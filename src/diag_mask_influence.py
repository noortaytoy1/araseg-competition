"""MASKED CROSS-SIDE-INFLUENCE DIAGNOSTIC for AraSeg NoPnx-NP. Additive,
analysis-only, CPU-ONLY (the GPU is owned by the integration battery).
Read-only on runs/ and probs/. Dev is used for EVALUATION of the signal only.

HONESTY FLAG (state the distinction, let the numbers speak):
  The killed surprise/OOV gates measured RAW SURPRISAL — "how improbable is
  this token given its context" — and landed at AUC ~0.5 / 0.52. This script
  measures something different: CROSS-SIDE INFLUENCE via masked context
  ablation — "how much does the OTHER side of a candidate cut help an MLM
  predict this token". Raw surprisal asks how odd a word is; influence asks
  whether the two sides inform each other. They can dissociate. The numbers
  below decide, not the framing.

THE QUANTITY. At candidate position i (a potential cut AFTER word i):
  RIGHT probe: target = word i+1 (first word of the right side).
      logP_full = MLM pseudo-logP of word i+1 (all its subwords masked) given
                  the full +/-ctx-word window around i.
      logP_abl  = same, but every subword of the LEFT side (words lo..i) is
                  additionally replaced by [MASK] up to the window edge.
      dP_right  = logP_full - logP_abl.
  LEFT probe: mirror — target = word i, ablate the right side (words i+1..hi).
      dP_left   = logP_full - logP_abl.
  dP = mean(dP_right, dP_left).
  HIGH dP  = the sides inform each other = same sentence.
  LOW  dP  = the sides are mutually uninformative = boundary.
  2 forward passes per probe direction = 4 forwards per position per model.

PROBES (two models, compared):
  (a) base : aubmindlab/bert-base-arabertv02 with its PRETRAINED MLM head.
  (b) tapt : runs/tapt_NoPnx-NP (MLM self-supervised on OUR 174 train docs —
      Noor's exact recipe). Verified to load as AutoModelForMaskedLM; if it
      does not, the script reports the failure and continues with (a) only.

SAMPLE (dev, evaluation only; positions deduped across groups):
  BOUND   1,500 true boundaries (gold 1)
  NONB    1,500 matched non-boundaries (same docs as the sampled boundaries,
          random interior gold-0 positions, per-doc matched counts)
  CONF-FP 1,000 tagger confident false positives (p>=0.8, gold 0) from the
          cached probs npz (cache_probs.py, union-NoPnx-NP-arabertv02-s42)
  CONF-TP 1,000 tagger confident true positives (p>=0.8, gold 1)
  Doc-final positions (i = n-1) are excluded everywhere: the RIGHT probe
  needs word i+1 to exist.

REPORT (failure-anatomy per standing order):
  (1) AUC of dP for boundary vs non-boundary overall
      (orientation: AUC = P(dP HIGHER on a non-boundary), i.e. -dP as the
       boundary score);
  (2) AUC for conf-FP vs conf-TP — the tagger-blind slice
      (orientation: AUC = P(dP HIGHER on a conf-FP); reference points:
       instability 0.711, 2D signal 0.6212, raw surprisal 0.52);
  (3) per-genre means (heuristic buckets from genre_buckets.bucket) — does
      the genealogy chain show HIGH cross-side influence at non-boundaries
      (the model knowing the links belong together) or is it blind there too;
  (4) dP distributions per group (mean/median/p10/p90), plus dP_right/dP_left
      separately.

PRE-REGISTERED READINGS (tick one, verbatim, per probe and for the best probe):
  [BUILD ] (2) >= 0.65 OR (1) >= 0.75 -> Noor's iterative grouping decoder
           gets built
  [PARK  ] (2) < 0.55 and (1) < 0.65 -> parked with the numbers
  [NOOR  ] between -> Noor's call

Usage (CPU only — battery owns the GPU; resume-safe via --cache):
  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python src/diag_mask_influence.py \
      --dev data/NoPnx-NP_dev.jsonl \
      --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --probe base --cache <scratch>/dmi_cache.json --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from genre_buckets import bucket


# --------------------------------------------------------------------------- #
# helpers (same estimator family as diag_stability.py / diag_unfamiliarity.py)
# --------------------------------------------------------------------------- #
def roc_auc(pos_vals, neg_vals):
    """ROC-AUC that pos_vals score HIGHER than neg_vals (Mann-Whitney U,
    ties get 0.5 credit)."""
    pos = np.asarray(pos_vals, dtype=float)
    neg = np.asarray(neg_vals, dtype=float)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank_per_val = (start + cum + 1) / 2.0
    ranks = avg_rank_per_val[inv]
    n_pos = pos.size
    sum_ranks_pos = ranks[:n_pos].sum()
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (pos.size * neg.size))


def summarize(name, vals):
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return f"{name:24s} n=0"
    return (f"{name:24s} n={a.size:5d}  mean={a.mean():8.3f}  "
            f"median={np.median(a):8.3f}  p10={np.percentile(a,10):8.3f}  "
            f"p90={np.percentile(a,90):8.3f}")


def load_probe(name, path_or_id):
    """Load (tokenizer, MLM model) or return (None, None, reason)."""
    try:
        tok = AutoTokenizer.from_pretrained(path_or_id)
        mdl, info = AutoModelForMaskedLM.from_pretrained(
            path_or_id, output_loading_info=True)
        mdl.eval()
        if tok.mask_token_id is None:
            return None, None, f"{name}: tokenizer has no [MASK] token"
        # a headless checkpoint would still "load" with a randomly initialised
        # MLM head — catch that via the loading report.
        bad = [k for k in info.get("missing_keys", [])
               if k.startswith("cls.predictions") and not k.endswith("decoder.bias")]
        if bad:
            return None, None, (f"{name}: MLM head weights missing from the "
                                f"checkpoint: {bad[:4]}")
        with torch.no_grad():
            out = mdl(input_ids=torch.tensor([[tok.cls_token_id,
                                               tok.mask_token_id,
                                               tok.sep_token_id]]))
        if out.logits.shape[-1] < len(tok) - 8:
            return None, None, (f"{name}: logits dim {out.logits.shape[-1]} "
                                f"!= vocab {len(tok)} — not an MLM head")
        return tok, mdl, None
    except Exception as e:  # noqa: BLE001 — report and fall back per spec
        return None, None, f"{name}: failed to load as MaskedLM: {e}"


# --------------------------------------------------------------------------- #
# core: dP for a batch of positions under one probe model
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_positions(positions, dev, tok, mdl, ctx, max_length, batch_positions,
                    cache, cache_path, probe_name, save_every=50):
    """Compute (dP_right, dP_left) for each (di, i) in `positions`.

    Results land in `cache` keyed "probe|di|i" -> [dpr, dpl]; NaNs mark
    positions whose target word produced zero subwords. Resume-safe: already
    cached keys are skipped; the cache is flushed every `save_every` chunks.
    """
    mask_id = tok.mask_token_id
    todo = [(di, i) for (di, i) in positions
            if f"{probe_name}|{di}|{i}" not in cache]
    print(f"# [{probe_name}] positions total={len(positions)}  "
          f"cached={len(positions) - len(todo)}  to-run={len(todo)}", flush=True)
    if not todo:
        return

    # pre-encode every window once; sort by subword length for tight batches
    encoded = []
    for (di, i) in todo:
        toks = dev[di]["tokens"]
        n = len(toks)
        lo = max(0, i - ctx)
        hi = min(n, i + 1 + ctx)
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                 for w in toks[lo:hi]]
        enc = tok(words, is_split_into_words=True, truncation=True,
                  max_length=max_length)
        word_ids = enc.word_ids(0) if hasattr(enc, "word_ids") else enc.word_ids()
        ids = enc["input_ids"]
        rel_i, rel_next = i - lo, i - lo + 1
        left_pos = [p for p, w in enumerate(word_ids)
                    if w is not None and w <= rel_i]
        right_pos = [p for p, w in enumerate(word_ids)
                     if w is not None and w >= rel_next]
        tgt_l = [p for p, w in enumerate(word_ids) if w == rel_i]
        tgt_r = [p for p, w in enumerate(word_ids) if w == rel_next]
        encoded.append((di, i, ids, left_pos, right_pos, tgt_l, tgt_r))
    encoded.sort(key=lambda e: len(e[2]))

    t0 = time.time()
    n_chunks = (len(encoded) + batch_positions - 1) // batch_positions
    for ci in range(n_chunks):
        chunk = encoded[ci * batch_positions:(ci + 1) * batch_positions]
        seqs, meta = [], []  # meta: (di, i, variant, tgt_positions, true_ids)
        for (di, i, ids, left_pos, right_pos, tgt_l, tgt_r) in chunk:
            if not tgt_l or not tgt_r:
                cache[f"{probe_name}|{di}|{i}"] = [float("nan"), float("nan")]
                continue
            base = list(ids)
            true_l = [ids[p] for p in tgt_l]
            true_r = [ids[p] for p in tgt_r]
            # RIGHT probe: mask word i+1; ablate = also mask whole left side
            r_full = list(base)
            for p in tgt_r:
                r_full[p] = mask_id
            r_abl = list(r_full)
            for p in left_pos:
                r_abl[p] = mask_id
            # LEFT probe: mask word i; ablate = also mask whole right side
            l_full = list(base)
            for p in tgt_l:
                l_full[p] = mask_id
            l_abl = list(l_full)
            for p in right_pos:
                l_abl[p] = mask_id
            for variant, seq, tgt, true in (("rf", r_full, tgt_r, true_r),
                                            ("ra", r_abl, tgt_r, true_r),
                                            ("lf", l_full, tgt_l, true_l),
                                            ("la", l_abl, tgt_l, true_l)):
                seqs.append(seq)
                meta.append((di, i, variant, tgt, true))
        if not seqs:
            continue
        maxlen = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, :len(s)] = 1
        logits = mdl(input_ids=input_ids, attention_mask=attn).logits
        lp = {}
        for r, (di, i, variant, tgt, true) in enumerate(meta):
            logprobs = torch.log_softmax(logits[r, tgt], dim=-1)
            lp[(di, i, variant)] = float(sum(
                logprobs[k, tid].item() for k, tid in enumerate(true)))
        done_pos = {(m[0], m[1]) for m in meta}
        for (di, i) in done_pos:
            dpr = lp[(di, i, "rf")] - lp[(di, i, "ra")]
            dpl = lp[(di, i, "lf")] - lp[(di, i, "la")]
            cache[f"{probe_name}|{di}|{i}"] = [dpr, dpl]
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            el = time.time() - t0
            rate = (ci + 1) * batch_positions / el
            eta = (n_chunks - ci - 1) * batch_positions / max(rate, 1e-9)
            print(f"# [{probe_name}] chunk {ci + 1}/{n_chunks}  "
                  f"({4 * (ci + 1) * batch_positions} fwds approx)  "
                  f"{el:.0f}s elapsed  ~{eta / 60:.1f} min left", flush=True)
        if cache_path and ((ci + 1) % save_every == 0 or ci + 1 == n_chunks):
            _save_cache(cache, cache_path)


def _save_cache(cache, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def report_probe(probe_name, groups, cache, genres):
    """Print the full failure-anatomy report for one probe. Returns (auc1, auc2)."""
    def vals(g, idx):
        out = []
        for (di, i) in groups[g]:
            v = cache.get(f"{probe_name}|{di}|{i}")
            if v is None:
                continue
            dpr, dpl = v
            if idx == "r":
                out.append(dpr)
            elif idx == "l":
                out.append(dpl)
            else:
                out.append(np.nanmean([dpr, dpl]))
        return out

    print(f"\n=================== PROBE [{probe_name}] ===================")
    print("\n--- (4) dP distributions per group (dP = mean of both directions) ---")
    for g in ["BOUND", "NONB", "CONF-FP", "CONF-TP"]:
        print("    " + summarize(g, vals(g, "m")))
    print("\n    dP_right only (left-side ablated):")
    for g in ["BOUND", "NONB", "CONF-FP", "CONF-TP"]:
        print("    " + summarize(g, vals(g, "r")))
    print("\n    dP_left only (right-side ablated):")
    for g in ["BOUND", "NONB", "CONF-FP", "CONF-TP"]:
        print("    " + summarize(g, vals(g, "l")))

    auc1 = roc_auc(vals("NONB", "m"), vals("BOUND", "m"))
    auc2 = roc_auc(vals("CONF-FP", "m"), vals("CONF-TP", "m"))
    auc1_r = roc_auc(vals("NONB", "r"), vals("BOUND", "r"))
    auc1_l = roc_auc(vals("NONB", "l"), vals("BOUND", "l"))
    auc2_r = roc_auc(vals("CONF-FP", "r"), vals("CONF-TP", "r"))
    auc2_l = roc_auc(vals("CONF-FP", "l"), vals("CONF-TP", "l"))

    print("\n--- (1) AUC boundary vs non-boundary  [AUC = P(dP higher on NONB)] ---")
    print(f"    dP mean-of-directions : {auc1:.4f}")
    print(f"    dP_right only         : {auc1_r:.4f}")
    print(f"    dP_left  only         : {auc1_l:.4f}")
    print("\n--- (2) AUC conf-FP vs conf-TP (tagger-blind slice)  "
          "[AUC = P(dP higher on conf-FP)] ---")
    print("    reference points on this slice: instability 0.711, "
          "2D signal 0.6212, raw surprisal 0.52")
    print(f"    dP mean-of-directions : {auc2:.4f}")
    print(f"    dP_right only         : {auc2_r:.4f}")
    print(f"    dP_left  only         : {auc2_l:.4f}")

    # ---------------- (3) per-genre means -------------------------------- #
    print("\n--- (3) per-genre mean dP (heuristic buckets; genealogy = the "
          "'X begat Y' chain) ---")
    print(f"    {'genre':20s} {'grp':8s} {'n':>6} {'mean dP':>9} "
          f"{'median':>9}   per-genre AUC(NONB>BOUND)")
    by = defaultdict(lambda: defaultdict(list))
    for g in ["BOUND", "NONB", "CONF-FP", "CONF-TP"]:
        for (di, i) in groups[g]:
            v = cache.get(f"{probe_name}|{di}|{i}")
            if v is None or np.all(np.isnan(v)):
                continue
            by[genres[di]][g].append(float(np.nanmean(v)))
    order = ["general prose", "legal/constitution", "hadith/isnad",
             "genealogy", "cloze/exercise"]
    for genre in order:
        if genre not in by:
            continue
        ga = roc_auc(by[genre].get("NONB", []), by[genre].get("BOUND", []))
        first = True
        for g in ["BOUND", "NONB", "CONF-FP", "CONF-TP"]:
            v = np.asarray(by[genre].get(g, []), dtype=float)
            if v.size == 0:
                continue
            tail = f"   AUC={ga:.3f}" if first else ""
            print(f"    {genre:20s} {g:8s} {v.size:6d} {v.mean():9.3f} "
                  f"{np.median(v):9.3f}{tail}")
            first = False

    return auc1, auc2


def tick_reading(probe_name, auc1, auc2):
    build = (auc2 >= 0.65) or (auc1 >= 0.75)
    park = (auc2 < 0.55) and (auc1 < 0.65)
    print(f"\n--- PRE-REGISTERED READING [{probe_name}]  "
          f"(1)={auc1:.4f}  (2)={auc2:.4f} ---")
    print(f"  [{'X' if build else ' '}] (2) >= 0.65 OR (1) >= 0.75 -> "
          "Noor's iterative grouping decoder gets built")
    print(f"  [{'X' if park else ' '}] (2) < 0.55 and (1) < 0.65 -> "
          "parked with the numbers")
    print(f"  [{'X' if (not build and not park) else ' '}] between -> Noor's call")
    return "BUILD" if build else ("PARK" if park else "NOOR")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--probs", default="probs/union-NoPnx-NP-arabertv02-s42_dev.npz")
    ap.add_argument("--base-model", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--tapt-model", default="runs/tapt_NoPnx-NP")
    ap.add_argument("--probe", choices=["base", "tapt", "both"], default="both")
    ap.add_argument("--n-bound", type=int, default=1500)
    ap.add_argument("--n-nonb", type=int, default=1500)
    ap.add_argument("--n-fp", type=int, default=1000)
    ap.add_argument("--n-tp", type=int, default=1000)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--ctx", type=int, default=60,
                    help="+/- context words around the candidate cut")
    ap.add_argument("--batch-positions", type=int, default=4,
                    help="positions per forward batch (4 sequences each)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=24,
                    help="torch CPU threads (leave headroom for the GPU battery)")
    ap.add_argument("--cache", default=None,
                    help="resume-safe JSON cache of per-position (dpr, dpl)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: cap positions per group (0 = full sample)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if torch.cuda.is_available():
        raise SystemExit("CPU ONLY: battery owns the GPU. Set CUDA_VISIBLE_DEVICES=-1.")
    torch.set_num_threads(max(1, args.threads))
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    print("# MASKED CROSS-SIDE-INFLUENCE DIAGNOSTIC (dev = evaluation only, CPU only)")
    print(f"# dev={args.dev}  probs={args.probs}  seed={args.seed}  "
          f"ctx=+/-{args.ctx}w  threads={args.threads}")
    print("# HONESTY FLAG: the killed surprise/OOV gates measured RAW surprisal")
    print("#   (AUC ~0.5); this measures CROSS-SIDE INFLUENCE via masked context")
    print("#   ablation (does one side help predict the other). Different quantity;")
    print("#   the numbers below decide.")

    dev = load_jsonl(args.dev)
    cache_npz = np.load(args.probs)
    genres = {di: bucket(d["tokens"]) for di, d in enumerate(dev)}

    # ------------------------------------------------ sampling (seeded)
    bound_pool, fp_pool, tp_pool = [], [], []
    nonb_by_doc = defaultdict(list)
    for di, d in enumerate(dev):
        toks, gold = d["tokens"], d["labels"]
        p = cache_npz[d["doc_id"]]
        n = len(toks)
        for i in range(n - 1):  # i = n-1 excluded: RIGHT probe needs word i+1
            if toks[i] == PARAGRAPH_TOKEN or toks[i + 1] == PARAGRAPH_TOKEN:
                continue
            if gold[i] == 1:
                bound_pool.append((di, i))
                if p[i] >= args.hi:
                    tp_pool.append((di, i))
            else:
                if 1 <= i <= n - 3:  # interior only for the matched negatives
                    nonb_by_doc[di].append((di, i))
                if p[i] >= args.hi:
                    fp_pool.append((di, i))

    def sample(pool, k):
        if len(pool) <= k:
            return list(pool)
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[j] for j in idx]

    n_bound = args.limit or args.n_bound
    n_nonb = args.limit or args.n_nonb
    n_fp = args.limit or args.n_fp
    n_tp = args.limit or args.n_tp

    bound_s = sample(bound_pool, n_bound)
    # matched non-boundaries: same docs, per-doc matched counts, random interior
    per_doc = defaultdict(int)
    for (di, _i) in bound_s:
        per_doc[di] += 1
    total_want = min(n_nonb, sum(len(v) for v in nonb_by_doc.values()))
    nonb_s, short = [], 0
    for di, cnt in per_doc.items():
        take = sample(nonb_by_doc[di], cnt)
        nonb_s.extend(take)
        short += cnt - len(take)
    if short:
        print(f"# note: {short} matched negatives short in their own doc; "
              "topped up from the other sampled docs")
    if len(nonb_s) > total_want:
        idx = rng.choice(len(nonb_s), size=total_want, replace=False)
        nonb_s = [nonb_s[j] for j in idx]
    elif len(nonb_s) < total_want:  # top up from the same sampled docs
        chosen = set(nonb_s)
        rest = [x for di in per_doc for x in nonb_by_doc[di] if x not in chosen]
        nonb_s.extend(sample(rest, total_want - len(nonb_s)))
    fp_s = sample(fp_pool, n_fp)
    tp_s = sample(tp_pool, n_tp)

    groups = {"BOUND": bound_s, "NONB": nonb_s, "CONF-FP": fp_s, "CONF-TP": tp_s}
    uniq = sorted({pos for g in groups.values() for pos in g})
    print(f"# dev docs={len(dev)}  pools: bound={len(bound_pool)} "
          f"conf-FP={len(fp_pool)} conf-TP={len(tp_pool)}")
    print(f"# sampled: BOUND={len(bound_s)} NONB={len(nonb_s)} "
          f"CONF-FP={len(fp_s)} CONF-TP={len(tp_s)}  "
          f"unique-positions={len(uniq)}  (~{4 * len(uniq)} forwards/probe)")

    # ------------------------------------------------ probes
    probes = []
    if args.probe in ("base", "both"):
        probes.append(("base", args.base_model))
    if args.probe in ("tapt", "both"):
        probes.append(("tapt", args.tapt_model))

    cache = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"# resume cache loaded: {len(cache)} entries from {args.cache}")

    results = {}
    for probe_name, path in probes:
        tok, mdl, err = load_probe(probe_name, path)
        if err is not None:
            print(f"\n# PROBE [{probe_name}] UNAVAILABLE — {err}")
            if probe_name == "tapt":
                print("# (per instruction: reporting the failure and using the "
                      "base probe only)")
            continue
        print(f"\n# probe [{probe_name}] = {path}  "
              f"(vocab={len(tok)}, mask_id={tok.mask_token_id})", flush=True)
        score_positions(uniq, dev, tok, mdl, args.ctx, args.max_length,
                        args.batch_positions, cache, args.cache, probe_name)
        del mdl
        auc1, auc2 = report_probe(probe_name, groups, cache, genres)
        verdict = tick_reading(probe_name, auc1, auc2)
        results[probe_name] = dict(auc1=auc1, auc2=auc2, verdict=verdict)

    if len(results) > 1:
        best = max(results, key=lambda k: max(results[k]["auc1"],
                                              results[k]["auc2"]))
        print(f"\n=========== OVERALL (best probe = [{best}]) ===========")
        tick_reading(f"best={best}", results[best]["auc1"], results[best]["auc2"])

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        payload = dict(args={k: v for k, v in vars(args).items()},
                       results=results,
                       groups={g: [[di, i] for (di, i) in v]
                               for g, v in groups.items()},
                       genres={str(k): v for k, v in genres.items()})
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"# wrote {args.out_json}")

    print(f"\n# done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
