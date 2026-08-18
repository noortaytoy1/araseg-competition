"""STABILITY-GATE DIAGNOSTIC (CCPS-style) for an AraSeg NoPnx single-encoder
checkpoint. Additive, analysis-only, CPU-only. Read-only on runs/ and probs/.

KILL-GATE QUESTION (pre-registered, same protocol as the OOV gate and the KD
diagnostic): does REPRESENTATION INSTABILITY under perturbation separate the
model's confident FALSE cuts (gold 0, p>=0.8) from its confident TRUE
boundaries (gold 1, p>=0.8)?  Logits cannot: both say 0.99.

METHOD (sampled, closed-legal — dev is used for EVALUATION only):
  1. From the cached dev probs npz (produced by cache_probs.py from this exact
     checkpoint with the predict.py windowing, window=180 overlap=60) plus the
     gold labels, sample N confident-FP and N confident-TP decision positions
     ("\n" positions excluded, as everywhere else).
  2. Re-enumerate the EXACT predict.py windows per doc (tokenization only) and
     dedupe: every window that contains at least one sampled position is run
     T times with dropout ACTIVE (model.train(), fixed per-window seed) as ONE
     batched forward of T copies — MC-dropout.
  3. Per sampled position and pass t, the boundary probability is the mean of
     the last-subword P(boundary) over all covering windows — identical
     aggregation to predict.py. Instability := std over the T passes (ddof=1).
  4. Secondary (near-free) signal: additive Gaussian noise (sigma) on the CLEAN
     final hidden states, re-decoded through the classifier head K times; std
     of the boundary probability across draws.

REPORT: mean/median instability for conf-FP vs conf-TP; ROC-AUC of instability
for separating them (AUC = P(instability higher on a conf-FP)); and the
operating table in the OOV-gate autopsy format: for each instability cutoff,
how many of the 2,236 confident FPs would a "suppress unstable fires" filter
kill vs how many of the ~10,985 confident TPs it would lose (sample rates
extrapolated to the full dev counts).

DECISION RULE (pre-registered):
  AUC <  0.60  ->  DEAD — park with the OOV gate.
  AUC >= 0.65  ->  build the decode-time filter.
  0.60-0.65    ->  gray zone: does NOT meet the build bar; no build.

Usage (CPU only — battery owns the GPU):
  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python src/diag_stability.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev data/NoPnx-NP_dev.jsonl \
      --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --hi 0.8 --n-per-group 400 --passes 8 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def roc_auc(pos_vals, neg_vals):
    """ROC-AUC that pos_vals score HIGHER than neg_vals (Mann-Whitney U,
    ties get 0.5 credit). Same estimator as diag_unfamiliarity.py."""
    pos = np.asarray(pos_vals, dtype=float)
    neg = np.asarray(neg_vals, dtype=float)
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
    if a.size == 0:
        return f"{name:26s} n=0"
    return (f"{name:26s} n={a.size:5d}  mean={a.mean():8.4f}  "
            f"median={np.median(a):8.4f}  p90={np.percentile(a, 90):8.4f}")


def enumerate_windows(words, tokenizer, window, overlap, max_length):
    """Replicate the EXACT predict.py window-start sequence (tokenization only).

    Returns a list of (start, coverage) where coverage maps ABSOLUTE word index
    -> token position of that word's LAST subword within the window encoding.
    """
    n = len(words)
    out = []
    start = 0
    while start < n:
        chunk = words[start:start + window]
        enc = tokenizer(chunk, is_split_into_words=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        word_ids = enc.word_ids(0)
        coverage = {}
        last_seen = -1
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
            if nxt != wid:  # last subword of this word
                coverage[start + wid] = pos
            last_seen = max(last_seen, wid)
        out.append((start, coverage))
        covered_through = start + last_seen
        if covered_through >= n - 1:
            break
        start = max(covered_through + 1 - overlap, start + 1)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/union-NoPnx-NP-arabertv02-s42")
    ap.add_argument("--dev", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--probs", default="probs/union-NoPnx-NP-arabertv02-s42_dev.npz")
    ap.add_argument("--hi", type=float, default=0.8,
                    help="confidence floor defining conf-FP / conf-TP groups")
    ap.add_argument("--n-per-group", type=int, default=400)
    ap.add_argument("--passes", type=int, default=8, help="MC-dropout passes T")
    ap.add_argument("--noise-sigma", type=float, default=0.02,
                    help="Gaussian sigma on final hidden states (secondary signal); 0 disables")
    ap.add_argument("--noise-draws", type=int, default=8)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None,
                    help="optional path for a JSON dump of per-position results")
    args = ap.parse_args()

    if torch.cuda.is_available():
        raise SystemExit("CPU ONLY: battery owns the GPU. Set CUDA_VISIBLE_DEVICES=-1.")
    device = "cpu"
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    print(f"# STABILITY-GATE DIAGNOSTIC  model={args.model}")
    print(f"# probs={args.probs}  dev={args.dev}")
    print(f"# hi={args.hi}  n_per_group={args.n_per_group}  T={args.passes}  "
          f"noise_sigma={args.noise_sigma}x{args.noise_draws}  seed={args.seed}")

    dev = load_jsonl(args.dev)
    cache = np.load(args.probs)

    # ------------------------------------------------ 1. pick sampled positions
    conf_fp, conf_tp = [], []  # (doc_idx, word_idx, cached_p)
    for di, d in enumerate(dev):
        p = cache[d["doc_id"]]
        toks, gold = d["tokens"], d["labels"]
        for i in range(len(toks)):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            if p[i] >= args.hi:
                (conf_tp if gold[i] == 1 else conf_fp).append((di, i, float(p[i])))
    print(f"# dev docs={len(dev)}  confident-FP={len(conf_fp)}  confident-TP={len(conf_tp)}")

    def sample(pool, n):
        if len(pool) <= n:
            return list(pool)
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[j] for j in idx]

    fp_s = sample(conf_fp, args.n_per_group)
    tp_s = sample(conf_tp, args.n_per_group)
    sampled = {(di, i): ("FP", cp) for di, i, cp in fp_s}
    sampled.update({(di, i): ("TP", cp) for di, i, cp in tp_s})
    print(f"# sampled: FP={len(fp_s)}  TP={len(tp_s)}")

    # ------------------------------------------------ 2. model + window dedupe
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    model.eval()

    docs_needed = sorted({di for di, _ in sampled})
    # windows to run: (di, window_index_in_doc) -> (start, coverage)
    win_jobs = {}
    pos2wins = defaultdict(list)  # (di,i) -> list of window keys covering it
    for di in docs_needed:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                 for w in dev[di]["tokens"]]
        wins = enumerate_windows(words, tokenizer, args.window, args.overlap,
                                 args.max_length)
        wanted = [i for (dj, i) in sampled if dj == di]
        for wi, (start, coverage) in enumerate(wins):
            hits = [i for i in wanted if i in coverage]
            if hits:
                win_jobs[(di, wi)] = (start, coverage)
                for i in hits:
                    pos2wins[(di, i)].append((di, wi))
    n_forwards = len(win_jobs)
    print(f"# unique windows to perturb: {n_forwards}  "
          f"(dedup over {len(sampled)} positions)")

    dropped = [k for k in sampled if not pos2wins[k]]
    if dropped:
        print(f"# WARNING: {len(dropped)} sampled positions had no covering window; dropped")
        for k in dropped:
            del sampled[k]

    # ------------------------------------------------ 3. perturbed forwards
    # per window: clean prob per covered word, T mc-dropout probs, K noise probs
    clean_p = defaultdict(list)   # (di,i) -> [per-window clean prob]
    mc_p = defaultdict(list)      # (di,i) -> [per-window array of T probs]
    noise_p = defaultdict(list)   # (di,i) -> [per-window array of K probs]

    for widx, ((di, wi), (start, coverage)) in enumerate(sorted(win_jobs.items())):
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                 for w in dev[di]["tokens"]]
        chunk = words[start:start + args.window]
        enc = tokenizer(chunk, is_split_into_words=True, truncation=True,
                        max_length=args.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        needed = [i for i in coverage if (di, i) in sampled]
        tokpos = [coverage[i] for i in needed]

        # --- clean eval pass (+ final hidden states for the noise variant)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=(args.noise_sigma > 0))
            pc = torch.softmax(out.logits[0], dim=-1)[:, 1]
            for i, tp in zip(needed, tokpos):
                clean_p[(di, i)].append(float(pc[tp]))
            if args.noise_sigma > 0:
                h = out.hidden_states[-1][0]  # (L, H) final hidden states
                torch.manual_seed(args.seed * 1_000_003 + widx * 2 + 1)
                with torch.no_grad():
                    draws = []
                    for _ in range(args.noise_draws):
                        hn = h + args.noise_sigma * torch.randn_like(h)
                        pn = torch.softmax(model.classifier(hn), dim=-1)[:, 1]
                        draws.append(pn)
                    draws = torch.stack(draws)  # (K, L)
                for i, tp in zip(needed, tokpos):
                    noise_p[(di, i)].append(draws[:, tp].numpy().copy())

        # --- MC-dropout: T copies in one batch, dropout active, fixed seed
        model.train()
        torch.manual_seed(args.seed * 1_000_003 + widx * 2)
        batch = {k: v.repeat(args.passes, 1) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**batch).logits  # (T, L, 2)
        model.eval()
        pt = torch.softmax(logits, dim=-1)[:, :, 1]  # (T, L)
        for i, tp in zip(needed, tokpos):
            mc_p[(di, i)].append(pt[:, tp].numpy().copy())

        if (widx + 1) % 50 == 0 or widx + 1 == n_forwards:
            print(f"#   window {widx + 1}/{n_forwards}  "
                  f"({time.time() - t_start:.0f}s elapsed)", flush=True)

    # ------------------------------------------------ 4. per-position metrics
    rows = []
    for (di, i), (grp, cached) in sampled.items():
        mc = np.stack(mc_p[(di, i)])          # (n_win, T)
        p_per_pass = mc.mean(axis=0)          # predict.py aggregation, per pass
        mc_std = float(np.std(p_per_pass, ddof=1))
        cl = float(np.mean(clean_p[(di, i)]))
        row = dict(di=di, i=i, grp=grp, cached=cached, clean=cl,
                   mc_std=mc_std, mc_mean=float(p_per_pass.mean()))
        if args.noise_sigma > 0:
            nz = np.stack(noise_p[(di, i)]).mean(axis=0)  # (K,)
            row["noise_std"] = float(np.std(nz, ddof=1))
        rows.append(row)

    FP = [r for r in rows if r["grp"] == "FP"]
    TP = [r for r in rows if r["grp"] == "TP"]

    # sanity: clean re-prediction should match the cache
    dif = [abs(r["clean"] - r["cached"]) for r in rows]
    print(f"\n# sanity: |clean re-predict - cached| mean={np.mean(dif):.4f}  "
          f"max={np.max(dif):.4f}")

    print("\n=========== STABILITY-GATE DIAGNOSTIC (conf-FP vs conf-TP, p>=%.2f) ==========="
          % args.hi)
    print("\n--- MC-DROPOUT instability: std of P(boundary) over %d passes ---" % args.passes)
    print("    " + summarize("conf-FP (gold 0)", [r["mc_std"] for r in FP]))
    print("    " + summarize("conf-TP (gold 1)", [r["mc_std"] for r in TP]))
    auc_mc = roc_auc([r["mc_std"] for r in FP], [r["mc_std"] for r in TP])
    print(f"\n  ROC-AUC (instability higher on conf-FP): {auc_mc:.3f}")

    auc_nz = float("nan")
    if args.noise_sigma > 0:
        print(f"\n--- HIDDEN-NOISE instability: std over {args.noise_draws} draws "
              f"(sigma={args.noise_sigma}) ---")
        print("    " + summarize("conf-FP (gold 0)", [r["noise_std"] for r in FP]))
        print("    " + summarize("conf-TP (gold 1)", [r["noise_std"] for r in TP]))
        auc_nz = roc_auc([r["noise_std"] for r in FP], [r["noise_std"] for r in TP])
        print(f"\n  ROC-AUC (instability higher on conf-FP): {auc_nz:.3f}")

    # ------------------------------------------------ 5. operating table
    n_fp_full, n_tp_full = len(conf_fp), len(conf_tp)
    fp_std = np.array([r["mc_std"] for r in FP])
    tp_std = np.array([r["mc_std"] for r in TP])
    print("\n--- OPERATING TABLE: 'suppress a p>=%.2f fire when mc_std >= cutoff' ---"
          % args.hi)
    print(f"(sample rates extrapolated to the full dev counts: "
          f"{n_fp_full} conf-FPs, {n_tp_full} conf-TPs)")
    print(f"\n  {'cutoff (mc_std >=)':22s} {'FP_kill%':>9} {'TP_loss%':>9} "
          f"{'~FP_killed':>11} {'~TP_lost':>9}")
    cutoffs = sorted(set(
        [round(c, 4) for c in np.percentile(tp_std, [99, 95, 90, 75, 50]).tolist()]
        + [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]))
    table = []
    for c in cutoffs:
        killp = 100.0 * float((fp_std >= c).mean())
        lossp = 100.0 * float((tp_std >= c).mean())
        table.append(dict(cutoff=c, fp_kill_pct=killp, tp_loss_pct=lossp,
                          fp_killed=killp / 100 * n_fp_full,
                          tp_lost=lossp / 100 * n_tp_full))
        print(f"  {c:<22.4f} {killp:9.1f} {lossp:9.1f} "
              f"{killp / 100 * n_fp_full:11.0f} {lossp / 100 * n_tp_full:9.0f}")

    # ------------------------------------------------ 6. pre-registered verdict
    print("\n--- PRE-REGISTERED DECISION RULE ---")
    print(f"MC-dropout AUC = {auc_mc:.3f}   (secondary hidden-noise AUC = {auc_nz:.3f})")
    if auc_mc < 0.60:
        verdict = "DEAD"
        print("VERDICT: AUC < 0.60 -> DEAD. Park the stability gate with the OOV gate.")
    elif auc_mc >= 0.65:
        verdict = "BUILD"
        print("VERDICT: AUC >= 0.65 -> build the decode-time instability filter.")
    else:
        verdict = "GRAY"
        print("VERDICT: 0.60 <= AUC < 0.65 -> gray zone. Does NOT meet the build bar; "
              "no build.")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(dict(args=vars(args), auc_mc=auc_mc, auc_noise=auc_nz,
                           verdict=verdict, n_fp_full=n_fp_full,
                           n_tp_full=n_tp_full, table=table, rows=rows), f)
        print(f"# wrote {args.out_json}")

    print(f"\n# done in {time.time() - t_start:.0f}s "
          f"({n_forwards} windows x {args.passes} passes, CPU)")


if __name__ == "__main__":
    main()
