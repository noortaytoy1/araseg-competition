"""UNFAMILIARITY-GATE DIAGNOSTIC for an AraSeg NoPnx-NP single-encoder checkpoint.

Analysis only. No training. Loads an existing fine-tuned checkpoint, predicts
per-token P(boundary) on the dev split with the exact windowing in predict.py
(same as diag_preposition.py), and asks ONE decisive question:

  Can the model's CONFIDENT OVER-FIRING be gated by an EXTERNAL "the model does
  not understand this token" (unfamiliarity) signal, WITHOUT suppressing real
  boundaries?

For every dev decision position it computes THREE cheap familiarity signals,
all closed-legal (train + AraBERTv02 tokenizer only; the MLM signal is an
inference-only forward pass, no training):

  1. SUBWORD-FRAGMENTATION  - how many AraBERTv02 subword pieces the word AT the
     position (frag_here) and the word JUST AFTER it (frag_next) split into.
     OOV proxy, no forward pass. Also max(here,next).
  2. TRAIN-FREQUENCY        - count of the token in NoPnx-NP_train.jsonl (174
     docs), count of the local (tok_i, tok_{i+1}) bigram, and binary
     seen/unseen flags for both. Rarer / unseen  = less familiar.
  3. MLM PSEUDO-SURPRISAL   - (optional, subsample only) AraBERTv02 masked-LM
     negative log-prob of the token's FIRST subword given its context, on a
     random subsample (~--mlm-sample positions) to avoid O(tokens) forward
     passes. Higher = more surprising / less familiar.

Then it compares the distribution of each signal across FOUR groups of dev
decision positions (position i, label g_i, prob p_i):

  (a) CONFIDENT FALSE POSITIVES  gold 0, p >= --hi   (the over-firing to suppress)
  (b) TRUE BOUNDARIES            gold 1               (a gate must NOT suppress these)
  (c) TRUE NON-BOUNDARIES        gold 0, p <  --lo
  (d) CONFIDENT FALSE NEGATIVES  gold 1, p <= --fn-hi

Reports mean/median of each signal per group and the ROC-AUC of each signal for
separating (a) from (b). Decisive test: are the confident FPs markedly MORE
unfamiliar (more subwords / rarer / higher surprisal) than TRUE boundaries?
  If yes  -> a "raise threshold when unfamiliar" gate can cut FPs without
             killing real boundaries.
  If no   -> the signal collides with real boundaries; the gate would hurt
             recall and the idea is weak.

Finally it estimates a concrete precision/recall tradeoff: for a family of
simple gate rules ("require higher confidence to fire when fragmentation is
high / bigram is unseen"), how many of the confident FPs get suppressed vs how
many TRUE boundaries are lost.

Usage (CUDA auto-detected):
  PYTHONIOENCODING=utf-8 python src/diag_unfamiliarity.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev data/NoPnx-NP_dev.jsonl \
      --train data/NoPnx-NP_train.jsonl \
      --hi 0.8 --lo 0.5 --fn-hi 0.2 \
      --mlm aubmindlab/bert-base-arabertv02 --mlm-sample 1500 --seed 0
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from transformers import (AutoModelForMaskedLM, AutoModelForTokenClassification,
                          AutoTokenizer)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def collect_probs(docs, tokenizer, model, device, window, overlap, max_length):
    """Return per-doc arrays of P(boundary) aligned to d['tokens']."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tokenizer, model, device, window, overlap, max_length)
        out.append(p)
    return out


def build_train_stats(train_docs):
    """Unigram counts, bigram counts over the train split."""
    uni = Counter()
    bi = Counter()
    for d in train_docs:
        toks = d["tokens"]
        for i, t in enumerate(toks):
            uni[t] += 1
            if i + 1 < len(toks):
                bi[(t, toks[i + 1])] += 1
    return uni, bi


def frag_count(word, tokenizer, cache):
    """Number of subword pieces `word` splits into (no special tokens)."""
    c = cache.get(word)
    if c is None:
        # [PAR] stand-in is a single special token; the raw "\n" never reaches here.
        c = len(tokenizer(word, add_special_tokens=False)["input_ids"])
        c = max(c, 1)  # UNK / empty -> 1
        cache[word] = c
    return c


def roc_auc(pos_vals, neg_vals):
    """ROC-AUC that `pos_vals` (group a) score HIGHER than `neg_vals` (group b).

    Rank-based Mann-Whitney U estimator; ties get 0.5 credit. Returns AUC in the
    orientation "P(random a-value > random b-value)". 0.5 = no separation.
    """
    pos = np.asarray(pos_vals, dtype=float)
    neg = np.asarray(neg_vals, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1, dtype=float)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank_per_val = (start + cum + 1) / 2.0  # mean of consecutive ranks in the tie block
    ranks = avg_rank_per_val[inv]
    n_pos = pos.size
    sum_ranks_pos = ranks[: pos.size].sum()
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (pos.size * neg.size))


def summarize(name, vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return f"{name:26s} n=0"
    return (f"{name:26s} n={a.size:6d}  mean={a.mean():8.3f}  median={np.median(a):7.3f}  "
            f"p90={np.percentile(a,90):7.3f}")


# --------------------------------------------------------------------------- #
# MLM pseudo-surprisal (subsample only, inference-only)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mlm_surprisal_for_positions(sample, dev, mlm_name, device, window_ctx=64):
    """For each (doc_idx, i) in `sample`, mask the FIRST subword of word i in a
    local context window and return the negative log-prob the MLM assigns to the
    true first-subword id. One forward pass per position (subsample kept small).

    Context is the raw NoPnx tokens around word i (no [PAR] substitution needed;
    [PAR]/"\\n" positions are never sampled). This is a familiarity proxy, not a
    calibrated language model score.
    """
    tok = AutoTokenizer.from_pretrained(mlm_name)
    mlm = AutoModelForMaskedLM.from_pretrained(mlm_name).to(device).eval()
    mask_id = tok.mask_token_id

    out = {}
    for (di, i) in sample:
        toks = dev[di]["tokens"]
        n = len(toks)
        lo = max(0, i - window_ctx)
        hi = min(n, i + window_ctx + 1)
        chunk = toks[lo:hi]
        rel = i - lo  # index of the target word within the chunk

        enc = tok(chunk, is_split_into_words=True, truncation=True,
                  max_length=512, return_tensors="pt")
        word_ids = enc.word_ids(0)
        input_ids = enc["input_ids"][0].clone()

        # subword positions belonging to the target word
        tgt_pos = [p for p, w in enumerate(word_ids) if w == rel]
        if not tgt_pos:
            out[(di, i)] = float("nan")
            continue
        first = tgt_pos[0]
        true_id = int(input_ids[first].item())
        input_ids[first] = mask_id

        logits = mlm(input_ids=input_ids.unsqueeze(0).to(device),
                     attention_mask=enc["attention_mask"].to(device)).logits[0]
        logp = torch.log_softmax(logits[first], dim=-1)
        out[(di, i)] = float(-logp[true_id].item())  # surprisal in nats
    del mlm
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--hi", type=float, default=0.8, help="confident-FP prob floor (group a)")
    ap.add_argument("--lo", type=float, default=0.5, help="true-non-boundary prob ceiling (group c)")
    ap.add_argument("--fn-hi", type=float, default=0.2, help="confident-FN prob ceiling (group d)")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--mlm", default="aubmindlab/bert-base-arabertv02",
                    help="MLM for pseudo-surprisal; empty string disables it")
    ap.add_argument("--mlm-sample", type=int, default=1500,
                    help="random # of positions to score with the MLM (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device={device}  model={args.model}")
    print(f"# thresholds: confident-FP p>={args.hi}  true-nonboundary p<{args.lo}  "
          f"confident-FN p<={args.fn_hi}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    dev = load_jsonl(args.dev)
    train = load_jsonl(args.train)
    print(f"# dev docs={len(dev)}  train docs={len(train)}")

    probs = collect_probs(dev, tokenizer, model, device,
                          args.window, args.overlap, args.max_length)
    uni, bi = build_train_stats(train)
    print(f"# train unigram types={len(uni)}  bigram types={len(bi)}")

    frag_cache = {}

    # ----------------------------------------------------------------- collect
    # one record per real decision position (exclude [PAR]/"\n" forced-0 sites).
    # group: 'a' conf-FP, 'b' true-boundary, 'c' true-nonboundary, 'd' conf-FN,
    #        'other' = leftover (gold0 lo<=p<hi, or gold1 fn_hi<p) -> kept for totals only
    recs = []  # dicts
    for di, (d, p) in enumerate(zip(dev, probs)):
        toks = d["tokens"]
        gold = d["labels"]
        n = len(toks)
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            pi = float(p[i])
            gi = int(gold[i])

            here = toks[i]
            nxt = toks[i + 1] if i + 1 < n else None

            frag_here = frag_count(MODEL_PAR_TOKEN if here == PARAGRAPH_TOKEN else here,
                                   tokenizer, frag_cache)
            if nxt is None or nxt == PARAGRAPH_TOKEN:
                frag_next = 0
            else:
                frag_next = frag_count(nxt, tokenizer, frag_cache)
            frag_max = max(frag_here, frag_next)

            uni_here = uni.get(here, 0)
            uni_next = uni.get(nxt, 0) if nxt is not None else 0
            bi_cnt = bi.get((here, nxt), 0) if nxt is not None else 0
            unseen_uni = 1 if uni_here == 0 else 0            # current token never seen in train
            unseen_bi = 1 if (nxt is not None and bi_cnt == 0) else 0  # local bigram never seen
            # log1p freq: higher = MORE familiar; we negate later so "unfamiliar high"
            neg_log_uni = -np.log1p(uni_here)                 # higher = rarer = less familiar
            neg_log_bi = -np.log1p(bi_cnt)

            if gi == 0 and pi >= args.hi:
                grp = "a"
            elif gi == 1:
                grp = "b"                          # all gold-1 are (b); conf-FN (d) is a subset flagged below
            elif gi == 0 and pi < args.lo:
                grp = "c"
            else:
                grp = "other"
            is_fn = (gi == 1 and pi <= args.fn_hi)  # confident FN, subset of (b)

            recs.append(dict(di=di, i=i, p=pi, g=gi, grp=grp, is_fn=is_fn,
                             frag_here=frag_here, frag_next=frag_next, frag_max=frag_max,
                             unseen_uni=unseen_uni, unseen_bi=unseen_bi,
                             neg_log_uni=neg_log_uni, neg_log_bi=neg_log_bi,
                             uni_here=uni_here, bi_cnt=bi_cnt))

    def group(g):
        return [r for r in recs if r["grp"] == g]

    A = group("a")              # confident FP
    B = group("b")              # ALL true boundaries (gold 1)
    C = group("c")              # true non-boundary
    D = [r for r in recs if r["is_fn"]]  # confident FN (subset of B)

    n_pos = len(recs)
    print(f"\n================= AraSeg NoPnx-NP UNFAMILIARITY-GATE DIAGNOSTIC =================")
    print(f"total decision positions (excl. [PAR]) = {n_pos}")
    print(f"(a) CONFIDENT FALSE POSITIVES gold0 p>={args.hi}   = {len(A)}")
    print(f"(b) TRUE BOUNDARIES           gold1              = {len(B)}")
    print(f"(c) TRUE NON-BOUNDARIES       gold0 p<{args.lo}      = {len(C)}")
    print(f"(d) CONFIDENT FALSE NEGATIVES gold1 p<={args.fn_hi}   = {len(D)}   (subset of b)")

    # ------------------------------------------------------------ MLM subsample
    mlm_scores = {}
    if args.mlm and args.mlm_sample > 0:
        # sample within each of the four groups so every group gets MLM coverage;
        # allocate the budget proportional to group size but with a floor.
        groups = {"a": A, "b": B, "c": C, "d": D}
        budget = args.mlm_sample
        keys = []
        # proportional allocation with a per-group floor of min(len, 150)
        floors = {g: min(len(v), 150) for g, v in groups.items()}
        base = sum(floors.values())
        for g, v in groups.items():
            if not v:
                continue
            take = floors[g]
            take = min(take, len(v))
            idx = rng.choice(len(v), size=take, replace=False)
            for j in idx:
                keys.append((v[j]["di"], v[j]["i"]))
        # de-dup (b and d overlap)
        keys = list(dict.fromkeys(keys))
        if len(keys) > budget:
            sel = rng.choice(len(keys), size=budget, replace=False)
            keys = [keys[j] for j in sel]
        print(f"\n# MLM pseudo-surprisal on {len(keys)} sampled positions "
              f"(model={args.mlm}) ...")
        mlm_scores = mlm_surprisal_for_positions(keys, dev, args.mlm, device)

    # ------------------------------------------------------------------ REPORT
    signals = [
        ("frag_here (subwords)",   "frag_here"),
        ("frag_next (subwords)",   "frag_next"),
        ("frag_max  (subwords)",   "frag_max"),
        ("unseen_uni (0/1)",       "unseen_uni"),
        ("unseen_bi  (0/1)",       "unseen_bi"),
        ("neg_log_uni_freq",       "neg_log_uni"),
        ("neg_log_bigram_freq",    "neg_log_bi"),
    ]

    print("\n--- MEAN / MEDIAN of each unfamiliarity signal per group ---")
    print("(higher value = LESS familiar for every signal, by construction)")
    for label, key in signals:
        print(f"\n  {label}")
        for gname, G in [("(a) conf-FP  ", A), ("(b) true-bound", B),
                         ("(c) true-nonb", C), ("(d) conf-FN  ", D)]:
            print("    " + summarize(gname, [r[key] for r in G]))

    # MLM group summaries
    if mlm_scores:
        print("\n  MLM pseudo-surprisal (nats, subsample)")
        def mlm_vals(G):
            return [mlm_scores[(r["di"], r["i"])] for r in G
                    if (r["di"], r["i"]) in mlm_scores
                    and not np.isnan(mlm_scores[(r["di"], r["i"])])]
        mlm_by = {}
        for gname, G in [("(a) conf-FP  ", A), ("(b) true-bound", B),
                         ("(c) true-nonb", C), ("(d) conf-FN  ", D)]:
            v = mlm_vals(G)
            mlm_by[gname.strip()] = v
            print("    " + summarize(gname, v))

    # --------------------------------------------------- ROC-AUC a-vs-b (KEY)
    print("\n--- ROC-AUC: does the signal SEPARATE (a) conf-FP from (b) true boundaries? ---")
    print("AUC = P(signal higher on a conf-FP than on a true boundary).")
    print("  >0.5 => conf-FPs are LESS familiar than real boundaries (gate can help).")
    print("  ~0.5 => no separation; true boundaries are just as unfamiliar (gate collides).")
    print("  <0.5 => true boundaries are MORE unfamiliar (gate would suppress them).")
    print(f"\n  {'signal':26s} {'AUC(a>b)':>9}")
    auc_tbl = {}
    for label, key in signals:
        auc = roc_auc([r[key] for r in A], [r[key] for r in B])
        auc_tbl[key] = auc
        print(f"  {label:26s} {auc:9.3f}")
    if mlm_scores:
        a_m = [mlm_scores[(r["di"], r["i"])] for r in A
               if (r["di"], r["i"]) in mlm_scores and not np.isnan(mlm_scores[(r["di"], r["i"])])]
        b_m = [mlm_scores[(r["di"], r["i"])] for r in B
               if (r["di"], r["i"]) in mlm_scores and not np.isnan(mlm_scores[(r["di"], r["i"])])]
        auc_m = roc_auc(a_m, b_m)
        auc_tbl["mlm"] = auc_m
        print(f"  {'MLM pseudo-surprisal':26s} {auc_m:9.3f}   (subsample n_a={len(a_m)} n_b={len(b_m)})")

    # For reference: AUC separating conf-FP (a) from true-nonboundary (c) — sanity
    print("\n--- (reference) ROC-AUC (a) conf-FP vs (c) true-nonboundary ---")
    for label, key in signals:
        auc = roc_auc([r[key] for r in A], [r[key] for r in C])
        print(f"  {label:26s} {auc:9.3f}")

    # ------------------------------------------------ gate precision/recall
    # Simple decode-time rule family: "require higher confidence to fire when the
    # position is UNFAMILIAR." We already only look at fires at p>=hi (group a).
    # A gate that suppresses a fire iff the position is unfamiliar would remove:
    #   - suppressed conf-FPs  = |{a : unfamiliar}|      (GOOD: FP killed)
    #   - lost true boundaries = |{b : fired AND unfamiliar}| (BAD: recall lost)
    # We must count true boundaries that the model ALSO fired on at p>=hi (only
    # those are at risk from a "suppress a high-confidence fire" gate).
    print("\n--- GATE TRADEOFF: 'suppress a p>=%.2f fire when the position is unfamiliar' ---"
          % args.hi)
    B_fired = [r for r in B if r["p"] >= args.hi]  # true boundaries the model fired on (at risk)
    print(f"true boundaries the model already fires on at p>={args.hi}: {len(B_fired)} "
          f"of {len(B)} ({100*len(B_fired)/max(len(B),1):.1f}%)")
    print(f"confident FPs available to suppress: {len(A)}")
    print(f"\n  {'rule (unfamiliar := ...)':38s} {'FP_suppressed':>13} {'TB_lost':>8} "
          f"{'FP_kill%':>9} {'TB_loss%':>9}")

    def rule_counts(pred):
        fp_supp = sum(1 for r in A if pred(r))
        tb_lost = sum(1 for r in B_fired if pred(r))
        return fp_supp, tb_lost

    rules = [
        ("frag_here >= 2",        lambda r: r["frag_here"] >= 2),
        ("frag_here >= 3",        lambda r: r["frag_here"] >= 3),
        ("frag_next >= 2",        lambda r: r["frag_next"] >= 2),
        ("frag_next >= 3",        lambda r: r["frag_next"] >= 3),
        ("frag_max  >= 3",        lambda r: r["frag_max"] >= 3),
        ("frag_max  >= 4",        lambda r: r["frag_max"] >= 4),
        ("unseen unigram (tok)",  lambda r: r["unseen_uni"] == 1),
        ("unseen bigram (i,i+1)", lambda r: r["unseen_bi"] == 1),
        ("unseen bi AND frag>=2", lambda r: r["unseen_bi"] == 1 and r["frag_max"] >= 2),
        ("unseen bi OR frag>=3",  lambda r: r["unseen_bi"] == 1 or r["frag_max"] >= 3),
    ]
    for name, pred in rules:
        fp_supp, tb_lost = rule_counts(pred)
        print(f"  {name:38s} {fp_supp:13d} {tb_lost:8d} "
              f"{100*fp_supp/max(len(A),1):9.1f} {100*tb_lost/max(len(B_fired),1):9.1f}")

    print("\n# interpretation guide:")
    print("#   A gate is BUILDABLE only if some rule kills a large share of conf-FPs")
    print("#   while losing a tiny share of the true boundaries the model fires on.")
    print("#   If every high-FP-kill rule also loses many true boundaries, the signal")
    print("#   collides with real boundaries and Noor's OOV-conservatism idea is weak.")

    # ----------------------------------------------------- verdict (auto)
    print("\n--- AUTO VERDICT ---")
    key_aucs = {k: auc_tbl[k] for k in ["frag_max", "unseen_bi", "neg_log_bi"] if k in auc_tbl}
    best_key = max(key_aucs, key=lambda k: key_aucs[k]) if key_aucs else None
    best_auc = key_aucs.get(best_key, float("nan")) if best_key else float("nan")
    print(f"best structural a-vs-b AUC among {{frag_max, unseen_bi, neg_log_bi}}: "
          f"{best_key}={best_auc:.3f}")
    if "mlm" in auc_tbl:
        print(f"MLM a-vs-b AUC: {auc_tbl['mlm']:.3f}")
    print("(An AUC near 0.5 on the strongest signal => the signal does NOT separate")
    print(" confident FPs from real boundaries => decode-time unfamiliarity gate is weak.)")
    print("\n# done")


if __name__ == "__main__":
    main()
