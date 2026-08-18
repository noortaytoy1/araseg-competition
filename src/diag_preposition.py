"""Preposition-error DIAGNOSTIC for an AraSeg NoPnx-NP single-encoder checkpoint.

Answers three questions about the model's boundary errors, grounded only in
measured per-token P(boundary):

  1. PREPOSITION SHARE OF ERROR  - of all false positives / false negatives at a
     threshold, what fraction sit where the NEXT token is a preposition/connective?
     What fraction of errors are NOT preposition-adjacent?  (is it PURELY prepositions?)
  2. CONFIDENCE PROFILE          - at positions whose next token is a preposition,
     is P(boundary) piled near 0.5 (INDECISION) or confidently high (OVER-FIRING)?
     Compared against non-preposition positions.
  3. SEEN vs UNSEEN CONTEXT      - split preposition-adjacent FALSE POSITIVES by
     whether the local bigram (prev,prep) and/or (prep,next) was seen in the 174
     train docs.  FP rate for seen- vs unseen-context prepositions.
  4. PER-GENRE (cheap)           - legal / hadith / genealogy / cloze / general.

This does NOT train anything.  It loads an existing checkpoint and predicts
per-token P(boundary) on the dev split using the exact windowing in predict.py.

Usage:
  PYTHONIOENCODING=utf-8 python src/diag_preposition.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev data/NoPnx-NP_dev.jsonl \
      --train data/NoPnx-NP_train.jsonl \
      --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc
from genre_buckets import bucket

# Preposition / connective set given by the diagnostic request (verbatim).
PREPOSITIONS = ["و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
                "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى"]
PREP_SET = set(PREPOSITIONS)


def collect_probs(docs, tokenizer, model, device, window, overlap, max_length):
    """Return per-doc arrays of P(boundary) aligned to d['tokens']."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tokenizer, model, device, window, overlap, max_length)
        out.append(p)
    return out


def build_train_bigrams(train_docs):
    """Set of adjacent (tok[i], tok[i+1]) token bigrams seen in train."""
    bg = set()
    for d in train_docs:
        toks = d["tokens"]
        for i in range(len(toks) - 1):
            bg.add((toks[i], toks[i + 1]))
    return bg


def hist_str(vals, edges):
    """Compact text histogram over the given bin edges."""
    vals = np.asarray(vals)
    if vals.size == 0:
        return "(no positions)"
    counts, _ = np.histogram(vals, bins=edges)
    parts = []
    for j in range(len(edges) - 1):
        parts.append(f"[{edges[j]:.1f},{edges[j+1]:.1f}):{counts[j]}")
    return "  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device={device}  model={args.model}  threshold={args.threshold}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    dev = load_jsonl(args.dev)
    train = load_jsonl(args.train)
    print(f"# dev docs={len(dev)}  train docs={len(train)}")

    probs = collect_probs(dev, tokenizer, model, device,
                          args.window, args.overlap, args.max_length)
    train_bg = build_train_bigrams(train)
    print(f"# distinct train token-bigrams={len(train_bg)}")

    thr = args.threshold

    # ---- token-level accumulators -------------------------------------------
    # A "position" i has: gold label g_i (1=boundary follows tok[i]), prob p_i,
    # and next token tok[i+1].  We classify positions by whether next token is a
    # preposition.  Positions where tok[i] == PARAGRAPH_TOKEN are excluded (the
    # model/eval force them to 0), matching predict.py.
    n_pos = 0
    n_prep_next = 0            # positions whose NEXT token is a preposition
    fp_total = 0
    fn_total = 0
    fp_prep = 0               # FP where next token is a preposition
    fn_prep = 0               # FN where next token is a preposition

    prep_probs = []           # P(boundary) at prep-next positions
    nonprep_probs = []        # P(boundary) at non-prep positions

    # per-genre over/under firing
    genre_stats = defaultdict(lambda: dict(pos=0, gold=0, pred=0, fp=0, fn=0,
                                           fp_prep=0, fn_prep=0))

    # SEEN vs UNSEEN for prep-adjacent false positives
    # context bigrams: left=(prev,prep), right=(prep,next)
    seen_bucket = defaultdict(lambda: dict(prep_pos=0, fp=0))  # key in {both,left,right,neither}
    # also need: prep_pos count per seen/unseen to get an FP RATE (fp / prep positions)

    # which prepositions dominate FPs
    fp_prep_tokens = Counter()

    for d, p in zip(dev, probs):
        toks = d["tokens"]
        gold = d["labels"]
        g_bucket = bucket(toks)
        n = len(toks)
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue  # forced-0 position, not a real decision
            nxt = toks[i + 1] if i + 1 < n else None
            is_prep_next = nxt in PREP_SET
            pi = float(p[i])
            gi = int(gold[i])
            pred = 1 if pi >= thr else 0

            n_pos += 1
            gs = genre_stats[g_bucket]
            gs["pos"] += 1
            gs["gold"] += gi
            gs["pred"] += pred

            if is_prep_next:
                n_prep_next += 1
                prep_probs.append(pi)
            else:
                nonprep_probs.append(pi)

            fp = (pred == 1 and gi == 0)
            fn = (pred == 0 and gi == 1)
            if fp:
                fp_total += 1
                gs["fp"] += 1
            if fn:
                fn_total += 1
                gs["fn"] += 1

            if is_prep_next:
                if fp:
                    fp_prep += 1
                    gs["fp_prep"] += 1
                    fp_prep_tokens[nxt] += 1
                if fn:
                    fn_prep += 1
                    gs["fn_prep"] += 1

                # SEEN/UNSEEN classification of the local context for prep positions
                prev_tok = toks[i - 1] if i - 1 >= 0 else None
                left_seen = (prev_tok is not None) and ((prev_tok, nxt) in train_bg)
                right_next = toks[i + 2] if i + 2 < n else None
                right_seen = (right_next is not None) and ((nxt, right_next) in train_bg)
                if left_seen and right_seen:
                    key = "both_seen"
                elif left_seen:
                    key = "left_seen_only"
                elif right_seen:
                    key = "right_seen_only"
                else:
                    key = "neither_seen"
                seen_bucket[key]["prep_pos"] += 1
                if fp:
                    seen_bucket[key]["fp"] += 1

    # ------------------------------------------------------------------ REPORT
    print("\n================= AraSeg NoPnx-NP PREPOSITION DIAGNOSTIC =================")
    print(f"total decision positions (excl. [PAR]) = {n_pos}")
    print(f"positions whose NEXT token is a preposition = {n_prep_next} "
          f"({100*n_prep_next/max(n_pos,1):.1f}% of positions)")

    # ---- 1. PREPOSITION SHARE OF ERROR --------------------------------------
    print("\n--- 1. PREPOSITION SHARE OF ERROR (threshold %.2f) ---" % thr)
    print(f"FALSE POSITIVES total = {fp_total}")
    if fp_total:
        print(f"   FP with next-token=preposition = {fp_prep} "
              f"({100*fp_prep/fp_total:.1f}% of all FP)")
        print(f"   FP NOT preposition-adjacent    = {fp_total-fp_prep} "
              f"({100*(fp_total-fp_prep)/fp_total:.1f}% of all FP)")
    print(f"FALSE NEGATIVES total = {fn_total}")
    if fn_total:
        print(f"   FN with next-token=preposition = {fn_prep} "
              f"({100*fn_prep/fn_total:.1f}% of all FN)")
        print(f"   FN NOT preposition-adjacent    = {fn_total-fn_prep} "
              f"({100*(fn_total-fn_prep)/fn_total:.1f}% of all FN)")
    tot_err = fp_total + fn_total
    err_prep = fp_prep + fn_prep
    if tot_err:
        print(f"ALL ERRORS = {tot_err}; preposition-adjacent = {err_prep} "
              f"({100*err_prep/tot_err:.1f}%); NOT prep-adjacent = {tot_err-err_prep} "
              f"({100*(tot_err-err_prep)/tot_err:.1f}%)")
    print("top prepositions among FP (next-token):")
    for tok, c in fp_prep_tokens.most_common(15):
        print(f"    {tok!r:12s} {c}")

    # ---- 2. CONFIDENCE PROFILE ----------------------------------------------
    print("\n--- 2. CONFIDENCE PROFILE: P(boundary) at prep-next vs non-prep positions ---")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    pp = np.asarray(prep_probs)
    npp = np.asarray(nonprep_probs)

    def frac_in(a, lo, hi):
        return float(((a >= lo) & (a < hi)).mean()) if a.size else float("nan")

    for name, a in [("prep-next    ", pp), ("non-prep     ", npp)]:
        if a.size == 0:
            print(f"{name}: (none)")
            continue
        print(f"{name}: n={a.size}  mean={a.mean():.3f}  median={np.median(a):.3f}  "
              f"frac[0.4,0.6)={frac_in(a,0.4,0.6):.3f}  "
              f"frac>=0.5={float((a>=0.5).mean()):.3f}  "
              f"frac>=0.8={float((a>=0.8).mean()):.3f}  "
              f"frac<0.2={float((a<0.2).mean()):.3f}")
    print("prep-next histogram:      " + hist_str(pp, edges))
    print("non-prep histogram:       " + hist_str(npp, edges))
    # Same but restricted to prep-next positions that are actually FP (gold 0)
    fp_prep_probs = []
    for d, p in zip(dev, probs):
        toks, gold, n = d["tokens"], d["labels"], len(d["tokens"])
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            nxt = toks[i + 1] if i + 1 < n else None
            if nxt in PREP_SET and int(gold[i]) == 0 and float(p[i]) >= thr:
                fp_prep_probs.append(float(p[i]))
    fpp = np.asarray(fp_prep_probs)
    if fpp.size:
        print(f"prep-next FALSE-POSITIVE fires: n={fpp.size}  mean={fpp.mean():.3f}  "
              f"median={np.median(fpp):.3f}  frac in[0.5,0.6)={frac_in(fpp,0.5,0.6):.3f}  "
              f"frac>=0.8={float((fpp>=0.8).mean()):.3f}")
        print("prep-next FP-fire histogram: " + hist_str(fpp, edges))

    # ---- 3. SEEN vs UNSEEN ---------------------------------------------------
    print("\n--- 3. SEEN vs UNSEEN local context for PREPOSITION-adjacent positions ---")
    print("context bigrams: left=(prev_tok, prep), right=(prep, next_tok); "
          "'seen' = present in the train token-bigram set")
    print(f"{'context':18s} {'prep_pos':>9} {'FP':>6} {'FP_rate':>8}")
    order = ["both_seen", "left_seen_only", "right_seen_only", "neither_seen"]
    agg_seen = dict(prep_pos=0, fp=0)   # any side seen
    agg_unseen = dict(prep_pos=0, fp=0) # neither seen
    for k in order:
        v = seen_bucket.get(k, {"prep_pos": 0, "fp": 0})
        rate = v["fp"] / v["prep_pos"] if v["prep_pos"] else float("nan")
        print(f"{k:18s} {v['prep_pos']:9d} {v['fp']:6d} {rate:8.3f}")
        if k == "neither_seen":
            agg_unseen["prep_pos"] += v["prep_pos"]; agg_unseen["fp"] += v["fp"]
        else:
            agg_seen["prep_pos"] += v["prep_pos"]; agg_seen["fp"] += v["fp"]
    rs = agg_seen["fp"]/agg_seen["prep_pos"] if agg_seen["prep_pos"] else float("nan")
    ru = agg_unseen["fp"]/agg_unseen["prep_pos"] if agg_unseen["prep_pos"] else float("nan")
    print(f"{'SEEN (any side)':18s} {agg_seen['prep_pos']:9d} {agg_seen['fp']:6d} {rs:8.3f}")
    print(f"{'UNSEEN (neither)':18s} {agg_unseen['prep_pos']:9d} {agg_unseen['fp']:6d} {ru:8.3f}")

    # ---- 4. PER-GENRE --------------------------------------------------------
    print("\n--- 4. PER-GENRE over/under-firing ---")
    print(f"{'genre':20s} {'pos':>7} {'gold_b':>7} {'pred_b':>7} {'FP':>6} {'FN':>6} "
          f"{'FP/pos':>7} {'FN/pos':>7} {'FPprep':>7} {'FNprep':>7}")
    for g, s in sorted(genre_stats.items(), key=lambda kv: -kv[1]["pos"]):
        pos = max(s["pos"], 1)
        print(f"{g:20s} {s['pos']:7d} {s['gold']:7d} {s['pred']:7d} {s['fp']:6d} "
              f"{s['fn']:6d} {s['fp']/pos:7.4f} {s['fn']/pos:7.4f} "
              f"{s['fp_prep']:7d} {s['fn_prep']:7d}")

    print("\n# done")


if __name__ == "__main__":
    main()
