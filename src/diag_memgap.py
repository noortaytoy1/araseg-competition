"""MEMORIZATION-GAP diagnostic for an AraSeg NoPnx-NP checkpoint (CPU-only).

Question it answers: how much better is the model on documents it was TRAINED
on than on dev documents it never saw?  If train-F1 is near-perfect while dev-F1
is ~83.5, the model has effectively memorized its 174 training documents, and
its confidence machinery was calibrated on that near-perfect world — the
mechanistic root of the measured overconfidence.

Analysis-only and ADDITIVE:
  * loads an existing HF checkpoint READ-ONLY (writes nothing to runs/),
  * reuses predict.predict_doc verbatim (the exact eval windowing:
    window 180, overlap 60, threshold 0.5),
  * scores with the same per-document macro P/R/F1 as eval_local.py
    (sklearn binary scores, positive class = 1, zero_division=0),
  * imports existing modules only; imported by nothing.

It also reports a confidence profile on both samples: what fraction of
positions sit at the confident extremes (P >= 0.9 or P <= 0.1), and the median
predicted probability of the positions the model fires on.

CPU is forced (torch device "cpu"); run with CUDA_VISIBLE_DEVICES="" as well
so nothing can touch a busy GPU.

Usage:
  CUDA_VISIBLE_DEVICES= PYTHONIOENCODING=utf-8 python src/diag_memgap.py \
      --model runs/frontier_battery_2026-07-09_v3/cons_baseline_s42 \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --n-docs 30 --seed 42
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc


def score_sample(docs, tokenizer, model, window, overlap, max_length, threshold):
    """predict_doc per doc -> per-document macro P/R/F1 + confidence profile."""
    per_doc = []
    n_pos_total = 0
    n_extreme = 0
    fire_probs = []          # P(boundary) at positions predicted 1
    fp = fn = tp = 0
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tokenizer, model, "cpu", window, overlap, max_length)
        lab = (p >= threshold).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        g = np.asarray(d["labels"], dtype=int)
        per_doc.append((
            precision_score(g, lab, average="binary", zero_division=0),
            recall_score(g, lab, average="binary", zero_division=0),
            f1_score(g, lab, average="binary", zero_division=0),
        ))
        keep = np.array([w != PARAGRAPH_TOKEN for w in d["tokens"]])
        pk, gk, lk = p[keep], g[keep], lab[keep]
        n_pos_total += len(pk)
        n_extreme += int(((pk >= 0.9) | (pk <= 0.1)).sum())
        fire_probs.extend(pk[lk == 1].tolist())
        tp += int(((lk == 1) & (gk == 1)).sum())
        fp += int(((lk == 1) & (gk == 0)).sum())
        fn += int(((lk == 0) & (gk == 1)).sum())
    arr = np.array(per_doc)
    return {
        "macro_P": float(arr[:, 0].mean()) * 100,
        "macro_R": float(arr[:, 1].mean()) * 100,
        "macro_F1": float(arr[:, 2].mean()) * 100,
        "n_docs": len(docs),
        "n_positions": n_pos_total,
        "pct_extreme": 100.0 * n_extreme / max(n_pos_total, 1),
        "median_fire_p": float(np.median(fire_probs)) if fire_probs else float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
    }


def report(name, m):
    print(f"--- {name} ({m['n_docs']} docs, {m['n_positions']} positions) ---")
    print(f"Precision: {m['macro_P']:.2f}   Recall: {m['macro_R']:.2f}   F1: {m['macro_F1']:.2f}")
    print(f"positions at confident extremes (P>=0.9 or P<=0.1): {m['pct_extreme']:.1f}%")
    print(f"median P(boundary) where the model fires: {m['median_fire_p']:.3f}")
    print(f"pooled counts: TP={m['tp']}  FP={m['fp']}  FN={m['fn']}"
          f"  (FP:FN = {m['fp'] / max(m['fn'], 1):.2f}:1)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF-loadable checkpoint dir (read-only)")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--dev-jsonl", default=None,
                    help="optional: also score a same-size dev sample for contrast")
    ap.add_argument("--n-docs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    torch.set_num_threads(max(torch.get_num_threads(), 4))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to("cpu").eval()

    rng = random.Random(args.seed)
    tr = load_jsonl(args.train_jsonl)
    tr_sample = rng.sample(tr, min(args.n_docs, len(tr)))
    print(f"model: {args.model}")
    print(f"train sample: {len(tr_sample)}/{len(tr)} docs (seed {args.seed}), "
          f"threshold {args.threshold}, window {args.window}, overlap {args.overlap}")
    m_tr = score_sample(tr_sample, tokenizer, model, args.window, args.overlap,
                        args.max_length, args.threshold)
    report("TRAIN sample (docs the model was trained on)", m_tr)

    if args.dev_jsonl:
        dv = load_jsonl(args.dev_jsonl)
        dv_sample = rng.sample(dv, min(args.n_docs, len(dv)))
        m_dv = score_sample(dv_sample, tokenizer, model, args.window, args.overlap,
                            args.max_length, args.threshold)
        report("DEV sample (never-seen docs)", m_dv)
        print(f"=== MEMORIZATION GAP (train F1 - dev-sample F1): "
              f"{m_tr['macro_F1'] - m_dv['macro_F1']:+.2f} F1 points ===")


if __name__ == "__main__":
    main()
