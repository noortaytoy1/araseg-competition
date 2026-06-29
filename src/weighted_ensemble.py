"""Greedy per-voter weight search for the ensemble (dev = model selection,
treated like threshold tuning: a handful of scalars, no learned model).

Loads the 6 individual voter probs, greedily searches integer weights in
[0..3], decodes with the frozen DP config, keeps the weight vector that
maximizes dev macro-F1. Compares to uniform averaging.

  python weighted_ensemble.py --task NoPnx-NP --lam 0.4 --bias 0.75
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from data import load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from eval_local import compute_metrics

VOTERS = ["", "-s1", "-s2", "-s3", "-arbertv2", "-araelectra"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--bias", type=float, default=0.0)
    args = ap.parse_args()

    low = args.task.lower()
    docs = load_jsonl(f"data/{args.task}_dev.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    len_lp = fit_length_logprob(load_jsonl(f"data/{args.task}_train.jsonl"))

    # per-voter cached probs
    voter_probs = []
    for v in VOTERS:
        p = np.load(f"probs/{args.task}_dev_voter{v}.npz")
        voter_probs.append(p)
    ids = [d["doc_id"] for d in docs]

    def score(weights):
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        preds = {}
        for d in docs:
            did = d["doc_id"]
            blended = sum(wi * vp[did] for wi, vp in zip(w, voter_probs))
            preds[did] = decode_doc(d["tokens"], blended, len_lp, args.lam, args.bias)
        return compute_metrics(gold, preds)["macro_f1"]

    uniform = score([1] * 6)
    print(f"uniform: {uniform*100:.2f}")

    # greedy: start uniform, try bumping each voter's weight
    best_w = [1] * 6
    best = uniform
    for _ in range(6):
        improved = False
        for i in range(6):
            for delta in (1, -1):
                w = list(best_w)
                w[i] += delta
                if w[i] < 0 or sum(w) == 0:
                    continue
                s = score(w)
                if s > best + 1e-5:
                    best, best_w, improved = s, w, True
        if not improved:
            break
    print(f"greedy weights {best_w}: {best*100:.2f}  (uniform {uniform*100:.2f})")


if __name__ == "__main__":
    main()
