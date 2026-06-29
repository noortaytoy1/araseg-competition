"""Boundary-level residual-error analysis for the frozen system.

For each task: compare gold vs prediction on dev, split errors into
  FN = missed boundary (gold 1, pred 0)
  FP = spurious boundary (gold 0, pred 1)
Bucket by the token AT the boundary and the PRECEDING token. Estimate
irreducibility: for each error's (prev, cur) context, look up how often that
same context is a boundary in TRAIN. High train-boundary-rate on a missed
context => systematic (fixable); low/ambiguous => closer to irreducible.

  python residual_errors.py --task NoPnx-NP --pred subs/NoPnx-NP_dev_adp.csv
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from data import PARAGRAPH_TOKEN, load_jsonl
from eval_local import load_predictions


def context_boundary_rate(train_docs):
    """P(boundary | (prev_token, cur_token)) and base counts from train."""
    pos, tot = Counter(), Counter()
    for d in train_docs:
        toks, labs = d["tokens"], d["labels"]
        for i in range(len(toks)):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            prev = toks[i - 1] if i > 0 else "<s>"
            key = (prev, toks[i])
            tot[key] += 1
            if labs[i] == 1:
                pos[key] += 1
    return pos, tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    train = load_jsonl(f"data/{args.task}_train.jsonl")
    pred = load_predictions(args.pred)
    pos, tot = context_boundary_rate(train)

    fn_tok, fp_tok = Counter(), Counter()       # token AT the boundary
    fn_prev, fp_prev = Counter(), Counter()     # preceding token
    fn_sys, fn_amb, fn_unseen = 0, 0, 0         # systematicity of misses
    n_fn = n_fp = n_gold = 0

    for d in docs:
        toks, g = d["tokens"], d["labels"]
        p = pred[d["doc_id"]]
        for i in range(len(toks)):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            prev = toks[i - 1] if i > 0 else "<s>"
            if g[i] == 1:
                n_gold += 1
            if g[i] == 1 and p[i] == 0:          # missed boundary
                n_fn += 1
                fn_tok[toks[i]] += 1
                fn_prev[prev] += 1
                key = (prev, toks[i])
                if tot[key] == 0:
                    fn_unseen += 1
                elif pos[key] / tot[key] >= 0.5:
                    fn_sys += 1                  # context usually IS a boundary -> fixable
                else:
                    fn_amb += 1                  # context rarely a boundary -> ambiguous
            elif g[i] == 0 and p[i] == 1:        # spurious boundary
                n_fp += 1
                fp_tok[toks[i]] += 1
                fp_prev[prev] += 1

    print(f"=== {args.task} ({args.split}) — gold boundaries={n_gold}, FN={n_fn}, FP={n_fp}")
    print(f"Missed-boundary (FN) systematicity:")
    print(f"  systematic (ctx ≥50% boundary in train): {fn_sys:4d} ({100*fn_sys/max(n_fn,1):.0f}%)")
    print(f"  ambiguous  (ctx <50% boundary in train): {fn_amb:4d} ({100*fn_amb/max(n_fn,1):.0f}%)")
    print(f"  unseen ctx (not in train)              : {fn_unseen:4d} ({100*fn_unseen/max(n_fn,1):.0f}%)")
    print(f"top tokens AT missed boundaries (FN):  {fn_tok.most_common(8)}")
    print(f"top tokens AT spurious boundaries (FP): {fp_tok.most_common(8)}")
    print(f"top tokens BEFORE missed boundaries:   {fn_prev.most_common(6)}")


if __name__ == "__main__":
    main()
