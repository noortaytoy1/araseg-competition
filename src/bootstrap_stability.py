"""Bootstrap stability of config selection. For each task, compare candidate
prediction CSVs by resampling documents (with replacement) B times and asking
how often each candidate wins on macro-F1. Tied win% (~50%) => the dev choice
is noise; prefer the lower-variance (fewer dev-fit knobs) config for the blind test.

  python bootstrap_stability.py --gold data/PA_dev.jsonl \
      --cands "3enc=subs/PA_dev_ens.csv" "mega+DP=subs/PA_dev_dp.csv" --B 2000
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from data import load_jsonl
from eval_local import load_predictions


def per_doc_f1(gold, pred):
    ids = list(gold)
    return ids, np.array([
        f1_score(gold[i], pred[i], average="binary", zero_division=0) for i in ids
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cands", nargs="+", required=True, help='name=path.csv ...')
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = load_jsonl(args.gold)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    names, f1s = [], []
    for c in args.cands:
        name, path = c.split("=", 1)
        pred = load_predictions(path)
        ids, f = per_doc_f1(gold, pred)
        names.append(name)
        f1s.append(f)
    F = np.vstack(f1s)  # [n_cands, n_docs]
    n = F.shape[1]
    print(f"gold={args.gold}  n_docs={n}  B={args.B}")
    for nm, f in zip(names, f1s):
        print(f"  {nm:18s} macro-F1 = {f.mean()*100:.2f}")

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.B, n))
    boot = F[:, idx].mean(axis=2)  # [n_cands, B]
    # pairwise win fractions and CI of the difference
    print("\npairwise: P(row beats col) and 95% CI of F1 difference")
    for i in range(len(names)):
        for j in range(len(names)):
            if i >= j:
                continue
            diff = boot[i] - boot[j]
            win = (diff > 0).mean()
            lo, hi = np.percentile(diff * 100, [2.5, 97.5])
            print(f"  {names[i]:14s} vs {names[j]:14s}: "
                  f"P(win)={win*100:5.1f}%  ΔF1 95%CI=[{lo:+.2f},{hi:+.2f}]")


if __name__ == "__main__":
    main()
