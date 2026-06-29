"""Length-prior semi-Markov (Viterbi) decoding over cached boundary probs.

Instead of thresholding each token independently, pick the boundary set that
maximizes  sum[ log p(b_i) ] + sum[ log(1-p) for non-boundaries ]
        + lam * sum[ log P_len(segment length) ]
with P_len fit on the task's TRAIN split (closed-track legal). Hard rules:
"\n" is never a boundary; the token before "\n" and the doc-final token are
forced boundaries (always true in gold).

  python dp_decode.py --task NoPnx-NP --probs probs/NoPnx-NP_dev.npz \
      --gold data/NoPnx-NP_dev.jsonl --train data/NoPnx-NP_train.jsonl \
      --lams 0 0.25 0.5 1.0 1.5 --out subs/NoPnx-NP_dev_dp.csv
"""
from __future__ import annotations

import argparse

import numpy as np

from baselines import write_submission
from data import PARAGRAPH_TOKEN, load_jsonl
from eval_local import compute_metrics

LMAX = 200
EPS = 1e-6


def fit_length_logprob(train_docs) -> np.ndarray:
    """log P(len) over 1..LMAX from train sentence lengths (\\n excluded)."""
    counts = np.ones(LMAX + 1)  # add-one smoothing; index 0 unused
    for d in train_docs:
        run = 0
        for tok, lab in zip(d["tokens"], d["labels"]):
            if tok == PARAGRAPH_TOKEN:
                continue
            run += 1
            if lab == 1:
                counts[min(run, LMAX)] += 1
                run = 0
    return np.log(counts / counts.sum())


def decode_doc(tokens, probs, len_lp, lam: float, bias: float = 0.0):
    """Viterbi over non-\\n positions with forced boundaries before \\n / at end."""
    idx = [i for i, t in enumerate(tokens) if t != PARAGRAPH_TOKEN]
    n = len(idx)
    if n == 0:
        return [0] * len(tokens)
    p = np.clip(probs[idx], EPS, 1 - EPS)
    lp, l1p = np.log(p) + bias, np.log(1 - p)  # bias>0 favors recall
    forced = np.zeros(n, dtype=bool)
    forced[-1] = True
    for j, i in enumerate(idx):
        if i + 1 < len(tokens) and tokens[i + 1] == PARAGRAPH_TOKEN:
            forced[j] = True

    NEG = -1e18
    # prefix sums of log(1-p) for O(1) segment interiors
    c1 = np.concatenate([[0.0], np.cumsum(l1p)])
    # best[j] = best score of a prefix whose last boundary is at position j-1;
    # segments may not cross a forced boundary position
    best = np.full(n + 1, NEG)
    back = np.zeros(n + 1, dtype=int)
    best[0] = 0.0
    forced_pos = np.where(forced)[0]
    for j in range(1, n + 1):
        lo = max(0, j - LMAX)
        # segment (k, j-1]: may not contain a forced position other than j-1
        prev_f = forced_pos[forced_pos < j - 1]
        if len(prev_f):
            lo = max(lo, int(prev_f[-1]) + 1)
        if lo >= j:
            continue
        ks = np.arange(lo, j)
        seg = best[ks] + (c1[j - 1] - c1[ks]) + lp[j - 1] + lam * len_lp[np.minimum(j - ks, LMAX)]
        k = int(np.argmax(seg))
        best[j], back[j] = seg[k], ks[k]
    # backtrace
    lab_n = np.zeros(n, dtype=int)
    j = n
    while j > 0:
        lab_n[j - 1] = 1
        j = back[j]
    out = [0] * len(tokens)
    for j, i in enumerate(idx):
        out[i] = int(lab_n[j])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 1.5])
    ap.add_argument("--biases", type=float, nargs="+", default=[0.0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs = load_jsonl(args.gold)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    probs = np.load(args.probs)
    len_lp = fit_length_logprob(load_jsonl(args.train))

    best = (None, None, -1.0, None)
    print(f"{'lam':>5} {'bias':>5} {'P':>7} {'R':>7} {'F1':>7}")
    for lam in args.lams:
        for bias in args.biases:
            preds = {d["doc_id"]: decode_doc(d["tokens"], probs[d["doc_id"]], len_lp, lam, bias)
                     for d in docs}
            m = compute_metrics(gold, preds)
            print(f"{lam:5.2f} {bias:5.2f} {m['macro_precision']*100:7.2f} "
                  f"{m['macro_recall']*100:7.2f} {m['macro_f1']*100:7.2f}")
            if m["macro_f1"] > best[2]:
                best = (lam, bias, m["macro_f1"], preds)

    lam, bias, f1, preds = best
    write_submission(docs, [preds[d["doc_id"]] for d in docs], args.out)
    print(f"\nDP {args.task}: best lam={lam} bias={bias} macro_f1={f1*100:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
