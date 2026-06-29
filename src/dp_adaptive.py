"""Two-pass document-adaptive length-prior decoding.

Pass 1: DP decode with the corpus-global train prior (current frozen system).
Pass 2: estimate THIS document's length distribution from pass-1 segments,
blend it with the global prior, re-decode with a per-doc prior.

  python dp_adaptive.py --task NoPnx-NP --probs probs/NoPnx-NP_dev_mega.npz \
      --gold data/NoPnx-NP_dev.jsonl --train data/NoPnx-NP_train.jsonl \
      --lam1 0.4 --bias 0.75 --lam2s 0.4 0.8 1.2 --blends 0.3 0.5 0.7 \
      --out subs/NoPnx-NP_dev_adp.csv
"""
from __future__ import annotations

import argparse

import numpy as np

from baselines import write_submission
from data import PARAGRAPH_TOKEN, load_jsonl
from dp_decode import LMAX, decode_doc, fit_length_logprob
from eval_local import compute_metrics


def doc_length_logprob(tokens, labels, global_lp, blend: float) -> np.ndarray:
    """Blend an empirical per-doc length histogram with the global prior."""
    counts = np.zeros(LMAX + 1)
    run = 0
    for tok, lab in zip(tokens, labels):
        if tok == PARAGRAPH_TOKEN:
            continue
        run += 1
        if lab == 1:
            counts[min(run, LMAX)] += 1
            run = 0
    if counts.sum() < 3:  # too few segments to trust
        return global_lp
    # kernel-smooth the per-doc histogram (triangular, width 2)
    sm = np.zeros_like(counts)
    for off, w in [(-2, 0.15), (-1, 0.5), (0, 1.0), (1, 0.5), (2, 0.15)]:
        sm[max(0, off):LMAX + 1 + min(0, off)] += w * counts[max(0, -off):LMAX + 1 - max(0, off)]
    sm += 1e-3
    doc_lp = np.log(sm / sm.sum())
    return blend * doc_lp + (1 - blend) * global_lp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--lam1", type=float, required=True, help="pass-1 lam (frozen)")
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--lam2s", type=float, nargs="+", default=[0.4, 0.8, 1.2])
    ap.add_argument("--blends", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs = load_jsonl(args.gold)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    probs = np.load(args.probs)
    global_lp = fit_length_logprob(load_jsonl(args.train))

    pass1 = {d["doc_id"]: decode_doc(d["tokens"], probs[d["doc_id"]], global_lp,
                                     args.lam1, args.bias) for d in docs}
    m1 = compute_metrics(gold, pass1)
    print(f"pass1 (lam={args.lam1}, bias={args.bias}): F1={m1['macro_f1']*100:.2f}")

    best = (None, None, m1["macro_f1"], pass1)
    print(f"{'lam2':>5} {'blend':>6} {'P':>7} {'R':>7} {'F1':>7}")
    for lam2 in args.lam2s:
        for blend in args.blends:
            preds = {}
            for d in docs:
                doc_lp = doc_length_logprob(d["tokens"], pass1[d["doc_id"]],
                                            global_lp, blend)
                preds[d["doc_id"]] = decode_doc(d["tokens"], probs[d["doc_id"]],
                                                doc_lp, lam2, args.bias)
            m = compute_metrics(gold, preds)
            print(f"{lam2:5.2f} {blend:6.2f} {m['macro_precision']*100:7.2f} "
                  f"{m['macro_recall']*100:7.2f} {m['macro_f1']*100:7.2f}")
            if m["macro_f1"] > best[2]:
                best = (lam2, blend, m["macro_f1"], preds)

    lam2, blend, f1, preds = best
    write_submission(docs, [preds[d["doc_id"]] for d in docs], args.out)
    tag = "pass1 kept" if lam2 is None else f"lam2={lam2} blend={blend}"
    print(f"\nADAPTIVE {args.task}: {tag} F1={f1*100:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
