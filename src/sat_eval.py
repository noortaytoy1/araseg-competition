"""Zero-shot SaT (wtpsplit) evaluation on AraSeg dev: map SaT's per-character
boundary probabilities to whitespace-token boundaries, sweep a threshold,
and (optionally) cache token-level probs for ensembling. Open-track tool.

Requires `wtpsplit` (torch>=2.6) — run in a separate env from the training one:

  python sat_eval.py --task NoPnx-NP --split dev \
      --model sat-12l-sm --out probs/NoPnx-NP_dev_sat.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from data import PARAGRAPH_TOKEN, load_jsonl
from eval_local import compute_metrics


def token_probs(sat, tokens, block=400):
    """P(boundary after token) = SaT char prob at each token's last char."""
    words = [t for t in tokens]  # \n stays a literal token (PA variants)
    text = " ".join(w if w != PARAGRAPH_TOKEN else "\n" for w in words)
    probs = np.asarray(sat.predict_proba(text))
    # token end positions in text
    out = np.zeros(len(words))
    pos = 0
    for i, w in enumerate(words):
        w_str = w if w != PARAGRAPH_TOKEN else "\n"
        end = pos + len(w_str) - 1
        lo, hi = end, min(end + 1, len(probs) - 1)  # last char or the space after
        out[i] = float(probs[lo:hi + 1].max())
        pos = end + 2  # +1 for the joining space
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="sat-12l-sm")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.025, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--out", default=None, help="optional .npz prob cache")
    args = ap.parse_args()

    from wtpsplit import SaT

    sat = SaT(args.model)
    try:
        sat.half().to("cuda")
    except Exception:
        pass

    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    probs = {d["doc_id"]: token_probs(sat, d["tokens"]) for d in docs}

    print(f"{'thr':>6} {'P':>7} {'R':>7} {'F1':>7}")
    best = (None, -1.0)
    for t in args.thresholds:
        preds = {}
        for d in docs:
            lab = (probs[d["doc_id"]] >= t).astype(int)
            for i, w in enumerate(d["tokens"]):
                if w == PARAGRAPH_TOKEN:
                    lab[i] = 0
            lab[-1] = 1 if d["tokens"][-1] != PARAGRAPH_TOKEN else lab[-1]
            preds[d["doc_id"]] = lab.tolist()
        m = compute_metrics(gold, preds)
        print(f"{t:6.3f} {m['macro_precision']*100:7.2f} {m['macro_recall']*100:7.2f} "
              f"{m['macro_f1']*100:7.2f}")
        if m["macro_f1"] > best[1]:
            best = (t, m["macro_f1"])
    print(f"\nSAT zero-shot {args.task}: best thr={best[0]} F1={best[1]*100:.2f}")

    if args.out:
        np.savez(args.out, **probs)
        print(f"cached -> {args.out}")


if __name__ == "__main__":
    main()
