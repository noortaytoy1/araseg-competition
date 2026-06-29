"""Cross-encoder ensemble: average P(boundary) from several fine-tuned models,
then sweep the decision threshold on a labeled split (dev).

  python ensemble_sweep.py --task PA --jsonl data/PA_dev.jsonl \
      --models runs/pa runs/pa-arbertv2 runs/pa-araelectra \
      --or-par --out subs/PA_dev_ens.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baselines import write_submission
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_jsonl
from eval_local import compute_metrics
from predict import predict_doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9])
    ap.add_argument("--or-par", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = load_jsonl(args.jsonl)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    words_per_doc = [
        [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]] for d in docs
    ]

    sum_probs = [np.zeros(len(d["tokens"])) for d in docs]
    for mdir in args.models:
        tokenizer = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForTokenClassification.from_pretrained(mdir).to(device).eval()
        for i, words in enumerate(words_per_doc):
            sum_probs[i] += predict_doc(words, tokenizer, model, device, 180, 60, 512)
        del model
        torch.cuda.empty_cache()
        print(f"done: {mdir}")
    probs = [s / len(args.models) for s in sum_probs]

    def labels_at(t: float):
        preds = {}
        for d, p in zip(docs, probs):
            lab = (p >= t).astype(int)
            for i, w in enumerate(d["tokens"]):
                if w == PARAGRAPH_TOKEN:
                    lab[i] = 0
                    if args.or_par and i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                        lab[i - 1] = 1
            preds[d["doc_id"]] = lab.tolist()
        return preds

    best = (None, -1.0)
    print(f"{'thr':>5} {'P':>7} {'R':>7} {'F1':>7}")
    for t in args.thresholds:
        m = compute_metrics(gold, labels_at(t))
        print(f"{t:5.2f} {m['macro_precision']*100:7.2f} {m['macro_recall']*100:7.2f} {m['macro_f1']*100:7.2f}")
        if m["macro_f1"] > best[1]:
            best = (t, m["macro_f1"])

    t, f1 = best
    preds = labels_at(t)
    write_submission(docs, [preds[d["doc_id"]] for d in docs], args.out)
    print(f"\nENSEMBLE {args.task}: best threshold={t} macro_f1={f1*100:.2f} "
          f"(models={len(args.models)}, or_par={args.or_par}) -> {args.out}")


if __name__ == "__main__":
    main()
