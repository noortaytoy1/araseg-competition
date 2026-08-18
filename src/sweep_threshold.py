"""Sweep decision thresholds for a fine-tuned model on a labeled split.

Runs inference ONCE (probabilities per token), then scores every threshold
with the document-level macro metric, optionally OR-ing in the paragraph
rule (token before "\n" -> 1; it is 100%-precise on dev). Writes the best
threshold's submission CSV.

  python sweep_threshold.py --model runs/pa --task PA --jsonl data/PA_dev.jsonl \
      --out subs/PA_dev_model.csv [--or-par]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baselines import write_submission
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from eval_local import compute_metrics
from predict import predict_doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--jsonl", required=True, help="labeled JSONL split to sweep on (dev)")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8])
    ap.add_argument("--or-par", action="store_true",
                    help="also force 1 on the token right before each \\n")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--out", required=True, help="submission CSV for the best threshold")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:  # modernbert decorates a layer with @torch.compile at import; this box's inductor is broken -> neuter
        from transformers import AutoConfig as _AC
        if getattr(_AC.from_pretrained(args.model), "model_type", "") == "modernbert":
            torch.compile = lambda model=None, *a, **k: (model if callable(model) else (lambda f: f))
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    docs = load_task(args.task, "dev", jsonl_path=args.jsonl)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    probs = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        probs.append(predict_doc(words, tokenizer, model, device,
                                 args.window, args.overlap, 512))

    def labels_at(t: float):
        preds = {}
        for d, p in zip(docs, probs):
            lab = (p >= t).astype(int)
            n = len(d["tokens"])
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
    print(f"\nbest threshold={t} macro_f1={f1*100:.2f}  (or_par={args.or_par}) -> {args.out}")


if __name__ == "__main__":
    main()
