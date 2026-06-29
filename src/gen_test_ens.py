"""Generate test-split submissions from the 3-encoder ensembles at the
dev-frozen thresholds (no tuning on test), and score vs open-test gold.

  python gen_test_ens.py
"""
from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baselines import write_submission
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from eval_local import compute_metrics
from predict import predict_doc

# (task, model dirs, dev-frozen threshold, or-par) — see EXPERIMENTS.md
CONFIGS = [
    ("PA", ["runs/pa", "runs/pa-arbertv2", "runs/pa-araelectra"], 0.60, True),
    ("NoPnx-PA", ["runs/nopnx-pa", "runs/nopnx-pa-arbertv2", "runs/nopnx-pa-araelectra"], 0.50, True),
    ("NP", ["runs/np", "runs/np-arbertv2", "runs/np-araelectra"], 0.50, False),
    ("NoPnx-NP", ["runs/nopnx-np", "runs/nopnx-np-arbertv2", "runs/nopnx-np-araelectra"], 0.60, False),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

for task, model_dirs, thr, or_par in CONFIGS:
    docs = load_jsonl(f"data/{task}_test.jsonl")
    words_per_doc = [
        [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]] for d in docs
    ]
    sum_probs = [None] * len(docs)
    for mdir in model_dirs:
        tokenizer = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForTokenClassification.from_pretrained(mdir).to(device).eval()
        for i, words in enumerate(words_per_doc):
            p = predict_doc(words, tokenizer, model, device, 180, 60, 512)
            sum_probs[i] = p if sum_probs[i] is None else sum_probs[i] + p
        del model
        torch.cuda.empty_cache()

    preds = []
    for d, s in zip(docs, sum_probs):
        lab = (s / len(model_dirs) >= thr).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
                if or_par and i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                    lab[i - 1] = 1
        preds.append(lab.tolist())

    out = f"subs/{task}_test_ens.csv"
    write_submission(docs, preds, out)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    m = compute_metrics(gold, {d["doc_id"]: p for d, p in zip(docs, preds)})
    print(f"{task:9s} thr={thr} or_par={or_par}  TEST-ENS  "
          f"P={m['macro_precision']*100:.2f} R={m['macro_recall']*100:.2f} "
          f"F1={m['macro_f1']*100:.2f}  -> {out}")
