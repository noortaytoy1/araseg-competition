"""Generate test-split submission CSVs for all 4 tasks using the dev-frozen
thresholds (no tuning on test), then score them against the open-test gold
as a preview of the CodaBench dev-phase leaderboard.

  python gen_test_subs.py
"""
from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baselines import write_submission
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from eval_local import compute_metrics
from predict import predict_doc

# (task, model dir, dev-frozen threshold, or-par) — see EXPERIMENTS.md
CONFIGS = [
    ("PA", "runs/pa", 0.55, True),
    ("NoPnx-PA", "runs/nopnx-pa", 0.55, True),
    ("NP", "runs/np", 0.50, False),
    ("NoPnx-NP", "runs/nopnx-np", 0.70, False),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

for task, model_dir, thr, or_par in CONFIGS:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
    docs = load_jsonl(f"data/{task}_test.jsonl")
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tokenizer, model, device, 180, 60, 512)
        lab = (p >= thr).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
                if or_par and i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                    lab[i - 1] = 1
        preds.append(lab.tolist())
    out = f"subs/{task}_test_model.csv"
    write_submission(docs, preds, out)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    m = compute_metrics(gold, {d["doc_id"]: p for d, p in zip(docs, preds)})
    print(f"{task:9s} thr={thr} or_par={or_par}  TEST  "
          f"P={m['macro_precision']*100:.2f} R={m['macro_recall']*100:.2f} "
          f"F1={m['macro_f1']*100:.2f}  -> {out}")
    del model
    torch.cuda.empty_cache()
