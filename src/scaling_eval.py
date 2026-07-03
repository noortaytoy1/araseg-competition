"""Evaluate the data-scaling models (doc-level dev macro-F1 at best dev threshold)
and plot the scaling curve. Run after train_scaling.sh completes.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from eval_local import compute_metrics
from predict import predict_doc

SIZES = [22, 44, 87, 130, 174]
device = "cuda" if torch.cuda.is_available() else "cpu"


def dev_f1(model_dir, task):
    docs = load_jsonl(f"data/{task}_dev.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    for d in docs:
        w = [MODEL_PAR_TOKEN if x == PARAGRAPH_TOKEN else x for x in d["tokens"]]
        probs.append(predict_doc(w, tok, mdl, device, 180, 60, 512))
    del mdl; torch.cuda.empty_cache()
    best = 0.0
    for t in np.linspace(0.3, 0.8, 11):
        preds = {}
        for d, p in zip(docs, probs):
            lab = (p >= t).astype(int)
            for i, x in enumerate(d["tokens"]):
                if x == PARAGRAPH_TOKEN: lab[i] = 0
            preds[d["doc_id"]] = lab.tolist()
        best = max(best, compute_metrics(gold, preds)["macro_f1"])
    return best * 100


def main():
    res = {}
    for task in ["PA", "NoPnx-NP"]:
        res[task] = []
        for n in SIZES:
            md = f"runs/{task.lower()}-n{n}"
            if not os.path.exists(os.path.join(md, "config.json")):
                print(f"missing {md}"); continue
            f = dev_f1(md, task)
            res[task].append((n, f))
            print(f"{task} n={n}: dev F1={f:.2f}")
    json.dump(res, open("scaling/scaling.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for task in res:
        xs = [n for n, _ in res[task]]; ys = [f for _, f in res[task]]
        ax.plot(xs, ys, "o-", label=task)
        # mark whether still rising at the end
        if len(ys) >= 2:
            print(f"{task}: last-step slope = {ys[-1]-ys[-2]:.2f} F1 / "
                  f"{xs[-1]-xs[-2]} docs")
    ax.set_xlabel("training documents"); ax.set_ylabel("dev macro-F1")
    ax.legend(fontsize=8); ax.set_title("Performance vs training-set size")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("paper/figs/scaling.pdf"); plt.close(fig)
    print("scaling.pdf written")


if __name__ == "__main__":
    main()
