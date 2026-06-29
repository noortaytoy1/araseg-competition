"""Local evaluation for AraSeg 2026 — mirrors the official scripts/eval.py.

Metric: per-document binary precision/recall/F1 on boundary labels
(sklearn, positive class = 1), macro-averaged over documents.

Differences from the official script (intentional):
  * Can read gold from a local JSONL (offline) as well as from HuggingFace.
  * Prints P and R under the right names. As of June 2026 the official script
    prints macro_recall after the word "Precision" and macro_precision after
    "Recall" (the F1 line is fine) — keep that in mind when comparing outputs.
  * zero_division=0 so empty-prediction docs score 0 instead of erroring.

Usage:
  python eval_local.py --task PA --split dev --predictions subs/PA_dev.csv
  python eval_local.py --gold-jsonl data/PA_dev.jsonl --predictions subs/PA_dev.csv
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from data import TASKS, load_task


def load_predictions(path: str) -> Dict[str, List[int]]:
    df = pd.read_csv(path, header=0, dtype={"Prediction": str})
    return {
        str(row["Document ID"]): [int(c) for c in str(row["Prediction"]).strip()]
        for _, row in df.iterrows()
    }


def compute_metrics(gold: Dict[str, List[int]], pred: Dict[str, List[int]]) -> dict:
    missing = set(gold) - set(pred)
    extra = set(pred) - set(gold)
    if missing or extra:
        raise SystemExit(f"Doc ID mismatch. missing={sorted(missing)[:5]}... extra={sorted(extra)[:5]}...")
    per_doc = {}
    for doc_id, g in gold.items():
        p = pred[doc_id]
        if len(g) != len(p):
            raise SystemExit(f"{doc_id}: gold has {len(g)} tokens but prediction has {len(p)}")
        per_doc[doc_id] = (
            precision_score(g, p, average="binary", zero_division=0),
            recall_score(g, p, average="binary", zero_division=0),
            f1_score(g, p, average="binary", zero_division=0),
        )
    arr = np.array(list(per_doc.values()))
    return {
        "macro_precision": float(arr[:, 0].mean()),
        "macro_recall": float(arr[:, 1].mean()),
        "macro_f1": float(arr[:, 2].mean()),
        "per_doc": per_doc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--gold-jsonl", default=None, help="local gold JSONL (skips HuggingFace)")
    ap.add_argument("--show-worst", type=int, default=0, help="print the N worst documents by F1")
    args = ap.parse_args()

    if not args.gold_jsonl and not args.task:
        ap.error("provide --task (HuggingFace gold) or --gold-jsonl (offline gold)")

    docs = load_task(args.task or "PA", args.split, jsonl_path=args.gold_jsonl)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    pred = load_predictions(args.predictions)
    m = compute_metrics(gold, pred)

    print("-" * 20)
    print(f"Predictions: {args.predictions}")
    print(f"Gold:        {args.gold_jsonl or (TASKS[args.task] + ':' + args.split)}")
    print("-" * 20)
    print(f"Precision: {m['macro_precision'] * 100:.2f}")
    print(f"Recall:    {m['macro_recall'] * 100:.2f}")
    print(f"F1:        {m['macro_f1'] * 100:.2f}")
    print("-" * 20)

    if args.show_worst:
        worst = sorted(m["per_doc"].items(), key=lambda kv: kv[1][2])[: args.show_worst]
        for doc_id, (p, r, f) in worst:
            print(f"{doc_id}  P={p * 100:5.1f} R={r * 100:5.1f} F1={f * 100:5.1f}")


if __name__ == "__main__":
    main()
