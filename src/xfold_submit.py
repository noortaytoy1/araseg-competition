"""Cross-view fold submission (ARMED — do not submit without the organizer ruling).

Folds a PA-track prediction CSV onto a stripped track's token grid via the
verified subsequence alignment (xview_align), producing a submission CSV for
the stripped track. Measured on dev 2026-07-10: PA_base_s42 folded onto
NoPnx-NP dev = 94.39 macro-F1 (vs 84.53 ensemble of record) — +9.86.

LEGALITY GATE: uses another track's test INPUT at inference. Closed-track
training rules do not forbid it explicitly, but it defeats the stripped
tracks' purpose — Noor must obtain the organizer's written ruling BEFORE any
leaderboard use. This script refuses to run without XFOLD_RULING=YES.

Usage:
  XFOLD_RULING=YES python src/xfold_submit.py \
      --pa-pred subs/PA_pool_test.csv --pa-jsonl data/PA_test.jsonl \
      --tgt-jsonl data/NoPnx-NP_test.jsonl --out subs/xfold_NoPnx-NP_test.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xview_align import align_subsequence, fold_labels  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa-pred", required=True, help="PA-track prediction CSV (Document ID, Prediction)")
    ap.add_argument("--pa-jsonl", required=True)
    ap.add_argument("--tgt-jsonl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.environ.get("XFOLD_RULING") != "YES":
        sys.exit("REFUSED: set XFOLD_RULING=YES only after the organizer's written ruling "
                 "that cross-track test inputs may be used at inference.")

    pa_docs = {d["doc_id"]: d for d in load_jsonl(a.pa_jsonl)}
    tgt_docs = load_jsonl(a.tgt_jsonl)
    preds = {}
    with open(a.pa_pred, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[row["Document ID"]] = [int(c) for c in row["Prediction"]]

    n_fail = 0
    rows = []
    for d in tgt_docs:
        twin = pa_docs[d["doc_id"]]
        pred = preds[d["doc_id"]]
        assert len(pred) == len(twin["tokens"]), f"{d['doc_id']}: pred/token length mismatch"
        keep = align_subsequence(twin["tokens"], d["tokens"])
        if keep is None:
            n_fail += 1
            folded = [0] * len(d["tokens"])   # fall back to no boundaries; caller should inspect
        else:
            folded = fold_labels(pred, keep, len(twin["tokens"]))
        folded[-1] = 1
        rows.append((d["doc_id"], "".join(str(x) for x in folded)))

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Document ID", "Prediction"])
        w.writerows(rows)
    print(f"wrote {len(rows)} docs -> {a.out}  (alignment failures: {n_fail})")


if __name__ == "__main__":
    main()
