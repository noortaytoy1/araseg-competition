"""Rule-based baselines for the AraSeg 2026 shared task.

Produces a submission-ready CSV (columns: "Document ID", "Prediction") for any
task/split. Rules can be combined:

  punct   1 on strong sentence-final punctuation tokens (PA / NP variants)
  verse   1 on ")" closing a "( <digits> )" verse/clause marker (Quranic style)
  par     1 on the token immediately BEFORE a "\n" paragraph token (PA variants)
  every-k 1 every k tokens (fallback floor for NoPnx-NP)
  doc-end 1 on the last non-"\n" token of the document (on by default)

"\n" tokens themselves are never predicted as boundaries (gold labels them 0).

Examples:
  python baselines.py --task PA       --split dev --rules punct verse par --out subs/PA_dev.csv
  python baselines.py --task NP       --split dev --rules punct verse     --out subs/NP_dev.csv
  python baselines.py --task NoPnx-PA --split dev --rules par             --out subs/NoPnx-PA_dev.csv
  python baselines.py --task NoPnx-NP --split dev --rules every-k --k 13  --out subs/NoPnx-NP_dev.csv
"""
from __future__ import annotations

import argparse
import os
from typing import List

from data import PARAGRAPH_TOKEN, SENT_FINAL_PUNCT, TASKS, load_task, mean_sentence_length


def predict_doc(tokens: List[str], rules: set, k: int, doc_end: bool) -> List[int]:
    n = len(tokens)
    pred = [0] * n
    since_last = 0
    for i, tok in enumerate(tokens):
        if tok == PARAGRAPH_TOKEN:
            since_last = 0
            continue
        if "punct" in rules and tok in SENT_FINAL_PUNCT:
            pred[i] = 1
        if (
            "verse" in rules
            and tok == ")"
            and i >= 2
            and tokens[i - 1].isdigit()
            and tokens[i - 2] == "("
        ):
            pred[i] = 1
        if "par" in rules and i + 1 < n and tokens[i + 1] == PARAGRAPH_TOKEN:
            pred[i] = 1
        since_last += 1
        if "every-k" in rules and since_last >= k and pred[i] == 0:
            pred[i] = 1
        if pred[i] == 1:
            since_last = 0
    if doc_end:
        for i in range(n - 1, -1, -1):
            if tokens[i] != PARAGRAPH_TOKEN:
                pred[i] = 1
                break
    return pred


def write_submission(docs, preds, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Document ID,Prediction\n")
        for d, p in zip(docs, preds):
            assert len(p) == len(d["tokens"]), f"length mismatch for {d['doc_id']}"
            f.write(f"{d['doc_id']},{''.join(map(str, p))}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--jsonl", default=None, help="optional local JSONL instead of HuggingFace")
    ap.add_argument("--rules", nargs="+", default=["punct", "verse", "par"],
                    choices=["punct", "verse", "par", "every-k"])
    ap.add_argument("--k", type=int, default=None,
                    help="period for every-k (default: mean train sentence length)")
    ap.add_argument("--no-doc-end", action="store_true", help="don't force a boundary at document end")
    ap.add_argument("--out", required=True, help="output submission CSV path")
    args = ap.parse_args()

    docs = load_task(args.task, args.split, jsonl_path=args.jsonl)
    rules = set(args.rules)

    k = args.k
    if "every-k" in rules and k is None:
        try:
            k = max(2, round(mean_sentence_length(load_task(args.task, "train"))))
        except Exception:
            k = 13  # ~ mean AraSeg sentence length
        print(f"[every-k] using k={k}")

    preds = [predict_doc(d["tokens"], rules, k or 13, not args.no_doc_end) for d in docs]
    write_submission(docs, preds, args.out)

    n_pos = sum(sum(p) for p in preds)
    n_tok = sum(len(p) for p in preds)
    print(f"wrote {len(docs)} docs ({n_tok} tokens, {n_pos} predicted boundaries, "
          f"rate={n_pos / max(n_tok, 1):.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
