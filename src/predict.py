"""Predict sentence boundaries with a fine-tuned encoder and write a
submission CSV ("Document ID","Prediction") for AraSeg 2026.

Every input token (including "\n") gets exactly one 0/1 in the output string,
as the official evaluator requires. "\n" tokens are always 0. Long documents
are handled with overlapping windows whose probabilities are averaged.

Examples:
  python predict.py --model runs/nopnx-np --task NoPnx-NP --split dev \
      --out subs/NoPnx-NP_dev.csv --check-ids path/to/examples/NoPnx-NP_dev.csv
  python predict.py --model runs/pa --task PA --split test --threshold 0.45 \
      --out subs/PA_test.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import List

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baselines import write_submission
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task


@torch.no_grad()
def predict_doc(words: List[str], tokenizer, model, device,
                window: int, overlap: int, max_length: int) -> np.ndarray:
    """Return P(boundary) per word, covering all words via overlapping windows."""
    n = len(words)
    probs = defaultdict(list)
    start = 0
    while start < n:
        chunk = words[start:start + window]
        enc = tokenizer(chunk, is_split_into_words=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        word_ids = enc.word_ids(0)
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
        p_boundary = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        last_seen = -1
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
            if nxt != wid:  # last subword of this word
                probs[start + wid].append(float(p_boundary[pos]))
            last_seen = max(last_seen, wid)

        covered_through = start + last_seen  # absolute index of last covered word
        if covered_through >= n - 1:
            break
        # step forward, keeping `overlap` already-covered words for context
        start = max(covered_through + 1 - overlap, start + 1)

    return np.array([np.mean(probs[i]) if probs[i] else 0.0 for i in range(n)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path or HF id of fine-tuned model")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--jsonl", default=None, help="optional local JSONL instead of HuggingFace")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(boundary) cutoff; tune on dev (lower -> higher recall)")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--force-doc-end", action="store_true",
                    help="force a boundary on the last non-\\n token of each document")
    ap.add_argument("--check-ids", default=None,
                    help="official example CSV for this task/split; verifies IDs and token counts match")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    docs = load_task(args.task, args.split, jsonl_path=args.jsonl)
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tokenizer, model, device,
                        args.window, args.overlap, args.max_length)
        lab = (p >= args.threshold).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        if args.force_doc_end:
            for i in range(len(lab) - 1, -1, -1):
                if d["tokens"][i] != PARAGRAPH_TOKEN:
                    lab[i] = 1
                    break
        preds.append(lab.tolist())

    write_submission(docs, preds, args.out)
    rate = sum(map(sum, preds)) / max(sum(map(len, preds)), 1)
    print(f"wrote {len(docs)} docs (boundary rate={rate:.3f}, threshold={args.threshold}) -> {args.out}")

    if args.check_ids:
        import pandas as pd
        ref = pd.read_csv(args.check_ids, dtype={"Prediction": str})
        ref_lens = {str(r["Document ID"]): len(str(r["Prediction"])) for _, r in ref.iterrows()}
        ours = {d["doc_id"]: len(p) for d, p in zip(docs, preds)}
        assert set(ref_lens) == set(ours), "doc ID set differs from the official example file"
        bad = [k for k in ours if ours[k] != ref_lens[k]]
        assert not bad, f"token-count mismatch for {len(bad)} docs, e.g. {bad[:3]}"
        print(f"check-ids OK: {len(ours)} doc IDs and token counts match {args.check_ids}")


if __name__ == "__main__":
    main()
