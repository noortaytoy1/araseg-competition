"""Cache per-token ensemble P(boundary) to .npz so decoding experiments
(threshold, Viterbi, calibration) run instantly on CPU.

  python cache_probs.py --task PA --split dev \
      --models runs/pa runs/pa-arbertv2 runs/pa-araelectra --out probs/PA_dev.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    words_per_doc = [
        [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]] for d in docs
    ]
    sums = [np.zeros(len(d["tokens"])) for d in docs]
    for mdir in args.models:
        try:  # modernbert @torch.compile-at-import vs this box's broken inductor -> neuter to eager
            from transformers import AutoConfig as _AC
            if getattr(_AC.from_pretrained(mdir), "model_type", "") == "modernbert":
                torch.compile = lambda model=None, *a, **k: (model if callable(model) else (lambda f: f))
        except Exception:
            pass
        tokenizer = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForTokenClassification.from_pretrained(mdir).to(device).eval()
        for i, words in enumerate(words_per_doc):
            sums[i] += predict_doc(words, tokenizer, model, device, 180, 60, 512)
        del model
        torch.cuda.empty_cache()
        print(f"done: {mdir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **{d["doc_id"]: s / len(args.models) for d, s in zip(docs, sums)})
    print(f"cached {len(docs)} docs ({len(args.models)} models) -> {args.out}")


if __name__ == "__main__":
    main()
