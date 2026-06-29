"""Build a self-supervised boundary-pretraining corpus from external Arabic
sentences (open-track only). Concatenate sentences into pseudo-documents:
the last whitespace token of each sentence is a gold boundary. For NoPnx
conditions, strip sentence-final punctuation so the model must recover
boundaries from lexical/syntactic cues alone — exactly the AraSeg NoPnx task.

  python build_pretrain.py --sents ext/opus_ar_sents.txt --variant NoPnx-NP \
      --n-docs 8000 --out ext/pretrain_NoPnx-NP.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

# AraSeg strong sentence-final punctuation (mirrors data.SENT_FINAL_PUNCT)
SENT_PUNCT = {".", "!", "?", "؟", "۔", "…", "؞", "،", ",", ";", "؛", ":"}


def strip_final_punct(tokens):
    """Drop trailing punctuation tokens (NoPnx conditions)."""
    return [t for t in tokens if t not in SENT_PUNCT]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sents", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--n-docs", type=int, default=8000)
    ap.add_argument("--min-sents", type=int, default=4)
    ap.add_argument("--max-sents", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.sents, encoding="utf-8") as f:
        sents = [ln.strip() for ln in f if ln.strip()]
    rng.shuffle(sents)
    nopnx = args.variant.startswith("NoPnx")
    par = args.variant in ("PA", "NoPnx-PA")  # paragraph-aware → emit \n tokens

    docs, i = [], 0
    while len(docs) < args.n_docs and i < len(sents):
        k = rng.randint(args.min_sents, args.max_sents)
        chunk = sents[i:i + k]
        i += k
        if len(chunk) < args.min_sents:
            break
        tokens, labels = [], []
        for si, s in enumerate(chunk):
            toks = s.split()
            if nopnx:
                toks = strip_final_punct(toks)
            if not toks:
                continue
            lab = [0] * len(toks)
            lab[-1] = 1  # sentence-final token = boundary
            tokens += toks
            labels += lab
            # PA variants: paragraph break (\n token, label 0) at ~30% of boundaries
            if par and si < len(chunk) - 1 and rng.random() < 0.3:
                tokens.append("\n")
                labels.append(0)
        if sum(labels) >= args.min_sents - 1:
            docs.append({"doc_id": f"ext_{len(docs):06d}", "tokens": tokens,
                         "labels": labels})

    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    tot = sum(len(d["labels"]) for d in docs)
    pos = sum(sum(d["labels"]) for d in docs)
    print(f"wrote {len(docs)} pseudo-docs ({tot} tokens, {pos} boundaries, "
          f"rate={pos/max(tot,1):.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
