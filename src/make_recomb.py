"""Closed-track data multiplication via sentence recombination.

The paper's Fig 2: NoPnx tracks keep improving with more data. We can't add
external data (closed track), but we CAN multiply boundary examples from the
SAME 174 docs: split every doc into its gold sentences, then build many new
synthetic docs by concatenating random sentences. Every boundary label is
exact (a sentence's last token is a boundary; internal tokens keep their
labels). Only training tokens from the closed set are used.

Usage: python make_recomb.py <task> <multiplier>
"""
import json
import os
import random
import statistics
import sys

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")


def sentences(docs):
    out = []
    for d in docs:
        t, lab = d["tokens"], d["labels"]
        start = 0
        for i, l in enumerate(lab):
            if l == 1:
                out.append((t[start:i + 1], lab[start:i + 1]))
                start = i + 1
        if start < len(t):                      # trailing (rare)
            out.append((t[start:], lab[start:]))
    return out


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "NoPnx-NP"
    mult = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    docs = [json.loads(l) for l in open(os.path.join(DATA, f"{task}_train.jsonl"),
            encoding="utf-8") if l.strip()]
    sents = sentences(docs)
    tgt = int(statistics.median(len(d["tokens"]) for d in docs))
    out = list(docs)                            # keep originals
    for k in range(mult * len(docs)):
        dt, dl = [], []
        while len(dt) < tgt:
            st, sl = random.choice(sents)
            dt += list(st); dl += list(sl)
        out.append({"doc_id": f"recomb_{k}", "tokens": dt, "labels": dl})
    dst = os.path.join(DATA, f"recomb_{task}_train.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # sanity: every synthetic doc ends on a boundary, label set unchanged
    bad = sum(1 for d in out if d["doc_id"].startswith("recomb") and d["labels"][-1] != 1)
    print(f"{len(docs)} orig + {mult}x = {len(out)} docs | {len(sents)} sentence pool "
          f"| target len {tgt} | synth-docs not ending on boundary: {bad} -> {dst}")


if __name__ == "__main__":
    main()
