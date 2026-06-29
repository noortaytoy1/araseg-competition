"""Heuristic genre bucketing of AraSeg dev docs (no gold genre labels exist) and
per-genre macro-F1 under a frozen prediction CSV. Buckets are signature-based and
approximate; used only for error-analysis reporting.

  python genre_buckets.py --task NoPnx-NP --pred subs/NoPnx-NP_dev_adp.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from data import load_jsonl
from eval_local import load_predictions

# signature keywords (priority order)
CLOZE = "فراغ"          # fill-in-the-blank placeholder
BEGAT = "ولد"           # "X begat Y" — Biblical genealogy
ARTICLE = "المادة"      # "Article N" — legal / constitutional
ISNAD = ("حدثنا", "أخبرنا", "حدثني", "روى", "رسول")  # hadith transmission


def bucket(tokens):
    c = {t: tokens.count(t) for t in [CLOZE, BEGAT, ARTICLE]}
    if c[CLOZE] >= 3:
        return "cloze/exercise"
    if c[BEGAT] >= 5:
        return "genealogy"
    if c[ARTICLE] >= 3:
        return "legal/constitution"
    if any(tokens.count(m) >= 1 for m in ISNAD) and tokens.count("بن") >= 2:
        return "hadith/isnad"
    return "general prose"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    pred = load_predictions(args.pred)
    by = defaultdict(list)
    for d in docs:
        g, p = d["labels"], pred[d["doc_id"]]
        f = f1_score(g, p, average="binary", zero_division=0)
        rate = sum(g) / max(len(g), 1)
        by[bucket(d["tokens"])].append((f, rate))

    print(f"{'genre':20s} {'docs':>5} {'meanF1':>7} {'b-rate':>7}")
    order = ["general prose", "legal/constitution", "hadith/isnad", "genealogy", "cloze/exercise"]
    for g in order:
        if g not in by:
            continue
        arr = np.array(by[g])
        print(f"{g:20s} {len(arr):5d} {arr[:,0].mean()*100:7.2f} {arr[:,1].mean():7.3f}")
    allf = np.array([f for v in by.values() for f, _ in v])
    print(f"{'ALL':20s} {len(allf):5d} {allf.mean()*100:7.2f}")


if __name__ == "__main__":
    main()
