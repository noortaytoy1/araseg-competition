"""Corruption augmentation for the NoPnx tasks (closed-track: derived solely
from AraSeg train).

NoPnx variants are deletion-only subsequences of PA. We recover the per-doc
deletion mask by alignment, verify our label-movement rule reproduces the
official NoPnx labels EXACTLY (p=0), then synthesize intermediate corruption
levels by re-inserting each deleted token with probability p.

  python augment.py --target NoPnx-PA --ps 0.2 0.4 0.6 --copies 1 \
      --out data/NoPnx-PA_train_aug.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from data import PARAGRAPH_TOKEN, load_jsonl


def deletion_mask(full: list, sub: list) -> list[bool]:
    """keep[i]=True iff full[i] survives into sub (greedy, deletion-only)."""
    keep = [False] * len(full)
    j = 0
    for i, tok in enumerate(full):
        if j < len(sub) and tok == sub[j]:
            keep[i] = True
            j += 1
    assert j == len(sub), "sub is not a subsequence of full"
    return keep


def project_labels(tokens, labels, keep):
    """Labels for the kept subsequence; a deleted 1 moves to the nearest
    preceding kept non-\\n token (else the next kept non-\\n token)."""
    kept_idx = [i for i in range(len(tokens)) if keep[i]]
    pos_of = {i: k for k, i in enumerate(kept_idx)}
    out = [0] * len(kept_idx)
    for i, lab in enumerate(labels):
        if lab != 1:
            continue
        if keep[i]:
            out[pos_of[i]] = 1
            continue
        j = i - 1
        while j >= 0 and not (keep[j] and tokens[j] != PARAGRAPH_TOKEN):
            j -= 1
        if j < 0:
            j = i + 1
            while j < len(tokens) and not (keep[j] and tokens[j] != PARAGRAPH_TOKEN):
                j += 1
            if j >= len(tokens):
                continue
        out[pos_of[j]] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=["NoPnx-PA", "NoPnx-NP"])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.2, 0.4, 0.6],
                    help="probability of re-inserting each deleted token")
    ap.add_argument("--copies", type=int, default=1, help="copies per p value")
    ap.add_argument("--extra-clean", type=int, default=1,
                    help="extra copies of the unmodified target train set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pa = load_jsonl("data/PA_train.jsonl")
    tgt = {d["doc_id"]: d for d in load_jsonl(f"data/{args.target}_train.jsonl")}

    # 1) recover masks + verify our projection reproduces official labels
    masks = {}
    for d in pa:
        t = tgt[d["doc_id"]]
        keep = deletion_mask(d["tokens"], t["tokens"])
        proj = project_labels(d["tokens"], d["labels"], keep)
        assert proj == list(t["labels"]), f"label projection mismatch in {d['doc_id']}"
        masks[d["doc_id"]] = keep
    print(f"verification OK: p=0 reproduces {args.target} train labels exactly "
          f"({len(pa)} docs)")

    # 2) synthesize
    out = []
    for _ in range(args.extra_clean):
        out.extend(tgt[d["doc_id"]] for d in pa)
    for p in args.ps:
        for c in range(args.copies):
            for d in pa:
                keep = list(masks[d["doc_id"]])
                for i in range(len(keep)):
                    if not keep[i] and rng.random() < p:
                        keep[i] = True
                toks = [t for t, k in zip(d["tokens"], keep) if k]
                labs = project_labels(d["tokens"], d["labels"], keep)
                out.append({"doc_id": f"{d['doc_id']}_p{p}_{c}", "tokens": toks,
                            "labels": labs})
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps({"doc_id": d["doc_id"], "tokens": d["tokens"],
                                "labels": d["labels"]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} docs ({args.extra_clean} clean + "
          f"{len(args.ps)}x{args.copies} corrupted copies) -> {args.out}")


if __name__ == "__main__":
    main()
