"""Data utilities for the AraSeg 2026 shared task.

Dataset schema (per document):
  doc_id    : str
  tokens    : list[str]   whitespace-tokenized; "\n" tokens mark paragraph breaks
  labels    : list[int]   1 = a sentence boundary FOLLOWS this token
  text      : str
  label_str : str         labels as a binary string

Tasks (HF dataset per variant), splits: train / dev / test.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List

TASKS: Dict[str, str] = {
    "PA": "MBZUAI/AraSeg-2026-Shared-Task-PA",
    "NoPnx-PA": "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA",
    "NP": "MBZUAI/AraSeg-2026-Shared-Task-NP",
    "NoPnx-NP": "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP",
}
SPLITS = ("train", "dev", "test")

# Paragraph break appears as a literal newline *token* in PA variants.
PARAGRAPH_TOKEN = "\n"
# Stand-in fed to the encoder instead of "\n" (added as a special token).
MODEL_PAR_TOKEN = "[PAR]"

# Strong sentence-final punctuation (Arabic + Latin). "," and ";" are NOT
# included by default: in AraSeg they are only sometimes boundaries (e.g.
# hadith chains) — that's what the learned model is for.
SENT_FINAL_PUNCT = {".", "!", "?", "؟", "۔", "…", "!", "؞"}


def load_task(task: str, split: str, jsonl_path: str | None = None) -> List[dict]:
    """Load one split of one task as a list of dicts.

    Preferred path: HuggingFace `datasets` (needs internet / cached data).
    Offline path:   a JSONL file previously written by `export_jsonl`.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(TASKS)}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}")

    if jsonl_path:
        return load_jsonl(jsonl_path)

    from datasets import load_dataset  # imported lazily

    ds = load_dataset(TASKS[task], split=split)
    return [dict(row) for row in ds]


def load_jsonl(path: str) -> List[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def export_jsonl(task: str, out_dir: str, splits: Iterable[str] = SPLITS) -> None:
    """Cache HF splits to local JSONL so everything else can run offline."""
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        docs = load_task(task, split)
        path = os.path.join(out_dir, f"{task}_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"wrote {len(docs):4d} docs -> {path}")


def boundary_rate(docs: List[dict]) -> float:
    pos = sum(sum(d["labels"]) for d in docs)
    tot = sum(len(d["labels"]) for d in docs)
    return pos / max(tot, 1)


def mean_sentence_length(docs: List[dict]) -> float:
    """Average number of tokens per gold sentence (incl. punct / \\n tokens)."""
    sents, toks = 0, 0
    for d in docs:
        toks += len(d["labels"])
        sents += sum(d["labels"])
    return toks / max(sents, 1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Export AraSeg splits to JSONL")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = ap.parse_args()
    export_jsonl(args.task, args.out_dir, args.splits)
