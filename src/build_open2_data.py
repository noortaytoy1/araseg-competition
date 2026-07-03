"""Open-track round 2 data: classical Arabic (Tashkeela) + large-scale mix.

Downloads 2 Tashkeela parquet shards (~800k classical passages, undiacritized
field), splits to sentences, writes:
  ext/classical_sents.txt  (<=500k sents, classical/hadith register)
  ext/scale_sents.txt      (~1M sents: classical + wiki + opus + ashaar mix)
then builds boundary-recovery pretrain jsonls via build_pretrain.py.
"""
from __future__ import annotations

import random
import re
import subprocess
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

SHARDS = [
    "data/train-00000-of-00004-24ddf0b9727c0b89.parquet",
    "data/train-00001-of-00004-557d7117f918bf60.parquet",
]
SPLIT_RE = re.compile(r"[.!?؟؛:…]+")
AR_RE = re.compile(r"[؀-ۿ]")


def sentences_from(text: str):
    for s in SPLIT_RE.split(text):
        s = " ".join(s.split())
        w = s.split()
        if 4 <= len(w) <= 60 and AR_RE.search(s):
            yield s


def main() -> None:
    rng = random.Random(0)
    seen, classical = set(), []
    for shard in SHARDS:
        p = hf_hub_download("asas-ai/Tashkeela", shard, repo_type="dataset")
        df = pd.read_parquet(p, columns=["text_no_taskheel"])
        for t in df["text_no_taskheel"].astype(str):
            for s in sentences_from(t):
                k = hash(s)
                if k not in seen:
                    seen.add(k)
                    classical.append(s)
        print(f"{shard}: total classical sents so far {len(classical)}", flush=True)
        if len(classical) >= 900_000:
            break
    rng.shuffle(classical)
    with open("ext/classical_sents.txt", "w") as f:
        f.write("\n".join(classical[:500_000]) + "\n")
    print("classical_sents.txt:", min(len(classical), 500_000))

    mix = classical[:600_000]
    for fn in ["ext/wiki_ar_sents.txt", "ext/opus_ar_sents.txt", "ext/ashaar_verses.txt"]:
        mix += [l.strip() for l in open(fn) if l.strip()]
    rng.shuffle(mix)
    with open("ext/scale_sents.txt", "w") as f:
        f.write("\n".join(mix[:1_000_000]) + "\n")
    print("scale_sents.txt:", min(len(mix), 1_000_000))

    PY = sys.executable
    jobs = [
        ("ext/classical_sents.txt", "NoPnx-NP", 12000, "ext/pretrain_clas_NoPnx-NP.jsonl"),
        ("ext/classical_sents.txt", "NoPnx-PA", 12000, "ext/pretrain_clas_NoPnx-PA.jsonl"),
        ("ext/scale_sents.txt", "NoPnx-NP", 40000, "ext/pretrain_scale_NoPnx-NP.jsonl"),
    ]
    for sents, variant, ndocs, out in jobs:
        subprocess.run([PY, "build_pretrain.py", "--sents", sents, "--variant", variant,
                        "--n-docs", str(ndocs), "--out", out], check=True)
        print("built", out, flush=True)


if __name__ == "__main__":
    main()
