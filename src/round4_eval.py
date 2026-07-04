"""Round-4 selection: the Qwen2.5-3B LoRA voter vs the current frozen systems.
 A. open NoPnx-NP: base = 4-voter (85.06/85.17); +qwen, and qwen-for-satft swap.
 B. closed NoPnx-NP: base = mega-6 (84.53/84.97); +qwen (closed-legal).
Same rule: dev CI excludes 0 AND test >= base.
"""
from __future__ import annotations

import numpy as np

from data import load_jsonl
from round3_eval import load, perdoc_adaptive, verdict
from dp_decode import fit_length_logprob

TASK = "NoPnx-NP"
NAMES = ["voter", "voter-s1", "voter-s2", "voter-s3", "voter-arbertv2",
         "voter-araelectra", "open", "satft", "qwen"]


def main() -> None:
    glp = fit_length_logprob(load_jsonl(f"data/{TASK}_train.jsonl"))
    dd, dc = load(TASK, "dev", NAMES)
    td, tc = load(TASK, "test", NAMES)
    if "qwen" not in dc:
        raise SystemExit("qwen probs missing")

    print("qwen solo dev:", perdoc_adaptive(TASK, dd, dc, ["qwen"], glp).mean().round(2))

    base4 = ["voter-s2", "voter-araelectra", "open", "satft"]
    b4d = perdoc_adaptive(TASK, dd, dc, base4, glp)
    b4t = perdoc_adaptive(TASK, td, tc, base4, glp).mean()
    print(f"\n[A] open base(4v): dev {b4d.mean():.2f} test {b4t:.2f}")
    for name, mem in [("add qwen", base4 + ["qwen"]),
                      ("swap satft->qwen", ["voter-s2", "voter-araelectra", "open", "qwen"])]:
        cd = perdoc_adaptive(TASK, dd, dc, mem, glp)
        ct = perdoc_adaptive(TASK, td, tc, mem, glp).mean()
        verdict(name, b4d, cd, b4t, ct)

    mega = ["voter", "voter-s1", "voter-s2", "voter-s3", "voter-arbertv2", "voter-araelectra"]
    bmd = perdoc_adaptive(TASK, dd, dc, mega, glp)
    bmt = perdoc_adaptive(TASK, td, tc, mega, glp).mean()
    print(f"\n[B] closed base(mega-6): dev {bmd.mean():.2f} test {bmt:.2f}")
    cd = perdoc_adaptive(TASK, dd, dc, mega + ["qwen"], glp)
    ct = perdoc_adaptive(TASK, td, tc, mega + ["qwen"], glp).mean()
    verdict("add qwen", bmd, cd, bmt, ct)


if __name__ == "__main__":
    main()
