"""Open-track round 2 selection: does any new voter beat the frozen open system?

Pool = 6 closed voters + candidates {ft(opus), mixft, wikift, clasft, scaleft,
xlmrl, satft}. Greedy forward selection on DEV with the frozen adaptive-DP decode
(lam1=.4 bias=.75 lam2=.6 blend=1.0); the winner is then confirmed on the public
TEST split. Decision rule (pre-registered): adopt only if dev > frozen dev AND
test >= frozen test (85.08). Prints solo F1s, greedy path, and the verdict.
Assumes per-voter prob caches probs/NoPnx-NP_{split}_{name}.npz exist.
"""
from __future__ import annotations

import os

import numpy as np

from data import load_jsonl
from dp_adaptive import doc_length_logprob
from dp_decode import decode_doc, fit_length_logprob
from eval_local import compute_metrics

TASK = "NoPnx-NP"
CLOSED = ["voter", "voter-s1", "voter-s2", "voter-s3", "voter-arbertv2", "voter-araelectra"]
FROZEN_EXT = ["open", "mixopen"]
CANDIDATES = ["wiki", "clasft", "scaleft", "xlmrl", "satft"]
LAM1, BIAS, LAM2, BLEND = 0.4, 0.75, 0.6, 1.0


def caches(split):
    out = {}
    for name in CLOSED + FROZEN_EXT + CANDIDATES:
        p = f"probs/{TASK}_{split}_{name}.npz"
        if os.path.exists(p):
            out[name] = np.load(p)
    return out


def score(members, split, docs, gold, cache, glp):
    preds = {}
    for d in docs:
        p = np.mean([cache[m][d["doc_id"]] for m in members], axis=0)
        pass1 = decode_doc(d["tokens"], p, glp, LAM1, BIAS)
        dlp = doc_length_logprob(d["tokens"], pass1, glp, BLEND)
        preds[d["doc_id"]] = decode_doc(d["tokens"], p, dlp, LAM2, BIAS)
    return compute_metrics(gold, preds)["macro_f1"] * 100


def main() -> None:
    glp = fit_length_logprob(load_jsonl(f"data/{TASK}_train.jsonl"))
    data = {}
    for split in ["dev", "test"]:
        docs = load_jsonl(f"data/{TASK}_{split}.jsonl")
        data[split] = (docs, {d["doc_id"]: list(d["labels"]) for d in docs}, caches(split))
    dev_docs, dev_gold, dev_cache = data["dev"]

    have = [c for c in CANDIDATES if c in dev_cache]
    print("available candidates:", have)
    print("\n== solo dev F1 (adaptive decode) ==")
    for c in have + ["voter"]:
        print(f"  {c:10s} {score([c], 'dev', dev_docs, dev_gold, dev_cache, glp):.2f}")

    frozen = CLOSED + FROZEN_EXT
    base_dev = score(frozen, "dev", dev_docs, dev_gold, dev_cache, glp)
    print(f"\nfrozen-8 dev = {base_dev:.2f}")

    print("\n== greedy add to frozen-8 (dev) ==")
    members, cur = list(frozen), base_dev
    while True:
        gains = []
        for c in have:
            if c in members:
                continue
            f = score(members + [c], "dev", dev_docs, dev_gold, dev_cache, glp)
            gains.append((f, c))
            print(f"  +{c:10s} -> {f:.2f}")
        if not gains or max(gains)[0] <= cur + 1e-9:
            break
        cur, best_c = max(gains)
        members.append(best_c)
        print(f"  ADD {best_c} (dev {cur:.2f})")

    print("\n== greedy from scratch over all voters (dev) ==")
    all_v = frozen + have
    m2, cur2 = [], -1.0
    while True:
        gains = [(score(m2 + [c], "dev", dev_docs, dev_gold, dev_cache, glp), c)
                 for c in all_v if c not in m2]
        if not gains or max(gains)[0] <= cur2 + 1e-9:
            break
        cur2, c = max(gains)
        m2.append(c)
    print(f"  scratch pool ({len(m2)}): {m2} dev={cur2:.2f}")

    finalists = {"frozen-8": frozen}
    if cur > base_dev:
        finalists[f"frozen+{'+'.join(m for m in members if m not in frozen)}"] = members
    if cur2 > max(base_dev, cur):
        finalists["scratch"] = m2
    print("\n== test confirmation ==")
    t_docs, t_gold, t_cache = data["test"]
    for name, mem in finalists.items():
        if not all(m in t_cache for m in mem):
            print(f"  {name}: missing test caches, skipped")
            continue
        f = score(mem, "test", t_docs, t_gold, t_cache, glp)
        print(f"  {name:30s} dev={score(mem, 'dev', dev_docs, dev_gold, dev_cache, glp):.2f} test={f:.2f}")
    print("\nfrozen open test reference: 85.08 — adopt a challenger only if dev AND test beat it.")


if __name__ == "__main__":
    main()
