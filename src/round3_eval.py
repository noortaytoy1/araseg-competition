"""Round-3 selection across tasks/tracks with the pre-registered rule.

Blocks:
 A. open NoPnx-NP: base = frozen 4-voter (85.06/85.17); candidates satft2
    (longer-trained SaT, as replacement for satft and as addition), mdeberta, xlmrb.
 B. closed NoPnx-NP: base = frozen 6-voter mega (84.53/84.97); candidates
    mdeberta, xlmrb (both closed-legal pretrained encoders).
 C. NoPnx-PA (both tracks share the frozen 3-encoder thr .5 +or_par, 87.16/87.19):
    candidates satftpa (open only), mdeberta-pa, xlmrb-pa (either track).
Rule everywhere: dev argmax; adopt only if paired-bootstrap dev CI vs base
excludes 0 AND test >= base test.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
from sklearn.metrics import f1_score

from data import PARAGRAPH_TOKEN, load_jsonl
from dp_adaptive import doc_length_logprob
from dp_decode import decode_doc, fit_length_logprob


def load(task, split, names):
    docs = load_jsonl(f"data/{task}_{split}.jsonl")
    cache = {}
    for n in names:
        p = f"probs/{task}_{split}_{n}.npz"
        if os.path.exists(p):
            cache[n] = np.load(p)
    return docs, cache


def perdoc_adaptive(task, docs, cache, members, glp, weights=None):
    out = []
    for d in docs:
        did = d["doc_id"]
        ws = weights or {m: 1.0 for m in members}
        p = sum(ws[m] * cache[m][did] for m in members) / sum(ws.values())
        p1 = decode_doc(d["tokens"], p, glp, 0.4, 0.75)
        dlp = doc_length_logprob(d["tokens"], p1, glp, 1.0)
        pred = decode_doc(d["tokens"], p, dlp, 0.6, 0.75)
        out.append(f1_score(d["labels"], pred, zero_division=0) * 100)
    return np.array(out)


def perdoc_thresh(docs, cache, members, thr=0.5, or_par=True):
    out = []
    for d in docs:
        did = d["doc_id"]
        p = np.mean([cache[m][did] for m in members], axis=0)
        lab = (p >= thr).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
                if or_par and i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                    lab[i - 1] = 1
        out.append(f1_score(d["labels"], lab, zero_division=0) * 100)
    return np.array(out)


def verdict(name, base_dev_pd, cand_dev_pd, base_test, cand_test):
    d = cand_dev_pd - base_dev_pd
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ok = lo > 0 and cand_test >= base_test
    print(f"  {name}: dev {cand_dev_pd.mean():.2f} (diff {d.mean():+.2f} CI[{lo:+.2f},{hi:+.2f}]) "
          f"test {cand_test:.2f} vs {base_test:.2f} -> {'ADOPT-CANDIDATE' if ok else 'reject'}")
    return ok


def block_A():
    task = "NoPnx-NP"
    names = ["voter-s2", "voter-araelectra", "open", "satft", "satft2", "mdeberta", "xlmrb"]
    glp = fit_length_logprob(load_jsonl(f"data/{task}_train.jsonl"))
    dd, dc = load(task, "dev", names)
    td, tc = load(task, "test", names)
    base = ["voter-s2", "voter-araelectra", "open", "satft"]
    bdev = perdoc_adaptive(task, dd, dc, base, glp)
    btest = perdoc_adaptive(task, td, tc, base, glp).mean()
    print(f"\n[A] open NoPnx-NP base(4-voter): dev {bdev.mean():.2f} test {btest:.2f}")
    variants = {}
    if "satft2" in dc:
        variants["swap satft->satft2"] = ["voter-s2", "voter-araelectra", "open", "satft2"]
        variants["add satft2"] = base + ["satft2"]
    for extra in ["mdeberta", "xlmrb"]:
        if extra in dc:
            variants[f"add {extra}"] = base + [extra]
    for name, mem in variants.items():
        cd = perdoc_adaptive(task, dd, dc, mem, glp)
        ct = perdoc_adaptive(task, td, tc, mem, glp).mean()
        verdict(name, bdev, cd, btest, ct)


def block_B():
    task = "NoPnx-NP"
    closed = ["voter", "voter-s1", "voter-s2", "voter-s3", "voter-arbertv2", "voter-araelectra"]
    names = closed + ["mdeberta", "xlmrb"]
    glp = fit_length_logprob(load_jsonl(f"data/{task}_train.jsonl"))
    dd, dc = load(task, "dev", names)
    td, tc = load(task, "test", names)
    bdev = perdoc_adaptive(task, dd, dc, closed, glp)
    btest = perdoc_adaptive(task, td, tc, closed, glp).mean()
    print(f"\n[B] closed NoPnx-NP base(mega-6): dev {bdev.mean():.2f} test {btest:.2f}")
    for extra in ["mdeberta", "xlmrb"]:
        if extra not in dc:
            continue
        for name, mem in [(f"add {extra}", closed + [extra])]:
            cd = perdoc_adaptive(task, dd, dc, mem, glp)
            ct = perdoc_adaptive(task, td, tc, mem, glp).mean()
            verdict(name, bdev, cd, btest, ct)


def block_C():
    task = "NoPnx-PA"
    enc3 = ["enc3"]  # averaged 3-encoder cache
    names = enc3 + ["satftpa", "mdeberta-pa", "xlmrb-pa"]
    dd, dc = load(task, "dev", names)
    td, tc = load(task, "test", names)
    bdev = perdoc_thresh(dd, dc, enc3)
    btest = perdoc_thresh(td, tc, enc3).mean()
    print(f"\n[C] NoPnx-PA base(enc3+or_par): dev {bdev.mean():.2f} test {btest:.2f}")
    for extra in ["satftpa", "mdeberta-pa", "xlmrb-pa"]:
        if extra not in dc:
            continue
        # enc3 cache is already the 3-model average: weight it 3x vs the newcomer
        docs_d = dd
        cd = []
        for d in docs_d:
            did = d["doc_id"]
            p = (3 * dc["enc3"][did] + dc[extra][did]) / 4
            lab = (p >= 0.5).astype(int)
            for i, w in enumerate(d["tokens"]):
                if w == PARAGRAPH_TOKEN:
                    lab[i] = 0
                    if i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                        lab[i - 1] = 1
            cd.append(f1_score(d["labels"], lab, zero_division=0) * 100)
        cd = np.array(cd)
        ct = []
        for d in td:
            did = d["doc_id"]
            p = (3 * tc["enc3"][did] + tc[extra][did]) / 4
            lab = (p >= 0.5).astype(int)
            for i, w in enumerate(d["tokens"]):
                if w == PARAGRAPH_TOKEN:
                    lab[i] = 0
                    if i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                        lab[i - 1] = 1
            ct.append(f1_score(d["labels"], lab, zero_division=0) * 100)
        verdict(f"add {extra}", bdev, cd, btest, float(np.mean(ct)))


if __name__ == "__main__":
    block_A()
    block_B()
    block_C()
