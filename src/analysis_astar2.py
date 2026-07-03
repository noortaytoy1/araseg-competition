"""A* round 2: address reviewer objections.
 (1) MODEL-FREE ambiguity: label inconsistency of repeated gold contexts.
 (2) Bootstrap CIs on the scaling-law exponents.
 (3) Krogh-Vedelsby ensemble ambiguity decomposition (closed vs +external).
 (4) Leave-one-out decoding ablation.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import f1_score

from data import PARAGRAPH_TOKEN, load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from dp_adaptive import doc_length_logprob
from eval_local import compute_metrics


# ---------- (1) model-free ambiguity from repeated gold contexts ----------
def model_free_ambiguity(task):
    docs = load_jsonl(f"data/{task}_train.jsonl") + load_jsonl(f"data/{task}_dev.jsonl")
    print(f"\n[{task}] model-free ambiguity (repeated-context label inconsistency)")
    for w in (1, 2, 3):
        groups = defaultdict(list)
        for d in docs:
            t, g = d["tokens"], d["labels"]
            idx = [i for i in range(len(t)) if t[i] != PARAGRAPH_TOKEN]
            seq = [t[i] for i in idx]; lab = [g[i] for i in idx]
            for j in range(len(seq)):
                lo = tuple(seq[max(0, j - w):j]); hi = tuple(seq[j + 1:j + 1 + w])
                groups[(lo, seq[j], hi)].append(lab[j])
        rep = {k: v for k, v in groups.items() if len(v) >= 2}
        inst = sum(len(v) for v in rep.values())
        mixed_inst = sum(len(v) for v in rep.values() if 0 < sum(v) < len(v))
        min_err = sum(min(sum(v), len(v) - sum(v)) for v in rep.values())
        tot = sum(len(v) for v in groups.values())
        print(f"  w=+-{w}: repeated-context tokens={inst} "
              f"({100*inst/tot:.0f}% of all); in label-MIXED contexts={100*mixed_inst/max(inst,1):.1f}%; "
              f"min achievable error on them={100*min_err/max(inst,1):.2f}%")


# ---------- (2) bootstrap CIs on scaling exponents ----------
def scaling_ci(B=2000):
    res = json.load(open("scaling/scaling_bands.json"))
    print("\n[scaling] power-law exponent CIs (parametric bootstrap over seed mean+-std)")
    rng = np.random.default_rng(0)
    def f(n, C, a, al): return C - a * np.power(n, -al)
    for task in res:
        ns = np.array(sorted(int(k) for k in res[task]), float)
        mu = np.array([res[task][str(int(n))][0] for n in ns])
        sd = np.array([max(res[task][str(int(n))][1], 0.05) for n in ns])
        als, Cs = [], []
        for _ in range(B):
            y = rng.normal(mu, sd)
            try:
                (C, a, al), _ = curve_fit(f, ns, y, p0=[mu[-1]+2, 50, .5], maxfev=8000,
                                          bounds=([mu[-1]-2,0,.05],[100,1e4,3]))
                als.append(al); Cs.append(C)
            except Exception:
                pass
        al_lo, al_hi = np.percentile(als, [2.5, 97.5])
        C_lo, C_hi = np.percentile(Cs, [2.5, 97.5])
        print(f"  {task:9s}: alpha={np.median(als):.2f} [{al_lo:.2f},{al_hi:.2f}]  "
              f"Chat={np.median(Cs):.1f} [{C_lo:.1f},{C_hi:.1f}]")


# ---------- (3) Krogh-Vedelsby decomposition (NoPnx-NP) ----------
def kv_decomposition():
    task = "NoPnx-NP"
    docs = load_jsonl(f"data/{task}_dev.jsonl")
    gold = {d["doc_id"]: np.array(d["labels"], float) for d in docs}
    def stack(files):
        cols = []
        for fsuf in files:
            p = np.load(f"probs/{task}_dev_{fsuf}.npz")
            cols.append(np.concatenate([p[d["doc_id"]] for d in docs]))
        return np.vstack(cols)
    g = np.concatenate([gold[d["doc_id"]] for d in docs])
    closed = ["voter", "voter-s1", "voter-s2", "voter-s3", "voter-arbertv2", "voter-araelectra"]
    ext = ["open", "mixopen"]
    print("\n[Krogh-Vedelsby] E_ens = mean voter MSE - diversity (ambiguity)")
    for name, files in [("closed-6", closed), ("closed-6 + 2 external", closed + ext)]:
        P = stack(files)                  # [K, N]
        pbar = P.mean(0)
        e_bar = ((P - g) ** 2).mean()     # mean individual MSE
        amb = ((P - pbar) ** 2).mean()    # diversity
        e_ens = ((pbar - g) ** 2).mean()  # = e_bar - amb
        print(f"  {name:22s}: mean-voter MSE={e_bar:.4f}  diversity={amb:.4f}  "
              f"ens MSE={e_ens:.4f}")


# ---------- (4) leave-one-out decoding ablation (NoPnx-NP) ----------
def loo_decode():
    task = "NoPnx-NP"
    docs = load_jsonl(f"data/{task}_dev.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    probs = np.load(f"probs/{task}_dev_mega.npz")
    glp = fit_length_logprob(load_jsonl(f"data/{task}_train.jsonl"))
    lam1, bias, lam2, blend = 0.4, 0.75, 0.6, 1.0
    flat = np.zeros(201)  # uniform length prior (disables prior)

    def score(decode_fn):
        return compute_metrics(gold, {d["doc_id"]: decode_fn(d) for d in docs})["macro_f1"] * 100

    full = lambda d: _adapt(d, probs[d["doc_id"]], glp, lam1, bias, lam2, blend)
    no_adapt = lambda d: decode_doc(d["tokens"], probs[d["doc_id"]], glp, lam1, bias)
    no_prior = lambda d: decode_doc(d["tokens"], probs[d["doc_id"]], flat, 0.0, bias)
    no_bias = lambda d: _adapt(d, probs[d["doc_id"]], glp, lam1, 0.0, lam2, blend)
    print("\n[decode ablation] NoPnx-NP, mega ensemble")
    print(f"  full (adaptive prior+bias)         : {score(full):.2f}")
    print(f"  - document-adaptive (static prior) : {score(no_adapt):.2f}")
    print(f"  - length prior (threshold-like)    : {score(no_prior):.2f}")
    print(f"  - recall bias                      : {score(no_bias):.2f}")


def _adapt(d, p, glp, lam1, bias, lam2, blend):
    pass1 = decode_doc(d["tokens"], p, glp, lam1, bias)
    dlp = doc_length_logprob(d["tokens"], pass1, glp, blend)
    return decode_doc(d["tokens"], p, dlp, lam2, bias)


if __name__ == "__main__":
    for t in ["PA", "NoPnx-NP"]:
        model_free_ambiguity(t)
    scaling_ci()
    kv_decomposition()
    loo_decode()
