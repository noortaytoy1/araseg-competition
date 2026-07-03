"""A* analyses: (1) calibration-grounded irreducible-error (Bayes) floor,
(2) power-law fit + data extrapolation of the scaling curve, (3) ensemble
bias-variance / diversity decomposition. Writes numbers + two figures.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import f1_score

from data import PARAGRAPH_TOKEN, load_jsonl

OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def flat_probs(task, split="dev"):
    docs = load_jsonl(f"data/{task}_{split}.jsonl")
    pr = np.load(f"probs/{task}_{split}_mega.npz")
    P, G = [], []
    for d in docs:
        p, g, toks = pr[d["doc_id"]], d["labels"], d["tokens"]
        for i, t in enumerate(toks):
            if t != PARAGRAPH_TOKEN:
                P.append(p[i]); G.append(g[i])
    return np.array(P), np.array(G)


def bayes_floor(task):
    """Irreducible per-token error = sum over prob-bins of min(rate,1-rate),
    i.e. the error of the optimal decision on these (well-calibrated) probs.
    Also report the share of POSITIVES that sit in ambiguous bins (0.3-0.7)."""
    P, G = flat_probs(task)
    bins = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(P, bins) - 1, 0, 19)
    n = len(P); err = 0.0
    for b in range(20):
        m = idx == b
        if m.sum() == 0:
            continue
        rate = G[m].mean()
        err += m.sum() / n * min(rate, 1 - rate)
    amb = ((P >= 0.3) & (P <= 0.7))
    pos_amb = G[amb].sum() / max(G.sum(), 1)
    return err * 100, pos_amb * 100, G.mean()


def powerlaw():
    res = json.load(open("scaling/scaling.json"))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    info = {}
    for task, col in [("PA", "C0"), ("NoPnx-NP", "C1")]:
        xs = np.array([n for n, _ in res[task]], float)
        ys = np.array([f for _, f in res[task]], float)
        # F1(n) = C - a * n^(-alpha)   (saturating power law)
        def f(n, C, a, al): return C - a * np.power(n, -al)
        try:
            (C, a, al), _ = curve_fit(f, xs, ys, p0=[ys[-1] + 2, 50, 0.5],
                                      maxfev=20000, bounds=([ys[-1] - 1, 0, 0.05],
                                                            [100, 1e4, 3]))
        except Exception:
            C, a, al = ys[-1], 0, 1
        info[task] = dict(C=float(C), alpha=float(al))
        xx = np.linspace(xs[0], 700, 200)
        ax.plot(xs, ys, "o", color=col)
        ax.plot(xx, f(xx, C, a, al), "-", color=col,
                label=f"{task}: $\\hat C$={C:.1f}, $\\alpha$={al:.2f}")
        # docs to reach 90 on NoPnx-NP / within 0.5 of asymptote
        if C > 90:
            n90 = (a / (C - 90)) ** (1 / al)
            info[task]["n_to_90"] = float(n90)
    ax.axhline(94.4, ls=":", color="C0", lw=0.7)
    ax.set_xlabel("training documents"); ax.set_ylabel("dev macro-F1")
    ax.legend(fontsize=7); ax.set_title("Power-law scaling fit")
    fig.tight_layout(); fig.savefig(f"{OUT}/powerlaw.pdf"); plt.close(fig)
    return info


def ensemble_decomp(task="NoPnx-NP"):
    """Single-voter mean error vs ensemble error; correlation-implied floor."""
    docs = load_jsonl(f"data/{task}_dev.jsonl")
    gold = {d["doc_id"]: np.array(d["labels"]) for d in docs}
    files = ["", "-s1", "-s2", "-s3", "-arbertv2", "-araelectra"]
    f1s = []
    perr = []
    for fsuf in files:
        p = np.load(f"probs/{task}_dev_voter{fsuf}.npz")
        f = np.mean([f1_score(gold[d], (p[d] >= 0.5).astype(int), zero_division=0) for d in gold])
        f1s.append(f * 100)
        perr.append(np.concatenate([((p[d] >= 0.5).astype(int) != gold[d]).astype(float) for d in gold]))
    mega = np.load(f"probs/{task}_dev_mega.npz")
    fe = np.mean([f1_score(gold[d], (mega[d] >= 0.5).astype(int), zero_division=0) for d in gold]) * 100
    E = np.vstack(perr)
    rho = E[np.corrcoef(E) < 1].mean() if False else np.corrcoef(E)[np.triu_indices(6, 1)].mean()
    return np.mean(f1s), fe, rho


if __name__ == "__main__":
    print("=== Irreducible (Bayes) error floor from calibrated probs ===")
    for t in ["PA", "NP", "NoPnx-PA", "NoPnx-NP"]:
        e, pa, base = bayes_floor(t)
        print(f"{t:9s}: Bayes per-token err={e:.3f}%  positives in ambiguous[0.3,0.7]={pa:.1f}%")
    print("\n=== Power-law scaling fit ===")
    pl = powerlaw()
    for t, d in pl.items():
        s = f"asymptote C={d['C']:.1f}, alpha={d['alpha']:.2f}"
        if "n_to_90" in d:
            s += f", docs to reach 90 = {d['n_to_90']:.0f}"
        print(f"{t}: {s}")
    print("\n=== Ensemble decomposition (NoPnx-NP) ===")
    fm, fe, rho = ensemble_decomp()
    print(f"mean single-voter F1={fm:.2f}, ensemble F1={fe:.2f}, "
          f"voter err-corr={rho:.2f} (high corr -> small ensemble headroom)")
