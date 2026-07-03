"""Multi-seed scaling: evaluate seeds {42,1,2} at each size, plot mean+-std bands,
refit the power law on the per-size means. Overwrites figs/scaling.pdf and
figs/powerlaw.pdf and updates scaling/scaling.json -> scaling_bands.json.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from scaling_eval import SIZES, dev_f1

SEEDS = {42: "", 1: "-s1", 2: "-s2"}


def collect():
    cache = {}
    if os.path.exists("scaling/bands.json"):
        cache = json.load(open("scaling/bands.json"))
    res = {}
    for task in ["PA", "NoPnx-NP"]:
        low = task.lower(); res[task] = {}
        for n in SIZES:
            vals = []
            for seed, suf in SEEDS.items():
                md = f"runs/{low}-n{n}{suf}"
                key = md
                if not os.path.exists(os.path.join(md, "model.safetensors")):
                    continue
                if key in cache:
                    vals.append(cache[key])
                else:
                    f = dev_f1(md, task); cache[key] = f; vals.append(f)
            if vals:
                res[task][n] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
                print(f"{task} n={n}: {np.mean(vals):.2f} +- {np.std(vals):.2f} (seeds={len(vals)})")
    json.dump(cache, open("scaling/bands.json", "w"))
    json.dump(res, open("scaling/scaling_bands.json", "w"), indent=2)
    return res


def plot(res):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fits = {}
    for task, col in [("PA", "C0"), ("NoPnx-NP", "C1")]:
        ns = sorted(res[task]); xs = np.array(ns, float)
        mu = np.array([res[task][n][0] for n in ns])
        sd = np.array([res[task][n][1] for n in ns])
        ax.fill_between(xs, mu - sd, mu + sd, color=col, alpha=0.2)
        ax.plot(xs, mu, "o", color=col)
        def f(n, C, a, al): return C - a * np.power(n, -al)
        try:
            (C, a, al), _ = curve_fit(f, xs, mu, p0=[mu[-1] + 2, 50, 0.5], maxfev=20000,
                                      bounds=([mu[-1] - 1, 0, 0.05], [100, 1e4, 3]))
        except Exception:
            C, a, al = mu[-1], 0, 1
        fits[task] = (C, al)
        xx = np.linspace(xs[0], 700, 200)
        ax.plot(xx, f(xx, C, a, al), "-", color=col,
                label=f"{task}: $\\hat C$={C:.1f}, $\\alpha$={al:.2f}")
    ax.set_xlabel("training documents"); ax.set_ylabel("dev macro-F1")
    ax.legend(fontsize=7); ax.set_title("Scaling (3 seeds, mean$\\pm$std) + power-law fit")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("paper/figs/scaling.pdf")
    fig.savefig("paper/figs/powerlaw.pdf"); plt.close(fig)
    print("fits:", {k: (round(v[0],1), round(v[1],2)) for k,v in fits.items()})
    return fits


if __name__ == "__main__":
    plot(collect())
