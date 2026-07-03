"""Generate analysis figures for the paper from cached probabilities and data.
Outputs PDFs in paper/figs/. English labels only (pdflatex-safe).
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from data import PARAGRAPH_TOKEN, load_jsonl

OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load_gold(task, split="dev"):
    docs = load_jsonl(f"data/{task}_{split}.jsonl")
    return docs, {d["doc_id"]: np.array(d["labels"]) for d in docs}


# ---------- Fig: voter error-correlation heatmap (NoPnx-NP) ----------
def fig_voter_corr():
    task = "NoPnx-NP"
    docs, gold = load_gold(task)
    names = ["AB-s42", "AB-s1", "AB-s2", "AB-s3", "ARBERT", "ELECTRA"]
    files = ["", "-s1", "-s2", "-s3", "-arbertv2", "-araelectra"]
    errs = []
    for f in files:
        p = np.load(f"probs/{task}_dev_voter{f}.npz")
        e = np.concatenate([( (p[d]>=0.5).astype(int) != gold[d] ).astype(float) for d in gold])
        errs.append(e)
    E = np.vstack(errs)
    C = np.corrcoef(E)
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    im = ax.imshow(C, vmin=0.3, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(names)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i,j]<0.8 else "black", fontsize=6.5)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Per-token error correlation (6 voters)")
    fig.tight_layout(); fig.savefig(f"{OUT}/voter_corr.pdf"); plt.close(fig)
    print(f"voter_corr: mean off-diag corr = {C[np.triu_indices(6,1)].mean():.3f}")


# ---------- Fig: reliability diagram + ECE (mega ensemble) ----------
def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, task in zip(axes, ["PA", "NoPnx-NP"]):
        docs, gold = load_gold(task)
        p = np.load(f"probs/{task}_dev_mega.npz")
        # restrict to non-\n tokens
        P, G = [], []
        for d in docs:
            toks = d["tokens"]; pr = p[d["doc_id"]]; g = gold[d["doc_id"]]
            for i, t in enumerate(toks):
                if t != PARAGRAPH_TOKEN:
                    P.append(pr[i]); G.append(g[i])
        P, G = np.array(P), np.array(G)
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(P, bins) - 1
        xs, ys, ece, n = [], [], 0.0, len(P)
        for b in range(10):
            m = idx == b
            if m.sum() == 0: continue
            conf, acc = P[m].mean(), G[m].mean()
            xs.append(conf); ys.append(acc)
            ece += m.sum()/n * abs(conf-acc)
        ax.plot([0,1],[0,1],"--",color="gray",lw=0.8)
        ax.plot(xs, ys, "o-", ms=3)
        ax.set_title(f"{task}  (ECE={ece:.3f})")
        ax.set_xlabel("predicted P(boundary)"); ax.set_ylabel("empirical rate")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        print(f"calibration {task}: ECE={ece:.4f}")
    fig.tight_layout(); fig.savefig(f"{OUT}/calibration.pdf"); plt.close(fig)


# ---------- Fig: precision/recall/F1 vs threshold (NoPnx-NP mega) ----------
def fig_threshold():
    task = "NoPnx-NP"
    docs, gold = load_gold(task)
    p = np.load(f"probs/{task}_dev_mega.npz")
    ths = np.linspace(0.2, 0.9, 29)
    Ps, Rs, Fs = [], [], []
    for t in ths:
        pr, rc, fs = [], [], []
        for d in docs:
            toks=d["tokens"]; lab=(p[d["doc_id"]]>=t).astype(int); g=gold[d["doc_id"]]
            for i,w in enumerate(toks):
                if w==PARAGRAPH_TOKEN: lab[i]=0
            pr.append(precision_score(g,lab,zero_division=0))
            rc.append(recall_score(g,lab,zero_division=0))
            fs.append(f1_score(g,lab,zero_division=0))
        Ps.append(np.mean(pr)*100); Rs.append(np.mean(rc)*100); Fs.append(np.mean(fs)*100)
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.plot(ths, Ps, label="Precision"); ax.plot(ths, Rs, label="Recall")
    ax.plot(ths, Fs, label="F1", lw=2)
    bi = int(np.argmax(Fs)); ax.axvline(ths[bi], color="gray", ls=":", lw=0.8)
    ax.set_xlabel("threshold"); ax.set_ylabel("macro score"); ax.legend(fontsize=7)
    ax.set_title(f"{task}: P/R/F1 vs threshold")
    fig.tight_layout(); fig.savefig(f"{OUT}/threshold.pdf"); plt.close(fig)
    print(f"threshold {task}: best F1={max(Fs):.2f} at thr={ths[bi]:.2f}")


# ---------- Fig: unseen-context coverage vs train size ----------
def fig_coverage():
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for task in ["PA", "NoPnx-NP"]:
        dev = load_jsonl(f"data/{task}_dev.jsonl")
        dev_ctx = set()
        for d in dev:
            t=d["tokens"]; g=d["labels"]
            for i in range(len(t)):
                if t[i]!=PARAGRAPH_TOKEN and g[i]==1:
                    dev_ctx.add((t[i-1] if i>0 else "<s>", t[i]))
        ns=[22,44,87,130,174]; cov=[]
        for n in ns:
            tr=load_jsonl(f"data/{task}_train_n{n}.jsonl")
            seen=set()
            for d in tr:
                t=d["tokens"]; g=d["labels"]
                for i in range(len(t)):
                    if t[i]!=PARAGRAPH_TOKEN and g[i]==1:
                        seen.add((t[i-1] if i>0 else "<s>", t[i]))
            cov.append(100*len(dev_ctx & seen)/len(dev_ctx))
        ax.plot(ns, cov, "o-", label=task)
        print(f"coverage {task}: {cov[-1]:.1f}% of dev boundary-contexts seen at n=174")
    ax.set_xlabel("training documents"); ax.set_ylabel("% dev boundary-contexts seen")
    ax.legend(fontsize=8); ax.set_title("Boundary-context coverage")
    fig.tight_layout(); fig.savefig(f"{OUT}/coverage.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_voter_corr()
    fig_calibration()
    fig_threshold()
    fig_coverage()
    print("figures ->", OUT)
