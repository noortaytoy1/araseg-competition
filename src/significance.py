"""Paired bootstrap significance for the main comparisons. Reports ΔF1 and a
two-sided bootstrap p-value (fraction of resamples where the sign flips).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from data import load_jsonl
from eval_local import load_predictions

PAIRS = {
    "PA":       ("subs/PA_dev_megadp.csv",        "rule"),
    "NoPnx-NP": ("subs/NoPnx-NP_dev_adp.csv",     "rule"),
}
# rule baselines regenerated as needed; here compare system vs single-encoder.
SYS = {
    "PA":       "subs/PA_dev_megadp.csv",
    "NoPnx-PA": "subs/NoPnx-PA_dev_ens.csv",
    "NP":       "subs/NP_dev_ens.csv",
    "NoPnx-NP": "subs/NoPnx-NP_dev_adp.csv",
}
SINGLE = {  # single AraBERTv02 dev predictions
    "PA":       "subs/PA_dev_model_orpar.csv",
    "NoPnx-PA": "subs/NoPnx-PA_dev_model_orpar.csv",
    "NP":       "subs/NP_dev_model.csv",
    "NoPnx-NP": "subs/NoPnx-NP_dev_model.csv",
}


def perdoc(gold, pred):
    return np.array([f1_score(gold[i], pred[i], average="binary", zero_division=0) for i in gold])


def paired_boot(a, b, B=5000, seed=0):
    n = len(a); rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    da = a[idx].mean(1) - b[idx].mean(1)
    obs = a.mean() - b.mean()
    p = 2 * min((da <= 0).mean(), (da >= 0).mean())
    lo, hi = np.percentile(da*100, [2.5, 97.5])
    return obs*100, p, lo, hi


if __name__ == "__main__":
    print(f"{'task':9s} {'system':>7} {'single':>7} {'ΔF1':>6} {'p':>8} {'95% CI':>16}")
    for task in ["PA", "NoPnx-PA", "NP", "NoPnx-NP"]:
        docs = load_jsonl(f"data/{task}_dev.jsonl")
        gold = {d["doc_id"]: list(d["labels"]) for d in docs}
        sys = load_predictions(SYS[task]); sgl = load_predictions(SINGLE[task])
        a, b = perdoc(gold, sys), perdoc(gold, sgl)
        d, p, lo, hi = paired_boot(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{task:9s} {a.mean()*100:7.2f} {b.mean()*100:7.2f} {d:+6.2f} "
              f"{p:8.4f} [{lo:+.2f},{hi:+.2f}] {sig}")
