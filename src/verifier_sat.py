"""Precision verifier v2 (closed-track): use SaT's BOUNDARY opinion (not
relatedness) as a decorrelated veto. SaT (wtpsplit) is a pretrained sentence
segmenter -> inference only, closed-track-legal. For each boundary the frozen
AraBERT ensemble fired, get SaT's boundary probability at that word; if SaT is
CONFIDENT there is no boundary (prob < tau), the fire is likely spurious ->
suppress. Post-hoc, threshold swept, scored with the official metric.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
TASK = "NoPnx-NP"
DEV = os.path.join(EX, "data", f"{TASK}_dev.jsonl")
POOL = os.path.join(EX, "probs", f"union-{TASK}-POOL_dev.npz")


def macro_f1(pred, gold):
    fs = []
    for d in gold:
        g = np.array(gold[d]); p = np.array(pred[d])
        tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * P * R / (P + R) if P + R else 0.0)
    return float(np.mean(fs))


def main():
    docs = {d["doc_id"]: d["tokens"] for d in
            (json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip())}
    gold = {d["doc_id"]: d["labels"] for d in
            (json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip())}
    z = np.load(POOL, allow_pickle=True)
    prob = {k: np.array(z[k], dtype=float) for k in z.files}
    # baseline operating point
    best, th = -1, 0.5
    for t in np.linspace(0.2, 0.9, 60):
        f = macro_f1({d: (prob[d] >= t).astype(int) for d in gold}, gold)
        if f > best:
            best, th = f, t
    base = {d: (prob[d] >= th).astype(int) for d in gold}
    f0 = best
    print(f"baseline pool: th={th:.3f} macro-F1={f0*100:.2f}")

    from wtpsplit import SaT
    sat = SaT("sat-3l")
    try:
        sat.half().to("cuda")
    except Exception:
        pass

    # SaT boundary prob per WORD: prob at the last char of each word
    satp = {}
    for d, toks in docs.items():
        text = " ".join(toks)
        ch = np.asarray(sat.predict_proba(text))   # per-character boundary prob
        pos, wp = 0, []
        for w in toks:
            end = pos + len(w) - 1
            wp.append(float(ch[end]) if 0 <= end < len(ch) else 0.0)
            pos += len(w) + 1                       # +1 for the space
        satp[d] = np.array(wp)
    # sanity: SaT's own boundary rate at gold boundaries vs non
    allp = np.concatenate([satp[d] for d in gold]); allg = np.concatenate([np.array(gold[d]) for d in gold])
    print(f"SaT prob: mean@gold-boundary {allp[allg == 1].mean():.3f}  mean@non {allp[allg == 0].mean():.3f} "
          f"(separation = signal quality)")

    print(f"\n{'tau':>5} {'kills':>6} {'macroF1':>8} {'dP':>6} {'dR':>6}")
    bestf = (f0, None)
    for tau in [-1, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        sup = {d: base[d].copy() for d in gold}
        kills = 0
        for d in gold:
            fired = np.where(base[d] == 1)[0]
            for i in fired:
                if i < len(satp[d]) and satp[d][i] < tau:   # SaT confident: no boundary
                    sup[d][i] = 0; kills += 1
        f = macro_f1(sup, gold)
        tp = sum(int(((sup[d] == 1) & (np.array(gold[d]) == 1)).sum()) for d in gold)
        fp = sum(int(((sup[d] == 1) & (np.array(gold[d]) == 0)).sum()) for d in gold)
        fn = sum(int(((sup[d] == 0) & (np.array(gold[d]) == 1)).sum()) for d in gold)
        print(f"{tau:>5.2f} {kills:>6d} {f*100:>8.2f} {100*tp/(tp+fp):>6.1f} {100*tp/(tp+fn):>6.1f}")
        if f > bestf[0]:
            bestf = (f, tau)
    print(f"\nBEST: macro-F1 {bestf[0]*100:.2f} at tau={bestf[1]}  (baseline {f0*100:.2f}, "
          f"gain {100*(bestf[0]-f0):+.2f}; noise floor +-0.45)")


if __name__ == "__main__":
    main()
