"""verify_deepdoc_2026_07_10.py — independent spot-check of the headline numbers in
docs/MODEL_PROBLEMS_DEEP_2026-07-10.md.

ADDITIVE, analysis-only, CPU-only, read-only on runs/ and probs/ and data/.
Recomputes, from primary sources (frozen probability cache + frozen prediction
CSVs + gold JSONL), every number it can reach WITHOUT model inference:

  [1] error balance at threshold 0.5 (FP / FN / ratio / over-cut ratio)
  [2] calibration profile (extremes, fire-side honesty gap, quiet side,
      per-genre ultra-confident wrong-cut share)
  [3] per-doc F1 distribution + worst-15 deficit share (prep battery s42)
  [4] cross-seed systematic-FP fraction (cons_baseline seeds 42/1/2)
  [5] burstiness / double-cuts
  [6] genealogy-document miss profile

Run from repo root:  CUDA_VISIBLE_DEVICES= python src/verify_deepdoc_2026_07_10.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only — GPU is busy

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genre_buckets import bucket  # noqa: E402

GOLD = "data/NoPnx-NP_dev.jsonl"
NPZ = "probs/union-NoPnx-NP-arabertv02-s42_dev.npz"
BATTERY = "runs/prep_battery_2026-07-09"
PRED = "pred_dev_t050.csv"


def load_gold(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_pred_csv(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if line:
                did, s = line.split(",", 1)
                out[did] = np.array([int(c) for c in s.strip()], dtype=int)
    return out


def prf1(g, p):
    tp = int(((g == 1) & (p == 1)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp, fp, fn


def main():
    docs = load_gold(GOLD)
    z = np.load(NPZ)
    n_docs = len(docs)
    print(f"dev docs = {n_docs}")

    # ---------------- [1] + [2]: union-s42 probability cache at t=0.5 ----------
    all_p, all_g, all_genre = [], [], []
    per_doc_pred_b, per_doc_gold_b = [], []
    for d in docs:
        p = z[d["doc_id"]]
        g = np.asarray(d["labels"], dtype=int)
        assert len(p) == len(g), d["doc_id"]
        all_p.append(p)
        all_g.append(g)
        gen = bucket(d["tokens"])
        all_genre.extend([gen] * len(g))
        per_doc_pred_b.append(int((p >= 0.5).sum()))
        per_doc_gold_b.append(int(g.sum()))
    P = np.concatenate(all_p)
    G = np.concatenate(all_g)
    GEN = np.array(all_genre)
    L = (P >= 0.5).astype(int)

    fp = int(((G == 0) & (L == 1)).sum())
    fn = int(((G == 1) & (L == 0)).sum())
    npos = len(P)
    nb = int(G.sum())
    npred = int(L.sum())
    print("\n[1] ERROR BALANCE (union-s42, t=0.5)")
    print(f"  positions={npos} (doc claims 127,722)  gold boundaries={nb} (claims 12,926)")
    print(f"  FP={fp} (claims 2,732)   FN={fn} (claims 1,531)   FP:FN={fp/fn:.2f}:1 (claims 1.78)")
    print(f"  pred boundaries={npred} (claims 14,127)   pred/gold={npred/nb:.3f} (claims 1.093)")
    print(f"  mean sent len: true={npos/nb:.2f} (claims 9.88)  pred={npos/npred:.2f} (claims 9.04)")
    over = sum(1 for a, b in zip(per_doc_pred_b, per_doc_gold_b) if a > b)
    print(f"  docs over-cut: {100*over/n_docs:.1f}% (claims 66.2%)")
    cfp = int(((G == 0) & (L == 1) & (P >= 0.8)).sum())
    cfn = int(((G == 1) & (L == 0) & (P <= 0.2)).sum())
    print(f"  confident FP (P>=0.8)={cfp} (claims 2,236) = {100*cfp/fp:.0f}% of FP (claims 82%)")
    print(f"  confident FN (P<=0.2)={cfn} (claims 1,245) = {100*cfn/fn:.0f}% of FN (claims 81%)")
    wrongcut_p = P[(G == 0) & (L == 1)]
    print(f"  median P on a wrong cut = {np.median(wrongcut_p):.3f} (claims 0.989)")

    print("\n[2] CALIBRATION (union-s42 cache)")
    ext = ((P >= 0.9) | (P <= 0.1)).mean() * 100
    ext99 = ((P >= 0.99) | (P <= 0.01)).mean() * 100
    print(f"  extremes (>=0.9 or <=0.1): {ext:.1f}% (claims 97.9%)")
    print(f"  beyond 0.99/0.01: {ext99:.1f}% (claims 95.6%)")
    m = P >= 0.9
    print(f"  fire side P>=0.9: claimed={100*P[m].mean():.1f}% (claims 99.5)  "
          f"true={100*G[m].mean():.1f}% (claims 84.1)  gap={100*(P[m].mean()-G[m].mean()):+.1f} (claims +15.3)")
    m99 = P >= 0.99
    print(f"  fire side P>=0.99: true={100*G[m99].mean():.1f}% (claims 87.5)")
    q = P <= 0.1
    print(f"  quiet side P<=0.1: claimed-no={100*(1-P[q]).mean():.1f}% (claims 99.9)  "
          f"true-no={100*(G[q] == 0).mean():.1f}% (claims 99.0)  "
          f"gap={100*((1-P[q]).mean()-(G[q] == 0).mean()):+.1f} (claims +0.9)")
    print("  ultra-confident cuts (P>=0.99) wrong-share by genre:")
    for gname, claim in [("cloze/exercise", 44.0), ("legal/constitution", 17.3),
                         ("general prose", 11.0)]:
        mm = (P >= 0.99) & (GEN == gname)
        if mm.sum():
            print(f"    {gname:20s}: {100*(G[mm] == 0).mean():.1f}% wrong of {int(mm.sum())} "
                  f"(claims {claim}%)")

    # ---------------- [6] genealogy document ----------------------------------
    print("\n[6] GENEALOGY DOC (union-s42)")
    for d in docs:
        if bucket(d["tokens"]) == "genealogy":
            p = z[d["doc_id"]]
            g = np.asarray(d["labels"], dtype=int)
            miss = int(((g == 1) & (p < 0.5)).sum())
            print(f"  {d['doc_id']}: gold boundaries={int(g.sum())} (claims 27)  "
                  f"missed={miss} (claims 17)  "
                  f"median P at true boundaries={np.median(p[g == 1]):.3f} (claims 0.078)  "
                  f"pred/gold={float((p >= .5).sum())/g.sum():.3f} (claims 0.889)")

    # ---------------- [3] per-doc distribution (prep battery s42) -------------
    print(f"\n[3] PER-DOC F1 (battery {BATTERY}, cons_baseline_s42, {PRED})")
    pred42 = load_pred_csv(os.path.join(BATTERY, "cons_baseline_s42", PRED))
    rows = []
    for d in docs:
        g = np.asarray(d["labels"], dtype=int)
        pr = pred42[d["doc_id"]]
        _, _, F, *_ = prf1(g, pr)
        rows.append((F * 100, d["doc_id"], int(g.sum())))
    rows.sort()
    f1s = np.array([r[0] for r in rows])
    macro = f1s.mean()
    print(f"  macro={macro:.2f} (claims 83.54)  median={np.median(f1s):.2f} (claims 83.84)  "
          f"min={f1s.min():.2f} (claims 49.28)  max={f1s.max():.2f} (claims 100)")
    print(f"  docs<50: {int((f1s < 50).sum())} (claims 1)   docs<70: {int((f1s < 70).sum())} (claims 13)")
    for lo, hi, claim in [(70, 80, 48), (80, 90, 106), (90, 100.0001, 55)]:
        print(f"  docs in [{lo},{int(hi)}): {int(((f1s >= lo) & (f1s < hi)).sum())} (claims {claim})")
    deficit = 100.0 - f1s
    td = deficit.sum()
    for k, claim in [(5, 5.9), (15, 14.8), (30, 26.0)]:
        print(f"  worst-{k} deficit share: {100*deficit[:k].sum()/td:.1f}% (claims {claim}%)")
    print("  worst-15 doc ids + F1:")
    for F, did, gb in rows[:15]:
        print(f"    {did}  {F:.1f}")
    legal = [(d["doc_id"], int(sum(d["labels"]))) for d in docs
             if bucket(d["tokens"]) == "legal/constitution"]
    print(f"  legal docs: {legal} (claims one doc with 1,161 boundaries)")

    # ---------------- [4] cross-seed systematicity ----------------------------
    print("\n[4] CROSS-SEED FP SYSTEMATICITY (cons_baseline s42/s1/s2)")
    preds = {s: load_pred_csv(os.path.join(BATTERY, f"cons_baseline_s{s}", PRED))
             for s in ("42", "1", "2")}
    cnt = defaultdict(int)
    for s, pd in preds.items():
        for d in docs:
            g = np.asarray(d["labels"], dtype=int)
            p = pd[d["doc_id"]]
            for pos in np.where((g == 0) & (p == 1))[0]:
                cnt[(d["doc_id"], int(pos))] += 1
    vals = np.array(list(cnt.values()))
    union = len(vals)
    in3 = int((vals == 3).sum())
    in1 = int((vals == 1).sum())
    print(f"  union FP={union}  all-3-seeds={100*in3/union:.1f}% (claims 65.4%)  "
          f"single-seed={100*in1/union:.1f}% (claims ~18.5%)")
    print(f"  of one seed's FPs, shared by all three: {100*3*in3/vals.sum():.1f}% (claims ~79.5%)")

    # ---------------- [5] burstiness ------------------------------------------
    print("\n[5] BURSTINESS (cons_baseline_s42)")
    total_fp = burst_fp = double_cuts = 0
    exp_burst = 0.0
    W = 5
    for d in docs:
        g = np.asarray(d["labels"], dtype=int)
        p = pred42[d["doc_id"]]
        fpos = np.where((g == 0) & (p == 1))[0]
        k, n = len(fpos), len(g)
        total_fp += k
        if k >= 2:
            gaps = np.diff(fpos)
            ib = np.zeros(k, bool)
            ib[1:] |= gaps <= W
            ib[:-1] |= gaps <= W
            burst_fp += int(ib.sum())
            double_cuts += int(((gaps >= 1) & (gaps <= 2)).sum())
            if n > 2 * W:
                exp_burst += k * (1.0 - (1.0 - (2 * W) / n) ** (k - 1))
    print(f"  FPs in burst: {100*burst_fp/total_fp:.1f}% (claims 36.4%)  "
          f"expected uniform: {100*exp_burst/total_fp:.1f}% (claims 30.6%)  "
          f"double-cut pairs: {double_cuts} (claims 288)")
    print(f"  prep-battery pred/gold ratio: "
          f"{sum(int(pred42[d['doc_id']].sum()) for d in docs)/nb:.3f} (claims 1.087)")


if __name__ == "__main__":
    main()
