"""diag_perdoc.py — NEW CUTS on frozen prediction CSVs (no model inference, CPU only).

Four diagnostics nobody has measured yet, all computed from prediction files
already on disk (written by baselines.write_submission) plus the local gold JSONL:

  1. PER-DOC F1 DISTRIBUTION   — the official metric is macro (a plain average of
     each document's F1), so a handful of catastrophic documents can drag the
     whole score. Histogram + worst-15 forensics.
  2. CROSS-SEED SYSTEMATICITY  — same training recipe, three random seeds.
     An error made by ALL three seeds is a defect of the recipe/data (fixable in
     principle); an error made by only one seed is lottery noise (not fixable by
     modelling alone).
  3. ERROR BURSTINESS          — do false cuts cluster (several within a 5-token
     window, or "double-cuts" 1-2 tokens apart) or spread out evenly?
  4. PREDICTED-VS-GOLD boundary counts per genre — the over/under-cut ratio,
     sanity-checked against the known 1.093 corpus-level over-cut ratio.

Read-only on runs/. Writes nothing anywhere. Additive script; touches no
existing code.

Usage (from repo root):
  python src/diag_perdoc.py
  python src/diag_perdoc.py --battery runs/prep_battery_2026-07-09 --pred pred_dev_t050.csv
"""
from __future__ import annotations

import argparse
import io
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only — GPU is busy

import json
from collections import defaultdict

import numpy as np

# Arabic text must survive the Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genre_buckets import bucket  # noqa: E402  (heuristic genre labels)


# ----------------------------------------------------------------------------- io
def load_gold(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_pred_csv(path):
    """Read a write_submission CSV -> {doc_id: [0/1,...]} without pandas."""
    out = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        assert header.strip().startswith("Document ID"), f"bad header in {path}"
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc_id, label_str = line.split(",", 1)
            out[doc_id] = [int(c) for c in label_str.strip()]
    return out


def prf1(g, p):
    g = np.asarray(g)
    p = np.asarray(p)
    tp = int(((g == 1) & (p == 1)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp, fp, fn


# ------------------------------------------------------------------- 1. per-doc F1
def cut1_perdoc_f1(docs, pred, arm_name):
    print("=" * 100)
    print(f"CUT 1 — PER-DOC F1 DISTRIBUTION   arm = {arm_name}")
    print("  (the official score is the plain average of every document's own F1,")
    print("   so a few terrible documents cost as much as many mediocre ones)")
    print("=" * 100)
    rows = []
    for d in docs:
        P, R, F, tp, fp, fn = prf1(d["labels"], pred[d["doc_id"]])
        rows.append(dict(doc_id=d["doc_id"], n=len(d["tokens"]), P=P, R=R, F=F,
                         fp=fp, fn=fn, gold_b=int(sum(d["labels"])),
                         pred_b=int(sum(pred[d["doc_id"]])),
                         genre=bucket(d["tokens"]), text=d.get("text", "")))
    rows.sort(key=lambda r: r["F"])
    f1s = np.array([r["F"] for r in rows]) * 100
    macro = f1s.mean()
    print(f"\ndocs={len(rows)}  macro-F1={macro:.2f}  median={np.median(f1s):.2f}  "
          f"min={f1s.min():.2f}  max={f1s.max():.2f}")

    print("\nHistogram of per-doc F1 (each # = 2 docs):")
    edges = list(range(0, 100, 10)) + [100.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = int(((f1s >= lo) & (f1s < hi)).sum())
        label = f"{lo:3d}-{min(int(hi),100):3d}"
        print(f"  {label}: {k:4d} {'#' * (k // 2 + (1 if k % 2 else 0))}")

    n_lt50 = int((f1s < 50).sum())
    n_lt70 = int((f1s < 70).sum())
    print(f"\ndocs with F1 < 50: {n_lt50}  ({100*n_lt50/len(rows):.1f}%)")
    print(f"docs with F1 < 70: {n_lt70}  ({100*n_lt70/len(rows):.1f}%)")

    deficit = 100.0 - f1s                     # what each doc costs vs a perfect 100
    total_def = deficit.sum()
    worst15_def = deficit[:15].sum()          # rows sorted ascending by F1
    print(f"\nmacro-F1 deficit = 100 - {macro:.2f} = {100-macro:.2f} points")
    print(f"share of that deficit carried by the worst 15 docs "
          f"({15}/{len(rows)} = {100*15/len(rows):.1f}% of docs): "
          f"{100*worst15_def/total_def:.1f}%")
    for k in (5, 10, 30):
        print(f"  (worst {k:2d} docs carry {100*deficit[:k].sum()/total_def:.1f}%)")

    print("\nWORST 15 DOCS")
    print(f"{'doc_id':18s} {'F1':>6} {'P':>6} {'R':>6} {'len':>6} {'gold_b':>6} "
          f"{'pred_b':>6} {'b-rate':>6}  genre")
    for r in rows[:15]:
        print(f"{r['doc_id']:18s} {100*r['F']:6.1f} {100*r['P']:6.1f} {100*r['R']:6.1f} "
              f"{r['n']:6d} {r['gold_b']:6d} {r['pred_b']:6d} "
              f"{r['gold_b']/max(r['n'],1):6.3f}  {r['genre']}")
        snippet = " ".join(r["text"].split()[:22])
        print(f"    {snippet} ...")
    return rows


# -------------------------------------------------------- 2. cross-seed systematicity
def cut2_systematicity(docs, preds_by_seed, arm_name):
    """preds_by_seed: {seed: {doc_id: [0/1]}} for the SAME arm."""
    gold = {d["doc_id"]: np.asarray(d["labels"]) for d in docs}
    seeds = sorted(preds_by_seed)
    fp_count = defaultdict(int)   # (doc_id, pos) -> in how many seeds it is an FP
    fn_count = defaultdict(int)
    for s in seeds:
        for did, g in gold.items():
            p = np.asarray(preds_by_seed[s][did])
            for pos in np.where((g == 0) & (p == 1))[0]:
                fp_count[(did, int(pos))] += 1
            for pos in np.where((g == 1) & (p == 0))[0]:
                fn_count[(did, int(pos))] += 1

    def summarize(cnt, kind):
        vals = np.array(list(cnt.values()))
        total = len(vals)
        in3 = int((vals == 3).sum())
        in2 = int((vals == 2).sum())
        in1 = int((vals == 1).sum())
        per_seed_avg = vals.sum() / len(seeds)
        print(f"  {kind}: union of positions wrong in >=1 seed = {total}   "
              f"(avg per seed = {per_seed_avg:.0f})")
        print(f"     wrong in all 3 seeds : {in3:5d}  ({100*in3/total:5.1f}% of union)  <- systematic, learnable defect")
        print(f"     wrong in exactly 2   : {in2:5d}  ({100*in2/total:5.1f}%)")
        print(f"     wrong in exactly 1   : {in1:5d}  ({100*in1/total:5.1f}%)  <- seed lottery / noise")
        # of a single seed's errors, how many are shared by all three?
        shared_of_perseed = 3 * in3 / vals.sum()
        print(f"     => of ONE seed's {kind}s, ~{100*shared_of_perseed:.1f}% are made by all three seeds")
        return dict(union=total, in3=in3, in2=in2, in1=in1)

    print(f"\narm = {arm_name}  seeds = {seeds}")
    fp_stats = summarize(fp_count, "FP")
    fn_stats = summarize(fn_count, "FN")
    return fp_stats, fn_stats


# ------------------------------------------------------------------ 3. burstiness
def cut3_burstiness(docs, pred, arm_name, window=5):
    print("=" * 100)
    print(f"CUT 3 — ERROR BURSTINESS   arm = {arm_name}   (window = +/-{window} tokens)")
    print("  a false cut is 'in a burst' if another false cut sits within "
          f"{window} tokens of it")
    print("=" * 100)
    total_fp = 0
    burst_fp = 0
    double_cuts = 0            # FP pairs 1-2 tokens apart (both FPs)
    exp_burst = 0.0            # expected bursty FPs if same #FPs were spread uniformly
    examples = []
    for d in docs:
        g = np.asarray(d["labels"])
        p = np.asarray(pred[d["doc_id"]])
        fp_pos = np.where((g == 0) & (p == 1))[0]
        k, n = len(fp_pos), len(g)
        total_fp += k
        if k >= 2:
            gaps_prev = np.diff(fp_pos)
            in_burst = np.zeros(k, bool)
            in_burst[1:] |= gaps_prev <= window
            in_burst[:-1] |= gaps_prev <= window
            burst_fp += int(in_burst.sum())
            double_cuts += int(((gaps_prev >= 1) & (gaps_prev <= 2)).sum())
            # chance level: k cuts thrown uniformly at n positions
            if n > 2 * window:
                p_neigh = 1.0 - (1.0 - (2 * window) / n) ** (k - 1)
                exp_burst += k * p_neigh
            # keep an example of the densest burst
            if (gaps_prev <= 2).any():
                i = int(np.argmax(gaps_prev <= 2))
                pos = int(fp_pos[i])
                lo, hi = max(0, pos - 6), min(n, pos + 9)
                toks = d["tokens"][lo:hi]
                marks = []
                for j, t in enumerate(toks, start=lo):
                    m = t
                    if p[j] == 1 and g[j] == 0:
                        m += "<[FALSE-CUT]"
                    elif p[j] == 1 and g[j] == 1:
                        m += "<[correct-cut]"
                    elif g[j] == 1:
                        m += "<[MISSED-cut]"
                    marks.append(m)
                examples.append((d["doc_id"], bucket(d["tokens"]), " ".join(marks)))
    print(f"\ntotal false cuts (FPs): {total_fp}")
    print(f"FPs in a burst (another FP within {window} tokens): {burst_fp}  "
          f"({100*burst_fp/max(total_fp,1):.1f}%)")
    print(f"expected if the same number of FPs were spread uniformly: "
          f"{exp_burst:.0f}  ({100*exp_burst/max(total_fp,1):.1f}%)")
    print(f"double-cuts (two false cuts only 1-2 tokens apart): {double_cuts} pairs")
    print("\nexamples of double-cut bursts (token<[FALSE-CUT] = model cut here, gold says no):")
    for did, gen, ex in examples[:4]:
        print(f"  {did} [{gen}]")
        print(f"    ... {ex} ...")
    return dict(total_fp=total_fp, burst_fp=burst_fp, exp_burst=exp_burst,
                double_cuts=double_cuts)


# --------------------------------------------------- 4. over/under-cut per genre
def cut4_cut_ratio(docs, pred, arm_name, known_ratio=1.093):
    print("=" * 100)
    print(f"CUT 4 — PREDICTED vs GOLD boundary counts per genre   arm = {arm_name}")
    print("  ratio > 1 means the model cuts MORE than the gold (over-cutting)")
    print("=" * 100)
    by = defaultdict(lambda: [0, 0, 0])   # genre -> [pred_b, gold_b, docs]
    per_doc_ratio = []
    for d in docs:
        g = int(sum(d["labels"]))
        p = int(sum(pred[d["doc_id"]]))
        b = bucket(d["tokens"])
        by[b][0] += p
        by[b][1] += g
        by[b][2] += 1
        if g:
            per_doc_ratio.append(p / g)
    tot_p = sum(v[0] for v in by.values())
    tot_g = sum(v[1] for v in by.values())
    print(f"\n{'genre':20s} {'docs':>5} {'pred_b':>7} {'gold_b':>7} {'ratio':>7}")
    order = ["general prose", "legal/constitution", "hadith/isnad", "genealogy",
             "cloze/exercise"]
    for gnr in order:
        if gnr not in by:
            continue
        p, g, nd = by[gnr]
        print(f"{gnr:20s} {nd:5d} {p:7d} {g:7d} {p/max(g,1):7.3f}")
    print(f"{'ALL':20s} {sum(v[2] for v in by.values()):5d} {tot_p:7d} {tot_g:7d} "
          f"{tot_p/tot_g:7.3f}")
    print(f"\ncorpus-level ratio {tot_p/tot_g:.3f} vs known corpus over-cut ratio "
          f"{known_ratio} (sanity check)")
    pr = np.array(per_doc_ratio)
    print(f"per-doc ratio: median {np.median(pr):.3f}, "
          f"docs over-cutting (>1) {int((pr>1).sum())}/{len(pr)}, "
          f"under-cutting (<1) {int((pr<1).sum())}/{len(pr)}")
    return tot_p / tot_g


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--battery", default="runs/prep_battery_2026-07-09",
                    help="battery dir with <arm>_s<seed>/ subdirs (read-only)")
    ap.add_argument("--arms", nargs="+",
                    default=["cons_baseline", "char_baseline", "enc_baseline"])
    ap.add_argument("--seeds", nargs="+", default=["42", "1", "2"])
    ap.add_argument("--pred", default="pred_dev_t050.csv",
                    choices=["pred_dev_t050.csv", "pred_dev_bestT.csv"])
    ap.add_argument("--main-arm", default="cons_baseline",
                    help="arm used for cuts 1, 3, 4 (seed = first of --seeds)")
    ap.add_argument("--window", type=int, default=5)
    args = ap.parse_args()

    docs = load_gold(args.gold)
    print(f"gold: {args.gold}  docs={len(docs)}  "
          f"tokens={sum(len(d['tokens']) for d in docs)}  "
          f"gold boundaries={sum(sum(d['labels']) for d in docs)}")

    # ---- load every requested arm x seed (skip missing files, say so)
    preds = {}   # (arm, seed) -> {doc_id: labels}
    for arm in args.arms:
        for s in args.seeds:
            path = os.path.join(args.battery, f"{arm}_s{s}", args.pred)
            if os.path.exists(path):
                preds[(arm, s)] = load_pred_csv(path)
            else:
                print(f"  [missing] {path}")

    main_key = (args.main_arm, args.seeds[0])
    main_pred = preds[main_key]
    main_name = f"{args.main_arm}_s{args.seeds[0]} @ {args.pred} ({args.battery})"

    rows = cut1_perdoc_f1(docs, main_pred, main_name)

    print("=" * 100)
    print("CUT 2 — CROSS-SEED SYSTEMATICITY (same recipe, seeds "
          f"{args.seeds}; error made by all seeds = real defect, by one = luck)")
    print("=" * 100)
    for arm in args.arms:
        by_seed = {s: preds[(arm, s)] for s in args.seeds if (arm, s) in preds}
        if len(by_seed) < 2:
            print(f"\narm = {arm}: fewer than 2 seeds on disk, skipped")
            continue
        cut2_systematicity(docs, by_seed, arm)

    cut3_burstiness(docs, main_pred, main_name, window=args.window)
    cut4_cut_ratio(docs, main_pred, main_name)
    return rows


if __name__ == "__main__":
    main()
