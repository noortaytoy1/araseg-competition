"""DEV-SIDE DECORRELATION GATE (additive, CPU-only, dev used for EVALUATION
only — diagnostic of an inference-time signal).

Follow-up to diag_simmatrix.py (train gate: block-contrast C(i), mid layer 8,
w=3 was the best config, AUC 0.7641). Question here: does C(i) separate the
tagger's CONFIDENT FALSE POSITIVES from its CONFIDENT TRUE POSITIVES on dev,
i.e. does the 2D similarity signal see differently exactly where the tagger
is blind?

  1. From cached decode-grid probs (probs/union-NoPnx-NP-arabertv02-s42_dev
     .npz) + gold (data/NoPnx-NP_dev.jsonl): all confident-FP positions
     (p >= 0.8, gold 0; n = 2,236 on record) and ~1,000 sampled confident-TP
     positions (p >= 0.8, gold 1; seed 42).
  2. CPU-encode ONLY the dev windows containing those positions (standard
     windowing 180/60 as in diag_simmatrix; each target assigned to the
     window where it sits most centrally; windows deduped). Mid-layer 8
     hidden states -> cosine matrix -> C(i) at the target edges, w in {3,5}.
  3. THE NUMBER: ROC-AUC of C(i) with confident-TP as the positive class
     (same orientation as the train gate's bound-vs-confFP column: high
     C(i) should mark REAL boundaries). Plus means/medians per group.

PRE-REGISTERED DECISION RULE (w=3, mid layer 8):
  AUC >= 0.62  -> the 2D U-Net arm gets built (sees differently where the
                  tagger is blind — ensemble value)
  AUC <  0.55  -> PARK the 2D arm (same seductive edges; a U-Net would
                  duplicate the tagger's errors); mid-layer LAYER-FUSION arm
                  remains as the salvage
  0.55 - 0.62  -> report for Noor's call

Usage (CPU only):
  CUDA_VISIBLE_DEVICES=-1 python src/diag_simmatrix_devgate.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from data import load_jsonl
from diag_simmatrix import (block_contrast, cosine_matrix, encode_window,
                            enumerate_windows, load_encoder, rank_auc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="runs/union-NoPnx-NP-arabertv02-s42")
    ap.add_argument("--jsonl", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--probs", default="probs/union-NoPnx-NP-arabertv02-s42_dev.npz")
    ap.add_argument("--mid-layer", type=int, default=8)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--w-list", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--conf-thresh", type=float, default=0.8)
    ap.add_argument("--n-tp", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wmax = max(args.w_list)
    docs = load_jsonl(args.jsonl)
    z = np.load(args.probs)
    assert all(d["doc_id"] in z for d in docs), "probs npz missing dev docs"

    # ---- 1. target positions from the cached decode-grid probs -------------
    fp_targets, tp_pool = [], []
    for di, d in enumerate(docs):
        p = z[d["doc_id"]]
        g = np.asarray(d["labels"], dtype=np.int8)
        assert len(p) == len(g)
        conf = p >= args.conf_thresh
        for pos in np.flatnonzero(conf & (g == 0)):
            fp_targets.append((di, int(pos)))
        for pos in np.flatnonzero(conf & (g == 1)):
            tp_pool.append((di, int(pos)))
    rng = np.random.default_rng(args.seed)
    take = min(args.n_tp, len(tp_pool))
    tp_targets = [tp_pool[j] for j in
                  sorted(rng.choice(len(tp_pool), size=take, replace=False))]
    print(f"# {len(docs)} dev docs; confident-FP = {len(fp_targets)}, "
          f"confident-TP pool = {len(tp_pool)} -> sampled {take} "
          f"(seed={args.seed}, p>={args.conf_thresh})", flush=True)

    # ---- 2. assign each target to its most-central window, dedupe ----------
    wins = enumerate_windows(docs, args.window, args.overlap,
                             min_len=2 * wmax + 2)
    wins_by_doc = {}
    for (di, start, L) in wins:
        wins_by_doc.setdefault(di, []).append((start, L))

    def candidates(di, gpos):
        """Windows where gpos is a valid wmax edge, most-central first."""
        out = []
        for (start, L) in wins_by_doc.get(di, []):
            i = gpos - start
            if wmax - 1 <= i <= L - 1 - wmax:
                out.append((abs(i - (L - 1) / 2.0), start, L))
        return [(s, L) for _, s, L in sorted(out)]

    targets = [(di, pos, 1) for di, pos in tp_targets] + \
              [(di, pos, 0) for di, pos in fp_targets]
    assign, dropped_margin = {}, 0
    for di, pos, lab in targets:
        cand = candidates(di, pos)
        if not cand:
            dropped_margin += 1
            continue
        key = (di,) + cand[0]
        assign.setdefault(key, []).append((pos, lab, cand))
    print(f"# {len(targets)} targets -> {len(assign)} unique windows to "
          f"encode ({dropped_margin} dropped: no window with wmax={wmax} "
          f"margin)", flush=True)

    # ---- 3. encode deduped windows, C(i) at target edges -------------------
    tokenizer, model = load_encoder(args.model)
    print(f"# encoder: {args.model} (mid layer = {args.mid_layer})", flush=True)

    rows = []          # (label, {w: C}, p_single, p_cached_absdiff)
    dropped_trunc = 0
    order = sorted(assign)
    t0 = time.time()
    for k, key in enumerate(order):
        di, start, L0 = key
        d = docs[di]
        chunk = d["tokens"][start:start + L0]
        _, v_mid, p_tag = encode_window(chunk, tokenizer, model,
                                        args.mid_layer)
        L = v_mid.shape[0]                       # truncation-trimmed
        S = cosine_matrix(v_mid)
        cw = {}
        for w in args.w_list:
            idx, C = block_contrast(S, w)
            cw[w] = (idx[0] if len(idx) else 0, C)
        p_cache = z[d["doc_id"]]
        for pos, lab, cand in assign[key]:
            i = pos - start
            if i > L - 1 - wmax:                 # lost to truncation
                dropped_trunc += 1
                continue
            vals = {}
            for w in args.w_list:
                lo, C = cw[w]
                vals[w] = float(C[i - lo])
            rows.append((lab, vals, float(p_tag[i]),
                         abs(float(p_tag[i]) - float(p_cache[pos]))))
        if (k + 1) % 50 == 0 or k == len(order) - 1:
            el = time.time() - t0
            print(f"  [{k + 1}/{len(order)}] {el:.0f}s elapsed, eta "
                  f"{el / (k + 1) * (len(order) - k - 1):.0f}s", flush=True)

    labs = np.array([r[0] for r in rows], dtype=bool)   # True = confident-TP
    p_single = np.array([r[2] for r in rows])
    p_diff = np.array([r[3] for r in rows])
    print(f"\n# scored {len(rows)} targets ({int(labs.sum())} TP, "
          f"{int((~labs).sum())} FP); dropped {dropped_margin} (margin) + "
          f"{dropped_trunc} (truncation)")
    print(f"# grid sanity: mean |p_single - p_cached| = {p_diff.mean():.4f} "
          f"(single-window vs overlap-averaged decode grid); "
          f"mean p_single: TP {p_single[labs].mean():.3f}, "
          f"FP {p_single[~labs].mean():.3f}")

    # ---- 4. THE NUMBER ------------------------------------------------------
    print("\n========== DEV DECORRELATION GATE: C(i) confident-TP vs "
          "confident-FP (mid layer %d) ==========" % args.mid_layer)
    print(f"{'w':>3} {'AUC(TP+)':>9} {'n_TP':>6} {'n_FP':>6} "
          f"{'meanC|TP':>9} {'meanC|FP':>9} {'medC|TP':>9} {'medC|FP':>9}")
    auc_by_w = {}
    for w in args.w_list:
        C = np.array([r[1][w] for r in rows])
        auc = rank_auc(C, labs)
        auc_by_w[w] = auc
        print(f"{w:3d} {auc:9.4f} {int(labs.sum()):6d} "
              f"{int((~labs).sum()):6d} {C[labs].mean():9.4f} "
              f"{C[~labs].mean():9.4f} {np.median(C[labs]):9.4f} "
              f"{np.median(C[~labs]):9.4f}")

    # ---- 5. pre-registered decision rule (w=3) ------------------------------
    a = auc_by_w.get(3, float("nan"))
    print("\n================ PRE-REGISTERED DECISION RULE (w=3) "
          "================")
    print(f"AUC = {a:.4f}")
    if a >= 0.62:
        print("VERDICT: AUC >= 0.62 -> the 2D U-Net arm gets built (it sees "
              "differently where the tagger is blind — ensemble value).")
    elif a < 0.55:
        print("VERDICT: AUC < 0.55 -> PARK the 2D arm (same seductive edges; "
              "a U-Net would duplicate the tagger's errors) — the mid-layer "
              "LAYER-FUSION arm remains as the salvage.")
    else:
        print("VERDICT: 0.55 <= AUC < 0.62 -> report for Noor's call.")


if __name__ == "__main__":
    main()
