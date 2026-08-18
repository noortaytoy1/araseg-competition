"""MULTI-STRIDE HEATMAP decode probe (Noor's idea) — ANALYSIS/DECODE ONLY.

Question
--------
The canonical inference runs a fixed-width window over the document and averages
each token's P(boundary) over the (few) windows that cover it.  Does accumulating
MORE / DENSER views — and stitching them with a CENTER-WEIGHTED mean instead of a
flat mean — move dev macro-F1 versus the stride-90 baseline?

This module is PURELY ADDITIVE.  It does NOT edit predict.py, eval_local.py, or
the canonical stitching.  It re-implements a small fixed-stride window accumulator
here (needed because we must record each token's position-within-window for the
center-weighting), and it *imports and calls* the frozen pieces:
    - predict.predict_doc            (arm (a*) sanity reference: canonical decode)
    - eval_local.compute_metrics     (official-mirror per-doc macro-F1 scoring)
    - genre_buckets.bucket           (per-genre FP/pos)
    - diag_preposition.PREPOSITIONS  (the over-firing prep-slice definition)
so every number here is directly comparable to the on-record diagnostics.

No training.  One deterministic forward pass over the dev split per unique window
start; the four strides all read from ONE cached prob-per-(start) table so we run
the model exactly once per distinct window, never once per arm.

Arms (all on the SAME model, own-best threshold swept on the SAME grid)
    (a) baseline           stride=90, flat mean over covering windows
    (b) multi uniform      strides {30,45,60,90}, flat mean over ALL covering windows
    (c) multi center-wtd   strides {30,45,60,90}, triangular center-weighted mean
    (d) dense single       stride=30, flat mean over covering windows

Usage:
  PYTHONUTF8=1 python src/multi_stride_decode.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev data/NoPnx-NP_dev.jsonl \
      --train data/NoPnx-NP_train.jsonl \
      --task NoPnx-NP
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from eval_local import compute_metrics
from genre_buckets import bucket
from predict import predict_doc
from diag_preposition import PREPOSITIONS

PREP_SET = set(PREPOSITIONS)

# The four strides Noor asked for, plus the baseline single stride.
MULTI_STRIDES = [30, 45, 60, 90]
BASELINE_STRIDE = 90
DENSE_STRIDE = 30
THRESHOLD_GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30..0.70 step .05


# --------------------------------------------------------------------------- #
# Model forward: cache P(boundary) for every DISTINCT window start, for every  #
# stride we need.  Because the strides are nested-ish, we simply take the      #
# union of all window-start indices across strides and run the model once per  #
# distinct start.  Each start yields, per covered token, (prob, dist-to-center)#
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _forward_window(words, start, window, tokenizer, model, device, max_length):
    """Run the model on words[start:start+window]; return {abs_word_idx: prob}
    using the SAME last-subword-of-word reduction and softmax[:,1] as predict.py."""
    chunk = words[start:start + window]
    enc = tokenizer(chunk, is_split_into_words=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    word_ids = enc.word_ids(0)
    logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
    p_boundary = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    out = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
        if nxt != wid:  # last subword of this word
            out[start + wid] = float(p_boundary[pos])
    return out


def _window_starts(n, window, stride):
    """Window starts for effective `stride` that EXACTLY reproduce predict.py's
    coverage-driven stepping (so arm (a) is bit-identical to the current decode).

    predict.py starts at 0 and, after a window that covers through `covered_through`
    (the last word it saw, = min(start+window, n) - 1), steps to
        start = max(covered_through + 1 - overlap, start + 1)
    with overlap = window - stride, breaking once covered_through >= n-1.
    We replay that exact loop and return the ordered list of starts."""
    if n <= 0:
        return [0]
    overlap = window - stride
    starts = []
    start = 0
    while start < n:
        starts.append(start)
        covered_through = min(start + window, n) - 1  # last word index this window sees
        if covered_through >= n - 1:
            break
        start = max(covered_through + 1 - overlap, start + 1)
    return starts


@torch.no_grad()
def collect_stride_views(words, tokenizer, model, device, window, strides, max_length):
    """For a single document, return a dict:
         token_idx -> list of (prob, weight_center) tuples,
       one entry per (stride, covering-window).  weight_center is a triangular
       centrality weight in (0,1]: 1.0 at the window center, decaying linearly to
       ~1/(window/2) at the window edges.  Flat-mean arms ignore the weight.

    We run the model ONCE per DISTINCT window start (union across strides), then
    replay that cached window for every stride whose start-set contains it — so a
    token covered by multiple strides that share a start is not double-counted for
    that start, exactly matching "accumulate over all covering windows"."""
    n = len(words)
    # distinct starts, and which strides use each start (to weight per-stride views)
    starts_by_stride = {s: _window_starts(n, window, s) for s in strides}
    # multiplicity: how many strides place a window at this start
    start_mult = defaultdict(int)
    for s in strides:
        for st in starts_by_stride[s]:
            start_mult[st] += 1
    distinct_starts = sorted(start_mult)

    cache = {}  # start -> {abs_idx: prob}
    for st in distinct_starts:
        cache[st] = _forward_window(words, st, window, tokenizer, model, device, max_length)

    half = window / 2.0
    views = defaultdict(list)
    for st in distinct_starts:
        wmap = cache[st]
        mult = start_mult[st]  # this window counts once per stride that visits it
        for abs_idx, prob in wmap.items():
            off = abs_idx - st                    # 0..window-1 position in window
            # triangular centrality: 1.0 at center, -> ~2/window at the edges
            centrality = 1.0 - abs(off - half) / half
            centrality = max(centrality, 1.0 / half)  # avoid zero weight at edge
            for _ in range(mult):
                views[abs_idx].append((prob, centrality))
    return views, starts_by_stride


def single_stride_probs(words, tokenizer, model, device, window, stride, max_length):
    """Flat-mean P(boundary) per token for ONE fixed stride (arms a and d)."""
    n = len(words)
    starts = _window_starts(n, window, stride)
    acc = defaultdict(list)
    for st in starts:
        wmap = _forward_window(words, st, window, tokenizer, model, device, max_length)
        for abs_idx, prob in wmap.items():
            acc[abs_idx].append(prob)
    return np.array([np.mean(acc[i]) if acc[i] else 0.0 for i in range(n)])


def multi_probs_from_views(views, n, center_weighted):
    """Collapse per-token view lists into one P(boundary) per token.
    center_weighted=False -> flat mean; True -> centrality-weighted mean."""
    out = np.zeros(n)
    for i in range(n):
        vs = views.get(i)
        if not vs:
            out[i] = 0.0
            continue
        if center_weighted:
            num = sum(p * w for p, w in vs)
            den = sum(w for _, w in vs)
            out[i] = num / den if den else 0.0
        else:
            out[i] = sum(p for p, _ in vs) / len(vs)
    return out


# --------------------------------------------------------------------------- #
# Scoring: build a prediction CSV-equivalent dict and score with the frozen    #
# eval_local.compute_metrics (per-doc macro-F1).                               #
# --------------------------------------------------------------------------- #
def labels_from_probs(docs, prob_by_doc, threshold):
    """Threshold per-token probs into 0/1, forcing [PAR]/\\n tokens to 0 exactly
    as predict.py does.  Returns {doc_id: [0/1,...]}."""
    pred = {}
    for d in docs:
        toks = d["tokens"]
        p = prob_by_doc[d["doc_id"]]
        lab = (p >= threshold).astype(int)
        for i, w in enumerate(toks):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        pred[d["doc_id"]] = lab.tolist()
    return pred


def sweep_threshold(docs, gold, prob_by_doc, grid):
    """Return (best_threshold, best_metrics_dict, full_curve) by macro-F1."""
    best_t, best_m, curve = None, None, []
    for t in grid:
        pred = labels_from_probs(docs, prob_by_doc, t)
        m = compute_metrics(gold, pred)
        curve.append((t, m["macro_precision"], m["macro_recall"], m["macro_f1"]))
        if best_m is None or m["macro_f1"] > best_m["macro_f1"]:
            best_t, best_m = t, m
    return best_t, best_m, curve


# --------------------------------------------------------------------------- #
# Diagnostics at the arm's own-best threshold: per-genre FP/pos + prep-slice.  #
# Mirrors diag_preposition.py so numbers line up with the on-record diagnostic.#
# --------------------------------------------------------------------------- #
def arm_diagnostics(docs, prob_by_doc, threshold):
    genre = defaultdict(lambda: dict(pos=0, gold=0, pred=0, fp=0, fn=0))
    prep_pos = 0
    prep_fp_fire = 0          # prep-next positions where gold=0 and we fired (over-fire)
    prep_fp_probs = []
    tot_fp = tot_fn = 0
    fp_prep = fn_prep = 0
    for d in docs:
        toks, gold = d["tokens"], d["labels"]
        p = prob_by_doc[d["doc_id"]]
        g_bucket = bucket(toks)
        n = len(toks)
        gs = genre[g_bucket]
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            nxt = toks[i + 1] if i + 1 < n else None
            is_prep = nxt in PREP_SET
            pi = float(p[i]); gi = int(gold[i]); pred = 1 if pi >= threshold else 0
            gs["pos"] += 1; gs["gold"] += gi; gs["pred"] += pred
            fp = (pred == 1 and gi == 0); fn = (pred == 0 and gi == 1)
            if fp:
                gs["fp"] += 1; tot_fp += 1
            if fn:
                gs["fn"] += 1; tot_fn += 1
            if is_prep:
                prep_pos += 1
                if fp:
                    fp_prep += 1; prep_fp_fire += 1; prep_fp_probs.append(pi)
                if fn:
                    fn_prep += 1
    return dict(genre=genre, prep_pos=prep_pos, prep_fp_fire=prep_fp_fire,
                prep_fp_probs=prep_fp_probs, tot_fp=tot_fp, tot_fn=tot_fn,
                fp_prep=fp_prep, fn_prep=fn_prep)


def print_arm(name, best_t, best_m, curve, diag, baseline_f1=None):
    f1 = best_m["macro_f1"] * 100
    p = best_m["macro_precision"] * 100
    r = best_m["macro_recall"] * 100
    line = f"\n=== ARM {name} ===  best_thr={best_t}  F1={f1:.3f}  P={p:.2f}  R={r:.2f}"
    if baseline_f1 is not None:
        line += f"   Δvs(a)={f1 - baseline_f1:+.3f}"
    print(line)
    print("  threshold sweep (t: P R F1):")
    for t, pp, rr, ff in curve:
        star = "  <-- best" if t == best_t else ""
        print(f"    {t:.2f}: {pp*100:6.2f} {rr*100:6.2f} {ff*100:6.3f}{star}")
    d = diag
    print(f"  errors @best_thr: FP={d['tot_fp']}  FN={d['tot_fn']}  "
          f"(prep-adjacent: FP={d['fp_prep']} FN={d['fn_prep']})")
    fpp = np.asarray(d["prep_fp_probs"])
    over = f"n={fpp.size} mean={fpp.mean():.3f}" if fpp.size else "n=0"
    print(f"  OVER-FIRING prep-slice: prep-next positions={d['prep_pos']}  "
          f"FP-fires={d['prep_fp_fire']}  ({over})")
    print(f"  {'genre':20s} {'pos':>7} {'goldB':>6} {'predB':>6} {'FP':>5} {'FN':>5} "
          f"{'FP/pos':>7} {'FN/pos':>7}")
    for g, s in sorted(d["genre"].items(), key=lambda kv: -kv[1]["pos"]):
        pos = max(s["pos"], 1)
        print(f"  {g:20s} {s['pos']:7d} {s['gold']:6d} {s['pred']:6d} {s['fp']:5d} "
              f"{s['fn']:5d} {s['fp']/pos:7.4f} {s['fn']/pos:7.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--train", default=None, help="unused; kept for CLI parity")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed-floor", type=float, default=0.45,
                    help="multi-seed noise floor for the plain verdict (report-only)")
    ap.add_argument("--json-out", default=None, help="optional path to dump summary JSON")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device={device}  model={args.model}")
    print(f"# window={args.window}  strides multi={MULTI_STRIDES}  baseline={BASELINE_STRIDE}  dense={DENSE_STRIDE}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    docs = load_jsonl(args.dev)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    print(f"# dev docs={len(docs)}")

    # Per-doc probability tables for each arm ------------------------------------
    prob_a, prob_b, prob_c, prob_d = {}, {}, {}, {}
    # canonical reference (predict.py at its default overlap=60 -> effective stride 120)
    prob_canon = {}
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        n = len(words)
        # (a) baseline stride 90, flat mean
        prob_a[d["doc_id"]] = single_stride_probs(
            words, tokenizer, model, device, args.window, BASELINE_STRIDE, args.max_length)
        # (d) dense single stride 30, flat mean
        prob_d[d["doc_id"]] = single_stride_probs(
            words, tokenizer, model, device, args.window, DENSE_STRIDE, args.max_length)
        # (b)/(c) multi-stride views (shared forward), flat and center-weighted
        views, _ = collect_stride_views(
            words, tokenizer, model, device, args.window, MULTI_STRIDES, args.max_length)
        prob_b[d["doc_id"]] = multi_probs_from_views(views, n, center_weighted=False)
        prob_c[d["doc_id"]] = multi_probs_from_views(views, n, center_weighted=True)
        # canonical predict_doc reference (overlap=60)
        prob_canon[d["doc_id"]] = predict_doc(
            words, tokenizer, model, device, args.window, 60, args.max_length)

    arms = [
        ("(a) baseline stride=90 mean", prob_a),
        ("(b) multi {30,45,60,90} uniform mean", prob_b),
        ("(c) multi {30,45,60,90} center-weighted", prob_c),
        ("(d) dense single stride=30 mean", prob_d),
    ]

    # sweep + score each arm
    results = {}
    for name, pb in arms:
        bt, bm, curve = sweep_threshold(docs, gold, pb, THRESHOLD_GRID)
        diag = arm_diagnostics(docs, pb, bt)
        results[name] = (bt, bm, curve, diag)

    # canonical reference (report-only; not one of the 4 arms)
    ct, cm, ccurve = sweep_threshold(docs, gold, prob_canon, THRESHOLD_GRID)

    baseline_f1 = results["(a) baseline stride=90 mean"][1]["macro_f1"] * 100

    print("\n########################  MULTI-STRIDE HEATMAP RESULTS  ########################")
    for name, pb in arms:
        bt, bm, curve, diag = results[name]
        bl = None if name.startswith("(a)") else baseline_f1
        print_arm(name, bt, bm, curve, diag, baseline_f1=bl)

    print("\n=== REFERENCE (not an arm): canonical predict_doc overlap=60 (eff. stride 120) ===")
    print(f"  best_thr={ct}  F1={cm['macro_f1']*100:.3f}  P={cm['macro_precision']*100:.2f}  "
          f"R={cm['macro_recall']*100:.2f}   Δvs(a)={cm['macro_f1']*100 - baseline_f1:+.3f}")

    # -------------------------------------------------- SUMMARY + PLAIN VERDICT
    print("\n########################  SUMMARY: Δ vs (a) stride-90 baseline  ########################")
    print(f"{'arm':44s} {'best_thr':>8} {'F1':>8} {'ΔF1':>8}  {'clears ±%.2f floor?' % args.seed_floor}")
    floor = args.seed_floor
    for name, pb in arms:
        bt, bm, curve, diag = results[name]
        f1 = bm["macro_f1"] * 100
        dv = f1 - baseline_f1
        clears = "n/a (baseline)" if name.startswith("(a)") else \
            ("YES" if abs(dv) >= floor else "no (sub-floor)")
        print(f"{name:44s} {bt:8.2f} {f1:8.3f} {dv:+8.3f}  {clears}")

    if args.json_out:
        summary = {
            "model": args.model, "dev": args.dev, "window": args.window,
            "strides_multi": MULTI_STRIDES, "seed_floor": floor,
            "baseline_f1": baseline_f1,
            "arms": {
                name: {
                    "best_thr": results[name][0],
                    "F1": results[name][1]["macro_f1"] * 100,
                    "P": results[name][1]["macro_precision"] * 100,
                    "R": results[name][1]["macro_recall"] * 100,
                    "delta_vs_a": results[name][1]["macro_f1"] * 100 - baseline_f1,
                } for name, _ in arms
            },
            "canonical_overlap60": {
                "best_thr": ct, "F1": cm["macro_f1"] * 100,
                "delta_vs_a": cm["macro_f1"] * 100 - baseline_f1,
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n# wrote summary -> {args.json_out}")

    print("\n# done")


if __name__ == "__main__":
    main()
