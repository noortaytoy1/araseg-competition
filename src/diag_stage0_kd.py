# -*- coding: utf-8 -*-
"""STAGE-0 KD feasibility diagnostic (external critique's decision gate).

Question: does the PA teacher's cross-view folded probability carry signal the
NoPnx-NP student does not already have?  Five numbers on the TRAIN docs
(held-in is fine -- this is a feasibility gate, not an eval):

  1. STUDENT CONFIDENT FALSE CUTS: positions with student P >= --fp-thresh
     (default 0.8) and gold = 0, plus all gold = 1 (true boundaries).
  2. THE AUC: ROC-AUC of the TEACHER's folded probability for separating
     student confident-FPs from true boundaries.
     Decision rule (critique, verbatim): AUC ~0.5 -> KD will null
     (deprioritize); AUC >> 0.5 (say >0.65) -> KD has frontier-pushing
     potential.
  3. MONOTONICITY: is the teacher's folded prob a monotone transform of the
     student's own prob at the same positions?  Spearman correlation on a
     large sample + fraction of position PAIRS where teacher and student
     RANK-DISAGREE (teacher re-ranks).  High monotonicity -> KD =
     calibration -> null.
  4. COINCIDENCE: fraction of gold NoPnx boundaries whose fold-group contains
     a punctuation token in the PA view (>= 95% -> redundant-with-gold trap).
  5. SHARPNESS: histogram of folded teacher probs (<0.05 / 0.05-0.95 / >0.95)
     on train (the memorization check).

ADDITIVE + CPU-ONLY by design: CUDA is masked before torch is imported (the
GPU belongs to the frontier battery).  Writes NOTHING under runs/.  If the
immutable xview teacher cache (runs/xview_teacher_2026-07-10) exists and its
meta.json attests guard g4, the folded teacher probs are REUSED read-only;
otherwise they are recomputed here through the ONE fold implementation of
record (augment.deletion_mask / project_labels routing via
xview_teacher_cache.route_map + fold_probs) with the same g4-style gold
round-trip asserted per doc before anything is used.  Optional --cache-dir
(e.g. a scratchpad) memoizes teacher/student probs between invocations.

  PYTHONUTF8=1 CUDA_VISIBLE_DEVICES= python src/diag_stage0_kd.py \
      --cache-dir /path/to/scratchpad/stage0_kd
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch import; battery owns the GPU

import numpy as np

from augment import deletion_mask, project_labels
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from xview_teacher_cache import fold_probs, or_fold_labels, route_map

XVIEW_CACHE_DIR = os.path.join("runs", "xview_teacher_2026-07-10")
XVIEW_PROBS_NAME = "teacher_pa_fold_NoPnx-NP_train.jsonl"


# ------------------------------------------------------------------ rank utils
def rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties, pure numpy (scipy-equivalent)."""
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    sa = a[sorter]
    obs = np.r_[True, sa[1:] != sa[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties counted 1/2)."""
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata_avg(np.asarray(scores, dtype=np.float64))
    u = r[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata_avg(x), rankdata_avg(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def pair_rank_disagreement(s: np.ndarray, t: np.ndarray, n_pairs: int,
                           seed: int) -> dict:
    """Random position pairs: fraction where student and teacher STRICTLY
    disagree on the order (teacher re-ranks), and fraction tied in either."""
    rng = np.random.default_rng(seed)
    n = len(s)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    ds = s[i] - s[j]
    dt = t[i] - t[j]
    disagree = int(np.sum(ds * dt < 0))
    tied = int(np.sum((ds == 0) | (dt == 0)))
    m = len(i)
    return {"n_pairs": m,
            "frac_rank_disagree": disagree / m,
            "frac_tied_either": tied / m}


# --------------------------------------------------------------- alignment/g4
def build_alignment(src_docs: list, tgt_by_id: dict, fold_mode: str) -> dict:
    """doc_id -> dict(keep, route, n_slots, gold); g4-style round-trip asserted
    per doc through the fold implementation of record before anything is used."""
    aligned = {}
    for d in src_docs:
        t = tgt_by_id[d["doc_id"]]
        toks, labs, gold = d["tokens"], list(d["labels"]), list(t["labels"])
        keep = deletion_mask(toks, t["tokens"])  # asserts subsequence
        route = route_map(toks, keep)
        n_slots = sum(keep)
        if n_slots != len(t["tokens"]) or n_slots != len(gold):
            raise SystemExit(f"g4-style FAIL {d['doc_id']}: slot count mismatch")
        if project_labels(toks, labs, keep) != gold:
            raise SystemExit(f"g4-style FAIL {d['doc_id']}: project_labels != gold")
        if or_fold_labels(labs, route, n_slots) != gold:
            raise SystemExit(f"g4-style FAIL {d['doc_id']}: routing OR fold != gold")
        if [int(p >= 0.5) for p in
                fold_probs([float(x) for x in labs], route, n_slots, fold_mode)] != gold:
            raise SystemExit(f"g4-style FAIL {d['doc_id']}: fold_probs({fold_mode}) "
                             "on gold indicators != gold")
        aligned[d["doc_id"]] = {"keep": keep, "route": route,
                                "n_slots": n_slots, "gold": gold}
    return aligned


# ------------------------------------------------------------------- inference
def _load_model(path: str):
    import torch  # lazy; CUDA already masked at module import
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path).to("cpu").eval()
    return tok, model


def run_model_on_docs(model_path: str, docs: list, window: int, overlap: int,
                      max_length: int, tag: str) -> dict:
    """doc_id -> list P(boundary) per token on the docs' own grid, canonical
    predict.predict_doc stitching, [PAR]/\\n forced to 0 (mirrors predict.py)."""
    from predict import predict_doc  # lazy (imports torch)
    tok, model = _load_model(model_path)
    out = {}
    t0 = time.time()
    for n, d in enumerate(docs, 1):
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tok, model, "cpu", window, overlap, max_length)
        out[d["doc_id"]] = [0.0 if w == PARAGRAPH_TOKEN else float(x)
                            for w, x in zip(d["tokens"], p)]
        if n % 25 == 0 or n == len(docs):
            print(f"   [{tag}] {n}/{len(docs)} docs, {time.time() - t0:.0f}s elapsed",
                  flush=True)
    return out


def cached(cache_dir: str | None, name: str, ident: dict, compute):
    """Memoize a {doc_id: list[float]} mapping as JSON in cache_dir (optional).
    The identity dict must match for a cache hit (stale caches are ignored)."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("ident") == ident:
                print(f"   cache hit: {path}")
                return blob["probs"]
            print(f"   cache STALE (ident mismatch), recomputing: {path}")
    probs = compute()
    if cache_dir:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ident": ident, "probs": probs}, f)
        print(f"   cached -> {path}")
    return probs


def load_xview_cache(cache_dir: str, tgt_docs: list, fold_mode: str,
                     window: int, overlap: int) -> dict | None:
    """Reuse the immutable xview teacher cache if present AND meta.json attests
    g4 AND its build parameters match this diagnostic's. Read-only."""
    probs_path = os.path.join(cache_dir, XVIEW_PROBS_NAME)
    meta_path = os.path.join(cache_dir, "meta.json")
    if not (os.path.exists(probs_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    g4 = meta.get("guards", {}).get("g4", "") or meta.get("round_trip_guard", "")
    if "PASSED" not in g4:
        print(f"   xview cache found but g4 NOT attested ({g4!r}) -- ignoring")
        return None
    if meta.get("fold_mode") != fold_mode or meta.get("window") != window \
            or meta.get("overlap") != overlap:
        print("   xview cache found but fold/window/overlap differ -- ignoring")
        return None
    rows = load_jsonl(probs_path)
    probs = {r["doc_id"]: r["teacher_probs"] for r in rows}
    tgt_by_id = {d["doc_id"]: d for d in tgt_docs}
    if set(probs) != set(tgt_by_id) or any(
            len(probs[i]) != len(tgt_by_id[i]["tokens"]) for i in probs):
        print("   xview cache doc ids / lengths do not match tgt -- ignoring")
        return None
    print(f"   REUSING immutable xview teacher cache: {probs_path}")
    print(f"   (g4 attestation: {g4})")
    return probs


# ------------------------------------------------------------------ punctuation
def is_punct(tok: str) -> bool:
    """PA-view punctuation: non-empty, not the paragraph token, and every char
    non-alphanumeric (covers . ؟ ! ؛ ، quotes brackets dashes dot-leaders)."""
    return tok != PARAGRAPH_TOKEN and len(tok) > 0 and \
        all(not c.isalnum() for c in tok)


# -------------------------------------------------------------------------- main
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="runs/opus_trdev_2026-07-08/PA_base_s42",
                    help="PA teacher checkpoint (train-only *_base_*)")
    ap.add_argument("--student",
                    default="runs/frontier_battery_2026-07-09_v3/cons_baseline_s42",
                    help="NoPnx-NP student checkpoint")
    ap.add_argument("--src", default="data/PA_train.jsonl")
    ap.add_argument("--tgt", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--xview-cache", default=XVIEW_CACHE_DIR,
                    help="immutable teacher-cache dir to REUSE if g4-attested")
    ap.add_argument("--cache-dir", default=None,
                    help="scratch dir to memoize computed probs (never runs/)")
    ap.add_argument("--fp-thresh", type=float, default=0.8)
    ap.add_argument("--fold", default="max", choices=["max", "noisy-or", "sum-cap"])
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--n-pairs", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None, help="optional results JSON path")
    args = ap.parse_args()

    if args.cache_dir and os.path.normpath(args.cache_dir).replace("\\", "/")\
            .lstrip("./").startswith("runs/"):
        raise SystemExit("--cache-dir must not be under runs/ (append-only)")

    # provenance guards mirrored from xview_teacher_cache (g0/g1 spirit)
    tb = os.path.basename(os.path.normpath(args.teacher))
    if "_trdev_" in tb or "_base_" not in tb:
        raise SystemExit(f"teacher basename {tb!r} must be *_base_* and not *_trdev_*")
    for p in (args.src, args.tgt):
        b = os.path.basename(p)
        if not b.endswith("_train.jsonl") or "trdev" in b:
            raise SystemExit(f"{b!r}: TRAIN docs only (no dev/test/trdev)")

    print("=== STAGE-0 KD DIAGNOSTIC (CPU only; feasibility gate, not an eval) ===")
    src_docs = load_jsonl(args.src)
    tgt_docs = load_jsonl(args.tgt)
    tgt_by_id = {d["doc_id"]: d for d in tgt_docs}
    if set(d["doc_id"] for d in src_docs) != set(tgt_by_id):
        raise SystemExit("src/tgt doc_id sets differ")
    print(f"docs: {len(src_docs)} in each view")

    print("aligning + g4-style gold round-trip (fold of record) ...")
    aligned = build_alignment(src_docs, tgt_by_id, args.fold)
    print(f"   PASSED {len(aligned)}/{len(src_docs)} docs")

    # ---- teacher folded probs on the NoPnx-NP grid --------------------------
    print("teacher folded probs ...")
    teacher = load_xview_cache(args.xview_cache, tgt_docs, args.fold,
                               args.window, args.overlap)
    if teacher is None:
        ident = {"model": args.teacher, "src": args.src, "tgt": args.tgt,
                 "window": args.window, "overlap": args.overlap,
                 "max_length": args.max_length, "fold": args.fold}

        def compute_teacher():
            raw = run_model_on_docs(args.teacher, src_docs, args.window,
                                    args.overlap, args.max_length, "teacher/PA")
            return {did: fold_probs(raw[did], aligned[did]["route"],
                                    aligned[did]["n_slots"], args.fold)
                    for did in raw}

        teacher = cached(args.cache_dir, "teacher_pa_fold_NoPnx-NP_train.json",
                         ident, compute_teacher)

    # ---- student probs on its own NoPnx-NP grid ------------------------------
    print("student probs ...")
    ident_s = {"model": args.student, "tgt": args.tgt, "window": args.window,
               "overlap": args.overlap, "max_length": args.max_length}
    student = cached(args.cache_dir, "student_NoPnx-NP_train.json", ident_s,
                     lambda: run_model_on_docs(args.student, tgt_docs, args.window,
                                               args.overlap, args.max_length,
                                               "student/NoPnx-NP"))

    # ---- flat arrays ----------------------------------------------------------
    order = [d["doc_id"] for d in tgt_docs]
    t_arr = np.concatenate([np.asarray(teacher[i], dtype=np.float64) for i in order])
    s_arr = np.concatenate([np.asarray(student[i], dtype=np.float64) for i in order])
    g_arr = np.concatenate([np.asarray(tgt_by_id[i]["labels"], dtype=np.int64)
                            for i in order])
    n_tok = len(g_arr)
    assert len(t_arr) == len(s_arr) == n_tok

    # ---- [1] student confident false cuts + true boundaries ------------------
    conf_fp = (s_arr >= args.fp_thresh) & (g_arr == 0)
    gold1 = g_arr == 1
    n_fp, n_pos = int(conf_fp.sum()), int(gold1.sum())

    # ---- [2] the AUC ----------------------------------------------------------
    sel = conf_fp | gold1
    auc = roc_auc(t_arr[sel], gold1[sel])
    mean_t_fp = float(t_arr[conf_fp].mean()) if n_fp else float("nan")
    mean_t_pos = float(t_arr[gold1].mean()) if n_pos else float("nan")

    # ---- [3] monotonicity -----------------------------------------------------
    rho_all = spearman(s_arr, t_arr)
    pairs = pair_rank_disagreement(s_arr, t_arr, args.n_pairs, args.seed)
    rho_dec = spearman(s_arr[sel], t_arr[sel]) if sel.sum() > 2 else float("nan")

    # ---- [4] coincidence ------------------------------------------------------
    n_b = n_b_punct = n_b_par = n_b_any_strip = 0
    for d in src_docs:
        a = aligned[d["doc_id"]]
        toks, keep, route = d["tokens"], a["keep"], a["route"]
        grp_punct = [False] * a["n_slots"]
        grp_par = [False] * a["n_slots"]
        grp_any = [False] * a["n_slots"]
        for i, tok in enumerate(toks):
            k = route[i]
            if k < 0 or keep[i]:
                continue
            grp_any[k] = True
            if tok == PARAGRAPH_TOKEN:
                grp_par[k] = True
            elif is_punct(tok):
                grp_punct[k] = True
        for j, lab in enumerate(a["gold"]):
            if lab == 1:
                n_b += 1
                n_b_punct += grp_punct[j]
                n_b_par += grp_punct[j] or grp_par[j]
                n_b_any_strip += grp_any[j]
    frac_punct = n_b_punct / max(n_b, 1)
    frac_punct_or_par = n_b_par / max(n_b, 1)
    frac_any_strip = n_b_any_strip / max(n_b, 1)

    # ---- [5] sharpness --------------------------------------------------------
    lo = int((t_arr < 0.05).sum())
    hi = int((t_arr > 0.95).sum())
    mid = n_tok - lo - hi

    # ---- report ---------------------------------------------------------------
    print("\n=== THE FIVE NUMBERS ===")
    print(f"[1] STUDENT CONFIDENT FALSE CUTS (P>={args.fp_thresh}, gold=0): "
          f"{n_fp}   |   true boundaries (gold=1): {n_pos}   "
          f"(of {n_tok} NoPnx-NP train positions)")
    print(f"[2] THE AUC (teacher folded prob separates conf-FPs from true "
          f"boundaries): {auc:.4f}")
    print(f"      mean teacher prob at conf-FPs: {mean_t_fp:.4f}   "
          f"at true boundaries: {mean_t_pos:.4f}")
    print(f"[3] MONOTONICITY: Spearman(student, teacher) all {n_tok} positions: "
          f"{rho_all:.4f}   on decision set (conf-FP + gold1): {rho_dec:.4f}")
    print(f"      random pairs n={pairs['n_pairs']}: teacher RANK-DISAGREES "
          f"(re-ranks) on {pairs['frac_rank_disagree']:.4f}; "
          f"tied in either on {pairs['frac_tied_either']:.4f}")
    print(f"[4] COINCIDENCE: gold NoPnx boundaries whose fold-group contains a "
          f"PA punctuation token: {n_b_punct}/{n_b} = {100 * frac_punct:.2f}%")
    print(f"      incl. \\n paragraph tokens: {100 * frac_punct_or_par:.2f}%   "
          f"any stripped token at all: {100 * frac_any_strip:.2f}%")
    print(f"[5] SHARPNESS of folded teacher probs (n={n_tok}):  "
          f"p<0.05: {lo} ({100 * lo / n_tok:.2f}%)   "
          f"0.05-0.95: {mid} ({100 * mid / n_tok:.2f}%)   "
          f"p>0.95: {hi} ({100 * hi / n_tok:.2f}%)")

    print("\n=== DECISION READING (critique's rule) ===")
    if np.isnan(auc):
        auc_read = "AUC UNDEFINED (empty class) -- gate unpowered"
    elif auc > 0.65:
        auc_read = ("AUC >> 0.5 (>0.65) -> KD has frontier-pushing potential: "
                    "the teacher's cross-view prob separates the student's "
                    "confident false cuts from true boundaries")
    elif auc >= 0.60:
        auc_read = "AUC in 0.60-0.65 -> borderline; weak but non-null KD signal"
    else:
        auc_read = "AUC ~0.5 (<0.60) -> KD will null; DEPRIORITIZE"
    print(f"[2] {auc_read}")
    mono_read = ("teacher is a near-monotone transform of the student -> "
                 "KD = calibration -> null"
                 if (rho_all > 0.95 and pairs["frac_rank_disagree"] < 0.02)
                 else "teacher genuinely re-ranks the student (not mere calibration)")
    print(f"[3] {mono_read}")
    coin_read = ("REDUNDANT-WITH-GOLD TRAP (>=95% of gold boundaries co-occur "
                 "with PA punctuation in their fold-group)"
                 if frac_punct >= 0.95 else
                 "below the 95% redundancy line -> teacher signal is not pure "
                 "punctuation restoration")
    print(f"[4] {coin_read}")
    sharp_read = (f"dark-knowledge mass (0.05-0.95) = {100 * mid / n_tok:.2f}% "
                  + ("-- teacher is saturated on train (memorization); soft "
                     "targets carry little beyond the gold OR-fold"
                     if mid / n_tok < 0.05 else
                     "-- non-trivial soft-target mass survives on train"))
    print(f"[5] {sharp_read}")
    if n_fp < 50:
        print(f"[!] POWER WARNING: only {n_fp} confident FPs at "
              f"P>={args.fp_thresh} on held-in train; AUC is low-powered")

    results = {
        "teacher": args.teacher, "student": args.student,
        "src": args.src, "tgt": args.tgt, "fold": args.fold,
        "fp_thresh": args.fp_thresh, "n_positions": n_tok,
        "n1_confident_fps": n_fp, "n1_true_boundaries": n_pos,
        "n2_auc_teacher_separates_fp_vs_boundary": auc,
        "n2_mean_teacher_at_conf_fp": mean_t_fp,
        "n2_mean_teacher_at_gold1": mean_t_pos,
        "n3_spearman_all": rho_all, "n3_spearman_decision_set": rho_dec,
        "n3_pair_rank_disagree": pairs["frac_rank_disagree"],
        "n3_pair_tied_either": pairs["frac_tied_either"],
        "n3_n_pairs": pairs["n_pairs"],
        "n4_coincidence_punct": frac_punct,
        "n4_coincidence_punct_or_par": frac_punct_or_par,
        "n4_coincidence_any_stripped": frac_any_strip,
        "n4_gold_boundaries": n_b,
        "n5_sharpness": {"n": n_tok, "frac_lt_0.05": lo / n_tok,
                         "frac_0.05_to_0.95": mid / n_tok,
                         "frac_gt_0.95": hi / n_tok},
        "readings": {"auc": auc_read, "monotonicity": mono_read,
                     "coincidence": coin_read, "sharpness": sharp_read},
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nresults JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
