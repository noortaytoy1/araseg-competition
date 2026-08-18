"""LEARNED ENSEMBLE / META-LEARNER on DEV (Noor's "who-to-trust" question):
does LEARNING a combiner over the 6 voters beat plain probability averaging
(the POOL of record, ~84.53)?

DISK-FREE design (no OOF voter retrains — the stacker.py OOF-train path needs
30 disk-blocked retrains and is NOT triggered here). The key honesty fact this
build leans on:

  the 6 voters were trained on TRAIN only; DEV was never in any voter's
  training set. Therefore the voters' DEV predictions
  (probs/union-NoPnx-NP-*_dev.npz) are HONEST held-out predictions. We can
  train a combiner directly on them, using k-fold CROSS-VALIDATION over the
  222 dev docs so the combiner is never evaluated on a doc it was fitted on.

METHOD (mirrors the task spec):
  1. Per-word features: 6 voter dev probs + derived (mean/std/min/max/range)
     + decorrelated extras when loadable (S-only U-Net pool; MC-dropout
     instability std) + distance-to-doc-end position features.
  2. k=5 fold CV over the 222 dev docs (seed 42, doc-LENGTH-stratified,
     DOC-LEVEL splits so no doc is in both the fit set and its eval fold).
     For each fold: fit the combiner on the other 4 folds, predict the
     held-out fold. Concatenate -> OOF-on-dev predictions for all 222 docs.
  3. Combiners: (a) logistic regression (primary), (b) small MLP (secondary).
  4. Score with the OFFICIAL metric (per-doc macro-F1, positive class = 1,
     macro over docs = knn_boundary.macro_f1 = eval_local convention).
     Own-best threshold over score_cut's grid. Compare THREE things on the
     SAME held-out folds: POOL plain average, LR combiner, MLP combiner.
     Tagger single-voter (arabertv02-s42) reported for reference.
  5. Report LR feature weights (who the combiner trusts).
  6. Pre-registered reading: learned ensemble WORKS if the CV combiner's
     held-out macro-F1 beats POOL average by > +0.2 (ensemble noise floor);
     MARGINAL if it beats at all; NULL if not.

LEAKAGE CAVEATS, reported straight (not swept under):
  * Voter dev probs are fully honest (dev never trained). The LR/MLP CV is
    clean for the voters+derived+position families.
  * The S-only U-Net checkpoints were SELECTED on dev F1@0.5 (their best.pt
    was picked using the whole dev set). So the U-Net dev-prob column carries
    a mild optimistic bias from checkpoint selection. We therefore report BOTH
    a "unet-in" arm and a "voters-only" (no-unet) arm; the honest headline is
    the voters-only arm, with the unet-in arm shown as an upper-bound probe.
  * The instability std dev cache is the confident-only cache (~89.6% NaN);
    NaN is filled with 0.0 and a coverage indicator column is added (the same
    convention stacker.feat_instability uses).

ADDITIVE, CPU-ONLY (the GPU is owned by the rmerge build — CUDA_VISIBLE_DEVICES
=-1). Read-only on probs/, runs/, stability/, unet2d/, data/. Artifacts land
under runs/learned_ensemble_2026-07-11/ (append-only). No voter retrain is ever
triggered. Reuses stacker.py's feature/combiner/fold helpers where they help.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np

# repo imports (run with cwd = arabic-sentence-segmentation, PYTHONPATH incl src)
from data import load_jsonl                                     # noqa: E402
from knn_boundary import macro_f1                               # noqa: E402
from stacker import (                                           # noqa: E402
    VOTERS,
    combiner_probs,
    feat_derived,
    feat_instability,
    feat_position,
    feat_unet,
    feat_voters,
    fit_lr,
    fit_mlp,
    length_stratified_folds,
    load_probs_map,
    flat,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# score_cut.py's exact decision grid (same values, same `>=` operator).
GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def assert_cpu_only():
    import torch
    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the GPU is owned by " \
        "the rmerge build)."


# --------------------------------------------------------------------------- #
# feature assembly (voters honest on dev; extras added when loadable)
# --------------------------------------------------------------------------- #
def build_features(docs, voter_dir, unet_spec=None, instab_npz=None):
    """Assemble the per-word feature matrix from DEV caches.

    Returns X [N, C] float64, names [C], families [C], and a dict of which
    optional signals were actually loaded."""
    voter_paths = {
        v: os.path.join(voter_dir, f"union-NoPnx-NP-{v}_dev.npz")
        for v in VOTERS
    }
    for v, p in voter_paths.items():
        assert os.path.exists(p), f"voter dev cache missing: {p}"

    blocks = []           # list of (matrix, names, family)
    Xv, nv, fv = feat_voters(docs, voter_paths)
    blocks.append((Xv, nv, fv))

    # derived stats across the 6 voters (mean/std/range) + explicit min/max so
    # the combiner can also learn "trust the extremes"
    Xd, nd, fd = feat_derived(Xv)
    vmin = Xv.min(axis=1)[:, None]
    vmax = Xv.max(axis=1)[:, None]
    Xd = np.concatenate([Xd, vmin, vmax], axis=1)
    nd = nd + ["v_min", "v_max"]
    blocks.append((Xd, nd, fd))

    loaded = {"unet": False, "instability": False}

    if unet_spec is not None:
        try:
            Xu, nu, fu = feat_unet(docs, unet_spec)
            # pool the S-only U-Net seeds into ONE decorrelated signal (matches
            # how the sonly phi 0.53 of record was measured on the pool) AND
            # keep per-seed cols for the combiner to weight.
            pool = Xu.mean(axis=1)[:, None]
            Xu = np.concatenate([Xu, pool], axis=1)
            nu = list(nu) + ["unet_sonly_pool"]
            blocks.append((Xu, nu, fu))
            loaded["unet"] = True
        except Exception as e:                                  # pragma: no cover
            print(f"# unet signal NOT loaded ({e}); proceeding without it")

    if instab_npz is not None and os.path.exists(instab_npz):
        try:
            Xi, ni, fi = feat_instability(docs, instab_npz)
            blocks.append((Xi, ni, fi))
            loaded["instability"] = True
        except Exception as e:                                  # pragma: no cover
            print(f"# instability signal NOT loaded ({e}); proceeding without")
    elif instab_npz is not None:
        print(f"# instability npz not found at {instab_npz}; proceeding without")

    # position (distance-to-doc-end etc.)
    Xp, np_, fp = feat_position(docs)
    blocks.append((Xp, np_, fp))

    X = np.concatenate([b[0] for b in blocks], axis=1)
    names = [n for b in blocks for n in b[1]]
    families = [b[2] for b in blocks for _ in b[1]]
    assert X.shape[1] == len(names) == len(families)
    assert np.isfinite(X).all(), "non-finite feature values after assembly"
    return X, names, families, loaded


def gold_and_docof(docs):
    gold = np.concatenate([np.asarray(d["labels"], dtype=np.int8)
                           for d in docs])
    doc_of = np.concatenate([np.full(len(d["labels"]), i, dtype=np.int64)
                             for i, d in enumerate(docs)])
    return gold, doc_of


# --------------------------------------------------------------------------- #
# k-fold CV on dev: fit combiner on other folds, predict held-out fold
# --------------------------------------------------------------------------- #
def cv_oof_predictions(X, y, doc_of, fold_of_pos, k, kind, seed,
                       drop_cols=None):
    """Return OOF-on-dev probs [N] plus, for LR, the per-fold coefficient
    matrix. `fold_of_pos` is the fold id per POSITION (broadcast from the
    doc-level fold assignment). `drop_cols` masks feature columns to 0-width
    for a leakage-clean variant."""
    N = X.shape[0]
    oof = np.full(N, np.nan)
    coefs = []
    if drop_cols is not None:
        keep = np.array([c not in drop_cols for c in range(X.shape[1])])
        Xu = X[:, keep]
    else:
        keep = np.ones(X.shape[1], dtype=bool)
        Xu = X

    for f in range(k):
        tr = fold_of_pos != f
        va = fold_of_pos == f
        # groups for LR's internal C-selection = the OTHER folds' ids, so the
        # held-out fold f never touches combiner fitting or C-selection.
        if kind == "lr":
            groups = fold_of_pos[tr]
            model = fit_lr(Xu[tr], y[tr], groups, seed=seed, quiet=True)
            coefs.append(model["coef"])
        else:
            model = fit_mlp(Xu[tr], y[tr], seed=seed)
        oof[va] = combiner_probs(model, Xu[va])
    assert not np.isnan(oof).any(), "some positions never held out"
    coef_mat = np.stack(coefs, axis=0) if coefs else None
    return oof, keep, coef_mat


def own_best(p, gold, doc_of, n_docs):
    fb, tb = max((macro_f1(p, gold, doc_of, n_docs, t), t) for t in GRID)
    return 100.0 * fb, tb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def cmd_run(args):
    assert_cpu_only()
    docs = load_jsonl(args.dev_jsonl)
    assert len(docs) == args.expect_docs, \
        f"--expect-docs {args.expect_docs} but loaded {len(docs)}"
    n_docs = len(docs)
    gold, doc_of = gold_and_docof(docs)

    # ---- feature assembly ------------------------------------------------- #
    unet_spec = None
    if args.unet_cache and args.unet_ckpts:
        unet_spec = {"mode": "ckpt", "cache": args.unet_cache,
                     "ckpts": args.unet_ckpts}
    X, names, families, loaded = build_features(
        docs, args.voter_dir, unet_spec=unet_spec,
        instab_npz=args.instab_npz)
    print(f"# features: {X.shape[0]} positions x {X.shape[1]} cols "
          f"({sorted(set(families))})")
    print(f"# decorrelated signals loaded: unet={loaded['unet']} "
          f"instability={loaded['instability']}")

    # ---- folds: doc-length-stratified, DOC-LEVEL, seed 42 ----------------- #
    ids = [d["doc_id"] for d in docs]
    lens = [len(d["tokens"]) for d in docs]
    fold_of_doc = length_stratified_folds(ids, lens, args.k, args.seed)
    fold_of_pos = np.array([fold_of_doc[i] for i in doc_of], dtype=np.int64)
    sizes = [int((np.array(fold_of_doc) == f).sum()) for f in range(args.k)]
    print(f"# k={args.k} folds (seed {args.seed}, doc-length-stratified, "
          f"doc-level); docs/fold={sizes}")

    # ---- (i) POOL plain average (the thing to beat) ----------------------- #
    pool = flat(docs, load_probs_map(
        os.path.join(args.voter_dir, "union-NoPnx-NP-POOL_dev.npz"), docs))
    pool_f1, pool_thr = own_best(pool, gold, doc_of, n_docs)

    # tagger single-voter reference
    tagger = flat(docs, load_probs_map(
        os.path.join(args.voter_dir,
                     "union-NoPnx-NP-arabertv02-s42_dev.npz"), docs))
    tag_f1, tag_thr = own_best(tagger, gold, doc_of, n_docs)

    # ---- (ii) LR combiner: full-feature + voters-only (clean) ------------- #
    # unet column ids to drop for the leakage-clean "voters-only" variant
    unet_col_ids = {i for i, fam in enumerate(families) if fam == "unet"}

    lr_oof, keep_full, coef_full = cv_oof_predictions(
        X, gold, doc_of, fold_of_pos, args.k, "lr", args.seed)
    lr_f1, lr_thr = own_best(lr_oof, gold, doc_of, n_docs)

    lr_clean_oof, keep_clean, coef_clean = cv_oof_predictions(
        X, gold, doc_of, fold_of_pos, args.k, "lr", args.seed,
        drop_cols=unet_col_ids)
    lrc_f1, lrc_thr = own_best(lr_clean_oof, gold, doc_of, n_docs)

    # ---- (iii) MLP combiner (secondary) ----------------------------------- #
    mlp_oof, _, _ = cv_oof_predictions(
        X, gold, doc_of, fold_of_pos, args.k, "mlp", args.seed)
    mlp_f1, mlp_thr = own_best(mlp_oof, gold, doc_of, n_docs)

    mlp_clean_oof, _, _ = cv_oof_predictions(
        X, gold, doc_of, fold_of_pos, args.k, "mlp", args.seed,
        drop_cols=unet_col_ids)
    mlpc_f1, mlpc_thr = own_best(mlp_clean_oof, gold, doc_of, n_docs)

    # ---- LR weights: who does the combiner trust? ------------------------- #
    # average |standardized coef| across folds -> importance; signed mean coef
    # -> direction. Names are on the FULL feature set.
    names_full = [names[i] for i in range(len(names)) if keep_full[i]]
    mean_coef = coef_full.mean(axis=0)
    std_coef = coef_full.std(axis=0)
    order = np.argsort(-np.abs(mean_coef))
    weights = [
        {"feature": names_full[i], "family": families[
            [j for j in range(len(names)) if keep_full[j]][i]],
         "mean_coef": float(mean_coef[i]),
         "abs_coef": float(abs(mean_coef[i])),
         "coef_std_across_folds": float(std_coef[i])}
        for i in order
    ]

    # ---- verdict (pre-registered) ----------------------------------------- #
    # Honest headline = leakage-clean voters-only LR vs POOL average.
    delta_lr = lr_f1 - pool_f1
    delta_lrc = lrc_f1 - pool_f1
    delta_mlp = mlp_f1 - pool_f1
    delta_mlpc = mlpc_f1 - pool_f1

    def verdict(delta):
        if delta > 0.2:
            return "WORKS"
        if delta > 0.0:
            return "MARGINAL"
        return "NULL"

    # ---- print report ----------------------------------------------------- #
    print("\n================ LEARNED ENSEMBLE: DEV CV REPORT ================")
    print(f"official metric = per-doc macro-F1 (positive=1), own-best over "
          f"grid {GRID}")
    print(f"{'arm':38s} {'held-out macroF1':>16s}  {'thr':>4s}  {'Δ vs POOL':>10s}")
    print(f"{'(ref) tagger single-voter s42':38s} {tag_f1:16.4f}  "
          f"{tag_thr:>4.2f}  {'':>10s}")
    print(f"{'(i)   POOL plain average':38s} {pool_f1:16.4f}  "
          f"{pool_thr:>4.2f}  {0.0:>+10.4f}")
    print(f"{'(ii)  LR combiner  [unet-in]':38s} {lr_f1:16.4f}  "
          f"{lr_thr:>4.2f}  {delta_lr:>+10.4f}  [{verdict(delta_lr)}]")
    print(f"{'(ii)  LR combiner  [voters-only]':38s} {lrc_f1:16.4f}  "
          f"{lrc_thr:>4.2f}  {delta_lrc:>+10.4f}  [{verdict(delta_lrc)}]")
    print(f"{'(iii) MLP combiner [unet-in]':38s} {mlp_f1:16.4f}  "
          f"{mlp_thr:>4.2f}  {delta_mlp:>+10.4f}  [{verdict(delta_mlp)}]")
    print(f"{'(iii) MLP combiner [voters-only]':38s} {mlpc_f1:16.4f}  "
          f"{mlpc_thr:>4.2f}  {delta_mlpc:>+10.4f}  [{verdict(delta_mlpc)}]")

    print("\n---- LR weights (unet-in fold-mean; who to trust) ----")
    for w in weights:
        print(f"  {w['feature']:26s} {w['family']:14s} "
              f"coef={w['mean_coef']:+.4f}  |coef|={w['abs_coef']:.4f}  "
              f"(fold-std {w['coef_std_across_folds']:.4f})")

    print("\n---- PRE-REGISTERED READING (verbatim) ----")
    print("learned ensemble WORKS if the CV combiner's held-out macro-F1 beats "
          "the POOL average by > +0.2 (ensemble noise floor); marginal if it "
          "beats at all; null if not.")
    print(f"HONEST HEADLINE (leakage-clean, voters-only LR): "
          f"Δ = {delta_lrc:+.4f} -> {verdict(delta_lrc)}")
    print(f"UPPER-BOUND PROBE (unet-in LR, mild dev-selection bias): "
          f"Δ = {delta_lr:+.4f} -> {verdict(delta_lr)}")

    # ---- artifacts -------------------------------------------------------- #
    report = {
        "task": "NoPnx-NP",
        "metric": "per-doc macro-F1 (knn_boundary.macro_f1 = eval_local conv)",
        "grid": GRID, "k": args.k, "seed": args.seed,
        "n_docs": n_docs, "n_positions": int(X.shape[0]),
        "signals_loaded": loaded,
        "feature_names": names, "families": families,
        "arms": {
            "tagger_single_voter_s42": {"f1": tag_f1, "thr": tag_thr},
            "pool_average": {"f1": pool_f1, "thr": pool_thr},
            "lr_unet_in": {"f1": lr_f1, "thr": lr_thr, "delta": delta_lr,
                           "verdict": verdict(delta_lr)},
            "lr_voters_only": {"f1": lrc_f1, "thr": lrc_thr, "delta": delta_lrc,
                               "verdict": verdict(delta_lrc)},
            "mlp_unet_in": {"f1": mlp_f1, "thr": mlp_thr, "delta": delta_mlp,
                            "verdict": verdict(delta_mlp)},
            "mlp_voters_only": {"f1": mlpc_f1, "thr": mlpc_thr,
                                "delta": delta_mlpc,
                                "verdict": verdict(delta_mlpc)},
        },
        "lr_weights_unet_in": weights,
        "honest_headline": {
            "arm": "lr_voters_only", "delta": delta_lrc,
            "verdict": verdict(delta_lrc)},
        "leakage_caveats": {
            "voters": "honest — dev never in voter training",
            "unet": "checkpoints selected on dev F1@0.5 -> mild optimistic "
                    "bias; unet-in arms are upper-bound probes",
            "instability": "confident-only cache (~89.6% NaN), NaN->0 + "
                           "coverage indicator col",
        },
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        assert not os.path.exists(args.out), \
            f"{args.out} exists — runs/ is append-only, pick a new name"
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\n# report -> {args.out}")
    return report


# --------------------------------------------------------------------------- #
# self-test (planted signal; runs with data/ hidden — no corpus needed)
# --------------------------------------------------------------------------- #
def cmd_self_test(_args):
    """Corpus-independent acceptance test on a PLANTED signal.

    Build 40 synthetic docs. Two 'voters' each see the true label through
    heavy noise; individually weak, but a learned combiner that trusts their
    AGREEMENT should beat their plain average when the noise is anti-correlated
    across voters. Verifies: (1) doc-level folds never split a doc; (2) OOF
    predictions cover every position exactly once; (3) LR CV runs end-to-end
    and produces finite probs; (4) the macro-F1 comparison path returns
    sensible numbers. Fast, CPU-only, no torch."""
    rng = np.random.default_rng(0)
    n_docs = 40
    docs = []
    labels = []
    for i in range(n_docs):
        n = int(rng.integers(30, 80))
        y = (rng.random(n) < 0.2).astype(np.int8)
        docs.append({"doc_id": f"d{i:03d}", "tokens": ["w"] * n,
                     "labels": y.tolist()})
        labels.append(y)
    gold = np.concatenate([np.asarray(d["labels"]) for d in docs])
    doc_of = np.concatenate([np.full(len(d["labels"]), i)
                             for i, d in enumerate(docs)])

    # two anti-correlated noisy voters
    def noisy(y, flip_pos, flip_neg, r):
        p = np.where(y == 1, 0.5 + 0.4 * (r.random(len(y)) - flip_pos),
                     0.5 - 0.4 * (r.random(len(y)) - flip_neg))
        return np.clip(p, 1e-4, 1 - 1e-4)

    r1, r2 = np.random.default_rng(1), np.random.default_rng(2)
    v1 = np.concatenate([noisy(np.asarray(d["labels"]), 0.3, 0.3, r1)
                         for d in docs])
    v2 = np.concatenate([noisy(np.asarray(d["labels"]), 0.35, 0.35, r2)
                         for d in docs])
    X = np.stack([v1, v2, 0.5 * (v1 + v2)], axis=1)

    ids = [d["doc_id"] for d in docs]
    lens = [len(d["tokens"]) for d in docs]
    k = 5
    fold_of_doc = length_stratified_folds(ids, lens, k, 42)
    fold_of_pos = np.array([fold_of_doc[i] for i in doc_of])

    # (1) doc-level: every position of a doc shares one fold id
    for i in range(n_docs):
        fp = fold_of_pos[doc_of == i]
        assert (fp == fp[0]).all(), "a doc was split across folds"

    # (3)+(2) OOF LR covers every position exactly once, finite
    oof, keep, coef = cv_oof_predictions(X, gold, doc_of, fold_of_pos, k,
                                         "lr", 42)
    assert oof.shape == gold.shape and np.isfinite(oof).all()
    assert coef.shape[0] == k

    # (4) macro-F1 path returns numbers in [0, 100]
    pool = 0.5 * (v1 + v2)
    pf, pt = own_best(pool, gold, doc_of, n_docs)
    lf, lt = own_best(oof, gold, doc_of, n_docs)
    assert 0.0 <= pf <= 100.0 and 0.0 <= lf <= 100.0
    print(f"self-test OK: folds doc-level, OOF full-coverage, LR CV runs; "
          f"planted pool={pf:.2f}@{pt} lr={lf:.2f}@{lt} (Δ={lf - pf:+.2f})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="CV meta-learner on dev voter predictions")
    r.add_argument("--dev-jsonl", required=True)
    r.add_argument("--voter-dir", default="probs",
                   help="dir holding union-NoPnx-NP-*_dev.npz")
    r.add_argument("--unet-cache", default=None,
                   help="S-only U-Net dev similarity-matrix cache (npz)")
    r.add_argument("--unet-ckpts", nargs="*", default=None,
                   help="S-only U-Net best.pt checkpoints (pooled)")
    r.add_argument("--instab-npz", default=None,
                   help="MC-dropout instability std dev npz (optional)")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--expect-docs", type=int, default=222)
    r.add_argument("--out", default=None, help="report json path")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("self-test", help="planted-signal acceptance test")
    s.set_defaults(fn=cmd_self_test)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
