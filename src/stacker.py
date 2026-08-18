"""STACKING META-COMBINER for AraSeg NoPnx-NP (Noor's standing meta-combiner
order): replace ensemble AVERAGING of the 6-voter pool (inter-voter error
correlation 0.74 of record) with a TRAINED combiner over per-position
features. ADDITIVE, CPU-ONLY here (the GPU is owned by the BoundRL pilot; all
GPU spend is staged in runs/stacker_battery_2026-07-11/run_stacker.sh behind
STACKER_GO=YES). Read-only on runs/, probs/, knn_store/, merge_judge/,
rae_gate/, rae_ft/, stability/, unet2d/, data/. Artifacts land under
runs/stacker_battery_2026-07-11/ — a NEW additive directory.

THE DESIGN TRAP (pre-identified; the reason this build is nontrivial):
the 6 full-fit voters MEMORIZED train (train probs ~= gold), so a combiner
fitted on their TRAIN predictions learns "trust the voters blindly" —
nothing transferable. Therefore the combiner trains ONLY on OUT-OF-FOLD
voter predictions:
  * k=5 fold split of the 174 TRAIN docs, seed 42, DOC-LENGTH-stratified
    (the train jsonl has NO genre field — verified: keys are doc_id / tokens /
    labels / text / label_str — so the pre-declared genre fallback applies).
  * each voter retrains on k-1 folds and predicts the held-out fold;
    concatenating the 5 held-fold predictions gives one OOF prob per voter
    per train position.
  * CHECKPOINT-SELECTION DECISION (documented): fold voters select their
    best epoch on an INNER-DEV slice (~15% of the k-1 fold-train docs,
    length-stratified, deterministic) — NOT on dev (dev enters ONLY for the
    final evaluation of the fitted combiner) and NOT on the held-out fold
    (that would leak the fold into its own OOF predictions).
  * DEV-SIDE FEATURES use the existing full-fit record caches
    (probs/union-NoPnx-NP-*_dev.npz). Standard stacking asymmetry: OOF
    features are slightly weaker (k-1 training data, no full-fit sharpness)
    than the dev-time features — the conservative direction.
  * OOF TRIPWIRE (pre-registered guard against silent leakage): fitting
    ABORTS if any voter column of the TRAIN matrix has pooled token-level
    ROC-AUC > 0.995 vs gold — that is the signature of full-fit
    (memorized) train probs entering by mistake.

PER-POSITION FEATURE FAMILIES (availability at build time is documented in
the battery runner header; loaders below):
  voters        6 cols  OOF probs (train) / record dev caches (dev)
  voter_derived 3 cols  mean / std / max-min across the 6 voters
  unet          3 cols  S-only 2D U-Net probs (runs/unet2d_sonly_2026-07-10/
                        s{1,2,42}r1); input = similarity matrices from the
                        FROZEN full-fit s42 encoder — memorization risk is
                        MEASURED, not assumed (`unet-sharpness` readout:
                        train-vs-dev prob sharpness + token AUC gap), and the
                        conditional OOF retrain (`unet-oof`, CPU) is
                        pre-registered in the runner on that measurement.
  instability   1 col   MC-dropout std (T=8). Dev cache of record covers only
                        confident positions (89.6% NaN) -> the runner stages
                        FULL-coverage recomputes on both sides
                        (`oof-predict --mc-passes`); the loader also accepts
                        the old NaN convention (fill 0 + coverage indicator,
                        2 cols) for the record cache.
  rae_E         1 col   Socher-RAE leaf reconstruction error E(i,i+1)
                        (rae_gate/rae_final.pt over knn_store vecs;
                        doc-final position filled with the doc mean).
  judge         1 col   rae_ft judge P(merge) in the ALL-CUT state
                        (rae_ft/judge_full_final.pt; state-independent so it
                        is defined at every junction on both sides; the
                        eraser-state variant of record depends on a tagger
                        segmentation and is train/dev-asymmetric). Included
                        on Noor's order but EXPECTED to be zeroed (measured:
                        eraser slice 0.7237 vs tagger's own 0.7298 — no
                        marginal value beyond p_tag).
  position      3 cols  distance-to-doc-end (log1p), relative position,
                        log doc token count.

COMBINER: logistic regression FIRST (pre-registered primary; data size and
literature contraindicate trees). Features standardized on the train matrix;
C selected by grouped CV where the groups are the SAME k=5 folds (dev never
touches combiner fitting). Optional small MLP behind --mlp (flagged
secondary). Per-position binary target. DECODE: fixed 0.5 headline PLUS
own-best threshold sweep 0.30-0.70 step 0.05 (score_cut.py convention:
same grid, same `>=`, eval_local.compute_metrics).

Stages runnable NOW on CPU:
  CUDA_VISIBLE_DEVICES=-1 python src/stacker.py self-test
  CUDA_VISIBLE_DEVICES=-1 python src/stacker.py make-folds \
      --train-jsonl data/NoPnx-NP_train.jsonl --k 5 --seed 42 \
      --out runs/stacker_battery_2026-07-11/folds.json \
      --jsonl-dir runs/stacker_battery_2026-07-11/folds
  CUDA_VISIBLE_DEVICES=-1 python src/stacker.py unet-sharpness \
      --ckpts runs/unet2d_sonly_2026-07-10/s1r1/best.pt \
              runs/unet2d_sonly_2026-07-10/s2r1/best.pt \
              runs/unet2d_sonly_2026-07-10/s42r1/best.pt \
      --train-cache unet2d/NoPnx-NP_s42_train_mid8.npz \
      --dev-cache unet2d/NoPnx-NP_s42_dev_mid8.npz \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --out runs/stacker_battery_2026-07-11/unet_sharpness.json

GPU / battery-only stages (assemble / fit / eval / oof-predict / unet-oof /
merge-npz) are orchestrated by run_stacker.sh — see its pre-registered
header for the gates of record.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

from data import PARAGRAPH_TOKEN, load_jsonl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# score_cut.py's exact decision grid (same values, same `>=` operator)
GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# fixed voter order of record (run_trdev_ensemble.sh MEMBERS order)
VOTERS = ["arabertv02-s1", "arabertv02-s2", "arabertv02-s3",
          "arabertv02-s42", "araelectra-s42", "arbertv2-s42"]

CONF = 0.8                  # "confident" per the standing order
OOF_AUC_TRIPWIRE = 0.995    # pre-registered full-fit-leakage abort bar

FAMILY_ORDER = ["voters", "voter_derived", "unet", "instability",
                "rae_E", "judge", "position"]


def assert_cpu_only():
    import torch
    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the GPU is owned by " \
        "the BoundRL pilot)"


# =========================================================================== #
# folds: k-fold split of the train docs, doc-length-stratified, seeded
# =========================================================================== #
def length_stratified_folds(doc_ids, lengths, k, seed):
    """fold id per doc. Docs are sorted by (length, doc_id) and consecutive
    blocks of k get a seeded permutation of fold ids -> every fold sees the
    same length spectrum. Deterministic in (doc set, k, seed)."""
    order = sorted(range(len(doc_ids)), key=lambda i: (lengths[i], doc_ids[i]))
    rng = np.random.default_rng(seed)
    fold_of = [None] * len(doc_ids)
    for b in range(0, len(order), k):
        block = order[b:b + k]
        perm = rng.permutation(k)[:len(block)]
        for slot, i in enumerate(block):
            fold_of[i] = int(perm[slot])
    return fold_of


def inner_dev_pick(doc_ids, lengths, frac, seed):
    """Deterministic length-stratified inner-dev subset (~frac of the docs):
    sort by (length, doc_id), take one seeded pick per block of
    round(1/frac). Used for fold-voter CHECKPOINT SELECTION so neither dev
    nor the held-out fold leaks into the OOF predictions."""
    step = max(2, int(round(1.0 / frac)))
    order = sorted(range(len(doc_ids)), key=lambda i: (lengths[i], doc_ids[i]))
    rng = np.random.default_rng(seed)
    picked = set()
    for b in range(0, len(order), step):
        block = order[b:b + step]
        picked.add(doc_ids[block[int(rng.integers(len(block)))]])
    return picked


def cmd_make_folds(args):
    docs = load_jsonl(args.train_jsonl)
    assert len(docs) == args.expect_docs, \
        f"--expect-docs {args.expect_docs} but loaded {len(docs)} from " \
        f"{args.train_jsonl} -> ABORT"
    ids = [d["doc_id"] for d in docs]
    lens = [len(d["tokens"]) for d in docs]
    has_genre = any("genre" in d for d in docs)
    strat = "genre" if has_genre else "doc-length"
    assert not has_genre, \
        "a genre field appeared in the train jsonl; genre stratification " \
        "was pre-declared for that case but is NOT implemented -> STOP, ask"
    fold_of = length_stratified_folds(ids, lens, args.k, args.seed)
    fold_by_id = dict(zip(ids, fold_of))
    len_by_id = dict(zip(ids, lens))

    folds = {}
    for f in range(args.k):
        held = [i for i in ids if fold_by_id[i] == f]  # order = jsonl order
        train_ids = [i for i in ids if fold_by_id[i] != f]
        tr_lens = [len_by_id[i] for i in train_ids]
        inner = inner_dev_pick(train_ids, tr_lens, args.inner_frac,
                               args.seed * 1000 + f)
        folds[str(f)] = {
            "held_out": held,
            "inner_dev": sorted(inner),
            "train": [i for i in train_ids if i not in inner],
        }
    payload = {
        "task": args.task, "train_jsonl": args.train_jsonl,
        "k": args.k, "seed": args.seed, "inner_frac": args.inner_frac,
        "stratification": strat, "n_docs": len(docs),
        "voters": VOTERS,
        "folds": folds,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    assert not os.path.exists(args.out), \
        f"{args.out} exists — battery dirs are append-only, refuse to clobber"
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    sizes = [(len(folds[str(f)]["train"]), len(folds[str(f)]["inner_dev"]),
              len(folds[str(f)]["held_out"])) for f in range(args.k)]
    print(f"folds.json -> {args.out}")
    for f, (a, b, c) in enumerate(sizes):
        print(f"  fold {f}: train={a}  inner_dev={b}  held_out={c}")
    lmeans = []
    for f in range(args.k):
        hl = [len_by_id[i] for i in folds[str(f)]["held_out"]]
        lmeans.append(sum(hl) / len(hl))
    print(f"  held-fold mean doc length per fold: "
          f"{['%.0f' % m for m in lmeans]} (stratification check)")

    if args.jsonl_dir:
        os.makedirs(args.jsonl_dir, exist_ok=True)
        by_id = {d["doc_id"]: d for d in docs}
        for f in range(args.k):
            for part in ("train", "inner_dev", "held_out"):
                p = os.path.join(args.jsonl_dir, f"f{f}_{part}.jsonl")
                assert not os.path.exists(p), \
                    f"{p} exists — append-only, refuse to clobber"
                with open(p, "w", encoding="utf-8") as fh:
                    for i in folds[str(f)][part]:
                        fh.write(json.dumps(by_id[i], ensure_ascii=False)
                                 + "\n")
        print(f"fold jsonls -> {args.jsonl_dir}/f<k>_{{train,inner_dev,"
              f"held_out}}.jsonl")


# =========================================================================== #
# npz helpers
# =========================================================================== #
def load_probs_map(path, docs, allow_missing=False):
    """word-grid npz -> {doc_id: float64 array}, length-checked."""
    z = np.load(path)
    out = {}
    for d in docs:
        if d["doc_id"] not in z.files:
            assert allow_missing, f"{path} missing {d['doc_id']}"
            continue
        v = z[d["doc_id"]].astype(np.float64)
        assert len(v) == len(d["tokens"]), \
            f"{path}: length mismatch for {d['doc_id']}"
        out[d["doc_id"]] = v
    return out


def flat(docs, m):
    return np.concatenate([m[d["doc_id"]] for d in docs])


def gold_arrays(docs):
    gold = np.concatenate([np.asarray(d["labels"], dtype=np.int8)
                           for d in docs])
    doc_of = np.concatenate([np.full(len(d["labels"]), i, dtype=np.int64)
                             for i, d in enumerate(docs)])
    return gold, doc_of


def cmd_merge_npz(args):
    """Merge disjoint word-grid npzs (per-fold OOF pieces) into one npz that
    must cover the reference jsonl exactly."""
    docs = load_jsonl(args.jsonl)
    merged = {}
    for p in args.parts:
        z = np.load(p)
        for k in z.files:
            assert k not in merged, f"doc {k} appears in two parts ({p})"
            merged[k] = z[k]
    missing = [d["doc_id"] for d in docs if d["doc_id"] not in merged]
    assert not missing, f"merged npz misses {len(missing)} docs, " \
                        f"e.g. {missing[:3]}"
    extra = set(merged) - {d["doc_id"] for d in docs}
    assert not extra, f"merged npz has {len(extra)} unknown docs"
    for d in docs:
        assert len(merged[d["doc_id"]]) == len(d["tokens"]), d["doc_id"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    assert not os.path.exists(args.out), f"{args.out} exists — append-only"
    np.savez(args.out, **merged)
    print(f"merged {len(args.parts)} parts -> {args.out} "
          f"({len(merged)} docs, full coverage)")


# =========================================================================== #
# feature builders — each returns (matrix [N, c] float64, [names], family)
# =========================================================================== #
def feat_voters(docs, voter_paths):
    cols, names = [], []
    for name in VOTERS:
        assert name in voter_paths, f"voter {name} missing from manifest"
        cols.append(flat(docs, load_probs_map(voter_paths[name], docs)))
        names.append(f"voter_{name}")
    return np.stack(cols, axis=1), names, "voters"


def feat_derived(voter_matrix):
    m = voter_matrix.mean(axis=1)
    s = voter_matrix.std(axis=1)
    r = voter_matrix.max(axis=1) - voter_matrix.min(axis=1)
    return (np.stack([m, s, r], axis=1),
            ["v_mean", "v_std", "v_maxmin"], "voter_derived")


def feat_unet(docs, spec):
    """S-only U-Net per-position probs. Two manifest modes so the train side
    can switch to OOF retrains when unet-sharpness demands it:
      {"mode": "ckpt", "ckpts": [...], "cache": <sim-matrix npz>}
          forward existing checkpoints through unet2d_train's own
          predict/stitch (eval convention identical to its cmd_eval);
          column name unet_s<seed-from-ckpt>.
      {"mode": "npz", "npz": {"s42": <word-grid npz>, ...}}
          pre-computed (e.g. merged unet-oof) probs; column name unet_<key>.
    Column NAMES must end up identical on both sides — asserted at eval."""
    if spec.get("mode", "ckpt") == "npz":
        cols, names = [], []
        for key in sorted(spec["npz"]):
            cols.append(flat(docs, load_probs_map(spec["npz"][key], docs)))
            names.append(f"unet_{key}")
        return np.stack(cols, axis=1), names, "unet"
    import torch
    from unet2d_train import (build_model, flatten_probs, load_cache,
                              predict_windows, stitch)
    cache = load_cache(spec["cache"])
    assert [str(x) for x in cache["doc_ids"]] == [d["doc_id"] for d in docs], \
        f"{spec['cache']}: cache doc order != jsonl doc order"
    cols, names = [], []
    for ck in spec["ckpts"]:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        in_ch = int(blob.get("in_ch", 2))
        model = build_model(base=blob["base"], in_ch=in_ch)
        model.load_state_dict(blob["state_dict"])
        pw = predict_windows(model, cache, no_tagger=(in_ch == 1))
        per_doc, _ = stitch(cache, pw)
        cols.append(flatten_probs(docs, per_doc).astype(np.float64))
        names.append(f"unet_s{blob['seed']}")
    order = np.argsort(names)
    return (np.stack(cols, axis=1)[:, order],
            [names[i] for i in order], "unet")


def feat_instability(docs, npz_path):
    """MC-dropout std. Full-coverage npz -> 1 col. Old confident-only cache
    (NaN elsewhere) -> fill 0 + coverage indicator (2 cols)."""
    m = load_probs_map(npz_path, docs)
    x = flat(docs, m)
    nan = np.isnan(x)
    if nan.any():
        has = (~nan).astype(np.float64)
        x = np.where(nan, 0.0, x)
        return (np.stack([x, has], axis=1),
                ["instab_std", "instab_has"], "instability")
    return x[:, None], ["instab_std"], "instability"


def feat_rae_E(docs, rae_ckpt, vecs_path):
    """Leaf reconstruction error E(i, i+1) per junction (rae_gate compose
    path). Doc-final position (no junction) filled with the doc's mean E."""
    from merge_judge import load_doc_vectors
    from rae_gate import leaf_errors, load_rae, normalize_rows
    vecs = load_doc_vectors(vecs_path, docs)
    model, _ = load_rae(rae_ckpt)
    cols = []
    for d in docs:
        V = normalize_rows(vecs[d["doc_id"]])
        n = len(V)
        e = np.zeros(n, dtype=np.float64)
        if n >= 2:
            le = leaf_errors(model, V)
            e[:n - 1] = le
            e[n - 1] = float(le.mean())
        cols.append(e)
    return np.concatenate(cols)[:, None], ["rae_E_final"], "rae_E"


def feat_judge(docs, judge_ckpt, vecs_path):
    """rae_ft judge P(merge) per junction in the ALL-CUT state (defined at
    every junction, train/dev-symmetric). Doc-final position -> 0.0 (gold is
    1 on 174/174 + 222/222 doc-final tokens; 0 merge prob = boundary-most)."""
    from merge_judge import (build_features, load_doc_vectors,
                             pairs_from_cuts, prefix_sum)
    from rae_finetune import judge_with_E, load_judge
    vecs = load_doc_vectors(vecs_path, docs)
    model, _ = load_judge(judge_ckpt)
    cols = []
    for d in docs:
        V = vecs[d["doc_id"]].astype(np.float32)
        n = len(V)
        s = np.zeros(n, dtype=np.float64)
        if n >= 2:
            X = build_features(V, prefix_sum(V),
                               pairs_from_cuts(np.arange(n - 1), n))
            pm, _e = judge_with_E(model, X)
            s[:n - 1] = pm.astype(np.float64)
        cols.append(s)
    return np.concatenate(cols)[:, None], ["judge_allcut_full_final"], "judge"


def feat_position(docs):
    cols = []
    for d in docs:
        n = len(d["tokens"])
        i = np.arange(n, dtype=np.float64)
        dist_end = np.log1p(n - 1 - i)
        rel = i / max(n - 1, 1)
        loglen = np.full(n, np.log1p(n))
        cols.append(np.stack([dist_end, rel, loglen], axis=1))
    return (np.concatenate(cols, axis=0),
            ["dist_doc_end_log", "rel_pos", "log_doc_len"], "position")


def assemble_matrix(docs, manifest):
    """manifest (dict) -> X, feature_names, family_of (per column), y,
    doc_of. Families absent from the manifest are skipped (train and dev
    matrices must end up with IDENTICAL name lists — asserted at fit/eval)."""
    blocks, names, fams = [], [], []

    def add(mat, nm, fam):
        blocks.append(np.asarray(mat, dtype=np.float64))
        names.extend(nm)
        fams.extend([fam] * len(nm))

    vX, vn, vf = feat_voters(docs, manifest["voters"])
    add(vX, vn, vf)
    add(*feat_derived(vX))
    if manifest.get("unet"):
        add(*feat_unet(docs, manifest["unet"]))
    if manifest.get("instability"):
        add(*feat_instability(docs, manifest["instability"]))
    if manifest.get("rae"):
        add(*feat_rae_E(docs, manifest["rae"]["ckpt"],
                        manifest["rae"]["vecs"]))
    if manifest.get("judge"):
        add(*feat_judge(docs, manifest["judge"]["ckpt"],
                        manifest["judge"]["vecs"]))
    add(*feat_position(docs))

    X = np.concatenate(blocks, axis=1)
    assert np.isfinite(X).all(), "non-finite feature values after assembly"
    y, doc_of = gold_arrays(docs)
    assert len(X) == len(y)
    return X, names, fams, y, doc_of


def cmd_assemble(args):
    assert_cpu_only()
    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    docs = load_jsonl(manifest["jsonl"])
    t0 = time.time()
    X, names, fams, y, doc_of = assemble_matrix(docs, manifest)
    doc_ids = np.array([d["doc_id"] for d in docs])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    assert not os.path.exists(args.out), f"{args.out} exists — append-only"
    np.savez_compressed(
        args.out, X=X.astype(np.float32), y=y, doc_of=doc_of,
        doc_ids=doc_ids, feature_names=np.array(names),
        families=np.array(fams), manifest=np.array(json.dumps(manifest)))
    print(f"assembled [{len(X)} positions x {X.shape[1]} features] from "
          f"{len(docs)} docs in {time.time() - t0:.0f}s -> {args.out}")
    for fam in FAMILY_ORDER:
        c = fams.count(fam)
        if c:
            print(f"  {fam:14s} {c} col(s): "
                  f"{[n for n, f in zip(names, fams) if f == fam]}")


# =========================================================================== #
# combiner: logistic regression primary (+ flagged MLP secondary)
# =========================================================================== #
def rank_auc_np(scores, labels):
    """Rank AUC with tie-averaged ranks (Mann-Whitney)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    uniq, inv, cnt = np.unique(scores, return_inverse=True,
                               return_counts=True)
    start = np.zeros(len(uniq))
    start[0] = 1
    start[1:] = np.cumsum(cnt)[:-1] + 1
    ranks = (start + (cnt - 1) / 2.0)[inv]
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oof_leak_tripwire(X, names, fams, y):
    """Pre-registered: ABORT if any voter column looks memorized."""
    lines, worst = [], 0.0
    for j, (nm, fam) in enumerate(zip(names, fams)):
        if fam != "voters":
            continue
        auc = rank_auc_np(X[:, j], y == 1)
        worst = max(worst, auc)
        lines.append(f"    {nm:28s} token AUC vs gold = {auc:.4f}")
    print("  OOF-leak tripwire (voter columns must NOT be memorized):")
    for ln in lines:
        print(ln)
    assert worst <= OOF_AUC_TRIPWIRE, (
        f"FULL-FIT LEAKAGE SUSPECTED: a voter column has train AUC "
        f"{worst:.4f} > {OOF_AUC_TRIPWIRE} — the combiner would learn "
        f"'trust the voters blindly'. Use OUT-OF-FOLD predictions. ABORT.")
    return worst


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.maximum(sd, 1e-8)
    return mu, sd


def fit_lr(X, y, groups, seed=42, c_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
           quiet=False):
    """Standardize + LogisticRegression; C picked by grouped CV (groups =
    the SAME k folds -> dev never touches combiner fitting)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    mu, sd = standardize_fit(X)
    Xs = (X - mu) / sd
    uniq = np.unique(groups)
    best = None
    for C in c_grid:
        losses = []
        for g in uniq:
            tr, va = groups != g, groups == g
            clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
            clf.fit(Xs[tr], y[tr])
            p = clf.predict_proba(Xs[va])[:, 1]
            losses.append(log_loss(y[va], p, labels=[0, 1]))
        m = float(np.mean(losses))
        if not quiet:
            print(f"    C={C:<5} grouped-CV logloss={m:.5f}")
        if best is None or m < best[1]:
            best = (C, m)
    C = best[0]
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(Xs, y)
    return {"kind": "lr", "coef": clf.coef_[0].astype(np.float64),
            "intercept": float(clf.intercept_[0]), "mu": mu, "sd": sd,
            "C": C, "cv_logloss": best[1], "seed": seed}


def fit_mlp(X, y, seed=42):
    """Flagged secondary: small sklearn MLP (no new dependency)."""
    from sklearn.neural_network import MLPClassifier
    mu, sd = standardize_fit(X)
    Xs = (X - mu) / sd
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300,
                        random_state=seed)
    clf.fit(Xs, y)
    return {"kind": "mlp", "sk_model": clf, "mu": mu, "sd": sd, "seed": seed}


def combiner_probs(model, X):
    Xs = (X - model["mu"]) / model["sd"]
    if model["kind"] == "lr":
        z = Xs @ model["coef"] + model["intercept"]
        return 1.0 / (1.0 + np.exp(-z))
    return model["sk_model"].predict_proba(Xs)[:, 1]


def load_features(path):
    z = np.load(path, allow_pickle=False)
    return {"X": z["X"].astype(np.float64), "y": z["y"],
            "doc_of": z["doc_of"],
            "doc_ids": [str(s) for s in z["doc_ids"]],
            "names": [str(s) for s in z["feature_names"]],
            "families": [str(s) for s in z["families"]],
            "manifest": json.loads(str(z["manifest"]))}


def cmd_fit(args):
    assert_cpu_only()
    tr = load_features(args.train_features)
    with open(args.folds, encoding="utf-8") as fh:
        fj = json.load(fh)
    fold_of_doc = {}
    for f, blob in fj["folds"].items():
        for i in blob["held_out"]:
            fold_of_doc[i] = int(f)
    groups = np.array([fold_of_doc[tr["doc_ids"][di]]
                       for di in tr["doc_of"]])
    print(f"# fit: [{tr['X'].shape[0]} x {tr['X'].shape[1]}], target rate "
          f"{tr['y'].mean():.4f}, groups=folds k={fj['k']}")
    oof_leak_tripwire(tr["X"], tr["names"], tr["families"], tr["y"])

    model = fit_lr(tr["X"], tr["y"], groups, seed=args.seed)
    model["feature_names"] = tr["names"]
    model["families"] = tr["families"]
    print(f"  LR fitted: C={model['C']} cv_logloss={model['cv_logloss']:.5f}")
    print("  standardized LR coefficients (positive -> boundary):")
    order = np.argsort(-np.abs(model["coef"]))
    for j in order:
        print(f"    {tr['names'][j]:28s} {model['coef'][j]:+8.4f}")

    if args.mlp:
        model2 = fit_mlp(tr["X"], tr["y"], seed=args.seed)
        model2["feature_names"] = tr["names"]
        model2["families"] = tr["families"]
        model = {"primary": model, "mlp": model2}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    assert not os.path.exists(args.out), f"{args.out} exists — append-only"
    with open(args.out, "wb") as fh:
        pickle.dump(model, fh)
    print(f"combiner -> {args.out}")


# =========================================================================== #
# dev evaluation (dev = evaluation/selection ONLY) + anatomy + ablation
# =========================================================================== #
def preds_at(docs, p_flat, t):
    """flat probs -> {doc_id: 0/1 list} at threshold t, PARAGRAPH forced 0
    (score_cut.py's exact reading; a no-op on NoPnx tracks)."""
    out, off = {}, 0
    for d in docs:
        n = len(d["tokens"])
        lab = (p_flat[off:off + n] >= t).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        out[d["doc_id"]] = lab.tolist()
        off += n
    assert off == len(p_flat)
    return out


def score_rows(docs, gold_map, p_flat, arm, seed, results_tsv=None):
    """Fixed 0.5 headline + own-best over GRID; appends a score_cut.py-layout
    row when results_tsv is given. Returns the report dict."""
    from eval_local import compute_metrics
    m_half = compute_metrics(gold_map, preds_at(docs, p_flat, 0.50))
    best_t, best_m = None, None
    for t in GRID:
        m = compute_metrics(gold_map, preds_at(docs, p_flat, t))
        if best_m is None or m["macro_f1"] > best_m["macro_f1"]:
            best_t, best_m = t, m
    line = (f"{arm}\t{seed}\t{m_half['macro_f1'] * 100:.4f}"
            f"\t{best_m['macro_f1'] * 100:.4f}\t{best_t:.2f}"
            f"\t{best_m['macro_precision'] * 100:.4f}"
            f"\t{best_m['macro_recall'] * 100:.4f}\tOK\n")
    if results_tsv:
        with open(results_tsv, "a", encoding="utf-8") as fh:
            fh.write(line)
    sys.stdout.write("SCORED " + line)
    return {"f1_05": round(m_half["macro_f1"] * 100, 4),
            "p_05": round(m_half["macro_precision"] * 100, 4),
            "r_05": round(m_half["macro_recall"] * 100, 4),
            "best_f1": round(best_m["macro_f1"] * 100, 4),
            "best_thr": best_t,
            "best_p": round(best_m["macro_precision"] * 100, 4),
            "best_r": round(best_m["macro_recall"] * 100, 4)}


def anatomy(docs, gold, p_comb, p_tag, p_pool):
    """Decorrelation / anatomy readouts (unet2d + cut-token conventions):
    conf-FP veto counts, phi vs the best single voter, per-genre FP/FN."""
    from genre_buckets import bucket
    rep = {}
    g1 = gold == 1
    conf_fp = (p_tag >= CONF) & ~g1
    conf_tp = (p_tag >= CONF) & g1
    veto_fp = int((p_comb[conf_fp] < 0.5).sum())
    veto_tp = int((p_comb[conf_tp] < 0.5).sum())
    rep["conf_fp"] = {"n": int(conf_fp.sum()), "vetoed": veto_fp}
    rep["conf_tp"] = {"n": int(conf_tp.sum()), "vetoed": veto_tp}
    print(f"  conf-FP veto: {veto_fp} of {int(conf_fp.sum())} tagger "
          f"confident FPs flipped off (gate bar: > 300 of 2,236)")
    print(f"  conf-TP cost: {veto_tp} of {int(conf_tp.sum())} confident TPs "
          f"lost")

    err_c = (p_comb >= 0.5) != g1
    for ref_name, p_ref in (("tagger_s42", p_tag), ("pool_mean", p_pool)):
        err_r = (p_ref >= 0.5) != g1
        phi = float(np.corrcoef(err_c.astype(float),
                                err_r.astype(float))[0, 1])
        fp_c = (p_comb >= 0.5) & ~g1
        fp_r = (p_ref >= 0.5) & ~g1
        jac = int((fp_c & fp_r).sum()) / max(int((fp_c | fp_r).sum()), 1)
        rep[f"phi_vs_{ref_name}"] = {"phi_all_errors": phi,
                                     "fp_jaccard": jac}
        print(f"  phi(all errors) vs {ref_name}: {phi:.4f}  "
              f"FP-Jaccard={jac:.4f} (pool inter-voter mean of record 0.74)")

    per_genre = defaultdict(lambda: [0, 0, 0, 0])   # FPc FNc FPt FNt
    off = 0
    for d in docs:
        n = len(d["tokens"])
        gg = gold[off:off + n] == 1
        pc = p_comb[off:off + n] >= 0.5
        pt = p_tag[off:off + n] >= 0.5
        b = bucket(d["tokens"])
        per_genre[b][0] += int((pc & ~gg).sum())
        per_genre[b][1] += int((~pc & gg).sum())
        per_genre[b][2] += int((pt & ~gg).sum())
        per_genre[b][3] += int((~pt & gg).sum())
        off += n
    rep["per_genre"] = {}
    print("  per-genre FP/FN (combiner@0.5 vs tagger-of-record@0.5):")
    for gname, (a, b, c, dd) in sorted(per_genre.items()):
        rep["per_genre"][gname] = {"comb_fp": a, "comb_fn": b,
                                   "tag_fp": c, "tag_fn": dd}
        print(f"    {gname:20s} comb FP={a:5d} FN={b:5d} | "
              f"tag FP={c:5d} FN={dd:5d}")
    return rep


def ablation_readout(tr, groups, docs_dev, gold_map, dv, seed,
                     full_row, results_tsv=None):
    """MARGINAL-FEATURE readout (pre-registered): refit with each feature
    family ablated, report the dev delta. This is how Noor learns which
    signals carried real weight."""
    fams_present = [f for f in FAMILY_ORDER if f in tr["families"]]
    # voters and voter_derived share one signal source: ablating the raw
    # columns alone leaves v_mean standing in for them (measured in the
    # self-test: -0.88 vs the real -3+), so the joint row is the honest
    # "does the voter signal carry" readout.
    if "voters" in fams_present and "voter_derived" in fams_present:
        fams_present.append("voters+voter_derived")
    rows = {}
    print("  marginal-feature readout (refit without each family):")
    for fam in fams_present:
        drop = set(fam.split("+"))
        keep = [j for j, f in enumerate(tr["families"]) if f not in drop]
        model = fit_lr(tr["X"][:, keep], tr["y"], groups, seed=seed,
                       quiet=True)
        p = combiner_probs(model, dv["X"][:, keep])
        r = score_rows(docs_dev, gold_map, p, f"stacker_lr_wo_{fam}", seed,
                       results_tsv)
        r["delta_f1_05"] = round(r["f1_05"] - full_row["f1_05"], 4)
        r["delta_best"] = round(r["best_f1"] - full_row["best_f1"], 4)
        rows[fam] = r
        print(f"    - {fam:14s} F1@0.5 {r['f1_05']:7.4f} "
              f"(d={r['delta_f1_05']:+.4f})  own-best {r['best_f1']:7.4f} "
              f"(d={r['delta_best']:+.4f})")
    return rows


def cmd_eval(args):
    assert_cpu_only()
    with open(args.model, "rb") as fh:
        model = pickle.load(fh)
    primary = model["primary"] if "primary" in model else model
    dv = load_features(args.dev_features)
    docs = load_jsonl(args.dev_jsonl)
    assert dv["doc_ids"] == [d["doc_id"] for d in docs], \
        "dev feature matrix doc order != dev jsonl"
    assert primary["feature_names"] == dv["names"], \
        "train/dev feature name mismatch — assemble both sides with the " \
        "same manifest families"
    gold, _ = gold_arrays(docs)
    gold_map = {d["doc_id"]: list(d["labels"]) for d in docs}

    z = np.load(args.tagger_probs)
    p_tag = np.concatenate([z[d["doc_id"]].astype(np.float64) for d in docs])
    zp = np.load(args.pool_probs)
    p_pool = np.concatenate([zp[d["doc_id"]].astype(np.float64)
                             for d in docs])

    report = {"model": args.model, "dev_features": args.dev_features}
    print("================ STACKER: DEV REPORT ================")
    report["tagger_ref"] = score_rows(docs, gold_map, p_tag,
                                      "tagger_s42_ref", args.seed)
    report["pool_ref"] = score_rows(docs, gold_map, p_pool,
                                    "pool_mean_ref", args.seed)

    p_comb = combiner_probs(primary, dv["X"])
    report["stacker_lr"] = score_rows(docs, gold_map, p_comb, "stacker_lr",
                                      args.seed, args.results_tsv)
    report["anatomy"] = anatomy(docs, gold, p_comb, p_tag, p_pool)

    if "mlp" in model:
        p_m = combiner_probs(model["mlp"], dv["X"])
        report["stacker_mlp"] = score_rows(docs, gold_map, p_m,
                                           "stacker_mlp", args.seed,
                                           args.results_tsv)

    if args.train_features and args.folds:
        tr = load_features(args.train_features)
        with open(args.folds, encoding="utf-8") as fh:
            fj = json.load(fh)
        fold_of_doc = {}
        for f, blob in fj["folds"].items():
            for i in blob["held_out"]:
                fold_of_doc[i] = int(f)
        groups = np.array([fold_of_doc[tr["doc_ids"][di]]
                           for di in tr["doc_of"]])
        report["ablation"] = ablation_readout(
            tr, groups, docs, gold_map, dv, args.seed,
            report["stacker_lr"], args.results_tsv)

    # ---- pre-registered gates, verbatim (see run_stacker.sh header) --------
    ens = args.ensemble_of_record
    fb = report["stacker_lr"]["best_f1"]
    veto = report["anatomy"]["conf_fp"]["vetoed"]
    nfp = report["anatomy"]["conf_fp"]["n"]
    adopt = fb > ens + 0.2
    interesting = (fb > ens) and (veto > 300)
    print("---------------- PRE-REGISTERED GATES ----------------")
    print(f"  own-best dev macro-F1 = {fb:.4f}; ensemble-of-record = {ens}; "
          f"noise floor = +0.2; conf-FP veto = {veto}/{nfp}")
    print(f"  [{'X' if adopt else ' '}] ADOPT-CANDIDATE iff dev macro-F1 > "
          f"ensemble-of-record 84.53 + 0.2 (ensemble noise floor)")
    print(f"  [{'X' if (interesting and not adopt) else ' '}] INTERESTING "
          f"iff it beats 84.53 at all with conf-FP veto count > 300 of the "
          f"2,236")
    print(f"  [{'X' if not (adopt or interesting) else ' '}] PARK with "
          f"numbers")
    report["gates"] = {"adopt_candidate": bool(adopt),
                       "interesting": bool(interesting and not adopt),
                       "park": bool(not (adopt or interesting)),
                       "ensemble_of_record": ens,
                       "gate_judged_on": "own-best row (like-for-like with "
                                         "the ensemble-of-record dev-selected "
                                         "reading); F1@0.5 headline reported "
                                         "alongside"}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=float)
        print(f"report -> {args.out}")


# =========================================================================== #
# oof-predict: encoder voter -> word-grid npz (+ optional MC-dropout std)
# =========================================================================== #
def cmd_oof_predict(args):
    """Predict a jsonl with a (fold-)voter checkpoint. Eval-mode probs go to
    --out (cache_probs convention). With --mc-passes T > 0 and --std-out,
    ALSO runs T dropout-active passes (model.train(); deterministic seeds per
    (seed, doc, pass)) and writes the per-position std (ddof=1) —
    full-coverage instability, the stacker's train/dev-symmetric variant of
    stability/instability_filter_2026-07-10 (which covers confident positions
    only). GPU allowed here — the battery runner owns scheduling."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from data import MODEL_PAR_TOKEN
    from predict import predict_doc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = load_jsonl(args.jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model = model.to(device).eval()
    words_per_doc = [
        [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        for d in docs]

    out = {}
    for d, words in zip(docs, words_per_doc):
        out[d["doc_id"]] = predict_doc(words, tokenizer, model, device,
                                       args.window, args.overlap,
                                       args.max_length)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    assert not os.path.exists(args.out), f"{args.out} exists — append-only"
    np.savez(args.out, **out)
    print(f"eval-mode probs: {len(docs)} docs -> {args.out}")

    if args.mc_passes > 0:
        assert args.std_out, "--mc-passes needs --std-out"
        model.train()                     # dropout ACTIVE; still no_grad
        stds = {}
        for di, (d, words) in enumerate(zip(docs, words_per_doc)):
            passes = []
            for t in range(args.mc_passes):
                torch.manual_seed(args.seed * 100003 + di * 1009 + t)
                passes.append(predict_doc(words, tokenizer, model, device,
                                          args.window, args.overlap,
                                          args.max_length))
            stds[d["doc_id"]] = np.std(np.stack(passes), axis=0,
                                       ddof=1).astype(np.float32)
        assert not os.path.exists(args.std_out), \
            f"{args.std_out} exists — append-only"
        np.savez(args.std_out, **stds)
        print(f"MC-dropout std (T={args.mc_passes}): {len(docs)} docs -> "
              f"{args.std_out}")


# =========================================================================== #
# unet-sharpness: MEASURE (not assume) the S-only U-Net's memorization
# =========================================================================== #
def cmd_unet_sharpness(args):
    """Train-vs-dev prob comparison for the S-only U-Net checkpoints. The
    U-Net trains with GOLD targets on train windows, so it CAN memorize; but
    its input is similarity matrices from a frozen encoder, which plausibly
    limits it. Readout per ckpt: token AUC vs gold and mean p at gold-1 /
    gold-0 (sharpness), train vs dev. PRE-REGISTERED DECISION RULE (consumed
    by run_stacker.sh): if mean (train AUC - dev AUC) > 0.05 OR mean train
    AUC > 0.98, the U-Net train-side features must come from the fold OOF
    retrains (`unet-oof`, CPU); else the existing s*r1 ckpts serve both
    sides with this measurement pasted as the justification."""
    assert_cpu_only()
    import torch
    from unet2d_train import (build_model, flatten_probs, load_cache,
                              predict_windows, stitch)

    rep = {"ckpts": {}, "rule": "OOF retrain iff mean(train_auc - dev_auc) "
                                "> 0.05 or mean train_auc > 0.98"}
    gaps, tr_aucs = [], []
    caches = {}
    for split, cpath, jpath in (("train", args.train_cache,
                                 args.train_jsonl),
                                ("dev", args.dev_cache, args.dev_jsonl)):
        caches[split] = (load_cache(cpath), load_jsonl(jpath))
        assert [str(x) for x in caches[split][0]["doc_ids"]] \
            == [d["doc_id"] for d in caches[split][1]], f"{split} order"

    for ck in args.ckpts:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        in_ch = int(blob.get("in_ch", 2))
        model = build_model(base=blob["base"], in_ch=in_ch)
        model.load_state_dict(blob["state_dict"])
        r = {}
        for split in ("train", "dev"):
            cache, docs = caches[split]
            pw = predict_windows(model, cache, no_tagger=(in_ch == 1))
            per_doc, _ = stitch(cache, pw)
            p = flatten_probs(docs, per_doc)
            gold, _ = gold_arrays(docs)
            r[split] = {
                "token_auc": round(rank_auc_np(p, gold == 1), 4),
                "mean_p_gold1": round(float(p[gold == 1].mean()), 4),
                "mean_p_gold0": round(float(p[gold == 0].mean()), 4),
                "frac_extreme": round(float(((p < 0.05) | (p > 0.95))
                                            .mean()), 4)}
        r["auc_gap_train_minus_dev"] = round(
            r["train"]["token_auc"] - r["dev"]["token_auc"], 4)
        gaps.append(r["auc_gap_train_minus_dev"])
        tr_aucs.append(r["train"]["token_auc"])
        rep["ckpts"][ck] = r
        print(f"{ck}:")
        for split in ("train", "dev"):
            print(f"  {split:5s} AUC={r[split]['token_auc']:.4f} "
                  f"mean_p@gold1={r[split]['mean_p_gold1']:.4f} "
                  f"mean_p@gold0={r[split]['mean_p_gold0']:.4f} "
                  f"frac_extreme={r[split]['frac_extreme']:.4f}")
        print(f"  AUC gap (train - dev) = "
              f"{r['auc_gap_train_minus_dev']:+.4f}")
    need = (float(np.mean(gaps)) > 0.05) or (float(np.mean(tr_aucs)) > 0.98)
    rep["mean_auc_gap"] = round(float(np.mean(gaps)), 4)
    rep["mean_train_auc"] = round(float(np.mean(tr_aucs)), 4)
    rep["oof_retrain_required"] = bool(need)
    print(f"MEASURED: mean AUC gap {rep['mean_auc_gap']:+.4f}, mean train "
          f"AUC {rep['mean_train_auc']:.4f} -> U-Net OOF retrain required: "
          f"{need}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print(f"-> {args.out}")


# =========================================================================== #
# unet-oof: fold-restricted S-only U-Net retrain -> OOF train probs (CPU)
# =========================================================================== #
def cmd_unet_oof(args):
    """One (fold, seed) cell: train the S-only U-Net on the fold-train docs'
    windows, select on the inner-dev docs' stitched F1@0.5, predict the
    held-out fold windows -> per-doc OOF probs npz (held docs only; merge
    with merge-npz). Reuses unet2d_train's model/stitch verbatim. NOTE OF
    RECORD: the similarity-matrix INPUT still comes from the frozen full-fit
    s42 encoder — this OOF pass removes the U-Net's own gold-target
    memorization only; the encoder-representation leak is measured by
    unet-sharpness and accepted as a caveat."""
    assert_cpu_only()
    import torch
    from knn_boundary import macro_f1
    from unet2d_train import (build_model, load_cache, make_inputs,
                              predict_windows, stitch)

    with open(args.folds, encoding="utf-8") as fh:
        fj = json.load(fh)
    blob = fj["folds"][str(args.fold)]
    cache = load_cache(args.train_cache)
    docs = load_jsonl(args.train_jsonl)
    assert [str(x) for x in cache["doc_ids"]] == [d["doc_id"] for d in docs]
    id2di = {d["doc_id"]: i for i, d in enumerate(docs)}
    sets = {part: np.array(sorted(id2di[i] for i in blob[part]))
            for part in ("train", "inner_dev", "held_out")}
    win_doc = cache["doc_idx"]
    idx_tr = np.flatnonzero(np.isin(win_doc, sets["train"]))
    idx_in = np.flatnonzero(np.isin(win_doc, sets["inner_dev"]))
    idx_he = np.flatnonzero(np.isin(win_doc, sets["held_out"]))
    print(f"# unet-oof fold={args.fold} seed={args.seed}: windows "
          f"train={len(idx_tr)} inner={len(idx_in)} held={len(idx_he)}")

    def sub(idx):
        c = {k: cache[k][idx] for k in ("S", "P", "G", "L", "doc_idx",
                                        "start")}
        c["doc_len"] = cache["doc_len"]
        c["doc_ids"] = cache["doc_ids"]
        return c

    tr, inn, he = sub(idx_tr), sub(idx_in), sub(idx_he)
    gold_in = np.concatenate([np.asarray(docs[di]["labels"], dtype=np.int8)
                              for di in sets["inner_dev"]])
    doc_of_in = np.concatenate([np.full(len(docs[di]["labels"]), j)
                                for j, di in enumerate(sets["inner_dev"])])

    torch.manual_seed(args.seed)
    model = build_model(base=args.base, in_ch=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(args.pos_weight)), reduction="none")
    rng = np.random.default_rng(args.seed)
    N = tr["S"].shape[0]
    best = (-1.0, -1, None)
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(N)
        t0 = time.time()
        for s in range(0, N, args.batch_size):
            idx = order[s:s + args.batch_size]
            x, vm = make_inputs(tr["S"][idx], tr["P"][idx], tr["L"][idx],
                                no_tagger=True)
            y = torch.from_numpy(tr["G"][idx].astype(np.float32))
            loss = (bce(model(x), y) * vm).sum() / vm.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        pw = predict_windows(model, inn, no_tagger=True)
        per_doc, _ = stitch(inn, pw)
        p_in = np.concatenate([per_doc[di] for di in sets["inner_dev"]])
        f05 = macro_f1(p_in, gold_in, doc_of_in, len(sets["inner_dev"]), 0.5)
        star = ""
        if f05 > best[0]:
            best = (f05, ep, {k: v.clone() for k, v in
                              model.state_dict().items()})
            star = " *best*"
        print(f"  epoch {ep:3d} inner F1@0.5 {100 * f05:.2f} "
              f"({time.time() - t0:.0f}s){star}", flush=True)
        if ep - best[1] >= args.patience:
            print(f"  early stop (patience {args.patience})")
            break
    model.load_state_dict(best[2])
    pw = predict_windows(model, he, no_tagger=True)
    per_doc, _ = stitch(he, pw)
    out = {docs[di]["doc_id"]: per_doc[di].astype(np.float64)
           for di in sets["held_out"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    assert not os.path.exists(args.out), f"{args.out} exists — append-only"
    np.savez(args.out, **out)
    print(f"OOF probs for {len(out)} held docs (inner-best epoch "
          f"{best[1]}, inner F1 {100 * best[0]:.2f}) -> {args.out}")


# =========================================================================== #
# self-test: synthetic end-to-end, planted complementary errors
# =========================================================================== #
def synth_docs(rng, n_docs, len_lo=60, len_hi=160):
    docs = []
    for k in range(n_docs):
        n = int(rng.integers(len_lo, len_hi))
        lab = np.zeros(n, dtype=int)
        i = 0
        while True:
            i += int(rng.integers(4, 13))
            if i >= n - 1:
                break
            lab[i] = 1
        lab[n - 1] = 1                       # doc-final boundary (gold rule)
        docs.append({"doc_id": f"synth_{k:04d}",
                     "tokens": [f"w{j}" for j in range(n)],
                     "labels": lab.tolist()})
    return docs


def synth_features(rng, docs, qualities=(0.90, 0.80, 0.70, 0.40)):
    """Fake voters with PLANTED COMPLEMENTARY reliabilities: voter j reads a
    noisy copy of gold with prob q_j, junk otherwise (per position). Also a
    weak planted signal (the 'unet' role) and a USELESS pure-noise column
    (the 'judge' role — its ablation delta must be ~0). Mimics the OOF
    regime: nothing is memorized."""
    gold, doc_of = gold_arrays(docs)
    N = len(gold)
    cols, names, fams = [], [], []
    for j, q in enumerate(qualities):
        informative = rng.random(N) < q
        sig = np.clip(gold * 0.72 + 0.14 + rng.normal(0, 0.16, N), 0, 1)
        junk = np.clip(rng.beta(2, 2, N), 0, 1)
        cols.append(np.where(informative, sig, junk))
        names.append(f"voter_q{q:.2f}_{j}")
        fams.append("voters")
    V = np.stack(cols, axis=1)
    d, dn, df = feat_derived(V)
    weak = np.clip(gold * 0.30 + 0.35 + rng.normal(0, 0.25, N), 0, 1)
    useless = rng.random(N)
    pos, pn, pf = feat_position(docs)
    X = np.concatenate([V, d, weak[:, None], useless[:, None], pos], axis=1)
    names = names + dn + ["weak_planted"] + ["useless_planted"] + pn
    fams = fams + [df] * 3 + ["unet"] + ["judge"] + [pf] * 3
    return X, names, fams, gold.astype(int), doc_of


def cmd_self_test(args):
    assert_cpu_only()
    rng = np.random.default_rng(0)
    docs = synth_docs(rng, args.n_docs)
    half = len(docs) // 2
    d_tr, d_ev = docs[:half], docs[half:]
    Xtr, names, fams, ytr, _ = synth_features(rng, d_tr)
    Xev, _, _, yev, _ = synth_features(rng, d_ev)
    gold_map = {d["doc_id"]: list(d["labels"]) for d in d_ev}

    # fold assignment machinery on synthetic ids (determinism checked)
    ids = [d["doc_id"] for d in d_tr]
    lens = [len(d["tokens"]) for d in d_tr]
    f1 = length_stratified_folds(ids, lens, 5, 42)
    f2 = length_stratified_folds(ids, lens, 5, 42)
    assert f1 == f2 and sorted(set(f1)) == [0, 1, 2, 3, 4]
    groups = np.concatenate([np.full(len(d["tokens"]), f1[i])
                             for i, d in enumerate(d_tr)])
    print(f"PASS 1: fold maker deterministic, k=5 all present "
          f"(sizes {[f1.count(v) for v in range(5)]})")

    # tripwire must fire on memorized ("full-fit") voter columns
    Xleak = Xtr.copy()
    Xleak[:, 0] = np.clip(ytr + rng.normal(0, 0.01, len(ytr)), 0, 1)
    try:
        oof_leak_tripwire(Xleak, names, fams, ytr)
        raise SystemExit("tripwire FAILED to fire on a memorized column")
    except AssertionError:
        print("PASS 2: OOF-leak tripwire fires on a memorized voter column")

    worst = oof_leak_tripwire(Xtr, names, fams, ytr)
    print(f"PASS 3: honest OOF-style columns clear the tripwire "
          f"(worst AUC {worst:.4f} <= {OOF_AUC_TRIPWIRE})")

    model = fit_lr(Xtr, ytr, groups, seed=42, quiet=True)
    model["feature_names"], model["families"] = names, fams

    # references: best single voter and the unweighted mean (the incumbents)
    rows = {}
    for j in range(4):
        rows[names[j]] = score_rows(d_ev, gold_map, Xev[:, j],
                                    names[j], 42)
    p_mean = Xev[:, :4].mean(axis=1)
    rows["mean"] = score_rows(d_ev, gold_map, p_mean, "voter_mean", 42)
    p_c = combiner_probs(model, Xev)
    rows["stacker"] = score_rows(d_ev, gold_map, p_c, "stacker_lr", 42)

    best_single = max(rows[names[j]]["best_f1"] for j in range(4))
    best_single_05 = max(rows[names[j]]["f1_05"] for j in range(4))
    ok_single = (rows["stacker"]["best_f1"] > best_single
                 and rows["stacker"]["f1_05"] > best_single_05)
    ok_mean = (rows["stacker"]["best_f1"] > rows["mean"]["best_f1"]
               and rows["stacker"]["f1_05"] > rows["mean"]["f1_05"])
    assert ok_single, "combiner failed to beat the best single voter"
    assert ok_mean, "combiner failed to beat the unweighted mean"
    print(f"PASS 4: combiner beats best single voter "
          f"({rows['stacker']['best_f1']:.2f} > {best_single:.2f}) and the "
          f"mean ({rows['stacker']['best_f1']:.2f} > "
          f"{rows['mean']['best_f1']:.2f}) at own-best AND @0.5")

    # planted-signal ablation discipline: the voter signal (raw + derived —
    # one source) must matter, the useless 'judge' column must not
    deltas = {}
    for fam in ("voters+voter_derived", "judge"):
        drop = set(fam.split("+"))
        keep = [j for j, f in enumerate(fams) if f not in drop]
        m2 = fit_lr(Xtr[:, keep], ytr, groups, seed=42, quiet=True)
        p2 = combiner_probs(m2, Xev[:, keep])
        r2 = score_rows(d_ev, gold_map, p2, f"ablate_{fam}", 42)
        deltas[fam] = r2["best_f1"] - rows["stacker"]["best_f1"]
    assert deltas["voters+voter_derived"] < -2.0, \
        f"ablating the voter signal should crater F1 " \
        f"(delta {deltas['voters+voter_derived']:+.2f})"
    assert abs(deltas["judge"]) < 0.6, \
        f"ablating the useless column moved F1 {deltas['judge']:+.2f}"
    print(f"PASS 5: ablation readout separates planted signal from noise "
          f"(voters {deltas['voters+voter_derived']:+.2f}, "
          f"useless {deltas['judge']:+.2f})")

    print("SELF-TEST PASS (5/5)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "ablation_deltas": deltas,
                       "n_docs": args.n_docs}, fh, indent=1, default=float)
        print(f"-> {args.out}")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("make-folds", help="k-fold split of the train docs")
    f.add_argument("--train-jsonl", required=True)
    f.add_argument("--k", type=int, default=5)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--inner-frac", type=float, default=0.15)
    f.add_argument("--expect-docs", type=int, default=174)
    f.add_argument("--task", default="NoPnx-NP")
    f.add_argument("--out", required=True)
    f.add_argument("--jsonl-dir", default=None,
                   help="also materialize f<k>_{train,inner_dev,held_out}"
                        ".jsonl here")
    f.set_defaults(fn=cmd_make_folds)

    m = sub.add_parser("merge-npz", help="merge disjoint word-grid npzs")
    m.add_argument("--parts", nargs="+", required=True)
    m.add_argument("--jsonl", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_merge_npz)

    a = sub.add_parser("assemble", help="manifest -> feature matrix npz")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_assemble)

    t = sub.add_parser("fit", help="fit the combiner on the OOF train matrix")
    t.add_argument("--train-features", required=True)
    t.add_argument("--folds", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--mlp", action="store_true",
                   help="also fit the flagged secondary small MLP")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_fit)

    e = sub.add_parser("eval", help="dev evaluation + anatomy + ablation")
    e.add_argument("--model", required=True)
    e.add_argument("--dev-features", required=True)
    e.add_argument("--dev-jsonl", required=True)
    e.add_argument("--tagger-probs",
                   default="probs/union-NoPnx-NP-arabertv02-s42_dev.npz")
    e.add_argument("--pool-probs",
                   default="probs/union-NoPnx-NP-POOL_dev.npz")
    e.add_argument("--train-features", default=None,
                   help="enables the marginal-feature ablation readout")
    e.add_argument("--folds", default=None)
    e.add_argument("--results-tsv", default=None)
    e.add_argument("--ensemble-of-record", type=float, default=84.53)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)

    o = sub.add_parser("oof-predict",
                       help="voter ckpt -> word-grid npz (+ MC std)")
    o.add_argument("--model", required=True)
    o.add_argument("--jsonl", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--window", type=int, default=180)
    o.add_argument("--overlap", type=int, default=60)
    o.add_argument("--max-length", type=int, default=512)
    o.add_argument("--mc-passes", type=int, default=0)
    o.add_argument("--std-out", default=None)
    o.add_argument("--seed", type=int, default=42)
    o.set_defaults(fn=cmd_oof_predict)

    u = sub.add_parser("unet-sharpness",
                       help="measure S-only U-Net train-vs-dev memorization")
    u.add_argument("--ckpts", nargs="+", required=True)
    u.add_argument("--train-cache", required=True)
    u.add_argument("--dev-cache", required=True)
    u.add_argument("--train-jsonl", required=True)
    u.add_argument("--dev-jsonl", required=True)
    u.add_argument("--out", default=None)
    u.set_defaults(fn=cmd_unet_sharpness)

    w = sub.add_parser("unet-oof",
                       help="one (fold, seed) S-only U-Net OOF cell (CPU)")
    w.add_argument("--folds", required=True)
    w.add_argument("--fold", type=int, required=True)
    w.add_argument("--train-cache", required=True)
    w.add_argument("--train-jsonl", required=True)
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--epochs", type=int, default=40)
    w.add_argument("--batch-size", type=int, default=8)
    w.add_argument("--lr", type=float, default=1e-3)
    w.add_argument("--pos-weight", type=float, default=8.0)
    w.add_argument("--patience", type=int, default=8)
    w.add_argument("--base", type=int, default=48)
    w.add_argument("--out", required=True)
    w.set_defaults(fn=cmd_unet_oof)

    s = sub.add_parser("self-test",
                       help="corpus-independent synthetic end-to-end")
    s.add_argument("--n-docs", type=int, default=80)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_self_test)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
