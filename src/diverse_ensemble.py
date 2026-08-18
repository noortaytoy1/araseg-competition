"""Diverse-pool ensemble comparison over Omar's DP decode + official macro-F1.

Purely ADDITIVE. Reuses (never modifies) the load-bearing machinery:
  * dp_decode.decode_doc / fit_length_logprob    (Omar's DP decode)
  * eval_local.compute_metrics                    (official per-doc macro-F1)
  * moe_metalearner: the corpus-independent meta-learner helpers
      (context_features_for_doc, build_design, fit_meta, build_train_vocab,
       length_stratified_folds via stacker) — imported, not edited.

moe_metalearner.py is hardwired to Omar's SIX AraBERT-heavy encoders (its
MEMBERS / VOTER_NAMES are pinned by tests/test_moe_metalearner.py). This script
does NOT touch those; it re-implements the tiny cache-loading / matrix-assembly
/ interaction wrappers with a *configurable* member list so we can drive the
same meta-learner over ANY pool (old / diverse / full), while every arm is
scored through the identical Omar DP decode with a single frozen (lam, bias).

Pools (task-specified):
  OLD    : arabertv02-s1, -s2, -s3, -s42, araelectra-s42, arbertv2-s42   (6)
  DIVERSE: arabertv02-s1, araelectra-s42, arbertv2-s42, + new encoders   (drop
           the redundant/harmful AraBERT seeds s2/s3/s42; add whichever of
           camelbert/marbertv2/qarib actually trained)
  FULL   : OLD 6 + the new ones

Everything is dev-only k=5 doc-level CV (length_stratified_folds, seed 42, no
doc across folds). Combiners per pool:
  uniform   : plain average of the pool's voters
  weighted  : Omar's greedy integer-weight search (weighted_ensemble.py logic),
              fit per-fold on the train fold, applied out-of-fold
  meta      : the context-gated meta-learner (LR), fit per-fold, applied OOF

Frozen decode: lam=0.0, bias=-0.25 (the config on record for this ensemble),
applied identically to every pool and every arm. Pass --refit-decode to instead
freeze (lam,bias) on each pool's own uniform-OOF over dp_decode's grid.

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python src/diverse_ensemble.py \
        --task NoPnx-NP --out-dir runs/diverse_ensemble_2026-07-11
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from data import PARAGRAPH_TOKEN, load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from eval_local import compute_metrics
from stacker import length_stratified_folds

# Reused, unmodified, from the load-bearing meta-learner:
from moe_metalearner import (
    CONTEXT_FEATURES,
    build_train_vocab,
    context_features_for_doc,
    fit_meta,
)

# --------------------------------------------------------------------------- #
# Pool definitions. New encoders are auto-filtered to those with a dev cache.
# --------------------------------------------------------------------------- #
OLD_POOL = [
    "arabertv02-s1", "arabertv02-s2", "arabertv02-s3", "arabertv02-s42",
    "araelectra-s42", "arbertv2-s42",
]
# Kept-diverse core (drop s2/s3/s42 per the recorded leave-one-out reading).
DIVERSE_CORE = ["arabertv02-s1", "araelectra-s42", "arbertv2-s42"]
# Candidate new encoders (whichever trained; missing ones are skipped).
NEW_CANDIDATES = ["camelbert-s42", "marbertv2-s42", "qarib-s42"]

# The frozen decode on record for this ensemble.
FROZEN_LAM = 0.0
FROZEN_BIAS = -0.25

# dp_decode's own default grid (only used with --refit-decode).
LAM_GRID = [0.0, 0.25, 0.5, 1.0, 1.5]
BIAS_GRID = [-0.5, -0.25, 0.0, 0.25, 0.5]


# --------------------------------------------------------------------------- #
# Cache loading / matrix assembly, parameterized on an arbitrary member list.
# (Mirrors moe_metalearner.load_voter_caches / voter_matrix_for_doc, but the
#  member list is an argument instead of a module global.)
# --------------------------------------------------------------------------- #
def cache_path(task, member, split, probs_dir):
    return os.path.join(probs_dir, f"union-{task}-{member}_{split}.npz")


def available_members(task, members, split="dev", probs_dir="probs"):
    return [m for m in members if os.path.exists(cache_path(task, m, split, probs_dir))]


def load_caches(task, members, split="dev", probs_dir="probs"):
    caches = {}
    for m in members:
        p = cache_path(task, m, split, probs_dir)
        if not os.path.exists(p):
            raise FileNotFoundError(f"voter cache missing: {p}")
        caches[m] = np.load(p)
    return caches


def voter_matrix_for_doc(caches, members, doc_id, n_tokens):
    cols = []
    for m in members:
        c = caches[m]
        if doc_id not in c.files:
            raise KeyError(f"{doc_id} absent from voter cache {m}")
        v = np.asarray(c[doc_id], dtype=np.float64)
        if v.shape[0] != n_tokens:
            raise ValueError(f"{m}:{doc_id} len {v.shape[0]} != {n_tokens}")
        cols.append(v)
    return np.stack(cols, axis=1)


def assemble(docs, caches, members, train_vocab):
    """Flat token-level arrays: V (N,M voters), C (N,F context), y, doc_of, spans."""
    Vs, Cs, ys, dof, spans = [], [], [], [], []
    ids = [d["doc_id"] for d in docs]
    cur = 0
    for di, d in enumerate(docs):
        toks = d["tokens"]
        n = len(toks)
        Vs.append(voter_matrix_for_doc(caches, members, d["doc_id"], n))
        Cs.append(context_features_for_doc(toks, train_vocab))
        ys.append(np.asarray(d["labels"], dtype=np.int64))
        dof.append(np.full(n, di, dtype=np.int64))
        spans.append((cur, cur + n))
        cur += n
    V = np.concatenate(Vs, axis=0)
    C = np.concatenate(Cs, axis=0)
    y = np.concatenate(ys, axis=0)
    doc_of = np.concatenate(dof, axis=0)
    return V, C, y, doc_of, ids, spans


def interaction_block(V, C, voter_names):
    """context x voter interactions (drop the bias context column)."""
    bias_idx = CONTEXT_FEATURES.index("bias")
    keep = [j for j in range(C.shape[1]) if j != bias_idx]
    blocks, names = [], []
    for vi in range(V.shape[1]):
        for cj in keep:
            blocks.append((V[:, vi] * C[:, cj])[:, None])
            names.append(f"{voter_names[vi]}×{CONTEXT_FEATURES[cj]}")
    return np.concatenate(blocks, axis=1), names


def build_design(V, C, voter_names, with_interactions=True):
    parts = [V, C]
    names = list(voter_names) + list(CONTEXT_FEATURES)
    if with_interactions:
        inter, inter_names = interaction_block(V, C, voter_names)
        parts.append(inter)
        names += inter_names
    return np.concatenate(parts, axis=1), names


# --------------------------------------------------------------------------- #
# Omar's greedy integer-weight search (weighted_ensemble.py logic), on train
# fold only, member-count agnostic.
# --------------------------------------------------------------------------- #
def omar_greedy_weights(train_docs, train_idx, V, spans, ids, gold, len_lp,
                        lam, bias, m):
    tr_index = {ids[i]: spans[i] for i in train_idx}

    def score(weights):
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        preds = {}
        for d in train_docs:
            s, e = tr_index[d["doc_id"]]
            preds[d["doc_id"]] = decode_doc(d["tokens"], V[s:e] @ w, len_lp, lam, bias)
        g = {d["doc_id"]: gold[d["doc_id"]] for d in train_docs}
        return compute_metrics(g, preds)["macro_f1"]

    best_w = [1] * m
    best = score(best_w)
    for _ in range(m):
        improved = False
        for i in range(m):
            for delta in (1, -1):
                w = list(best_w)
                w[i] += delta
                if w[i] < 0 or sum(w) == 0:
                    continue
                s = score(w)
                if s > best + 1e-5:
                    best, best_w, improved = s, w, True
        if not improved:
            break
    w = np.array(best_w, dtype=float)
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Decode config: frozen (default) or refit on each pool's uniform-OOF.
# --------------------------------------------------------------------------- #
def pick_decode_config(docs, blended_by_doc, len_lp, gold):
    best = (LAM_GRID[0], BIAS_GRID[0], -1.0)
    for lam in LAM_GRID:
        for bias in BIAS_GRID:
            preds = {d["doc_id"]: decode_doc(d["tokens"], blended_by_doc[d["doc_id"]],
                                             len_lp, lam, bias) for d in docs}
            f1 = compute_metrics(gold, preds)["macro_f1"]
            if f1 > best[2]:
                best = (lam, bias, f1)
    return best[0], best[1]


# --------------------------------------------------------------------------- #
# One pool: k=5 CV over uniform / weighted / meta, all through Omar's DP decode.
# --------------------------------------------------------------------------- #
def run_pool(docs, caches, members, train_vocab, len_lp, gold, fold_of, ids,
             spans, lam, bias, do_meta=True, meta_kind="lr", seed=42,
             refit_decode=False):
    m = len(members)
    V, C, y, doc_of, ids2, spans2 = assemble(docs, caches, members, train_vocab)
    assert ids2 == ids and spans2 == spans
    X_full, feat_names = build_design(V, C, members, with_interactions=do_meta)

    # optionally refit the decode config on this pool's own uniform-OOF
    if refit_decode:
        uni_by_doc = {ids[di]: V[s:e].mean(axis=1) for di, (s, e) in enumerate(spans)}
        lam, bias = pick_decode_config(docs, uni_by_doc, len_lp, gold)

    preds_uniform, preds_omar, preds_meta = {}, {}, {}
    omar_w_folds = []
    meta_coef_accum = np.zeros(X_full.shape[1])
    meta_coef_folds = 0

    for f in sorted(set(fold_of)):
        held = [i for i, ff in enumerate(fold_of) if ff == f]
        train = [i for i, ff in enumerate(fold_of) if ff != f]
        train_docs = [docs[i] for i in train]
        tr_mask = np.isin(doc_of, np.array(train))

        # uniform
        for di in held:
            s, e = spans[di]
            preds_uniform[ids[di]] = decode_doc(
                docs[di]["tokens"], V[s:e].mean(axis=1), len_lp, lam, bias)

        # omar greedy
        w = omar_greedy_weights(train_docs, train, V, spans, ids, gold, len_lp,
                                lam, bias, m)
        omar_w_folds.append(w)
        for di in held:
            s, e = spans[di]
            preds_omar[ids[di]] = decode_doc(
                docs[di]["tokens"], V[s:e] @ w, len_lp, lam, bias)

        # meta
        if do_meta:
            prob_fn, coef = fit_meta(X_full[tr_mask], y[tr_mask], meta_kind, seed)
            if coef is not None:
                meta_coef_accum += coef
                meta_coef_folds += 1
            for di in held:
                s, e = spans[di]
                preds_meta[ids[di]] = decode_doc(
                    docs[di]["tokens"], prob_fn(X_full[s:e]), len_lp, lam, bias)

    out = {
        "members": list(members),
        "m": m,
        "lam": lam,
        "bias": bias,
        "uniform": compute_metrics(gold, preds_uniform),
        "omar": compute_metrics(gold, preds_omar),
        "omar_weights_mean": np.mean(omar_w_folds, axis=0).tolist(),
        "omar_weights_per_fold": [wi.tolist() for wi in omar_w_folds],
    }
    if do_meta:
        out["meta"] = compute_metrics(gold, preds_meta)
        out["meta_coef_mean"] = (
            (meta_coef_accum / meta_coef_folds).tolist() if meta_coef_folds else None)
        out["feat_names"] = feat_names
    return out


def solo_f1(docs, caches, member, len_lp, gold, lam, bias):
    """Single-voter decoded macro-F1 (the encoder alone through Omar's DP)."""
    c = caches[member]
    preds = {d["doc_id"]: decode_doc(d["tokens"],
                                     np.asarray(c[d["doc_id"]], dtype=np.float64),
                                     len_lp, lam, bias) for d in docs}
    return compute_metrics(gold, preds)["macro_f1"] * 100


def f1_of(res, key):
    return res[key]["macro_f1"] * 100


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--meta", default="lr", choices=["lr", "mlp"])
    ap.add_argument("--refit-decode", action="store_true",
                    help="freeze (lam,bias) per-pool on uniform-OOF grid instead "
                         "of the recorded lam=0.0 bias=-0.25")
    ap.add_argument("--out-dir", default="runs/diverse_ensemble_2026-07-11")
    args = ap.parse_args()

    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    train_docs = load_jsonl(f"data/{args.task}_train.jsonl")
    train_vocab = build_train_vocab(train_docs)
    len_lp = fit_length_logprob(train_docs)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    ids = [d["doc_id"] for d in docs]
    lens = [len(d["tokens"]) for d in docs]
    fold_of = length_stratified_folds(ids, lens, args.k, args.seed)
    spans, cur = [], 0
    for d in docs:
        spans.append((cur, cur + len(d["tokens"])))
        cur += len(d["tokens"])

    # resolve which new encoders actually have a cache
    new_present = available_members(args.task, NEW_CANDIDATES, args.split)
    new_absent = [m for m in NEW_CANDIDATES if m not in new_present]
    diverse_pool = DIVERSE_CORE + new_present
    full_pool = OLD_POOL + new_present

    lam0, bias0 = FROZEN_LAM, FROZEN_BIAS

    # load every cache we might touch, once
    all_members = sorted(set(OLD_POOL + diverse_pool + full_pool))
    caches = load_caches(args.task, all_members, args.split)

    # ------------------------------------------------------------------ #
    # (1) OLD POOL: uniform + weighted   (should reproduce ~84.70/~84.94)
    # ------------------------------------------------------------------ #
    old = run_pool(docs, caches, OLD_POOL, train_vocab, len_lp, gold, fold_of,
                   ids, spans, lam0, bias0, do_meta=False,
                   refit_decode=args.refit_decode)

    # ------------------------------------------------------------------ #
    # (2) DIVERSE POOL: uniform + weighted + meta
    # ------------------------------------------------------------------ #
    diverse = run_pool(docs, caches, diverse_pool, train_vocab, len_lp, gold,
                       fold_of, ids, spans, lam0, bias0, do_meta=True,
                       meta_kind=args.meta, seed=args.seed,
                       refit_decode=args.refit_decode)

    # ------------------------------------------------------------------ #
    # (3) FULL POOL: uniform + weighted (+ meta for completeness)
    # ------------------------------------------------------------------ #
    full = run_pool(docs, caches, full_pool, train_vocab, len_lp, gold, fold_of,
                    ids, spans, lam0, bias0, do_meta=True, meta_kind=args.meta,
                    seed=args.seed, refit_decode=args.refit_decode)

    # ------------------------------------------------------------------ #
    # Solo F1 of every member (single encoder through Omar's DP decode)
    # ------------------------------------------------------------------ #
    solos = {m: solo_f1(docs, caches, m, len_lp, gold, lam0, bias0)
             for m in all_members}

    # ------------------------------------------------------------------ #
    # (4) LEAVE-ONE-OUT on the FULL pool: drop-delta of each member (the
    #     drop is measured on the WEIGHTED combiner, the "let the weighter
    #     keep whatever helps" version). Positive delta => member helps.
    # ------------------------------------------------------------------ #
    loo_pool = full_pool
    base_full_weighted = f1_of(full, "omar")
    loo = {}
    for m in loo_pool:
        sub = [x for x in loo_pool if x != m]
        r = run_pool(docs, caches, sub, train_vocab, len_lp, gold, fold_of, ids,
                     spans, lam0, bias0, do_meta=False,
                     refit_decode=args.refit_decode)
        f1_without = f1_of(r, "omar")
        loo[m] = {
            "weighted_without": f1_without,
            "drop_delta": base_full_weighted - f1_without,  # >0 => member helps
        }

    # weight the weighter gave each NEW encoder in the FULL pool
    full_w = full["omar_weights_mean"]
    new_weight_in_full = {m: full_w[full["members"].index(m)] for m in new_present}
    diverse_w = diverse["omar_weights_mean"]
    new_weight_in_diverse = {m: diverse_w[diverse["members"].index(m)]
                             for m in new_present}

    # ------------------------------------------------------------------ #
    # PRE-REGISTERED READING
    # ------------------------------------------------------------------ #
    old_best = max(f1_of(old, "uniform"), f1_of(old, "omar"))
    diverse_best = max(f1_of(diverse, "uniform"), f1_of(diverse, "omar"),
                       f1_of(diverse, "meta"))
    delta_best = diverse_best - old_best
    if delta_best > 0.2:
        verdict = "WIN"
    elif delta_best > 0.0:
        verdict = "MARGINAL"
    else:
        verdict = "NULL"

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    def pr(res, key):
        return [res[key]["macro_precision"] * 100, res[key]["macro_recall"] * 100]

    report = {
        "task": args.task, "split": args.split, "k": args.k, "seed": args.seed,
        "meta": args.meta, "frozen_decode": {"lam": lam0, "bias": bias0},
        "refit_decode": args.refit_decode,
        "new_candidates": NEW_CANDIDATES,
        "new_present": new_present, "new_absent": new_absent,
        "pools": {
            "old": OLD_POOL, "diverse": diverse_pool, "full": full_pool,
        },
        "old": {"uniform": f1_of(old, "uniform"), "weighted": f1_of(old, "omar"),
                "uniform_PR": pr(old, "uniform"), "weighted_PR": pr(old, "omar"),
                "omar_weights_mean": old["omar_weights_mean"]},
        "diverse": {"uniform": f1_of(diverse, "uniform"),
                    "weighted": f1_of(diverse, "omar"),
                    "meta": f1_of(diverse, "meta"),
                    "uniform_PR": pr(diverse, "uniform"),
                    "weighted_PR": pr(diverse, "omar"),
                    "meta_PR": pr(diverse, "meta"),
                    "omar_weights_mean": diverse["omar_weights_mean"]},
        "full": {"uniform": f1_of(full, "uniform"),
                 "weighted": f1_of(full, "omar"),
                 "meta": f1_of(full, "meta"),
                 "uniform_PR": pr(full, "uniform"),
                 "weighted_PR": pr(full, "omar"),
                 "omar_weights_mean": full["omar_weights_mean"]},
        "solo_f1": solos,
        "leave_one_out_full": loo,
        "loo_base_full_weighted": base_full_weighted,
        "new_weight_in_full": new_weight_in_full,
        "new_weight_in_diverse": new_weight_in_diverse,
        "old_best": old_best, "diverse_best": diverse_best,
        "delta_diverse_best_minus_old_best": delta_best,
        "verdict": verdict,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.task}_diverse_ensemble.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- console table ----
    print("=" * 74)
    print(f"DIVERSE ENSEMBLE  task={args.task}  k={args.k}  seed={args.seed}  "
          f"meta={args.meta}")
    print(f"decode: lam={lam0} bias={bias0}"
          + ("  (per-pool refit)" if args.refit_decode else "  (frozen, on record)"))
    print(f"new encoders present: {new_present or 'NONE'}   absent: {new_absent}")
    print("-" * 74)
    print(f"{'POOL':10s} {'members':>7s}  {'uniform':>8s} {'weighted':>9s} {'meta':>8s}")
    print(f"{'OLD':10s} {old['m']:7d}  {f1_of(old,'uniform'):8.2f} "
          f"{f1_of(old,'omar'):9.2f} {'—':>8s}")
    print(f"{'DIVERSE':10s} {diverse['m']:7d}  {f1_of(diverse,'uniform'):8.2f} "
          f"{f1_of(diverse,'omar'):9.2f} {f1_of(diverse,'meta'):8.2f}")
    print(f"{'FULL':10s} {full['m']:7d}  {f1_of(full,'uniform'):8.2f} "
          f"{f1_of(full,'omar'):9.2f} {f1_of(full,'meta'):8.2f}")
    print("-" * 74)
    print("SOLO F1 (single encoder through Omar DP):")
    for mm in all_members:
        tag = " (NEW)" if mm in new_present else ""
        print(f"    {mm:16s} {solos[mm]:6.2f}{tag}")
    print("-" * 74)
    print("LEAVE-ONE-OUT on FULL pool (weighted; drop_delta>0 => member HELPS):")
    for mm in loo_pool:
        tag = " (NEW)" if mm in new_present else ""
        print(f"    drop {mm:16s} weighted={loo[mm]['weighted_without']:6.2f}  "
              f"delta={loo[mm]['drop_delta']:+.3f}{tag}")
    print("-" * 74)
    print("Greedy weight given to NEW encoders:")
    for mm in new_present:
        print(f"    {mm:16s}  in DIVERSE={new_weight_in_diverse[mm]:.3f}  "
              f"in FULL={new_weight_in_full[mm]:.3f}")
    print("-" * 74)
    print(f"old_best      = {old_best:.2f}")
    print(f"diverse_best  = {diverse_best:.2f}")
    print(f"delta         = {delta_best:+.2f}")
    print(f"VERDICT (diverse pool): {verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
