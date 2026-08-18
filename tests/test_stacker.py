"""Unit tests for src/stacker.py — corpus-independent (pass with data/ hidden).

The STACKING META-COMBINER. These tests exercise the pure, CPU-only pieces that
carry the design trap (OOF discipline) and the decode/ablation conventions:
  * fold maker: deterministic, length-stratified, k folds all present, no doc
    appears in two of its own {train, inner_dev, held_out} splits;
  * inner-dev pick: deterministic, disjoint from nothing it must not be (it is
    a subset of the fold-train ids only);
  * OOF-leak tripwire: FIRES on a memorized (full-fit) voter column, PASSES on
    honest OOF-style columns — this is the guard the whole build rests on;
  * combiner: standardized LR beats the best single voter AND the unweighted
    mean on planted complementary errors, at own-best AND @0.5;
  * ablation discipline: dropping the (raw+derived) voter signal craters F1,
    dropping a pure-noise column moves it ~0;
  * decode helpers: GRID matches score_cut.py, preds_at forces PARAGRAPH->0,
    rank AUC agrees with a brute-force Mann-Whitney on a tiny case.
No file in data/ is read; everything is synthetic.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from stacker import (  # noqa: E402
    GRID,
    OOF_AUC_TRIPWIRE,
    combiner_probs,
    feat_derived,
    feat_position,
    fit_lr,
    gold_arrays,
    inner_dev_pick,
    length_stratified_folds,
    oof_leak_tripwire,
    preds_at,
    rank_auc_np,
    score_rows,
    synth_docs,
    synth_features,
)


# --------------------------------------------------------------------------- #
# fold maker
# --------------------------------------------------------------------------- #
def _toy_ids_lens(n=40, seed=1):
    rng = np.random.default_rng(seed)
    lens = [int(x) for x in rng.integers(30, 400, n)]
    ids = [f"doc_{i:03d}" for i in range(n)]
    return ids, lens


def test_folds_deterministic_and_complete():
    ids, lens = _toy_ids_lens()
    f1 = length_stratified_folds(ids, lens, 5, 42)
    f2 = length_stratified_folds(ids, lens, 5, 42)
    assert f1 == f2                                   # deterministic
    assert sorted(set(f1)) == [0, 1, 2, 3, 4]         # all k present
    # every doc assigned exactly one fold
    assert len(f1) == len(ids) and all(v is not None for v in f1)


def test_folds_length_stratified():
    """Each fold must see a similar length spectrum: per-fold mean length
    spread should be small relative to the global spread (blocks of k share
    one length neighbourhood)."""
    ids, lens = _toy_ids_lens(n=100, seed=7)
    fold_of = length_stratified_folds(ids, lens, 5, 42)
    lens = np.array(lens)
    means = [lens[[i for i, f in enumerate(fold_of) if f == k]].mean()
             for k in range(5)]
    # per-fold means cluster far tighter than the raw doc spread
    assert (max(means) - min(means)) < 0.5 * (lens.max() - lens.min())


def test_folds_seed_changes_assignment():
    ids, lens = _toy_ids_lens(n=60, seed=3)
    a = length_stratified_folds(ids, lens, 5, 42)
    b = length_stratified_folds(ids, lens, 5, 43)
    assert a != b                                     # seed actually bites


def test_inner_dev_is_subset_and_deterministic():
    ids, lens = _toy_ids_lens(n=80, seed=5)
    p1 = inner_dev_pick(ids, lens, 0.15, 123)
    p2 = inner_dev_pick(ids, lens, 0.15, 123)
    assert p1 == p2                                   # deterministic
    assert p1.issubset(set(ids))
    assert 0 < len(p1) < len(ids)                     # non-trivial subset


# --------------------------------------------------------------------------- #
# OOF-leak tripwire — the guard the whole build rests on
# --------------------------------------------------------------------------- #
def test_tripwire_fires_on_memorized_column():
    rng = np.random.default_rng(0)
    docs = synth_docs(rng, 30)
    X, names, fams, y, _ = synth_features(rng, docs)
    Xleak = X.copy()
    j0 = fams.index("voters")
    Xleak[:, j0] = np.clip(y + rng.normal(0, 0.005, len(y)), 0, 1)  # ~ gold
    with pytest.raises(AssertionError):
        oof_leak_tripwire(Xleak, names, fams, y)


def test_tripwire_passes_on_honest_oof_columns():
    rng = np.random.default_rng(1)
    docs = synth_docs(rng, 40)
    X, names, fams, y, _ = synth_features(rng, docs)
    worst = oof_leak_tripwire(X, names, fams, y)
    assert worst <= OOF_AUC_TRIPWIRE


# --------------------------------------------------------------------------- #
# combiner beats the incumbents on planted complementary errors
# --------------------------------------------------------------------------- #
def test_combiner_beats_best_single_and_mean():
    rng = np.random.default_rng(0)
    docs = synth_docs(rng, 60)
    half = len(docs) // 2
    d_tr, d_ev = docs[:half], docs[half:]
    Xtr, names, fams, ytr, _ = synth_features(rng, d_tr)
    Xev, _, _, yev, _ = synth_features(rng, d_ev)
    gold_map = {d["doc_id"]: list(d["labels"]) for d in d_ev}

    ids = [d["doc_id"] for d in d_tr]
    lens = [len(d["tokens"]) for d in d_tr]
    fold_of = length_stratified_folds(ids, lens, 5, 42)
    groups = np.concatenate([np.full(len(d["tokens"]), fold_of[i])
                             for i, d in enumerate(d_tr)])

    model = fit_lr(Xtr, ytr, groups, seed=42, quiet=True)
    p_c = combiner_probs(model, Xev)
    r_c = score_rows(d_ev, gold_map, p_c, "stacker", 42)

    nv = fams.count("voters")
    best_single = max(score_rows(d_ev, gold_map, Xev[:, j], f"v{j}", 42)["best_f1"]
                      for j in range(nv))
    r_mean = score_rows(d_ev, gold_map, Xev[:, :nv].mean(axis=1), "mean", 42)
    assert r_c["best_f1"] > best_single
    assert r_c["best_f1"] > r_mean["best_f1"]


def test_ablation_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    docs = synth_docs(rng, 60)
    half = len(docs) // 2
    d_tr, d_ev = docs[:half], docs[half:]
    Xtr, names, fams, ytr, _ = synth_features(rng, d_tr)
    Xev, _, _, yev, _ = synth_features(rng, d_ev)
    gold_map = {d["doc_id"]: list(d["labels"]) for d in d_ev}
    ids = [d["doc_id"] for d in d_tr]
    lens = [len(d["tokens"]) for d in d_tr]
    fold_of = length_stratified_folds(ids, lens, 5, 42)
    groups = np.concatenate([np.full(len(d["tokens"]), fold_of[i])
                             for i, d in enumerate(d_tr)])

    full = fit_lr(Xtr, ytr, groups, seed=42, quiet=True)
    r_full = score_rows(d_ev, gold_map, combiner_probs(full, Xev), "full", 42)

    def ablate(drop):
        keep = [j for j, f in enumerate(fams) if f not in drop]
        m = fit_lr(Xtr[:, keep], ytr, groups, seed=42, quiet=True)
        return score_rows(d_ev, gold_map, combiner_probs(m, Xev[:, keep]),
                          "abl", 42)["best_f1"]

    d_voter = ablate({"voters", "voter_derived"}) - r_full["best_f1"]
    d_noise = ablate({"judge"}) - r_full["best_f1"]     # 'judge' col is pure noise here
    assert d_voter < -2.0                                # voter signal carries
    assert abs(d_noise) < 0.6                            # useless column ~ 0


# --------------------------------------------------------------------------- #
# decode / metric helpers match the score_cut.py conventions
# --------------------------------------------------------------------------- #
def test_grid_matches_score_cut_convention():
    assert GRID == [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def test_preds_at_forces_paragraph_zero():
    from data import PARAGRAPH_TOKEN
    docs = [{"doc_id": "d0",
             "tokens": ["a", PARAGRAPH_TOKEN, "b", "c"],
             "labels": [0, 1, 0, 1]}]
    p = np.array([0.9, 0.99, 0.1, 0.8])                 # paragraph pos is high
    out = preds_at(docs, p, 0.5)
    assert out["d0"] == [1, 0, 0, 1]                    # paragraph forced to 0


def test_rank_auc_matches_bruteforce():
    rng = np.random.default_rng(2)
    s = rng.random(200)
    y = (rng.random(200) < 0.3)
    pos = s[y]
    neg = s[~y]
    # brute-force Mann-Whitney with tie halves
    wins = 0.0
    for a in pos:
        wins += (neg < a).sum() + 0.5 * (neg == a).sum()
    brute = wins / (len(pos) * len(neg))
    assert abs(rank_auc_np(s, y) - brute) < 1e-9


def test_rank_auc_degenerate_labels():
    s = np.array([0.1, 0.2, 0.3])
    assert np.isnan(rank_auc_np(s, np.array([False, False, False])))
    assert np.isnan(rank_auc_np(s, np.array([True, True, True])))


# --------------------------------------------------------------------------- #
# feature builders are shape/name-stable
# --------------------------------------------------------------------------- #
def test_feat_derived_shapes_and_names():
    V = np.array([[0.1, 0.2, 0.9], [0.5, 0.5, 0.5]])
    mat, names, fam = feat_derived(V)
    assert mat.shape == (2, 3)
    assert names == ["v_mean", "v_std", "v_maxmin"]
    assert fam == "voter_derived"
    assert np.isclose(mat[0, 2], 0.8)                   # max-min of row 0


def test_feat_position_monotone_and_docfinal_zero():
    docs = [{"doc_id": "d", "tokens": list("abcde"),
             "labels": [0, 0, 0, 0, 1]}]
    mat, names, fam = feat_position(docs)
    assert names == ["dist_doc_end_log", "rel_pos", "log_doc_len"]
    assert mat.shape == (5, 3)
    # distance-to-doc-end is 0 at the final token, strictly larger earlier
    assert mat[-1, 0] == 0.0
    assert (np.diff(mat[:, 0]) < 0).all()               # strictly decreasing
    assert mat[0, 1] == 0.0 and np.isclose(mat[-1, 1], 1.0)  # rel pos 0..1


def test_gold_arrays_doc_of_indexing():
    docs = [{"doc_id": "a", "tokens": ["x", "y"], "labels": [0, 1]},
            {"doc_id": "b", "tokens": ["z"], "labels": [1]}]
    y, doc_of = gold_arrays(docs)
    assert y.tolist() == [0, 1, 1]
    assert doc_of.tolist() == [0, 0, 1]
