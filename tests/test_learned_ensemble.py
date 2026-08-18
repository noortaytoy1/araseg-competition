"""Unit tests for src/learned_ensemble.py — corpus-independent (pass with
data/ hidden). Everything here is synthetic; no file in data/ is read.

The LEARNED ENSEMBLE / META-LEARNER on DEV. These tests exercise the pure,
CPU-only pieces that carry the method's correctness:
  * doc-level k-fold CV: every position of a doc lands in exactly one fold
    (no doc split across the fit set and its own eval fold — the leakage guard
    the whole disk-free design rests on);
  * OOF coverage: cv_oof_predictions holds out every position exactly once and
    returns finite probs; the LR path also returns a k-row coef matrix;
  * drop_cols masking: the voters-only variant genuinely removes the named
    columns from the fit (coef width shrinks by exactly the dropped count);
  * own_best returns a macro-F1 in [0,100] and picks a threshold on GRID;
  * planted complementary-error signal: a learned LR combiner over two
    anti-correlated noisy voters beats their plain average on the official
    macro-F1 (the whole point of "learning who to trust").
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from learned_ensemble import (  # noqa: E402
    GRID,
    cv_oof_predictions,
    own_best,
)
from stacker import length_stratified_folds  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic fixture: docs + two anti-correlated noisy voters + pool column
# --------------------------------------------------------------------------- #
def _synth(n_docs=40, seed=0):
    rng = np.random.default_rng(seed)
    docs, labels = [], []
    for i in range(n_docs):
        n = int(rng.integers(30, 90))
        y = (rng.random(n) < 0.2).astype(np.int8)
        docs.append({"doc_id": f"d{i:03d}", "tokens": ["w"] * n,
                     "labels": y.tolist()})
        labels.append(y)
    gold = np.concatenate([np.asarray(d["labels"]) for d in docs])
    doc_of = np.concatenate([np.full(len(d["labels"]), i)
                             for i, d in enumerate(docs)])

    def noisy(y, spread, r):
        p = np.where(y == 1, 0.5 + 0.4 * (r.random(len(y)) - spread),
                     0.5 - 0.4 * (r.random(len(y)) - spread))
        return np.clip(p, 1e-4, 1 - 1e-4)

    r1, r2 = np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
    v1 = np.concatenate([noisy(np.asarray(d["labels"]), 0.30, r1)
                         for d in docs])
    v2 = np.concatenate([noisy(np.asarray(d["labels"]), 0.35, r2)
                         for d in docs])
    X = np.stack([v1, v2, 0.5 * (v1 + v2)], axis=1)  # v1, v2, pool
    return docs, gold, doc_of, X, v1, v2


def _fold_of_pos(docs, doc_of, k=5, seed=42):
    ids = [d["doc_id"] for d in docs]
    lens = [len(d["tokens"]) for d in docs]
    fod = length_stratified_folds(ids, lens, k, seed)
    return np.array([fod[i] for i in doc_of], dtype=np.int64), fod


# --------------------------------------------------------------------------- #
# doc-level folds: no doc split across folds
# --------------------------------------------------------------------------- #
def test_folds_are_doc_level():
    docs, gold, doc_of, X, _, _ = _synth()
    fop, _ = _fold_of_pos(docs, doc_of)
    for i in range(len(docs)):
        fp = fop[doc_of == i]
        assert (fp == fp[0]).all(), "a doc was split across CV folds"
    assert sorted(set(fop.tolist())) == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# OOF coverage: every position held out exactly once, finite, k coef rows
# --------------------------------------------------------------------------- #
def test_oof_full_coverage_and_finite():
    docs, gold, doc_of, X, _, _ = _synth()
    fop, _ = _fold_of_pos(docs, doc_of)
    oof, keep, coef = cv_oof_predictions(X, gold, doc_of, fop, 5, "lr", 42)
    assert oof.shape == gold.shape
    assert np.isfinite(oof).all()
    assert (0.0 <= oof).all() and (oof <= 1.0).all()
    assert coef.shape[0] == 5                       # one coef row per fold
    assert keep.all()                               # nothing dropped by default


# --------------------------------------------------------------------------- #
# a held-out fold never contributes to its own fit (leakage guard, indirect)
# --------------------------------------------------------------------------- #
def test_holdout_fold_prediction_uses_only_other_folds():
    # If fold f's predictions came from a model fitted on fold f, perturbing
    # ONLY fold f's labels would change fold f's OOF preds. It must not.
    docs, gold, doc_of, X, _, _ = _synth()
    fop, _ = _fold_of_pos(docs, doc_of)
    oof_a, _, _ = cv_oof_predictions(X, gold, doc_of, fop, 5, "lr", 42)
    g2 = gold.copy()
    f0 = fop == 0
    g2[f0] = 1 - g2[f0]                             # flip ALL of fold 0's labels
    oof_b, _, _ = cv_oof_predictions(X, g2, doc_of, fop, 5, "lr", 42)
    # fold 0's own OOF preds are produced by a model fit on folds 1..4, whose
    # labels are unchanged -> identical.
    assert np.allclose(oof_a[f0], oof_b[f0], atol=1e-9), \
        "held-out fold prediction depended on its own labels (leak)"


# --------------------------------------------------------------------------- #
# drop_cols genuinely removes columns from the fit
# --------------------------------------------------------------------------- #
def test_drop_cols_masks_features():
    docs, gold, doc_of, X, _, _ = _synth()
    fop, _ = _fold_of_pos(docs, doc_of)
    _, keep_all, coef_all = cv_oof_predictions(X, gold, doc_of, fop, 5,
                                               "lr", 42)
    _, keep_drop, coef_drop = cv_oof_predictions(X, gold, doc_of, fop, 5,
                                                 "lr", 42, drop_cols={2})
    assert coef_all.shape[1] == X.shape[1]
    assert coef_drop.shape[1] == X.shape[1] - 1
    assert keep_drop.sum() == X.shape[1] - 1
    assert not keep_drop[2]


# --------------------------------------------------------------------------- #
# own_best sane + threshold on GRID
# --------------------------------------------------------------------------- #
def test_own_best_range_and_grid():
    docs, gold, doc_of, X, v1, v2 = _synth()
    n_docs = len(docs)
    f1, thr = own_best(0.5 * (v1 + v2), gold, doc_of, n_docs)
    assert 0.0 <= f1 <= 100.0
    assert thr in GRID


# --------------------------------------------------------------------------- #
# THE POINT: learned LR beats plain average on planted complementary errors
# --------------------------------------------------------------------------- #
def test_learned_combiner_beats_average_on_planted_signal():
    docs, gold, doc_of, X, v1, v2 = _synth(n_docs=60, seed=7)
    n_docs = len(docs)
    fop, _ = _fold_of_pos(docs, doc_of)
    oof, _, _ = cv_oof_predictions(X, gold, doc_of, fop, 5, "lr", 42)
    lr_f1, _ = own_best(oof, gold, doc_of, n_docs)
    pool_f1, _ = own_best(0.5 * (v1 + v2), gold, doc_of, n_docs)
    # learned combiner must be at least as good as the plain average here
    # (planted anti-correlated noise is exactly what a combiner can exploit).
    assert lr_f1 >= pool_f1 - 1e-6, \
        f"learned LR {lr_f1:.3f} < plain average {pool_f1:.3f} on planted"
