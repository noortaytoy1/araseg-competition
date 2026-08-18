"""Unit tests for src/merge_judge.py — corpus-independent (pass with data/ hidden)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from merge_judge import (  # noqa: E402
    N_SCALARS,
    agglomerative_cuts,
    build_features,
    curriculum_states,
    decorrelation,
    eraser_merge_probs,
    gold_cut_positions,
    pairs_from_cuts,
    pooled_prf,
    prefix_sum,
    sentence_spans,
)

LAB = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1])          # 3 sentences, n=9


def test_gold_cuts_exclude_doc_final():
    cuts = gold_cut_positions(LAB)
    assert cuts.tolist() == [2, 4]                    # position 8 is doc-final
    assert sentence_spans(LAB) == [(0, 2), (3, 4), (5, 8)]


def test_pairs_from_cuts_segments():
    prs = pairs_from_cuts([2, 4], 9)
    assert prs.tolist() == [[0, 2, 4], [3, 4, 8]]
    # all-cut: every adjacent word pair
    prs = pairs_from_cuts(np.arange(8), 9)
    assert len(prs) == 8
    assert (prs[:, 0] == prs[:, 1]).all()
    assert (prs[:, 2] == prs[:, 1] + 1).all()
    assert pairs_from_cuts([], 9).shape == (0, 3)


def test_pair_supervision_counts():
    """MERGE=1 iff no gold boundary after i; dense at the all-cut state."""
    prs = pairs_from_cuts(np.arange(len(LAB) - 1), len(LAB))
    y = 1 - LAB[prs[:, 1]]
    assert y.tolist() == [1, 1, 0, 1, 0, 1, 1, 1]     # 6 merge / 2 keep


def test_curriculum_states_refine_gold():
    rng = np.random.default_rng(7)
    gset = set(gold_cut_positions(LAB).tolist())
    names = []
    for name, cuts in curriculum_states(LAB, rng, n_random_states=3):
        names.append(name)
        assert gset <= set(cuts.tolist()), f"{name} lost a gold cut"
        assert all(0 <= c <= len(LAB) - 2 for c in cuts)
    assert names == ["allcut", "rand0", "rand1", "rand2", "gold", "half"]
    # gold state is exactly the gold cuts; allcut is every position
    states = dict(curriculum_states(LAB, np.random.default_rng(7), 3))
    assert set(states["gold"].tolist()) == gset
    assert states["allcut"].tolist() == list(range(len(LAB) - 1))
    # half adds at most one interior split per sentence of len >= 2
    extra = set(states["half"].tolist()) - gset
    assert 1 <= len(extra) <= 3


def test_build_features_matches_naive():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(9, 5)).astype(np.float32)
    P = prefix_sum(V)
    prs = pairs_from_cuts([2, 4], 9)
    X = build_features(V, P, prs)
    assert X.shape == (2, 4 * 5 + N_SCALARS)
    a0, c, b1 = prs[1]
    np.testing.assert_allclose(X[1, :5], V[a0:c + 1].mean(0), rtol=1e-5)
    np.testing.assert_allclose(X[1, 5:10], V[c + 1:b1 + 1].mean(0), rtol=1e-5)
    np.testing.assert_allclose(X[1, 10:15], V[c], rtol=1e-6)
    np.testing.assert_allclose(X[1, 15:20], V[c + 1], rtol=1e-6)
    lenA, lenB = c - a0 + 1, b1 - c
    np.testing.assert_allclose(X[1, 20], np.log1p(lenA), rtol=1e-5)
    np.testing.assert_allclose(X[1, 21], np.log1p(lenB), rtol=1e-5)
    assert X[1, 22] == float(lenA == 1) and X[1, 23] == float(lenB == 1)
    assert -1.001 <= X[1, 24] <= 1.001 and -1.001 <= X[1, 25] <= 1.001


def test_eraser_one_shot_bookkeeping():
    """Mock judge erases a cut iff the LEFT segment has >= 2 words."""
    V = np.ones((7, 4), dtype=np.float32)
    P = prefix_sum(V)
    pred = np.array([1, 1, 0, 1, 1, 0, 1], dtype=bool)

    def mock(X):
        return (np.expm1(X[:, -N_SCALARS]) >= 2).astype(np.float32)

    cuts, mp = eraser_merge_probs(V, P, pred, mock)
    assert cuts.tolist() == [0, 1, 3, 4]              # position 6 is doc-final
    assert (mp > 0.5).tolist() == [False, False, True, False]


def test_agglomerative_planted_segmentation():
    """Separable topic vectors + a cosine-rule judge recover the gold cuts."""
    rng = np.random.default_rng(3)
    lab = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1])
    topics = rng.normal(0, 1, (3, 8)) * 5
    tid = np.repeat([0, 1, 2], [4, 2, 3])
    V = (topics[tid] + rng.normal(0, 0.1, (9, 8))).astype(np.float32)

    def judge(X):                                     # merge iff pools align
        return (X[:, -2] > 0.5).astype(np.float32)    # cos(poolA, poolB)

    got = agglomerative_cuts(V, prefix_sum(V), judge, 0.5)
    assert got.tolist() == gold_cut_positions(lab).tolist()


def test_agglomerative_stops_at_tau():
    V = np.ones((5, 3), dtype=np.float32)

    def never(X):
        return np.full(len(X), 0.4, dtype=np.float32)

    assert agglomerative_cuts(V, prefix_sum(V), never, 0.5).tolist() == [0, 1, 2, 3]

    def always(X):
        return np.full(len(X), 0.9, dtype=np.float32)

    assert agglomerative_cuts(V, prefix_sum(V), always, 0.5).tolist() == []


def test_pooled_prf_and_decorrelation():
    gold = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
    a = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
    t = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    prf = pooled_prf(a, gold)
    assert (prf["TP"], prf["FP"], prf["FN"]) == (2, 1, 1)
    d = decorrelation(a, t, gold)
    assert d["n_fp_a"] == 1 and d["n_fp_tagger"] == 2 and d["n_fp_shared"] == 1
    assert abs(d["fp_jaccard"] - 0.5) < 1e-9
    assert -1.0 <= d["phi_all_errors"] <= 1.0
