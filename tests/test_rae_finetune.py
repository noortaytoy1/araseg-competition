"""Unit tests for src/rae_finetune.py — corpus-independent (pass with data/
hidden). The decisive test plants an INVERTED composability prior in a small
RAE and proves supervised fine-tuning flips it (the pipeline cell's
mechanism), before any real-data number is trusted."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from merge_judge import (  # noqa: E402
    N_SCALARS,
    _synth_doc,
    agglomerative_cuts,
    build_features,
    eraser_merge_probs,
    gold_cut_positions,
    pairs_from_cuts,
    prefix_sum,
)
from rae_finetune import (  # noqa: E402
    ARMS,
    FTJudge,
    allcut_slice_auc,
    finetune_toy,
    fit_scaler,
    judge_with_E,
    load_judge,
    make_judge,
    make_judge_head,
    plant_inverted_rae,
    toy_pair_table,
)
from rae_gate import make_rae, normalize_rows  # noqa: E402

H = 12


def _docs(seeds, **kw):
    kw.setdefault("n_sent", 5)
    kw.setdefault("max_len", 5)
    kw.setdefault("h", H)
    kw.setdefault("sep", 4.0)
    return [_synth_doc(np.random.default_rng(s), **kw) for s in seeds]


def test_judge_inputs_match_naive_composition():
    import torch
    rae = make_rae(H, seed=0)
    model = make_judge(rae, H, seed=0)
    _w, lab, V = _docs([7])[0]
    prs = pairs_from_cuts(gold_cut_positions(lab), len(lab))
    X = build_features(V, prefix_sum(V), prs)
    with torch.no_grad():
        J = model.judge_inputs(torch.from_numpy(X)).numpy()
    assert J.shape == (len(prs), 2 * H + 2 + N_SCALARS)
    a0, c, b1 = prs[0]
    with torch.no_grad():
        p1, e1 = rae.compose(
            torch.from_numpy(normalize_rows(V[a0:c + 1].mean(0)[None])),
            torch.from_numpy(normalize_rows(V[c + 1:b1 + 1].mean(0)[None])))
        p2, e2 = rae.compose(torch.from_numpy(normalize_rows(V[c][None])),
                             torch.from_numpy(normalize_rows(V[c + 1][None])))
    np.testing.assert_allclose(J[0, :H], p1[0].numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(J[0, H:2 * H], p2[0].numpy(), rtol=1e-4,
                               atol=1e-5)
    np.testing.assert_allclose(J[0, 2 * H], float(e1[0]), rtol=1e-4)
    np.testing.assert_allclose(J[0, 2 * H + 1], float(e2[0]), rtol=1e-4)
    np.testing.assert_allclose(J[0, 2 * H + 2:], X[0, 4 * H:], rtol=1e-5)


def test_allcut_state_recovers_leaf_gate():
    """Single-word segments: p_seg == p_word and E_seg == E_word."""
    import torch
    model = make_judge(make_rae(H, seed=1), H, seed=1)
    _w, lab, V = _docs([3])[0]
    prs = pairs_from_cuts(np.arange(len(lab) - 1), len(lab))
    X = build_features(V, prefix_sum(V), prs)
    with torch.no_grad():
        J = model.judge_inputs(torch.from_numpy(X)).numpy()
    np.testing.assert_allclose(J[:, :H], J[:, H:2 * H], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(J[:, 2 * H], J[:, 2 * H + 1], rtol=1e-4,
                               atol=1e-5)


def test_ftjudge_probs_and_empty():
    model = make_judge(make_rae(H, seed=2), H, seed=2)
    _w, lab, V = _docs([4])[0]
    X = build_features(V, prefix_sum(V),
                       pairs_from_cuts(np.arange(len(lab) - 1), len(lab)))
    fit_scaler(model, [X])
    p = FTJudge(model)(X)
    assert p.shape == (len(X),) and (p >= 0).all() and (p <= 1).all()
    empty = FTJudge(model)(np.zeros((0, 4 * H + N_SCALARS), dtype=np.float32))
    assert empty.shape == (0,)
    p2, e2 = judge_with_E(model, X)
    np.testing.assert_allclose(p, p2, rtol=1e-5, atol=1e-6)
    assert (e2 > 0).all()


def test_frozen_arm_keeps_composer_bit_identical():
    import torch
    docs = _docs(range(6))
    rae = make_rae(H, seed=3)
    w0 = {k: v.clone() for k, v in rae.state_dict().items()}
    model = finetune_toy(rae, docs, "frozenW", epochs=1, seed=3)
    for k, v in model.rae.state_dict().items():
        assert torch.equal(w0[k], v), f"frozenW moved composer weight {k}"
    h0 = make_judge_head(H, seed=3).state_dict()
    assert any(not torch.equal(h0[k], v)
               for k, v in model.head.state_dict().items()), \
        "head did not train"


def test_full_arm_moves_composer():
    import torch
    docs = _docs(range(6))
    rae = make_rae(H, seed=3)
    w0 = {k: v.clone() for k, v in rae.state_dict().items()}
    model = finetune_toy(rae, docs, "full", epochs=1, seed=3)
    assert any(not torch.equal(w0[k], v)
               for k, v in model.rae.state_dict().items()), \
        "full arm did not move the composer"


def test_planted_inverted_prior_is_flipped_by_supervision():
    """THE commissioned proof: an RAE pretrained to compose well ACROSS
    boundaries (inverted prior, slice AUC < 0.4) must be un-inverted by the
    supervised fine-tune (judge AUC >= 0.85 on held-out docs, both arms)."""
    train_docs = _docs(range(100, 118), n_sent=6, max_len=6)
    held_docs = _docs(range(900, 906), n_sent=6, max_len=6)
    rae_inv = plant_inverted_rae(H, train_docs, steps=250, seed=0)
    auc_prior = allcut_slice_auc(rae_inv, held_docs, use_judge=False)
    assert auc_prior < 0.40, f"prior not inverted ({auc_prior:.3f})"
    for arm in ARMS:
        rae_c = make_rae(H, seed=0)
        rae_c.load_state_dict(rae_inv.state_dict())
        model = finetune_toy(rae_c, train_docs, arm, epochs=3, seed=0)
        auc_ft = allcut_slice_auc(model, held_docs, use_judge=True)
        assert auc_ft >= 0.85, \
            f"[{arm}] did not un-invert the planted prior ({auc_ft:.3f})"


def test_wrapper_drives_merge_judge_policies():
    """The fine-tuned judge, through FTJudge, must drive merge_judge's
    eraser and agglomerative decode unchanged and recover planted cuts."""
    train_docs = _docs(range(200, 224), n_sent=6, max_len=6)
    model = finetune_toy(make_rae(H, seed=5), train_docs, "full", epochs=4,
                         seed=5)
    judge = FTJudge(model)
    _w, lab, V = _docs([998], n_sent=6, max_len=6)[0]
    want = set(gold_cut_positions(lab).tolist())
    got = agglomerative_cuts(V, prefix_sum(V), judge, 0.5)
    assert set(got.tolist()) == want
    pred = np.ones(len(lab), dtype=bool)
    cuts, mps = eraser_merge_probs(V, prefix_sum(V), pred, judge)
    assert cuts.tolist() == list(range(len(lab) - 1))
    kept = set(cuts.tolist()) - set(cuts[mps > 0.5].tolist())
    assert kept == want


def test_checkpoint_roundtrip(tmp_path):
    import torch
    docs = _docs(range(300, 306))
    model = finetune_toy(make_rae(H, seed=6), docs, "frozenW", epochs=1,
                         seed=6)
    _w, lab, V = docs[0]
    X = build_features(V, prefix_sum(V),
                       pairs_from_cuts(np.arange(len(lab) - 1), len(lab)))
    p_before = FTJudge(model)(X)
    pth = os.path.join(str(tmp_path), "j.pt")
    torch.save({"rae_state": model.rae.state_dict(),
                "head_state": model.head.state_dict(),
                "mu": model.mu.numpy(), "sd": model.sd.numpy(), "h": H,
                "layer": "final", "arm": "frozenW", "history": [],
                "config": {}}, pth)
    m2, ck = load_judge(pth)
    assert ck["arm"] == "frozenW"
    p_after = FTJudge(m2)(X)
    np.testing.assert_allclose(p_before, p_after, rtol=1e-5, atol=1e-6)


def test_toy_pair_table_labels_exact():
    """Curriculum labels: MERGE=1 iff the cut is not a gold boundary."""
    docs = _docs([11])
    X, y = toy_pair_table(docs, np.random.default_rng(0))
    assert X.shape[1] == 4 * H + N_SCALARS
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}
    _w, lab, V = docs[0]
    n = len(lab)
    prs = pairs_from_cuts(np.arange(n - 1), n)
    want = (1 - lab[prs[:, 1]]).astype(np.float32)
    np.testing.assert_array_equal(X[:n - 1, 4 * H].round(4),
                                  np.log1p(prs[:, 1] - prs[:, 0] + 1)
                                  .astype(np.float32).round(4))
    np.testing.assert_array_equal(y[:n - 1], want)
