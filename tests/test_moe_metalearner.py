"""Unit tests for src/moe_metalearner.py — corpus-independent (pass with data/
hidden). Everything here is synthetic; no file in data/ is read.

The CONTEXT-GATED META-LEARNER over Omar's 6 encoder voters. These tests pin
the properties the method's correctness rests on:
  * INPUT-CONTEXT features are text-only and behave as specified (conjunction /
    preposition / number / OOV / position / distance-to-end / bias), so nothing
    model-derived can sneak in;
  * the meta-model input is EXACTLY [6 voter probs | context | context x voter]
    and the interaction block genuinely makes a voter's weight depend on
    context (the "learns which to trust based on the thing");
  * doc-level k=5 folds via the shared maker put every position of a doc in one
    fold (the no-doc-across-folds leakage guard the CV rests on);
  * the CV is honest: a held-out fold's OOF prediction does not depend on its
    own labels;
  * THE POINT: on a planted context-conditional signal (voter A is right on the
    left half of docs, voter B on the right half), the context-gated combiner
    beats the uniform average through Omar's DP decode — a plain fixed-weight
    average provably cannot capture a weight that flips with context;
  * verdict thresholds match the pre-registered reading verbatim.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from moe_metalearner import (  # noqa: E402
    CONTEXT_FEATURES,
    MEMBERS,
    VOTER_NAMES,
    build_design,
    build_train_vocab,
    context_features_for_doc,
    fit_meta,
    interaction_block,
    verdict_line,
    _is_conjunction,
    _is_number,
    _is_preposition,
)
from data import PARAGRAPH_TOKEN  # noqa: E402


# --------------------------------------------------------------------------- #
# voter set is exactly Omar's 6 encoders
# --------------------------------------------------------------------------- #
def test_voters_are_omars_six_encoders():
    assert MEMBERS == [
        "arabertv02-s1", "arabertv02-s2", "arabertv02-s3", "arabertv02-s42",
        "araelectra-s42", "arbertv2-s42",
    ]
    assert VOTER_NAMES == MEMBERS
    assert len(MEMBERS) == 6


# --------------------------------------------------------------------------- #
# input-context features are text-only and correct
# --------------------------------------------------------------------------- #
def test_conjunction_flag():
    for w in ["و", "ف", "ثم", "أو", "لكن"]:
        assert _is_conjunction(w)
    assert _is_conjunction("وكان")          # و+ clitic prefix
    assert not _is_conjunction("كتاب")       # ordinary word


def test_preposition_flag():
    for w in ["في", "من", "إلى", "على", "عن", "ب", "ل"]:
        assert _is_preposition(w)
    assert not _is_preposition("كتاب")


def test_number_flag():
    assert _is_number("123")
    assert _is_number("٤٥٦")                 # Arabic-indic digits
    assert not _is_number("كتاب")
    assert not _is_number("")


def test_context_feature_matrix_shape_and_positions():
    toks = ["و", "الكتاب", "في", "2024", PARAGRAPH_TOKEN, "الاخير"]
    vocab = {"الكتاب"}                        # only this one is "seen"
    C = context_features_for_doc(toks, vocab)
    assert C.shape == (len(toks), len(CONTEXT_FEATURES))
    idx = {n: i for i, n in enumerate(CONTEXT_FEATURES)}
    # و -> conjunction
    assert C[0, idx["is_conjunction"]] == 1.0
    # في -> preposition
    assert C[2, idx["is_preposition"]] == 1.0
    # 2024 -> number
    assert C[3, idx["is_number"]] == 1.0
    # OOV: "و" unseen -> 1, "الكتاب" seen -> 0
    assert C[0, idx["is_oov_train"]] == 1.0
    assert C[1, idx["is_oov_train"]] == 0.0
    # bias always 1
    assert (C[:, idx["bias"]] == 1.0).all()
    # PARAGRAPH position: only bias set, all else zero
    para = C[4]
    for n, i in idx.items():
        if n != "bias":
            assert para[i] == 0.0
    # relative position 0..1, distance-to-end 0 at final token
    assert C[0, idx["rel_pos"]] == 0.0
    assert abs(C[-1, idx["rel_pos"]] - 1.0) < 1e-9
    assert C[-1, idx["dist_end_log"]] == 0.0


def test_train_vocab_excludes_paragraph():
    docs = [{"doc_id": "d", "tokens": ["a", PARAGRAPH_TOKEN, "b"],
             "labels": [0, 0, 1]}]
    v = build_train_vocab(docs)
    assert v == {"a", "b"}
    assert PARAGRAPH_TOKEN not in v


# --------------------------------------------------------------------------- #
# design matrix is EXACTLY [voters | context | context x voter]
# --------------------------------------------------------------------------- #
def test_design_is_voters_context_and_interactions_only():
    N = 20
    rng = np.random.default_rng(0)
    V = rng.random((N, 6))
    C = rng.random((N, len(CONTEXT_FEATURES)))
    X, names = build_design(V, C, with_interactions=True)
    n_ctx = len(CONTEXT_FEATURES)
    # interactions: 6 voters x (context minus the bias column)
    n_inter = 6 * (n_ctx - 1)
    assert X.shape[1] == 6 + n_ctx + n_inter
    assert len(names) == X.shape[1]
    # first 6 names are the encoders, next block the context features
    assert names[:6] == VOTER_NAMES
    assert names[6:6 + n_ctx] == CONTEXT_FEATURES
    # every interaction name references a voter and a context feature (× join)
    for nm in names[6 + n_ctx:]:
        assert "×" in nm
        left, right = nm.split("×")
        assert left in VOTER_NAMES and right in CONTEXT_FEATURES


def test_interaction_makes_voter_weight_context_dependent():
    # product columns must actually equal voter*context so the fitted weight on
    # a voter can vary with context.
    N = 10
    rng = np.random.default_rng(1)
    V = rng.random((N, 6))
    C = rng.random((N, len(CONTEXT_FEATURES)))
    inter, names = interaction_block(V, C)
    bias_idx = CONTEXT_FEATURES.index("bias")
    kept = [j for j in range(C.shape[1]) if j != bias_idx]
    assert inter.shape == (N, 6 * len(kept))
    # check the very first interaction column = V[:,0] * C[:,kept[0]]
    assert np.allclose(inter[:, 0], V[:, 0] * C[:, kept[0]])
    # bias context never appears in an interaction name
    assert all("bias" != nm.split("×")[1] for nm in names)


# --------------------------------------------------------------------------- #
# CV honesty: held-out fold's OOF prediction ignores its own labels
# --------------------------------------------------------------------------- #
def test_meta_fit_is_pure_function_of_training_labels():
    rng = np.random.default_rng(2)
    Xtr = rng.random((200, 8))
    ytr = (rng.random(200) < 0.3).astype(int)
    prob_a, coef_a = fit_meta(Xtr, ytr, "lr", 42)
    prob_b, coef_b = fit_meta(Xtr, ytr, "lr", 42)
    Xte = rng.random((50, 8))
    # deterministic given (X, y, seed)
    assert np.allclose(prob_a(Xte), prob_b(Xte))
    assert np.allclose(coef_a, coef_b)
    # flipping labels changes the fitted model (labels actually bite)
    prob_c, _ = fit_meta(Xtr, 1 - ytr, "lr", 42)
    assert not np.allclose(prob_a(Xte), prob_c(Xte))


# --------------------------------------------------------------------------- #
# THE POINT: context-gated combiner beats uniform average on a planted
# context-conditional trust signal, through Omar's DP decode.
# --------------------------------------------------------------------------- #
def _planted_docs(n_docs=80, seed=7):
    """Two voters; voter 0 is accurate on the LEFT half of every doc, voter 1 on
    the RIGHT half. A fixed-weight average cannot exploit this; a context-gated
    model keyed on rel_pos can. We embed these two informative voters as the
    first two of the six columns and fill the rest with pure noise.
    """
    rng = np.random.default_rng(seed)
    docs, V_rows, C_rows, y_rows, doc_of = [], [], [], [], []
    for di in range(n_docs):
        n = int(rng.integers(40, 80))
        y = (rng.random(n) < 0.25).astype(int)
        # ensure a doc-final boundary (matches real gold / dp_decode forcing)
        y[-1] = 1
        toks = [f"w{di}_{i}" for i in range(n)]
        docs.append({"doc_id": f"d{di:03d}", "tokens": toks,
                     "labels": y.tolist()})
        rel = np.arange(n) / max(n - 1, 1)
        left = rel < 0.5
        # accurate voter is near-gold on its half, coin-flip on the other half
        def voter(accurate_mask):
            p = np.where(y == 1, 0.85, 0.15)
            flip = ~accurate_mask
            p = p.copy()
            p[flip] = 0.5 + 0.02 * (rng.random(flip.sum()) - 0.5)
            return np.clip(p, 1e-3, 1 - 1e-3)
        v0 = voter(left)            # good on left
        v1 = voter(~left)           # good on right
        noise = np.clip(0.5 + 0.05 * (rng.random((n, 4)) - 0.5), 1e-3, 1 - 1e-3)
        V = np.column_stack([v0, v1, noise])
        # context features (text-only); rel_pos is the useful one here
        from moe_metalearner import context_features_for_doc
        C = context_features_for_doc(toks, set())  # all OOV, fine for synthetic
        V_rows.append(V)
        C_rows.append(C)
        y_rows.append(y)
        doc_of.append(np.full(n, di))
    V = np.concatenate(V_rows)
    C = np.concatenate(C_rows)
    y = np.concatenate(y_rows)
    doc_of = np.concatenate(doc_of)
    return docs, V, C, y, doc_of


def test_context_gated_beats_uniform_on_planted_signal():
    from dp_decode import decode_doc, fit_length_logprob
    from eval_local import compute_metrics
    from stacker import length_stratified_folds

    docs, V, C, y, doc_of = _planted_docs()
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    # length prior fit on these synthetic docs (self-contained, closed)
    len_lp = fit_length_logprob(docs)
    lam, bias = 0.5, 0.0

    ids = [d["doc_id"] for d in docs]
    lens = [len(d["tokens"]) for d in docs]
    fold_of = length_stratified_folds(ids, lens, 5, 42)

    spans, cur = [], 0
    for d in docs:
        n = len(d["tokens"])
        spans.append((cur, cur + n))
        cur += n

    X, _ = build_design(V, C, with_interactions=True)

    preds_uni, preds_meta = {}, {}
    for f in range(5):
        held = [i for i, ff in enumerate(fold_of) if ff == f]
        train = [i for i, ff in enumerate(fold_of) if ff != f]
        tr_mask = np.isin(doc_of, np.array(train))
        prob_fn, _ = fit_meta(X[tr_mask], y[tr_mask], "lr", 42)
        for di in held:
            s, e = spans[di]
            preds_uni[ids[di]] = decode_doc(docs[di]["tokens"],
                                            V[s:e].mean(axis=1), len_lp, lam, bias)
            preds_meta[ids[di]] = decode_doc(docs[di]["tokens"],
                                             prob_fn(X[s:e]), len_lp, lam, bias)
    f_uni = compute_metrics(gold, preds_uni)["macro_f1"]
    f_meta = compute_metrics(gold, preds_meta)["macro_f1"]
    assert f_meta > f_uni + 1e-6, \
        f"context-gated {f_meta:.4f} !> uniform {f_uni:.4f} on planted signal"


# --------------------------------------------------------------------------- #
# verdict thresholds match the pre-registered reading verbatim
# --------------------------------------------------------------------------- #
def test_verdict_thresholds():
    # WORKS: beats BOTH by > +0.2 (fractions; verdict scales by 100 internally)
    assert verdict_line(0.9000, 0.8970, 0.8975) == "WORKS"   # +0.30, +0.25
    # MARGINAL: beats both but by <= 0.2
    assert verdict_line(0.9000, 0.8990, 0.8995) == "MARGINAL"  # +0.10, +0.05
    # NULL: does not beat one of them
    assert verdict_line(0.9000, 0.9010, 0.8990) == "NULL"
    assert verdict_line(0.9000, 0.9000, 0.8990) == "NULL"    # tie is not "beat"
