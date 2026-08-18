"""Unit tests for src/forest_train.py — corpus-independent (pass with data/
hidden) and network-free (no HF encoder download). The mechanism under test —
the composer + keep/erase head + easy-first supervised merge, and gradient flow
into the (stand-in) word vectors — is exercised on synthetic vectors; the real
encoder is covered by src/smoke_forest.py (which reads the offline HF cache)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from forest_train import (  # noqa: E402
    Forest,
    _adjacent_cosine_baseline_f1,
    _adjacent_pair_baseline_f1,
    _bracket_stream,
    _bracket_vectors,
    _forest_boundary_f1,
    _free_run_start_cuts,
    _make_bracket_docs,
    _restore_weights,
    _snapshot_weights,
    _token_local_baseline_f1,
    _train_forest_on_docs,
    forest_predict_window,
    forest_window_loss,
    forest_window_loss_agglo,
    head_logits,
    llrd_forest_param_groups,
    make_head,
    rdrop_symmetric_kl,
    self_test,
)
from rae_gate import make_rae  # noqa: E402


# --------------------------------------------------------------------------- #
# head shape: in_features = 3H + 1 ([parent ; c1 ; c2 ; recon_error])
# --------------------------------------------------------------------------- #
def test_head_in_features():
    for h in (8, 16, 64, 768):
        head = make_head(h, seed=0)
        assert head.in_features == 3 * h + 1
    # the head actually consumes exactly 3H+1 features
    import torch
    h = 16
    head = make_head(h, seed=0)
    comp = make_rae(h, seed=0)
    c1 = torch.randn(3, h); c2 = torch.randn(3, h)
    parent, e = comp.compose(c1, c2)
    out = head_logits(head, parent, c1, c2, e)
    assert out.shape == (3,)


# --------------------------------------------------------------------------- #
# gradient flows keep/erase -> composer -> word vectors (the encoder path)
# --------------------------------------------------------------------------- #
def test_grad_flows_to_word_vectors_and_composer():
    import torch
    h = 16
    comp = make_rae(h, seed=0)
    head = make_head(h, seed=0)
    forest = Forest(None, None, comp, head, h, "cpu")
    wv = torch.tensor(np.random.default_rng(1).normal(size=(7, h)),
                      dtype=torch.float32, requires_grad=True)
    labels = [0, 0, 1, 0, 0, 1, 1]
    loss, n_leaf, n_pos = forest_window_loss(
        forest, wv, labels, pos_weight=3.0, rng=np.random.default_rng(0))
    assert loss.requires_grad and loss.ndim == 0 and float(loss.detach()) > 0
    loss.backward()
    # word vectors = stand-in for encoder outputs; nonzero grad here == the e2e
    # path would push gradient into the encoder.
    assert wv.grad is not None and float(wv.grad.abs().sum()) > 0
    assert comp.enc.weight.grad is not None and \
        float(comp.enc.weight.grad.abs().sum()) > 0
    assert head[0].weight.grad is not None and \
        float(head[0].weight.grad.abs().sum()) > 0


def test_gold_keep_always_supervised_and_par_excluded():
    import torch
    h = 8
    forest = Forest(None, None, make_rae(h, 0), make_head(h, 0), h, "cpu")
    wv = torch.tensor(np.random.default_rng(2).normal(size=(7, h)),
                      dtype=torch.float32)
    # gold boundaries after w2 and w5 (w6 is doc-final: no junction)
    labels = [0, 0, 1, 0, 0, 1, 1]
    for s in range(8):
        _l, n_leaf, n_pos = forest_window_loss(
            forest, wv, labels, 3.0, rng=np.random.default_rng(s))
        assert n_pos == 2, (s, n_pos)                 # both gold KEEPs supervised
        assert n_leaf >= n_pos                        # curriculum may add ERASE cuts
    # a -100 ([PAR]) junction is never supervised and never counted positive
    labels_par = [0, -100, 1, 0, 1]
    for s in range(8):
        _l, n_leaf, n_pos = forest_window_loss(
            forest, wv[:5], labels_par, 3.0, rng=np.random.default_rng(s))
        assert n_pos == 1, (s, n_pos)                 # only the after-w2 gold cut


# --------------------------------------------------------------------------- #
# the agglomerative-from-singletons training primitive + the construction:
# grad flows keep/erase -> composer -> word vectors, and the left-context
# bigram rule holds on _bracket_stream / _bracket_vectors output.
# --------------------------------------------------------------------------- #
def test_agglo_loss_grad_and_construction():
    import torch
    # (a) the construction obeys the left-context (A,B) boundary rule exactly
    r = np.random.default_rng(1000)
    ids, lab, _ = _bracket_stream(r, n_sent=4)
    for j in range(len(lab) - 1):
        want = 1 if (j >= 1 and ids[j - 1] == 0 and ids[j] == 1) else 0
        assert lab[j] == want, (j, ids[j - 1:j + 1], lab[j])
    V = _bracket_vectors(ids, 12, r)
    assert V.shape == (len(ids), 12) and V.dtype == np.float32
    # (b) forest_window_loss_agglo returns a grad-carrying scalar; grad reaches
    #     the word vectors (the encoder path) AND the composer AND the head.
    h = 12
    comp = make_rae(h, seed=0); head = make_head(h, seed=0)
    forest = Forest(None, None, comp, head, h, "cpu")
    wv = torch.tensor(V[:9], dtype=torch.float32, requires_grad=True)
    loss, n_j, n_pos = forest_window_loss_agglo(
        forest, wv, lab[:9], pos_weight=3.0)
    assert loss is not None and loss.requires_grad and loss.ndim == 0
    loss.backward()
    assert wv.grad is not None and float(wv.grad.abs().sum()) > 0
    assert comp.enc.weight.grad is not None and \
        float(comp.enc.weight.grad.abs().sum()) > 0
    assert head[0].weight.grad is not None


# --------------------------------------------------------------------------- #
# predict readout shape / range / doc-final convention (both readouts)
# --------------------------------------------------------------------------- #
def test_predict_window_shape_and_range():
    import torch
    h = 16
    forest = Forest(None, None, make_rae(h, 0), make_head(h, 0), h, "cpu")
    forest.eval()                                     # inference: dropout off
    wv = torch.tensor(np.random.default_rng(4).normal(size=(9, h)),
                      dtype=torch.float32)
    # both readouts + the historical boolean alias must agree on shape/range/rule
    for readout in ("agglom", "leaf"):
        p = forest_predict_window(forest, wv, readout=readout)
        assert p.shape == (9,)
        assert np.all((p >= 0) & (p <= 1))
        assert p[-1] == 0.0                           # doc-final word has no junction
    # descend= alias: True->agglom, False->leaf (backward compatible)
    np.testing.assert_array_equal(
        forest_predict_window(forest, wv, descend=True),
        forest_predict_window(forest, wv, readout="agglom"))
    np.testing.assert_array_equal(
        forest_predict_window(forest, wv, descend=False),
        forest_predict_window(forest, wv, readout="leaf"))
    # n < 2 -> all zeros (default readout)
    assert forest_predict_window(forest, wv[:1]).tolist() == [0.0]


def test_predict_window_widths_track_composition():
    """return_widths exposes the composed-unit width each junction was read at.
    The leaf readout reads every junction on a 1+1 pair (width 2); the default
    agglom readout reads at least one junction on a WIDER composed unit (> 2) —
    the distinction that removes the train/decode state mismatch (BLOCKER-2)."""
    import torch
    h = 16
    forest = Forest(None, None, make_rae(h, 0), make_head(h, 0), h, "cpu")
    forest.eval()                                     # inference: dropout off
    wv = torch.tensor(np.random.default_rng(5).normal(size=(9, h)),
                      dtype=torch.float32)
    _, wl = forest_predict_window(forest, wv, readout="leaf", return_widths=True)
    assert set(int(x) for x in wl[:-1]) == {2}      # every junction on a raw pair
    assert wl[-1] == 0
    _, wa = forest_predict_window(forest, wv, readout="agglom", return_widths=True)
    assert wa.shape == (9,)
    assert wa[-1] == 0
    assert wa[:-1].max() > 2                          # composition consumed at decode


# --------------------------------------------------------------------------- #
# COMPOSITIONAL PLANTED-SIGNAL contract. The boundary is the LEFT-CONTEXT ordered
# bigram (A,B): junction j is a boundary iff (symbol[j-1], symbol[j]) == (A, B).
# This is provably invisible to EVERY pair-local view of the junction (its own
# token, its own adjacent pair, its own adjacent cosine — all < 0.6 F1), because
# the trigger's LEFT arm (symbol[j-1]) is not in the junction's own pair. Only
# COMPOSING the left context into the left unit reveals it. The forest recovers
# it (> 0.9), a FROZEN-RANDOM composer does NOT (< 0.7) — recursion is load-
# bearing. These are the gates that must pass before any GPU spend. Docs come
# from the module's own _make_bracket_docs (the same the self-test uses).
# --------------------------------------------------------------------------- #
def test_token_local_baseline_is_at_chance():
    """A token-local classifier fit ON the training set (an upper bound) stays
    < 0.6 F1 — the junction's own single word is uninformative by construction."""
    h = 24
    docs = _make_bracket_docs(h, range(80))
    f1 = _token_local_baseline_f1(docs, h)
    assert f1 < 0.6, f"single-token not at chance (F1={f1:.3f})"


def test_adjacent_pair_baseline_is_at_chance():
    """The verifier's PAIR-LOCAL attack at full strength: a logistic classifier
    reading the junction's OWN adjacent pair [v_j ; v_{j+1}], fit ON the training
    set (upper bound), still stays < 0.6 — the boundary depends on symbol[j-1],
    which the (j, j+1) pair cannot see. Guards against a merely pair-local signal
    (a frozen-random composer + head previously scored 0.96 on such a signal)."""
    h = 24
    docs = _make_bracket_docs(h, range(80))
    f1 = _adjacent_pair_baseline_f1(docs, h)
    assert f1 < 0.6, f"adjacent-pair not at chance (F1={f1:.3f}) -> pair-local!"


def test_adjacent_cosine_baseline_is_at_chance():
    """The verifier's raw-cosine attack: best-threshold F1 on cosine(v_j, v_{j+1})
    (which scored 1.00 on the OLD pair-local construction) must be < 0.6 here."""
    h = 24
    docs = _make_bracket_docs(h, range(80))
    f1 = _adjacent_cosine_baseline_f1(docs)
    assert f1 < 0.6, f"adjacent-cosine not at chance (F1={f1:.3f}) -> pair-local!"


def test_forest_recovers_compositional_boundaries():
    """A forest (composer LEARNS) trained with the agglomerative-from-singletons
    objective recovers the left-context boundary on HELD-OUT docs at > 0.9 F1
    under the composed-unit readout — proof the architecture USES composition.
    (This is the gate that must pass before any GPU spend.)"""
    h = 24
    train_docs = _make_bracket_docs(h, range(80))
    test_docs = _make_bracket_docs(h, range(80, 100))
    forest, first, last = _train_forest_on_docs(train_docs, h, seed=3,
                                                freeze_composer=False)
    assert last < first, f"training did not reduce the loss ({first:.3f}->{last:.3f})"
    f1 = _forest_boundary_f1(forest, test_docs, readout="agglom")
    assert f1 > 0.9, f"forest failed the compositional contract (F1={f1:.3f})"


def test_frozen_random_composer_fails_control():
    """LOAD-BEARING-RECURSION control: with the composer FROZEN at its random init
    (only the head trains), the same forest CANNOT recover the left-context
    boundary (< 0.7 F1) — a random composer builds no left representation that
    exposes the (A,B) conjunction. This proves the compositional recovery above
    comes from LEARNED recursion, not from the head reading a pair-local shortcut.
    """
    h = 24
    train_docs = _make_bracket_docs(h, range(80))
    test_docs = _make_bracket_docs(h, range(80, 100))
    full, _, _ = _train_forest_on_docs(train_docs, h, seed=3,
                                       freeze_composer=False)
    frozen, _, _ = _train_forest_on_docs(train_docs, h, seed=3,
                                         freeze_composer=True)
    full_f1 = _forest_boundary_f1(full, test_docs, readout="agglom")
    frozen_f1 = _forest_boundary_f1(frozen, test_docs, readout="agglom")
    assert frozen_f1 < 0.7, \
        f"frozen-random composer did NOT fail (F1={frozen_f1:.3f}) -> signal is " \
        "solvable without learned composition"
    assert full_f1 > frozen_f1 + 0.2, \
        f"learned composer {full_f1:.3f} not clearly above frozen {frozen_f1:.3f}"


def test_self_test_runs_clean():
    """The module's own self-test (superset of the above) runs end-to-end."""
    self_test()


# --------------------------------------------------------------------------- #
# _es MECHANISMS (Noor, 2026-07-11): LLRD ladder, R-Drop KL, SAM perturb/restore,
# early-stop trigger + best-restore. All corpus-independent & network-free (a tiny
# fake encoder module stands in for the BertModel layout the LLRD grouper reads).
# --------------------------------------------------------------------------- #
class _FakeLayer:
    def __init__(self, h):
        import torch.nn as nn
        self.lin = nn.Linear(h, h)

    def parameters(self):
        return self.lin.parameters()


class _FakeEncoder:
    """Minimal stand-in exposing the BertModel attributes llrd_forest_param_groups
    reads: `embeddings`, `encoder.layer` (list), optional `pooler`, `.parameters()`."""
    def __init__(self, h=8, n_layers=12):
        import torch.nn as nn
        self.embeddings = nn.Linear(h, h)
        self.pooler = nn.Linear(h, h)

        class _Enc:
            pass
        self.encoder = _Enc()
        self.encoder.layer = [_FakeLayer(h) for _ in range(n_layers)]

    def parameters(self):
        ps = list(self.embeddings.parameters()) + list(self.pooler.parameters())
        for ly in self.encoder.layer:
            ps += list(ly.parameters())
        return ps


def test_llrd_ladder_decays():
    """LLRD builds >1 group with a STRICTLY-decaying encoder lr ladder (top=
    encoder_lr_top, embeddings lowest) and a flat composer+head group at
    composer_lr; a frozen encoder collapses to the single composer+head group."""
    h = 8
    enc = _FakeEncoder(h=h, n_layers=12)
    comp = make_rae(h, seed=0)
    head = make_head(h, seed=0)
    forest = Forest(enc, None, comp, head, h, "cpu")
    groups, ladder = llrd_forest_param_groups(forest, 2.5e-5, 0.874, 1e-3)
    assert len(groups) > 1
    names = [n for n, _, _ in ladder]
    assert names[0] == "composer+head"
    comp_lr = [lr for n, lr, _ in ladder if n == "composer+head"][0]
    assert abs(comp_lr - 1e-3) < 1e-12
    enc_lrs = [lr for n, lr, _ in ladder if "encoder.layer" in n or n == "embeddings"]
    assert enc_lrs == sorted(enc_lrs, reverse=True), enc_lrs
    assert abs(enc_lrs[0] - 2.5e-5) < 1e-9         # top layer at encoder_lr_top
    assert 4e-6 < enc_lrs[-1] < 6e-6               # embeddings ~5e-6 (12-layer decay)
    # frozen encoder -> only the composer+head group
    import torch
    for p in enc.parameters():
        p.requires_grad_(False)
    fg, fl = llrd_forest_param_groups(forest, 2.5e-5, 0.874, 1e-3)
    assert len(fg) == 1 and fl[0][0] == "composer+head"


def test_rdrop_kl_positive_with_dropout_zero_without():
    """R-Drop symmetric-KL over the matched gold/curriculum junction logits is > 0
    when dropout is ON (the two passes differ) and EXACTLY 0 when dropout is OFF
    (the passes are identical), and it carries gradient to the composer."""
    import torch
    h = 16
    comp = make_rae(h, seed=0)
    head = make_head(h, seed=0)
    forest = Forest(None, None, comp, head, h, "cpu")
    wv = torch.tensor(np.random.default_rng(3).normal(size=(11, h)),
                      dtype=torch.float32)
    labels = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    forest.train()                                    # dropout ON
    _l1, _n1, _p1, s1 = forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    _l2, _n2, _p2, s2 = forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    assert s1 and set(s1) == set(s2)                  # same supervised junctions
    kl = rdrop_symmetric_kl(s1, s2)
    assert float(kl.detach()) > 0
    kl.backward()
    assert comp.enc.weight.grad is not None and \
        float(comp.enc.weight.grad.abs().sum()) > 0
    forest.eval()                                     # dropout OFF -> identical
    _, _, _, e1 = forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    _, _, _, e2 = forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    assert abs(float(rdrop_symmetric_kl(e1, e2).detach())) < 1e-9


def test_sam_perturb_then_restore_and_uses_perturbed_grad():
    """First-order SAM: the ascent step PERTURBS the weights (nonzero move), the
    descent gradient at w+eps DIFFERS from the plain gradient at w, and restoring
    from the snapshot returns the weight bit-for-bit."""
    import torch
    h = 16
    comp = make_rae(h, seed=0)
    head = make_head(h, seed=0)
    forest = Forest(None, None, comp, head, h, "cpu")
    forest.eval()                                     # deterministic (no dropout)
    params = list(comp.parameters()) + list(head.parameters())
    wv = torch.tensor(np.random.default_rng(21).normal(size=(9, h)),
                      dtype=torch.float32)
    labels = [0, 1, 0, 0, 1, 0, 1, 0, 1]
    watch = comp.enc.weight
    w_before = watch.detach().clone()

    def bl():
        l, _, _ = forest_window_loss(forest, wv, labels, 3.0,
                                     rng=np.random.default_rng(2))
        return l

    for p in params:
        p.grad = None
    bl().backward()
    g_ascent = watch.grad.detach().clone()
    gnorm = sum(float((p.grad.detach() ** 2).sum())
                for p in params if p.grad is not None) ** 0.5
    scale = 0.05 / (gnorm + 1e-12)
    w_orig = {}
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            w_orig[p] = p.detach().clone()
            p.add_(p.grad * scale)
    assert float((watch.detach() - w_before).abs().sum()) > 0   # perturbed
    for p in params:
        p.grad = None
    bl().backward()
    g_descent = watch.grad.detach().clone()
    assert float((g_descent - g_ascent).abs().sum()) > 0        # different grad
    with torch.no_grad():
        for p, w in w_orig.items():
            p.copy_(w)
    assert torch.equal(watch.detach(), w_before)                # restored exactly


def test_early_stop_best_epoch_and_restore():
    """Early-stop monitor rule on [0.5,0.6,0.55,0.54,0.53] (patience 3, min-epochs
    2) -> best epoch 2 (0.6), stop after epoch 5; and BEST-restore: after mutating
    the model to a WORSE 'last epoch', _restore_weights(best_snapshot) returns the
    BEST weights (tensor compare), NOT the last."""
    import torch
    monitor = [0.5, 0.6, 0.55, 0.54, 0.53]
    patience, min_epochs = 3, 2
    best_f1, best_epoch, no_improve, stopped_at = -1.0, 0, 0, len(monitor)
    for ep, f1 in enumerate(monitor):
        if f1 > best_f1 + 1e-6:
            best_f1, best_epoch, no_improve = f1, ep + 1, 0
        else:
            no_improve += 1
            if (ep + 1) >= min_epochs and no_improve >= patience:
                stopped_at = ep + 1
                break
    assert best_epoch == 2 and abs(best_f1 - 0.6) < 1e-9
    assert stopped_at == 5

    h = 12
    forest = Forest(None, None, make_rae(h, 0), make_head(h, 0), h, "cpu")
    best_snap = _snapshot_weights(forest)
    best_w = forest.composer.enc.weight.detach().clone()
    with torch.no_grad():                             # simulate a worse last epoch
        forest.composer.enc.weight.add_(1.234)
        forest.head[0].weight.add_(-0.5)
    last_w = forest.composer.enc.weight.detach().clone()
    assert not torch.equal(best_w, last_w)
    _restore_weights(forest, best_snap, "cpu")
    restored_w = forest.composer.enc.weight.detach().clone()
    assert torch.equal(restored_w, best_w)            # == BEST snapshot
    assert not torch.equal(restored_w, last_w)        # != last epoch


def test_default_window_loss_signature_unchanged():
    """The default (collect_sup_logits=False) still returns the 3-tuple, so all
    existing callers are unaffected; the 4-tuple only appears when opted in."""
    import torch
    h = 8
    forest = Forest(None, None, make_rae(h, 0), make_head(h, 0), h, "cpu")
    wv = torch.tensor(np.random.default_rng(2).normal(size=(7, h)),
                      dtype=torch.float32)
    labels = [0, 0, 1, 0, 0, 1, 1]
    out = forest_window_loss(forest, wv, labels, 3.0, rng=np.random.default_rng(0))
    assert len(out) == 3
    out4 = forest_window_loss(forest, wv, labels, 3.0,
                              rng=np.random.default_rng(0), collect_sup_logits=True)
    assert len(out4) == 4 and isinstance(out4[3], dict)


# --------------------------------------------------------------------------- #
# EXPOSURE-BIAS FIX (Noor, 2026-07-11): --curriculum-include-allcut and
# --scheduled-sampling. Both DEFAULT OFF must reduce to the pre-fix path bit-for-
# bit; both ON must ENGAGE (change the supervised start-state), keep every gold
# junction supervised, and still carry gradient to word-vectors + composer + head.
# Corpus-independent (synthetic vectors; no encoder, no data/).
# --------------------------------------------------------------------------- #
_EB_LABELS = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]


def _eb_forest(h=16, seed=0, train=False):
    f = Forest(None, None, make_rae(h, seed), make_head(h, seed), h, "cpu")
    f.train() if train else f.eval()
    return f


def test_curriculum_include_allcut_starts_from_all_separate():
    """--curriculum-include-allcut with allcut_prob=1.0 starts a window from ALL
    WORDS SEPARATE: n_leaf == the number of valid junctions (n-1, minus [PAR]),
    strictly MORE than the default near-gold curriculum refine, while the gold-KEEP
    count is identical (allcut keeps every gold boundary)."""
    import torch
    h = 16
    f = _eb_forest(h)                                 # eval -> deterministic
    n = len(_EB_LABELS)
    wv = torch.tensor(np.random.default_rng(1).normal(size=(n, h)),
                      dtype=torch.float32)
    _l0, nl0, np0 = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                       rng=np.random.default_rng(42))
    _la, nla, npa = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                       rng=np.random.default_rng(7),
                                       include_allcut=True, allcut_prob=1.0)
    n_valid = sum(1 for i in range(n - 1) if _EB_LABELS[i] != -100)
    assert nla == n_valid, (nla, n_valid)             # every valid junction is a cut
    assert nla > nl0, (nla, nl0)                       # more than the refine default
    assert npa == np0                                  # gold-KEEPs unchanged


def test_curriculum_include_allcut_prob_zero_is_default():
    """allcut_prob=0.0 (even with the flag set) is bit-for-bit the default path."""
    import torch
    h = 16
    f = _eb_forest(h)
    n = len(_EB_LABELS)
    wv = torch.tensor(np.random.default_rng(1).normal(size=(n, h)),
                      dtype=torch.float32)
    l_def, nl_def, _ = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                          rng=np.random.default_rng(42))
    l_ac0, nl_ac0, _ = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                          rng=np.random.default_rng(42),
                                          include_allcut=True, allcut_prob=0.0)
    assert abs(float(l_def.detach()) - float(l_ac0.detach())) < 1e-9
    assert nl_def == nl_ac0


def test_free_run_never_crosses_gold_and_compresses():
    """The scheduled-sampling free-run (a no_grad state selector) returns a survivor
    junction set that (a) is a SUPERSET of the gold-KEEP set (never crosses a gold
    boundary) and (b) is a strict COMPRESSION of all-separate (it actually merges)."""
    import torch
    h = 16
    f = _eb_forest(h)
    n = len(_EB_LABELS)
    wv = torch.tensor(np.random.default_rng(1).normal(size=(n, h)),
                      dtype=torch.float32)
    keep_set = set(i for i in range(n - 1) if _EB_LABELS[i] == 1)
    all_valid = [i for i in range(n - 1) if _EB_LABELS[i] != -100]
    surv = _free_run_start_cuts(f, wv, keep_set, all_valid, n_free_steps=None)
    assert surv is not None
    assert keep_set.issubset(surv)                     # never crossed a gold boundary
    assert len(surv) < len(all_valid)                  # merged something
    assert surv.issubset(set(all_valid))               # only valid junctions survive


def test_scheduled_sampling_engages_and_grad_flows():
    """--scheduled-sampling P=1.0 supervises from the model's OWN free-run state
    (n_leaf == the survivor count, distinct from both default and all-separate),
    keeps all gold junctions, and carries gradient to word-vectors + composer +
    head (the free-run is no_grad; the supervised scoring carries grad)."""
    import torch
    h = 16
    f = _eb_forest(h)
    n = len(_EB_LABELS)
    wv = torch.tensor(np.random.default_rng(1).normal(size=(n, h)),
                      dtype=torch.float32)
    keep_set = set(i for i in range(n - 1) if _EB_LABELS[i] == 1)
    all_valid = [i for i in range(n - 1) if _EB_LABELS[i] != -100]
    surv = _free_run_start_cuts(f, wv, keep_set, all_valid, n_free_steps=None)
    _l0, nl0, np0 = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                       rng=np.random.default_rng(42))
    _ls, nls, nps = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                       rng=np.random.default_rng(3),
                                       scheduled_sampling=1.0)
    assert nls == len(surv)                            # supervises the own-reached state
    assert nps == np0                                  # gold-KEEPs preserved
    assert nls != n - 1                                # not all-separate (it merged)

    # grad flows: fresh trainable forest, backward through the SS loss
    ft = _eb_forest(h, seed=1, train=True)
    wvg = torch.tensor(np.random.default_rng(2).normal(size=(n, h)),
                       dtype=torch.float32, requires_grad=True)
    lg, _n, _p = forest_window_loss(ft, wvg, _EB_LABELS, 3.0,
                                    rng=np.random.default_rng(5),
                                    scheduled_sampling=1.0)
    assert lg is not None and lg.requires_grad
    lg.backward()
    assert wvg.grad is not None and float(wvg.grad.abs().sum()) > 0
    assert ft.composer.enc.weight.grad is not None and \
        float(ft.composer.enc.weight.grad.abs().sum()) > 0
    assert ft.head[0].weight.grad is not None and \
        float(ft.head[0].weight.grad.abs().sum()) > 0


def test_scheduled_sampling_zero_is_default():
    """scheduled_sampling=0.0 is bit-for-bit the default path."""
    import torch
    h = 16
    f = _eb_forest(h)
    n = len(_EB_LABELS)
    wv = torch.tensor(np.random.default_rng(1).normal(size=(n, h)),
                      dtype=torch.float32)
    l_def, nl_def, _ = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                          rng=np.random.default_rng(42))
    l_ss0, nl_ss0, _ = forest_window_loss(f, wv, _EB_LABELS, 3.0,
                                          rng=np.random.default_rng(42),
                                          scheduled_sampling=0.0)
    assert abs(float(l_def.detach()) - float(l_ss0.detach())) < 1e-9
    assert nl_def == nl_ss0
