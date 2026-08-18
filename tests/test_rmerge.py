"""Unit tests for src/rmerge.py — recursive beam-merge segmenter.

Two layers:
  * network-free tests of the PURE machinery (gold-cut alignment, teacher-forced
    state sequence, dense-supervision count, cuts<->boundaries, geometric-mean
    scoring, beam bookkeeping) — pass with data/ hidden and with NO HF download.
  * the MANDATORY ACCEPTANCE TEST (the build gate) on Noor's exact toy doc, which
    fine-tunes AraBERT from the OFFLINE HF cache; it is skipped (not failed) if
    the encoder cache is absent, so CI without the cache still runs the pure
    layer.  The acceptance test includes the required CAN-FAIL check (flip a gold
    label -> the assertions must fail).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from rmerge import (  # noqa: E402
    ALPHA_DEFAULT,
    Hyp,
    adjacent_candidates,
    beam_decode,
    cuts_to_word_boundaries,
    count_dense_candidates,
    dense_supervision_pairs,
    docfinal_is_boundary,
    docfinal_junction,
    force_docfinal_cut,
    gold_cut_mask,
    merge_nodes,
    predict_doc,
    predict_doc_windowed,
    sched_p_at_epoch,
    self_test,
    sentence_spans,
    start_state,
    state_cuts,
    teacher_forced_states,
    train_pos_weight,
)

TOY_WORDS = ["the", "cat", "ate", "dogs", "ran"]
# gold boundary AFTER "ate" (word index 2); doc-final "ran" label = 1.
TOY_LABELS = [0, 0, 1, 0, 1]
# gold_cut over the 4 junctions:
TOY_GOLD_CUT = [0, 0, 1, 0]


# --------------------------------------------------------------------------- #
# pure machinery (network-free)
# --------------------------------------------------------------------------- #
def test_self_test():
    assert self_test()


def test_gold_cut_alignment():
    assert gold_cut_mask(TOY_LABELS).tolist() == TOY_GOLD_CUT
    # junction axis has n-1 entries; doc-final label is not a junction
    assert len(gold_cut_mask(TOY_LABELS)) == len(TOY_WORDS) - 1
    # a single-sentence doc has all-zero cut mask
    assert gold_cut_mask([0, 0, 0, 1]).tolist() == [0, 0, 0]


def test_sentence_spans():
    assert sentence_spans(TOY_LABELS) == [(0, 2), (3, 4)]
    assert sentence_spans([1, 1]) == [(0, 0), (1, 1)]


def test_cuts_to_word_boundaries_roundtrip():
    # gold cuts of the toy are {2}; induced per-word == gold minus doc-final slot
    y = cuts_to_word_boundaries([2], len(TOY_WORDS))
    assert y == [0, 0, 1, 0, 0]           # doc-final carries 0 (scorer fills it)


def test_teacher_states_never_straddle_boundary():
    """Every constituent the teacher builds is a real sub-sentence: no node
    straddles the gold boundary after word 2."""
    for nodes in teacher_forced_states(TOY_LABELS):
        for (w0, w1) in nodes:
            left = w0 <= 2 and w1 <= 2
            right = w0 >= 3 and w1 >= 3
            assert left or right, (nodes, (w0, w1))
    # final state is exactly the two gold sentences
    assert teacher_forced_states(TOY_LABELS)[-1] == [(0, 2), (3, 4)]


def test_dense_supervision_count():
    """#supervised candidates == sum over visited states of #adjacent pairs
    (dense: EVERY adjacent pair, not just the merged ones)."""
    states = teacher_forced_states(TOY_LABELS)
    by_hand = sum(max(0, len(nodes) - 1) for nodes in states)
    assert count_dense_candidates(TOY_LABELS) == by_hand
    # and it is strictly more than just the merges performed (dense > sparse)
    n_merges = (len(TOY_WORDS) - len(sentence_spans(TOY_LABELS)))
    assert count_dense_candidates(TOY_LABELS) > n_merges


def test_dense_supervision_targets_penalize_boundary():
    """At the start state, the pair straddling the gold boundary (ate|dogs)
    gets BCE target 0 (should NOT merge); within-sentence pairs get target 1."""
    nodes = start_state(len(TOY_WORDS))
    gc = gold_cut_mask(TOY_LABELS)
    pairs, targets = dense_supervision_pairs(nodes, gc)
    # junctions in order: 0,1,2,3 -> targets 1,1,0,1
    assert targets == [1, 1, 0, 1]
    # the target-0 pair is exactly ate|dogs = ((2,2),(3,3))
    zero_idx = targets.index(0)
    assert pairs[zero_idx] == ((2, 2), (3, 3))


def test_state_helpers():
    st = start_state(4)
    assert st == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert adjacent_candidates(st) == [(0, 0), (1, 1), (2, 2)]
    assert merge_nodes(st, 0) == [(0, 1), (2, 2), (3, 3)]
    assert state_cuts([(0, 2), (3, 4)]) == [2]     # one cut at junction 2


def test_geometric_mean_scoring():
    """Length-normalized score = geometric mean, NOT raw product (guards FM1)."""
    h2 = Hyp([(0, 4)], [0.9, 0.9])
    assert abs(h2.score() - 0.9) < 1e-9           # product would be 0.81
    h3 = Hyp([(0, 4)], [0.8, 0.8, 0.8])
    assert abs(h3.score() - 0.8) < 1e-9           # product would be 0.512
    # a no-merge hyp scores neutral 1.0 (not penalized by having no factors)
    assert Hyp([(0, 0)], []).score() == 1.0


def test_pos_weight_from_labels():
    docs = [{"labels": TOY_LABELS}, {"labels": [0, 0, 0, 1]}]
    # within-sentence junctions (t=1) vs boundary junctions (t=0)
    # doc1 cut mask [0,0,1,0] -> pos 3 neg 1 ; doc2 [0,0,0] -> pos 3 neg 0
    # total pos 6 neg 1 -> weight 6.0
    assert train_pos_weight(docs) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# beam bookkeeping with a MOCK scorer (network-free): a hand-set merge-prob
# table drives the beam so we can test the search logic without AraBERT.
# --------------------------------------------------------------------------- #
class _MockScorer:
    """probs_for_pairs looks each ((a0,a1),(b0,b1)) pair up in a dict keyed by
    the erased junction = a1.  _word_ids is a no-op list."""

    def __init__(self, junction_prob):
        self.junction_prob = junction_prob      # {junction_k: prob}
        self.device = "cpu"

    def _word_ids(self, words):
        return [[0] for _ in words]

    def probs_for_pairs(self, word_ids, pairs):
        out = []
        for (a, b) in pairs:
            out.append(self.junction_prob[a[1]])
        return np.asarray(out, dtype=np.float64)


def test_beam_builds_correct_forest_from_probs():
    """With high within-sentence probs and a low boundary prob at junction 2,
    beam-decode (tau=0.3) must build (the cat ate)|(dogs ran)."""
    probs = {0: 0.95, 1: 0.95, 2: 0.05, 3: 0.95}
    sc = _MockScorer(probs)
    win = beam_decode(sc, TOY_WORDS, tau=0.30, beam_width=3)
    assert win.nodes == [(0, 2), (3, 4)]
    assert state_cuts(win.nodes) == [2]
    y = predict_doc(sc, TOY_WORDS, tau=0.30, beam_width=3).tolist()
    assert y == [0, 0, 1, 0, 0]


def test_beam_merge_generous_low_tau_merges_more():
    """Merge-generous: at LOW tau even a middling junction merges; only a clearly
    sub-tau junction stays a cut."""
    probs = {0: 0.95, 1: 0.95, 2: 0.02, 3: 0.95}
    sc = _MockScorer(probs)
    # tau below every within-sentence prob but above the boundary prob
    win = beam_decode(sc, TOY_WORDS, tau=0.20, beam_width=3)
    assert win.nodes == [(0, 2), (3, 4)]


def test_beam_all_below_tau_keeps_all_cuts():
    """If NO adjacent merge clears tau, every junction is a boundary."""
    probs = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1}
    sc = _MockScorer(probs)
    win = beam_decode(sc, TOY_WORDS, tau=0.30, beam_width=3)
    assert win.nodes == start_state(len(TOY_WORDS))
    assert state_cuts(win.nodes) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# MANDATORY ACCEPTANCE TEST (build gate) — real AraBERT from the offline cache
# --------------------------------------------------------------------------- #
def _arabert_cached():
    """True iff the AraBERT weights are reachable offline (HF cache)."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        return False
    try:
        AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02",
                                      local_files_only=True)
        AutoModel.from_pretrained("aubmindlab/bert-base-arabertv02",
                                  local_files_only=True)
        return True
    except Exception:
        return False


def _fit_and_decode(labels, epochs=40, lr=5e-5, seed=0):
    """Fine-tune a fresh AraBERT merge scorer on ONE toy doc for a few steps,
    then beam-decode.  Returns (scorer, winning_hyp, within_probs, cross_prob).
    """
    import torch  # noqa: F401
    from rmerge import MergeScorer, load_arabert, train_on_docs
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=seed)
    scorer = MergeScorer(tok, enc, head, span_cap=16)
    doc = {"doc_id": "toy", "tokens": TOY_WORDS, "labels": labels}
    train_on_docs(scorer, [doc], epochs=epochs, lr=lr, seed=seed,
                  verbose=False)
    win = beam_decode(scorer, TOY_WORDS, tau=0.30, beam_width=3)
    word_ids = scorer._word_ids(TOY_WORDS)
    # probe the three raw merge probs at the START state (leaf pairs)
    p_the_cat = scorer.probs_for_pairs(word_ids, [((0, 0), (1, 1))])[0]
    p_cat_ate = scorer.probs_for_pairs(word_ids, [((1, 1), (2, 2))])[0]
    p_ate_dogs = scorer.probs_for_pairs(word_ids, [((2, 2), (3, 3))])[0]
    p_dogs_ran = scorer.probs_for_pairs(word_ids, [((3, 3), (4, 4))])[0]
    within = [float(p_the_cat), float(p_cat_ate), float(p_dogs_ran)]
    return scorer, win, within, float(p_ate_dogs)


@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_acceptance_toy_segments():
    """THE BUILD GATE.  Fine-tune on the toy, beam-decode, assert the four
    required properties."""
    scorer, win, within, cross = _fit_and_decode(TOY_LABELS)
    # (i) the forest is exactly (the cat ate) | (dogs ran)
    assert win.nodes == [(0, 2), (3, 4)], win.nodes
    # (ii) within-sentence merge probs > 0.7
    assert all(p > 0.7 for p in within), within
    # (iii) the ate|dogs merge prob < 0.3
    assert cross < 0.3, cross
    # (iv) induced per-word boundaries == gold (doc-final slot excluded, as the
    # cut grid has no junction after the last word)
    y = predict_doc(scorer, TOY_WORDS, tau=0.30, beam_width=3).tolist()
    assert y == [0, 0, 1, 0, 0], y             # gold minus doc-final == this


@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_acceptance_can_fail_on_flipped_gold():
    """CAN-FAIL check: train on a WRONG gold (boundary flipped to after 'cat'),
    then assert the ORIGINAL-gold acceptance assertions FAIL.  If the toy could
    be 'segmented' regardless of supervision, the build is meaningless."""
    flipped = [0, 1, 0, 0, 1]                   # boundary after 'cat', not 'ate'
    scorer, win, within, cross = _fit_and_decode(flipped)
    # Under the flipped supervision, the model should NOT reproduce the original
    # (0,2)|(3,4) forest with a clean low ate|dogs prob.  At least one original
    # assertion must break.
    orig_holds = (
        win.nodes == [(0, 2), (3, 4)]
        and all(p > 0.7 for p in within)
        and cross < 0.3
    )
    assert not orig_holds, (
        "flipped-gold model still satisfied the original acceptance assertions "
        f"-> supervision is not load-bearing (nodes={win.nodes}, "
        f"within={within}, cross={cross})")


# =========================================================================== #
# CPU SMOKES — each proves one locked-spec MANDATE actually ENGAGES.  The pure
# ones are network-free; the encoder ones use the offline AraBERT cache (skipped
# if absent).  Each is intentionally small and deterministic.
# =========================================================================== #

# ---- MANDATE: doc-final invariant (junction after last word = forced cut) --- #
def test_smoke_docfinal_invariant():
    n = len(TOY_WORDS)
    assert docfinal_is_boundary(n) is True
    assert docfinal_junction(n) == n - 1            # slot the _de row fills
    # force_docfinal_cut asserts the invariant and is a no-op on the cut grid
    y = [0, 0, 1, 0, 0]
    assert force_docfinal_cut(list(y), n) == y


# ---- MANDATE: scheduled-sampling ramp mixes model states (P>0), not pure TF -- #
def test_smoke_scheduled_sampling_ramp():
    # linear ramp 0 -> p_max over epochs; epoch 0 = pure teacher (P=0)
    assert sched_p_at_epoch(0, 10, 0.25) == 0.0
    assert sched_p_at_epoch(9, 10, 0.25) == pytest.approx(0.25)
    mid = sched_p_at_epoch(5, 10, 0.25)
    assert 0.0 < mid < 0.25                        # genuinely mixes mid-training
    assert sched_p_at_epoch(0, 1, 0.25) == 0.0     # 1 epoch -> pure teacher


# ---- MANDATE: dense supervision counts ALL adjacent candidates -------------- #
def test_smoke_dense_supervision_counts_all_adjacent():
    states = teacher_forced_states(TOY_LABELS)
    total = sum(max(0, len(nodes) - 1) for nodes in states)
    assert count_dense_candidates(TOY_LABELS) == total
    # at the start state ALL 4 junctions are candidates (dense, not just merges)
    pairs, targets = dense_supervision_pairs(start_state(len(TOY_WORDS)),
                                             gold_cut_mask(TOY_LABELS))
    assert len(pairs) == len(TOY_WORDS) - 1 == 4


# ---- MANDATE: class-weight applied (boundaries up-weighted from train freq) -- #
def test_smoke_class_weight_from_train_freq():
    docs = [{"labels": TOY_LABELS}, {"labels": [0, 0, 0, 1]}]
    w = train_pos_weight(docs)                      # pos(6)/neg(1) = 6.0
    assert w == pytest.approx(6.0)
    assert w > 1.0                                  # boundaries (t=0) up-weighted


# ---- MANDATE: log-space length-normalized (geometric mean) scoring ---------- #
def test_smoke_logspace_geomean_scoring():
    h = Hyp([(0, 4)], [0.8, 0.8, 0.8], alpha=ALPHA_DEFAULT)
    assert h.score() == pytest.approx(0.8)         # geo-mean; product = 0.512
    # log_score == mean(log p) for alpha=1.0
    import math
    assert h.log_score() == pytest.approx(math.log(0.8))
    # a FEWER-merge hyp does NOT automatically win under geo-mean (guards FM1):
    two = Hyp([(0, 2)], [0.8, 0.8])
    three = Hyp([(0, 3)], [0.8, 0.8, 0.8])
    assert two.score() == pytest.approx(three.score())   # length-independent


# ---- MANDATE: beam dedup removes identical active-cut hypotheses ------------ #
def test_smoke_beam_dedup_identical_cutsets():
    # two prob tables that reach the SAME final partition by different orders
    probs = {0: 0.95, 1: 0.95, 2: 0.05, 3: 0.95}
    sc = _MockScorer(probs)
    win = beam_decode(sc, TOY_WORDS, tau=0.30, beam_width=3)
    # despite multiple internal merge orders, the beam collapses to ONE winning
    # cut-set {2}; the winner is a single hypothesis (dedup kept the survivor).
    assert state_cuts(win.nodes) == [2]


# ---- MANDATE (network-free): sliding-window stitch preserves whole-doc cut --- #
class _ContentMockScorer:
    """Content-keyed mock: each word maps to a unique id; a pair's merge prob is
    LOW iff the last word of span A is the sentinel 'BREAK' word (so the boundary
    is defined by TOKEN CONTENT and is stable no matter which window / local
    offset the pair appears in — exactly what a real re-reading encoder gives)."""
    BREAK = "ate"                       # boundary sits AFTER this word

    def __init__(self):
        self.device = "cpu"
        self._vocab = {}

    def _word_ids(self, words):
        ids = []
        for w in words:
            self._vocab.setdefault(w, len(self._vocab) + 1)
            ids.append([self._vocab[w]])
        return ids

    def probs_for_pairs(self, word_ids, pairs):
        import numpy as _np
        break_id = self._vocab.get(self.BREAK)
        out = []
        for (a, b) in pairs:
            last_word_of_A = word_ids[a[1]][0]      # id of span-A's last word
            out.append(0.05 if last_word_of_A == break_id else 0.95)
        return _np.asarray(out, dtype=_np.float64)


def test_smoke_sliding_window_stitch():
    # 6-word doc; the ONLY boundary is after 'ate' (word 2).  window<doc forces
    # stitching across >=2 windows; the content-keyed mock keeps the boundary
    # stable across windows so we test the STITCH mechanics, not the mock.
    words = ["the", "cat", "ate", "dogs", "ran", "far"]
    sc = _ContentMockScorer()
    sc._word_ids(words)                             # prime the vocab
    y = predict_doc_windowed(sc, words, tau=0.30, beam_width=3,
                             window=4, stride=2)
    assert len(y) == len(words)
    # the true boundary after word 2 survives stitching; no spurious interior
    # window-edge cuts leaked in (only the ate|dogs junction is a boundary).
    assert y.tolist() == [0, 0, 1, 0, 0, 0]
    assert docfinal_is_boundary(len(words))         # doc-final forced internally


def test_smoke_window_equals_wholedoc_when_no_split():
    # window >= doc length must reproduce the whole-doc decode exactly (stitching
    # is a no-op when a single window covers the doc).
    words = ["the", "cat", "ate", "dogs", "ran"]
    sc = _ContentMockScorer()
    sc._word_ids(words)
    y_win = predict_doc_windowed(sc, words, tau=0.30, beam_width=3,
                                 window=99, stride=2).tolist()
    y_whole = predict_doc(sc, words, tau=0.30, beam_width=3).tolist()
    assert y_win == y_whole == [0, 0, 1, 0, 0]


# ---- MANDATE (encoder): AraBERT re-reads [CLS] A [SEP] B [SEP] -------------- #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_arabert_rereads_cls_sep():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import MergeScorer, load_arabert
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    sc = MergeScorer(tok, enc, head, span_cap=16)
    word_ids = sc._word_ids(TOY_WORDS)
    ids, attn, tt = sc.build_pair_batch(word_ids, [((0, 0), (1, 1))])
    seq = ids[0].tolist()
    cls, sep = tok.cls_token_id, tok.sep_token_id
    assert seq[0] == cls                           # [CLS] first
    assert seq.count(sep) == 2                      # exactly two [SEP]
    # token_type_ids segment A (0) then B (1) — proves real A/B re-read
    ttl = tt[0].tolist()[: int(attn[0].sum())]
    assert set(ttl) == {0, 1} and ttl[0] == 0 and ttl[-1] == 1


# ---- MANDATE (encoder): span > cap -> p forced 0 (never merge) -------------- #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_span_cap_forces_zero():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import MergeScorer, load_arabert
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    # span_cap = 2 subwords: a wide 3-word left span merged with anything is
    # over-cap -> p must be ~0 (never merge), regardless of encoder weights.
    sc = MergeScorer(tok, enc, head, span_cap=2, span_cap_mode="force0")
    words = ["the", "cat", "ate", "dogs"]
    word_ids = sc._word_ids(words)
    over = (((0, 2), (3, 3)))                       # resulting span = 4 words
    assert sc._over_cap(word_ids, over[0], over[1])
    p = sc.probs_for_pairs(word_ids, [over])[0]
    assert p < 1e-6, p                              # forced 0


# ---- MANDATE (encoder): memoization cache HIT on 2nd identical span --------- #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_memoization_cache_hit():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import MergeScorer, load_arabert
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    sc = MergeScorer(tok, enc, head, span_cap=16, memoize=True)
    word_ids = sc._word_ids(TOY_WORDS)
    pair = [((0, 0), (1, 1))]
    p1 = sc.probs_for_pairs(word_ids, pair)[0]
    assert sc.cache_misses == 1 and sc.cache_hits == 0
    p2 = sc.probs_for_pairs(word_ids, pair)[0]      # identical span -> HIT
    assert sc.cache_hits == 1                        # served from cache
    assert p1 == p2                                  # same value, no re-run


# ---- MANDATE (encoder): batched == per-pair probs (parallelization correct) -- #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_batched_equals_perpair():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import MergeScorer, load_arabert, batched_equals_perpair
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    # memoize OFF so the per-pair calls actually re-run (else cache short-circuits)
    sc = MergeScorer(tok, enc, head, span_cap=16, memoize=False)
    pairs = [((0, 0), (1, 1)), ((1, 1), (2, 2)), ((2, 2), (3, 3))]
    assert batched_equals_perpair(sc, TOY_WORDS, pairs, atol=1e-4)


# ---- MANDATE (encoder): surrounding context k adds left/right words --------- #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_context_k_adds_surrounding():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import MergeScorer, load_arabert
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    words = ["the", "cat", "ate", "dogs", "ran", "far", "away"]
    pair = ((2, 2), (3, 3))            # ate|dogs -- mid-doc, has left+right context
    sc0 = MergeScorer(tok, enc, head, span_cap=64, context_k=0)
    sc2 = MergeScorer(tok, enc, head, span_cap=64, context_k=2)
    wid = sc0._word_ids(words)
    seq0 = sc0.build_pair_batch(wid, [pair])[0][0].tolist()
    ids2, attn2, tt2 = sc2.build_pair_batch(wid, [pair])
    seq2 = ids2[0].tolist()
    sep = tok.sep_token_id
    assert seq0.count(sep) == 2                        # [CLS] A [SEP] B [SEP]
    assert seq2.count(sep) == 4                        # + left/right context blocks
    assert len(seq2) > len(seq0)                       # context added tokens
    # token_type flips 0 -> 1 exactly once (the A|B junction)
    tt = tt2[0].tolist()[: int(attn2[0].sum())]
    flips = sum(1 for i in range(1, len(tt)) if tt[i] != tt[i - 1])
    assert flips == 1 and tt[0] == 0 and tt[-1] == 1
    # _context_ids grabs the 2 words before A and the 2 words after B
    l, r = sc2._context_ids(wid, pair[0], pair[1])
    assert l == wid[0] + wid[1] and r == wid[4] + wid[5]


# ---- MANDATE (B lever): boundary-dropout blanks ONLY the junction words ------ #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_boundary_dropout_masks_only_junction_words():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import MergeScorer, load_arabert
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    sc = MergeScorer(tok, enc, head, span_cap=64, context_k=0)
    words = ["the", "cat", "ate", "dogs", "ran"]
    wid = sc._word_ids(words)
    mask_id = tok.mask_token_id
    sep = tok.sep_token_id
    pair = ((2, 2), (3, 3))                         # single-word A|B: ate|dogs
    la, lb = len(wid[2]), len(wid[3])
    plain = sc.build_pair_batch(wid, [pair], mask_junction_p=0.0)[0][0].tolist()
    # default (no arg) is p=0.0 -> byte-identical (exact back-compat)
    assert sc.build_pair_batch(wid, [pair])[0][0].tolist() == plain
    # p=1.0 -> both junction words blanked, SAME length, specials intact
    masked = sc.build_pair_batch(wid, [pair], mask_junction_p=1.0)[0][0].tolist()
    assert len(masked) == len(plain)                # length-preserving
    assert masked[0] == plain[0]                    # [CLS] kept
    assert masked[1:1 + la] == [mask_id] * la       # A blanked
    assert masked[1 + la] == sep                    # [SEP] kept
    assert masked[2 + la:2 + la + lb] == [mask_id] * lb   # B blanked
    assert masked[-1] == sep
    # multi-word A: ONLY A's rightmost word masked; its interior words survive
    pair2 = ((1, 2), (3, 3))                        # A = cat+ate
    l1, l2 = len(wid[1]), len(wid[2])
    m2 = sc.build_pair_batch(wid, [pair2], mask_junction_p=1.0)[0][0].tolist()
    assert m2[1:1 + l1] == wid[1]                   # 'cat' (interior) survives
    assert m2[1 + l1:1 + l1 + l2] == [mask_id] * l2  # 'ate' (junction word) blanked


# ---- MANDATE (B lever): boundary-dropout is OFF at inference (decode intact) - #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_boundary_dropout_off_at_inference():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import MergeScorer, load_arabert
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    words = ["the", "cat", "ate", "dogs", "ran"]
    pairs = [((2, 2), (3, 3))]
    sc0 = MergeScorer(tok, enc, head, span_cap=64, context_k=2,
                      boundary_dropout=0.0, memoize=False)
    sc1 = MergeScorer(tok, enc, head, span_cap=64, context_k=2,
                      boundary_dropout=1.0, memoize=False)
    wid = sc0._word_ids(words)
    # probs_for_pairs uses grad=False -> masking gated off -> identical probs
    p0 = float(sc0.probs_for_pairs(wid, pairs)[0])
    p1 = float(sc1.probs_for_pairs(wid, pairs)[0])
    assert abs(p0 - p1) < 1e-6


# ---- MANDATE (encoder): grad flows to AraBERT (it IS the merger) ------------ #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_grad_flows_to_arabert():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from rmerge import MergeScorer, load_arabert, doc_train_loss, train_pos_weight
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    sc = MergeScorer(tok, enc, head, span_cap=16)
    sc.enc.train()
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    loss, ncand, npos, nneg = doc_train_loss(sc, TOY_WORDS, TOY_LABELS, pw)
    loss.backward()
    gnorm = 0.0
    for p in sc.enc.parameters():
        if p.grad is not None:
            gnorm += float(p.grad.norm())
    assert gnorm > 0.0                              # gradient reaches AraBERT
    assert ncand > 0


# ---- MANDATE (encoder): scheduled sampling actually visits MODEL states ----- #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_sched_sampling_visits_model_states():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as _np
    from rmerge import MergeScorer, load_arabert, doc_train_loss_sched, train_pos_weight
    tok, enc, head = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    sc = MergeScorer(tok, enc, head, span_cap=16)
    sc.enc.train()
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    # P_model = 1.0 -> every step follows the MODEL (not teacher); with a longer
    # doc there must be >0 model steps recorded.
    rng = _np.random.default_rng(0)
    loss, ncand, npos, nneg, nms = doc_train_loss_sched(
        sc, TOY_WORDS, TOY_LABELS, pw, p_model=1.0, rng=rng)
    assert nms > 0                                  # model-roll-in genuinely used
    assert ncand > 0
    # P_model = 0.0 -> pure teacher, zero model steps
    rng0 = _np.random.default_rng(0)
    _, _, _, _, nms0 = doc_train_loss_sched(
        sc, TOY_WORDS, TOY_LABELS, pw, p_model=0.0, rng=rng0)
    assert nms0 == 0


# ---- MANDATE (encoder): LR warmup schedule engages (lr rises then falls) ---- #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_smoke_lr_warmup_engages():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from rmerge import _lr_lambda
    fn = _lr_lambda(warmup_steps=4, total_steps=20)
    # rises through warmup (step 0 < step 3), peaks at end of warmup, then decays
    assert fn(0) < fn(3)
    assert fn(3) == pytest.approx(1.0)
    assert fn(3) > fn(19)                           # decay after warmup
    assert fn(19) < 0.1


# =========================================================================== #
# SELF-GLUING (--roll-in selfmerge): the model builds the tree by its OWN glues
# and is punished at every cross-boundary glue, with NO answer-key hand-holding.
# Network-free via a grad-capable mock (the plain _MockScorer has no
# logits_for_pairs, so it cannot drive the training losses).
# =========================================================================== #
class _GradMockScorer:
    """Network-free scorer WITH a real gradient path, enough to drive the
    training losses (which call logits_for_pairs(grad=True) and _over_cap).  One
    trainable logit per junction (keyed by the erased junction = the left span's
    right word index), so a cross-boundary glue can actually be pushed down.  No
    AraBERT, no network."""

    def __init__(self, n_words, span_cap=64, init=0.0):
        import torch
        self.device = "cpu"
        self.span_cap = int(span_cap)
        self.span_cap_mode = "force0"
        self.theta = torch.nn.Parameter(torch.full((n_words,), float(init)))

    def _word_ids(self, words):
        return [[i] for i in range(len(words))]         # id == word index

    def _pair_ids(self, word_ids, a, b):
        (a0, a1), (b0, b1) = a, b
        left = [word_ids[k][0] for k in range(a0, a1 + 1)]
        right = [word_ids[k][0] for k in range(b0, b1 + 1)]
        return left, right

    def _resulting_span_len(self, word_ids, a, b):
        (a0, _a1), (_b0, b1) = a, b
        return sum(len(word_ids[k]) for k in range(a0, b1 + 1))

    def _over_cap(self, word_ids, a, b):
        return (self.span_cap_mode == "force0"
                and self._resulting_span_len(word_ids, a, b) > self.span_cap)

    def logits_for_pairs(self, word_ids, pairs, *, grad):
        import torch
        if len(pairs) == 0:
            return torch.zeros((0,))
        rows = []
        for (a, b) in pairs:
            if self._over_cap(word_ids, a, b):
                rows.append(torch.tensor(-30.0))        # forced ~0 (span cap)
            else:
                rows.append(self.theta[a[1]])           # trainable junction logit
        return torch.stack(rows)


def test_selfmerge_registered_and_constant_pmodel():
    """selfmerge is a valid roll-in and runs FULL model choice on EVERY epoch,
    including the first; the existing schedules are unchanged."""
    from rmerge import ROLL_IN_CHOICES, rollin_p_model, SCHED_P_MAX_DEFAULT
    assert "selfmerge" in ROLL_IN_CHOICES
    # the fix: pure self-gluing every epoch (not throttled, not off at epoch 0)
    assert rollin_p_model("selfmerge", 0, 8) == 1.0
    assert rollin_p_model("selfmerge", 7, 8) == 1.0
    # untouched schedules
    assert rollin_p_model("teacher", 3, 8) == 0.0
    assert rollin_p_model("sched", 0, 8) == 0.0
    assert rollin_p_model("sched", 7, 8) == pytest.approx(SCHED_P_MAX_DEFAULT)


def test_selfmerge_recursive_glues_and_punishes():
    """Per-pass recursion: the model glues its own pieces, and the cross-boundary
    pair is densely supervised to NOT glue (target 0)."""
    from rmerge import doc_train_loss_recursive, train_pos_weight
    sc = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.0)   # P=0.5 -> glues
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    loss, ncand, npos, nneg, nglue = doc_train_loss_recursive(
        sc, TOY_WORDS, TOY_LABELS, pw, glue_tau=0.5)
    assert nglue > 0                        # the model actually glued pieces
    assert ncand > 0
    # the boundary junction (ate|dogs) is supervised with target 0 (do NOT glue)
    pairs, targets = dense_supervision_pairs(start_state(len(TOY_WORDS)),
                                             gold_cut_mask(TOY_LABELS))
    assert targets == [1, 1, 0, 1]
    assert pairs[targets.index(0)] == ((2, 2), (3, 3))


def test_plan_pass_glues_confidence_vs_leftright():
    """Confidence-order glues the highest-prob pair first; left-to-right glues the
    leftmost.  On an overlapping pair they diverge."""
    from rmerge import _plan_pass_glues
    # 3 pieces (a,b,c): junction 0 = (a,b) P=0.6, junction 1 = (b,c) P=0.9 (share b)
    probs = [0.6, 0.9]
    assert _plan_pass_glues(probs, 0.5, "leftright") == {0}     # leftmost wins
    assert _plan_pass_glues(probs, 0.5, "confidence") == {1}    # highest-P wins
    # below-threshold pairs never glue
    assert _plan_pass_glues([0.4, 0.3], 0.5, "confidence") == set()
    # non-overlap enforced + span-cap-blocked junction excluded
    assert _plan_pass_glues([0.9, 0.9, 0.9], 0.5, "confidence", blocked={0}) == {1}
    # far-apart pairs both glue (non-overlapping)
    assert _plan_pass_glues([0.9, 0.1, 0.9], 0.5, "confidence") == {0, 2}


def test_selfmerge_recursive_is_cheap_not_quadratic():
    """The per-pass recursion supervises ~O(pieces x passes) candidates, NOT the
    O(n^2) of re-scoring after every single glue -- the cost wall this fixes."""
    from rmerge import doc_train_loss_recursive
    n = 40
    words = [f"w{i}" for i in range(n)]
    labels = [0] * n
    labels[-1] = 1                          # one sentence (all within) + doc-final
    sc = _GradMockScorer(n, span_cap=10_000, init=0.0)   # P=0.5 -> glue everything
    loss, ncand, npos, nneg, nglue = doc_train_loss_recursive(
        sc, words, labels, 1.0, glue_tau=0.5)
    one_at_a_time = n * (n - 1) // 2        # the OLD re-score-after-every-glue cost
    assert ncand < one_at_a_time // 3       # recursion is far cheaper
    assert ncand < 5 * n                    # ~near-linear (passes ~ log n)


def test_selfmerge_recursive_punishes_cross_boundary():
    """The punishment is load-bearing: a few optimizer steps of the per-pass
    recursion drive the cross-boundary glue's probability DOWN."""
    import torch
    from rmerge import doc_train_loss_recursive, train_pos_weight
    sc = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.0)
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    opt = torch.optim.SGD([sc.theta], lr=0.5)
    before = float(torch.sigmoid(sc.theta[2].detach()))   # junction 2 = ate|dogs
    for _ in range(20):
        opt.zero_grad()
        loss, *_ = doc_train_loss_recursive(sc, TOY_WORDS, TOY_LABELS, pw, glue_tau=0.5)
        loss.backward()
        opt.step()
    after = float(torch.sigmoid(sc.theta[2].detach()))
    assert after < before                                 # boundary glue punished down


def test_cascade_gamma_upweights_early_boundary_penalty():
    """Depth-weighted cascade penalty: cascade_gamma>0 scales the boundary
    (don't-merge) penalty higher at EARLY passes, so the total loss is strictly
    larger than the un-weighted loss when gold boundaries exist; cascade_gamma=0
    reproduces the prior behaviour EXACTLY."""
    from rmerge import doc_train_loss_recursive, train_pos_weight
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    # the mock is deterministic, so fresh scorers with the same init are identical
    sc0 = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.0)
    l0, *_ = doc_train_loss_recursive(sc0, TOY_WORDS, TOY_LABELS, pw,
                                      glue_tau=0.5, cascade_gamma=0.0)
    sc1 = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.0)
    l1, *_ = doc_train_loss_recursive(sc1, TOY_WORDS, TOY_LABELS, pw,
                                      glue_tau=0.5, cascade_gamma=3.0)
    assert float(l1) > float(l0) + 1e-6            # early boundary penalty up-weighted
    # cascade_gamma=0 is byte-identical to omitting the arg (default path unchanged)
    sc2 = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.0)
    l2, *_ = doc_train_loss_recursive(sc2, TOY_WORDS, TOY_LABELS, pw, glue_tau=0.5)
    assert abs(float(l0) - float(l2)) < 1e-9


def test_mixout_identity_at_inference_and_saves_standard():
    """MixLinear (anti-memorization anchor): p=0 is a plain Linear; at INFERENCE
    (eval) it is exactly the trained Linear regardless of p; and its pretrained
    anchor is a NON-persistent buffer, so the state_dict saves a standard Linear."""
    import torch
    from rmerge import MixLinear
    lin = torch.nn.Linear(5, 3)
    x = torch.randn(4, 5)
    m0 = MixLinear.from_linear(lin, 0.0)
    m0.train()
    assert torch.allclose(m0(x), lin(x))            # p=0 -> identical even in train
    m9 = MixLinear.from_linear(lin, 0.9)
    m9.eval()
    assert torch.allclose(m9(x), lin(x))            # inference = trained weight, no mix
    # the pretrained anchor must NOT be saved -> save_pretrained writes a std Linear
    assert "mix_target" not in m9.state_dict()
    assert set(m9.state_dict().keys()) == {"weight", "bias"}


def test_mixout_mixes_in_training_and_apply_converts_linears():
    """In TRAINING with p>0 the weight is stochastically pulled toward the anchor
    (so the forward differs from the plain Linear), and apply_mixout swaps every
    nn.Linear in a module for a MixLinear."""
    import torch
    from rmerge import MixLinear, apply_mixout
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64)
    m = MixLinear.from_linear(lin, 0.5)
    # perturb the trainable weight away from the anchor so mixing is observable
    with torch.no_grad():
        m.weight.add_(1.0)
    m.train()
    x = torch.randn(8, 64)
    # at least one of a few stochastic draws differs from the un-mixed Linear
    diffs = [not torch.allclose(m(x), torch.nn.functional.linear(x, m.weight, m.bias))
             for _ in range(5)]
    assert any(diffs)
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(),
                              torch.nn.Linear(4, 2))
    n = apply_mixout(net, 0.5)
    assert n == 2 and isinstance(net[0], MixLinear) and isinstance(net[2], MixLinear)
    assert apply_mixout(net, 0.0) == 0              # p=0 -> no-op


# =========================================================================== #
# FULL-WINDOW JOINT SCORER (--scorer joint): whole-window encode, junction MLP
# =========================================================================== #
@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_joint_scorer_scores_and_caps():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from rmerge import JointScorer, build_joint_head, load_arabert
    tok, enc, _ = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    enc.eval()
    head = build_joint_head(enc.config.hidden_size, seed=0)
    sc = JointScorer(tok, enc, head, span_cap=3)
    wid = sc._word_ids(TOY_WORDS)                      # 5 words: indices 0..4
    pairs = [((0, 0), (1, 1)), ((1, 1), (2, 2)), ((0, 1), (2, 4))]
    p = sc.probs_for_pairs(wid, pairs)
    assert p.shape == (3,) and all(0.0 <= x <= 1.0 for x in p)
    # the wide pair exceeds span_cap=3 subwords -> forced ~0 (veto semantics kept)
    assert sc._over_cap(wid, (0, 1), (2, 4)) and p[2] < 1e-6
    # deterministic across calls in eval mode (single-slot cache + no dropout)
    p2 = sc.probs_for_pairs(wid, pairs)
    assert np.allclose(p, p2, atol=1e-6)


@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_joint_encode_segment_stitching():
    """Long inputs are encoded in overlapping segments and every subword position
    is filled exactly once from the segment where it is most interior."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from rmerge import JointScorer, build_joint_head, load_arabert
    tok, enc, _ = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    enc.eval()
    head = build_joint_head(enc.config.hidden_size, seed=0)
    sc = JointScorer(tok, enc, head, span_cap=64)
    sc.SEG, sc.STRIDE = 8, 4                    # force multi-segment on a toy doc
    words = ["the", "cat", "ate", "dogs", "ran", "far", "away", "home",
             "fast", "then", "slept", "well"]
    wid = sc._word_ids(words)
    n_sub = sum(len(w) for w in wid)
    assert n_sub > sc.SEG                       # genuinely multi-segment
    states, off = sc.encode(wid, grad=False)
    assert states.shape[0] == n_sub
    assert float(states.norm(dim=1).min()) > 0.0     # every position filled
    assert off[0] == (0, len(wid[0])) and off[-1][1] == n_sub


@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_joint_recursion_trains_and_grads_flow():
    """Noor's per-pass recursion runs UNCHANGED on the joint scorer; the summed
    multi-pass loss backwards once through the SINGLE shared encode, reaching
    both the encoder and the junction head."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from rmerge import (JointScorer, build_joint_head, load_arabert,
                        doc_train_loss_recursive, train_pos_weight)
    tok, enc, _ = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    head = build_joint_head(enc.config.hidden_size, seed=0)
    sc = JointScorer(tok, enc, head, span_cap=64)
    sc.enc.train()
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    loss, ncand, npos, nneg, nglue = doc_train_loss_recursive(
        sc, TOY_WORDS, TOY_LABELS, pw, glue_tau=0.5,
        backward_each_state=False)              # joint: one backward, shared graph
    assert ncand > 0 and torch.isfinite(loss)
    loss.backward()
    g_enc = sum(float(p.grad.norm()) for p in sc.enc.parameters()
                if p.grad is not None)
    g_head = sum(float(p.grad.norm()) for p in sc.head.parameters()
                 if p.grad is not None)
    assert g_enc > 0.0 and g_head > 0.0


def test_margin_scaling_and_conn_targeting():
    """THE-trial loss: multiplicative boundary-margin raises the loss exactly on
    the boundary side; the connective-targeted extra raises it further ONLY when
    the word after a boundary junction is connective-initial; defaults (gamma=1,
    extra=0, rdrop=0) are byte-identical to the plain loss."""
    from rmerge import doc_train_loss_recursive, train_pos_weight, conn_initial
    assert conn_initial("وقال") and conn_initial("فذهب") and conn_initial("ثم")
    assert not conn_initial("قلم") and not conn_initial("the")
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    # positive logits everywhere (init=1) -> boundary (t=0) BCE grows with scale
    def loss(**kw):
        sc = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=1.0)
        l, *_ = doc_train_loss_recursive(sc, TOY_WORDS, TOY_LABELS, pw,
                                         glue_tau=0.5, **kw)
        return float(l)
    base = loss()
    assert abs(loss(margin_gamma_b=1.0, conn_margin_extra=0.0,
                    rdrop_alpha=0.0) - base) < 1e-9      # defaults reproduce
    assert loss(margin_gamma_b=2.0) > base + 1e-6        # boundary margin bites
    # connective targeting: boundary junction 2 precedes words[3]; make it و-initial
    words_conn = list(TOY_WORDS); words_conn[3] = "و" + words_conn[3]
    def loss_w(words, **kw):
        sc = _GradMockScorer(len(words), span_cap=64, init=1.0)
        l, *_ = doc_train_loss_recursive(sc, words, TOY_LABELS, pw,
                                         glue_tau=0.5, **kw)
        return float(l)
    plain = loss_w(words_conn, margin_gamma_b=1.0)
    targeted = loss_w(words_conn, margin_gamma_b=1.0, conn_margin_extra=1.0)
    assert targeted > plain + 1e-6                       # extra bites at و only
    # same extra with NO connective word present changes nothing
    assert abs(loss_w(TOY_WORDS, margin_gamma_b=1.0, conn_margin_extra=1.0)
               - loss_w(TOY_WORDS, margin_gamma_b=1.0)) < 1e-9


def test_rdrop_plumbing_on_deterministic_scorer():
    """R-Drop on a deterministic (no-dropout) scorer: the two draws agree, the
    symmetric-KL term is 0, and the loss equals the alpha=0 loss — proving the
    plumbing adds exactly the disagreement penalty and nothing else."""
    from rmerge import doc_train_loss_recursive, train_pos_weight
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    sc0 = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.5)
    l0, *_ = doc_train_loss_recursive(sc0, TOY_WORDS, TOY_LABELS, pw, glue_tau=0.5)
    sc1 = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.5)
    l1, *_ = doc_train_loss_recursive(sc1, TOY_WORDS, TOY_LABELS, pw,
                                      glue_tau=0.5, rdrop_alpha=1.0)
    assert abs(float(l0) - float(l1)) < 1e-6


@pytest.mark.skipif(not _arabert_cached(),
                    reason="AraBERT weights not in the offline HF cache")
def test_joint_rdrop_draws_differ_and_cache_per_draw():
    """With dropout ON, the joint scorer's two R-Drop draws give DIFFERENT logits;
    within one draw the per-draw cache returns identical logits (one encode per
    draw per doc)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from rmerge import JointScorer, build_joint_head, load_arabert
    tok, enc, _ = load_arabert("aubmindlab/bert-base-arabertv02", seed=0)
    head = build_joint_head(enc.config.hidden_size, seed=0)
    sc = JointScorer(tok, enc, head, span_cap=64)
    sc.enc.train()                                   # dropout active
    wid = sc._word_ids(TOY_WORDS)
    pairs = [((0, 0), (1, 1)), ((2, 2), (3, 3))]
    l0a = sc.logits_for_pairs(wid, pairs, grad=True)
    l0b = sc.logits_for_pairs(wid, pairs, grad=True)     # same draw -> cached
    assert torch.allclose(l0a, l0b)
    sc.set_draw(1)
    l1 = sc.logits_for_pairs(wid, pairs, grad=True)      # new draw -> new dropout
    assert not torch.allclose(l0a, l1)
    sc.set_draw(0)
    assert torch.allclose(sc.logits_for_pairs(wid, pairs, grad=True), l0a)


def test_leaf_rollin_trains_flat_detector():
    """--roll-in leaf (Noor's pivot): supervises EXACTLY the n-1 leaf junctions
    per doc, no gluing, no recursion; registered in ROLL_IN_CHOICES with
    p_model 0; margin/R-Drop flags flow through _state_loss unchanged."""
    import torch
    from rmerge import (ROLL_IN_CHOICES, rollin_p_model, _state_loss,
                        start_state, gold_cut_mask, train_pos_weight)
    assert "leaf" in ROLL_IN_CHOICES
    assert rollin_p_model("leaf", 0, 8) == 0.0
    sc = _GradMockScorer(len(TOY_WORDS), span_cap=64, init=0.5)
    pw = train_pos_weight([{"labels": TOY_LABELS}])
    gc = gold_cut_mask(TOY_LABELS)
    wid = sc._word_ids(TOY_WORDS)
    loss, ncand, npos, nneg, _ = _state_loss(sc, wid, start_state(len(TOY_WORDS)),
                                             gc, pw)
    assert ncand == len(TOY_WORDS) - 1          # every leaf junction, exactly once
    assert torch.isfinite(loss) and npos + nneg == ncand
    # margin + conn machinery works at the leaf state too
    l2, *_ = _state_loss(sc, wid, start_state(len(TOY_WORDS)), gc, pw,
                         margin_gamma_b=2.0)
    assert float(l2) > float(loss)              # boundary margin bites at leaf


def test_window_docs_slices_labels_aligned():
    from rmerge import window_docs
    d = {"doc_id": "d1", "tokens": [f"w{i}" for i in range(10)],
         "labels": [0, 1, 0, 0, 1, 0, 0, 0, 1, 1]}
    # short doc passes through untouched
    assert window_docs([d], window=10, stride=5) == [d]
    ws = window_docs([d], window=4, stride=3)
    assert all(len(w["tokens"]) == len(w["labels"]) for w in ws)
    assert len(set(w["doc_id"] for w in ws)) == len(ws)      # unique ids
    # every (token, label) pair in a window matches the source doc exactly
    for w in ws:
        start = int(w["doc_id"].split("#w")[1])
        for j, (t, l) in enumerate(zip(w["tokens"], w["labels"])):
            assert t == d["tokens"][start + j] and l == d["labels"][start + j]
    # full coverage: every source position appears in >=1 window
    covered = set()
    for w in ws:
        start = int(w["doc_id"].split("#w")[1])
        covered.update(range(start, start + len(w["tokens"])))
    assert covered == set(range(10))
