"""Unit tests for src/boundrl_reward.py — corpus-independent (pass with data/ hidden).

Required behaviours (pilot ruling 2026-07-10):
  * perfect prediction        -> r == 1
  * off-by-one-word starts    -> partial credit (0 < r < 1)
  * unlocatable starts        -> segment dropped, rho_rec falls
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from boundrl_data import format_target, spans_from_starts  # noqa: E402
from boundrl_reward import (  # noqa: E402
    char_f1,
    completion_reward,
    exact_match_f1,
    reconstruction_ratio,
    segmentation_reward,
)

# Synthetic "document": 12 words, 3 gold sentences of 4 words each.
WORDS = "aa bb cc dd ee ff gg hh ii jj kk ll".split()
GOLD = spans_from_starts([0, 4, 8], len(WORDS))  # [(0,4),(4,8),(8,12)]


def test_perfect_prediction_reward_is_one():
    m = segmentation_reward(WORDS, GOLD, GOLD)
    assert m["rho_rec"] == 1.0
    assert m["em"] == 1.0
    assert m["f1char"] == 1.0
    assert m["reward"] == 1.0


def test_perfect_prediction_via_completion_text():
    completion = format_target([WORDS[0:2], WORDS[4:6], WORDS[8:10]])
    m = completion_reward(WORDS, GOLD, completion)
    assert m["reward"] == 1.0
    assert m["n_dropped"] == 0


def test_off_by_one_word_start_gets_partial_credit():
    # second sentence's start shifted one word right: boundary at 5 not 4
    pred = [(0, 5), (5, 8), (8, 12)]
    m = segmentation_reward(WORDS, GOLD, pred)
    assert m["rho_rec"] == 1.0          # still a full tiling
    assert 0.0 < m["em"] < 1.0          # two segments no longer exact
    assert 0.0 < m["f1char"] < 1.0      # only chars of one word mislabelled
    assert 0.0 < m["reward"] < 1.0
    # char F1 must give MORE credit than exact match (soft vs strict)
    assert m["f1char"] > m["em"]


def test_unlocatable_start_drops_segment_and_rho_falls():
    completion = format_target([WORDS[0:2], ["zz", "qq"], WORDS[8:10]])
    m = completion_reward(WORDS, GOLD, completion)
    # middle start unlocatable: paper discards segment i (and the segment
    # whose end it defines), so reconstruction is no longer complete
    assert m["n_dropped"] >= 1
    assert m["rho_rec"] < 1.0
    assert m["reward"] < 1.0


def test_all_unlocatable_reward_zero():
    m = completion_reward(WORDS, GOLD, "xx yy [SEP] zz qq")
    assert m["reward"] == 0.0


def test_empty_completion_reward_zero():
    m = completion_reward(WORDS, GOLD, "")
    assert m["reward"] == 0.0


def test_rho_counts_separator_chars_between_adjacent_recovered_words():
    # only first gold sentence recovered: 4 words of 2 chars + 3 inner seps
    # = 11 chars of |d| = 12*2 + 11 = 35
    m = reconstruction_ratio(WORDS, [(0, 4)])
    assert abs(m - 11 / 35) < 1e-12


def test_em_multiset_matching_handles_duplicate_texts():
    words = "aa bb aa bb cc dd".split()
    gold = [(0, 2), (2, 4), (4, 6)]  # "aa bb", "aa bb", "cc dd"
    pred = [(0, 2), (2, 4), (4, 6)]
    assert exact_match_f1(words, gold, pred) == 1.0
    # only one of the two duplicate segments produced
    pred2 = [(0, 4), (4, 6)]  # "aa bb aa bb" (no match), "cc dd"
    em2 = exact_match_f1(words, gold, pred2)
    assert 0.0 < em2 < 1.0


def test_char_f1_alignment_ties_break_to_earlier_gold():
    words = "aa bb cc dd".split()
    gold = [(0, 2), (2, 4)]
    pred = [(1, 3)]  # overlaps both gold segments equally (one word each)
    f = char_f1(words, gold, pred)
    assert 0.0 < f < 1.0


def test_reward_monotone_in_dropped_segments():
    full = segmentation_reward(WORDS, GOLD, GOLD)["reward"]
    partial = segmentation_reward(WORDS, GOLD, [GOLD[0], None, GOLD[2]])["reward"]
    assert partial < full
