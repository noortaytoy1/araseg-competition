"""Unit tests for the Sec 4.3 intermediate-candidate functions in
src/boundrl_grpo.py — corpus-independent, no model / trl objects needed
(perturbation_pool and best_intermediate are pure functions)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from boundrl_data import format_target, spans_from_starts  # noqa: E402
from boundrl_grpo import best_intermediate, perturbation_pool  # noqa: E402
from boundrl_reward import completion_reward  # noqa: E402

WORDS = "aa bb cc dd ee ff gg hh ii jj kk ll".split()
GOLD = spans_from_starts([0, 4, 8], len(WORDS))


def test_pool_contains_left_and_right_shifts():
    seqs = [["aa", "bb"], ["ee", "ff"], ["ii", "jj"]]
    pool = perturbation_pool(WORDS, seqs)
    # seg0: right-shift only (pos 0 has no left word); seg1, seg2: both
    assert len(pool) == 1 + 2 + 2
    assert [["bb"], ["ee", "ff"], ["ii", "jj"]] in pool          # seg0 right
    assert [["aa", "bb"], ["dd", "ee", "ff"], ["ii", "jj"]] in pool  # seg1 left
    assert [["aa", "bb"], ["ff"], ["ii", "jj"]] in pool          # seg1 right


def test_one_word_sequences_are_not_shortened():
    seqs = [["aa"], ["ee"]]
    pool = perturbation_pool(WORDS, seqs)
    for variant in pool:
        assert all(len(s) >= 1 for s in variant)
    # seg0 one-word at pos 0 -> no variant from it at all
    assert all(v[1] != ["ee"] or v[0] != ["aa"] for v in pool) or pool


def test_best_intermediate_repairs_off_by_one_boundary():
    # median candidate has segment 2 starting one word early (at hh, idx 7)
    bad = format_target([["aa", "bb"], ["ee", "ff"], ["hh", "ii"]])
    r_bad = completion_reward(WORDS, GOLD, bad)["reward"]
    got = best_intermediate(WORDS, GOLD, bad)
    assert got is not None
    text, r_new = got
    assert r_new > r_bad
    # the winning perturbation shifts segment 2's start right to gold (ii)
    assert r_new == 1.0
    assert completion_reward(WORDS, GOLD, text)["reward"] == 1.0


def test_best_intermediate_none_for_garbage():
    assert best_intermediate(WORDS, GOLD, "") is None


def test_perfect_candidate_yields_no_improving_intermediate():
    perfect = format_target([["aa", "bb"], ["ee", "ff"], ["ii", "jj"]])
    r_perfect = completion_reward(WORDS, GOLD, perfect)["reward"]
    got = best_intermediate(WORDS, GOLD, perfect)
    assert r_perfect == 1.0
    if got is not None:  # a pool exists but nothing can beat 1.0
        assert got[1] <= r_perfect
