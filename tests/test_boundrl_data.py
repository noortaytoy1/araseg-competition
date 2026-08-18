"""Unit tests for src/boundrl_data.py — corpus-independent (pass with data/ hidden)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from boundrl_data import (  # noqa: E402
    MAX_START_LEN,
    MIN_START_LEN,
    build_examples,
    chunk_doc,
    count_occurrences,
    format_target,
    gold_segment_starts,
    labels_from_spans,
    locate_starts,
    parse_target,
    reconstruct,
    roundtrip_selftest,
    sample_start_sequences,
    spans_from_starts,
)


def make_doc(n_sent=5, sent_len=6, prefix="w"):
    words, labels = [], []
    for s in range(n_sent):
        for i in range(sent_len):
            words.append(f"{prefix}{s}_{i}")
            labels.append(0)
        labels[-1] = 1
    return words, labels


def test_gold_starts_and_labels_roundtrip():
    words, labels = make_doc()
    starts = gold_segment_starts(labels)
    assert starts == [0, 6, 12, 18, 24]
    spans = spans_from_starts(starts, len(words))
    assert labels_from_spans(spans, len(words)) == labels


def test_final_label_produces_no_extra_start():
    labels = [0, 0, 1, 0, 1]  # boundary after idx2 and doc-end
    assert gold_segment_starts(labels) == [0, 3]


def test_sampled_length_in_range_and_capped():
    words, labels = make_doc(n_sent=3, sent_len=3)
    rng = random.Random(0)
    seqs = sample_start_sequences(words, gold_segment_starts(labels), rng)
    for seq in seqs:
        assert MIN_START_LEN <= len(seq) <= 3  # capped at sentence length


def test_extension_until_unique_within_doc():
    # two sentences sharing a long identical prefix force extension past 2
    words = "a b c d X a b c d Y".split()
    labels = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    rng = random.Random(1)
    seqs = sample_start_sequences(words, gold_segment_starts(labels), rng)
    for seq in seqs:
        assert count_occurrences(words, seq) == 1


def test_format_parse_roundtrip():
    seqs = [["aa", "bb"], ["cc"], ["dd", "ee", "ff"]]
    assert parse_target(format_target(seqs)) == seqs


def test_iterative_leftmost_ordering_disambiguates_duplicates():
    # identical start sequence for sentences 1 and 2: ordering constraint
    words = "a b c a b c".split()
    positions = locate_starts(words, [["a", "b"], ["a", "b"]])
    assert positions == [0, 3]


def test_reconstruct_perfect():
    words, labels = make_doc()
    starts = gold_segment_starts(labels)
    seqs = sample_start_sequences(words, starts, random.Random(2))
    spans, dropped = reconstruct(words, seqs)
    assert dropped == 0
    assert spans == spans_from_starts(starts, len(words))


def test_reconstruct_drops_unlocatable_and_its_predecessor():
    words = "a b c d e f".split()
    spans, dropped = reconstruct(words, [["a", "b"], ["zz"], ["e", "f"]])
    # segment 0 discarded (its end s_1 unlocatable), segment 1 discarded
    assert spans[0] is None and spans[1] is None
    assert spans[2] == (4, 6)
    assert dropped == 2


def test_chunking_is_sentence_aligned_and_within_budget():
    words, labels = make_doc(n_sent=10, sent_len=8)
    count = lambda s: len(s.split())  # 1 token per word for the test
    chunks = chunk_doc(words, labels, count, budget=20)
    assert sum(len(c["words"]) for c in chunks) == len(words)
    off = 0
    for c in chunks:
        assert c["word_offset"] == off
        off += len(c["words"])
        assert c["labels"][-1] == 1  # sentence-aligned split
        assert not c["over_budget"]
        assert count(" ".join(c["words"])) <= 20


def test_oversized_single_sentence_becomes_own_chunk():
    words, labels = make_doc(n_sent=1, sent_len=30)
    count = lambda s: len(s.split())
    chunks = chunk_doc(words, labels, count, budget=10)
    assert len(chunks) == 1 and chunks[0]["over_budget"]


def test_build_examples_and_synthetic_roundtrip():
    docs = []
    for d in range(3):
        words, labels = make_doc(n_sent=4, sent_len=5, prefix=f"d{d}w")
        docs.append({"doc_id": f"doc{d}", "tokens": words, "labels": labels})
    count = lambda s: len(s.split())
    exs = build_examples(docs, seed=0, count_tokens=count, budget=12)
    assert all(ex["prompt"].endswith("Segmentation:\n") for ex in exs)
    rep = roundtrip_selftest(docs, seed=0, count_tokens=count, budget=12)
    assert rep["n_failures"] == 0
