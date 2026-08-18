"""BoundRL data formatting for AraSeg sentence segmentation (single label class).

Implements Section 4.1 of "BoundRL: Efficient Token-level Structured Text
Segmentation through Reinforced Boundary Generation" (arXiv:2510.20151v2),
adapted per the approved pilot ruling of 2026-07-10:

  * single label class ("sentence") -> no [category] tags in the output;
  * the target for a document is the per-sentence starting-word sequences
    joined by " [SEP] ";
  * starting-sequence length is sampled uniformly in [2, 10] words (paper:
    "randomly sampled between 2 and 10 tokens"), capped at the full sentence
    length, then EXTENDED until the sequence occurs exactly once in the
    document (word-level occurrence) or the full sentence is used.
    NOTE: the paper's extension criterion is weaker (extend until the start
    sequences of different segments become pairwise different); the pilot
    ruling says "extended until unique within the doc", which is strictly
    stronger and is what is implemented here.  残 remaining ambiguity (a full
    sentence that appears several times verbatim) is handled by the paper's
    iterative leftmost-match reconstruction with its ordering constraint.

Word/token granularity: AraSeg documents are whitespace-tokenized and
`text == " ".join(tokens)` holds for every document of every track (verified
on all 4 tracks x 174 train docs on 2026-07-10).  "Tokens" in the paper's
Sec. 4.1 sense are therefore whitespace words here; model-tokenizer tokens
are only used for the windowing budget.

Label convention (src/data.py): labels[i] == 1  <=>  a sentence boundary
FOLLOWS token i.  The last token of every document carries label 1.
Segment starts are therefore 0 plus {i+1 : labels[i] == 1, i < n-1}.

Windowing (documented choice): documents up to 18,267 words exist; they are
split into chunks whose *document text* measures at most --budget (default
2048) model-tokenizer tokens, with splits allowed only at gold sentence
boundaries (sentence-aligned).  A single sentence longer than the budget
becomes its own over-budget chunk (never split mid-sentence; such chunks are
counted and reported).  When no tokenizer is given, tokens are estimated as
ceil(TOKENS_PER_WORD_EST * words); the real pilot passes the actual model
tokenizer.  Each chunk is an independent training example: start sampling and
the uniqueness extension are performed within the chunk.

Round-trip self-test (CLI): format -> parse -> iterative leftmost-match
reconstruct -> boundary labels must reproduce the gold segmentation exactly
on all 174 train docs; any doc where the start tokens cannot disambiguate is
reported.

Usage:
  python src/boundrl_data.py --selftest --train-jsonl data/NoPnx-NP_train.jsonl --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Callable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_jsonl  # noqa: E402  (repo-local, load-bearing, reused not copied)

SEP = "[SEP]"
SEP_JOIN = " [SEP] "
MIN_START_LEN = 2
MAX_START_LEN = 10
TOKENS_PER_WORD_EST = 2.3  # rough Arabic word -> subword ratio; documented estimate only

# Meta instruction, adapted from the paper's Fig. 5 to a single "sentence"
# class and the pilot's [SEP]-joined output format.
META_PROMPT = (
    "You are requested to segment an Arabic document into sentences.\n"
    "Output only the starting words of each sentence, in the order the "
    "sentences appear, separated by \" [SEP] \".\n"
    "Copy the starting words of each sentence exactly as they appear in the "
    "document.\n"
    "Document:\n{document}\n"
    "Segmentation:\n"
)


# --------------------------------------------------------------------------
# gold segmentation <-> spans <-> labels
# --------------------------------------------------------------------------

def gold_segment_starts(labels: Sequence[int]) -> List[int]:
    """Word indices at which gold sentences start (always includes 0)."""
    starts = [0]
    for i, lab in enumerate(labels[:-1]):  # final label closes the doc, opens nothing
        if lab == 1:
            starts.append(i + 1)
    return starts


def spans_from_starts(starts: Sequence[int], n_words: int) -> List[Tuple[int, int]]:
    """Half-open word spans [(b, e), ...] from sorted start indices."""
    spans = []
    for j, b in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else n_words
        spans.append((b, e))
    return spans


def labels_from_spans(spans: Sequence[Tuple[int, int]], n_words: int) -> List[int]:
    """Boundary labels from word spans: label[e-1] = 1 for every span end e.

    With full-coverage spans this reproduces the gold convention, including
    label 1 on the final word (last span ends at n_words).
    """
    labels = [0] * n_words
    for _, e in spans:
        if 0 < e <= n_words:
            labels[e - 1] = 1
    return labels


# --------------------------------------------------------------------------
# start-sequence sampling (Sec 4.1)
# --------------------------------------------------------------------------

def count_occurrences(words: Sequence[str], seq: Sequence[str]) -> int:
    """Number of word-level occurrences of `seq` in `words` (overlaps count)."""
    L = len(seq)
    if L == 0 or L > len(words):
        return 0
    first = seq[0]
    tup = tuple(seq)
    return sum(
        1
        for i in range(len(words) - L + 1)
        if words[i] == first and tuple(words[i : i + L]) == tup
    )


def sample_start_sequences(
    words: Sequence[str],
    starts: Sequence[int],
    rng: random.Random,
    n_words: Optional[int] = None,
) -> List[List[str]]:
    """Sample one starting-word sequence per segment (Sec 4.1).

    Length ~ U[2, 10] capped at segment length; extended word-by-word until
    the sequence occurs exactly once in `words` or the full segment is used.
    """
    n = n_words if n_words is not None else len(words)
    spans = spans_from_starts(starts, n)
    seqs: List[List[str]] = []
    for b, e in spans:
        seg_len = e - b
        length = min(rng.randint(MIN_START_LEN, MAX_START_LEN), seg_len)
        while length < seg_len and count_occurrences(words, words[b : b + length]) > 1:
            length += 1
        seqs.append(list(words[b : b + length]))
    return seqs


# --------------------------------------------------------------------------
# target formatting / parsing
# --------------------------------------------------------------------------

def format_target(start_seqs: Sequence[Sequence[str]]) -> str:
    return SEP_JOIN.join(" ".join(seq) for seq in start_seqs)


def parse_target(text: str) -> List[List[str]]:
    """Inverse of format_target; tolerant of stray whitespace and empty parts."""
    parts = [p.strip() for p in text.split(SEP)]
    return [p.split() for p in parts if p]


def build_prompt(doc_text: str) -> str:
    return META_PROMPT.format(document=doc_text)


# --------------------------------------------------------------------------
# iterative leftmost-match reconstruction (Sec 4.1, ordering constraint)
# --------------------------------------------------------------------------

def find_subsequence(words: Sequence[str], seq: Sequence[str], from_idx: int) -> Optional[int]:
    """Leftmost word index >= from_idx where `seq` occurs, else None."""
    L = len(seq)
    if L == 0:
        return None
    first = seq[0]
    tup = tuple(seq)
    for i in range(max(from_idx, 0), len(words) - L + 1):
        if words[i] == first and tuple(words[i : i + L]) == tup:
            return i
    return None


def locate_starts(words: Sequence[str], start_seqs: Sequence[Sequence[str]]) -> List[Optional[int]]:
    """Iterative leftmost-match localisation with the paper's ordering
    constraint ("each match must occur later in the input text than the
    previous one"): each sequence is located at its leftmost occurrence at or
    after the END of the previously located start sequence (prev_pos +
    len(prev_seq)).  Rationale: a segment is at least as long as its start
    sequence, so the true next start can never precede that point; searching
    from prev_pos + 1 instead would let repeated phrases that straddle a
    boundary match inside the previous segment's start tokens (observed on
    train docs with repeated song-chorus lines).  An unlocatable sequence
    yields None and does not advance the search position."""
    positions: List[Optional[int]] = []
    search_from = 0
    for seq in start_seqs:
        pos = find_subsequence(words, seq, search_from)
        positions.append(pos)
        if pos is not None:
            search_from = pos + len(seq)
    return positions


def reconstruct(
    words: Sequence[str], start_seqs: Sequence[Sequence[str]]
) -> Tuple[List[Optional[Tuple[int, int]]], int]:
    """Reconstruct segment word spans from starting sequences (Sec 4.1).

    Segment i spans from the position of its start to the position of the
    next segment's start (last segment: to the end of the document).  If the
    position of s_i or s_{i+1} cannot be found, segment i is discarded (span
    None), exactly as in the paper.

    Returns (spans, n_dropped); spans is aligned with start_seqs.
    """
    positions = locate_starts(words, start_seqs)
    n = len(start_seqs)
    spans: List[Optional[Tuple[int, int]]] = [None] * n
    dropped = 0
    for i in range(n):
        pos = positions[i]
        if pos is None:
            dropped += 1
            continue
        if i + 1 < n:
            nxt = positions[i + 1]
            if nxt is None:
                dropped += 1
                continue
            spans[i] = (pos, nxt)
        else:
            spans[i] = (pos, len(words))
    return spans, dropped


# --------------------------------------------------------------------------
# sentence-aligned windowing
# --------------------------------------------------------------------------

def estimate_count_tokens(text: str) -> int:
    return int(math.ceil(TOKENS_PER_WORD_EST * max(len(text.split()), 1)))


def chunk_doc(
    words: Sequence[str],
    labels: Sequence[int],
    count_tokens: Callable[[str], int] = estimate_count_tokens,
    budget: int = 2048,
) -> List[dict]:
    """Split a document into sentence-aligned chunks of <= budget model tokens.

    Greedy packing of whole gold sentences; a single over-budget sentence
    becomes its own chunk (flagged 'over_budget').  Chunk-local labels are a
    slice of the gold labels; because splits are sentence-aligned the final
    label of every chunk is 1.
    """
    n = len(words)
    sent_spans = spans_from_starts(gold_segment_starts(labels), n)
    chunks: List[dict] = []
    cur: List[Tuple[int, int]] = []

    def flush():
        if not cur:
            return
        b, e = cur[0][0], cur[-1][1]
        c_words = list(words[b:e])
        chunks.append(
            {
                "word_offset": b,
                "words": c_words,
                "labels": list(labels[b:e]),
                "n_sentences": len(cur),
                "over_budget": count_tokens(" ".join(c_words)) > budget,
            }
        )
        cur.clear()

    cur_tokens = 0
    for b, e in sent_spans:
        sent_text = " ".join(words[b:e])
        t = count_tokens(sent_text)
        if cur and cur_tokens + t > budget:
            flush()
            cur_tokens = 0
        cur.append((b, e))
        cur_tokens += t
        if cur_tokens > budget:  # single over-budget sentence
            flush()
            cur_tokens = 0
    flush()
    return chunks


# --------------------------------------------------------------------------
# SFT / GRPO example construction
# --------------------------------------------------------------------------

def build_examples(
    docs: List[dict],
    seed: int = 0,
    count_tokens: Callable[[str], int] = estimate_count_tokens,
    budget: int = 2048,
    max_docs: Optional[int] = None,
) -> List[dict]:
    """One example per chunk: prompt, completion, plus everything the reward
    function needs (chunk words and gold word spans)."""
    rng = random.Random(seed)
    examples: List[dict] = []
    for doc in docs[: max_docs if max_docs else len(docs)]:
        words, labels = doc["tokens"], doc["labels"]
        for ci, ch in enumerate(chunk_doc(words, labels, count_tokens, budget)):
            c_words, c_labels = ch["words"], ch["labels"]
            starts = gold_segment_starts(c_labels)
            seqs = sample_start_sequences(c_words, starts, rng)
            examples.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_id": f"{doc['doc_id']}#c{ci}",
                    "word_offset": ch["word_offset"],
                    "over_budget": ch["over_budget"],
                    "words": c_words,
                    "gold_starts": starts,
                    "gold_spans": spans_from_starts(starts, len(c_words)),
                    "prompt": build_prompt(" ".join(c_words)),
                    "completion": format_target(seqs),
                }
            )
    return examples


def make_count_tokens(tokenizer_name: Optional[str]) -> Callable[[str], int]:
    if not tokenizer_name:
        return estimate_count_tokens
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda s: len(tok(s, add_special_tokens=False).input_ids)


# --------------------------------------------------------------------------
# round-trip self-test
# --------------------------------------------------------------------------

def roundtrip_selftest(
    docs: List[dict],
    seed: int = 0,
    count_tokens: Callable[[str], int] = estimate_count_tokens,
    budget: int = 2048,
) -> dict:
    """format -> parse -> reconstruct must reproduce gold labels exactly.

    Docs where the start tokens provably CANNOT disambiguate are reported
    separately (pilot ruling): a mislocated segment counts as inherent
    ambiguity iff its start sequence already uses the FULL gold segment text
    (no further extension possible per Sec 4.1) and that text occurs more
    than once in the chunk.  Anything else is an unexpected failure (a bug).
    """
    failures, cannot_disambiguate, ambiguous, n_chunks, n_over = [], [], [], 0, 0
    examples = build_examples(docs, seed=seed, count_tokens=count_tokens, budget=budget)
    by_doc: dict = {}
    for ex in examples:
        by_doc.setdefault(ex["doc_id"], []).append(ex)
        n_chunks += 1
        n_over += int(ex["over_budget"])
    doc_by_id = {d["doc_id"]: d for d in docs}
    for doc_id, exs in by_doc.items():
        gold = doc_by_id[doc_id]["labels"]
        pred = [0] * len(gold)
        ok = True
        inherent = True  # every mismatch (if any) is provably unavoidable
        for ex in exs:
            seqs = parse_target(ex["completion"])
            if len(seqs) != len(ex["gold_starts"]):
                ok = False
                inherent = False
            spans, dropped = reconstruct(ex["words"], seqs)
            if dropped:
                ok = False
                inherent = False
            positions = locate_starts(ex["words"], seqs)
            for i, (pos, gb) in enumerate(zip(positions, ex["gold_starts"])):
                if pos == gb:
                    continue
                ok = False
                b, e = ex["gold_spans"][i]
                full_seg = len(seqs[i]) == e - b
                multi = count_occurrences(ex["words"], seqs[i]) > 1
                if not (full_seg and multi):
                    inherent = False
            # non-unique starts (even after extension) in this chunk?
            for s in seqs:
                if count_occurrences(ex["words"], s) > 1:
                    ambiguous.append(ex["chunk_id"])
                    break
            kept = [sp for sp in spans if sp is not None]
            for b, e in kept:
                off = ex["word_offset"]
                lab = labels_from_spans([(b, e)], len(ex["words"]))
                for i, v in enumerate(lab):
                    if v:
                        pred[off + i] = 1
        if pred != list(gold) or not ok:
            diff = [i for i, (a, b) in enumerate(zip(pred, gold)) if a != b]
            rec = {"doc_id": doc_id, "first_diffs": diff[:5], "n_diffs": len(diff)}
            if inherent:
                cannot_disambiguate.append(rec)
            else:
                failures.append(rec)
    return {
        "n_docs": len(docs),
        "n_chunks": n_chunks,
        "n_over_budget_chunks": n_over,
        "n_chunks_with_nonunique_start": len(set(ambiguous)),
        "nonunique_chunk_ids": sorted(set(ambiguous)),
        "n_cannot_disambiguate": len(cannot_disambiguate),
        "cannot_disambiguate": cannot_disambiguate,
        "n_failures": len(failures),
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for the window budget (default: word-count estimate)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-examples", default=None, help="write formatted examples to this JSONL")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    docs = load_jsonl(args.train_jsonl)
    if args.max_docs:
        docs = docs[: args.max_docs]
    count_tokens = make_count_tokens(args.tokenizer)

    if args.dump_examples:
        examples = build_examples(docs, seed=args.seed, count_tokens=count_tokens, budget=args.budget)
        with open(args.dump_examples, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote {len(examples)} examples -> {args.dump_examples}")

    if args.selftest:
        rep = roundtrip_selftest(docs, seed=args.seed, count_tokens=count_tokens, budget=args.budget)
        print(json.dumps({k: v for k, v in rep.items() if k != "nonunique_chunk_ids"}, ensure_ascii=False, indent=2))
        n_amb = rep["n_cannot_disambiguate"]
        if rep["n_failures"] == 0 and n_amb == 0:
            print(f"ROUND-TRIP OK: gold segmentation reproduced exactly on all {rep['n_docs']} docs.")
        elif rep["n_failures"] == 0:
            ids = ", ".join(r["doc_id"] for r in rep["cannot_disambiguate"])
            print(
                f"ROUND-TRIP OK on {rep['n_docs'] - n_amb}/{rep['n_docs']} docs; "
                f"{n_amb} doc(s) REPORTED where start tokens provably cannot "
                f"disambiguate (full-segment start text repeats in the doc): {ids}"
            )
        else:
            print(f"ROUND-TRIP FAILED on {rep['n_failures']} docs (see above).")
            sys.exit(1)


if __name__ == "__main__":
    main()
