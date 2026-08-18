"""BoundRL reward (Sec 4.2 of arXiv:2510.20151v2), single-label-class adaptation.

    r(T̂L) = rho_rec(T̂L) * (EM(T̂L) + F1char(T̂L)) / 2

* rho_rec — reconstruction ratio: proportion of the input text d (in
  CHARACTERS) recovered by the reconstructed segments.  Implementation
  detail: d == " ".join(words); a separator space between two consecutive
  words is counted as recovered iff both adjacent words are recovered.  With
  a perfect full-coverage tiling every character of d is recovered and
  rho_rec == 1 exactly (a naive sum of segment-string lengths would top out
  at (|d| - (n_seg - 1)) / |d| because the joining spaces between adjacent
  segments belong to neither segment).

* EM — F1 of exact segment matches: precision = fraction of reconstructed
  segments whose text exactly equals some annotated segment's text, recall =
  fraction of annotated segments exactly produced (multiset matching, since
  the single class removes label comparison).  Dropped (unlocatable)
  segments are absent from T̂L per the paper, so they lower rho_rec and
  recall but not precision.

* F1char — character-level labeling F1.  With a single class the paper's
  label-per-character metric degenerates (every character would carry the
  label "sentence"), so per the pilot ruling it is adapted to SEGMENTATION
  AGREEMENT: the character-level "label" is the segment identity.  Each
  character of d gets its gold segment index as the gold label; each
  reconstructed segment is aligned to the gold segment with maximal
  character overlap (ties -> earliest gold segment) and all its characters
  get that index as the predicted label; characters not covered by any
  reconstructed segment get the null label -1.  F1 is the support-weighted
  mean over gold classes of the per-class F1 (the paper's "weighted F-1
  score"), computed over word units weighted by their character length
  (word + following separator), which is exactly the char-level score
  because segment boundaries never fall inside a word.

Self-check requirements (covered by tests/test_boundrl_reward.py):
  perfect prediction -> r == 1;  off-by-one-word starts -> partial credit;
  unlocatable starts -> segment dropped and rho_rec falls.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundrl_data import parse_target, reconstruct  # noqa: E402

Span = Tuple[int, int]


def _char_weights(words: Sequence[str]) -> List[int]:
    """Character weight of each word position in d = " ".join(words):
    len(word) plus 1 for the separator that follows (none after the last)."""
    n = len(words)
    return [len(w) + (1 if i < n - 1 else 0) for i, w in enumerate(words)]


def reconstruction_ratio(words: Sequence[str], pred_spans: Sequence[Optional[Span]]) -> float:
    """rho_rec = recovered characters / |d| (see module docstring)."""
    n = len(words)
    if n == 0:
        return 0.0
    covered = [False] * n
    for sp in pred_spans:
        if sp is None:
            continue
        b, e = sp
        for i in range(max(b, 0), min(e, n)):
            covered[i] = True
    total = sum(len(w) for w in words) + (n - 1)
    got = sum(len(w) for i, w in enumerate(words) if covered[i])
    got += sum(1 for i in range(n - 1) if covered[i] and covered[i + 1])
    return got / total if total else 0.0


def exact_match_f1(
    words: Sequence[str], gold_spans: Sequence[Span], pred_spans: Sequence[Optional[Span]]
) -> float:
    """F1 of exact segment (text) matches, multiset counting."""
    gold_texts = Counter(" ".join(words[b:e]) for b, e in gold_spans)
    kept = [sp for sp in pred_spans if sp is not None]
    pred_texts = Counter(" ".join(words[b:e]) for b, e in kept)
    n_match = sum((gold_texts & pred_texts).values())
    if not kept or not gold_spans:
        return 0.0
    precision = n_match / len(kept)
    recall = n_match / len(gold_spans)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def char_f1(
    words: Sequence[str], gold_spans: Sequence[Span], pred_spans: Sequence[Optional[Span]]
) -> float:
    """Support-weighted char-level F1 with segment-identity labels."""
    n = len(words)
    if n == 0 or not gold_spans:
        return 0.0
    w = _char_weights(words)
    total = sum(w)

    gold_id = [-2] * n  # every word belongs to exactly one gold span
    for gi, (b, e) in enumerate(gold_spans):
        for i in range(b, min(e, n)):
            gold_id[i] = gi

    pred_id = [-1] * n  # -1 = not covered by any reconstructed segment
    for sp in pred_spans:
        if sp is None:
            continue
        b, e = max(sp[0], 0), min(sp[1], n)
        # align this predicted segment to the max-char-overlap gold segment
        overlap: Dict[int, int] = {}
        for i in range(b, e):
            overlap[gold_id[i]] = overlap.get(gold_id[i], 0) + w[i]
        if not overlap:
            continue
        best = max(sorted(overlap), key=lambda g: overlap[g])
        for i in range(b, e):
            pred_id[i] = best

    f1_weighted = 0.0
    for gi, (b, e) in enumerate(gold_spans):
        support = sum(w[i] for i in range(b, min(e, n)))
        tp = sum(w[i] for i in range(n) if gold_id[i] == gi and pred_id[i] == gi)
        fp = sum(w[i] for i in range(n) if gold_id[i] != gi and pred_id[i] == gi)
        fn = support - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        f1_weighted += (support / total) * f1
    return f1_weighted


def segmentation_reward(
    words: Sequence[str], gold_spans: Sequence[Span], pred_spans: Sequence[Optional[Span]]
) -> Dict[str, float]:
    """r = rho_rec * (EM + F1char) / 2   (Eq. 2 of the paper)."""
    rho = reconstruction_ratio(words, pred_spans)
    em = exact_match_f1(words, gold_spans, pred_spans)
    f1c = char_f1(words, gold_spans, pred_spans)
    return {"rho_rec": rho, "em": em, "f1char": f1c, "reward": rho * (em + f1c) / 2.0}


def completion_reward(
    words: Sequence[str], gold_spans: Sequence[Span], completion_text: str
) -> Dict[str, object]:
    """Parse a raw model completion, reconstruct, and score.  Returns the
    metric dict plus the reconstructed spans and drop count."""
    seqs = parse_target(completion_text)
    if not seqs:
        return {"rho_rec": 0.0, "em": 0.0, "f1char": 0.0, "reward": 0.0,
                "spans": [], "n_dropped": 0, "start_seqs": []}
    spans, dropped = reconstruct(words, seqs)
    out = segmentation_reward(words, gold_spans, spans)
    out["spans"] = spans
    out["n_dropped"] = dropped
    out["start_seqs"] = seqs
    return out
