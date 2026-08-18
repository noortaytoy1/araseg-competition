"""Quality filter for SYNTHETIC segmentation data (closed-track law enforced).

Purpose
-------
Synthetic training docs come out of the closed-track augmenters (``make_recomb.py``,
and the not-yet-materialized ``daga_gen.py`` / ``block_aug.py``). Left unfiltered,
degenerate or off-distribution synthetic docs poison the segmenter — recomb was
scored -1.56 precisely because it ignored the real train length distribution and
went OOD. This module reads a synthetic JSONL and writes a *filtered* JSONL,
rejecting docs that are degenerate, off-distribution, non-Arabic, or — worst of
all — leaking dev/test material.

CLOSED-TRACK LAW (non-negotiable, enforced here)
-------------------------------------------------
The distribution profile (length band, boundary-rate band) is fit ONLY on
``data/{TASK}_train.jsonl`` (the 174 train docs). Dev and test files are read for
EXACTLY ONE purpose: to build an 8-gram *tripwire* that DROPS any synthetic doc
sharing an 8-gram with dev/test token streams. That is a leakage guard, not a fit
on dev/test — no threshold, band, or statistic is derived from dev/test content.
No external corpus and no generative LM world-knowledge is consulted. Boundary
labels are validated as correct-by-construction (each synthetic doc must end on a
boundary and have len(tokens)==len(labels)); they are never guessed or re-segmented.

Schema (mirrors ``make_recomb.py`` output)
------------------------------------------
JSONL, one doc per line: ``{"doc_id": str, "tokens": [str, ...], "labels": [0/1, ...]}``
where label 1 == a sentence boundary AFTER that token. Extra keys on a doc (e.g.
``text``, ``label_str`` on the original train docs) are preserved verbatim on kept
docs. Only ``tokens`` and ``labels`` are required.

Filters (each with an explicit rule + default threshold)
--------------------------------------------------------
0. schema/label validity  : len(tokens)==len(labels)>0, labels are 0/1, and the
                            doc ends on a boundary (labels[-1]==1). By construction
                            this should never fire; it is a fail-closed guard.
1. degeneracy/repetition  : reject if the max n-gram repetition ratio exceeds a cap
                            (default: 4-gram distinct-ratio < 0.55 => reject; also a
                            top-token-frequency cap).
2. length-band            : reject if token length is outside the real-train band
                            [p01 .. p99] of {TASK}_train (keeps the heavy right tail
                            without letting a 20k-token monster through).
3. boundary-rate band     : reject if per-doc boundary rate is outside a sane band
                            around the real-train rate (default: train_rate * [0.5 .. 1.6],
                            clamped to a hard [0.03 .. 0.30]).
4. arabic script-ratio    : reject if the Arabic-script character ratio over all
                            token text is < 0.90.
5. 8-gram dev/test tripwire: reject if the doc shares ANY 8-gram (token 8-gram) with
                            the dev OR test token stream of {TASK}. Leakage guard.
6. (optional) ensemble    : if a pooled ensemble-prob npz is supplied and a doc's
   agreement                tokens can be aligned to it, score agreement between the
                            by-construction labels and ensemble boundary probs; reject
                            docs whose agreement is below a floor. Skipped gracefully
                            (never fails the run) if no npz is given or alignment is
                            impossible — this hook is advisory only.

Usage
-----
    python -m src.synth_filter --task NoPnx-NP \
        --in data/recomb_NoPnx-NP_train.jsonl \
        --out data/recomb_NoPnx-NP_train.filtered.jsonl

    python -m src.synth_filter --self-test        # CPU smoke test, no data needed

The filter is generator-agnostic: it operates on any synthetic JSONL regardless of
which augmenter produced it (recomb / daga_gen / block_aug / hand-made). It NEVER
modifies its input file and NEVER touches any existing repo file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# On Windows the default console codec (cp1252) cannot encode Arabic; force UTF-8
# so the smoke report and any Arabic doc content print cleanly. Best-effort.
try:  # pragma: no cover - platform shim
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

# --------------------------------------------------------------------------- #
# Locate the repo data dir the same way make_recomb.py does, so this module    #
# works no matter the current working directory.                               #
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")

# Reuse the project's own JSONL loader when importable; otherwise fall back to a
# local reader. Importing must not require `data/` to be present (tests hide it).
try:  # pragma: no cover - trivial import shim
    from src.data import load_jsonl as _load_jsonl  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from data import load_jsonl as _load_jsonl  # type: ignore
    except Exception:  # noqa: BLE001
        _load_jsonl = None  # defined below


def load_jsonl(path: str) -> List[dict]:
    """Load a JSONL file of docs. Prefers src.data.load_jsonl if available."""
    if _load_jsonl is not None:
        return _load_jsonl(path)
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def write_jsonl(path: str, docs: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Arabic script detection (no external deps).                                  #
# Ranges: Arabic (0600-06FF), Supplement (0750-077F), Extended-A (08A0-08FF),  #
# Presentation Forms-A (FB50-FDFF), Forms-B (FE70-FEFF).                        #
# --------------------------------------------------------------------------- #
def _is_arabic_char(ch: str) -> bool:
    o = ord(ch)
    return (
        0x0600 <= o <= 0x06FF
        or 0x0750 <= o <= 0x077F
        or 0x08A0 <= o <= 0x08FF
        or 0xFB50 <= o <= 0xFDFF
        or 0xFE70 <= o <= 0xFEFF
    )


def arabic_ratio(tokens: Sequence[str]) -> float:
    """Fraction of *letter* characters that are Arabic script.

    Digits, punctuation, whitespace and the paragraph token are ignored so that a
    legitimately Arabic doc peppered with Arabic-Indic digits or parentheses is not
    penalized. If a doc has no letters at all, ratio is 0.0 (degenerate -> reject).
    """
    ar = 0
    letters = 0
    for tok in tokens:
        for ch in tok:
            if ch.isalpha():
                letters += 1
                if _is_arabic_char(ch):
                    ar += 1
    if letters == 0:
        return 0.0
    return ar / letters


# --------------------------------------------------------------------------- #
# n-gram repetition / degeneracy.                                              #
# --------------------------------------------------------------------------- #
def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_ngram_ratio(tokens: Sequence[str], n: int) -> float:
    """distinct n-grams / total n-grams. 1.0 == no repetition; low == degenerate.

    For docs too short to have an n-gram, returns 1.0 (nothing to reject on).
    """
    grams = _ngrams(tokens, n)
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def top_token_frequency(tokens: Sequence[str]) -> float:
    """Fraction of the doc made up of its single most common token.

    Catches the ``the the the ...`` failure mode that a distinct-n-gram ratio can
    miss when the repeated span is longer than n.
    """
    if not tokens:
        return 1.0
    c = Counter(tokens)
    return c.most_common(1)[0][1] / len(tokens)


# --------------------------------------------------------------------------- #
# Length + boundary-rate profile fit on TRAIN ONLY.                            #
# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]. sorted_vals must be sorted."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class TrainProfile:
    """Distribution profile fit ONLY on the closed-track train split.

    Holds the length band and boundary-rate band that in-distribution synthetic
    docs must fall within.
    """

    def __init__(
        self,
        len_lo: int,
        len_hi: int,
        brate_lo: float,
        brate_hi: float,
        train_brate: float,
        n_train: int,
    ) -> None:
        self.len_lo = len_lo
        self.len_hi = len_hi
        self.brate_lo = brate_lo
        self.brate_hi = brate_hi
        self.train_brate = train_brate
        self.n_train = n_train

    def as_dict(self) -> Dict[str, float]:
        return {
            "len_lo": self.len_lo,
            "len_hi": self.len_hi,
            "brate_lo": round(self.brate_lo, 4),
            "brate_hi": round(self.brate_hi, 4),
            "train_brate": round(self.train_brate, 4),
            "n_train": self.n_train,
        }


def fit_train_profile(
    train_docs: Sequence[dict],
    len_low_q: float = 0.01,
    len_high_q: float = 0.99,
    brate_lo_mult: float = 0.5,
    brate_hi_mult: float = 1.6,
    brate_hard_lo: float = 0.03,
    brate_hard_hi: float = 0.30,
) -> TrainProfile:
    """Fit the length band [p01..p99] and boundary-rate band on TRAIN docs only.

    The boundary-rate band is anchored on the pooled train boundary rate times
    [brate_lo_mult .. brate_hi_mult], then clamped to a hard [0.03 .. 0.30] so a
    pathological train estimate can never open the gate arbitrarily wide.
    """
    lens = sorted(len(d["tokens"]) for d in train_docs)
    len_lo = int(round(_percentile(lens, len_low_q)))
    len_hi = int(round(_percentile(lens, len_high_q)))

    pos = sum(sum(d["labels"]) for d in train_docs)
    tot = sum(len(d["labels"]) for d in train_docs)
    train_brate = pos / max(tot, 1)

    brate_lo = max(brate_hard_lo, train_brate * brate_lo_mult)
    brate_hi = min(brate_hard_hi, train_brate * brate_hi_mult)
    return TrainProfile(
        len_lo=len_lo,
        len_hi=len_hi,
        brate_lo=brate_lo,
        brate_hi=brate_hi,
        train_brate=train_brate,
        n_train=len(train_docs),
    )


# --------------------------------------------------------------------------- #
# 8-gram dev/test tripwire.                                                    #
# --------------------------------------------------------------------------- #
def build_ngram_tripwire(
    docs_streams: Sequence[Sequence[dict]], n: int = 8
) -> set:
    """Build a set of token n-grams appearing in the given doc streams (dev/test).

    A synthetic doc sharing any of these n-grams is treated as leakage and dropped.
    """
    grams: set = set()
    for docs in docs_streams:
        for d in docs:
            toks = d["tokens"]
            for g in _ngrams(toks, n):
                grams.add(g)
    return grams


def shares_ngram(tokens: Sequence[str], tripwire: set, n: int = 8) -> bool:
    if not tripwire:
        return False
    for g in _ngrams(tokens, n):
        if g in tripwire:
            return True
    return False


# --------------------------------------------------------------------------- #
# Optional ensemble-agreement hook (advisory; skips gracefully).               #
# --------------------------------------------------------------------------- #
def _load_pool_probs(npz_path: str) -> Optional[dict]:
    """Load a pooled-ensemble npz of per-token boundary probs, if possible.

    Returns a dict mapping doc_id -> 1-D prob array, or None if the artifact
    cannot be interpreted. NEVER raises — the hook is advisory.
    """
    try:
        import numpy as np  # local import; numpy may be absent in a bare env
    except Exception:  # noqa: BLE001
        return None
    if not npz_path or not os.path.exists(npz_path):
        return None
    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:  # noqa: BLE001
        return None
    keys = list(z.keys())
    # Common layouts: {doc_id: probs} or {"doc_ids": [...], "probs": [...]}.
    out: Dict[str, "np.ndarray"] = {}
    if "doc_ids" in keys and "probs" in keys:
        try:
            ids = list(z["doc_ids"])
            probs = z["probs"]
            for i, did in enumerate(ids):
                out[str(did)] = np.asarray(probs[i]).ravel()
            return out
        except Exception:  # noqa: BLE001
            return None
    # Fallback: treat each key as a doc_id whose value is a prob array.
    for k in keys:
        try:
            out[str(k)] = np.asarray(z[k]).ravel()
        except Exception:  # noqa: BLE001
            continue
    return out or None


def ensemble_agreement(
    doc: dict, pool: Optional[dict], thr: float = 0.5
) -> Optional[float]:
    """Fraction of tokens where the by-construction label agrees with pool>thr.

    Returns None (skip) when there is no pool, no matching doc_id, or a length
    mismatch — the hook never rejects on a can't-align condition.
    """
    if not pool:
        return None
    did = str(doc.get("doc_id", ""))
    probs = pool.get(did)
    if probs is None:
        return None
    labels = doc["labels"]
    if len(probs) != len(labels):
        return None
    agree = sum(1 for p, y in zip(probs, labels) if int(p >= thr) == int(y))
    return agree / max(len(labels), 1)


# --------------------------------------------------------------------------- #
# Per-doc filtering.                                                           #
# --------------------------------------------------------------------------- #
FILTER_ORDER = [
    "invalid_labels",
    "degenerate",
    "length_band",
    "boundary_rate",
    "arabic_ratio",
    "leak_8gram",
    "ensemble_disagree",
]


class FilterConfig:
    def __init__(
        self,
        # degeneracy
        ngram_n: int = 4,
        min_distinct_ngram: float = 0.55,
        max_top_token_freq: float = 0.35,
        # arabic
        min_arabic_ratio: float = 0.90,
        # tripwire
        tripwire_n: int = 8,
        # ensemble (advisory)
        min_ensemble_agreement: float = 0.60,
    ) -> None:
        self.ngram_n = ngram_n
        self.min_distinct_ngram = min_distinct_ngram
        self.max_top_token_freq = max_top_token_freq
        self.min_arabic_ratio = min_arabic_ratio
        self.tripwire_n = tripwire_n
        self.min_ensemble_agreement = min_ensemble_agreement


def _valid_labels(doc: dict) -> bool:
    toks = doc.get("tokens")
    labs = doc.get("labels")
    if not isinstance(toks, list) or not isinstance(labs, list):
        return False
    if len(toks) == 0 or len(toks) != len(labs):
        return False
    if any(l not in (0, 1) for l in labs):
        return False
    # correct-by-construction: a synthetic doc must end on a boundary
    if labs[-1] != 1:
        return False
    return True


def classify_doc(
    doc: dict,
    profile: TrainProfile,
    tripwire: set,
    cfg: FilterConfig,
    pool: Optional[dict] = None,
) -> Optional[str]:
    """Return the name of the FIRST filter that rejects this doc, or None if kept.

    Filters are applied in FILTER_ORDER; the first failure is the recorded reason.
    """
    # 0. schema / label validity (fail-closed)
    if not _valid_labels(doc):
        return "invalid_labels"

    tokens = doc["tokens"]
    labels = doc["labels"]

    # 1. degeneracy / repetition
    if distinct_ngram_ratio(tokens, cfg.ngram_n) < cfg.min_distinct_ngram:
        return "degenerate"
    if top_token_frequency(tokens) > cfg.max_top_token_freq:
        return "degenerate"

    # 2. length band
    n = len(tokens)
    if n < profile.len_lo or n > profile.len_hi:
        return "length_band"

    # 3. boundary-rate band
    brate = sum(labels) / max(len(labels), 1)
    if brate < profile.brate_lo or brate > profile.brate_hi:
        return "boundary_rate"

    # 4. arabic script ratio
    if arabic_ratio(tokens) < cfg.min_arabic_ratio:
        return "arabic_ratio"

    # 5. 8-gram dev/test tripwire (leakage)
    if shares_ngram(tokens, tripwire, cfg.tripwire_n):
        return "leak_8gram"

    # 6. optional ensemble agreement (advisory; None => skip)
    if pool is not None:
        agr = ensemble_agreement(doc, pool)
        if agr is not None and agr < cfg.min_ensemble_agreement:
            return "ensemble_disagree"

    return None


# --------------------------------------------------------------------------- #
# Top-level filter over a JSONL.                                               #
# --------------------------------------------------------------------------- #
class FilterResult:
    def __init__(self) -> None:
        self.kept: List[dict] = []
        self.dropped_by: Counter = Counter()
        self.n_in = 0

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        return self.n_in - self.n_kept


def filter_docs(
    synth_docs: Sequence[dict],
    profile: TrainProfile,
    tripwire: set,
    cfg: FilterConfig,
    pool: Optional[dict] = None,
    protect_originals: bool = True,
) -> FilterResult:
    """Apply all filters to a list of synthetic docs.

    ``protect_originals``: docs whose doc_id does NOT look synthetic (i.e. carry a
    ``text``/``label_str`` field, the signature of the real 174 train docs mixed
    into a recomb file) are kept unconditionally — the augmenters copy the real
    train docs into their output and those are ground truth, never filtered.
    """
    res = FilterResult()
    res.n_in = len(synth_docs)
    for doc in synth_docs:
        if protect_originals and _looks_like_original(doc):
            res.kept.append(doc)
            continue
        reason = classify_doc(doc, profile, tripwire, cfg, pool)
        if reason is None:
            res.kept.append(doc)
        else:
            res.dropped_by[reason] += 1
    return res


def _looks_like_original(doc: dict) -> bool:
    """A real train doc copied into the synthetic file (has text + label_str)."""
    return "text" in doc and "label_str" in doc


def report_breakdown(res: FilterResult, profile: TrainProfile, cfg: FilterConfig) -> str:
    lines = []
    lines.append("PER-FILTER KEPT/DROPPED BREAKDOWN")
    lines.append(f"  input docs         : {res.n_in}")
    lines.append(f"  kept               : {res.n_kept}")
    lines.append(f"  dropped            : {res.n_dropped}")
    lines.append("  dropped by filter  :")
    for name in FILTER_ORDER:
        cnt = res.dropped_by.get(name, 0)
        lines.append(f"    {name:<18}: {cnt}")
    # any unexpected reasons
    for name, cnt in res.dropped_by.items():
        if name not in FILTER_ORDER:
            lines.append(f"    {name:<18}: {cnt}  (unexpected)")
    lines.append("  profile (train-fit):")
    for k, v in profile.as_dict().items():
        lines.append(f"    {k:<12}: {v}")
    lines.append("  thresholds:")
    lines.append(f"    degeneracy   : distinct-{cfg.ngram_n}gram >= {cfg.min_distinct_ngram}, "
                 f"top-token-freq <= {cfg.max_top_token_freq}")
    lines.append(f"    arabic-ratio : >= {cfg.min_arabic_ratio}")
    lines.append(f"    tripwire     : no shared {cfg.tripwire_n}-gram with dev/test")
    lines.append(f"    ensemble     : agreement >= {cfg.min_ensemble_agreement} (if npz)")
    return "\n".join(lines)


def run_filter(
    task: str,
    in_path: str,
    out_path: str,
    data_dir: str = DATA,
    pool_npz: Optional[str] = None,
    cfg: Optional[FilterConfig] = None,
) -> FilterResult:
    """End-to-end: load train profile + dev/test tripwire, filter, write output."""
    cfg = cfg or FilterConfig()

    train_path = os.path.join(data_dir, f"{task}_train.jsonl")
    dev_path = os.path.join(data_dir, f"{task}_dev.jsonl")
    test_path = os.path.join(data_dir, f"{task}_test.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Closed-track train split not found: {train_path} — the profile MUST "
            f"be fit on train only; refusing to proceed."
        )
    train_docs = load_jsonl(train_path)
    profile = fit_train_profile(train_docs)

    # Tripwire is built from dev + test if present. Absence is allowed (e.g. a bare
    # smoke env) but is loudly noted — a missing tripwire means NO leakage guard.
    streams = []
    for p, name in ((dev_path, "dev"), (test_path, "test")):
        if os.path.exists(p):
            streams.append(load_jsonl(p))
        else:
            print(f"WARNING: {name} split absent ({p}) — 8-gram tripwire will not "
                  f"cover it. Leakage against {name} is NOT guarded this run.",
                  file=sys.stderr)
    tripwire = build_ngram_tripwire(streams, n=cfg.tripwire_n)

    pool = _load_pool_probs(pool_npz) if pool_npz else None
    if pool_npz and pool is None:
        print(f"NOTE: ensemble npz {pool_npz} could not be used — ensemble-agreement "
              f"filter skipped (advisory only).", file=sys.stderr)

    synth_docs = load_jsonl(in_path)
    res = filter_docs(synth_docs, profile, tripwire, cfg, pool)

    write_jsonl(out_path, res.kept)
    print(report_breakdown(res, profile, cfg))
    print(f"\nwrote {res.n_kept} kept docs -> {out_path}")
    return res


# --------------------------------------------------------------------------- #
# Self-test / SMOKE REPORT (CPU, tiny scale).                                  #
# --------------------------------------------------------------------------- #
def _make_fake_train() -> Tuple[List[dict], list]:
    """A tiny hand-made 'real train' set.

    Sentences are ~10 tokens each (boundary rate ~0.10, matching the real ~0.104),
    so the fitted boundary-rate band is realistic and each planted bad doc can be
    made in-distribution on every axis except the one it is meant to trip.
    """
    base = [
        (["القطة", "الصغيرة", "تجلس", "بهدوء", "على", "الحصير",
          "الملون", "قرب", "النافذة", "المفتوحة"], [0] * 9 + [1]),
        (["الولد", "المجتهد", "يقرأ", "الكتاب", "المفيد", "في",
          "المدرسة", "كل", "صباح", "باكرا"], [0] * 9 + [1]),
        (["ذهب", "الرجل", "الكريم", "إلى", "السوق", "الكبير",
          "لشراء", "الخضار", "الطازجة", "اليوم"], [0] * 9 + [1]),
        (["الشمس", "الساطعة", "تشرق", "من", "جهة", "الشرق",
          "لتضيء", "القرية", "الجميلة", "الوادعة"], [0] * 9 + [1]),
        (["نحن", "جميعا", "نحب", "بلادنا", "الغالية", "ونعمل",
          "بجد", "لبناء", "مستقبلها", "المشرق"], [0] * 9 + [1]),
        (["المعلم", "القدير", "يشرح", "الدرس", "الجديد", "للطلاب",
          "بوضوح", "وصبر", "داخل", "الفصل"], [0] * 9 + [1]),
    ]
    docs = []
    for i in range(30):  # 30 'train' docs by joining varied numbers of sentences
        toks: List[str] = []
        labs: List[int] = []
        for j in range((i % 5) + 4):  # 4..8 sentences per doc -> len 40..80
            s, l = base[(i + j) % len(base)]
            toks += s
            labs += l
        docs.append({
            "doc_id": f"train_{i}",
            "tokens": toks,
            "labels": labs,
            "text": " ".join(toks),
            "label_str": "".join(map(str, labs)),
        })
    return docs, base


def _make_fake_synth(base) -> List[dict]:
    """A tiny synthetic set: a few good docs + planted bad ones.

    Every bad doc is in-distribution on length (~40-80) and Arabic-script EXCEPT on
    the single axis it is designed to trip, so the smoke test proves each filter
    fires on its own target rather than a doc being incidentally caught elsewhere.
    """
    def concat(seqs):
        toks, labs = [], []
        for s, l in seqs:
            toks += list(s)
            labs += list(l)
        return toks, labs

    docs = []

    # 3 GOOD in-distribution docs (recombination of the same base sentences)
    for k in range(3):
        seqs = [base[(k + j) % len(base)] for j in range(5 + k)]  # len 50..70
        t, l = concat(seqs)
        docs.append({"doc_id": f"synth_good_{k}", "tokens": t, "labels": l})

    # BAD: degenerate repetition — same sentence repeated (in length band, ~brate)
    t, l = concat([base[0]] * 6)  # 60 tokens, brate 0.10, but 4-grams all repeat
    docs.append({"doc_id": "synth_degen", "tokens": t, "labels": l})

    # BAD: too short (below length band) — Arabic, ends on boundary
    short_seq = base[0][0][:6]
    docs.append({
        "doc_id": "synth_tooshort",
        "tokens": list(short_seq),
        "labels": [0] * 5 + [1],
    })

    # BAD: boundary rate far too high — length in band, Arabic, distinct tokens,
    # but a boundary after almost every token (brate ~0.9).
    hb_tokens = [base[i % len(base)][0][j % 10]
                 for i in range(5) for j in range(10)]  # 50 varied Arabic tokens
    hb_labels = [1] * len(hb_tokens)  # boundary after every token -> brate 1.0
    docs.append({"doc_id": "synth_hibrate", "tokens": hb_tokens, "labels": hb_labels})

    # BAD: non-Arabic (Latin) content — length in band, brate ~0.10, and all
    # tokens distinct (unique index suffix) so it is NOT degenerate and therefore
    # must be caught specifically by the Arabic-script-ratio filter.
    latin_words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
                   "adipiscing", "elit", "sed", "do"]
    lat_tokens = [f"{latin_words[k % len(latin_words)]}{k}" for k in range(50)]
    lat_labels = [1 if (k + 1) % 10 == 0 else 0 for k in range(len(lat_tokens))]
    docs.append({"doc_id": "synth_latin", "tokens": lat_tokens, "labels": lat_labels})

    # BAD: invalid labels — in-distribution but does NOT end on a boundary
    seqs = [base[j % len(base)] for j in range(5)]
    t, l = concat(seqs)
    l = list(l)
    l[-1] = 0  # break correct-by-construction
    docs.append({"doc_id": "synth_noend", "tokens": t, "labels": l})

    return docs


def _make_fake_devtest():
    """A dev/test stream carrying a unique 8-gram to trip a planted leak doc."""
    leak = ["سلسلة", "من", "ثماني", "كلمات", "فريدة", "جدا", "للاختبار", "هنا"]
    dev = [{"doc_id": "dev_0", "tokens": leak + ["إضافي"], "labels": [0] * 8 + [1]}]
    test = [{"doc_id": "test_0", "tokens": ["مقطع"] + leak, "labels": [0] * 9}]
    return dev, test, leak


def self_test() -> int:
    """Run a CPU-only smoke test on a synthesized tiny fake set. Prints a report."""
    print("=" * 70)
    print("SYNTH_FILTER SMOKE TEST (CPU, tiny scale, no real data required)")
    print("=" * 70)

    train_docs, base = _make_fake_train()
    dev, test, leak = _make_fake_devtest()
    synth = _make_fake_synth(base)

    # Plant one doc that leaks a dev/test 8-gram, guaranteed in-distribution otherwise
    good_body = []
    for j in range(5):
        s, l = base[j % len(base)]
        good_body += [(s, l)]

    def concat(seqs):
        toks, labs = [], []
        for s, l in seqs:
            toks += list(s)
            labs += list(l)
        return toks, labs

    body_t, body_l = concat(good_body)
    leak_doc = {
        "doc_id": "synth_leak",
        "tokens": body_t + leak + ["ختام"],
        "labels": body_l + [0] * 8 + [1],
    }
    synth.append(leak_doc)

    # ------- (a) validate labels of every VALID synthetic doc by construction -----
    print("\n(a) LABEL VALIDITY of synthetic docs (each valid doc must end on a")
    print("    boundary; len(tokens)==len(labels); labels in {0,1}):")
    for d in synth:
        ok = _valid_labels(d)
        ends = d["labels"][-1] if d["labels"] else None
        match = len(d["tokens"]) == len(d["labels"])
        flag = "VALID  " if ok else "INVALID"
        print(f"    {flag} {d['doc_id']:<16} len={len(d['tokens']):>3} "
              f"ends_on_boundary={ends} len_match={match}")

    # ------- fit profile + tripwire ----------------------------------------------
    profile = fit_train_profile(train_docs)
    cfg = FilterConfig()
    tripwire = build_ngram_tripwire([dev, test], n=cfg.tripwire_n)

    # ------- (b) distribution match: train vs kept-synth ------------------------
    res = filter_docs(synth, profile, tripwire, cfg, pool=None)

    def profile_of(docs):
        lens = sorted(len(d["tokens"]) for d in docs)
        pos = sum(sum(d["labels"]) for d in docs)
        tot = sum(len(d["labels"]) for d in docs)
        med = lens[len(lens) // 2] if lens else 0
        return (lens[0] if lens else 0, med, lens[-1] if lens else 0,
                pos / max(tot, 1))

    tr = profile_of(train_docs)
    kept_syn = [d for d in res.kept]
    ks = profile_of(kept_syn) if kept_syn else (0, 0, 0, 0)
    print("\n(b) DISTRIBUTION MATCH (kept synth should resemble real train):")
    print(f"    real train : len min/med/max = {tr[0]}/{tr[1]}/{tr[2]}  brate={tr[3]:.3f}")
    print(f"    kept synth : len min/med/max = {ks[0]}/{ks[1]}/{ks[2]}  brate={ks[3]:.3f}")
    print(f"    train band : len[{profile.len_lo}..{profile.len_hi}] "
          f"brate[{profile.brate_lo:.3f}..{profile.brate_hi:.3f}]")

    # ------- (c) round-trip integrity -------------------------------------------
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "synth_filter_smoke_out.jsonl")
    n_written = write_jsonl(tmp, res.kept)
    reread = load_jsonl(tmp)
    rt_ok = (
        n_written == len(res.kept) == len(reread)
        and all(a["doc_id"] == b["doc_id"] and a["tokens"] == b["tokens"]
                and a["labels"] == b["labels"]
                for a, b in zip(res.kept, reread))
    )
    print("\n(c) ROUND-TRIP INTEGRITY (write -> read -> compare):")
    print(f"    wrote {n_written}, reread {len(reread)}, identical={rt_ok} -> {tmp}")

    # ------- breakdown + examples ------------------------------------------------
    print()
    print(report_breakdown(res, profile, cfg))

    print("\nEXAMPLE KEPT DOC:")
    if res.kept:
        ek = res.kept[0]
        print(f"    id={ek['doc_id']} tokens[:8]={ek['tokens'][:8]} "
              f"labels[:8]={ek['labels'][:8]} ends={ek['labels'][-1]}")
    print("EXAMPLE DROPPED DOCS (id -> reason):")
    for d in synth:
        if _valid_labels(d):
            r = classify_doc(d, profile, tripwire, cfg)
        else:
            r = "invalid_labels"
        if r is not None:
            print(f"    {d['doc_id']:<16} -> {r}")

    # ------- assertions: the planted docs must be caught, good ones kept ---------
    reasons = {d["doc_id"]: (classify_doc(d, profile, tripwire, cfg)
                             if _valid_labels(d) else "invalid_labels")
               for d in synth}
    expect = {
        "synth_good_0": None, "synth_good_1": None, "synth_good_2": None,
        "synth_degen": "degenerate",
        "synth_tooshort": "length_band",
        "synth_hibrate": "boundary_rate",
        "synth_latin": "arabic_ratio",
        "synth_noend": "invalid_labels",
        "synth_leak": "leak_8gram",
    }
    failures = []
    for did, want in expect.items():
        got = reasons.get(did, "MISSING")
        if got != want:
            failures.append(f"{did}: expected {want!r}, got {got!r}")
    if not rt_ok:
        failures.append("round-trip integrity failed")

    print("\n" + "=" * 70)
    if failures:
        print("SMOKE TEST: FAIL")
        for f in failures:
            print("   -", f)
        print("=" * 70)
        return 1
    print("SMOKE TEST: PASS — all planted bad docs rejected by the expected filter,")
    print("all good docs kept, labels valid by construction, round-trip intact.")
    print("=" * 70)
    return 0


# --------------------------------------------------------------------------- #
# CLI.                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Quality filter for synthetic segmentation data (closed-track).")
    ap.add_argument("--task", default="NoPnx-NP",
                    help="Task whose train/dev/test define the profile + tripwire.")
    ap.add_argument("--in", dest="in_path", help="Input synthetic JSONL.")
    ap.add_argument("--out", dest="out_path", help="Output filtered JSONL.")
    ap.add_argument("--data-dir", default=DATA, help="Dir holding {task}_*.jsonl.")
    ap.add_argument("--pool-npz", default=None,
                    help="Optional pooled-ensemble prob npz for the agreement hook.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the CPU smoke test on a synthesized fake set and exit.")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.in_path or not args.out_path:
        ap.error("--in and --out are required unless --self-test is given.")

    if os.path.abspath(args.in_path) == os.path.abspath(args.out_path):
        ap.error("--in and --out must differ; this filter never overwrites its input.")

    run_filter(
        task=args.task,
        in_path=args.in_path,
        out_path=args.out_path,
        data_dir=args.data_dir,
        pool_npz=args.pool_npz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
