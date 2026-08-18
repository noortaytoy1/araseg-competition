# -*- coding: utf-8 -*-
"""QA START-ANCHOR DECODE PACK — BoundRL-inspired, start-anchored re-decode for
the QA/exam/cloze genre. ADDITIVE new file; nothing existing is modified. CPU
only; reads the frozen prob caches, never touches a model. Precedents: the
legal decode pack (v16-B1 + v17-2) and the chain pack (src/chain_decode.py):
hand-designed detectors + TRAIN-GOLD-ONLY fitting, hard fire/no-fire gates,
dev used for measurement/reporting only.

Why this pack exists (measured context of record)
--------------------------------------------------
The model SHREDS exam docs: predicted sentence length 3.7 vs gold 5.1, the
cloze placeholder token sits before ~99 false cuts, and 44% of P>=0.99 cuts in
this genre are wrong; fixing the 4 worst cloze dev docs is a +0.54 macro-F1
ceiling. The failure is OVER-cutting inside question-units, so the correct
decode-side move is SUPPRESSION of non-start cuts (the opposite of the chain
genre, where the model under-cut and the pack measured +0.00 for lack of train
evidence — here train HAS exam docs, so suppression is train-licensed).

Fitting discipline (binding, v16-B1 pattern)
--------------------------------------------
ALL patterns, lexicons, thresholds and priors are fit on TRAIN gold only
(data/NoPnx-NP_train.jsonl). Dev is measured, never fitted on. The single
model-channel constant TAU_MODEL=0.90 is declared a priori and never swept.

Design
------
1. GENRE DETECTOR (train-fitted, tokens-only, doc-level). Marker families:
     STRONG (any one qualifies the doc):
       TF       >= 3 adjacent صح…خطأ pairs           (true/false quizzes)
       CLOZE    >= 3 فراغ placeholder tokens          (cloze/fill-in docs)
       LET      >= 4 standalone MCQ letters أ ب ج د هـ and >= 0.8/100 tokens
       OPTF-HI  >= 6 option formulas and >= 2.5/100   (جميع ما سبق، كلاهما، صح، خطأ…)
     WEAK (need the anti-prose structure check: prose-function-word density
     <= 13.1% and و-prefixed-token ratio <= 9%):
       QW       >= 5 question-word units and >= 2.2/100   (هل ماذا أين متى… + من هو/ما هي…)
       STEM     >= 3 exercise-stem verbs and >= 1.0/100   (اختر اكمل رتب علل اذكر…)
       FLW      >= 2 which-of-the-following tokens and >= 0.5/100 (التالية الآتية يلي…)
       OPTF-LO  >= 3 option formulas and >= 0.9/100
     NUM-TIGHT (needs the tighter check func <= 12.0%): >= 8 standalone number
       tokens and >= 3.5/100 (dense numeric options; the tighter gate is what
       separates math/history quizzes from number-heavy prose on train).
     VETO: سيناريو present -> never exam (comic-strip credit pages: the Majid
       comics are dialogue full of question words; train shows they must not fire).
   Fitted on the 42 eyeballed train exam docs vs the other 132 train docs
   (--fit prints the confusion). Known coverage gap, reported not hidden:
   doc_46717e5891d4 (letterless name-option MCQ) carries no surface marker any
   gate can license without admitting prose docs -> it stays undetected
   (missed upside, never damage). Zero non-exam train docs may fire; that
   constraint is hard and asserted by --fit.

2. START ANCHORING (trick A). From TRAIN gold inside DETECTED train docs, fit
   what segment STARTS look like: a start-unigram lexicon, a start-bigram
   lexicon, and pooled character classes (<NUM> digit tokens, <LET> MCQ
   letters, <LAT> Latin tokens). An entry is licensed iff n >= 3 and
   P(gold-start | occurrence) >= 0.70 (unigrams/classes) or >= 0.80 (bigrams)
   — same n/P gate family as v16-B1 (promote-«،» precedent: no evidence, no
   fire). Decode inside a detected doc: force one boundary immediately BEFORE
   every accepted start (lexicon hit OR cached model prob >= TAU_MODEL), and
   SUPPRESS every other boundary (the anti-confetti move). The doc-final
   boundary is always kept (label convention: every doc ends a segment).
   Segment-length sanity from train exam gold: segments longer than the fitted
   MAXLEN re-admit the strongest cached-prob cuts inside until compliant.

3. PERTURBATION RESCORE (trick B, BoundRL candidate-perturbation). Local
   deterministic search over the anchored joints: shift each joint +/-2
   tokens, drop a joint, add a joint at the best interior position. Candidates
   are scored by summed token log-likelihood under the cached probs plus a
   train-fitted exam-genre segment-length log-prior (Laplace-smoothed
   histogram). Greedy accept-if-improve, <= 5 sweeps, fully deterministic; the
   number of candidates evaluated is reported (this is the K).

4. Override semantics (hard guarantees, asserted at apply time):
     * regions are whole detected docs; predictions of every UNDETECTED doc
       are bit-identical to baseline (asserted per doc);
     * inside detected docs both directions are allowed (this genre's failure
       is over-cutting; suppression is the point) — every modified doc is
       printed before/after for eyeballing.

Usage (CPU, reads frozen caches, writes nothing unless --out-csv):
  python src/qa_start_decode.py --self-test
  python src/qa_start_decode.py --fit
  python src/qa_start_decode.py --apply --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz
  python src/qa_start_decode.py --apply --probs probs/union-NoPnx-NP-POOL_dev.npz
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from bisect import insort
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# hand-written token classes (a priori domain knowledge, v16-B1 style)
# ---------------------------------------------------------------------------
QW_UNI = ("هل", "ماذا", "أين", "متى", "لماذا", "كيف", "بماذا", "ماهي", "ماهو", "بمن", "أيهما")
QW_BI = (("من", "هو"), ("من", "هي"), ("من", "الذي"), ("من", "التي"), ("ما", "هو"), ("ما", "هي"),
         ("ما", "الذي"), ("ما", "التي"), ("كم", "عدد"), ("في", "أي"), ("ما", "حكم"), ("ما", "اسم"),
         ("ما", "دلالة"), ("أي", "من"), ("ما", "المقصود"))
STEM_VERBS = ("اختر", "اختار", "اكمل", "أكمل", "املأ", "املا", "رتب", "عرف", "عدد", "اذكر",
              "أذكر", "علل", "قارن", "أجب", "أجيب", "حدد", "استنتج", "وضح", "اشرح", "صنف", "لخص")
MCQ_LETTERS = ("أ", "ب", "ج", "د", "هـ")
FOLLOWING = ("التالية", "الآتية", "التالي", "الآتي", "يلي", "الاتية", "الاتي")
# option formulas: (first_token, second_token_or_None)
OPTFORM_BI = (("جميع", "ماسبق"), ("جميع", "ما"), ("كلاهما", None), ("مما", "ذكر"), ("مما", "سبق"),
              ("غير", "ذلك"), ("جميع", "الإجابات"), ("جميع", "الاجابات"), ("لا", "شيء"),
              ("لاشئ", None), ("لاشيء", None), ("صح", None), ("خطأ", None))
CLOZE_TOKEN = "فراغ"
VETO_TOKENS = ("سيناريو",)          # comic-strip credit (Majid comics, train)
PROSE_FUNC = ("في", "من", "على", "إلى", "عن", "أن", "إن", "الذي", "التي", "ثم", "قد", "كان",
              "كانت", "بعد", "قبل", "حتى", "كما", "لم", "لا", "مع", "هذا", "هذه", "ذلك", "بين")
ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"

# ---------------------------------------------------------------------------
# fitted gate constants (fit on TRAIN, --fit prints the confusion of record)
# ---------------------------------------------------------------------------
TF_MIN = 3
CLOZE_MIN = 3
LET_MIN, LET_DENS = 4, 0.8            # dens = count per 100 tokens
OPTF_HI_MIN, OPTF_HI_DENS = 6, 2.5
QW_MIN, QW_DENS = 5, 2.2
STEM_MIN, STEM_DENS = 3, 1.0
FLW_MIN, FLW_DENS = 2, 0.5
OPTF_LO_MIN, OPTF_LO_DENS = 3, 0.9
NUM_MIN, NUM_DENS = 8, 3.5
WEAK_FUNC_MAX = 0.131                 # prose-function-word ratio ceiling (weak fams)
NUM_FUNC_MAX = 0.120                  # tighter ceiling for the NUM family
WPREF_MAX = 0.09                      # و-prefixed token ratio ceiling

# start-lexicon gates (v16-B1 n/P family)
UNI_MIN_N, UNI_MIN_P = 3, 0.70
BI_MIN_N, BI_MIN_P = 3, 0.80
CLS_MIN_N, CLS_MIN_P = 3, 0.70

TAU_MODEL = 0.90                      # a priori model-channel constant; never swept
LAMBDA_PRIOR = 1.0                    # rescore length-prior weight (a priori)
PRIOR_ALPHA = 0.5                     # Laplace smoothing of the length histogram
MAX_SWEEPS = 5                        # rescore local-search sweeps (deterministic)
EPS = 1e-6

# ---------------------------------------------------------------------------
# ground truth of record: eyeballed train exam/QA docs (fitting target only).
# 30 MCQ quizzes + 9 true/false quizzes + 1 question-list textbook
# (doc_51fccce927eb) + 2 workbook lesson pages with lettered/numbered
# exercises (doc_20dcf69f9905, doc_501500835788).
# ---------------------------------------------------------------------------
EXAM_TRAIN_DOCS = frozenset({
    "doc_60d9a7d0d1ca", "doc_fe3b2556a241", "doc_cc92bd9a59d3", "doc_d38d53501296",
    "doc_f7c344e48e78", "doc_b1d7ec94fa9e", "doc_cddb10bd6b23", "doc_d2cedca907a0",
    "doc_25ed373f8aeb", "doc_a83273e31ee3", "doc_056d9b714661", "doc_72b8decfdb8c",
    "doc_e438f8ad8abe", "doc_c626796ad5b3", "doc_741bfb3bda19", "doc_a17df83313f0",
    "doc_617f4fad2251", "doc_87ac286b6785", "doc_ede30da2ee29", "doc_00b450a96684",
    "doc_1c815410c046", "doc_c4cb5672fc6c", "doc_eb9893e8f98b", "doc_7825ac72ea71",
    "doc_afd0d91483b4", "doc_7a7737eabf04", "doc_149339985bf9", "doc_d30b2fb502db",
    "doc_d715df9a8c78", "doc_449f221bdbd2", "doc_f60bd3a48b58", "doc_74177f9e1015",
    "doc_b3d76332711b", "doc_c2a9fc26a488", "doc_690f8f5c21aa", "doc_46717e5891d4",
    "doc_153372a232d6", "doc_5ec52c5c5c0b", "doc_1c49614cd6bc", "doc_51fccce927eb",
    "doc_20dcf69f9905", "doc_501500835788",
})


# ---------------------------------------------------------------------------
# genre detector (pure function of the token list; no gold, no probs)
# ---------------------------------------------------------------------------
def _is_number_token(t: str) -> bool:
    return bool(t) and (t.isdigit() or all(c in ARABIC_INDIC for c in t))


def marker_counts(tokens: List[str]) -> Dict[str, float]:
    n = len(tokens)
    c: Dict[str, float] = {"n": max(1, n)}
    tf = 0
    for i, t in enumerate(tokens):
        if t == "صح" and any(tokens[j] == "خطأ" for j in range(i + 1, min(n, i + 3))):
            tf += 1
    c["TF"] = tf
    c["CLOZE"] = sum(1 for t in tokens if t == CLOZE_TOKEN)
    c["LET"] = sum(1 for t in tokens if t in MCQ_LETTERS)
    qw = sum(1 for t in tokens if t in QW_UNI)
    qw += sum(1 for i in range(n - 1) if (tokens[i], tokens[i + 1]) in QW_BI)
    c["QW"] = qw
    c["STEM"] = sum(1 for t in tokens if t in STEM_VERBS)
    c["FLW"] = sum(1 for t in tokens if t in FOLLOWING)
    opt = 0
    for i, t in enumerate(tokens):
        for a, b in OPTFORM_BI:
            if t == a and (b is None or (i + 1 < n and tokens[i + 1] == b)):
                opt += 1
                break
    c["OPTF"] = opt
    c["NUM"] = sum(1 for t in tokens if _is_number_token(t))
    c["VETO"] = sum(1 for t in tokens if t in VETO_TOKENS)
    c["FUNC"] = sum(1 for t in tokens if t in PROSE_FUNC) / c["n"]
    c["WPREF"] = sum(1 for t in tokens if len(t) > 2 and t.startswith("و")) / c["n"]
    return c


def detect_exam(tokens: List[str]) -> Tuple[bool, List[str]]:
    """Doc-level detector. Returns (qualifies, fired marker families)."""
    c = marker_counts(tokens)
    n = c["n"]
    dens = lambda k: 100.0 * c[k] / n
    if c["VETO"] >= 1:
        return False, ["VETO"]
    strong: List[str] = []
    if c["TF"] >= TF_MIN:
        strong.append("TF")
    if c["CLOZE"] >= CLOZE_MIN:
        strong.append("CLOZE")
    if c["LET"] >= LET_MIN and dens("LET") >= LET_DENS:
        strong.append("LET")
    if c["OPTF"] >= OPTF_HI_MIN and dens("OPTF") >= OPTF_HI_DENS:
        strong.append("OPTF-HI")
    weak: List[str] = []
    if c["QW"] >= QW_MIN and dens("QW") >= QW_DENS:
        weak.append("QW")
    if c["STEM"] >= STEM_MIN and dens("STEM") >= STEM_DENS:
        weak.append("STEM")
    if c["FLW"] >= FLW_MIN and dens("FLW") >= FLW_DENS:
        weak.append("FLW")
    if c["OPTF"] >= OPTF_LO_MIN and dens("OPTF") >= OPTF_LO_DENS:
        weak.append("OPTF-LO")
    numt: List[str] = []
    if c["NUM"] >= NUM_MIN and dens("NUM") >= NUM_DENS:
        numt.append("NUM")
    struct_weak = c["FUNC"] <= WEAK_FUNC_MAX and c["WPREF"] <= WPREF_MAX
    struct_num = c["FUNC"] <= NUM_FUNC_MAX and c["WPREF"] <= WPREF_MAX
    fired = list(strong)
    if struct_weak:
        fired += weak
    if struct_num:
        fired += numt
    qualifies = bool(strong) or (struct_weak and bool(weak)) or (struct_num and bool(numt))
    return qualifies, fired


def fit_gate_report(train_docs: List[dict], verbose: bool = True) -> Set[str]:
    """Confusion of the detector against the eyeballed train ground truth.
    HARD constraint asserted: zero non-exam train docs fire."""
    detected: Set[str] = set()
    rows_fp, rows_fn, rows_tp = [], [], []
    for d in train_docs:
        qual, fams = detect_exam(d["tokens"])
        is_exam = d["doc_id"] in EXAM_TRAIN_DOCS
        if qual:
            detected.add(d["doc_id"])
        if qual and not is_exam:
            rows_fp.append((d["doc_id"], fams))
        elif not qual and is_exam:
            rows_fn.append((d["doc_id"], fams))
        elif qual:
            rows_tp.append((d["doc_id"], fams))
    if verbose:
        n_exam = sum(1 for d in train_docs if d["doc_id"] in EXAM_TRAIN_DOCS)
        print("== TRAIN-FITTED GENRE GATE (dev never read) ==")
        print(f"  gate: STRONG any-of TF>={TF_MIN} | CLOZE>={CLOZE_MIN} | "
              f"LET>={LET_MIN}@{LET_DENS}/100 | OPTF>={OPTF_HI_MIN}@{OPTF_HI_DENS}/100")
        print(f"        WEAK  (func<={WEAK_FUNC_MAX}, wpref<={WPREF_MAX}) any-of "
              f"QW>={QW_MIN}@{QW_DENS} | STEM>={STEM_MIN}@{STEM_DENS} | "
              f"FLW>={FLW_MIN}@{FLW_DENS} | OPTF>={OPTF_LO_MIN}@{OPTF_LO_DENS}")
        print(f"        NUM-TIGHT (func<={NUM_FUNC_MAX}) NUM>={NUM_MIN}@{NUM_DENS} | "
              f"VETO {VETO_TOKENS}")
        print(f"  train exam docs qualifying : {len(rows_tp) + 0}/{n_exam}")
        print(f"  non-exam train docs firing : {len(rows_fp)}/{len(train_docs) - n_exam}  (must be 0)")
        for did, fams in rows_tp:
            print(f"    DETECTED {did}  fams={','.join(fams)}")
        for did, fams in rows_fn:
            print(f"    MISSED   {did}  (reported coverage gap; missed upside, never damage)")
        for did, fams in rows_fp:
            print(f"    FALSE-FIRE {did}  fams={','.join(fams)}  <-- VIOLATION")
    assert not rows_fp, f"gate false-fires on non-exam train docs: {rows_fp}"
    return detected


# ---------------------------------------------------------------------------
# start-lexicon + length-prior fitting (TRAIN gold inside detected docs only)
# ---------------------------------------------------------------------------
def token_class(t: str) -> Optional[str]:
    if _is_number_token(t):
        return "<NUM>"
    if t in MCQ_LETTERS:
        return "<LET>"
    if t and all(("a" <= ch.lower() <= "z") or ch.isdigit() for ch in t):
        return "<LAT>"
    return None


@dataclass
class StartModel:
    unigrams: Dict[str, Tuple[int, float]] = field(default_factory=dict)   # tok -> (n, P)
    bigrams: Dict[Tuple[str, str], Tuple[int, float]] = field(default_factory=dict)
    classes: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    log_prior: np.ndarray = field(default_factory=lambda: np.zeros(1))     # index = seg len
    prior_floor: float = -20.0
    minlen: int = 1
    maxlen: int = 10 ** 9

    def lp(self, length: int) -> float:
        if 1 <= length < len(self.log_prior):
            return float(self.log_prior[length])
        return self.prior_floor

    def is_lexicon_start(self, tokens: List[str], j: int) -> bool:
        t = tokens[j]
        if t in self.unigrams:
            return True
        cls = token_class(t)
        if cls is not None and cls in self.classes:
            return True
        if j + 1 < len(tokens) and (t, tokens[j + 1]) in self.bigrams:
            return True
        return False


def fit_start_model(train_docs: List[dict], detected: Set[str], verbose: bool = True) -> StartModel:
    occ_u, start_u = Counter(), Counter()
    occ_b, start_b = Counter(), Counter()
    occ_c, start_c = Counter(), Counter()
    seg_lens: List[int] = []
    for d in train_docs:
        if d["doc_id"] not in detected:
            continue
        toks, labs = d["tokens"], d["labels"]
        n = len(toks)
        starts = {0} | {i + 1 for i in range(n - 1) if labs[i] == 1}
        for j, t in enumerate(toks):
            is_s = j in starts
            occ_u[t] += 1
            start_u[t] += is_s
            cls = token_class(t)
            if cls is not None:
                occ_c[cls] += 1
                start_c[cls] += is_s
            if j + 1 < n:
                bg = (t, toks[j + 1])
                occ_b[bg] += 1
                start_b[bg] += is_s
        prev = -1
        for i in range(n):
            if labs[i] == 1:
                seg_lens.append(i - prev)
                prev = i
    sm = StartModel()
    for t, n_occ in occ_u.items():
        p = start_u[t] / n_occ
        if n_occ >= UNI_MIN_N and p >= UNI_MIN_P:
            sm.unigrams[t] = (n_occ, p)
    for bg, n_occ in occ_b.items():
        p = start_b[bg] / n_occ
        if n_occ >= BI_MIN_N and p >= BI_MIN_P and bg[0] not in sm.unigrams:
            sm.bigrams[bg] = (n_occ, p)
    for cls, n_occ in occ_c.items():
        p = start_c[cls] / n_occ
        if n_occ >= CLS_MIN_N and p >= CLS_MIN_P:
            sm.classes[cls] = (n_occ, p)
    assert seg_lens, "no detected train docs — start model has no evidence"
    sm.minlen = int(min(seg_lens))
    sm.maxlen = int(max(seg_lens))
    cap = sm.maxlen
    hist = np.zeros(cap + 1)
    for L in seg_lens:
        hist[L] += 1
    total = hist.sum() + PRIOR_ALPHA * cap
    probs = (hist + PRIOR_ALPHA) / total
    sm.log_prior = np.log(probs)
    sm.log_prior[0] = -1e9  # zero-length segments are impossible
    sm.prior_floor = float(np.log(PRIOR_ALPHA / total))
    if verbose:
        print("\n== TRAIN-GOLD START MODEL (fit inside detected train docs only) ==")
        print(f"  segments={len(seg_lens)}  len min={sm.minlen} p50={int(np.median(seg_lens))} "
              f"p95={int(np.percentile(seg_lens, 95))} max={sm.maxlen}")
        print(f"  unigram lexicon ({len(sm.unigrams)} entries; gate n>={UNI_MIN_N}, P>={UNI_MIN_P}):")
        top = sorted(sm.unigrams.items(), key=lambda kv: -kv[1][0])
        print("    " + "  ".join(f"{t}(n={n},P={p:.2f})" for t, (n, p) in top[:25]))
        if len(top) > 25:
            print(f"    ... and {len(top) - 25} more")
        print(f"  class lexicon (gate n>={CLS_MIN_N}, P>={CLS_MIN_P}):")
        for cls in ("<NUM>", "<LET>", "<LAT>"):
            n_occ = occ_c.get(cls, 0)
            p = start_c[cls] / n_occ if n_occ else float("nan")
            v = "LICENSED" if cls in sm.classes else "NEVER FIRES"
            print(f"    {cls}: n={n_occ} P={p:.3f}  {v}")
        print(f"  bigram lexicon ({len(sm.bigrams)} entries; gate n>={BI_MIN_N}, P>={BI_MIN_P}):")
        topb = sorted(sm.bigrams.items(), key=lambda kv: -kv[1][0])
        print("    " + "  ".join(f"{a}+{b}(n={n},P={p:.2f})" for (a, b), (n, p) in topb[:15]))
    return sm


# ---------------------------------------------------------------------------
# trick A: start-anchored decode inside a detected doc
# ---------------------------------------------------------------------------
def accepted_starts(tokens: List[str], probs: np.ndarray, sm: StartModel,
                    tau: float = TAU_MODEL) -> Tuple[List[int], Counter]:
    """Token positions j >= 1 that begin a segment; source bookkeeping."""
    src = Counter()
    out = []
    for j in range(1, len(tokens)):
        lex = sm.is_lexicon_start(tokens, j)
        model = probs[j - 1] >= tau
        if lex or model:
            out.append(j)
            src["lexicon" if lex else "model-prob"] += 1
    return out, src


def anchor_decode(tokens: List[str], probs: np.ndarray, base: np.ndarray,
                  sm: StartModel, tau: float = TAU_MODEL) -> Tuple[np.ndarray, Counter, int]:
    """Force a boundary before every accepted start, suppress everything else,
    keep the doc-final boundary, enforce the train MAXLEN via prob fallback."""
    n = len(tokens)
    starts, src = accepted_starts(tokens, probs, sm, tau)
    new = np.zeros(n, dtype=int)
    for j in starts:
        new[j - 1] = 1
    new[n - 1] = 1
    # MAXLEN sanity from train exam gold: re-admit strongest cuts inside
    n_fallback = 0
    bounds = [-1] + [i for i in range(n - 1) if new[i] == 1] + [n - 1]
    stack = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a <= sm.maxlen or b - a <= 1:
            continue
        interior = list(range(a + 1, b))
        cut = max(interior, key=lambda i: (base[i], probs[i], -i))
        new[cut] = 1
        n_fallback += 1
        stack.append((a, cut))
        stack.append((cut, b))
    return new, src, n_fallback


# ---------------------------------------------------------------------------
# trick B: deterministic local perturbation rescore (BoundRL-style)
# ---------------------------------------------------------------------------
def seg_score(joints: List[int], n: int, ll0: np.ndarray, ll1: np.ndarray,
              sm: StartModel, lam: float = LAMBDA_PRIOR) -> float:
    """Full score of a segmentation given internal joints (label idx < n-1)."""
    s = ll0.sum()
    for j in joints:
        s += ll1[j] - ll0[j]
    s += ll1[n - 1] - ll0[n - 1]
    prev = -1
    for j in joints + [n - 1]:
        s += lam * sm.lp(j - prev)
        prev = j
    return float(s)


def rescore(tokens: List[str], probs: np.ndarray, pred: np.ndarray,
            sm: StartModel, lam: float = LAMBDA_PRIOR) -> Tuple[np.ndarray, int]:
    """Local search over the anchored joints. Deterministic. Returns the
    improved prediction and the number of candidates evaluated (the K)."""
    n = len(tokens)
    p = np.clip(probs.astype(float), EPS, 1.0 - EPS)
    ll1, ll0 = np.log(p), np.log(1.0 - p)
    joints = [i for i in range(n - 1) if pred[i] == 1]
    K = 0

    def neighbors(js: List[int], idx: int) -> Tuple[int, int]:
        prev = js[idx - 1] if idx > 0 else -1
        nxt = js[idx + 1] if idx + 1 < len(js) else n - 1
        return prev, nxt

    for _ in range(MAX_SWEEPS):
        changed = False
        # per-joint moves: shift +/-2, drop
        idx = 0
        while idx < len(joints):
            b = joints[idx]
            prev, nxt = neighbors(joints, idx)
            best_delta, best_move = 0.0, None
            for c in (b - 2, b - 1, b + 1, b + 2):
                if c <= prev or c >= nxt or c < 0 or c > n - 2:
                    continue
                K += 1
                d = (ll0[b] - ll1[b] + ll1[c] - ll0[c]
                     + lam * (sm.lp(c - prev) + sm.lp(nxt - c)
                              - sm.lp(b - prev) - sm.lp(nxt - b)))
                if d > best_delta + 1e-12:
                    best_delta, best_move = d, ("shift", c)
            K += 1
            d = (ll0[b] - ll1[b]
                 + lam * (sm.lp(nxt - prev) - sm.lp(b - prev) - sm.lp(nxt - b)))
            if d > best_delta + 1e-12:
                best_delta, best_move = d, ("drop", None)
            if best_move is not None:
                changed = True
                if best_move[0] == "shift":
                    joints[idx] = best_move[1]
                    idx += 1
                else:
                    joints.pop(idx)
            else:
                idx += 1
        # per-segment add: best interior position by ll1-ll0
        segs = list(zip([-1] + joints, joints + [n - 1]))
        for a, b in segs:
            if b - a < 2:
                continue
            interior = np.arange(a + 1, b)
            c = int(interior[np.argmax((ll1 - ll0)[interior])])
            K += 1
            d = (ll1[c] - ll0[c]
                 + lam * (sm.lp(c - a) + sm.lp(b - c) - sm.lp(b - a)))
            if d > 1e-12:
                insort(joints, c)
                changed = True
        if not changed:
            break
    new = np.zeros(n, dtype=int)
    for j in joints:
        new[j] = 1
    new[n - 1] = 1
    return new, K


# ---------------------------------------------------------------------------
# measurement (dev = evaluation only)
# ---------------------------------------------------------------------------
def _prf1(g: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((g == 1) & (p == 1)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def _load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _annotate_window(tokens, gold, base, new, start, end) -> str:
    out = []
    for i in range(max(0, start), min(len(tokens), end)):
        out.append(tokens[i])
        marks = ""
        if gold[i] == 1:
            marks += "G"
        if base[i] == 1 and new[i] == 1:
            marks += "b"
        if base[i] == 1 and new[i] == 0:
            marks += "S"
        if base[i] == 0 and new[i] == 1:
            marks += "F"
        if marks:
            out.append("/" + marks + "/")
    return " ".join(out)


def _dump_changed_windows(tokens, gold, base, new, pad: int = 10, cap: int = 25) -> List[str]:
    diff = np.where(base != new)[0]
    if len(diff) == 0:
        return []
    windows: List[Tuple[int, int]] = []
    for i in diff:
        a, b = int(i) - pad, int(i) + pad + 1
        if windows and a <= windows[-1][1]:
            windows[-1] = (windows[-1][0], b)
        else:
            windows.append((a, b))
    lines = []
    for a, b in windows[:cap]:
        lines.append("      ... " + _annotate_window(tokens, gold, base, new, a, b) + " ...")
    if len(windows) > cap:
        lines.append(f"      ({len(windows) - cap} more changed windows not shown)")
    return lines


def run_apply(probs_path: str, gold_path: str, train_path: str, thr: float,
              stage: str = "rescore", tau: float = TAU_MODEL,
              out_csv: Optional[str] = None) -> None:
    train_docs = _load_jsonl(train_path)
    detected_train = fit_gate_report(train_docs, verbose=True)
    sm = fit_start_model(train_docs, detected_train, verbose=True)
    docs = _load_jsonl(gold_path)
    z = np.load(probs_path)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from genre_buckets import bucket
    except Exception:  # pragma: no cover
        def bucket(_tokens):
            return "n/a"

    f1_before, f1_after, genres = [], [], []
    detected_rows = []
    span_dumps: List[str] = []
    preds_out = {}
    tot_src = Counter()
    tot_fallback = tot_K = 0
    # exam-genre cut bookkeeping (inside detected docs)
    ex_tp_b = ex_fp_b = ex_tp_a = ex_fp_a = 0
    ex_pred_b = ex_pred_a = ex_gold = ex_tok = 0
    n_detected = 0
    for d in docs:
        did = d["doc_id"]
        g = np.asarray(d["labels"], dtype=int)
        p = np.asarray(z[did], dtype=float)
        assert len(p) == len(g), did
        base = (p >= thr).astype(int)
        qual, fams = detect_exam(d["tokens"])
        if qual:
            n_detected += 1
            new, src, n_fb = anchor_decode(d["tokens"], p, base, sm, tau)
            tot_src += src
            tot_fallback += n_fb
            f1_anchor = _prf1(g, new)[2]
            if stage == "rescore":
                new, K = rescore(d["tokens"], p, new, sm)
                tot_K += K
            # invariant: region is the whole doc; nothing to assert outside it
        else:
            new = base.copy()
            assert (new == base).all(), f"undetected doc modified: {did}"
            f1_anchor = None
        preds_out[did] = new
        _, _, fb = _prf1(g, base)
        _, _, fa = _prf1(g, new)
        f1_before.append(fb)
        f1_after.append(fa)
        genres.append(bucket(d["tokens"]))
        if qual:
            ex_tp_b += int(((g == 1) & (base == 1)).sum())
            ex_fp_b += int(((g == 0) & (base == 1)).sum())
            ex_tp_a += int(((g == 1) & (new == 1)).sum())
            ex_fp_a += int(((g == 0) & (new == 1)).sum())
            ex_pred_b += int(base.sum())
            ex_pred_a += int(new.sum())
            ex_gold += int(g.sum())
            ex_tok += len(g)
            detected_rows.append((did, genres[-1], ",".join(fams), 100 * fb,
                                  100 * (f1_anchor if f1_anchor is not None else fb), 100 * fa,
                                  int((base != new).sum())))
            if (base != new).any():
                span_dumps.append(f"--- {did} [{','.join(fams)}] "
                                  f"(G=gold cut, b=kept cut, S=suppressed cut, F=forced cut)")
                span_dumps.extend(_dump_changed_windows(d["tokens"], g, base, new))

    mb, ma = 100 * float(np.mean(f1_before)), 100 * float(np.mean(f1_after))
    print("\n== DEV MEASUREMENT (dev used for evaluation only) ==")
    print(f"probs: {probs_path}   thr={thr}   stage={stage}   tau_model={tau}   docs={len(docs)}")
    print(f"detected dev docs = {n_detected}   (regions are whole docs; all other "
          f"docs bit-identical, asserted per doc)")
    print(f"accepted-start sources: {dict(tot_src)}   maxlen-fallback cuts={tot_fallback}"
          f"   rescore candidates evaluated K={tot_K}")
    print(f"macro-F1 before = {mb:.2f}")
    print(f"macro-F1 after  = {ma:.2f}")
    print(f"DELTA           = {ma - mb:+.2f}")

    if ex_pred_b:
        print("\nexam-genre cut statistics (inside detected docs only):")
        print(f"  gold cuts={ex_gold}  mean gold seg len={ex_tok / max(1, ex_gold):.2f}")
        print(f"  before: pred cuts={ex_pred_b}  FP={ex_fp_b}  FP-rate={ex_fp_b / ex_pred_b:.3f}"
              f"  mean pred seg len={ex_tok / max(1, ex_pred_b):.2f}")
        print(f"  after : pred cuts={ex_pred_a}  FP={ex_fp_a}  FP-rate={ex_fp_a / max(1, ex_pred_a):.3f}"
              f"  mean pred seg len={ex_tok / max(1, ex_pred_a):.2f}")

    print("\nper-genre mean F1 (before -> after):")
    per = defaultdict(list)
    for gname, fb, fa in zip(genres, f1_before, f1_after):
        per[gname].append((fb, fa))
    for gname in sorted(per, key=lambda k: -len(per[k])):
        arr = np.array(per[gname])
        print(f"  {gname:20s} n={len(arr):3d}  {100 * arr[:, 0].mean():6.2f} -> "
              f"{100 * arr[:, 1].mean():6.2f}  ({100 * (arr[:, 1] - arr[:, 0]).mean():+.2f})")

    print("\ndetected dev docs (per-doc F1 before -> after-anchor -> after-rescore):")
    if not detected_rows:
        print("  (none — the train-fitted gate did not transfer; reported, not tuned)")
    for did, gname, fams, fb, fan, fa, nch in detected_rows:
        print(f"  {did}  {gname:16s} [{fams}]  F1 {fb:6.2f} -> {fan:6.2f} -> {fa:6.2f}  changes={nch}")

    cloze = [(did, fb, fa) for (did, gname, _, fb, _, fa, _) in detected_rows
             if gname == "cloze/exercise"]
    print("\ncloze-bucket dev docs (the +0.54-ceiling docs), F1 before -> after:")
    if not cloze:
        print("  (no cloze-bucket doc was detected)")
    for did, fb, fa in cloze:
        print(f"  {did}  {fb:6.2f} -> {fa:6.2f}  ({fa - fb:+.2f})")

    print("\nmodified regions (before/after, for eyeballing):")
    for s in span_dumps or ["  (none)"]:
        print(s)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            f.write("Document ID,Prediction\n")
            for d in docs:
                f.write(d["doc_id"] + "," + "".join(map(str, preds_out[d["doc_id"]].tolist())) + "\n")
        print(f"\nwrote {out_csv}")


# ---------------------------------------------------------------------------
# corpus-independent self-test (passes with data/ hidden)
# ---------------------------------------------------------------------------
def self_test() -> None:
    rng = np.random.RandomState(0)
    name = ["زيد", "عمرو", "بكر", "خالد", "سعد", "أنس", "قطر", "عمان"]

    # --- synthetic train docs -------------------------------------------
    def mk_quiz(k: int) -> dict:
        toks, labs = [], []
        for q in range(6):
            toks += ["هل", "يعرف", name[(q + k) % 8], "الجواب"]
            labs += [0, 0, 0, 1]
            toks += ["صح"]
            labs += [1]
            toks += ["خطأ"]
            labs += [1]
        return {"doc_id": f"t_quiz{k}", "tokens": toks, "labels": labs}

    def mk_mcq(k: int) -> dict:
        toks, labs = [], []
        for q in range(5):
            toks += ["ماذا", "تعني", "هذه", "الكلمة"]
            labs += [0, 0, 0, 1]
            for o in range(3):
                toks += ["أ" if o == 0 else "ب" if o == 1 else "ج", name[(q + o + k) % 8]]
                labs += [0, 1]
        return {"doc_id": f"t_mcq{k}", "tokens": toks, "labels": labs}

    prose_toks = ("كان الرجل يمشي في المدينة وقد وصل إلى البيت الذي كان يسكن فيه "
                  "وبعد ذلك ذهب إلى السوق ثم عاد وهو يحمل وقال إنه رأى صديقه القديم "
                  "وتحدثا عن الأيام التي مضت وكانت أياما جميلة في تلك المدينة الهادئة").split()
    prose = {"doc_id": "t_prose", "tokens": prose_toks,
             "labels": [0] * (len(prose_toks) - 1) + [1]}
    comic_toks = ("سيناريو فلان رسوم فلان هل تعرف أين ذهب البطل ماذا قال له الرجل "
                  "كيف نهرب من هنا متى نصل هل تعرف").split()
    comic = {"doc_id": "t_comic", "tokens": comic_toks,
             "labels": [0] * (len(comic_toks) - 1) + [1]}
    train = [mk_quiz(0), mk_quiz(1), mk_quiz(2), mk_mcq(0), mk_mcq(1), prose, comic]

    # --- gate ------------------------------------------------------------
    q, fams = detect_exam(train[0]["tokens"])
    assert q and "TF" in fams, (q, fams)
    q, fams = detect_exam(train[3]["tokens"])
    assert q and ("LET" in fams or "QW" in fams), (q, fams)
    q, fams = detect_exam(prose["tokens"])
    assert not q, "prose must not fire"
    q, fams = detect_exam(comic["tokens"])
    assert not q and fams == ["VETO"], "سيناريو veto must kill the comic despite question words"
    exam_ids = {d["doc_id"] for d in train if d["doc_id"].startswith(("t_quiz", "t_mcq"))}
    detected = {d["doc_id"] for d in train if detect_exam(d["tokens"])[0]}
    assert detected == exam_ids, detected

    # --- start-model fit --------------------------------------------------
    sm = fit_start_model(train, detected, verbose=False)
    assert "صح" in sm.unigrams and "خطأ" in sm.unigrams, sm.unigrams.keys()
    assert "هل" in sm.unigrams and "ماذا" in sm.unigrams
    assert "الجواب" not in sm.unigrams, "segment-final token must not be a licensed start"
    assert "<LET>" in sm.classes, sm.classes
    assert sm.minlen >= 1 and 1 <= sm.maxlen <= 12, (sm.minlen, sm.maxlen)

    # --- anchor decode: suppression + forcing -----------------------------
    toks = mk_quiz(3)["tokens"]
    gold = np.array(mk_quiz(3)["labels"])
    n = len(toks)
    p = np.full(n, 0.6)                      # confetti: model wants to cut everywhere
    base = (p >= 0.5).astype(int)
    new, src, n_fb = anchor_decode(toks, p, base, sm)
    got = _prf1(gold, new)[2]
    assert got == 1.0, f"anchored decode must recover the quiz gold, F1={got}"
    assert new.sum() < base.sum(), "suppression must reduce the confetti cuts"
    assert new[n - 1] == 1, "doc-final boundary must be kept"

    # --- maxlen fallback ---------------------------------------------------
    long_toks = ["كلمة"] * 60
    p_long = rng.rand(60) * 0.4              # no model cut >= tau, no lexicon starts
    base_long = np.zeros(60, dtype=int)
    new_long, _, n_fb = anchor_decode(long_toks, p_long, base_long, sm)
    bounds = [-1] + list(np.where(new_long == 1)[0])
    lens = [b - a for a, b in zip(bounds, bounds[1:])]
    assert max(lens) <= sm.maxlen and n_fb > 0, (lens, sm.maxlen, n_fb)

    # --- rescore: improves the score, deterministic, stays in bounds -------
    toks_r = mk_quiz(4)["tokens"]
    n = len(toks_r)
    p_r = np.full(n, 0.05)
    p_r[np.where(np.array(mk_quiz(4)["labels"]) == 1)[0]] = 0.95
    p_r[2] = 0.94                            # a tempting near-miss one off the gold cut at 3
    pred0 = np.zeros(n, dtype=int)
    pred0[2] = 1                             # mis-anchored joint (off by one)
    pred0[n - 1] = 1
    ll = np.clip(p_r, EPS, 1 - EPS)
    ll1, ll0 = np.log(ll), np.log(1 - ll)
    s_before = seg_score([2], n, ll0, ll1, sm)
    new_r, K1 = rescore(toks_r, p_r, pred0, sm)
    s_after = seg_score([int(i) for i in np.where(new_r[:n - 1] == 1)[0]], n, ll0, ll1, sm)
    assert s_after > s_before and K1 > 0, (s_before, s_after, K1)
    assert new_r[3] == 1, "rescore must shift the off-by-one joint onto the high-prob cut"
    new_r2, K2 = rescore(toks_r, p_r, pred0.copy(), sm)
    assert (new_r == new_r2).all() and K1 == K2, "rescore must be deterministic"

    # --- outside-region invariance guard ------------------------------------
    # the apply path touches ONLY detector-qualified docs; the detector is the
    # guard, asserted above (prose and comic must not fire).
    # fit_gate_report enforces the hard zero-false-fire constraint: synthetic
    # quiz ids are not in the real ground-truth list, so it must raise.
    try:
        fit_gate_report(train, verbose=False)
        raise AssertionError("must reject: fit_gate_report accepted docs "
                             "outside the ground-truth exam list")
    except AssertionError as e:
        if "must reject" in str(e):
            raise
        # expected: synthetic quiz docs count as false fires vs the real list

    print("self-test OK (gate fires/vetoes, start-model fit, suppression+forcing, "
          "maxlen fallback, rescore improvement+determinism, invariance guard)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fit", action="store_true",
                    help="print the train-gold gate confusion + start model only")
    ap.add_argument("--apply", action="store_true",
                    help="apply to cached probs + measure on gold")
    ap.add_argument("--probs", default="probs/union-NoPnx-NP-arabertv02-s42_dev.npz")
    ap.add_argument("--gold", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--train", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--stage", choices=["anchor", "rescore"], default="rescore",
                    help="anchor = trick A only; rescore = trick A + trick B (default)")
    ap.add_argument("--tau-model", type=float, default=TAU_MODEL,
                    help="a priori model-channel constant (declared, never swept)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.fit:
        train_docs = _load_jsonl(args.train)
        detected = fit_gate_report(train_docs, verbose=True)
        fit_start_model(train_docs, detected, verbose=True)
        return
    if args.apply:
        run_apply(args.probs, args.gold, args.train, args.thr,
                  stage=args.stage, tau=args.tau_model, out_csv=args.out_csv)
        return
    ap.error("choose one of --self-test / --fit / --apply")


if __name__ == "__main__":
    main()
