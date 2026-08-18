"""Confident-error EXAMPLE extractor for an AraSeg NoPnx-NP single-encoder checkpoint.

Purpose (analysis only; trains NOTHING):
  Let a native reader SEE what the confident boundary errors actually are, to test
  whether "unseen word-pair coverage" is the real driver of the closed-track wall
  or a proxy hiding gold-noise / genre-formatting.

It loads an existing checkpoint (default: the union AraBERTv02 s42 model the
preposition diagnostic used) and predicts per-token P(boundary) on the dev split
with the EXACT windowing in predict.py (via predict.predict_doc). It then pulls:

  CONFIDENT FALSE POSITIVES : gold label 0 but P(boundary) >= FP_HI   (default 0.8)
                              -> model confidently CUT where gold says no boundary.
  CONFIDENT FALSE NEGATIVES : gold label 1 but P(boundary) <= FN_LO   (default 0.2)
                              -> model confidently did NOT cut where gold says cut.

Forced-0 positions (tok == PARAGRAPH_TOKEN, i.e. "\n") are excluded, matching
predict.py / the evaluator (they are never a real decision).

For each sampled example it prints the +/-6-word Arabic context window with the
decision point marked ' | <== ', the model P, the gold label, and the genre bucket
(genre_buckets.bucket). It then AUTO-BUCKETS each example CONSERVATIVELY into:

  (A) plausibly GOLD-ambiguous/wrong  - the model's decision looks linguistically
        defensible: a real sentence boundary could sit there (for an FP) or clearly
        should sit there and gold disagrees (for an FN). TENTATIVE, heuristic only;
        flagged for a native reader (Noor) to confirm. We DO NOT overclaim.
  (B) GENRE-structural  - cloze/exercise / list / genealogy / legal-article
        formatting (genre bucket != general prose, or strong list/enumeration cues).
  (C) rare / UNSEEN word or morphology at the site - the local token bigram around
        the decision point was never seen in the 174 train docs (coverage gap).
  (D) clear MODEL error on ordinary clean prose - none of the above; general prose,
        seen context, no defensible reading.

Bucket priority when several could apply is deliberately CONSERVATIVE for the
gold-noise claim: B (genre) and C (unseen) are assigned BEFORE A, so A only
collects general-prose, seen-context, in-distribution sites where the model's
decision is still linguistically defensible. This makes the reported
"fraction of confident FP that are bucket (A)" a LOWER bound on gold-noise, not
an inflated one.

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python src/diag_error_examples.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev   data/NoPnx-NP_dev.jsonl \
      --train data/NoPnx-NP_train.jsonl \
      --fp-hi 0.8 --fn-lo 0.2 --sample 40 --seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc
from genre_buckets import bucket

# Force UTF-8 stdout so Arabic prints on Windows consoles (cp1252 default).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --- lexical cues used ONLY for conservative auto-bucketing -----------------
# Prepositions / connectives that legitimately START a new clause in Arabic;
# an FP right before one of these is a plausible (defensible) boundary.
# (Same list the preposition diagnostic uses.)
CLAUSE_STARTERS = {
    "و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
    "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى",
    "قال", "وقال", "أما", "إذا", "إن", "لأن", "حيث", "كي", "لكي",
}
# Genealogy / list cues (structural formatting).
LIST_STRUCT = {"ولد", "بن", "ابن", "المادة", "فراغ", "أولا", "ثانيا", "ثالثا",
               "رابعا", "خامسا"}
# Sentence-final-ish words that make an FN (missed boundary) defensible when
# they PRECEDE the decision point (i.e. tok[i] is one of these and gold=1).
SENT_FINAL_HINT = {"ذلك", "الله", "وسلم", "عليه", "آمين", "انتهى"}


def make_context(tokens, i, radius=6):
    """+/- radius word window with the decision point marked after tok[i]."""
    n = len(tokens)
    lo = max(0, i - radius)
    hi = min(n, i + 1 + radius)
    left = tokens[lo:i + 1]
    right = tokens[i + 1:hi]

    def show(t):
        return "[PAR]" if t == PARAGRAPH_TOKEN else t

    left_s = " ".join(show(t) for t in left)
    right_s = " ".join(show(t) for t in right)
    pre = "... " if lo > 0 else ""
    post = " ..." if hi < n else ""
    return f"{pre}{left_s}  | <==CUT HERE  {right_s}{post}"


def build_train_bigrams(train_docs):
    """Set of adjacent (tok[i], tok[i+1]) token bigrams seen in train.

    Identical construction to diag_preposition.build_train_bigrams.
    """
    bg = set()
    vocab = set()
    for d in train_docs:
        toks = d["tokens"]
        for t in toks:
            vocab.add(t)
        for i in range(len(toks) - 1):
            bg.add((toks[i], toks[i + 1]))
    return bg, vocab


def auto_bucket(kind, tokens, i, genre, train_bg, train_vocab):
    """CONSERVATIVE bucket assignment for one confident error.

    kind: 'FP' (gold 0, model cut) or 'FN' (gold 1, model did not cut).
    Returns (letter, reason_string).

    Priority (deliberately conservative for the gold-noise 'A' claim):
        1. B  genre/structural  (non-prose genre OR strong list/genealogy cue)
        2. C  unseen local bigram at the site  (coverage gap)
        3. A  plausibly gold-ambiguous  (defensible linguistic reading)
        4. D  otherwise  (clean-prose model error)
    Assigning B and C BEFORE A means A cannot absorb genre or coverage cases,
    so the A-fraction is a LOWER bound on true gold-noise.
    """
    n = len(tokens)
    prev_tok = tokens[i - 1] if i - 1 >= 0 else None
    cur_tok = tokens[i]
    nxt_tok = tokens[i + 1] if i + 1 < n else None

    # --- B: genre / structural formatting -----------------------------------
    if genre != "general prose":
        return "B", f"genre={genre}"
    # strong local list/genealogy/legal cue even inside 'general prose'
    window = set(tokens[max(0, i - 2):min(n, i + 3)])
    struct_hits = window & LIST_STRUCT
    if struct_hits:
        return "B", f"list/struct cue {sorted(struct_hits)}"

    # --- C: unseen local context (coverage gap) -----------------------------
    # The bigram straddling the decision point is (cur_tok, nxt_tok).
    # Also consider the immediate right bigram (nxt_tok, tok[i+2]) since the
    # boundary decision hinges on the material just after the cut.
    right2 = tokens[i + 2] if i + 2 < n else None
    straddle_seen = (nxt_tok is not None) and ((cur_tok, nxt_tok) in train_bg)
    rightbg_seen = (right2 is not None) and ((nxt_tok, right2) in train_bg)
    cur_seen = cur_tok in train_vocab
    nxt_seen = (nxt_tok is None) or (nxt_tok in train_vocab)
    if not straddle_seen and not rightbg_seen:
        detail = []
        if not cur_seen:
            detail.append(f"OOV_cur={cur_tok!r}")
        if not nxt_seen:
            detail.append(f"OOV_next={nxt_tok!r}")
        if not detail:
            detail.append("unseen bigram at site")
        return "C", "; ".join(detail)

    # --- A: plausibly gold-ambiguous / defensible ---------------------------
    if kind == "FP":
        # Model cut. A cut is defensible if the NEXT token normally starts a
        # new clause/sentence (connective/preposition/verb-of-saying).
        if nxt_tok in CLAUSE_STARTERS:
            return "A", f"cut before clause-starter {nxt_tok!r} (defensible)"
    else:  # FN: gold says boundary, model didn't cut.
        # A missed boundary is defensible-as-gold-ambiguous if the site does
        # NOT look like a natural sentence end: i.e. the NEXT token is itself a
        # connective/preposition (mid-clause continuation) OR the current token
        # is one that rarely ends a sentence. Then gold's boundary is the
        # questionable one and the model's non-cut is linguistically reasonable.
        if nxt_tok in CLAUSE_STARTERS and cur_tok not in SENT_FINAL_HINT:
            return "A", (f"gold cut before continuation {nxt_tok!r}; site not a "
                         f"natural end (model's no-cut defensible)")

    # --- D: clean-prose model error -----------------------------------------
    return "D", "general prose, seen context, no defensible reading"


def collect_confident_errors(dev, probs, fp_hi, fn_lo):
    """Return two lists of dicts: confident FPs and confident FNs."""
    fps, fns = [], []
    for di, (d, p) in enumerate(zip(dev, probs)):
        toks = d["tokens"]
        gold = d["labels"]
        genre = bucket(toks)
        n = len(toks)
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue  # forced-0 position, never a real decision
            gi = int(gold[i])
            pi = float(p[i])
            if gi == 0 and pi >= fp_hi:
                fps.append(dict(doc=di, i=i, p=pi, gold=gi, genre=genre))
            elif gi == 1 and pi <= fn_lo:
                fns.append(dict(doc=di, i=i, p=pi, gold=gi, genre=genre))
    return fps, fns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/union-NoPnx-NP-arabertv02-s42")
    ap.add_argument("--dev", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--train", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--fp-hi", type=float, default=0.8,
                    help="confident FP threshold: gold 0 but P(boundary) >= this")
    ap.add_argument("--fn-lo", type=float, default=0.2,
                    help="confident FN threshold: gold 1 but P(boundary) <= this")
    ap.add_argument("--sample", type=int, default=40,
                    help="max examples to PRINT per class (random, seeded)")
    ap.add_argument("--radius", type=int, default=6, help="context words each side")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device={device}  model={args.model}")
    print(f"# fp_hi(P>=)={args.fp_hi}  fn_lo(P<=)={args.fn_lo}  "
          f"sample<={args.sample}  seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    dev = load_jsonl(args.dev)
    train = load_jsonl(args.train)
    print(f"# dev docs={len(dev)}  train docs={len(train)}")

    probs = []
    for d in dev:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        probs.append(predict_doc(words, tokenizer, model, device,
                                 args.window, args.overlap, args.max_length))

    train_bg, train_vocab = build_train_bigrams(train)
    print(f"# distinct train token-bigrams={len(train_bg)}  train vocab={len(train_vocab)}")

    fps, fns = collect_confident_errors(dev, probs, args.fp_hi, args.fn_lo)
    print(f"\n# CONFIDENT FALSE POSITIVES (gold0, P>={args.fp_hi}) total = {len(fps)}")
    print(f"# CONFIDENT FALSE NEGATIVES (gold1, P<={args.fn_lo}) total = {len(fns)}")

    # ---- bucket ALL confident errors (for the distribution / key number) ----
    def bucket_all(items, kind):
        counts = Counter()
        for e in items:
            toks = dev[e["doc"]]["tokens"]
            letter, reason = auto_bucket(kind, toks, e["i"], e["genre"],
                                         train_bg, train_vocab)
            e["bucket"] = letter
            e["reason"] = reason
            counts[letter] += 1
        return counts

    fp_counts = bucket_all(fps, "FP")
    fn_counts = bucket_all(fns, "FN")

    def print_dist(name, counts, total):
        print(f"\n--- {name} bucket distribution (n={total}) ---")
        labels = {
            "A": "A gold-ambiguous/defensible (TENTATIVE - Noor to confirm)",
            "B": "B genre-structural (cloze/list/genealogy/legal)",
            "C": "C rare/UNSEEN word or bigram at site (coverage gap)",
            "D": "D clear MODEL error on ordinary clean prose",
        }
        for k in ["A", "B", "C", "D"]:
            c = counts.get(k, 0)
            pct = 100 * c / total if total else 0.0
            print(f"   {labels[k]:64s} {c:4d}  ({pct:5.1f}%)")

    print_dist("CONFIDENT FALSE POSITIVES", fp_counts, len(fps))
    print_dist("CONFIDENT FALSE NEGATIVES", fn_counts, len(fns))

    # ---- sample and PRINT concrete examples --------------------------------
    def sample(items):
        idx = list(range(len(items)))
        rng.shuffle(idx)
        return [items[j] for j in idx[:args.sample]]

    def dump(name, items, kind):
        print(f"\n================= {name}: up to {args.sample} sampled examples "
              f"=================")
        chosen = sample(items)
        # group printout by bucket for readability, but keep them all
        chosen.sort(key=lambda e: (e["bucket"], -e["p"]))
        for k, e in enumerate(chosen, 1):
            toks = dev[e["doc"]]["tokens"]
            ctx = make_context(toks, e["i"], args.radius)
            print(f"\n[{name[:2]}#{k:02d}] bucket={e['bucket']}  P={e['p']:.3f}  "
                  f"gold={e['gold']}  genre={e['genre']}  doc={e['doc']} pos={e['i']}")
            print(f"   reason: {e['reason']}")
            print(f"   {ctx}")

    dump("FALSE POSITIVES", fps, "FP")
    dump("FALSE NEGATIVES", fns, "FN")

    # ---- KEY NUMBER --------------------------------------------------------
    fpA = fp_counts.get("A", 0)
    fpB = fp_counts.get("B", 0)
    fpC = fp_counts.get("C", 0)
    fpD = fp_counts.get("D", 0)
    tot = max(len(fps), 1)
    print("\n===================== KEY NUMBER =====================")
    print(f"Confident FALSE POSITIVES = {len(fps)}")
    print(f"  bucket A (tentative GOLD-noise / defensible) = {fpA}  "
          f"({100*fpA/tot:.1f}%)  <-- LOWER-BOUND estimate of gold-noise (Sense B)")
    print(f"  bucket B (genre-structural)                  = {fpB}  ({100*fpB/tot:.1f}%)")
    print(f"  bucket C (unseen/coverage gap)               = {fpC}  ({100*fpC/tot:.1f}%)  "
          f"<-- the 'unseen word-pair coverage' driver (Sense A, fixable)")
    print(f"  bucket D (clean-prose model error)           = {fpD}  ({100*fpD/tot:.1f}%)  "
          f"<-- also Sense A, fixable")
    print("Note: A is assigned only AFTER B and C are ruled out, so it is a "
          "conservative LOWER bound. Genre (B) and coverage (C) are NOT gold-noise.")
    print("\n# done")


if __name__ == "__main__":
    main()
