"""Confident-error EXAMPLE extractor for the PUNCTUATED tracks (PA / NP).

Analysis only; trains NOTHING. This is an ADDITIVE standalone file: it imports
existing modules (data, predict, genre_buckets) and is imported by nothing. It
adapts src/diag_error_examples.py (which is hard-wired to NoPnx-NP) to the two
punctuated tracks so Noor can SEE whether the PA/NP "headroom" is real fixable
model error or defensible gold ambiguity.

Motivation: the "~44% ambiguous gold" figure for PA/NP traces to residual_errors.py
("25-29% ambiguous, 65-70% unseen") re-expressed in PLAN_data_scarcity as
"29% ambiguous gold + 15% no-evidence". That number is a TRAIN-FREQUENCY heuristic
(context seen in train but boundary <50% of the time), NOT a linguistic judgment
and NOT annotator disagreement. This script produces a checkpoint-grounded,
conservative bucket distribution over the CONFIDENT errors instead.

  CONFIDENT FALSE POSITIVES : gold 0 but P(boundary) >= FP_HI (default 0.8)
  CONFIDENT FALSE NEGATIVES : gold 1 but P(boundary) <= FN_LO (default 0.2)

Forced-0 positions (tok == PARAGRAPH_TOKEN "\n") are excluded (never a real
decision). Buckets (CONSERVATIVE, B and C assigned BEFORE A so A is a LOWER bound):
  (A) plausibly gold-ambiguous / defensible  - the model's call is linguistically
        reasonable; a real boundary could sit there (FP) or gold's boundary looks
        arguable (FN). TENTATIVE heuristic flag for a NATIVE reader to confirm.
        We have NO inter-annotator data; A is NOT proven annotator disagreement.
  (B) genre / structural  - non-prose genre or list/genealogy/legal cue.
  (C) rare / UNSEEN word or bigram at the site  - coverage gap.
  (D) clear MODEL error on ordinary clean prose  - fixable = real headroom.

For the punctuated tracks the boundary decision is dominated by PUNCTUATION, so
the A/D heuristics are punctuation-aware (see PUNCT_* sets below).

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python src/diag_error_examples_panp.py \
      --task PA --model runs/opus_trdev_2026-07-08/PA_base_s42 \
      --fp-hi 0.8 --fn-lo 0.2 --sample 40 --seed 42
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python src/diag_error_examples_panp.py \
      --task NP --model runs/opus_trdev_2026-07-08/NP_base_s42
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc
from genre_buckets import bucket

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Default base (train-only) checkpoint per task, so dev is genuinely unseen.
DEFAULT_MODEL = {
    "PA": "runs/opus_trdev_2026-07-08/PA_base_s42",
    "NP": "runs/opus_trdev_2026-07-08/NP_base_s42",
}

# --- lexical / punctuation cues used ONLY for conservative auto-bucketing ---
# Arabic + Latin sentence-final punctuation. On the PUNCTUATED tracks a boundary
# essentially always lands right AFTER one of these; the interesting errors are
# where the model cut/failed-to-cut around a punctuation mark.
PUNCT_STRONG = {".", "؟", "!", "?", "،،", "؛"}          # strong sentence enders
PUNCT_WEAK = {"،", ",", ":", "：", "؛", ";", "-", "—", "…", "..", "..."}  # ambiguous (comma etc.)
ALL_PUNCT = PUNCT_STRONG | PUNCT_WEAK | {'"', "»", "«", ")", "(", "]", "["}

CLAUSE_STARTERS = {
    "و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
    "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى",
    "قال", "وقال", "أما", "إذا", "إن", "لأن", "حيث", "كي", "لكي",
}
LIST_STRUCT = {"ولد", "بن", "ابن", "المادة", "فراغ", "أولا", "ثانيا", "ثالثا",
               "رابعا", "خامسا"}


def is_punct(tok):
    if tok is None:
        return False
    t = tok.strip()
    if not t:
        return False
    return all((not ch.isalnum()) for ch in t) or t in ALL_PUNCT


def make_context(tokens, i, radius=6):
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
    """CONSERVATIVE bucket for one confident error on a PUNCTUATED track.

    Priority: B (genre/struct) -> C (unseen) -> A (defensible) -> D (clean error).
    B and C before A => A is a LOWER bound on gold-ambiguity.

    Punctuation-aware A/D logic (the crux for PA/NP):
      * FP (model cut, gold 0):
          - If the token AT the cut is a WEAK/ambiguous punct (comma، colon:
            semicolon؛ dash), a sentence split there is often linguistically
            defensible -> A. (Weak punct is exactly the arguable-gold zone.)
          - If the cut is right after a STRONG ender (. ؟ !) but gold says 0,
            that is also defensible (strong punct usually ends a sentence) -> A.
          - Otherwise, cut before a clause-starter with no punct at all -> A only
            if a connective clause-start; else D.
      * FN (gold 1, model didn't cut):
          - If there is NO strong ender at/just-before the site (i.e. gold put a
            boundary with only a weak comma or no punctuation), the boundary is
            arguable and the model's non-cut is defensible -> A.
          - If a STRONG ender sits at/just-before the site and the model still
            missed it, that is an ordinary model error on clear evidence -> D.
    """
    n = len(tokens)
    prev_tok = tokens[i - 1] if i - 1 >= 0 else None
    cur_tok = tokens[i]
    nxt_tok = tokens[i + 1] if i + 1 < n else None

    # --- B: genre / structural formatting -----------------------------------
    if genre != "general prose":
        return "B", f"genre={genre}"
    window = set(tokens[max(0, i - 2):min(n, i + 3)])
    struct_hits = window & LIST_STRUCT
    if struct_hits:
        return "B", f"list/struct cue {sorted(struct_hits)}"

    # --- C: unseen local context (coverage gap) -----------------------------
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

    # --- punctuation context at the decision site ---------------------------
    cur_strong = cur_tok in PUNCT_STRONG
    cur_weak = cur_tok in PUNCT_WEAK
    prev_strong = prev_tok in PUNCT_STRONG
    prev_weak = prev_tok in PUNCT_WEAK
    site_has_strong = cur_strong or prev_strong
    site_has_weak = cur_weak or prev_weak

    # --- A: plausibly gold-ambiguous / defensible ---------------------------
    if kind == "FP":
        if cur_weak or prev_weak:
            return "A", (f"cut at/after WEAK punct {cur_tok if cur_weak else prev_tok!r}"
                         f" (comma/colon-split defensible)")
        if cur_strong or prev_strong:
            return "A", (f"cut at/after STRONG ender {cur_tok if cur_strong else prev_tok!r}"
                         f" (gold 0 but ender present -> arguable)")
        if nxt_tok in CLAUSE_STARTERS:
            return "A", f"cut before clause-starter {nxt_tok!r} (defensible)"
    else:  # FN: gold says boundary, model didn't cut
        if not site_has_strong:
            # gold boundary with only weak/no punctuation evidence -> arguable
            if site_has_weak:
                return "A", ("gold boundary on WEAK punct only "
                             "(comma/colon; model no-cut defensible)")
            if nxt_tok in CLAUSE_STARTERS:
                return "A", (f"gold boundary before continuation {nxt_tok!r}, no ender "
                             f"(model no-cut defensible)")
            return "A", "gold boundary with NO sentence-ender at site (arguable)"

    # --- D: clean-prose model error -----------------------------------------
    if kind == "FN" and site_has_strong:
        return "D", (f"missed boundary despite STRONG ender "
                     f"{cur_tok if cur_strong else prev_tok!r} (clear model error)")
    return "D", "general prose, seen context, no defensible reading"


def collect_confident_errors(dev, probs, fp_hi, fn_lo):
    fps, fns = [], []
    for di, (d, p) in enumerate(zip(dev, probs)):
        toks = d["tokens"]
        gold = d["labels"]
        genre = bucket(toks)
        n = len(toks)
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            gi = int(gold[i])
            pi = float(p[i])
            if gi == 0 and pi >= fp_hi:
                fps.append(dict(doc=di, i=i, p=pi, gold=gi, genre=genre))
            elif gi == 1 and pi <= fn_lo:
                fns.append(dict(doc=di, i=i, p=pi, gold=gi, genre=genre))
    return fps, fns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["PA", "NP"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--fp-hi", type=float, default=0.8)
    ap.add_argument("--fn-lo", type=float, default=0.2)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    model_path = args.model or DEFAULT_MODEL[args.task]
    dev_path = args.dev or f"data/{args.task}_dev.jsonl"
    train_path = args.train or f"data/{args.task}_train.jsonl"

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# task={args.task}  device={device}  model={model_path}")
    print(f"# fp_hi(P>=)={args.fp_hi}  fn_lo(P<=)={args.fn_lo}  "
          f"sample<={args.sample}  seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device).eval()

    dev = load_jsonl(dev_path)
    train = load_jsonl(train_path)
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
            "A": "A gold-ambiguous/defensible (TENTATIVE - native reader to confirm)",
            "B": "B genre-structural (cloze/list/genealogy/legal)",
            "C": "C rare/UNSEEN word or bigram at site (coverage gap)",
            "D": "D clear MODEL error = real fixable headroom",
        }
        for k in ["A", "B", "C", "D"]:
            c = counts.get(k, 0)
            pct = 100 * c / total if total else 0.0
            print(f"   {labels[k]:70s} {c:4d}  ({pct:5.1f}%)")

    print_dist("CONFIDENT FALSE POSITIVES", fp_counts, len(fps))
    print_dist("CONFIDENT FALSE NEGATIVES", fn_counts, len(fns))

    # combined (FP+FN) distribution
    comb = Counter()
    for k in ["A", "B", "C", "D"]:
        comb[k] = fp_counts.get(k, 0) + fn_counts.get(k, 0)
    tot_comb = len(fps) + len(fns)
    print_dist("ALL CONFIDENT ERRORS (FP+FN combined)", comb, tot_comb)

    def sample(items):
        idx = list(range(len(items)))
        rng.shuffle(idx)
        return [items[j] for j in idx[:args.sample]]

    def dump(name, items, kind):
        print(f"\n================= {name}: up to {args.sample} sampled examples "
              f"=================")
        chosen = sample(items)
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
    def kn(label, counts, total):
        a, b, c, d = (counts.get("A", 0), counts.get("B", 0),
                      counts.get("C", 0), counts.get("D", 0))
        t = max(total, 1)
        print(f"\n----- KEY NUMBER: {label} (n={total}) -----")
        print(f"  A defensible/ambiguous = {a}  ({100*a/t:.1f}%)  <- LOWER bound, unfixable-ish")
        print(f"  B genre-structural     = {b}  ({100*b/t:.1f}%)")
        print(f"  C unseen/coverage      = {c}  ({100*c/t:.1f}%)  <- fixable via coverage")
        print(f"  D clean model error    = {d}  ({100*d/t:.1f}%)  <- fixable = real headroom")
        print(f"  FIXABLE (C+D)          = {c+d}  ({100*(c+d)/t:.1f}%)")
    print("\n===================== KEY NUMBERS =====================")
    kn("CONFIDENT FALSE POSITIVES", fp_counts, len(fps))
    kn("CONFIDENT FALSE NEGATIVES", fn_counts, len(fns))
    kn("ALL CONFIDENT ERRORS (FP+FN)", comb, tot_comb)
    print("\nNote: A is assigned only AFTER B and C are ruled out => conservative "
          "LOWER bound on gold-ambiguity. A is a TENTATIVE heuristic flag; we have "
          "NO inter-annotator data, so A is NOT proven annotator disagreement.")
    print("\n# done")


if __name__ == "__main__":
    main()
