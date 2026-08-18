"""EXTRA mistake DIAGNOSTIC for an AraSeg NoPnx-NP single-encoder checkpoint.

Analysis only. Trains NOTHING, writes NOTHING to runs/. It loads an existing
checkpoint (default: the union AraBERTv02 s42 model the other diagnostics use)
and predicts per-token P(boundary) on the dev split with the EXACT windowing in
predict.py (via predict.predict_doc), then measures the error dimensions the
preposition / example diagnostics did NOT already cover:

  1. OVER-SEGMENTATION RATE       - per doc and per genre, predicted boundaries
       (at threshold) vs gold boundaries: the ratio pred/gold (how many MORE cuts
       than gold), plus mean predicted vs gold sentence length (tokens/sentence).
  2. SENTENCE-LENGTH EFFECTS      - bin gold SENTENCES by their length (token span
       between consecutive gold boundaries) and report, for the decision positions
       inside each bin, the FP-rate (over-cutting) and, for the bin's true boundary
       position, the FN-rate (missed end). Does it over-cut long sentences / miss
       the end of short or very long ones?
  3. POSITION-IN-DOC             - FP-rate and FN-rate by relative position within a
       doc: first 10% / middle 80% / last 10% of decision positions.
  4. STRUCTURAL TRIGGERS         - FP-rate immediately before/at: a number/digit
       token, a quotation mark, an enumeration/list marker; PLUS the top-20 surface
       tokens by raw FP count, keyed on the NEXT token (the token the wrong cut
       lands in front of) and also on the CURRENT token (the token the wrong cut
       lands after). Which surface tokens most trigger a wrong cut?
  5. FALSE-NEGATIVE TAXONOMY     - mirror of the FP structural view for the MISSES:
       what structures does it FAIL to cut (hadith isnad chains, Quranic verse
       starts, genealogy 'begat', enumeration, ...)? Counts + ~10 real Arabic
       examples per category.

Label semantics (data.py): labels[i]==1 means a sentence boundary FOLLOWS tok[i].
Positions where tok[i] == PARAGRAPH_TOKEN ("\n") are forced-0 by predict.py / the
evaluator and are NEVER a real decision, so they are excluded everywhere below.
(The NoPnx-NP dev split contains zero "\n" tokens, but we exclude defensively so
this file is safe on the PA variants too.)

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python src/diag_mistake_extra.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev   data/NoPnx-NP_dev.jsonl \
      --threshold 0.5
"""
from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Surface-structure detectors (regex-free, token-level). All operate on the raw
# whitespace token strings exactly as stored in the JSONL.
# ---------------------------------------------------------------------------
ARABIC_INDIC_DIGITS = set(chr(c) for c in range(0x0660, 0x066A))  # ٠-٩ (U+0660..0669)
ASCII_DIGITS = set("0123456789")
# Quote / speech marks: ASCII " and ', plus common Unicode pairs. Defined by
# code point to avoid embedding literal quote glyphs in the source string.
QUOTE_CHARS = {
    '"', "'",
    "“", "”",   # “ ”  left/right double
    "‘", "’",   # ‘ ’  left/right single
    "«", "»",   # « »  guillemets
    "‹", "›",   # ‹ ›  single guillemets
}

# Enumeration / list ordinals and markers common in legal + exercise text.
ENUM_TOKENS = {
    "أولا", "أولاً", "ثانيا", "ثانياً", "ثالثا", "ثالثاً", "رابعا", "رابعاً",
    "خامسا", "خامساً", "سادسا", "سادساً", "سابعا", "سابعاً", "ثامنا", "ثامناً",
    "تاسعا", "تاسعاً", "عاشرا", "عاشراً",
    "المادة", "الفقرة", "البند", "الفصل", "الباب",
    "أ", "ب", "ج", "د", "ه", "و",   # single-letter list labels (weak; see note)
    "-", "•", "*", ")", "(",
}

# Hadith / isnad transmission verbs and chain cues.
ISNAD_TOKENS = {"حدثنا", "حدثني", "أخبرنا", "أخبرني", "أنبأنا", "روى", "رواه",
                "عن", "قال", "بن", "ابن", "رسول"}
# Quranic verse / recitation cues.
QURAN_CUES = {"قال", "تعالى", "بسم", "الله", "الرحمن", "الرحيم", "سبحانه",
              "عز", "وجل", "الآية", "سورة"}
# Genealogy "begat".
BEGAT_TOKENS = {"ولد", "أنجب", "بن", "ابن"}


def is_number_token(t: str) -> bool:
    """True if the token is (mostly) a numeral: ASCII digits or Arabic-Indic."""
    if not t:
        return False
    digitish = sum(1 for c in t if c in ASCII_DIGITS or c in ARABIC_INDIC_DIGITS)
    # allow decimal/thousand separators and a trailing % without disqualifying
    other = sum(1 for c in t if c not in ASCII_DIGITS and c not in ARABIC_INDIC_DIGITS
                and c not in ".,%/-")
    return digitish >= 1 and digitish >= other


def has_quote(t: str) -> bool:
    return any(c in QUOTE_CHARS for c in t)


def is_enum_marker(t: str) -> bool:
    return t in ENUM_TOKENS


def collect_probs(docs, tokenizer, model, device, window, overlap, max_length):
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        out.append(predict_doc(words, tokenizer, model, device,
                               window, overlap, max_length))
    return out


def gold_sentence_spans(tokens, gold):
    """Yield (start, end, length) for each gold sentence.

    A gold sentence is the token span ending at each position with gold==1
    (inclusive), starting right after the previous boundary. The length counts
    tokens in the span (including any \\n, matching data.mean_sentence_length).
    A trailing run with no final boundary is NOT counted as a completed sentence
    (mirrors data.mean_sentence_length, which divides by sum(labels)).
    """
    start = 0
    for i, g in enumerate(gold):
        if int(g) == 1:
            yield (start, i, i - start + 1)
            start = i + 1


def make_context(tokens, i, radius=6):
    """+/- radius word window with the decision point marked after tok[i]."""
    n = len(tokens)
    lo = max(0, i - radius)
    hi = min(n, i + 1 + radius)

    def show(t):
        return "[PAR]" if t == PARAGRAPH_TOKEN else t

    left_s = " ".join(show(t) for t in tokens[lo:i + 1])
    right_s = " ".join(show(t) for t in tokens[i + 1:hi])
    pre = "... " if lo > 0 else ""
    post = " ..." if hi < n else ""
    return f"{pre}{left_s}  |<==  {right_s}{post}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/union-NoPnx-NP-arabertv02-s42")
    ap.add_argument("--dev", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--examples", type=int, default=10,
                    help="real Arabic examples to print per FN category")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device={device}  model={args.model}  threshold={args.threshold}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    dev = load_jsonl(args.dev)
    print(f"# dev docs={len(dev)}")
    probs = collect_probs(dev, tokenizer, model, device,
                          args.window, args.overlap, args.max_length)
    thr = args.threshold

    # Precompute per-doc predictions (thresholded, with forced-0 at \n) so every
    # section uses the SAME decision the evaluator would.
    preds = []
    for d, p in zip(dev, probs):
        lab = (p >= thr).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        preds.append(lab)

    # =====================================================================
    # 1. OVER-SEGMENTATION RATE (per doc + per genre) & mean sentence length
    # =====================================================================
    print("\n" + "=" * 78)
    print("1. OVER-SEGMENTATION RATE  (pred boundaries vs gold boundaries)")
    print("=" * 78)

    per_doc_ratio = []
    genre_agg = defaultdict(lambda: dict(docs=0, gold=0, pred=0, toks=0))
    corpus_gold = corpus_pred = corpus_tok = 0
    for d, pr in zip(dev, preds):
        toks = d["tokens"]
        gold = d["labels"]
        # decision positions exclude \n
        dec_tok = sum(1 for w in toks if w != PARAGRAPH_TOKEN)
        g = int(sum(gold))
        pc = int(sum(pr))
        genre = bucket(toks)
        ga = genre_agg[genre]
        ga["docs"] += 1
        ga["gold"] += g
        ga["pred"] += pc
        ga["toks"] += dec_tok
        corpus_gold += g
        corpus_pred += pc
        corpus_tok += dec_tok
        if g > 0:
            per_doc_ratio.append(pc / g)

    print(f"corpus gold boundaries = {corpus_gold}   "
          f"pred boundaries = {corpus_pred}   "
          f"ratio pred/gold = {corpus_pred / max(corpus_gold, 1):.3f}")
    print(f"corpus mean GOLD sentence length  = "
          f"{corpus_tok / max(corpus_gold, 1):.2f} tok/sent")
    print(f"corpus mean PRED sentence length  = "
          f"{corpus_tok / max(corpus_pred, 1):.2f} tok/sent")
    if per_doc_ratio:
        arr = np.array(per_doc_ratio)
        print(f"per-doc pred/gold ratio: mean={arr.mean():.3f}  "
              f"median={np.median(arr):.3f}  "
              f"p10={np.percentile(arr,10):.3f}  p90={np.percentile(arr,90):.3f}  "
              f"frac docs OVER-cut(>1)={float((arr>1).mean()):.3f}  "
              f"frac UNDER-cut(<1)={float((arr<1).mean()):.3f}")

    print(f"\n{'genre':20s} {'docs':>5} {'gold_b':>7} {'pred_b':>7} "
          f"{'p/g':>6} {'goldLen':>8} {'predLen':>8}")
    order = ["general prose", "legal/constitution", "hadith/isnad",
             "genealogy", "cloze/exercise"]
    for g in order + [k for k in genre_agg if k not in order]:
        if g not in genre_agg:
            continue
        s = genre_agg[g]
        ratio = s["pred"] / max(s["gold"], 1)
        glen = s["toks"] / max(s["gold"], 1)
        plen = s["toks"] / max(s["pred"], 1)
        print(f"{g:20s} {s['docs']:5d} {s['gold']:7d} {s['pred']:7d} "
              f"{ratio:6.3f} {glen:8.2f} {plen:8.2f}")

    # =====================================================================
    # 2. SENTENCE-LENGTH EFFECTS
    # =====================================================================
    print("\n" + "=" * 78)
    print("2. SENTENCE-LENGTH EFFECTS  (bin gold sentences by length)")
    print("=" * 78)
    print("For each gold-sentence-length bin:")
    print("  FN-rate  = fraction of that bin's TRUE end-boundaries the model MISSED")
    print("             (gold==1 but pred==0 at the sentence's final token).")
    print("  FP-rate  = fraction of the bin's INTERIOR positions (gold==0, the")
    print("             non-final tokens of the sentence) the model WRONGLY cut.")
    print("  -> high interior FP-rate on long bins = over-cutting long sentences;")
    print("     high FN-rate on short/very-long bins = missing those ends.")

    # bin edges by sentence length in tokens
    bins = [(1, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 40), (41, 10**9)]

    def bin_of(L):
        for lo, hi in bins:
            if lo <= L <= hi:
                return (lo, hi)
        return bins[-1]

    binstat = {b: dict(sents=0, end_miss=0, interior_pos=0, interior_fp=0)
               for b in bins}
    for d, pr in zip(dev, preds):
        toks = d["tokens"]
        gold = d["labels"]
        for (s0, s1, L) in gold_sentence_spans(toks, gold):
            b = bin_of(L)
            bs = binstat[b]
            bs["sents"] += 1
            # final token = the true boundary position s1 (gold==1)
            if toks[s1] != PARAGRAPH_TOKEN:
                if int(pr[s1]) == 0:
                    bs["end_miss"] += 1
            # interior positions s0..s1-1 are gold==0 (no boundary yet)
            for j in range(s0, s1):
                if toks[j] == PARAGRAPH_TOKEN:
                    continue
                bs["interior_pos"] += 1
                if int(pr[j]) == 1:
                    bs["interior_fp"] += 1

    print(f"\n{'len_bin':>10} {'#sents':>7} {'FN_end':>7} {'FN_rate':>8} "
          f"{'int_pos':>8} {'int_FP':>7} {'FP_rate':>8}")
    for b in bins:
        s = binstat[b]
        lo, hi = b
        name = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        fnr = s["end_miss"] / max(s["sents"], 1)
        fpr = s["interior_fp"] / max(s["interior_pos"], 1)
        print(f"{name:>10} {s['sents']:7d} {s['end_miss']:7d} {fnr:8.4f} "
              f"{s['interior_pos']:8d} {s['interior_fp']:7d} {fpr:8.4f}")

    # =====================================================================
    # 3. POSITION-IN-DOC
    # =====================================================================
    print("\n" + "=" * 78)
    print("3. POSITION-IN-DOC  (relative position of decision within its doc)")
    print("=" * 78)
    print("Decision positions (excl. \\n) split by relative index r=i/n_decisions:")
    print("  first 10% (r<0.10) / middle (0.10<=r<0.90) / last 10% (r>=0.90).")

    pos_stat = {k: dict(pos=0, gold=0, pred=0, fp=0, fn=0)
                for k in ("first10", "middle", "last10")}
    for d, pr in zip(dev, preds):
        toks = d["tokens"]
        gold = d["labels"]
        # enumerate real decision positions in order
        dec_idx = [i for i in range(len(toks)) if toks[i] != PARAGRAPH_TOKEN]
        m = len(dec_idx)
        for rank, i in enumerate(dec_idx):
            r = rank / max(m, 1)
            if r < 0.10:
                key = "first10"
            elif r >= 0.90:
                key = "last10"
            else:
                key = "middle"
            gi = int(gold[i]); pi = int(pr[i])
            ps = pos_stat[key]
            ps["pos"] += 1
            ps["gold"] += gi
            ps["pred"] += pi
            if pi == 1 and gi == 0:
                ps["fp"] += 1
            if pi == 0 and gi == 1:
                ps["fn"] += 1

    print(f"\n{'region':10s} {'pos':>8} {'gold_b':>7} {'pred_b':>7} "
          f"{'FP':>6} {'FN':>6} {'FP/neg':>8} {'FN/pos':>8}")
    for key in ("first10", "middle", "last10"):
        s = pos_stat[key]
        neg = s["pos"] - s["gold"]     # gold-negative positions
        posb = s["gold"]               # gold-positive positions
        fpr = s["fp"] / max(neg, 1)
        fnr = s["fn"] / max(posb, 1)
        print(f"{key:10s} {s['pos']:8d} {s['gold']:7d} {s['pred']:7d} "
              f"{s['fp']:6d} {s['fn']:6d} {fpr:8.4f} {fnr:8.4f}")
    print("(FP/neg = FP / gold-negative positions; FN/pos = FN / gold-positive.)")

    # =====================================================================
    # 4. STRUCTURAL TRIGGERS for FALSE POSITIVES
    # =====================================================================
    print("\n" + "=" * 78)
    print("4. STRUCTURAL TRIGGERS  (what surface features trigger a WRONG cut)")
    print("=" * 78)
    print("A wrong cut at position i (FP: gold==0, pred==1) 'lands after' tok[i]")
    print("and 'before' tok[i+1]. We test structural features of BOTH sides.")

    # feature accumulators: for each feature, count gold-negative positions that
    # match it (the denominator) and how many of those are FP (the numerator).
    feat = defaultdict(lambda: dict(neg=0, fp=0))

    def note(featname, is_fp):
        feat[featname]["neg"] += 1
        if is_fp:
            feat[featname]["fp"] += 1

    fp_next_tok = Counter()   # next token (cut lands before)
    fp_cur_tok = Counter()    # current token (cut lands after)
    fp_examples_by_feat = defaultdict(list)

    for d, pr in zip(dev, preds):
        toks = d["tokens"]
        gold = d["labels"]
        n = len(toks)
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            gi = int(gold[i])
            if gi != 0:
                continue  # only gold-negative positions can be FP
            is_fp = int(pr[i]) == 1
            cur = toks[i]
            nxt = toks[i + 1] if i + 1 < n else None
            # structural features keyed on the NEXT token (what the cut precedes)
            if nxt is not None:
                if is_number_token(nxt):
                    note("next=NUMBER", is_fp)
                    if is_fp:
                        fp_examples_by_feat["next=NUMBER"].append((d["doc_id"], toks, i))
                if has_quote(nxt):
                    note("next=QUOTE", is_fp)
                    if is_fp:
                        fp_examples_by_feat["next=QUOTE"].append((d["doc_id"], toks, i))
                if is_enum_marker(nxt):
                    note("next=ENUM", is_fp)
                    if is_fp:
                        fp_examples_by_feat["next=ENUM"].append((d["doc_id"], toks, i))
            # structural features keyed on the CURRENT token (what the cut follows)
            if is_number_token(cur):
                note("cur=NUMBER", is_fp)
            if has_quote(cur):
                note("cur=QUOTE", is_fp)
            if is_enum_marker(cur):
                note("cur=ENUM", is_fp)
            if is_fp:
                if nxt is not None:
                    fp_next_tok[nxt] += 1
                fp_cur_tok[cur] += 1

    # baseline FP rate over all gold-negative positions
    all_neg = sum(1 for d, pr in zip(dev, preds) for i, w in enumerate(d["tokens"])
                  if w != PARAGRAPH_TOKEN and int(d["labels"][i]) == 0)
    all_fp = sum(1 for d, pr in zip(dev, preds) for i, w in enumerate(d["tokens"])
                 if w != PARAGRAPH_TOKEN and int(d["labels"][i]) == 0 and int(pr[i]) == 1)
    base_fpr = all_fp / max(all_neg, 1)
    print(f"\nBASELINE FP-rate over ALL gold-negative positions = "
          f"{all_fp}/{all_neg} = {base_fpr:.4f}")
    print(f"\n{'feature':16s} {'neg_pos':>8} {'FP':>6} {'FP_rate':>8} {'vs_base':>8}")
    for fn in ["next=NUMBER", "next=QUOTE", "next=ENUM",
               "cur=NUMBER", "cur=QUOTE", "cur=ENUM"]:
        s = feat.get(fn, {"neg": 0, "fp": 0})
        rate = s["fp"] / max(s["neg"], 1)
        lift = rate / base_fpr if base_fpr > 0 else float("nan")
        print(f"{fn:16s} {s['neg']:8d} {s['fp']:6d} {rate:8.4f} {lift:7.2f}x")

    print("\ntop-20 tokens by raw FP count keyed on the NEXT token "
          "(the wrong cut lands BEFORE this token):")
    for tok, c in fp_next_tok.most_common(20):
        print(f"    {tok!r:16s} {c}")
    print("\ntop-20 tokens by raw FP count keyed on the CURRENT token "
          "(the wrong cut lands AFTER this token):")
    for tok, c in fp_cur_tok.most_common(20):
        print(f"    {tok!r:16s} {c}")

    # a few concrete FP examples for the strongest structural triggers
    print("\n--- sample FP examples for structural triggers ---")
    for fn in ["next=NUMBER", "next=QUOTE", "next=ENUM"]:
        exs = fp_examples_by_feat.get(fn, [])
        if not exs:
            print(f"[{fn}] (no FP examples)")
            continue
        rng.shuffle(exs)
        print(f"[{fn}] {len(exs)} FP total; up to 4 examples:")
        for doc_id, toks, i in exs[:4]:
            print(f"    doc={doc_id} pos={i}: {make_context(toks, i, 6)}")

    # =====================================================================
    # 5. FALSE-NEGATIVE TAXONOMY  (what structures does it FAIL to cut)
    # =====================================================================
    print("\n" + "=" * 78)
    print("5. FALSE-NEGATIVE TAXONOMY  (missed boundaries: gold==1, pred==0)")
    print("=" * 78)
    print("Each missed boundary is categorized by the local structure at the")
    print("gold-cut site (window tok[i-3..i+3]). A miss can match several; we")
    print("record ALL matches for counts and keep examples per category.")

    # category detectors operate on a small window around the gold boundary i
    def fn_categories(toks, i, genre):
        n = len(toks)
        cur = toks[i]
        nxt = toks[i + 1] if i + 1 < n else None
        win = toks[max(0, i - 3):min(n, i + 4)]
        winset = set(win)
        cats = []
        # hadith isnad: transmission verb or chain nearby
        if winset & {"حدثنا", "حدثني", "أخبرنا", "أخبرني", "أنبأنا", "روى", "رواه"} \
                or (("عن" in winset or "قال" in winset) and "بن" in winset):
            cats.append("hadith/isnad")
        # Quranic verse start: تعالى / بسم / سبحانه / عز وجل near, or next starts sura-ish
        if winset & {"تعالى", "بسم", "سبحانه", "الآية", "سورة"} \
                or (("قال" in win or "قال" == cur) and "الله" in winset):
            cats.append("quran/verse")
        # genealogy 'begat'
        if (nxt in BEGAT_TOKENS) or (cur in {"ولد", "أنجب"}) \
                or (winset & {"ولد", "أنجب"} and "بن" in winset):
            cats.append("genealogy/begat")
        # enumeration / list marker just after the missed cut
        if nxt is not None and (is_enum_marker(nxt) or nxt in ENUM_TOKENS):
            cats.append("enumeration/list")
        # next token is a number/digit (e.g. a new numbered item / verse no.)
        if nxt is not None and is_number_token(nxt):
            cats.append("next=number")
        # next token is a connective/clause-starter (continuation-looking end)
        if nxt in {"و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن", "فقال",
                   "فإن", "كما", "لكن", "أو", "بل", "حتى", "قال", "وقال",
                   "أما", "إذا", "إن", "حيث"}:
            cats.append("next=connective")
        # quotation boundary
        if (nxt is not None and has_quote(nxt)) or has_quote(cur):
            cats.append("quote-adjacent")
        if not cats:
            cats.append("other/plain-prose")
        return cats

    fn_cat_count = Counter()
    fn_cat_examples = defaultdict(list)
    total_fn = 0
    for d, pr, p in zip(dev, preds, probs):
        toks = d["tokens"]
        gold = d["labels"]
        genre = bucket(toks)
        n = len(toks)
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            if int(gold[i]) == 1 and int(pr[i]) == 0:
                total_fn += 1
                for c in fn_categories(toks, i, genre):
                    fn_cat_count[c] += 1
                    fn_cat_examples[c].append((d["doc_id"], toks, i, float(p[i])))

    print(f"\nTOTAL false negatives (missed gold boundaries) = {total_fn}")
    print(f"{'category':20s} {'count':>7} {'% of FN':>8}")
    for c, n in fn_cat_count.most_common():
        print(f"{c:20s} {n:7d} {100*n/max(total_fn,1):8.1f}")

    print(f"\n--- up to {args.examples} real Arabic examples per FN category ---")
    order_cats = ["hadith/isnad", "quran/verse", "genealogy/begat",
                  "enumeration/list", "next=number", "next=connective",
                  "quote-adjacent", "other/plain-prose"]
    for c in order_cats + [k for k in fn_cat_examples if k not in order_cats]:
        exs = fn_cat_examples.get(c, [])
        if not exs:
            continue
        rng.shuffle(exs)
        print(f"\n[FN category: {c}]  (n={fn_cat_count[c]})")
        for doc_id, toks, i, pmiss in exs[:args.examples]:
            print(f"    doc={doc_id} pos={i} P={pmiss:.3f}: {make_context(toks, i, 6)}")

    print("\n# done")


if __name__ == "__main__":
    main()
