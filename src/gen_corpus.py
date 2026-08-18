"""In-genre Arabic CORPUS GENERATOR for AraSeg 2026 — PRETRAINING corpus only.

WHAT THIS IS (and is NOT)
=========================
This script generates a large, in-genre, *unlabeled* Arabic text corpus with a
local AraGPT2 model. The corpus is used for exactly ONE thing: unsupervised
masked-LM (MLM) pretraining / continued-pretraining of the AraBERTv02 encoder.
It is NEVER used as labeled boundary data. There is no boundary label anywhere
in the output. Boundary supervision stays 100% the real 174 AraSeg docs.

WHY
===
Data scarcity. The labeled AraSeg corpus is tiny (~174 docs). An earlier TAPT
attempt (MLM on only ~74k of our own tokens for 50 epochs) OVERFIT and hurt dev
(-0.61). The fix is a MUCH larger in-genre corpus so the encoder can adapt to
the AraSeg genres (MSA prose/news, legal/constitutional, Quran/religious,
hadith/isnad) with a controlled step budget instead of memorizing 74k tokens.

DOWNSTREAM COMPATIBILITY
========================
Output is `data/gen_corpus.txt`: ONE document per line, plain UTF-8 text,
whitespace-tokenizable. An MLM pretrainer joins/segments these lines itself.
This script writes NOTHING but text; it does not touch train/dev/test JSONL,
the shared pipeline, or any load-bearing file. It is fully self-contained and
additive (CLAUDE.md rules 2 + 5).

LEAKAGE GUARD (hard requirement)
================================
Every generated doc is checked against EVERY `data/*_dev.jsonl` and
`data/*_test.jsonl` token stream with an 8-gram tripwire: any generated doc
sharing a single token 8-gram with dev/test is DROPPED. Reuses the audited
primitives in `src/synth_filter.py` (build_ngram_tripwire / shares_ngram),
which is the repo's canonical leakage guard.

QUALITY FILTERS (all reuse src/synth_filter.py primitives)
==========================================================
  * Arabic-script ratio >= 0.90                    (arabic_ratio)
  * distinct-4gram ratio >= threshold  (anti-loop) (distinct_ngram_ratio)
  * top-token frequency  <= threshold  (anti-loop) (top_token_frequency)
  * exact-dedup across generated docs              (hash of normalized text)
  * 8-gram dev/test tripwire                        (build_ngram_tripwire)

PREPOSITION-EMPHASIS MODE (additive, flag-gated: --prep-mode)
============================================================
For self-supervised (BYOL-style) pretraining we sometimes want a HUGE corpus
whose passages are RICH in the exact preposition/connective words the segmenter
confuses (و ف ثم من في وكان وقد ولكن فقال فإن كما لكن أو بل حتى). Passing
--prep-mode:
  * swaps the genre seeds for preposition-front-loaded seeds (PREP_GENRE_SEEDS),
  * adds an OVERSAMPLING gate: a passage is kept only if it carries >= K target
    prepositions (--min-preps) AND prep-density >= floor (--min-prep-density),
  * lowers the Arabic-script floor to 0.80 (task requirement; --min-arabic-ratio
    still overrides), keeping all the other reused quality filters
    (distinct-ngram anti-loop, top-token anti-loop, exact-dedup, 8-gram
    dev/test leakage tripwire) unchanged,
  * writes UNLABELED jsonl: one {"text": ...} per line, NO boundary labels,
    defaulting to data/gen_corpus_prep.jsonl (never clobbers gen_corpus.txt).
Default behavior (no --prep-mode) is byte-for-byte identical to before.

USAGE
=====
GPU smoke (~30 short docs, prints samples + stats):
  python src/gen_corpus.py --smoke

Full ~1M-token corpus (orchestrator; see printed command at end of smoke):
  python src/gen_corpus.py --target-tokens 1000000 --out data/gen_corpus.txt

Preposition-emphasis smoke (~200 passages, prints prep stats + scale-up cmd):
  python src/gen_corpus.py --prep-mode --smoke

Every default reproduces the same recipe; nothing here changes on re-run except
via explicit flags.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

# Force UTF-8 stdout so Arabic prints on Windows consoles (cp1252 default).
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# Reuse the repo's audited filter primitives — do NOT reimplement.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from synth_filter import (  # noqa: E402
    arabic_ratio,
    build_ngram_tripwire,
    distinct_ngram_ratio,
    load_jsonl,
    shares_ngram,
    top_token_frequency,
)

# aragpt2-medium (369M) is the largest AraGPT2 that loads CLEANLY with the stock
# AutoModelForCausalLM (no trust_remote_code) and generates fluent in-genre Arabic.
# It uses only ~0.8GB VRAM in fp16, leaving huge batch headroom on the 5090.
# aragpt2-mega is intentionally NOT the default: it requires trust_remote_code=True
# (executes custom repo modeling code) — a scope/dependency decision that is out of
# bounds here (CLAUDE.md scope guards; "no new dependencies without approval"). It
# can still be selected explicitly via --model aubmindlab/aragpt2-mega if approved
# (the loader honors trust_remote_code only when that model id is passed).
DEFAULT_MODEL = "aubmindlab/aragpt2-medium"
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")
DEFAULT_OUT = os.path.join(DATA_DIR, "gen_corpus.txt")
# Preposition-emphasis mode writes UNLABELED jsonl ({"text": ...}) here by
# default so it never clobbers the plain-text gen_corpus.txt.
DEFAULT_PREP_OUT = os.path.join(DATA_DIR, "gen_corpus_prep.jsonl")

# --------------------------------------------------------------------------- #
# Genre seed prompts. These prime generation toward the four AraSeg genres.   #
# Weights control the genre mix of the corpus (roughly uniform-ish, with MSA  #
# prose/news slightly dominant since it is the broadest genre).               #
# --------------------------------------------------------------------------- #
GENRE_SEEDS: Dict[str, List[str]] = {
    # General MSA prose / news
    "msa_news": [
        "أعلنت وزارة",
        "في تطور جديد،",
        "قال مصدر مسؤول إن",
        "كشف تقرير حديث أن",
        "شهدت الأسواق العالمية",
        "أكد الخبراء أن",
        "وفي سياق متصل،",
        "أوضح البيان الرسمي أن",
    ],
    # Legal / constitutional — anchored on the "المادة" article phrasing
    "legal": [
        "المادة الأولى:",
        "المادة الثانية:",
        "المادة الثالثة:",
        "يحظر على كل شخص",
        "مع مراعاة أحكام هذا النظام،",
        "تلتزم الجهة المختصة بأن",
        "المادة الرابعة: لا يجوز",
        "وفقاً لأحكام المادة السابقة،",
    ],
    # Quran / religious
    "quran": [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "يا أيها الذين آمنوا",
        "إن الله",
        "قال تعالى",
        "وإذ قال ربك",
        "ذلك الكتاب لا ريب فيه",
    ],
    # Hadith / isnad — anchored on isnad openers حدثنا / عن / قال
    "hadith": [
        "حدثنا",
        "عن أبي هريرة رضي الله عنه قال",
        "قال رسول الله صلى الله عليه وسلم",
        "حدثنا محمد بن",
        "عن ابن عباس رضي الله عنهما",
        "أخبرنا",
        "عن أنس بن مالك قال",
    ],
}
GENRE_WEIGHTS: Dict[str, float] = {
    "msa_news": 0.34,
    "legal": 0.25,
    "quran": 0.16,
    "hadith": 0.25,
}

# --------------------------------------------------------------------------- #
# PREPOSITION-EMPHASIS MODE (additive, flag-gated: --prep-mode).               #
# ========================================================================== #
# Purpose: build a HUGE *unlabeled* Arabic corpus whose passages are rich in  #
# the exact preposition/connective words the segmenter confuses, for          #
# self-supervised (BYOL-style) pretraining. NO boundary labels are produced   #
# — output is {"text": ...} jsonl (one passage/line) only.                    #
#                                                                             #
# The preposition list is the SINGLE source of truth already used by the      #
# connective-augmentation path (src/connective_aug.CONNECTIVE_WORDS); we      #
# import it rather than duplicate a second list that could silently drift.    #
# The task's PREPOSITIONS list is asserted equal to it at import time so a     #
# future edit to either side fails loudly instead of quietly diverging.       #
# --------------------------------------------------------------------------- #
try:
    from connective_aug import CONNECTIVE_WORDS as _CONNECTIVE_WORDS  # noqa: E402
except Exception:  # pragma: no cover - fallback if import path differs
    _CONNECTIVE_WORDS = None

# The list the task pins us to (verbatim). Kept explicit so this file is
# self-documenting even if connective_aug is unavailable.
PREPOSITIONS: List[str] = [
    "و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
    "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى",
]
if _CONNECTIVE_WORDS is not None:
    # Fail loudly if the two ever diverge (single source of truth guard).
    assert list(_CONNECTIVE_WORDS) == PREPOSITIONS, (
        "PREPOSITIONS drifted from connective_aug.CONNECTIVE_WORDS; "
        "reconcile the two lists before generating a corpus."
    )
PREPOSITION_SET = set(PREPOSITIONS)

# Preposition-heavy seed prompts. Each seed front-loads several target
# prepositions/connectives so the model's continuation stays in a
# preposition-dense register. Grouped by the same four AraSeg genres so the
# corpus still spans the in-genre distribution while biasing toward
# preposition-rich prose. (Used ONLY when --prep-mode is set.)
PREP_GENRE_SEEDS: Dict[str, List[str]] = {
    "msa_news": [
        "وكان من المقرر أن",
        "ومن المعروف أن",
        "وقد أكد المصدر أنه، ومن ثم",
        "وفي هذا السياق، ومن جهة أخرى",
        "ولكن، وعلى الرغم من ذلك، فإن",
        "ومن ناحية أخرى، فقد",
        "وقال المتحدث إن، كما أضاف أن",
        "ثم إن الأمر، وبناء على ذلك، فإن",
    ],
    "legal": [
        "ومع مراعاة أحكام هذا النظام، فإنه",
        "ولكن يجوز، بل يتعين، على الجهة أن",
        "وفي حال، وبناء على ما تقدم، فإن",
        "ومن ثم، وطبقا لأحكام المادة، فإنه",
        "أو أن، وحتى في هذه الحالة، فإن",
        "وقد نصت المادة على أنه، ومن ثم",
        "لكن، ومع ذلك، وفقا لما سبق، فإن",
    ],
    "quran": [
        "وإذ قال ربك، ومن ثم",
        "ولكن أكثر الناس، بل إن",
        "فقال لهم، ومن بعد ذلك",
        "وفي ذلك، ومن آياته أن",
        "ثم إن ربك، وقد",
        "وكان الله، ومن رحمته أن",
        "فإن مع العسر، ومن ثم",
    ],
    "hadith": [
        "وعن أبي هريرة، ومن ثم قال",
        "فقال رسول الله، ثم إن",
        "وحدثنا، ومن طريق آخر، عن",
        "ولكن في رواية، فقال",
        "وقد روي، ومن وجه آخر، أنه",
        "وكان من دعائه، ومن ثم",
        "فإن النبي، وقد قال",
    ],
}


def preposition_count(tokens: List[str]) -> int:
    """Number of target-preposition occurrences in a whitespace token list.

    A token counts if it exactly equals one of PREPOSITIONS. We match on the
    whole whitespace token (not substrings) so that e.g. the standalone
    connective «و» is counted while «وزارة» is not — the same matching
    convention connective_aug uses (word-level, not clitic-splitting).
    """
    n = 0
    for t in tokens:
        if t in PREPOSITION_SET:
            n += 1
    return n


def preposition_density(tokens: List[str]) -> float:
    """Target prepositions per token. 0.0 for an empty passage."""
    if not tokens:
        return 0.0
    return preposition_count(tokens) / len(tokens)


def _normalize_for_dedup(text: str) -> str:
    """Whitespace-collapsed lowercase key for exact-dedup (Arabic has no case,
    but this also folds any stray Latin/digits)."""
    return " ".join(text.split()).lower()


def _doc_hash(text: str) -> str:
    return hashlib.sha1(_normalize_for_dedup(text).encode("utf-8")).hexdigest()


def collect_devtest_files(data_dir: str) -> List[str]:
    """Every data/*_dev.jsonl and data/*_test.jsonl file (leakage sources)."""
    files = sorted(
        glob.glob(os.path.join(data_dir, "*_dev.jsonl"))
        + glob.glob(os.path.join(data_dir, "*_test.jsonl"))
    )
    return files


def build_tripwire_from_devtest(data_dir: str, n: int = 8) -> Tuple[set, List[str]]:
    """Build the token 8-gram tripwire over every dev/test token stream."""
    files = collect_devtest_files(data_dir)
    streams = []
    for fp in files:
        try:
            docs = load_jsonl(fp)
        except Exception:
            continue
        # Keep only docs that actually carry a token list.
        streams.append([d for d in docs if isinstance(d.get("tokens"), list)])
    tripwire = build_ngram_tripwire(streams, n=n)
    return tripwire, files


def clean_generated_text(raw: str) -> str:
    """Light surface cleanup of a raw generation.

    - collapse all whitespace (including the model's newlines) to single spaces
      so each doc is exactly one line in the output file;
    - strip leading/trailing space.
    We deliberately do NOT strip punctuation — the encoder should see natural
    surface form. Boundaries are irrelevant here (no labels).
    """
    return " ".join(raw.split()).strip()


def pick_genre(rng) -> str:
    genres = list(GENRE_WEIGHTS.keys())
    weights = [GENRE_WEIGHTS[g] for g in genres]
    return rng.choices(genres, weights=weights, k=1)[0]


def build_prompt(rng, genre: str, prep_mode: bool = False) -> str:
    """Pick a seed prompt for a genre.

    In --prep-mode we draw from PREP_GENRE_SEEDS (preposition-front-loaded
    seeds) so the model's continuation stays in a preposition-dense register.
    Default (prep_mode=False) is unchanged: the original GENRE_SEEDS.
    """
    if prep_mode:
        return rng.choice(PREP_GENRE_SEEDS[genre])
    return rng.choice(GENRE_SEEDS[genre])


def generate_batch(model, tokenizer, prompts: List[str], device: str,
                   max_new_tokens: int, temperature: float, top_p: float,
                   repetition_penalty: float, no_repeat_ngram_size: int, seed: int):
    import torch

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts


def load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # aragpt2-mega ships custom modeling code and only loads with
    # trust_remote_code=True. We enable it ONLY when the operator explicitly asks
    # for mega on the command line (that is their approval); the clean-loading
    # default (medium) never executes remote code.
    trust = "aragpt2-mega" in model_name
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left padding so sampled continuations align for batched generate.
    tok.padding_side = "left"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=trust
    ).to(device)
    model.eval()
    return model, tok


def run(args) -> int:
    import torch

    # Defensive attribute resolution so run() is safe to call directly (not only
    # via main()): fills the prep-mode additions with their inert defaults.
    if not hasattr(args, "prep_mode"):
        args.prep_mode = False
    if not hasattr(args, "min_preps"):
        args.min_preps = 3
    if not hasattr(args, "min_prep_density"):
        args.min_prep_density = 0.06
    if getattr(args, "min_arabic_ratio", None) is None:
        args.min_arabic_ratio = 0.80 if args.prep_mode else 0.90

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    rng_seed = args.seed
    import random
    rng = random.Random(rng_seed)

    t0 = time.time()
    print(f"[gen_corpus] device={device} model={args.model}")
    model, tok = load_model(args.model, device)
    if device == "cuda":
        print(f"[gen_corpus] model loaded in {time.time()-t0:.1f}s | "
              f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB | "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    print("[gen_corpus] building 8-gram dev/test tripwire ...")
    tripwire, devtest_files = build_tripwire_from_devtest(DATA_DIR, n=args.tripwire_n)
    print(f"[gen_corpus] tripwire: {len(tripwire):,} unique {args.tripwire_n}-grams "
          f"from {len(devtest_files)} dev/test files")

    prep_mode = args.prep_mode
    if prep_mode:
        print(f"[gen_corpus] PREPOSITION-EMPHASIS MODE ON | "
              f"min_preps(K)={args.min_preps} min_density={args.min_prep_density} "
              f"prepositions={len(PREPOSITIONS)} | output=jsonl {{'text':...}} "
              f"(UNLABELED)")

    # Counters
    kept = 0
    approx_tokens = 0
    dropped = {
        "arabic_ratio": 0,
        "loop_distinct_ngram": 0,
        "loop_top_token": 0,
        "too_short": 0,
        "dedup": 0,
        "leak_8gram": 0,
    }
    if prep_mode:
        # Additional preposition-emphasis reject bucket (oversampling gate).
        dropped["too_few_preps"] = 0
    per_genre_kept = {g: 0 for g in GENRE_SEEDS}
    seen_hashes = set()
    sample_docs: List[Tuple[str, str]] = []  # (genre, text) for smoke printing
    # Prep-mode aggregate stats (accumulated over KEPT passages only).
    prep_dens_sum = 0.0
    prep_count_sum = 0
    ar_ratio_kept: List[float] = []  # script-ratio distribution over kept passages

    out_path = args.out
    # In prep-mode, if the operator did not override --out, default to a jsonl
    # path so we never clobber the plain-text gen_corpus.txt with jsonl content.
    if prep_mode and out_path == DEFAULT_OUT:
        out_path = DEFAULT_PREP_OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    target_tokens = args.target_tokens
    max_docs = args.max_docs  # smoke cap on KEPT docs (None -> unbounded)

    seed_counter = 0
    total_generated = 0  # every passage the model produced (kept or dropped)
    # Optional hard cap on GENERATED passages (prep-mode smoke uses this so the
    # throughput estimate is over a fixed ~200-passage budget). None -> no cap.
    gen_cap = getattr(args, "smoke_gen_cap", None)
    fout = None if args.dry_run else open(out_path, "w", encoding="utf-8")
    try:
        while approx_tokens < target_tokens:
            if max_docs is not None and kept >= max_docs:
                break
            if gen_cap is not None and total_generated >= gen_cap:
                break
            # Build a batch of prompts (may span genres).
            batch_genres = [pick_genre(rng) for _ in range(args.batch_size)]
            prompts = [build_prompt(rng, g, prep_mode=prep_mode)
                       for g in batch_genres]
            seed_counter += 1
            texts = generate_batch(
                model, tok, prompts, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                seed=rng_seed + seed_counter,
            )
            for genre, raw in zip(batch_genres, texts):
                total_generated += 1
                text = clean_generated_text(raw)
                toks = text.split()
                # too short (nothing to learn from)
                if len(toks) < args.min_tokens:
                    dropped["too_short"] += 1
                    continue
                # arabic script ratio. In prep-mode the floor is
                # args.min_arabic_ratio too, but its default is lowered to 0.80
                # by the prep-mode arg overrides (task requirement: >= 0.8),
                # so the same line enforces the correct threshold in both modes.
                ar = arabic_ratio(toks)
                if ar < args.min_arabic_ratio:
                    dropped["arabic_ratio"] += 1
                    continue
                # anti-loop: distinct 4-gram ratio
                if distinct_ngram_ratio(toks, args.ngram_n) < args.min_distinct_ngram:
                    dropped["loop_distinct_ngram"] += 1
                    continue
                # anti-loop: top-token frequency
                if top_token_frequency(toks) > args.max_top_token_freq:
                    dropped["loop_top_token"] += 1
                    continue
                # PREPOSITION-EMPHASIS gate (oversampling): keep only passages
                # carrying >= K target prepositions AND a density above the
                # floor. This is what biases the KEPT corpus toward
                # preposition-rich text on top of the biased seed prompts.
                # (No-op unless --prep-mode.)
                pc = 0
                pd = 0.0
                if prep_mode:
                    pc = preposition_count(toks)
                    pd = pc / max(len(toks), 1)
                    if pc < args.min_preps or pd < args.min_prep_density:
                        dropped["too_few_preps"] += 1
                        continue
                # exact dedup
                h = _doc_hash(text)
                if h in seen_hashes:
                    dropped["dedup"] += 1
                    continue
                # leakage tripwire (8-gram against dev/test)
                if shares_ngram(toks, tripwire, n=args.tripwire_n):
                    dropped["leak_8gram"] += 1
                    continue

                seen_hashes.add(h)
                kept += 1
                approx_tokens += len(toks)
                per_genre_kept[genre] += 1
                if prep_mode:
                    prep_dens_sum += pd
                    prep_count_sum += pc
                    ar_ratio_kept.append(ar)
                if fout is not None:
                    if prep_mode:
                        # UNLABELED jsonl: one {"text": ...} per line, NO labels.
                        fout.write(json.dumps({"text": text},
                                              ensure_ascii=False) + "\n")
                    else:
                        fout.write(text + "\n")
                if len(sample_docs) < args.n_samples:
                    sample_docs.append((genre, text))

            if kept and kept % args.log_every == 0:
                elapsed = time.time() - t0
                print(f"[gen_corpus] kept={kept} approx_tokens={approx_tokens:,} "
                      f"({approx_tokens/max(elapsed,1e-9):.0f} tok/s) elapsed={elapsed:.0f}s")
    finally:
        if fout is not None:
            fout.close()

    total_seen = kept + sum(dropped.values())
    dedup_rate = dropped["dedup"] / max(total_seen, 1)
    # Recompute corpus-level script ratio over kept docs (samples are a proxy;
    # for exact we scan kept text file if written).
    elapsed = time.time() - t0

    print("\n===== CORPUS STATS =====")
    print(f"model               : {args.model}")
    print(f"device              : {device}")
    print(f"kept docs           : {kept}")
    print(f"approx tokens (ws)  : {approx_tokens:,}")
    print(f"total generated     : {total_seen}")
    print(f"elapsed             : {elapsed:.1f}s "
          f"({approx_tokens/max(elapsed,1e-9):.0f} tok/s)")
    print(f"dedup rate          : {dedup_rate:.3f}")
    print("dropped by filter   :")
    for k, v in dropped.items():
        print(f"   {k:22s}: {v}")
    print("kept per genre      :")
    for g, v in per_genre_kept.items():
        print(f"   {g:22s}: {v}")

    # ------------------------------------------------------------------- #
    # PREPOSITION-EMPHASIS smoke report (only in --prep-mode).            #
    # Reports the four numbers the task asks for: survivors, mean prep    #
    # density, script-ratio distribution, and throughput (passages/min). #
    # ------------------------------------------------------------------- #
    if prep_mode:
        passages_per_min = kept / max(elapsed / 60.0, 1e-9)
        gen_passages_per_min = total_seen / max(elapsed / 60.0, 1e-9)
        survival_rate = kept / max(total_seen, 1)
        mean_density = prep_dens_sum / max(kept, 1)
        mean_preps = prep_count_sum / max(kept, 1)
        print("\n===== PREPOSITION-EMPHASIS STATS =====")
        print(f"survived (kept)     : {kept} / {total_seen} generated "
              f"({survival_rate*100:.1f}% survival)")
        print(f"mean prep density   : {mean_density:.4f} "
              f"(target prepositions per token, over kept passages)")
        print(f"mean preps / passage: {mean_preps:.2f} "
              f"(K threshold = {args.min_preps}, density floor = "
              f"{args.min_prep_density})")
        print(f"throughput          : {passages_per_min:.1f} kept passages/min "
              f"| {gen_passages_per_min:.1f} generated passages/min")
        # Script-ratio distribution over KEPT passages.
        if ar_ratio_kept:
            srt = sorted(ar_ratio_kept)
            def _q(p):
                if len(srt) == 1:
                    return srt[0]
                idx = p * (len(srt) - 1)
                lo = int(idx); hi = min(lo + 1, len(srt) - 1)
                return srt[lo] * (1 - (idx - lo)) + srt[hi] * (idx - lo)
            n_lt_08 = sum(1 for x in ar_ratio_kept if x < 0.80)
            n_lt_09 = sum(1 for x in ar_ratio_kept if x < 0.90)
            n_eq_1 = sum(1 for x in ar_ratio_kept if x >= 0.999)
            print("script-ratio dist   : "
                  f"min={srt[0]:.3f} p05={_q(0.05):.3f} p25={_q(0.25):.3f} "
                  f"median={_q(0.50):.3f} p75={_q(0.75):.3f} "
                  f"p95={_q(0.95):.3f} max={srt[-1]:.3f} "
                  f"mean={sum(srt)/len(srt):.3f}")
            print("script-ratio buckets: "
                  f"[<0.80]={n_lt_08} [0.80-0.90)={n_lt_09-n_lt_08} "
                  f"[0.90-1.0)={kept-n_lt_09-n_eq_1} [==1.0]={n_eq_1}")

    if sample_docs:
        print(f"\n===== SAMPLE DOCS (first {len(sample_docs)}) =====")
        for i, (genre, text) in enumerate(sample_docs):
            snippet = text if len(text) <= 220 else text[:220] + " …"
            stoks = text.split()
            ar = arabic_ratio(stoks)
            if prep_mode:
                pc = preposition_count(stoks)
                pd = pc / max(len(stoks), 1)
                print(f"[{i}] genre={genre} ar_ratio={ar:.2f} "
                      f"preps={pc} density={pd:.3f}")
            else:
                print(f"[{i}] genre={genre} ar_ratio={ar:.2f}")
            print(f"    {snippet}")

    if not args.dry_run:
        print(f"\n[gen_corpus] wrote {kept} docs -> {out_path}")

    if args.smoke:
        # Print the exact full-run command for the orchestrator.
        if prep_mode:
            print("\n===== SCALE-UP COMMAND (prep-emphasis, ~50M tokens) =====")
            print("# UNLABELED preposition-rich jsonl for BYOL pretraining. "
                  "Every kept line is {\"text\": ...}; no boundary labels.")
            print(f"python src/gen_corpus.py --prep-mode --model {args.model} "
                  f"--target-tokens 50000000 --out data/gen_corpus_prep.jsonl "
                  f"--batch-size 48 --max-new-tokens 180 "
                  f"--min-preps {args.min_preps} "
                  f"--min-prep-density {args.min_prep_density} "
                  f"--min-arabic-ratio {args.min_arabic_ratio}")
            print("# Tune corpus size via --target-tokens (100M for a HUGE run). "
                  "Estimate wall-clock from the passages/min above.")
        else:
            print("\n===== FULL-RUN COMMAND (~1M tokens) =====")
            # Batch 48 @ 180 new tokens measured at 1.82GB peak VRAM / 1223 tok/s
            # on the 5090 — huge headroom, so we recommend the fast batch-48
            # recipe.
            print(f"python src/gen_corpus.py --model {args.model} "
                  f"--target-tokens 1000000 --out data/gen_corpus.txt "
                  f"--batch-size 48 --max-new-tokens 180")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF causal-LM id or local dir (largest AraGPT2 that loads)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output corpus (one doc/line)")
    ap.add_argument("--target-tokens", type=int, default=1_000_000,
                    help="stop after ~this many kept whitespace tokens")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="cap on KEPT docs (smoke sets this small)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="force CPU (debug only)")
    # generation
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    # quality filters
    ap.add_argument("--min-tokens", type=int, default=20)
    # Sentinel default: resolved in main() to 0.90 (normal) or 0.80 (prep-mode,
    # per the task's >= 0.8 requirement) unless the operator sets it explicitly.
    ap.add_argument("--min-arabic-ratio", type=float, default=None,
                    help="min Arabic-script ratio (default 0.90; 0.80 in "
                         "--prep-mode unless overridden)")
    ap.add_argument("--ngram-n", type=int, default=4)
    ap.add_argument("--min-distinct-ngram", type=float, default=0.55)
    ap.add_argument("--max-top-token-freq", type=float, default=0.35)
    ap.add_argument("--tripwire-n", type=int, default=8)
    # preposition-emphasis mode (additive; default OFF -> byte-identical to
    # yesterday's behavior on every existing flag)
    ap.add_argument("--prep-mode", action="store_true",
                    help="preposition-emphasis mode: prep-heavy seeds + "
                         "prep-density oversampling gate; writes UNLABELED "
                         "jsonl {\"text\": ...} (no boundary labels)")
    ap.add_argument("--min-preps", type=int, default=3,
                    help="prep-mode: keep only passages with >= this many "
                         "target prepositions (K)")
    ap.add_argument("--min-prep-density", type=float, default=0.06,
                    help="prep-mode: keep only passages with >= this "
                         "prepositions-per-token density")
    # smoke / logging
    ap.add_argument("--smoke", action="store_true",
                    help="GPU smoke: ~30 kept docs, print samples + full-run cmd")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="do not write the file")
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    # Resolve the Arabic-ratio sentinel: 0.80 in prep-mode (task requires
    # >= 0.8), else 0.90 (unchanged default). An explicit --min-arabic-ratio
    # on the command line always wins.
    if args.min_arabic_ratio is None:
        args.min_arabic_ratio = 0.80 if args.prep_mode else 0.90
    if args.smoke:
        if args.prep_mode:
            # Prep-mode smoke: the task asks for ~200 generated passages so we
            # can estimate scaling. We cap on GENERATED count (via max-docs on
            # kept is too tight because prep-gate rejects a lot), so we instead
            # set a generous kept cap and a token target large enough that the
            # ~200-passage generation budget (not the token target) is the
            # binding limit. batch 20 x 10 rounds ~= 200 generated.
            if args.max_docs is None:
                args.max_docs = 200          # cap on KEPT passages
            args.target_tokens = min(args.target_tokens, 200_000)
            args.max_new_tokens = min(args.max_new_tokens, 120)
            args.batch_size = min(args.batch_size, 20)
            args.smoke_gen_cap = 200         # hard cap on GENERATED passages
        else:
            # Smoke overrides: small kept-doc cap, short docs, quick.
            if args.max_docs is None:
                args.max_docs = 30
            args.target_tokens = min(args.target_tokens, 10_000)
            args.max_new_tokens = min(args.max_new_tokens, 80)
            args.batch_size = min(args.batch_size, 10)
    if not hasattr(args, "smoke_gen_cap"):
        args.smoke_gen_cap = None
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
