"""DAPT — Domain-Adaptive continued PreTraining for AraSeg 2026.

Continued **whole-word masked-LM** on a LARGE unsupervised Arabic corpus, then
the existing pipeline (``train_encoder.py --model-name runs/dapt_gen``) fine-tunes
the boundary head from the domain-adapted encoder. This is the DAPT half of
Gururangan et al. (ACL 2020, "Don't Stop Pretraining"): continued MLM on a broad
*domain* corpus (as opposed to TAPT, which is MLM on the task's own tiny train
text). AdaptaBERT (Han & Eisenstein, EMNLP 2019) is the same recipe.

WHY THIS EXISTS (the bug we are fixing)
---------------------------------------
An earlier TAPT run HURT the segmenter (-0.61 F1). The cause was overfitting:
50 epochs of MLM on only ~74k tokens (the 174 labeled train docs) let the encoder
memorize the tiny corpus and drift off the AraBERT manifold. The two fixes,
both enforced here:
  1. MUCH larger corpus — an arbitrary plain-text generated corpus, not the 174
     labeled docs. Default ``data/gen_corpus.txt``, one document per line.
  2. A CONTROLLED STEP BUDGET — defaults (``--epochs 3``) plus an explicit
     ``--max-steps`` cap so the model sees the corpus only ~2-4 times total.
     With a big corpus 3 epochs is a *short* schedule, the opposite of the
     50-epochs-on-74k-tokens regime that overfit.

WHAT THE GENERATED CORPUS IS (and is NOT)
-----------------------------------------
The generated text is used ONLY for unsupervised masked-LM pretraining of the
encoder. It is NEVER used as labeled boundary data. Boundary supervision stays
100% the real 174 AraSeg docs, fine-tuned afterwards by ``train_encoder.py`` on
``data/{TASK}_train.jsonl`` exactly as before. This script never reads any
labeled jsonl for training; it reads exactly one plain-text corpus file.

LEAKAGE GUARD (non-negotiable)
------------------------------
Before any training, the corpus is filtered against the dev/test splits: any
generated document sharing an 8-gram (token 8-gram) with ANY
``data/*_dev.jsonl`` or ``data/*_test.jsonl`` token stream is DROPPED. The
8-gram tripwire is the SAME mechanism used by ``synth_filter.py`` (reused here,
adapted to plain text by whitespace-tokenizing each corpus line). Dev/test are
read for this one purpose only — no statistic, threshold, or label is derived
from them. Absence of a dev/test file is loud but allowed (a bare smoke env);
a missing file simply is not covered by the tripwire.

MASKING (identical to src/tapt.py)
----------------------------------
15% masking, whole-word (all subwords of a chosen word masked together — the
AraBERT convention), 80/10/10 mask/random/keep via the standard
``DataCollatorForWholeWordMask`` (which implements exactly that scheme at the
whole-word level). ``return_special_tokens_mask=True`` excludes
[CLS]/[SEP]/[PAR]/pad from the 15% pool, as the authors' ``mask_tokens`` does.
Paragraph tokens ("\\n") are mapped to the [PAR] special token the fine-tuner
uses, so the encoder sees the same surface vocabulary at adaptation and fine-tune
time. One corpus line == one MLM "document" (the authors' --line_by_line regime).

ADDITIVE + FLAG-GATED
---------------------
NEW, standalone file. It does not import from, modify, or change any default of
train_encoder.py, tapt.py, or any other load-bearing file. With DAPT simply not
run, the pipeline reproduces today's baseline bit-for-bit; DAPT only takes effect
when the operator points ``train_encoder.py --model-name`` at the adapted-encoder
directory this script writes (default ``runs/dapt_gen``). ``runs/`` is
append-only — this script refuses to write into a non-empty output dir.

Output: a self-contained HF encoder directory ``runs/dapt_gen/`` (config +
weights + tokenizer) loadable by ``train_encoder.py --model-name runs/dapt_gen``
exactly like the hub checkpoint.

ORCHESTRATOR — the 2-command sequence
-------------------------------------
  # 1) domain-adapt the encoder on the generated corpus (unsupervised MLM):
  python src/dapt.py --corpus data/gen_corpus.txt --out-dir runs/dapt_gen \
      --epochs 3 --lr 5e-5 --batch-size 16
  # 2) fine-tune the boundary head on the REAL 174 labeled docs:
  python src/train_encoder.py --task NoPnx-NP --model-name runs/dapt_gen \
      --out-dir runs/nopnx-np-dapt

GPU SMOKE (tiny, ~30s on the RTX 5090):
  python src/dapt.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Match the fine-tuner's paragraph symbol. Import from src.data when importable
# (running as `python src/dapt.py` puts src/ on the path); fall back to the
# literal so this module never hard-depends on data/ being present.
try:  # pragma: no cover - import shim
    from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN
except Exception:  # noqa: BLE001
    try:
        from src.data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN  # type: ignore
    except Exception:  # noqa: BLE001
        MODEL_PAR_TOKEN, PARAGRAPH_TOKEN = "[PAR]", "\n"

# Reuse the EXACT 8-gram tripwire mechanism from synth_filter.py so the leakage
# guard is identical to the one already on record. If the import fails (bare
# env), a local copy of the two tiny helpers is defined below — same code.
try:  # pragma: no cover - import shim
    from synth_filter import _ngrams as _sf_ngrams  # type: ignore
    from synth_filter import build_ngram_tripwire as _sf_build_tripwire  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from src.synth_filter import _ngrams as _sf_ngrams  # type: ignore
        from src.synth_filter import build_ngram_tripwire as _sf_build_tripwire  # type: ignore
    except Exception:  # noqa: BLE001
        _sf_ngrams = None
        _sf_build_tripwire = None

# On Windows the default console codec (cp1252) cannot encode Arabic; force UTF-8.
try:  # pragma: no cover - platform shim
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")


# --------------------------------------------------------------------------- #
# 8-gram leakage tripwire (reused from synth_filter; local fallback = same code)
# --------------------------------------------------------------------------- #
def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if _sf_ngrams is not None:
        return _sf_ngrams(tokens, n)
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_devtest_tripwire(data_dir: str, tripwire_n: int = 8,
                           tasks: Sequence[str] | None = None) -> set:
    """Build a token-8-gram set from every ``*_dev.jsonl`` / ``*_test.jsonl`` in
    ``data_dir``. A generated doc sharing any of these 8-grams is leakage.

    Reads the dev/test files directly (one JSON doc per line, ``tokens`` field),
    so it does not require src.data / the HF datasets package. Files that are
    absent are simply not covered — noted loudly by the caller.
    """
    import glob
    import json

    streams: List[List[dict]] = []
    covered: List[str] = []
    patterns = ["*_dev.jsonl", "*_test.jsonl"]
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(data_dir, pat))):
            base = os.path.basename(path)
            if tasks is not None:
                # keep only files whose {task}_ prefix is in the requested set
                if not any(base == f"{t}_dev.jsonl" or base == f"{t}_test.jsonl"
                           for t in tasks):
                    continue
            if path in seen:
                continue
            seen.add(path)
            docs = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        if "tokens" in d:
                            docs.append({"tokens": d["tokens"]})
            streams.append(docs)
            covered.append(base)
    if covered:
        print(f"[dapt] leakage tripwire covers {len(covered)} dev/test file(s): "
              f"{', '.join(covered)}")
    else:
        print("[dapt] WARNING: no *_dev.jsonl / *_test.jsonl found under "
              f"{data_dir} — leakage tripwire is EMPTY (no guard this run).",
              file=sys.stderr)
    if _sf_build_tripwire is not None:
        return _sf_build_tripwire(streams, n=tripwire_n)
    grams: set = set()
    for docs in streams:
        for d in docs:
            for g in _ngrams(d["tokens"], tripwire_n):
                grams.add(g)
    return grams


def filter_corpus_lines(lines: List[str], tripwire: set,
                        tripwire_n: int = 8) -> tuple[List[str], int]:
    """Drop any corpus line sharing an 8-gram with dev/test.

    Each plain-text line is whitespace-tokenized (the same surface tokenization
    the AraSeg jsonl uses) and checked against the tripwire. Returns
    (kept_lines, n_dropped). An empty tripwire keeps everything (and the caller
    has already warned that there is no guard).
    """
    if not tripwire:
        return list(lines), 0
    kept, dropped = [], 0
    for ln in lines:
        toks = ln.split()
        leaks = False
        for g in _ngrams(toks, tripwire_n):
            if g in tripwire:
                leaks = True
                break
        if leaks:
            dropped += 1
        else:
            kept.append(ln)
    return kept, dropped


# --------------------------------------------------------------------------- #
# Corpus reading + MLM line construction.                                      #
# --------------------------------------------------------------------------- #
def read_corpus(path: str, max_docs: int | None = None) -> List[str]:
    """Read an arbitrary plain-text corpus: one document per (non-blank) line."""
    lines: List[str] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s:
                lines.append(s)
                if max_docs is not None and len(lines) >= max_docs:
                    break
    return lines


def normalize_par(lines: List[str]) -> List[str]:
    """Map any literal paragraph tokens ("\\n") to the fine-tuner's [PAR] symbol.

    A plain-text corpus rarely contains a bare newline TOKEN, but if a generator
    emits ``\\n`` as a standalone whitespace-separated token (matching the PA
    jsonl convention) we normalize it so the encoder sees [PAR], exactly like
    tapt.py's build_mlm_lines does for the labeled docs. Real embedded newlines
    were already stripped by read_corpus (line-per-doc), so this only touches an
    explicit ``\\n`` token.
    """
    out = []
    for ln in lines:
        toks = [MODEL_PAR_TOKEN if t == PARAGRAPH_TOKEN else t for t in ln.split()]
        s = " ".join(toks).strip()
        if s:
            out.append(s)
    return out


def encode_lines(lines: List[str], tokenizer, max_length: int):
    """Tokenize each line for MLM (same as tapt.py.encode_lines)."""
    from datasets import Dataset

    ds = Dataset.from_dict({"text": lines})

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])


# --------------------------------------------------------------------------- #
# Core: run DAPT.                                                              #
# --------------------------------------------------------------------------- #
def run_dapt(args) -> dict:
    set_seed(args.seed)

    # ---- corpus (plain text, NOT labeled jsonl) ----
    if not os.path.exists(args.corpus):
        raise SystemExit(
            f"REFUSED: corpus file not found: {args.corpus!r}. DAPT trains on an "
            "arbitrary plain-text corpus (one doc per line), e.g. data/gen_corpus.txt.")
    base = os.path.basename(args.corpus).lower()
    if base.endswith(".jsonl") or "_train" in base or "_dev" in base or "_test" in base:
        raise SystemExit(
            f"REFUSED: --corpus {args.corpus!r} looks like a labeled AraSeg split. "
            "DAPT is UNSUPERVISED and must read a plain-text corpus, never the "
            "labeled boundary data. Point --corpus at generated plain text.")

    raw_lines = read_corpus(args.corpus, max_docs=args.max_docs)
    print(f"[dapt] read corpus={args.corpus}  docs(lines)={len(raw_lines)}")

    # ---- leakage guard: drop lines sharing an 8-gram with dev/test ----
    tasks = args.filter_tasks.split(",") if args.filter_tasks else None
    tripwire = build_devtest_tripwire(args.data_dir, tripwire_n=args.tripwire_n,
                                      tasks=tasks)
    print(f"[dapt] tripwire size = {len(tripwire)} distinct {args.tripwire_n}-grams")
    lines, n_leak = filter_corpus_lines(raw_lines, tripwire, args.tripwire_n)
    print(f"[dapt] leakage filter: dropped {n_leak} / {len(raw_lines)} docs "
          f"(shared an {args.tripwire_n}-gram with dev/test); kept {len(lines)}")

    lines = normalize_par(lines)
    if not lines:
        raise SystemExit(
            "REFUSED: 0 docs remain after leakage filtering — nothing to train on. "
            "Check the corpus and the dev/test tripwire coverage.")
    n_words = sum(len(ln.split()) for ln in lines)
    print(f"[dapt] training on {len(lines)} docs  ~{n_words} whitespace-tokens "
          f"(UNSUPERVISED masked-LM; no boundary labels involved)")

    # ---- model + tokenizer (start from AraBERTv02 by default) ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    ds = encode_lines(lines, tokenizer, args.max_length)

    if args.whole_word_mask:
        collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    else:
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # runs/ is append-only: refuse to write into a non-empty out-dir.
    if os.path.isdir(args.out_dir) and any(
            not f.startswith(".") for f in os.listdir(args.out_dir)):
        raise SystemExit(
            f"REFUSED: out-dir {args.out_dir!r} exists and is non-empty. runs/ is "
            "append-only — choose a new, dated output directory.")

    # CONTROLLED STEP BUDGET. Default epochs=3 on a LARGE corpus = a short
    # schedule (~2-4 passes). --max-steps, when >0, hard-caps optimizer steps
    # (HF honors max_steps OVER num_train_epochs), the explicit anti-overfit lever.
    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ckpts"),
        overwrite_output_dir=False,          # runs/ is append-only
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        learning_rate=args.lr,               # 5e-5 (task-specified DAPT default)
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,      # 0.06
        weight_decay=args.weight_decay,      # 0.01
        logging_steps=args.logging_steps,
        save_strategy="no",                  # save the final encoder once, below
        report_to="none",
        fp16=use_cuda,                       # fp16 on GPU
        no_cuda=not use_cuda,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    steps_planned = trainer.args.max_steps if trainer.args.max_steps > 0 else None
    print(f"[dapt] recipe: epochs={args.epochs} max_steps={args.max_steps} "
          f"lr={args.lr} bs={args.batch_size} warmup={args.warmup_ratio} "
          f"wd={args.weight_decay} wwm={args.whole_word_mask} "
          f"mlm_p={args.mlm_probability} fp16={use_cuda} "
          f"device={'cuda' if use_cuda else 'cpu'}")

    out = trainer.train()

    # MLM loss trajectory — the smoke uses this to PROVE the loss decreases.
    hist = [h for h in trainer.state.log_history if "loss" in h]
    first = last = None
    if hist:
        first, last = hist[0]["loss"], hist[-1]["loss"]
        print(f"[dapt] MLM train loss: first-logged={first:.4f}  "
              f"final-logged={last:.4f}  delta={last - first:+.4f}")
    print(f"[dapt] final train loss (mean over run) = {out.training_loss:.4f}")

    # ---- save the adapted encoder (loadable by train_encoder.py --model-name) ----
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.save_model(args.out_dir)         # config + weights
    tokenizer.save_pretrained(args.out_dir)  # tokenizer + [PAR] special token
    print(f"[dapt] saved adapted encoder -> {args.out_dir}")
    print("[dapt] next: python src/train_encoder.py --task <TASK> "
          f"--model-name {args.out_dir} --out-dir runs/<finetune-out>")

    return {
        "n_docs_in": len(raw_lines),
        "n_leak_dropped": n_leak,
        "n_docs_trained": len(lines),
        "n_words": n_words,
        "first_loss": first,
        "last_loss": last,
        "train_loss": out.training_loss,
        "out_dir": args.out_dir,
        "global_step": trainer.state.global_step,
    }


# --------------------------------------------------------------------------- #
# GPU SMOKE.                                                                   #
# --------------------------------------------------------------------------- #
def _make_dummy_corpus(path: str, n_lines: int = 200) -> None:
    """Write a tiny multi-genre Arabic dummy corpus (one doc per line).

    Covers the AraSeg genres so masking sees the real surface vocabulary:
    general MSA prose/news, legal ("المادة"), Quran/religious, hadith/isnad
    openers (حدثنا / عن / قال). NOT labeled — plain text only.
    """
    prose = ("قالت الوزارة في بيان لها إن الخطة الجديدة تهدف إلى تطوير قطاع "
             "التعليم ورفع كفاءة المعلمين في جميع المدارس الحكومية خلال العام المقبل")
    news = ("أعلن المتحدث الرسمي أمس أن الحكومة وافقت على مشروع الميزانية بعد "
            "مناقشات طويلة استمرت عدة أيام في مجلس النواب")
    legal = ("المادة الأولى يعمل بأحكام هذا القانون من تاريخ نشره في الجريدة "
             "الرسمية وتلغى كل الأحكام المخالفة له المادة الثانية على الوزراء "
             "المختصين تنفيذ هذا القرار")
    quran = ("الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين إياك نعبد وإياك "
             "نستعين اهدنا الصراط المستقيم")
    hadith = ("حدثنا أبو بكر بن أبي شيبة قال حدثنا عبد الله بن إدريس عن ربيعة عن "
              "القاسم عن عائشة رضي الله عنها قالت كان النبي صلى الله عليه وسلم يذكر "
              "الله على كل أحيانه")
    isnad = ("عن ابن عمر رضي الله عنهما قال قال رسول الله صلى الله عليه وسلم بني "
             "الإسلام على خمس شهادة أن لا إله إلا الله وإقام الصلاة")
    bank = [prose, news, legal, quran, hadith, isnad]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            # vary each line a little so the corpus is not 200 identical docs
            body = bank[i % len(bank)]
            f.write(f"{body} رقم الوثيقة {i}\n")


def smoke(args) -> int:
    """Tiny GPU smoke: 200-line dummy corpus, 2 epochs, prove loss decreases,
    encoder saves and reloads as AutoModelForTokenClassification(num_labels=2)."""
    print("=" * 72)
    print("DAPT GPU SMOKE (tiny dummy corpus; real device)")
    print("=" * 72)
    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"[smoke] cuda_available={torch.cuda.is_available()} using_cuda={use_cuda} "
          f"device={torch.cuda.get_device_name(0) if use_cuda else 'cpu'}")
    if not use_cuda and not args.cpu:
        print("[smoke] WARNING: CUDA not available — task requires GPU-SMOKE on the "
              "REAL GPU. Proceeding on CPU only because it was forced.",
              file=sys.stderr)

    import tempfile
    tmp = tempfile.mkdtemp(prefix="dapt_smoke_")
    corpus = os.path.join(tmp, "dummy_corpus.txt")
    _make_dummy_corpus(corpus, n_lines=args.smoke_lines)
    print(f"[smoke] wrote {args.smoke_lines}-line dummy corpus -> {corpus}")

    out_dir = args.out_dir if args.out_dir != DEFAULT_OUT else os.path.join(
        tmp, "dapt_smoke_out")

    smoke_args = argparse.Namespace(
        corpus=corpus,
        model_name=args.model_name,
        out_dir=out_dir,
        data_dir=args.data_dir,
        filter_tasks=args.filter_tasks,
        tripwire_n=args.tripwire_n,
        mlm_probability=args.mlm_probability,
        whole_word_mask=args.whole_word_mask,
        max_length=128,               # short blocks -> fast smoke
        epochs=2.0,                   # task: run 2 epochs on GPU
        max_steps=args.max_steps,     # honored if operator passed one
        lr=args.lr,
        batch_size=8,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_docs=None,
        logging_steps=1,              # trace the loss curve
        cpu=args.cpu,
    )

    stats = run_dapt(smoke_args)

    # ---- prove the encoder reloads as a token-classifier (num_labels=2) ----
    print("\n[smoke] reload check: AutoModelForTokenClassification(num_labels=2) "
          f"from {out_dir}")
    tok = AutoTokenizer.from_pretrained(out_dir)
    clf = AutoModelForTokenClassification.from_pretrained(out_dir, num_labels=2)
    if use_cuda:
        clf = clf.to("cuda")
    # one tiny forward pass on the real device to catch device/OOM bugs
    enc = tok(["حدثنا أبو بكر عن عائشة قالت كان النبي يذكر الله",
               "المادة الأولى يعمل بأحكام هذا القانون"],
              is_split_into_words=False, padding=True, truncation=True,
              max_length=64, return_tensors="pt")
    if use_cuda:
        enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        logits = clf(**enc).logits
    print(f"[smoke] token-classifier forward ok: logits shape={tuple(logits.shape)} "
          f"(expect [..,..,2]); num_labels={clf.config.num_labels}")

    # ---- verdict ----
    loss_ok = (stats["first_loss"] is not None and stats["last_loss"] is not None
               and stats["last_loss"] < stats["first_loss"])
    shape_ok = logits.shape[-1] == 2 and clf.config.num_labels == 2
    saved_ok = os.path.exists(os.path.join(out_dir, "config.json")) and (
        os.path.exists(os.path.join(out_dir, "model.safetensors"))
        or os.path.exists(os.path.join(out_dir, "pytorch_model.bin")))

    print("\n" + "=" * 72)
    print("SMOKE RESULTS")
    print(f"  corpus docs in / leak-dropped / trained : "
          f"{stats['n_docs_in']} / {stats['n_leak_dropped']} / {stats['n_docs_trained']}")
    print(f"  MLM loss first -> last                  : "
          f"{stats['first_loss']:.4f} -> {stats['last_loss']:.4f} "
          f"(delta {stats['last_loss'] - stats['first_loss']:+.4f})")
    print(f"  global steps                            : {stats['global_step']}")
    print(f"  loss decreased                          : {loss_ok}")
    print(f"  encoder saved (config + weights)        : {saved_ok}")
    print(f"  reloads as TokenClassification(2)       : {shape_ok}")
    print(f"  device                                  : "
          f"{'cuda' if use_cuda else 'cpu'} (no device/OOM error)")
    ok = loss_ok and shape_ok and saved_ok
    print("=" * 72)
    print("DAPT SMOKE: PASS" if ok else "DAPT SMOKE: FAIL")
    print("=" * 72)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
DEFAULT_OUT = "runs/dapt_gen"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/gen_corpus.txt",
                    help="plain-text corpus, ONE doc per line (NOT labeled jsonl). "
                         "Default: data/gen_corpus.txt")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02",
                    help="base encoder to continue-pretrain (default: AraBERTv02)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="adapted-encoder dir to write (default: runs/dapt_gen)")
    # ---- leakage guard ----
    ap.add_argument("--data-dir", default=DATA,
                    help="dir holding *_dev.jsonl / *_test.jsonl for the 8-gram "
                         "leakage tripwire")
    ap.add_argument("--filter-tasks", default=None,
                    help="comma-separated task prefixes to build the tripwire from "
                         "(e.g. 'NoPnx-NP,PA'); default = ALL *_dev/*_test files")
    ap.add_argument("--tripwire-n", type=int, default=8,
                    help="n-gram size for the dev/test leakage tripwire (default 8)")
    # ---- MLM recipe (masking identical to tapt.py) ----
    ap.add_argument("--mlm-probability", type=float, default=0.15,
                    help="masking ratio (BERT/AraBERT default 0.15)")
    ap.add_argument("--whole-word-mask", dest="whole_word_mask",
                    action="store_true", default=True,
                    help="mask all subwords of a chosen word together "
                         "(AraBERT convention; default ON)")
    ap.add_argument("--no-whole-word-mask", dest="whole_word_mask",
                    action="store_false",
                    help="fall back to plain subword MLM")
    ap.add_argument("--max-length", type=int, default=512,
                    help="MLM block/sequence length")
    # ---- CONTROLLED STEP BUDGET (the anti-overfit fix) ----
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="MLM epochs (DEFAULT 3: on a LARGE corpus this is a SHORT "
                         "~2-4-pass schedule — the opposite of the 50-epochs-on-74k "
                         "run that overfit and hurt TAPT by -0.61)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="hard cap on optimizer steps (>0 OVERRIDES --epochs; the "
                         "explicit anti-overfit lever for a fixed step budget)")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="MLM learning rate (task-specified DAPT default 5e-5)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="per-device MLM batch size")
    ap.add_argument("--warmup-ratio", type=float, default=0.06,
                    help="warmup fraction (task-specified 0.06)")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="weight decay (task-specified 0.01)")
    ap.add_argument("--seed", type=int, default=42)
    # ---- smoke / plumbing ----
    ap.add_argument("--max-docs", type=int, default=None,
                    help="cap #corpus docs used (default: all)")
    ap.add_argument("--logging-steps", type=int, default=20,
                    help="log MLM loss every N steps")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (default uses CUDA if available)")
    ap.add_argument("--smoke", action="store_true",
                    help="run the tiny GPU smoke (200-line dummy corpus, 2 epochs) "
                         "and exit")
    ap.add_argument("--smoke-lines", type=int, default=200,
                    help="dummy-corpus size for --smoke (default 200)")
    return ap


def main(argv: List[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.smoke:
        return smoke(args)
    run_dapt(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
