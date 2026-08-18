"""TAPT — Task-Adaptive PreTraining for AraSeg 2026 (Gururangan et al., ACL 2020).

Continued **masked-LM** on our OWN task's training text, then the existing
pipeline fine-tunes the boundary head from the adapted encoder. This is the
"Don't Stop Pretraining" TAPT recipe (MLM on the task's own unlabeled text),
which is also the AdaptaBERT recipe (Han & Eisenstein, EMNLP 2019: MLM on
target-domain text -> task fine-tune).

Transferred faithfully from the authors' repo
  vendor/dont-stop-pretraining  (allenai/dont-stop-pretraining, ACL 2020)
specifically ADAPTIVE_PRETRAINING.md (the TAPT invocation) and
scripts/run_language_modeling.py::mask_tokens (the 80/10/10 masking rule). We
reproduce that recipe on HuggingFace `Trainer` + the standard whole-word-mask
data collator (which implements the identical 15% / 80-10-10 scheme at the
whole-word level — the AraBERT pretraining convention).

HARD CONSTRAINTS honored here (see CLAUDE.md, DIRECTIVES):
  * TRAIN-ONLY. This script reads exactly one file: data/{TASK}_train.jsonl
    (the 174 labeled docs). It STRIPS the labels — MLM is unsupervised, it uses
    only the raw token text. It never opens a dev/test/held/external file. There
    is no code path here that can touch a non-train split.
  * ADDITIVE + FLAG-GATED at the pipeline level. This is a NEW, standalone
    script. It does not import, modify, or change any default of
    train_encoder.py or any other load-bearing file. With TAPT simply not run,
    the pipeline reproduces today's baseline bit-for-bit; TAPT only takes effect
    when the operator explicitly points `train_encoder.py --model-name` at the
    adapted-encoder directory this script writes.

Input text construction (matched to fine-tuning): `train_encoder.py` feeds the
AraSeg whitespace tokens to the encoder RAW via `is_split_into_words=True` with
no AraBERT/Farasa preprocessing. We therefore build each MLM line by joining a
doc's tokens with single spaces and NOTHING else, so the encoder adapts to the
exact surface form it will later be fine-tuned and evaluated on. Paragraph
tokens ("\\n") are mapped to the same [PAR] special token the fine-tuner uses.

Output: a self-contained HF encoder directory `runs/tapt_{TASK}/` (config +
weights + tokenizer) that `train_encoder.py --model-name runs/tapt_{TASK}` loads
exactly like the hub checkpoint.

Example (full GPU run, orchestrator):
  python src/tapt.py --task NoPnx-NP --out-dir runs/tapt_NoPnx-NP \
      --epochs 100 --lr 1e-4 --batch-size 16

CPU smoke:
  python src/tapt.py --task NoPnx-NP --out-dir runs/tapt_smoke_NoPnx-NP \
      --epochs 2 --batch-size 4 --max-docs 12 --cpu --seed 42
"""
from __future__ import annotations

import argparse
import os
from typing import List

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task


def build_mlm_lines(docs: List[dict]) -> List[str]:
    """Join each doc's tokens into ONE raw-text line for line-by-line MLM.

    Labels are dropped entirely (MLM is unsupervised). Paragraph "\\n" tokens
    are replaced by the fine-tuner's [PAR] special token so the encoder sees the
    same surface vocabulary at adaptation time and at fine-tune time. One line
    per document = the authors' `--line_by_line` regime (each doc is a "document"
    in run_language_modeling.py terms).
    """
    lines = []
    for d in docs:
        toks = [MODEL_PAR_TOKEN if t == PARAGRAPH_TOKEN else t for t in d["tokens"]]
        line = " ".join(toks).strip()
        if line:
            lines.append(line)
    return lines


def encode_lines(lines: List[str], tokenizer, max_length: int):
    """Tokenize each line for MLM. `return_special_tokens_mask=True` lets the
    collator exclude [CLS]/[SEP]/[PAR]/pad from the 15% masking pool, exactly as
    the authors' mask_tokens() does via get_special_tokens_mask()."""
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", required=True, choices=sorted(TASKS),
                    help="AraSeg variant; reads data/{TASK}_train.jsonl ONLY")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02",
                    help="base encoder to continue-pretrain (default: AraBERTv02)")
    ap.add_argument("--train-jsonl", default=None,
                    help="override path to the TRAIN jsonl (train split only; "
                         "never point this at dev/test)")
    ap.add_argument("--out-dir", required=True,
                    help="adapted-encoder dir to write, e.g. runs/tapt_NoPnx-NP")
    # --- MLM recipe (defaults = the paper's TAPT config, ADAPTIVE_PRETRAINING.md) ---
    ap.add_argument("--mlm-probability", type=float, default=0.15,
                    help="masking ratio (Devlin/BERT + paper default 0.15)")
    ap.add_argument("--whole-word-mask", dest="whole_word_mask",
                    action="store_true", default=True,
                    help="mask all subwords of a chosen whole word together "
                         "(AraBERT convention; default ON)")
    ap.add_argument("--no-whole-word-mask", dest="whole_word_mask",
                    action="store_false",
                    help="fall back to plain subword MLM (authors' base collator)")
    ap.add_argument("--max-length", type=int, default=512,
                    help="MLM block/sequence length")
    ap.add_argument("--epochs", type=float, default=100.0,
                    help="MLM epochs (paper uses 100 for tiny corpora; our "
                         "corpus is ~74-98k tokens, so a short schedule suffices)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="MLM learning rate (paper TAPT = 1e-4)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="per-device MLM batch size (paper = 16)")
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    # --- smoke / plumbing ---
    ap.add_argument("--max-docs", type=int, default=None,
                    help="CPU smoke: cap #train docs used for MLM (default: all)")
    ap.add_argument("--logging-steps", type=int, default=10,
                    help="log MLM loss every N steps (smoke uses 1 to trace the "
                         "loss curve)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (smoke); default uses CUDA if available")
    args = ap.parse_args()

    set_seed(args.seed)

    # ---- TRAIN-ONLY guard: load the train split and NOTHING else. ----
    if args.train_jsonl is not None:
        base = os.path.basename(args.train_jsonl)
        if "dev" in base or "test" in base or "held" in base:
            raise SystemExit(
                f"REFUSED: --train-jsonl {args.train_jsonl!r} looks like a "
                "non-train split. TAPT is TRAIN-ONLY (organizer-ruled).")
    train_docs = load_task(args.task, "train", jsonl_path=args.train_jsonl)
    if args.max_docs is not None:
        train_docs = train_docs[: args.max_docs]

    lines = build_mlm_lines(train_docs)
    n_tokens = sum(len(d["tokens"]) for d in train_docs)
    print(f"[tapt] task={args.task}  docs={len(lines)}  raw-tokens~{n_tokens}  "
          f"(TRAIN split only; labels stripped)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # match the fine-tuner: [PAR] is a real special token there, so it must be
    # one here too (otherwise the encoder never sees the paragraph symbol).
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
    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ckpts"),
        overwrite_output_dir=False,          # runs/ is append-only
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="no",                  # we save the final encoder once, below
        report_to="none",
        fp16=use_cuda,
        no_cuda=not use_cuda,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    out = trainer.train()
    # report MLM loss trajectory so the smoke can PROVE the loss decreases
    hist = [h for h in trainer.state.log_history if "loss" in h]
    if hist:
        first, last = hist[0]["loss"], hist[-1]["loss"]
        print(f"[tapt] MLM train loss: first-logged={first:.4f}  "
              f"final-logged={last:.4f}  delta={last - first:+.4f}")
    print(f"[tapt] final train loss (mean over run) = "
          f"{out.training_loss:.4f}")

    # ---- SAVE the adapted encoder (loadable by train_encoder.py --model-name) ----
    os.makedirs(args.out_dir, exist_ok=True)
    trainer.save_model(args.out_dir)        # config + weights
    tokenizer.save_pretrained(args.out_dir)  # tokenizer + the [PAR] special token
    print(f"[tapt] saved adapted encoder -> {args.out_dir}")
    print("[tapt] next: python src/train_encoder.py --task", args.task,
          "--model-name", args.out_dir, "--out-dir runs/<finetune-out>")


if __name__ == "__main__":
    main()
