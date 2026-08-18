"""BoundRL Stage-1 SFT (Sec 5.2 / A.2 of arXiv:2510.20151v2) for AraSeg.

LoRA SFT with peft (r=16, alpha=32, attention + MLP projections), 1 epoch,
completion-only loss on the [SEP]-joined starting-word targets built by
boundrl_data.build_examples.

Paper-fixed hyperparameters (A.2): max_grad_norm 0.1, weight_decay 0.01,
linear LR schedule with warmup ratio 0.03.  Learning rate is a flag with the
approved pilot range [2e-6, 1e-5] (paper: 2e-6 for Qwen3); values outside the
range abort unless --allow-any-lr.

Model: default Qwen/Qwen3-1.7B; if it cannot be loaded, the fallback
Qwen/Qwen2.5-1.5B-Instruct is tried and the model actually used is recorded
in <out>/boundrl_sft_config.json.  Note (2026-07-10): neither Qwen model is
in the local HF cache; the guarded pilot script downloads at run time.

Windowing (documented choice, see boundrl_data): documents are split into
sentence-aligned chunks whose document text is at most --window-tokens
(default 2048) tokens of the *actual model tokenizer*; --max-len (default
4096) then bounds prompt+completion for training.  Arithmetic: a 2048-token
chunk of Arabic (~890 words at ~2.3 tok/word) contains ~76 sentences of
~11.7 words; with mean start length 6 words the completion adds roughly
6*76*2.3 ~ 1050 tokens plus [SEP]s, so 4096 covers meta prompt (~90) + 2048
+ completion with margin.

LoRA target modules: attention + MLP projections of the model family —
Qwen/Llama: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj;
GPT-2 (CPU smoke only): c_attn,c_proj,c_fc.

Usage (CPU smoke):
  CUDA_VISIBLE_DEVICES=-1 python src/boundrl_sft.py --model sshleifer/tiny-gpt2 \
    --max-docs 5 --max-steps 2 --window-tokens 256 --max-len 512 \
    --out runs/boundrl_pilot_2026-07-10/smoke_sft
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LORA_TARGETS = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
}

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LR_RANGE = (2e-6, 1e-5)


def load_model_and_tokenizer(model_name: str, fallback: str | None, dtype: str):
    """Try the primary model, then the fallback; return (model, tok, used_name)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    for name in [model_name] + ([fallback] if fallback else []):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return model, tok, name
        except Exception as e:  # noqa: BLE001 — deliberately broad: any load failure -> fallback
            print(f"[boundrl_sft] could not load {name!r}: {e}", file=sys.stderr)
    raise SystemExit("no model could be loaded (primary and fallback both failed)")


def lora_config_for(model, r: int, alpha: int):
    from peft import LoraConfig, TaskType

    mt = getattr(model.config, "model_type", "")
    targets = LORA_TARGETS.get(mt)
    if targets is None:
        raise SystemExit(f"no LoRA target-module map for model_type {mt!r}; add it to LORA_TARGETS")
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=0.0,
        target_modules=targets,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fallback-model", default=FALLBACK_MODEL)
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-6, help=f"approved range {LR_RANGE}")
    ap.add_argument("--allow-any-lr", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="override for smoke tests")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16, help="paper batch size 16")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--window-tokens", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None, help="smoke: only first N docs")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = ap.parse_args()

    if not args.allow_any_lr and not (LR_RANGE[0] <= args.lr <= LR_RANGE[1]):
        raise SystemExit(f"--lr {args.lr} outside approved range {LR_RANGE}; pass --allow-any-lr to override")

    import datasets as hf_datasets
    import peft
    import transformers
    import trl
    from trl import SFTConfig, SFTTrainer

    from boundrl_data import build_examples
    from data import load_jsonl

    model, tok, used_model = load_model_and_tokenizer(args.model, args.fallback_model, args.dtype)
    count_tokens = lambda s: len(tok(s, add_special_tokens=False).input_ids)  # noqa: E731

    docs = load_jsonl(args.train_jsonl)
    if args.max_docs:
        docs = docs[: args.max_docs]
    examples = build_examples(docs, seed=args.seed, count_tokens=count_tokens, budget=args.window_tokens)
    ds = hf_datasets.Dataset.from_list(
        [{"prompt": ex["prompt"], "completion": ex["completion"]} for ex in examples]
    )
    print(f"[boundrl_sft] model={used_model} docs={len(docs)} chunks/examples={len(examples)}")

    os.makedirs(args.out, exist_ok=True)
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.1,
        max_length=args.max_len,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        bf16=args.dtype == "bfloat16",
        use_cpu=os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1",
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora_config_for(model, args.lora_r, args.lora_alpha),
    )
    result = trainer.train()
    trainer.save_model(args.out)  # saves the LoRA adapter
    tok.save_pretrained(args.out)

    record = {
        "model_used": used_model,
        "model_requested": args.model,
        "fallback_model": args.fallback_model,
        "train_jsonl": args.train_jsonl,
        "n_docs": len(docs),
        "n_examples": len(examples),
        "lr": args.lr,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "effective_batch": args.per_device_batch * args.grad_accum,
        "max_len": args.max_len,
        "window_tokens": args.window_tokens,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "train_loss": float(result.training_loss),
        "versions": {
            "trl": trl.__version__,
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "datasets": hf_datasets.__version__,
        },
    }
    with open(os.path.join(args.out, "boundrl_sft_config.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[boundrl_sft] done; adapter + boundrl_sft_config.json -> {args.out}")


if __name__ == "__main__":
    main()
