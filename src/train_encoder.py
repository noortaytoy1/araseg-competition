"""Fine-tune an Arabic encoder for AraSeg 2026 sentence-boundary prediction.

Approach: binary token classification. Each whitespace token gets its label on
its LAST subword ("a boundary follows this token"); other subwords are -100.
Paragraph "\n" tokens are mapped to a [PAR] special token and excluded from
the loss (gold always labels them 0; predict.py forces them to 0).
Long documents are split into overlapping word windows.

Defaults are tuned for a single 16 GB GPU (Colab T4 works; ~minutes per epoch
on this small corpus). Examples:

  python train_encoder.py --task NoPnx-NP --out-dir runs/nopnx-np
  python train_encoder.py --task PA --model-name aubmindlab/bert-base-arabertv02 \
      --epochs 10 --out-dir runs/pa-arabert
"""
from __future__ import annotations

import argparse
import os
import random
from typing import List

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task


def make_windows(docs: List[dict], window: int, stride: int) -> List[dict]:
    """Cut each doc into overlapping word windows of `window` words."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labels = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        n = len(words)
        start = 0
        while start < n:
            end = min(start + window, n)
            out.append({"words": words[start:end], "word_labels": labels[start:end]})
            if end == n:
                break
            start += stride
    return out


def encode_batch(examples, tokenizer, max_length: int):
    enc = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_labels = []
    for i, word_labels in enumerate(examples["word_labels"]):
        word_ids = enc.word_ids(batch_index=i)
        labels = [-100] * len(word_ids)
        # label the LAST subword of each word
        for pos in range(len(word_ids)):
            wid = word_ids[pos]
            if wid is None:
                continue
            nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
            if nxt != wid:
                labels[pos] = word_labels[wid]
        all_labels.append(labels)
    enc["labels"] = all_labels
    return enc


class WeightedTrainer(Trainer):
    """Class-weighted CE with optional boundary label smoothing and FGM.

    boundary_smooth: tokens adjacent to a gold boundary get soft target
    `smooth_eps` for the boundary class (boundary-smoothing trick, 1-D).
    fgm_eps: one-step FGM adversarial perturbation on the word embeddings.
    """

    def __init__(self, *args, pos_weight: float = 1.0, smooth_eps: float = 0.0,
                 fgm_eps: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight]), ignore_index=-100
        )
        self._pos_weight = pos_weight
        self._smooth_eps = smooth_eps
        self._fgm_eps = fgm_eps

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._smooth_eps > 0:
            mask = labels != -100
            hard = labels.clamp(min=0).float()
            # neighbors of a boundary (within the window) get a soft target
            soft = hard.clone()
            nb = torch.zeros_like(hard)
            nb[:, :-1] += hard[:, 1:]
            nb[:, 1:] += hard[:, :-1]
            soft = torch.where((hard == 0) & (nb > 0),
                               torch.full_like(soft, self._smooth_eps), soft)
            logp = torch.log_softmax(logits, dim=-1)
            w = torch.where(hard == 1, self._pos_weight,
                            torch.ones_like(hard))
            tok_loss = -(soft * logp[..., 1] + (1 - soft) * logp[..., 0]) * w
            loss = tok_loss[mask].mean()
        else:
            loss = self._loss.to(logits.device)(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._fgm_eps <= 0:
            return loss
        emb = model.get_input_embeddings().weight
        if emb.grad is None:
            return loss
        # FGM: perturb embeddings along grad sign, re-backward, restore
        norm = emb.grad.norm()
        if torch.isfinite(norm) and norm > 0:
            delta = self._fgm_eps * emb.grad / norm
            emb.data.add_(delta)
            adv_loss = self.compute_loss(model, {k: v for k, v in inputs.items()})
            if self.use_apex:
                adv_loss.backward()
            else:
                self.accelerator.backward(adv_loss)
            emb.data.sub_(delta)
        return loss


def token_f1_metrics(eval_pred):
    """Token-level boundary P/R/F1 on windows — a fast proxy for model selection.
    Run predict.py + eval_local.py for the true document-level metric."""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    mask = labels != -100
    p, g = preds[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--dev-jsonl", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=int, default=180, help="words per training window")
    ap.add_argument("--stride", type=int, default=90)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="loss weight for the boundary class (default: auto from train stats, capped at 8)")
    ap.add_argument("--smooth-eps", type=float, default=0.0,
                    help="soft boundary target for tokens adjacent to a gold boundary")
    ap.add_argument("--fgm-eps", type=float, default=0.0,
                    help="FGM adversarial perturbation size on word embeddings (0=off)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    train_docs = load_task(args.task, "train", jsonl_path=args.train_jsonl)
    dev_docs = load_task(args.task, "dev", jsonl_path=args.dev_jsonl)

    pos = sum(sum(d["labels"]) for d in train_docs)
    tot = sum(len(d["labels"]) for d in train_docs)
    pos_weight = args.pos_weight if args.pos_weight is not None else min((tot - pos) / max(pos, 1), 8.0)
    print(f"train boundary rate={pos / tot:.3f}  pos_weight={pos_weight:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    from datasets import Dataset

    train_ds = Dataset.from_list(make_windows(train_docs, args.window, args.stride))
    dev_ds = Dataset.from_list(make_windows(dev_docs, args.window, args.window))  # no overlap for eval
    fn = lambda ex: encode_batch(ex, tokenizer, args.max_length)
    train_ds = train_ds.map(fn, batched=True, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(fn, batched=True, remove_columns=dev_ds.column_names)

    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ckpts"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=token_f1_metrics,
        pos_weight=pos_weight,
        smooth_eps=args.smooth_eps,
        fgm_eps=args.fgm_eps,
    )
    trainer.train()
    print("best window-level dev metrics:", trainer.evaluate())

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"saved model -> {args.out_dir}")
    print("next: python predict.py --model", args.out_dir, "--task", args.task,
          "--split dev --out subs/" + args.task + "_dev.csv")


if __name__ == "__main__":
    main()
