"""Coherence-auxiliary tagger (Yu 2023 style; the 'understand-first' idea).
Self-contained — does NOT touch the shared train_encoder/predict pipeline.

Multi-task: (1) per-token boundary head (the real task); (2) a pooled coherence
head that predicts whether a window's sentences are in their ORIGINAL order or
were SHUFFLED. Shuffling reorders whole sentences, so every boundary label stays
valid (a sentence still ends on a boundary) — the boundary task is unaffected;
only the coherence signal changes. This forces the encoder to represent sentence
flow, the semantic prior, which should sharpen boundary decisions. The coherence
head is discarded at inference.

Usage:
  python coherence_train.py train --task NoPnx-NP --coh 0.5 --out runs/coh
  python coherence_train.py predict --task NoPnx-NP --split dev --model runs/coh --out subs/x.csv
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))


def sentences_of(words, labs):
    """Split a (words, labels) window into sentences (each ends on label 1)."""
    out, start = [], 0
    for i, l in enumerate(labs):
        if l == 1:
            out.append((words[start:i + 1], labs[start:i + 1])); start = i + 1
    if start < len(words):
        out.append((words[start:], labs[start:]))
    return out


class CohModel(nn.Module):
    def __init__(self, model_name, pos_weight=1.0, coh=0.5, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.bound = nn.Linear(h, 2)
        self.cohh = nn.Linear(h, 2)
        self.coh = coh
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                coh_label=None, **kw):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.bound(self.drop(hs))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(hs.device), ignore_index=-100)
            if self.coh > 0 and coh_label is not None:
                m = attention_mask.unsqueeze(-1).float()
                pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1)   # mean-pool
                closs = nn.functional.cross_entropy(
                    self.cohh(pooled).float(), coh_label.view(-1))
                loss = loss + self.coh * closs
        return {"loss": loss, "logits": logits}


def make_examples(docs, window, stride, shuffle_p=0.0):
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            w_, l_ = words[s:e], labs[s:e]
            coh = 0
            if shuffle_p > 0 and random.random() < shuffle_p:
                sents = sentences_of(w_, l_)
                if len(sents) >= 2:
                    random.shuffle(sents)
                    w_ = [t for st, _ in sents for t in st]
                    l_ = [x for _, sl in sents for x in sl]
                    coh = 1
            out.append({"words": w_, "word_labels": l_, "coh_label": coh})
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, max_len):
    enc = tok(examples["words"], is_split_into_words=True, truncation=True, max_length=max_len)
    all_lab = []
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        lab = [-100] * len(wid)
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:
                lab[pos] = wl[w]
        all_lab.append(lab)
    enc["labels"] = all_lab
    enc["coh_label"] = examples["coh_label"]
    return enc


class Collator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        coh = [f.pop("coh_label") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        batch["coh_label"] = torch.tensor(coh, dtype=torch.long)
        return batch


def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  coh={a.coh}")
    model = CohModel(a.model_name, pw, a.coh, n_extra_tok=1)
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride, a.coh_shuffle))
    dvds = Dataset.from_list(make_examples(dv, a.window, a.window, 0.0))  # clean eval
    fn = lambda ex: encode(ex, tok, a.max_length)
    trds = trds.map(fn, batched=True, remove_columns=trds.column_names)
    dvds = dvds.map(fn, batched=True, remove_columns=dvds.column_names)
    targs = TrainingArguments(
        output_dir=os.path.join(a.out, "ckpts"), num_train_epochs=a.epochs,
        learning_rate=a.lr, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1",
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=20, save_total_limit=1, report_to="none", seed=a.seed,
        remove_unused_columns=False, label_names=["labels"])
    tr_ = Trainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                  data_collator=Collator(tok), compute_metrics=metrics)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name}, open(os.path.join(a.out, "cfg.json"), "w"))
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = CohModel(cfg["model_name"], n_extra_tok=1)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))
    model.to(dev).eval()
    docs = load_task(a.task, a.split, jsonl_path=a.jsonl)
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        n = len(words); probs = {i: [] for i in range(n)}; s = 0
        while s < n:
            e = min(s + a.window, n)
            enc = tok(words[s:e], is_split_into_words=True, truncation=True,
                      max_length=a.max_length, return_tensors="pt")
            wid = enc.word_ids(0)
            logit = model(input_ids=enc["input_ids"].to(dev),
                          attention_mask=enc["attention_mask"].to(dev))["logits"][0]
            pb = torch.softmax(logit, -1)[:, 1].cpu().numpy()
            last = -1
            for pos, w in enumerate(wid):
                if w is None:
                    continue
                nxt = wid[pos + 1] if pos + 1 < len(wid) else None
                if nxt != w:
                    probs[s + w].append(float(pb[pos]))
                last = max(last, w)
            if s + last >= n - 1:
                break
            s = max(s + last + 1 - a.overlap, s + 1)
        lab = [1 if (np.mean(probs[i]) if probs[i] else 0) >= a.threshold else 0 for i in range(n)]
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        preds.append(lab)
    write_submission(docs, preds, a.out)
    print(f"wrote {len(docs)} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--coh", type=float, default=0.5, help="coherence aux loss weight")
    t.add_argument("--coh-shuffle", type=float, default=0.5, help="fraction of windows shuffled")
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--task", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(); a.func(a)
