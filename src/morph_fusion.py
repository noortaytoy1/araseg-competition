"""Morphology-feature fusion tagger (self-contained; does NOT touch the shared
train_encoder/predict pipeline).

Each word carries (POS, prc0, prc1, prc2) from camel_tools (see extract_morph.py).
These are embedded and concatenated to the word's final BERT hidden state before
the boundary head, so the model USES the morphological signal at inference — the
signal every subword-BERT is blind to (the «و»-clause-vs-list disambiguation).

Usage:
  python morph_fusion.py train --task NoPnx-NP --out runs/morph_nopnxnp
  python morph_fusion.py predict --task NoPnx-NP --split dev --model runs/morph_nopnxnp --out subs/x.csv
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
NFEAT = 4
EMB = 16


def load_morph(task, split):
    return pickle.load(open(os.path.join(DATA, f"morph_{task}_{split}.pkl"), "rb"))


def build_vocabs(morph):
    v = [{"<pad>": 0} for _ in range(NFEAT)]
    for feats in morph.values():
        for f in feats:
            for i in range(NFEAT):
                v[i].setdefault(str(f[i]), len(v[i]))
    return v


class MorphFused(nn.Module):
    def __init__(self, model_name, vocab_sizes, pos_weight=1.0, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.membs = nn.ModuleList([nn.Embedding(v, EMB, padding_idx=0)
                                    for v in vocab_sizes])
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.cls = nn.Linear(h + NFEAT * EMB, 2)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config  # for Trainer save

    def forward(self, input_ids=None, attention_mask=None, morph_ids=None,
                labels=None, **kw):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        me = torch.cat([self.membs[i](morph_ids[..., i]) for i in range(NFEAT)], dim=-1)
        logits = self.cls(self.drop(torch.cat([h, me], dim=-1)))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(logits.device), ignore_index=-100)
        return {"loss": loss, "logits": logits}


def make_examples(docs, morph, window, stride):
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        mf = morph[d["doc_id"]]
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            out.append({"words": words[s:e], "word_labels": labs[s:e],
                        "word_morph": [list(mf[j]) for j in range(s, e)]})
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, vocabs, max_len, zero_morph=False):
    enc = tok(examples["words"], is_split_into_words=True, truncation=True,
              max_length=max_len)
    all_lab, all_m = [], []
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        lab = [-100] * len(wid)
        m = [[0] * NFEAT for _ in wid]
        wm = examples["word_morph"][i]
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:  # last subword
                lab[pos] = wl[w]
                if not zero_morph:   # control: keep morph ids at 0 (zero embedding)
                    m[pos] = [vocabs[k].get(str(wm[w][k]), 0) for k in range(NFEAT)]
        all_lab.append(lab); all_m.append(m)
    enc["labels"] = all_lab
    enc["morph_ids"] = all_m
    return enc


class Collator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        morph = [f.pop("morph_ids") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        batch["morph_ids"] = torch.tensor(
            [mm + [[0] * NFEAT] * (m - len(mm)) for mm in morph], dtype=torch.long)
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
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    m_tr = load_morph(a.task, "train"); m_dv = load_morph(a.task, "dev")
    vocabs = build_vocabs(m_tr)
    pickle.dump(vocabs, open(os.path.join(HERE, os.pardir, "data",
                f"morphvocab_{a.task}.pkl"), "wb"))
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  vocab sizes={[len(v) for v in vocabs]}")
    model = MorphFused(a.model_name, [len(v) for v in vocabs], pw, n_extra_tok=1)
    trds = Dataset.from_list(make_examples(tr, m_tr, a.window, a.stride))
    dvds = Dataset.from_list(make_examples(dv, m_dv, a.window, a.window))
    fn = lambda ex: encode(ex, tok, vocabs, a.max_length, a.zero_morph)
    trds = trds.map(fn, batched=True, remove_columns=trds.column_names)
    dvds = dvds.map(fn, batched=True, remove_columns=dvds.column_names)
    targs = TrainingArguments(
        output_dir=os.path.join(a.out, "ckpts"), num_train_epochs=a.epochs,
        learning_rate=a.lr, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", warmup_ratio=0.1, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=20, save_total_limit=1,
        report_to="none", seed=a.seed, remove_unused_columns=False)
    tr_ = Trainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                  data_collator=Collator(tok), compute_metrics=metrics)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name, "vocab_sizes": [len(v) for v in vocabs],
               "zero_morph": a.zero_morph},
              open(os.path.join(a.out, "cfg.json"), "w"))
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    vocabs = pickle.load(open(os.path.join(HERE, os.pardir, "data",
                         f"morphvocab_{a.task}.pkl"), "rb"))
    model = MorphFused(cfg["model_name"], cfg["vocab_sizes"], n_extra_tok=1)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))
    model.to(dev).eval()
    zm = cfg.get("zero_morph", False)
    docs = load_task(a.task, a.split, jsonl_path=a.jsonl)
    morph = load_morph(a.task, a.split)
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        mf = morph[d["doc_id"]]
        n = len(words); probs = {i: [] for i in range(n)}; s = 0
        W, OV = a.window, a.overlap
        while s < n:
            e = min(s + W, n)
            enc = tok(words[s:e], is_split_into_words=True, truncation=True,
                      max_length=a.max_length, return_tensors="pt")
            wid = enc.word_ids(0)
            mids = [[0] * NFEAT for _ in wid]
            for pos in range(len(wid)):
                w = wid[pos]
                if w is None:
                    continue
                nxt = wid[pos + 1] if pos + 1 < len(wid) else None
                if nxt != w and not zm:
                    mids[pos] = [vocabs[k].get(str(mf[s + w][k]), 0) for k in range(NFEAT)]
            mt = torch.tensor([mids], dtype=torch.long).to(dev)
            logit = model(input_ids=enc["input_ids"].to(dev),
                          attention_mask=enc["attention_mask"].to(dev),
                          morph_ids=mt)["logits"][0]
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
            s = max(s + last + 1 - OV, s + 1)
        lab = [1 if (np.mean(probs[i]) if probs[i] else 0) >= a.threshold else 0
               for i in range(n)]
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        preds.append(lab)
    write_submission(docs, preds, a.out)
    print(f"wrote {len(docs)} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--zero-morph", action="store_true", help="control: zero the morph signal (same architecture)")
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--task", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)
    args = ap.parse_args()
    args.func(args)
