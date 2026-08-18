"""Augmentation-consistency contrastive tagger (SimCSE-style; the 'coverage'
idea). Self-contained — does NOT touch the shared train_encoder/predict pipeline.

Transferred from princeton-nlp/SimCSE (Gao, Yao & Chen, EMNLP 2021),
vendor/SimCSE/simcse/models.py::cl_forward. SimCSE's central trick is that a
SINGLE forward pass, run TWICE with the encoder's own independent dropout,
yields two slightly different "views" of the same input; an InfoNCE loss then
pulls the two views of the same item together and pushes different items apart.
No external augmentation, no label leakage — the positive pair is literally the
same tokens seen under two dropout masks.

We move SimCSE from the SENTENCE (pooled) level to the TOKEN-POSITION level,
which is what a boundary tagger needs:

  * View A = standard forward (independent dropout), same as today's baseline.
  * View B = a SECOND forward of the SAME window under an independent dropout
    mask, optionally with LABEL-PRESERVING perturbations that never move a
    boundary and never change the token grid (so position j in A aligns exactly
    to position j in B):
      - emb-noise: small Gaussian jitter on the input word embeddings;
      - mask-drop: a few NON-boundary, non-boundary-adjacent tokens have their
        attention_mask zeroed (the token is still there, its slot and label are
        unchanged — the model just cannot attend to it in view B).
    Both keep every boundary label EXACT (SimCSE-safe augmentation only).
  * InfoNCE consistency loss: for each boundary-eligible token position, the
    view-A and view-B representations of the SAME position are a positive pair;
    all other positions in the (flattened, in-batch) valid-position set are
    negatives. This teaches the encoder that a boundary decision at a site is
    INVARIANT to augmentation, so it generalizes to unseen contexts — exactly
    the coverage bottleneck (88-89% NoPnx errors on unseen bigrams).

The boundary cross-entropy stays the MAIN objective; the InfoNCE term is a
flag-gated ADD-ON. With --consistency 0 (default) the second view is never
built and the loss is BIT-FOR-BIT the plain class-weighted CE — same as today's
baseline. The projection head and view B are discarded at inference.

TRAIN-ONLY: fits/reads only the train split. dev is used solely by the existing
eval, never here.

Usage:
  python contrastive_train.py train --task NoPnx-NP --consistency 0.1 --out runs/cl
  python contrastive_train.py predict --task NoPnx-NP --split dev --model runs/cl --out subs/x.csv
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


class CLModel(nn.Module):
    """Boundary tagger + SimCSE-style projection head for token-position InfoNCE.

    The boundary head is the real task. The projection head (proj) maps hidden
    states to the contrastive space; it is used ONLY by the consistency loss and
    is discarded at inference. With consistency weight 0, proj and the second
    view are never touched, and forward() returns exactly the baseline logits.
    """

    def __init__(self, model_name, pos_weight=1.0, consistency=0.0,
                 proj_dim=128, temp=0.05, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.bound = nn.Linear(h, 2)
        # SimCSE projection head: Linear -> Tanh (mirrors vendor MLPLayer).
        self.proj = nn.Sequential(nn.Linear(h, proj_dim), nn.Tanh())
        self.consistency = consistency
        self.temp = temp
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def _encode(self, input_ids, attention_mask):
        """One forward pass. Independent dropout is applied by the encoder each
        call, so calling this twice on the same input yields two views."""
        return self.enc(input_ids=input_ids,
                        attention_mask=attention_mask).last_hidden_state

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                attention_mask_b=None, emb_noise=0.0, **kw):
        # ---- View A: standard forward (this is the baseline path) ----
        hs = self._encode(input_ids, attention_mask)
        logits = self.bound(self.drop(hs))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(hs.device), ignore_index=-100)
            # ---- View B + InfoNCE: only when consistency > 0 ----
            if self.consistency > 0:
                loss = loss + self.consistency * self._infonce(
                    input_ids, attention_mask, attention_mask_b, labels,
                    hs, emb_noise)
        return {"loss": loss, "logits": logits}

    def _infonce(self, input_ids, attention_mask, attention_mask_b, labels,
                 hs_a, emb_noise):
        """Token-position InfoNCE between view A (hs_a) and a fresh view B.

        Transplant of vendor/SimCSE cl_forward (lines 154-212): build a
        similarity matrix between the two views over the boundary-eligible token
        positions, with arange() targets so each position's positive is its own
        counterpart in the other view and every other position is a negative.
        """
        am_b = attention_mask_b if attention_mask_b is not None else attention_mask
        if emb_noise and emb_noise > 0:
            # LABEL-PRESERVING: jitter the input embeddings for view B only.
            emb = self.enc.get_input_embeddings()(input_ids)
            emb = emb + emb_noise * torch.randn_like(emb)
            hs_b = self.enc(inputs_embeds=emb,
                            attention_mask=am_b).last_hidden_state
        else:
            hs_b = self._encode(input_ids, am_b)   # 2nd dropout mask = view B

        # only boundary-eligible positions carry a real label (last subword);
        # padding / non-final subwords / [PAR] are ignored (-100).
        valid = (labels.view(-1) != -100)
        if valid.sum() < 2:            # need >=2 anchors for a contrastive batch
            return hs_b.sum() * 0.0
        za = self.proj(hs_a).view(-1, self.proj[0].out_features)[valid]
        zb = self.proj(hs_b).view(-1, self.proj[0].out_features)[valid]
        za = nn.functional.normalize(za, dim=-1)
        zb = nn.functional.normalize(zb, dim=-1)
        # cosine sim matrix / temperature; diagonal = same position across views
        sim = (za @ zb.t()) / self.temp
        tgt = torch.arange(sim.size(0), device=sim.device)
        return nn.functional.cross_entropy(sim, tgt)


def make_examples(docs, window, stride, augment=False, mask_p=0.0, seed=0):
    """Cut docs into overlapping windows. If augment, precompute a view-B
    attention mask that drops a few SAFE tokens (non-boundary and not adjacent
    to a boundary), so the InfoNCE view B is perturbed but every label is exact.
    The mask is at WORD granularity here; encode() expands it to subwords."""
    rng = random.Random(seed)
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            w_, l_ = words[s:e], labs[s:e]
            keep_b = [1] * len(w_)
            if augment and mask_p > 0:
                for i in range(len(w_)):
                    if l_[i] == 1:                       # never drop a boundary
                        continue
                    prev_b = i > 0 and l_[i - 1] == 1     # nor its neighbours
                    next_b = i + 1 < len(l_) and l_[i + 1] == 1
                    if prev_b or next_b:
                        continue
                    if rng.random() < mask_p:
                        keep_b[i] = 0
            out.append({"words": w_, "word_labels": l_, "keep_b": keep_b})
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, max_len):
    enc = tok(examples["words"], is_split_into_words=True, truncation=True, max_length=max_len)
    all_lab, all_keep = [], []
    has_keep = "keep_b" in examples
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        lab = [-100] * len(wid)
        keep = [1] * len(wid)
        kb = examples["keep_b"][i] if has_keep else None
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            if kb is not None:
                keep[pos] = kb[w]           # view-B mask follows the word
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:
                lab[pos] = wl[w]            # label the LAST subword of each word
        all_lab.append(lab)
        all_keep.append(keep)
    enc["labels"] = all_lab
    enc["keep_b"] = all_keep
    return enc


class Collator:
    """Pad labels and the view-B keep mask to the batch length."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        keep = [f.pop("keep_b") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        # view-B attention mask = base attention mask AND the safe-drop keep mask
        keep_t = torch.tensor([k + [0] * (m - len(k)) for k in keep])
        batch["attention_mask_b"] = batch["attention_mask"] * keep_t
        return batch


def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


class CLTrainer(Trainer):
    """Passes the InfoNCE knobs (emb_noise) into the model's forward and lets the
    model return its own loss. With consistency=0 in the model, this is a plain
    class-weighted CE trainer — identical to the baseline path."""

    def __init__(self, *args, emb_noise=0.0, **kw):
        super().__init__(*args, **kw)
        self._emb_noise = emb_noise

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs = dict(inputs)
        inputs["emb_noise"] = self._emb_noise
        out = model(**inputs)
        loss = out["loss"]
        return (loss, out) if return_outputs else loss


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)   # TRAIN-ONLY
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)       # eval selection only
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  consistency={a.consistency}  "
          f"temp={a.temp}  mask_p={a.mask_p}  emb_noise={a.emb_noise}")
    model = CLModel(a.model_name, pw, a.consistency, a.proj_dim, a.temp, n_extra_tok=1)
    augment = a.consistency > 0 and a.mask_p > 0
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride, augment, a.mask_p, a.seed))
    dvds = Dataset.from_list(make_examples(dv, a.window, a.window))  # clean eval
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
    tr_ = CLTrainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                    data_collator=Collator(tok), compute_metrics=metrics,
                    emb_noise=a.emb_noise)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name, "proj_dim": a.proj_dim, "temp": a.temp},
              open(os.path.join(a.out, "cfg.json"), "w"))
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    # torch.cuda.is_available() can be True while device_count()==0 (e.g.
    # CUDA_VISIBLE_DEVICES=""); gate on an actually-usable device so CPU
    # prediction never tries to deserialize onto a phantom cuda:0.
    dev = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = CLModel(cfg["model_name"], proj_dim=cfg.get("proj_dim", 128),
                    temp=cfg.get("temp", 0.05), n_extra_tok=1)
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
    t.add_argument("--consistency", type=float, default=0.0,
                   help="InfoNCE consistency loss weight (0=off => plain CE)")
    t.add_argument("--temp", type=float, default=0.05, help="InfoNCE temperature")
    t.add_argument("--proj-dim", type=int, default=128, help="projection head dim")
    t.add_argument("--mask-p", type=float, default=0.0,
                   help="view-B safe-token drop prob (0=dropout-only SimCSE)")
    t.add_argument("--emb-noise", type=float, default=0.0,
                   help="view-B input-embedding Gaussian noise std (0=off)")
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
