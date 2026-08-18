"""Anti-overconfidence boundary tagger for AraSeg (closed track).

Self-contained — does NOT touch the shared train_encoder/predict/eval pipeline.
The data / window / label / [PAR] handling MIRRORS src/consistency_train.py
exactly (AraBERTv02 token classifier, window 180 / stride 90, epochs 8,
batch 16, lr 5e-5, weighted CE with pos_weight capped ~8, labels on the LAST
subword of each word, MODEL_PAR_TOKEN for the "\n" paragraph token).

Base model: AutoModelForTokenClassification(aubmindlab/bert-base-arabertv02,
num_labels=2). Two FLAG-GATED anti-overconfidence penalties are added on top of
the plain weighted-CE baseline. Both are OFF by default, and with BOTH weights
0.0 the trainer reproduces the plain weighted-CE baseline BIT-FOR-BIT (a single
forward pass, identical loss — verified with torch.equal in the GPU smoke).

MOTIVATION (Noor's caution — the overconfidence is ONE-SIDED):
  A confident CUT is wrong ~1-in-6 (+15.9pt over-firing at prepositions /
  connectives); a confident NO-CUT is honest. So we must punish confidence ONLY
  where a cut would be WRONG, and NEVER reduce confidence on correct cuts. That
  is NOT the same as raising the negative-class weight (which suppresses ALL
  firing and hurts recall) — the fire-side penalty targets only CONFIDENT
  false-cut pressure.

PENALTY 1 — ASYMMETRIC FIRE-SIDE CONFIDENCE PENALTY  (Noor's explicit ask)
  --fire-penalty-weight B (default 0.0), --fire-penalty-gamma G (default 2.0):
    On positions with GOLD label 0 (no boundary) only, add
        B * mean( p_boundary ** G )
    where p_boundary = softmax(logits)[..., 1]. A focal-style extra fine that
    grows with how confidently the model wants to CUT a non-boundary token.
    On GOLD == 1 positions: NO penalty (plain CE, confidence on correct cuts is
    untouched). label == -100 (subwords / pad / [PAR]) is ignored.
    loss = weighted_CE + fire_penalty.
  This is asymmetric by construction: the mean is taken over gold==0 positions
  ONLY, so it can never push down p_boundary on a true boundary.

PENALTY 2 — SYMMETRIC CONFIDENCE / ENTROPY PENALTY  (comparison arm)
  --conf-penalty-weight W (default 0.0):
    Over all VALID (label != -100) positions, add
        loss += -W * H(p)
    where H(p) = -sum_c p_c log p_c is the per-token entropy of the 2-class
    softmax. This is the standard confidence-penalty / entropy-regularization
    term (Pereyra et al. 2017): it discourages LOW entropy (over-confidence) on
    BOTH classes symmetrically — a comparison arm against the one-sided fire
    penalty. Maximizing entropy == subtracting entropy from the loss (hence the
    minus sign; a POSITIVE W is a penalty).

The two flags are independent; they may be combined, but the battery runs them
one at a time. Only one is expected to be > 0 in any given run.

Usage:
  # baseline (bit-for-bit identical to plain weighted CE):
  python fire_penalty_train.py train --task NoPnx-NP --out runs/fire_base \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # asymmetric fire-side penalty (Noor's ask):
  python fire_penalty_train.py train --task NoPnx-NP --fire-penalty-weight 1.0 \
      --fire-penalty-gamma 2.0 --out runs/fire_asym \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # symmetric entropy penalty (comparison arm):
  python fire_penalty_train.py train --task NoPnx-NP --conf-penalty-weight 0.1 \
      --out runs/fire_entropy \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  python fire_penalty_train.py predict --task NoPnx-NP --split dev \
      --model runs/fire_asym --jsonl data/NoPnx-NP_dev.jsonl \
      --out subs/fire_asym_dev.csv
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py exactly)              #
# --------------------------------------------------------------------------- #
def make_examples(docs, window, stride):
    """Slide a (window, stride) window over each doc; [PAR] positions -> -100."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            out.append({"words": words[s:e], "word_labels": labs[s:e]})
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, max_len):
    """Sub-word encode; put each word's label on its LAST subword only."""
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
            if nxt != w:  # last subword of word w
                lab[pos] = wl[w]
        all_lab.append(lab)
    enc["labels"] = all_lab
    return enc


# --------------------------------------------------------------------------- #
# model wrapper: plain token-classification + two flag-gated penalties         #
# --------------------------------------------------------------------------- #
class FirePenaltyModel(nn.Module):
    """AraBERTv02 token classifier plus two flag-gated anti-overconfidence
    penalties. Neither penalty adds any parameters, so `self.enc` is exactly the
    inference model (see _save_hf_checkpoint)."""

    def __init__(self, model_name, pos_weight=1.0, fire_weight=0.0,
                 fire_gamma=2.0, conf_weight=0.0, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.fire_weight = float(fire_weight)
        self.fire_gamma = float(fire_gamma)
        self.conf_weight = float(conf_weight)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def _logits(self, input_ids, attention_mask, inputs_embeds=None):
        if inputs_embeds is not None:
            return self.enc(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask).logits
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).logits

    def _ce(self, logits, labels):
        return nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        logits = self._logits(input_ids, attention_mask)
        loss = None
        if labels is not None:
            ce = self._ce(logits, labels)
            loss = ce

            # -- PENALTY 1: ASYMMETRIC FIRE-SIDE CONFIDENCE PENALTY ----------- #
            # Gated OFF at weight 0: the block is skipped, loss is exactly plain
            # weighted CE (bit-for-bit). When on: a focal-style extra fine on
            # GOLD==0 positions only, growing with p_boundary ** gamma. It NEVER
            # touches gold==1 positions, so confidence on correct cuts is intact.
            if self.fire_weight > 0:
                flat_labels = labels.view(-1)
                p_boundary = torch.softmax(logits.view(-1, 2).float(), -1)[..., 1]
                neg = (flat_labels == 0)  # gold no-boundary, excludes -100 and gold==1
                if neg.any():
                    fire = (p_boundary[neg] ** self.fire_gamma).mean()
                else:
                    fire = (p_boundary * 0.0).sum()  # keep graph, contribute 0
                loss = loss + self.fire_weight * fire

            # -- PENALTY 2: SYMMETRIC CONFIDENCE / ENTROPY PENALTY ------------ #
            # Gated OFF at weight 0: skipped, loss stays plain weighted CE.
            # When on: subtract mean per-token entropy over VALID positions
            # (Pereyra et al. 2017 confidence penalty); a POSITIVE weight
            # discourages over-confidence on BOTH classes symmetrically.
            if self.conf_weight > 0:
                flat_labels = labels.view(-1)
                valid = (flat_labels != -100)
                logp = torch.log_softmax(logits.view(-1, 2).float(), -1)
                p = logp.exp()
                ent = -(p * logp).sum(-1)  # per-token entropy H(p)
                if valid.any():
                    ent_mean = ent[valid].mean()
                else:
                    ent_mean = (ent * 0.0).sum()
                loss = loss - self.conf_weight * ent_mean

        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# metrics / train / predict                                                   #
# --------------------------------------------------------------------------- #
def metrics(ep):
    logits, labels = ep
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def _save_hf_checkpoint(model, tok, out_dir):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / eval_local.py / fair_threshold.py / diag_preposition.py
    load with AutoModelForTokenClassification.from_pretrained(out_dir) +
    AutoTokenizer, which the custom .pt cannot satisfy. Here `model.enc` IS the
    trained AutoModelForTokenClassification (encoder + boundary classifier); the
    fire / entropy penalties add NO parameters, so `model.enc` is exactly the
    inference model. We write it verbatim. This does not touch training, the loss,
    the flags, or the existing .pt/cfg.json outputs — it only adds config.json +
    weights + tokenizer files so the downstream tools can read the run directory.
    """
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  fire_penalty_weight={a.fire_penalty_weight}  "
          f"fire_penalty_gamma={a.fire_penalty_gamma}  "
          f"conf_penalty_weight={a.conf_penalty_weight}")
    model = FirePenaltyModel(a.model_name, pw, a.fire_penalty_weight,
                             a.fire_penalty_gamma, a.conf_penalty_weight,
                             n_extra_tok=1)
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride))
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
    tr_ = Trainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                  data_collator=DataCollatorForTokenClassification(tok),
                  compute_metrics=metrics)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name}, open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: also emit an HF checkpoint so predict.py / eval_local.py /
    # diag_preposition.py can load this run dir. Nothing above is changed.
    _save_hf_checkpoint(model, tok, a.out)
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = FirePenaltyModel(cfg["model_name"], n_extra_tok=1)
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
    t.add_argument("--fire-penalty-weight", type=float, default=0.0,
                   help="PENALTY 1 asymmetric FIRE-SIDE confidence penalty weight "
                        "B (0 = off). On gold==0 positions only, adds "
                        "B*mean(p_boundary**gamma); never touches gold==1.")
    t.add_argument("--fire-penalty-gamma", type=float, default=2.0,
                   help="PENALTY 1 focal exponent gamma on p_boundary (default 2.0)")
    t.add_argument("--conf-penalty-weight", type=float, default=0.0,
                   help="PENALTY 2 SYMMETRIC entropy confidence penalty weight W "
                        "(0 = off). Comparison arm: subtracts W*mean(H(p)) over "
                        "valid positions (Pereyra et al. 2017).")
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
