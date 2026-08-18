"""Teacher-consistency taggers against Arabic preposition/connective OVER-firing.
Self-contained — does NOT touch the shared train_encoder/predict pipeline. It
mirrors the data/window/label pipeline of coherence_train.py / contrastive_train.py
(window 180 / stride 90, last-subword labels, [PAR] handling, weighted CE with a
pos_weight capped ~8), and adds TWO flag-gated consistency methods that both aim
at the same failure: the model fires a boundary just because it saw a preposition
(و, ف, ثم, من, في, ...), instead of deciding by meaning.

Both methods hide the prepositions and force the boundary decision to survive
without them:

METHOD 3 — MEAN TEACHER  (--mt-weight, default 0)
  A TEACHER model is kept as an exponential-moving-average (EMA) copy of the
  student's weights (--ema-decay, default 0.999; no gradient ever flows to it).
  Each step:
    * the TEACHER predicts on the CLEAN window (detached -> stable targets);
    * the STUDENT predicts on a PERTURBED window where every preposition token is
      replaced by [MASK];
    * mt_loss = MSE between the student's boundary probabilities and the teacher's
      boundary probabilities, over valid (labelled) positions.
  loss = weighted_CE(student on CLEAN window, labels) + mt_weight * mt_loss.
  After each optimizer step the teacher is updated by EMA from the student.
  (The teacher gives a stable target; the student must reproduce it even though
  the preposition it used to lean on is gone.)

METHOD 4 — CROSS-VIEW TRAINING / CVT  (--cvt-weight, default 0)
  A small AUXILIARY boundary head is added. The window is encoded TWICE:
    * FULL view (all tokens) -> the MAIN head predicts here; this is the real
      prediction and carries the cross-entropy;
    * RESTRICTED view (prepositions replaced by [MASK]) -> the AUX head predicts
      on this preposition-hidden representation and is trained to MATCH the main
      head: target = softmax(main logits).detach().
  cvt_loss = KL( softmax(aux logits) || target ) over valid positions.
  loss = weighted_CE(main logits on FULL view, labels) + cvt_weight * cvt_loss.
  The aux head is discarded at inference.
  (Forces a boundary decision that is recoverable WITHOUT the preposition.)

FLAG-GATING / BIT-FOR-FOR-BIT GUARANTEE
  With --mt-weight 0 AND --cvt-weight 0 (the defaults) NEITHER extra view is
  built, NO teacher is constructed, the aux head is never touched, and no extra
  randomness is drawn — forward() returns exactly the baseline class-weighted-CE
  logits/loss, BIT-FOR-BIT identical to the plain weighted-CE tagger. The [MASK]
  perturbation is computed deterministically from token identity (which words are
  prepositions), so it consumes no RNG and never perturbs the dropout stream.

CLOSED-TRACK / TRAIN-ONLY: fits/reads only the train split (the 174 docs). dev is
used solely by the existing eval for checkpoint selection, never as training data.

Usage:
  python teacher_train.py train --task NoPnx-NP --mt-weight 1.0  --out runs/mt
  python teacher_train.py train --task NoPnx-NP --cvt-weight 1.0 --out runs/cvt
  python teacher_train.py train --task NoPnx-NP --out runs/base          # plain CE
  python teacher_train.py predict --task NoPnx-NP --split dev --model runs/mt --out subs/x.csv
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))

# The Arabic prepositions / connectives the model OVER-fires on. Kept identical
# to connective_aug.CONNECTIVE_WORDS so the two attacks target the same set.
PREPOSITIONS = ["و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
                "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى"]
PREPOSITIONS_SET = set(PREPOSITIONS)


class TeacherModel(nn.Module):
    """Boundary tagger with two flag-gated consistency heads.

    * bound : the real per-token boundary head (2-way). Always used.
    * aux   : CVT auxiliary boundary head (2-way). Exists as parameters but is
              only exercised when cvt_weight > 0; discarded at inference.

    With mt_weight == 0 and cvt_weight == 0, forward() runs ONLY view A (the
    clean input) through the encoder + bound head + weighted CE — bit-for-bit the
    plain weighted-CE baseline. No second forward, no teacher, no aux head.
    """

    def __init__(self, model_name, pos_weight=1.0, mt_weight=0.0, cvt_weight=0.0,
                 n_extra_tok=1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.bound = nn.Linear(h, 2)
        # CVT auxiliary head (preposition-hidden view -> match main head).
        self.aux = nn.Linear(h, 2)
        self.mt_weight = mt_weight
        self.cvt_weight = cvt_weight
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def _encode(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids,
                        attention_mask=attention_mask).last_hidden_state

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                input_ids_masked=None, teacher_probs=None, **kw):
        # ---- View A: standard forward on the CLEAN input (the baseline path) ----
        hs = self._encode(input_ids, attention_mask)
        logits = self.bound(self.drop(hs))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(hs.device), ignore_index=-100)
            valid = (labels.view(-1) != -100)

            # ---- Method 3: MEAN TEACHER (only when mt_weight > 0) ----
            # teacher_probs are supplied by the trainer (EMA teacher on the CLEAN
            # input, detached). The student re-encodes the PREPOSITION-MASKED input
            # and must match the teacher's boundary probabilities.
            if self.mt_weight > 0 and teacher_probs is not None and valid.any():
                hs_m = self._encode(input_ids_masked, attention_mask)
                stu_logits = self.bound(self.drop(hs_m))
                stu_p = torch.softmax(stu_logits.view(-1, 2).float(), -1)[valid]
                tea_p = teacher_probs.view(-1, 2).float()[valid]
                mt_loss = nn.functional.mse_loss(stu_p, tea_p)
                loss = loss + self.mt_weight * mt_loss

            # ---- Method 4: CROSS-VIEW TRAINING / CVT (only when cvt_weight > 0) ----
            # AUX head on the RESTRICTED (preposition-hidden) view is trained to
            # match the MAIN head's decision on the FULL view (detached target).
            if self.cvt_weight > 0 and input_ids_masked is not None and valid.any():
                hs_r = self._encode(input_ids_masked, attention_mask)
                aux_logits = self.aux(self.drop(hs_r))
                target = torch.softmax(logits.view(-1, 2).float(), -1).detach()[valid]
                logp = nn.functional.log_softmax(aux_logits.view(-1, 2).float(), -1)[valid]
                cvt_loss = nn.functional.kl_div(logp, target, reduction="batchmean")
                loss = loss + self.cvt_weight * cvt_loss

        return {"loss": loss, "logits": logits}


def _save_hf_checkpoint(model, tok, out_dir, model_name, n_extra_tok=1):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / fair_threshold.py / diag_preposition.py load with
    AutoModelForTokenClassification.from_pretrained(out_dir) + AutoTokenizer, which
    the custom .pt cannot satisfy. The inference model here is ONLY the main
    boundary path: the bare encoder `model.enc` followed by the `model.bound`
    Linear(hidden, 2) head. The CVT aux head (`model.aux`) and the EMA teacher are
    discarded at inference, so they are NOT copied.

    We assemble a standard AutoModelForTokenClassification(model_name, num_labels=2),
    resize its embeddings to match the trainer (vocab + n_extra_tok for [PAR]), then
    load the TRAINED encoder body into `.bert` (strict=False tolerates the encoder's
    unused pooler) and copy the TRAINED `bound` weights into `.classifier`. The
    result reproduces the trainer's inference logits exactly. This does not touch
    training, the loss, the flags, or the existing .pt/cfg.json outputs.
    """
    hf = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    hf.resize_token_embeddings(hf.config.vocab_size + n_extra_tok)
    # trained BERT body -> hf.bert (pooler in enc is unused for token-cls; ignore it)
    hf.bert.load_state_dict(model.enc.state_dict(), strict=False)
    # trained boundary head -> hf.classifier (Linear(hidden, 2), same shape as bound)
    with torch.no_grad():
        hf.classifier.weight.copy_(model.bound.weight)
        hf.classifier.bias.copy_(model.bound.bias)
    hf.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def make_examples(docs, window, stride):
    """Cut docs into overlapping windows (window / stride), mirroring the
    reference trainers. is_prep marks preposition words at WORD granularity;
    encode() expands it to the [MASK]-replacement subword mask deterministically
    (no RNG), so it never perturbs the dropout stream."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        is_prep = [1 if w in PREPOSITIONS_SET else 0 for w in words]
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            out.append({"words": words[s:e], "word_labels": labs[s:e],
                        "is_prep": is_prep[s:e]})
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, max_len):
    enc = tok(examples["words"], is_split_into_words=True, truncation=True, max_length=max_len)
    all_lab, all_prep = [], []
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        pw = examples["is_prep"][i]
        lab = [-100] * len(wid)
        prep = [0] * len(wid)
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            prep[pos] = pw[w]                       # every subword of a prep word
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:
                lab[pos] = wl[w]                    # label the LAST subword of each word
        all_lab.append(lab)
        all_prep.append(prep)
    enc["labels"] = all_lab
    enc["is_prep"] = all_prep
    return enc


class Collator:
    """Pad labels + the preposition mask, then build input_ids_masked by
    replacing every preposition subword with [MASK]. Deterministic (no RNG)."""

    def __init__(self, tok):
        self.tok = tok
        self.mask_id = tok.mask_token_id

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        prep = [f.pop("is_prep") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        prep_t = torch.tensor([p + [0] * (m - len(p)) for p in prep], dtype=torch.bool)
        # PREPOSITION-MASKED copy of the input: prep subwords -> [MASK].
        ids = batch["input_ids"].clone()
        if self.mask_id is not None:
            ids[prep_t] = self.mask_id
        batch["input_ids_masked"] = ids
        return batch


def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


class TeacherTrainer(Trainer):
    """Owns the EMA teacher for Method 3. With mt_weight == 0 the teacher is never
    built and compute_loss reduces to the model's plain weighted-CE path — no EMA
    bookkeeping, no extra forward. cvt_weight is handled entirely inside the model.
    """

    def __init__(self, *args, mt_weight=0.0, ema_decay=0.999, **kw):
        super().__init__(*args, **kw)
        self._mt_weight = mt_weight
        self._ema_decay = ema_decay
        self._teacher = None
        if mt_weight > 0:
            # EMA teacher = a frozen deep copy of the student, updated after each
            # optimizer step. No gradient ever flows to it.
            self._teacher = copy.deepcopy(self.model)
            for p in self._teacher.parameters():
                p.requires_grad_(False)
            self._teacher.eval()
            # Update the teacher EXACTLY ONCE per optimizer step (standard
            # Mean-Teacher, Tarvainen & Valpola 2017). on_optimizer_step fires
            # after each real optimizer.step(), so with gradient_accumulation_steps
            # > 1 the teacher no longer blends on the intermediate micro-batches
            # (where the student has not moved yet). With accumulation == 1 (the
            # cmd_train default) this is one EMA per micro-batch — identical to the
            # previous behaviour, so the default path is unchanged.
            self.add_callback(_EMACallback(self))

    @torch.no_grad()
    def _ema_update(self):
        d = self._ema_decay
        for tp, sp in zip(self._teacher.parameters(), self.model.parameters()):
            tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)
        # buffers (e.g. LayerNorm running stats, the pos_weight buffer) tracked hard
        for tb, sb in zip(self._teacher.buffers(), self.model.buffers()):
            tb.copy_(sb)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs = dict(inputs)
        if self._mt_weight > 0 and self._teacher is not None:
            # TEACHER predicts on the CLEAN input; detach -> stable target probs.
            if next(self._teacher.parameters()).device != inputs["input_ids"].device:
                self._teacher.to(inputs["input_ids"].device)
            with torch.no_grad():
                t_logits = self._teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"])["logits"]
                inputs["teacher_probs"] = torch.softmax(t_logits.float(), -1).detach()
        else:
            # Ensure the baseline path never sees teacher targets.
            inputs.pop("teacher_probs", None)
        out = model(**inputs)
        loss = out["loss"]
        return (loss, out) if return_outputs else loss


class _EMACallback(TrainerCallback):
    """Fire the EMA teacher update once per OPTIMIZER step (not per micro-batch).

    on_optimizer_step runs immediately after optimizer.step(), i.e. after the
    student weights have actually moved and exactly once per real training step,
    regardless of gradient_accumulation_steps. This matches the Mean-Teacher
    definition (teacher = EMA of the student, updated each optimizer step)."""

    def __init__(self, trainer):
        self._trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        t = self._trainer
        if t._mt_weight > 0 and t._teacher is not None:
            t._ema_update()


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)   # TRAIN-ONLY
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)       # eval selection only
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  mt_weight={a.mt_weight}  ema_decay={a.ema_decay}  "
          f"cvt_weight={a.cvt_weight}")
    model = TeacherModel(a.model_name, pw, a.mt_weight, a.cvt_weight, n_extra_tok=1)
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
    tr_ = TeacherTrainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                         data_collator=Collator(tok), compute_metrics=metrics,
                         mt_weight=a.mt_weight, ema_decay=a.ema_decay)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name}, open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: also emit an HF checkpoint (main boundary path only; CVT aux head +
    # EMA teacher discarded) so predict.py / fair_threshold.py / diag_preposition.py
    # can load this run dir. Nothing above is changed.
    _save_hf_checkpoint(model, tok, a.out, a.model_name, n_extra_tok=1)
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    # Gate on an actually-usable device so CPU prediction never deserializes onto
    # a phantom cuda:0 (mirrors contrastive_train.cmd_predict).
    dev = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = TeacherModel(cfg["model_name"], n_extra_tok=1)
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
    t.add_argument("--mt-weight", type=float, default=0.0,
                   help="Method 3 mean-teacher MSE weight (0=off => plain CE)")
    t.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for the mean teacher")
    t.add_argument("--cvt-weight", type=float, default=0.0,
                   help="Method 4 CVT KL weight (0=off => plain CE)")
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
