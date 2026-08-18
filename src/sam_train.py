"""Sharpness-Aware Minimization (SAM) boundary tagger for AraSeg (closed track).

Self-contained trainer — does NOT touch the shared train_encoder / predict /
eval_local / stitching logic (it only imports data.py + baselines.write_submission).
The data / window / label / [PAR] handling MIRRORS src/consistency_train.py exactly
(window 180, stride 90, labels on the LAST subword of each word, MODEL_PAR_TOKEN for
the "\n" paragraph token, weighted CE with pos_weight capped ~8, epochs 8, batch 16,
lr 5e-5). Base model: AutoModelForTokenClassification(aubmindlab/bert-base-arabertv02,
num_labels=2).

METHOD — SHARPNESS-AWARE MINIMIZATION  [Foret et al., ICLR 2021]  (--sam-rho, default 0.0 = OFF)
--------------------------------------------------------------------------------------------------
SAM is an OPTIMIZER WRAPPER: each training step does two forward-backwards.
  (1) g = ∇L(w)                              (ascent gradient at the current weights w)
  (2) eps = rho * g / ||g||_2                 (||·|| = a SINGLE global L2 norm over ALL params)
  (3) second forward-backward at w + eps      (the "perturbed" / sharp point)
  (4) apply THAT (perturbed) gradient to the ORIGINAL weights w via optimizer.step()
  (5) restore w  (undo the eps perturbation) BEFORE the optimizer update touches it
This minimizes the loss in a neighbourhood (a flat basin) rather than at a sharp point,
which tends to reduce over-confident over-firing on the boundary class.

  * --sam-rho == 0.0  (DEFAULT / OFF):  no ascent step at all — a single forward-backward,
    bit-for-bit identical to plain weighted-CE (same loss, same logits, and IDENTICAL RNG
    consumption: exactly one model forward per training step, so dropout draws match).
    Verified with torch.equal + a param-hash check in src/smoke_sam.py.
  * --sam-rho  > 0.0:  two distinct forward-backwards; rho sweep 0.01–0.05.

CLASS-ASYMMETRIC variant  (--sam-class-alpha, default 1.0 = symmetric)
----------------------------------------------------------------------
Specialization of Focal-SAM / class-asymmetric SAM to the C=2 boundary problem.
Inside the ASCENT (perturbation) loss ONLY, boundary tokens (gold == 1) are weighted by
alpha (> 1) and non-boundary tokens (gold == 0) by 1. This flattens the POSITIVE-class
basin harder, so the perturbation attacks over-confident boundary firing specifically.
The DESCENT loss (step 3's second forward, the one whose grad is actually applied) stays
the NORMAL weighted CE — alpha changes WHERE we look for sharpness, not the objective.

  * --sam-class-alpha == 1.0:  ascent loss == descent loss == plain weighted CE.
  * --sam-class-alpha  > 1.0:  ascent loss re-weights gold==1 positions by alpha; the
    descent loss is unchanged. Only meaningful when --sam-rho > 0 (no ascent otherwise).

Usage:
  # baseline (bit-for-bit identical to plain weighted CE):
  python sam_train.py train --task NoPnx-NP --out runs/sam_base \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # SAM, symmetric:
  python sam_train.py train --task NoPnx-NP --sam-rho 0.02 --out runs/sam_r02 \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # SAM, class-asymmetric (flatten the boundary basin harder):
  python sam_train.py train --task NoPnx-NP --sam-rho 0.02 --sam-class-alpha 2.0 \
      --out runs/sam_r02_a2 \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  python sam_train.py predict --task NoPnx-NP --split dev --model runs/sam_r02 \
      --jsonl data/NoPnx-NP_dev.jsonl --out subs/sam_r02_dev.csv
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
                          Trainer, TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py verbatim)             #
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


class Collator:
    """Pads input_ids / attention_mask / labels to a rectangular batch.

    Plain (mirrors consistency_train.Collator minus the counterfactual masks —
    SAM needs no extra per-batch tensors)."""
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        return batch


# --------------------------------------------------------------------------- #
# model wrapper: plain token-classification + class-asymmetric ascent loss    #
# --------------------------------------------------------------------------- #
class SAMModel(nn.Module):
    """Thin wrapper over a single AutoModelForTokenClassification.

    `self.enc` IS the trained inference model (encoder + boundary classifier); SAM
    adds NO parameters, so `self.enc` is exactly what gets written by
    `_save_hf_checkpoint` (mirrors consistency_train.ConsistencyModel). The
    optimizer perturbs `self.enc`'s parameters directly during the SAM ascent step.

    forward() returns the DESCENT loss (normal weighted CE) — this is the standard
    objective used both by the HF Trainer at rho=0 and as SAM's second (descent)
    pass. The class-asymmetric ASCENT loss is a separate method `ascent_loss`, used
    only by SAMTrainer during the perturbation step.
    """
    def __init__(self, model_name, pos_weight=1.0, class_alpha=1.0, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.class_alpha = float(class_alpha)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def _logits(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).logits

    def _ce(self, logits, labels):
        """Normal weighted CE (the DESCENT objective)."""
        return nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100)

    def _asym_ce(self, logits, labels):
        """Class-asymmetric CE for the ASCENT step: on top of the usual class
        weights, up-weight gold==1 (boundary) tokens by class_alpha.

        PyTorch's weighted cross_entropy with reduction='mean' computes
            Σ_i w[y_i]·loss_i  /  Σ_i w[y_i]      (i over valid tokens).
        We fold class_alpha into BOTH the numerator and that denominator as a
        per-token multiplier a_i (a_i = class_alpha where gold==1 else 1):
            Σ_i a_i·w[y_i]·loss_i  /  Σ_i a_i·w[y_i].
        With class_alpha == 1.0, a_i ≡ 1 and this collapses to exactly the
        weighted-mean above — bit-for-bit equal to self._ce (verified in
        src/smoke_sam.py with torch.equal). class_alpha > 1 simply re-weights the
        positive class inside the ascent objective, so the perturbation direction
        leans on boundary sharpness.
        """
        flat_logits = logits.view(-1, 2).float()
        flat_labels = labels.view(-1)
        wdev = self.w.to(logits.device)
        # per-token weighted CE, no reduction; cross_entropy returns 0 at ignored.
        per = nn.functional.cross_entropy(
            flat_logits, flat_labels, weight=wdev,
            ignore_index=-100, reduction="none")
        valid = flat_labels != -100
        # a_i · w[y_i], the effective per-token weight; a_i = alpha at gold==1.
        alpha_vec = torch.where(flat_labels == 1,
                                torch.as_tensor(self.class_alpha, device=wdev.device, dtype=per.dtype),
                                torch.ones((), device=wdev.device, dtype=per.dtype))
        # w[y_i] for valid tokens (0 where ignored so it never enters the sums).
        w_tok = torch.where(valid, wdev[flat_labels.clamp_min(0)], torch.zeros((), device=wdev.device, dtype=per.dtype))
        eff = alpha_vec * w_tok
        num = (eff * per)[valid].sum()
        den = eff[valid].sum().clamp_min(1e-12)
        return num / den

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        logits = self._logits(input_ids, attention_mask)
        loss = None if labels is None else self._ce(logits, labels)
        return {"loss": loss, "logits": logits}

    def ascent_loss(self, input_ids=None, attention_mask=None, labels=None, **kw):
        """The perturbation objective: normal CE when class_alpha == 1, otherwise
        the boundary-upweighted CE. Consumes exactly ONE forward pass."""
        logits = self._logits(input_ids, attention_mask)
        if self.class_alpha == 1.0:
            return self._ce(logits, labels)
        return self._asym_ce(logits, labels)


# --------------------------------------------------------------------------- #
# SAM optimizer, implemented as a Trainer subclass                            #
# --------------------------------------------------------------------------- #
class SAMTrainer(Trainer):
    """HF Trainer whose training_step performs the SAM two-step update.

    rho == 0.0  -> falls through to the EXACT stock Trainer.training_step: a single
                   forward-backward, identical loss/logits/RNG. Bit-for-bit baseline.
    rho  > 0.0  -> ascent (perturb w -> w+eps) then descent (grad at w+eps), restore
                   w, and hand the descent grad to the stock optimizer.step() (called
                   by the training loop AFTER training_step returns). Weights are
                   restored to the ORIGINAL w before optimizer.step() touches them.
    """
    def __init__(self, *args, sam_rho=0.0, **kw):
        super().__init__(*args, **kw)
        self.sam_rho = float(sam_rho)

    @torch.no_grad()
    def _grad_norm(self, params):
        """Single global L2 norm over the gradients of all params (Foret et al.
        use one shared norm across the whole parameter vector)."""
        device = params[0].grad.device
        return torch.norm(torch.stack([
            p.grad.detach().norm(p=2).to(device) for p in params if p.grad is not None
        ]), p=2)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # OFF path: exact stock behavior, bit-for-bit. One forward-backward only.
        if self.sam_rho == 0.0:
            return super().training_step(model, inputs, num_items_in_batch)

        # ---- SAM ON path -------------------------------------------------- #
        model.train()
        inputs = self._prepare_inputs(inputs)
        core = self.accelerator.unwrap_model(model)
        params = [p for p in model.parameters() if p.requires_grad]

        # (1) ascent gradient g = ∇L_ascent(w)  (class-asymmetric if alpha>1)
        model.zero_grad(set_to_none=True)
        with self.compute_loss_context_manager():
            ascent = core.ascent_loss(**inputs)
        self.accelerator.backward(ascent)

        # (2) eps = rho * g / (||g|| + 1e-12); (3) climb to w + eps.
        #     Snapshot the ORIGINAL w so step (5) restores it EXACTLY via copy_
        #     (not w+eps-eps, which drifts in floating point). This makes the
        #     recipe's "param hash before == after ascent+restore" hold exactly.
        grad_norm = self._grad_norm(params)
        scale = self.sam_rho / (grad_norm + 1e-12)
        w_orig = {}
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                w_orig[p] = p.detach().clone()               # exact original w
                p.add_(p.grad * scale.to(p.grad.device))     # w <- w + eps

        # (4) descent forward-backward at the perturbed point w + eps
        model.zero_grad(set_to_none=True)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        self.accelerator.backward(loss)

        # (5) restore w EXACTLY, BEFORE the training loop calls optimizer.step().
        #     The gradients now sitting on the params are the descent grads at
        #     w+eps, which the stock optimizer.step() applies to the restored w.
        with torch.no_grad():
            for p in params:
                if p in w_orig:
                    p.copy_(w_orig[p])   # w <- original w (bit-for-bit)

        # Match the stock training_step's accumulation scaling exactly. The
        # training loop sets current_gradient_accumulation_steps per batch; fall
        # back to the static arg when training_step is called directly (smoke).
        gacc = getattr(self, "current_gradient_accumulation_steps",
                       self.args.gradient_accumulation_steps)
        return loss.detach() / gacc


# --------------------------------------------------------------------------- #
# metrics / checkpoint / train / predict                                      #
# --------------------------------------------------------------------------- #
def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def _save_hf_checkpoint(model, tok, out_dir):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / eval_local.py / diag_preposition.py load with
    AutoModelForTokenClassification.from_pretrained(out_dir) + AutoTokenizer, which
    the custom .pt cannot satisfy. Here `model.enc` IS the trained
    AutoModelForTokenClassification (encoder + boundary classifier); SAM adds NO
    parameters (it is an optimizer wrapper), so `model.enc` is exactly the
    inference model. We write it verbatim. This mirrors
    consistency_train._save_hf_checkpoint; it does not touch training, the loss,
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
    print(f"pos_weight={pw:.2f}  sam_rho={a.sam_rho}  sam_class_alpha={a.sam_class_alpha}")
    model = SAMModel(a.model_name, pw, class_alpha=a.sam_class_alpha, n_extra_tok=1)
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
    tr_ = SAMTrainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                     data_collator=Collator(tok), compute_metrics=metrics,
                     sam_rho=a.sam_rho)
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
    model = SAMModel(cfg["model_name"], n_extra_tok=1)
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
    t.add_argument("--sam-rho", type=float, default=0.0,
                   help="SAM neighbourhood radius rho (0.0 = OFF = plain weighted CE, "
                        "bit-for-bit; sweep 0.01-0.05)")
    t.add_argument("--sam-class-alpha", type=float, default=1.0,
                   help="class-asymmetric ascent weight on gold==1 (boundary) tokens "
                        "(1.0 = symmetric; >1 flattens the positive-class basin harder; "
                        "only affects the ascent/perturbation loss)")
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
