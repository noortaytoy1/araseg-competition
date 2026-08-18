"""Consistency-regularized boundary tagger for AraSeg NoPnx-NP (closed track).

Self-contained — does NOT touch the shared train_encoder/predict pipeline. The
data / window / label / [PAR] handling MIRRORS src/coherence_train.py and
src/contrastive_train.py exactly (window 180, stride 90, labels on the LAST
subword of each word, MODEL_PAR_TOKEN for the "\n" paragraph token, weighted CE
with pos_weight capped ~8).

Base model: AutoModelForTokenClassification(aubmindlab/bert-base-arabertv02,
num_labels=2). Two FLAG-GATED consistency methods are added on top of the plain
weighted-CE baseline. Both are OFF by default, and with their weight 0.0 the
trainer reproduces the plain weighted-CE baseline BIT-FOR-BIT (a single forward
pass, identical loss — verified with torch.equal in the GPU smoke).

GOAL of both methods: stop the model OVER-firing at Arabic prepositions /
connectives (the model learns "a boundary tends to precede و / ف / من / ..." and
fires spuriously). The two methods attack that reliance from different angles.

METHOD 1 — COUNTERFACTUAL preposition-swap penalty  (--cf-weight, default 0):
  Forward the batch twice.
    view A = the batch as-is.
    view B = the SAME batch, but every token whose id is a PREPOSITION id is
             replaced by the tokenizer's [MASK] id (positions unchanged).
  Let p_A, p_B = softmax(logits)[..., 1] (boundary prob per token).
  cf_loss = mean, over label positions whose NEXT token is a preposition, of
            (p_A - p_B) ** 2.
  i.e. penalize the boundary decision for CHANGING when the following
  preposition is hidden. This punishes reliance on the preposition.
  loss = weighted_CE(view A logits, labels) + cf_weight * cf_loss.

METHOD 2 — R-DROP  (--rdrop-weight, default 0):
  Forward the SAME input twice with independent dropout -> logits1, logits2.
  rdrop = 0.5 * (KL(p1 || p2) + KL(p2 || p1)) averaged over valid
          (label != -100) positions, p = softmax over the 2 classes.
  loss = 0.5 * (CE1 + CE2) + rdrop_weight * rdrop.

METHOD 3 — SAM optimizer wrapper  (--sam-rho, default 0.0 = OFF)  [ADDITIVE, 2026-07-10]:
  Sharpness-Aware Minimization [Foret et al., ICLR 2021], mirroring
  src/sam_train.py's SAMTrainer mechanism EXACTLY:
    (1) first forward-backward: g = grad of L(w) at the current weights w
    (2) eps = rho * g / (||g||_2 + 1e-12)   (single global L2 norm over ALL params)
    (3) second forward-backward at w + eps  (the perturbed point)
    (4) restore w EXACTLY (snapshot copy_, not w+eps-eps) BEFORE optimizer.step()
    (5) the stock optimizer.step() applies the perturbed-point gradient to w
  COMPOSITION: unlike sam_train.py there is NO separate ascent objective here —
  BOTH the ascent and the descent forward-backward use the model's FULL loss,
  i.e. weighted CE PLUS any active consistency term. With --rdrop-weight > 0 the
  R-Drop term is inside the loss of BOTH SAM passes (each pass then does two
  encoder forwards internally, one per dropout view).
    * --sam-rho == 0.0 (DEFAULT / OFF): the trainer is the plain stock HF Trainer,
      bit-for-bit and RNG-identical to the pre-flag trainer (single-pass).
    * --sam-rho  > 0.0: two full forward-backwards per training step.

METHOD 4 — Child-Tuning-D  (--child-tune, default 0.0 = OFF)  [ADDITIVE, 2026-07-10]:
  Xu et al., "Raise a Child in Large Language Model: Towards Effective and
  Generalizable Fine-tuning", EMNLP 2021 — task-driven (D) variant.
  BEFORE training: ONE pass over the (windowed, encoded) train set in eval mode
  accumulates grad^2 of the plain weighted-CE loss for every ENCODER parameter
  (model.enc.* EXCEPT the classifier head) -> empirical Fisher information.
  The top-P fraction of encoder weight ENTRIES by Fisher (exact global top-k,
  k = round(P * n_encoder_entries)) becomes the fixed CHILD mask.
  DURING training: per-parameter backward hooks multiply each incoming encoder
  gradient elementwise by its 0/1 mask (i.e. "after backward" the accumulated
  .grad is already masked — non-child entries get ZERO gradient). The
  head/classifier is NEVER masked. The mask is computed ONCE and stored (plain
  tensors on the model, deliberately NOT register_buffer so model.pt keeps its
  exact legacy key set and cmd_predict's strict load_state_dict still works).
  Grad-masking ONLY, exactly like the reference implementation's
  ChildTuningAdamW-D: AdamW's decoupled weight decay is untouched.
  With --sam-rho > 0 the hooks fire in BOTH SAM passes, so the ascent
  perturbation eps is also confined to the child network.
  --child-tune 0.0 (DEFAULT / OFF): no Fisher pass, no hooks — nothing runs.

METHOD 5 — SWA, stochastic weight averaging  (--swa off/on, default off)
  [ADDITIVE, 2026-07-10]:
  From --swa-start-frac (default 0.6) of training onward, a read-only callback
  snapshots the live weights at the END of every epoch e with
  e >= swa_start_frac * epochs and maintains their running EQUAL average
  IN MEMORY (float64 accumulator; the trainers keep no ckpts on disk). Training
  itself is untouched — the callback only reads. At the end, AFTER the normal
  best checkpoint is saved exactly as before, the SWA-averaged weights are
  ALSO saved as a second, self-contained HF checkpoint variant in <out>/swa
  (config + weights + tokenizer + swa_meta.json), so the scorer can score both.
  No BatchNorm anywhere in this model (LayerNorm only), so no BN-recalibration
  pass is needed. --swa off (DEFAULT): no callback, no swa/ dir — bit-for-bit
  legacy.

The flags are independent; they may be combined (the sequel battery runs
--rdrop-weight 1.0 --sam-rho 0.05; the integration battery's Stage-1 arms run
--child-tune 0.3, --child-tune 0.3 --swa on, and --rdrop-weight 1.0
--sam-rho 0.05 --swa on), but the earlier batteries run them one at a time.
Only the pre-registered combinations are run.

Usage:
  # baseline (bit-for-bit identical to plain weighted CE):
  python consistency_train.py train --task NoPnx-NP --out runs/cons_base \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # counterfactual:
  python consistency_train.py train --task NoPnx-NP --cf-weight 1.0 --out runs/cons_cf \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # r-drop:
  python consistency_train.py train --task NoPnx-NP --rdrop-weight 1.0 --out runs/cons_rdrop \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # r-drop + SAM (the sequel-battery composition):
  python consistency_train.py train --task NoPnx-NP --rdrop-weight 1.0 --sam-rho 0.05 \
      --out runs/cons_rdrop_sam \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # child-tuning-D (integration Stage-1):
  python consistency_train.py train --task NoPnx-NP --child-tune 0.3 --out runs/cons_ct \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # child-tuning-D + SWA (integration Stage-1; swa/ variant saved alongside best):
  python consistency_train.py train --task NoPnx-NP --child-tune 0.3 --swa on \
      --out runs/cons_ct_swa \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  python consistency_train.py predict --task NoPnx-NP --split dev --model runs/cons_cf \
      --jsonl data/NoPnx-NP_dev.jsonl --out subs/cons_cf_dev.csv
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
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))

# Arabic prepositions / connectives the model tends to OVER-fire in front of.
# Each of these is a single AraBERTv02 subword (verified), so masking the token
# whose id matches one of these hides exactly that word.
PREPOSITIONS = ["و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
                "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى"]


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors coherence_train.py / contrastive_train.py)  #
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
    """Pads, and precomputes two boolean masks used by the counterfactual loss:

      prep_tok_mask     : True where the input id IS a preposition id
                          (these positions get [MASK]'d for view B).
      next_is_prep_mask : True at a label position whose NEXT token (the first
                          real token after it) is a preposition — these are the
                          positions cf_loss is averaged over.
    """
    def __init__(self, tok, prep_ids):
        self.tok = tok
        self.prep_ids = set(prep_ids)

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])

        ids = batch["input_ids"]
        attn = batch["attention_mask"].bool()
        prep_tok = torch.zeros_like(ids, dtype=torch.bool)
        for pid in self.prep_ids:
            prep_tok |= (ids == pid)
        prep_tok &= attn  # never touch padding
        batch["prep_tok_mask"] = prep_tok

        # next_is_prep: position i is flagged if the next *real* (attended,
        # non-pad) token i+1 is a preposition token. Shift prep_tok left by one.
        nxt = torch.zeros_like(prep_tok)
        nxt[:, :-1] = prep_tok[:, 1:]
        batch["next_is_prep_mask"] = nxt & attn
        return batch


# --------------------------------------------------------------------------- #
# model wrapper: plain token-classification + two flag-gated consistency terms #
# --------------------------------------------------------------------------- #
class ConsistencyModel(nn.Module):
    def __init__(self, model_name, pos_weight=1.0, cf_weight=0.0, rdrop_weight=0.0,
                 mask_id=None, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.cf_weight = float(cf_weight)
        self.rdrop_weight = float(rdrop_weight)
        self.mask_id = mask_id
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def _logits(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).logits

    def _ce(self, logits, labels):
        return nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                prep_tok_mask=None, next_is_prep_mask=None, **kw):
        logits = self._logits(input_ids, attention_mask)   # view A / pass 1
        loss = None
        if labels is not None:
            ce = self._ce(logits, labels)
            loss = ce

            # -- METHOD 1: counterfactual preposition-swap penalty ------------ #
            # Gated OFF at weight 0: no second forward pass at all, so the loss
            # is exactly plain weighted CE (bit-for-bit).
            if self.cf_weight > 0 and prep_tok_mask is not None:
                masked_ids = input_ids.clone()
                masked_ids[prep_tok_mask] = self.mask_id
                logits_b = self._logits(masked_ids, attention_mask)  # view B
                p_a = torch.softmax(logits, -1)[..., 1]
                p_b = torch.softmax(logits_b, -1)[..., 1]
                sel = next_is_prep_mask
                if sel is not None and sel.any():
                    cf = ((p_a[sel] - p_b[sel]) ** 2).mean()
                else:
                    cf = (p_a * 0.0).sum()  # no eligible positions -> 0, keeps graph
                loss = loss + self.cf_weight * cf

            # -- METHOD 2: R-DROP -------------------------------------------- #
            # Gated OFF at weight 0: no second forward pass, loss == plain CE.
            if self.rdrop_weight > 0:
                logits2 = self._logits(input_ids, attention_mask)  # pass 2 (fresh dropout)
                ce2 = self._ce(logits2, labels)
                valid = (labels.view(-1) != -100)
                lp1 = torch.log_softmax(logits.view(-1, 2).float(), -1)[valid]
                lp2 = torch.log_softmax(logits2.view(-1, 2).float(), -1)[valid]
                p1, p2 = lp1.exp(), lp2.exp()
                kl_12 = (p1 * (lp1 - lp2)).sum(-1)
                kl_21 = (p2 * (lp2 - lp1)).sum(-1)
                if valid.any():
                    rdrop = 0.5 * (kl_12 + kl_21).mean()
                else:
                    rdrop = (logits2 * 0.0).sum()
                loss = 0.5 * (ce + ce2) + self.rdrop_weight * rdrop

        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# SAM optimizer wrapper, implemented as a Trainer subclass                     #
# (ADDITIVE 2026-07-10; mirrors src/sam_train.py::SAMTrainer verbatim except   #
# that the ASCENT pass uses the model's FULL loss — CE + any active            #
# consistency term (cf / rdrop) — instead of sam_train's separate              #
# class-asymmetric ascent_loss. Ascent objective == descent objective here.)   #
# --------------------------------------------------------------------------- #
class ConsistencySAMTrainer(Trainer):
    """HF Trainer whose training_step performs the SAM two-step update.

    sam_rho == 0.0 -> falls through to the EXACT stock Trainer.training_step: a
                   single forward-backward, identical loss/logits/RNG.
                   Bit-for-bit baseline (cmd_train additionally keeps the plain
                   Trainer class at the 0.0 default, so the legacy path is the
                   untouched pre-flag code object).
    sam_rho  > 0.0 -> ascent (perturb w -> w+eps) then descent (grad at w+eps),
                   restore w, and hand the descent grad to the stock
                   optimizer.step() (called by the training loop AFTER
                   training_step returns). Weights are restored to the ORIGINAL
                   w before optimizer.step() touches them. BOTH passes compute
                   the FULL model loss, so --rdrop-weight (and --cf-weight)
                   compose with SAM: the R-Drop term is inside the ascent AND
                   the descent objective.
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

        # ---- SAM ON path (mirrors sam_train.SAMTrainer.training_step) ----- #
        model.train()
        inputs = self._prepare_inputs(inputs)
        params = [p for p in model.parameters() if p.requires_grad]

        # (1) ascent gradient g = ∇L_full(w). L_full is the model's forward
        #     loss — weighted CE PLUS the active consistency terms (cf/rdrop).
        model.zero_grad(set_to_none=True)
        with self.compute_loss_context_manager():
            ascent = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        self.accelerator.backward(ascent)

        # (2) eps = rho * g / (||g|| + 1e-12); (3) climb to w + eps.
        #     Snapshot the ORIGINAL w so step (5) restores it EXACTLY via copy_
        #     (not w+eps-eps, which drifts in floating point).
        grad_norm = self._grad_norm(params)
        scale = self.sam_rho / (grad_norm + 1e-12)
        w_orig = {}
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                w_orig[p] = p.detach().clone()               # exact original w
                p.add_(p.grad * scale.to(p.grad.device))     # w <- w + eps

        # (4) descent forward-backward at the perturbed point w + eps —
        #     the SAME full loss (CE + cf + rdrop), fresh dropout draws.
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
# METHOD 4 — Child-Tuning-D: Fisher pass + fixed gradient-mask hooks           #
# (ADDITIVE 2026-07-10; --child-tune 0.0 default = none of this code runs)     #
# --------------------------------------------------------------------------- #
def _encoder_named_params(model):
    """ENCODER parameters of the ConsistencyModel = everything under model.enc
    EXCEPT the classification head (model.enc.classifier.*). Child-Tuning-D
    masks the pretrained network only; the task head is NEVER masked."""
    return [(n, p) for n, p in model.enc.named_parameters()
            if not n.startswith("classifier.")]


def compute_child_mask(model, loader, top_p, device):
    """ONE pass over the (windowed, encoded) train set in eval mode: accumulate
    grad^2 of the PLAIN weighted-CE loss (no cf / rdrop terms) per encoder
    parameter -> empirical Fisher information. Keep the exact global top-k
    weight ENTRIES, k = round(top_p * n_encoder_entries), as the fixed 0/1
    CHILD mask. Computed ONCE before training; never recomputed per step.
    Returns {param_name: float 0/1 mask shaped like the param}."""
    was_training = model.training
    model.eval()  # dropout OFF: Fisher of the deterministic task loss
    enc_params = _encoder_named_params(model)
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in enc_params}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        model.zero_grad(set_to_none=True)
        logits = model._logits(batch["input_ids"], batch["attention_mask"])
        ce = model._ce(logits, batch["labels"])       # plain weighted CE ONLY
        ce.backward()
        with torch.no_grad():
            for n, p in enc_params:
                if p.grad is not None:
                    fisher[n] += p.grad.detach().float() ** 2
        n_batches += 1
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    flat = torch.cat([fisher[n].reshape(-1) for n, _ in enc_params])
    n_entries = flat.numel()
    k = int(round(top_p * n_entries))
    k = max(1, min(k, n_entries))
    idx = torch.topk(flat, k, sorted=False).indices    # exact global top-k
    mask_flat = torch.zeros(n_entries, dtype=torch.float32, device=flat.device)
    mask_flat[idx] = 1.0
    masks, off = {}, 0
    for n, p in enc_params:
        cnt = p.numel()
        masks[n] = mask_flat[off:off + cnt].view_as(p).to(p.device)
        off += cnt
    print(f"child-tune Fisher pass: {n_batches} batches, {n_entries} encoder "
          f"entries, kept k={k} ({k / n_entries:.4f}); head/classifier unmasked")
    return masks


def apply_child_hooks(model, masks):
    """Per-parameter backward hooks: every incoming ENCODER gradient is
    multiplied elementwise by its fixed 0/1 mask, so after backward the
    accumulated .grad is already masked (non-child entries get ZERO gradient).
    Head/classifier params get NO hook. The masks are stored ONCE on the model
    as a plain attribute (deliberately NOT register_buffer, so model.pt keeps
    its exact legacy key set and cmd_predict's strict load_state_dict still
    works). With --sam-rho > 0 the hooks fire in BOTH SAM passes."""
    model._child_masks = masks          # plain attribute, not a buffer/param
    for n, p in _encoder_named_params(model):
        m = masks[n]
        p.register_hook(lambda grad, m=m: grad * m.to(grad.device))


# --------------------------------------------------------------------------- #
# METHOD 5 — SWA: in-memory equal average of per-epoch weight snapshots        #
# (ADDITIVE 2026-07-10; --swa off default = callback never constructed)        #
# --------------------------------------------------------------------------- #
class SWACallback(TrainerCallback):
    """READ-ONLY callback. At the END of each training epoch e (1-based int
    counter; state.epoch is float) with e >= swa_start_frac * total_epochs,
    add the LIVE model.enc weights to a float64 running-sum accumulator held
    IN MEMORY (the trainers keep no ckpts on disk). Training itself is never
    touched. swa_state_dict() returns the equal average (sum / n_snapshots)
    cast back to the checkpoint dtypes."""

    def __init__(self, model, swa_start_frac, total_epochs):
        self.model = model
        self.start_after = float(swa_start_frac) * float(total_epochs)
        self.n_snapshots = 0
        self.epochs_in = []
        self._sum = None
        self._epoch = 0

    def on_epoch_end(self, args, state, control, **kw):
        self._epoch += 1
        if self._epoch < self.start_after:
            return
        sd = self.model.enc.state_dict()
        if self._sum is None:
            self._sum = {k: v.detach().to("cpu", torch.float64).clone()
                         for k, v in sd.items()}
        else:
            for k, v in sd.items():
                self._sum[k] += v.detach().to("cpu", torch.float64)
        self.n_snapshots += 1
        self.epochs_in.append(self._epoch)

    def swa_state_dict(self, like):
        """Equal average of the snapshots, dtype-matched to `like` (the live
        enc state_dict), ready for load_state_dict."""
        assert self.n_snapshots > 0, \
            "SWA: no snapshots taken (swa-start-frac too high for this epoch count?)"
        return {k: (self._sum[k] / self.n_snapshots).to(like[k].dtype)
                for k in self._sum}


# --------------------------------------------------------------------------- #
# metrics / train / predict                                                   #
# --------------------------------------------------------------------------- #
def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def _prep_ids(tok):
    """The single-subword id of each preposition (skip any that are multi-piece)."""
    ids = []
    for p in PREPOSITIONS:
        pieces = tok(p, add_special_tokens=False)["input_ids"]
        if len(pieces) == 1:
            ids.append(pieces[0])
    return sorted(set(ids))


def _save_hf_checkpoint(model, tok, out_dir):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / fair_threshold.py / diag_preposition.py load with
    AutoModelForTokenClassification.from_pretrained(out_dir) + AutoTokenizer, which
    the custom .pt cannot satisfy. Here `model.enc` IS the trained
    AutoModelForTokenClassification (encoder + boundary classifier); the aux
    consistency terms (cf / rdrop) add NO parameters, so `model.enc` is exactly the
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
    prep_ids = _prep_ids(tok)
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  cf_weight={a.cf_weight}  rdrop_weight={a.rdrop_weight}  "
          f"n_prep_ids={len(prep_ids)}")
    if a.sam_rho > 0.0:  # ADDITIVE: silent at the 0.0 default (legacy stdout unchanged)
        print(f"sam_rho={a.sam_rho}  (SAM wrapper ON: 2 fwd-bwd/step, full loss in both passes)")
    if a.child_tune > 0.0:  # ADDITIVE: silent at the 0.0 default
        print(f"child_tune={a.child_tune}  (Child-Tuning-D ON: encoder grads masked to the "
              f"top-{a.child_tune} Fisher entries; head unmasked)")
    if a.swa == "on":  # ADDITIVE: silent at the 'off' default
        print(f"swa=on  swa_start_frac={a.swa_start_frac}  (epoch-end snapshots from epoch "
              f">= {a.swa_start_frac * a.epochs:.2f} equal-averaged in memory -> <out>/swa)")
    model = ConsistencyModel(a.model_name, pw, a.cf_weight, a.rdrop_weight,
                             mask_id=tok.mask_token_id, n_extra_tok=1)
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
    if a.sam_rho > 0.0:
        # ADDITIVE SAM path: the SAM wrapper trainer (2 fwd-bwd per step).
        tr_ = ConsistencySAMTrainer(model=model, args=targs, train_dataset=trds,
                                    eval_dataset=dvds, data_collator=Collator(tok, prep_ids),
                                    compute_metrics=metrics, sam_rho=a.sam_rho)
    else:
        # DEFAULT 0.0: the untouched pre-flag trainer, bit-for-bit legacy.
        tr_ = Trainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                      data_collator=Collator(tok, prep_ids), compute_metrics=metrics)
    if a.child_tune > 0.0:
        # ADDITIVE METHOD 4 path (never entered at the 0.0 default). Runs AFTER
        # the Trainer has placed the model on its device and BEFORE any training
        # step: one eval-mode Fisher pass over the train windows, then fixed
        # grad-mask hooks. Fisher pass is deterministic (no dropout, no shuffle,
        # no RNG draws) and computed ONCE — no per-step recompute.
        from torch.utils.data import DataLoader
        fisher_loader = DataLoader(trds, batch_size=a.batch_size, shuffle=False,
                                   collate_fn=Collator(tok, prep_ids))
        masks = compute_child_mask(model, fisher_loader, a.child_tune, tr_.args.device)
        apply_child_hooks(model, masks)
    swa_cb = None
    if a.swa == "on":
        # ADDITIVE METHOD 5 path (never entered at the 'off' default). add_callback
        # keeps both Trainer constructor calls above byte-identical to legacy.
        swa_cb = SWACallback(model, a.swa_start_frac, a.epochs)
        tr_.add_callback(swa_cb)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name}, open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: also emit an HF checkpoint so predict.py / fair_threshold.py /
    # diag_preposition.py can load this run dir. Nothing above is changed.
    _save_hf_checkpoint(model, tok, a.out)
    print(f"saved -> {a.out}")
    if a.swa == "on":
        # ADDITIVE METHOD 5 second checkpoint: AFTER the normal best checkpoint
        # is saved exactly as before, ALSO save the SWA equal average as a
        # self-contained HF checkpoint variant in <out>/swa (config + weights +
        # tokenizer + swa_meta.json) so the scorer can score both. LayerNorm
        # only in this model — no BatchNorm-recalibration pass needed. The live
        # (best) weights are restored afterwards, so nothing else changes.
        swa_dir = os.path.join(a.out, "swa")
        best_sd = {k: v.detach().clone() for k, v in model.enc.state_dict().items()}
        model.enc.load_state_dict(swa_cb.swa_state_dict(best_sd))
        _save_hf_checkpoint(model, tok, swa_dir)
        json.dump({"swa": "on", "swa_start_frac": a.swa_start_frac,
                   "epochs": a.epochs, "n_snapshots": swa_cb.n_snapshots,
                   "epochs_averaged": swa_cb.epochs_in},
                  open(os.path.join(swa_dir, "swa_meta.json"), "w"))
        model.enc.load_state_dict(best_sd)   # restore the normal best exactly
        print(f"saved SWA variant (equal avg of {swa_cb.n_snapshots} epoch-end "
              f"snapshots, epochs {swa_cb.epochs_in}) -> {swa_dir}")


@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = ConsistencyModel(cfg["model_name"], mask_id=tok.mask_token_id, n_extra_tok=1)
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
    t.add_argument("--cf-weight", type=float, default=0.0,
                   help="METHOD 1 counterfactual preposition-swap penalty weight (0 = off)")
    t.add_argument("--rdrop-weight", type=float, default=0.0,
                   help="METHOD 2 R-DROP consistency weight (0 = off)")
    t.add_argument("--sam-rho", type=float, default=0.0,
                   help="METHOD 3 SAM neighbourhood radius rho (0.0 = OFF = the exact "
                        "pre-flag single-pass trainer, bit-for-bit; >0 wraps each "
                        "training step with SAM exactly as src/sam_train.py — both "
                        "SAM passes use the FULL loss incl. --cf-weight/--rdrop-weight)")
    t.add_argument("--child-tune", type=float, default=0.0,
                   help="METHOD 4 Child-Tuning-D top-P fraction of ENCODER weight entries "
                        "kept by task-Fisher (0.0 = OFF = no Fisher pass, no hooks — the "
                        "exact legacy trainer; e.g. 0.3 keeps the top 30%% of encoder "
                        "entries and zeroes every other encoder gradient after backward; "
                        "the classifier head is NEVER masked; mask computed once, stored, "
                        "never recomputed per step)")
    t.add_argument("--swa", choices=["on", "off"], default="off",
                   help="METHOD 5 stochastic weight averaging (off = OFF = bit-for-bit "
                        "legacy, no callback, no swa/ dir; on = equal-average the epoch-end "
                        "weight snapshots from --swa-start-frac of training onward, held in "
                        "memory, and ALSO save them as a second HF checkpoint in <out>/swa "
                        "alongside the normal best so the scorer can score both)")
    t.add_argument("--swa-start-frac", type=float, default=0.6,
                   help="METHOD 5: snapshots enter the SWA average from epoch >= "
                        "swa-start-frac * epochs (1-based; inert unless --swa on)")
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
