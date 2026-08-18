"""BYOL self-distillation pretraining of the AraBERTv02 encoder.
======================================================================

Bootstrap Your Own Latent (Grill et al. 2020, "Bootstrap Your Own Latent:
A New Approach to Self-Supervised Learning", NeurIPS 2020) adapted to *token
level* representations of an Arabic transformer encoder, with a
PREPOSITION-AWARE two-view augmentation.

WHAT THIS IS FOR
----------------
Pretrain (self-supervised, NO labels) the `aubmindlab/bert-base-arabertv02`
encoder on a plain-text UNLABELED Arabic corpus, then hand the resulting
encoder to the existing boundary fine-tuner:

    python byol_pretrain.py --corpus data/unlabeled.txt --out runs/byol-arabert
    python train_encoder.py --task NoPnx-NP --model-name runs/byol-arabert \
        --out-dir runs/nopnx-np-byol

The projector and predictor are BYOL scaffolding; they are DISCARDED when the
encoder is saved (train_encoder.py loads an AutoModelForTokenClassification and
attaches a fresh boundary head).

WHY BYOL (no negatives)
-----------------------
BYOL trains WITHOUT negative pairs. Two augmented views of the same passage go
through an ONLINE network and a TARGET network. The online network's predictor
tries to predict the target network's projection of the OTHER view. The target
network is an exponential moving average (EMA) of the online weights with
stop-gradient — it is never trained by backprop. The asymmetry (predictor on
online only) + EMA target is what prevents representational collapse without
needing negatives. See Grill et al. 2020 §3 and the BYOL loss (their Eq. 2).

EXACTLY BYOL — the mechanics implemented here
---------------------------------------------
ONLINE net f_theta -> g_theta (projector) -> q_theta (predictor)
TARGET net f_xi   -> g_xi   (projector)     [NO predictor], xi = EMA(theta)
  * projector g and predictor q are both MLPs: Linear -> BN -> ReLU -> Linear.
  * STOP-GRADIENT on every target output (target params requires_grad_(False),
    target forward under torch.no_grad()).
  * NO negative pairs anywhere in the loss.

TWO VIEWS of each passage (token positions kept ALIGNED between views):
  * view A = LIGHT aug: random token masking of ~`--mask-prob` (default 0.15)
    of the real (non-special) tokens -> [MASK].
  * view B = PREPOSITION-AWARE aug: every token that IS one of the Arabic
    PREPOSITIONS is replaced by [MASK] (positions preserved), PLUS the same
    random token masking. This hides the exact connective cues the boundary
    model over-relies on, forcing the representation to survive without them.
  Both views share the SAME input_ids grid (same length, same [CLS]/[SEP]/[PAD]
  positions), so the per-token targets line up one-to-one.

TOKEN-LEVEL BYOL LOSS (symmetrized):
  For each aligned VALID token position i (a real token, not special/pad, in
  BOTH views), let
      p_A = normalize( q_theta( online(view_A)_i ) )       (online, with grad)
      z_B = normalize( sg( g_xi( target(view_B)_i ) ) )     (target, detached)
  loss_dir = mean_i ( 2 - 2 * <p_A, z_B> )                  (Grill Eq. 2)
  Symmetrize: also predict target(view_A) from online(view_B) and average:
      loss = 0.5 * (loss(A->B) + loss(B->A))
  EMA target update runs AFTER each optimizer step:  xi <- m*xi + (1-m)*theta,
  with m the EMA decay (default 0.996, optionally ramped 1 - (1-m0)*(cos+1)/2).

COLLAPSE GUARD:
  Every `--log-every` steps we log the mean per-dimension standard deviation of
  the L2-normalized ONLINE projections over the batch's valid tokens. Under
  collapse this trends toward 0 (all tokens map to one point); a healthy run
  keeps it well away from 0. We WARN if it drops below `--collapse-std` (default
  1e-3) or falls to < 25% of the first logged value.

SELF-CONTAINED. This module does NOT import or modify train_encoder / predict /
eval / data pipelines. Its ONLY contract with the rest of the repo is the saved
checkpoint: `--out` receives the online encoder via `save_pretrained` plus the
tokenizer, so `AutoModel.from_pretrained(out)` and
`AutoModelForTokenClassification.from_pretrained(out, num_labels=2)` both load
it (a fresh classification head is initialized by the fine-tuner, as usual).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, set_seed

# The Arabic prepositions / connectives the boundary model over-fires on. Kept
# identical to connective_aug.CONNECTIVE_WORDS / teacher_train.PREPOSITIONS so
# the preposition-aware view targets exactly the same set.
PREPOSITIONS = ["و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
                "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "aubmindlab/bert-base-arabertv02"


# --------------------------------------------------------------------------- #
# Corpus loading                                                              #
# --------------------------------------------------------------------------- #
def load_corpus(path: str, min_chars: int = 1) -> List[str]:
    """Load an UNLABELED text corpus.

    Accepts either a plain-text file (one passage per non-empty line) or a JSONL
    file (one object per line with a ``text`` field, else the first string
    value). Blank lines and passages shorter than ``min_chars`` are dropped.
    """
    passages: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = line
            if line[0] in "{[":
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        text = obj.get("text")
                        if text is None:  # first string field as a fallback
                            text = next((v for v in obj.values()
                                         if isinstance(v, str)), "")
                    elif isinstance(obj, list):
                        text = " ".join(str(x) for x in obj)
                except json.JSONDecodeError:
                    text = line  # not actually JSON: treat as raw text
            text = (text or "").strip()
            if len(text) >= min_chars:
                passages.append(text)
    return passages


def preposition_token_ids(tokenizer) -> set:
    """The set of vocabulary ids that a preposition word maps to when it is a
    SINGLE WordPiece (add_special_tokens=False). Multi-piece prepositions are
    handled at the whole-word level in the augmenter (see build_views); this set
    is the fast path for the common single-piece connectives (و, ف, من, ...)."""
    ids = set()
    for w in PREPOSITIONS:
        pieces = tokenizer.encode(w, add_special_tokens=False)
        if len(pieces) == 1:
            ids.add(pieces[0])
    return ids


# --------------------------------------------------------------------------- #
# BYOL networks                                                               #
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """BYOL projector / predictor MLP: Linear -> BN -> ReLU -> Linear.

    Same shape is used for the projector g and the predictor q (Grill et al.
    2020 §3, "Implementation details": both are a linear layer to a 4096-d
    hidden, BatchNorm, ReLU, then a linear layer to the output dim). Here the
    hidden and output dims are configurable and default smaller than the paper's
    image-net sizes because we operate on 768-d BERT token vectors."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim). BatchNorm1d needs (N, C); flatten leading dims.
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.net(x)
        return x.reshape(*shape[:-1], x.shape[-1])


class OnlineNet(nn.Module):
    """f_theta (encoder) -> g_theta (projector) -> q_theta (predictor)."""

    def __init__(self, model_name: str, proj_hidden: int, proj_dim: int,
                 n_extra_tok: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if n_extra_tok:
            self.encoder.resize_token_embeddings(
                self.encoder.config.vocab_size + n_extra_tok)
        h = self.encoder.config.hidden_size
        self.projector = MLP(h, proj_hidden, proj_dim)
        self.predictor = MLP(proj_dim, proj_hidden, proj_dim)

    def represent(self, input_ids, attention_mask) -> torch.Tensor:
        """Raw encoder token representations (pre-projection). Used by the
        collapse guard so it measures the ENCODER's spread, which is what we
        actually keep."""
        return self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden)

    def predict(self, projected: torch.Tensor) -> torch.Tensor:
        return self.predictor(projected)


class TargetNet(nn.Module):
    """f_xi (encoder) -> g_xi (projector). No predictor. EMA of the online net;
    never receives gradients."""

    def __init__(self, online: OnlineNet):
        super().__init__()
        self.encoder = copy.deepcopy(online.encoder)
        self.projector = copy.deepcopy(online.projector)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        return self.projector(hidden)

    @torch.no_grad()
    def update_ema(self, online: OnlineNet, decay: float) -> None:
        """xi <- decay*xi + (1-decay)*theta for encoder + projector params AND
        buffers (BatchNorm running stats, LayerNorm has none but be safe)."""
        for tp, op in zip(self.encoder.parameters(), online.encoder.parameters()):
            tp.mul_(decay).add_(op.detach(), alpha=1.0 - decay)
        for tp, op in zip(self.projector.parameters(), online.projector.parameters()):
            tp.mul_(decay).add_(op.detach(), alpha=1.0 - decay)
        # Buffers (e.g. BatchNorm running_mean/var) are copied, not blended:
        # they are statistics, not weights; EMA of the online buffers is fine
        # and keeps the target's BN consistent with its (EMA) weights.
        for tb, ob in zip(self.encoder.buffers(), online.encoder.buffers()):
            if tb.dtype.is_floating_point:
                tb.mul_(decay).add_(ob.detach(), alpha=1.0 - decay)
            else:
                tb.copy_(ob)
        for tb, ob in zip(self.projector.buffers(), online.projector.buffers()):
            if tb.dtype.is_floating_point:
                tb.mul_(decay).add_(ob.detach(), alpha=1.0 - decay)
            else:
                tb.copy_(ob)


# --------------------------------------------------------------------------- #
# Two-view augmentation                                                       #
# --------------------------------------------------------------------------- #
def build_views(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                special_mask: torch.Tensor, prep_id_mask: torch.Tensor,
                mask_token_id: int, mask_prob: float, rng: torch.Generator):
    """Build the two aligned augmented views.

    Args
    ----
    input_ids       : (B, L) long, the clean batch.
    attention_mask  : (B, L) long/bool, 1 for real+special tokens, 0 for pad.
    special_mask    : (B, L) bool, True where the token is a special token
                      ([CLS]/[SEP]/[PAD]); these are NEVER masked.
    prep_id_mask    : (B, L) bool, True where the token is a preposition token.
    mask_token_id   : int, the [MASK] id.
    mask_prob       : float, per-token random masking probability (~0.15).
    rng             : torch.Generator (CPU) for deterministic, reproducible aug.

    Returns
    -------
    view_a, view_b : (B, L) long tensors (masked input_ids).
    valid          : (B, L) bool, positions that are real (maskable) tokens in
                     BOTH views' shared grid — used to select loss positions.

    Positions are IDENTICAL across the two views (same grid); only the token
    IDENTITY differs, so per-token targets align one-to-one.
    """
    # "valid" = a real content token (attended, not special). The BYOL loss is
    # computed only on these positions. Prepositions ARE valid positions (we
    # want a target for them too); they are simply masked in view B.
    valid = (attention_mask.bool()) & (~special_mask)

    # Random-masking decisions are drawn on CPU with an explicit generator so a
    # given seed reproduces the exact same views (no interference with the model
    # dropout RNG stream). Draw ONE random field and derive both views from it
    # so the light-mask positions coincide where possible (keeps the views as
    # aligned as BYOL's two-crop, differing only by the preposition hiding).
    rand = torch.rand(input_ids.shape, generator=rng)  # (B, L) on CPU
    rand = rand.to(input_ids.device)
    light_mask = valid & (rand < mask_prob)

    # View A: light random masking only.
    view_a = input_ids.clone()
    view_a[light_mask] = mask_token_id

    # View B: preposition tokens hidden (positions preserved) + the SAME light
    # random masking. Preposition hiding is deterministic from token identity.
    view_b = input_ids.clone()
    view_b[light_mask] = mask_token_id
    view_b[valid & prep_id_mask] = mask_token_id

    return view_a, view_b, valid


def word_level_prep_mask(tokenizer, input_ids: torch.Tensor,
                        word_ids_batch: List[List[Optional[int]]],
                        word_is_prep_batch: List[List[bool]]) -> torch.Tensor:
    """Build a (B, L) bool mask that is True on EVERY subword of a preposition
    WORD (covers multi-piece prepositions like وكان/فقال that the fast single-id
    path misses). Falls back gracefully when word_ids are unavailable."""
    prep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b, (wids, is_prep) in enumerate(zip(word_ids_batch, word_is_prep_batch)):
        for pos, wid in enumerate(wids):
            if wid is not None and wid < len(is_prep) and is_prep[wid]:
                prep_mask[b, pos] = True
    return prep_mask


# --------------------------------------------------------------------------- #
# Batching                                                                    #
# --------------------------------------------------------------------------- #
class Batcher:
    """Tokenizes passages once, yields padded batches with the per-token
    metadata the augmenter needs (special-token mask, preposition word mask)."""

    def __init__(self, passages: List[str], tokenizer, max_length: int,
                 batch_size: int, seed: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.prep_words = set(PREPOSITIONS)
        self.rng = random.Random(seed)

        # Pre-tokenize each passage at the WORD level so we can flag preposition
        # words and align subwords to words (for multi-piece preposition hiding).
        self.examples = []
        for text in passages:
            words = text.split()
            if not words:
                continue
            is_prep = [w in self.prep_words for w in words]
            enc = tokenizer(words, is_split_into_words=True,
                            truncation=True, max_length=max_length,
                            add_special_tokens=True)
            wids = enc.word_ids()
            self.examples.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "word_ids": wids,
                "word_is_prep": is_prep,
            })

    def __len__(self):
        return len(self.examples)

    def epoch(self, shuffle: bool = True):
        idx = list(range(len(self.examples)))
        if shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            chunk = idx[start:start + self.batch_size]
            yield self._collate([self.examples[i] for i in chunk])

    def _collate(self, batch):
        maxlen = max(len(ex["input_ids"]) for ex in batch)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attn, word_ids_b, prep_b = [], [], [], []
        for ex in batch:
            ids = ex["input_ids"]
            am = ex["attention_mask"]
            n_pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * n_pad)
            attn.append(am + [0] * n_pad)
            word_ids_b.append(ex["word_ids"] + [None] * n_pad)
            prep_b.append(ex["word_is_prep"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "word_ids": word_ids_b,
            "word_is_prep": prep_b,
        }


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #
def _normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def byol_token_loss(online: OnlineNet, target: TargetNet,
                    view_a, view_b, attn, valid) -> torch.Tensor:
    """Symmetrized token-level BYOL loss over aligned valid positions.

    loss = 0.5 * ( D(pred_online(A), sg target(B)) + D(pred_online(B), sg target(A)) )
    with D(p, z) = 2 - 2 * cos(p, z) applied per valid token and averaged.
    """
    valid_flat = valid.reshape(-1)
    if valid_flat.sum() == 0:
        # Degenerate batch (all special/pad) — return a zero that still has grad.
        return online.project(online.represent(view_a, attn)).sum() * 0.0

    # ---- Online forward on BOTH views (with gradient) ----
    ha = online.represent(view_a, attn)
    hb = online.represent(view_b, attn)
    pa = online.predict(online.project(ha))   # predictor output, view A
    pb = online.predict(online.project(hb))   # predictor output, view B

    # ---- Target forward on BOTH views (no grad, stop-gradient) ----
    with torch.no_grad():
        za = target(view_a, attn)             # target projection, view A
        zb = target(view_b, attn)             # target projection, view B

    D = pa.shape[-1]
    pa = _normalize(pa.reshape(-1, D)[valid_flat])
    pb = _normalize(pb.reshape(-1, D)[valid_flat])
    za = _normalize(za.reshape(-1, D)[valid_flat]).detach()
    zb = _normalize(zb.reshape(-1, D)[valid_flat]).detach()

    # online-A predicts target-B ; online-B predicts target-A  (symmetric)
    loss_ab = (2 - 2 * (pa * zb).sum(-1)).mean()
    loss_ba = (2 - 2 * (pb * za).sum(-1)).mean()
    return 0.5 * (loss_ab + loss_ba)


@torch.no_grad()
def online_rep_std(online: OnlineNet, view, attn, valid) -> float:
    """Mean per-dimension std of the L2-normalized ONLINE projection over valid
    tokens. The collapse guard's health signal: ->0 means collapse."""
    valid_flat = valid.reshape(-1)
    if valid_flat.sum() < 2:
        return float("nan")
    h = online.represent(view, attn)
    z = _normalize(online.project(h))
    D = z.shape[-1]
    z = z.reshape(-1, D)[valid_flat]
    return z.std(dim=0).mean().item()


# --------------------------------------------------------------------------- #
# EMA schedule                                                                #
# --------------------------------------------------------------------------- #
def ema_decay_at(step: int, total_steps: int, base_decay: float,
                 ramp: bool) -> float:
    """BYOL's target decay. Constant `base_decay` by default; with `ramp`, follow
    Grill Eq. 8: tau = 1 - (1 - tau_base) * (cos(pi*k/K) + 1) / 2, i.e. the decay
    starts smaller and rises toward 1 over training."""
    if not ramp or total_steps <= 1:
        return base_decay
    cos = math.cos(math.pi * step / total_steps)
    return 1.0 - (1.0 - base_decay) * (cos + 1.0) / 2.0


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train(args) -> dict:
    set_seed(args.seed)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported()
                and not args.no_bf16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise SystemExit("tokenizer has no [MASK] token; BYOL masking needs one")

    passages = load_corpus(args.corpus)
    if len(passages) < args.batch_size:
        raise SystemExit(
            f"corpus has only {len(passages)} usable passages, need >= "
            f"--batch-size ({args.batch_size})")
    print(f"[byol] {len(passages)} passages | device={device} bf16={use_bf16} "
          f"| model={args.model_name}")

    batcher = Batcher(passages, tokenizer, args.max_length, args.batch_size,
                      args.seed)

    online = OnlineNet(args.model_name, args.proj_hidden, args.proj_dim).to(device)
    target = TargetNet(online).to(device)
    online.train()
    target.eval()

    # Special-token ids that must never be masked (CLS/SEP/PAD/UNK-as-special).
    special_ids = set(tokenizer.all_special_ids)
    single_prep_ids = preposition_token_ids(tokenizer)

    steps_per_epoch = math.ceil(len(batcher) / 1)  # batcher.epoch yields batches
    total_steps = args.max_steps if args.max_steps > 0 else \
        args.epochs * math.ceil(len(batcher) / args.batch_size)
    total_steps = max(total_steps, 1)

    optim = torch.optim.AdamW(
        [p for p in online.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    aug_rng = torch.Generator()  # CPU generator for the augmentation
    aug_rng.manual_seed(args.seed + 1)

    history = []          # (step, loss, std) triples
    first_std = None
    step = 0
    done = False
    for epoch in range(args.epochs):
        for batch in batcher.epoch(shuffle=True):
            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            # special-token mask (CLS/SEP/PAD/...): never masked.
            special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for sid in special_ids:
                special_mask |= (input_ids == sid)

            # preposition mask: single-piece fast path (by id) OR any subword of
            # a multi-piece preposition WORD (by word_ids alignment).
            prep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for pid in single_prep_ids:
                prep_mask |= (input_ids == pid)
            prep_mask |= word_level_prep_mask(
                tokenizer, input_ids, batch["word_ids"],
                batch["word_is_prep"]).to(device)

            view_a, view_b, valid = build_views(
                input_ids, attn, special_mask, prep_mask, mask_id,
                args.mask_prob, aug_rng)

            optim.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = byol_token_loss(online, target, view_a, view_b,
                                          attn, valid)
            else:
                loss = byol_token_loss(online, target, view_a, view_b, attn,
                                      valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in online.parameters() if p.requires_grad],
                args.clip_norm)
            optim.step()
            sched.step()

            # ---- EMA target update AFTER the optimizer step (BYOL order) ----
            m = ema_decay_at(step, total_steps, args.ema_decay, args.ema_ramp)
            target.update_ema(online, m)

            if step % args.log_every == 0 or step == total_steps - 1:
                std = online_rep_std(online, view_a, attn, valid)
                if first_std is None and not math.isnan(std):
                    first_std = std
                warn = ""
                if not math.isnan(std):
                    if std < args.collapse_std:
                        warn = "  *** COLLAPSE WARNING: std < %.1e ***" % args.collapse_std
                    elif first_std and std < 0.25 * first_std:
                        warn = ("  *** COLLAPSE WARNING: std fell to %.0f%% of "
                                "initial ***" % (100.0 * std / first_std))
                history.append((step, float(loss.item()), float(std)))
                print("[byol] step %5d/%d  loss %.5f  online_std %.5f  ema %.5f  "
                      "lr %.2e%s" % (step, total_steps, loss.item(), std, m,
                                     sched.get_last_lr()[0], warn))
            step += 1
        if done or (args.max_steps > 0 and step >= args.max_steps):
            break

    # ---- Save the ONLINE encoder in HF format (projector/predictor dropped) --
    os.makedirs(args.out, exist_ok=True)
    online.encoder.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    meta = {
        "byol": True,
        "base_model": args.model_name,
        "steps_trained": step,
        "final_loss": history[-1][1] if history else None,
        "final_online_std": history[-1][2] if history else None,
        "first_online_std": first_std,
        "ema_decay": args.ema_decay,
        "ema_ramp": args.ema_ramp,
        "mask_prob": args.mask_prob,
        "prepositions": PREPOSITIONS,
        "proj_hidden": args.proj_hidden,
        "proj_dim": args.proj_dim,
    }
    with open(os.path.join(args.out, "byol_pretrain_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[byol] saved online encoder + tokenizer -> %s" % args.out)
    print("[byol] fine-tune with:  python train_encoder.py --task NoPnx-NP "
          "--model-name %s --out-dir runs/nopnx-np-byol" % args.out)

    return {"history": history, "meta": meta, "out": args.out}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True,
                    help="UNLABELED text corpus: one passage per line, or JSONL "
                         "with a 'text' field")
    ap.add_argument("--out", required=True,
                    help="output dir for the pretrained online encoder (HF format)")
    ap.add_argument("--model-name", default=DEFAULT_MODEL,
                    help="base encoder to pretrain (default AraBERTv02)")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0,
                    help=">0 caps the number of optimizer steps (for smoke runs)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    # BYOL target EMA
    ap.add_argument("--ema-decay", type=float, default=0.996,
                    help="base EMA decay for the target network (Grill 0.996)")
    ap.add_argument("--ema-ramp", action="store_true",
                    help="ramp the EMA decay toward 1 over training (Grill Eq. 8)")
    # Augmentation
    ap.add_argument("--mask-prob", type=float, default=0.15,
                    help="random token-masking probability for both views")
    # Projector / predictor MLP shape (Linear->BN->ReLU->Linear)
    ap.add_argument("--proj-hidden", type=int, default=2048)
    ap.add_argument("--proj-dim", type=int, default=256)
    # Collapse guard / logging
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--collapse-std", type=float, default=1e-3,
                    help="warn if online-projection std drops below this")
    # Runtime
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast")
    return ap


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
