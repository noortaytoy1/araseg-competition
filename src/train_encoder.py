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
import json
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
from dice_loss import DiceLoss  # ShannonAI/dice_loss_for_NLP, verbatim (Apache-2.0)
from supcon_head import (  # vendored HobbitLong/SupContrast (Khosla et al. 2020)
    ProjectionHead,
    SupConLoss,
    supcon_token_loss,
    triplet_token_loss,
)
from auxlm_head import (  # Rei ACL-2017 lmcost, transferred (marekrei/sequence-labeler)
    AuxLMHead,
    auxlm_token_loss,
    build_neighbor_targets,
)
from seed_stability import (  # Zhang et al. ICLR-2021, transferred (revisit-bert-finetuning)
    llrd_param_groups,
    reinit_top_layers,
)


_PA_OF = {"NP": "PA", "NoPnx-NP": "NoPnx-PA", "PA": "PA", "NoPnx-PA": "NoPnx-PA"}


def paragraph_word_labels(task: str, split: str, jsonl_path: str = None) -> dict:
    """Per-word coarse paragraph-boundary labels for the HIERARCHICAL aux head.

    Word i -> 1 if it is immediately followed by a paragraph break, else 0.
    Derived from the paragraph-aware counterpart of `task` (PA for NP, NoPnx-PA
    for NoPnx-NP), whose token stream equals this task's stream with `\\n`
    paragraph tokens inserted (verified: PA − \\n == NP, 174/174 docs). Only
    meaningful for the paragraph-LESS tracks, where the model cannot see the
    structure it is asked to predict."""
    pa_task = _PA_OF[task]
    path = jsonl_path or os.path.join(os.path.dirname(__file__), os.pardir,
                                      "data", f"{pa_task}_{split}.jsonl")
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        toks = d["tokens"]
        wl = []
        for j, w in enumerate(toks):
            if w == PARAGRAPH_TOKEN:
                continue
            nxt = toks[j + 1] if j + 1 < len(toks) else None
            wl.append(1 if nxt == PARAGRAPH_TOKEN else 0)
        out[d["doc_id"]] = wl
    return out


def load_pos(task: str, split: str) -> dict:
    """Load cached per-token POS tags (camel_tools CALIMA-MSA) for the MORPHOLOGICAL
    aux head. features/pos_<task>_<split>.json = {doc_id: [pos_str per token]}.
    Returns None if not built (aux then simply off)."""
    p = os.path.join(os.path.dirname(__file__), os.pardir, "features",
                     f"pos_{task}_{split}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def build_pos_vocab(pos_dict: dict) -> dict:
    """Deterministic POS-tag -> id map. 0=<pad>/ignore, 1=<unk> (dev tags unseen in
    train), 2.. = sorted train tags."""
    tags = sorted({t for v in pos_dict.values() for t in v})
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in tags:
        vocab[t] = len(vocab)
    return vocab


def make_windows(docs: List[dict], window: int, stride: int,
                 para: dict = None, pos: dict = None) -> List[dict]:
    """Cut each doc into overlapping word windows of `window` words.
    If `para` (doc_id -> per-word paragraph labels) is given, carry aligned
    paragraph-boundary labels for the hierarchical aux head."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labels = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        pw = None
        if para is not None:
            pl = para.get(d["doc_id"])
            # para labels are in no-\n word order; align to words, -100 at [PAR]
            pw, k = [], 0
            for w in words:
                if w == MODEL_PAR_TOKEN or pl is None or k >= len(pl):
                    pw.append(-100)
                    if w != MODEL_PAR_TOKEN and pl is not None:
                        k += 1
                else:
                    pw.append(pl[k]); k += 1
        # POS ids (already vocab-mapped) are aligned 1:1 to d["tokens"]; mask [PAR]
        pos_w = None
        if pos is not None:
            pi = pos.get(d["doc_id"])
            if pi is not None:
                pos_w = [(-100 if w == MODEL_PAR_TOKEN else pi[j])
                         for j, w in enumerate(words)]
        n = len(words)
        start = 0
        while start < n:
            end = min(start + window, n)
            ex = {"words": words[start:end], "word_labels": labels[start:end]}
            if pw is not None:
                ex["para_word_labels"] = pw[start:end]
            if pos_w is not None:
                ex["pos_word_labels"] = pos_w[start:end]
            out.append(ex)
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
    def align(word_labels, word_ids):
        labels = [-100] * len(word_ids)
        for pos in range(len(word_ids)):
            wid = word_ids[pos]
            if wid is None:
                continue
            nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
            if nxt != wid:            # label the LAST subword of each word
                labels[pos] = word_labels[wid]
        return labels

    all_labels, all_para, all_pos = [], [], []
    has_para = "para_word_labels" in examples
    has_pos = "pos_word_labels" in examples
    for i, word_labels in enumerate(examples["word_labels"]):
        word_ids = enc.word_ids(batch_index=i)
        all_labels.append(align(word_labels, word_ids))
        if has_para:
            all_para.append(align(examples["para_word_labels"][i], word_ids))
        if has_pos:
            all_pos.append(align(examples["pos_word_labels"][i], word_ids))
    enc["labels"] = all_labels
    if has_para:
        enc["para_labels"] = all_para
    if has_pos:
        enc["pos_labels"] = all_pos
    return enc


class ParaCollator:
    """Wrap the token-classification collator; right-pad the extra aux label fields
    (para_labels, pos_labels) with -100 to the batch's label length, the same way."""

    def __init__(self, tokenizer):
        self.base = DataCollatorForTokenClassification(tokenizer)

    def __call__(self, features):
        aux_keys = [k for k in ("para_labels", "pos_labels") if k in features[0]]
        if not aux_keys:                        # eval batches carry no aux labels
            return self.base(features)
        popped = {k: [list(f.pop(k)) for f in features] for k in aux_keys}
        batch = self.base(features)
        m = batch["labels"].shape[1]
        for k, rows in popped.items():
            batch[k] = torch.tensor(
                [p + [-100] * (m - len(p)) for p in rows], dtype=torch.long)
        return batch


class WeightedTrainer(Trainer):
    """Class-weighted CE with optional boundary label smoothing and FGM.

    boundary_smooth: tokens adjacent to a gold boundary get soft target
    `smooth_eps` for the boundary class (boundary-smoothing trick, 1-D).
    fgm_eps: one-step FGM adversarial perturbation on the word embeddings.
    """

    def __init__(self, *args, pos_weight: float = 1.0, smooth_eps: float = 0.0,
                 fgm_eps: float = 0.0, soft_f1: float = 0.0, para_aux: float = 0.0,
                 pos_aux: float = 0.0,
                 dice_alpha: float = -1.0, supcon_weight: float = 0.0,
                 supcon_temp: float = 0.07, supcon_neg_cap: float = 3.0,
                 supcon_seed: int = 42, triplet_weight: float = 0.0,
                 triplet_margin: float = 0.3, rank_margin_weight: float = 0.0,
                 rank_margin: float = 2.0, rank_hard_k: int = 10,
                 rank_neg_only: bool = False,
                 auxlm_weight: float = 0.0,
                 auxlm_max_vocab: int = 7500, auxlm_special_ids=None,
                 r3f_weight: float = 0.0, r3f_eps: float = 1e-5,
                 r3f_noise: str = "uniform", llrd_decay: float = 1.0,
                 llrd_lr: float = 5e-5, llrd_weight_decay: float = 0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight]), ignore_index=-100
        )
        self._pos_weight = pos_weight
        self._smooth_eps = smooth_eps
        self._fgm_eps = fgm_eps
        self._soft_f1 = soft_f1
        self._para_aux = para_aux
        self._para_loss = nn.CrossEntropyLoss(ignore_index=-100)
        # MORPHOLOGICAL POS aux (train-only multi-task; makes the encoder POS-aware
        # without changing the inference interface). Inert at weight 0.
        self._pos_aux = pos_aux
        self._pos_loss = nn.CrossEntropyLoss(ignore_index=-100)
        # supervised contrastive aux (Khosla et al. 2020). Inert at weight 0:
        # the SupConLoss/head are built but never touched below unless >0.
        self._supcon_weight = supcon_weight
        self._supcon_neg_cap = supcon_neg_cap
        self._supcon = SupConLoss(temperature=supcon_temp,
                                  base_temperature=supcon_temp) \
            if supcon_weight > 0 else None
        # deterministic negative-sampling RNG (CPU generator; index ops only)
        self._supcon_gen = torch.Generator().manual_seed(supcon_seed) \
            if supcon_weight > 0 else None
        # TRIPLET metric-margin aux (label-only, distinct geometry from SupCon).
        # Inert at weight 0 (head built below only when >0; baseline bit-identical).
        self._triplet_weight = triplet_weight
        self._triplet_margin = triplet_margin
        # PAIRWISE RANKING MARGIN on the boundary log-odds (audit Fix 2). Inert at 0.
        self._rank_margin_weight = rank_margin_weight
        self._rank_margin = rank_margin
        self._rank_hard_k = rank_hard_k
        self._rank_neg_only = rank_neg_only
        self._triplet_gen = torch.Generator().manual_seed(supcon_seed + 1) \
            if triplet_weight > 0 else None
        # AUXILIARY LANGUAGE-MODEL aux (Rei ACL 2017, transferred). Inert at
        # weight 0: heads/state are built but never touched below unless >0.
        self._auxlm_weight = auxlm_weight
        self._auxlm_max_vocab = auxlm_max_vocab
        self._auxlm_special_ids = (auxlm_special_ids
                                   if auxlm_special_ids is not None
                                   else torch.tensor([], dtype=torch.long))
        # real Dice loss (ShannonAI), NER-SOTA config; dice_alpha<0 = off
        self._dice = None
        if dice_alpha is not None and dice_alpha >= 0:
            self._dice = DiceLoss(with_logits=True, smooth=1.0,
                                  square_denominator=True, ohem_ratio=0.0,
                                  alpha=dice_alpha, reduction="mean",
                                  index_label_position=True)
        # R3F consistency regularizer (Aghajanyan et al. ACL 2021). Inert at
        # weight 0: the noise sampler is built but never used below unless >0.
        # Matches fairseq examples/rxf sentence_prediction_r3f.py: Normal(0,eps)
        # or Uniform(-eps,eps) noise added to the input embeddings.
        self._r3f_weight = r3f_weight
        self._r3f_eps = r3f_eps
        self._r3f_sampler = None
        if r3f_weight > 0:
            if r3f_noise == "normal":
                self._r3f_sampler = torch.distributions.normal.Normal(
                    loc=0.0, scale=r3f_eps)
            else:
                self._r3f_sampler = torch.distributions.uniform.Uniform(
                    low=-r3f_eps, high=r3f_eps)
        # layer-wise LR decay (Zhang et al. ICLR 2021). decay==1.0 => off: the
        # HF-default optimizer is kept so the baseline stays bit-for-bit (see
        # create_optimizer below). Only consulted there.
        self._llrd_decay = llrd_decay
        self._llrd_lr = llrd_lr
        self._llrd_weight_decay = llrd_weight_decay

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        para_labels = inputs.pop("para_labels", None)
        pos_labels = inputs.pop("pos_labels", None)
        # All three auxiliaries below are TRAIN-ONLY and are the only consumers of
        # output_hidden_states. Gate the flag on model.training so hidden_states
        # are NEVER requested at eval: at eval HF Trainer's prediction path returns
        # every non-loss output as `logits`, so a leaked hidden_states tensor turns
        # `logits` into a tuple (crashing token_f1_metrics) AND accumulates the
        # whole-dev hidden states across batches (CUDA OOM). Requesting them only
        # during training fixes both. At weight 0 this is a no-op (all three False),
        # so the baseline stays bit-for-bit.
        need_hidden = model.training and (
            (self._para_aux > 0) or (self._supcon_weight > 0)
            or (self._triplet_weight > 0) or (self._auxlm_weight > 0)
            or (self._pos_aux > 0))
        outputs = model(**inputs, output_hidden_states=need_hidden)
        logits = outputs.logits
        if self._dice is not None:
            # authors use BINARY dice (num_classes=1, sigmoid) for token labeling,
            # not softmax-over-2. Feed the boundary log-odds as a single logit.
            valid = labels.view(-1) != -100
            if valid.any():
                bl = (logits[..., 1] - logits[..., 0]).reshape(-1)[valid].view(-1, 1)
                ft = labels.view(-1)[valid].float()
                loss = self._dice(bl, ft)
            else:
                loss = logits.sum() * 0.0
        elif self._smooth_eps > 0:
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
        if self._rank_margin_weight > 0 and model.training:
            # PAIRWISE RANKING MARGIN aux (audit Fix 2): every gold boundary's score
            # must exceed the HARDEST non-boundary scores (the confident false cuts,
            # median p=0.998) by a margin, per window. Directly attacks confident
            # over-firing that no threshold/decode can move. Additive — weight 0 = off.
            s = (logits[..., 1] - logits[..., 0]).float()      # boundary log-odds [B,T]
            mask_v = labels != -100
            rank_losses = []
            for b in range(s.size(0)):
                sb, lb = s[b][mask_v[b]], labels[b][mask_v[b]]
                pos_s = sb[lb == 1]
                neg_s = sb[lb == 0]
                if pos_s.numel() == 0 or neg_s.numel() == 0:
                    continue
                k = min(int(self._rank_hard_k), neg_s.numel())
                hard_neg = neg_s.topk(k).values                # hardest (most FP-like)
                # hinge over all pos x hard-neg pairs in this window
                anchor = pos_s.detach() if self._rank_neg_only else pos_s
                # neg-only: freeze the boundary side -> gradient ONLY pushes the
                # confident fakes DOWN (precision-targeted; symmetric variant
                # measured to inflate recall at the cost of precision)
                gap = anchor.unsqueeze(1) - hard_neg.unsqueeze(0)
                rank_losses.append(torch.clamp(self._rank_margin - gap, min=0).mean())
            if rank_losses:
                loss = loss + self._rank_margin_weight * torch.stack(rank_losses).mean()
        if self._soft_f1 > 0:
            # macro (per-window) soft-F1 surrogate on the boundary class: closes
            # the train(BCE)/eval(boundary-F1) gap by optimizing a differentiable
            # F1 directly. TP/FP/FN are computed from soft boundary probabilities;
            # averaged over windows (macro), added to the base loss. Additive —
            # soft_f1=0 leaves the loss above bit-identical to the CE/smooth path.
            mask = (labels != -100).float()
            hard = labels.clamp(min=0).float()
            p = torch.softmax(logits, dim=-1)[..., 1]
            tp = (p * hard * mask).sum(dim=1)
            fp = (p * (1.0 - hard) * mask).sum(dim=1)
            fn = ((1.0 - p) * hard * mask).sum(dim=1)
            soft_f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-6)
            # macro over boundary-containing windows only (F1 is undefined for an
            # all-negative window; the weighted CE already suppresses its FPs).
            has_pos = (hard * mask).sum(dim=1) > 0
            if has_pos.any():
                loss = loss + self._soft_f1 * (1.0 - soft_f1[has_pos].mean())
        if self._para_aux > 0 and para_labels is not None and model.training:
            # HIERARCHICAL aux: predict coarse paragraph boundaries from the
            # shared encoder (a separate head), regularizing toward document
            # structure. Additive — para_aux=0 skips this entirely.
            head = model.para_head if hasattr(model, "para_head") else \
                self.accelerator.unwrap_model(model).para_head
            h = outputs.hidden_states[-1]
            para_logits = head(h)
            para_loss = self._para_loss(
                para_logits.view(-1, 2).float(), para_labels.view(-1))
            loss = loss + self._para_aux * para_loss
        if self._pos_aux > 0 and pos_labels is not None and model.training:
            # MORPHOLOGICAL aux: predict each word's POS (camel_tools CALIMA) from
            # the shared encoder, so the representation encodes "verb/noun/prep/..."
            # -> generalizes past specific tokens (attacks the unseen-bigram wall).
            # Additive — pos_aux=0 skips this entirely. Inference interface unchanged.
            phead = model.pos_head if hasattr(model, "pos_head") else \
                self.accelerator.unwrap_model(model).pos_head
            h = outputs.hidden_states[-1]
            pos_logits = phead(h)
            pos_loss = self._pos_loss(
                pos_logits.view(-1, pos_logits.size(-1)).float(), pos_labels.view(-1))
            loss = loss + self._pos_aux * pos_loss
        if self._supcon_weight > 0 and model.training:
            # SUPERVISED CONTRASTIVE aux (Khosla et al. 2020, vendored verbatim):
            # training-only: eval needs only the boundary head, and the M x M
            # similarity on the doubled eval batch OOMs. Skipping on eval also
            # keeps the reported dev metric a pure boundary-F1.
            # project each token's last-layer hidden state to an L2-normalized
            # embedding and pull boundary-token embeddings together / push them
            # apart from non-boundary tokens, using the boundary LABEL as the
            # contrastive class. Non-boundary anchors are capped (hard negatives
            # = tokens adjacent to a true boundary kept first) to handle the
            # ~10% imbalance. Labels come only from the training batch. Additive
            # — supcon_weight=0 skips this entirely (baseline is bit-identical).
            head = model.supcon_head if hasattr(model, "supcon_head") else \
                self.accelerator.unwrap_model(model).supcon_head
            h = outputs.hidden_states[-1]
            sc_loss = supcon_token_loss(
                h, labels, head, self._supcon,
                neg_cap_ratio=self._supcon_neg_cap,
                generator=self._supcon_gen)
            loss = loss + self._supcon_weight * sc_loss
        if self._triplet_weight > 0 and model.training:
            # TRIPLET metric-margin aux (training-only; own projection head).
            # additive — triplet_weight=0 skips this entirely (baseline identical).
            thead = model.triplet_head if hasattr(model, "triplet_head") else \
                self.accelerator.unwrap_model(model).triplet_head
            h = outputs.hidden_states[-1]
            tr_loss = triplet_token_loss(
                h, labels, thead, margin=self._triplet_margin,
                generator=self._triplet_gen)
            loss = loss + self._triplet_weight * tr_loss
        if self._auxlm_weight > 0 and model.training:
            # AUXILIARY LANGUAGE-MODEL aux (Rei ACL 2017, "lmcost", transferred):
            # from each word's last-subword encoder state, predict the NEXT and
            # PREVIOUS word's id through two SEPARATE tanh->vocab heads (not the
            # tagger head), summed CE, capped vocab. This is the paper's
            # "separate" forward/backward LM objective, adapted from the authors'
            # bi-LSTM to our BERT grid. Targets come only from the batch's own
            # input_ids (train-only). Additive — auxlm_weight=0 skips this
            # entirely (baseline is bit-identical).
            fw_head = model.auxlm_fw_head if hasattr(model, "auxlm_fw_head") \
                else self.accelerator.unwrap_model(model).auxlm_fw_head
            bw_head = model.auxlm_bw_head if hasattr(model, "auxlm_bw_head") \
                else self.accelerator.unwrap_model(model).auxlm_bw_head
            h = outputs.hidden_states[-1]
            fw_t, bw_t = build_neighbor_targets(
                inputs["input_ids"], labels,
                self._auxlm_special_ids.to(labels.device))
            lm_loss = auxlm_token_loss(
                h, fw_t, bw_t, fw_head, bw_head, self._auxlm_max_vocab)
            loss = loss + self._auxlm_weight * lm_loss
        if self._r3f_weight > 0 and model.training and "input_ids" in inputs:
            # R3F consistency regularizer (Aghajanyan et al. ACL 2021, "Better
            # Fine-Tuning by Reducing Representational Collapse"), transferred
            # from fairseq examples/rxf sentence_prediction_r3f.py. Add small
            # noise to the INPUT EMBEDDINGS and penalize the SYMMETRIC KL between
            # the clean and noised output distributions — a data-free consistency
            # term. Additive — r3f_weight=0 skips this entirely (baseline exact).
            emb_layer = model.get_input_embeddings()
            token_embeddings = emb_layer(inputs["input_ids"])
            noise = self._r3f_sampler.sample(
                sample_shape=token_embeddings.shape).to(token_embeddings)
            noised_embeddings = token_embeddings.detach().clone() + noise
            noise_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
            noised_logits = model(inputs_embeds=noised_embeddings,
                                  **noise_inputs).logits
            # symmetric KL over VALID token positions only (subwords/pad/[PAR]
            # carry -100 and must not contribute to the consistency target).
            valid = (labels.view(-1) != -100)
            cl = logits.reshape(-1, logits.size(-1))[valid]
            nl = noised_logits.reshape(-1, noised_logits.size(-1))[valid]
            if cl.size(0) > 0:
                symm_kl = self._symm_kl(nl, cl)
                loss = loss + self._r3f_weight * symm_kl
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _symm_kl(noised_logits, input_logits):
        """Symmetric KL between clean and noised output distributions.
        Verbatim from fairseq sentence_prediction_r3f.py `_get_symm_kl`:
        KL(noised||clean)+KL(clean||noised), summed then divided by rows."""
        return (
            nn.functional.kl_div(
                nn.functional.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                nn.functional.softmax(input_logits, dim=-1, dtype=torch.float32),
                None, None, "sum",
            )
            + nn.functional.kl_div(
                nn.functional.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                nn.functional.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None, None, "sum",
            )
        ) / noised_logits.size(0)

    def create_optimizer(self):
        """Layer-wise-LR-decay optimizer (Zhang et al. ICLR 2021) when enabled.

        llrd_decay == 1.0 (default) -> defer to HuggingFace's optimizer so the
        baseline is bit-for-bit. Otherwise build the authors' per-layer grouped
        params (revisit-bert-finetuning get_optimizer_grouped_parameters) using
        the SAME optimizer class/kwargs the Trainer would use, so only the
        per-group lr/weight_decay differ. The Trainer still owns the scheduler."""
        if self._llrd_decay == 1.0 or self.optimizer is not None:
            return super().create_optimizer()
        opt_model = self.model
        grouped = llrd_param_groups(opt_model, self._llrd_lr,
                                    self._llrd_decay, self._llrd_weight_decay)
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model)
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        # per-group lr/weight_decay come from `grouped`; drop the scalar
        # weight_decay default so it can't override our per-group values.
        optimizer_kwargs.pop("weight_decay", None)
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        return self.optimizer

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
    # Defense-in-depth: if any model output beyond `logits` leaks into the eval
    # predictions (e.g. hidden_states from an aux head), HF Trainer hands us a
    # tuple here. Keep only the boundary logits (element 0) before argmax.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
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
    ap.add_argument("--expect-docs", type=int, default=None,
                    help="if set, ASSERT the train split loaded exactly this many "
                         "documents (tripwire against a clobbered / wrong "
                         "--train-jsonl; A/B runners should pass the frozen doc "
                         "count). Off by default = baseline behaviour unchanged.")
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
    ap.add_argument("--soft-f1", type=float, default=0.0,
                    help="weight on the macro (per-window) soft-F1 surrogate loss "
                         "added to CE (0=off; targets the train-BCE/eval-F1 gap)")
    ap.add_argument("--para-aux", type=float, default=0.0,
                    help="weight on the hierarchical paragraph-boundary auxiliary "
                         "head (0=off; NP/NoPnx-NP only, derives coarse paragraph "
                         "labels from the paragraph-aware counterpart)")
    ap.add_argument("--pos-aux", type=float, default=0.0,
                    help="weight on the MORPHOLOGICAL POS auxiliary head (0=off): "
                         "predicts each word's camel_tools POS from the shared "
                         "encoder (train-only; needs features/pos_<task>_train.json). "
                         "Inference interface unchanged.")
    ap.add_argument("--dice-loss", type=float, default=-1.0,
                    help="use the authors' Dice loss (ShannonAI) as the main loss; "
                         "value = alpha (NER-SOTA=0.01); <0 = off (keep CE)")
    ap.add_argument("--supcon-weight", type=float, default=0.0,
                    help="weight on the token-level SUPERVISED CONTRASTIVE aux "
                         "loss (Khosla et al. 2020) added to CE (0=off, baseline "
                         "exact). Clusters boundary-token embeddings vs "
                         "non-boundary; train labels only")
    ap.add_argument("--supcon-proj-dim", type=int, default=128,
                    help="projection-head output dim for --supcon-weight")
    ap.add_argument("--supcon-temp", type=float, default=0.07,
                    help="SupCon temperature (Khosla et al. default 0.07)")
    ap.add_argument("--supcon-neg-cap", type=float, default=3.0,
                    help="cap non-boundary contrastive anchors at this multiple "
                         "of the boundary count (hard negatives kept first)")
    ap.add_argument("--triplet-weight", type=float, default=0.0,
                    help="weight on the token-level TRIPLET metric-margin aux loss "
                         "(0=off, baseline exact). Boundary anchor closer to another "
                         "boundary than to a hard (adjacent) non-boundary; train labels only")
    ap.add_argument("--triplet-margin", type=float, default=0.3,
                    help="margin for --triplet-weight (squared-euclidean, unit-norm space)")
    ap.add_argument("--rank-margin-weight", type=float, default=0.0,
                    help="weight on the PAIRWISE RANKING MARGIN aux (0=off, baseline exact): "
                         "each gold boundary's log-odds must beat the window's hardest "
                         "non-boundaries by --rank-margin (targets confident false cuts)")
    ap.add_argument("--rank-margin", type=float, default=2.0,
                    help="log-odds margin for --rank-margin-weight")
    ap.add_argument("--rank-hard-k", type=int, default=10,
                    help="number of hardest negatives per window for the ranking margin")
    ap.add_argument("--rank-neg-only", action="store_true",
                    help="one-sided ranking margin: detach the boundary side so the "
                         "gradient only pushes confident non-boundaries DOWN "
                         "(precision-targeted; default two-sided)")
    ap.add_argument("--auxlm-weight", type=float, default=0.0,
                    help="weight (gamma) on the AUXILIARY LANGUAGE-MODEL aux loss "
                         "(Rei ACL 2017 'lmcost', transferred) added to CE (0=off, "
                         "baseline exact). From each word's encoder state, predicts "
                         "the next+previous word id via separate heads; train text "
                         "only. Authors' default gamma=0.1")
    ap.add_argument("--auxlm-max-vocab", type=int, default=7500,
                    help="max vocab for the aux-LM softmax; rarer word ids collapse "
                         "to one bucket (authors' lmcost_max_vocab_size=7500)")
    ap.add_argument("--auxlm-hidden", type=int, default=50,
                    help="hidden size of the aux-LM MLP (authors' "
                         "lmcost_hidden_layer_size=50)")
    # ---- SEED-STABILITY REGULARIZATION (small-data fine-tuning) ----
    # Zhang et al. ICLR 2021 (revisit-bert-finetuning) + Aghajanyan et al.
    # ACL 2021 (R3F). All three default to OFF => baseline bit-for-bit.
    ap.add_argument("--reinit-top-layers", type=int, default=0,
                    help="RE-INITIALIZE the top K transformer encoder layers "
                         "before fine-tuning (Zhang et al. ICLR 2021; 0=off, "
                         "baseline exact). Stabilizes few-sample fine-tuning by "
                         "discarding pretraining-specialized top layers")
    ap.add_argument("--llrd-decay", type=float, default=1.0,
                    help="LAYER-WISE LR DECAY factor (Zhang et al. ICLR 2021; "
                         "1.0=off, baseline exact). e.g. 0.9 => each layer from "
                         "output to input gets lr*0.9**depth; embeddings decayed "
                         "most. Debiased Adam is HF's default (correct_bias)")
    ap.add_argument("--r3f-weight", type=float, default=0.0,
                    help="weight (lambda) on the R3F symmetric-KL consistency "
                         "regularizer (Aghajanyan et al. ACL 2021; 0=off, "
                         "baseline exact). Penalizes divergence between clean and "
                         "embedding-noised output distributions; NO extra data")
    ap.add_argument("--r3f-eps", type=float, default=1e-5,
                    help="R3F input-embedding noise scale (authors' default 1e-5)")
    ap.add_argument("--r3f-noise", choices=["uniform", "normal"],
                    default="uniform",
                    help="R3F noise distribution (authors' default: uniform "
                         "[-eps,eps]; 'normal' = N(0,eps))")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-bottom-layers", type=int, default=0,
                    help="freeze embeddings + bottom N encoder layers (big models on a 32GB GPU)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing (memory saver for big models like XLM-R-XL)")
    ap.add_argument("--save-only-model", action="store_true",
                    help="write only model weights in each checkpoint, not optimizer/scheduler "
                         "state (~3x smaller ckpts). Training math and load_best_model_at_end "
                         "selection are unchanged; only mid-run resumability is lost. Default off.")
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="checkpoints to keep; <=0 keeps ALL epochs (for post-hoc best-epoch selection on the real metric)")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="allow custom model code (e.g. NeoBERT/NeoAraBERT)")
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)

    train_docs = load_task(args.task, "train", jsonl_path=args.train_jsonl)
    dev_docs = load_task(args.task, "dev", jsonl_path=args.dev_jsonl)

    # Doc-count tripwire: always PRINT how many train docs loaded so a clobbered
    # or wrong --train-jsonl leaves a trace in the log; ASSERT when --expect-docs
    # is given (A/B runners pass the frozen count). Off by default -> unchanged.
    print(f"loaded train docs={len(train_docs)}  from={args.train_jsonl or '(default)'}")
    if args.expect_docs is not None and len(train_docs) != args.expect_docs:
        raise SystemExit(
            f"--expect-docs {args.expect_docs} but loaded {len(train_docs)} train "
            f"docs from {args.train_jsonl!r} -> ABORT (data clobber / wrong path)")

    pos = sum(sum(d["labels"]) for d in train_docs)
    tot = sum(len(d["labels"]) for d in train_docs)
    pos_weight = args.pos_weight if args.pos_weight is not None else min((tot - pos) / max(pos, 1), 8.0)
    print(f"train boundary rate={pos / tot:.3f}  pos_weight={pos_weight:.2f}")

    from transformers import AutoConfig
    import torch as _torch
    trc = args.trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=trc)
    tokenizer.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    _mk = {"num_labels": 2}
    if trc:
        _mk["trust_remote_code"] = True
    try:  # ModernBERT decorates ModernBertEmbeddings with @torch.compile(dynamic=True) AT IMPORT TIME.
          # This box's torch._inductor is internally broken (duplicate flex_attention template registration
          # + missing persistent_grid symbol), so ANY compile probe crashes model load. Neuter torch.compile
          # to an identity for modernbert only -> the layer runs eager (identical math, just no fusion).
        if getattr(AutoConfig.from_pretrained(args.model_name, trust_remote_code=trc), "model_type", "") == "modernbert":
            _mk["reference_compile"] = False
            def _noc(model=None, *a, **k):
                return model if callable(model) else (lambda f: f)
            _torch.compile = _noc
    except Exception:
        pass
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, **_mk)
    model.resize_token_embeddings(len(tokenizer))

    if args.freeze_bottom_layers > 0:
        # Big-model memory saver: freeze embeddings + bottom N encoder layers so
        # only the top layers + classifier head get grads/optimizer state. Frozen
        # params (requires_grad=False) receive no grad -> AdamW allocates no state
        # for them, so this is what makes a 3.5B model fit a 32GB card.
        base = model.base_model
        for p in base.embeddings.parameters():
            p.requires_grad = False
        enc_layers = base.encoder.layer
        n = min(args.freeze_bottom_layers, len(enc_layers))
        for layer in enc_layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False
        ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze embeddings + bottom {n}/{len(enc_layers)} layers -> "
              f"trainable={ntr/1e6:.1f}M params")
    if args.grad_checkpoint:
        model.config.use_cache = False
        # with frozen bottom layers, checkpointing needs the embedding output to
        # require grad or backprop through the trainable top layers fails
        model.enable_input_require_grads()

    if args.reinit_top_layers > 0:
        # SEED-STABILITY: re-initialize the top K encoder layers (Zhang et al.
        # ICLR 2021). Done here, before the optimizer/heads are built, exactly
        # as the authors do it prior to fine-tuning. Off by default (K=0).
        k = reinit_top_layers(model, args.reinit_top_layers)
        print(f"re-initialized top {k} transformer layer(s) "
              f"(std={model.config.initializer_range}; Zhang et al. ICLR 2021)")

    para_tr = None
    if args.para_aux > 0:
        if args.task not in ("NP", "NoPnx-NP"):
            raise SystemExit("--para-aux only applies to NP / NoPnx-NP (the "
                             "paragraph-less tracks)")
        para_tr = paragraph_word_labels(args.task, "train")   # aux is train-only
        model.para_head = nn.Linear(model.config.hidden_size, 2)
        print(f"hierarchical paragraph-aux head on (weight {args.para_aux}); "
              f"para docs train={len(para_tr)}")

    pos_tr = None
    if args.pos_aux > 0:
        pos_raw = load_pos(args.task, "train")
        if pos_raw is None:
            raise SystemExit(f"--pos-aux needs features/pos_{args.task}_train.json "
                             "(run the POS tagger first)")
        pos_vocab = build_pos_vocab(pos_raw)
        pos_tr = {did: [pos_vocab.get(t, 1) for t in tags]
                  for did, tags in pos_raw.items()}
        model.pos_head = nn.Linear(model.config.hidden_size, len(pos_vocab))
        print(f"morphological POS-aux head on (weight {args.pos_aux}); "
              f"num_pos={len(pos_vocab)}, docs={len(pos_tr)}")

    if args.supcon_weight > 0:
        # projection head g(.) attached to the model so the Trainer optimizer
        # picks up its parameters (same pattern as para_head above).
        model.supcon_head = ProjectionHead(model.config.hidden_size,
                                           proj_dim=args.supcon_proj_dim)
        print(f"supervised-contrastive aux head on (weight {args.supcon_weight}; "
              f"proj_dim={args.supcon_proj_dim}, temp={args.supcon_temp}, "
              f"neg_cap={args.supcon_neg_cap})")

    if args.triplet_weight > 0:
        # own projection head g(.) for the triplet metric loss (same attach pattern)
        model.triplet_head = ProjectionHead(model.config.hidden_size,
                                            proj_dim=args.supcon_proj_dim)
        print(f"triplet metric-margin aux head on (weight {args.triplet_weight}; "
              f"margin={args.triplet_margin}, proj_dim={args.supcon_proj_dim})")

    auxlm_special_ids = None
    if args.auxlm_weight > 0:
        # two SEPARATE forward/backward LM heads attached to the model so the
        # Trainer optimizer picks up their params (same pattern as supcon_head).
        cap = min(args.auxlm_max_vocab, len(tokenizer))
        model.auxlm_fw_head = AuxLMHead(model.config.hidden_size, cap,
                                        lm_hidden_size=args.auxlm_hidden)
        model.auxlm_bw_head = AuxLMHead(model.config.hidden_size, cap,
                                        lm_hidden_size=args.auxlm_hidden)
        # never predict special/pad tokens as a "neighbour word"
        auxlm_special_ids = torch.tensor(
            sorted(set(tokenizer.all_special_ids)), dtype=torch.long)
        print(f"aux-LM (Rei 2017) head on (weight {args.auxlm_weight}; "
              f"vocab_cap={cap}, hidden={args.auxlm_hidden})")

    from datasets import Dataset

    train_ds = Dataset.from_list(make_windows(train_docs, args.window, args.stride, para_tr, pos_tr))
    dev_ds = Dataset.from_list(make_windows(dev_docs, args.window, args.window))  # no overlap; no aux at eval
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
        # Dice's (1-p)^alpha focal gradient overflows in fp16 -> train it in fp32
        fp16=torch.cuda.is_available() and args.dice_loss < 0,
        logging_steps=20,
        save_total_limit=(None if args.save_total_limit <= 0 else args.save_total_limit),
        save_only_model=args.save_only_model,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=args.grad_checkpoint,
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=(ParaCollator(tokenizer) if (args.para_aux > 0 or args.pos_aux > 0)
                       else DataCollatorForTokenClassification(tokenizer)),
        compute_metrics=token_f1_metrics,
        pos_weight=pos_weight,
        smooth_eps=args.smooth_eps,
        fgm_eps=args.fgm_eps,
        soft_f1=args.soft_f1,
        para_aux=args.para_aux,
        pos_aux=args.pos_aux,
        dice_alpha=args.dice_loss,
        supcon_weight=args.supcon_weight,
        supcon_temp=args.supcon_temp,
        supcon_neg_cap=args.supcon_neg_cap,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        rank_margin_weight=args.rank_margin_weight,
        rank_margin=args.rank_margin,
        rank_hard_k=args.rank_hard_k,
        rank_neg_only=args.rank_neg_only,
        supcon_seed=args.seed,
        auxlm_weight=args.auxlm_weight,
        auxlm_max_vocab=min(args.auxlm_max_vocab, len(tokenizer)),
        auxlm_special_ids=auxlm_special_ids,
        r3f_weight=args.r3f_weight,
        r3f_eps=args.r3f_eps,
        r3f_noise=args.r3f_noise,
        llrd_decay=args.llrd_decay,
        llrd_lr=args.lr,
        llrd_weight_decay=targs.weight_decay,
    )
    if args.r3f_weight > 0:
        print(f"R3F consistency regularizer on (lambda={args.r3f_weight}; "
              f"noise={args.r3f_noise} eps={args.r3f_eps}; Aghajanyan et al. ACL 2021)")
    if args.llrd_decay != 1.0:
        print(f"layer-wise LR decay on (decay={args.llrd_decay}; "
              f"base_lr={args.lr}; Zhang et al. ICLR 2021)")
    trainer.train()
    print("best window-level dev metrics:", trainer.evaluate())

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"saved model -> {args.out_dir}")
    print("next: python predict.py --model", args.out_dir, "--task", args.task,
          "--split dev --out subs/" + args.task + "_dev.csv")


if __name__ == "__main__":
    main()
