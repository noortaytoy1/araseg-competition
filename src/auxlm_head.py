"""Auxiliary language-model head + token-level aux-LM adapter (Rei, ACL 2017).

Transfers the "lmcost" objective from Marek Rei, "Semi-supervised Multitask
Learning for Sequence Labeling" (ACL 2017; github.com/marekrei/sequence-labeler,
commit 00ff795, labeler.py::construct_lmcost / _construct_lmcost). That code is
AGPL-3.0, so we do NOT vendor it verbatim; we re-implement the *formulation*,
adapted from the authors' bi-LSTM tagger to our BERT token-classifier. The math
transferred is the authors' exactly:

  authors' _construct_lmcost(input_tensor):
      h = Dense(lmcost_hidden_layer_size, tanh)(input_tensor)   # separate LM MLP
      logits = Dense(lmcost_max_vocab_size)(h)                  # NOT the tagger head
      loss = sparse_softmax_cross_entropy(logits, target_ids)   # masked, summed
  authors' construct_lmcost(..., "separate"):
      forward state at t  -> predict token at t+1  (next word)
      backward state at t -> predict token at t-1  (previous word)
      end positions masked out via sentence_mask[:,1:] / [:,:-1]
      target_ids capped: ids >= max_vocab-1 collapse to a single bucket
  total loss += lmcost_gamma * (fw_cost + bw_cost)     # gamma default 0.1

BERT adaptation (the only change of substance):
  The authors have distinct forward/backward LSTM hidden states; a BERT encoder
  produces ONE bidirectional hidden state per position. We keep the authors'
  "separate" design of TWO predictors with distinct parameters -- a forward head
  that predicts the NEXT word's id and a backward head that predicts the PREVIOUS
  word's id -- both reading the shared encoder hidden state (the same last-layer
  grid the boundary CE uses, at each word's LAST subword). This preserves the
  paper's objective (predict surrounding tokens jointly with the tagger, from the
  same contextual representation, no extra data) while fitting a transformer.

Everything here is inert unless train_encoder.py is run with --auxlm-weight>0.
Targets/text come only from the training batch's own input (train-only by
construction).
"""
from __future__ import annotations

import torch
from torch import nn


# Sentinel matching the token-classification convention: positions with this
# target are excluded from the aux-LM loss (word endpoints, padding, [PAR],
# non-last subwords -- i.e. anything that is not a real neighbour prediction).
IGNORE = -100


class AuxLMHead(nn.Module):
    """Rei-2017 lmcost predictor: a separate tanh->vocab MLP over hidden states.

    Mirrors the authors' `_construct_lmcost` two-layer design exactly:
        Linear(hidden -> lm_hidden) -> tanh -> Linear(lm_hidden -> capped_vocab)
    One instance per direction (forward = predict next word, backward = predict
    previous word), giving the paper's "separate" forward/backward LM objective
    with distinct parameters -- NOT the tagger's output head.
    """

    def __init__(self, hidden_size: int, vocab_size: int,
                 lm_hidden_size: int = 50):
        super().__init__()
        self.proj = nn.Linear(hidden_size, lm_hidden_size)
        self.out = nn.Linear(lm_hidden_size, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.proj(h)))


def cap_targets(target_ids: torch.Tensor, max_vocab: int) -> torch.Tensor:
    """Collapse rare ids into a single bucket (authors' vocab capping).

    Authors: `target_ids = where(target_ids >= max_vocab-1, max_vocab-1, ...)`.
    Only applied to non-ignore entries so the IGNORE sentinel is preserved.
    """
    keep = target_ids != IGNORE
    capped = torch.clamp(target_ids, max=max_vocab - 1)
    return torch.where(keep, capped, target_ids)


def auxlm_token_loss(hidden_states: torch.Tensor,
                     fw_targets: torch.Tensor,
                     bw_targets: torch.Tensor,
                     fw_head: AuxLMHead,
                     bw_head: AuxLMHead,
                     max_vocab: int) -> torch.Tensor:
    """Rei-2017 "separate" aux-LM loss adapted to a BERT token grid.

    hidden_states : [B, T, H] encoder last hidden states.
    fw_targets    : [B, T] long, id of the NEXT word (first subword), IGNORE=skip.
    bw_targets    : [B, T] long, id of the PREVIOUS word (first subword), IGNORE.
    Returns a scalar (0-dim) loss. Returns a differentiable zero when no valid
    neighbour target exists in the batch. Mean over valid positions (both
    directions pooled) so the weight scale is batch-size independent -- the only
    departure from the authors' `reduce_sum` (they compensate with a per-token
    learning-rate schedule; we keep CE-comparable means like the rest of our
    losses).
    """
    fw_t = cap_targets(fw_targets, max_vocab)
    bw_t = cap_targets(bw_targets, max_vocab)

    fw_logits = fw_head(hidden_states)          # [B, T, V]
    bw_logits = bw_head(hidden_states)          # [B, T, V]
    V = fw_logits.size(-1)

    ce = nn.functional.cross_entropy
    # sum over both directions, divide by total valid count -> mean CE per
    # neighbour prediction. cross_entropy already ignores IGNORE (-100).
    n_fw = int((fw_t != IGNORE).sum())
    n_bw = int((bw_t != IGNORE).sum())
    n = n_fw + n_bw
    if n == 0:
        return hidden_states.sum() * 0.0
    loss_fw = ce(fw_logits.reshape(-1, V), fw_t.reshape(-1),
                 ignore_index=IGNORE, reduction="sum")
    loss_bw = ce(bw_logits.reshape(-1, V), bw_t.reshape(-1),
                 ignore_index=IGNORE, reduction="sum")
    return (loss_fw + loss_bw) / n


def build_neighbor_targets(input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           special_ids: torch.Tensor) -> tuple:
    """Derive next/previous WORD-id targets from the batch itself (train-only).

    We predict a neighbouring *word*, represented by that word's FIRST subword
    id -- the natural word-level target for the authors' word-LM (our tokens are
    words split into subwords). We reuse the boundary-loss grid: `labels != -100`
    marks each word's LAST subword. From that grid:
      * anchor positions = last-subword positions of real words (labels != -100);
      * a word's own id = its FIRST subword id, which is the token right after
        the previous anchor (or sequence start). We recover per-word first-subword
        ids by walking the anchors in order.

    fw_targets[a_k] = first-subword id of word k+1  (next word), IGNORE for last.
    bw_targets[a_k] = first-subword id of word k-1  (prev word), IGNORE for first.
    Positions that are not anchors, or whose neighbour is a special token
    ([PAR]/[CLS]/[SEP]/pad), get IGNORE.

    input_ids   : [B, T] long.
    labels      : [B, T] long, boundary labels with -100 on non-anchor subwords.
    special_ids : 1-D long tensor of ids to never predict (specials + pad).
    Returns (fw_targets, bw_targets), each [B, T] long.
    """
    B, T = input_ids.shape
    device = input_ids.device
    fw = torch.full((B, T), IGNORE, dtype=torch.long, device=device)
    bw = torch.full((B, T), IGNORE, dtype=torch.long, device=device)
    special = set(int(x) for x in special_ids.tolist())

    for b in range(B):
        # anchors = last-subword positions in order; first-subword id of the
        # word ending at anchor a is the token following the previous anchor.
        anchors = (labels[b] != IGNORE).nonzero(as_tuple=True)[0].tolist()
        if not anchors:
            continue
        first_ids, keep = [], []
        prev_anchor = -1
        for a in anchors:
            fid = int(input_ids[b, prev_anchor + 1])   # word's first subword id
            first_ids.append(fid)
            keep.append(fid not in special)
            prev_anchor = a
        for k, a in enumerate(anchors):
            if k + 1 < len(anchors) and keep[k + 1]:
                fw[b, a] = first_ids[k + 1]
            if k - 1 >= 0 and keep[k - 1]:
                bw[b, a] = first_ids[k - 1]
    return fw, bw
