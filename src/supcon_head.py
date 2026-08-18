"""Supervised-contrastive projection head + token-level SupCon adapter.

Transferred from HobbitLong/SupContrast (Khosla et al., NeurIPS 2020),
vendored verbatim at vendor/SupContrast/losses.py. This module adds only the
glue needed to apply that loss at the TOKEN level for our binary
boundary-after-token task; the contrastive math itself is the authors' code,
imported unchanged.

Wiring (train_encoder.py):
  * `ProjectionHead` maps each token's encoder hidden state -> a small
    L2-normalized embedding (the SimCLR/SupCon "projection network g(.)",
    authors' networks/resnet_big.py `head='mlp'` design: Linear -> ReLU ->
    Linear, then F.normalize). Applied to the LAST-subword position of each
    word (the same grid the CE boundary loss uses).
  * `supcon_token_loss` selects the valid (label != -100) token embeddings,
    balances the ~10% boundary / ~90% non-boundary split by capping the
    non-boundary anchors (hard negatives = non-boundary tokens ADJACENT to a
    true boundary are kept first), and calls the vendored `SupConLoss` with
    n_views=1 and the boundary/non-boundary LABEL as the contrastive class.

Everything here is inert unless train_encoder.py is run with --supcon-weight>0.
Labels come only from the training batch (train-only by construction).
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

# Import the AUTHORS' SupConLoss verbatim from the vendored clone. We do not
# re-implement the contrastive math; only the token-level plumbing lives here.
_VENDOR = os.path.join(os.path.dirname(__file__), os.pardir, "vendor", "SupContrast")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from losses import SupConLoss  # noqa: E402  (vendored, Khosla et al. 2020)


class ProjectionHead(nn.Module):
    """Small MLP projection head g(.) -> L2-normalized embedding.

    Mirrors SupContrast's `networks/resnet_big.py` head='mlp':
        Linear(hidden -> hidden) -> ReLU -> Linear(hidden -> proj_dim) -> L2 norm
    Input:  [..., hidden_size] encoder hidden states.
    Output: [..., proj_dim] unit-norm embeddings.
    """

    def __init__(self, hidden_size: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.net(h)
        return F.normalize(z, dim=-1)


def _balanced_token_index(labels_flat: torch.Tensor,
                          adj_flat: torch.Tensor,
                          neg_cap_ratio: float,
                          generator: torch.Generator | None) -> torch.Tensor:
    """Pick which valid tokens become contrastive samples.

    labels_flat: 1-D long tensor of 0/1 boundary labels for VALID tokens only.
    adj_flat   : 1-D bool tensor, True where a valid token is adjacent (within
                 the window) to a true boundary token -> a HARD NEGATIVE.
    Keeps ALL boundary tokens; caps non-boundary tokens at
    neg_cap_ratio * (#boundary), filling hard negatives first, then random
    other non-boundary tokens. Returns indices into labels_flat.
    """
    device = labels_flat.device
    pos_idx = (labels_flat == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels_flat == 0).nonzero(as_tuple=True)[0]
    n_pos = int(pos_idx.numel())
    if n_pos == 0:
        # No boundary anchors in this batch: contrastive loss is undefined
        # (no positive pairs across classes). Signal caller to skip.
        return pos_idx  # empty
    cap = int(round(neg_cap_ratio * n_pos))
    if neg_idx.numel() <= cap:
        keep_neg = neg_idx
    else:
        # hard negatives first (adjacent to a boundary), then random fill
        is_hard = adj_flat[neg_idx]
        hard = neg_idx[is_hard]
        rest = neg_idx[~is_hard]
        if hard.numel() >= cap:
            perm = torch.randperm(hard.numel(), generator=generator).to(device)
            keep_neg = hard[perm[:cap]]
        else:
            need = cap - int(hard.numel())
            perm = torch.randperm(rest.numel(), generator=generator).to(device)
            keep_neg = torch.cat([hard, rest[perm[:need]]])
    return torch.cat([pos_idx, keep_neg])


def supcon_token_loss(hidden_states: torch.Tensor,
                      labels: torch.Tensor,
                      head: ProjectionHead,
                      loss_fn: SupConLoss,
                      neg_cap_ratio: float = 3.0,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Token-level supervised contrastive loss on boundary vs non-boundary.

    hidden_states: [B, T, H] encoder last hidden states.
    labels       : [B, T] long, 0/1 boundary labels, -100 = ignore (subwords
                   that are not a word's last subword, padding, [PAR]).
    Returns a scalar loss (0-dim tensor). Returns a differentiable zero when
    the batch has no boundary token (no cross-class positives possible).
    """
    B, T, H = hidden_states.shape
    valid = labels != -100                              # [B, T]
    # adjacency: a valid token next to a boundary token (within the window)
    is_b = (labels == 1)
    adj = torch.zeros_like(is_b)
    adj[:, :-1] |= is_b[:, 1:]
    adj[:, 1:] |= is_b[:, :-1]
    adj = adj & valid & (labels == 0)                   # hard negatives only

    lab_flat = labels[valid]                            # [N]
    if lab_flat.numel() == 0:
        return hidden_states.sum() * 0.0
    h_flat = hidden_states[valid]                       # [N, H]
    adj_flat = adj[valid]                               # [N] bool

    sel = _balanced_token_index(lab_flat, adj_flat, neg_cap_ratio, generator)
    if sel.numel() == 0 or int((lab_flat[sel] == 1).sum()) == 0:
        # no boundary anchor survived -> no positive pairs -> skip (grad-safe 0)
        return hidden_states.sum() * 0.0

    if sel.numel() > 512:                               # bound the M x M similarity
        perm = torch.randperm(sel.numel(), generator=generator).to(sel.device)
        sel = sel[perm[:512]]                           # matrix -> avoid CUDA OOM
        if int((lab_flat[sel] == 1).sum()) == 0:
            return hidden_states.sum() * 0.0            # unlucky: no positive left

    z = head(h_flat[sel])                               # [M, proj_dim], unit norm
    feats = z.unsqueeze(1)                              # [M, n_views=1, proj_dim]
    lbl = lab_flat[sel]                                 # [M]
    return loss_fn(feats, labels=lbl)


def triplet_token_loss(hidden_states: torch.Tensor,
                       labels: torch.Tensor,
                       head: ProjectionHead,
                       margin: float = 0.3,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Token-level TRIPLET margin loss (metric-learning geometry, distinct from
    SupCon's InfoNCE). Anchor = a boundary token; positive = a DIFFERENT boundary
    token; negative = a non-boundary token, HARD negatives (adjacent to a true
    boundary -> the preposition/clause over-fire sites) sampled first. Pushes each
    boundary embedding closer to another boundary than to a nearby non-boundary by
    `margin`, in the same L2-normalized projection space. Label-only (no augmented
    views). Returns a differentiable zero when a batch lacks a boundary pair."""
    valid = labels != -100
    is_b = (labels == 1)
    adj = torch.zeros_like(is_b)
    adj[:, :-1] |= is_b[:, 1:]
    adj[:, 1:] |= is_b[:, :-1]
    adj = adj & valid & (labels == 0)                   # hard negatives = adjacent
    lab = labels[valid]
    if lab.numel() == 0:
        return hidden_states.sum() * 0.0
    h = hidden_states[valid]
    adjf = adj[valid]
    pos_i = (lab == 1).nonzero(as_tuple=True)[0]
    neg_i = (lab == 0).nonzero(as_tuple=True)[0]
    if pos_i.numel() < 2 or neg_i.numel() == 0:
        return hidden_states.sum() * 0.0                # need >=2 boundaries + a negative
    if pos_i.numel() > 256:                             # cap for memory
        pos_i = pos_i[torch.randperm(pos_i.numel(), generator=generator)[:256].to(pos_i.device)]
    za = head(h[pos_i])                                 # anchors [P, d], unit norm
    zp = torch.roll(za, 1, 0)                           # positive = a different boundary
    hard = neg_i[adjf[neg_i]]                           # adjacent non-boundary = hard neg
    pool = hard if hard.numel() >= pos_i.numel() else neg_i
    negsel = pool[torch.randperm(pool.numel(), generator=generator)[:pos_i.numel()].to(pool.device)]
    zn = head(h[negsel])                                # negatives [P, d]
    d_pos = (za - zp).pow(2).sum(-1)                    # squared euclidean (unit-norm => 2-2cos)
    d_neg = (za - zn).pow(2).sum(-1)
    return torch.clamp(d_pos - d_neg + margin, min=0.0).mean()
