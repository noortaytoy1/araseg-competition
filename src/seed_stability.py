"""Seed-stability regularizers for small-data fine-tuning (transferred).

Two independent tricks from Zhang et al., "Revisiting Few-sample BERT
Fine-tuning" (ICLR 2021), cloned at vendor/revisit-bert-finetuning
(asappresearch/revisit-bert-finetuning, run_glue.py):

  * `reinit_top_layers(model, K)` — RE-INITIALIZE the top K transformer blocks
    (closest to the output) before fine-tuning. The paper's finding: the top
    pretrained layers are specialized to the pretraining objective and transfer
    poorly; re-initializing them stabilizes and speeds few-sample fine-tuning.
    Weights are drawn N(0, config.initializer_range); LayerNorm is reset to
    (bias 0, weight 1); Linear biases to 0 — exactly run_glue.py's `reinit_layers`
    block (lines 748-764), generalized from `getattr(model, args.model_type)` to
    `getattr(model, model.base_model_prefix)` so it also covers AraELECTRA
    ("electra") and ARBERTv2 ("bert").

  * `llrd_param_groups(model, base_lr, decay, weight_decay)` — LAYER-WISE LR
    DECAY: the embeddings + each encoder layer get their own learning rate,
    decayed by `decay` per layer going from the output down to the input
    (top layer = base_lr*decay, next = base_lr*decay**2, ...). This is
    run_glue.py's `get_optimizer_grouped_parameters` (lines 90-119), verbatim
    grouping logic (no_decay = bias/LayerNorm.weight; classifier at base_lr).

The debiased-Adam knob from the same paper ("BERTAdam omits the bias
correction; restoring it helps small data") is HuggingFace's default: torch/HF
`AdamW` applies Adam's bias correction (the paper's BERTAdam == correct_bias
False; the fix is correct_bias True, which is the default here). We therefore
rely on that default rather than re-adding a flag, and "train longer" is the
existing --epochs. See the report for the exact mapping.

Everything here is a no-op unless train_encoder.py is run with
--reinit-top-layers>0 or --llrd-decay!=1.0. Uses train data only.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn


def _base_encoder(model: nn.Module):
    """The pretrained encoder submodule (model.bert / model.electra / ...).

    Mirrors run_glue.py's `getattr(model, args.model_type)` but derived
    generically from HuggingFace's `base_model_prefix`, so one code path covers
    AraBERT/ARBERTv2 (prefix 'bert') and AraELECTRA (prefix 'electra')."""
    return getattr(model, model.base_model_prefix)


def reinit_top_layers(model: nn.Module, k: int) -> int:
    """Re-initialize the top `k` transformer encoder blocks in place.

    Verbatim port of asappresearch/revisit-bert-finetuning run_glue.py lines
    748-764 (the `reinit_layers` block for bert/roberta/electra). Returns the
    number of layers actually re-initialized (clamped to the model depth).
    `k<=0` is a no-op and returns 0 (baseline preserved bit-for-bit).
    """
    if k <= 0:
        return 0
    encoder = _base_encoder(model)
    layers = encoder.encoder.layer                       # BERT/ELECTRA layout
    n = len(layers)
    k = min(k, n)
    std = encoder.config.initializer_range
    for layer in layers[-k:]:
        for module in layer.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses
                # truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    return k


def llrd_param_groups(model: nn.Module, base_lr: float, decay: float,
                      weight_decay: float) -> List[dict]:
    """Layer-wise-learning-rate-decay optimizer parameter groups.

    Verbatim grouping logic from asappresearch/revisit-bert-finetuning
    run_glue.py `get_optimizer_grouped_parameters` (lines 90-119):

      * classifier/pooler params -> base_lr, weight_decay 0
      * embeddings, then encoder layers reversed (output -> input): each block's
        lr is multiplied by `decay` cumulatively; within a block, params split
        into decay / no-decay (no_decay = bias, LayerNorm.weight).

    `decay == 1.0` should not call this (caller keeps the HF default optimizer
    so the baseline stays bit-for-bit); this function assumes decay != 1.0.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": base_lr,
        },
    ]
    encoder = _base_encoder(model)
    layers = [encoder.embeddings] + list(encoder.encoder.layer)
    layers.reverse()                                     # output side first
    lr = base_lr
    for layer in layers:
        lr *= decay
        groups += [
            {
                "params": [p for n, p in layer.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return groups
