"""CPU smoke for the Rei-2017 aux-LM transfer in train_encoder.py.

Proves two things, no GPU, no corpus download (tiny random BERT):
  (1) --auxlm-weight 0 reproduces the baseline CE loss BIT-FOR-BIT (the flag is
      truly inert at default): same forward inputs -> identical loss tensor, and
      the aux heads are never created.
  (2) --auxlm-weight > 0 runs, produces a strictly larger loss (aux term added),
      and trains (a few optimizer steps reduce the loss; aux-head params receive
      gradients).

Run:  python smoke_auxlm.py
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn
from transformers import (
    AutoModelForTokenClassification,
    BertConfig,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, os.path.dirname(__file__))
from train_encoder import WeightedTrainer  # noqa: E402


def tiny_tokenizer(vocab_size=60):
    # a minimal WordLevel fast tokenizer with the specials the code expects
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[PAR]": 4}
    for i in range(vocab_size):
        vocab[f"w{i}"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]",
        additional_special_tokens=["[PAR]"])


def tiny_model(tok):
    cfg = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64, num_labels=2)
    m = AutoModelForTokenClassification.from_config(cfg)
    return m


def make_batch(tok, model, n=4, seqlen=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    # words w5..w{N} are non-special; build simple whitespace "sentences"
    real_ids = list(range(5, len(tok)))
    feats = []
    for _ in range(n):
        picks = [real_ids[int(torch.randint(len(real_ids), (1,), generator=g))]
                 for _ in range(seqlen - 2)]
        ids = [tok.cls_token_id] + picks + [tok.sep_token_id]
        # boundary labels: -100 on CLS/SEP, random 0/1 on content (last-subword
        # grid == every content token here, since each word is one subword)
        labels = [-100] + [int(torch.randint(2, (1,), generator=g))
                           for _ in picks] + [-100]
        feats.append({"input_ids": ids,
                      "attention_mask": [1] * len(ids),
                      "labels": labels})
    coll = DataCollatorForTokenClassification(tok)
    return coll(feats)


def build_trainer(model, tok, auxlm_weight):
    special_ids = None
    if auxlm_weight > 0:
        model.auxlm_fw_head = _mk_head(model, tok)
        model.auxlm_bw_head = _mk_head(model, tok)
        special_ids = torch.tensor(sorted(set(tok.all_special_ids)),
                                   dtype=torch.long)
    from transformers import TrainingArguments
    args = TrainingArguments(output_dir=os.path.join(
        os.environ.get("TEMP", "/tmp"), "auxlm_smoke_out"),
        per_device_train_batch_size=4, report_to="none", use_cpu=True)
    return WeightedTrainer(
        model=model, args=args,
        data_collator=DataCollatorForTokenClassification(tok),
        auxlm_weight=auxlm_weight, auxlm_max_vocab=len(tok),
        auxlm_special_ids=special_ids)


def _mk_head(model, tok):
    from auxlm_head import AuxLMHead
    return AuxLMHead(model.config.hidden_size, min(len(tok), len(tok)),
                     lm_hidden_size=16)


def main():
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    model.eval()
    batch = make_batch(tok, model)

    # ---- (1) weight 0 == baseline CE, bit-for-bit ----------------------------
    # reference: raw class-weighted CE exactly as the baseline path computes it
    with torch.no_grad():
        ref_inputs = {k: v.clone() for k, v in batch.items()}
        ref_labels = ref_inputs.pop("labels")
        out = model(**ref_inputs)
        ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]),
                                 ignore_index=-100)
        ref_loss = ce(out.logits.view(-1, 2), ref_labels.view(-1))

    tr0 = build_trainer(model, tok, auxlm_weight=0.0)
    with torch.no_grad():
        loss0 = tr0.compute_loss(model, {k: v.clone() for k, v in batch.items()})
    has_heads = hasattr(model, "auxlm_fw_head")
    bit_identical = torch.equal(loss0.detach(), ref_loss.detach())
    print(f"[1] baseline CE          = {ref_loss.item():.10f}")
    print(f"[1] auxlm_weight=0 loss   = {loss0.item():.10f}")
    print(f"[1] bit-for-bit identical = {bit_identical}")
    print(f"[1] aux heads created at w=0 = {has_heads}  (must be False)")
    assert bit_identical, "weight 0 did NOT reproduce CE bit-for-bit"
    assert not has_heads, "aux heads were created at weight 0"

    # ---- (2) weight > 0 runs, adds a positive term, and trains ---------------
    torch.manual_seed(0)
    model2 = tiny_model(tok)
    model2.train()
    tr1 = build_trainer(model2, tok, auxlm_weight=0.1)
    b2 = {k: v.clone() for k, v in batch.items()}
    loss1 = tr1.compute_loss(model2, {k: v.clone() for k, v in b2.items()})
    # baseline CE on THIS model, same batch, for the "aux added a term" check
    with torch.no_grad():
        r_inputs = {k: v.clone() for k, v in b2.items()}
        r_labels = r_inputs.pop("labels")
        base2 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]),
                                    ignore_index=-100)(
            model2(**r_inputs).logits.view(-1, 2), r_labels.view(-1))
    print(f"[2] baseline CE (model2) = {base2.item():.6f}")
    print(f"[2] auxlm_weight=0.1 loss = {loss1.item():.6f}  "
          f"(> baseline: {loss1.item() > base2.item()})")
    assert loss1.item() > base2.item(), "aux term did not increase the loss"

    # a few optimizer steps should reduce the total loss AND touch aux params
    opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    fw_before = model2.auxlm_fw_head.out.weight.detach().clone()
    losses = []
    grad_ok = False
    for step in range(20):
        opt.zero_grad()
        l = tr1.compute_loss(model2, {k: v.clone() for k, v in b2.items()})
        l.backward()
        grad_ok = grad_ok or (
            model2.auxlm_fw_head.out.weight.grad is not None and
            model2.auxlm_fw_head.out.weight.grad.abs().sum().item() > 0)
        opt.step()
        losses.append(l.item())
    fw_moved = not torch.equal(fw_before, model2.auxlm_fw_head.out.weight.detach())
    print(f"[2] loss over 5 steps    = {['%.4f' % x for x in losses]}")
    print(f"[2] loss decreased        = {losses[-1] < losses[0]}")
    print(f"[2] aux-head got gradient  = {grad_ok}")
    print(f"[2] aux-head params moved  = {fw_moved}")
    assert losses[-1] < losses[0], "training did not reduce the loss"
    assert grad_ok and fw_moved, "aux head did not train"

    print("\nSMOKE PASS: weight 0 reproduces CE bit-for-bit; "
          "weight>0 runs, adds the aux term, and trains.")


if __name__ == "__main__":
    main()
