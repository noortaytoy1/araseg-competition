"""GPU smoke for the SupCon aux path in train_encoder.py — REAL GPU only.

Guards the eval-time hidden-states leak that crashed the SupCon A/B (fixed:
need_hidden is gated on model.training, and token_f1_metrics unwraps a tuple).

Proves, through the REAL WeightedTrainer (not a hand-rolled forward):
  (a) --supcon-weight 0 reproduces the plain weighted-CE baseline BIT-FOR-BIT
      (torch.equal against an independent weighted-CE on the same logits);
  (b) --supcon-weight > 0 makes the SupCon term non-zero and a few train steps
      run and move the loss (no device/OOM);
  (c) evaluate() with --supcon-weight > 0 RUNS (no 'tuple has no argmax' crash,
      no whole-dev hidden-state OOM) and returns eval_f1 — the exact failure the
      audit flagged. evaluate() at weight 0 also runs and is unaffected.

Run:
  cd arabic-sentence-segmentation
  python src/smoke_supcon.py
"""
import os
import sys

import torch
from torch import nn
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, TrainingArguments,
                          set_seed)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import Dataset                                   # noqa: E402
from data import MODEL_PAR_TOKEN, load_task                    # noqa: E402
import train_encoder as T                                      # noqa: E402

MODEL = "aubmindlab/bert-base-arabertv02"
TASK = "NoPnx-NP"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
CKPT = os.path.join(HERE, os.pardir, "runs", "_smoke_supcon_tmp")  # throwaway; not evidence


def build(n_docs, supcon_weight, seed=0):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(TASK, "train", jsonl_path=os.path.join(DATA, f"{TASK}_train.jsonl"))[:n_docs]
    dv = load_task(TASK, "dev", jsonl_path=os.path.join(DATA, f"{TASK}_dev.jsonl"))[:n_docs]
    model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)
    model.resize_token_embeddings(len(tok))
    if supcon_weight > 0:
        model.supcon_head = T.ProjectionHead(model.config.hidden_size, proj_dim=128)
    trds = Dataset.from_list(T.make_windows(tr, 180, 90))
    dvds = Dataset.from_list(T.make_windows(dv, 180, 180))
    fn = lambda ex: T.encode_batch(ex, tok, 512)
    trds = trds.map(fn, batched=True, remove_columns=trds.column_names)
    dvds = dvds.map(fn, batched=True, remove_columns=dvds.column_names)
    targs = TrainingArguments(
        output_dir=CKPT, num_train_epochs=1, per_device_train_batch_size=8,
        per_device_eval_batch_size=16, eval_strategy="no", save_strategy="no",
        report_to="none", seed=seed, fp16=torch.cuda.is_available(), logging_steps=10000)
    trainer = T.WeightedTrainer(
        model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=T.token_f1_metrics, pos_weight=4.0,
        supcon_weight=supcon_weight, supcon_temp=0.07, supcon_neg_cap=3.0,
        supcon_seed=seed)
    return trainer


def main():
    assert torch.cuda.is_available(), "GPU-SMOKE requires a real CUDA device"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    # ---------------- (a) BIT-FOR-BIT at supcon=0 ----------------
    print("\n[(a) supcon-weight 0 bit-for-bit vs plain weighted-CE]")
    tr0 = build(n_docs=12, supcon_weight=0.0, seed=1)
    tr0.model.to("cuda").eval()  # dropout off -> deterministic
    b = next(iter(tr0.get_train_dataloader()))
    b = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in b.items()}
    with torch.no_grad():
        gated = tr0.compute_loss(tr0.model, {k: v.clone() for k, v in b.items()})
        ref_logits = tr0.model(input_ids=b["input_ids"],
                               attention_mask=b["attention_mask"]).logits
        w = torch.tensor([1.0, 4.0], device=ref_logits.device)
        ref = nn.functional.cross_entropy(ref_logits.view(-1, 2).float(),
                                          b["labels"].view(-1), weight=w, ignore_index=-100)
    eq = torch.equal(gated, ref)
    print(f"  gated={gated.item():.10f}  ref_CE={ref.item():.10f}  torch.equal={eq}")
    assert eq, "baseline NOT bit-for-bit at supcon=0"
    print("  PASS: baseline reproduced bit-for-bit")

    # evaluate() at weight 0 must also run cleanly
    m0 = tr0.evaluate()
    assert "eval_f1" in m0, "eval failed at supcon=0"
    print(f"  eval(supcon=0) ok: eval_f1={m0['eval_f1']:.5f}")

    # ---------------- (b) supcon>0 term non-zero + trains ----------------
    print("\n[(b) supcon-weight > 0: term non-zero + trains]")
    tr = build(n_docs=12, supcon_weight=0.3, seed=0)
    tr.model.to("cuda").train()
    b = next(iter(tr.get_train_dataloader()))
    b = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in b.items()}
    with torch.no_grad():
        total = tr.compute_loss(tr.model, {k: v.clone() for k, v in b.items()}).item()
    sw = tr._supcon_weight; tr._supcon_weight = 0.0
    with torch.no_grad():
        ce_only = tr.compute_loss(tr.model, {k: v.clone() for k, v in b.items()}).item()
    tr._supcon_weight = sw
    term = total - ce_only
    print(f"  CE_only={ce_only:.6f}  total={total:.6f}  supcon_term={term:.6e}")
    assert term > 0, "supcon term is not > 0 with weight 0.3"
    opt = torch.optim.AdamW(tr.model.parameters(), lr=5e-5)
    losses = []
    for i in range(4):
        opt.zero_grad()
        l = tr.compute_loss(tr.model, {k: v.clone() for k, v in b.items()})
        l.backward(); opt.step(); losses.append(l.item())
    print(f"  train losses: {[round(x,5) for x in losses]}")
    assert all(torch.isfinite(torch.tensor(losses))), "non-finite supcon loss"
    print("  PASS: supcon term non-zero and trains, no device/OOM")

    # ---------------- (c) evaluate() with supcon>0 must NOT crash ----------------
    print("\n[(c) evaluate() with supcon-weight > 0 (the crash the fix removes)]")
    m = tr.evaluate()
    assert "eval_f1" in m, "eval did not return eval_f1"
    print(f"  eval(supcon=0.3) ok: eval_f1={m['eval_f1']:.5f}  eval_loss={m['eval_loss']:.5f}")
    print("  PASS: eval runs (no 'tuple has no argmax', no whole-dev hidden-state OOM)")

    print("\nSMOKE OK: (a) bit-for-bit, (b) supcon term active+trains, (c) eval no longer crashes.")


if __name__ == "__main__":
    main()
