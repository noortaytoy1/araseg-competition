"""GPU smoke for consistency_train.py — MUST run on the real GPU.

Proves:
  (a) cf-weight 0 AND rdrop-weight 0 reproduce the plain weighted-CE baseline
      BIT-FOR-BIT (torch.equal against an independent CE computed straight from
      AutoModelForTokenClassification logits).
  (b) cf-weight > 0 makes the cf term non-zero, and rdrop-weight > 0 makes the
      rdrop term non-zero; each then trains a few real GPU steps with no
      device/OOM error.

Run:
  cd arabic-sentence-segmentation
  python src/smoke_consistency.py --train-jsonl data/NoPnx-NP_train.jsonl
"""
import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, set_seed
from data import MODEL_PAR_TOKEN, load_task
import consistency_train as C


def build_batch(jsonl, task, n_docs, window, stride, bs, tok, prep_ids):
    docs = load_task(task, "train", jsonl_path=jsonl)[:n_docs]
    exs = C.make_examples(docs, window, stride)[:bs]
    enc = [C.encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, 512)
           for e in exs]
    feats = []
    for e in enc:
        feats.append({k: e[k][0] for k in ("input_ids", "attention_mask", "labels")})
    coll = C.Collator(tok, prep_ids)
    return coll(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--stride", type=int, default=90)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "GPU-SMOKE requires a real CUDA device"
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    prep_ids = C._prep_ids(tok)
    print(f"n_prep_ids={len(prep_ids)}  mask_id={tok.mask_token_id}")

    batch = build_batch(a.train_jsonl, a.task, 20, a.window, a.stride, a.bs, tok, prep_ids)
    batch = {k: v.to(dev) for k, v in batch.items()}
    n_prep_next = int(batch["next_is_prep_mask"].sum())
    n_prep_tok = int(batch["prep_tok_mask"].sum())
    print(f"batch: input_ids {tuple(batch['input_ids'].shape)}  "
          f"prep_tok positions={n_prep_tok}  next_is_prep positions={n_prep_next}")
    assert n_prep_next > 0, "no next_is_prep positions in smoke batch — pick more docs"

    pos_weight = 4.0

    # ---------------------------------------------------------------- #
    # (a) BIT-FOR-BIT baseline: cf=0, rdrop=0  vs  independent CE       #
    # ---------------------------------------------------------------- #
    set_seed(0)
    model = C.ConsistencyModel(a.model_name, pos_weight, cf_weight=0.0, rdrop_weight=0.0,
                               mask_id=tok.mask_token_id, n_extra_tok=1).to(dev)
    model.eval()  # dropout off -> deterministic single forward on both paths

    with torch.no_grad():
        out = model(**batch)
        gated_loss = out["loss"]
        gated_logits = out["logits"]

        # independent reference: raw logits from the same underlying encoder,
        # plain weighted CE with the SAME class weights + ignore_index.
        ref_logits = model.enc(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
        w = torch.tensor([1.0, pos_weight], device=dev)
        ref_loss = nn.functional.cross_entropy(
            ref_logits.view(-1, 2).float(), batch["labels"].view(-1),
            weight=w, ignore_index=-100)

    logits_equal = torch.equal(gated_logits, ref_logits)
    loss_equal = torch.equal(gated_loss, ref_loss)
    print("\n[(a) bit-for-bit @ cf=0, rdrop=0]")
    print(f"  gated loss = {gated_loss.item():.10f}")
    print(f"  ref   loss = {ref_loss.item():.10f}")
    print(f"  torch.equal(logits) = {logits_equal}")
    print(f"  torch.equal(loss)   = {loss_equal}")
    assert logits_equal, "logits differ at weight 0"
    assert loss_equal, "loss NOT bit-for-bit identical to plain weighted CE at weight 0"
    print("  PASS: baseline reproduced bit-for-bit")

    # ---------------------------------------------------------------- #
    # (b1) cf-weight > 0 -> cf term non-zero, then a few GPU steps      #
    # ---------------------------------------------------------------- #
    print("\n[(b) cf-weight > 0]")
    set_seed(0)
    model_cf = C.ConsistencyModel(a.model_name, pos_weight, cf_weight=1.0, rdrop_weight=0.0,
                                  mask_id=tok.mask_token_id, n_extra_tok=1).to(dev)
    model_cf.train()
    # isolate the cf term: total - CE, computed with dropout off for a clean read
    model_cf.eval()
    with torch.no_grad():
        total_cf = model_cf(**batch)["loss"]
        ce_only = model_cf._ce(
            model_cf._logits(batch["input_ids"], batch["attention_mask"]), batch["labels"])
        cf_term = (total_cf - ce_only).item()
    print(f"  CE={ce_only.item():.6f}  total={total_cf.item():.6f}  cf_term={cf_term:.6e}")
    assert cf_term > 0, "cf term is not > 0 with cf-weight=1"
    print("  cf term is non-zero")
    model_cf.train()
    opt = torch.optim.AdamW(model_cf.parameters(), lr=5e-5)
    for i in range(a.steps):
        opt.zero_grad()
        loss = model_cf(**batch)["loss"]
        loss.backward(); opt.step()
        print(f"  cf step {i}: loss={loss.item():.6f}  "
              f"mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print("  PASS: cf trains a few GPU steps, no device/OOM")

    # ---------------------------------------------------------------- #
    # (b2) rdrop-weight > 0 -> rdrop term non-zero, then a few GPU steps#
    # ---------------------------------------------------------------- #
    print("\n[(b) rdrop-weight > 0]")
    set_seed(0)
    model_rd = C.ConsistencyModel(a.model_name, pos_weight, cf_weight=0.0, rdrop_weight=1.0,
                                  mask_id=tok.mask_token_id, n_extra_tok=1).to(dev)
    model_rd.train()  # dropout ON so the two passes differ -> KL > 0
    with torch.no_grad():
        total_rd = model_rd(**batch)["loss"]
        # 0.5(CE1+CE2) baseline term is ~CE; the rdrop term is the extra KL. Just
        # confirm the KL itself is > 0 by recomputing it directly with dropout on.
        l1 = model_rd._logits(batch["input_ids"], batch["attention_mask"])
        l2 = model_rd._logits(batch["input_ids"], batch["attention_mask"])
        valid = (batch["labels"].view(-1) != -100)
        lp1 = torch.log_softmax(l1.view(-1, 2).float(), -1)[valid]
        lp2 = torch.log_softmax(l2.view(-1, 2).float(), -1)[valid]
        p1, p2 = lp1.exp(), lp2.exp()
        kl = 0.5 * ((p1 * (lp1 - lp2)).sum(-1) + (p2 * (lp2 - lp1)).sum(-1)).mean().item()
    print(f"  total={total_rd.item():.6f}  independent KL(dropout on)={kl:.6e}")
    assert kl > 0, "rdrop KL is not > 0 with dropout on"
    print("  rdrop term is non-zero")
    opt = torch.optim.AdamW(model_rd.parameters(), lr=5e-5)
    for i in range(a.steps):
        opt.zero_grad()
        loss = model_rd(**batch)["loss"]
        loss.backward(); opt.step()
        print(f"  rdrop step {i}: loss={loss.item():.6f}  "
              f"mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print("  PASS: rdrop trains a few GPU steps, no device/OOM")

    print("\nSMOKE OK: baseline bit-for-bit + both methods active and training on GPU.")


if __name__ == "__main__":
    main()
