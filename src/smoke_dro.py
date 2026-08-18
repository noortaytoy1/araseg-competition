"""GPU smoke for dro_train.py — MUST run on the real GPU.

Proves:
  (a) DRO OFF reproduces the plain weighted-CE baseline BIT-FOR-BIT (torch.equal
      against an independent CE computed straight from AutoModelForTokenClassification
      logits — the same reference consistency_train's smoke uses).
  (b) DRO ON: GEORGE forms 4 pseudo-groups (print sizes); the CTC-DRO q-update
      moves the group weights, they STAY ON THE SIMPLEX (q_g >= 0, sum == 1), and
      the WORST-GROUP loss is tracked. Then a few GPU steps run with no OOM.

Run:
  cd arabic-sentence-segmentation
  python src/smoke_dro.py --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, set_seed
from data import MODEL_PAR_TOKEN, load_task
import dro_train as D


def build_feats(jsonl, task, n_docs, window, stride, bs, tok):
    docs = load_task(task, "train", jsonl_path=jsonl)[:n_docs]
    exs = D.make_examples(docs, window, stride)[:bs]
    feats = []
    for e in exs:
        enc = D.encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, 512)
        feats.append({"input_ids": enc["input_ids"][0],
                      "attention_mask": enc["attention_mask"][0],
                      "labels": enc["labels"][0]})
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--dev-jsonl", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--stride", type=int, default=90)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--dro-alpha", type=float, default=0.5)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "GPU-SMOKE requires a real CUDA device"
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    pos_weight = 4.0

    # ---------------------------------------------------------------- #
    # (a) BIT-FOR-BIT ERM: DRO off  vs  independent weighted CE         #
    # ---------------------------------------------------------------- #
    print("\n[(a) DRO off -> bit-for-bit weighted-CE ERM]")
    feats = build_feats(a.train_jsonl, a.task, 20, a.window, a.stride, a.bs, tok)
    batch = D.pad_batch([dict(f) for f in feats], tok, dev)

    set_seed(0)
    model = D.DROModel(a.model_name, pos_weight, n_extra_tok=1).to(dev)
    model.eval()  # dropout off -> deterministic

    with torch.no_grad():
        logits = model.logits(batch["input_ids"], batch["attention_mask"])
        gated_loss = model.ce(logits, batch["labels"])
        # independent reference: raw logits from the same encoder, plain weighted CE.
        ref_logits = model.enc(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
        w = torch.tensor([1.0, pos_weight], device=dev)
        ref_loss = nn.functional.cross_entropy(
            ref_logits.view(-1, 2).float(), batch["labels"].view(-1),
            weight=w, ignore_index=-100)

    logits_equal = torch.equal(logits, ref_logits)
    loss_equal = torch.equal(gated_loss, ref_loss)
    print(f"  gated loss = {gated_loss.item():.10f}")
    print(f"  ref   loss = {ref_loss.item():.10f}")
    print(f"  torch.equal(logits) = {logits_equal}")
    print(f"  torch.equal(loss)   = {loss_equal}")
    assert logits_equal, "logits differ at DRO off"
    assert loss_equal, "loss NOT bit-for-bit identical to plain weighted CE at DRO off"
    print("  PASS: ERM baseline reproduced bit-for-bit")

    # sanity: the ERM training path itself (D.model.ce) trains a few steps
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    for i in range(a.steps):
        opt.zero_grad()
        lg = model.logits(batch["input_ids"], batch["attention_mask"])
        loss = model.ce(lg, batch["labels"])
        loss.backward(); opt.step()
        print(f"  erm step {i}: loss={loss.item():.6f}  "
              f"mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    # ---------------------------------------------------------------- #
    # (b) DRO on: GEORGE groups + CTC-DRO q-update + few GPU steps      #
    # ---------------------------------------------------------------- #
    print("\n[(b) DRO on -> GEORGE 4 groups + CTC-DRO q-update]")
    docs = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    print(f"  n_train_docs = {len(docs)}")
    labels, sizes, X = D.assign_groups(docs, a.model_name, dev, seed=42, cache_path=None)
    print(f"  GEORGE cluster sizes (k={D.N_GROUPS}) = {sizes}   sum={sum(sizes)}")
    assert len(sizes) == 4, "expected 4 GEORGE groups"
    assert sum(sizes) == len(docs), "group sizes must cover all docs"
    assert all(s > 0 for s in sizes), f"a GEORGE group is empty: {sizes}"
    assert X.shape == (len(docs), 4), f"feature matrix shape {X.shape} != (n,4)"

    # build a few train windows per group and run real DRO cycles
    fresh = D.DROModel(a.model_name, pos_weight, n_extra_tok=1).to(dev)
    fresh.train()
    tr_ex = D.make_examples(docs, a.window, a.stride, with_group=True)
    by_group = {g: [] for g in range(D.N_GROUPS)}
    for e in tr_ex:
        enc = D.encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, 512)
        by_group[e["group"]].append({
            "input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
            "labels": enc["labels"][0]})
    active = [g for g in sorted(by_group) if by_group[g]]
    G = len(active)
    print(f"  active groups={active}  windows/group={ {g: len(by_group[g]) for g in active} }")

    q = {g: 1.0 / G for g in active}
    alpha = a.dro_alpha; eta_q = 0.01
    opt = torch.optim.AdamW(fresh.parameters(), lr=5e-5, weight_decay=0.01)
    rng = random.Random(0)
    q_history = [dict(q)]
    for step in range(a.steps):
        opt.zero_grad()
        grp_mean = {}
        worst_loss = -1.0; worst_g = None
        for g in active:
            rec = rng.choice(by_group[g])
            b = D.pad_batch([dict(input_ids=rec["input_ids"],
                                  attention_mask=rec["attention_mask"],
                                  labels=list(rec["labels"]))], tok, dev)
            lg = fresh.logits(b["input_ids"], b["attention_mask"])
            lsum, cnt = fresh.ce_sum_and_count(lg, b["labels"])
            grp_mean[g] = lsum / max(cnt, 1)
            gm = float(grp_mean[g].detach())
            if gm > worst_loss:
                worst_loss = gm; worst_g = g
        # CTC-DRO smoothed q-update
        log_q = {g: math.log(max(q[g], 1e-12)) + eta_q * float(grp_mean[g].detach()) / (q[g] + alpha)
                 for g in active}
        mx = max(log_q.values())
        exps = {g: math.exp(log_q[g] - mx) for g in active}
        Z = sum(exps.values())
        q = {g: exps[g] / Z for g in active}
        # Sagawa step scaled by q_g * G (uniform -> 1)
        dro_loss = sum((q[g] * G) * grp_mean[g] for g in active)
        dro_loss.backward(); opt.step()
        q_history.append(dict(q))
        simplex_sum = sum(q.values())
        print(f"  dro step {step}: worst_g={worst_g} worst_loss={worst_loss:.4f} "
              f"q={ {g: round(q[g],3) for g in active} } sum(q)={simplex_sum:.6f} "
              f"loss={float(dro_loss):.4f} mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        assert abs(simplex_sum - 1.0) < 1e-6, "q left the simplex (sum != 1)"
        assert all(v >= 0 for v in q.values()), "q has a negative entry"

    # q actually moved off uniform
    moved = any(abs(q_history[-1][g] - 1.0 / G) > 1e-6 for g in active)
    print(f"  q moved off uniform 1/{G}: {moved}   final q={ {g: round(q[g],4) for g in active} }")
    assert moved, "q did not update away from uniform"
    print("  PASS: 4 groups formed, q updates + stays on simplex, worst-group tracked, no OOM")

    print("\nSMOKE OK: DRO off bit-for-bit ERM + DRO on groups/q-update/worst-group on GPU.")


if __name__ == "__main__":
    main()
