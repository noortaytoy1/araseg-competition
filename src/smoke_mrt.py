"""GPU smoke for mrt_train.py — MUST run on the real GPU.

Proves the four self-test properties the MRT build requires:
  (a) MRT OFF (mrt off / weight 0) reproduces plain weighted-CE BIT-FOR-BIT:
      torch.equal on both loss and logits against an independent weighted CE
      computed straight from AutoModelForTokenClassification logits (the same
      check smoke_consistency.py runs for the CE baseline).
  (b) MRT ON: the k=6 candidate labelings are DIVERSE (not all identical).
  (c) the reward / Delta is computed on the STITCHED, doc-level boundary sequence
      (per-doc macro-F1 via the eval_local scorer), and the REINFORCE-with-
      baseline surrogate produces a NON-ZERO gradient into the model parameters.
  (d) a few real MRT optimizer steps run on the GPU with no device / OOM error.

Run:
  cd arabic-sentence-segmentation
  python src/smoke_mrt.py --train-jsonl data/NoPnx-NP_train.jsonl
"""
import argparse
import os
import sys
import types

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, set_seed
from data import MODEL_PAR_TOKEN, load_task
import mrt_train as M


def build_window_batch(jsonl, task, n_docs, window, stride, bs, tok):
    docs = load_task(task, "train", jsonl_path=jsonl)[:n_docs]
    exs = M.make_examples(docs, window, stride)[:bs]
    enc = [M.encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, 512)
           for e in exs]
    feats = [{k: e[k][0] for k in ("input_ids", "attention_mask", "labels")} for e in enc]
    return M.Collator(tok)(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--stride", type=int, default=90)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "GPU-SMOKE requires a real CUDA device"
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    pos_weight = 4.0

    # ---------------------------------------------------------------- #
    # (a) BIT-FOR-BIT baseline: MRT off  vs  independent weighted CE    #
    # ---------------------------------------------------------------- #
    batch = build_window_batch(a.train_jsonl, a.task, 20, a.window, a.stride, a.bs, tok)
    batch = {k: v.to(dev) for k, v in batch.items()}
    print(f"batch: input_ids {tuple(batch['input_ids'].shape)}")

    set_seed(0)
    model = M.MRTModel(a.model_name, pos_weight, n_extra_tok=1).to(dev)
    model.eval()  # dropout off -> deterministic forward
    with torch.no_grad():
        out = model(**batch)
        gated_loss, gated_logits = out["loss"], out["logits"]
        ref_logits = model.enc(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
        w = torch.tensor([1.0, pos_weight], device=dev)
        ref_loss = nn.functional.cross_entropy(
            ref_logits.view(-1, 2).float(), batch["labels"].view(-1),
            weight=w, ignore_index=-100)
    logits_equal = torch.equal(gated_logits, ref_logits)
    loss_equal = torch.equal(gated_loss, ref_loss)
    print("\n[(a) bit-for-bit @ mrt off]")
    print(f"  gated loss = {gated_loss.item():.10f}")
    print(f"  ref   loss = {ref_loss.item():.10f}")
    print(f"  torch.equal(logits) = {logits_equal}")
    print(f"  torch.equal(loss)   = {loss_equal}")
    assert logits_equal, "logits differ with MRT off"
    assert loss_equal, "loss NOT bit-for-bit identical to plain weighted CE with MRT off"

    # also confirm _mrt_enabled gating: off, and on-with-weight-0, are both OFF.
    off1 = types.SimpleNamespace(mrt="off", mrt_weight=1.0)
    off2 = types.SimpleNamespace(mrt="on", mrt_weight=0.0)
    on1 = types.SimpleNamespace(mrt="on", mrt_weight=1.0)
    assert not M._mrt_enabled(off1) and not M._mrt_enabled(off2) and M._mrt_enabled(on1)
    print("  gate: mrt=off -> OFF, mrt=on weight=0 -> OFF, mrt=on weight>0 -> ON")
    print("  PASS: baseline reproduced bit-for-bit")

    # ---------------------------------------------------------------- #
    # (b/c/d) MRT ON: diverse samples, stitched reward, non-zero grad,  #
    #         a few GPU steps.                                          #
    # ---------------------------------------------------------------- #
    print("\n[(b/c/d) mrt on]")
    # pick a doc long enough to have multiple windows (exercises the stitcher)
    docs = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    doc = max(docs[:40], key=lambda d: len(d["tokens"]))
    print(f"  doc {doc['doc_id']}: {len(doc['tokens'])} tokens "
          f"(> window {a.window} => multi-window stitch)")

    args = types.SimpleNamespace(
        window=a.window, stride=a.stride, overlap=a.overlap, max_length=512,
        mrt_k=6, mrt_alpha=5e-3, mrt_temp=1.0, threshold=0.5,
        mrt_weight=1.0, mrt_ce_weight=1.0, max_grad_norm=1.0)

    set_seed(0)
    model = M.MRTModel(a.model_name, pos_weight, n_extra_tok=1).to(dev)
    model.train()

    # inspect the candidate set + reward on the STITCHED doc sequence
    import numpy as np
    from sklearn.metrics import f1_score
    from data import PARAGRAPH_TOKEN
    words = [MODEL_PAR_TOKEN if wtok == PARAGRAPH_TOKEN else wtok for wtok in doc["tokens"]]
    avg, _, _ = M._doc_word_probs(model, tok, words, a.window, a.overlap, 512, dev,
                                  with_grad=False)
    assert len(avg) == len(doc["tokens"]), "stitched prob length != doc length"
    greedy = M._apply_par_zeros((avg >= 0.5).astype(int).tolist(), doc["tokens"])
    greedy_f1 = f1_score(list(doc["labels"]), greedy, average="binary", zero_division=0)
    print(f"  stitched greedy doc-F1 = {greedy_f1:.4f} "
          f"(reward on the DOC-LEVEL sequence, len={len(avg)})")

    surrogate, st = M._mrt_doc_loss(model, tok, doc, args, dev)
    print(f"  candidates: n={st['n_cand']} unique={st['n_unique']}  "
          f"delta[min,max]=[{st['delta_min']:.3f},{st['delta_max']:.3f}]  "
          f"reward_mean_F1={st['reward_mean_f1']:.3f}")
    assert st["n_unique"] >= 2, "samples are NOT diverse (all candidates identical)"
    print("  (b) PASS: candidate labelings are diverse")

    # non-zero gradient from the surrogate into the encoder params
    model.zero_grad()
    surrogate.backward()
    gnorm = 0.0
    for pn in model.parameters():
        if pn.grad is not None:
            gnorm += float(pn.grad.detach().float().norm() ** 2)
    gnorm = gnorm ** 0.5
    print(f"  surrogate = {float(surrogate.detach().cpu()):+.6e}  "
          f"grad L2-norm into params = {gnorm:.6e}")
    assert gnorm > 0, "REINFORCE surrogate produced a ZERO gradient"
    print("  (c) PASS: reward computed on stitched doc sequence + non-zero gradient")

    # a few full MRT optimizer steps (CE + MRT surrogate) on the GPU
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for i in range(a.steps):
        opt.zero_grad()
        ce = torch.zeros((), device=dev)
        exs = M.make_examples([doc], a.window, a.stride)
        enc = [M.encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, 512)
               for e in exs]
        feats = [{k: e[k][0] for k in ("input_ids", "attention_mask", "labels")} for e in enc]
        coll = M.Collator(tok)
        ntok = 0
        for j in range(0, len(feats), 16):
            b = coll([dict(f) for f in feats[j:j + 16]])
            b = {k: v.to(dev) for k, v in b.items()}
            o = model(**b)
            m = int((b["labels"] != -100).sum())
            ce = ce + o["loss"] * m; ntok += m
        ce = ce / max(ntok, 1)
        mrt_loss, _ = M._mrt_doc_loss(model, tok, doc, args, dev)
        loss = ce + mrt_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"  mrt step {i}: ce={float(ce.detach().cpu()):.4f} "
              f"mrt={float(mrt_loss.detach().cpu()):+.4e} "
              f"mem={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print("  (d) PASS: MRT trains a few GPU steps, no device/OOM")

    print("\nSMOKE OK: mrt off bit-for-bit CE; mrt on diverse + stitched-reward + "
          "non-zero grad + GPU steps.")


if __name__ == "__main__":
    main()
