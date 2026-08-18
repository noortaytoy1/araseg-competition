"""GPU smoke for semicrf_train.py — Filtered Semi-Markov CRF. MUST run on GPU.

Proves the pre-registered self-test:
  (a) --semicrf off reproduces the plain weighted-CE baseline BIT-FOR-BIT
      (torch.equal against an independent weighted CE straight from
      AutoModelForTokenClassification logits — same reference as
      smoke_consistency.py).
  (b) --semicrf on:
        * logZ is FINITE and logZ >= gold-path score (normalization sanity),
        * Viterbi decode yields a VALID segmentation (a 0/1 boundary sequence
          whose induced spans tile the window, each length <= L, last pos = 1),
        * the DURATION bias table RECEIVES GRADIENT after a backward pass,
        * a few real GPU steps train with no device / OOM error.

Run:
  cd arabic-sentence-segmentation
  python src/smoke_semicrf.py --train-jsonl data/NoPnx-NP_train.jsonl
"""
import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, set_seed
from data import MODEL_PAR_TOKEN, load_task
import semicrf_train as S


def build_batch(jsonl, task, n_docs, window, stride, bs, tok):
    docs = load_task(task, "train", jsonl_path=jsonl)[:n_docs]
    exs = S.make_examples(docs, window, stride)[:bs]
    enc = [S.encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, 512)
           for e in exs]
    feats = []
    for e in enc:
        feats.append({k: e[k][0] for k in ("input_ids", "attention_mask", "labels")})
    coll = S.Collator(tok)
    return coll(feats)


def spans_from_labels(labels):
    """Induce spans from a 0/1 boundary sequence (boundary = span END)."""
    spans, prev = [], 0
    for k, b in enumerate(labels):
        if b == 1:
            spans.append((prev, k)); prev = k + 1
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--stride", type=int, default=90)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--max-span", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "GPU-SMOKE requires a real CUDA device"
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})

    batch = build_batch(a.train_jsonl, a.task, 20, a.window, a.stride, a.bs, tok)
    batch = {k: v.to(dev) for k, v in batch.items()}
    n_lab = int((batch["labels"] != -100).sum())
    n_pos = int((batch["labels"] == 1).sum())
    print(f"batch: input_ids {tuple(batch['input_ids'].shape)}  "
          f"label positions={n_lab}  gold boundaries={n_pos}")
    assert n_pos > 0, "no gold boundaries in smoke batch — pick more docs"

    pos_weight = 4.0

    # ---------------------------------------------------------------- #
    # (a) BIT-FOR-BIT: --semicrf off  vs  independent weighted CE       #
    # ---------------------------------------------------------------- #
    set_seed(0)
    model = S.SemiCRFModel(a.model_name, pos_weight, semicrf=False,
                           max_span=a.max_span, n_extra_tok=1).to(dev)
    model.eval()  # dropout off -> deterministic single forward
    with torch.no_grad():
        out = model(**batch)
        gated_loss = out["loss"]; gated_logits = out["logits"]
        ref_logits = model.enc(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
        w = torch.tensor([1.0, pos_weight], device=dev)
        ref_loss = nn.functional.cross_entropy(
            ref_logits.view(-1, 2).float(), batch["labels"].view(-1),
            weight=w, ignore_index=-100)
    logits_equal = torch.equal(gated_logits, ref_logits)
    loss_equal = torch.equal(gated_loss, ref_loss)
    print("\n[(a) bit-for-bit @ --semicrf off]")
    print(f"  gated loss = {gated_loss.item():.10f}")
    print(f"  ref   loss = {ref_loss.item():.10f}")
    print(f"  torch.equal(logits) = {logits_equal}")
    print(f"  torch.equal(loss)   = {loss_equal}")
    assert logits_equal, "logits differ with --semicrf off"
    assert loss_equal, "loss NOT bit-for-bit identical to plain weighted CE with --semicrf off"
    print("  PASS: baseline reproduced bit-for-bit")

    # ---------------------------------------------------------------- #
    # (b) --semicrf on: normalization, valid decode, duration gradient #
    # ---------------------------------------------------------------- #
    print("\n[(b) --semicrf on]")
    set_seed(0)
    model_on = S.SemiCRFModel(a.model_name, pos_weight, semicrf=True,
                              max_span=a.max_span, n_extra_tok=1).to(dev)

    # --- normalization sanity on every window in the batch: logZ finite & >= gold
    model_on.eval()
    with torch.no_grad():
        outc = model_on.enc(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            output_hidden_states=True)
        h = outc.hidden_states[-1]
        B = batch["input_ids"].shape[0]
        worst_gap = float("inf")
        for b in range(B):
            lab = batch["labels"][b]
            keep = (lab != -100)
            pos = torch.nonzero(keep, as_tuple=False).flatten()
            h_last = h[b][pos]
            gold_bound = lab[pos].clamp(min=0)
            l_loc, l_glob, logZ, gold_score, ns = model_on._window_losses(h_last, gold_bound)
            gap = float(logZ) - float(gold_score)
            worst_gap = min(worst_gap, gap)
            assert torch.isfinite(logZ), f"logZ not finite in window {b}"
            assert gap >= -1e-3, (f"logZ < gold score in window {b}: "
                                  f"logZ={float(logZ):.4f} gold={float(gold_score):.4f}")
            # valid decode
            P = h_last.shape[0]
            win_lab = model_on.decode_window(h_last)
            assert len(win_lab) == P, "decode length mismatch"
            assert win_lab[-1] == 1, "last position must be a segment end (boundary)"
            spans = spans_from_labels(win_lab)
            # spans must tile 0..P-1 with each length <= L
            cur = 0
            for (i, j) in spans:
                assert i == cur, f"non-tiling span in window {b}: {spans}"
                assert 1 <= (j - i + 1) <= a.max_span, f"span length out of [1,L]: {(i,j)}"
                cur = j + 1
            assert cur == P, f"decoded spans do not cover the window {b}: {spans}"
        print(f"  logZ finite on all {B} windows; min(logZ - gold_score) = {worst_gap:.4f} (>= 0)")
        print(f"  Viterbi decode valid on all {B} windows (tiling, len<=L, last=boundary)")

    # --- duration bias table receives gradient ---
    model_on.train()
    model_on.zero_grad()
    loss = model_on(**batch)["loss"]
    loss.backward()
    dgrad = model_on.scorer.duration_bias.grad
    assert dgrad is not None, "duration_bias.grad is None — table received NO gradient"
    dgrad_norm = float(dgrad.norm())
    print(f"  loss={float(loss):.6f}  duration_bias.grad norm={dgrad_norm:.6e}")
    assert dgrad_norm > 0, "duration_bias gradient is zero — table not learning"
    print("  PASS: duration bias table receives gradient")

    # --- a few real GPU training steps, no device / OOM ---
    opt = torch.optim.AdamW(model_on.parameters(), lr=5e-5)
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.steps):
        opt.zero_grad()
        loss = model_on(**batch)["loss"]
        loss.backward(); opt.step()
        print(f"  semicrf step {i}: loss={loss.item():.6f}  "
              f"mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print("  PASS: semi-CRF trains a few GPU steps, no device/OOM")

    print("\nSMOKE OK: off bit-for-bit + on (logZ>=gold, valid Viterbi, duration "
          "grad, GPU steps).")


if __name__ == "__main__":
    main()
