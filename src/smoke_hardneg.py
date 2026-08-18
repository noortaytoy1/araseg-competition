"""CPU smoke for hardneg_contrastive_train.py — runs entirely on CPU
(the GPU is owned by the frontier/sequel batteries; run with
CUDA_VISIBLE_DEVICES=-1).

Proves, per the build order:
  (a) --hardneg-weight 0 reproduces the plain weighted-CE baseline BIT-FOR-BIT:
      * every parameter/buffer equals consistency_train.ConsistencyModel's
        after identical seeding (same RNG stream -> same init -> same
        trajectory under the identical recipe), torch.equal per tensor;
      * on a real batch, loss and logits are torch.equal to (i) an independent
        CE computed straight from AutoModelForTokenClassification logits and
        (ii) the ConsistencyModel all-off forward.
  (b) mining runs against a real trained checkpoint on a small doc slice and
      prints a sane count + examples (the FULL-train count is measured by the
      `mine` subcommand, reported separately).
  (c) with --hardneg-weight 0.3 the L_hn term is NON-ZERO and DECREASES over a
      few CPU optimization steps on a tiny slice (enc warm-loaded from the
      checkpoint so dynamic negatives are realistic; eval mode = dropout off
      for a clean monotone read, mirroring smoke_consistency's term isolation).
  (d) --arctanh on is FINITE at extreme cosines: the score function at
      cos = -1, -0.99999, 0.99999, 1 (clamped) and a full forward/backward.

Run:
  cd arabic-sentence-segmentation
  CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 python src/smoke_hardneg.py \
      --init-model runs/frontier_battery_2026-07-09_v3/cons_baseline_s42
"""
import argparse
import math
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          set_seed)
from data import MODEL_PAR_TOKEN, load_task
import consistency_train as C
import hardneg_contrastive_train as H


def build_batch(jsonl, task, n_docs, window, stride, bs, tok, hn_sets=None):
    docs = load_task(task, "train", jsonl_path=jsonl)[:n_docs]
    exs = H.make_examples(docs, window, stride, hn_sets)[:bs]
    enc = [H.encode({"words": [e["words"]], "word_labels": [e["word_labels"]],
                     "word_hn": [e["word_hn"]]}, tok, 512) for e in exs]
    feats = []
    for e in enc:
        feats.append({k: e[k][0] for k in ("input_ids", "attention_mask",
                                           "labels", "hn_mask")})
    return H.Collator(tok)(feats)


def warm_load_enc(model, init_dir):
    """mrt_train._load_init_model pattern: load an HF checkpoint into model.enc."""
    src = AutoModelForTokenClassification.from_pretrained(init_dir, num_labels=2)
    sd = src.state_dict()
    tgt = model.enc.state_dict()
    matched = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
    model.enc.load_state_dict({**tgt, **matched}, strict=False)
    return len(matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--init-model",
                    default="runs/frontier_battery_2026-07-09_v3/cons_baseline_s42")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--stride", type=int, default=90)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--mine-docs", type=int, default=6)
    a = ap.parse_args()

    dev = "cpu"  # CPU-ONLY smoke; the GPU belongs to the running batteries
    print(f"device: {dev} (forced)  torch {torch.__version__}  "
          f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    pos_weight = 4.0

    batch = build_batch(a.train_jsonl, a.task, 20, a.window, a.stride, a.bs, tok)
    print(f"batch: input_ids {tuple(batch['input_ids'].shape)}  "
          f"anchors(gold-1)={int((batch['labels'] == 1).sum())}  "
          f"gold-0={int((batch['labels'] == 0).sum())}")

    # ---------------------------------------------------------------- #
    # (a) BIT-FOR-BIT at weight 0: params vs ConsistencyModel + loss    #
    # ---------------------------------------------------------------- #
    print("\n[(a) bit-for-bit @ hardneg-weight 0]")
    set_seed(0)
    ref = C.ConsistencyModel(a.model_name, pos_weight, cf_weight=0.0, rdrop_weight=0.0,
                             mask_id=tok.mask_token_id, n_extra_tok=1).to(dev)
    set_seed(0)
    hn0 = H.HardNegModel(a.model_name, pos_weight, hardneg_weight=0.0,
                         n_extra_tok=1).to(dev)
    sd_r, sd_h = ref.state_dict(), hn0.state_dict()
    assert set(sd_r.keys()) == set(sd_h.keys()), \
        f"key mismatch: {set(sd_r) ^ set(sd_h)}"
    n_eq = sum(int(torch.equal(sd_r[k], sd_h[k])) for k in sd_r)
    print(f"  state-dict tensors equal: {n_eq}/{len(sd_r)}")
    assert n_eq == len(sd_r), "params differ at weight 0 (RNG stream diverged)"

    ref.eval(); hn0.eval()
    with torch.no_grad():
        out_h = hn0(**batch)
        out_c = ref(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        raw = hn0.enc(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"]).logits
        w = torch.tensor([1.0, pos_weight])
        ref_ce = nn.functional.cross_entropy(
            raw.view(-1, 2).float(), batch["labels"].view(-1),
            weight=w, ignore_index=-100)
    print(f"  hardneg loss = {out_h['loss'].item():.10f}")
    print(f"  indep CE     = {ref_ce.item():.10f}")
    print(f"  cons loss    = {out_c['loss'].item():.10f}")
    print(f"  torch.equal(logits vs raw)  = {torch.equal(out_h['logits'], raw)}")
    print(f"  torch.equal(loss vs indepCE)= {torch.equal(out_h['loss'], ref_ce)}")
    print(f"  torch.equal(loss vs cons)   = {torch.equal(out_h['loss'], out_c['loss'])}")
    assert torch.equal(out_h["logits"], raw)
    assert torch.equal(out_h["loss"], ref_ce)
    assert torch.equal(out_h["loss"], out_c["loss"])
    assert not hasattr(hn0, "proj"), "projection head must NOT exist at weight 0"
    print("  PASS: weight 0 is bit-for-bit plain weighted CE (and == cons_baseline)")
    del ref, hn0, out_h, out_c, raw

    # ---------------------------------------------------------------- #
    # (b) mining mechanics on a small slice (real checkpoint, CPU)      #
    # ---------------------------------------------------------------- #
    print(f"\n[(b) mining on first {a.mine_docs} train docs, real checkpoint]")
    docs = load_task(a.task, "train", jsonl_path=a.train_jsonl)[:a.mine_docs]
    hn_sets, stats, rows = H.mine_hard_negatives(
        a.init_model, docs, tok, dev, source="auto", thresh=0.8,
        topn=300, min_count=200)
    assert stats["n_mined"] > 0, "mining produced nothing on the slice"
    assert all(docs[r["doc_index"]]["labels"][r["word_index"]] == 0 for r in rows), \
        "a mined position is not gold-0"
    print(f"  PASS: slice mining sane (fired={stats['source_fired']}, "
          f"n={stats['n_mined']}; full-train count comes from the mine subcommand)")

    # ---------------------------------------------------------------- #
    # (c) L_hn non-zero and decreasing over a few CPU steps             #
    # ---------------------------------------------------------------- #
    print("\n[(c) hardneg-weight 0.3: term non-zero + decreases]")
    set_seed(0)
    hn = H.HardNegModel(a.model_name, pos_weight, hardneg_weight=0.3,
                        margin=0.3, dyn_negs=64, n_extra_tok=1).to(dev)
    nm = warm_load_enc(hn, a.init_model)
    print(f"  enc warm-loaded from {a.init_model}: {nm} tensors "
          f"(miner-quality probs for dynamic negatives)")
    cbatch = build_batch(a.train_jsonl, a.task, a.mine_docs, a.window, a.stride,
                         a.bs, tok, hn_sets)
    n_mined_in_batch = int(((cbatch["hn_mask"] == 1) & (cbatch["labels"] == 0)).sum())
    print(f"  mined hard negatives present in batch: {n_mined_in_batch}")
    hn.eval()  # dropout off -> clean monotone read (grads still flow)
    opt = torch.optim.AdamW(hn.parameters(), lr=5e-5)
    hist = []
    for i in range(a.steps):
        opt.zero_grad()
        out = hn(**cbatch)
        out["loss"].backward()
        opt.step()
        hist.append(hn.last_hn)
        print(f"  step {i}: total={out['loss'].item():.6f}  L_hn={hn.last_hn:.6f}  "
              f"anchors={hn.last_n_anchor}  negs={hn.last_n_neg}")
    assert all(math.isfinite(v) for v in hist)
    assert hist[0] > 0, "L_hn is zero at start (fresh head should hinge ~margin)"
    assert hist[-1] < hist[0], f"L_hn did not decrease: {hist[0]:.6f} -> {hist[-1]:.6f}"
    print(f"  PASS: L_hn {hist[0]:.6f} -> {hist[-1]:.6f} over {a.steps} CPU steps")
    del hn, opt

    # ---------------------------------------------------------------- #
    # (d) arctanh mode finite at extreme cosines                        #
    # ---------------------------------------------------------------- #
    print("\n[(d) arctanh finite at extreme cosines]")
    set_seed(0)
    hna = H.HardNegModel(a.model_name, pos_weight, hardneg_weight=0.3,
                         margin=0.3, arctanh=True, n_extra_tok=1).to(dev)
    ext = torch.tensor([-1.0, -0.99999, 0.0, 0.99999, 1.0])
    s = hna._sim(ext)
    print(f"  _sim({ext.tolist()}) = {[round(float(v), 4) for v in s]}")
    assert torch.isfinite(s).all(), "arctanh score not finite at extreme cosines"
    hna.eval()
    out = hna(**batch)
    out["loss"].backward()
    gnorm = torch.nn.utils.clip_grad_norm_(hna.parameters(), 1e9)
    print(f"  arctanh forward: total={out['loss'].item():.6f}  L_hn={hna.last_hn:.6f}  "
          f"grad_norm={float(gnorm):.4f}")
    assert torch.isfinite(out["loss"]) and math.isfinite(float(gnorm))
    assert hna.last_hn > 0
    print("  PASS: arctanh mode finite (loss + grads) with non-zero term")

    print("\nSMOKE OK: (a) bit-for-bit CE at weight 0, (b) mining sane, "
          "(c) L_hn non-zero and decreasing, (d) arctanh finite.")


if __name__ == "__main__":
    main()
