"""CPU smoke for consistency_train.py METHOD 4 (Child-Tuning-D, --child-tune)
and METHOD 5 (SWA, --swa) — 2026-07-10, integration battery Stage-1 build.

CPU ONLY (the GPU is owned by the sequel battery): run with
CUDA_VISIBLE_DEVICES=-1. The smoke ABORTS if a CUDA device is visible.

Proves, on a tiny truncated corpus:
  (a) both flags at their defaults (--child-tune 0.0, --swa off) reproduce the
      plain weighted-CE baseline BIT-FOR-BIT: torch.equal on loss AND logits
      against an independent CE computed straight from the encoder logits.
      (The full legacy-vs-new TRAINING bit-for-bit — identical model.pt,
      identical log_history, identical torch/numpy/python RNG states after
      cmd_train — is run by the session's bitcheck driver against the frozen
      pre-edit snapshot; this smoke re-proves the forward-level identity.)
  (b) --child-tune P: the child mask covers EXACTLY k = round(P * n_encoder_
      entries) entries and they are the global Fisher top-k (verified against
      an INDEPENDENT hand-computed Fisher pass); the classifier head is never
      masked; after backward every masked encoder entry has grad exactly 0;
      after one AdamW step (wd=0) every masked entry is bit-for-bit unchanged
      while unmasked encoder entries and the head move.
  (c) --swa on: the SWA average equals the HAND-COMPUTED float64 mean of the
      epoch-end snapshot weights (unit-level, torch.equal), AND an end-to-end
      tiny cmd_train writes <out>/swa whose weights equal the hand mean of the
      live epoch-end snapshots recorded by a spy, loadable via
      AutoModelForTokenClassification.from_pretrained.

Run:
  cd arabic-sentence-segmentation
  CUDA_VISIBLE_DEVICES=-1 python src/smoke_childtune_swa.py \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed
from data import MODEL_PAR_TOKEN, load_task
import consistency_train as C


def write_tiny(src_jsonl, dst_jsonl, task, n_docs, n_tok):
    docs = load_task(task, "train", jsonl_path=src_jsonl)[:n_docs]
    with open(dst_jsonl, "w", encoding="utf-8") as f:
        for d in docs:
            d = dict(d)
            d["tokens"] = d["tokens"][:n_tok]
            d["labels"] = d["labels"][:n_tok]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return docs


def build_loader(jsonl, task, tok, prep_ids, window, stride, max_len, bs):
    docs = load_task(task, "train", jsonl_path=jsonl)
    ds = Dataset.from_list(C.make_examples(docs, window, stride))
    ds = ds.map(lambda ex: C.encode(ex, tok, max_len), batched=True,
                remove_columns=ds.column_names)
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      collate_fn=C.Collator(tok, prep_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--dev-jsonl", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--n-docs", type=int, default=6)
    ap.add_argument("--n-tok", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=0.3)
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for tiny jsonl + the (c) end-to-end run "
                         "(default: a fresh tempdir; NEVER runs/)")
    a = ap.parse_args()

    assert not torch.cuda.is_available(), \
        "CPU smoke: hide the GPU (CUDA_VISIBLE_DEVICES=-1) — the GPU is owned by the sequel battery"
    dev = "cpu"
    work = a.workdir or tempfile.mkdtemp(prefix="smoke_childtune_swa_")
    os.makedirs(work, exist_ok=True)
    print(f"device: cpu  torch {torch.__version__}  workdir: {work}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    prep_ids = C._prep_ids(tok)

    tiny_tr = os.path.join(work, "tiny_train.jsonl")
    tiny_dv = os.path.join(work, "tiny_dev.jsonl")
    write_tiny(a.train_jsonl, tiny_tr, a.task, a.n_docs, a.n_tok)
    write_tiny(a.dev_jsonl, tiny_dv, a.task, max(2, a.n_docs // 2), a.n_tok)

    loader = build_loader(tiny_tr, a.task, tok, prep_ids, 180, 90, 512, 4)
    batches = [{k: v for k, v in b.items()} for b in loader]
    print(f"tiny corpus: {a.n_docs} docs -> {sum(b['input_ids'].shape[0] for b in batches)} "
          f"windows in {len(batches)} batches")

    pos_weight = 4.0
    set_seed(0)
    model = C.ConsistencyModel(a.model_name, pos_weight, cf_weight=0.0, rdrop_weight=0.0,
                               mask_id=tok.mask_token_id, n_extra_tok=1).to(dev)

    # ---------------------------------------------------------------- #
    # (a) BOTH FLAGS OFF -> bit-for-bit plain weighted CE               #
    # ---------------------------------------------------------------- #
    print("\n[(a) defaults --child-tune 0.0 --swa off = bit-for-bit plain CE]")
    model.eval()
    batch = batches[0]
    with torch.no_grad():
        out = model(**batch)
        ref_logits = model.enc(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
        w = torch.tensor([1.0, pos_weight])
        ref_loss = nn.functional.cross_entropy(
            ref_logits.view(-1, 2).float(), batch["labels"].view(-1),
            weight=w, ignore_index=-100)
    assert torch.equal(out["logits"], ref_logits), "logits differ at defaults"
    assert torch.equal(out["loss"], ref_loss), "loss differs at defaults"
    print(f"  torch.equal(logits)=True  torch.equal(loss)=True  loss={out['loss'].item():.10f}")
    assert not hasattr(model, "_child_masks"), "child masks exist without --child-tune"
    print("  PASS: defaults reproduce plain weighted CE bit-for-bit; no masks, no hooks")

    # ---------------------------------------------------------------- #
    # (b) CHILD-TUNING-D                                                #
    # ---------------------------------------------------------------- #
    print(f"\n[(b) --child-tune {a.top_p}]")
    masks = C.compute_child_mask(model, loader, a.top_p, dev)

    # (b1) head never masked; mask covers EXACTLY round(P * n_entries)
    assert all(not n.startswith("classifier.") for n in masks), "classifier key in masks"
    enc_names = [n for n, _ in C._encoder_named_params(model)]
    assert set(masks) == set(enc_names), "mask keys != encoder param set"
    n_entries = sum(m.numel() for m in masks.values())
    n_kept = int(sum(m.sum().item() for m in masks.values()))
    k_expected = int(round(a.top_p * n_entries))
    print(f"  encoder entries={n_entries}  kept={n_kept}  expected k={k_expected}")
    assert n_kept == k_expected, "mask does not cover exactly round(P*n) entries"
    assert all(((m == 0) | (m == 1)).all() for m in masks.values()), "mask not 0/1"

    # (b2) INDEPENDENT hand Fisher -> the kept set is the exact global top-k
    hand = {n: torch.zeros_like(p, dtype=torch.float32)
            for n, p in C._encoder_named_params(model)}
    model.eval()
    for b in batches:
        model.zero_grad(set_to_none=True)
        ce = model._ce(model._logits(b["input_ids"], b["attention_mask"]), b["labels"])
        ce.backward()
        for n, p in C._encoder_named_params(model):
            if p.grad is not None:
                hand[n] += p.grad.detach().float() ** 2
    model.zero_grad(set_to_none=True)
    hand_flat = torch.cat([hand[n].reshape(-1) for n in enc_names])
    mask_flat = torch.cat([masks[n].reshape(-1) for n in enc_names]).bool()
    kept_min = hand_flat[mask_flat].min().item()
    drop_max = hand_flat[~mask_flat].max().item()
    print(f"  hand-Fisher: min(kept)={kept_min:.6e}  max(dropped)={drop_max:.6e}")
    assert kept_min >= drop_max, "mask is NOT the global Fisher top-k (kept < dropped)"
    hand_topk = torch.zeros_like(mask_flat)
    hand_topk[torch.topk(hand_flat, k_expected, sorted=False).indices] = True
    n_disagree = int((hand_topk != mask_flat).sum())
    print(f"  exact top-k index agreement: {n_entries - n_disagree}/{n_entries} "
          f"(disagreements={n_disagree}, ties only)")
    assert n_disagree == 0 or kept_min == drop_max, "top-k disagreement beyond ties"
    del hand, hand_flat, hand_topk
    print("  PASS: mask == exact global top-P Fisher entries, head unmasked")

    # (b3) hooks: masked grads exactly zero after backward; head grads live
    C.apply_child_hooks(model, masks)
    model.zero_grad(set_to_none=True)
    model.eval()  # deterministic; grad flow identical to train for this check
    loss = model(**batches[0])["loss"]
    loss.backward()
    n_zero_ok, n_kept_nonzero = 0, 0
    for n, p in C._encoder_named_params(model):
        g, m = p.grad, masks[n].bool()
        assert g is not None
        assert (g[~m] == 0).all(), f"masked entries of {n} received gradient"
        n_zero_ok += int((~m).sum())
        n_kept_nonzero += int((g[m] != 0).sum())
    head_g = [p.grad for n, p in model.enc.named_parameters() if n.startswith("classifier.")]
    assert all(g is not None and g.abs().sum() > 0 for g in head_g), "head grads dead"
    print(f"  masked entries with grad==0: {n_zero_ok}/{n_entries - n_kept} (all)  "
          f"kept entries with grad!=0: {n_kept_nonzero}  head grad nonzero: True")

    # (b4) one AdamW step (wd=0): masked entries bit-for-bit unchanged, rest move
    before = {n: p.detach().clone() for n, p in model.enc.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    opt.step()
    n_moved_kept, n_moved_head = 0, 0
    for n, p in C._encoder_named_params(model):
        m = masks[n].bool()
        assert torch.equal(p.detach()[~m], before[n][~m]), \
            f"masked entries of {n} moved during the update"
        n_moved_kept += int((p.detach()[m] != before[n][m]).sum())
    for n, p in model.enc.named_parameters():
        if n.startswith("classifier."):
            n_moved_head += int((p.detach() != before[n]).sum())
    assert n_moved_kept > 0, "no unmasked encoder entry moved"
    assert n_moved_head > 0, "head did not move"
    print(f"  after AdamW(wd=0) step: masked entries unchanged (torch.equal), "
          f"unmasked moved={n_moved_kept}, head moved={n_moved_head}")
    print("  PASS: zero grad-update on masked entries; unmasked + head move")
    del before, opt

    # ---------------------------------------------------------------- #
    # (c) SWA                                                           #
    # ---------------------------------------------------------------- #
    print("\n[(c) --swa on]")
    # (c1) UNIT: average == hand-computed float64 mean of the snapshots
    total_epochs, start_frac = 4, 0.5
    cb = C.SWACallback(model, start_frac, total_epochs)   # start_after = 2.0
    hand_snaps = []
    g = torch.Generator().manual_seed(7)
    for e in range(1, total_epochs + 1):
        with torch.no_grad():
            for p in model.enc.parameters():
                p.add_(torch.randn(p.shape, generator=g) * 1e-3)
        if e >= start_frac * total_epochs:
            hand_snaps.append({k: v.detach().to("cpu", torch.float64).clone()
                               for k, v in model.enc.state_dict().items()})
        cb.on_epoch_end(None, None, None)
    assert cb.n_snapshots == 3 and cb.epochs_in == [2, 3, 4], \
        f"snapshot schedule wrong: n={cb.n_snapshots} epochs={cb.epochs_in}"
    like = model.enc.state_dict()
    swa_sd = cb.swa_state_dict(like)
    for k in like:
        acc = hand_snaps[0][k].clone()
        for s in hand_snaps[1:]:
            acc += s[k]
        hand_mean = (acc / len(hand_snaps)).to(like[k].dtype)
        assert torch.equal(swa_sd[k], hand_mean), f"SWA average != hand mean at {k}"
    print(f"  unit: snapshots at epochs {cb.epochs_in} (start_after="
          f"{start_frac * total_epochs}); swa_state_dict == hand float64 mean "
          f"(torch.equal) on all {len(like)} tensors")

    # (c2) END-TO-END tiny cmd_train --swa on; spy records the live snapshots
    out_dir = os.path.join(work, "swa_run")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    recorded = []
    orig_hook = C.SWACallback.on_epoch_end

    def spy(self, args, state, control, **kw):
        n0 = self.n_snapshots
        r = orig_hook(self, args, state, control, **kw)
        if self.n_snapshots > n0:
            recorded.append({k: v.detach().to("cpu", torch.float64).clone()
                             for k, v in self.model.enc.state_dict().items()})
        return r

    C.SWACallback.on_epoch_end = spy
    try:
        args = argparse.Namespace(
            task=a.task, out=out_dir, model_name=a.model_name,
            train_jsonl=tiny_tr, dev_jsonl=tiny_dv,
            cf_weight=0.0, rdrop_weight=0.0, sam_rho=0.0,
            child_tune=0.0, swa="on", swa_start_frac=0.5,
            window=180, stride=90, max_length=512, epochs=2, lr=5e-5,
            batch_size=4, pos_weight=None, seed=42)
        C.cmd_train(args)
    finally:
        C.SWACallback.on_epoch_end = orig_hook

    swa_dir = os.path.join(out_dir, "swa")
    meta = json.load(open(os.path.join(swa_dir, "swa_meta.json")))
    print(f"  end-to-end: swa_meta.json = {meta}")
    assert meta["n_snapshots"] == len(recorded) == 2 and meta["epochs_averaged"] == [1, 2]
    loaded = AutoModelForTokenClassification.from_pretrained(swa_dir)
    print("  swa/ loads via AutoModelForTokenClassification: True")
    lsd = loaded.state_dict()
    n_cmp = 0
    for k, v in lsd.items():
        acc = recorded[0][k].clone()
        for s in recorded[1:]:
            acc += s[k]
        hand_mean = (acc / len(recorded)).to(v.dtype)
        assert torch.equal(v, hand_mean), f"swa/ checkpoint != hand mean at {k}"
        n_cmp += 1
    print(f"  swa/ weights == hand mean of the {len(recorded)} live epoch-end "
          f"snapshots (torch.equal) on all {n_cmp} tensors")
    # the normal best checkpoint sits alongside, untouched
    best = AutoModelForTokenClassification.from_pretrained(out_dir)
    n_diff = sum(1 for k in lsd if not torch.equal(lsd[k], best.state_dict()[k]))
    print(f"  normal best checkpoint also present; tensors differing from swa: {n_diff}")
    print("  PASS: SWA average verified + swa/ checkpoint loads")

    print("\nSMOKE OK: defaults bit-for-bit, child-tune mask/hooks/update verified, "
          "SWA average verified end-to-end on CPU.")


if __name__ == "__main__":
    main()
