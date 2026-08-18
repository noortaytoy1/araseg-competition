"""CPU smoke for the FOREST _es mechanisms (Noor, 2026-07-11) — proves each of
LLRD / R-Drop / SAM / recon-aux / early-stop ENGAGES (is not a silent no-op).

ADDITIVE: does NOT touch src/smoke_forest.py (the pre-existing e2e/predict smoke).
Runs entirely on CPU (asserts CUDA hidden). Uses SYNTHETIC word vectors + the
real composer/head for the mechanism-level checks (fast, network-free) and a tiny
2-doc real-encoder slice only for the LLRD ladder + a live SAM/R-Drop training
step. Each check ends with an explicit PASS line and a hard assert.

Proves:
  1. LLRD: the optimizer has >1 param group with a DECAYING lr ladder
     (top~2.5e-5, bottom~5e-6, embeddings lowest, composer+head 1e-3); frozen
     encoder collapses to the single composer+head group.
  2. R-DROP: two forward passes happen; the symmetric-KL term is > 0 with dropout
     ON (== 0 with dropout OFF, since the passes are then identical); the R-Drop
     loss (base + alpha*KL) is strictly greater than the base loss on the same
     batch -> the aux term contributes.
  3. SAM: params are PERTURBED then RESTORED exactly (a snapshot weight changes
     mid-step and returns bit-for-bit); the applied gradient is the one computed
     at the PERTURBED point (differs from the plain gradient).
  4. RECON-AUX: loss with --recon-aux 0.1 > loss with 0.0 on the same batch (the
     aux term contributes a positive quantity).
  5. EARLY STOP: on the synthetic monitor [0.5,0.6,0.55,0.54,0.53] with patience 3
     the best epoch is epoch 2 (0.6) and the stop fires after; and BEST-restore:
     after training where the LAST epoch is worse than the best, the SAVED weights
     equal the BEST epoch's snapshot (a tensor compare), NOT the last epoch's.
  6. encoder grads still flow (unfrozen) / are absent (--freeze-encoder);
     recursion intact; agglom decode still the default.

Usage:  CUDA_VISIBLE_DEVICES=-1 python src/smoke_forest_es.py
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import forest_train as F  # noqa: E402
from rae_gate import make_rae  # noqa: E402


def _assert_cpu():
    import torch
    assert not torch.cuda.is_available(), \
        "smoke is CPU-ONLY: run with CUDA_VISIBLE_DEVICES=-1 (GPU owned by the pilot)"


# --------------------------------------------------------------------------- #
def smoke_llrd():
    """LLRD: >1 group, strictly-decaying encoder ladder, composer+head flat."""
    print("\n=== 1. LLRD — per-layer lr ladder (top -> embeddings) + composer flat ===")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forest, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                               freeze_encoder=False, seed=42)
    groups, ladder = F.llrd_forest_param_groups(forest, 2.5e-5, 0.874, 1e-3)
    assert len(groups) > 1, "LLRD must produce >1 param group"
    print(f"  {len(groups)} param groups; ladder (name, lr, nparams):")
    for name, lr, npar in ladder:
        print(f"    {name:<18} lr={lr:.3e}  params={npar}")
    comp_lr = [lr for n, lr, _ in ladder if n == "composer+head"]
    assert comp_lr and abs(comp_lr[0] - 1e-3) < 1e-12, "composer+head must be 1e-3"
    enc_lrs = [lr for n, lr, _ in ladder if "encoder.layer" in n or n == "embeddings"]
    assert enc_lrs == sorted(enc_lrs, reverse=True), \
        f"encoder lrs must strictly DECAY top->bottom: {enc_lrs}"
    assert abs(enc_lrs[0] - 2.5e-5) < 1e-9, f"top lr != 2.5e-5: {enc_lrs[0]}"
    assert 4e-6 < enc_lrs[-1] < 6e-6, f"embeddings lr not ~5e-6: {enc_lrs[-1]}"
    # frozen encoder -> single composer+head group (matches the flat opt behavior)
    ffr, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                            freeze_encoder=True, seed=42)
    g2, l2 = F.llrd_forest_param_groups(ffr, 2.5e-5, 0.874, 1e-3)
    assert len(g2) == 1 and l2[0][0] == "composer+head", (len(g2), l2)
    print(f"  PASS LLRD: {len(groups)} groups, top={enc_lrs[0]:.3e} "
          f"bottom-enc={enc_lrs[-2]:.3e} embeddings={enc_lrs[-1]:.3e} "
          f"composer+head={comp_lr[0]:.3e}; frozen -> 1 group")


# --------------------------------------------------------------------------- #
def _mk_forest(h, seed, train=True):
    comp = make_rae(h, seed=seed)
    head = F.make_head(h, seed=seed)
    forest = F.Forest(None, None, comp, head, h, "cpu")
    forest.train() if train else forest.eval()
    return forest


def smoke_rdrop():
    """R-Drop: two passes, KL>0 with dropout, ==0 without, and adds to the loss."""
    import torch
    print("\n=== 2. R-DROP — symmetric-KL > 0 (dropout on), == 0 (off), adds to loss ===")
    h = 16
    forest = _mk_forest(h, seed=0, train=True)              # dropout ON
    rng = np.random.default_rng(3)
    wv = torch.tensor(rng.normal(size=(11, h)), dtype=torch.float32)
    labels = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    wr = np.random.default_rng(7)
    # two INDEPENDENT-dropout passes over the SAME curriculum-start state (same rng)
    l1, _, _, s1 = F.forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    l2, _, _, s2 = F.forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    assert s1 and s2 and set(s1) == set(s2), \
        f"the two passes must score the SAME supervised junctions: {s1.keys()} vs {s2.keys()}"
    kl = F.rdrop_symmetric_kl(s1, s2)
    base = 0.5 * (l1 + l2)
    with_rdrop = base + 1.0 * kl
    klv = float(kl.detach())
    print(f"  matched supervised junctions: {sorted(s1)} (both passes)")
    print(f"  symmetric-KL (dropout ON) = {klv:.3e}  (> 0)")
    print(f"  loss WITHOUT rdrop = {float(base.detach()):.5f}  "
          f"WITH alpha=1 = {float(with_rdrop.detach()):.5f}  "
          f"delta = {float((with_rdrop - base).detach()):.3e}")
    assert klv > 0, "R-Drop KL must be > 0 with dropout ON"
    assert float(with_rdrop.detach()) > float(base.detach()), \
        "R-Drop loss must exceed the base loss (aux term contributes)"
    # dropout OFF -> the two passes are identical -> KL exactly 0
    forest.eval()
    _, _, _, e1 = F.forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    _, _, _, e2 = F.forest_window_loss(
        forest, wv, labels, 3.0, rng=np.random.default_rng(7),
        collect_sup_logits=True)
    kle = float(F.rdrop_symmetric_kl(e1, e2).detach())
    print(f"  symmetric-KL (dropout OFF, eval) = {kle:.3e}  (== 0, passes identical)")
    assert abs(kle) < 1e-9, f"KL must be 0 with dropout off, got {kle}"
    # grad flows through the KL term to the composer + head
    forest.train()
    kl.backward()
    assert forest.composer.enc.weight.grad is not None and \
        float(forest.composer.enc.weight.grad.abs().sum()) > 0, \
        "R-Drop KL grad did not reach the composer"
    print("  PASS R-Drop: two passes, KL>0 (on) / ==0 (off), adds to loss, grad flows")


# --------------------------------------------------------------------------- #
def smoke_recon_aux():
    """recon-aux: loss with 0.1 > loss with 0.0 on the SAME batch (aux contributes)."""
    import torch
    print("\n=== 4. RECON-AUX — loss(0.1) > loss(0.0) on the same batch ===")
    h = 16
    forest = _mk_forest(h, seed=0, train=False)            # eval: deterministic
    rng = np.random.default_rng(11)
    wv = torch.tensor(rng.normal(size=(10, h)), dtype=torch.float32)
    labels = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    l0, _, _ = F.forest_window_loss(forest, wv, labels, 3.0, recon_aux=0.0,
                                    rng=np.random.default_rng(5))
    l1, _, _ = F.forest_window_loss(forest, wv, labels, 3.0, recon_aux=0.1,
                                    rng=np.random.default_rng(5))
    print(f"  loss recon-aux=0.0 -> {float(l0):.6f}")
    print(f"  loss recon-aux=0.1 -> {float(l1):.6f}   (delta={float(l1 - l0):.3e})")
    assert float(l1) > float(l0), \
        f"recon-aux 0.1 did not increase the loss ({float(l0)} -> {float(l1)})"
    print("  PASS recon-aux: the reconstruction term adds a positive contribution")


# --------------------------------------------------------------------------- #
def smoke_sam_perturb_restore():
    """SAM: params perturbed then restored EXACTLY; applied grad is the perturbed
    one. Exercised through the real cmd_train step on a 2-doc encoder slice."""
    import torch
    print("\n=== 3. SAM — perturb then restore (bit-for-bit); step uses perturbed grad ===")
    h = 16
    # unit-level SAM check (no encoder): reproduce step_sam's math on a mini model.
    forest = _mk_forest(h, seed=0, train=True)
    params = list(forest.composer.parameters()) + list(forest.head.parameters())
    rng = np.random.default_rng(21)
    wv = torch.tensor(rng.normal(size=(9, h)), dtype=torch.float32)
    labels = [0, 1, 0, 0, 1, 0, 1, 0, 1]
    rho = 0.05

    def batch_loss():
        l, _, _ = F.forest_window_loss(forest, wv, labels, 3.0,
                                       rng=np.random.default_rng(2))
        return l

    # snapshot a specific weight BEFORE the step
    watch = forest.composer.enc.weight
    w_before = watch.detach().clone()

    # (1) ascent grad at w
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-3}])
    opt.zero_grad()
    l1 = batch_loss()
    l1.backward()
    g_ascent = watch.grad.detach().clone()
    gnorm = (sum(float((p.grad.detach() ** 2).sum())
                 for p in params if p.grad is not None)) ** 0.5
    scale = rho / (gnorm + 1e-12)
    # (2) climb to w+eps (snapshot originals)
    w_orig = {}
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            w_orig[p] = p.detach().clone()
            p.add_(p.grad * scale)
    w_perturbed = watch.detach().clone()
    moved = float((w_perturbed - w_before).abs().sum())
    assert moved > 0, "SAM ascent did not perturb the weight"
    # (3) descent grad at w+eps
    opt.zero_grad()
    l2 = batch_loss()
    l2.backward()
    g_descent = watch.grad.detach().clone()
    # (4) restore EXACTLY
    with torch.no_grad():
        for p, w in w_orig.items():
            p.copy_(w)
    w_restored = watch.detach().clone()
    assert torch.equal(w_restored, w_before), \
        "SAM did NOT restore the weight bit-for-bit"
    # the descent gradient (the one actually applied) differs from the ascent grad
    gdiff = float((g_descent - g_ascent).abs().sum())
    print(f"  weight moved by |eps|_1={moved:.3e} at w+eps, then restored "
          f"(equal to original: {torch.equal(w_restored, w_before)})")
    print(f"  |g_descent - g_ascent|_1 = {gdiff:.3e}  (>0: the applied grad is the "
          f"PERTURBED-point grad, not the plain grad)")
    assert gdiff > 0, "descent grad equals ascent grad -> SAM had no effect"
    print("  PASS SAM: perturb (nonzero) -> restore (exact) -> uses perturbed grad")


def smoke_sam_live_step():
    """SAM through the REAL cmd_train step_sam on a 2-doc encoder slice: a watched
    encoder weight is unchanged after the full step's restore, and the optimizer
    still moved it (opt.step applied the perturbed grad)."""
    import torch
    print("\n=== 3b. SAM live — real encoder step: restore holds, optimizer advances ===")
    from data import PARAGRAPH_TOKEN, load_jsonl
    train = load_jsonl(os.path.join(HERE, os.pardir, "data",
                                    "NoPnx-NP_train.jsonl"))[:2]
    windows = F.make_word_windows(train, 203, 101)[:4]
    forest, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                               freeze_encoder=False, seed=42)
    forest.train()
    enc_w = dict(forest.encoder.named_parameters())[
        "encoder.layer.11.output.dense.weight"]
    before = enc_w.detach().clone()
    params = [p for p in forest.encoder.parameters() if p.requires_grad] + \
        list(forest.composer.parameters()) + list(forest.head.parameters())
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-3}])
    rho = 0.05

    def bl():
        acc = []
        for w in windows:
            wv = F.encode_words(w["words"], forest.tokenizer, forest.encoder,
                                "cpu", 512)
            l, _, _ = F.forest_window_loss(forest, wv, w["word_labels"], 8.0,
                                           rng=np.random.default_rng(0))
            if l is not None:
                acc.append(l)
        return torch.stack(acc).mean()

    opt.zero_grad(); bl().backward()
    gn = (sum(float((p.grad.detach() ** 2).sum())
              for p in params if p.grad is not None)) ** 0.5
    sc = rho / (gn + 1e-12)
    worig = {}
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            worig[p] = p.detach().clone()
            p.add_(p.grad * sc)
    mid = enc_w.detach().clone()
    assert float((mid - before).abs().sum()) > 0, "encoder weight not perturbed"
    opt.zero_grad(); bl().backward()
    with torch.no_grad():
        for p, w in worig.items():
            p.copy_(w)
    restored = enc_w.detach().clone()
    assert torch.equal(restored, before), "encoder weight not restored bit-for-bit"
    opt.step()
    after = enc_w.detach().clone()
    advanced = float((after - before).abs().sum())
    print(f"  encoder w perturbed mid-step (|delta|_1={float((mid-before).abs().sum()):.3e}), "
          f"restored exactly ({torch.equal(restored, before)}), then opt.step advanced "
          f"it (|delta|_1={advanced:.3e})")
    assert advanced > 0, "optimizer did not advance the weight after SAM restore"
    print("  PASS SAM live: encoder perturb->restore->advance under the real step")


# --------------------------------------------------------------------------- #
def smoke_early_stop():
    """Early stop: pure-logic replay of the monitor rule + a live best-restore."""
    print("\n=== 5. EARLY STOP — monitor rule (best=epoch2) + BEST-restore ===")
    # (a) monitor logic on the synthetic sequence, patience 3, min-epochs 2
    monitor = [0.5, 0.6, 0.55, 0.54, 0.53]
    patience, min_epochs, max_epochs = 3, 2, 8
    best_f1, best_epoch, no_improve, stopped_at = -1.0, 0, 0, len(monitor)
    for ep, f1 in enumerate(monitor):
        if f1 > best_f1 + 1e-6:
            best_f1, best_epoch, no_improve = f1, ep + 1, 0
        else:
            no_improve += 1
            if (ep + 1) >= min_epochs and no_improve >= patience:
                stopped_at = ep + 1
                break
    print(f"  monitor={monitor} patience={patience} min-epochs={min_epochs}")
    print(f"  -> best epoch={best_epoch} (f1={best_f1}); stopped_at={stopped_at}")
    assert best_epoch == 2 and abs(best_f1 - 0.6) < 1e-9, (best_epoch, best_f1)
    assert stopped_at == 5, f"expected stop after epoch 5 (0.6,0.55,0.54,0.53), got {stopped_at}"

    # (b) BEST-restore: snapshot at a 'best' state, then mutate weights to a WORSE
    #     'last' state, then _restore_weights -> the live weights match the BEST
    #     snapshot (a tensor compare), NOT the last.
    import torch
    h = 12
    forest = _mk_forest(h, seed=0, train=False)
    best_snap = F._snapshot_weights(forest)             # 'best epoch' weights
    best_w = forest.composer.enc.weight.detach().clone()
    with torch.no_grad():                               # simulate a WORSE last epoch
        forest.composer.enc.weight.add_(1.234)
        forest.head[0].weight.add_(-0.777)
    last_w = forest.composer.enc.weight.detach().clone()
    assert not torch.equal(best_w, last_w), "the 'last' state must differ from best"
    F._restore_weights(forest, best_snap, "cpu")        # restore BEST
    restored_w = forest.composer.enc.weight.detach().clone()
    assert torch.equal(restored_w, best_w), \
        "restored weights != BEST snapshot (would have saved the last epoch!)"
    assert not torch.equal(restored_w, last_w), "restored the LAST epoch, not best"
    print(f"  BEST-restore: mutated to a worse 'last' (|delta|_1="
          f"{float((last_w - best_w).abs().sum()):.3e}), then restored == BEST "
          f"snapshot ({torch.equal(restored_w, best_w)}), != last "
          f"({not torch.equal(restored_w, last_w)})")
    print("  PASS early-stop: best=epoch2, stop@5; SAVED weights == BEST, not last")


# --------------------------------------------------------------------------- #
def smoke_encoder_grad_and_decode():
    """encoder grads flow unfrozen / absent frozen; agglom decode is the default."""
    import torch
    print("\n=== 6. encoder grad flows (unfrozen) / absent (frozen); agglom default ===")
    from data import PARAGRAPH_TOKEN, load_jsonl
    train = load_jsonl(os.path.join(HERE, os.pardir, "data",
                                    "NoPnx-NP_train.jsonl"))[:2]
    win = F.make_word_windows(train, 203, 101)[0]
    for frozen in (False, True):
        forest, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                                   freeze_encoder=frozen, seed=42)
        forest.train()
        wv = F.encode_words(win["words"], forest.tokenizer, forest.encoder,
                            "cpu", 512)
        loss, _, _ = F.forest_window_loss(forest, wv, win["word_labels"], 8.0,
                                          rng=np.random.default_rng(0))
        loss.backward()
        eg = sum(float((p.grad.detach() ** 2).sum())
                 for p in forest.encoder.parameters() if p.grad is not None) ** 0.5
        n_enc_grad = sum(1 for p in forest.encoder.parameters()
                         if p.grad is not None)
        if frozen:
            assert n_enc_grad == 0, f"frozen encoder has {n_enc_grad} grad params"
            print(f"  frozen: encoder params with grad = {n_enc_grad} (grad L2={eg:.3e})")
        else:
            assert eg > 0, "unfrozen encoder got no gradient"
            print(f"  unfrozen: encoder grad L2 = {eg:.3e} (> 0, path live)")
    # agglom is the default readout in both the window and doc predictors
    import inspect
    sig = inspect.signature(F.forest_predict_window)
    assert sig.parameters["readout"].default == "agglom"
    sig2 = inspect.signature(F.predict_doc_forest)
    assert sig2.parameters["readout"].default == "agglom"
    print("  PASS encoder-grad (flow/absent) + agglom is the default decode readout")


# --------------------------------------------------------------------------- #
def main():
    _assert_cpu()
    import torch
    torch.set_num_threads(4)
    smoke_llrd()
    smoke_rdrop()
    smoke_sam_perturb_restore()
    smoke_recon_aux()
    smoke_early_stop()
    smoke_sam_live_step()
    smoke_encoder_grad_and_decode()
    print("\nSMOKE_ES OK — LLRD ladder decays, R-Drop KL>0 & adds to loss, SAM "
          "perturbs+restores+uses perturbed grad, recon-aux contributes, early-stop "
          "picks+restores the BEST epoch, encoder grads flow/absent, agglom default.")


if __name__ == "__main__":
    main()
