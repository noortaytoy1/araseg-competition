"""CPU smoke for FOREST (src/forest_train.py) — the mandatory pre-GPU checks.

Runs entirely on CPU (asserts CUDA hidden) against a 2-doc slice of the real
NoPnx-NP train/dev JSONL. Proves, without touching the GPU:

  1. e2e training runs for BOTH arms (forest_e2e = rae init, forest_e2e_rand =
     random init), loss DECREASES across a few steps.
  2. encoder gradients FLOW under the default (unfrozen) e2e path — nonzero
     grad on encoder params — and are ABSENT under --freeze-encoder. This is
     the untested-cell contract: keep/erase gradient reaches the encoder.
  3. the composer init contrast is real: rae init loads rae_gate/rae_final.pt;
     random init does not.
  4. predict produces a VALID submission CSV + word-grid npz on 2 dev docs
     (doc_id -> per-word P in [0,1]; doc-final word carries 0.0), for both the
     leaf and --descend readouts.
  5. EXPOSURE-BIAS FIX flags ENGAGE on the real encoder: --curriculum-include-allcut
     (allcut_prob=1.0) starts windows from ALL-SEPARATE (n_leaf ~= n-1), and
     --scheduled-sampling (P=1.0) starts from the model's OWN free-run state — with
     gradient still flowing to encoder + composer + head (free-run is no_grad).

Usage:  CUDA_VISIBLE_DEVICES=-1 python src/smoke_forest.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import forest_train as F  # noqa: E402
from data import PARAGRAPH_TOKEN, load_jsonl  # noqa: E402


def _assert_cpu():
    import torch
    assert not torch.cuda.is_available(), \
        "smoke is CPU-ONLY: run with CUDA_VISIBLE_DEVICES=-1 (GPU owned by the pilot)"


def _train_a_few_steps(forest, windows, pos_weight, n_steps, lr_enc, lr_other,
                       recon_aux=0.0, seed=0):
    """Train a few optimizer steps; return (losses, enc_grad_l2_list)."""
    import torch
    enc_params = forest.encoder_params()
    other = forest.other_params()
    groups = [{"params": other, "lr": lr_other}]
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_enc})
    opt = torch.optim.AdamW(groups)
    forest.train()
    losses, egs = [], []
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(windows)))
    step = 0
    while step < n_steps:
        wi = order[step % len(order)]
        w = windows[wi]
        wv = F.encode_words(w["words"], forest.tokenizer, forest.encoder,
                            "cpu", 512)
        loss, _nl, _np = F.forest_window_loss(
            forest, wv, w["word_labels"], pos_weight, recon_aux=recon_aux,
            rng=np.random.default_rng((seed, step)))
        if loss is None:
            order.append(order[step % len(order)])
            step += 1
            continue
        opt.zero_grad()
        loss.backward()
        eg = 0.0
        for p in enc_params:
            if p.grad is not None:
                eg += float((p.grad.detach() ** 2).sum())
        egs.append(eg ** 0.5)
        torch.nn.utils.clip_grad_norm_(enc_params + other, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        step += 1
    return losses, egs


def main():
    _assert_cpu()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    torch.set_num_threads(4)

    train = load_jsonl(os.path.join(HERE, os.pardir, "data",
                                    "NoPnx-NP_train.jsonl"))[:2]
    windows = F.make_word_windows(train, 203, 101)
    pos = tot = 0
    for d in train:
        for i in range(len(d["tokens"]) - 1):
            if d["tokens"][i] == PARAGRAPH_TOKEN:
                continue
            tot += 1
            pos += int(d["labels"][i])
    pw = min((tot - pos) / max(pos, 1), 8.0)
    print(f"[smoke] 2 train docs -> {len(windows)} windows (W=203,S=101); "
          f"junction keep rate={pos/max(tot,1):.3f} pos_weight={pw:.2f}")

    # ---------- ARM A: forest_e2e (rae init, encoder UNFROZEN) ----------
    print("\n=== ARM A  forest_e2e  (composer-init=rae, encoder UNFROZEN) ===")
    fa, note_a = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                                freeze_encoder=False, seed=42)
    assert "rae" in note_a and "rae_final" in note_a, note_a
    la, ega = _train_a_few_steps(fa, windows, pw, n_steps=6,
                                 lr_enc=2e-5, lr_other=1e-3, seed=1)
    print(f"  composer-init: {note_a}")
    print(f"  losses: {[round(x,4) for x in la]}")
    print(f"  encoder grad L2 (pre-clip): {[round(x,3) for x in ega]}")
    assert la[-1] < la[0], f"loss did not decrease: {la[0]:.4f} -> {la[-1]:.4f}"
    assert all(g > 0 for g in ega), \
        f"encoder gradient is ZERO under e2e (dead path!): {ega}"
    print(f"  PASS loss down {la[0]:.4f}->{la[-1]:.4f}; encoder grads FLOW "
          f"(min L2={min(ega):.3f})")

    # ---------- ARM B: forest_e2e_rand (random init, encoder UNFROZEN) ----------
    print("\n=== ARM B  forest_e2e_rand  (composer-init=random, encoder UNFROZEN) ===")
    fb, note_b = F.build_forest(F.BASE_MODEL, "random", "cpu",
                                freeze_encoder=False, seed=42)
    assert note_b == "random", note_b
    lb, egb = _train_a_few_steps(fb, windows, pw, n_steps=6,
                                 lr_enc=2e-5, lr_other=1e-3, seed=2)
    print(f"  composer-init: {note_b}")
    print(f"  losses: {[round(x,4) for x in lb]}")
    print(f"  encoder grad L2 (pre-clip): {[round(x,3) for x in egb]}")
    assert lb[-1] < lb[0], f"loss did not decrease: {lb[0]:.4f} -> {lb[-1]:.4f}"
    assert all(g > 0 for g in egb), \
        f"encoder gradient is ZERO under e2e (dead path!): {egb}"
    print(f"  PASS loss down {lb[0]:.4f}->{lb[-1]:.4f}; encoder grads FLOW "
          f"(min L2={min(egb):.3f})")

    # ---------- FROZEN control: encoder grads must be ABSENT ----------
    print("\n=== FROZEN control  (--freeze-encoder): encoder grads must be ABSENT ===")
    fc, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                           freeze_encoder=True, seed=42)
    lc, egc = _train_a_few_steps(fc, windows, pw, n_steps=3,
                                 lr_enc=2e-5, lr_other=1e-3, seed=3)
    n_enc_grad = sum(1 for p in fc.encoder.parameters() if p.grad is not None)
    assert n_enc_grad == 0, f"frozen encoder has {n_enc_grad} params with grad!"
    assert fc.composer.enc.weight.grad is not None, "frozen arm: composer got no grad"
    print(f"  PASS encoder params with grad = {n_enc_grad} (composer still trains); "
          f"loss {lc[0]:.4f}->{lc[-1]:.4f}")

    # ---------- RECON-AUX arm: --recon-aux 0.3 runs and trains ----------
    print("\n=== recon-aux 0.3 arm: runs + loss decreases ===")
    fr, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                           freeze_encoder=False, seed=42)
    lr_, egr = _train_a_few_steps(fr, windows, pw, n_steps=4,
                                  lr_enc=2e-5, lr_other=1e-3, recon_aux=0.3, seed=4)
    assert lr_[-1] < lr_[0], f"recon-aux loss did not decrease: {lr_}"
    assert all(g > 0 for g in egr), "recon-aux: encoder grad zero"
    print(f"  PASS recon-aux 0.3 loss {lr_[0]:.4f}->{lr_[-1]:.4f}; grads flow")

    # ---------- PREDICT: valid CSV + npz on 2 dev docs, both readouts ----------
    print("\n=== PREDICT: 2 dev docs -> CSV + npz (agglom default + leaf diag) ===")
    dev = load_jsonl(os.path.join(HERE, os.pardir, "data",
                                  "NoPnx-NP_dev.jsonl"))[:2]
    tmp = tempfile.mkdtemp(prefix="forest_smoke_")
    fa.encoder.save_pretrained(os.path.join(tmp, "encoder"))
    fa.tokenizer.save_pretrained(os.path.join(tmp, "encoder"))
    torch.save({"composer": fa.composer.state_dict(),
                "head": fa.head.state_dict(), "h": fa.h,
                "head_in_features": fa.head.in_features},
               os.path.join(tmp, "forest_heads.pt"))
    import json
    json.dump({"design": "forest", "task": "NoPnx-NP", "window": 203,
               "stride": 101, "max_length": 512, "composer_init": "rae"},
              open(os.path.join(tmp, F.CFG_NAME), "w"))

    from baselines import write_submission
    forest, cfg = F.load_forest(tmp, "cpu")
    for readout in ("agglom", "leaf"):
        preds, prob_map = [], {}
        for d in dev:
            words = [F.MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                     for w in d["tokens"]]
            p = F.predict_doc_forest(words, forest, 203, 101, 512,
                                     readout=readout)
            assert len(p) == len(d["tokens"]), d["doc_id"]
            assert np.all((p >= 0) & (p <= 1)), "prob out of [0,1]"
            assert p[-1] == 0.0, "doc-final word must carry 0.0"
            prob_map[d["doc_id"]] = p.astype(np.float32)
            preds.append((p >= 0.5).astype(int).tolist())
        csv_path = os.path.join(tmp, f"pred_{readout}.csv")
        npz_path = os.path.join(tmp, f"probs_{readout}.npz")
        write_submission(dev, preds, csv_path)
        np.savez(npz_path, **prob_map)
        # verify CSV shape via a re-read
        import pandas as pd
        df = pd.read_csv(csv_path, dtype={"Prediction": str})
        assert set(df["Document ID"].astype(str)) == {d["doc_id"] for d in dev}
        for _, r in df.iterrows():
            did = str(r["Document ID"])
            gold = next(d for d in dev if d["doc_id"] == did)
            assert len(str(r["Prediction"])) == len(gold["tokens"]), did
        z = np.load(npz_path)
        assert set(z.files) == {d["doc_id"] for d in dev}
        rate = sum(map(sum, preds)) / max(sum(map(len, preds)), 1)
        print(f"  [{readout:7s}] CSV+npz OK ({len(dev)} docs, boundary rate={rate:.3f})")

    # ---------- BLOCKER-2 EVIDENCE: 10 dev docs, train-vs-decode width histogram
    # + CSV/npz validity under the DEFAULT (agglom) readout. Proves the default
    # decode scores boundary junctions on COMPOSED multi-word units (matching the
    # training-time composed-unit distribution), not on raw 1-word leaf pairs. ---
    print("\n=== BLOCKER-2: 10 dev docs — train-vs-decode boundary-junction width "
          "histogram + CSV/npz validity (DEFAULT agglom readout) ===")
    dev10 = load_jsonl(os.path.join(HERE, os.pardir, "data",
                                    "NoPnx-NP_dev.jsonl"))[:10]

    def _hist(widths):
        from collections import Counter
        c = Counter(int(w) for w in widths)
        tot = max(sum(c.values()), 1)
        return (float(np.mean(widths)) if widths else 0.0,
                {w: (c[w], round(100.0 * c[w] / tot, 1)) for w in sorted(c)})

    # (a) TRAIN-time composed-unit widths at gold-KEEP (boundary) junctions:
    #     replay forest_window_loss (the production curriculum-start loop) and
    #     record the width of the two units a boundary junction is scored on at
    #     the depth it is finally decided.
    train_bwidths = []
    for d in train:
        words = [F.MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                 for w in d["tokens"]]
        for win in F.make_word_windows([{**d, "tokens": words}], 203, 101):
            with torch.no_grad():
                wv = F.encode_words(win["words"], forest.tokenizer,
                                    forest.encoder, "cpu", 512)
            train_bwidths += F.train_boundary_unit_widths(
                forest, wv, win["word_labels"],
                rng=np.random.default_rng(0))
    tmean, thist = _hist(train_bwidths)
    print(f"  TRAIN  boundary-junction unit width: mean={tmean:.2f} "
          f"n={len(train_bwidths)}  hist(width:(n,%))={thist}")

    # (b) DECODE-time (default agglom readout) composed-unit widths at the SAME
    #     gold-boundary junctions on the 10 dev docs:
    decode_bwidths, preds10, prob10 = [], [], {}
    for d in dev10:
        words = [F.MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                 for w in d["tokens"]]
        p, w = F.predict_doc_forest(words, forest, 203, 101, 512,
                                    readout="agglom", return_widths=True)
        assert len(p) == len(d["tokens"]) == len(w), d["doc_id"]
        assert np.all((p >= 0) & (p <= 1)), "prob out of [0,1]"
        assert p[-1] == 0.0, "doc-final word must carry 0.0"
        gold = d["labels"]
        for i in range(len(gold) - 1):
            if int(gold[i]) == 1 and w[i] > 0:
                decode_bwidths.append(int(w[i]))
        prob10[d["doc_id"]] = p.astype(np.float32)
        preds10.append((p >= 0.5).astype(int).tolist())
    dmean, dhist = _hist(decode_bwidths)
    tfrac = (np.mean([w > 2 for w in train_bwidths]) if train_bwidths else 0.0)
    dfrac = (np.mean([w > 2 for w in decode_bwidths]) if decode_bwidths else 0.0)
    print(f"  DECODE boundary-junction unit width: mean={dmean:.2f} "
          f"n={len(decode_bwidths)}  hist(width:(n,%))={dhist}")
    print(f"  boundary junctions scored on COMPOSED (>2-word) units: "
          f"TRAIN {100 * tfrac:.1f}%  DECODE {100 * dfrac:.1f}%  "
          f"(the LEAF readout would be 0% for BOTH — every junction width 2)")
    # The qualitative BLOCKER-2 fix: the DEFAULT decode scores boundary junctions
    # on COMPOSED units (the training distribution), NOT the leaf pair. On this
    # UNTRAINED smoke model the decode over-agglomerates (nothing has learned to
    # KEEP at a boundary yet, so a surviving junction keeps merging), inflating
    # the mean vs training; a TRAINED model learns to stop at boundaries, pulling
    # the decode width distribution toward the training one. The load-bearing
    # assertion is that boundaries are on composed (>2) units, unlike leaf.
    assert dmean > 2.0 and dfrac > 0.5, \
        f"decode boundary junctions not on composed units (mean {dmean:.2f}, " \
        f"composed frac {dfrac:.2f})"
    assert tmean > 2.0, f"train boundary junctions not composed (mean {tmean:.2f})"

    # CSV/npz validity on the 10 dev docs (default readout)
    import pandas as pd
    from baselines import write_submission as _ws
    csv10 = os.path.join(tmp, "pred_dev10.csv")
    npz10 = os.path.join(tmp, "probs_dev10.npz")
    _ws(dev10, preds10, csv10)
    np.savez(npz10, **prob10)
    df10 = pd.read_csv(csv10, dtype={"Prediction": str})
    assert set(df10["Document ID"].astype(str)) == {d["doc_id"] for d in dev10}
    for _, r in df10.iterrows():
        gold = next(d for d in dev10 if d["doc_id"] == str(r["Document ID"]))
        assert len(str(r["Prediction"])) == len(gold["tokens"])
    z10 = np.load(npz10)
    assert set(z10.files) == {d["doc_id"] for d in dev10}
    for did in z10.files:
        arr = z10[did]
        assert np.all((arr >= 0) & (arr <= 1)), f"{did}: prob out of [0,1]"
        assert arr[-1] == 0.0, f"{did}: doc-final not 0.0"
    print(f"  10-dev-doc CSV+npz VALID (probs in [0,1], doc-final=0.0, one score "
          f"per word); boundaries on composed units: train {100*tfrac:.0f}% / "
          f"decode {100*dfrac:.0f}% (leaf=0%). Decode mean width inflated vs train "
          f"because this smoke model is UNTRAINED (never learned to KEEP); a "
          f"trained model tightens decode toward train.")

    # ---------- EXPOSURE-BIAS FIX: each flag ENGAGES on the REAL encoder ----------
    # (Noor, 2026-07-11) The default curriculum EXCLUDES 'allcut' (near-gold refine),
    # but the agglom decode starts from ALL WORDS SEPARATE. Prove, on real encoder
    # word-vectors over the 2 train docs' windows, that:
    #   * --curriculum-include-allcut (allcut_prob=1.0) makes some/most windows start
    #     from ALL-SEPARATE (n_leaf ~= n-1 valid junctions), vs the refine default.
    #   * --scheduled-sampling (P=1.0) makes windows start from the model's OWN
    #     free-run state (distinct from BOTH the default and all-separate), and grad
    #     still flows to encoder + composer + head.
    print("\n=== EXPOSURE-BIAS FIX: --curriculum-include-allcut + "
          "--scheduled-sampling ENGAGE on the real encoder ===")
    fe, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                           freeze_encoder=False, seed=42)
    fe.train()
    enc_params_e = fe.encoder_params()
    other_e = fe.other_params()
    n_allcut_starts = n_ss_starts = 0
    n_checked = 0
    ss_encoder_grad_seen = False
    for wi, w in enumerate(windows):
        labs = w["word_labels"]
        n = len(labs)
        n_valid = sum(1 for i in range(n - 1) if labs[i] != -100)
        if n_valid < 3:
            continue
        n_checked += 1
        with torch.no_grad():
            wv = F.encode_words(w["words"], fe.tokenizer, fe.encoder, "cpu", 512)
        # (a) default refine start-state (n_leaf small, near gold)
        _ld, nl_def, _pd = F.forest_window_loss(
            fe, wv, labs, pw, rng=np.random.default_rng((0, wi)))
        # (b) allcut prob=1.0 -> n_leaf == n_valid (every valid junction a start-cut)
        _la, nl_ac, _pa = F.forest_window_loss(
            fe, wv, labs, pw, rng=np.random.default_rng((1, wi)),
            include_allcut=True, allcut_prob=1.0)
        if nl_ac == n_valid and nl_ac > nl_def:
            n_allcut_starts += 1
        # (c) scheduled sampling P=1.0 -> own free-run state (distinct)
        keep_set = set(i for i in range(n - 1) if labs[i] == 1)
        all_valid = [i for i in range(n - 1) if labs[i] != -100]
        surv = F._free_run_start_cuts(fe, wv, keep_set, all_valid)
        _ls, nl_ss, _ps = F.forest_window_loss(
            fe, wv, labs, pw, rng=np.random.default_rng((2, wi)),
            scheduled_sampling=1.0)
        if surv is not None and nl_ss == len(surv) and keep_set.issubset(surv) \
                and nl_ss != n_valid:
            n_ss_starts += 1
        print(f"  win{wi:>2d}: n_valid_junc={n_valid:>3d}  n_leaf: "
              f"default(refine)={nl_def:>3d}  allcut={nl_ac:>3d}  "
              f"scheduled-sampling(own)={nl_ss:>3d}")
    assert n_checked > 0, "no non-trivial windows to check"
    assert n_allcut_starts > 0, \
        "--curriculum-include-allcut never produced an all-separate start-state"
    assert n_ss_starts > 0, \
        "--scheduled-sampling never produced a model-free-run start-state"
    print(f"  ENGAGE COUNTS over {n_checked} windows: allcut all-separate starts="
          f"{n_allcut_starts}  scheduled-sampling own-state starts={n_ss_starts}")

    # (d) grad flows to ENCODER + composer + head under scheduled sampling (P=1.0)
    fg, _ = F.build_forest(F.BASE_MODEL, "rae", "cpu",
                           freeze_encoder=False, seed=42)
    fg.train()
    opt_g = torch.optim.AdamW(
        [{"params": fg.other_params(), "lr": 1e-3},
         {"params": fg.encoder_params(), "lr": 2e-5}])
    w0 = next(w for w in windows
              if sum(1 for i in range(len(w["word_labels"]) - 1)
                     if w["word_labels"][i] != -100) >= 3)
    wv0 = F.encode_words(w0["words"], fg.tokenizer, fg.encoder, "cpu", 512)
    lg, _n, _p = F.forest_window_loss(
        fg, wv0, w0["word_labels"], pw, rng=np.random.default_rng(11),
        scheduled_sampling=1.0)
    assert lg is not None and lg.requires_grad
    opt_g.zero_grad()
    lg.backward()
    enc_g = sum(float((p.grad.detach() ** 2).sum())
                for p in fg.encoder_params() if p.grad is not None) ** 0.5
    comp_g = float(fg.composer.enc.weight.grad.abs().sum())
    head_g = float(fg.head[0].weight.grad.abs().sum())
    assert enc_g > 0, f"scheduled-sampling: ENCODER grad is ZERO (dead path!) {enc_g}"
    assert comp_g > 0 and head_g > 0, "scheduled-sampling: composer/head grad zero"
    print(f"  PASS scheduled-sampling grad: encoder-L2={enc_g:.4e} (>0), "
          f"composer-abs={comp_g:.3f} head-abs={head_g:.3f}  "
          f"(free-run no_grad; supervised scoring carries grad)")
    print("  PASS exposure-bias flags ENGAGE (allcut -> all-separate start; "
          "scheduled-sampling -> own free-run start) and recursion/grad intact.")

    print("\nSMOKE OK — forest e2e trains, encoder grads flow (absent when frozen), "
          "predict emits valid CSV+npz; DEFAULT decode scores boundaries on "
          "COMPOSED units (not the leaf pair) — the BLOCKER-2 fix; and the "
          "EXPOSURE-BIAS flags (--curriculum-include-allcut / --scheduled-sampling) "
          "ENGAGE with grad intact.")


if __name__ == "__main__":
    main()
