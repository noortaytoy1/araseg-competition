"""2D U-NET ARM over mid-layer-8 similarity matrices (additive, CPU-only).

EARNED by the pre-registered gates on record:
  * train gate  (diag_simmatrix.py):        block-contrast AUC 0.7641 (mid8, w=3)
  * dev gate    (diag_simmatrix_devgate.py): confident-TP vs confident-FP
                                             AUC 0.6212 >= 0.62 -> "the 2D U-Net
                                             arm gets built"

The encoder stays FROZEN (runs/union-NoPnx-NP-arabertv02-s42); only the small
U-Net on top of its similarity matrices is trained. Train windows only; dev is
used for EVALUATION and for checkpoint SELECTION only (dev doc-F1 selection is
the repo convention for every trainer — documented dev-selection caveat).

DATA (per window, standard windowing 180/60 exactly as diag_simmatrix.py
enumerates it — actual count: 821 train / 1,078 dev windows, not the ~1,020
estimated in the diag docstring):
  channel 1: 180x180 cosine-similarity matrix of mid-layer-8 word states
             (last-subword grid, L2-normalized) — float16 on disk.
  channel 2: the tagger's per-token boundary prob as a ROW-PLUS-COLUMN OUTER
             TRACK:  X2[i,j] = 0.5 * (p[i] + p[j]).   DECISION: this (single
             channel) rather than separate row+col tiled channels, keeping the
             input at exactly 2 channels as specified.
             p is the SINGLE-WINDOW tagger prob from the same frozen forward
             pass that produced the hidden states (self-consistent between
             train and dev; the overlap-averaged decode grid is used only for
             the eval-time confident-error slices and the ensemble probe).
  target:    the window's gold per-position boundary vector (labels[i]=1 =
             boundary FOLLOWS token i). Padded/truncated tails are masked out
             of the loss.

MODEL — small 2D U-Net, 2 down/up blocks (3 scales: 180 -> 90 -> 45),
base=48 channels => 1,050,673 params (within the specified ~1-3M):
  enc1(2->48) -> pool -> enc2(48->96) -> pool -> enc3(96->192)
  -> up+skip dec2(->96) -> up+skip dec1(->48) -> 1x1 conv -> 180x180 map.
READOUT DECISION: per-position boundary logits = the DIAGONAL of the final
1x1-conv output map (logit[i] = map[i,i]) — the simpler of the two allowed
readouts (vs a separate vector head off bottleneck+skips); the block corner
for a boundary after token i sits at (i,i), and the receptive field at the
diagonal spans both neighbouring blocks.

LOSS: BCEWithLogits, pos_weight=8 (train boundary rate 0.104 -> neg/pos ~ 8.6),
masked to valid positions. Adam, constant lr. Early stopping + checkpoint
selection on dev doc-macro-F1 @0.5 (stitched), patience 8.

ABLATION FLAG (additive, default off): train --no-tagger-channel drops
channel 2 entirely — input = the similarity matrix ONLY (in_ch=1; the first
conv adapts). The checkpoint records in_ch=1 and eval auto-detects it (old
2-channel ckpts have no in_ch key and load exactly as before). This arm is
NOT expected to win solo; it exists to test decorrelation from the tagger
(veto rate at the tagger's confident FPs vs endorsement at confident TPs).

EVAL / STITCHING: window probs are merged back onto documents by AVERAGING
over all covering windows — the same vote-merging convention as
predict.predict_doc / cache_probs.py. Positions never covered (truncation
tails) get p=0.0 and are counted in the report.

Stages (CPU only — the battery owns the GPU; caches under unet2d/, ckpts under
runs/unet2d_<date>/ which is a NEW dir, runs/ stays append-only):

  CUDA_VISIBLE_DEVICES=-1 python src/unet2d_train.py self-test

  CUDA_VISIBLE_DEVICES=-1 python src/unet2d_train.py cache \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --jsonl data/NoPnx-NP_train.jsonl --out unet2d/NoPnx-NP_s42_train_mid8.npz

  CUDA_VISIBLE_DEVICES=-1 python src/unet2d_train.py cache \
      --jsonl data/NoPnx-NP_dev.jsonl --out unet2d/NoPnx-NP_s42_dev_mid8.npz

  CUDA_VISIBLE_DEVICES=-1 python src/unet2d_train.py train \
      --train-cache unet2d/NoPnx-NP_s42_train_mid8.npz \
      --dev-cache unet2d/NoPnx-NP_s42_dev_mid8.npz \
      --dev-jsonl data/NoPnx-NP_dev.jsonl --seed 42 \
      --out-dir runs/unet2d_2026-07-10/s42

  CUDA_VISIBLE_DEVICES=-1 python src/unet2d_train.py eval \
      --ckpts runs/unet2d_2026-07-10/s42/best.pt \
      --dev-cache unet2d/NoPnx-NP_s42_dev_mid8.npz \
      --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --out runs/unet2d_2026-07-10/eval_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from data import PARAGRAPH_TOKEN, load_jsonl
from diag_simmatrix import (cosine_matrix, encode_window, enumerate_windows,
                            load_encoder, rank_auc)
from knn_boundary import THRESH_GRID, macro_f1

WIDE_GRID = [round(t, 2) for t in np.arange(0.05, 0.9501, 0.05)]


def assert_cpu_only():
    import torch
    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the battery owns the GPU)"


# --------------------------------------------------------------------------- #
# cache: one frozen-encoder CPU pass -> fp16 similarity matrices on disk
# --------------------------------------------------------------------------- #
def cmd_cache(args):
    assert_cpu_only()
    docs = load_jsonl(args.jsonl)
    wins = enumerate_windows(docs, args.window, args.overlap,
                             min_len=args.min_len)
    W = args.window
    N = len(wins)
    print(f"# {len(docs)} docs -> {N} windows (window={W}, "
          f"stride={W - args.overlap}, min_len={args.min_len})", flush=True)

    tokenizer, model = load_encoder(args.model)
    n_layers = model.config.num_hidden_layers
    assert 0 < args.mid_layer <= n_layers, "bad --mid-layer"
    print(f"# encoder: {args.model} ({n_layers} layers, "
          f"mid layer = {args.mid_layer})", flush=True)

    S = np.zeros((N, W, W), dtype=np.float16)
    P = np.zeros((N, W), dtype=np.float16)
    G = np.zeros((N, W), dtype=np.int8)
    Lv = np.zeros(N, dtype=np.int32)
    doc_idx = np.zeros(N, dtype=np.int32)
    start_a = np.zeros(N, dtype=np.int32)
    trunc = 0
    t0 = time.time()
    for k, (di, start, L0) in enumerate(wins):
        d = docs[di]
        chunk = d["tokens"][start:start + L0]
        _, v_mid, p_tag = encode_window(chunk, tokenizer, model,
                                        args.mid_layer,
                                        max_length=args.max_length)
        L = v_mid.shape[0]                       # truncation-trimmed
        trunc += L0 - L
        S[k, :L, :L] = cosine_matrix(v_mid).astype(np.float16)
        P[k, :L] = p_tag.astype(np.float16)
        G[k, :L] = np.asarray(d["labels"][start:start + L], dtype=np.int8)
        Lv[k], doc_idx[k], start_a[k] = L, di, start
        if (k + 1) % 50 == 0 or k == N - 1:
            el = time.time() - t0
            print(f"  [{k + 1}/{N}] {el:.0f}s elapsed, "
                  f"eta {el / (k + 1) * (N - k - 1):.0f}s", flush=True)

    meta = {"model": args.model, "jsonl": args.jsonl,
            "mid_layer": args.mid_layer, "window": args.window,
            "overlap": args.overlap, "min_len": args.min_len,
            "max_length": args.max_length}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, S=S, P=P, G=G, L=Lv, doc_idx=doc_idx, start=start_a,
        doc_ids=np.array([d["doc_id"] for d in docs]),
        doc_len=np.array([len(d["tokens"]) for d in docs], dtype=np.int32),
        meta=np.array(json.dumps(meta)))
    print(f"# cached {N} windows ({trunc} tokens lost to truncation, "
          f"{S.nbytes / 2**20:.0f} MiB fp16 pre-compress) -> {args.out}",
          flush=True)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(base=48, in_ch=2):
    import torch.nn as nn

    def block(ci, co):
        g = 8
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(g, co), nn.ReLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(g, co), nn.ReLU())

    class UNet2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(in_ch, base)
            self.enc2 = block(base, base * 2)
            self.enc3 = block(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):                    # x: [B, 2, W, W]
            import torch
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            m = self.head(d1)[:, 0]              # [B, W, W]
            return m.diagonal(dim1=1, dim2=2)    # [B, W] per-position logits

    return UNet2D()


def make_inputs(S16, P16, Lv, no_tagger=False):
    """fp16 cache slices -> float32 [B,C,W,W] input tensor + [B,W] mask.

    no_tagger=False (default): C=2 (similarity + tagger outer track) —
    unchanged behavior. no_tagger=True (--no-tagger-channel ablation arm):
    C=1, the similarity matrix channel ONLY."""
    import torch
    S = torch.from_numpy(S16.astype(np.float32))
    p = torch.from_numpy(P16.astype(np.float32))
    x2 = 0.5 * (p[:, :, None] + p[:, None, :])
    W = S.shape[1]
    vm = (torch.arange(W)[None, :] < torch.from_numpy(
        Lv.astype(np.int64))[:, None]).float()          # [B, W] valid mask
    sq = vm[:, :, None] * vm[:, None, :]
    if no_tagger:
        x = (S * sq)[:, None]                # [B, 1, W, W] similarity only
    else:
        x = torch.stack([S * sq, x2 * sq], dim=1)
    return x, vm


def stitch(cache, probs_win):
    """Average per-window probs back onto documents (predict.py convention).

    probs_win: [N, W] float array (only [:L] of each row is meaningful).
    Returns (per_doc list of np arrays, n_uncovered)."""
    doc_len = cache["doc_len"]
    sums = [np.zeros(int(n)) for n in doc_len]
    cnts = [np.zeros(int(n)) for n in doc_len]
    for k in range(probs_win.shape[0]):
        di, s, L = int(cache["doc_idx"][k]), int(cache["start"][k]), \
            int(cache["L"][k])
        sums[di][s:s + L] += probs_win[k, :L]
        cnts[di][s:s + L] += 1
    out, unc = [], 0
    for sm, ct in zip(sums, cnts):
        p = np.where(ct > 0, sm / np.maximum(ct, 1), 0.0)
        unc += int((ct == 0).sum())
        out.append(p)
    return out, unc


def predict_windows(model, cache, batch=16, no_tagger=False):
    """Forward the whole cache, return sigmoid probs [N, W] (numpy)."""
    import torch
    model.eval()
    N = cache["S"].shape[0]
    out = np.zeros((N, cache["S"].shape[1]), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s + batch, N)
            x, _ = make_inputs(cache["S"][s:e], cache["P"][s:e],
                               cache["L"][s:e], no_tagger=no_tagger)
            out[s:e] = torch.sigmoid(model(x)).numpy()
    return out


def load_cache(path):
    z = np.load(path)
    c = {k: z[k] for k in ("S", "P", "G", "L", "doc_idx", "start",
                           "doc_ids", "doc_len")}
    c["meta"] = json.loads(str(z["meta"]))
    return c


def dev_gold_arrays(docs):
    """Concatenated gold + doc_of arrays in doc order (all tokens)."""
    gold = np.concatenate([np.asarray(d["labels"], dtype=np.int8)
                           for d in docs])
    doc_of = np.concatenate([np.full(len(d["labels"]), i, dtype=np.int64)
                             for i, d in enumerate(docs)])
    return gold, doc_of


def flatten_probs(docs, per_doc):
    """per_doc probs -> concatenated array, forcing p=0 on PARAGRAPH tokens
    (same rule predict.py applies to its labels; a no-op on NoPnx-NP)."""
    out = []
    for d, p in zip(docs, per_doc):
        q = p.copy()
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                q[i] = 0.0
        out.append(q)
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args):
    assert_cpu_only()
    import torch

    assert not os.path.exists(args.out_dir), \
        f"{args.out_dir} exists — runs/ is append-only, pick a new name"
    os.makedirs(args.out_dir)

    tr = load_cache(args.train_cache)
    dv = load_cache(args.dev_cache)
    docs = load_jsonl(args.dev_jsonl)
    assert [str(x) for x in dv["doc_ids"]] == [d["doc_id"] for d in docs], \
        "dev cache doc order != dev jsonl doc order"
    gold, doc_of = dev_gold_arrays(docs)
    n_docs = len(docs)

    torch.manual_seed(args.seed)
    in_ch = 1 if args.no_tagger_channel else 2
    model = build_model(base=args.base, in_ch=in_ch)
    n_par = sum(p.numel() for p in model.parameters())
    note = (" [--no-tagger-channel: in_ch=1, similarity ONLY]"
            if in_ch == 1 else "")
    print(f"# UNet2D base={args.base}{note}: {n_par:,} params "
          f"({tr['S'].shape[0]} train windows, {dv['S'].shape[0]} dev "
          f"windows); seed={args.seed}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(args.pos_weight)), reduction="none")

    rng = np.random.default_rng(args.seed)
    N = tr["S"].shape[0]
    best = (-1.0, -1)                            # (dev F1@0.5, epoch)
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(N)
        tot, cnt, t0 = 0.0, 0, time.time()
        for s in range(0, N, args.batch_size):
            idx = order[s:s + args.batch_size]
            x, vm = make_inputs(tr["S"][idx], tr["P"][idx], tr["L"][idx],
                                no_tagger=args.no_tagger_channel)
            y = torch.from_numpy(tr["G"][idx].astype(np.float32))
            logits = model(x)
            loss = (bce(logits, y) * vm).sum() / vm.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * int(len(idx))
            cnt += int(len(idx))
        tr_loss = tot / max(cnt, 1)

        pw = predict_windows(model, dv, batch=2 * args.batch_size,
                             no_tagger=args.no_tagger_channel)
        per_doc, unc = stitch(dv, pw)
        p_flat = flatten_probs(docs, per_doc)
        f05 = macro_f1(p_flat, gold, doc_of, n_docs, 0.5)
        el = time.time() - t0
        star = ""
        if f05 > best[0]:
            best = (f05, ep)
            blob = {"state_dict": model.state_dict(),
                    "base": args.base, "seed": args.seed,
                    "epoch": ep, "dev_f1_05": f05,
                    "train_cache": args.train_cache,
                    "dev_cache": args.dev_cache}
            if args.no_tagger_channel:
                blob["in_ch"] = 1                # flag-gated: absent = 2
            torch.save(blob, os.path.join(args.out_dir, "best.pt"))
            star = "  *best*"
        hist.append({"epoch": ep, "train_loss": round(tr_loss, 5),
                     "dev_f1_05": round(100 * f05, 4),
                     "seconds": round(el, 1)})
        print(f"epoch {ep:3d}  loss {tr_loss:.4f}  dev F1@0.5 "
              f"{100 * f05:.2f}  ({el:.0f}s, uncovered={unc}){star}",
              flush=True)
        if ep - best[1] >= args.patience:
            print(f"early stop: no dev F1@0.5 improvement for "
                  f"{args.patience} epochs", flush=True)
            break

    with open(os.path.join(args.out_dir, "metrics.csv"), "w",
              newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(hist[0]))
        wcsv.writeheader()
        wcsv.writerows(hist)
    arg_d = {k: v for k, v in vars(args).items() if k != "fn"}
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({"args": arg_d, "params": n_par,
                   "best_epoch": best[1],
                   "best_dev_f1_05": round(100 * best[0], 4),
                   "epochs_run": len(hist)}, f, indent=2)
    print(f"BEST dev F1@0.5 = {100 * best[0]:.2f} (epoch {best[1]}) "
          f"-> {args.out_dir}/best.pt", flush=True)
    print("(dev used for checkpoint SELECTION only — repo convention; "
          "training loss saw train windows only)", flush=True)


# --------------------------------------------------------------------------- #
# eval: solo F1, decorrelation numbers, ensemble probe
# --------------------------------------------------------------------------- #
def _f1_line(name, p, gold, doc_of, n_docs, grid):
    f05 = macro_f1(p, gold, doc_of, n_docs, 0.5)
    fb, tb = max((macro_f1(p, gold, doc_of, n_docs, t), t) for t in grid)
    print(f"{name:34s} F1@0.5={100 * f05:6.2f}  own-best={100 * fb:6.2f} "
          f"@thr={tb}")
    return {"f1_05": round(100 * f05, 4), "best_f1": round(100 * fb, 4),
            "best_thr": tb}


def cmd_eval(args):
    assert_cpu_only()
    import torch

    dv = load_cache(args.dev_cache)
    docs = load_jsonl(args.dev_jsonl)
    assert [str(x) for x in dv["doc_ids"]] == [d["doc_id"] for d in docs], \
        "dev cache doc order != dev jsonl doc order"
    gold, doc_of = dev_gold_arrays(docs)
    n_docs = len(docs)

    z = np.load(args.probs)
    assert all(d["doc_id"] in z for d in docs), "probs npz missing dev docs"
    p_tag = np.concatenate([z[d["doc_id"]].astype(np.float64) for d in docs])
    assert len(p_tag) == len(gold)

    rep = {"seeds": {}, "probs": args.probs, "dev_cache": args.dev_cache}
    print("================ 2D U-NET ARM: DEV REPORT ================")
    base = _f1_line("tagger baseline (cached probs)", p_tag, gold, doc_of,
                    n_docs, THRESH_GRID)
    rep["baseline"] = base

    unet_flat = {}
    for ck in args.ckpts:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        in_ch = int(blob.get("in_ch", 2))        # absent in old ckpts = 2
        model = build_model(base=blob["base"], in_ch=in_ch)
        model.load_state_dict(blob["state_dict"])
        pw = predict_windows(model, dv, no_tagger=(in_ch == 1))
        per_doc, unc = stitch(dv, pw)
        p_u = flatten_probs(docs, per_doc)
        name = f"s{blob['seed']}"
        unet_flat[name] = p_u
        ch_note = " [no-tagger-channel: similarity ONLY]" if in_ch == 1 \
            else ""
        print(f"# {ck}: seed={blob['seed']} best-epoch={blob['epoch']}"
              f"{ch_note} (uncovered positions -> p=0: {unc})")
        rep["seeds"][name] = {"ckpt": ck, "epoch": blob["epoch"],
                              "uncovered": unc}
        if in_ch == 1:
            rep["seeds"][name]["in_ch"] = 1      # flag-gated: absent = 2

    variants = dict(unet_flat)
    if len(unet_flat) > 1:
        variants["pool"] = np.mean(list(unet_flat.values()), axis=0)

    conf_fp = (p_tag >= args.conf_thresh) & (gold == 0)
    conf_tp = (p_tag >= args.conf_thresh) & (gold == 1)
    err_t = (p_tag >= 0.5) != (gold == 1)
    fp_t = (p_tag >= 0.5) & (gold == 0)
    print(f"\n# tagger confident-FP n={int(conf_fp.sum())}, confident-TP "
          f"n={int(conf_tp.sum())} (p>={args.conf_thresh} on cached decode "
          f"grid; 2,236 FP expected on record)")

    for name, p_u in variants.items():
        print(f"\n-------- U-Net [{name}] --------")
        r = {}
        r["solo"] = _f1_line(f"U-Net {name} solo", p_u, gold, doc_of, n_docs,
                             WIDE_GRID)
        r["solo"]["note"] = "own-best over wide grid 0.05-0.95"

        # ---- decorrelation numbers ---------------------------------------
        pf, pt = p_u[conf_fp], p_u[conf_tp]
        r["conf_fp"] = {"n": int(conf_fp.sum()), "mean_p": float(pf.mean()),
                        "median_p": float(np.median(pf)),
                        "frac_below_05": float((pf < 0.5).mean())}
        r["conf_tp"] = {"n": int(conf_tp.sum()), "mean_p": float(pt.mean()),
                        "median_p": float(np.median(pt)),
                        "frac_below_05": float((pt < 0.5).mean())}
        sub = conf_fp | conf_tp
        r["auc_tp_vs_fp"] = rank_auc(p_u[sub], conf_tp[sub])
        print(f"at tagger conf-FP (n={r['conf_fp']['n']}): mean p="
              f"{r['conf_fp']['mean_p']:.4f} median={r['conf_fp']['median_p']:.4f} "
              f"frac<0.5={r['conf_fp']['frac_below_05']:.4f}")
        print(f"at tagger conf-TP (n={r['conf_tp']['n']}): mean p="
              f"{r['conf_tp']['mean_p']:.4f} median={r['conf_tp']['median_p']:.4f} "
              f"frac<0.5={r['conf_tp']['frac_below_05']:.4f}")
        print(f"U-Net p separates conf-TP vs conf-FP: AUC="
              f"{r['auc_tp_vs_fp']:.4f} (dev gate C(i) was 0.6212)")

        # ---- error correlation vs the tagger ------------------------------
        err_u = (p_u >= 0.5) != (gold == 1)
        fp_u = (p_u >= 0.5) & (gold == 0)
        phi_err = float(np.corrcoef(err_u.astype(float),
                                    err_t.astype(float))[0, 1])
        inter = int((fp_u & fp_t).sum())
        union = int((fp_u | fp_t).sum())
        jac = inter / max(union, 1)
        phi_fp = float(np.corrcoef(fp_u[gold == 0].astype(float),
                                   fp_t[gold == 0].astype(float))[0, 1])
        r["error_correlation"] = {
            "phi_all_errors": phi_err, "fp_jaccard": jac,
            "phi_fp_given_gold0": phi_fp,
            "n_fp_unet": int(fp_u.sum()), "n_fp_tagger": int(fp_t.sum()),
            "n_fp_shared": inter}
        print(f"error corr vs tagger: phi(all errors)={phi_err:.4f} "
              f"(existing-pool inter-voter mean = 0.74)")
        print(f"FP sets: unet={int(fp_u.sum())} tagger={int(fp_t.sum())} "
              f"shared={inter}  Jaccard={jac:.4f}  "
              f"phi(FP|gold=0)={phi_fp:.4f}")

        # ---- ensemble probe ------------------------------------------------
        blend = 0.5 * (p_tag + p_u)
        r["blend_mean"] = _f1_line(f"mean(tagger, unet-{name})", blend, gold,
                                   doc_of, n_docs, THRESH_GRID)
        r["blend_mean"]["delta_at_05_vs_baseline"] = round(
            r["blend_mean"]["f1_05"] - base["f1_05"], 4)
        sweep = {}
        for a in [round(x, 2) for x in np.arange(0.1, 0.91, 0.1)]:
            pb = (1 - a) * p_tag + a * p_u
            sweep[a] = round(100 * macro_f1(pb, gold, doc_of, n_docs, 0.5), 4)
        r["blend_alpha_sweep_f1_05"] = sweep
        besta = max(sweep, key=sweep.get)
        print(f"alpha sweep F1@0.5 (alpha=unet weight, dev-selected, "
              f"caveat): best alpha={besta} -> {sweep[besta]:.2f}")
        rep[f"unet_{name}"] = r

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, default=float)
        print(f"\nwrote {args.out}")
    print("\nCAVEAT: own-best thresholds and alpha are selected on dev; "
          "F1@0.5 with alpha=0.5 is the honest headline for the blend.")


# --------------------------------------------------------------------------- #
# self-test (corpus-independent; no data/, no encoder)
# --------------------------------------------------------------------------- #
def cmd_self_test(_args):
    assert_cpu_only()
    import torch

    # 1. param count + shapes at the real size
    model = build_model(base=48)
    n_par = sum(p.numel() for p in model.parameters())
    assert 1_000_000 <= n_par <= 3_000_000, f"param count {n_par} outside 1-3M"
    x = torch.zeros(2, 2, 180, 180)
    out = model(x)
    assert out.shape == (2, 180), f"bad output shape {tuple(out.shape)}"
    print(f"PASS 1: UNet2D base=48 has {n_par:,} params (1-3M), "
          f"[2,2,180,180] -> {tuple(out.shape)}")

    # 2. synthetic block matrices are learnable (W=64, tiny net)
    rng = np.random.default_rng(0)
    W, N = 64, 96
    S = np.zeros((N, W, W), dtype=np.float16)
    P = np.zeros((N, W), dtype=np.float16)
    G = np.zeros((N, W), dtype=np.int8)
    Lv = np.full(N, W, dtype=np.int32)
    for n in range(N):
        g = np.zeros(W, dtype=np.int8)
        i = 0
        while True:
            i += int(rng.integers(4, 14))
            if i >= W - 1:
                break
            g[i] = 1
        seg = np.cumsum(np.concatenate([[0], g[:-1]]))
        M = (seg[:, None] == seg[None, :]).astype(np.float32) * 0.7 + 0.2
        M += rng.normal(0, 0.05, size=(W, W)).astype(np.float32)
        np.fill_diagonal(M, 1.0)
        S[n] = M.astype(np.float16)
        P[n] = np.clip(g * 0.5 + 0.2
                       + rng.normal(0, 0.15, W), 0, 1).astype(np.float16)
        G[n] = g
    tiny = build_model(base=8)
    opt = torch.optim.Adam(tiny.parameters(), lr=2e-3)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0),
                                     reduction="none")
    torch.manual_seed(0)
    losses = []
    for ep in range(15):
        for s in range(0, N, 16):
            xx, vm = make_inputs(S[s:s + 16], P[s:s + 16], Lv[s:s + 16])
            y = torch.from_numpy(G[s:s + 16].astype(np.float32))
            loss = (bce(tiny(xx), y) * vm).sum() / vm.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    with torch.no_grad():
        xx, _ = make_inputs(S, P, Lv)
        p = torch.sigmoid(tiny(xx)).numpy().ravel()
    auc = rank_auc(p, G.ravel().astype(bool))
    assert losses[-1] < 0.6 * losses[0], \
        f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert auc > 0.85, f"synthetic AUC too low: {auc:.3f}"
    print(f"PASS 2: synthetic blocks learnable — loss {losses[0]:.3f} -> "
          f"{losses[-1]:.3f}, AUC={auc:.3f} (>0.85)")

    # 3. stitching: overlap-average + uncovered counting
    cache = {"doc_len": np.array([10]), "doc_idx": np.array([0, 0]),
             "start": np.array([0, 4]), "L": np.array([6, 4])}
    pw = np.zeros((2, 6))
    pw[0, :6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pw[1, :4] = [0.8, 0.6, 0.4, 0.2]
    per_doc, unc = stitch(cache, pw)
    exp = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.6, 0.4, 0.2, 0.0, 0.0])
    assert np.allclose(per_doc[0], exp), per_doc[0]
    assert unc == 2, unc
    print("PASS 3: stitch() averages overlaps and zero-fills uncovered "
          "positions (predict.py convention)")

    # 4. masking: padded positions contribute nothing
    x_a, vm = make_inputs(S[:4], P[:4], np.array([W, W, 40, 40]))
    assert float(x_a[2, :, 40:, :].abs().sum()) == 0.0
    assert float(x_a[2, :, :, 40:].abs().sum()) == 0.0
    assert vm[2, 40:].sum() == 0 and vm[2, :40].sum() == 40
    print("PASS 4: valid-length masking zeroes padded channels and loss mask")

    print("SELF-TEST PASS (4/4)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="frozen-encoder pass -> fp16 matrices")
    c.add_argument("--model", default="runs/union-NoPnx-NP-arabertv02-s42")
    c.add_argument("--jsonl", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--mid-layer", type=int, default=8)
    c.add_argument("--window", type=int, default=180)
    c.add_argument("--overlap", type=int, default=60)
    c.add_argument("--min-len", type=int, default=4)
    c.add_argument("--max-length", type=int, default=512)
    c.set_defaults(fn=cmd_cache)

    t = sub.add_parser("train", help="train one seed of the U-Net")
    t.add_argument("--train-cache", required=True)
    t.add_argument("--dev-cache", required=True)
    t.add_argument("--dev-jsonl", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--pos-weight", type=float, default=8.0)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--base", type=int, default=48)
    t.add_argument("--no-tagger-channel", action="store_true",
                   help="ablation arm: drop the tagger outer-track channel; "
                        "input = the similarity matrix ONLY (in_ch=1). "
                        "Default off = 2-channel behavior, unchanged.")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="solo F1 + decorrelation + blend probe")
    e.add_argument("--ckpts", nargs="+", required=True)
    e.add_argument("--dev-cache", required=True)
    e.add_argument("--dev-jsonl", required=True)
    e.add_argument("--probs", required=True,
                   help="cached decode-grid tagger probs npz")
    e.add_argument("--conf-thresh", type=float, default=0.8)
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("self-test", help="corpus-independent smoke test")
    s.set_defaults(fn=cmd_self_test)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
