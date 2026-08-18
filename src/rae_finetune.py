"""RAE-FINETUNE — the untested pipeline cell from Noor's research report:
SELF-SUPERVISED-PRETRAINED COMPOSER, THEN SUPERVISED FINE-TUNING.
ADDITIVE, CPU-ONLY (the GPU is owned by the BoundRL pilot). Read-only on
runs/, probs/, knn_store/, merge_judge/, rae_gate/, data/. Artifacts land
under rae_ft/ — a NEW additive directory. rae_gate.py and merge_judge.py are
IMPORTED, never modified.

THE TWO HALVES ON DISK
  (a) prior  : rae_gate/rae_{final,mid8}.pt — Socher-2011 Recursive
      Autoencoder trained label-free on the 174 TRAIN docs
      (p = tanh(W[c1;c2]+b) 1536->768 unit-norm; reconstruction head W';
      greedy lowest-E adjacent merge). Its dev slice AUC was 0.3062 —
      INVERTED (composability higher on confident-FP junctions' complements).
  (b) harness: merge_judge.py — from-scratch pair judge on dense curriculum
      pair supervision (~290K pairs: allcut/rand0-2/gold/half states from the
      174 TRAIN docs). ITS RESULT IS THE ABLATION BASELINE:
      eraser-on-tagger +0.05 (122 FP / 79 TP erased; 23 of 2,236 conf-FPs vs
      15 of 10,985 conf-TPs, ratio 1.53), solo 83.62 (-0.34), phi 0.90.

THE MODEL (decisions of record, resolving the commissioning order's options)
  Per curriculum pair (A=[a0..c], B=[c+1..b1]) the judge sees the junction
  ONLY THROUGH THE COMPOSER — two RAE parents plus their reconstruction
  errors, never the raw children:
    p_seg, E_seg = compose(unit(mean V[A]), unit(mean V[B]))   (segment level)
    p_word,E_word= compose(unit(V[c]),      unit(V[c+1]))      (leaf level)
    head input   = [ p_seg 768 | p_word 768 | E_seg | E_word |
                     the same 6 merge_judge scalars ]           (dim 2h+8)
  Raw pre-merge children are EXCLUDED deliberately: with raw children the
  head could re-learn merge_judge's from-scratch judge and the frozen arm
  would no longer test "does the self-supervised prior + labels suffice".
  At the all-cut state mean(A)==V[c] so p_seg==p_word — the leaf gate of
  rae_gate is recovered exactly. Head = MLP 2h+8 -> 256 -> 64 -> 1 ->
  P(merge) (same shape family as merge_judge's make_head; ~410K params on
  top of the 2.4M-param RAE). Inputs standardized by a scaler fitted ONCE
  with the pretrained RAE weights and then FROZEN (an affine transform; kept
  fixed even while arm B moves the composer, so both arms share one scale).

TWO ARMS (both 5 epochs CPU, batch 1024, Adam, seed 42 — merge_judge's
training recipe, same curriculum builder, same w_keep class balancing):
  A rae_ft_frozenW : RAE composition weights FROZEN, only the head trains
                     (lr 1e-3). Tests whether the prior + labels suffice.
  B rae_ft_full    : composition weights fine-tune too at a SMALL learning
                     rate (1e-4; head 1e-3). Tests whether labels can fix
                     the inverted slice inside the composer itself.

EVAL on dev (222 docs; EVALUATION ONLY) — exactly the merge_judge protocol,
headline tau=0.5 UNTUNED; tagger reference cache
probs/union-NoPnx-NP-arabertv02-s42_dev.npz (macro-F1 83.96 of record):
  (1) SLICE UN-INVERSION (decisive): AUC of the judge separating conf-FP
      junctions (should say MERGE) from conf-TP junctions (should say KEEP).
      PRIMARY (gate) number = the ERASER-STATE slice: P(merge) of each
      confident junction judged under the tagger's own segmentation — the
      state in which the score would join the stacker feature list, and the
      state in which merge_judge's supervise-only judge implicitly sat at
      ~0.5 (ratio 1.53 ~ random). Reported alongside: the ALL-CUT-state
      judge slice (the state of the RAE-alone 0.3062 record) and the raw
      -E_word slice (composer diagnostic; for arm A this must reproduce
      ~0.3062, a built-in harness check — computed on ALL conf-TP junctions
      where rae_gate sampled 1,000, so small deviation is expected).
      Slice sets are junction-level (doc-final tokens carry no pair).
  (2) ERASER-ON-TAGGER: one-shot judging of the tagger's cuts in the
      pre-erase state, erase where P(merge) > tau; tau grid + headline 0.5;
      net macro-F1 delta, erased FP/TP, conf-FP vs conf-TP erasure ratio,
      per-genre deltas, P/R decomposition (merge_judge's exact bookkeeping,
      full-token-grid confident sets of record: 2,236 / 10,985).
  (3) SOLO agglomerative decode F1 + decorrelation phi_all_errors /
      FP-Jaccard vs the tagger cache (unet2d convention; references: pool
      mean 0.74, S-only U-Net 0.53). Default solo tau list = the 0.5
      headline only (runtime; --solo-taus restores merge_judge's 0.3,0.5,0.7).

PRE-REGISTERED GATES (tick verbatim, per arm):
  [ ] ADOPT-CANDIDATE iff eraser net F1 > +0.15 AND conf-FP:conf-TP erasure
      ratio > 3 (same bars merge-judge failed).
  [ ] SLICE-FIXED (interesting even sub-bar) iff slice AUC >= 0.60 in the
      CORRECT direction — supervision un-inverted the prior; then the judge
      score joins the stacker feature list.
  [ ] VOTER iff solo phi < 0.6.
  [ ] else PARK with the numbers, stating explicitly whether the pipeline
      beat its supervise-only ablation (+0.05 / 1.53 / 0.90) at all.

Stages (CPU only; artifacts under rae_ft/):

  CUDA_VISIBLE_DEVICES=-1 python src/rae_finetune.py self-test

  CUDA_VISIBLE_DEVICES=-1 python src/rae_finetune.py train --layer final \
      --arm frozenW --rae rae_gate/rae_final.pt \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --train-vecs knn_store/NoPnx-NP_s42_train.npz \
      --out rae_ft/judge_frozenW_final.pt
  (arm full: --arm full --out rae_ft/judge_full_final.pt; mid8: --layer mid8
   --rae rae_gate/rae_mid8.pt --train-vecs merge_judge/NoPnx-NP_s42_train_mid8vec.npz)

  CUDA_VISIBLE_DEVICES=-1 python src/rae_finetune.py eval --layer final \
      --judge rae_ft/judge_frozenW_final.pt \
      --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --dev-vecs knn_store/NoPnx-NP_s42_dev_q.npz \
      --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --out rae_ft/report_frozenW_final.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

from data import PARAGRAPH_TOKEN, load_jsonl
from diag_mask_influence import roc_auc
from genre_buckets import bucket
from knn_boundary import macro_f1
from merge_judge import (
    BASELINE_F1_REF,
    CONF,
    N_SCALARS,
    TAU_EXTRA,
    TAU_GRID,
    TAU_HEADLINE,
    _features_idx,
    _synth_doc,
    agglomerative_cuts,
    assemble_pairs,
    assert_cpu_only,
    build_features,
    curriculum_states,
    decorrelation,
    eraser_merge_probs,
    pairs_from_cuts,
    pooled_prf,
    prefix_sum,
)
from rae_gate import load_rae, make_rae, normalize_rows

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

ARMS = ["frozenW", "full"]
GENRE_ORDER = ["general prose", "legal/constitution", "hadith/isnad",
               "genealogy", "cloze/exercise"]
ABLATION = {"eraser_delta": 0.05, "erasure_ratio": 1.53, "solo_phi": 0.90}


# --------------------------------------------------------------------------- #
# model: RAE composer + keep/erase head on the parent vectors
# --------------------------------------------------------------------------- #
def make_judge_head(h, seed=42):
    """Head over [p_seg | p_word | E_seg | E_word | 6 scalars] (dim 2h+8);
    same shape family as merge_judge.make_head."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2 * h + 2 + N_SCALARS, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 64), nn.ReLU(),
        nn.Linear(64, 1))


def make_judge(rae, h, seed=42):
    """RAEJudge module: composes both granularities with the (shared) RAE and
    scores P(merge) from the parents + reconstruction errors + scalars.
    Consumes merge_judge's raw feature matrices (m, 4h+6) unchanged, so
    eraser_merge_probs / agglomerative_cuts drive it without modification."""
    import torch
    import torch.nn as nn

    class RAEJudge(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = h
            self.rae = rae
            self.head = make_judge_head(h, seed=seed)
            self.register_buffer("mu", torch.zeros(2 * h + 2 + N_SCALARS))
            self.register_buffer("sd", torch.ones(2 * h + 2 + N_SCALARS))

        @staticmethod
        def _unit(x):
            return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        def judge_inputs(self, X):
            """X (m, 4h+6) raw merge_judge features -> (m, 2h+8) head input
            [p_seg | p_word | E_seg | E_word | scalars] (unstandardized)."""
            hh = self.h
            p1, e1 = self.rae.compose(self._unit(X[:, :hh]),
                                      self._unit(X[:, hh:2 * hh]))
            p2, e2 = self.rae.compose(self._unit(X[:, 2 * hh:3 * hh]),
                                      self._unit(X[:, 3 * hh:4 * hh]))
            return torch.cat([p1, p2, e1[:, None], e2[:, None],
                              X[:, 4 * hh:]], dim=1)

        def forward(self, X):
            """(m, 4h+6) -> (m,) merge logits."""
            z = self.head((self.judge_inputs(X) - self.mu) / self.sd)
            return z[:, 0]

    return RAEJudge()


class FTJudge:
    """Frozen callable over merge_judge feature matrices -> P(merge);
    plug-compatible with eraser_merge_probs / agglomerative_cuts.
    Forces eval mode (deterministic inference is the contract)."""

    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, X):
        import torch
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            z = self.model(torch.from_numpy(
                np.ascontiguousarray(X, dtype=np.float32)))
        return torch.sigmoid(z).numpy()


def judge_with_E(model, X, batch=8192):
    """(P(merge), E_word) for a raw feature matrix, batched, no grad,
    eval mode forced (deterministic)."""
    import torch
    model.eval()
    if len(X) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    ps, es = [], []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xt = torch.from_numpy(
                np.ascontiguousarray(X[s:s + batch], dtype=np.float32))
            J = model.judge_inputs(xt)
            z = model.head((J - model.mu) / model.sd)[:, 0]
            ps.append(torch.sigmoid(z).numpy())
            es.append(J[:, 2 * model.h + 1].numpy())
    return np.concatenate(ps), np.concatenate(es)


def fit_scaler(model, batches):
    """mu/sd buffers over judge_inputs of an iterable of raw feature batches
    (fitted ONCE with the pretrained RAE weights, then frozen — decision of
    record; an affine transform shared by both arms)."""
    import torch
    d = 2 * model.h + 2 + N_SCALARS
    tot = np.zeros(d, dtype=np.float64)
    tot2 = np.zeros(d, dtype=np.float64)
    seen = 0
    with torch.no_grad():
        for X in batches:
            J = model.judge_inputs(torch.from_numpy(
                np.ascontiguousarray(X, dtype=np.float32))).numpy()
            tot += J.sum(axis=0, dtype=np.float64)
            tot2 += (J.astype(np.float64) ** 2).sum(axis=0)
            seen += len(J)
    mu = tot / seen
    sd = np.sqrt(np.maximum(tot2 / seen - mu ** 2, 1e-8))
    model.mu.copy_(torch.from_numpy(mu.astype(np.float32)))
    model.sd.copy_(torch.from_numpy(sd.astype(np.float32)))
    return seen


def make_optimizer(model, arm, lr, rae_lr):
    """frozenW: RAE params frozen, head-only Adam. full: two param groups,
    small lr on the composer."""
    import torch
    assert arm in ARMS, f"unknown arm {arm}"
    if arm == "frozenW":
        for p in model.rae.parameters():
            p.requires_grad_(False)
        return torch.optim.Adam(model.head.parameters(), lr=lr)
    return torch.optim.Adam([
        {"params": model.head.parameters(), "lr": lr},
        {"params": model.rae.parameters(), "lr": rae_lr}])


def train_epochs(model, opt, batch_feats, y_all, epochs, batch, w_keep, rng,
                 verbose=True):
    """merge_judge's training loop over (feature-provider, labels): weighted
    BCE with keep pairs up-weighted by w_keep. Returns history rows."""
    import torch
    import torch.nn as nn
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    history = []
    for ep in range(epochs):
        perm = rng.permutation(len(y_all))
        ep_loss = ep_n = 0.0
        acc_m = acc_k = seen_m = seen_k = 0
        t0 = time.time()
        model.train()
        for s in range(0, len(perm), batch):
            sel = perm[s:s + batch]
            X = batch_feats(sel)
            y = y_all[sel]
            Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            yt = torch.from_numpy(y)
            z = model(Xt)
            w = torch.where(yt > 0.5, torch.ones_like(yt),
                            torch.full_like(yt, w_keep))
            loss = (lossf(z, yt) * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()) * len(y)
            ep_n += len(y)
            pred = (torch.sigmoid(z.detach()) > 0.5).numpy()
            acc_m += int((pred & (y > 0.5)).sum())
            acc_k += int((~pred & (y < 0.5)).sum())
            seen_m += int((y > 0.5).sum())
            seen_k += int((y < 0.5).sum())
        bal = 0.5 * (acc_m / max(seen_m, 1) + acc_k / max(seen_k, 1))
        row = {"loss": ep_loss / max(ep_n, 1),
               "merge_acc": acc_m / max(seen_m, 1),
               "keep_acc": acc_k / max(seen_k, 1), "bal_acc": bal}
        history.append(row)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs}: loss={row['loss']:.4f} "
                  f"merge-acc={row['merge_acc']:.4f} "
                  f"keep-acc={row['keep_acc']:.4f} bal-acc={bal:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return history


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args):
    assert_cpu_only()
    import torch
    torch.set_num_threads(max(1, args.threads))
    from merge_judge import load_doc_vectors

    docs = load_jsonl(args.train_jsonl)
    for d in docs:
        assert all(w != PARAGRAPH_TOKEN for w in d["tokens"]), \
            "unexpected PAR token in a NoPnx track"
    vecs = load_doc_vectors(args.train_vecs, docs)
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    rae, rck = load_rae(args.rae)
    assert rck["layer"] == args.layer, \
        f"RAE was trained on layer {rck['layer']}, train asked for {args.layer}"
    h = rck["h"]
    rng = np.random.default_rng(args.seed)

    table, counts = assemble_pairs(docs, rng, args.n_random_states)
    n1 = int(table[:, 4].sum())
    n0 = len(table) - n1
    w_keep = n1 / max(n0, 1)
    print(f"# RAE-FINETUNE train [{args.arm}/{args.layer}]  "
          f"prior={args.rae} (last pretrain E="
          f"{rck['loss_history'][-1]:.5f})", flush=True)
    print("# supervision: TRAIN boundary labels via merge_judge's curriculum "
          "(dev untouched):")
    for name in ["allcut"] + [f"rand{r}" for r in range(args.n_random_states)] \
            + ["gold", "half"]:
        m, k = counts.get(name, (0, 0))
        print(f"    state {name:7s}: merge={m:6d} keep={k:6d}")
    print(f"    TOTAL: merge={n1} keep={n0} -> keep weight {w_keep:.3f} "
          f"(classes balanced in the loss)", flush=True)

    # global arrays (merge_judge.cmd_train's pattern of record)
    keys = [d["doc_id"] for d in docs]
    lens = np.array([len(vecs[k]) for k in keys], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    poff = off + np.arange(len(keys))
    Vg = np.concatenate([vecs[k] for k in keys])
    Pg = np.concatenate([prefix_sum(vecs[k]) for k in keys])

    di, a0, c, b1 = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
    y_all = table[:, 4].astype(np.float32)
    IDX = np.stack([off[di] + c, off[di] + c + 1,
                    poff[di] + a0, poff[di] + c + 1,
                    poff[di] + b1 + 1,
                    c - a0 + 1, b1 - c], axis=1)

    def batch_feats(sel):
        cols = IDX[sel]
        return _features_idx(Vg, Pg, cols[:, 0], cols[:, 1], cols[:, 2],
                             cols[:, 3], cols[:, 4], cols[:, 5], cols[:, 6])

    model = make_judge(rae, h, seed=args.seed)
    n_head = sum(p.numel() for p in model.head.parameters())
    n_rae = sum(p.numel() for p in model.rae.parameters())
    print(f"# judge: head {n_head} params on RAE {n_rae} params  "
          f"arm={args.arm} (frozenW: composer frozen; full: composer "
          f"lr={args.rae_lr})  epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} seed={args.seed} threads={args.threads}", flush=True)

    t0 = time.time()
    seen = fit_scaler(model, (batch_feats(np.arange(s, min(s + 8192,
                                                           len(table))))
                              for s in range(0, len(table), 8192)))
    print(f"# scaler over {seen} pairs (pretrained-RAE stats, frozen) in "
          f"{time.time() - t0:.0f}s", flush=True)

    opt = make_optimizer(model, args.arm, args.lr, args.rae_lr)
    history = train_epochs(model, opt, batch_feats, y_all, args.epochs,
                           args.batch, w_keep, rng)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    import torch as _t
    _t.save({"rae_state": model.rae.state_dict(),
             "head_state": model.head.state_dict(),
             "mu": model.mu.numpy(), "sd": model.sd.numpy(),
             "h": h, "layer": args.layer, "arm": args.arm,
             "history": history,
             "config": {"seed": args.seed, "epochs": args.epochs,
                        "batch": args.batch, "lr": args.lr,
                        "rae_lr": args.rae_lr,
                        "n_random_states": args.n_random_states,
                        "rae_ckpt": args.rae,
                        "train_jsonl": args.train_jsonl,
                        "train_vecs": args.train_vecs,
                        "w_keep": w_keep,
                        "curriculum_counts": counts}}, args.out)
    print(f"fine-tuned judge [{args.arm}/{args.layer}] -> {args.out}",
          flush=True)


def load_judge(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rae = make_rae(ck["h"])
    rae.load_state_dict(ck["rae_state"])
    model = make_judge(rae, ck["h"])
    model.head.load_state_dict(ck["head_state"])
    model.mu.copy_(torch.from_numpy(np.asarray(ck["mu"], dtype=np.float32)))
    model.sd.copy_(torch.from_numpy(np.asarray(ck["sd"], dtype=np.float32)))
    model.eval()
    return model, ck


# --------------------------------------------------------------------------- #
# eval (dev = evaluation only) — merge_judge's protocol + the slice readout
# --------------------------------------------------------------------------- #
def tick_gates(tag, delta, ratio, slice_primary, phi_solo):
    """The commissioning order's pre-registered gates, verbatim."""
    adopt = (delta > 0.15) and (ratio > 3)
    sliced = slice_primary >= 0.60
    voter = (phi_solo == phi_solo) and (phi_solo < 0.6)
    park = not (adopt or sliced or voter)
    print(f"\n=========== PRE-REGISTERED GATES [{tag}] (verbatim) ===========")
    print(f"  [{'X' if adopt else ' '}] ADOPT-CANDIDATE iff eraser net F1 > "
          f"+0.15 AND conf-FP:conf-TP erasure ratio > 3 (same bars "
          f"merge-judge failed).  (measured: {delta:+.2f}; ratio "
          f"{ratio:.2f})")
    print(f"  [{'X' if sliced else ' '}] SLICE-FIXED (interesting even "
          f"sub-bar) iff slice AUC >= 0.60 in the CORRECT direction — "
          f"supervision un-inverted the prior; then the judge score joins "
          f"the stacker feature list.  (measured: {slice_primary:.4f})")
    print(f"  [{'X' if voter else ' '}] VOTER iff solo phi < 0.6.  "
          f"(measured: phi={phi_solo:.4f})")
    print(f"  [{'X' if park else ' '}] else PARK with the numbers, stating "
          f"explicitly whether the pipeline beat its supervise-only "
          f"ablation (+0.05 / 1.53 / 0.90) at all.")
    return {"adopt_candidate": bool(adopt), "slice_fixed": bool(sliced),
            "voter": bool(voter), "park": bool(park)}


def ablation_verdict(delta, ratio, phi_solo):
    beats = {"eraser_delta": bool(delta > ABLATION["eraser_delta"]),
             "erasure_ratio": bool(ratio > ABLATION["erasure_ratio"]),
             "solo_phi": bool(phi_solo == phi_solo
                              and phi_solo < ABLATION["solo_phi"])}
    print("\n--- ABLATION COMPARISON vs merge_judge supervise-only "
          "(+0.05 / 1.53 / 0.90) ---")
    print(f"  eraser net F1 delta : {delta:+.2f} vs +0.05  -> "
          f"{'BEATS' if beats['eraser_delta'] else 'does NOT beat'}")
    print(f"  cFP:cTP erasure ratio: {ratio:.2f} vs 1.53  -> "
          f"{'BEATS' if beats['erasure_ratio'] else 'does NOT beat'}")
    print(f"  solo phi (lower=better): {phi_solo:.4f} vs 0.90 -> "
          f"{'BEATS' if beats['solo_phi'] else 'does NOT beat'}")
    beats["any"] = bool(any(beats.values()))
    print(f"  pipeline beat its supervise-only ablation at all: "
          f"{'YES' if beats['any'] else 'NO'}")
    return beats


def cmd_eval(args):
    assert_cpu_only()
    import torch
    torch.set_num_threads(max(1, args.threads))
    from merge_judge import load_doc_vectors
    t_start = time.time()

    docs = load_jsonl(args.dev_jsonl)
    vecs = load_doc_vectors(args.dev_vecs, docs)
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    model, ck = load_judge(args.judge)
    assert ck["layer"] == args.layer, \
        f"judge was trained on layer {ck['layer']}, eval asked for {args.layer}"
    judge = FTJudge(model)
    pz = np.load(args.probs)
    n_docs = len(docs)
    arm = ck["arm"]
    report = {"layer": args.layer, "arm": arm, "judge": args.judge,
              "probs": args.probs, "dev_jsonl": args.dev_jsonl,
              "n_docs": n_docs, "tau_headline": TAU_HEADLINE,
              "train_history": ck["history"], "config": ck["config"]}
    print(f"# RAE-FINETUNE eval [{arm}/{args.layer}]  judge={args.judge}")
    print(f"# dev={args.dev_jsonl} ({n_docs} docs)  probs={args.probs}  "
          f"headline tau={TAU_HEADLINE} UNTUNED", flush=True)

    # pooled arrays (merge_judge.cmd_eval's exact bookkeeping)
    golds, preds_t, pconf, doc_of, genres, junction = [], [], [], [], [], []
    for di, d in enumerate(docs):
        assert all(w != PARAGRAPH_TOKEN for w in d["tokens"]), \
            "unexpected PAR token in a NoPnx track"
        g = np.asarray(d["labels"], dtype=np.int64)
        p = pz[d["doc_id"]].astype(np.float64)
        assert len(g) == len(p) == len(vecs[d["doc_id"]])
        golds.append(g)
        preds_t.append(p >= 0.5)
        pconf.append(p)
        doc_of.append(np.full(len(g), di, dtype=np.int64))
        genres.append(bucket(d["tokens"]))
        j = np.ones(len(g), dtype=bool)
        j[-1] = False                              # doc-final token: no pair
        junction.append(j)
    gold = np.concatenate(golds).astype(bool)
    pred_t = np.concatenate(preds_t)
    p_tag = np.concatenate(pconf)
    doc_of = np.concatenate(doc_of)
    junction = np.concatenate(junction)

    base_f1 = 100 * macro_f1(pred_t.astype(np.float64), gold.astype(np.int8),
                             doc_of, n_docs, 0.5)
    conf_fp = (p_tag >= CONF) & ~gold              # full-grid sets of record
    conf_tp = (p_tag >= CONF) & gold
    print(f"# BASELINE tagger@0.5: macro-F1={base_f1:.2f} "
          f"(record: {BASELINE_F1_REF}); confident-FP n={int(conf_fp.sum())} "
          f"(record: 2236), confident-TP n={int(conf_tp.sum())} "
          f"(record: 10985)", flush=True)
    report["baseline"] = {"macro_f1": base_f1,
                          "n_conf_fp": int(conf_fp.sum()),
                          "n_conf_tp": int(conf_tp.sum()),
                          "pooled": pooled_prf(pred_t, gold)}

    # ---------------- (2) ERASER-ON-TAGGER ---------------------------------
    P = {k: prefix_sum(vecs[k]) for k in vecs}
    t0 = time.time()
    mp = np.full(len(gold), -1.0)
    off = 0
    for di, d in enumerate(docs):
        k = d["doc_id"]
        n = len(d["labels"])
        cuts, probs_m = eraser_merge_probs(vecs[k], P[k], preds_t[di], judge)
        mp[off + cuts] = probs_m
        off += n
    print(f"# eraser: judged {int((mp >= 0).sum())} tagger cuts in "
          f"{time.time() - t0:.0f}s", flush=True)

    def erased_pred(tau):
        out = pred_t.copy()
        out[mp > tau] = False
        return out

    print("\n=========== (2) ERASER-ON-TAGGER (dev, %d docs) ===========" % n_docs)
    print(f"{'tau':>6} {'macroF1':>8} {'delta':>7} {'erased':>7} "
          f"{'cFP-erased':>10} {'cTP-erased':>10} {'ratio':>6}")
    rows_a = {}
    for tau in TAU_GRID + TAU_EXTRA:
        pe = erased_pred(tau)
        f1 = 100 * macro_f1(pe.astype(np.float64), gold.astype(np.int8),
                            doc_of, n_docs, 0.5)
        e = pred_t & ~pe
        ce_fp = int((e & conf_fp).sum())
        ce_tp = int((e & conf_tp).sum())
        ratio = ce_fp / max(ce_tp, 1)
        tag = " *extra*" if tau in TAU_EXTRA else \
            ("  <-- HEADLINE (untuned)" if tau == TAU_HEADLINE else "")
        print(f"{tau:6.2f} {f1:8.2f} {f1 - base_f1:+7.2f} {int(e.sum()):7d} "
              f"{ce_fp:10d} {ce_tp:10d} {ratio:6.2f}{tag}")
        rows_a[round(tau, 2)] = {
            "macro_f1": f1, "delta": f1 - base_f1, "n_erased": int(e.sum()),
            "erased_fp_all": int((e & ~gold).sum()),
            "erased_tp_all": int((e & gold).sum()),
            "erased_conf_fp": ce_fp, "erased_conf_tp": ce_tp,
            "pooled": pooled_prf(pe, gold)}
    report["eraser_taus"] = rows_a

    pe_h = erased_pred(TAU_HEADLINE)
    hh = rows_a[TAU_HEADLINE]
    bp = report["baseline"]["pooled"]
    hp = hh["pooled"]
    ratio_h = hh["erased_conf_fp"] / max(hh["erased_conf_tp"], 1)
    print(f"\nHEADLINE tau=0.5 FAILURE ANATOMY (standing order):")
    print(f"  macro-F1 {base_f1:.2f} -> {hh['macro_f1']:.2f} "
          f"(delta {hh['delta']:+.2f} vs {BASELINE_F1_REF})")
    print(f"  erased {hh['n_erased']} cuts: {hh['erased_fp_all']} were FPs "
          f"(good), {hh['erased_tp_all']} were TPs (bad)")
    print(f"  of the {int(conf_fp.sum())} confident FPs (p>=0.8, gold=0): "
          f"{hh['erased_conf_fp']} erased")
    print(f"  of the {int(conf_tp.sum())} confident TPs (p>=0.8, gold=1): "
          f"{hh['erased_conf_tp']} wrongly erased "
          f"(ratio cFP/cTP = {ratio_h:.2f}, gate needs > 3; "
          f"ablation record 1.53)")
    print(f"  pooled P/R decomposition: P {bp['P']:.4f} -> {hp['P']:.4f}, "
          f"R {bp['R']:.4f} -> {hp['R']:.4f} "
          f"(TP {bp['TP']}->{hp['TP']}, FP {bp['FP']}->{hp['FP']}, "
          f"FN {bp['FN']}->{hp['FN']})")

    print("\n  per-genre (mean per-doc F1, before -> after; FP/FN pooled):")
    print(f"  {'genre':20s} {'docs':>5} {'F1 before':>10} {'F1 after':>9} "
          f"{'delta':>7} {'dFP':>6} {'dFN':>6}")
    genre_rows = {}
    for gname in GENRE_ORDER:
        idx = [i for i, x in enumerate(genres) if x == gname]
        if not idx:
            continue
        m = np.isin(doc_of, idx)
        fb = 100 * macro_f1(pred_t[m].astype(np.float64),
                            gold[m].astype(np.int8),
                            np.searchsorted(np.array(idx), doc_of[m]),
                            len(idx), 0.5)
        fa = 100 * macro_f1(pe_h[m].astype(np.float64),
                            gold[m].astype(np.int8),
                            np.searchsorted(np.array(idx), doc_of[m]),
                            len(idx), 0.5)
        dfp = int((pe_h[m] & ~gold[m]).sum()) - int((pred_t[m] & ~gold[m]).sum())
        dfn = int((~pe_h[m] & gold[m]).sum()) - int((~pred_t[m] & gold[m]).sum())
        print(f"  {gname:20s} {len(idx):5d} {fb:10.2f} {fa:9.2f} "
              f"{fa - fb:+7.2f} {dfp:+6d} {dfn:+6d}")
        genre_rows[gname] = {"n_docs": len(idx), "f1_before": fb,
                             "f1_after": fa, "delta": fa - fb,
                             "d_fp": dfp, "d_fn": dfn}
    report["eraser_headline_genres"] = genre_rows

    # ---------------- (1) SLICE UN-INVERSION --------------------------------
    # junction-level confident sets (doc-final tokens carry no pair)
    cfj = conf_fp & junction
    ctj = conf_tp & junction
    assert (mp[cfj] >= 0).all() and (mp[ctj] >= 0).all(), \
        "confident junctions must all be tagger cuts (p>=0.8 -> p>=0.5)"
    slice_eraser = roc_auc(mp[cfj], mp[ctj])

    # all-cut state pass (the state of the RAE-alone 0.3062 record)
    pj = np.full(len(gold), np.nan)
    ej = np.full(len(gold), np.nan)
    off = 0
    t0 = time.time()
    for d in docs:
        k = d["doc_id"]
        n = len(d["labels"])
        if n >= 2:
            X = build_features(vecs[k], P[k],
                               pairs_from_cuts(np.arange(n - 1), n))
            pr, ee = judge_with_E(model, X)
            pj[off:off + n - 1] = pr
            ej[off:off + n - 1] = ee
        off += n
    slice_allcut = roc_auc(pj[cfj], pj[ctj])
    slice_negE = roc_auc(-ej[cfj], -ej[ctj])
    print(f"\n=========== (1) SLICE UN-INVERSION (conf-FP vs conf-TP "
          f"junctions: {int(cfj.sum())} vs {int(ctj.sum())}) ===========")
    print(f"  [AUC = P(judge says MERGE more on a conf-FP); correct "
          f"direction > 0.5; RAE-alone record 0.3062 (inverted); "
          f"merge-judge supervise-only implicitly ~0.5]")
    print(f"  PRIMARY  eraser-state judge slice AUC : {slice_eraser:.4f}")
    print(f"           all-cut-state judge slice AUC: {slice_allcut:.4f}")
    print(f"  diagnostic -E_word (composer) slice   : {slice_negE:.4f}  "
          f"(arm frozenW must sit near the 0.3062 record — harness check; "
          f"arm full shows whether the composer itself moved)  "
          f"({time.time() - t0:.0f}s)", flush=True)
    report["slice"] = {"n_conf_fp_junction": int(cfj.sum()),
                       "n_conf_tp_junction": int(ctj.sum()),
                       "judge_eraser_state": float(slice_eraser),
                       "judge_allcut_state": float(slice_allcut),
                       "negE_allcut_state": float(slice_negE),
                       "rae_alone_record": 0.3062}

    # ---------------- (3) SOLO AGGLOMERATIVE --------------------------------
    solo_taus = [float(t) for t in args.solo_taus.split(",")] \
        if args.solo_taus else [TAU_HEADLINE]
    report["solo"] = {}
    pred_solo_h = None
    if not args.skip_solo:
        print("\n=========== (3) SOLO AGGLOMERATIVE (all-cut start) ===========")
        for tau in solo_taus:
            t0 = time.time()
            chunks = []
            for d in docs:
                k = d["doc_id"]
                n = len(d["labels"])
                cuts = agglomerative_cuts(vecs[k], P[k], judge, tau)
                pr = np.zeros(n, dtype=bool)
                pr[cuts] = True
                pr[n - 1] = True                  # doc end (gold convention)
                chunks.append(pr)
            ps = np.concatenate(chunks)
            f1 = 100 * macro_f1(ps.astype(np.float64), gold.astype(np.int8),
                                doc_of, n_docs, 0.5)
            pl = pooled_prf(ps, gold)
            tag = "  <-- HEADLINE (untuned)" if tau == TAU_HEADLINE else ""
            print(f"  tau={tau:.2f}: macro-F1={f1:.2f} "
                  f"(delta {f1 - base_f1:+.2f} vs tagger@0.5; ablation solo "
                  f"record 83.62)  P={pl['P']:.4f} R={pl['R']:.4f} "
                  f"pred-cuts={int(ps.sum())} ({time.time() - t0:.0f}s){tag}",
                  flush=True)
            report["solo"][round(tau, 2)] = {"macro_f1": f1,
                                             "delta": f1 - base_f1,
                                             "pooled": pl}
            if abs(tau - TAU_HEADLINE) < 1e-9:
                pred_solo_h = ps

    print("\n=========== DECORRELATION vs tagger errors (unet2d "
          "convention) ===========")
    dec_a = decorrelation(pe_h, pred_t, gold)
    print(f"  eraser@0.5 : phi(all errors)={dec_a['phi_all_errors']:.4f}  "
          f"FP-Jaccard={dec_a['fp_jaccard']:.4f}  "
          f"phi(FP|gold=0)={dec_a['phi_fp_given_gold0']:.4f}")
    report["decorrelation_eraser"] = dec_a
    phi_solo = float("nan")
    if pred_solo_h is not None:
        dec_b = decorrelation(pred_solo_h, pred_t, gold)
        phi_solo = dec_b["phi_all_errors"]
        print(f"  solo@0.5   : phi(all errors)={dec_b['phi_all_errors']:.4f}  "
              f"FP-Jaccard={dec_b['fp_jaccard']:.4f}  "
              f"phi(FP|gold=0)={dec_b['phi_fp_given_gold0']:.4f}")
        report["decorrelation_solo"] = dec_b
    print("  reference points: existing-pool inter-voter mean phi = 0.74; "
          "S-only U-Net = 0.53; ablation solo phi record = 0.90")

    # ---------------- GATES + ABLATION VERDICT ------------------------------
    gates = tick_gates(f"{arm}/{args.layer}", hh["delta"], ratio_h,
                       slice_eraser, phi_solo)
    beats = ablation_verdict(hh["delta"], ratio_h, phi_solo)
    verdict = ("ADOPT-CANDIDATE" if gates["adopt_candidate"] else
               "SLICE-FIXED" if gates["slice_fixed"] else
               "VOTER" if gates["voter"] else "PARK with the numbers")
    print(f"\n  VERDICT [{arm}/{args.layer}]: {verdict}")
    report["gates"] = dict(gates, verdict=verdict)
    report["ablation_comparison"] = dict(
        beats, record=ABLATION,
        measured={"eraser_delta": hh["delta"], "erasure_ratio": ratio_h,
                  "solo_phi": phi_solo})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nreport -> {args.out}")
    print(f"# done in {(time.time() - t_start) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
# synthetic machinery for the planted-inversion proof (also used by pytest)
# --------------------------------------------------------------------------- #
def plant_inverted_rae(h, docs, steps=300, lr=1e-2, seed=0, margin=2.0):
    """Plant an INVERTED composability prior in a small RAE: reconstruction
    error is pushed DOWN on cross-boundary adjacent pairs and UP (hinge to
    `margin`) on within-sentence pairs — low E exactly where composition
    should fail, mirroring the real slice inversion of record (0.3062)."""
    import torch
    rae = make_rae(h, seed=seed)
    opt = torch.optim.Adam(rae.parameters(), lr=lr)
    xc, wc = [], []
    for _w, lab, V in docs:
        Vn = normalize_rows(V)
        for i in range(len(lab) - 1):
            (xc if lab[i] == 1 else wc).append((Vn[i], Vn[i + 1]))
    c1x = torch.from_numpy(np.stack([a for a, _ in xc]))
    c2x = torch.from_numpy(np.stack([b for _, b in xc]))
    c1w = torch.from_numpy(np.stack([a for a, _ in wc]))
    c2w = torch.from_numpy(np.stack([b for _, b in wc]))
    for _ in range(steps):
        _, e_cross = rae.compose(c1x, c2x)
        _, e_within = rae.compose(c1w, c2w)
        loss = e_cross.mean() \
            + torch.relu(margin - e_within).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    rae.eval()
    return rae


def toy_pair_table(docs, rng, n_random_states=2):
    """Curriculum features/labels over synthetic (_synth_doc) docs.
    Returns (X (m,4h+6) float32, y (m,) float32)."""
    feats, ys = [], []
    for _w, lab, V in docs:
        Pd = prefix_sum(V)
        for _name, cset in curriculum_states(lab, rng, n_random_states):
            prs = pairs_from_cuts(cset, len(lab))
            if len(prs) == 0:
                continue
            feats.append(build_features(V, Pd, prs))
            ys.append((1 - lab[prs[:, 1]]).astype(np.float32))
    return np.concatenate(feats), np.concatenate(ys)


def allcut_slice_auc(model_or_rae, docs, use_judge=True):
    """On synthetic docs, all-cut-state slice AUC in the judge convention:
    P(score-for-MERGE higher on a within-sentence pair than on a true
    boundary pair). use_judge=False scores composability -E of a bare RAE."""
    import torch
    sm, sk = [], []
    for _w, lab, V in docs:
        n = len(lab)
        prs = pairs_from_cuts(np.arange(n - 1), n)
        if use_judge:
            X = build_features(V, prefix_sum(V), prs)
            s, _ = judge_with_E(model_or_rae, X)
        else:
            Vn = normalize_rows(V)
            with torch.no_grad():
                _, e = model_or_rae.compose(torch.from_numpy(Vn[:-1]),
                                            torch.from_numpy(Vn[1:]))
            s = -e.numpy()
        y = lab[:-1]
        sm.extend(s[y == 0].tolist())             # within-sentence: MERGE
        sk.extend(s[y == 1].tolist())             # boundary: KEEP
    return roc_auc(np.asarray(sm), np.asarray(sk))


def finetune_toy(rae, docs, arm, epochs=4, seed=0, lr=1e-3, rae_lr=1e-3):
    """Fine-tune a judge on synthetic curriculum pairs; returns the model."""
    rng = np.random.default_rng(seed)
    X, y = toy_pair_table(docs, rng)
    h = rae.h
    model = make_judge(rae, h, seed=seed)
    fit_scaler(model, [X])
    opt = make_optimizer(model, arm, lr, rae_lr)
    w_keep = float(y.sum() / max((1 - y).sum(), 1))
    train_epochs(model, opt, lambda sel: X[sel], y, epochs, 256, w_keep,
                 rng, verbose=False)
    return model


# --------------------------------------------------------------------------- #
# self-test (corpus-independent; no data/, no caches, no encoder)
# --------------------------------------------------------------------------- #
def self_test():
    import torch
    rng = np.random.default_rng(0)
    h = 16

    # 1. judge input slicing matches a naive construction
    rae = make_rae(h, seed=0)
    model = make_judge(rae, h, seed=0)
    _w, lab, V = _synth_doc(rng, n_sent=4, max_len=5, h=h)
    prs = pairs_from_cuts(gold_cuts(lab), len(lab))
    X = build_features(V, prefix_sum(V), prs)
    with torch.no_grad():
        J = model.judge_inputs(torch.from_numpy(X)).numpy()
    a0, c, b1 = prs[0]
    poolA = V[a0:c + 1].mean(axis=0)
    poolB = V[c + 1:b1 + 1].mean(axis=0)
    with torch.no_grad():
        p1, e1 = rae.compose(
            torch.from_numpy(normalize_rows(poolA[None])),
            torch.from_numpy(normalize_rows(poolB[None])))
        p2, e2 = rae.compose(
            torch.from_numpy(normalize_rows(V[c][None])),
            torch.from_numpy(normalize_rows(V[c + 1][None])))
    np.testing.assert_allclose(J[0, :h], p1[0].numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(J[0, h:2 * h], p2[0].numpy(), rtol=1e-4,
                               atol=1e-5)
    np.testing.assert_allclose(J[0, 2 * h], float(e1[0]), rtol=1e-4)
    np.testing.assert_allclose(J[0, 2 * h + 1], float(e2[0]), rtol=1e-4)
    np.testing.assert_allclose(J[0, 2 * h + 2:], X[0, 4 * h:], rtol=1e-5)
    print("PASS judge inputs = [p_seg | p_word | E_seg | E_word | scalars]")

    # 2. all-cut state: p_seg == p_word (single-word segments); probs valid
    prs_ac = pairs_from_cuts(np.arange(len(lab) - 1), len(lab))
    X_ac = build_features(V, prefix_sum(V), prs_ac)
    with torch.no_grad():
        J_ac = model.judge_inputs(torch.from_numpy(X_ac)).numpy()
    np.testing.assert_allclose(J_ac[:, :h], J_ac[:, h:2 * h], rtol=1e-4,
                               atol=1e-5)
    p = FTJudge(model)(X_ac)
    assert p.shape == (len(prs_ac),) and (p >= 0).all() and (p <= 1).all()
    assert FTJudge(model)(np.zeros((0, 4 * h + N_SCALARS),
                                   dtype=np.float32)).shape == (0,)
    print("PASS all-cut state recovers the leaf gate (p_seg == p_word)")

    # 3. arm mechanics: frozenW leaves the composer bit-identical, head moves;
    #    full moves both
    docs = [_synth_doc(np.random.default_rng(s), n_sent=5, max_len=5, h=h,
                       sep=4.0) for s in range(10)]
    rae_a = make_rae(h, seed=1)
    w0 = {k: v.clone() for k, v in rae_a.state_dict().items()}
    m_a = finetune_toy(rae_a, docs, "frozenW", epochs=1, seed=1)
    for k, v in m_a.rae.state_dict().items():
        assert torch.equal(w0[k], v), f"frozenW moved composer weight {k}"
    h0 = make_judge_head(h, seed=1).state_dict()
    moved = any(not torch.equal(h0[k], v)
                for k, v in m_a.head.state_dict().items())
    assert moved, "frozenW arm did not train the head"
    rae_b = make_rae(h, seed=1)
    w0 = {k: v.clone() for k, v in rae_b.state_dict().items()}
    m_b = finetune_toy(rae_b, docs, "full", epochs=1, seed=1)
    moved = any(not torch.equal(w0[k], v)
                for k, v in m_b.rae.state_dict().items())
    assert moved, "full arm did not move the composer"
    print("PASS arm mechanics (frozenW: composer frozen; full: composer moves)")

    # 4. THE PLANTED-INVERSION PROOF (the commissioned synthetic case):
    #    supervision must flip a planted inverted prior
    train_docs = [_synth_doc(np.random.default_rng(100 + s), n_sent=6,
                             max_len=6, h=h, sep=4.0) for s in range(30)]
    held_docs = [_synth_doc(np.random.default_rng(900 + s), n_sent=6,
                            max_len=6, h=h, sep=4.0) for s in range(8)]
    rae_inv = plant_inverted_rae(h, train_docs, steps=300, seed=0)
    auc_prior = allcut_slice_auc(rae_inv, held_docs, use_judge=False)
    assert auc_prior < 0.40, \
        f"prior not inverted on toy data (composability AUC {auc_prior:.3f})"
    for arm in ARMS:
        rae_c = make_rae(h, seed=0)
        rae_c.load_state_dict(rae_inv.state_dict())
        m = finetune_toy(rae_c, train_docs, arm, epochs=4, seed=0)
        auc_ft = allcut_slice_auc(m, held_docs, use_judge=True)
        print(f"  planted prior AUC={auc_prior:.3f} (inverted) -> "
              f"fine-tuned [{arm}] judge AUC={auc_ft:.3f}")
        assert auc_ft >= 0.90, \
            f"[{arm}] failed to un-invert the planted prior ({auc_ft:.3f})"
    print("PASS supervision flips a planted inverted prior (both arms)")

    # 5. wrapper drives merge_judge's policies: eraser bookkeeping + the
    #    agglomerative decode recovers the planted segmentation
    m = finetune_toy(make_rae(h, seed=2), train_docs, "full", epochs=4,
                     seed=2)
    judge = FTJudge(m)
    _w, lab_t, V_t = _synth_doc(np.random.default_rng(999), n_sent=6,
                                max_len=6, h=h, sep=4.0)
    got = agglomerative_cuts(V_t, prefix_sum(V_t), judge, 0.5)
    want = gold_cuts(lab_t)
    assert set(got.tolist()) == set(want.tolist()), \
        f"agglomerative failed: got {got}, want {want}"
    pred = np.ones(len(lab_t), dtype=bool)
    cuts, mps = eraser_merge_probs(V_t, prefix_sum(V_t), pred, judge)
    assert cuts.tolist() == list(range(len(lab_t) - 1))
    kept = np.asarray(sorted(set(cuts) - set(cuts[mps > 0.5])))
    assert set(kept.tolist()) == set(want.tolist()), "eraser missed the gold"
    print("PASS wrapper drives eraser + agglomerative (recovers planted cuts)")

    # 6. checkpoint round trip preserves the judge exactly
    import tempfile
    X_t = build_features(V_t, prefix_sum(V_t),
                         pairs_from_cuts(np.arange(len(lab_t) - 1),
                                         len(lab_t)))
    p_before = judge(X_t)
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "j.pt")
        torch.save({"rae_state": m.rae.state_dict(),
                    "head_state": m.head.state_dict(),
                    "mu": m.mu.numpy(), "sd": m.sd.numpy(), "h": h,
                    "layer": "final", "arm": "full", "history": [],
                    "config": {}}, pth)
        m2, _ = load_judge(pth)
        p_after = FTJudge(m2)(X_t)
    np.testing.assert_allclose(p_before, p_after, rtol=1e-5, atol=1e-6)
    print("PASS checkpoint round trip")
    print("SELF-TEST OK")


def gold_cuts(labels):
    from merge_judge import gold_cut_positions
    return gold_cut_positions(labels)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test")

    t = sub.add_parser("train")
    t.add_argument("--layer", choices=["final", "mid8"], required=True)
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--rae", required=True)
    t.add_argument("--train-jsonl", required=True)
    t.add_argument("--train-vecs", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--rae-lr", type=float, default=1e-4,
                   help="small composer learning rate (arm full only)")
    t.add_argument("--n-random-states", type=int, default=3)
    t.add_argument("--threads", type=int, default=6)
    t.add_argument("--limit-docs", type=int, default=0)

    v = sub.add_parser("eval")
    v.add_argument("--layer", choices=["final", "mid8"], required=True)
    v.add_argument("--judge", required=True)
    v.add_argument("--dev-jsonl", required=True)
    v.add_argument("--dev-vecs", required=True)
    v.add_argument("--probs", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--solo-taus", default="",
                   help="default: headline 0.5 only (runtime); "
                        "'0.3,0.5,0.7' restores merge_judge's list")
    v.add_argument("--skip-solo", action="store_true")
    v.add_argument("--threads", type=int, default=6)
    v.add_argument("--limit-docs", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "self-test":
        self_test()
    elif args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
