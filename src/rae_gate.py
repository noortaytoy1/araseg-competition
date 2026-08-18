"""RAE-GATE — Socher-style Recursive Autoencoder composability gate for AraSeg
NoPnx-NP (Noor's hint). ADDITIVE, CPU-ONLY (the GPU is owned by the integration
battery). Read-only on runs/, probs/, knn_store/, merge_judge/, data/.
Artifacts land under rae_gate/ — a NEW additive directory.

MODEL (Socher, Pennington, Huang, Ng, Manning — EMNLP 2011):
    composition     p = tanh(W [c1; c2] + b)          (768+768 -> 768)
    reconstruction  [c1'; c2'] = W' p + b'            (768 -> 768+768)
    loss            E = ||c1 - c1'||^2 + ||c2 - c2'||^2
  Children and the composed parent are unit-normalized (the standard Socher
  trick against the shrinking-hidden degenerate solution). ~2.4M params.

TRAINING (self-supervised — NO boundary labels touch training; train docs
only; dev is EVALUATION of the signal): each train doc is chunked into
fixed-length windows of cached word vectors; within a window the adjacent
pair with the LOWEST reconstruction error is merged repeatedly (greedy,
selection under no_grad, the chosen merge recomputed with grad, parents feed
later merges) and the loss is accumulated over the whole merge sequence.
Sentence structure is never given — the RAE just learns what composes.

WORD VECTORS (reused caches; frozen union-NoPnx-NP-arabertv02-s42 encoder):
  final : knn_store/NoPnx-NP_s42_train.npz + knn_store/NoPnx-NP_s42_dev_q.npz
  mid8  : merge_judge/NoPnx-NP_s42_{train,dev}_mid8vec.npz
          (NOTE OF RECORD, inherited from merge_judge.py: the unet2d/ mid-8
          caches store only 180x180 cosine MATRICES — pooled word vectors are
          not recoverable from them, so the mid-8 word-vector caches that
          merge_judge encoded additively are the reusable mid-8 source.)
  Compared: mid-8 carried the coherence structure before (0.76 vs 0.56).

THE GATE (dev, evaluation only): for every adjacent word pair, the trained
RAE's reconstruction error E(i, i+1) at the LEAF level, and — reported
alongside — E of the pair spanning the junction after k greedy merges within
a +/-10-word neighborhood (the crossing pair itself is never merged).

AUCs (orientation stated so the pre-registered bars read in the expected
direction: composition should FAIL, i.e. E should be HIGH, at boundaries):
  (1) OVERALL: E at TRUE-boundary junctions vs within-sentence junctions;
      AUC = P(E higher at a true boundary).
  (2) THE SLICE: the tagger's 2,236 confident-FP junctions (p>=0.8, gold=0)
      vs 1,000 sampled confident-TP junctions (p>=0.8, gold=1);
      AUC = P(composability = -E higher on a conf-FP)  =  P(E higher on a
      conf-TP). Reference points on this slice: instability 0.711,
      2D signal 0.6212, raw surprisal 0.52.
  (3) per-genre E means (genealogy chains: does the 'X ولد Y' link compose?
      hadith: the 'بن' chains; cloze: does فراغ compose with what follows?).

FAILURE ANATOMY (standing order): E distributions per group, the highest-E
within-sentence pairs and the lowest-E cross-boundary pairs (the failure
cases), both layer choices.

PRE-REGISTERED READINGS (tick one, verbatim, per layer/variant + best):
  [BUILD] slice >= 0.65 OR overall >= 0.75 -> build the recursive
          erase-decoder (greedy merge until E exceeds threshold — Noor's
          full pipeline) + feed E as a stacker feature
  [PARK ] slice < 0.55 AND overall < 0.65 -> park with the numbers
  [NOOR ] between -> Noor's call

Stages (CPU only; resume-free, minutes-per-epoch scale):

  CUDA_VISIBLE_DEVICES=-1 python src/rae_gate.py self-test

  CUDA_VISIBLE_DEVICES=-1 python src/rae_gate.py train --layer final \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --train-vecs knn_store/NoPnx-NP_s42_train.npz \
      --out rae_gate/rae_final.pt
  CUDA_VISIBLE_DEVICES=-1 python src/rae_gate.py train --layer mid8 \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --train-vecs merge_judge/NoPnx-NP_s42_train_mid8vec.npz \
      --out rae_gate/rae_mid8.pt

  CUDA_VISIBLE_DEVICES=-1 python src/rae_gate.py eval --layer final \
      --rae rae_gate/rae_final.pt \
      --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --dev-vecs knn_store/NoPnx-NP_s42_dev_q.npz \
      --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --out rae_gate/report_final.json
  (same with --layer mid8 / rae_mid8.pt / merge_judge dev mid8 vecs.)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from data import PARAGRAPH_TOKEN, load_jsonl
from diag_mask_influence import roc_auc, summarize
from genre_buckets import bucket
from merge_judge import load_doc_vectors

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

CONF = 0.8                # "confident" per the standing order
NEIGH_DEFAULT = 10        # +/- words around the junction for the merged gate
KS_DEFAULT = "4,8"        # greedy-merge depths reported for the merged gate
GENRE_ORDER = ["general prose", "legal/constitution", "hadith/isnad",
               "genealogy", "cloze/exercise"]
PROBE_WORDS = {"genealogy": ["ولد"], "hadith/isnad": ["بن"],
               "cloze/exercise": ["فراغ"]}


def assert_cpu_only():
    import torch
    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the battery owns the GPU)"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def make_rae(h, seed=42):
    import torch
    import torch.nn as nn

    class RAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = h
            self.enc = nn.Linear(2 * h, h)     # W, b
            self.dec = nn.Linear(h, 2 * h)     # W', b'

        def compose(self, c1, c2):
            """(m,h),(m,h) unit-norm children -> (parent (m,h) unit-norm,
            reconstruction error (m,))."""
            x = torch.cat([c1, c2], dim=-1)
            p = torch.tanh(self.enc(x))
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            r = self.dec(p)
            e = ((c1 - r[..., :self.h]) ** 2).sum(-1) \
                + ((c2 - r[..., self.h:]) ** 2).sum(-1)
            return p, e

    torch.manual_seed(seed)
    return RAE()


def normalize_rows(V):
    """float32 unit-norm rows (cached vectors arrive float16)."""
    V = np.asarray(V, dtype=np.float32)
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    return V / np.maximum(n, 1e-8)


# --------------------------------------------------------------------------- #
# greedy training (unsupervised; structure never given)
# --------------------------------------------------------------------------- #
def greedy_window_loss(model, V):
    """One window's accumulated loss: repeatedly merge the adjacent pair with
    the lowest reconstruction error; selection under no_grad, the chosen merge
    recomputed with grad; parents feed later merges. Returns mean E over the
    len(V)-1 merges (torch scalar with grad)."""
    import torch
    nodes = [V[i] for i in range(len(V))]
    losses = []
    while len(nodes) > 1:
        if len(nodes) == 2:
            j = 0
        else:
            with torch.no_grad():
                _, e_sel = model.compose(torch.stack(nodes[:-1]),
                                         torch.stack(nodes[1:]))
            j = int(torch.argmin(e_sel))
        p, e = model.compose(nodes[j].unsqueeze(0), nodes[j + 1].unsqueeze(0))
        losses.append(e[0])
        nodes[j:j + 2] = [p[0]]
    return torch.stack(losses).mean()


def greedy_group_loss(model, Vs):
    """Lockstep-batched greedy_window_loss over a GROUP of windows: identical
    merge decisions and loss values, but one selection compose + one merge
    compose per step for the whole group (CPU-speed necessity). Returns the
    mean over windows of the per-window mean merge E (torch scalar, grad)."""
    import torch
    nodes = [[V[i] for i in range(len(V))] for V in Vs]
    terms = [[] for _ in Vs]
    while True:
        active = [w for w in range(len(Vs)) if len(nodes[w]) > 1]
        if not active:
            break
        segs, c1, c2 = [], [], []
        for w in active:
            nd = nodes[w]
            segs.append(len(nd) - 1)
            c1.extend(nd[:-1])
            c2.extend(nd[1:])
        with torch.no_grad():
            _, e = model.compose(torch.stack(c1), torch.stack(c2))
        js, off = [], 0
        for m in segs:
            js.append(int(torch.argmin(e[off:off + m])) if m > 1 else 0)
            off += m
        m1 = torch.stack([nodes[w][j] for w, j in zip(active, js)])
        m2 = torch.stack([nodes[w][j + 1] for w, j in zip(active, js)])
        p, em = model.compose(m1, m2)
        for r, (w, j) in enumerate(zip(active, js)):
            terms[w].append(em[r])
            nodes[w][j:j + 2] = [p[r]]
    return torch.stack([torch.stack(t).mean() for t in terms]).mean()


def doc_windows(V, window):
    """Non-overlapping windows of unit-norm rows; a length-1 remainder is
    dropped (a single word offers no pair)."""
    out = []
    for s in range(0, len(V), window):
        chunk = V[s:s + window]
        if len(chunk) >= 2:
            out.append(chunk)
    return out


def cmd_train(args):
    assert_cpu_only()
    import torch
    torch.set_num_threads(max(1, args.threads))

    docs = load_jsonl(args.train_jsonl)
    for d in docs:
        assert all(w != PARAGRAPH_TOKEN for w in d["tokens"]), \
            "unexpected PAR token in a NoPnx track"
    vecs = load_doc_vectors(args.train_vecs, docs)   # needs the full doc list
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    h = next(iter(vecs.values())).shape[1]

    windows = []
    for d in docs:
        windows.extend(doc_windows(normalize_rows(vecs[d["doc_id"]]),
                                   args.window))
    n_words = sum(len(w) for w in windows)
    print(f"# RAE-GATE train [{args.layer}]: {len(docs)} docs -> "
          f"{len(windows)} windows of <= {args.window} words "
          f"({n_words} words, h={h})", flush=True)
    print("# self-supervised: boundary labels are never read here", flush=True)

    model = make_rae(h, seed=args.seed)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"# params={n_par}  epochs={args.epochs}  lr={args.lr}  "
          f"windows/step={args.group}  seed={args.seed}  "
          f"threads={args.threads}", flush=True)

    rng = np.random.default_rng(args.seed)
    history = []
    for ep in range(args.epochs):
        order = rng.permutation(len(windows))
        model.train()
        ep_loss, t0 = 0.0, time.time()
        for s in range(0, len(order), args.group):
            batch = [torch.from_numpy(windows[j])
                     for j in order[s:s + args.group]]
            loss = greedy_group_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()) * len(batch)
        ep_loss /= len(windows)
        history.append(ep_loss)
        print(f"  epoch {ep + 1}/{args.epochs}: mean-merge-E={ep_loss:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        model.eval()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "h": h,
                    "layer": args.layer, "loss_history": history,
                    "config": {"seed": args.seed, "epochs": args.epochs,
                               "window": args.window, "group": args.group,
                               "lr": args.lr,
                               "train_jsonl": args.train_jsonl,
                               "train_vecs": args.train_vecs}}, args.out)
    print(f"# loss first->last epoch: {history[0]:.5f} -> {history[-1]:.5f}")
    print(f"RAE [{args.layer}] -> {args.out}", flush=True)


def load_rae(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = make_rae(ck["h"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


# --------------------------------------------------------------------------- #
# the gate: leaf-level E and E after k greedy neighborhood merges
# --------------------------------------------------------------------------- #
def leaf_errors(model, V):
    """E(i, i+1) for every adjacent pair of one doc; V (n,h) unit-norm
    float32 numpy. Returns (n-1,) float64."""
    import torch
    with torch.no_grad():
        t = torch.from_numpy(V)
        _, e = model.compose(t[:-1], t[1:])
    return e.numpy().astype(np.float64)


def cross_error_after_merges(model, V, i, ks, neigh):
    """E of the pair SPANNING the junction after-word-i, after k greedy merges
    within words [i-neigh, i+neigh+1); the crossing pair itself is never
    merged. Returns {k: E} for each k in ks (early-stops when only the
    crossing pair remains, carrying the current E forward)."""
    import torch
    n = len(V)
    assert 0 <= i <= n - 2, "junction needs a right-hand word"
    lo, hi = max(0, i - neigh), min(n - 1, i + neigh)
    t = torch.from_numpy(V)
    nodes = [t[w] for w in range(lo, hi + 1)]
    ends = list(range(lo, hi + 1))       # last word index under each node
    ks = sorted(set(int(k) for k in ks))
    res, done = {}, 0
    with torch.no_grad():
        while True:
            _, e = model.compose(torch.stack(nodes[:-1]),
                                 torch.stack(nodes[1:]))
            cross = ends.index(i)        # pair (cross, cross+1) spans i/i+1
            if done in ks:
                res[done] = float(e[cross])
            if done >= ks[-1] or len(nodes) == 2:
                for k in ks:             # early stop: carry current E forward
                    res.setdefault(k, float(e[cross]))
                return res
            e[cross] = float("inf")
            j = int(torch.argmin(e))
            p, _ = model.compose(nodes[j].unsqueeze(0),
                                 nodes[j + 1].unsqueeze(0))
            nodes[j:j + 2] = [p[0]]
            ends[j:j + 2] = [ends[j + 1]]
            done += 1


# --------------------------------------------------------------------------- #
# evaluation (dev = evaluation of the signal only)
# --------------------------------------------------------------------------- #
def tick_reading(tag, overall, slice_):
    build = (slice_ >= 0.65) or (overall >= 0.75)
    park = (slice_ < 0.55) and (overall < 0.65)
    print(f"\n--- PRE-REGISTERED READING [{tag}]  "
          f"overall={overall:.4f}  slice={slice_:.4f} ---")
    print(f"  [{'X' if build else ' '}] slice >= 0.65 OR overall >= 0.75 -> "
          "build the recursive erase-decoder (greedy merge until E exceeds "
          "threshold — Noor's full pipeline) + feed E as a stacker feature")
    print(f"  [{'X' if park else ' '}] slice < 0.55 AND overall < 0.65 -> "
          "park with the numbers")
    print(f"  [{'X' if (not build and not park) else ' '}] between -> "
          "Noor's call")
    return "BUILD" if build else ("PARK" if park else "NOOR")


def cmd_eval(args):
    assert_cpu_only()
    import torch
    torch.set_num_threads(max(1, args.threads))
    t_start = time.time()

    docs = load_jsonl(args.dev_jsonl)
    vecs = load_doc_vectors(args.dev_vecs, docs)     # needs the full doc list
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    model, ck = load_rae(args.rae)
    assert ck["layer"] == args.layer, \
        f"RAE was trained on layer {ck['layer']}, eval asked for {args.layer}"
    pz = np.load(args.probs)
    rng = np.random.default_rng(args.seed)
    ks = [int(k) for k in args.ks.split(",")]

    print(f"# RAE-GATE eval [{args.layer}]  rae={args.rae}  "
          f"train-loss last epoch={ck['loss_history'][-1]:.5f}")
    print(f"# dev={args.dev_jsonl} ({len(docs)} docs)  probs={args.probs}  "
          f"seed={args.seed}  neigh=+/-{args.neigh}w  ks={ks}", flush=True)

    # ---- leaf-level E for EVERY adjacent pair, pooled -----------------------
    E, G, PT, DI, POS = [], [], [], [], []
    genres = {}
    Vn = {}
    for di, d in enumerate(docs):
        assert all(w != PARAGRAPH_TOKEN for w in d["tokens"]), \
            "unexpected PAR token in a NoPnx track"
        V = normalize_rows(vecs[d["doc_id"]])
        Vn[di] = V
        g = np.asarray(d["labels"], dtype=np.int64)
        p = pz[d["doc_id"]].astype(np.float64)
        assert len(g) == len(p) == len(V)
        e = leaf_errors(model, V)                 # junctions i = 0 .. n-2
        E.append(e)
        G.append(g[:-1])                          # doc-final label is no pair
        PT.append(p[:-1])
        DI.append(np.full(len(e), di, dtype=np.int64))
        POS.append(np.arange(len(e), dtype=np.int64))
        genres[di] = bucket(d["tokens"])
    E = np.concatenate(E)
    G = np.concatenate(G)
    PT = np.concatenate(PT)
    DI = np.concatenate(DI)
    POS = np.concatenate(POS)
    print(f"# pooled junctions={len(E)}  true-boundary={int(G.sum())}  "
          f"within-sentence={int((G == 0).sum())}  "
          f"({time.time() - t_start:.0f}s)", flush=True)

    # ---- groups (seeded; mirrors diag_mask_influence's sampling) ------------
    fp_mask = (PT >= args.hi) & (G == 0)
    tp_mask = (PT >= args.hi) & (G == 1)

    def sample_idx(idx, k):
        if k <= 0 or len(idx) <= k:
            return idx
        return idx[rng.choice(len(idx), size=k, replace=False)]

    idx_fp = np.flatnonzero(fp_mask)              # ALL confident FPs (record: 2236)
    idx_fp = sample_idx(idx_fp, args.n_fp)
    idx_tp = sample_idx(np.flatnonzero(tp_mask), args.n_tp)
    idx_bound = sample_idx(np.flatnonzero(G == 1), args.n_bound)
    # matched non-boundaries: same docs as sampled boundaries, interior only
    nonb_by_doc = defaultdict(list)
    for j in np.flatnonzero((G == 0) & (POS >= 1)):
        nonb_by_doc[DI[j]].append(j)
    per_doc = defaultdict(int)
    for j in idx_bound:
        per_doc[DI[j]] += 1
    nonb = []
    for di, cnt in per_doc.items():
        pool = np.asarray(nonb_by_doc.get(di, []), dtype=np.int64)
        nonb.extend(sample_idx(pool, cnt).tolist())
    want = min(args.n_nonb, sum(len(v) for v in nonb_by_doc.values()))
    if len(nonb) > want:
        nonb = [nonb[j] for j in
                rng.choice(len(nonb), size=want, replace=False)]
    elif len(nonb) < want:
        chosen = set(nonb)
        rest = np.asarray([x for di in per_doc for x in nonb_by_doc[di]
                           if x not in chosen], dtype=np.int64)
        nonb.extend(sample_idx(rest, want - len(nonb)).tolist())
    idx_nonb = np.asarray(sorted(nonb), dtype=np.int64)
    groups = {"BOUND": idx_bound, "NONB": idx_nonb,
              "CONF-FP": idx_fp, "CONF-TP": idx_tp}
    print(f"# groups: BOUND={len(idx_bound)} NONB={len(idx_nonb)} "
          f"CONF-FP={len(idx_fp)} (pool {int(fp_mask.sum())}, record 2236) "
          f"CONF-TP={len(idx_tp)} (pool {int(tp_mask.sum())})", flush=True)

    # ---- merged-gate E for the sampled groups (leaf gate is global) ---------
    if args.limit:
        groups = {gname: idx[:args.limit] for gname, idx in groups.items()}
    uniq = sorted({int(j) for idx in groups.values() for j in idx})
    t0 = time.time()
    Ek = {k: np.full(len(E), np.nan) for k in ks}
    for cnt, j in enumerate(uniq):
        res = cross_error_after_merges(model, Vn[int(DI[j])], int(POS[j]),
                                       ks, args.neigh)
        for k in ks:
            Ek[k][j] = res[k]
        if (cnt + 1) % 2000 == 0 or cnt + 1 == len(uniq):
            print(f"#   merged gate {cnt + 1}/{len(uniq)} junctions "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- (4) distributions / failure anatomy --------------------------------
    variants = [("leaf", E)] + [(f"k={k}", Ek[k]) for k in ks]
    print("\n--- E distributions per group (per variant) ---")
    for vname, arr in variants:
        print(f"  [{vname}]")
        for gname in ["BOUND", "NONB", "CONF-FP", "CONF-TP"]:
            print("    " + summarize(gname, arr[groups[gname]]))

    # ---- (1) + (2) AUCs ------------------------------------------------------
    aucs = {}
    print("\n--- (1) OVERALL AUC: E at true-boundary vs within-sentence "
          "junctions  [AUC = P(E higher at boundary)] ---")
    for vname, arr in variants:
        if vname == "leaf":
            a_full = roc_auc(arr[G == 1], arr[G == 0])
            print(f"    {vname:6s} ALL junctions ({int(G.sum())} vs "
                  f"{int((G == 0).sum())}): {a_full:.4f}")
            aucs[f"overall_{vname}_all"] = a_full
        a = roc_auc(arr[groups["BOUND"]], arr[groups["NONB"]])
        print(f"    {vname:6s} sampled BOUND vs NONB: {a:.4f}")
        aucs[f"overall_{vname}_sampled"] = a
    print("\n--- (2) SLICE AUC: conf-FP vs conf-TP (the tagger-blind slice) "
          "[AUC = P(composability = -E higher on conf-FP)] ---")
    print("    reference points on this slice: instability 0.711, "
          "2D signal 0.6212, raw surprisal 0.52")
    for vname, arr in variants:
        a = roc_auc(-arr[groups["CONF-FP"]], -arr[groups["CONF-TP"]])
        print(f"    {vname:6s}: {a:.4f}")
        aucs[f"slice_{vname}"] = a

    # ---- (3) per-genre -------------------------------------------------------
    print("\n--- (3) per-genre leaf E (BOUND = true-boundary junctions, "
          "NONB = within-sentence) ---")
    print(f"    {'genre':20s} {'grp':6s} {'n':>7} {'mean E':>9} "
          f"{'median':>9}   per-genre AUC(E: BOUND>NONB)")
    genre_stats = {}
    for gname in GENRE_ORDER:
        dis = [di for di, gg in genres.items() if gg == gname]
        if not dis:
            continue
        m = np.isin(DI, dis)
        eb, ew = E[m & (G == 1)], E[m & (G == 0)]
        ga = roc_auc(eb, ew)
        genre_stats[gname] = {"auc": ga, "n_bound": int(len(eb)),
                              "n_within": int(len(ew)),
                              "mean_bound": float(np.mean(eb)) if len(eb) else None,
                              "mean_within": float(np.mean(ew)) if len(ew) else None}
        for tag, v in [("BOUND", eb), ("NONB", ew)]:
            tail = f"   AUC={ga:.3f}" if tag == "BOUND" else ""
            print(f"    {gname:20s} {tag:6s} {len(v):7d} "
                  f"{np.mean(v):9.4f} {np.median(v):9.4f}{tail}")
        for w in PROBE_WORDS.get(gname, []):
            left = [j for j in np.flatnonzero(m)
                    if docs[DI[j]]["tokens"][POS[j]] == w]
            right = [j for j in np.flatnonzero(m)
                     if docs[DI[j]]["tokens"][POS[j] + 1] == w]
            for side, jj in [(f"('{w}' X)", left), (f"(X '{w}')", right)]:
                if jj:
                    v = E[np.asarray(jj)]
                    gg = G[np.asarray(jj)]
                    print(f"    {gname:20s}   probe {side}: n={len(v)} "
                          f"mean E={v.mean():.4f} "
                          f"(bound-rate {gg.mean():.2f})")
                    genre_stats[gname][f"probe {side}"] = {
                        "n": int(len(v)), "mean_E": float(v.mean()),
                        "bound_rate": float(gg.mean())}

    # ---- failure examples ----------------------------------------------------
    def examples(idx_pool, arr, best_high, n=10):
        order = np.argsort(arr[idx_pool])
        pick = idx_pool[order[::-1][:n]] if best_high else idx_pool[order[:n]]
        out = []
        for j in pick:
            d = docs[DI[j]]
            i = int(POS[j])
            out.append({"doc_id": d["doc_id"], "i": i,
                        "left": d["tokens"][i], "right": d["tokens"][i + 1],
                        "E": float(arr[j]), "p_tag": float(PT[j]),
                        "gold": int(G[j])})
        return out

    print("\n--- failure cases (leaf E) ---")
    ex_hi_within = examples(np.flatnonzero(G == 0), E, best_high=True)
    print("  highest-E WITHIN-SENTENCE pairs (composition fails inside a "
          "sentence — false alarms of the gate):")
    for x in ex_hi_within:
        print(f"    E={x['E']:.4f} p_tag={x['p_tag']:.3f} "
              f"{x['doc_id']} i={x['i']}: '{x['left']}' | '{x['right']}'")
    ex_lo_cross = examples(np.flatnonzero(G == 1), E, best_high=False)
    print("  lowest-E CROSS-BOUNDARY pairs (true boundaries that compose "
          "fine — the gate's misses):")
    for x in ex_lo_cross:
        print(f"    E={x['E']:.4f} p_tag={x['p_tag']:.3f} "
              f"{x['doc_id']} i={x['i']}: '{x['left']}' | '{x['right']}'")

    # ---- readings ------------------------------------------------------------
    verdicts = {}
    for vname, _arr in variants:
        overall = aucs[f"overall_{vname}_all"] if vname == "leaf" \
            else aucs[f"overall_{vname}_sampled"]
        verdicts[vname] = tick_reading(f"{args.layer}/{vname}", overall,
                                       aucs[f"slice_{vname}"])
    best = max(variants, key=lambda v: max(
        aucs[f"overall_{v[0]}_all"] if v[0] == "leaf"
        else aucs[f"overall_{v[0]}_sampled"], aucs[f"slice_{v[0]}"]))[0]
    overall_b = aucs[f"overall_{best}_all"] if best == "leaf" \
        else aucs[f"overall_{best}_sampled"]
    verdicts["best"] = tick_reading(f"{args.layer}/best={best}", overall_b,
                                    aucs[f"slice_{best}"])

    report = {"layer": args.layer, "rae": args.rae, "probs": args.probs,
              "dev_jsonl": args.dev_jsonl, "dev_vecs": args.dev_vecs,
              "seed": args.seed, "neigh": args.neigh, "ks": ks,
              "hi": args.hi, "n_junctions": int(len(E)),
              "group_sizes": {g: int(len(idx)) for g, idx in groups.items()},
              "conf_fp_pool": int(fp_mask.sum()),
              "conf_tp_pool": int(tp_mask.sum()),
              "aucs": {k: float(v) for k, v in aucs.items()},
              "per_genre": genre_stats,
              "examples_high_E_within": ex_hi_within,
              "examples_low_E_cross": ex_lo_cross,
              "train_loss_history": ck["loss_history"],
              "verdicts": verdicts}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nreport -> {args.out}")
    print(f"# done in {(time.time() - t_start) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- #
# self-test (corpus-independent; no data/, no caches, no encoder)
# --------------------------------------------------------------------------- #
def self_test():
    import torch
    from merge_judge import _synth_doc
    rng = np.random.default_rng(0)

    # 1. composition/reconstruction math matches a naive numpy replica
    model = make_rae(8, seed=0)
    c1 = normalize_rows(rng.normal(size=(5, 8)))
    c2 = normalize_rows(rng.normal(size=(5, 8)))
    with torch.no_grad():
        p, e = model.compose(torch.from_numpy(c1), torch.from_numpy(c2))
    W = model.enc.weight.detach().numpy()
    b = model.enc.bias.detach().numpy()
    Wp = model.dec.weight.detach().numpy()
    bp = model.dec.bias.detach().numpy()
    x = np.concatenate([c1, c2], axis=1)
    pn = np.tanh(x @ W.T + b)
    pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-8)
    r = pn @ Wp.T + bp
    en = ((c1 - r[:, :8]) ** 2).sum(1) + ((c2 - r[:, 8:]) ** 2).sum(1)
    np.testing.assert_allclose(p.numpy(), pn, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(e.numpy(), en, rtol=1e-4, atol=1e-5)
    assert abs(float(np.linalg.norm(p.numpy()[0])) - 1.0) < 1e-4
    print("PASS composition p=tanh(W[c1;c2]+b) + reconstruction loss math")

    # 2. greedy window loss: torch scalar with grad; params update
    V = torch.from_numpy(normalize_rows(rng.normal(size=(7, 8))
                                        ).astype(np.float32))
    loss = greedy_window_loss(model, V)
    assert loss.requires_grad and loss.ndim == 0 and float(loss.detach()) > 0
    g0 = model.enc.weight.clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    opt.zero_grad(); loss.backward(); opt.step()
    assert not torch.equal(g0, model.enc.weight), "no parameter update"
    print("PASS greedy merge-sequence loss (grad flows through parents)")

    # 2b. lockstep-batched group loss == sequential per-window losses
    Vs = [torch.from_numpy(normalize_rows(rng.normal(size=(L, 8))))
          for L in (5, 2, 9)]
    with torch.no_grad():
        want = torch.stack([greedy_window_loss(model, V) for V in Vs]).mean()
        got = greedy_group_loss(model, Vs)
    np.testing.assert_allclose(float(got), float(want), rtol=1e-4)
    print("PASS lockstep group loss == sequential greedy loss")

    # 3. neighborhood merged gate: k=0 equals the leaf E at the junction;
    #    the crossing pair is never merged; early-stop carries E forward
    Vn = normalize_rows(rng.normal(size=(30, 8)))
    i = 14
    res = cross_error_after_merges(model, Vn, i, ks=[0, 3], neigh=5)
    leaf = leaf_errors(model, Vn)
    assert abs(res[0] - leaf[i]) < 1e-5, "k=0 must equal the leaf gate"
    assert res[3] > 0
    res_edge = cross_error_after_merges(model, Vn, 0, ks=[50], neigh=2)
    assert 50 in res_edge          # early-stopped, value carried forward
    print("PASS neighborhood merged gate bookkeeping (k=0 == leaf; edges)")

    # 4. end-to-end on synthetic clustered docs: training reduces the loss and
    #    leaf E separates cluster boundaries from within-cluster pairs
    docs = [_synth_doc(np.random.default_rng(s), n_sent=6, max_len=6,
                       h=16, sep=4.0) for s in range(24)]
    model = make_rae(16, seed=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    windows = []
    for _w, _lab, Vd in docs:
        windows.extend(doc_windows(normalize_rows(Vd), 12))
    first = last = None
    for ep in range(25):
        ep_loss = 0.0
        for w in windows:
            loss = greedy_window_loss(model, torch.from_numpy(w))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.detach())
        ep_loss /= len(windows)
        first = ep_loss if first is None else first
        last = ep_loss
    assert last < first, f"training did not reduce the loss ({first}->{last})"
    model.eval()
    _w, lab_t, V_t = _synth_doc(np.random.default_rng(99), n_sent=6,
                                max_len=6, h=16, sep=4.0)
    e_t = leaf_errors(model, normalize_rows(V_t))
    g_t = np.asarray(lab_t[:-1])
    auc = roc_auc(e_t[g_t == 1], e_t[g_t == 0])
    assert auc >= 0.75, f"synthetic boundary-vs-within AUC too low: {auc:.3f}"
    print(f"PASS synthetic end-to-end (loss {first:.4f}->{last:.4f}, "
          f"boundary AUC={auc:.3f})")
    print("SELF-TEST OK")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test")

    t = sub.add_parser("train")
    t.add_argument("--layer", choices=["final", "mid8"], required=True)
    t.add_argument("--train-jsonl", required=True)
    t.add_argument("--train-vecs", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--window", type=int, default=32)
    t.add_argument("--group", type=int, default=16,
                   help="windows per optimizer step (lockstep-batched)")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--threads", type=int, default=6,
                   help="torch CPU threads (leave headroom for other jobs)")
    t.add_argument("--limit-docs", type=int, default=0)

    v = sub.add_parser("eval")
    v.add_argument("--layer", choices=["final", "mid8"], required=True)
    v.add_argument("--rae", required=True)
    v.add_argument("--dev-jsonl", required=True)
    v.add_argument("--dev-vecs", required=True)
    v.add_argument("--probs", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--hi", type=float, default=CONF)
    v.add_argument("--n-bound", type=int, default=1500)
    v.add_argument("--n-nonb", type=int, default=1500)
    v.add_argument("--n-fp", type=int, default=0,
                   help="0 = ALL confident FPs (the 2,236 of record)")
    v.add_argument("--n-tp", type=int, default=1000)
    v.add_argument("--neigh", type=int, default=NEIGH_DEFAULT)
    v.add_argument("--ks", default=KS_DEFAULT)
    v.add_argument("--threads", type=int, default=6)
    v.add_argument("--limit-docs", type=int, default=0)
    v.add_argument("--limit", type=int, default=0,
                   help="smoke: cap merged-gate positions per group")
    args = ap.parse_args()

    if args.cmd == "self-test":
        self_test()
    elif args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
