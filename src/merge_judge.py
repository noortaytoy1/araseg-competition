"""MERGE-JUDGE — Noor's erase-policy for AraSeg NoPnx-NP (additive, CPU-only).

A pairwise judge that looks at two ADJACENT SEGMENTS and outputs
P(merge) = P(the cut between them should not exist). The frozen union
encoder (runs/union-NoPnx-NP-arabertv02-s42) supplies word vectors — only a
small MLP head trains, on CPU. Dev is EVALUATION ONLY; all supervision comes
from the 174 TRAIN docs.

1. PAIR SUPERVISION (dense)
   From train gold, every adjacent word pair (i, i+1): MERGE=1 if no gold
   boundary after i, MERGE=0 (KEEP the cut) if there is one. On the 174-doc
   train set that is 88,210 merge / 10,073 keep pairs at the all-cut state
   (10,247 gold boundaries minus the 174 doc-final ones, which never form a
   pair). A candidate pair is represented at SEGMENT level under a current
   segmentation state: given adjacent segments A=[a0..c], B=[c+1..b1],
     features = [ mean(V[A]) 768 | mean(V[B]) 768 | V[c] 768 | V[c+1] 768 |
                  log1p(lenA), log1p(lenB), lenA==1, lenB==1,
                  cos(meanA, meanB), cos(V[c], V[c+1]) ]        (dim 3078)
   V comes from either the FINAL-layer cache (knn_store/*.npz, reused as-is)
   or MID-LAYER-8 vectors. NOTE OF RECORD: the unet2d/ mid-8 caches store
   only 180x180 cosine MATRICES — pooled word vectors are NOT recoverable
   from them, so the mid-8 arm re-runs the frozen encoder once on CPU
   (`encode` below, same ~2-minute pass that built knn_store) and caches the
   vectors additively under merge_judge/. Nothing is retrained.

2. TRAINING CURRICULUM (matches inference granularities) — documented sampling
   Every training state is a REFINEMENT of gold (cuts = gold cuts + a subset
   of sentence-interior positions), so each pair's label is exact:
   MERGE=1 iff the separating cut is not a gold boundary. Per doc:
     * state "allcut" : every adjacent word pair            (~88.2K m / 10.1K k)
     * states "rand0..2" (seeded): per gold sentence draw q~U(0,1), keep each
       interior cut with prob q -> mixed granularities       (~44K m / 10.1K k each)
     * state "gold"   : adjacent FULL SENTENCES at true boundaries -> keep
     * state "half"   : each sentence (len>=2) split at one uniform interior
       point -> half-sentence merge pairs, plus the gold-cut pairs seen with
       half-sentence context
   Class weighting: keep pairs are up-weighted by n_merge/n_keep so both
   classes contribute equally to the BCE loss (decision of record: tau=0.5 is
   then the balanced-prior operating point, i.e. the CONSERVATIVE reading of
   "untuned tau" — erring against erasing true positives).

3. THREE EVALUATIONS on dev (the verdict):
   (a) ERASER-ON-TAGGER: cached tagger cuts (probs>=0.5, the 83.96 baseline);
       each proposed cut is judged against the two segments the TAGGER's own
       segmentation puts around it (one-shot: all cuts judged in the pre-erase
       state, then erased simultaneously — decision of record); erase if
       P(merge) > tau. tau swept on the standard grid (0.30..0.70 step 0.05;
       high-confidence extras 0.80/0.90/0.95/0.99 reported as clearly-marked
       extras); HEADLINE = untuned tau=0.5. Reported per the standing order:
       macro-F1 delta vs 83.96, how many of the 2,236 confident FPs
       (p>=0.8, gold=0) were erased vs how many of the 10,985 confident TPs
       (p>=0.8, gold=1) were wrongly erased, per-genre deltas, P/R
       decomposition.
   (b) SOLO AGGLOMERATIVE: start all-cut, repeatedly erase the highest-
       P(merge) cut (recomputing the two affected neighbour pairs after each
       merge) until no pair exceeds tau. Doc-final position is always a
       boundary (gold convention: every doc ends with label 1).
   (c) DECORRELATION: phi(all errors) + FP-Jaccard of (a)'s and (b)'s dev
       errors vs the tagger's, the exact unet2d_train.py convention
       (reference points: existing-pool inter-voter mean = 0.74; the S-only
       U-Net achieved 0.53).

PRE-REGISTERED GATES (from the commissioning order; evaluated at tau=0.5):
   * eraser adopt-candidate  iff net F1 > +0.15 AND conf-FP erasures > 3x
     conf-TP erasures
   * solo interesting        iff phi(all errors) < 0.6
   * else PARK with the numbers.

Stages (CPU only — the battery owns the GPU; artifacts under merge_judge/,
which is a NEW additive dir; runs/, probs/, knn_store/, unet2d/ untouched):

  CUDA_VISIBLE_DEVICES=-1 python src/merge_judge.py self-test

  CUDA_VISIBLE_DEVICES=-1 python src/merge_judge.py encode \
      --model runs/union-NoPnx-NP-arabertv02-s42 --layer 8 \
      --jsonl data/NoPnx-NP_train.jsonl \
      --out merge_judge/NoPnx-NP_s42_train_mid8vec.npz
  CUDA_VISIBLE_DEVICES=-1 python src/merge_judge.py encode \
      --model runs/union-NoPnx-NP-arabertv02-s42 --layer 8 \
      --jsonl data/NoPnx-NP_dev.jsonl \
      --out merge_judge/NoPnx-NP_s42_dev_mid8vec.npz

  CUDA_VISIBLE_DEVICES=-1 python src/merge_judge.py train --layer final \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --train-vecs knn_store/NoPnx-NP_s42_train.npz \
      --out merge_judge/head_final.pt
  CUDA_VISIBLE_DEVICES=-1 python src/merge_judge.py train --layer mid8 \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --train-vecs merge_judge/NoPnx-NP_s42_train_mid8vec.npz \
      --out merge_judge/head_mid8.pt

  CUDA_VISIBLE_DEVICES=-1 python src/merge_judge.py eval --layer final \
      --head merge_judge/head_final.pt \
      --dev-jsonl data/NoPnx-NP_dev.jsonl \
      --dev-vecs knn_store/NoPnx-NP_s42_dev_q.npz \
      --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --out merge_judge/report_final.json
  (same with --layer mid8 / head_mid8.pt / *_dev_mid8vec.npz — compare both.)
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from collections import defaultdict

import numpy as np

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from genre_buckets import bucket
from knn_boundary import THRESH_GRID, load_encoder, macro_f1

TAU_GRID = THRESH_GRID                       # standard grid: 0.30..0.70 by 0.05
TAU_EXTRA = [0.80, 0.90, 0.95, 0.99]         # clearly-marked extras
TAU_HEADLINE = 0.5
CONF = 0.8                                   # "confident" per the standing order
N_SCALARS = 6
BASELINE_F1_REF = 83.96                      # tagger cache @0.5, record of note


def assert_cpu_only():
    import torch
    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the battery owns the GPU)"


# --------------------------------------------------------------------------- #
# pure segmentation-state helpers (corpus-independent, unit-tested)
# --------------------------------------------------------------------------- #
def gold_cut_positions(labels):
    """Cut positions: i in [0, n-2] with labels[i]==1 (doc-final label is not
    a cut — there is no pair after the last word)."""
    lab = np.asarray(labels, dtype=np.int64)
    return np.flatnonzero(lab[:len(lab) - 1] == 1)


def sentence_spans(labels):
    """Inclusive (start, end) word spans of gold sentences."""
    n = len(labels)
    cuts = gold_cut_positions(labels)
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts, [n - 1]])
    return list(zip(starts.tolist(), ends.tolist()))


def pairs_from_cuts(cuts, n):
    """(m,3) int64 [a0, c, b1]: for each cut c, segment A=[a0..c], B=[c+1..b1]
    under the segmentation whose full cut set is `cuts`."""
    cuts = np.asarray(sorted(int(c) for c in cuts), dtype=np.int64)
    if cuts.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    assert cuts[0] >= 0 and cuts[-1] <= n - 2, "cut out of range"
    a0 = np.concatenate([[0], cuts[:-1] + 1])
    b1 = np.concatenate([cuts[1:], [n - 1]])
    return np.stack([a0, cuts, b1], axis=1)


def curriculum_states(labels, rng, n_random_states=3):
    """Sampled segmentation states for one doc (all refine gold — see module
    docstring, section 2). Returns [(name, cuts int64 array), ...]."""
    n = len(labels)
    g = gold_cut_positions(labels)
    gset = set(g.tolist())
    spans = sentence_spans(labels)
    states = [("allcut", np.arange(n - 1, dtype=np.int64))]
    for r in range(n_random_states):
        cuts = set(gset)
        for (s, e) in spans:
            if e > s:
                q = float(rng.random())
                for i in range(s, e):                    # interior cut slots
                    if rng.random() < q:
                        cuts.add(i)
        states.append((f"rand{r}", np.asarray(sorted(cuts), dtype=np.int64)))
    states.append(("gold", g.copy()))
    half = set(gset)
    for (s, e) in spans:
        if e > s:
            half.add(int(rng.integers(s, e)))            # one split per sentence
    states.append(("half", np.asarray(sorted(half), dtype=np.int64)))
    return states


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def prefix_sum(V):
    """(n+1, h) float64 prefix sums of word vectors."""
    P = np.zeros((V.shape[0] + 1, V.shape[1]), dtype=np.float64)
    np.cumsum(V.astype(np.float64), axis=0, out=P[1:])
    return P


def _cos_rows(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return ((a * b).sum(axis=1) / np.maximum(na * nb, 1e-9)).astype(np.float32)


def _features_idx(V, P, iva, ivb, ipa, ipc, ipb, lenA, lenB):
    """Single implementation of the feature math over (possibly global) index
    columns: iva/ivb index V (last of A / first of B); ipa/ipc/ipb index the
    prefix-sum rows (start of A / start of B / one-past-end of B)."""
    lenA = lenA.astype(np.float64)
    lenB = lenB.astype(np.float64)
    poolA = ((P[ipc] - P[ipa]) / lenA[:, None]).astype(np.float32)
    poolB = ((P[ipb] - P[ipc]) / lenB[:, None]).astype(np.float32)
    vlast = V[iva].astype(np.float32)
    vfirst = V[ivb].astype(np.float32)
    scal = np.stack([
        np.log1p(lenA).astype(np.float32), np.log1p(lenB).astype(np.float32),
        (lenA == 1).astype(np.float32), (lenB == 1).astype(np.float32),
        _cos_rows(poolA, poolB), _cos_rows(vlast, vfirst)], axis=1)
    return np.concatenate([poolA, poolB, vlast, vfirst, scal], axis=1)


def build_features(V, P, pairs):
    """Features for pairs [a0, c, b1] against one doc's V (n,h) / P (n+1,h).
    Returns (m, 4h+6) float32."""
    if len(pairs) == 0:
        return np.zeros((0, 4 * V.shape[1] + N_SCALARS), dtype=np.float32)
    a0, c, b1 = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    return _features_idx(V, P, c, c + 1, a0, c + 1, b1 + 1,
                         c - a0 + 1, b1 - c)


# --------------------------------------------------------------------------- #
# judge head (small MLP; the only thing that trains)
# --------------------------------------------------------------------------- #
def make_head(in_dim, seed=42):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 64), nn.ReLU(),
        nn.Linear(64, 1))


class Judge:
    """Frozen wrapper: standardized features -> P(merge)."""

    def __init__(self, head, mu, sd):
        self.head, self.mu, self.sd = head, mu, sd

    def __call__(self, X):
        import torch
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        Xs = (X - self.mu) / self.sd
        with torch.no_grad():
            z = self.head(torch.from_numpy(Xs))
        return torch.sigmoid(z[:, 0]).numpy()

    @staticmethod
    def load(path):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        head = make_head(ck["in_dim"])
        head.load_state_dict(ck["state_dict"])
        head.eval()
        return Judge(head, ck["mu"], ck["sd"]), ck


# --------------------------------------------------------------------------- #
# vector-cache loading (reuse knn_store as-is; merge_judge/ for mid-8)
# --------------------------------------------------------------------------- #
def load_doc_vectors(vec_path, docs):
    """Per-doc float32 (n,h) word vectors aligned to the token grid.
    Accepts either the flat knn_store datastore format (keys/doc_idx/word_pos)
    or the per-doc 'v::<doc_id>' format (knn dev queries / merge_judge encode)."""
    z = np.load(vec_path)
    out = {}
    if "keys" in z.files:                                # flat datastore
        keys = z["keys"]
        doc_idx = z["doc_idx"]
        word_pos = z["word_pos"]
        doc_ids = [str(s) for s in z["doc_ids"]]
        by_id = {d["doc_id"]: d for d in docs}
        for di, doc_id in enumerate(doc_ids):
            m = doc_idx == di
            d = by_id[doc_id]
            n = len(d["tokens"])
            assert m.sum() == n and (word_pos[m] == np.arange(n)).all(), \
                f"datastore grid mismatch for {doc_id} (PAR-token doc?)"
            out[doc_id] = keys[m].astype(np.float32)
    else:                                                # per-doc v:: format
        for d in docs:
            out[d["doc_id"]] = z["v::" + d["doc_id"]].astype(np.float32)
    for d in docs:
        assert d["doc_id"] in out, f"no vectors for {d['doc_id']}"
        assert len(out[d["doc_id"]]) == len(d["tokens"]), \
            f"vector/token length mismatch for {d['doc_id']}"
    return out


# --------------------------------------------------------------------------- #
# encode: frozen-encoder CPU pass at an arbitrary hidden layer
# (knn_boundary.encode_doc's exact window walk; that function hardcodes the
#  final layer, so the walk is reproduced here verbatim with `layer` swapped)
# --------------------------------------------------------------------------- #
def encode_doc_layer(words, tokenizer, model, layer,
                     window=180, overlap=60, max_length=512):
    import torch
    n = len(words)
    hidden = model.config.hidden_size
    vec_sum = np.zeros((n, hidden), dtype=np.float64)
    vec_cnt = np.zeros(n, dtype=np.int64)
    probs = defaultdict(list)
    start = 0
    with torch.no_grad():
        while start < n:
            chunk = words[start:start + window]
            enc = tokenizer(chunk, is_split_into_words=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            word_ids = enc.word_ids(0)
            out = model(**enc, output_hidden_states=True)
            logits = out.logits[0]
            h = out.hidden_states[layer][0]            # <- the only change
            p_boundary = torch.softmax(logits, dim=-1)[:, 1].numpy()
            h_np = h.numpy()
            last_seen = -1
            for pos, wid in enumerate(word_ids):
                if wid is None:
                    continue
                nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
                if nxt != wid:
                    probs[start + wid].append(float(p_boundary[pos]))
                    vec_sum[start + wid] += h_np[pos]
                    vec_cnt[start + wid] += 1
                last_seen = max(last_seen, wid)
            covered_through = start + last_seen
            if covered_through >= n - 1:
                break
            start = max(covered_through + 1 - overlap, start + 1)
    vecs = np.zeros((n, hidden), dtype=np.float32)
    got = vec_cnt > 0
    vecs[got] = (vec_sum[got] / vec_cnt[got, None]).astype(np.float32)
    p = np.array([np.mean(probs[i]) if probs[i] else 0.0 for i in range(n)],
                 dtype=np.float32)
    return vecs, p


def cmd_encode(args):
    assert_cpu_only()
    tokenizer, model = load_encoder(args.model)
    docs = load_jsonl(args.jsonl)
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    print(f"# encoding {len(docs)} docs from {args.jsonl} at hidden layer "
          f"{args.layer}", flush=True)
    out = {}
    t0 = time.time()
    for di, d in enumerate(docs):
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                 for w in d["tokens"]]
        vecs, p = encode_doc_layer(words, tokenizer, model, args.layer)
        out["v::" + d["doc_id"]] = vecs.astype(np.float16)
        out["p::" + d["doc_id"]] = p                     # alignment guard
        if (di + 1) % 10 == 0 or di == len(docs) - 1:
            el = time.time() - t0
            print(f"  [{di+1}/{len(docs)}] {el:.0f}s elapsed, "
                  f"eta {el/(di+1)*(len(docs)-di-1):.0f}s", flush=True)
    out["meta"] = np.array(json.dumps(
        {"model": args.model, "jsonl": args.jsonl, "layer": args.layer}))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"vectors for {len(docs)} docs (layer {args.layer}) -> {args.out}",
          flush=True)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def assemble_pairs(docs, rng, n_random_states=3):
    """Curriculum pair table over all docs.
    Returns (doc_key list, int64 table [doc_i, a0, c, b1, y], per-state counts)."""
    rows = []
    counts = defaultdict(lambda: [0, 0])                 # state -> [merge, keep]
    for di, d in enumerate(docs):
        lab = np.asarray(d["labels"], dtype=np.int64)
        for name, cuts in curriculum_states(lab, rng, n_random_states):
            prs = pairs_from_cuts(cuts, len(lab))
            if len(prs) == 0:
                continue
            y = 1 - lab[prs[:, 1]]                       # MERGE=1 iff no gold cut
            counts[name][0] += int(y.sum())
            counts[name][1] += int((1 - y).sum())
            rows.append(np.concatenate(
                [np.full((len(prs), 1), di, dtype=np.int64), prs,
                 y[:, None]], axis=1))
    return np.concatenate(rows, axis=0), dict(counts)


def cmd_train(args):
    assert_cpu_only()
    import torch
    import torch.nn as nn

    docs = load_jsonl(args.train_jsonl)
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    vecs = load_doc_vectors(args.train_vecs, docs)
    rng = np.random.default_rng(args.seed)

    table, counts = assemble_pairs(docs, rng, args.n_random_states)
    n1 = int(table[:, 4].sum())
    n0 = len(table) - n1
    w_keep = n1 / max(n0, 1)
    print("# CURRICULUM (documented sampling; states refine gold):")
    for name in ["allcut"] + [f"rand{r}" for r in range(args.n_random_states)] \
            + ["gold", "half"]:
        m, k = counts.get(name, (0, 0))
        print(f"    state {name:7s}: merge={m:6d} keep={k:6d}")
    print(f"    TOTAL: merge={n1} keep={n0} -> keep weight {w_keep:.3f} "
          f"(classes balanced in the loss)", flush=True)

    # global arrays: one vectorized feature call per batch, no per-doc loops
    keys = [d["doc_id"] for d in docs]
    lens = np.array([len(vecs[k]) for k in keys], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])        # V row offsets
    poff = off + np.arange(len(keys))                        # prefix row offsets
    Vg = np.concatenate([vecs[k] for k in keys])
    Pg = np.concatenate([prefix_sum(vecs[k]) for k in keys])
    in_dim = 4 * Vg.shape[1] + N_SCALARS

    di, a0, c, b1 = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
    y_all = table[:, 4].astype(np.float32)
    IDX = np.stack([off[di] + c, off[di] + c + 1,            # iva, ivb
                    poff[di] + a0, poff[di] + c + 1,         # ipa, ipc
                    poff[di] + b1 + 1,                       # ipb
                    c - a0 + 1, b1 - c], axis=1)             # lenA, lenB

    def batch_feats(sel):
        cols = IDX[sel]
        return _features_idx(Vg, Pg, cols[:, 0], cols[:, 1], cols[:, 2],
                             cols[:, 3], cols[:, 4], cols[:, 5],
                             cols[:, 6]), y_all[sel]

    # standardization over the full curriculum (chunked, no materialization)
    t0 = time.time()
    tot = np.zeros(in_dim, dtype=np.float64)
    tot2 = np.zeros(in_dim, dtype=np.float64)
    seen = 0
    for s in range(0, len(table), 8192):
        X, _ = batch_feats(np.arange(s, min(s + 8192, len(table))))
        tot += X.sum(axis=0, dtype=np.float64)
        tot2 += (X.astype(np.float64) ** 2).sum(axis=0)
        seen += len(X)
    mu = (tot / seen).astype(np.float32)
    sd = np.sqrt(np.maximum(tot2 / seen - (tot / seen) ** 2, 1e-8)) \
        .astype(np.float32)
    print(f"# scaler over {seen} pairs in {time.time()-t0:.0f}s", flush=True)

    torch.manual_seed(args.seed)
    head = make_head(in_dim, seed=args.seed)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    n_par = sum(p.numel() for p in head.parameters())
    print(f"# judge head: in_dim={in_dim}, {n_par} params, "
          f"{args.epochs} epochs, batch {args.batch}, lr {args.lr}", flush=True)

    for ep in range(args.epochs):
        perm = rng.permutation(len(table))
        ep_loss = ep_n = 0.0
        acc_m = acc_k = seen_m = seen_k = 0
        t0 = time.time()
        head.train()
        for s in range(0, len(perm), args.batch):
            X, y = batch_feats(perm[s:s + args.batch])
            Xt = torch.from_numpy((X - mu) / sd)
            yt = torch.from_numpy(y)
            z = head(Xt)[:, 0]
            w = torch.where(yt > 0.5, torch.ones_like(yt),
                            torch.full_like(yt, w_keep))
            loss = (lossf(z, yt) * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach()) * len(y)
            ep_n += len(y)
            pred = (torch.sigmoid(z) > 0.5).numpy()
            acc_m += int((pred & (y > 0.5)).sum())
            acc_k += int((~pred & (y < 0.5)).sum())
            seen_m += int((y > 0.5).sum())
            seen_k += int((y < 0.5).sum())
        bal = 0.5 * (acc_m / max(seen_m, 1) + acc_k / max(seen_k, 1))
        print(f"  epoch {ep+1}/{args.epochs}: loss={ep_loss/ep_n:.4f} "
              f"merge-acc={acc_m/max(seen_m,1):.4f} "
              f"keep-acc={acc_k/max(seen_k,1):.4f} bal-acc={bal:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    head.eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "mu": mu, "sd": sd,
                "in_dim": in_dim, "layer": args.layer,
                "config": {"seed": args.seed, "epochs": args.epochs,
                           "batch": args.batch, "lr": args.lr,
                           "n_random_states": args.n_random_states,
                           "train_jsonl": args.train_jsonl,
                           "train_vecs": args.train_vecs,
                           "w_keep": w_keep,
                           "curriculum_counts": counts}}, args.out)
    print(f"judge head -> {args.out}", flush=True)


# --------------------------------------------------------------------------- #
# inference policies (pure given a judge callable; unit-tested with mocks)
# --------------------------------------------------------------------------- #
def eraser_merge_probs(V, P, pred, judge):
    """(cuts, merge_probs) for the tagger's cuts under the tagger's own
    segmentation (one-shot judging; see docstring 3a)."""
    n = len(pred)
    cuts = np.flatnonzero(pred[:n - 1])
    pairs = pairs_from_cuts(cuts, n)
    return cuts, judge(build_features(V, P, pairs))


def agglomerative_cuts(V, P, judge, tau):
    """Start all-cut; repeatedly erase the highest-P(merge) cut until no pair
    exceeds tau. Returns the surviving cut positions (sorted int64)."""
    n = len(V)
    if n <= 1:
        return np.zeros(0, dtype=np.int64)
    left = np.arange(n - 1, dtype=np.int64)        # seg start left of cut c
    right = np.arange(1, n, dtype=np.int64)        # seg end right of cut c
    alive = np.ones(n - 1, dtype=bool)
    ver = np.zeros(n - 1, dtype=np.int64)
    init = judge(build_features(
        V, P, pairs_from_cuts(np.arange(n - 1), n)))
    heap = [(-float(p), int(c), 0) for c, p in enumerate(init) if p > tau]
    heapq.heapify(heap)
    while heap:
        negp, c, v = heapq.heappop(heap)
        if not alive[c] or v != ver[c] or -negp <= tau:
            continue
        alive[c] = False                            # erase the cut = merge
        lo, hi = left[c], right[c]
        cl, cr = lo - 1, hi                         # neighbour cuts, if any
        upd = []
        if cl >= 0 and alive[cl]:
            right[cl] = hi
            upd.append(cl)
        if cr <= n - 2 and alive[cr]:
            left[cr] = lo
            upd.append(cr)
        if upd:
            prs = np.stack([[left[u], u, right[u]] for u in upd]).astype(np.int64)
            ps = judge(build_features(V, P, prs))
            for u, p in zip(upd, ps):
                ver[u] += 1
                if p > tau:
                    heapq.heappush(heap, (-float(p), int(u), int(ver[u])))
    return np.flatnonzero(alive).astype(np.int64)


# --------------------------------------------------------------------------- #
# scoring helpers (repo conventions: doc-macro F1, unet2d phi/Jaccard)
# --------------------------------------------------------------------------- #
def pooled_prf(pred, gold):
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    return {"P": tp / max(tp + fp, 1), "R": tp / max(tp + fn, 1),
            "F1": 2 * tp / max(2 * tp + fp + fn, 1),
            "TP": tp, "FP": fp, "FN": fn}


def decorrelation(pred_a, pred_t, gold):
    err_a = pred_a != gold
    err_t = pred_t != gold
    fp_a = pred_a & ~gold
    fp_t = pred_t & ~gold
    phi = float(np.corrcoef(err_a.astype(float), err_t.astype(float))[0, 1]) \
        if err_a.any() and err_t.any() else float("nan")
    inter = int((fp_a & fp_t).sum())
    union = int((fp_a | fp_t).sum())
    g0 = ~gold
    phi_fp = float(np.corrcoef(fp_a[g0].astype(float),
                               fp_t[g0].astype(float))[0, 1]) \
        if fp_a[g0].any() and fp_t[g0].any() else float("nan")
    return {"phi_all_errors": phi, "fp_jaccard": inter / max(union, 1),
            "phi_fp_given_gold0": phi_fp, "n_fp_a": int(fp_a.sum()),
            "n_fp_tagger": int(fp_t.sum()), "n_fp_shared": inter}


def cmd_eval(args):
    assert_cpu_only()
    docs = load_jsonl(args.dev_jsonl)
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    vecs = load_doc_vectors(args.dev_vecs, docs)
    judge, ck = Judge.load(args.head)
    assert ck["layer"] == args.layer, \
        f"head was trained on layer {ck['layer']}, eval asked for {args.layer}"
    pz = np.load(args.probs)
    n_docs = len(docs)
    report = {"layer": args.layer, "head": args.head, "probs": args.probs,
              "dev_jsonl": args.dev_jsonl, "n_docs": n_docs,
              "tau_headline": TAU_HEADLINE}

    # pooled arrays (repo grid: PAR-filtered — NoPnx tracks contain no PAR)
    golds, preds_t, pconf, doc_of, doc_key, genres = [], [], [], [], [], []
    for di, d in enumerate(docs):
        keep = np.array([w != PARAGRAPH_TOKEN for w in d["tokens"]])
        assert keep.all(), "unexpected PAR token in a NoPnx track"
        g = np.asarray(d["labels"], dtype=np.int64)
        p = pz[d["doc_id"]].astype(np.float64)
        assert len(g) == len(p) == len(vecs[d["doc_id"]])
        golds.append(g)
        preds_t.append(p >= 0.5)
        pconf.append(p)
        doc_of.append(np.full(len(g), di, dtype=np.int64))
        doc_key.append(d["doc_id"])
        genres.append(bucket(d["tokens"]))
    gold = np.concatenate(golds).astype(bool)
    pred_t = np.concatenate(preds_t)
    p_tag = np.concatenate(pconf)
    doc_of = np.concatenate(doc_of)

    base_f1 = 100 * macro_f1(pred_t.astype(np.float64), gold.astype(np.int8),
                             doc_of, n_docs, 0.5)
    conf_fp = (p_tag >= CONF) & ~gold
    conf_tp = (p_tag >= CONF) & gold
    print(f"# BASELINE tagger@0.5: macro-F1={base_f1:.2f} "
          f"(record: {BASELINE_F1_REF}); confident-FP n={int(conf_fp.sum())} "
          f"(record: 2236), confident-TP n={int(conf_tp.sum())} "
          f"(record: 10985)", flush=True)
    report["baseline"] = {"macro_f1": base_f1,
                          "n_conf_fp": int(conf_fp.sum()),
                          "n_conf_tp": int(conf_tp.sum()),
                          "pooled": pooled_prf(pred_t, gold)}

    # ---------------- (a) ERASER-ON-TAGGER ---------------------------------
    P = {k: prefix_sum(vecs[k]) for k in vecs}
    t0 = time.time()
    mp = np.full(len(gold), -1.0)          # merge prob of the cut at position i
    off = 0
    for di, d in enumerate(docs):
        k = d["doc_id"]
        n = len(d["labels"])
        cuts, probs_m = eraser_merge_probs(vecs[k], P[k], preds_t[di], judge)
        mp[off + cuts] = probs_m
        off += n
    print(f"# eraser: judged {int((mp >= 0).sum())} tagger cuts in "
          f"{time.time()-t0:.0f}s", flush=True)

    def erased_pred(tau):
        out = pred_t.copy()
        out[mp > tau] = False
        return out

    print("\n=========== (a) ERASER-ON-TAGGER (dev, %d docs) ===========" % n_docs)
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
        print(f"{tau:6.2f} {f1:8.2f} {f1-base_f1:+7.2f} {int(e.sum()):7d} "
              f"{ce_fp:10d} {ce_tp:10d} {ratio:6.2f}{tag}")
        rows_a[round(tau, 2)] = {
            "macro_f1": f1, "delta": f1 - base_f1, "n_erased": int(e.sum()),
            "erased_fp_all": int((e & ~gold).sum()),
            "erased_tp_all": int((e & gold).sum()),
            "erased_conf_fp": ce_fp, "erased_conf_tp": ce_tp,
            "pooled": pooled_prf(erased_pred(tau), gold)}
    report["eraser_taus"] = rows_a

    pe_h = erased_pred(TAU_HEADLINE)
    hh = rows_a[TAU_HEADLINE]
    bp = report["baseline"]["pooled"]
    hp = hh["pooled"]
    print(f"\nHEADLINE tau=0.5 FAILURE ANATOMY (standing order):")
    print(f"  macro-F1 {base_f1:.2f} -> {hh['macro_f1']:.2f} "
          f"(delta {hh['delta']:+.2f} vs {BASELINE_F1_REF})")
    print(f"  erased {hh['n_erased']} cuts: {hh['erased_fp_all']} were FPs "
          f"(good), {hh['erased_tp_all']} were TPs (bad)")
    print(f"  of the {int(conf_fp.sum())} confident FPs (p>=0.8, gold=0): "
          f"{hh['erased_conf_fp']} erased")
    print(f"  of the {int(conf_tp.sum())} confident TPs (p>=0.8, gold=1): "
          f"{hh['erased_conf_tp']} wrongly erased "
          f"(ratio cFP/cTP = {hh['erased_conf_fp']/max(hh['erased_conf_tp'],1):.2f}, "
          f"gate needs > 3)")
    print(f"  pooled P/R decomposition: P {bp['P']:.4f} -> {hp['P']:.4f}, "
          f"R {bp['R']:.4f} -> {hp['R']:.4f} "
          f"(TP {bp['TP']}->{hp['TP']}, FP {bp['FP']}->{hp['FP']}, "
          f"FN {bp['FN']}->{hp['FN']})")

    # per-genre deltas at headline tau
    print("\n  per-genre (mean per-doc F1, before -> after; FP/FN pooled):")
    print(f"  {'genre':20s} {'docs':>5} {'F1 before':>10} {'F1 after':>9} "
          f"{'delta':>7} {'dFP':>6} {'dFN':>6}")
    genre_rows = {}
    for gname in ["general prose", "legal/constitution", "hadith/isnad",
                  "genealogy", "cloze/exercise"]:
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
              f"{fa-fb:+7.2f} {dfp:+6d} {dfn:+6d}")
        genre_rows[gname] = {"n_docs": len(idx), "f1_before": fb,
                             "f1_after": fa, "delta": fa - fb,
                             "d_fp": dfp, "d_fn": dfn}
    report["eraser_headline_genres"] = genre_rows

    # ---------------- (b) SOLO AGGLOMERATIVE -------------------------------
    solo_taus = [float(t) for t in args.solo_taus.split(",")] \
        if args.solo_taus else [0.3, 0.5, 0.7]
    report["solo"] = {}
    pred_solo_h = None
    if not args.skip_solo:
        print("\n=========== (b) SOLO AGGLOMERATIVE (all-cut start) ===========")
        for tau in solo_taus:
            t0 = time.time()
            chunks = []
            for d in docs:
                k = d["doc_id"]
                n = len(d["labels"])
                cuts = agglomerative_cuts(vecs[k], P[k], judge, tau)
                pr = np.zeros(n, dtype=bool)
                pr[cuts] = True
                pr[n - 1] = True                      # doc end (gold convention)
                chunks.append(pr)
            ps = np.concatenate(chunks)
            f1 = 100 * macro_f1(ps.astype(np.float64), gold.astype(np.int8),
                                doc_of, n_docs, 0.5)
            pl = pooled_prf(ps, gold)
            tag = "  <-- HEADLINE (untuned)" if tau == TAU_HEADLINE else ""
            print(f"  tau={tau:.2f}: macro-F1={f1:.2f} "
                  f"(delta {f1-base_f1:+.2f} vs tagger@0.5)  "
                  f"P={pl['P']:.4f} R={pl['R']:.4f} "
                  f"pred-cuts={int(ps.sum())} ({time.time()-t0:.0f}s){tag}",
                  flush=True)
            report["solo"][round(tau, 2)] = {"macro_f1": f1,
                                             "delta": f1 - base_f1,
                                             "pooled": pl}
            if abs(tau - TAU_HEADLINE) < 1e-9:
                pred_solo_h = ps

    # ---------------- (c) DECORRELATION ------------------------------------
    print("\n=========== (c) DECORRELATION vs tagger errors ===========")
    dec_a = decorrelation(pe_h, pred_t, gold)
    print(f"  (a) eraser@0.5 : phi(all errors)={dec_a['phi_all_errors']:.4f}  "
          f"FP-Jaccard={dec_a['fp_jaccard']:.4f}  "
          f"phi(FP|gold=0)={dec_a['phi_fp_given_gold0']:.4f}")
    print("      (by construction the eraser only removes tagger cuts, so "
          "high correlation is expected here)")
    report["decorrelation_eraser"] = dec_a
    if pred_solo_h is not None:
        dec_b = decorrelation(pred_solo_h, pred_t, gold)
        print(f"  (b) solo@0.5   : phi(all errors)={dec_b['phi_all_errors']:.4f}  "
              f"FP-Jaccard={dec_b['fp_jaccard']:.4f}  "
              f"phi(FP|gold=0)={dec_b['phi_fp_given_gold0']:.4f}")
        report["decorrelation_solo"] = dec_b
    print("  reference points: existing-pool inter-voter mean phi = 0.74; "
          "S-only U-Net = 0.53")

    # ---------------- PRE-REGISTERED GATES ---------------------------------
    print("\n=========== PRE-REGISTERED GATES (verbatim) ===========")
    gate_a = (hh["delta"] > 0.15 and
              hh["erased_conf_fp"] > 3 * hh["erased_conf_tp"])
    print(f"  [{'x' if gate_a else ' '}] eraser adopt-candidate iff net F1 > "
          f"+0.15 with conf-FP erasures > 3x conf-TP erasures  "
          f"(measured: {hh['delta']:+.2f}; "
          f"{hh['erased_conf_fp']} vs {hh['erased_conf_tp']})")
    phi_b = report.get("decorrelation_solo", {}).get("phi_all_errors",
                                                     float("nan"))
    gate_b = bool(phi_b == phi_b and phi_b < 0.6)
    print(f"  [{'x' if gate_b else ' '}] solo interesting iff phi < 0.6  "
          f"(measured: phi={phi_b:.4f})" if phi_b == phi_b else
          "  [ ] solo interesting iff phi < 0.6  (solo not run)")
    if not gate_a and not gate_b:
        print("  [x] else PARK with the numbers.")
    report["gates"] = {"eraser_adopt_candidate": bool(gate_a),
                       "solo_interesting": gate_b,
                       "verdict": ("ADOPT-CANDIDATE (eraser)" if gate_a else
                                   ("INTERESTING (solo)" if gate_b else
                                    "PARK with the numbers"))}
    print(f"  VERDICT [{args.layer}]: {report['gates']['verdict']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nreport -> {args.out}", flush=True)


# --------------------------------------------------------------------------- #
# self-test (corpus-independent; no data/, no encoder, no caches)
# --------------------------------------------------------------------------- #
def _synth_doc(rng, n_sent=6, max_len=7, h=16, sep=4.0):
    """Sentences = clusters around distinct topic vectors (separable)."""
    words, labels, V = [], [], []
    for s in range(n_sent):
        L = int(rng.integers(2, max_len + 1))
        topic = rng.normal(0, 1, h) * sep
        for i in range(L):
            V.append(topic + rng.normal(0, 0.3, h))
            labels.append(1 if i == L - 1 else 0)
            words.append(f"w{s}_{i}")
    return words, np.array(labels), np.array(V, dtype=np.float32)


def self_test():
    rng = np.random.default_rng(0)

    # 1. pair supervision at the all-cut state
    _, lab, V = _synth_doc(rng)
    n = len(lab)
    cuts = np.arange(n - 1)
    prs = pairs_from_cuts(cuts, n)
    assert len(prs) == n - 1
    assert (prs[:, 0] == prs[:, 1]).all() and (prs[:, 2] == prs[:, 1] + 1).all()
    g = gold_cut_positions(lab)
    assert lab[-1] == 1 and len(g) == lab.sum() - 1

    # 2. curriculum: every state refines gold; labels well-defined
    for name, c in curriculum_states(lab, rng):
        assert set(g.tolist()) <= set(c.tolist()), f"state {name} loses gold"
        if name == "gold":
            assert set(c.tolist()) == set(g.tolist())
    print("PASS pair supervision + curriculum invariants")

    # 3. features: pooled means match a naive loop
    P = prefix_sum(V)
    prs = pairs_from_cuts(g, n)
    X = build_features(V, P, prs)
    assert X.shape == (len(prs), 4 * V.shape[1] + N_SCALARS)
    a0, c, b1 = prs[0]
    np.testing.assert_allclose(X[0, :V.shape[1]],
                               V[a0:c + 1].mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(X[0, V.shape[1]:2 * V.shape[1]],
                               V[c + 1:b1 + 1].mean(axis=0), rtol=1e-5)
    print("PASS feature construction (pooled means, endpoints, scalars)")

    # 4. a trained judge separates synthetic merge/keep and the agglomerative
    #    policy recovers the planted segmentation
    import torch
    import torch.nn as nn
    docs = [_synth_doc(rng) for _ in range(30)]
    feats, ys = [], []
    for _, lab_d, V_d in docs:
        P_d = prefix_sum(V_d)
        for _, cset in curriculum_states(lab_d, rng, 2):
            prs_d = pairs_from_cuts(cset, len(lab_d))
            feats.append(build_features(V_d, P_d, prs_d))
            ys.append(1 - lab_d[prs_d[:, 1]])
    X = np.concatenate(feats)
    y = np.concatenate(ys).astype(np.float32)
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-6)
    head = make_head(X.shape[1], seed=0)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy((X - mu) / sd)
    yt = torch.from_numpy(y)
    for _ in range(200):
        idx = torch.from_numpy(
            np.random.default_rng(_).integers(0, len(y), 256))
        loss = lf(head(Xt[idx])[:, 0], yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    judge = Judge(head, mu.astype(np.float32), sd.astype(np.float32))
    p = judge(X)
    bal = 0.5 * (((p > 0.5) & (y > 0.5)).sum() / max((y > 0.5).sum(), 1)
                 + ((p <= 0.5) & (y < 0.5)).sum() / max((y < 0.5).sum(), 1))
    assert bal > 0.9, f"synthetic judge balanced-acc too low: {bal:.3f}"
    _, lab_t, V_t = _synth_doc(np.random.default_rng(99))
    got = agglomerative_cuts(V_t, prefix_sum(V_t), judge, 0.5)
    want = gold_cut_positions(lab_t)
    assert set(got.tolist()) == set(want.tolist()), \
        f"agglomerative failed: got {got}, want {want}"
    print(f"PASS judge training (bal-acc={bal:.3f}) + agglomerative recovery")

    # 5. eraser bookkeeping with a mock judge (erase iff left seg len >= 2)
    lab_e = np.array([0, 1, 0, 0, 1, 0, 1])
    Ve = np.ones((7, 4), dtype=np.float32)
    Pe = prefix_sum(Ve)
    pred = np.array([1, 1, 0, 1, 1, 0, 1], dtype=bool)

    def mock(Xf):
        lenA = np.expm1(Xf[:, -N_SCALARS])
        return (lenA >= 2).astype(np.float32)

    cuts_e, mps = eraser_merge_probs(Ve, Pe, pred, mock)
    assert cuts_e.tolist() == [0, 1, 3, 4]
    assert (mps > 0.5).tolist() == [False, False, True, False]
    print("PASS eraser one-shot bookkeeping (mock judge)")
    print("SELF-TEST OK")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test")

    e = sub.add_parser("encode")
    e.add_argument("--model", required=True)
    e.add_argument("--jsonl", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--layer", type=int, default=8)
    e.add_argument("--limit-docs", type=int, default=0)

    t = sub.add_parser("train")
    t.add_argument("--layer", choices=["final", "mid8"], required=True)
    t.add_argument("--train-jsonl", required=True)
    t.add_argument("--train-vecs", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--n-random-states", type=int, default=3)
    t.add_argument("--limit-docs", type=int, default=0)

    v = sub.add_parser("eval")
    v.add_argument("--layer", choices=["final", "mid8"], required=True)
    v.add_argument("--head", required=True)
    v.add_argument("--dev-jsonl", required=True)
    v.add_argument("--dev-vecs", required=True)
    v.add_argument("--probs", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--solo-taus", default="")
    v.add_argument("--skip-solo", action="store_true")
    v.add_argument("--limit-docs", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "self-test":
        self_test()
    elif args.cmd == "encode":
        cmd_encode(args)
    elif args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
