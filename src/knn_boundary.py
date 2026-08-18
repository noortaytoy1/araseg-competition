"""kNN-BOUNDARY VOTER for AraSeg NoPnx-NP (closed track, CPU-only, additive).

Non-parametric memory on top of the frozen union-NoPnx-NP-arabertv02-s42
encoder: store every TRAIN word's final-layer hidden vector (last-subword
position, the exact grid predict.py labels on) with its 0/1 boundary label;
at dev time, retrieve k nearest train words and let them vote:

    p_kNN(i)  = sum_j softmax(-d_ij^2 / T) * y_j        (j over k neighbors)
    p(i)      = lam * p_kNN(i) + (1 - lam) * p_param(i)

p_param comes from the CACHED parametric probs (probs/*.npz) — nothing is
retrained, dev is used for EVALUATION only.

Stages (run in order; artifacts under knn_store/, never under runs/ or probs/):

  CUDA_VISIBLE_DEVICES=-1 python src/knn_boundary.py build-datastore \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --jsonl data/NoPnx-NP_train.jsonl --out knn_store/NoPnx-NP_s42_train.npz

  CUDA_VISIBLE_DEVICES=-1 python src/knn_boundary.py encode-queries \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --jsonl data/NoPnx-NP_dev.jsonl --out knn_store/NoPnx-NP_s42_dev_q.npz

  CUDA_VISIBLE_DEVICES=-1 python src/knn_boundary.py eval \
      --datastore knn_store/NoPnx-NP_s42_train.npz \
      --queries   knn_store/NoPnx-NP_s42_dev_q.npz \
      --param-probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --jsonl data/NoPnx-NP_dev.jsonl

Windowing mirrors predict.predict_doc verbatim (window=180, overlap=60,
max_length=512); vectors of words covered by several windows are averaged,
exactly like the probabilities are. The recomputed parametric probs are saved
alongside the query vectors and checked against the cached npz at eval time
(grid-alignment guard).

Honesty: lam / k / T are decode-time hyperparameters like the threshold;
tuned numbers carry a dev-selection caveat. The pre-declared honest headline
is lam=0.2, k=16, T=T0 (T0 estimated from the datastore itself, no dev
tuning), no whitening, threshold 0.5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from genre_buckets import bucket

# same set as diag_preposition.py (verbatim)
PREPOSITIONS = ["و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
                "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى"]
PREP_SET = set(PREPOSITIONS)

THRESH_GRID = [round(t, 2) for t in np.arange(0.30, 0.7001, 0.05)]


# --------------------------------------------------------------------------- #
# encoding (CPU) — predict.predict_doc's exact window walk, but also grabbing
# the final-layer hidden state at each word's last subword position
# --------------------------------------------------------------------------- #
def encode_doc(words, tokenizer, model, window=180, overlap=60, max_length=512):
    """Return (vecs [n,768] float32 mean-pooled over covering windows,
    probs [n] float32 mean over covering windows) on the last-subword grid."""
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
            h_last = out.hidden_states[-1][0]          # (seq, hidden) final layer
            p_boundary = torch.softmax(logits, dim=-1)[:, 1].numpy()
            h_np = h_last.numpy()

            last_seen = -1
            for pos, wid in enumerate(word_ids):
                if wid is None:
                    continue
                nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
                if nxt != wid:                          # last subword of this word
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


def load_encoder(model_dir):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the battery owns the GPU)"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    return tokenizer, model


def cmd_build_datastore(args):
    tokenizer, model = load_encoder(args.model)
    docs = load_jsonl(args.jsonl)
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    print(f"# datastore over {len(docs)} docs from {args.jsonl}", flush=True)
    keys, labels, doc_idx, word_pos = [], [], [], []
    t0 = time.time()
    for di, d in enumerate(docs):
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        vecs, _ = encode_doc(words, tokenizer, model)
        keep = [i for i, w in enumerate(d["tokens"]) if w != PARAGRAPH_TOKEN]
        keys.append(vecs[keep].astype(np.float16))
        labels.append(np.asarray(d["labels"], dtype=np.uint8)[keep])
        doc_idx.append(np.full(len(keep), di, dtype=np.int32))
        word_pos.append(np.asarray(keep, dtype=np.int32))
        if (di + 1) % 10 == 0 or di == len(docs) - 1:
            el = time.time() - t0
            print(f"  [{di+1}/{len(docs)}] {el:.0f}s elapsed, "
                  f"eta {el/(di+1)*(len(docs)-di-1):.0f}s", flush=True)
    K = np.concatenate(keys)
    y = np.concatenate(labels)
    np.savez_compressed(args.out, keys=K, labels=y,
                        doc_idx=np.concatenate(doc_idx),
                        word_pos=np.concatenate(word_pos),
                        doc_ids=np.array([d["doc_id"] for d in docs]))
    print(f"datastore: {K.shape[0]} vectors x {K.shape[1]} (float16), "
          f"boundary rate {y.mean():.4f} -> {args.out}", flush=True)


def cmd_encode_queries(args):
    tokenizer, model = load_encoder(args.model)
    docs = load_jsonl(args.jsonl)
    if args.limit_docs:
        docs = docs[:args.limit_docs]
    print(f"# encoding {len(docs)} query docs from {args.jsonl}", flush=True)
    out = {}
    t0 = time.time()
    for di, d in enumerate(docs):
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        vecs, p = encode_doc(words, tokenizer, model)
        out["v::" + d["doc_id"]] = vecs.astype(np.float16)
        out["p::" + d["doc_id"]] = p          # recomputed param probs (guard)
        if (di + 1) % 10 == 0 or di == len(docs) - 1:
            el = time.time() - t0
            print(f"  [{di+1}/{len(docs)}] {el:.0f}s elapsed, "
                  f"eta {el/(di+1)*(len(docs)-di-1):.0f}s", flush=True)
    np.savez_compressed(args.out, **out)
    print(f"queries for {len(docs)} docs -> {args.out}", flush=True)


# --------------------------------------------------------------------------- #
# retrieval + scoring (pure numpy)
# --------------------------------------------------------------------------- #
def zca(K):
    """Whitening transform (mean, W) fit on datastore keys K (float32)."""
    mu = K.mean(axis=0)
    C = np.cov((K - mu).T)
    ev, U = np.linalg.eigh(C)
    ev = np.clip(ev, 1e-6, None)
    W = (U / np.sqrt(ev)) @ U.T
    return mu.astype(np.float32), W.astype(np.float32)


def topk_neighbors(Q, K, kmax, chunk=2048, tag=""):
    """Sorted top-kmax squared-L2 distances and indices of K for each row of Q."""
    Kn2 = (K.astype(np.float32) ** 2).sum(axis=1)
    D_out = np.empty((Q.shape[0], kmax), dtype=np.float32)
    I_out = np.empty((Q.shape[0], kmax), dtype=np.int64)
    t0 = time.time()
    for s in range(0, Q.shape[0], chunk):
        q = Q[s:s + chunk].astype(np.float32)
        d2 = q @ (-2.0 * K.T.astype(np.float32))
        d2 += Kn2[None, :]
        d2 += (q ** 2).sum(axis=1)[:, None]
        part = np.argpartition(d2, kmax - 1, axis=1)[:, :kmax]
        pd = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(pd, axis=1)
        I_out[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        D_out[s:s + chunk] = np.clip(np.take_along_axis(pd, order, axis=1), 0, None)
        done = min(s + chunk, Q.shape[0])
        el = time.time() - t0
        print(f"  kNN{tag} [{done}/{Q.shape[0]}] {el:.0f}s "
              f"eta {el/done*(Q.shape[0]-done):.0f}s", flush=True)
    return D_out, I_out


def knn_probs(D, nbr_labels, k, T):
    """softmax(-d^2/T) label vote over the k nearest of the stored neighbors."""
    d = D[:, :k] / T
    w = np.exp(-(d - d.min(axis=1, keepdims=True)))
    w /= w.sum(axis=1, keepdims=True)
    return (w * nbr_labels[:, :k]).sum(axis=1)


def macro_f1(p, gold, doc_of, n_docs, thr):
    pred = p >= thr
    g = gold.astype(bool)
    tp = np.bincount(doc_of[pred & g], minlength=n_docs)
    fp = np.bincount(doc_of[pred & ~g], minlength=n_docs)
    fn = np.bincount(doc_of[~pred & g], minlength=n_docs)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def cmd_eval(args):
    docs = load_jsonl(args.jsonl)
    qz = np.load(args.queries)
    have = {k[3:] for k in qz.files if k.startswith("v::")}
    docs = [d for d in docs if d["doc_id"] in have]
    print(f"# eval on {len(docs)} docs (of {len(have)} encoded)", flush=True)

    pz = np.load(args.param_probs)
    Q_list, pparam_list, gold_list, doc_of_list = [], [], [], []
    max_dp = 0.0
    for di, d in enumerate(docs):
        v = qz["v::" + d["doc_id"]]
        p_re = qz["p::" + d["doc_id"]]
        p_c = pz[d["doc_id"]].astype(np.float32)
        assert len(v) == len(d["tokens"]) == len(p_c), f"align fail {d['doc_id']}"
        max_dp = max(max_dp, float(np.abs(p_re - p_c).max()))
        keep = np.array([w != PARAGRAPH_TOKEN for w in d["tokens"]])
        Q_list.append(v[keep])
        pparam_list.append(p_c[keep])
        gold_list.append(np.asarray(d["labels"])[keep])
        doc_of_list.append(np.full(int(keep.sum()), di, dtype=np.int64))
    print(f"grid-alignment guard: max |p_recomputed - p_cached| = {max_dp:.4f} "
          f"(same grid if small)", flush=True)

    Q = np.concatenate(Q_list).astype(np.float32)
    p_param = np.concatenate(pparam_list).astype(np.float64)
    gold = np.concatenate(gold_list).astype(np.int8)
    doc_of = np.concatenate(doc_of_list)
    n_docs = len(docs)
    print(f"{Q.shape[0]} query positions", flush=True)

    dz = np.load(args.datastore)
    K = dz["keys"].astype(np.float32)
    y = dz["labels"].astype(np.float32)
    print(f"datastore {K.shape}, boundary rate {y.mean():.4f}", flush=True)

    base_05 = macro_f1(p_param, gold, doc_of, n_docs, 0.5)
    base_best = max((macro_f1(p_param, gold, doc_of, n_docs, t), t) for t in THRESH_GRID)
    print(f"\nBASELINE lam=0 parametric: F1@0.5={100*base_05:.2f}  "
          f"own-best={100*base_best[0]:.2f} @thr={base_best[1]}", flush=True)

    kmax = 64
    rng = np.random.default_rng(0)
    settings = ["off", "on"] if args.whiten == "both" else [args.whiten]
    results = []          # rows: (whiten,k,T,lam,f1@0.5,best_thr,f1@best)
    pknn_cache = {}       # (whiten,k,T) -> p_kNN vector

    for wh in settings:
        if wh == "on":
            mu, W = zca(K)
            Kw, Qw = (K - mu) @ W, (Q - mu) @ W
        else:
            Kw, Qw = K, Q
        # T0: median squared-distance to nearest non-self neighbor, estimated
        # from a 4096-key sample of the DATASTORE (no dev involved)
        samp = rng.choice(K.shape[0], size=min(4096, K.shape[0]), replace=False)
        Ds, Is = topk_neighbors(Kw[samp], Kw, 2, tag=f"/T0[{wh}]")
        nn = np.where(Is[:, 0] == samp, Ds[:, 1], Ds[:, 0])
        T0 = float(np.median(nn))
        print(f"[whiten={wh}] T0 (median NN squared dist on datastore) = {T0:.3f}",
              flush=True)

        D, I = topk_neighbors(Qw, Kw, kmax, tag=f"[{wh}]")
        nbr_lab = y[I]

        for k in (8, 16, 32, 64):
            for Tmul in (0.25, 0.5, 1.0, 2.0, 4.0):
                T = Tmul * T0
                pk = knn_probs(D, nbr_lab, k, T)
                pknn_cache[(wh, k, Tmul)] = pk
                for lam in (0.1, 0.2, 0.3, 0.5, 1.0):
                    p = lam * pk + (1 - lam) * p_param
                    f05 = macro_f1(p, gold, doc_of, n_docs, 0.5)
                    fb, tb = max((macro_f1(p, gold, doc_of, n_docs, t), t)
                                 for t in THRESH_GRID)
                    results.append((wh, k, Tmul, lam, f05, tb, fb))

    # ---------------- table: per (whiten,k,lam) show the best-T row ----------
    print("\n================ kNN-VOTER RESULTS (dev, 222 docs) ================")
    print("(T tuned on dev within {0.25,0.5,1,2,4}xT0 — selection caveat; "
          "baseline F1@0.5=%.2f own-best=%.2f)" % (100 * base_05, 100 * base_best[0]))
    print(f"{'whiten':6s} {'k':>3} {'lam':>5} {'T/T0':>5} {'F1@0.5':>8} "
          f"{'d@0.5':>7} {'bestThr':>8} {'F1@best':>8} {'d@best':>7}")
    best_row = None
    for wh in settings:
        for k in (8, 16, 32, 64):
            for lam in (0.1, 0.2, 0.3, 0.5, 1.0):
                rows = [r for r in results if r[0] == wh and r[1] == k and r[3] == lam]
                r = max(rows, key=lambda r: r[6])
                print(f"{r[0]:6s} {r[1]:3d} {r[3]:5.1f} {r[2]:5.2f} "
                      f"{100*r[4]:8.2f} {100*(r[4]-base_05):+7.2f} "
                      f"{r[5]:8.2f} {100*r[6]:8.2f} {100*(r[6]-base_best[0]):+7.2f}")
                if lam < 1.0 and (best_row is None or r[6] > best_row[6]):
                    best_row = r

    # ---------------- honest headline (nothing tuned on dev) -----------------
    wh0 = settings[0]
    pk_h = pknn_cache[(wh0, 16, 1.0)]
    p_h = 0.2 * pk_h + 0.8 * p_param
    h05 = macro_f1(p_h, gold, doc_of, n_docs, 0.5)
    hb, ht = max((macro_f1(p_h, gold, doc_of, n_docs, t), t) for t in THRESH_GRID)
    print(f"\nHONEST HEADLINE (pre-declared, untuned: lam=0.2 k=16 T=T0 "
          f"whiten={wh0}): F1@0.5={100*h05:.2f} ({100*(h05-base_05):+.2f}); "
          f"own-best={100*hb:.2f} @thr={ht} ({100*(hb-base_best[0]):+.2f})")

    # ---------------- mechanism check ---------------------------------------
    def mech(pk, name):
        m_fp = (p_param >= 0.8) & (gold == 0)
        m_tp = (p_param >= 0.8) & (gold == 1)
        print(f"\nMECHANISM CHECK [{name}]")
        print(f"  confident-FP (p_param>=0.8, gold=0): n={int(m_fp.sum())}  "
              f"p_kNN mean={pk[m_fp].mean():.3f} median={np.median(pk[m_fp]):.3f}  "
              f"frac<0.5={float((pk[m_fp] < 0.5).mean()):.3f}  "
              f"frac<0.2={float((pk[m_fp] < 0.2).mean()):.3f}")
        print(f"  confident-TP (p_param>=0.8, gold=1): n={int(m_tp.sum())}  "
              f"p_kNN mean={pk[m_tp].mean():.3f} median={np.median(pk[m_tp]):.3f}  "
              f"frac>=0.5={float((pk[m_tp] >= 0.5).mean()):.3f}  "
              f"frac<0.2={float((pk[m_tp] < 0.2).mean()):.3f}")

    mech(pk_h, f"headline k=16 T=T0 whiten={wh0}")
    bw, bk, bT, blam, _, bthr, bf1 = best_row
    pk_b = pknn_cache[(bw, bk, bT)]
    if (bw, bk, bT) != (wh0, 16, 1.0):
        mech(pk_b, f"best combo k={bk} T={bT}xT0 whiten={bw}")

    # ---------------- prep-slice + per-genre deltas at the best combo --------
    p_best = blam * pk_b + (1 - blam) * p_param
    print(f"\nBEST COMBO: whiten={bw} k={bk} T={bT}xT0 lam={blam} "
          f"thr={bthr} F1={100*bf1:.2f} ({100*(bf1-base_best[0]):+.2f} vs baseline own-best)")

    def slices(p, thr, tag):
        fp_tot = fp_prep = fn_tot = 0
        genre = defaultdict(lambda: [0, 0, 0])          # pos, fp, fn
        off = 0
        for d in docs:
            toks = d["tokens"]
            n = len(toks)
            g = bucket(toks)
            for i in range(n):
                if toks[i] == PARAGRAPH_TOKEN:
                    continue
                pi, gi = p[off], int(gold[off])
                off += 1
                pred = 1 if pi >= thr else 0
                genre[g][0] += 1
                if pred == 1 and gi == 0:
                    fp_tot += 1
                    genre[g][1] += 1
                    nxt = toks[i + 1] if i + 1 < n else None
                    if nxt in PREP_SET:
                        fp_prep += 1
                elif pred == 0 and gi == 1:
                    fn_tot += 1
                    genre[g][2] += 1
        print(f"  [{tag}] thr={thr}: FP={fp_tot} (prep-next FP={fp_prep}, "
              f"{100*fp_prep/max(fp_tot,1):.1f}%)  FN={fn_tot}")
        return fp_tot, fp_prep, fn_tot, genre

    print("\nPREP-SLICE + PER-GENRE (at the best combo's own threshold)")
    b_fp, b_fpp, b_fn, b_gen = slices(p_param, bthr, "parametric")
    c_fp, c_fpp, c_fn, c_gen = slices(p_best, bthr, "kNN-interp")
    print(f"  prep-next FP delta: {c_fpp - b_fpp:+d}  total FP delta: "
          f"{c_fp - b_fp:+d}  FN delta: {c_fn - b_fn:+d}")
    print(f"  {'genre':20s} {'pos':>7} {'FP/pos par':>11} {'FP/pos knn':>11} "
          f"{'dFP/pos':>9} {'FN par':>7} {'FN knn':>7}")
    for g in sorted(b_gen, key=lambda g: -b_gen[g][0]):
        pos = max(b_gen[g][0], 1)
        print(f"  {g:20s} {b_gen[g][0]:7d} {b_gen[g][1]/pos:11.4f} "
              f"{c_gen[g][1]/pos:11.4f} {(c_gen[g][1]-b_gen[g][1])/pos:+9.4f} "
              f"{b_gen[g][2]:7d} {c_gen[g][2]:7d}")

    print("\nCAVEAT: k, T, lam and the own-best threshold are selected on dev; "
          "treat tuned rows as an upper bound. The honest headline above uses "
          "the pre-declared untuned setting.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-datastore")
    b.add_argument("--model", required=True)
    b.add_argument("--jsonl", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit-docs", type=int, default=0)

    q = sub.add_parser("encode-queries")
    q.add_argument("--model", required=True)
    q.add_argument("--jsonl", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--limit-docs", type=int, default=0)

    e = sub.add_parser("eval")
    e.add_argument("--datastore", required=True)
    e.add_argument("--queries", required=True)
    e.add_argument("--param-probs", required=True)
    e.add_argument("--jsonl", required=True)
    e.add_argument("--whiten", choices=["off", "on", "both"], default="both")

    args = ap.parse_args()
    if args.cmd == "build-datastore":
        cmd_build_datastore(args)
    elif args.cmd == "encode-queries":
        cmd_encode_queries(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
