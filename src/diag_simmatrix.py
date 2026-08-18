"""SIGNAL DIAGNOSTIC (additive, CPU-only): are sentence boundaries visible as
BLOCK EDGES in token-similarity matrices from the frozen
union-NoPnx-NP-arabertv02-s42 encoder?

For ~60 sampled train windows (180 tokens, standard windowing = window 180 /
overlap 60, i.e. nominal stride 120 exactly as predict.predict_doc walks when
truncation does not bite), build S = cosine similarity of the L2-normalized
word-level (last-subword) hidden states, for BOTH the final layer and a mid
layer (default 8).

EDGE STATISTIC per edge position i (edge between token i and i+1; gold
labels[i]=1 means a sentence boundary FOLLOWS token i):

    C(i) = mean(S[a,b] same-side pairs within a +/-w local box)
         - mean(S[a,b] pairs crossing i within the box)

with the box = tokens {i-w+1..i} (left) and {i+1..i+w} (right); same-side =
all within-left + within-right off-diagonal pairs, crossing = the w x w
left-x-right block. High C(i) = position i looks like a block edge.

GATE NUMBER: pooled token-level ROC-AUC of C(i) separating gold-boundary
edges from non-boundary edges, for each layer and w in {3,5,8}. Also: AUC
restricted to {true boundaries} vs {token-tagger confident-FP edges}
(p_tagger >= 0.8, gold 0 — same 0.8 convention as knn_boundary's mechanism
check) — does the 2D signal see what the 1D tagger misses?

REUSE: the final-layer vectors are read from the completed kNN datastore
(knn_store/NoPnx-NP_s42_train.npz — same encoder, same last-subword grid,
window-averaged, float16). The mid layer is not cached anywhere, so the
sampled windows get ONE CPU forward pass each; that same pass also yields a
fresh (single-window) final layer as a cross-check and the tagger
probabilities for the confident-error slice.

PRE-REGISTERED DECISION RULE:
  best AUC < 0.60  -> PARK (blocks too faint at sentence granularity, as the
                      semantic-closeness measurement predicted)
  best AUC >= 0.65 -> the 2D arm earns a build (small U-Net over matrices,
                      ~1-5M params, 1020 training windows)
  0.60 - 0.65      -> report and let Noor call it.

Usage (CPU only — the battery owns the GPU):

  CUDA_VISIBLE_DEVICES=-1 python src/diag_simmatrix.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --jsonl data/NoPnx-NP_train.jsonl \
      --datastore knn_store/NoPnx-NP_s42_train.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl

ASCII_RAMP = " .:-=+*#%@"


# --------------------------------------------------------------------------- #
# encoder (CPU) — same loading guard as knn_boundary.load_encoder
# --------------------------------------------------------------------------- #
def load_encoder(model_dir):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the battery owns the GPU)"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    return tokenizer, model


def encode_window(chunk, tokenizer, model, mid_layer, max_length=512):
    """One forward pass over one window of words.

    Returns (v_final [L,768], v_mid [L,768], p_tag [L]) on the last-subword
    grid, where L <= len(chunk) (tail words lost to truncation are trimmed).
    """
    import torch

    words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in chunk]
    with torch.no_grad():
        enc = tokenizer(words, is_split_into_words=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        h_fin = out.hidden_states[-1][0].numpy()
        h_mid = out.hidden_states[mid_layer][0].numpy()
        p_all = torch.softmax(out.logits[0], dim=-1)[:, 1].numpy()

    word_ids = enc.word_ids(0)
    last_pos = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
        if nxt != wid:                       # last subword of this word
            last_pos[wid] = pos
    L = max(last_pos) + 1 if last_pos else 0
    assert all(i in last_pos for i in range(L)), "non-contiguous word coverage"
    idx = np.array([last_pos[i] for i in range(L)])
    return (h_fin[idx].astype(np.float32), h_mid[idx].astype(np.float32),
            p_all[idx].astype(np.float32))


# --------------------------------------------------------------------------- #
# similarity matrices + block-contrast edge statistic
# --------------------------------------------------------------------------- #
def cosine_matrix(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    Vn = V / np.maximum(n, 1e-8)
    return Vn @ Vn.T


def block_contrast(S, w):
    """C(i) for every valid edge i in [w-1, L-1-w]; returns (idx, C)."""
    L = S.shape[0]
    lo, hi = w - 1, L - 1 - w
    if hi < lo:
        return np.empty(0, dtype=np.int64), np.empty(0)
    iu = np.triu_indices(w, 1)
    idx = np.arange(lo, hi + 1)
    C = np.empty(len(idx))
    for j, i in enumerate(idx):
        B = S[i - w + 1:i + w + 1, i - w + 1:i + w + 1]
        left, right = B[:w, :w], B[w:, w:]
        cross = B[:w, w:]
        if w > 1:
            same = 0.5 * (left[iu].mean() + right[iu].mean())
        else:
            same = 0.0
        C[j] = same - cross.mean()
    return idx, C


def rank_auc(scores, pos_mask):
    """Mann-Whitney ROC-AUC with tie-averaged ranks (pure numpy)."""
    scores = np.asarray(scores, dtype=np.float64)
    pos_mask = np.asarray(pos_mask, dtype=bool)
    n_pos, n_neg = int(pos_mask.sum()), int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    uniq, inv, counts = np.unique(scores, return_inverse=True,
                                  return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = (csum - counts + 1 + csum) / 2.0     # 1-based average ranks
    r = avg_rank[inv]
    return float((r[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# ASCII rendering
# --------------------------------------------------------------------------- #
def ascii_matrix(S, gold, max_cols=60):
    """Downsampled similarity matrix with gold boundaries marked.

    Header/footer rows carry '|' at boundary columns; the left margin carries
    '|' on rows whose token block contains a boundary token.
    """
    L = S.shape[0]
    ds = max(1, int(np.ceil(L / max_cols)))
    m = int(np.ceil(L / ds))
    D = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            D[a, b] = S[a * ds:(a + 1) * ds, b * ds:(b + 1) * ds].mean()
    off = D[~np.eye(m, dtype=bool)]
    lo, hi = np.percentile(off, 5), np.percentile(off, 95)
    span = max(hi - lo, 1e-8)
    q = np.clip((D - lo) / span, 0, 1)
    chars = np.array(list(ASCII_RAMP))
    idx = np.minimum((q * len(ASCII_RAMP)).astype(int), len(ASCII_RAMP) - 1)

    bcols = sorted({int(i) // ds for i in np.flatnonzero(gold[:L])})
    marks = [("|" if c in bcols else " ") for c in range(m)]
    lines = ["    " + "".join(marks) + "   <- gold boundaries (downsample x%d)" % ds]
    for a in range(m):
        margin = "|" if a in bcols else " "
        lines.append("  %s " % margin + "".join(chars[idx[a]]))
    lines.append("    " + "".join(marks))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def enumerate_windows(docs, window, overlap, min_len):
    stride = window - overlap
    wins = []
    for di, d in enumerate(docs):
        n = len(d["tokens"])
        start = 0
        while start < n:
            L = min(window, n - start)
            if L >= min_len:
                wins.append((di, start, L))
            if start + window >= n:
                break
            start += stride
    return wins


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="runs/union-NoPnx-NP-arabertv02-s42")
    ap.add_argument("--jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--datastore", default="knn_store/NoPnx-NP_s42_train.npz")
    ap.add_argument("--n-windows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mid-layer", type=int, default=8)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--w-list", type=int, nargs="+", default=[3, 5, 8])
    ap.add_argument("--conf-thresh", type=float, default=0.8)
    ap.add_argument("--n-ascii", type=int, default=3)
    args = ap.parse_args()

    docs = load_jsonl(args.jsonl)
    wmax = max(args.w_list)
    wins = enumerate_windows(docs, args.window, args.overlap,
                             min_len=2 * wmax + 2)
    rng = np.random.default_rng(args.seed)
    take = min(args.n_windows, len(wins))
    sample = [wins[j] for j in sorted(rng.choice(len(wins), size=take,
                                                 replace=False))]
    print(f"# {len(docs)} docs -> {len(wins)} windows "
          f"(window={args.window}, stride={args.window - args.overlap}); "
          f"sampled {take} (seed={args.seed})", flush=True)

    # ---------------- REUSED final-layer vectors from the kNN datastore ------
    dz = np.load(args.datastore)
    store_ids = [str(x) for x in dz["doc_ids"]]
    assert store_ids == [d["doc_id"] for d in docs], \
        "datastore doc order != jsonl doc order"
    K = dz["keys"]
    doc_idx = dz["doc_idx"]
    word_pos = dz["word_pos"]
    doc_off = np.searchsorted(doc_idx, np.arange(len(docs)))
    print(f"# reusing datastore {K.shape} (final layer, window-averaged, "
          f"float16) from {args.datastore}", flush=True)

    def store_slice(di, start, L):
        o = doc_off[di]
        seg = K[o + start:o + start + L].astype(np.float32)
        assert (word_pos[o + start:o + start + L]
                == np.arange(start, start + L)).all(), "grid mismatch"
        return seg

    # ---------------- one CPU pass per window for mid layer + tagger p -------
    tokenizer, model = load_encoder(args.model)
    n_layers = model.config.num_hidden_layers
    assert 0 < args.mid_layer <= n_layers, "bad --mid-layer"
    print(f"# encoder: {args.model} ({n_layers} layers, "
          f"mid layer = {args.mid_layer})", flush=True)

    reps = ["final(store)", "final(fresh)", f"mid{args.mid_layer}(fresh)"]
    pooled = {(r, w): ([], []) for r in reps for w in args.w_list}  # (C, gold)
    conf_fp = {(r, w): [] for r in reps for w in args.w_list}       # mask
    per_win = []                                    # for ASCII selection
    cos_agree = []
    t0 = time.time()
    for k, (di, start, L0) in enumerate(sample):
        d = docs[di]
        chunk = d["tokens"][start:start + L0]
        v_fin, v_mid, p_tag = encode_window(chunk, tokenizer, model,
                                            args.mid_layer)
        L = v_fin.shape[0]                          # truncation-trimmed
        gold = np.asarray(d["labels"][start:start + L], dtype=np.int8)
        v_store = store_slice(di, start, L)

        fs = (v_store / np.maximum(np.linalg.norm(v_store, axis=1,
                                                  keepdims=True), 1e-8))
        ff = (v_fin / np.maximum(np.linalg.norm(v_fin, axis=1,
                                                keepdims=True), 1e-8))
        cos_agree.append(float((fs * ff).sum(axis=1).mean()))

        mats = {"final(store)": cosine_matrix(v_store),
                "final(fresh)": cosine_matrix(v_fin),
                f"mid{args.mid_layer}(fresh)": cosine_matrix(v_mid)}
        for r, S in mats.items():
            for w in args.w_list:
                idx, C = block_contrast(S, w)
                pooled[(r, w)][0].append(C)
                pooled[(r, w)][1].append(gold[idx])
                cf = (p_tag[idx] >= args.conf_thresh) & (gold[idx] == 0)
                conf_fp[(r, w)].append(cf)
        per_win.append((di, start, L, int(gold[:L - 1].sum()), gold, mats))
        if (k + 1) % 10 == 0 or k == take - 1:
            el = time.time() - t0
            print(f"  [{k + 1}/{take}] {el:.0f}s elapsed, "
                  f"eta {el / (k + 1) * (take - k - 1):.0f}s", flush=True)

    print(f"# store-vs-fresh final-layer agreement: mean cosine = "
          f"{np.mean(cos_agree):.4f} (store is window-averaged float16; "
          f"high = same grid)", flush=True)

    # ---------------- AUC table ----------------------------------------------
    print("\n================ BLOCK-CONTRAST ROC-AUC (pooled over windows) "
          "================")
    print(f"{'layer':16s} {'w':>3} {'AUC':>7} {'n_pos':>7} {'n_neg':>7} "
          f"{'C|bound':>9} {'C|non':>8}   {'AUC(bound vs confFP)':>21} "
          f"{'n_confFP':>8} {'C|confFP':>9}")
    best = (float("-inf"), None, None)
    for r in reps:
        for w in args.w_list:
            C = np.concatenate(pooled[(r, w)][0])
            g = np.concatenate(pooled[(r, w)][1]).astype(bool)
            cf = np.concatenate(conf_fp[(r, w)])
            auc = rank_auc(C, g)
            sub = g | cf
            auc_cf = rank_auc(C[sub], g[sub])
            print(f"{r:16s} {w:3d} {auc:7.4f} {int(g.sum()):7d} "
                  f"{int((~g).sum()):7d} {C[g].mean():9.4f} "
                  f"{C[~g].mean():8.4f}   {auc_cf:21.4f} "
                  f"{int(cf.sum()):8d} "
                  f"{C[cf].mean() if cf.any() else float('nan'):9.4f}")
            if auc > best[0]:
                best = (auc, r, w)
    print(f"\nBEST: AUC={best[0]:.4f} at layer={best[1]}, w={best[2]}")
    print(f"(tagger confident-FP = single-window p >= {args.conf_thresh} & "
          f"gold 0; single-window probs, not the overlap-averaged decode "
          f"grid — diagnostic only)")

    # ---------------- ASCII examples -----------------------------------------
    br, bw = best[1], best[2]
    show = sorted(per_win, key=lambda t: -t[3])[:args.n_ascii]
    for di, start, L, nb, gold, mats in show:
        print(f"\n---- EXAMPLE doc={docs[di]['doc_id']} tokens "
              f"[{start}:{start + L}]  {nb} gold boundaries  "
              f"layer={br} ----")
        print(ascii_matrix(mats[br], gold))

    # ---------------- pre-registered decision rule ---------------------------
    print("\n================ PRE-REGISTERED DECISION RULE ================")
    print(f"best AUC = {best[0]:.4f}")
    if best[0] < 0.60:
        print("VERDICT: PARK — the blocks are too faint at sentence "
              "granularity, as the semantic-closeness measurement predicted "
              "(best AUC < 0.60).")
    elif best[0] >= 0.65:
        print("VERDICT: the 2D arm earns a build (small U-Net over matrices, "
              "~1-5M params, 1020 training windows) (best AUC >= 0.65).")
    else:
        print("VERDICT: 0.60 <= best AUC < 0.65 — report and let Noor "
              "call it.")


if __name__ == "__main__":
    main()
