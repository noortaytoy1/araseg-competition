"""Context-gated META-LEARNER over Omar's 6 encoder voters (NoPnx-NP).

Noor's exact ask: a meta-learner that learns WHICH of Omar's encoder voters to
trust, CONDITIONED ON THE INPUT CONTEXT, evaluated OVER OMAR'S OWN ensemble
pipeline (his DP decode + official per-doc macro-F1).

What this is (and is NOT)
-------------------------
VOTERS are ONLY Omar's 6 encoder models, from his cached probs:
    arabertv02-s1, arabertv02-s2, arabertv02-s3, arabertv02-s42,
    araelectra-s42, arbertv2-s42
loaded from probs/union-NoPnx-NP-<member>_dev.npz — the 6 members of
run_trdev_ensemble.sh. (weighted_ensemble.py's probs/<task>_dev_voter{...}.npz
do not exist on disk; the task pre-authorised this exact fallback set.)

The gate/meta-model sees, per token position:
    [6 voter probs]  +  [INPUT-CONTEXT features]  +  [context x voter]
where INPUT-CONTEXT features are properties of the TEXT ITSELF only:
    is-conjunction (و/ف/ثم/أو/لكن and و+ prefix), is-preposition,
    token length (log), is-number, is-OOV-vs-train, relative position,
    distance-to-doc-end (log), plus a bias column.
Because the design includes context x voter interaction terms, the effective
weight placed on each voter VARIES WITH CONTEXT — that is the "learns which to
trust based on the thing." A logistic-regression meta-model (default) or a small
MLP maps these to P(boundary). NO U-Net, NO instability/MC-dropout, NO RAE, NO
judge, NO kNN, NO SaT — nothing derived from any other MODEL. Encoders + input
context ONLY.

DECODE is OMAR'S DP decode: dp_decode.decode_doc with a length prior fit by
dp_decode.fit_length_logprob on train, exactly as weighted_ensemble.py uses it.
A single (lam, bias) is frozen ONCE on the uniform baseline's OOF predictions
(dp_decode's own grid) and applied identically to all three arms so the arms are
compared on the same decode.

EVAL is k=5 doc-level CV on dev (seed 42, no doc across folds; folds via
stacker.length_stratified_folds — the same maker the other combiners use). All
three arms are scored through Omar's DP decode on the SAME held-out folds with
eval_local.compute_metrics (official per-doc macro-F1):
    (i)   UNIFORM average of the 6                              (baseline)
    (ii)  OMAR's weighted_ensemble greedy integer weights       (his combiner)
    (iii) the context-gated META-LEARNER                        (ours)
Each arm's per-fold decoded preds are collected over all folds and scored once,
so every dev doc is scored exactly once, out-of-fold.

Pre-registered reading (verbatim from the task): the meta-learner WORKS if
held-out macro-F1 beats BOTH uniform and Omar's weighted by > +0.2; MARGINAL if
it beats them at all; NULL if not.

    CUDA_VISIBLE_DEVICES=-1 python src/moe_metalearner.py --task NoPnx-NP \
        --out-dir runs/moe_metalearner_2026-07-11
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from data import PARAGRAPH_TOKEN, load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from eval_local import compute_metrics
from stacker import length_stratified_folds

# Omar's 6 encoder voters, in a fixed order. These are the members of
# run_trdev_ensemble.sh; their cached dev probs live at
# probs/union-<task>-<member>_dev.npz.
MEMBERS = [
    "arabertv02-s1",
    "arabertv02-s2",
    "arabertv02-s3",
    "arabertv02-s42",
    "araelectra-s42",
    "arbertv2-s42",
]

# Human-readable names for the report (which encoder is which column).
VOTER_NAMES = list(MEMBERS)

# INPUT-CONTEXT feature names, in column order (text-only; no model signal).
CONTEXT_FEATURES = [
    "is_conjunction",     # و / ف / ثم / أو / لكن, or a و+ / ف+ prefix
    "is_preposition",     # في / من / إلى / على / عن / ب / ل / ك ...
    "tok_len_log",        # log(1 + character length of the token)
    "is_number",          # token is (Arabic/ASCII) digits
    "is_oov_train",       # token unseen in the TRAIN split vocabulary
    "rel_pos",            # position / (doc_len - 1)  in [0, 1]
    "dist_end_log",       # log(1 + tokens remaining until doc end)
    "bias",               # constant 1.0 (intercept / always-on context)
]

# dp_decode's own default grid (see dp_decode.main defaults + trdev script).
LAM_GRID = [0.0, 0.25, 0.5, 1.0, 1.5]
BIAS_GRID = [-0.5, -0.25, 0.0, 0.25, 0.5]

# --------------------------------------------------------------------------- #
# Arabic function-word lexicons (TEXT properties, not model outputs).
# Conjunctions per the task: و ف ثم أو لكن and the و+ (waw) prefix family.
# --------------------------------------------------------------------------- #
_CONJ_WORDS = {"و", "ف", "ثم", "أو", "لكن", "بل", "أم", "حتى"}
_CONJ_PREFIXES = ("و", "ف")  # clitic و+/ف+ attached to the next word
_PREP_WORDS = {
    "في", "من", "إلى", "على", "عن", "عند", "مع", "بين", "لدى", "حول",
    "ب", "ل", "ك", "منذ", "خلال", "نحو", "دون", "بعد", "قبل", "فوق", "تحت",
}
_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"


def _is_number(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return False
    return all((c.isdigit() or c in _ARABIC_INDIC or c in ".,٫٬") for c in t) and any(
        (c.isdigit() or c in _ARABIC_INDIC) for c in t
    )


def _is_conjunction(tok: str) -> bool:
    if tok in _CONJ_WORDS:
        return True
    # و+ / ف+ clitic: a longer token that starts with the conjunction letter.
    # Kept conservative: only flag the و/ف prefix (the task's "و+" family).
    return len(tok) > 1 and tok[0] in _CONJ_PREFIXES


def _is_preposition(tok: str) -> bool:
    return tok in _PREP_WORDS


def build_train_vocab(train_docs) -> set:
    """Vocabulary of the TRAIN split (closed-track legal) for the OOV flag."""
    vocab = set()
    for d in train_docs:
        for tok in d["tokens"]:
            if tok != PARAGRAPH_TOKEN:
                vocab.add(tok)
    return vocab


def context_features_for_doc(tokens, train_vocab) -> np.ndarray:
    """(n_tokens, len(CONTEXT_FEATURES)) matrix of TEXT-ONLY features.

    Row per token; \n (PARAGRAPH_TOKEN) positions get all-zero context except
    the bias column, and are never boundaries anyway (dp_decode forces that).
    """
    n = len(tokens)
    rel = np.zeros(n)
    dist_end = np.zeros(n)
    denom = max(n - 1, 1)
    for i in range(n):
        rel[i] = i / denom
        dist_end[i] = np.log1p(n - 1 - i)
    feats = np.zeros((n, len(CONTEXT_FEATURES)), dtype=np.float64)
    for i, tok in enumerate(tokens):
        if tok == PARAGRAPH_TOKEN:
            feats[i, CONTEXT_FEATURES.index("bias")] = 1.0
            continue
        feats[i, 0] = 1.0 if _is_conjunction(tok) else 0.0
        feats[i, 1] = 1.0 if _is_preposition(tok) else 0.0
        feats[i, 2] = np.log1p(len(tok))
        feats[i, 3] = 1.0 if _is_number(tok) else 0.0
        feats[i, 4] = 0.0 if tok in train_vocab else 1.0
        feats[i, 5] = rel[i]
        feats[i, 6] = dist_end[i]
        feats[i, 7] = 1.0
    return feats


# --------------------------------------------------------------------------- #
# Cache loading
# --------------------------------------------------------------------------- #
def load_voter_caches(task: str, split: str, probs_dir: str = "probs"):
    """Return dict[member] -> np.load handle for the 6 union caches.

    Raises with a clear message if any of the 6 caches is missing, so we never
    silently fall back to fewer than Omar's 6 encoders.
    """
    caches = {}
    for m in MEMBERS:
        path = os.path.join(probs_dir, f"union-{task}-{m}_{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"voter cache missing: {path} (need all 6 of Omar's encoders)"
            )
        caches[m] = np.load(path)
    return caches


def voter_matrix_for_doc(caches, doc_id, n_tokens) -> np.ndarray:
    """(n_tokens, 6) matrix of the 6 encoder boundary probs for one doc."""
    cols = []
    for m in MEMBERS:
        c = caches[m]
        if doc_id not in c.files:
            raise KeyError(f"{doc_id} absent from voter cache {m}")
        v = np.asarray(c[doc_id], dtype=np.float64)
        if v.shape[0] != n_tokens:
            raise ValueError(f"{m}:{doc_id} len {v.shape[0]} != {n_tokens}")
        cols.append(v)
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------- #
# Design matrices
# --------------------------------------------------------------------------- #
def assemble_arrays(docs, caches, train_vocab):
    """Stack per-doc arrays into flat token-level arrays.

    Returns
      V        (N, 6)   the 6 voter probs
      C        (N, F)   input-context features
      y        (N,)     gold boundary labels
      doc_of   (N,)     doc index per token
      ids      list     doc_ids in order
      per_doc  list of (start, stop) slices into the flat arrays
    """
    Vs, Cs, ys, dof = [], [], [], []
    ids = [d["doc_id"] for d in docs]
    spans = []
    cursor = 0
    for di, d in enumerate(docs):
        toks = d["tokens"]
        n = len(toks)
        V = voter_matrix_for_doc(caches, d["doc_id"], n)
        C = context_features_for_doc(toks, train_vocab)
        Vs.append(V)
        Cs.append(C)
        ys.append(np.asarray(d["labels"], dtype=np.int64))
        dof.append(np.full(n, di, dtype=np.int64))
        spans.append((cursor, cursor + n))
        cursor += n
    V = np.concatenate(Vs, axis=0)
    C = np.concatenate(Cs, axis=0)
    y = np.concatenate(ys, axis=0)
    doc_of = np.concatenate(dof, axis=0)
    return V, C, y, doc_of, ids, spans


def interaction_block(V, C):
    """context x voter interactions: for each voter column and each NON-bias
    context column, the product. Lets a voter's weight depend on context.
    """
    # drop the constant bias context column from the interaction (it would just
    # duplicate the raw voter columns).
    bias_idx = CONTEXT_FEATURES.index("bias")
    keep = [j for j in range(C.shape[1]) if j != bias_idx]
    blocks = []
    names = []
    for vi in range(V.shape[1]):
        for cj in keep:
            blocks.append((V[:, vi] * C[:, cj])[:, None])
            names.append(f"{VOTER_NAMES[vi]}×{CONTEXT_FEATURES[cj]}")
    return np.concatenate(blocks, axis=1), names


def build_design(V, C, with_interactions=True):
    """Meta-model input: [6 voter probs | context | context x voter]."""
    parts = [V, C]
    names = list(VOTER_NAMES) + list(CONTEXT_FEATURES)
    if with_interactions:
        inter, inter_names = interaction_block(V, C)
        parts.append(inter)
        names += inter_names
    X = np.concatenate(parts, axis=1)
    return X, names


# --------------------------------------------------------------------------- #
# Meta-models (encoders + input-context ONLY)
# --------------------------------------------------------------------------- #
def _standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def fit_meta(X, y, kind="lr", seed=42):
    """Fit the meta-model. Standardize, then LR (default) or small MLP.

    Returns a callable prob(Xnew) -> P(boundary), plus a coef vector for the LR
    path (None for MLP) so we can report which voter is trusted in which context.
    """
    mu, sd = _standardize_fit(X)
    Xs = (X - mu) / sd
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=seed
        )
        clf.fit(Xs, y)
        coef = clf.coef_.ravel().copy()

        def prob(Xnew):
            return clf.predict_proba((Xnew - mu) / sd)[:, 1]

        return prob, coef
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            hidden_layer_sizes=(32,), activation="relu", alpha=1e-3,
            max_iter=400, random_state=seed,
        )
        clf.fit(Xs, y)

        def prob(Xnew):
            return clf.predict_proba((Xnew - mu) / sd)[:, 1]

        return prob, None
    raise ValueError(f"unknown meta kind {kind!r}")


# --------------------------------------------------------------------------- #
# Omar's greedy weighted-ensemble combiner (his logic, reused, fit per fold)
# --------------------------------------------------------------------------- #
def omar_greedy_weights(train_docs, train_span, V, spans, ids, gold, len_lp,
                        lam, bias):
    """Reproduce weighted_ensemble.py's greedy integer-weight search, but on the
    supplied TRAIN docs only (so it is honest under CV). Returns weight vector.

    Mirrors weighted_ensemble.main: start uniform, try +/-1 bumps in [0..],
    keep whatever improves dev(=train-fold) macro-F1 decoded with the frozen DP.
    """
    tr_index = {ids[i]: spans[i] for i in train_span}

    def score(weights):
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        preds = {}
        for d in train_docs:
            did = d["doc_id"]
            s, e = tr_index[did]
            blended = V[s:e] @ w
            preds[did] = decode_doc(d["tokens"], blended, len_lp, lam, bias)
        g = {d["doc_id"]: gold[d["doc_id"]] for d in train_docs}
        return compute_metrics(g, preds)["macro_f1"]

    n = V.shape[1]                       # number of voters (generalizes past 6)
    best_w = [1] * n
    best = score(best_w)
    for _ in range(n):
        improved = False
        for i in range(n):
            for delta in (1, -1):
                w = list(best_w)
                w[i] += delta
                if w[i] < 0 or sum(w) == 0:
                    continue
                s = score(w)
                if s > best + 1e-5:
                    best, best_w, improved = s, w, True
        if not improved:
            break
    w = np.array(best_w, dtype=float)
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Decode helpers
# --------------------------------------------------------------------------- #
def decode_probs(docs_subset, prob_by_doc, len_lp, lam, bias):
    """Decode a set of docs given a dict doc_id -> per-token prob vector."""
    return {
        d["doc_id"]: decode_doc(d["tokens"], prob_by_doc[d["doc_id"]], len_lp,
                                lam, bias)
        for d in docs_subset
    }


def pick_decode_config(docs, blended_by_doc, len_lp, gold):
    """Freeze (lam, bias) on the UNIFORM baseline's OOF preds over dp_decode's
    grid. Chosen once, applied identically to all three arms."""
    best = (LAM_GRID[0], BIAS_GRID[0], -1.0)
    for lam in LAM_GRID:
        for bias in BIAS_GRID:
            preds = decode_probs(docs, blended_by_doc, len_lp, lam, bias)
            f1 = compute_metrics(gold, preds)["macro_f1"]
            if f1 > best[2]:
                best = (lam, bias, f1)
    return best[0], best[1]


# --------------------------------------------------------------------------- #
# The k=5 doc-level CV, three arms, all through Omar's DP decode
# --------------------------------------------------------------------------- #
def run_cv(docs, caches, train_vocab, len_lp, k=5, seed=42, meta_kind="lr",
           meta_own_decode=False):
    ids = [d["doc_id"] for d in docs]
    lens = [len(d["tokens"]) for d in docs]
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}

    V, C, y, doc_of, ids2, spans = assemble_arrays(docs, caches, train_vocab)
    assert ids2 == ids
    X_full, feat_names = build_design(V, C, with_interactions=True)

    fold_of = length_stratified_folds(ids, lens, k, seed)  # per-doc fold id
    # doc-level leakage guard: no doc appears in two folds (it can't — one id).
    doc_fold = np.array(fold_of, dtype=np.int64)

    # ---- freeze the decode config on the UNIFORM baseline, out-of-fold ----
    uni_by_doc = {}
    for di, (s, e) in enumerate(spans):
        uni_by_doc[ids[di]] = V[s:e].mean(axis=1)
    lam, bias = pick_decode_config(docs, uni_by_doc, len_lp, gold)

    # per-arm OOF decoded predictions (each doc decoded exactly once)
    preds_uniform, preds_omar, preds_meta = {}, {}, {}
    meta_prob_by_doc = {}            # meta OOF probs (decoded after the fold loop)
    omar_weights_per_fold = []
    meta_coef_accum = np.zeros(X_full.shape[1])
    meta_coef_folds = 0

    for f in range(k):
        held_docs = [d for d, ff in zip(docs, fold_of) if ff == f]
        train_docs = [d for d, ff in zip(docs, fold_of) if ff != f]
        held_idx = [i for i, ff in enumerate(fold_of) if ff == f]
        train_idx = [i for i, ff in enumerate(fold_of) if ff != f]

        # token masks
        tr_mask = np.isin(doc_of, np.array(train_idx))
        # ---- arm (i): uniform (no fit) ----
        for di in held_idx:
            s, e = spans[di]
            preds_uniform[ids[di]] = decode_doc(
                docs[di]["tokens"], V[s:e].mean(axis=1), len_lp, lam, bias)

        # ---- arm (ii): Omar greedy weights, fit on train fold only ----
        w = omar_greedy_weights(train_docs, train_idx, V, spans, ids, gold,
                                len_lp, lam, bias)
        omar_weights_per_fold.append(w)
        for di in held_idx:
            s, e = spans[di]
            preds_omar[ids[di]] = decode_doc(
                docs[di]["tokens"], V[s:e] @ w, len_lp, lam, bias)

        # ---- arm (iii): context-gated meta-learner, fit on train fold ----
        prob_fn, coef = fit_meta(X_full[tr_mask], y[tr_mask], meta_kind, seed)
        if coef is not None:
            meta_coef_accum += coef
            meta_coef_folds += 1
        for di in held_idx:
            s, e = spans[di]
            meta_prob_by_doc[ids[di]] = prob_fn(X_full[s:e])   # collect OOF probs

    # The meta arm's LR output has a DIFFERENT probability scale than a uniform
    # blend (sharper, shifted by class_weight=balanced). Decoding it with the
    # config frozen on the uniform blend handicaps it. When meta_own_decode, pick
    # (lam,bias) on the meta arm's OWN out-of-fold probs -- the same tuning the
    # uniform arm gets on its own blend -- so the comparison is fair.
    if meta_own_decode:
        lam_m, bias_m = pick_decode_config(docs, meta_prob_by_doc, len_lp, gold)
    else:
        lam_m, bias_m = lam, bias
    for di, (s, e) in enumerate(spans):
        did = ids[di]
        preds_meta[did] = decode_doc(docs[di]["tokens"],
                                     meta_prob_by_doc[did], len_lp, lam_m, bias_m)

    m_uniform = compute_metrics(gold, preds_uniform)
    m_omar = compute_metrics(gold, preds_omar)
    m_meta = compute_metrics(gold, preds_meta)

    mean_coef = (meta_coef_accum / meta_coef_folds) if meta_coef_folds else None
    return {
        "lam": lam,
        "bias": bias,
        "lam_meta": lam_m,
        "bias_meta": bias_m,
        "uniform": m_uniform,
        "omar": m_omar,
        "meta": m_meta,
        "omar_weights_per_fold": [wi.tolist() for wi in omar_weights_per_fold],
        "omar_weights_mean": np.mean(omar_weights_per_fold, axis=0).tolist(),
        "meta_coef_mean": None if mean_coef is None else mean_coef.tolist(),
        "feat_names": feat_names,
    }


def summarize_coef(feat_names, coef, top=12):
    """Which encoder is trusted in which context: rank |coef| among the
    interaction and voter terms."""
    order = np.argsort(-np.abs(np.asarray(coef)))
    rows = []
    for j in order[:top]:
        rows.append((feat_names[j], float(coef[j])))
    return rows


def verdict_line(m_meta, m_uniform, m_omar):
    dm_u = (m_meta - m_uniform) * 100
    dm_o = (m_meta - m_omar) * 100
    beats_both = (dm_u > 0.2) and (dm_o > 0.2)
    beats_any = (m_meta > m_uniform) and (m_meta > m_omar)
    if beats_both:
        return "WORKS"
    if beats_any:
        return "MARGINAL"
    return "NULL"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--meta", default="lr", choices=["lr", "mlp"])
    ap.add_argument("--members", default=None,
                    help="comma-separated voter member names (default: Omar's 6). "
                         "Each must have probs/union-<task>-<member>_<split>.npz.")
    ap.add_argument("--meta-own-decode", action="store_true",
                    help="tune the DP (lam,bias) on the meta arm's OWN out-of-fold "
                         "probs instead of borrowing the uniform-tuned config "
                         "(fair decode for the meta arm's probability scale).")
    ap.add_argument("--out-dir", default="runs/moe_metalearner_2026-07-11")
    args = ap.parse_args()

    # Additive override: point the meta-learner at a different (e.g. diverse) pool
    # without changing the default 6-encoder behavior. Rebinds the module globals
    # that load_voter_caches / voter_matrix_for_doc / interaction naming read.
    if args.members:
        global MEMBERS, VOTER_NAMES
        MEMBERS = [m.strip() for m in args.members.split(",") if m.strip()]
        VOTER_NAMES = list(MEMBERS)

    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    train_docs = load_jsonl(f"data/{args.task}_train.jsonl")
    train_vocab = build_train_vocab(train_docs)
    len_lp = fit_length_logprob(train_docs)
    caches = load_voter_caches(args.task, args.split)

    res = run_cv(docs, caches, train_vocab, len_lp, args.k, args.seed, args.meta,
                 meta_own_decode=args.meta_own_decode)

    fu = res["uniform"]["macro_f1"] * 100
    fo = res["omar"]["macro_f1"] * 100
    fm = res["meta"]["macro_f1"] * 100
    verdict = verdict_line(res["meta"]["macro_f1"], res["uniform"]["macro_f1"],
                           res["omar"]["macro_f1"])

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 70)
    print(f"CONTEXT-GATED META-LEARNER  task={args.task}  k={args.k}  "
          f"seed={args.seed}  meta={args.meta}")
    print(f"DP decode: uniform/Omar lam={res['lam']} bias={res['bias']}"
          f"  |  meta arm lam={res['lam_meta']} bias={res['bias_meta']}"
          f"  ({'meta tuned on its OWN OOF' if args.meta_own_decode else 'all arms share uniform-tuned'})")
    print("-" * 70)
    print(f"(i)   uniform average of 6      macro-F1 = {fu:6.2f}")
    print(f"(ii)  Omar weighted (greedy)    macro-F1 = {fo:6.2f}")
    print(f"(iii) context-gated meta-learner macro-F1 = {fm:6.2f}")
    print("-" * 70)
    print(f"delta meta - uniform = {fm - fu:+.2f}")
    print(f"delta meta - Omar    = {fm - fo:+.2f}")
    print(f"VERDICT: {verdict}")
    print("-" * 70)
    print("Omar greedy weights (mean over folds), 6 encoders:")
    for name, w in zip(VOTER_NAMES, res["omar_weights_mean"]):
        print(f"    {name:16s} {w:.3f}")
    if res["meta_coef_mean"] is not None:
        print("Which encoder trusted in which context (top |coef|, LR):")
        for name, c in summarize_coef(res["feat_names"], res["meta_coef_mean"]):
            print(f"    {name:34s} {c:+.3f}")

    out = {
        "task": args.task, "k": args.k, "seed": args.seed, "meta": args.meta,
        "lam": res["lam"], "bias": res["bias"],
        "voters": VOTER_NAMES,
        "context_features": CONTEXT_FEATURES,
        "uniform_macro_f1": fu, "omar_macro_f1": fo, "meta_macro_f1": fm,
        "delta_meta_minus_uniform": fm - fu,
        "delta_meta_minus_omar": fm - fo,
        "verdict": verdict,
        "omar_weights_mean": res["omar_weights_mean"],
        "omar_weights_per_fold": res["omar_weights_per_fold"],
        "meta_top_context_voter_terms":
            (summarize_coef(res["feat_names"], res["meta_coef_mean"], top=20)
             if res["meta_coef_mean"] is not None else None),
        "uniform_PR": [res["uniform"]["macro_precision"] * 100,
                       res["uniform"]["macro_recall"] * 100],
        "omar_PR": [res["omar"]["macro_precision"] * 100,
                    res["omar"]["macro_recall"] * 100],
        "meta_PR": [res["meta"]["macro_precision"] * 100,
                    res["meta"]["macro_recall"] * 100],
    }
    out_path = os.path.join(args.out_dir, f"{args.task}_metalearner_{args.meta}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
