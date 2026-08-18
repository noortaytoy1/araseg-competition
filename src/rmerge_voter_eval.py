"""Evaluate the recursive merge model AS AN ENSEMBLE VOTER (NoPnx-NP).

The whole point of the recursive model is to be a STRUCTURALLY DIFFERENT,
DECORRELATED voter -- the diagnostic showed the encoder pool is all redundant
twins (mean corr 0.80) and is missing exactly this. So we measure:

  solo_AUC / solo_F1   is the recursive model any good on its own?
  mean_corr vs pool     is it actually decorrelated? (phi < 0.65 = the missing kind)
  pool vs pool+rmerge   does ADDING it to the 84.94 AraBERT core help?
                        (uniform + Omar-greedy; Krogh-Vedelsby dividend)

Reads the recursive model's per-word dev probs from an .npz (rmerge predict
--npz-out) and the AraBERT-core caches. Reuses the audited primitives.

  python src/rmerge_voter_eval.py --rmerge-npz probs/rmerge-NoPnx-NP-seed42_dev.npz \
      --pool arabertv02-s1,arabertv02-s2,arabertv02-s3,arabertv02-s42,araelectra-s42,arbertv2-s42
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from data import load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from eval_local import compute_metrics
from moe_metalearner import pick_decode_config, omar_greedy_weights
from loo_uniform import load_pool, uniform_blend_by_doc, score_pool


def blend_dict(docs, prob_by_doc_list):
    """Uniform-average a list of {doc_id->probs} dicts, aligned per doc."""
    out = {}
    for d in docs:
        did = d["doc_id"]
        cols = [pb[did] for pb in prob_by_doc_list]
        out[did] = np.mean(np.stack(cols, axis=1), axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--rmerge-npz", required=True)
    ap.add_argument("--pool", required=True, help="comma-separated AraBERT-core members")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    members = [m.strip() for m in args.pool.split(",") if m.strip()]
    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    len_lp = fit_length_logprob(load_jsonl(f"data/{args.task}_train.jsonl"))
    caches = load_pool(args.task, args.split, members)

    rm = np.load(args.rmerge_npz)
    # rmerge probs per doc, aligned to token count (assert -> crash on mismatch)
    rmerge = {}
    for d in docs:
        did = d["doc_id"]
        n = len(d["tokens"])
        if did not in rm.files:
            raise KeyError(f"{did} missing from {args.rmerge_npz}")
        v = np.asarray(rm[did], dtype=np.float64)
        if v.shape[0] != n:
            raise ValueError(f"rmerge {did}: len {v.shape[0]} != {n}")
        rmerge[did] = v

    # --- solo AUC + solo F1 for the recursive model ---
    ps, gs = [], []
    for d in docs:
        n = len(d["tokens"])
        if n < 2:
            continue
        ps.append(rmerge[d["doc_id"]][: n - 1])
        gs.append(np.asarray(d["labels"], dtype=np.int64)[: n - 1])
    rj, gj = np.concatenate(ps), np.concatenate(gs)
    solo_auc = roc_auc_score(gj, rj) if len(set(gj.tolist())) > 1 else float("nan")

    lam_r, bias_r = pick_decode_config(docs, rmerge, len_lp, gold)
    solo_f1 = score_pool(docs, rmerge, len_lp, lam_r, bias_r, gold)["macro_f1"] * 100

    # --- correlation vs each pool member (decorrelation = the whole point) ---
    corrs = {}
    for m in members:
        mp = []
        for d in docs:
            n = len(d["tokens"])
            if n < 2:
                continue
            mp.append(np.asarray(caches[m][d["doc_id"]], dtype=np.float64)[: n - 1])
        mv = np.concatenate(mp)
        corrs[m] = float(np.corrcoef(rj, mv)[0, 1])
    mean_corr = float(np.mean(list(corrs.values())))

    # --- pool alone vs pool + rmerge (uniform + greedy) ---
    pool_probs = [{d["doc_id"]: np.asarray(caches[m][d["doc_id"]], dtype=np.float64)
                   for d in docs} for m in members]
    pool_blend = blend_dict(docs, pool_probs)
    lam0, bias0 = pick_decode_config(docs, pool_blend, len_lp, gold)
    pool_f1 = score_pool(docs, pool_blend, len_lp, lam0, bias0, gold)["macro_f1"] * 100

    plus_blend = blend_dict(docs, pool_probs + [rmerge])
    plus_f1 = score_pool(docs, plus_blend, len_lp, lam0, bias0, gold)["macro_f1"] * 100

    # greedy over pool + rmerge (does the combiner give rmerge weight?)
    aug = members + ["rmerge"]
    ids = [d["doc_id"] for d in docs]
    spans, V, off = [], [], 0
    for d in docs:
        n = len(d["tokens"])
        cols = [np.asarray(caches[m][d["doc_id"]], dtype=np.float64) for m in members]
        cols.append(rmerge[d["doc_id"]])
        V.append(np.stack(cols, axis=1)); spans.append((off, off + n)); off += n
    V = np.concatenate(V, axis=0)
    gw = omar_greedy_weights(docs, list(range(len(docs))), V, spans, ids, gold, len_lp, lam0, bias0)

    print("=" * 74)
    print(f"RECURSIVE MODEL AS A VOTER  task={args.task}  pool={len(members)} AraBERT-core")
    print("-" * 74)
    print(f"recursive solo:  AUC={solo_auc:.3f}   F1={solo_f1:.2f}   (decode lam={lam_r} bias={bias_r})")
    print(f"mean corr vs pool = {mean_corr:.3f}   "
          f"[{'DECORRELATED (phi<0.65) — the missing kind!' if mean_corr < 0.65 else 'still correlated (>=0.65)'}]")
    for m, c in sorted(corrs.items(), key=lambda kv: kv[1]):
        print(f"     corr(rmerge, {m:18s}) = {c:+.3f}")
    print("-" * 74)
    print(f"pool alone        uniform F1 = {pool_f1:.2f}")
    print(f"pool + recursive  uniform F1 = {plus_f1:.2f}   (delta {plus_f1 - pool_f1:+.2f})")
    print(f"greedy weight the combiner gives the recursive voter = {gw[-1]:.3f}")
    print("-" * 74)
    verdict = ("ADDS VALUE — decorrelated voter helps the pool" if plus_f1 > pool_f1 + 0.02
               else "no ensemble gain (absorbed or too weak)")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
