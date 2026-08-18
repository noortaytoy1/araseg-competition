"""Per-voter diagnostics for an ensemble pool (NoPnx-NP).

For every voter in the pool, report BOTH its individual quality AND its
contribution inside the ensemble, so we can see which models are dead weight,
which are redundant, and whether the pool is MISSING a decorrelated voter:

  solo_AUC   token-level ROC-AUC of that voter's boundary probs vs gold junctions
             (individual boundary-detection quality; high = good on its own)
  solo_F1    that voter decoded ALONE (its own best lam/bias) -> doc macro-F1
  mean_corr  mean Pearson corr of its prob vector with the OTHER voters
             (high = redundant twin; low = diverse / decorrelated)
  greedy_wt  Omar greedy integer-weight this voter earns in the full pool
             (0.0 = the weighted ensemble discards it)
  loo_delta  drop this voter from the uniform pool -> F1 change
             (negative = removing it HURTS => it earns its seat; positive = it HURTS the pool)

Sorted by loo_delta (most valuable first). A voter that is low-AUC, high-corr,
0-weight, and +loo_delta is pure dead weight. If EVERY voter is high-corr, the
pool's problem is that it lacks a decorrelated member (add one, don't stack twins).

  python src/voter_diagnostics.py --task NoPnx-NP --members <comma-separated>
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


def pooled_junction_arrays(docs, caches, member):
    """Concatenate (prob, gold) over all junctions (exclude the forced doc-final
    position) for one voter -> for ROC-AUC."""
    ps, gs = [], []
    c = caches[member]
    for d in docs:
        did = d["doc_id"]
        n = len(d["tokens"])
        if n < 2:
            continue
        v = np.asarray(c[did], dtype=np.float64)[: n - 1]   # junctions 0..n-2
        g = np.asarray(d["labels"], dtype=np.int64)[: n - 1]
        ps.append(v)
        gs.append(g)
    return np.concatenate(ps), np.concatenate(gs)


def solo_f1(docs, caches, member, len_lp, gold, lam, bias):
    """Decode this voter ALONE with the shared pool config (fast; AUC already
    captures per-voter quality, so a per-voter sweep isn't worth the cost)."""
    blend = {d["doc_id"]: np.asarray(caches[member][d["doc_id"]], dtype=np.float64)
             for d in docs}
    return score_pool(docs, blend, len_lp, lam, bias, gold)["macro_f1"] * 100


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--members", required=True)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    len_lp = fit_length_logprob(load_jsonl(f"data/{args.task}_train.jsonl"))
    caches = load_pool(args.task, args.split, members)

    # per-voter pooled junction arrays (for AUC + correlation)
    pooled = {m: pooled_junction_arrays(docs, caches, m) for m in members}
    prob_mat = np.stack([pooled[m][0] for m in members], axis=1)   # (Njunc, M)
    goldvec = pooled[members[0]][1]
    corr = np.corrcoef(prob_mat, rowvar=False)                      # (M, M)

    # full-pool uniform + frozen config for LOO
    full_blend = uniform_blend_by_doc(docs, caches, members)
    lam0, bias0 = pick_decode_config(docs, full_blend, len_lp, gold)
    full_f1 = score_pool(docs, full_blend, len_lp, lam0, bias0, gold)["macro_f1"] * 100

    # Omar greedy weights over the full pool (reuse the audited combiner)
    ids = [d["doc_id"] for d in docs]
    spans, V = [], []
    off = 0
    for d in docs:
        n = len(d["tokens"])
        cols = [np.asarray(caches[m][d["doc_id"]], dtype=np.float64) for m in members]
        V.append(np.stack(cols, axis=1))
        spans.append((off, off + n))
        off += n
    V = np.concatenate(V, axis=0)
    all_idx = list(range(len(docs)))
    gw = omar_greedy_weights(docs, all_idx, V, spans, ids, gold, len_lp, lam0, bias0)

    rows = []
    for i, m in enumerate(members):
        auc = roc_auc_score(goldvec, prob_mat[:, i])
        sf1 = solo_f1(docs, caches, m, len_lp, gold, lam0, bias0)
        mean_corr = (corr[i].sum() - 1.0) / (len(members) - 1) if len(members) > 1 else 0.0
        # LOO: drop m, re-decode uniform with frozen config
        rest = [x for x in members if x != m]
        drop_blend = uniform_blend_by_doc(docs, caches, rest)
        drop_f1 = score_pool(docs, drop_blend, len_lp, lam0, bias0, gold)["macro_f1"] * 100
        rows.append((m, auc, sf1, mean_corr, float(gw[i]), drop_f1 - full_f1))
    rows.sort(key=lambda r: r[5])   # most valuable (most negative loo_delta) first

    print("=" * 92)
    print(f"VOTER DIAGNOSTICS  task={args.task}  pool={len(members)}  "
          f"full-pool uniform F1={full_f1:.2f}  (decode lam={lam0} bias={bias0})")
    print("-" * 92)
    print(f"{'voter':20s} {'solo_AUC':>9s} {'solo_F1':>8s} {'mean_corr':>10s} "
          f"{'greedy_wt':>10s} {'loo_delta':>10s}   {'read':s}")
    for m, auc, sf1, mc, gwt, dl in rows:
        if dl < -0.02:
            read = "earns seat"
        elif dl > 0.02:
            read = "HURTS pool"
        else:
            read = "neutral"
        if gwt <= 0.001:
            read += " / greedy-drops"
        print(f"{m:20s} {auc:9.3f} {sf1:8.2f} {mc:10.3f} {gwt:10.3f} {dl:+10.2f}   {read}")
    print("-" * 92)
    print("loo_delta = F1(pool without voter) - F1(full pool).  negative => voter earns its seat.")
    print(f"mean pairwise corr across pool = {(corr.sum()-len(members))/(len(members)*(len(members)-1)):.3f} "
          f"(high => redundant twins; a decorrelated voter would read LOW here)")


if __name__ == "__main__":
    main()
