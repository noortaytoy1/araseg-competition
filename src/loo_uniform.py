"""Leave-one-out contribution diagnostic for the uniform ensemble (NoPnx-NP).

For a given pool of encoder voters, freezes the DP decode config (lam, bias)
ONCE on the FULL-pool uniform blend (moe_metalearner.pick_decode_config, its own
grid), then measures each voter's marginal value: recompute the uniform blend
over the pool WITHOUT that voter, decode with the SAME frozen config, score.

    drop_delta(v) = F1(pool \\ {v}) - F1(pool)

A NEGATIVE delta means removing v HURT -> v was pulling its weight (diverse,
error-correcting). A POSITIVE delta means removing v HELPED -> v was dead weight
or actively harmful. This is the same diagnostic used to prune the inbred pool;
here it ranks the diverse+domain voters so we keep only the ones that earn a seat.

Uniform (not greedy/meta) is used on purpose: no per-fold fitting, so the delta
isolates the voter's raw contribution, not a re-tuned combiner. Config is held
fixed across all drops so the delta reflects the pool change alone.

  python src/loo_uniform.py --task NoPnx-NP \
      --members arabertv02-s1,araelectra-s42,arbertv2-s42,camelbert-s42,qarib-s42,camelbertca-s42
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from data import load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from eval_local import compute_metrics
from moe_metalearner import pick_decode_config


def load_pool(task: str, split: str, members, probs_dir: str = "probs"):
    caches = {}
    for m in members:
        path = os.path.join(probs_dir, f"union-{task}-{m}_{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"voter cache missing: {path}")
        caches[m] = np.load(path)
    return caches


def uniform_blend_by_doc(docs, caches, members):
    """dict doc_id -> (N_tokens,) mean of the given members' boundary probs.

    Asserts every member's cache aligns per-token with the doc (crashes loud on a
    length mismatch rather than silently averaging misaligned probs)."""
    out = {}
    for d in docs:
        did = d["doc_id"]
        n = len(d["tokens"])
        cols = []
        for m in members:
            c = caches[m]
            if did not in c.files:
                raise KeyError(f"{did} absent from voter cache {m}")
            v = np.asarray(c[did], dtype=np.float64)
            if v.shape[0] != n:
                raise ValueError(f"{m}:{did} len {v.shape[0]} != {n} (misaligned cache)")
            cols.append(v)
        out[did] = np.mean(np.stack(cols, axis=1), axis=1)
    return out


def score_pool(docs, blend_by_doc, len_lp, lam, bias, gold):
    preds = {d["doc_id"]: decode_doc(d["tokens"], blend_by_doc[d["doc_id"]],
                                     len_lp, lam, bias) for d in docs}
    return compute_metrics(gold, preds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--members", required=True,
                    help="comma-separated voter member names")
    args = ap.parse_args()

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    docs = load_jsonl(f"data/{args.task}_{args.split}.jsonl")
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    len_lp = fit_length_logprob(load_jsonl(f"data/{args.task}_train.jsonl"))
    caches = load_pool(args.task, args.split, members)

    # freeze (lam, bias) on the FULL-pool uniform blend, then hold fixed for all drops
    full_blend = uniform_blend_by_doc(docs, caches, members)
    lam, bias = pick_decode_config(docs, full_blend, len_lp, gold)  # returns a 2-tuple
    full = score_pool(docs, full_blend, len_lp, lam, bias, gold)
    full_f1 = full["macro_f1"] * 100

    rows = []
    for m in members:
        remaining = [x for x in members if x != m]
        blend = uniform_blend_by_doc(docs, caches, remaining)
        s = score_pool(docs, blend, len_lp, lam, bias, gold)
        f1 = s["macro_f1"] * 100
        rows.append((m, f1, f1 - full_f1))  # delta = drop - full; negative => helps
    rows.sort(key=lambda r: r[2])  # most-helpful (most negative delta) first

    print("=" * 68)
    print(f"LEAVE-ONE-OUT (uniform)  task={args.task}  pool={len(members)} voters")
    print(f"frozen DP: lam={lam}  bias={bias}   full-pool uniform F1 = {full_f1:.2f}")
    print("-" * 68)
    print(f"{'drop this voter':22s} {'F1 without it':>14s} {'delta':>9s}   verdict")
    for m, f1, dl in rows:
        verdict = "HELPS (keep)" if dl < -0.02 else ("HURTS (drop)" if dl > 0.02 else "neutral")
        print(f"{m:22s} {f1:14.2f} {dl:+9.2f}   {verdict}")
    print("-" * 68)
    print("delta = F1(pool without voter) - F1(full pool).  negative => voter earns its seat.")


if __name__ == "__main__":
    main()
