"""Threshold-FAIR A/B reporting: F1@0.5 vs F1@own-best, SAME grid for every arm.

Why this exists (audit MAJOR: threshold fairness)
--------------------------------------------------
Every A/B runner scores its arms with predict.py at the hardcoded --threshold 0.5.
But different arms move their calibrated operating point, so a fixed-0.5 firing
point can mis-rank two models whose own-best thresholds differ (a ~0.10 F1 swing
was observed between baseline th~0.56 and connaug-fix th~0.60). This module makes
the fairness explicit and COMMITTED: for each arm it reports

    F1@0.5              (the number the old runners reported)
    F1@own-best         (each arm at its own best threshold on the SAME grid)

using the SAME threshold grid and the SAME `>=` operator as predict.py /
sweep_threshold.py, and the SAME document-level macro metric as eval_local.py.

HONESTY ABOUT own-best
----------------------
Picking the threshold on the very split you score on is OPTIMISTIC (the number is
selected in-sample). Two honest modes are offered:

  * default (--sweep-jsonl == --score-jsonl or --sweep-jsonl omitted): own-best is
    selected AND scored on the same dev split. The report labels this
    "own-best(dev-SELECTED, optimistic)".
  * held-out (--sweep-jsonl a DIFFERENT slice, e.g. NoPnx-NP_devheld.jsonl): the
    threshold is chosen on the sweep slice and applied, unchanged, to the score
    slice. The report labels this "own-best(held-out)" — the honest number for a
    leaderboard claim.

This does NOT change training, predict.py, eval_local.py, or any existing metric.
It only reports; nothing here fires at train time. Inference reuses
predict.predict_doc verbatim, so probabilities match predict.py exactly.

Usage
-----
  # one arm, dev-selected (optimistic) own-best:
  python fair_threshold.py --task NoPnx-NP \
      --score-jsonl data/NoPnx-NP_dev.jsonl \
      --arm baseline=runs/opus_aug_2026-07-08/base_s42 \
      --arm connaug=runs/opus_connaug_fix_2026-07-09/cf_s42

  # honest held-out threshold selection:
  python fair_threshold.py --task NoPnx-NP \
      --sweep-jsonl data/NoPnx-NP_devheld.jsonl \
      --score-jsonl data/NoPnx-NP_dev.jsonl \
      --arm baseline=... --arm connaug=...
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from eval_local import compute_metrics
from predict import predict_doc

# The SAME grid sweep_threshold.py uses. 0.5 is present so F1@0.5 comes from the
# identical operator/metric path as F1@own-best (no separate code for the two).
DEFAULT_GRID = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]


def doc_probs(model_dir, task, jsonl, window, overlap, max_length, device):
    """P(boundary) per word for every doc in `jsonl`, via predict.predict_doc."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()
    docs = load_task(task, "dev", jsonl_path=jsonl)
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        out.append(predict_doc(words, tok, model, device, window, overlap, max_length))
    return docs, out


def labels_at(docs, probs, t):
    """Apply threshold t with the SAME `>=` operator predict.py uses; force [PAR]=0."""
    preds = {}
    for d, p in zip(docs, probs):
        lab = (p >= t).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        preds[d["doc_id"]] = lab.tolist()
    return preds


def f1_at(docs, probs, gold, t):
    return compute_metrics(gold, labels_at(docs, probs, t))["macro_f1"]


def own_best(docs, probs, gold, grid):
    """Return (best_threshold, best_f1) over the grid on this (docs, probs, gold)."""
    best_t, best_f1 = None, -1.0
    for t in grid:
        f1 = f1_at(docs, probs, gold, t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--score-jsonl", required=True,
                    help="labeled dev slice every arm is SCORED on")
    ap.add_argument("--sweep-jsonl", default=None,
                    help="labeled slice the own-best threshold is SELECTED on "
                         "(default: the score slice -> dev-SELECTED/optimistic). "
                         "Pass a held-out slice for the honest number.")
    ap.add_argument("--arm", action="append", default=[], required=True,
                    metavar="NAME=MODEL_DIR",
                    help="repeatable; e.g. --arm baseline=runs/.../base_s42")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = sorted(args.thresholds)
    held_out = args.sweep_jsonl is not None and args.sweep_jsonl != args.score_jsonl
    sweep_jsonl = args.sweep_jsonl or args.score_jsonl

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            ap.error(f"--arm must be NAME=MODEL_DIR, got {spec!r}")
        name, model_dir = spec.split("=", 1)
        arms.append((name, model_dir))

    ob_label = "own-best(held-out)" if held_out else "own-best(dev-SELECTED, optimistic)"
    print(f"=== THRESHOLD-FAIR A/B ({args.task}) ===")
    print(f"  score on : {args.score_jsonl}")
    print(f"  select on: {sweep_jsonl}  [{'held-out' if held_out else 'SAME as score = optimistic'}]")
    print(f"  grid     : {grid}")
    print(f"  {'arm':<16} {'F1@0.5':>8} {'ownT':>6} {ob_label:>34}")

    rows = []
    for name, model_dir in arms:
        # score slice
        s_docs, s_probs = doc_probs(model_dir, args.task, args.score_jsonl,
                                    args.window, args.overlap, args.max_length, device)
        s_gold = {d["doc_id"]: list(d["labels"]) for d in s_docs}
        f1_half = f1_at(s_docs, s_probs, s_gold, 0.5)
        # select threshold on the sweep slice
        if held_out:
            w_docs, w_probs = doc_probs(model_dir, args.task, sweep_jsonl,
                                        args.window, args.overlap, args.max_length, device)
            w_gold = {d["doc_id"]: list(d["labels"]) for d in w_docs}
            t_star, _ = own_best(w_docs, w_probs, w_gold, grid)
            f1_ob = f1_at(s_docs, s_probs, s_gold, t_star)   # apply held-out t to score slice
        else:
            t_star, f1_ob = own_best(s_docs, s_probs, s_gold, grid)
        rows.append((name, f1_half, t_star, f1_ob))
        print(f"  {name:<16} {f1_half*100:8.2f} {t_star:6.2f} {f1_ob*100:34.2f}")

    if len(rows) >= 2:
        b = rows[0]
        print("\n  GAINS vs first arm "
              f"({b[0]}: F1@0.5={b[1]*100:.2f}, {ob_label}={b[3]*100:.2f}):")
        for name, f1_half, t_star, f1_ob in rows[1:]:
            print(f"    {name:<16} @0.5 {(f1_half-b[1])*100:+6.2f}   "
                  f"@own-best {(f1_ob-b[3])*100:+6.2f}")
        print("\n  Note: a +GAIN that only appears @0.5 (and not @own-best) is a "
              "calibration artifact, not a real improvement.")


if __name__ == "__main__":
    main()
