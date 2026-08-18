"""INSTABILITY DECODE-FILTER for an AraSeg NoPnx single-encoder checkpoint.

Additive, flag-gated, CPU-only. Read-only on runs/ and probs/. Built because
the stability-gate diagnostic (src/diag_stability.py) cleared its
pre-registered build bar: MC-dropout AUC 0.711 >= 0.65.

WHAT IT DOES
  1. From the cached dev probs npz + dev jsonl, take EVERY decision position
     with cached P(boundary) >= --hi (confident fires; "\n" excluded) — no
     sampling, full coverage.
  2. Re-enumerate the EXACT predict.py windows (window=180 overlap=60), keep
     only windows covering >= 1 confident fire, dedupe, and run each ONE time
     as a batched forward of T dropout-active copies (MC-dropout, fixed
     per-window seed derived from (--seed, doc index, window index) so the
     result is independent of processing order). Per position and pass t the
     boundary probability is the mean over covering windows — identical
     aggregation to predict.py. Instability := std over the T passes (ddof=1).
  3. Filter: baseline prediction = cached p >= --threshold ("\n" forced 0);
     for each cutoff, suppress every confident fire whose mc_std >= cutoff.
     One submission CSV per cutoff, scorable by eval_local.py.
  4. Report: per cutoff, macro-P/R/F1 (eval_local.compute_metrics — per-doc,
     macro-averaged) and the macro-F1 delta vs the unfiltered baseline, plus
     gold-side accounting (conf-FPs killed vs conf-TPs lost). Gold labels are
     used ONLY in this report section — the filter itself is gold-free and
     reusable at test time.
  5. Dumps the per-position mc_std to a reusable npz in the probs-cache format
     (key = doc_id, value = float32 array over tokens, NaN where not computed)
     — doubles as a stacking feature later.

PRE-REGISTERED DECISION RULE (this build):
  cutoffs evaluated in the net-positive tail ONLY: 0.26..0.40 step 0.02.
  adopt-candidate  iff  best-cutoff macro-F1 delta >= +0.10 points.
  (Composability with other levers is judged at the ensemble level; the
  ±0.45 floor applies there, not here.)

CHECKPOINTED EXECUTION: the ~thousands of window forwards take >10 min of
CPU, so progress accumulates in a --state pickle. Each invocation processes
pending windows for up to --budget-s seconds, saves state, and exits printing
"PROGRESS m/n". Re-run the same command until it prints "ALL WINDOWS DONE",
at which point it finalizes (npz + CSVs + report). Deterministic: per-window
dropout seeds do not depend on chunk boundaries.

Usage (CPU only — battery owns the GPU):
  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python src/instability_filter.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev data/NoPnx-NP_dev.jsonl \
      --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz \
      --state <scratch>/instab_state.pkl \
      --out-dir stability/instability_filter_2026-07-10 \
      --budget-s 420
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from diag_stability import enumerate_windows
from eval_local import compute_metrics


# --------------------------------------------------------------------------- #
# deterministic problem setup (identical on every invocation)
# --------------------------------------------------------------------------- #
def build_positions(dev, cache, hi):
    """All confident-fire positions, in (doc, word) order.

    Returns (positions, gold, cached_p): positions is a list of (di, i);
    gold and cached_p are aligned float/int arrays.
    """
    positions, gold, cached_p = [], [], []
    for di, d in enumerate(dev):
        p = cache[d["doc_id"]]
        toks, lab = d["tokens"], d["labels"]
        for i in range(len(toks)):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            if p[i] >= hi:
                positions.append((di, i))
                gold.append(int(lab[i]))
                cached_p.append(float(p[i]))
    return positions, np.array(gold), np.array(cached_p)


def window_seed(base_seed, di, wi):
    """Fixed per-window dropout seed, independent of processing order."""
    return base_seed * 1_000_003 + di * 100_000 + wi * 2


# --------------------------------------------------------------------------- #
# checkpoint state
# --------------------------------------------------------------------------- #
def state_fingerprint(args, n_pos):
    return dict(model=args.model, dev=args.dev, probs=args.probs, hi=args.hi,
                passes=args.passes, seed=args.seed, window=args.window,
                overlap=args.overlap, max_length=args.max_length, n_pos=n_pos)


def load_state(path, fp, n_pos, passes):
    if os.path.exists(path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        if st["fingerprint"] != fp:
            raise SystemExit(f"state file {path} was built with different "
                             f"arguments: {st['fingerprint']} != {fp}")
        return st
    return dict(fingerprint=fp,
                done=set(),                                   # {(di, wi)}
                pos_sum=np.zeros((n_pos, passes), dtype=np.float64),
                pos_cnt=np.zeros(n_pos, dtype=np.int32),
                clean_sum=np.zeros(n_pos, dtype=np.float64))


def save_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/union-NoPnx-NP-arabertv02-s42")
    ap.add_argument("--dev", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--probs", default="probs/union-NoPnx-NP-arabertv02-s42_dev.npz")
    ap.add_argument("--hi", type=float, default=0.8,
                    help="confidence floor defining 'confident fire'")
    ap.add_argument("--passes", type=int, default=8, help="MC-dropout passes T")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="baseline decision threshold on the cached probs")
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40],
                    help="pre-registered net-positive tail: 0.26..0.40 step 0.02")
    ap.add_argument("--adopt-bar", type=float, default=0.10,
                    help="adopt-candidate iff best macro-F1 delta >= this (points)")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state", required=True, help="checkpoint pickle path")
    ap.add_argument("--budget-s", type=float, default=420.0,
                    help="stop processing new windows after this many seconds "
                         "(state is saved; re-run to continue). 0 = no limit")
    ap.add_argument("--save-every", type=int, default=150,
                    help="save state every N windows (crash protection)")
    ap.add_argument("--out-dir", default="stability/instability_filter")
    args = ap.parse_args()

    if torch.cuda.is_available():
        raise SystemExit("CPU ONLY: battery owns the GPU. Set CUDA_VISIBLE_DEVICES=-1.")
    device = "cpu"
    t0 = time.time()

    print(f"# INSTABILITY DECODE-FILTER  model={args.model}")
    print(f"# probs={args.probs}  dev={args.dev}")
    print(f"# hi={args.hi}  T={args.passes}  seed={args.seed}  "
          f"threshold={args.threshold}  cutoffs={args.cutoffs}", flush=True)

    dev = load_jsonl(args.dev)
    cache = np.load(args.probs)

    positions, gold, cached_p = build_positions(dev, cache, args.hi)
    n_pos = len(positions)
    pos_index = {k: j for j, k in enumerate(positions)}
    n_fp = int((gold == 0).sum())
    n_tp = int((gold == 1).sum())
    print(f"# dev docs={len(dev)}  confident fires (p>={args.hi}): {n_pos} "
          f"({n_fp} gold-0 / {n_tp} gold-1)", flush=True)

    fp = state_fingerprint(args, n_pos)
    st = load_state(args.state, fp, n_pos, args.passes)

    # ---------------------------------------------------- window enumeration
    # Docs with >= 1 confident fire; per doc, windows covering >= 1 such fire.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    docs_needed = sorted({di for di, _ in positions})
    doc_wanted = defaultdict(set)
    for di, i in positions:
        doc_wanted[di].add(i)
    print(f"# docs with confident fires: {len(docs_needed)}", flush=True)

    # jobs: (di, wi) -> (start, {abs word idx -> token pos}) restricted to fires
    jobs = {}
    pos2wins = defaultdict(list)
    for di in docs_needed:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                 for w in dev[di]["tokens"]]
        wins = enumerate_windows(words, tokenizer, args.window, args.overlap,
                                 args.max_length)
        for wi, (start, coverage) in enumerate(wins):
            hits = {i: coverage[i] for i in doc_wanted[di] if i in coverage}
            if hits:
                jobs[(di, wi)] = (start, hits)
                for i in hits:
                    pos2wins[(di, i)].append((di, wi))
    n_win = len(jobs)
    uncovered = [k for k in positions if not pos2wins[k]]
    print(f"# unique windows covering >=1 confident fire: {n_win}  "
          f"(dedup over {n_pos} positions; uncovered positions: {len(uncovered)})",
          flush=True)

    pending = sorted(k for k in jobs if k not in st["done"])
    print(f"# windows done={n_win - len(pending)}  pending={len(pending)}", flush=True)

    # ---------------------------------------------------- MC-dropout forwards
    if pending:
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
        model.eval()
        t_work = time.time()
        n_done_this_call = 0
        for (di, wi) in pending:
            start, hits = jobs[(di, wi)]
            words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                     for w in dev[di]["tokens"]]
            chunk = words[start:start + args.window]
            enc = tokenizer(chunk, is_split_into_words=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            needed = sorted(hits)
            tokpos = [hits[i] for i in needed]

            # clean eval pass (sanity anchor vs the cache)
            with torch.no_grad():
                pc = torch.softmax(model(**enc).logits[0], dim=-1)[:, 1]

            # MC-dropout: T copies in one batch, dropout active, fixed seed
            model.train()
            torch.manual_seed(window_seed(args.seed, di, wi))
            batch = {k: v.repeat(args.passes, 1) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**batch).logits          # (T, L, 2)
            model.eval()
            pt = torch.softmax(logits, dim=-1)[:, :, 1]  # (T, L)

            for i, tp in zip(needed, tokpos):
                j = pos_index[(di, i)]
                st["pos_sum"][j] += pt[:, tp].numpy()
                st["pos_cnt"][j] += 1
                st["clean_sum"][j] += float(pc[tp])
            st["done"].add((di, wi))
            n_done_this_call += 1

            if n_done_this_call % 50 == 0:
                print(f"#   window {len(st['done'])}/{n_win}  "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
            if n_done_this_call % args.save_every == 0:
                save_state(args.state, st)
            if args.budget_s > 0 and time.time() - t_work > args.budget_s:
                break
        save_state(args.state, st)

    if len(st["done"]) < n_win:
        print(f"PROGRESS {len(st['done'])}/{n_win} windows — state saved to "
              f"{args.state}. Re-run the same command to continue.", flush=True)
        return
    print(f"ALL WINDOWS DONE ({n_win}/{n_win}) — finalizing.", flush=True)

    # ---------------------------------------------------- per-position std
    cnt = np.maximum(st["pos_cnt"], 1)
    p_per_pass = st["pos_sum"] / cnt[:, None]            # (n_pos, T)
    mc_std = p_per_pass.std(axis=1, ddof=1)
    mc_mean = p_per_pass.mean(axis=1)
    clean = st["clean_sum"] / cnt
    covered = st["pos_cnt"] > 0
    mc_std = np.where(covered, mc_std, np.nan)           # never suppress uncovered

    dif = np.abs(clean[covered] - cached_p[covered])
    print(f"# sanity: |clean re-predict - cached| mean={dif.mean():.6f}  "
          f"max={dif.max():.6f}  (over {int(covered.sum())} covered positions)")

    # ---------------------------------------------------- reusable npz dump
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.probs))[0]
    std_by_doc = {}
    for di, d in enumerate(dev):
        std_by_doc[d["doc_id"]] = np.full(len(d["tokens"]), np.nan, dtype=np.float32)
    for j, (di, i) in enumerate(positions):
        std_by_doc[dev[di]["doc_id"]][i] = mc_std[j]
    npz_path = os.path.join(args.out_dir,
                            f"instability_std_{tag}_T{args.passes}_s{args.seed}.npz")
    np.savez_compressed(npz_path, **std_by_doc)
    print(f"# wrote per-position mc_std npz -> {npz_path}")

    # ---------------------------------------------------- filter + evaluate
    gold_by_doc = {d["doc_id"]: list(d["labels"]) for d in dev}

    def base_pred(d):
        p = cache[d["doc_id"]]
        lab = (p >= args.threshold).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        return lab

    def write_csv(preds_by_doc, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("Document ID,Prediction\n")
            for d in dev:
                f.write(f"{d['doc_id']},{''.join(map(str, preds_by_doc[d['doc_id']]))}\n")

    base = {d["doc_id"]: base_pred(d) for d in dev}
    base_csv = os.path.join(args.out_dir, f"dev_unfiltered_t{args.threshold:.2f}.csv")
    write_csv(base, base_csv)
    m0 = compute_metrics(gold_by_doc, {k: list(v) for k, v in base.items()})
    print(f"\n# UNFILTERED baseline (threshold={args.threshold}): "
          f"P={m0['macro_precision']*100:.3f} R={m0['macro_recall']*100:.3f} "
          f"F1={m0['macro_f1']*100:.3f}  -> {base_csv}")

    print(f"\n--- CUTOFF SWEEP (suppress p>={args.hi} fire when mc_std >= cutoff; "
          f"pre-registered tail) ---")
    hdr = (f"  {'cutoff':>7} {'suppressed':>10} {'FP_killed':>9} {'TP_lost':>8} "
           f"{'macroP':>8} {'macroR':>8} {'macroF1':>8} {'dF1':>8} "
           f"{'docs+':>6} {'docs-':>6}")
    print(hdr)
    results = []
    for c in args.cutoffs:
        supp = covered & (np.nan_to_num(mc_std, nan=-1.0) >= c)
        pred = {k: v.copy() for k, v in base.items()}
        for j in np.flatnonzero(supp):
            di, i = positions[j]
            pred[dev[di]["doc_id"]][i] = 0
        csv_path = os.path.join(args.out_dir, f"dev_filtered_std{c:.2f}.csv")
        write_csv(pred, csv_path)
        m = compute_metrics(gold_by_doc, {k: list(v) for k, v in pred.items()})
        d_f1 = (m["macro_f1"] - m0["macro_f1"]) * 100
        docs_up = sum(1 for k in m["per_doc"]
                      if m["per_doc"][k][2] > m0["per_doc"][k][2] + 1e-12)
        docs_dn = sum(1 for k in m["per_doc"]
                      if m["per_doc"][k][2] < m0["per_doc"][k][2] - 1e-12)
        fp_kill = int((supp & (gold == 0)).sum())
        tp_lost = int((supp & (gold == 1)).sum())
        results.append(dict(cutoff=c, suppressed=int(supp.sum()),
                            fp_killed=fp_kill, tp_lost=tp_lost,
                            macro_p=m["macro_precision"] * 100,
                            macro_r=m["macro_recall"] * 100,
                            macro_f1=m["macro_f1"] * 100, delta_f1=d_f1,
                            docs_improved=docs_up, docs_hurt=docs_dn,
                            csv=csv_path))
        print(f"  {c:7.2f} {int(supp.sum()):10d} {fp_kill:9d} {tp_lost:8d} "
              f"{m['macro_precision']*100:8.3f} {m['macro_recall']*100:8.3f} "
              f"{m['macro_f1']*100:8.3f} {d_f1:+8.3f} {docs_up:6d} {docs_dn:6d}",
              flush=True)

    # ---------------------------------------------------- decision
    best = max(results, key=lambda r: r["delta_f1"])
    print(f"\n--- PRE-REGISTERED DECISION RULE (this build) ---")
    print(f"best cutoff = {best['cutoff']:.2f}  macro-F1 delta = "
          f"{best['delta_f1']:+.3f} points  (bar: >= +{args.adopt_bar:.2f})")
    if best["delta_f1"] >= args.adopt_bar:
        verdict = "ADOPT-CANDIDATE"
        print("VERDICT: ADOPT-CANDIDATE — clears the +0.10 bar; final adoption "
              "vs the ±0.45 floor is judged at the ensemble level.")
    else:
        verdict = "NO-ADOPT"
        print("VERDICT: NO-ADOPT — best cutoff does not clear the +0.10 bar.")

    report = dict(args=vars(args), n_pos=n_pos, n_fp=n_fp, n_tp=n_tp,
                  n_windows=n_win, uncovered=len(uncovered),
                  sanity_mean_absdiff=float(dif.mean()),
                  sanity_max_absdiff=float(dif.max()),
                  baseline=dict(macro_p=m0["macro_precision"] * 100,
                                macro_r=m0["macro_recall"] * 100,
                                macro_f1=m0["macro_f1"] * 100, csv=base_csv),
                  sweep=results, best_cutoff=best["cutoff"],
                  best_delta_f1=best["delta_f1"], verdict=verdict,
                  npz=npz_path)
    rep_path = os.path.join(args.out_dir, "report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"# wrote {rep_path}")
    print(f"\n# done in {time.time() - t0:.0f}s this call "
          f"({n_win} windows x {args.passes} passes total, CPU)")


if __name__ == "__main__":
    main()
