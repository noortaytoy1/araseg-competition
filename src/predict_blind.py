"""Label-free final predictor for the AraSeg blind test.

Applies the FROZEN, bootstrap-selected config per (track, task) to a blind-test
JSONL ({"doc_id","tokens"} per line — labels NOT required) and writes a ready
submission CSV. Verified to reproduce the open-test outputs exactly.

  # one task:
  python predict_blind.py --track closed --task NP --jsonl blind/NP_test.jsonl --out sub_NP.csv
  # all four tasks of a track at once:
  python predict_blind.py --track closed --all --blind-dir blind --out-dir blind_subs

Then: cp sub_NP.csv prediction && zip prediction.zip prediction  (upload prediction.zip)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baselines import write_submission
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from dp_adaptive import doc_length_logprob
from predict import predict_doc

# Frozen per-(track,task) config. method ∈ {dp, thresh, adaptive}.
def M(low, extra=()):  # model dir list helper
    base = [f"runs/{low}", f"runs/{low}-s1", f"runs/{low}-s2", f"runs/{low}-s3",
            f"runs/{low}-arbertv2", f"runs/{low}-araelectra"]
    return base + list(extra)

ENC3 = lambda low: [f"runs/{low}", f"runs/{low}-arbertv2", f"runs/{low}-araelectra"]

CONFIG = {
    # PA: 6-model mega + DP (lam .2, bias 0); forces doc-end + pre-\n
    ("closed", "PA"): dict(models=M("pa"), method="dp", lam=0.2, bias=0.0),
    ("open",   "PA"): dict(models=M("pa"), method="dp", lam=0.2, bias=0.0),
    # NoPnx-PA & NP: 3-encoder ensemble + threshold (bootstrap-selected, simpler)
    ("closed", "NoPnx-PA"): dict(models=ENC3("nopnx-pa"), method="thresh", thr=0.50, or_par=True),
    ("open",   "NoPnx-PA"): dict(models=ENC3("nopnx-pa"), method="thresh", thr=0.50, or_par=True),
    ("closed", "NP"): dict(models=ENC3("np"), method="thresh", thr=0.50, or_par=False),
    ("open",   "NP"): dict(models=ENC3("np"), method="thresh", thr=0.50, or_par=False),
    # NoPnx-NP: mega + adaptive length prior (closed); + 2 external voters (open)
    ("closed", "NoPnx-NP"): dict(models=M("nopnx-np"), method="adaptive",
                                 lam1=0.4, bias=0.75, lam2=0.6, blend=1.0),
    ("open",   "NoPnx-NP"): dict(models=M("nopnx-np", ("open_runs/nopnx-np-ft",
                                 "open_runs/nopnx-np-mixft")),
                                 method="adaptive", lam1=0.4, bias=0.75, lam2=0.6, blend=1.0),
}


def ensemble_probs(docs, model_dirs, device):
    words = [[MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]] for d in docs]
    sums = [np.zeros(len(d["tokens"])) for d in docs]
    for mdir in model_dirs:
        tok = AutoTokenizer.from_pretrained(mdir)
        mdl = AutoModelForTokenClassification.from_pretrained(mdir).to(device).eval()
        for i, w in enumerate(words):
            sums[i] += predict_doc(w, tok, mdl, device, 180, 60, 512)
        del mdl
        torch.cuda.empty_cache()
    return [s / len(model_dirs) for s in sums]


def decode_one(task, cfg, docs, probs):
    method = cfg["method"]
    if method == "thresh":
        out = []
        for d, p in zip(docs, probs):
            lab = (p >= cfg["thr"]).astype(int)
            for i, w in enumerate(d["tokens"]):
                if w == PARAGRAPH_TOKEN:
                    lab[i] = 0
                    if cfg.get("or_par") and i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                        lab[i - 1] = 1
            out.append(lab.tolist())
        return out
    # dp / adaptive need a train length prior
    train = load_jsonl(f"data/{task}_train.jsonl")
    glp = fit_length_logprob(train)
    if method == "dp":
        return [decode_one_dp(d, p, glp, cfg["lam"], cfg["bias"]) for d, p in zip(docs, probs)]
    # adaptive: two-pass
    out = []
    for d, p in zip(docs, probs):
        pass1 = decode_one_dp(d, p, glp, cfg["lam1"], cfg["bias"])
        dlp = doc_length_logprob(d["tokens"], pass1, glp, cfg["blend"])
        out.append(decode_doc(d["tokens"], p, dlp, cfg["lam2"], cfg["bias"]))
    return out


def decode_one_dp(d, p, glp, lam, bias):
    return decode_doc(d["tokens"], p, glp, lam, bias)


def run_task(track, task, jsonl, out, device):
    cfg = CONFIG[(track, task)]
    docs = load_jsonl(jsonl)
    probs = ensemble_probs(docs, cfg["models"], device)
    preds = decode_one(task, cfg, docs, probs)
    write_submission(docs, preds, out)
    rate = sum(map(sum, preds)) / max(sum(map(len, preds)), 1)
    print(f"[{track}/{task}] {len(docs)} docs, {cfg['method']}, "
          f"{len(cfg['models'])} models, rate={rate:.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=["closed", "open"])
    ap.add_argument("--task", choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--jsonl", help="blind-test JSONL for one task")
    ap.add_argument("--out", help="output CSV for one task")
    ap.add_argument("--all", action="store_true", help="run all 4 tasks")
    ap.add_argument("--blind-dir", help="dir with {task}_test.jsonl (for --all)")
    ap.add_argument("--out-dir", help="output dir (for --all)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.all:
        os.makedirs(args.out_dir, exist_ok=True)
        for t in ["PA", "NoPnx-PA", "NP", "NoPnx-NP"]:
            run_task(args.track, t, f"{args.blind_dir}/{t}_test.jsonl",
                     f"{args.out_dir}/{t}.csv", device)
    else:
        run_task(args.track, args.task, args.jsonl, args.out, device)


if __name__ == "__main__":
    main()
