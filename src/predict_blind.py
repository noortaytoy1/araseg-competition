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

# Round-5 re-freeze (2026-07-05): slot-preserving seed averaging (pre-registered
# rule, EXPERIMENTS.md). Slot weights unchanged from the earlier configs; each
# slot is now the mean of its seeds. "weights" is parallel to "models".
PA10 = ["runs/pa", "runs/pa-s1", "runs/pa-s2", "runs/pa-s3", "runs/pa-s4", "runs/pa-s5",
        "runs/pa-arbertv2", "runs/pa-arbertv2-s1",
        "runs/pa-araelectra", "runs/pa-araelectra-s1"]
PA10_W = [4, 4, 4, 4, 4, 4, 3, 3, 3, 3]  # arabert slot 4/6, others 1/6 each
NPA8 = ["runs/nopnx-pa", "runs/nopnx-pa-s1",
        "runs/nopnx-pa-arbertv2", "runs/nopnx-pa-arbertv2-s1",
        "runs/nopnx-pa-araelectra", "runs/nopnx-pa-araelectra-s1",
        "runs/nopnx-pa-mdeberta", "runs/nopnx-pa-mdeberta-s1"]
NP6 = ["runs/np", "runs/np-s1", "runs/np-arbertv2", "runs/np-arbertv2-s1",
       "runs/np-araelectra", "runs/np-araelectra-s1"]
ONNP = ["runs/nopnx-np", "runs/nopnx-np-s1", "runs/nopnx-np-s2", "runs/nopnx-np-s3",
        "runs/nopnx-np-s4", "runs/nopnx-np-s5",
        "runs/nopnx-np-araelectra", "runs/nopnx-np-araelectra-s1",
        "open_runs/nopnx-np-ft", "open_runs/nopnx-np-ft-s1"]
ONNP_W = [1, 1, 1, 1, 1, 1, 3, 3, 3, 3]  # 4 uniform slots; satft slot adds weight 6

CONFIG = {
    ("closed", "PA"): dict(models=PA10, weights=PA10_W, method="dp", lam=0.2, bias=0.0),
    ("open",   "PA"): dict(models=PA10, weights=PA10_W, method="dp", lam=0.2, bias=0.0),
    ("closed", "NoPnx-PA"): dict(models=NPA8, method="thresh", thr=0.50, or_par=True),
    ("open",   "NoPnx-PA"): dict(models=NPA8, method="thresh", thr=0.50, or_par=True),
    ("closed", "NP"): dict(models=NP6, method="thresh", thr=0.50, or_par=False),
    ("open",   "NP"): dict(models=NP6, method="thresh", thr=0.50, or_par=False),
    # closed NoPnx-NP: round-5 variant breached the dev guard -> ORIGINAL frozen kept
    ("closed", "NoPnx-NP"): dict(models=M("nopnx-np"), method="adaptive",
                                 lam1=0.4, bias=0.75, lam2=0.6, blend=1.0),
    # open NoPnx-NP: 4 uniform slots {arabert(6 seeds), araelectra(2), opus-ft(2),
    # satft(2 states)}. SaT probs: dump BOTH states with satft_blind.py (venv27)
    # and pass --satft-npz a.npz,b.npz (they are averaged into the satft slot).
    ("open",   "NoPnx-NP"): dict(models=ONNP, weights=ONNP_W, satft=True, satft_weight=6,
                                 method="adaptive", lam1=0.4, bias=0.75, lam2=0.6, blend=1.0),
}


def ensemble_probs(docs, model_dirs, device, weights=None):
    words = [[MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]] for d in docs]
    ws = weights or [1.0] * len(model_dirs)
    sums = [np.zeros(len(d["tokens"])) for d in docs]
    for mdir, mw in zip(model_dirs, ws):
        tok = AutoTokenizer.from_pretrained(mdir)
        mdl = AutoModelForTokenClassification.from_pretrained(mdir).to(device).eval()
        for i, w in enumerate(words):
            sums[i] += mw * predict_doc(w, tok, mdl, device, 180, 60, 512)
        del mdl
        torch.cuda.empty_cache()
    return [s / sum(ws) for s in sums]


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


def run_task(track, task, jsonl, out, device, satft_npz=None):
    cfg = CONFIG[(track, task)]
    docs = load_jsonl(jsonl)
    weights = cfg.get("weights")
    probs = ensemble_probs(docs, cfg["models"], device, weights)
    n_vote = len(cfg["models"])
    wsum = sum(weights) if weights else n_vote
    if cfg.get("satft"):
        if not satft_npz:
            raise SystemExit(f"[{track}/{task}] config includes the SaT-ft slot: dump "
                             "each state with satft_blind.py (venv27) and pass "
                             "--satft-npz state1.npz,state2.npz")
        sats = [np.load(p) for p in satft_npz.split(",")]
        sw = cfg.get("satft_weight", 1)
        probs = [(p * wsum + sw * np.mean([s[d["doc_id"]] for s in sats], axis=0))
                 / (wsum + sw) for d, p in zip(docs, probs)]
        n_vote += len(sats)
    preds = decode_one(task, cfg, docs, probs)
    write_submission(docs, preds, out)
    rate = sum(map(sum, preds)) / max(sum(map(len, preds)), 1)
    print(f"[{track}/{task}] {len(docs)} docs, {cfg['method']}, "
          f"{n_vote} voters, rate={rate:.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=["closed", "open"])
    ap.add_argument("--task", choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--jsonl", help="blind-test JSONL for one task")
    ap.add_argument("--out", help="output CSV for one task")
    ap.add_argument("--all", action="store_true", help="run all 4 tasks")
    ap.add_argument("--blind-dir", help="dir with {task}_test.jsonl (for --all)")
    ap.add_argument("--out-dir", help="output dir (for --all)")
    ap.add_argument("--satft-npz", help="SaT-ft voter probs npz (open NoPnx-NP; "
                    "dump with satft_blind.py in venv27)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.all:
        os.makedirs(args.out_dir, exist_ok=True)
        for t in ["PA", "NoPnx-PA", "NP", "NoPnx-NP"]:
            run_task(args.track, t, f"{args.blind_dir}/{t}_test.jsonl",
                     f"{args.out_dir}/{t}.csv", device, satft_npz=args.satft_npz)
    else:
        run_task(args.track, args.task, args.jsonl, args.out, device,
                 satft_npz=args.satft_npz)


if __name__ == "__main__":
    main()
