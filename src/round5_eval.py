"""Round-5 evaluation: slot-preserving seed averaging, per the pre-registered
rule in EXPERIMENTS.md (adopt iff dev >= frozen-0.10 AND test >= frozen-0.10).
Exactly ONE variant per task; no grids, no selection.

Slots (weights identical to frozen configs):
  PA (dp lam=.2 bias=0):        arabert(w4)=mean(6 seeds) arbertv2(w1)=mean(2) araelectra(w1)=mean(2)
  closed NoPnx-NP (adaptive):   same slot structure
  NP (thresh .5):               3 slots x mean(2 seeds), uniform
  NoPnx-PA (thresh .5 +or_par): 4 slots x mean(2 seeds), uniform
  open NoPnx-NP (adaptive):     arabert=mean(6) araelectra=mean(2) opus-ft=mean(2) satft=mean(2), uniform
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import f1_score

from data import PARAGRAPH_TOKEN, load_jsonl
from dp_adaptive import doc_length_logprob
from dp_decode import decode_doc, fit_length_logprob

P = "probs"


def slot_probs(task, split, slots):
    """slots: list of (weight, [cache names]) -> dict doc_id -> weighted slot-mean."""
    caches = {}
    for _, names in slots:
        for n in names:
            f = f"{P}/{task}_{split}_{n}.npz"
            if not os.path.exists(f):
                raise FileNotFoundError(f)
            caches[n] = np.load(f)
    docs = load_jsonl(f"data/{task}_{split}.jsonl")
    out = {}
    wsum = sum(w for w, _ in slots)
    for d in docs:
        did = d["doc_id"]
        acc = None
        for w, names in slots:
            sm = np.mean([caches[n][did] for n in names], axis=0) * w
            acc = sm if acc is None else acc + sm
        out[did] = acc / wsum
    return docs, out


def score_dp(task, docs, probs, lam, bias):
    glp = fit_length_logprob(load_jsonl(f"data/{task}_train.jsonl"))
    return np.array([f1_score(d["labels"], decode_doc(d["tokens"], probs[d["doc_id"]], glp, lam, bias),
                              zero_division=0) * 100 for d in docs])


def score_adaptive(task, docs, probs):
    glp = fit_length_logprob(load_jsonl(f"data/{task}_train.jsonl"))
    out = []
    for d in docs:
        p = probs[d["doc_id"]]
        p1 = decode_doc(d["tokens"], p, glp, 0.4, 0.75)
        dlp = doc_length_logprob(d["tokens"], p1, glp, 1.0)
        out.append(f1_score(d["labels"], decode_doc(d["tokens"], p, dlp, 0.6, 0.75), zero_division=0) * 100)
    return np.array(out)


def score_thresh(docs, probs, or_par):
    out = []
    for d in docs:
        lab = (probs[d["doc_id"]] >= 0.5).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
                if or_par and i > 0 and d["tokens"][i - 1] != PARAGRAPH_TOKEN:
                    lab[i - 1] = 1
        out.append(f1_score(d["labels"], lab, zero_division=0) * 100)
    return np.array(out)


def report(name, frozen_slots, seed_slots, task, scorer, frozen_ref):
    row = {}
    for label, slots in [("frozen", frozen_slots), ("seed-avg", seed_slots)]:
        dd, dp_ = slot_probs(task, "dev", slots)
        td, tp = slot_probs(task, "test", slots)
        row[label] = (scorer(task, dd, dp_).mean() if scorer is not score_thresh
                      else scorer(dd, dp_, frozen_ref).mean(),
                      scorer(task, td, tp).mean() if scorer is not score_thresh
                      else scorer(td, tp, frozen_ref).mean())
    fd, ft = row["frozen"]
    sd, st = row["seed-avg"]
    ok = sd >= fd - 0.10 and st >= ft - 0.10
    print(f"{name}: frozen dev {fd:.2f}/test {ft:.2f}  ->  seed-avg dev {sd:.2f}/test {st:.2f}"
          f"  ({'ADOPT' if ok else 'KEEP FROZEN'})")
    return ok


def main() -> None:
    # PA slots — frozen: 4x arabert seeds + arbertv2 + araelectra, uniform over 6
    pa_frozen = [(4, ["pa-voter", "pa-voter-s1", "pa-voter-s2", "pa-voter-s3"]),
                 (1, ["pa-voter-arbertv2"]), (1, ["pa-voter-araelectra"])]
    pa_seed = [(4, ["pa-voter", "pa-voter-s1", "pa-voter-s2", "pa-voter-s3", "pa-s4", "pa-s5"]),
               (1, ["pa-voter-arbertv2", "pa-arbertv2-s1"]),
               (1, ["pa-voter-araelectra", "pa-araelectra-s1"])]

    def sc_dp(task, docs, probs):
        return score_dp(task, docs, probs, 0.2, 0.0)

    report("PA (dp lam=.2)", pa_frozen, pa_seed, "PA", sc_dp, None)

    nnp_frozen = [(4, ["voter", "voter-s1", "voter-s2", "voter-s3"]),
                  (1, ["voter-arbertv2"]), (1, ["voter-araelectra"])]
    nnp_seed = [(4, ["voter", "voter-s1", "voter-s2", "voter-s3", "nopnx-np-s4", "nopnx-np-s5"]),
                (1, ["voter-arbertv2", "nopnx-np-arbertv2-s1"]),
                (1, ["voter-araelectra", "nopnx-np-araelectra-s1"])]
    report("closed NoPnx-NP (adaptive)", nnp_frozen, nnp_seed, "NoPnx-NP", score_adaptive, None)

    onnp_frozen = [(1, ["voter-s2"]), (1, ["voter-araelectra"]), (1, ["open"]), (1, ["satft"])]
    onnp_seed = [(1, ["voter", "voter-s1", "voter-s2", "voter-s3", "nopnx-np-s4", "nopnx-np-s5"]),
                 (1, ["voter-araelectra", "nopnx-np-araelectra-s1"]),
                 (1, ["open", "nopnx-np-ft-s1"]),
                 (1, ["satft", "satft-s1"])]
    report("open NoPnx-NP (adaptive)", onnp_frozen, onnp_seed, "NoPnx-NP", score_adaptive, None)

    np_frozen = [(1, ["np-voter"]), (1, ["np-voter-arbertv2"]), (1, ["np-voter-araelectra"])]
    np_seed = [(1, ["np-voter", "np-s1"]), (1, ["np-voter-arbertv2", "np-arbertv2-s1"]),
               (1, ["np-voter-araelectra", "np-araelectra-s1"])]

    def sc_th(task, docs, probs):
        return score_thresh(docs, probs, or_par=False)

    report("NP (thresh .5)", np_frozen, np_seed, "NP", sc_th, None)

    npa_frozen = [(1, ["npa-voter"]), (1, ["npa-voter-arbertv2"]),
                  (1, ["npa-voter-araelectra"]), (1, ["mdeberta-pa"])]
    npa_seed = [(1, ["npa-voter", "nopnx-pa-s1"]),
                (1, ["npa-voter-arbertv2", "nopnx-pa-arbertv2-s1"]),
                (1, ["npa-voter-araelectra", "nopnx-pa-araelectra-s1"]),
                (1, ["mdeberta-pa", "nopnx-pa-mdeberta-s1"])]

    def sc_thp(task, docs, probs):
        return score_thresh(docs, probs, or_par=True)

    report("NoPnx-PA (thresh .5 +or_par)", npa_frozen, npa_seed, "NoPnx-PA", sc_thp, None)


if __name__ == "__main__":
    main()
