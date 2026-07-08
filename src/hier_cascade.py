"""Hierarchical doc->paragraph->sentence cascade helpers (NP / NoPnx-NP).

Three subcommands:
  para-target  : write a paragraph-boundary target jsonl (stage-1 predictor).
  inject       : inject predicted \\n paragraph tokens into the NP data
                 (stage-2 input), so the sentence model sees paragraph structure.
  score        : strip the injected \\n from a stage-2 prediction, re-align to
                 the ORIGINAL NP token stream, and score macro-F1 vs NP gold.

Grounded: paragraph info is worth ~+2.8 F1 (NoPnx-NP->NoPnx-PA) in our pipeline;
this recovers a fraction of it with PREDICTED paragraphs, adding new structural
signal the ensemble+DP does not have. Additive, quarantined (new file).
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_encoder import paragraph_word_labels  # noqa: E402

PARA = "\n"


def _docs(task, split):
    return [json.loads(l) for l in open(
        os.path.join(os.path.dirname(__file__), os.pardir, "data",
                     f"{task}_{split}.jsonl"), encoding="utf-8") if l.strip()]


def _read_pred(path):
    out = {}
    for r in csv.reader(open(path, encoding="utf-8")):
        if not r or r[0] == "Document ID" or len(r) < 2:
            continue
        out[r[0]] = r[1]
    return out


def para_target(task, split, out):
    para = paragraph_word_labels(task, split)
    docs = _docs(task, split)
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            pl = para.get(d["doc_id"])
            if pl is None or len(pl) != len(d["tokens"]):
                pl = [0] * len(d["tokens"])
            f.write(json.dumps({"doc_id": d["doc_id"], "tokens": d["tokens"],
                                "labels": pl}, ensure_ascii=False) + "\n")
    print(f"para-target: {len(docs)} docs -> {out}")


def inject(task, split, pred_csv, out):
    """Insert a \\n token AFTER every word predicted to be a paragraph end."""
    pred = _read_pred(pred_csv)
    docs = _docs(task, split)
    n_ins = 0
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            bits = pred.get(d["doc_id"], "0" * len(d["tokens"]))
            toks, labs = [], []
            for i, (w, l) in enumerate(zip(d["tokens"], d["labels"])):
                toks.append(w); labs.append(l)
                if i < len(bits) and bits[i] == "1":
                    toks.append(PARA); labs.append(0); n_ins += 1
            f.write(json.dumps({"doc_id": d["doc_id"], "tokens": toks,
                                "labels": labs}, ensure_ascii=False) + "\n")
    print(f"inject: {len(docs)} docs, {n_ins} paragraph tokens inserted -> {out}")


def score(task, split, hier_jsonl, pred_csv):
    """Strip injected \\n from the stage-2 prediction, realign to the ORIGINAL
    NP tokens, macro-F1 vs NP gold."""
    import numpy as np
    hier = {d["doc_id"]: d["tokens"] for d in
            (json.loads(l) for l in open(hier_jsonl, encoding="utf-8") if l.strip())}
    pred = _read_pred(pred_csv)
    gold = {d["doc_id"]: d["labels"] for d in _docs(task, split)}
    fs = []
    for did, g in gold.items():
        bits = pred[did]; toks = hier[did]
        # keep only predictions at non-\n positions -> original NP order
        p = [int(bits[i]) for i in range(len(toks)) if toks[i] != PARA]
        g = list(g)
        assert len(p) == len(g), f"{did}: {len(p)} vs {len(g)}"
        p = np.array(p); g = np.array(g)
        tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * P * R / (P + R) if P + R else 0.0)
    print(f"CASCADE {task} {split}: macro-F1 = {100 * float(np.mean(fs)):.2f}  "
          f"({len(fs)} docs)")


def make_folds(jsonl, k, outprefix):
    """Deterministic K-fold split (by doc index) for out-of-fold paragraph
    prediction on TRAIN, so train [PAR] carries the same noise as dev [PAR]
    (avoids the overfit-train-predictions leak)."""
    docs = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    k = int(k)
    for f in range(k):
        rest = [d for i, d in enumerate(docs) if i % k != f]
        held = [d for i, d in enumerate(docs) if i % k == f]
        for name, part in (("train", rest), ("held", held)):
            with open(f"{outprefix}_{name}_{f}.jsonl", "w", encoding="utf-8") as g:
                for d in part:
                    g.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"make-folds: {len(docs)} docs -> {k} folds ({outprefix})")


def concat(out, *csvs):
    """Merge prediction CSVs (one header, then all rows)."""
    rows = {}
    for c in csvs:
        rows.update(_read_pred(c))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Document ID", "Prediction"])
        for did, b in rows.items():
            w.writerow([did, b])
    print(f"concat: {len(rows)} docs -> {out}")


def para_csv(task, split, out):
    """Gold paragraph bitstring CSV (for the ceiling inject)."""
    para = paragraph_word_labels(task, split)
    docs = _docs(task, split)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Document ID", "Prediction"])
        for d in docs:
            pl = para.get(d["doc_id"]) or [0] * len(d["tokens"])
            if len(pl) != len(d["tokens"]):
                pl = [0] * len(d["tokens"])
            w.writerow([d["doc_id"], "".join(map(str, pl))])
    print(f"para-csv: {len(docs)} docs -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "para-target":
        para_target(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "para-csv":
        para_csv(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "make-folds":
        make_folds(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "concat":
        concat(sys.argv[2], *sys.argv[3:])
    elif cmd == "inject":
        inject(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
