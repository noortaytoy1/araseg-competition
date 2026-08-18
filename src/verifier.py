"""Precision verifier (closed-track): suppress OVER-FIRED sentence boundaries
whose two adjacent spans are still semantically RELATED.

Decorrelated + closed-track-legal: uses a PRETRAINED multilingual sentence
encoder (LaBSE) for inference only -- no training on any data. For each boundary
the frozen ensemble already fired, embed the span to its left and the span to
its right; if their cosine similarity is high (the spans belong together), the
boundary is a spurious fire -> flip 1->0. Post-hoc: touches no training, no
eval/stitching; sweeps the suppression threshold on dev.
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
MODEL = "sentence-transformers/LaBSE"   # multilingual, strong on Arabic; pretrained (inference only)
TASK = "NoPnx-NP"
DEV = os.path.join(EX, "data", f"{TASK}_dev.jsonl")
POOL = os.path.join(EX, "probs", f"union-{TASK}-POOL_dev.npz")


def macro_f1(pred, gold):
    fs = []
    for d in gold:
        g = np.array(gold[d]); p = np.array(pred[d])
        tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * P * R / (P + R) if P + R else 0.0)
    return float(np.mean(fs))


def best_threshold(prob, gold):
    best, bt = -1, 0.5
    for t in np.linspace(0.2, 0.9, 60):
        f = macro_f1({d: (np.array(prob[d]) >= t).astype(int) for d in gold}, gold)
        if f > best:
            best, bt = f, t
    return bt, best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = {d["doc_id"]: d["tokens"] for d in
            (json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip())}
    gold = {d["doc_id"]: d["labels"] for d in
            (json.loads(l) for l in open(DEV, encoding="utf-8") if l.strip())}
    z = np.load(POOL, allow_pickle=True)
    prob = {k: np.array(z[k], dtype=float) for k in z.files}

    th, f0 = best_threshold(prob, gold)
    base = {d: (prob[d] >= th).astype(int) for d in gold}
    # baseline P/R
    tp = sum(int(((base[d] == 1) & (np.array(gold[d]) == 1)).sum()) for d in gold)
    fp = sum(int(((base[d] == 1) & (np.array(gold[d]) == 0)).sum()) for d in gold)
    fn = sum(int(((base[d] == 0) & (np.array(gold[d]) == 1)).sum()) for d in gold)
    print(f"baseline pool: th={th:.3f} macro-F1={f0*100:.2f}  P={100*tp/(tp+fp):.1f} R={100*tp/(tp+fn):.1f}  ({fp} false fires)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).to(device).eval()

    @torch.no_grad()
    def embed(texts):
        out = []
        for i in range(0, len(texts), 128):
            b = tok(texts[i:i+128], padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, enc.config.hidden_size)

    # collect the relatedness of every fired boundary, span window W=6
    W = 6
    cand = []          # (doc_id, idx)
    left, right = [], []
    for d in gold:
        toks = docs[d]; fired = np.where(base[d] == 1)[0]
        for i in fired:
            l = " ".join(toks[max(0, i - W + 1):i + 1])
            r = " ".join(toks[i + 1:i + 1 + W])
            if not r.strip():          # doc-final fire: keep (never suppress)
                continue
            cand.append((d, int(i))); left.append(l); right.append(r)
    vl = embed(left); vr = embed(right)
    rel = (vl * vr).sum(-1).numpy()    # cosine (both normalized)
    print(f"scored {len(cand)} fired boundaries with LaBSE relatedness "
          f"(mean {rel.mean():.3f})")

    # sweep suppression threshold tau: kill fires with relatedness > tau
    relmap = {}
    for (d, i), r in zip(cand, rel):
        relmap.setdefault(d, {})[i] = r
    print(f"\n{'tau':>5} {'kills':>6} {'macroF1':>8} {'dP':>6} {'dR':>6}")
    best = (f0, None)
    for tau in [1.01, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
        sup = {d: base[d].copy() for d in gold}
        kills = 0
        for d in gold:
            for i, r in relmap.get(d, {}).items():
                if r > tau:
                    sup[d][i] = 0; kills += 1
        f = macro_f1(sup, gold)
        tp = sum(int(((sup[d] == 1) & (np.array(gold[d]) == 1)).sum()) for d in gold)
        fp = sum(int(((sup[d] == 1) & (np.array(gold[d]) == 0)).sum()) for d in gold)
        fn = sum(int(((sup[d] == 0) & (np.array(gold[d]) == 1)).sum()) for d in gold)
        print(f"{tau:>5.2f} {kills:>6d} {f*100:>8.2f} {100*tp/(tp+fp):>6.1f} {100*tp/(tp+fn):>6.1f}")
        if f > best[0]:
            best = (f, tau)
    print(f"\nBEST: macro-F1 {best[0]*100:.2f} at tau={best[1]}  (baseline {f0*100:.2f}, "
          f"gain {100*(best[0]-f0):+.2f}; noise floor +-0.45)")


if __name__ == "__main__":
    main()
