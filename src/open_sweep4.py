"""Re-tune decode params + SaT weight for the NEW 4-voter open NoPnx-NP pool.

The frozen adaptive params (lam1=.4 bias=.75 lam2=.6) were tuned on the 8-voter
mega pool; the 4-voter pool is differently calibrated. Small grid (w x lam1 x
bias, lam2=lam1+.2), dev-argmax, paired-bootstrap CI vs the current config, test
confirmation. Adopt only if dev CI excludes 0 AND test >= 85.17.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from data import load_jsonl
from dp_adaptive import doc_length_logprob
from dp_decode import decode_doc, fit_length_logprob

TASK = "NoPnx-NP"
POOL = ["voter-s2", "voter-araelectra", "open", "satft"]
FROZEN = dict(w=1.0, lam1=0.4, bias=0.75, lam2=0.6)
glp = fit_length_logprob(load_jsonl(f"data/{TASK}_train.jsonl"))
DATA = {}
for split in ["dev", "test"]:
    docs = load_jsonl(f"data/{TASK}_{split}.jsonl")
    DATA[split] = (docs, {m: np.load(f"probs/{TASK}_{split}_{m}.npz") for m in POOL})


def perdoc(split, w, lam1, bias, lam2):
    docs, cache = DATA[split]
    out = []
    for d in docs:
        did = d["doc_id"]
        p = (cache["voter-s2"][did] + cache["voter-araelectra"][did]
             + cache["open"][did] + w * cache["satft"][did]) / (3 + w)
        p1 = decode_doc(d["tokens"], p, glp, lam1, bias)
        dlp = doc_length_logprob(d["tokens"], p1, glp, 1.0)
        pred = decode_doc(d["tokens"], p, dlp, lam2, bias)
        out.append(f1_score(d["labels"], pred, zero_division=0) * 100)
    return np.array(out)


def main() -> None:
    base_dev = perdoc("dev", **FROZEN)
    print(f"current config dev={base_dev.mean():.2f}")
    results = []
    for w in [0.5, 0.75, 1.0, 1.25]:
        for lam1 in [0.2, 0.4, 0.6]:
            for bias in [0.5, 0.75, 1.0]:
                f = perdoc("dev", w=w, lam1=lam1, bias=bias, lam2=lam1 + 0.2)
                results.append((f.mean(), dict(w=w, lam1=lam1, bias=bias, lam2=lam1 + 0.2), f))
    results.sort(key=lambda r: -r[0])
    for m, cfg, _ in results[:5]:
        print(f"  dev {m:.2f}  {cfg}")
    best_m, best_cfg, best_pd = results[0]
    d = best_pd - base_dev
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nbest vs current: dev diff {d.mean():+.2f} CI[{lo:+.2f},{hi:+.2f}]")
    t_new = perdoc("test", **best_cfg).mean()
    t_cur = perdoc("test", **FROZEN).mean()
    print(f"test: best-cfg {t_new:.2f} vs current 85.17 ({t_cur:.2f} recomputed)")
    verdict = "ADOPT" if lo > 0 and t_new >= t_cur else "KEEP CURRENT"
    print(f"VERDICT: {verdict}  cfg={best_cfg}")


if __name__ == "__main__":
    main()
