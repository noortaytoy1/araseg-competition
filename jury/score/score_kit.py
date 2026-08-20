"""Strict scorer for a replication kit: verdicts in <kit>/out/ against the public gold.

An edit applies ONLY if the stated index holds exactly the stated token (no tolerance).
Reports ensemble (draft) and jury macro F1 over the documents that have verdicts.

Usage:
  python jury/score/score_kit.py --track NoPnx-NP --kit kit_NoPnx-NP \
      --gold data/NoPnx-NP_test.jsonl --draft jury/draft_rows/NoPnx-NP.json
"""
import argparse, json, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["NoPnx-NP", "NoPnx-PA", "NP", "PA"])
    ap.add_argument("--kit", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--draft", required=True)
    a = ap.parse_args()
    hasnl = a.track in ("NoPnx-PA", "PA")
    gold = {}
    for l in open(a.gold, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); gold[d["doc_id"]] = d
    rows = json.load(open(a.draft, encoding="utf-8"))["rows"]

    def norm(T, r):
        r = r.copy(); r[-1] = 1
        if hasnl:
            for i, t in enumerate(T):
                if t == "\n":
                    r[i] = 0
                    if i > 0 and T[i - 1] != "\n": r[i - 1] = 1
        return r

    def f1(g, p):
        tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        return 100 * 2 * tp / max(2 * tp + fp + fn, 1)

    E, J = [], []
    applied = rejected = 0
    outdir = os.path.join(a.kit, "out")
    for fn in sorted(os.listdir(outdir)):
        if not fn.endswith(".json"):
            continue
        did = fn[:-5]
        if did not in gold or did not in rows:
            continue
        T = gold[did]["tokens"]; g = np.array(gold[did]["labels"])
        base = norm(T, np.array([int(c) for c in rows[did]]))
        j = base.copy()
        v = json.load(open(os.path.join(outdir, fn), encoding="utf-8"))
        for kind in ("remove", "add"):
            for b in v.get(kind, []):
                i = int(b.get("i", -1)); w = b.get("w", "")
                if not (0 <= i < len(T) and w and T[i] == w):
                    rejected += 1; continue
                if kind == "remove":
                    if j[i] != 1 or i == len(T) - 1: continue
                    if hasnl and (T[i] == "\n" or (i + 1 < len(T) and T[i + 1] == "\n")): continue
                    j[i] = 0; applied += 1
                else:
                    if j[i] != 0 or T[i] == "\n": continue
                    j[i] = 1; applied += 1
        j = norm(T, j)
        E.append(f1(g, base)); J.append(f1(g, j))
    E, J = np.array(E), np.array(J)
    print(f"{a.track}: n={len(E)} documents with verdicts")
    print(f"  ensemble (draft): {E.mean():.2f}")
    print(f"  + juries        : {J.mean():.2f}   delta {J.mean()-E.mean():+.2f}")
    print(f"  edits applied {applied}, rejected (non-exact) {rejected}")
    d = J - E
    print(f"  per-doc: improved {(d>0).sum()} / degraded {(d<0).sum()} / unchanged {(d==0).sum()}")

if __name__ == "__main__":
    main()
