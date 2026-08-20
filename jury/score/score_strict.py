"""Strict scorer (Noor's rule 2026-08-19): an edit applies ONLY if the jury's index holds exactly the
named token. No snap. Usage: python score_strict.py --track NoPnx-NP --outdir exam_out_rerun [--docs N]
Compares to the published exam_out verdicts on the SAME documents. Gold read in memory from the locked zip."""
import argparse, json, os, zipfile
import numpy as np
from scipy import stats
ap = argparse.ArgumentParser()
ap.add_argument("--track", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--docs", type=int, default=0, help="restrict to the first N docs of manifest order")
a = ap.parse_args()
HASNL = a.track in ("NoPnx-PA", "PA")
P = f"scratch_exo/papereval/{a.track}"
Z = zipfile.ZipFile(r"C:/Users/pc/araseg_heldout_LOCKED.zip")
gold = {}
for l in Z.read(f"heldout_data/{a.track}_test.jsonl").decode("utf-8").split("\n"):
    if l.strip(): d = json.loads(l); gold[d["doc_id"]] = d
rows = json.load(open(f"{P}/draft_rows.json", encoding="utf-8"))["rows"]
order = json.load(open(f"{P}/manifest.json", encoding="utf-8"))["order"]
if a.docs: order = order[:a.docs]
def norm(T, r):
    r = r.copy(); r[-1] = 1
    if HASNL:
        for i, t in enumerate(T):
            if t == "\n":
                r[i] = 0
                if i > 0 and T[i-1] != "\n": r[i-1] = 1
    return r
def apply(T, base, v):
    r = base.copy(); rej = 0; n = 0
    for b in v.get("remove", []):
        i = int(b.get("i", -1)); w = b.get("w", "")
        if not (0 <= i < len(T) and w and T[i] == w): rej += 1; continue
        if r[i] != 1 or i == len(T)-1: continue
        if HASNL and (T[i] == "\n" or (i+1 < len(T) and T[i+1] == "\n")): continue
        r[i] = 0; n += 1
    for b in v.get("add", []):
        i = int(b.get("i", -1)); w = b.get("w", "")
        if not (0 <= i < len(T) and w and T[i] == w): rej += 1; continue
        if r[i] != 0 or T[i] == "\n": continue
        r[i] = 1; n += 1
    return norm(T, r), n, rej
def f1(g, p):
    tp = int(((p==1)&(g==1)).sum()); fp = int(((p==1)&(g==0)).sum()); fn = int(((p==0)&(g==1)).sum())
    return 100*2*tp/max(2*tp+fp+fn, 1)
E, A, B, ids = [], [], [], []; nA = nB = rejA = rejB = 0
for did in order:
    pa = f"{P}/{a.outdir}/{did}.json"; pb = f"{P}/exam_out/{did}.json"
    if not os.path.exists(pa): continue
    T = gold[did]["tokens"]; g = np.array(gold[did]["labels"])
    base = norm(T, np.array([int(c) for c in rows[did]]))
    ja, n1, r1 = apply(T, base, json.load(open(pa, encoding="utf-8"))); nA += n1; rejA += r1
    E.append(f1(g, base)); A.append(f1(g, ja)); ids.append(did)
    if os.path.exists(pb):
        jb, n2, r2 = apply(T, base, json.load(open(pb, encoding="utf-8"))); nB += n2; rejB += r2
        B.append(f1(g, jb))
    else:
        B.append(None)
E, A = np.array(E), np.array(A)
print(f"=== {a.track} / {a.outdir} (strict, exact index+token) n={len(E)} ===")
print(f"  ensemble        : {E.mean():.2f}")
print(f"  {a.outdir:16s}: {A.mean():.2f}   delta vs ensemble {A.mean()-E.mean():+.2f}   edits applied {nA}, rejected (non-exact) {rejA}")
if all(b is not None for b in B):
    B = np.array(B)
    print(f"  doctrine (pub.) : {B.mean():.2f}   delta vs ensemble {B.mean()-E.mean():+.2f}   edits applied {nB}, rejected {rejB}")
    d = B - A
    t = stats.ttest_rel(B, A); w = stats.wilcoxon(B, A) if (d != 0).any() else None
    print(f"  doctrine - {a.outdir}: {d.mean():+.2f}  paired t p={t.pvalue:.2e}" + (f"  Wilcoxon p={w.pvalue:.2e}" if w else "") + f"  | docs where doctrine better/worse/same: {(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()}")
up = int((A-E>0).sum()); dn = int((A-E<0).sum()); sm = int((A-E==0).sum())
print(f"  {a.outdir} vs ensemble per-doc: improved {up} / degraded {dn} / unchanged {sm}")
