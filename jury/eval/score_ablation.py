"""Paper ablation: ensemble alone vs ensemble + trained juries, on the released TEST split.

MAIN LOOP ONLY. Test gold is read in memory from the locked zip and never written anywhere.
No agent ever sees this file's inputs or outputs.

  python scratch_exo/papereval/score_ablation.py --track NoPnx-PA [--all]

--all scores every document in the packet (missing verdicts = draft row unchanged).
Default scores only documents that have a jury verdict, which is the honest paired comparison
while a run is still in progress.
"""
import argparse, json, os, sys, zipfile
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("--track", required=True, choices=["NoPnx-NP", "NoPnx-PA", "NP", "PA"])
ap.add_argument("--all", action="store_true")
a = ap.parse_args()

HASNL = a.track in ("NoPnx-PA", "PA")
P = f"scratch_exo/papereval/{a.track}"
Z = zipfile.ZipFile(r"<HELDOUT_ZIP>")

gold = {}
for line in Z.read(f"heldout_data/{a.track}_test.jsonl").decode("utf-8").split("\n"):
    if line.strip():
        d = json.loads(line)
        gold[d["doc_id"]] = d

draft = json.load(open(f"{P}/draft_rows.json", encoding="utf-8"))
rows = draft["rows"]
order = json.load(open(f"{P}/manifest.json", encoding="utf-8"))["order"]


def snap(T, i, w):
    n = len(T)
    if w and 0 <= i < n and T[i] == w:
        return i
    if w:
        near = [k for k in range(max(0, i - 2), min(n, i + 3)) if T[k] == w]
        if near:
            return min(near, key=lambda x: abs(x - i))
    return i if 0 <= i < n else None


def apply_verdict(T, row, v):
    row = row.copy()
    n_rm = n_add = 0
    for b in v.get("remove", []):
        k = snap(T, int(b.get("i", -1)), b.get("w", ""))
        if k is None or row[k] != 1 or k == len(T) - 1:
            continue
        if HASNL and (T[k] in ("<NL>", "\n") or (k + 1 < len(T) and T[k + 1] in ("<NL>", "\n"))):
            continue
        row[k] = 0
        n_rm += 1
    for b in v.get("add", []):
        k = snap(T, int(b.get("i", -1)), b.get("w", ""))
        if k is None or row[k] != 0 or T[k] in ("<NL>", "\n"):
            continue
        row[k] = 1
        n_add += 1
    row[-1] = 1
    if HASNL:
        for i, t in enumerate(T):
            if t == "\n":
                row[i] = 0
                if i > 0 and T[i - 1] != "\n":
                    row[i - 1] = 1
    return row, n_add, n_rm


def f1(g, p):
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    return 100 * 2 * tp / max(2 * tp + fp + fn, 1)


ens, jur, deltas, ids = [], [], [], []
adds = rms = 0
skipped_len = 0
for did in order:
    if did not in gold or did not in rows:
        continue
    vp = f"{P}/exam_out/{did}.json"
    has = os.path.exists(vp)
    if not has and not a.all:
        continue
    T = gold[did]["tokens"]
    g = np.array(gold[did]["labels"])
    base = np.array([int(c) for c in rows[did]])
    if len(base) != len(T):
        skipped_len += 1
        continue
    b2 = base.copy()
    b2[-1] = 1
    if HASNL:
        for i, t in enumerate(T):
            if t == "\n":
                b2[i] = 0
                if i > 0 and T[i - 1] != "\n":
                    b2[i - 1] = 1
    if has:
        v = json.load(open(vp, encoding="utf-8"))
        j, na, nr = apply_verdict(T, b2, v)
        adds += na
        rms += nr
    else:
        j = b2
    ens.append(f1(g, b2))
    jur.append(f1(g, j))
    deltas.append(jur[-1] - ens[-1])
    ids.append(did)

ens, jur, deltas = np.array(ens), np.array(jur), np.array(deltas)
n = len(ens)
print(f"=== {a.track} — ensemble vs ensemble+jury on the released TEST split ===")
print(f"documents scored: {n}" + (f"   (packet has {len(order)})" if n != len(order) else ""))
if skipped_len:
    print(f"  skipped for length mismatch: {skipped_len}")
if n == 0:
    sys.exit(0)
print(f"  ensemble alone   : {ens.mean():.2f}")
print(f"  ensemble + jury  : {jur.mean():.2f}")
print(f"  DELTA            : {jur.mean() - ens.mean():+.2f}")
w = int((deltas > 0).sum()); l = int((deltas < 0).sum()); t = int((deltas == 0).sum())
print(f"  per-document     : {w} improved / {l} degraded / {t} unchanged")
print(f"  jury edits applied: +{adds} boundaries added, -{rms} removed")
try:
    from scipy import stats
    if w + l > 0:
        tt = stats.ttest_rel(jur, ens)
        wl = stats.wilcoxon(jur, ens) if (w + l) > 5 else None
        ci = stats.t.interval(0.95, n - 1, loc=deltas.mean(), scale=stats.sem(deltas)) if deltas.std() > 0 else (0, 0)
        print(f"  paired t         : t={tt.statistic:+.3f}  p={tt.pvalue:.2e}")
        if wl:
            print(f"  Wilcoxon         : p={wl.pvalue:.2e}")
        print(f"  95% CI on delta  : [{ci[0]:+.3f}, {ci[1]:+.3f}]")
except ImportError:
    pass
big = np.argsort(deltas)
print(f"  best  : {ids[big[-1]]} {deltas[big[-1]]:+.1f}   worst : {ids[big[0]]} {deltas[big[0]]:+.1f}")
json.dump(
    {"track": a.track, "n": n, "ensemble": float(ens.mean()), "jury": float(jur.mean()),
     "delta": float(jur.mean() - ens.mean()), "improved": w, "degraded": l, "unchanged": t,
     "adds": adds, "removes": rms},
    open(f"{P}/ablation_result.json", "w"),
)
print(f"  written: {P}/ablation_result.json")
