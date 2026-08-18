"""Adversarial audit: run the checks an organizer or reviewer could plausibly run.

Each test states in advance what a GUILTY system would look like and what an INNOCENT one
looks like, then reports which pattern the evidence matches. No test is designed to pass.

  python paper/adversarial_audit.py
"""
import io, json, os, re, sys, zipfile
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
from data import load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from dp_adaptive import doc_length_logprob

Z = zipfile.ZipFile(r"C:/Users/pc/araseg_heldout_LOCKED.zip")
V = ["imp_fgm", "nat_arabertv02_base", "xlmr-2e5", "boost_supcon01", "nat_asafaya_large"]
verdicts = []

def f1(g, p):
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum()); fn = int(((p == 0) & (g == 1)).sum())
    return 100 * 2 * tp / max(2 * tp + fp + fn, 1)

def zgold(track, split):
    out = {}
    for l in Z.read(f"heldout_data/{track}_{split}.jsonl").decode("utf-8").split("\n"):
        if l.strip():
            d = json.loads(l); out[d["doc_id"]] = d
    return out

def decode_all(track, docs, probs, hasnl):
    GLP = fit_length_logprob(load_jsonl(f"data/{track}_train.jsonl"))
    rows = {}
    for did, d in docs.items():
        if did not in probs: continue
        T = d["tokens"]; em = probs[did]
        p1 = decode_doc(T, em, GLP, lam=0.4, bias=0.75)
        lp2 = doc_length_logprob(T, p1, GLP, blend=0.7)
        r = np.array(decode_doc(T, em, lp2, lam=0.4, bias=0.75)); r[-1] = 1
        if hasnl:
            for i, t in enumerate(T):
                if t == "\n":
                    r[i] = 0
                    if i > 0 and T[i - 1] != "\n": r[i - 1] = 1
        rows[did] = r
    return rows

# =============================================================== TEST 1
print("=" * 70)
print("TEST 1  Can the submitted predictions be reproduced from the model alone?")
print("  GUILTY: rows contain hand edits or gold-derived corrections and will not reproduce.")
print("  INNOCENT: rows reproduce exactly from cached probabilities plus the published decode.")
for track, pref in [("NoPnx-NP", "nonp"), ("NoPnx-PA", "nopa")]:
    try:
        P = {v: dict(np.load(f"probs/blind/{pref}_{v}_blind.npz")) for v in V}
    except Exception as e:
        print(f"  {track}: caches unavailable ({type(e).__name__}), skipped"); continue
    bl = {d["doc_id"]: d for d in load_jsonl(f"data/{track}_blind.jsonl")}
    em = {did: np.mean([P[v][did] for v in V], 0) for did in bl if all(did in P[v] for v in V)}
    rows = decode_all(track, bl, em, hasnl=(track == "NoPnx-PA"))
    raw = open(f"subs_blind/prediction_{track}", "rb").read().decode("utf-8").replace("\r\n", "\n").strip().split("\n")
    sub = {}
    for l in raw[1:]:
        if l:
            did, s = l.split(","); sub[did] = np.array([int(c) for c in s])
    exact = sum(1 for d in sub if d in rows and len(rows[d]) == len(sub[d]) and (rows[d] == sub[d]).all())
    ok = exact == len(sub)
    print(f"  {track}: {exact}/{len(sub)} rows reproduced exactly  -> {'INNOCENT' if ok else 'ANOMALY'}")
    verdicts.append(("reproducibility " + track, ok))

# =============================================================== TEST 2
print("\n" + "=" * 70)
print("TEST 2  Memorisation profile across splits (submitted 5-voter system).")
print("  GUILTY: held-out splits score close to training fit, or far above the blind set.")
print("  INNOCENT: held-out scores sit below training fit and near the blind score.")
BLIND_REPORTED = {"NoPnx-NP": 89.1, "NoPnx-PA": 89.70}
for track, pref in [("NoPnx-NP", "nonp"), ("NoPnx-PA", "nopa")]:
    hasnl = track == "NoPnx-PA"
    line = {}
    for split in ["dev", "test"]:
        names = [n for n in Z.namelist() if n.endswith(f"/{'pa_' if hasnl else ''}{V[0]}_{split}.npz")]
        try:
            P = {}
            for v in V:
                c = [n for n in Z.namelist() if n.endswith(f"/{'pa_' if hasnl else ''}{v}_{split}.npz")]
                P[v] = dict(np.load(io.BytesIO(Z.read(c[0]))))
        except Exception:
            line[split] = None; continue
        g = zgold(track, split)
        em = {did: np.mean([P[v][did] for v in V], 0) for did in g if all(did in P[v] for v in V)}
        rows = decode_all(track, g, em, hasnl)
        line[split] = float(np.mean([f1(np.array(g[d]["labels"]), rows[d]) for d in rows]))
    dv, te = line.get("dev"), line.get("test")
    bl = BLIND_REPORTED[track]
    if dv is None or te is None:
        print(f"  {track}: caches unavailable, skipped"); continue
    ok = (te <= bl + 1.0)
    print(f"  {track}: dev {dv:.2f} | test {te:.2f} | blind (reported) {bl:.2f}")
    print(f"     held-out is {'BELOW' if te < bl else 'ABOVE'} blind by {abs(te-bl):.2f} -> {'INNOCENT' if ok else 'ANOMALY'}")
    verdicts.append(("memorisation profile " + track, ok))

# =============================================================== TEST 3
print("\n" + "=" * 70)
print("TEST 3  Perfect-document rate on the held-out test split.")
print("  GUILTY: a system holding test gold reproduces many documents exactly.")
print("  INNOCENT: perfect documents are rare and concentrated in trivially short texts.")
for track in ["NoPnx-NP", "NoPnx-PA"]:
    P = f"scratch_exo/papereval/{track}"
    g = zgold(track, "test")
    rows = json.load(open(f"{P}/draft_rows.json", encoding="utf-8"))["rows"]
    sc = []
    for did, s in rows.items():
        if did not in g: continue
        gg = np.array(g[did]["labels"]); r = np.array([int(c) for c in s])
        if len(gg) != len(r): continue
        sc.append(f1(gg, r))
    sc = np.array(sc)
    perfect = int((sc >= 99.99).sum())
    ok = perfect / len(sc) < 0.15
    print(f"  {track}: {perfect}/{len(sc)} documents at F1 100 ({100*perfect/len(sc):.1f}%), median {np.median(sc):.1f}"
          f"  -> {'INNOCENT' if ok else 'ANOMALY'}")
    verdicts.append(("perfect-doc rate " + track, ok))

# =============================================================== TEST 4
print("\n" + "=" * 70)
print("TEST 4  Do the learned policies contain text lifted from held-out documents?")
print("  GUILTY: word sequences appear in a doctrine that occur in test/dev but never in train.")
print("  INNOCENT: no such sequence exists.")
def ngrams(toks, n=8):
    return {" ".join(toks[i:i+n]) for i in range(max(0, len(toks) - n + 1))}
train_ng = set()
for t in ["NoPnx-NP", "NoPnx-PA"]:
    for l in open(f"data/{t}_train.jsonl", encoding="utf-8"):
        if l.strip(): train_ng |= ngrams(json.loads(l)["tokens"])
held_only = set()
for t in ["NoPnx-NP", "NoPnx-PA"]:
    for sp in ["dev", "test"]:
        for d in zgold(t, sp).values():
            held_only |= (ngrams(d["tokens"]) - train_ng)
print(f"  built {len(held_only)} 8-grams unique to dev/test (absent from train)")
docs = ["scratch_exo/retrain4/nopa/doctrine_j0.md", "scratch_exo/retrain4/nopa/doctrine_j1.md",
        "scratch_exo/retrain3/nonp/doctrine_j0.md", "scratch_exo/retrain3/nonp/doctrine_j1.md"]
tot_hits = 0
for f in docs:
    txt = open(f, encoding="utf-8", errors="replace").read()
    toks = txt.split()
    hits = [g for g in ngrams(toks) if g in held_only]
    tot_hits += len(hits)
    print(f"  {os.path.basename(f)}: {len(hits)} held-out-only 8-grams" + (f"  e.g. {hits[0][:60]}" if hits else ""))
ok = tot_hits == 0
print(f"  total {tot_hits} -> {'INNOCENT' if ok else 'ANOMALY'}")
verdicts.append(("no held-out text in policies", ok))

# =============================================================== TEST 5
print("\n" + "=" * 70)
print("TEST 5  Does the blind evaluation gold exist anywhere in this project?")
print("  GUILTY: any file carrying labels for blind documents.")
print("  INNOCENT: blind files carry tokens only, and no blind gold exists in the archive.")
bad = 0
for t in ["NoPnx-NP", "NoPnx-PA", "NP", "PA"]:
    for l in open(f"data/{t}_blind.jsonl", encoding="utf-8"):
        if l.strip() and "labels" in json.loads(l): bad += 1
inzip = [n for n in Z.namelist() if "blind" in n.lower()]
ok = bad == 0 and len(inzip) == 0
print(f"  blind docs carrying labels: {bad} | blind entries in locked archive: {len(inzip)} -> {'INNOCENT' if ok else 'ANOMALY'}")
verdicts.append(("no blind gold on disk", ok))

# =============================================================== SUMMARY
print("\n" + "=" * 70)
fails = [v for v in verdicts if not v[1]]
for name, ok in verdicts:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{len(verdicts)-len(fails)}/{len(verdicts)} adversarial tests clean")
sys.exit(1 if fails else 0)
