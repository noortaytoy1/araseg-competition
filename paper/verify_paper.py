"""Re-derive every number in the paper from source data and check it against the .tex.

Run:  python paper/verify_paper.py
Exit code 0 = every claim verified. Non-zero = at least one FAIL.

Nothing here is copied from the draft. Each value is recomputed from the corpus, the
cached probabilities, the jury verdicts, or the training code, and only then compared to
what the paper says. Test/dev gold is read in memory from the locked archive and is never
written to disk.
"""
import io, json, os, re, sys, zipfile
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

TEX = "paper/ensar_araseg.tex"
ZIP = r"C:/Users/pc/araseg_heldout_LOCKED.zip"
tex = open(TEX, encoding="utf-8").read()
Z = zipfile.ZipFile(ZIP)

results = []
def check(label, claimed, actual, tol=0.005):
    if isinstance(actual, float) and isinstance(claimed, float):
        ok = abs(claimed - actual) <= tol
    else:
        ok = claimed == actual
    results.append((ok, label, claimed, actual))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: paper={claimed}  recomputed={actual}")

def in_tex(s):
    return s in tex

def note(label, ok, detail=""):
    results.append((ok, label, "", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {detail}")

# ---------------------------------------------------------------- 1. corpus sizes
print("\n=== 1. Corpus sizes ===")
for t, blind_n in [("NoPnx-NP", 212), ("NoPnx-PA", 100), ("NP", 212), ("PA", 100)]:
    tr = sum(1 for l in open(f"data/{t}_train.jsonl", encoding="utf-8") if l.strip())
    bl = sum(1 for l in open(f"data/{t}_blind.jsonl", encoding="utf-8") if l.strip())
    dv = sum(1 for l in Z.read(f"heldout_data/{t}_dev.jsonl").decode("utf-8").split("\n") if l.strip())
    te = sum(1 for l in Z.read(f"heldout_data/{t}_test.jsonl").decode("utf-8").split("\n") if l.strip())
    check(f"{t} train=174", 174, tr)
    check(f"{t} dev=222", 222, dv)
    check(f"{t} test=262", 262, te)
    check(f"{t} blind={blind_n}", blind_n, bl)

# ---------------------------------------------------------------- 2. class balance
print("\n=== 2. Boundary rates and class weight ===")
claimed_rate = {"NoPnx-NP": 0.104, "NoPnx-PA": 0.101, "NP": 0.086, "PA": 0.083}
claimed_tok = {"NoPnx-NP": 98457, "NoPnx-PA": 101784, "NP": 124444, "PA": 128173}
for t in claimed_rate:
    pos = tot = 0
    for l in open(f"data/{t}_train.jsonl", encoding="utf-8"):
        if l.strip():
            d = json.loads(l); pos += sum(d["labels"]); tot += len(d["labels"])
    check(f"{t} boundary rate", claimed_rate[t], round(pos / tot, 3), tol=0.0006)
    check(f"{t} train tokens", claimed_tok[t], tot)
    check(f"{t} pos_weight cap 8", 8.0, round(min((tot - pos) / pos, 8.0), 2))

# ---------------------------------------------------------------- 3. training config
print("\n=== 3. Training hyperparameters (from code, not memory) ===")
src = open("src/train_encoder.py", encoding="utf-8").read()
w = re.search(r'"--window".*?default=(\d+)', src, re.S)
s = re.search(r'"--stride".*?default=(\d+)', src, re.S)
ml = re.search(r'"--max-length".*?default=(\d+)', src, re.S)
check("window default 180", 180, int(w.group(1)))
check("stride default 90", 90, int(s.group(1)))
check("max-length default 512", 512, int(ml.group(1)))
sh = open("scratch_exo/train_pnx_voters.sh", encoding="utf-8").read()
note("no --window override in launch script", "--window" not in sh)
check("epochs 10", 10, int(re.search(r"--epochs (\d+)", sh).group(1)))
note("imp_fgm = bert-large-arabertv02 + fgm 1.0",
     "imp_fgm" in sh and "aubmindlab/bert-large-arabertv02 --fgm-eps 1.0" in sh)
note("boost_supcon01 = bert-large-arabertv02 + supcon 0.1",
     "aubmindlab/bert-large-arabertv02 --supcon-weight 0.1" in sh)
note("xlmr-2e5 = xlm-roberta-large lr 2e-5", "FacebookAI/xlm-roberta-large --lr 2e-5" in sh)
note("nat_asafaya_large = asafaya/bert-large-arabic", "asafaya/bert-large-arabic" in sh)
note("nat_arabertv02_base = bert-base-arabertv02", "aubmindlab/bert-base-arabertv02" in sh)
note("class-weighted CE, weight=min(neg/pos,8)",
     "min((tot - pos) / max(pos, 1), 8.0)" in src and "weight=torch.tensor([1.0, pos_weight])" in src)

# ---------------------------------------------------------------- 4. jury training coverage
print("\n=== 4. Jury training coverage (NoPnx-PA) ===")
sp = json.load(open("scratch_exo/retrain3/nopa/allsplit.json", encoding="utf-8"))
j0 = [d for c in sp["0"] for d in c]; j1 = [d for c in sp["1"] for d in c]
train_ids = {json.loads(l)["doc_id"] for l in open("data/NoPnx-PA_train.jsonl", encoding="utf-8") if l.strip()}
check("docs per juror 87", 87, len(j0))
check("docs per juror 87 (j1)", 87, len(j1))
check("union covers train corpus", 174, len(set(j0) | set(j1)))
check("juror overlap 0", 0, len(set(j0) & set(j1)))
note("union == train split exactly", (set(j0) | set(j1)) == train_ids)
for j in ("0", "1"):
    n = len(os.listdir(f"scratch_exo/retrain4/nopa/ans/j{j}"))
    check(f"juror {j} answers banked = 87", 87, n)
for j, claimed in (("0", 1398), ("1", 2098)):
    n = sum(1 for _ in open(f"scratch_exo/retrain4/nopa/doctrine_j{j}.md", encoding="utf-8"))
    check(f"doctrine_j{j} lines", claimed, n)

# ---------------------------------------------------------------- 5. ablation
print("\n=== 5. Ablation, recomputed end to end ===")
def snap(T, i, w_):
    n = len(T)
    if w_ and 0 <= i < n and T[i] == w_: return i
    if w_:
        near = [k for k in range(max(0, i - 2), min(n, i + 3)) if T[k] == w_]
        if near: return min(near, key=lambda x: abs(x - i))
    return i if 0 <= i < n else None

def score(track):
    hasnl = track in ("NoPnx-PA", "PA")
    gold = {}
    for l in Z.read(f"heldout_data/{track}_test.jsonl").decode("utf-8").split("\n"):
        if l.strip():
            d = json.loads(l); gold[d["doc_id"]] = d
    P = f"scratch_exo/papereval/{track}"
    rows = json.load(open(f"{P}/draft_rows.json", encoding="utf-8"))["rows"]
    def f1(g, p):
        tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum()); fn = int(((p == 0) & (g == 1)).sum())
        return 100 * 2 * tp / max(2 * tp + fp + fn, 1), tp, fp, fn
    E, J, adds, rms = [], [], 0, 0
    mTP = mFP = mFN = 0
    for did, srow in rows.items():
        if did not in gold: continue
        T = gold[did]["tokens"]; g = np.array(gold[did]["labels"])
        r = np.array([int(c) for c in srow])
        if len(r) != len(g): continue
        r = r.copy(); r[-1] = 1
        if hasnl:
            for i, tk in enumerate(T):
                if tk == "\n":
                    r[i] = 0
                    if i > 0 and T[i - 1] != "\n": r[i - 1] = 1
        base = r.copy()
        vp = f"{P}/exam_out/{did}.json"
        if os.path.exists(vp):
            v = json.load(open(vp, encoding="utf-8"))
            for b in v.get("remove", []):
                k = snap(T, int(b.get("i", -1)), b.get("w", ""))
                if k is None or r[k] != 1 or k == len(T) - 1: continue
                if hasnl and (T[k] == "\n" or (k + 1 < len(T) and T[k + 1] == "\n")): continue
                r[k] = 0; rms += 1
            for b in v.get("add", []):
                k = snap(T, int(b.get("i", -1)), b.get("w", ""))
                if k is None or r[k] != 0 or T[k] == "\n": continue
                r[k] = 1; adds += 1
            r[-1] = 1
        e, tp, fp, fn = f1(g, base); mTP += tp; mFP += fp; mFN += fn
        jj, _, _, _ = f1(g, r)
        E.append(e); J.append(jj)
    E, J = np.array(E), np.array(J)
    d = J - E
    micro = 100 * 2 * mTP / max(2 * mTP + mFP + mFN, 1)
    from scipy import stats
    tt = stats.ttest_rel(J, E)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return dict(n=len(E), ens=E.mean(), jur=J.mean(), delta=d.mean(), micro=micro,
                up=int((d > 0).sum()), dn=int((d < 0).sum()), same=int((d == 0).sum()),
                adds=adds, rms=rms, p=tt.pvalue, ci=ci, best=d.max())

claims = {
    "NoPnx-NP": dict(ens=84.64, jur=89.49, delta=4.85, micro=86.23, up=186, dn=28, same=48,
                     adds=923, rms=417, ci=(3.63, 6.08), ratio=6.6, pmax=1e-12, bestmin=65),
    "NoPnx-PA": dict(ens=87.05, jur=92.28, delta=5.23, micro=89.30, up=164, dn=35, same=63,
                     adds=554, rms=702, ci=(3.87, 6.59), ratio=4.7, pmax=1e-12, bestmin=65),
    "NP":       dict(ens=92.83, jur=93.61, delta=0.77, micro=93.76, up=80, dn=38, same=144,
                     adds=192, rms=203, ci=(0.38, 1.17), ratio=2.1, pmax=1e-3, bestmin=25),
    "PA":       dict(ens=94.56, jur=94.92, delta=0.36, micro=95.76, up=58, dn=44, same=160,
                     adds=141, rms=113, ci=(0.05, 0.67), ratio=1.3, pmax=0.05, bestmin=10),
}
edit_totals = {}
for track, c in claims.items():
    r = score(track)
    print(f" -- {track} (n={r['n']})")
    check(f"{track} n=262", 262, r["n"])
    check(f"{track} ensemble macro", c["ens"], round(r["ens"], 2))
    check(f"{track} jury macro", c["jur"], round(r["jur"], 2))
    check(f"{track} delta", c["delta"], round(r["delta"], 2))
    check(f"{track} ensemble micro", c["micro"], round(r["micro"], 2))
    check(f"{track} improved", c["up"], r["up"])
    check(f"{track} degraded", c["dn"], r["dn"])
    check(f"{track} unchanged", c["same"], r["same"])
    check(f"{track} boundaries added", c["adds"], r["adds"])
    check(f"{track} boundaries removed", c["rms"], r["rms"])
    check(f"{track} CI low", c["ci"][0], round(r["ci"][0], 2), tol=0.01)
    check(f"{track} CI high", c["ci"][1], round(r["ci"][1], 2), tol=0.01)
    check(f"{track} improve:degrade ratio", c["ratio"], round(r["up"] / r["dn"], 1), tol=0.05)
    note(f"{track} p < {c['pmax']}", r["p"] < c["pmax"], f"(p={r['p']:.2e})")
    note(f"{track} best doc gain > {c['bestmin']}", r["best"] > c["bestmin"], f"(best={r['best']:.1f})")
    edit_totals[track] = r["adds"] + r["rms"]
    if track in ("NP", "PA"):
        note(f"{track} more than half of documents untouched", r["same"] > 131, f"(same={r['same']})")

check("prose: NP total edits 395", 395, edit_totals["NP"])
check("prose: PA total edits 254", 254, edit_totals["PA"])
check("prose: NoPnx-NP total edits 1340", 1340, edit_totals["NoPnx-NP"])
check("prose: NoPnx-PA total edits 1256", 1256, edit_totals["NoPnx-PA"])

# ---------------------------------------------------------------- 5b. intro error analysis
print("\n=== 5b. Intro error analysis (NoPnx-NP dev, threshold 0.5) ===")
import io as _io
_V = ["imp_fgm", "nat_arabertv02_base", "xlmr-2e5", "boost_supcon01", "nat_asafaya_large"]
_gold = {}
for _l in Z.read("heldout_data/NoPnx-NP_dev.jsonl").decode("utf-8").split("\n"):
    if _l.strip():
        _d = json.loads(_l); _gold[_d["doc_id"]] = _d
_P = {v: dict(np.load(_io.BytesIO(Z.read(f"swept/arabic-sentence-segmentation/probs/{v}_dev.npz")))) for v in _V}
_fp, _aw, _ne = [], 0, 0
for _did, _d in _gold.items():
    _g = np.array(_d["labels"])
    _ps = np.stack([_P[v][_did] for v in _V])
    _em = _ps.mean(0)
    _pred = (_em >= 0.5).astype(int); _pred[-1] = 1
    _err = _pred != _g
    _ne += int(_err.sum())
    _fp += _em[(_pred == 1) & (_g == 0)].tolist()
    _aw += int((_err & (((_ps >= 0.5).astype(int) != _g).all(0))).sum())
check("median ensemble prob at FP sites (dev)", 0.86, round(float(np.median(_fp)), 2))
check("share of errors with every voter wrong (dev)", 48.3, round(100 * _aw / _ne, 1))
check("dev total error sites", 3749, _ne)
check("dev all-five-wrong count", 1811, _aw)

# appendix error-analysis section: solo voters, threshold overlap, concentration, test fixes
def _f1v(_g, _p):
    _tp = int(((_p == 1) & (_g == 1)).sum()); _fpx = int(((_p == 1) & (_g == 0)).sum()); _fnx = int(((_p == 0) & (_g == 1)).sum())
    return 100 * 2 * _tp / max(2 * _tp + _fpx + _fnx, 1)
_solo = {v: [] for v in _V}; _ensf = []; _derr = []; _tp_probs = []
for _did, _d in _gold.items():
    _g = np.array(_d["labels"])
    _ps = np.stack([_P[v][_did] for v in _V])
    _em = _ps.mean(0)
    _pred = (_em >= 0.5).astype(int); _pred[-1] = 1
    _ensf.append(_f1v(_g, _pred))
    _derr.append(int((_pred != _g).sum()))
    _tp_probs += _em[_g == 1].tolist()
    for _i, v in enumerate(_V):
        _pv = (_ps[_i] >= 0.5).astype(int); _pv[-1] = 1
        _solo[v].append(_f1v(_g, _pv))
_solos = [round(float(np.mean(_solo[v])), 2) for v in _V]
check("dev ensemble macro F1 (threshold 0.5)", 85.62, round(float(np.mean(_ensf)), 2))
check("dev solo voter min", 80.74, min(_solos))
check("dev solo voter max", 84.56, max(_solos))
check("share of true boundaries below median FP prob", 22.7, round(100 * float(np.mean(np.array(_tp_probs) < np.median(_fp))), 1))
_de = np.array(sorted(_derr, reverse=True)); _k = max(1, int(0.2 * len(_de)))
check("worst 20 pct of dev docs hold pct of errors", 51.4, round(100 * _de[:_k].sum() / _de.sum(), 1))

_gt = {}
for _l in Z.read("heldout_data/NoPnx-NP_test.jsonl").decode("utf-8").split("\n"):
    if _l.strip():
        _d = json.loads(_l); _gt[_d["doc_id"]] = _d
_Pt = {v: dict(np.load(_io.BytesIO(Z.read(f"swept/arabic-sentence-segmentation/probs/{v}_test.npz")))) for v in _V}
_rows = json.load(open("scratch_exo/papereval/NoPnx-NP/draft_rows.json", encoding="utf-8"))["rows"]
_tot = _fixed = _fixed_aw = _intro = 0
for _did, _srow in _rows.items():
    if _did not in _gt: continue
    _T = _gt[_did]["tokens"]; _g = np.array(_gt[_did]["labels"])
    _base = np.array([int(c) for c in _srow]); _base[-1] = 1
    _jury = _base.copy()
    _vp = f"scratch_exo/papereval/NoPnx-NP/exam_out/{_did}.json"
    if os.path.exists(_vp):
        _v = json.load(open(_vp, encoding="utf-8"))
        for _b in _v.get("remove", []):
            _kk = snap(_T, int(_b.get("i", -1)), _b.get("w", ""))
            if _kk is None or _jury[_kk] != 1 or _kk == len(_T) - 1: continue
            _jury[_kk] = 0
        for _b in _v.get("add", []):
            _kk = snap(_T, int(_b.get("i", -1)), _b.get("w", ""))
            if _kk is None or _jury[_kk] != 0: continue
            _jury[_kk] = 1
        _jury[-1] = 1
    _tot += int((_base != _g).sum())
    _psx = np.stack([_Pt[v][_did] for v in _V])
    _vw = ((_psx >= 0.5).astype(int) != _g)
    _fx = (_base != _g) & (_jury == _g)
    _fixed += int(_fx.sum())
    _intro += int(((_base == _g) & (_jury != _g)).sum())
    _fixed_aw += int((_fx & _vw.all(0)).sum())
check("test total base error sites", 3413, _tot)
check("test errors fixed by juries", 1110, _fixed)
check("test errors introduced by juries", 230, _intro)
check("test fixed errors that were all-five-wrong", 299, _fixed_aw)
check("pct of fixed errors that were all-five-wrong", 26.9, round(100 * _fixed_aw / max(_fixed, 1), 1))

# zero-shot arm fix stats (appendix B integration)
_totz = _fixedz = _introz = _fixed_awz = 0
for _did, _srow in _rows.items():
    if _did not in _gt: continue
    _T = _gt[_did]["tokens"]; _g = np.array(_gt[_did]["labels"])
    _base = np.array([int(c) for c in _srow]); _base[-1] = 1
    _jury = _base.copy()
    _vp = f"scratch_exo/papereval/NoPnx-NP/exam_out_zeroshot/{_did}.json"
    if os.path.exists(_vp):
        _v = json.load(open(_vp, encoding="utf-8"))
        for _b in _v.get("remove", []):
            _i2 = int(_b.get("i", -1)); _w2 = _b.get("w", "")
            if 0 <= _i2 < len(_T) and _w2 and _T[_i2] == _w2 and _jury[_i2] == 1 and _i2 != len(_T) - 1: _jury[_i2] = 0
        for _b in _v.get("add", []):
            _i2 = int(_b.get("i", -1)); _w2 = _b.get("w", "")
            if 0 <= _i2 < len(_T) and _w2 and _T[_i2] == _w2 and _jury[_i2] == 0: _jury[_i2] = 1
        _jury[-1] = 1
    _psx = np.stack([_Pt[v][_did] for v in _V])
    _vw = ((_psx >= 0.5).astype(int) != _g)
    _fx = (_base != _g) & (_jury == _g)
    _fixedz += int(_fx.sum()); _introz += int(((_base == _g) & (_jury != _g)).sum())
    _fixed_awz += int((_fx & _vw.all(0)).sum())
check("zero-shot errors fixed", 917, _fixedz)
check("zero-shot errors introduced", 187, _introz)
check("zero-shot fixed all-five-wrong", 227, _fixed_awz)
check("zero-shot pct all-five-wrong of fixes", 24.8, round(100 * _fixed_awz / max(_fixedz, 1), 1))

# trace-document claims (appendix D)
def _apply_strict(_T, _base, _v):
    _r = _base.copy()
    for _b in _v.get("remove", []):
        _i3 = int(_b.get("i", -1)); _w3 = _b.get("w", "")
        if 0 <= _i3 < len(_T) and _w3 and _T[_i3] == _w3 and _r[_i3] == 1 and _i3 != len(_T) - 1: _r[_i3] = 0
    for _b in _v.get("add", []):
        _i3 = int(_b.get("i", -1)); _w3 = _b.get("w", "")
        if 0 <= _i3 < len(_T) and _w3 and _T[_i3] == _w3 and _r[_i3] == 0: _r[_i3] = 1
    _r[-1] = 1
    return _r
def _f1s(_g, _p):
    _tp = int(((_p == 1) & (_g == 1)).sum()); _fpq = int(((_p == 1) & (_g == 0)).sum()); _fnq = int(((_p == 0) & (_g == 1)).sum())
    return 100 * 2 * _tp / max(2 * _tp + _fpq + _fnq, 1)
for _did, _bf, _jf in [("doc_00f88da2b078", 28.6, 100.0), ("doc_23175663d44d", 77.1, 87.5)]:
    _T = _gt[_did]["tokens"]; _g = np.array(_gt[_did]["labels"])
    _base = np.array([int(c) for c in _rows[_did]]); _base[-1] = 1
    _v = json.load(open(f"scratch_exo/papereval/NoPnx-NP/exam_out_rerun/{_did}.json", encoding="utf-8"))
    check(f"trace {_did} draft F1", _bf, round(_f1s(_g, _base), 1))
    check(f"trace {_did} jury F1", _jf, round(_f1s(_g, _apply_strict(_T, _base, _v)), 1))

# ---------------------------------------------------------------- 5c. appendix reasoning excerpts are verbatim
print("\n=== 5c. Appendix excerpts trace to the doctrine files ===")
_d0 = open("scratch_exo/retrain4/nopa/doctrine_j0.md", encoding="utf-8").read()
_d1 = open("scratch_exo/retrain4/nopa/doctrine_j1.md", encoding="utf-8").read()
note("J0 batch-2 arc: error-profile inversion quote", "batch 2 inverted my error profile from 4.8:1 under-cutting" in _d0)
note("J0 batch-2 arc: base-rate sentence", "a base rate is not transferable" in _d0.replace("\n> ", " ").replace("\n", " "))
note("J0 batch-2 arc: 354 false cuts attribution", "354 false cuts" in _d0 and "300 of my" in _d0)
note("J0 batch-13 arc: WELD default falsified", "falsified the WELD default" in _d0)
note("J0 batch-13 arc: most expensive sentence quote", "the single most expensive sentence I have ever written in this file" in _d0)
note("J0 batch-13 arc: not a return to batch 1", "This is not a return to batch 1" in _d0)
note("J1 batch-2 audit: no new evidence quote", "batch 2 contains **no new evidence**" in _d1)
note("J1 batch-2 audit: rather than pretending", "rather than pretending to a second" in _d1.replace("\n", " "))
note("J1 batch-5: K1 IS WRONG revision", "K1 IS WRONG" in _d1 and "each hemistich is a unit" in _d1)

# figure: 10-document moving average endpoints
_sp5 = json.load(open("scratch_exo/retrain3/nopa/allsplit.json", encoding="utf-8"))
_trn5 = {}
for _l in open("data/NoPnx-PA_train.jsonl", encoding="utf-8"):
    if _l.strip():
        _d = json.loads(_l); _trn5[_d["doc_id"]] = _d
def _att(_j, _did):
    _d = _trn5[_did]; _T = _d["tokens"]; _g = np.array(_d["labels"])
    _v = json.load(open(f"scratch_exo/retrain4/nopa/ans/j{_j}/{_did}.json", encoding="utf-8"))
    _row = np.zeros(len(_T), int)
    for _b in _v.get("boundaries", []):
        _kk = snap(_T, int(_b["i"]), _b.get("w", ""))
        if _kk is not None and _T[_kk] != "\n": _row[_kk] = 1
    _row[-1] = 1
    for _i, _t in enumerate(_T):
        if _t == "\n":
            _row[_i] = 0
            if _i > 0 and _T[_i - 1] != "\n": _row[_i - 1] = 1
    _tp = int(((_row == 1) & (_g == 1)).sum()); _fpx = int(((_row == 1) & (_g == 0)).sum()); _fnx = int(((_row == 0) & (_g == 1)).sum())
    return 100 * 2 * _tp / max(2 * _tp + _fpx + _fnx, 1)
for _j, (_first, _last) in {"0": (85.1, 90.9), "1": (88.5, 94.5)}.items():
    _f = [_att(_j, d) for c in _sp5[_j] for d in c]
    _bm = np.array([np.mean(_f[i:i + 5]) for i in range(0, len(_f), 5)])
    _ma = np.convolve(_bm, np.ones(10) / 10, mode="valid")
    check(f"jury {_j} first 10-batch window", _first, round(float(_ma[0]), 1))
    check(f"jury {_j} last 10-batch window", _last, round(float(_ma[-1]), 1))
    note(f"jury {_j} final window is its maximum", round(float(_ma[-1]), 1) == round(float(_ma.max()), 1))
note("figure file exists", os.path.exists("paper/figs/learning_trend.pdf"))

# ---------------------------------------------------------------- 5d. claim consistency
print("\n=== 5d. Rank claims and no-punctuation fact ===")
tex_nocomment = re.sub(r"(?m)^%.*$", "", tex)
note("no clean-sweep rank claim anywhere in tex (comments excluded)", ("first on all four" not in tex_nocomment) and ("first place on all" not in tex_nocomment))
note("three-of-four rank stated somewhere", ("3 out of 4 tracks" in tex) or ("three of the four tracks" in tex) or ("three of four" in tex))
_pun = set("،.؟!:؛,;?\"()«»[]{}")
_ptok = 0
for _l in open("data/NoPnx-PA_train.jsonl", encoding="utf-8"):
    if _l.strip():
        for _t in json.loads(_l)["tokens"]:
            if any(c in _pun for c in _t): _ptok += 1
check("NoPnx-PA train punctuation tokens = 0 (appendix C reading note)", 0, _ptok)
note("caption no longer says 'submitted predictions reproduce'", "our submitted predictions reproduce" not in tex)

# ---------------------------------------------------------------- 6. no rule-based claim
print("\n=== 6. Submitted system is purely neural (exact reproduction) ===")
from data import load_jsonl
from dp_decode import decode_doc, fit_length_logprob
from dp_adaptive import doc_length_logprob
V = ["imp_fgm", "nat_arabertv02_base", "xlmr-2e5", "boost_supcon01", "nat_asafaya_large"]
Pb = {v: dict(np.load(f"probs/blind/nonp_{v}_blind.npz")) for v in V}
bl = {d["doc_id"]: d for d in load_jsonl("data/NoPnx-NP_blind.jsonl")}
GLP = fit_length_logprob(load_jsonl("data/NoPnx-NP_train.jsonl"))
raw = open("subs_blind/prediction_NoPnx-NP", "rb").read().decode("utf-8").replace("\r\n", "\n").strip().split("\n")
sub = {}
for l in raw[1:]:
    if l:
        did, ss = l.split(","); sub[did] = np.array([int(c) for c in ss])
exact = 0
for did, d in bl.items():
    if did not in sub: continue
    T = d["tokens"]
    em = np.mean([Pb[v][did] for v in V], 0)
    p1 = decode_doc(T, em, GLP, lam=0.4, bias=0.75)
    lp2 = doc_length_logprob(T, p1, GLP, blend=0.7)
    r = np.array(decode_doc(T, em, lp2, lam=0.4, bias=0.75)); r[-1] = 1
    if len(r) == len(sub[did]) and (r == sub[did]).all(): exact += 1
check("blind rows reproduced from probabilities alone", 212, exact)

# ---------------------------------------------------------------- 7. sealing / leakage
print("\n=== 7. Leakage checks ===")
ids = set()
for t in ("NoPnx-NP", "NoPnx-PA", "NP", "PA"):
    for sp_ in ("dev", "test"):
        for l in Z.read(f"heldout_data/{t}_{sp_}.jsonl").decode("utf-8").split("\n"):
            if l.strip(): ids.add(json.loads(l)["doc_id"])
bad = []
for f in ["scratch_exo/retrain4/nopa/doctrine_j0.md", "scratch_exo/retrain4/nopa/doctrine_j1.md",
          "scratch_exo/retrain3/nonp/doctrine_j0.md", "scratch_exo/retrain3/nonp/doctrine_j1.md"]:
    txt = open(f, encoding="utf-8", errors="replace").read()
    hits = [i for i in ids if i in txt]
    if hits: bad.append((f, hits[:3]))
note("no dev/test doc id appears in any doctrine", not bad, str(bad[:2]))
leak = 0
for t in ("NoPnx-NP", "NoPnx-PA"):
    for f in os.listdir(f"scratch_exo/papereval/{t}/docs"):
        if '"labels"' in open(f"scratch_exo/papereval/{t}/docs/{f}", encoding="utf-8", errors="replace").read():
            leak += 1
check("exam packets containing labels", 0, leak)

# ---------------------------------------------------------------- 8. style constraints
print("\n=== 8. Draft style constraints ===")
check("em dashes in tex", 0, tex.count("\u2014"))
check("en dashes in tex", 0, tex.count("\u2013"))
print(f"  [INFO] tildes in tex: {tex.count(chr(126))} (allowed in this draft: non-breaking refs)")
print(f"  [INFO] TODO markers remaining: {tex.count('TODO')}")

# ---------------------------------------------------------------- summary
fails = [r for r in results if not r[0]]
print("\n" + "=" * 62)
print(f"{len(results) - len(fails)}/{len(results)} checks PASSED")
if fails:
    print("FAILURES:")
    for _, label, c, a in fails:
        print(f"  - {label}: paper={c} recomputed={a}")
sys.exit(1 if fails else 0)
