# -*- coding: utf-8 -*-
"""FM3 CHAIN DECODE PACK — deterministic, hand-written decode-side rules for
repetition-chain genres (genealogy, hadith isnad, verse lists). ADDITIVE new
file; nothing existing is modified. CPU only. Precedent: the legal decode pack
(v16-B1 + v17-2): hand-designed detectors + rule arms, TRAIN-GOLD-ONLY
fitting, hard fire/no-fire gates, dev used for measurement/reporting only.

Fitting discipline (binding, v16-B1 pattern)
--------------------------------------------
Every arm forces boundaries only at "joint" positions of a detected repetition
chain, and every joint FAMILY is gated by TRAIN gold statistics measured
inside detected train regions:

    ENABLED  iff  n_train >= 2  and  P(gold boundary at joint | train) >= 0.9

An arm whose train evidence is absent (n=0/1) or contradicts the forced cut
NEVER FIRES on dev — exactly like v16-B1's promote-«،» arm (train P=0.136 →
never fired). Dev gold is never read by the fitter.

The four arms (design notes of record)
--------------------------------------
ISNAD      hadith transmission chain. Opener in {حدثنا, حدثني, أخبرنا, أخبرني}
           (family = standard hadith transmission verbs, hand-listed a priori;
           v16-B1 precedent for hand-designed token classes). Region runs from
           the opener to the first قال/فقال NOT followed by a transmission
           verb (the matn-قال), inclusive. Joint families:
             qal+trans : boundary BEFORE every قال/فقال followed by a
                         transmission verb   (train: cuts before every one)
             qal-matn  : boundary BEFORE the terminal matn-قال
           Train evidence: docs doc_0d81543aa818, doc_26ebb09caa38,
           doc_b8fa69b09c88, doc_e1435023282b — gold cuts before every قال.
           عن-links are never joints (gold keeps them fused).
GEN-WALAD  Genesis-5 begat formula (train doc doc_b58385fe50df):
           "وعاش X ... وولد Y || وعاش X بعد ما ولد Y ... || فكانت كل أيام X".
           Joints = formula-head tokens (candidates وعاش فكانت وكانت وسار
           وكان ودعا), each gated on train; additionally a head must repeat
           >= 2 times INSIDE the region to fire there (a chain is repetition;
           this keeps a lone mid-sentence فكانت in a non-Genesis genealogy
           from being forced).
GEN-BN     nasab/genealogy بن-run: >= 3 repetitions of بن at gap <= 3 tokens.
           Candidate joint: boundary before each بن. Train evidence
           (doc_a9c7d1c861a3 "محمد بن عبد الله بن الحسن بن الحسن بن علي بن
           أبي طالب", 0 internal cuts; quiz-option runs, 0 cuts before بن):
           P(train) = 0 → the arm NEVER FIRES. Encoded and reported, not
           dropped silently. (The dev Luke-3 chain cuts every FIFTH بن-link —
           a verse rhythm with no surface marker; closed-track train-only
           fitting cannot license it. Same verdict as v16-B1: rare-convention
           collision, untreatable decode-side.)
VERSE-LIST قوله تعالى / (ف|و)أنزل الله openings, >= 3 repetitions per doc.
           Candidate joint: boundary before each opening. Train has no doc
           with >= 3 repetitions (the bigrams appear mid-sentence in quiz
           stems / Quran prose) → no train region → NEVER FIRES. Reported.

Override semantics (hard guarantees, asserted at apply time)
------------------------------------------------------------
  * predictions may change ONLY at enabled joint positions inside detected
    regions, and only 0 -> 1 (force a boundary; never remove one);
  * outside detected chain spans the baseline prediction is bit-identical
    (asserted per doc, and reported as "false-cut cost outside chains = 0").

Usage (CPU, reads frozen caches, writes nothing unless --out-csv):
  python src/chain_decode.py --self-test
  python src/chain_decode.py --fit
  python src/chain_decode.py --apply --probs probs/union-NoPnx-NP-arabertv02-s42_dev.npz
  python src/chain_decode.py --apply --probs probs/union-NoPnx-NP-POOL_dev.npz
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# hand-written token classes (a priori domain knowledge, v16-B1 style)
# ---------------------------------------------------------------------------
TRANS = ("حدثنا", "حدثني", "أخبرنا", "أخبرني")            # transmission verbs
TRANS_PREF = tuple("و" + t for t in TRANS)                  # و-prefixed forms
QAL = ("قال", "فقال")                                       # "he said" forms
# Said-paradigm handover forms: when the isnad hands over to reported speech
# via "…سمع فلانا يقول…" / "…أنها قالت…", gold does NOT cut at the handover
# and the chain is over — the region must END there with no forced joint.
# Train evidence: يقول in doc_dfc470ebeb16 ("يقول قال رسول الله", gold 0) and
# doc_27a114d4e376 ("يقول جاء رجل", matn starts mid-flow). The rest of the
# paradigm (feminine/plural/imperfect forms) is an a priori morphological
# class, v16-B1-style hand-designed token class.
SAID_HANDOVER = ("يقول", "تقول", "قالت", "قالوا", "قلت", "قلنا", "نقول")
WALAD_DETECT = ("ولد", "وولد", "فولدت", "يلد", "مواليد")   # begat-chain markers
WALAD_HEADS = ("وعاش", "فكانت", "وكانت", "وسار", "وكان", "ودعا")  # formula heads
BN = "بن"
VERSE_BIGRAMS = (("قوله", "تعالى"), ("فأنزل", "الله"), ("وأنزل", "الله"), ("أنزل", "الله"))

# fire/no-fire gate on train gold (v16-B1: promote arm parked at P=0.136)
GATE_MIN_N = 2
GATE_MIN_P = 0.9

# detector constants
ISNAD_WINDOW = 60          # opener qualification window
ISNAD_MIN_TRANS = 2        # >= 2 transmission verbs near opener (>= 2 links)
ISNAD_REGION_CAP = 150     # max region length if no matn-قال found
WALAD_MIN_REP = 4          # begat markers per cluster
WALAD_MAX_GAP = 40         # max token gap inside a begat cluster
WALAD_HEAD_MIN_REP = 2     # a head must repeat inside the region to fire
BN_MIN_REP = 3             # بن repetitions (spec of the pack)
BN_MAX_GAP = 3             # بن ... بن gap ("بن <name>" unit is 2 tokens)
VERSE_MIN_REP = 3          # verse-list openings per doc


@dataclass
class Region:
    arm: str
    start: int   # token span [start, end)  — override may only touch it
    end: int
    joints: List[Tuple[str, int]] = field(default_factory=list)
    # joints: (family_key, token_pos j) meaning "force boundary BEFORE token j"
    # i.e. labels[j-1] = 1.  family_key is what the train gate is keyed on.


# ---------------------------------------------------------------------------
# detectors (pure functions of the token list; no gold, no probs)
# ---------------------------------------------------------------------------
def _detect_isnad(tokens: List[str]) -> List[Region]:
    regions: List[Region] = []
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i] not in TRANS:
            i += 1
            continue
        win = tokens[i:i + ISNAD_WINDOW]
        n_trans = sum(1 for t in win if t in TRANS or t in TRANS_PREF)
        n_qal = sum(1 for t in win if t in QAL)
        if n_trans < ISNAD_MIN_TRANS or n_qal < 1:
            i += 1
            continue
        # walk to the matn-قال (first قال/فقال not followed by a transmission verb)
        joints: List[Tuple[str, int]] = []
        end = min(n, i + ISNAD_REGION_CAP)
        j = i + 1
        while j < min(n, i + ISNAD_REGION_CAP):
            if tokens[j] in SAID_HANDOVER:
                # handover to reported speech without a cuttable قال joint
                # (train: doc_dfc470ebeb16, doc_27a114d4e376) — end the region,
                # force nothing.
                end = j + 1
                break
            if tokens[j] in QAL:
                if j + 1 < n and tokens[j + 1] in TRANS:
                    joints.append(("ISNAD/qal+trans", j))
                else:
                    # matn-قال: the chain hands over to the reported speech;
                    # gold cuts before it in every remaining train case.
                    joints.append(("ISNAD/qal-matn", j))
                    end = j + 1
                    break
            j += 1
        regions.append(Region("ISNAD", i, end, joints))
        i = end
    return regions


def _detect_walad(tokens: List[str]) -> List[Region]:
    pos = [i for i, t in enumerate(tokens) if t in WALAD_DETECT]
    regions: List[Region] = []
    cluster: List[int] = []
    for p in pos + [10 ** 9]:  # sentinel flushes the last cluster
        if cluster and p - cluster[-1] > WALAD_MAX_GAP:
            if len(cluster) >= WALAD_MIN_REP:
                start, end = cluster[0], cluster[-1] + 1
                heads = defaultdict(list)
                for j in range(start, end):
                    if tokens[j] in WALAD_HEADS:
                        heads[tokens[j]].append(j)
                joints = [(f"GEN-WALAD/{h}", j)
                          for h, js in heads.items() if len(js) >= WALAD_HEAD_MIN_REP
                          for j in js]
                regions.append(Region("GEN-WALAD", start, end, sorted(joints, key=lambda x: x[1])))
            cluster = []
        cluster.append(p)
    return regions


def _detect_bn(tokens: List[str]) -> List[Region]:
    pos = [i for i, t in enumerate(tokens) if t == BN]
    regions: List[Region] = []
    run: List[int] = []
    for p in pos + [10 ** 9]:
        if run and p - run[-1] > BN_MAX_GAP:
            if len(run) >= BN_MIN_REP:
                start, end = max(0, run[0] - 1), min(len(tokens), run[-1] + 2)
                joints = [("GEN-BN/before-بن", j) for j in run]
                regions.append(Region("GEN-BN", start, end, joints))
            run = []
        run.append(p)
    return regions


def _detect_verse(tokens: List[str]) -> List[Region]:
    pos = [i for i in range(len(tokens) - 1)
           if any(tokens[i] == a and tokens[i + 1] == b for a, b in VERSE_BIGRAMS)]
    if len(pos) < VERSE_MIN_REP:
        return []
    joints = [("VERSE/opening", j) for j in pos]
    return [Region("VERSE-LIST", pos[0], min(len(tokens), pos[-1] + 2), joints)]


def detect_regions(tokens: List[str]) -> List[Region]:
    return (_detect_isnad(tokens) + _detect_walad(tokens)
            + _detect_bn(tokens) + _detect_verse(tokens))


# ---------------------------------------------------------------------------
# train-gold fitting: gate every joint family (dev gold never enters here)
# ---------------------------------------------------------------------------
def fit_gates(train_docs: List[dict], verbose: bool = True) -> Dict[str, bool]:
    stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # fam -> [n, correct]
    fires: List[str] = []
    for d in train_docs:
        toks, labs = d["tokens"], d["labels"]
        for r in detect_regions(toks):
            fam_names = sorted({f for f, _ in r.joints})
            fires.append(f"    {d.get('doc_id', '?')}  {r.arm:9s} span=[{r.start},{r.end}) "
                         f"joints={len(r.joints)} {fam_names}")
            for fam, j in r.joints:
                if j <= 0:
                    continue
                stats[fam][0] += 1
                stats[fam][1] += int(labs[j - 1] == 1)
    enabled = {fam: (n >= GATE_MIN_N and c / n >= GATE_MIN_P)
               for fam, (n, c) in stats.items()}
    if verbose:
        print("== TRAIN-GOLD FIT (gate: n>=%d and P>=%.2f; dev never read) ==" % (GATE_MIN_N, GATE_MIN_P))
        print("  detected train regions:")
        for line in fires or ["    (none)"]:
            print(line)
        print(f"  {'joint family':28s} {'n':>4s} {'gold-correct':>12s} {'P(train)':>9s}  verdict")
        for fam in sorted(stats):
            n, c = stats[fam]
            v = "ENABLED" if enabled[fam] else "NEVER FIRES"
            print(f"  {fam:28s} {n:4d} {c:12d} {c / n:9.3f}  {v}")
        # families that never even occurred in a train region (no evidence at all)
        for fam in ("GEN-BN/before-بن", "VERSE/opening"):
            if fam not in stats:
                print(f"  {fam:28s} {0:4d} {0:12d} {'--':>9s}  NEVER FIRES (no train region)")
        print("  (unconditional, report-only) P(gold boundary immediately before marker), whole train:")
        for name, mark in [("verse-opening bigram", None)]:
            n = c = 0
            for d in train_docs:
                toks, labs = d["tokens"], d["labels"]
                for i in range(1, len(toks) - 1):
                    if any(toks[i] == a and toks[i + 1] == b for a, b in VERSE_BIGRAMS):
                        n += 1
                        c += int(labs[i - 1] == 1)
            print(f"    {name}: n={n} correct={c} P={c / n if n else float('nan'):.3f}")
    return dict(enabled)


# ---------------------------------------------------------------------------
# override application (prediction-level; only 0->1 at enabled joints)
# ---------------------------------------------------------------------------
def apply_doc(tokens: List[str], base_pred: np.ndarray, enabled: Dict[str, bool]):
    regions = detect_regions(tokens)
    new = base_pred.copy()
    forced: List[Tuple[str, int]] = []       # (family, label_index touched)
    for r in regions:
        for fam, j in r.joints:
            if j <= 0 or not enabled.get(fam, False):
                continue
            if not (r.start <= j - 1 < r.end):
                continue
            forced.append((fam, j - 1))
            new[j - 1] = 1
    # hard guarantees
    diff = np.where(new != base_pred)[0]
    forced_idx = {i for _, i in forced}
    assert set(diff.tolist()) <= forced_idx, "override touched a non-joint position"
    in_region = np.zeros(len(tokens), dtype=bool)
    for r in regions:
        in_region[r.start:r.end] = True
    assert in_region[diff].all() if len(diff) else True, "override outside detected span"
    assert (new >= base_pred).all(), "override removed a boundary"
    return new, forced, regions


# ---------------------------------------------------------------------------
# measurement (dev = evaluation only)
# ---------------------------------------------------------------------------
def _prf1(g: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((g == 1) & (p == 1)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def _annotate_span(tokens, gold, base, new, start, end) -> str:
    out = []
    for i in range(max(0, start), min(len(tokens), end)):
        out.append(tokens[i])
        marks = ""
        if gold[i] == 1:
            marks += "G"
        if base[i] == 1:
            marks += "b"
        if new[i] == 1 and base[i] == 0:
            marks += "F"
        if marks:
            out.append("/" + marks + "/")
    return " ".join(out)


def run_apply(probs_path: str, gold_path: str, train_path: str, thr: float,
              out_csv: Optional[str] = None) -> None:
    train_docs = _load_jsonl(train_path)
    enabled = fit_gates(train_docs, verbose=True)
    docs = _load_jsonl(gold_path)
    z = np.load(probs_path)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from genre_buckets import bucket
    except Exception:  # pragma: no cover
        def bucket(_tokens):
            return "n/a"

    f1_before, f1_after, genres = [], [], []
    touched_rows = []
    span_dumps = []
    tot_forced = tot_flip = tot_fn_rec = tot_fp_add = tot_already = 0
    preds_out = {}
    for d in docs:
        did = d["doc_id"]
        g = np.asarray(d["labels"], dtype=int)
        p = z[did]
        assert len(p) == len(g), did
        base = (p >= thr).astype(int)
        new, forced, regions = apply_doc(d["tokens"], base, enabled)
        preds_out[did] = new
        _, _, fb = _prf1(g, base)
        _, _, fa = _prf1(g, new)
        f1_before.append(fb)
        f1_after.append(fa)
        genres.append(bucket(d["tokens"]))
        # honesty bookkeeping — outside-chain invariance is asserted in apply_doc
        flips = [(fam, i) for fam, i in forced if base[i] == 0]
        tot_forced += len(forced)
        tot_already += len(forced) - len(flips)
        tot_flip += len(flips)
        fn_rec = sum(1 for _, i in flips if g[i] == 1)
        fp_add = sum(1 for _, i in flips if g[i] == 0)
        tot_fn_rec += fn_rec
        tot_fp_add += fp_add
        if flips:
            touched_rows.append((did, genres[-1], 100 * fb, 100 * fa, len(flips), fn_rec, fp_add))
            for r in regions:
                if any(base[i] == 0 and new[i] == 1 and r.start <= i < r.end for _, i in forced):
                    span_dumps.append(
                        f"--- {did} [{r.arm}] span=[{r.start},{r.end})  "
                        f"(G=gold cut, b=baseline cut, F=newly forced cut)\n    "
                        + _annotate_span(d["tokens"], g, base, new, r.start - 2, r.end + 2))

    mb, ma = 100 * float(np.mean(f1_before)), 100 * float(np.mean(f1_after))
    print("\n== DEV MEASUREMENT ==")
    print(f"probs: {probs_path}   thr={thr}   docs={len(docs)}")
    print(f"macro-F1 before = {mb:.2f}")
    print(f"macro-F1 after  = {ma:.2f}")
    print(f"DELTA           = {ma - mb:+.2f}")
    print(f"forced joints={tot_forced}  already-predicted={tot_already}  flipped 0->1={tot_flip}"
          f"  -> FN recovered={tot_fn_rec}  FP added (inside chains)={tot_fp_add}")
    print("false-cut cost outside chains = 0 (asserted per doc: overrides only inside detected spans)")

    print("\nper-genre mean F1 (before -> after):")
    per = defaultdict(list)
    for gname, fb, fa in zip(genres, f1_before, f1_after):
        per[gname].append((fb, fa))
    for gname in sorted(per, key=lambda k: -len(per[k])):
        arr = np.array(per[gname])
        print(f"  {gname:20s} n={len(arr):3d}  {100 * arr[:, 0].mean():6.2f} -> {100 * arr[:, 1].mean():6.2f}"
              f"  ({100 * (arr[:, 1] - arr[:, 0]).mean():+.2f})")

    print("\ntouched docs (any prediction flipped):")
    if not touched_rows:
        print("  (none)")
    for did, gname, fb, fa, nf, fnr, fpa in touched_rows:
        print(f"  {did}  {gname:16s} F1 {fb:6.2f} -> {fa:6.2f}  flips={nf} (FN-recovered={fnr}, FP-added={fpa})")

    print("\noverridden spans (before/after, for eyeballing):")
    for s in span_dumps or ["  (none)"]:
        print(s)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            f.write("Document ID,Prediction\n")
            for d in docs:
                f.write(d["doc_id"] + "," + "".join(map(str, preds_out[d["doc_id"]].tolist())) + "\n")
        print(f"\nwrote {out_csv}")


def _load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# corpus-independent self-test (passes with data/ hidden)
# ---------------------------------------------------------------------------
def self_test() -> None:
    name = ["زيد", "عمرو", "بكر", "خالد", "سعد", "أنس"]

    # synthetic isnad train doc: حدثنا A || قال حدثنا B عن C عن D || قال <matn>
    isnad_toks = ["حدثنا", name[0], "قال", "حدثنا", name[1], "عن", name[2],
                  "عن", name[3], "قال", "الحديث", "هنا", "انتهى"]
    isnad_labs = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    # synthetic Genesis doc: (وعاش X سنة وولد Y || فكانت أيامه سنين ومات ||) x5
    gen_toks, gen_labs = [], []
    for k in range(5):
        gen_toks += ["وعاش", name[k], "سنة", "وولد", name[k + 1]]
        gen_labs += [0, 0, 0, 0, 1]
        gen_toks += ["فكانت", "أيامه", "سنين", "ومات"]
        gen_labs += [0, 0, 0, 1]
    # synthetic nasab: فلان بن A بن B بن C — gold has NO internal cut
    bn_toks = [name[0], BN, name[1], BN, name[2], BN, name[3]]
    bn_labs = [0, 0, 0, 0, 0, 0, 1]
    train = [
        {"doc_id": "t_isnad", "tokens": isnad_toks, "labels": isnad_labs},
        {"doc_id": "t_isnad2", "tokens": list(isnad_toks), "labels": list(isnad_labs)},
        {"doc_id": "t_gen", "tokens": gen_toks, "labels": gen_labs},
        {"doc_id": "t_bn", "tokens": bn_toks, "labels": bn_labs},
    ]
    enabled = fit_gates(train, verbose=False)
    assert enabled.get("ISNAD/qal+trans") is True, enabled
    assert enabled.get("ISNAD/qal-matn") is True, enabled
    assert enabled.get("GEN-WALAD/وعاش") is True, enabled
    assert enabled.get("GEN-WALAD/فكانت") is True, enabled
    assert enabled.get("GEN-BN/before-بن") is False, "بن arm must NEVER fire (train P=0)"

    # apply on a fresh isnad doc with an empty baseline: joints get forced
    base = np.zeros(len(isnad_toks), dtype=int)
    new, forced, _ = apply_doc(isnad_toks, base, enabled)
    assert sorted(i for _, i in forced) == [1, 8], forced   # before قال@2 and قال@9
    assert new.sum() == 2 and new[1] == 1 and new[8] == 1

    # بن chain must stay untouched even with an eager detector
    base = np.zeros(len(bn_toks), dtype=int)
    new, forced, regs = apply_doc(bn_toks, base, enabled)
    assert any(r.arm == "GEN-BN" for r in regs), "بن run must be detected"
    assert len(forced) == 0 and (new == base).all(), "disabled arm fired"

    # outside-chain invariance: prose far from any region is bit-identical
    prose = ["كلمة"] * 30 + isnad_toks + ["كلمة"] * 30
    rng = np.random.RandomState(0)
    base = (rng.rand(len(prose)) > 0.7).astype(int)
    new, forced, regs = apply_doc(prose, base, enabled)
    in_region = np.zeros(len(prose), dtype=bool)
    for r in regs:
        in_region[r.start:r.end] = True
    diff = np.where(new != base)[0]
    assert in_region[diff].all() if len(diff) else True
    assert (new[~in_region] == base[~in_region]).all()
    assert (new >= base).all()

    # handover via said-paradigm form: region ends there, nothing forced past it
    hand = ["حدثنا", name[0], "قال", "حدثنا", name[1], "عن", name[2],
            "يقول", "قال", "الرسول", "شيئا", "فقال", "الرجل", "شيئا"]
    base = np.zeros(len(hand), dtype=int)
    new, forced, regs = apply_doc(hand, base, enabled)
    assert sorted(i for _, i in forced) == [1], forced  # only before قال@2 (qal+trans)
    assert regs[0].end == 8, regs  # region ends at يقول@7 (inclusive)

    # a lone فكانت inside a begat cluster without repetition must NOT fire
    matthew = []
    for k in range(5):
        matthew += [name[k % 6], "ولد", name[(k + 1) % 6], "و" + name[(k + 1) % 6], "ولد", name[(k + 2) % 6]]
    matthew += ["أما", "الولادة", "فكانت", "هكذا"]
    base = np.zeros(len(matthew), dtype=int)
    new, forced, regs = apply_doc(matthew, base, enabled)
    assert len(forced) == 0 and (new == base).all(), "single فكانت must not be forced"

    print("self-test OK (fit gates, forced joints, disabled-arm silence, outside-chain invariance)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fit", action="store_true", help="print the train-gold fitting report only")
    ap.add_argument("--apply", action="store_true", help="apply to cached probs + measure on gold")
    ap.add_argument("--probs", default="probs/union-NoPnx-NP-arabertv02-s42_dev.npz")
    ap.add_argument("--gold", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--train", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.fit:
        fit_gates(_load_jsonl(args.train), verbose=True)
        return
    if args.apply:
        run_apply(args.probs, args.gold, args.train, args.thr, args.out_csv)
        return
    ap.error("choose one of --self-test / --fit / --apply")


if __name__ == "__main__":
    main()
