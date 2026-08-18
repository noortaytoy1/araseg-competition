"""CALIBRATION / "COCKINESS" DIAGNOSTIC for an AraSeg NoPnx-NP single-encoder
checkpoint.  Quantifies OVERCONFIDENCE / MISCALIBRATION of per-token P(boundary):
the model's "0.99" — is it a promise or a lie?

Analysis-only.  Loads an existing checkpoint and predicts per-token P(boundary)
on the dev split using the EXACT windowing in predict.predict_doc (same as
diag_preposition.py).  It trains nothing and writes nothing to runs/.

Reports:
  1. OVERALL CALIBRATION  - reliability diagram: 10 equal-width bins on predicted
     P(boundary); per bin: mean predicted confidence vs empirical boundary-rate
     and bin count.  Expected Calibration Error (ECE, count-weighted) and Maximum
     Calibration Error (MCE).  Over-confidence stated separately for the
     high-confidence FIRE band (P>=0.8) and the high-confidence NO-FIRE band
     (P<=0.2).
  2. "COCKY-WRONG" RATE   - of confident positions (P>=0.9 OR P<=0.1), what
     fraction are WRONG, split by direction (confident-fire-but-gold-0 vs
     confident-nofire-but-gold-1).
  3. PER-GENRE            - ECE and cocky-wrong-rate per genre (genre_buckets.bucket);
     genres RANKED by how untrustworthy high confidence is.
  4. PER-TRIGGER          - for named structural triggers (after-number,
     cloze-blank فراغ, document-end, genealogy بن-chain, verse-opening,
     connective-initial): mean confidence of the ERRORS at that site.

A "position" i is a decision "does a sentence boundary follow token i?", with
gold g_i and predicted p_i.  Positions where tok[i] == PARAGRAPH_TOKEN are
excluded (predict.py forces them to 0), matching diag_preposition.py.

Usage:
  PYTHONIOENCODING=utf-8 python src/diag_calibration.py \
      --model runs/union-NoPnx-NP-arabertv02-s42 \
      --dev data/NoPnx-NP_dev.jsonl \
      --threshold 0.5
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc
from genre_buckets import bucket

# ---- structural trigger vocabularies (verbatim tokens seen in dev) ----------
FARAGH = "فراغ"          # cloze fill-in-the-blank placeholder
BN = "بن"                # genealogy "X bn Y" chain link
VERSE_OPENERS = {"سورة", "بسم"}                 # verse / basmala openings
# sentence-initial connectives (over-fire traps): report/quote verbs + conjunctions
CONNECTIVES = {"ثم", "و", "ف", "فقال", "وقال", "قال", "فإن", "وقد", "وكان",
               "ولكن", "لكن", "بل", "حتى", "أو", "كما", "وكانت", "فكان"}

_DIGIT = re.compile(r"[0-9٠-٩]")
_PURE_NUM = re.compile(r"^[0-9٠-٩][0-9٠-٩.,/\-]*$")


def is_number(tok: str) -> bool:
    """True if the token is a (mostly-)numeric run, Arabic-Indic or ASCII digits."""
    return bool(_PURE_NUM.match(tok)) or (
        bool(_DIGIT.search(tok)) and sum(bool(_DIGIT.match(ch)) for ch in tok) >= max(1, len(tok) // 2)
    )


def collect_probs(docs, tokenizer, model, device, window, overlap, max_length):
    """Return per-doc arrays of P(boundary) aligned to d['tokens']."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tokenizer, model, device, window, overlap, max_length)
        out.append(p)
    return out


# ------------------------------------------------------------------ calibration
def reliability(probs, golds, n_bins=10):
    """Equal-width reliability bins on [0,1].

    Returns list of dicts (lo, hi, count, conf=mean predicted p, acc=empirical
    boundary-rate) plus ECE (count-weighted |conf-acc|) and MCE (max |conf-acc|).
    """
    probs = np.asarray(probs, dtype=float)
    golds = np.asarray(golds, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    N = len(probs)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # last bin closed on the right so p==1.0 lands somewhere
        if b == n_bins - 1:
            m = (probs >= lo) & (probs <= hi + 1e-9)
        else:
            m = (probs >= lo) & (probs < hi)
        cnt = int(m.sum())
        if cnt:
            conf = float(probs[m].mean())
            acc = float(golds[m].mean())        # empirical P(boundary) in bin
            gap = abs(conf - acc)
            ece += (cnt / N) * gap
            mce = max(mce, gap)
        else:
            conf = float("nan")
            acc = float("nan")
        bins.append(dict(lo=lo, hi=hi, count=cnt, conf=conf, acc=acc))
    return bins, ece, mce


def cocky_wrong(probs, golds, hi=0.9, lo=0.1):
    """Confident positions (p>=hi OR p<=lo): counts of confident, wrong, and the
    two directional wrongs.  Returns a dict of raw counts + rates."""
    probs = np.asarray(probs, dtype=float)
    golds = np.asarray(golds, dtype=int)
    conf_fire = probs >= hi          # model confidently says BOUNDARY
    conf_nofire = probs <= lo        # model confidently says NO boundary
    n_conf = int(conf_fire.sum() + conf_nofire.sum())
    # confident-fire but gold 0  => wrong
    wrong_fire = int(((probs >= hi) & (golds == 0)).sum())
    n_fire = int(conf_fire.sum())
    # confident-nofire but gold 1 => wrong
    wrong_nofire = int(((probs <= lo) & (golds == 1)).sum())
    n_nofire = int(conf_nofire.sum())
    n_wrong = wrong_fire + wrong_nofire
    return dict(
        n_conf=n_conf, n_wrong=n_wrong,
        wrong_rate=(n_wrong / n_conf if n_conf else float("nan")),
        n_fire=n_fire, wrong_fire=wrong_fire,
        fire_wrong_rate=(wrong_fire / n_fire if n_fire else float("nan")),
        n_nofire=n_nofire, wrong_nofire=wrong_nofire,
        nofire_wrong_rate=(wrong_nofire / n_nofire if n_nofire else float("nan")),
    )


def print_reliability(bins, ece, mce, title):
    print(f"\n{title}")
    print(f"{'bin':>12} {'count':>7} {'pred_conf':>10} {'emp_rate':>9} {'gap':>7}")
    for bdct in bins:
        if bdct["count"]:
            gap = bdct["conf"] - bdct["acc"]
            print(f"[{bdct['lo']:.1f},{bdct['hi']:.1f}) {bdct['count']:7d} "
                  f"{bdct['conf']:10.3f} {bdct['acc']:9.3f} {gap:+7.3f}")
        else:
            print(f"[{bdct['lo']:.1f},{bdct['hi']:.1f}) {bdct['count']:7d} "
                  f"{'--':>10} {'--':>9} {'--':>7}")
    print(f"ECE={ece:.4f}   MCE={mce:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="operating threshold; only used to LABEL fire/no-fire "
                         "for reporting. Calibration/cocky metrics are threshold-free.")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--hi", type=float, default=0.9, help="confident-fire band lower edge")
    ap.add_argument("--lo", type=float, default=0.1, help="confident-nofire band upper edge")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# device={device}  model={args.model}  dev={args.dev}")
    print(f"# threshold(report-only)={args.threshold}  confident bands: p>={args.hi} or p<={args.lo}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()

    dev = load_jsonl(args.dev)
    print(f"# dev docs={len(dev)}")
    probs = collect_probs(dev, tokenizer, model, device,
                          args.window, args.overlap, args.max_length)

    # ---- flatten all valid decision positions -------------------------------
    P = []          # predicted p
    G = []          # gold 0/1
    GEN = []        # genre bucket
    # per-trigger error confidences: trigger -> dict(err_confs=[], n_pos, n_err, n_gold)
    trig = defaultdict(lambda: dict(err_confs=[], all_confs=[], n_pos=0, n_err=0, n_gold=0))
    thr = args.threshold

    for d, p in zip(dev, probs):
        toks = d["tokens"]
        gold = d["labels"]
        g_bucket = bucket(toks)
        n = len(toks)
        # index of last non-[PAR] token = document-end decision site
        last_real = None
        for j in range(n - 1, -1, -1):
            if toks[j] != PARAGRAPH_TOKEN:
                last_real = j
                break
        for i in range(n):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            pi = float(p[i])
            gi = int(gold[i])
            pred = 1 if pi >= thr else 0
            err = (pred != gi)
            P.append(pi)
            G.append(gi)
            GEN.append(g_bucket)

            nxt = toks[i + 1] if i + 1 < n else None
            fires = []                                  # which triggers this pos matches
            if is_number(toks[i]):
                fires.append("after-number")
            if nxt == FARAGH or toks[i] == FARAGH:
                fires.append("cloze-blank فراغ")
            if i == last_real:
                fires.append("document-end")
            if nxt == BN:
                fires.append("genealogy بن-chain")
            if nxt in VERSE_OPENERS:
                fires.append("verse-opening")
            if nxt in CONNECTIVES:
                fires.append("connective-initial")

            for t in fires:
                s = trig[t]
                s["n_pos"] += 1
                s["n_gold"] += gi
                s["all_confs"].append(pi)
                if err:
                    s["n_err"] += 1
                    # "how cocky at the failure site": confidence = distance from 0.5
                    # toward whichever pole the model committed to.
                    s["err_confs"].append(pi)

    P = np.asarray(P); G = np.asarray(G); GEN = np.asarray(GEN)
    n_pos = len(P)
    base_rate = float(G.mean())

    print("\n================= AraSeg NoPnx-NP CALIBRATION / COCKINESS DIAGNOSTIC =================")
    print(f"total decision positions (excl. [PAR]) = {n_pos}   gold boundary base-rate = {base_rate:.3f}")

    # ---- 1. OVERALL CALIBRATION ---------------------------------------------
    bins, ece, mce = reliability(P, G, n_bins=10)
    print_reliability(bins, ece, mce,
                      "--- 1. OVERALL CALIBRATION: reliability diagram (10 equal-width bins) ---")
    print("  (gap = pred_conf - emp_rate;  POSITIVE gap => OVER-confident)")

    # FIRE band P>=0.8
    fire = P >= 0.8
    if fire.sum():
        fconf = float(P[fire].mean()); frate = float(G[fire].mean())
        print(f"\nHIGH-CONFIDENCE FIRE band (P>=0.8): n={int(fire.sum())}  "
              f"mean_pred={fconf:.3f}  actual_boundary_rate={frate:.3f}  "
              f"OVER-confidence={fconf-frate:+.3f}")
        print(f"    => when it says ~{fconf:.2f} boundary, it is right {100*frate:.1f}% of the time "
              f"({'OVER-confident by '+format(100*(fconf-frate),'.1f')+' pts' if fconf>frate else 'well/under-calibrated'})")
    # NO-FIRE band P<=0.2 : confidence in the NO-boundary call = 1 - P
    nofire = P <= 0.2
    if nofire.sum():
        nconf = float((1.0 - P[nofire]).mean())       # confidence in "no boundary"
        nacc = float((G[nofire] == 0).mean())          # right => gold 0
        print(f"HIGH-CONFIDENCE NO-FIRE band (P<=0.2): n={int(nofire.sum())}  "
              f"mean_pred_noboundary={nconf:.3f}  actual_no-boundary_rate={nacc:.3f}  "
              f"OVER-confidence={nconf-nacc:+.3f}")
        print(f"    => when it says ~{nconf:.2f} NO-boundary, it is right {100*nacc:.1f}% of the time "
              f"({'OVER-confident by '+format(100*(nconf-nacc),'.1f')+' pts' if nconf>nacc else 'well/under-calibrated'})")

    # ---- 2. COCKY-WRONG RATE -------------------------------------------------
    cw = cocky_wrong(P, G, hi=args.hi, lo=args.lo)
    print(f"\n--- 2. COCKY-WRONG RATE (confident = P>={args.hi} OR P<={args.lo}) ---")
    print(f"confident positions            = {cw['n_conf']}  "
          f"({100*cw['n_conf']/max(n_pos,1):.1f}% of all positions)")
    print(f"confident AND wrong            = {cw['n_wrong']}  "
          f"=> COCKY-WRONG RATE = {100*cw['wrong_rate']:.2f}%")
    print(f"  confident-FIRE (P>={args.hi}), gold=0 : {cw['wrong_fire']:6d} / {cw['n_fire']:6d} "
          f"= {100*cw['fire_wrong_rate']:.2f}%   (cocky over-fire)")
    print(f"  confident-NOFIRE (P<={args.lo}), gold=1: {cw['wrong_nofire']:6d} / {cw['n_nofire']:6d} "
          f"= {100*cw['nofire_wrong_rate']:.2f}%   (cocky missed boundary)")

    # ---- 3. PER-GENRE --------------------------------------------------------
    print("\n--- 3. PER-GENRE CALIBRATION (ranked: most untrustworthy high-confidence first) ---")
    rows = []
    for g in sorted(set(GEN.tolist())):
        m = GEN == g
        pg, gg = P[m], G[m]
        _, gece, gmce = reliability(pg, gg, n_bins=10)
        gcw = cocky_wrong(pg, gg, hi=args.hi, lo=args.lo)
        rows.append(dict(genre=g, n=int(m.sum()), brate=float(gg.mean()),
                         ece=gece, mce=gmce,
                         n_conf=gcw["n_conf"], cocky=gcw["wrong_rate"],
                         fire_wr=gcw["fire_wrong_rate"], nofire_wr=gcw["nofire_wrong_rate"]))
    # rank by cocky-wrong-rate (nan -> -1 so empty genres sink), tie-break ECE
    rows.sort(key=lambda r: (r["cocky"] if r["cocky"] == r["cocky"] else -1.0,
                             r["ece"]), reverse=True)
    print(f"{'genre':20s} {'pos':>7} {'b-rate':>7} {'ECE':>7} {'MCE':>7} "
          f"{'conf_n':>7} {'cocky%':>7} {'fireWR%':>8} {'nofireWR%':>10}")
    for r in rows:
        cocky = 100 * r["cocky"] if r["cocky"] == r["cocky"] else float("nan")
        fwr = 100 * r["fire_wr"] if r["fire_wr"] == r["fire_wr"] else float("nan")
        nwr = 100 * r["nofire_wr"] if r["nofire_wr"] == r["nofire_wr"] else float("nan")
        print(f"{r['genre']:20s} {r['n']:7d} {r['brate']:7.3f} {r['ece']:7.4f} "
              f"{r['mce']:7.4f} {r['n_conf']:7d} {cocky:7.2f} {fwr:8.2f} {nwr:10.2f}")

    # ---- 4. PER-TRIGGER ------------------------------------------------------
    print("\n--- 4. PER-TRIGGER: mean confidence of ERRORS at each failure site ---")
    print("  err_conf = mean P(boundary) over the ERROR positions at that trigger.")
    print("  For over-fire errors (gold 0) high err_conf means it fired cockily;")
    print("  for missed-boundary errors (gold 1) LOW err_conf means it dismissed cockily.")
    print(f"{'trigger':22s} {'n_pos':>7} {'n_gold':>7} {'n_err':>7} {'err_rate':>9} "
          f"{'mean_err_conf':>13} {'mean_all_conf':>13} {'cocky_err%':>11}")
    trig_order = ["after-number", "cloze-blank فراغ", "document-end",
                  "genealogy بن-chain", "verse-opening", "connective-initial"]
    for t in trig_order:
        s = trig.get(t)
        if not s or s["n_pos"] == 0:
            print(f"{t:22s} {'--':>7}  (no positions)")
            continue
        ec = np.asarray(s["err_confs"])
        ac = np.asarray(s["all_confs"])
        err_rate = s["n_err"] / s["n_pos"]
        mean_err_conf = float(ec.mean()) if ec.size else float("nan")
        mean_all_conf = float(ac.mean()) if ac.size else float("nan")
        # "cocky errors" = error positions that were ALSO confident (>=hi or <=lo)
        cocky_err = int(((ec >= args.hi) | (ec <= args.lo)).sum()) if ec.size else 0
        cocky_err_rate = cocky_err / ec.size if ec.size else float("nan")
        mec = f"{mean_err_conf:13.3f}" if mean_err_conf == mean_err_conf else f"{'--':>13}"
        cer = f"{100*cocky_err_rate:11.2f}" if cocky_err_rate == cocky_err_rate else f"{'--':>11}"
        print(f"{t:22s} {s['n_pos']:7d} {s['n_gold']:7d} {s['n_err']:7d} {err_rate:9.4f} "
              f"{mec} {mean_all_conf:13.3f} {cer}")

    print("\n# done")


if __name__ == "__main__":
    main()
