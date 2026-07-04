"""Blind-day input preparation + validation. The biggest risk to the blind score
is a malformed conversion of the organisers' inputs — this script makes that
step mechanical and LOUD on any anomaly.

Usage (after downloading blind inputs):
  # from a HuggingFace dataset dump / jsonl with doc_id+tokens fields:
  python prepare_blind.py --task PA --in raw/PA_blind.jsonl --out blind/PA_test.jsonl
  # from raw text files (one doc per file, whitespace tokens, real newlines):
  python prepare_blind.py --task PA --raw-dir raw/PA_docs --out blind/PA_test.jsonl
  # cross-variant consistency check once all four exist:
  python prepare_blind.py --check-all --blind-dir blind

Validations (hard failures unless noted):
  per file : doc count > 0, unique doc_ids, no empty/whitespace tokens,
             Arabic-character ratio sane, token counts in plausible range (warn),
             \n tokens present iff PA-variant, punctuation tokens present iff
             punctuated variant (warn if ratio unusual)
  check-all: same doc_ids across all 4 variants; NP == PA minus \n tokens;
             NoPnx-PA == PA minus punctuation; NoPnx-NP == NP minus punctuation
             (subsequence identity — catches tokenisation drift immediately)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

AR = re.compile(r"[؀-ۿ]")
PUNCT = set(".,!?؟،؛:…»«()\"'-—")


def fail(msg):
    print(f"VALIDATION FAILED: {msg}")
    sys.exit(1)


def warn(msg):
    print(f"  warning: {msg}")


def load_any(path):
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = r.get("doc_id") or r.get("id") or r.get("document_id")
            toks = r.get("tokens")
            if toks is None and "text" in r:
                toks = []
                for seg in r["text"].split("\n"):
                    toks.extend(seg.split())
                    toks.append("\n")
                toks = toks[:-1]
            if did is None or toks is None:
                fail(f"row missing doc_id/tokens in {path}: keys={list(r)[:6]}")
            docs.append({"doc_id": str(did), "tokens": [str(t) for t in toks]})
    return docs


def load_raw_dir(d):
    docs = []
    for fn in sorted(os.listdir(d)):
        text = open(os.path.join(d, fn), encoding="utf-8").read()
        toks = []
        for seg in text.split("\n"):
            toks.extend(seg.split())
            toks.append("\n")
        while toks and toks[-1] == "\n":
            toks.pop()
        docs.append({"doc_id": os.path.splitext(fn)[0], "tokens": toks})
    return docs


def validate(task, docs):
    if not docs:
        fail("no documents")
    ids = [d["doc_id"] for d in docs]
    if len(set(ids)) != len(ids):
        fail("duplicate doc_ids")
    pa_variant = task.endswith("PA")
    punct_variant = not task.startswith("NoPnx")
    n_par = n_punct = n_tok = 0
    for d in docs:
        for t in d["tokens"]:
            if t == "":
                fail(f"empty token in {d['doc_id']}")
            if t != "\n" and any(c.isspace() for c in t):
                fail(f"whitespace inside token {t!r} in {d['doc_id']}")
            n_tok += 1
            n_par += t == "\n"
            n_punct += t in PUNCT
        if not (30 <= len(d["tokens"]) <= 20000):
            warn(f"{d['doc_id']}: unusual length {len(d['tokens'])}")
    ar_ratio = sum(1 for d in docs for t in d["tokens"] if AR.search(t)) / n_tok
    if ar_ratio < 0.5:
        fail(f"Arabic-token ratio {ar_ratio:.2f} < 0.5 — wrong file or encoding?")
    if pa_variant and n_par == 0:
        fail("PA-variant but no \\n tokens found")
    if not pa_variant and n_par > 0:
        fail("no-paragraph variant but \\n tokens present")
    if punct_variant and n_punct / n_tok < 0.01:
        warn(f"punctuated variant but punct ratio only {n_punct/n_tok:.3f}")
    if not punct_variant and n_punct / n_tok > 0.002:
        warn(f"NoPnx variant but punct ratio {n_punct/n_tok:.3f}")
    print(f"OK {task}: {len(docs)} docs, {n_tok} tokens, par={n_par}, "
          f"punct={n_punct}, arabic={ar_ratio:.2f}")


def strip_par(seq):
    return [t for t in seq if t != "\n"]


def is_subseq(small, big):
    it = iter(big)
    return all(tok in it for tok in (iter(small),)) if False else _subseq(small, big)


def _subseq(small, big):
    j = 0
    for t in big:
        if j < len(small) and small[j] == t:
            j += 1
    return j == len(small)


def check_all(blind_dir):
    # Invariants (verified to hold on the public test set):
    #   NP == PA minus \n tokens (exact equality)
    #   NoPnx-NP == NoPnx-PA minus \n tokens (exact equality)
    #   NoPnx-PA is an order-preserving SUBSEQUENCE of PA (organisers drop
    #   punctuation AND assorted marker tokens — digits, enumeration letters)
    D = {}
    for t in ["PA", "NoPnx-PA", "NP", "NoPnx-NP"]:
        p = os.path.join(blind_dir, f"{t}_test.jsonl")
        if not os.path.exists(p):
            fail(f"missing {p}")
        D[t] = {d["doc_id"]: d["tokens"] for d in load_any(p)}
        validate(t, [{"doc_id": k, "tokens": v} for k, v in D[t].items()])
    base = set(D["PA"])
    for t in ["NoPnx-PA", "NP", "NoPnx-NP"]:
        if set(D[t]) != base:
            warn(f"{t}: doc_id set differs from PA (OK if organisers de-aliased the blind IDs)")
            return
    bad = 0
    for did in base:
        if strip_par(D["PA"][did]) != D["NP"][did]:
            print(f"  MISMATCH NP != PA-minus-par: {did}"); bad += 1
        if strip_par(D["NoPnx-PA"][did]) != D["NoPnx-NP"][did]:
            print(f"  MISMATCH NoPnx-NP != NoPnx-PA-minus-par: {did}"); bad += 1
        if not _subseq(D["NoPnx-PA"][did], D["PA"][did]):
            print(f"  MISMATCH NoPnx-PA not a subsequence of PA: {did}"); bad += 1
    if bad:
        fail(f"{bad} cross-variant mismatches — tokenisation drift, DO NOT SUBMIT")
    print(f"CROSS-VARIANT OK: {len(base)} aligned docs, all invariants hold")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--raw-dir")
    ap.add_argument("--out")
    ap.add_argument("--check-all", action="store_true")
    ap.add_argument("--blind-dir", default="blind")
    a = ap.parse_args()
    if a.check_all:
        check_all(a.blind_dir)
        return
    if not (a.task and a.out and (a.inp or a.raw_dir)):
        ap.error("need --task, --out and one of --in/--raw-dir (or --check-all)")
    docs = load_any(a.inp) if a.inp else load_raw_dir(a.raw_dir)
    validate(a.task, docs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
