# -*- coding: utf-8 -*-
"""Cross-view alignment + label folding between an AraSeg punctuated view (PA)
and a punctuation-stripped view (NoPnx-NP).  ADDITIVE utility — verification
and reusable fold machinery for cross-view distillation.  CPU only.

Mechanics (verified on data/PA_train.jsonl vs data/NoPnx-NP_train.jsonl,
174 docs, 2026-07):

  * The NoPnx-NP token sequence of every doc is an exact ordered subsequence
    of the PA token sequence.  Alignment = greedy earliest-match subsequence
    injection (each NoPnx token matched to the first identical unconsumed PA
    token).  The stripped material is NOT purely lexical punctuation: besides
    punctuation glyphs and the "\\n" paragraph token it contains contextual
    enumeration/header material (MCQ option letters inside parentheses,
    legal-article headers like "المادة ( 28 )" including the bare number).
    Because stripped types (numbers, "المادة") also occur as kept tokens,
    alignment MUST be positional (subsequence matching), not token-class
    filtering.

  * Labels are boundary-AFTER-token.  Fold rule (verified exact, 174/174):
        group(j) = kept PA token a(j)  +  the run of stripped PA tokens
                   strictly between a(j) and a(j+1)   (doc tail included)
        folded_label[j] = OR of PA gold labels over group(j)
    Boundaries in a stripped run BEFORE the first kept token would be
    dropped (empirically zero such boundaries in train).

  * Teacher mapping: for per-token boundary probabilities from a PA model,
        teacher_prob[j] = REDUCE over group(j) of p_PA
    with REDUCE = max by default.  On binary labels max == OR == min(1,sum),
    so every such reduction reproduces the verified gold folding; `max` is
    the recommended one because stripped runs can be long (up to 62 tokens,
    e.g. dot leaders "........."), where noisy-OR / capped-sum inflate the
    folded probability from many small per-token masses.

Usage:
    python -m src.xview_align --src data/PA_train.jsonl --tgt data/NoPnx-NP_train.jsonl
    python -m src.xview_align --src ... --tgt ... --probs probs/pa_train.json --fold max --out folded.json

`--probs` (optional) is a JSON mapping doc_id -> list[float] with one
P(boundary) per SOURCE-view token; the script writes doc_id -> list[float]
with one teacher probability per TARGET-view token.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- io
def load_jsonl(path: str) -> List[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# --------------------------------------------------------------------- alignment
def align_subsequence(src_tokens: Sequence[str], tgt_tokens: Sequence[str]) -> Optional[List[int]]:
    """Greedy earliest-match injection of tgt into src.

    Returns keep[j] = index into src_tokens aligned to tgt_tokens[j],
    strictly increasing; None if tgt is not a subsequence of src.
    """
    keep: List[int] = []
    i, n = 0, len(src_tokens)
    for tok in tgt_tokens:
        while i < n and src_tokens[i] != tok:
            i += 1
        if i == n:
            return None
        keep.append(i)
        i += 1
    return keep


def groups_from_alignment(keep: Sequence[int], n_src: int) -> List[Tuple[int, int]]:
    """Half-open src-index ranges [start, end) per target token:
    kept token + the stripped run that FOLLOWS it (doc tail included)."""
    out = []
    for j, ki in enumerate(keep):
        end = keep[j + 1] if j + 1 < len(keep) else n_src
        out.append((ki, end))
    return out


# ----------------------------------------------------------------------- folding
def fold_labels(src_labels: Sequence[int], keep: Sequence[int], n_src: int) -> List[int]:
    """OR of boundary-after-token gold labels over each group (verified rule)."""
    return [1 if any(src_labels[t] for t in range(s, e)) else 0
            for s, e in groups_from_alignment(keep, n_src)]


def fold_probs(src_probs: Sequence[float], keep: Sequence[int], n_src: int,
               mode: str = "max") -> List[float]:
    """Fold per-src-token P(boundary) to per-tgt-token teacher probability.

    mode='max'      : max over group          (recommended; robust to long runs)
    mode='noisy-or' : 1 - prod(1 - p)         (independence assumption)
    mode='sum-cap'  : min(1, sum(p))          (mass-additive)
    All three reduce to the verified gold OR-fold on {0,1} inputs.
    """
    out = []
    for s, e in groups_from_alignment(keep, n_src):
        ps = src_probs[s:e]
        if mode == "max":
            v = max(ps)
        elif mode == "noisy-or":
            v = 1.0
            for p in ps:
                v *= (1.0 - p)
            v = 1.0 - v
        elif mode == "sum-cap":
            v = min(1.0, float(sum(ps)))
        else:
            raise ValueError(f"unknown fold mode {mode!r}")
        out.append(float(v))
    return out


# ------------------------------------------------------------------ verification
def verify(src_docs: List[dict], tgt_docs: List[dict]) -> Dict:
    """Align + fold every doc; return a report dict (printed by main)."""
    src_by_id = {d["doc_id"]: d for d in src_docs}
    tgt_by_id = {d["doc_id"]: d for d in tgt_docs}
    rep: Dict = {
        "n_src_docs": len(src_docs), "n_tgt_docs": len(tgt_docs),
        "doc_ids_match": set(src_by_id) == set(tgt_by_id),
        "align_ok_docs": 0, "align_fail_docs": [],
        "fold_ok_docs": 0, "fold_fail_docs": [],
        "fold_mismatch_positions": 0,
        "stripped_inventory": Counter(),
        "n_src_tokens": 0, "n_tgt_tokens": 0,
        "src_boundaries": 0, "tgt_boundaries": 0, "folded_boundaries": 0,
        "boundaries_on_stripped": 0, "boundaries_on_kept": 0,
        "merged_groups_by_multiplicity": Counter(),
        "prefix_stripped_tokens": 0, "prefix_dropped_boundaries": 0,
        "max_stripped_run": 0, "groups_with_stripped_run": 0,
    }
    for did in src_by_id:
        if did not in tgt_by_id:
            continue
        A, LA = src_by_id[did]["tokens"], src_by_id[did]["labels"]
        B, LB = tgt_by_id[did]["tokens"], tgt_by_id[did]["labels"]
        rep["n_src_tokens"] += len(A)
        rep["n_tgt_tokens"] += len(B)
        rep["src_boundaries"] += sum(LA)
        rep["tgt_boundaries"] += sum(LB)
        keep = align_subsequence(A, B)
        if keep is None:
            rep["align_fail_docs"].append(did)
            continue
        rep["align_ok_docs"] += 1
        kept_set = set(keep)
        for i, tok in enumerate(A):
            if i not in kept_set:
                rep["stripped_inventory"][tok] += 1
        for i, lab in enumerate(LA):
            if lab:
                if i in kept_set:
                    rep["boundaries_on_kept"] += 1
                else:
                    rep["boundaries_on_stripped"] += 1
        first = keep[0]
        rep["prefix_stripped_tokens"] += first
        rep["prefix_dropped_boundaries"] += sum(LA[:first])
        for s, e in groups_from_alignment(keep, len(A)):
            run = e - s - 1
            if run > 0:
                rep["groups_with_stripped_run"] += 1
                rep["max_stripped_run"] = max(rep["max_stripped_run"], run)
            n1 = sum(LA[s:e])
            if n1 >= 2:
                rep["merged_groups_by_multiplicity"][n1] += 1
        folded = fold_labels(LA, keep, len(A))
        rep["folded_boundaries"] += sum(folded)
        if folded == LB:
            rep["fold_ok_docs"] += 1
        else:
            bad = [j for j in range(len(B)) if folded[j] != LB[j]]
            rep["fold_mismatch_positions"] += len(bad)
            rep["fold_fail_docs"].append((did, bad[:10]))
    return rep


def print_report(rep: Dict, top_inventory: int = 40) -> None:
    print(f"docs: src={rep['n_src_docs']} tgt={rep['n_tgt_docs']} "
          f"doc_id sets match={rep['doc_ids_match']}")
    print(f"[1] token alignment (tgt is ordered subsequence of src): "
          f"{rep['align_ok_docs']}/{rep['n_src_docs']} docs OK; "
          f"failures: {rep['align_fail_docs'] or 'none'}")
    print(f"    src tokens={rep['n_src_tokens']}  tgt tokens={rep['n_tgt_tokens']}  "
          f"stripped={sum(rep['stripped_inventory'].values())}")
    inv = rep["stripped_inventory"]
    print(f"    stripped inventory: {len(inv)} types; top {top_inventory}:")
    for tok, c in inv.most_common(top_inventory):
        print(f"      {tok!r}: {c}")
    print(f"[2] label folding (OR over kept token + following stripped run): "
          f"{rep['fold_ok_docs']}/{rep['align_ok_docs']} docs EXACT; "
          f"mismatching positions={rep['fold_mismatch_positions']}; "
          f"failures: {[d for d, _ in rep['fold_fail_docs']] or 'none'}")
    print(f"    src boundaries={rep['src_boundaries']}  folded={rep['folded_boundaries']}  "
          f"tgt gold={rep['tgt_boundaries']}")
    print(f"    boundaries on stripped tokens={rep['boundaries_on_stripped']}  "
          f"on kept tokens={rep['boundaries_on_kept']}")
    print(f"    merged groups (>=2 src boundaries -> 1): "
          f"{dict(rep['merged_groups_by_multiplicity']) or 'none'}")
    print(f"    prefix stripped tokens={rep['prefix_stripped_tokens']}  "
          f"prefix-dropped boundaries={rep['prefix_dropped_boundaries']}")
    print(f"    groups with stripped run={rep['groups_with_stripped_run']}  "
          f"max run length={rep['max_stripped_run']}")


# --------------------------------------------------------------------------- cli
def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):  # Windows cp1252 console guard
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="punctuated-view JSONL (e.g. data/PA_train.jsonl)")
    ap.add_argument("--tgt", required=True, help="stripped-view JSONL (e.g. data/NoPnx-NP_train.jsonl)")
    ap.add_argument("--probs", default=None,
                    help="optional JSON {doc_id: [P(boundary) per src token]} to fold")
    ap.add_argument("--fold", default="max", choices=["max", "noisy-or", "sum-cap"],
                    help="probability fold reduction (default: max)")
    ap.add_argument("--out", default=None, help="where to write folded probs JSON")
    args = ap.parse_args()

    src_docs = load_jsonl(args.src)
    tgt_docs = load_jsonl(args.tgt)
    rep = verify(src_docs, tgt_docs)
    print_report(rep)

    if args.probs:
        with open(args.probs, encoding="utf-8") as f:
            probs = json.load(f)
        src_by_id = {d["doc_id"]: d for d in src_docs}
        tgt_by_id = {d["doc_id"]: d for d in tgt_docs}
        folded_all = {}
        for did, p in probs.items():
            A = src_by_id[did]["tokens"]
            B = tgt_by_id[did]["tokens"]
            if len(p) != len(A):
                raise ValueError(f"{did}: {len(p)} probs vs {len(A)} src tokens")
            keep = align_subsequence(A, B)
            if keep is None:
                raise ValueError(f"{did}: target not a subsequence of source")
            folded_all[did] = fold_probs(p, keep, len(A), mode=args.fold)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(folded_all, f)
            print(f"wrote folded teacher probs ({args.fold}) for "
                  f"{len(folded_all)} docs -> {args.out}")


if __name__ == "__main__":
    main()
