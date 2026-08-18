"""Offline cross-view teacher cache builder (PA teacher -> NoPnx-NP student grid).

Runs the PA boundary teacher ONCE over the PA train view, folds per-token
P(boundary) onto the NoPnx-NP token grid using the alignment / label-routing
semantics of record (src/augment.py: deletion_mask + project_labels), and
writes an immutable cache for the cross-view distillation trainer
(src/xview_distill_train.py). Also writes model-free punctuation-after
targets for the auxiliary head. The student trainer never loads the teacher
model; it reads only this cache.

ONE fold implementation of record: this module imports augment.deletion_mask
and augment.project_labels and derives the shared position->slot routing from
project_labels' semantics; guard g4 asserts per doc that the routing-based OR
fold, project_labels itself, and fold_probs on gold indicators all reproduce
the official NoPnx-NP gold bit-exactly BEFORE anything is written.
src/xview_align.py is retained as verification evidence only and must never
be imported by the pipeline.

Closed-track guards (all hard aborts; nothing is written on failure):
  g0  teacher checkpoint basename must contain "_base_" and must NOT contain
      "_trdev_" (trdev checkpoints saw dev -> using one would launder
      dev-training through the teacher; rule-illegal).
  g1  --src and --tgt basenames must end in "_train.jsonl" and must not
      contain "trdev" (teacher inference on TRAIN docs only; dev/test caches
      are refused).
  g2  both files hold exactly --expect-docs docs and identical doc_id sets.
  g3  --out dir must not already exist (runs/ is append-only; no --force).
  g4  gold round-trip on ALL docs before writing anything (see above).

CPU-only by design: the cache is built once (~1.1k windows) and the GPU is
reserved for the frontier battery. No torch.cuda call is ever made.

  PYTHONUTF8=1 CUDA_VISIBLE_DEVICES= python src/xview_teacher_cache.py \
      --teacher runs/opus_trdev_2026-07-08/PA_base_s42 \
      --src data/PA_train.jsonl --tgt data/NoPnx-NP_train.jsonl \
      --out runs/xview_teacher_2026-07-10/
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from collections import Counter

from augment import deletion_mask, project_labels
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl

# Punctuation-after classes for the auxiliary head (ruling of the build order
# of record, 2026-07-10): classify the FIRST stripped token after the kept
# word that is not "\n" ("\n" is not punctuation; a stripped paragraph alone
# is NONE -- the KD signal already carries paragraph information).
PUNCT_STRONG = {".", "؟", "!", "؛"}   # class 1
PUNCT_COMMA = {",", "،"}              # class 2
# class 3 = any other first stripped token whose chars are all non-alphanumeric
# class 0 = empty run / only "\n" / first non-"\n" stripped token contains
#           an alphanumeric char (e.g. MCQ option letters, article numbers:
#           stripped, but not punctuation)


def route_map(tokens: list, keep: list) -> list:
    """PA position -> NoPnx slot index (-1 = dropped).

    Exactly project_labels' routing: a kept position maps to its own slot; a
    deleted position maps to the slot of the nearest PRECEDING kept non-"\\n"
    token, else the NEXT kept non-"\\n" token, else is dropped. Guard g4
    asserts equivalence with augment.project_labels on every doc.
    """
    kept_idx = [i for i in range(len(tokens)) if keep[i]]
    pos_of = {i: k for k, i in enumerate(kept_idx)}
    route = [-1] * len(tokens)
    for i in range(len(tokens)):
        if keep[i]:
            route[i] = pos_of[i]
            continue
        j = i - 1
        while j >= 0 and not (keep[j] and tokens[j] != PARAGRAPH_TOKEN):
            j -= 1
        if j < 0:
            j = i + 1
            while j < len(tokens) and not (keep[j] and tokens[j] != PARAGRAPH_TOKEN):
                j += 1
            if j >= len(tokens):
                continue  # dropped (no kept non-\n token anywhere)
        route[i] = pos_of[j]
    return route


def fold_probs(probs: list, route: list, n_slots: int, mode: str = "max") -> list:
    """Combine per-PA-position probabilities into per-NoPnx-slot probabilities.

    max      : order-invariant OR analog (default; conservative on long runs)
    noisy-or : 1 - prod(1 - p) over the slot's group
    sum-cap  : min(sum(p), 1) over the slot's group
    All three reduce exactly to the OR fold on {0,1} inputs.
    """
    if mode == "max":
        out = [0.0] * n_slots
        for i, p in enumerate(probs):
            k = route[i]
            if k >= 0 and p > out[k]:
                out[k] = p
        return out
    if mode == "noisy-or":
        prod = [1.0] * n_slots
        for i, p in enumerate(probs):
            k = route[i]
            if k >= 0:
                prod[k] *= 1.0 - p
        return [1.0 - q for q in prod]
    if mode == "sum-cap":
        acc = [0.0] * n_slots
        for i, p in enumerate(probs):
            k = route[i]
            if k >= 0:
                acc[k] += p
        return [min(s, 1.0) for s in acc]
    raise ValueError(f"unknown fold mode {mode!r}")


def or_fold_labels(labels: list, route: list, n_slots: int) -> list:
    """OR of {0,1} labels per slot via the shared routing (g4 comparator)."""
    out = [0] * n_slots
    for i, lab in enumerate(labels):
        if lab == 1 and route[i] >= 0:
            out[route[i]] = 1
    return out


def punct_after_classes(tokens: list, keep: list) -> list:
    """Per kept slot: class of the FIRST non-"\\n" stripped token that follows
    it (before the next kept token; doc tail included). Model-free."""
    kept_idx = [i for i in range(len(tokens)) if keep[i]]
    out = []
    for k, i in enumerate(kept_idx):
        end = kept_idx[k + 1] if k + 1 < len(kept_idx) else len(tokens)
        cls = 0
        for j in range(i + 1, end):
            t = tokens[j]
            if t == PARAGRAPH_TOKEN:
                continue  # "\n" runs skipped; alone -> NONE
            if t in PUNCT_STRONG:
                cls = 1
            elif t in PUNCT_COMMA:
                cls = 2
            elif all(not c.isalnum() for c in t):
                cls = 3
            else:
                cls = 0  # stripped but alphanumeric -> not punctuation
            break
        out.append(cls)
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_fail(name: str, msg: str) -> None:
    print(f"GUARD {name} FAILED: {msg}")
    print("HARD ABORT -- nothing was written.")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True,
                    help="dir of the PA teacher checkpoint (*_base_* only)")
    ap.add_argument("--src", required=True, help="teacher-view train jsonl (PA)")
    ap.add_argument("--tgt", required=True, help="student-view train jsonl (NoPnx-NP)")
    ap.add_argument("--out", required=True, help="output dir (must not exist)")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--expect-docs", type=int, default=174)
    ap.add_argument("--fold", default="max", choices=["max", "noisy-or", "sum-cap"],
                    help="per-slot probability reduction (default: max, the OR analog)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- g0: teacher provenance -------------------------------------------
    tb = os.path.basename(os.path.normpath(args.teacher))
    if "_trdev_" in tb:
        guard_fail("g0", f"teacher basename {tb!r} contains '_trdev_' -- this "
                         "checkpoint was trained on train+dev; using it as a "
                         "teacher would launder dev-training. FORBIDDEN.")
    if "_base_" not in tb:
        guard_fail("g0", f"teacher basename {tb!r} does not contain '_base_' "
                         "(only train-only *_base_* checkpoints are legal teachers)")
    print(f"g0 PASS: teacher basename {tb!r} matches *_base_* and is not *_trdev_*")

    # ---- g1: train-only inputs --------------------------------------------
    for role, path in (("--src", args.src), ("--tgt", args.tgt)):
        b = os.path.basename(path)
        if not b.endswith("_train.jsonl"):
            guard_fail("g1", f"{role} basename {b!r} does not end in '_train.jsonl' "
                             "(teacher inference on TRAIN docs only)")
        if "trdev" in b:
            guard_fail("g1", f"{role} basename {b!r} contains 'trdev' (train+dev "
                             "file -- dev must never pass through the teacher)")
    print(f"g1 PASS: src={os.path.basename(args.src)!r} tgt={os.path.basename(args.tgt)!r} "
          "are *_train.jsonl and not trdev")

    # ---- g3: append-only out dir (checked early; re-checked at mkdir) -----
    out_dir = os.path.normpath(args.out)
    if os.path.exists(out_dir):
        guard_fail("g3", f"out dir {out_dir!r} already exists (runs/ is append-only; "
                         "there is no --force)")
    print(f"g3 PASS: out dir {out_dir!r} does not exist")

    src_task = os.path.basename(args.src)[: -len("_train.jsonl")]
    tgt_task = os.path.basename(args.tgt)[: -len("_train.jsonl")]

    # ---- g2: doc counts + id sets ------------------------------------------
    src_docs = load_jsonl(args.src)
    tgt_docs = load_jsonl(args.tgt)
    if len(src_docs) != args.expect_docs or len(tgt_docs) != args.expect_docs:
        guard_fail("g2", f"doc counts src={len(src_docs)} tgt={len(tgt_docs)} != "
                         f"expected {args.expect_docs}")
    src_ids = [d["doc_id"] for d in src_docs]
    tgt_by_id = {d["doc_id"]: d for d in tgt_docs}
    if set(src_ids) != set(tgt_by_id):
        only_s = sorted(set(src_ids) - set(tgt_by_id))[:3]
        only_t = sorted(set(tgt_by_id) - set(src_ids))[:3]
        guard_fail("g2", f"doc_id sets differ (src-only e.g. {only_s}, tgt-only e.g. {only_t})")
    print(f"g2 PASS: {len(src_docs)} docs in each file, identical doc_id sets")

    # ---- g4: gold round-trip through the ONE fold implementation ----------
    aligned = {}  # doc_id -> dict(keep, route, n_slots, gold)
    pa_boundaries = 0
    boundary_on_stripped = 0
    folded_boundaries = 0
    dropped_label1 = 0
    merges_2to1 = Counter()  # multiplicity of PA label-1s per slot, where >= 2
    for d in src_docs:
        t = tgt_by_id[d["doc_id"]]
        toks, labs, gold = d["tokens"], list(d["labels"]), list(t["labels"])
        try:
            keep = deletion_mask(toks, t["tokens"])
        except AssertionError:
            guard_fail("g4", f"doc {d['doc_id']}: tgt tokens are not a subsequence "
                             "of src tokens")
        route = route_map(toks, keep)
        n_slots = sum(keep)
        if n_slots != len(t["tokens"]) or n_slots != len(gold):
            guard_fail("g4", f"doc {d['doc_id']}: slot count {n_slots} != "
                             f"tgt tokens {len(t['tokens'])} / labels {len(gold)}")
        proj = project_labels(toks, labs, keep)          # fold of record (labels)
        via_route = or_fold_labels(labs, route, n_slots)  # same routing, OR
        via_probs = [int(p >= 0.5) for p in
                     fold_probs([float(x) for x in labs], route, n_slots, args.fold)]
        if proj != gold:
            guard_fail("g4", f"doc {d['doc_id']}: augment.project_labels does not "
                             "reproduce official gold")
        if via_route != gold:
            guard_fail("g4", f"doc {d['doc_id']}: routing-based OR fold != official gold")
        if via_probs != gold:
            guard_fail("g4", f"doc {d['doc_id']}: fold_probs({args.fold}) on gold "
                             "indicators != official gold")
        pa_boundaries += sum(labs)
        boundary_on_stripped += sum(1 for i, l in enumerate(labs) if l == 1 and not keep[i])
        folded_boundaries += sum(gold)
        dropped_label1 += sum(1 for i, l in enumerate(labs) if l == 1 and route[i] < 0)
        per_slot = Counter(route[i] for i, l in enumerate(labs) if l == 1 and route[i] >= 0)
        for m in per_slot.values():
            if m >= 2:
                merges_2to1[m] += 1
        aligned[d["doc_id"]] = {"keep": keep, "route": route,
                                "n_slots": n_slots, "gold": gold}
    n_docs = len(src_docs)
    g4_att = (f"PASSED {n_docs}/{n_docs} docs: augment.project_labels == routing-OR == "
              f"fold_probs({args.fold}) on gold indicators == official {tgt_task} gold, "
              "bit-exact")
    print(f"g4 PASS: gold round-trip {g4_att}")
    print(f"   PA gold boundaries: {pa_boundaries}; on stripped tokens: "
          f"{boundary_on_stripped} ({100.0 * boundary_on_stripped / max(pa_boundaries, 1):.1f}%)")
    print(f"   folded gold boundaries: {folded_boundaries}; merge multiplicities "
          f"(>=2 PA boundaries -> 1 slot): {dict(sorted(merges_2to1.items()))}; "
          f"label-1 dropped in prefix runs: {dropped_label1}")

    # ---- punctuation-after targets (model-free) ----------------------------
    punct = {d["doc_id"]: punct_after_classes(d["tokens"], aligned[d["doc_id"]]["keep"])
             for d in src_docs}
    punct_freq = Counter(c for v in punct.values() for c in v)

    # ---- teacher inference (CPU; canonical decode via predict.predict_doc) --
    print(f"loading teacher {args.teacher!r} on CPU ...")
    from transformers import AutoModelForTokenClassification, AutoTokenizer  # lazy

    from predict import predict_doc  # canonical window/overlap stitching

    device = "cpu"  # by design; the GPU belongs to the frontier battery
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForTokenClassification.from_pretrained(args.teacher).to(device).eval()

    probs_out = {}  # doc_id -> folded probs (full precision, on tgt grid)
    t0 = time.time()
    for n, d in enumerate(src_docs, 1):
        a = aligned[d["doc_id"]]
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc(words, tokenizer, model, device,
                        args.window, args.overlap, args.max_length)
        p = [0.0 if w == PARAGRAPH_TOKEN else float(x)          # mirror predict.py's
             for w, x in zip(d["tokens"], p)]                    # forced 0 at [PAR]
        probs_out[d["doc_id"]] = fold_probs(p, a["route"], a["n_slots"], args.fold)
        if n % 25 == 0 or n == n_docs:
            print(f"   {n}/{n_docs} docs, {time.time() - t0:.0f}s elapsed")

    # ---- decision numbers ---------------------------------------------------
    lo = mid = hi = 0
    agree = pos_agree = pos_tot = neg_agree = neg_tot = 0
    for d in src_docs:
        gold = aligned[d["doc_id"]]["gold"]
        for pv, g in zip(probs_out[d["doc_id"]], gold):
            if pv < 0.05:
                lo += 1
            elif pv > 0.95:
                hi += 1
            else:
                mid += 1
            pred = int(pv >= 0.5)
            agree += int(pred == g)
            if g == 1:
                pos_tot += 1
                pos_agree += int(pred == 1)
            else:
                neg_tot += 1
                neg_agree += int(pred == 0)
    n_tok = lo + mid + hi

    # ---- write (only now; guards all green) ---------------------------------
    os.makedirs(out_dir)  # no exist_ok: re-asserts g3 at write time
    probs_name = f"teacher_{src_task.lower()}_fold_{tgt_task}_train.jsonl"
    punct_name = f"punct_after_{tgt_task}_train.jsonl"
    probs_path = os.path.join(out_dir, probs_name)
    punct_path = os.path.join(out_dir, punct_name)
    with open(probs_path, "w", encoding="utf-8") as f:
        for d in src_docs:
            a = aligned[d["doc_id"]]
            n_deleted_folded = sum(1 for i in range(len(d["tokens"]))
                                   if not a["keep"][i] and a["route"][i] >= 0)
            f.write(json.dumps({
                "doc_id": d["doc_id"],
                "n_tokens": a["n_slots"],
                "teacher_probs": [round(x, 6) for x in probs_out[d["doc_id"]]],
                "n_pa_tokens": len(d["tokens"]),
                "n_deleted_folded": n_deleted_folded,
            }, ensure_ascii=False) + "\n")
    with open(punct_path, "w", encoding="utf-8") as f:
        for d in src_docs:
            f.write(json.dumps({
                "doc_id": d["doc_id"],
                "n_tokens": aligned[d["doc_id"]]["n_slots"],
                "punct_after": punct[d["doc_id"]],
            }, ensure_ascii=False) + "\n")

    decision_numbers = {
        "pa_gold_boundaries": pa_boundaries,
        "pa_gold_boundaries_on_stripped": boundary_on_stripped,
        "folded_gold_boundaries": folded_boundaries,
        "merge_multiplicities": {str(k): v for k, v in sorted(merges_2to1.items())},
        "sharpness_histogram": {
            "n_tokens": n_tok,
            "p_lt_0.05": lo, "p_0.05_to_0.95": mid, "p_gt_0.95": hi,
            "frac_lt_0.05": round(lo / n_tok, 6),
            "frac_0.05_to_0.95": round(mid / n_tok, 6),
            "frac_gt_0.95": round(hi / n_tok, 6),
        },
        "teacher_vs_folded_gold_agreement_at_0.5": round(agree / n_tok, 6),
        "agreement_on_gold_1": round(pos_agree / max(pos_tot, 1), 6),
        "agreement_on_gold_0": round(neg_agree / max(neg_tot, 1), 6),
        "punct_class_frequencies": {str(k): punct_freq.get(k, 0) for k in (0, 1, 2, 3)},
    }
    meta = {
        "created": datetime.date.today().isoformat(),
        "teacher": args.teacher,
        "teacher_basename": tb,
        "teacher_task": src_task,
        "student_task": tgt_task,
        "src_jsonl": args.src, "src_sha256": sha256_file(args.src),
        "tgt_jsonl": args.tgt, "tgt_sha256": sha256_file(args.tgt),
        "n_docs": n_docs,
        "window": args.window, "overlap": args.overlap, "max_length": args.max_length,
        "device": device,
        "fold_mode": args.fold,
        "fold_rule": ("augment.deletion_mask + project_labels routing (kept token -> "
                      "own slot; deleted token -> nearest preceding kept non-\\n token, "
                      "else next kept non-\\n token, else dropped); per-slot probability "
                      f"reduction = {args.fold}"),
        "par_forced_zero": True,
        "teacher_probs_rounded_to": 6,
        "punct_classes": {
            "0": "none (empty run / only \\n / first non-\\n stripped token alphanumeric)",
            "1": "strong: . ؟ ! ؛",
            "2": "comma: , ،",
            "3": "other (first non-\\n stripped token, all chars non-alphanumeric)",
        },
        "guards": {
            "g0": f"PASS: teacher basename {tb!r} matches *_base_*, not *_trdev_*",
            "g1": "PASS: src/tgt basenames end in _train.jsonl, no trdev",
            "g2": f"PASS: {n_docs} docs each, identical doc_id sets",
            "g3": f"PASS: out dir {out_dir!r} freshly created",
            "g4": g4_att,
        },
        "round_trip_guard": g4_att,
        "decision_numbers": decision_numbers,
        "outputs": {probs_name: sha256_file(probs_path),
                    punct_name: sha256_file(punct_path)},
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- report -------------------------------------------------------------
    print("\n=== DECISION NUMBERS (paste into the GPU-approval request) ===")
    print(f"PA gold boundaries {pa_boundaries}; ON stripped tokens "
          f"{boundary_on_stripped} ({100.0 * boundary_on_stripped / max(pa_boundaries, 1):.1f}%); "
          f"folded gold {folded_boundaries}")
    print(f"SHARPNESS HISTOGRAM of folded teacher probs (n={n_tok}):")
    print(f"   p < 0.05    : {lo:7d}  ({100.0 * lo / n_tok:6.2f}%)")
    print(f"   0.05 - 0.95 : {mid:7d}  ({100.0 * mid / n_tok:6.2f}%)   <-- KD dark-knowledge mass")
    print(f"   p > 0.95    : {hi:7d}  ({100.0 * hi / n_tok:6.2f}%)")
    print(f"teacher-vs-folded-gold agreement @0.5: {agree / n_tok:.4f} "
          f"(on gold-1: {pos_agree / max(pos_tot, 1):.4f}, "
          f"on gold-0: {neg_agree / max(neg_tot, 1):.4f})")
    print("punct-after class frequencies: " +
          ", ".join(f"{k}={punct_freq.get(k, 0)}" for k in (0, 1, 2, 3)) +
          f"  (0=none 1=strong 2=comma 3=other; total {sum(punct_freq.values())})")
    print("\nwrote:")
    for pth in (probs_path, punct_path, meta_path):
        print(f"   {pth}")
    print("cache is IMMUTABLE (runs/ append-only). Done.")


if __name__ == "__main__":
    main()
