# -*- coding: utf-8 -*-
"""CPU smoke suite for the cross-view distillation build (xview).

Corpus-independent: synthetic documents only — no data/, no GPU, no network
(the s4 tokenizer comes from the local HF cache; run with HF_HUB_OFFLINE=1).

Sections (build order of record, 2026-07-10, step 2):
  s1  FOLD == LABEL-PROJECTION. 200 random synthetic PA/NoPnx doc pairs with
      adversarial strip patterns (doc-prefix strips incl. forward-folds,
      all-stripped sentences, stripped runs up to length 60, kept "\\n"
      adjacency, fully-stripped docs). The PIPELINE fold
      (xview_teacher_cache.route_map + or_fold_labels + fold_probs in all
      three modes, applied to {0,1} gold indicators) must equal
      augment.project_labels (the label-movement rule of record) bit-exactly
      on every doc.
  s2  PUNCT-CLASS DERIVATION TABLE for xview_teacher_cache.punct_after_classes,
      incl. the spec-pinned newline-run edge cases (["\\n", "."] -> STRONG with
      the newline skipped; only-"\\n" run -> NONE), tail runs, first-token
      precedence, alphanumeric-first -> NONE; plus an independent re-derivation
      on 50 random docs.
  s3  KD MATH IDENTITIES (pure torch, the exact loss of spec section 3):
      z_s == z_t -> KD term exactly 0; T=1 reduces to the plain binary KL;
      the T^2 factor is applied and matches the large-T asymptote
      (z_t - z_s)^2 / 8; the gate selects exactly {valid & teacher_p >= gate}
      (gate 0 -> all valid); entropy weights H2(q)/ln2 are in [0,1] and
      maximal at q = 0.5 (all-0.5 weights reduce to the plain mean); clamps
      keep everything finite at q in {0, 1} and |z_s| = 100; an empty
      selection returns a grad-safe 0 (grad_fn present, backward OK).
  s4  PLUMBING (real tokenizer, offline): per-word teacher_p / punct_cls ride
      the exact make_examples/encode/collate pipeline of record
      (consistency_train.py window/stride loop, labels on the LAST subword,
      [PAR] -> sentinel) and land on EXACTLY the positions where
      labels != -100, in windowed AND padded batches, with and without
      truncation; sentinels (-1.0 / -100) are never selected; poisoned [PAR]
      values are always replaced by sentinels; a global word-index round-trip
      proves each value sits on the right word. Label placement is asserted
      identical to consistency_train.encode on the same windows.

s3/s4 validate the REFERENCE math/plumbing that src/xview_distill_train.py
(build step 3) must implement; the deferred GPU smokes s5-s8 then close the
loop bit-for-bit against the actual trainer. Nothing here reads data/ or
touches the GPU (CUDA_VISIBLE_DEVICES is forced empty unless --gpu is passed,
reserved for the future s5-s8 sections).

Run (from arabic-sentence-segmentation/):
  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONUTF8=1 python src/smoke_xview.py
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import traceback

# CPU-only by default: the GPU belongs to the frontier battery. Must happen
# BEFORE torch import. GPU sections s5-s8 (deferred) will pass --gpu.
if "--gpu" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from augment import deletion_mask, project_labels
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN
import xview_teacher_cache as XTC

AR_WORDS = ["قال", "الرجل", "كتاب", "بيت", "ذهب", "علم", "يوم", "كبير",
            "مدينة", "ولد", "شمس", "قمر", "بحر", "جبل", "طريق", "باب",
            "نهر", "ليل", "صباح", "كلام", "استقلالية", "فاستمعوا",
            "المستشفيات", "يتحدثون"]
PUNCT_TOKENS = [".", "،", "؟", "!", "؛", ")", "(", ":", "\"", "-", "…"]
STRONG_LIKE = [".", "؟", "!", "؛"]


# =========================================================================== #
# s1 — fold == label-projection on synthetic doc pairs                        #
# =========================================================================== #
def gen_pa_doc(rng, force_prefix=False, force_long_run=False, force_empty=False):
    """One synthetic PA doc: (tokens, labels, intended_keep).

    Coverage by construction: boundaries ON stripped tokens and ON "\\n",
    doc-prefix stripped runs (some containing a boundary -> forward-fold),
    all-stripped sentences (2->1 merges), stripped runs up to length 60,
    kept "\\n" adjacent to stripped runs, fully-stripped docs (empty target).
    """
    tokens, labels, keep = [], [], []

    def add(tok, lab, k):
        tokens.append(tok)
        labels.append(lab)
        keep.append(k)

    if force_empty:  # fully stripped doc -> empty target, dropped boundaries
        for _ in range(rng.randint(3, 12)):
            add(rng.choice(AR_WORDS + PUNCT_TOKENS + [PARAGRAPH_TOKEN]),
                1 if rng.random() < 0.3 else 0, False)
        return tokens, labels, keep

    if force_prefix or rng.random() < 0.15:
        # doc-prefix stripped run, often containing a boundary (forward-fold)
        for _ in range(rng.randint(1, 8)):
            add(rng.choice(PUNCT_TOKENS + AR_WORDS + [PARAGRAPH_TOKEN]),
                1 if rng.random() < 0.4 else 0, False)

    for si in range(rng.randint(2, 12)):
        if rng.random() < 0.15:
            # ALL-STRIPPED "sentence" (header/MCQ analog): merges 2 -> 1
            for _ in range(rng.randint(1, 5)):
                add(rng.choice(AR_WORDS + PUNCT_TOKENS), 0, False)
            add(rng.choice(PUNCT_TOKENS), 1, False)
            continue
        for _ in range(rng.randint(1, 10)):
            r = rng.random()
            if r < 0.08:
                add(PARAGRAPH_TOKEN, 1 if rng.random() < 0.05 else 0,
                    rng.random() < 0.5)          # "\n" kept or stripped
            elif r < 0.2:
                add(rng.choice(PUNCT_TOKENS),
                    1 if rng.random() < 0.15 else 0, False)
            else:
                add(rng.choice(AR_WORDS),
                    1 if rng.random() < 0.05 else 0, True)
        if rng.random() < 0.7:
            add(rng.choice(STRONG_LIKE), 1, False)   # boundary ON stripped punct
        else:
            if tokens and keep[-1]:
                labels[-1] = 1                       # boundary on the kept word
            add(rng.choice([".", ")", "،"]),
                1 if rng.random() < 0.3 else 0, False)
        run = 60 if (force_long_run and si == 0) else rng.randint(0, 6)
        for _ in range(run):                          # dot leaders / newline runs
            add(rng.choice([".", ".", PARAGRAPH_TOKEN, ")", "("]),
                1 if rng.random() < 0.02 else 0, False)
    return tokens, labels, keep


def s1_fold_equals_projection(n_docs=200, seed=42):
    rng = random.Random(seed)
    plan = (["prefix"] * 40 + ["longrun"] * 4 + ["empty"] * 4
            + ["random"] * (n_docs - 48))
    st = {"docs": 0, "pa_tokens": 0, "stripped": 0, "boundaries": 0,
          "boundaries_on_stripped": 0, "prefix_docs": 0,
          "forward_folded_1s": 0, "dropped_1s": 0, "merges_ge2": 0,
          "max_run": 0, "empty_target_docs": 0, "mismatch_docs": 0,
          "mismatch_positions": 0}
    for kind in plan:
        toks, labs, keep0 = gen_pa_doc(
            rng, force_prefix=(kind == "prefix"),
            force_long_run=(kind == "longrun"), force_empty=(kind == "empty"))
        sub = [t for t, k in zip(toks, keep0) if k]
        keep = deletion_mask(toks, sub)              # recovered mask (pipeline)
        route = XTC.route_map(toks, keep)
        n_slots = sum(keep)
        assert n_slots == len(sub)

        proj = project_labels(toks, labs, keep)      # projection of record
        assert len(proj) == n_slots
        via_or = XTC.or_fold_labels(labs, route, n_slots)
        folds = {m: [int(p >= 0.5) for p in
                     XTC.fold_probs([float(x) for x in labs], route, n_slots, m)]
                 for m in ("max", "noisy-or", "sum-cap")}

        bad = via_or != proj or any(folds[m] != proj for m in folds)
        if bad:
            st["mismatch_docs"] += 1
            st["mismatch_positions"] += sum(
                1 for j in range(n_slots)
                if via_or[j] != proj[j] or any(folds[m][j] != proj[j] for m in folds))

        # stats
        kept_idx = [i for i in range(len(toks)) if keep[i]]
        st["docs"] += 1
        st["pa_tokens"] += len(toks)
        st["stripped"] += len(toks) - n_slots
        st["boundaries"] += sum(labs)
        st["boundaries_on_stripped"] += sum(
            1 for i, l in enumerate(labs) if l == 1 and not keep[i])
        first_kept = kept_idx[0] if kept_idx else len(toks)
        if first_kept > 0 and kind != "empty":
            st["prefix_docs"] += 1
        for i, l in enumerate(labs):
            if l != 1 or keep[i]:
                continue
            if route[i] < 0:
                st["dropped_1s"] += 1
            elif kept_idx[route[i]] > i:
                st["forward_folded_1s"] += 1
        from collections import Counter
        per_slot = Counter(route[i] for i, l in enumerate(labs)
                           if l == 1 and route[i] >= 0)
        st["merges_ge2"] += sum(1 for m in per_slot.values() if m >= 2)
        run = 0
        for k in keep + [True]:
            if not k:
                run += 1
            else:
                st["max_run"] = max(st["max_run"], run)
                run = 0
        if n_slots == 0:
            st["empty_target_docs"] += 1

    print(f"  docs={st['docs']}  PA tokens={st['pa_tokens']}  "
          f"stripped={st['stripped']}  boundaries={st['boundaries']} "
          f"(on stripped: {st['boundaries_on_stripped']})")
    print(f"  prefix-strip docs={st['prefix_docs']}  "
          f"forward-folded 1s={st['forward_folded_1s']}  "
          f"dropped 1s={st['dropped_1s']}  merged slots(>=2 -> 1)={st['merges_ge2']}  "
          f"max stripped run={st['max_run']}  empty-target docs={st['empty_target_docs']}")
    print(f"  fold(or/max/noisy-or/sum-cap on gold indicators) vs "
          f"augment.project_labels: mismatching docs={st['mismatch_docs']}, "
          f"positions={st['mismatch_positions']}")
    # the suite must actually exercise every adversarial pattern:
    assert st["prefix_docs"] >= 30, "coverage: too few prefix-strip docs"
    assert st["forward_folded_1s"] >= 5, "coverage: no forward-fold cases generated"
    assert st["dropped_1s"] >= 1, "coverage: no dropped-boundary cases generated"
    assert st["merges_ge2"] >= 20, "coverage: no all-stripped-sentence merges"
    assert st["max_run"] >= 60, "coverage: no stripped run of length 60"
    assert st["empty_target_docs"] >= 4, "coverage: no fully-stripped docs"
    assert st["mismatch_docs"] == 0 and st["mismatch_positions"] == 0, \
        "PIPELINE FOLD != augment.project_labels on {0,1} indicators"
    return st


# =========================================================================== #
# s2 — punct-class derivation table                                            #
# =========================================================================== #
def _expected_class(run):
    """Independent re-derivation of the documented rule: classify the FIRST
    non-'\\n' stripped token of the run; sets from xview_teacher_cache."""
    for t in run:
        if t == PARAGRAPH_TOKEN:
            continue
        if t in XTC.PUNCT_STRONG:
            return 1
        if t in XTC.PUNCT_COMMA:
            return 2
        if all(not c.isalnum() for c in t):
            return 3
        return 0
    return 0


def s2_punct_class_table():
    NL = PARAGRAPH_TOKEN
    # (kept word, stripped run after it, expected class, note)
    table = [
        ("قال",  [],                            0, "empty run -> NONE"),
        ("ذهب",  ["."],                         1, "strong ."),
        ("بيت",  ["؟"],                         1, "strong ؟"),
        ("علم",  ["!"],                         1, "strong !"),
        ("يوم",  ["؛"],                         1, "strong ؛ (ruling of record)"),
        ("كبير", [","],                         2, "comma ,"),
        ("ولد",  ["،"],                         2, "comma ،"),
        ("بحر",  [")"],                         3, "other )"),
        ("جبل",  [":"],                         3, "other : (not comma; ruling)"),
        ("نهر",  ["-"],                         3, "other -"),
        ("كتاب", ["…"],                         3, "other … (not in strong set)"),
        ("ليل",  [NL, "."],                     1, "EDGE: newline skipped -> STRONG"),
        ("قمر",  [NL],                          0, "EDGE: only-newline run -> NONE"),
        ("شمس",  [NL, NL],                      0, "EDGE: newline run -> NONE"),
        ("باب",  [NL, "،"],                     2, "EDGE: newline then comma"),
        ("طريق", ["أ"],                         0, "alphanumeric first (MCQ) -> NONE"),
        ("مدينة", ["المادة", "(", "28", ")"],   0, "alnum first (header) -> NONE"),
        ("كلام", ["28"],                        0, "digits -> NONE"),
        ("صباح", ["أ)"],                        0, "mixed alnum+punct token -> NONE"),
        ("الرجل", [".", "،"],                   1, "first token decides"),
        ("علم",  [".", ".", "."],               1, "tail dot-leader run (doc end)"),
    ]
    tokens, keep, expected = [], [], []
    for w, run, cls, _ in table:
        tokens.append(w)
        keep.append(True)
        expected.append(cls)
        for r in run:
            tokens.append(r)
            keep.append(False)
    got = XTC.punct_after_classes(tokens, keep)
    assert len(got) == len(expected)
    n_bad = 0
    for k, (w, run, cls, note) in enumerate(table):
        ok = got[k] == cls
        n_bad += 0 if ok else 1
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] slot {k:2d} run={run!r:32s} expected={cls} "
              f"got={got[k]}  ({note})")
    # cross-check every expectation against the independent re-derivation
    for k, (w, run, cls, _) in enumerate(table):
        assert _expected_class(run) == cls, f"table row {k} self-inconsistent"
    assert n_bad == 0, f"{n_bad} punct-class rows disagree with the pipeline"

    # independent re-derivation on random docs
    rng = random.Random(7)
    n_slots = n_mm = 0
    for _ in range(50):
        toks, keep2 = [], []
        for _ in range(rng.randint(5, 60)):
            if keep2 and not keep2[-1] and rng.random() < 0.4:
                pass  # allow multi-token stripped runs
            kflag = rng.random() < 0.55
            toks.append(rng.choice(AR_WORDS + PUNCT_TOKENS + [PARAGRAPH_TOKEN, "28", "أ"]))
            keep2.append(kflag)
        if not any(keep2):
            keep2[0] = True
        got2 = XTC.punct_after_classes(toks, keep2)
        kept_idx = [i for i in range(len(toks)) if keep2[i]]
        exp2 = []
        for k, i in enumerate(kept_idx):
            end = kept_idx[k + 1] if k + 1 < len(kept_idx) else len(toks)
            exp2.append(_expected_class(toks[i + 1:end]))
        n_slots += len(exp2)
        n_mm += sum(1 for a, b in zip(got2, exp2) if a != b)
    print(f"  random cross-check: 50 docs, {n_slots} kept slots, "
          f"mismatches={n_mm}")
    assert n_mm == 0
    return {"rows": len(table), "row_failures": n_bad,
            "random_slots": n_slots, "random_mismatches": n_mm}


# =========================================================================== #
# s3 — KD math identities (pure torch; the exact loss of spec section 3)      #
# =========================================================================== #
EPS = 1e-6


def kd_pointwise(z_s, teacher_p, T):
    """Per-position forward-KL distillation term of the spec:
    q=clamp(p); z_t=log(q/(1-q)); qT=sigmoid(z_t/T); pT=sigmoid(z_s/T);
    kd_i = qT*log(qT/pT) + (1-qT)*log((1-qT)/(1-pT)).  pT is clamped like q so
    extreme student logits stay finite."""
    q = teacher_p.clamp(EPS, 1 - EPS)
    z_t = torch.log(q / (1 - q))
    qT = torch.sigmoid(z_t / T)
    pT = torch.sigmoid(z_s / T).clamp(EPS, 1 - EPS)
    return qT * torch.log(qT / pT) + (1 - qT) * torch.log((1 - qT) / (1 - pT))


def kd_term(z_s, teacher_p, labels, T, gate=0.0, entropy_weight=False):
    """The spec's batch KD term: T^2-scaled mean over the gated selection, or
    the entropy-weighted average over valid; empty selection -> grad-safe 0."""
    valid = (labels != -100) & (teacher_p >= 0)
    kd_i = kd_pointwise(z_s, teacher_p, T)
    if entropy_weight:
        q = teacher_p.clamp(EPS, 1 - EPS)
        h = -(q * torch.log(q) + (1 - q) * torch.log(1 - q)) / math.log(2.0)
        w = h * valid.float()
        return T * T * (w * kd_i).sum() / w.sum().clamp(min=1e-9), valid
    sel = valid & (teacher_p >= gate)
    if sel.any():
        return T * T * kd_i[sel].mean(), sel
    return (z_s * 0.0).sum(), sel


def s3_kd_math():
    g = torch.Generator().manual_seed(42)
    q0 = torch.rand(4, 9, generator=g) * 0.8 + 0.1          # (0.1, 0.9)
    labels = torch.zeros(4, 9, dtype=torch.long)

    # (a) z_s == z_t  ->  kd_i exactly 0
    z_t = torch.log(q0 / (1 - q0))
    kd_i = kd_pointwise(z_t.clone(), q0, T=2.0)
    m = kd_i.abs().max().item()
    print(f"  (a) z_s=z_t: max|kd_i| = {m:.3e} (exact-zero required)")
    assert torch.all(kd_i == 0.0)

    # (b) T=1 reduces to the plain binary KL(q || p)
    z_s = torch.randn(4, 9, generator=g) * 3
    p = torch.sigmoid(z_s).clamp(EPS, 1 - EPS)
    q = q0.clamp(EPS, 1 - EPS)
    kl_ref = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    kd_t1 = kd_pointwise(z_s, q0, T=1.0)
    d = (kd_t1 - kl_ref).abs().max().item()
    print(f"  (b) T=1 vs plain binary KL: max abs diff = {d:.3e}")
    assert torch.allclose(kd_t1, kl_ref, atol=1e-5)

    # (c) T^2 scaling: term == T^2 * mean(kd_i); large-T asymptote (dz)^2/8
    T = 3.0
    term, sel = kd_term(z_s, q0, labels, T)
    manual = T * T * kd_pointwise(z_s, q0, T)[labels != -100].mean()
    print(f"  (c) kd(T=3) = {term.item():.6f}  ==  9 * mean(kd_i) = {manual.item():.6f}")
    assert torch.allclose(term, manual, rtol=1e-6)
    zt_a = torch.tensor([1.5, -0.8, 2.0, 0.3])
    zs_a = torch.tensor([-0.5, 1.0, 0.0, -1.7])
    q_a = torch.sigmoid(zt_a)
    big_T = 64.0
    asym = big_T * big_T * kd_pointwise(zs_a, q_a, big_T)
    ref = (zt_a - zs_a) ** 2 / 8.0
    ratio = (asym / ref)
    print(f"  (c) large-T asymptote T^2*kd_i / ((zt-zs)^2/8) at T=64: "
          f"{[round(float(r), 4) for r in ratio]}")
    assert torch.all((ratio > 0.98) & (ratio < 1.02))

    # (d) gate selection is exactly {valid & teacher_p >= gate}
    tp = torch.tensor([[-1.0, 0.0, 0.049, 0.05, 0.2, 0.95, 1.0, 0.5, 0.7]])
    lb = torch.tensor([[0, 0, 0, 0, -100, 0, 0, -100, 0]])
    zz = torch.zeros_like(tp)
    _, sel = kd_term(zz, tp, lb, T=2.0, gate=0.05)
    got = sel.nonzero()[:, 1].tolist()
    expect = [3, 5, 6, 8]  # >= 0.05, valid label, not the -1 sentinel
    print(f"  (d) gate=0.05 selects columns {got} (expected {expect})")
    assert got == expect
    _, sel0 = kd_term(zz, tp, lb, T=2.0, gate=0.0)
    valid = (lb != -100) & (tp >= 0)
    print(f"  (d) gate=0.0 -> selection == valid: {bool(torch.equal(sel0, valid))}")
    assert torch.equal(sel0, valid)

    # (e) entropy weights in [0,1], maximal at q=0.5; all-0.5 == plain mean
    qs = torch.linspace(0.0, 1.0, 101)
    qc = qs.clamp(EPS, 1 - EPS)
    h = -(qc * torch.log(qc) + (1 - qc) * torch.log(1 - qc)) / math.log(2.0)
    print(f"  (e) H2/ln2 range = [{h.min().item():.3e}, {h.max().item():.6f}], "
          f"argmax at q = {qs[h.argmax()].item():.2f}")
    assert torch.all((h >= 0) & (h <= 1.0 + 1e-6))
    assert abs(qs[h.argmax()].item() - 0.5) < 1e-6
    assert abs(h[50].item() - 1.0) < 1e-6
    tp_half = torch.full((2, 6), 0.5)
    lb_half = torch.zeros(2, 6, dtype=torch.long)
    zs_half = torch.randn(2, 6, generator=g)
    kd_ew, _ = kd_term(zs_half, tp_half, lb_half, T=2.0, entropy_weight=True)
    kd_pl, _ = kd_term(zs_half, tp_half, lb_half, T=2.0, gate=0.0)
    print(f"  (e) all-q=0.5 entropy-weighted {kd_ew.item():.6f} == "
          f"plain mean {kd_pl.item():.6f}")
    assert torch.allclose(kd_ew, kd_pl, rtol=1e-5)

    # (f) clamps: q in {0,1} and |z_s| = 100 stay finite on every path
    tp_x = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    zs_x = torch.tensor([[100.0, -100.0, -100.0, 100.0]])
    lb_x = torch.zeros(1, 4, dtype=torch.long)
    kd_ix = kd_pointwise(zs_x, tp_x, T=2.0)
    k1, _ = kd_term(zs_x, tp_x, lb_x, T=2.0, gate=0.0)
    k2, _ = kd_term(zs_x, tp_x, lb_x, T=2.0, gate=0.05)
    k3, _ = kd_term(zs_x, tp_x, lb_x, T=2.0, entropy_weight=True)
    fin = all(bool(torch.isfinite(t).all()) for t in (kd_ix, k1, k2, k3))
    print(f"  (f) q in {{0,1}}, |z_s|=100: kd_i={ [round(float(v),4) for v in kd_ix[0]] } "
          f"all finite = {fin}")
    assert fin

    # (g) empty selection -> 0 with a live graph
    zg = torch.randn(2, 5, generator=g, requires_grad=True)
    lb_none = torch.full((2, 5), -100, dtype=torch.long)
    tp_g = torch.rand(2, 5, generator=g)
    kd_e, sel_e = kd_term(zg, tp_g, lb_none, T=2.0)
    print(f"  (g) empty selection: kd = {kd_e.item()}  grad_fn = "
          f"{type(kd_e.grad_fn).__name__ if kd_e.grad_fn else None}  "
          f"selected = {int(sel_e.sum())}")
    assert kd_e.item() == 0.0 and kd_e.grad_fn is not None
    kd_e.backward()
    assert zg.grad is not None and float(zg.grad.abs().sum()) == 0.0
    return {"checks": 7}


# =========================================================================== #
# s4 — plumbing: teacher_p / punct_cls land exactly where labels != -100      #
# =========================================================================== #
def make_examples_x(docs, window, stride):
    """Spec section 3 data plumbing: the SAME window/stride loop as
    consistency_train.make_examples, with two per-word parallel columns
    (teacher_p sentinel -1.0 at [PAR]; punct_cls sentinel -100 at [PAR])."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l
                for w, l in zip(words, d["labels"])]
        tp = [-1.0 if w == MODEL_PAR_TOKEN else float(p)
              for w, p in zip(words, d["teacher_p"])]
        pc = [-100 if w == MODEL_PAR_TOKEN else int(c)
              for w, c in zip(words, d["punct_cls"])]
        n = len(words)
        s = 0
        while s < n:
            e = min(s + window, n)
            out.append({"words": words[s:e], "word_labels": labs[s:e],
                        "teacher_p": tp[s:e], "punct_cls": pc[s:e]})
            if e == n:
                break
            s += stride
    return out


def encode_x(examples, tok, max_len):
    """consistency_train.encode + the two parallel columns on the LAST subword."""
    enc = tok(examples["words"], is_split_into_words=True, truncation=True,
              max_length=max_len)
    all_lab, all_tp, all_pc = [], [], []
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        wt = examples["teacher_p"][i]
        wc = examples["punct_cls"][i]
        lab = [-100] * len(wid)
        tp = [-1.0] * len(wid)
        pc = [-100] * len(wid)
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:                       # last subword of word w
                lab[pos] = wl[w]
                tp[pos] = wt[w]
                pc[pos] = wc[w]
        all_lab.append(lab)
        all_tp.append(tp)
        all_pc.append(pc)
    enc["labels"] = all_lab
    enc["teacher_p"] = all_tp
    enc["punct_cls"] = all_pc
    return enc


def collate_x(enc, tok):
    feats = [{"input_ids": enc["input_ids"][i],
              "attention_mask": enc["attention_mask"][i]}
             for i in range(len(enc["input_ids"]))]
    batch = tok.pad(feats, return_tensors="pt")
    m = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor(
        [l + [-100] * (m - len(l)) for l in enc["labels"]])
    batch["teacher_p"] = torch.tensor(
        [t + [-1.0] * (m - len(t)) for t in enc["teacher_p"]])
    batch["punct_cls"] = torch.tensor(
        [c + [-100] * (m - len(c)) for c in enc["punct_cls"]])
    return batch


def s4_plumbing(window=24, stride=12):
    from transformers import AutoTokenizer
    import consistency_train as C

    tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})

    # synthetic docs; every token carries a GLOBAL index g so values can be
    # traced back to the exact word: teacher_p = g/10000, punct_cls = g % 4.
    # [PAR] positions are POISONED (0.777 / 9 / label 1) and must come out as
    # sentinels (-1.0 / -100 / -100) after make_examples_x.
    rng = random.Random(42)
    docs, flat_tokens, flat_labels = [], [], []
    g = 0
    for n_words in (5, 30, 47, 80, 120, 61):
        tokens, labels, tps, pcs = [], [], [], []
        for _ in range(n_words):
            if rng.random() < 0.1:
                tokens.append(PARAGRAPH_TOKEN)
                labels.append(1)      # poison: must become -100
                tps.append(0.777)     # poison: must become -1.0
                pcs.append(9)         # poison: must become -100
            else:
                tokens.append(rng.choice(AR_WORDS))
                labels.append(1 if rng.random() < 0.12 else 0)
                tps.append(g / 10000.0)
                pcs.append(g % 4)
            flat_tokens.append(tokens[-1])
            flat_labels.append(labels[-1])
            g += 1
        docs.append({"tokens": tokens, "labels": labels,
                     "teacher_p": tps, "punct_cls": pcs})
    assert g < 1000  # keeps g/10000 exactly recoverable in float32

    # global index of each word: parallel windows over the flat arrays
    doc_offsets = []
    off = 0
    for d in docs:
        doc_offsets.append(off)
        off += len(d["tokens"])

    results = {}
    for tag, max_len in (("no-trunc", 512), ("trunc", 24)):
        exs = make_examples_x(docs, window, stride)
        # window offsets, replicating the loop:
        offsets = []
        for di, d in enumerate(docs):
            n = len(d["tokens"])
            s = 0
            while s < n:
                offsets.append(doc_offsets[di] + s)
                if min(s + window, n) == n:
                    break
                s += stride
        assert len(offsets) == len(exs)

        enc = encode_x({k: [e[k] for e in exs] for k in exs[0]}, tok, max_len)

        # label placement must equal the pipeline of record bit-exactly
        ref = C.encode({"words": [e["words"] for e in exs],
                        "word_labels": [e["word_labels"] for e in exs]},
                       tok, max_len)
        assert enc["labels"] == ref["labels"], \
            "labels differ from consistency_train.encode"

        batch = collate_x(enc, tok)
        lab_pos = batch["labels"] != -100
        tp_pos = batch["teacher_p"] >= 0.0
        pc_pos = batch["punct_cls"] != -100
        eq_tp = bool(torch.equal(tp_pos, lab_pos))
        eq_pc = bool(torch.equal(pc_pos, lab_pos))
        n_lab = int(lab_pos.sum())
        n_pad = int((batch["attention_mask"] == 0).sum())
        # poison must never survive
        n_poison_tp = int(((batch["teacher_p"] - 0.777).abs() < 1e-6).sum())
        n_poison_pc = int((batch["punct_cls"] == 9).sum())
        # value round-trip: each value sits on the RIGHT word
        n_checked = n_val_bad = 0
        tpv = batch["teacher_p"]
        pcv = batch["punct_cls"]
        lbv = batch["labels"]
        idx = lab_pos.nonzero()
        for r, cpos in idx.tolist():
            gi = int(round(float(tpv[r, cpos]) * 10000.0))
            ok = (flat_tokens[gi] != PARAGRAPH_TOKEN
                  and int(pcv[r, cpos]) == gi % 4
                  and int(lbv[r, cpos]) == flat_labels[gi])
            n_checked += 1
            n_val_bad += 0 if ok else 1
        print(f"  [{tag}] windows={len(exs)}  batch={tuple(batch['input_ids'].shape)}  "
              f"pad positions={n_pad}  label positions={n_lab}")
        print(f"  [{tag}] (teacher_p>=0)==(labels!=-100): {eq_tp}   "
              f"(punct_cls!=-100)==(labels!=-100): {eq_pc}   "
              f"labels == consistency_train.encode: True")
        print(f"  [{tag}] poison survivors: teacher_p={n_poison_tp} punct_cls={n_poison_pc}   "
              f"value round-trip: {n_checked} positions checked, {n_val_bad} wrong")
        assert eq_tp and eq_pc
        assert n_poison_tp == 0 and n_poison_pc == 0
        assert n_lab > 0 and n_checked == n_lab and n_val_bad == 0
        results[tag] = {"windows": len(exs), "label_positions": n_lab,
                        "checked": n_checked, "bad": n_val_bad}
    # the truncated pass must actually truncate
    assert results["trunc"]["label_positions"] < results["no-trunc"]["label_positions"], \
        "truncation sub-check did not truncate anything"
    return results


# =========================================================================== #
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-docs", type=int, default=200, help="s1 synthetic docs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", action="store_true",
                    help="reserved for the deferred GPU sections s5-s8 "
                         "(not built yet; hard-stops)")
    args = ap.parse_args()
    if args.gpu:
        print("GPU sections s5-s8 are DEFERRED until the frontier battery "
              "releases the GPU and spend is approved. Nothing to run.")
        sys.exit(2)

    print(f"torch {torch.__version__}  cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}  "
          f"cuda.is_available={torch.cuda.is_available()}")
    sections = [
        ("s1 fold == label-projection",
         lambda: s1_fold_equals_projection(args.n_docs, args.seed)),
        ("s2 punct-class derivation table", s2_punct_class_table),
        ("s3 KD math identities", s3_kd_math),
        ("s4 plumbing (teacher_p / punct_cls placement)", s4_plumbing),
    ]
    failed = []
    for name, fn in sections:
        print(f"\n=== {name} ===")
        try:
            fn()
            print(f"--- {name}: PASS")
        except Exception:
            traceback.print_exc()
            print(f"--- {name}: FAIL")
            failed.append(name)
    print("\n================ SMOKE SUMMARY ================")
    for name, _ in sections:
        print(f"  {'FAIL' if name in failed else 'PASS'}  {name}")
    if failed:
        print("SMOKE FAILED")
        sys.exit(1)
    print("ALL CPU SECTIONS PASS (s5-s8 deferred to GPU approval)")


if __name__ == "__main__":
    main()
