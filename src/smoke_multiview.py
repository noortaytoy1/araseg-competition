# -*- coding: utf-8 -*-
"""CPU smoke suite for src/multiview_train.py (multi-view joint training).

Corpus-independent: synthetic documents only -- no data/, no GPU, no network
(the tokenizer/model come from the local HF cache; run with HF_HUB_OFFLINE=1).
CUDA_VISIBLE_DEVICES is forced empty BEFORE torch import: the GPU belongs to
the frontier battery.

Sections (the three pre-registered self-checks of the build order):
  (a) SINGLE-VIEW = BIT-FOR-BIT PLAIN BASELINE. --views NoPnx-NP with default
      flags must reproduce the consistency_train.py all-off baseline exactly:
      same tokenizer length (NO extra view tokens added), identical initial
      weights under the same set_seed (torch.equal on EVERY state-dict
      tensor), identical windows + encoded features on the same synthetic
      docs, identical padded batch tensors, and torch.equal loss AND logits
      on a forward pass. Plus: predict_probs(view_word=None) is the shared
      predict.predict_doc (delegated; asserted equal element-for-element).
  (b) MULTI-VIEW MECHANICS. 50 synthetic PA docs and their derived NP /
      NoPnx-PA / NoPnx-NP twins (stripped via the verified fold):
      window counts per view match an independent count; the view-indicator
      token is a single tokenizer id and appears EXACTLY once per encoded
      window (word position 0, right after [CLS]) with label -100; the
      punct-dropout re-fold is verified EXACTLY against an independent naive
      walker (not fold_labels) on every window of the punctuated views at
      rates 0.3 / 0.5 / 1.0 -- kept sequence, folded labels, and lost-boundary
      accounting all match; [PAR] is never dropped; kept -100 stays -100;
      pd=0 windows pass through unchanged; the OnTheFlyCollator batch equals
      a manual dropout+prepend+encode+pad replication bit-for-bit.
  (c) TINY CPU TRAIN. 2 docs per view, all four views, indicator tokens,
      --punct-dropout 0.5, one epoch on CPU through the real cmd_train
      (Trainer, eval, HF-checkpoint save), then reload + view-token predict +
      submission CSV round-trip through eval_local.compute_metrics.

Run (from arabic-sentence-segmentation/):
  HF_HUB_OFFLINE=1 PYTHONUTF8=1 python src/smoke_multiview.py
(the script itself forces CUDA_VISIBLE_DEVICES=-1 before the torch import and
hard-asserts torch.cuda.is_available() is False; an EMPTY value is NOT enough
on Windows -- observed on this box leaving the GPU visible)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import traceback

if "--gpu" not in sys.argv:                     # GPU belongs to the frontier battery
    # "-1", not "": on Windows an EMPTY CUDA_VISIBLE_DEVICES can still leave
    # torch.cuda.is_available() == True (observed on this box 2026-07-10);
    # "-1" is the documented always-hide sentinel. main() hard-asserts below.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN
from xview_align import fold_labels
import consistency_train as C
import multiview_train as MV

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

AR_WORDS = ["قال", "الرجل", "كتاب", "بيت", "ذهب", "علم", "يوم", "كبير",
            "مدينة", "ولد", "شمس", "قمر", "بحر", "جبل", "طريق", "باب",
            "نهر", "ليل", "صباح", "كلام", "استقلالية", "فاستمعوا",
            "المستشفيات", "يتحدثون"]
PUNCT = [".", "،", "؟", "!", "؛", ")", "(", ":", "-", "…", "«", "»"]
STRONG = [".", "؟", "!", "؛"]


# --------------------------------------------------------------------------- #
# synthetic 4-view corpus (PA original; twins derived via the verified fold)   #
# --------------------------------------------------------------------------- #
def gen_pa_doc(rng, doc_id):
    """One synthetic PA doc: punctuated, with '\\n' paragraph tokens.
    Boundary-after-token labels; boundaries mostly ON the closing punct
    (mirrors real PA), sometimes on the last word."""
    tokens, labels = [], []
    for _ in range(rng.randint(3, 10)):                       # sentences
        for _ in range(rng.randint(2, 8)):                    # words
            tokens.append(rng.choice(AR_WORDS)); labels.append(0)
            if rng.random() < 0.15:                           # mid-sentence punct
                tokens.append(rng.choice(["،", ":", "(", ")", "«", "»"]))
                labels.append(0)
        if rng.random() < 0.75:                               # boundary on punct
            tokens.append(rng.choice(STRONG)); labels.append(1)
        else:                                                 # boundary on word
            labels[-1] = 1
        if rng.random() < 0.25:                               # paragraph break
            tokens.append(PARAGRAPH_TOKEN); labels.append(0)
        if rng.random() < 0.1:                                # dot-leader run
            for _ in range(rng.randint(2, 6)):
                tokens.append("."); labels.append(0)
    return {"doc_id": doc_id, "tokens": tokens, "labels": labels}


def strip_view(doc, drop_punct, drop_par):
    """Derive a stripped twin with the verified fold (xview_align.fold_labels).
    Mirrors the organizer PA -> NoPnx/NP relationship."""
    toks, labs = doc["tokens"], doc["labels"]
    keep = [i for i, w in enumerate(toks)
            if not ((drop_punct and MV.is_punct_token(w)) or
                    (drop_par and w == PARAGRAPH_TOKEN))]
    return {"doc_id": doc["doc_id"],
            "tokens": [toks[i] for i in keep],
            "labels": fold_labels(labs, keep, len(toks))}


def gen_views(n_docs, seed):
    rng = random.Random(seed)
    pa = [gen_pa_doc(rng, f"d{i:03d}") for i in range(n_docs)]
    return {
        "PA": pa,
        "NP": [strip_view(d, drop_punct=False, drop_par=True) for d in pa],
        "NoPnx-PA": [strip_view(d, drop_punct=True, drop_par=False) for d in pa],
        "NoPnx-NP": [strip_view(d, drop_punct=True, drop_par=True) for d in pa],
    }


# =========================================================================== #
# (a) single-view default == bit-for-bit plain baseline                       #
# =========================================================================== #
def s_a_bit_for_bit():
    docs = gen_views(12, seed=42)["NoPnx-NP"]
    pw = 4.0

    # tokenizers: multiview single-view/none vs the consistency pipeline
    assert MV.resolve_indicator("auto", ["NoPnx-NP"]) == "none"
    assert MV.resolve_indicator("auto", ["NoPnx-NP", "PA"]) == "token"
    tok_m, n_extra = MV.build_tokenizer(MODEL_NAME, "none")
    tok_c = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok_c.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    print(f"  tokenizer: len(mv)={len(tok_m)}  len(cons)={len(tok_c)}  "
          f"n_extra_tok={n_extra}")
    assert n_extra == 1 and len(tok_m) == len(tok_c), "extra tokens added in single-view mode"
    for v in MV.ALL_VIEWS:
        assert MV.VIEW_TOKENS[v] not in tok_m.get_vocab(), f"{v} token leaked into baseline vocab"

    # identical initial weights under the same seed
    set_seed(0)
    model_c = C.ConsistencyModel(MODEL_NAME, pw, cf_weight=0.0, rdrop_weight=0.0,
                                 mask_id=tok_c.mask_token_id, n_extra_tok=1)
    set_seed(0)
    model_m = MV.MultiViewModel(MODEL_NAME, pw, n_extra_tok=1)
    sd_c, sd_m = model_c.enc.state_dict(), model_m.enc.state_dict()
    assert set(sd_c) == set(sd_m)
    n_neq = sum(0 if torch.equal(sd_c[k], sd_m[k]) else 1 for k in sd_c)
    print(f"  init weights: {len(sd_c)} tensors compared, {n_neq} differ (torch.equal)")
    assert n_neq == 0, "initial weights are NOT bit-for-bit identical"
    assert torch.equal(model_c.w, model_m.w)

    # identical windows + encoded features (both paths run the code of record)
    exs_c = C.make_examples(docs, 24, 12)
    exs_m = MV.make_view_examples(docs, 24, 12, "NoPnx-NP", "none", raw=False)
    assert exs_c == exs_m, "windows differ from consistency_train.make_examples"
    cols = {k: [e[k] for e in exs_c] for k in ("words", "word_labels")}
    enc_c = C.encode(cols, tok_c, 64)
    enc_m = C.encode(cols, tok_m, 64)
    assert enc_c["input_ids"] == enc_m["input_ids"] and enc_c["labels"] == enc_m["labels"]
    print(f"  windows={len(exs_c)}  encoded features identical (input_ids + labels)")

    # identical padded batches + torch.equal forward loss/logits
    nb = min(8, len(exs_c))
    feats_m = [{"input_ids": enc_m["input_ids"][i], "token_type_ids": enc_m["token_type_ids"][i],
                "attention_mask": enc_m["attention_mask"][i], "labels": enc_m["labels"][i]}
               for i in range(nb)]
    feats_c = [dict(f) for f in feats_m]
    batch_m = MV.PadCollator(tok_m)(feats_m)
    batch_c = C.Collator(tok_c, C._prep_ids(tok_c))(feats_c)
    for k in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(batch_m[k], batch_c[k]), f"batch tensor {k} differs"
    model_c.eval(); model_m.eval()
    with torch.no_grad():
        out_c = model_c(**batch_c)
        out_m = model_m(**batch_m)
    eq_logits = torch.equal(out_c["logits"], out_m["logits"])
    eq_loss = torch.equal(out_c["loss"], out_m["loss"])
    print(f"  forward: loss(cons)={out_c['loss'].item():.10f}  "
          f"loss(mv)={out_m['loss'].item():.10f}")
    print(f"  torch.equal(logits)={eq_logits}  torch.equal(loss)={eq_loss}")
    assert eq_logits and eq_loss, "single-view default is NOT bit-for-bit the plain baseline"

    # predict path: view_word=None IS predict.predict_doc
    from predict import predict_doc
    d = docs[0]
    words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
    model_m.enc.eval()
    p_ref = predict_doc(words, tok_m, model_m.enc, "cpu", 24, 8, 64)
    p_mv = MV.predict_probs(words, tok_m, model_m.enc, "cpu", 24, 8, 64, view_word=None)
    assert np.array_equal(p_ref, p_mv), "predict_probs(None) != predict.predict_doc"
    print(f"  predict_probs(view_word=None) == predict.predict_doc on {len(words)} tokens: True")
    return {"tensors": len(sd_c), "windows": len(exs_c), "batch": nb}


# =========================================================================== #
# (b) multi-view mechanics: windows, view tokens, punct-dropout re-fold       #
# =========================================================================== #
def independent_window_count(n, window, stride):
    """Independent re-derivation of the make_examples window count."""
    if n <= window:
        return 1 if n > 0 else 0
    k = 1
    s = 0
    while True:
        s += stride
        k += 1
        if s + window >= n:
            return k


def naive_drop_and_refold(words, labels, rate, rng):
    """INDEPENDENT re-derivation of punct_dropout (explicit walker, no
    fold_labels / groups_from_alignment): each dropped punct token's boundary
    folds LEFT onto the nearest kept token; kept -100 stays -100 (a boundary
    folding onto it is lost); boundaries dropped before the first kept token
    are lost. Same rng draw order (one draw per punct token, in order)."""
    n = len(words)
    dropped = set()
    for i, w in enumerate(words):
        if MV.is_punct_token(w) and rng.random() < rate:
            dropped.add(i)
    kept = [i for i in range(n) if i not in dropped]
    if not kept or not dropped:
        return list(words), list(labels), 0
    out = []
    lost = sum(1 for t in range(kept[0]) if labels[t] == 1)    # dropped prefix
    for j, i in enumerate(kept):
        end = kept[j + 1] if j + 1 < len(kept) else n
        if labels[i] == -100:
            out.append(-100)
            lost += sum(1 for t in range(i + 1, end) if labels[t] == 1)
        else:
            out.append(1 if any(labels[t] == 1 for t in range(i, end)) else 0)
    return [words[i] for i in kept], out, lost


def s_b_multiview_mechanics(n_docs=50, seed=7):
    views = gen_views(n_docs, seed)
    window, stride, max_len = 24, 12, 64
    tok, n_extra = MV.build_tokenizer(MODEL_NAME, "token")
    assert n_extra == 5
    vt_ids = {}
    for v in MV.ALL_VIEWS:
        ids = tok(MV.VIEW_TOKENS[v], add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"view token {v} is not a single id: {ids}"
        vt_ids[v] = ids[0]
    print(f"  view tokens -> single ids: { {v: vt_ids[v] for v in MV.ALL_VIEWS} }")

    # ---- window counts per view vs independent count ----
    for v in MV.ALL_VIEWS:
        exs = MV.make_view_examples(views[v], window, stride, v, "token", raw=True)
        expect = sum(independent_window_count(len(d["tokens"]), window, stride)
                     for d in views[v])
        print(f"  windows[{v}]: got={len(exs)}  independent={expect}")
        assert len(exs) == expect, f"window count mismatch for view {v}"
        assert all(e["view"] == v for e in exs)
        pd_want = 1 if v in MV.PUNCTUATED_VIEWS else 0
        assert all(e["pd"] == pd_want for e in exs), f"pd flag wrong for {v}"

    # ---- view token exactly once per encoded window, label -100, position 1 ----
    tot_win = 0
    for v in MV.ALL_VIEWS:
        exs = MV.make_view_examples(views[v], window, stride, v, "token", raw=False)
        cols = {k: [e[k] for e in exs] for k in ("words", "word_labels")}
        enc = C.encode(cols, tok, max_len)
        for i in range(len(exs)):
            ids = enc["input_ids"][i]
            labs = enc["labels"][i]
            occ = [p for p, t in enumerate(ids) if t in vt_ids.values()]
            assert occ == [1], f"view token not exactly once at pos 1: {occ}"
            assert ids[1] == vt_ids[v], "wrong view's token"
            assert labs[1] == -100, "view token label must be -100"
            tot_win += 1
    print(f"  view-token placement: {tot_win} windows checked, all exactly-once "
          f"@pos1 with label -100")

    # ---- punct-dropout re-fold vs the independent naive walker ----
    stats = {"windows": 0, "positions": 0, "mismatch": 0, "dropped": 0,
             "droppable": 0, "lost": 0, "par_kept": 0}
    for rate in (0.3, 0.5, 1.0):
        for v in sorted(MV.PUNCTUATED_VIEWS):
            exs = MV.make_view_examples(views[v], window, stride, v, "token", raw=True)
            for k, ex in enumerate(exs):
                sd = hash((rate, v, k)) & 0x7FFFFFFF
                w1, l1, lost1 = MV.punct_dropout(
                    ex["words"], ex["word_labels"], rate, random.Random(sd))
                w2, l2, lost2 = naive_drop_and_refold(
                    ex["words"], ex["word_labels"], rate, random.Random(sd))
                stats["windows"] += 1
                stats["positions"] += len(l1)
                if not (w1 == w2 and l1 == l2 and lost1 == lost2):
                    stats["mismatch"] += 1
                # structural: only punct dropped, [PAR] intact, -100 preserved
                n_punct = sum(1 for w in ex["words"] if MV.is_punct_token(w))
                stats["droppable"] += n_punct
                stats["dropped"] += len(ex["words"]) - len(w1)
                stats["lost"] += lost1
                assert sum(1 for w in w1 if w == MODEL_PAR_TOKEN) == \
                    sum(1 for w in ex["words"] if w == MODEL_PAR_TOKEN), "[PAR] dropped"
                assert all(not MV.is_punct_token(w) for w in w1) or rate < 1.0, \
                    "punct survived rate=1.0"
                assert sum(1 for x in l1 if x == -100) == \
                    sum(1 for x in ex["word_labels"] if x == -100), "-100 count changed"
                # boundary accounting: folded 1s + lost <= original 1s (the
                # difference = legitimate >=2 -> 1 merges, exactly as in the
                # verified gold fold); exact lost equality already asserted
                # against the independent walker above.
                orig1 = sum(1 for x in ex["word_labels"] if x == 1)
                assert sum(1 for x in l1 if x == 1) + lost1 <= orig1, "boundary accounting"
    print(f"  punct-dropout re-fold: windows={stats['windows']} (PA+NP x rates .3/.5/1.0)  "
          f"label positions checked={stats['positions']}  MISMATCHES={stats['mismatch']}")
    print(f"  dropped punct tokens={stats['dropped']}/{stats['droppable']} droppable  "
          f"lost boundaries (prefix/[PAR]-fold)={stats['lost']}")
    assert stats["mismatch"] == 0, "punct_dropout != independent naive re-fold"
    assert stats["dropped"] > 0 and stats["lost"] >= 0

    # pd=0 / rate=0 identity
    ex0 = MV.make_view_examples(views["PA"], window, stride, "PA", "token", raw=True)[0]
    w0, l0, lost0 = MV.punct_dropout(ex0["words"], ex0["word_labels"], 0.0, random.Random(1))
    assert w0 == ex0["words"] and l0 == ex0["word_labels"] and lost0 == 0

    # ---- OnTheFlyCollator == manual replication, bit-for-bit ----
    feats = []
    for v in MV.ALL_VIEWS:
        feats.extend(MV.make_view_examples(views[v][:3], window, stride, v,
                                           "token", raw=True)[:4])
    coll = MV.OnTheFlyCollator(tok, max_len, 0.5, "token", seed=123)
    batch = coll([{k: f[k] for k in f} for f in feats])
    rng2 = random.Random(123)
    words_l, labs_l = [], []
    for f in feats:
        w, l = list(f["words"]), list(f["word_labels"])
        if f["pd"] == 1:
            w, l, _ = MV.punct_dropout(w, l, 0.5, rng2)
        words_l.append([MV.VIEW_TOKENS[f["view"]]] + w)
        labs_l.append([-100] + l)
    enc2 = C.encode({"words": words_l, "word_labels": labs_l}, tok, max_len)
    labs2 = enc2.pop("labels")
    feats2 = [{k: enc2[k][i] for k in enc2.keys()} for i in range(len(words_l))]
    ref = tok.pad(feats2, return_tensors="pt")
    m = ref["input_ids"].shape[1]
    ref["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs2])
    for k in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(batch[k], ref[k]), f"collator batch {k} differs from manual"
    print(f"  OnTheFlyCollator: batch {tuple(batch['input_ids'].shape)} == manual "
          f"dropout+prepend+encode+pad replication (torch.equal)")
    return stats


# =========================================================================== #
# (c) tiny CPU train step: 2 docs per view, all four views, pd 0.5            #
# =========================================================================== #
def s_c_tiny_train(tmp_root=None):
    import argparse as ap_mod
    from eval_local import compute_metrics

    views = gen_views(6, seed=11)
    tmp = tempfile.mkdtemp(prefix="mv_smoke_", dir=tmp_root or None)
    paths = {}
    for v in MV.ALL_VIEWS:
        p = os.path.join(tmp, f"{v}_train.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for d in views[v][:2]:                       # 2 docs per view
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        paths[v] = p
    devp = os.path.join(tmp, "NoPnx-NP_dev.jsonl")
    with open(devp, "w", encoding="utf-8") as f:
        for d in views["NoPnx-NP"][2:4]:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    out = os.path.join(tmp, "mv_tiny")
    order = ["NoPnx-NP", "NoPnx-PA", "NP", "PA"]
    args = ap_mod.Namespace(
        task="NoPnx-NP", out=out, model_name=MODEL_NAME,
        views=",".join(order), train_jsonl=",".join(paths[v] for v in order),
        dev_jsonl=devp, view_indicator="auto", punct_dropout=0.5,
        expect_docs_per_view=2, window=24, stride=12, max_length=64,
        epochs=1, lr=5e-5, batch_size=2, pos_weight=None, seed=42)
    MV.cmd_train(args)

    # reload the HF checkpoint + view-token predict + eval_local round-trip
    cfg = json.load(open(os.path.join(out, "cfg.json")))
    assert cfg["view_indicator"] == "token" and cfg["view_token"] == "[VIEW_NoPnx-NP]"
    assert MV.load_view_word(out) == "[VIEW_NoPnx-NP]"
    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForTokenClassification.from_pretrained(out).eval()
    pargs = ap_mod.Namespace(task="NoPnx-NP", split="dev", model=out, jsonl=devp,
                             out=os.path.join(tmp, "pred_dev.csv"), view=None,
                             threshold=0.5, window=24, overlap=8, max_length=64)
    MV.cmd_predict(pargs)
    gold, pred = {}, {}
    for d in views["NoPnx-NP"][2:4]:
        gold[d["doc_id"]] = d["labels"]
    with open(pargs.out, encoding="utf-8") as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            if row:
                pred[row[0]] = [int(c) for c in row[1].strip()]
    assert set(gold) == set(pred)
    for k in gold:
        assert len(gold[k]) == len(pred[k]), "prediction length mismatch"
    m = compute_metrics(gold, pred)
    print(f"  tiny train done; predict CSV -> eval_local.compute_metrics: "
          f"macro_f1={m['macro_f1']:.4f} P={m['macro_precision']:.4f} "
          f"R={m['macro_recall']:.4f} (tiny-run value, mechanics only)")
    # direct probability sanity via the view-aware stitcher
    d = views["NoPnx-NP"][2]
    words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
    p = MV.predict_probs(words, tok, model, "cpu", 24, 8, 64,
                         view_word="[VIEW_NoPnx-NP]")
    assert len(p) == len(words) and np.isfinite(p).all() and \
        (p >= 0).all() and (p <= 1).all()
    print(f"  predict_probs with view token: {len(p)} probs, all finite in [0,1]")
    return {"tmp": tmp, "macro_f1": m["macro_f1"]}


# =========================================================================== #
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-docs", type=int, default=50, help="section (b) synthetic docs")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tmp-root", default=os.environ.get("MV_SMOKE_TMP"),
                    help="where section (c) writes its temp corpus/checkpoint")
    ap.add_argument("--gpu", action="store_true",
                    help="reserved; this smoke is CPU-only by design")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}  "
          f"cuda.is_available={torch.cuda.is_available()}")
    if not args.gpu:
        # HARD guard: the whole suite (incl. the section (c) Trainer, whose
        # device pick follows torch.cuda.is_available()) must stay on CPU --
        # the GPU belongs to the frontier battery.
        assert not torch.cuda.is_available(), \
            "CUDA still visible -- refusing to run the CPU smoke on a GPU box"
    sections = [
        ("(a) single-view default == plain baseline, bit-for-bit", s_a_bit_for_bit),
        ("(b) multi-view mechanics: windows / view tokens / punct-dropout re-fold",
         lambda: s_b_multiview_mechanics(args.n_docs, args.seed)),
        ("(c) tiny CPU train (2 docs/view, 4 views, pd=0.5)",
         lambda: s_c_tiny_train(args.tmp_root)),
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
    print("ALL CPU SECTIONS PASS")


if __name__ == "__main__":
    main()
