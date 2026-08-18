"""CPU smokes for src/cuttoken_train.py (Noor's cut-token grouping model).

Runs the four pre-registered checks, CPU-ONLY (CUDA hidden with -1 before
torch import — the GPU is battery-owned):

  (a) INPUT BUILDER ROUND-TRIP on 20 real docs: per window, #[CUT] units ==
      #words - 1; every [CUT]'s label equals the gold label of its LEFT word
      (exact absolute alignment against data/NoPnx-NP_train.jsonl); after
      tokenization every [CUT] is a single subtoken carrying that label and
      every word subtoken is -100.
  (b) TINY CPU TRAIN STEP for BOTH init modes (base and tapt; the tapt smoke
      also exercises --cut-mlm-epochs 1, i.e. the full stage-2a -> stage-2
      path through the real CLI).
  (c) PREDICT on 2 dev docs -> a valid submission CSV that eval_local accepts
      (compute_metrics in-process AND the eval_local.py CLI), plus the
      word-grid npz export (lengths, [0,1] range, doc-final 0.0).
  (d) THE CUT-MLM STAGE MASKS ONLY WORDS: over many collator draws, every
      supervised position (label != -100) was a WORD token; [CUT]/[PAR]/CLS/
      SEP/pad are never masked, never predicted, and [CUT] inputs are never
      corrupted; masked fraction is ~15% of word tokens.

Artifacts land in --out-root (default runs/cuttoken_smoke_2026-07-10; the dir
must NOT pre-exist — runs/ is append-only).

Usage:  python src/smoke_cuttoken.py [--out-root runs/cuttoken_smoke_...]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only; -1 (not "") hides CUDA

import argparse
import json
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)

from cuttoken_train import (CUT_TOKEN, CutMLMCollator, build_cut_units,
                            encode_cut_batch, make_cut_windows)
from data import MODEL_PAR_TOKEN, load_jsonl
from eval_local import compute_metrics

TRAIN_JSONL = os.path.join(ROOT, "data", "NoPnx-NP_train.jsonl")
DEV_JSONL = os.path.join(ROOT, "data", "NoPnx-NP_dev.jsonl")

PASS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    PASS.append((name, ok))
    if not ok:
        raise SystemExit(f"SMOKE FAILED at: {name}")


def smoke_a_builder(tokenizer) -> None:
    print("\n=== (a) input builder round-trip on 20 real docs ===")
    docs = load_jsonl(TRAIN_JSONL)[:20]
    W, S = 203, 101                                  # the chosen window math
    windows = make_cut_windows(docs, W, S)
    gold = {d["doc_id"]: d["labels"] for d in docs}
    toks = {d["doc_id"]: d["tokens"] for d in docs}

    n_cut_total = 0
    for w in windows:
        units, ul = build_cut_units(w["words"], w["word_labels"])
        n_words = len(w["words"])
        n_cuts = sum(1 for u in units if u == CUT_TOKEN)
        assert n_cuts == n_words - 1, (n_cuts, n_words)
        n_cut_total += n_cuts
        # exact absolute alignment to gold: [CUT] unit 2i+1 carries the gold
        # label of absolute word (start + i)
        g = gold[w["doc_id"]]
        t = toks[w["doc_id"]]
        for j, (u, l) in enumerate(zip(units, ul)):
            if u == CUT_TOKEN:
                abs_i = w["start"] + (j - 1) // 2
                expect = -100 if t[abs_i] == "\n" else int(g[abs_i])
                assert l == expect, (w["doc_id"], abs_i, l, expect)
            else:
                assert l == -100
    check("(a1) [CUT] count == words-1 in every window",
          True, f"{len(windows)} windows, {n_cut_total} [CUT]s")
    check("(a2) every [CUT] label == gold label of its left word (20 docs)", True)

    # encoded alignment: labels only on single-subtoken [CUT] units
    cut_id = tokenizer.convert_tokens_to_ids(CUT_TOKEN)
    ex = {"words": [w["words"] for w in windows[:40]],
          "word_labels": [w["word_labels"] for w in windows[:40]]}
    enc = encode_cut_batch(ex, tokenizer, 512)
    n_lab = 0
    for i in range(len(ex["words"])):
        ids, labs = enc["input_ids"][i], enc["labels"][i]
        uids = enc.word_ids(batch_index=i)
        for pos, (tid, l) in enumerate(zip(ids, labs)):
            if l != -100:
                n_lab += 1
                assert tid == cut_id, f"label on non-[CUT] token id {tid}"
                assert uids[pos] is not None and uids[pos] % 2 == 1
        # every [CUT] occupies exactly one subtoken
        from collections import Counter
        c = Counter(u for u, t_ in zip(uids, ids) if u is not None and t_ == cut_id)
        assert all(v == 1 for v in c.values()), "multi-subtoken [CUT]"
    check("(a3) encoded: labels sit ONLY on single-subtoken [CUT] positions",
          True, f"{n_lab} labeled [CUT] positions in 40 windows")


def run_cli(args_list, log_path):
    print("  $", " ".join(args_list))
    env = dict(os.environ, PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1")
    with open(log_path, "w", encoding="utf-8") as lf:
        r = subprocess.run(args_list, cwd=ROOT, env=env, stdout=lf,
                           stderr=subprocess.STDOUT)
    tail = open(log_path, encoding="utf-8").read().splitlines()[-25:]
    print("  --- log tail ---")
    for ln in tail:
        print("   ", ln)
    return r.returncode


def smoke_b_train(out_root: str) -> str:
    print("\n=== (b) tiny CPU train step, BOTH init modes ===")
    py = sys.executable
    common = ["--task", "NoPnx-NP", "--train-jsonl", TRAIN_JSONL,
              "--dev-jsonl", DEV_JSONL, "--max-docs", "2", "--max-dev-docs", "1",
              "--window", "40", "--epochs", "1", "--batch-size", "2",
              "--seed", "42", "--cpu"]
    base_dir = os.path.join(out_root, "tiny_base")
    rc = run_cli([py, os.path.join("src", "cuttoken_train.py"), "train",
                  "--init-mode", "base", "--out-dir", base_dir] + common,
                 os.path.join(out_root, "tiny_base.log"))
    check("(b1) tiny CPU train, --init-mode base", rc == 0
          and os.path.exists(os.path.join(base_dir, "model.safetensors")))

    tapt_dir = os.path.join(out_root, "tiny_tapt_mlm1")
    rc = run_cli([py, os.path.join("src", "cuttoken_train.py"), "train",
                  "--init-mode", "tapt", "--cut-mlm-epochs", "1",
                  "--out-dir", tapt_dir] + common,
                 os.path.join(out_root, "tiny_tapt_mlm1.log"))
    log = open(os.path.join(out_root, "tiny_tapt_mlm1.log"), encoding="utf-8").read()
    check("(b2) tiny CPU train, --init-mode tapt (+ --cut-mlm-epochs 1)",
          rc == 0 and os.path.exists(os.path.join(tapt_dir, "model.safetensors")))
    check("(b3) tapt stage-1 encoder VERIFIED at load (no fallback)",
          "tapt encoder verified" in log and "TAPT_INCOMPATIBLE" not in log)
    check("(b4) stage 2a ran and saved the cut-adapted encoder",
          "saved cut-adapted encoder" in log
          and os.path.exists(os.path.join(tapt_dir, "cutmlm", "model.safetensors")))
    return base_dir


def smoke_c_predict(out_root: str, model_dir: str) -> None:
    print("\n=== (c) predict on 2 dev docs -> CSV eval_local accepts ===")
    py = sys.executable
    csv_path = os.path.join(out_root, "pred_dev2.csv")
    npz_path = os.path.join(out_root, "pred_dev2.npz")
    rc = run_cli([py, os.path.join("src", "cuttoken_train.py"), "predict",
                  "--model", model_dir, "--task", "NoPnx-NP",
                  "--split", "dev", "--jsonl", DEV_JSONL, "--max-docs", "2",
                  "--out", csv_path, "--npz-out", npz_path, "--cpu"],
                 os.path.join(out_root, "predict2.log"))
    check("(c1) predict CLI ran", rc == 0 and os.path.exists(csv_path))

    docs = load_jsonl(DEV_JSONL)[:2]
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    from eval_local import load_predictions
    pred = load_predictions(csv_path)
    m = compute_metrics(gold, pred)                 # raises on any invalid CSV
    check("(c2) eval_local.compute_metrics accepts the CSV",
          0.0 <= m["macro_f1"] <= 1.0,
          f"2-doc macro F1 = {m['macro_f1']*100:.2f} (tiny 2-doc model; "
          f"plumbing check, not a quality claim)")

    # the eval_local CLI end-to-end, against a 2-doc gold jsonl
    g2 = os.path.join(out_root, "dev2_gold.jsonl")
    with open(g2, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    env = dict(os.environ, PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([py, os.path.join("src", "eval_local.py"),
                        "--predictions", csv_path, "--gold-jsonl", g2],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    print("  eval_local.py stdout:", " | ".join(
        ln for ln in r.stdout.splitlines() if ln.strip()))
    check("(c3) eval_local.py CLI scores the CSV", r.returncode == 0
          and "F1:" in r.stdout)

    z = np.load(npz_path)
    ok = True
    for d in docs:
        p = z[d["doc_id"]]
        ok &= len(p) == len(d["tokens"])
        ok &= bool((p >= 0).all() and (p <= 1).all())
        ok &= float(p[-1]) == 0.0                   # doc-final gap: no [CUT]
    check("(c4) npz word-grid: per-doc lengths, [0,1] probs, doc-final 0.0", ok)


def smoke_d_mlm_masks_words_only(tokenizer) -> None:
    print("\n=== (d) cut-MLM collator masks WORD tokens only ===")
    docs = load_jsonl(TRAIN_JSONL)[:6]
    windows = make_cut_windows(docs, 60, 30)
    ex = {"words": [w["words"] for w in windows],
          "word_labels": [w["word_labels"] for w in windows]}
    enc = encode_cut_batch(ex, tokenizer, 512, with_labels=False)
    feats = [{"input_ids": enc["input_ids"][i],
              "attention_mask": enc["attention_mask"][i],
              "token_type_ids": enc["token_type_ids"][i]}
             for i in range(len(enc["input_ids"]))]

    torch.manual_seed(42)
    coll = CutMLMCollator(tokenizer, 0.15)
    special_ids = set(tokenizer.all_special_ids)
    cut_id = tokenizer.convert_tokens_to_ids(CUT_TOKEN)
    assert cut_id in special_ids and \
        tokenizer.convert_tokens_to_ids(MODEL_PAR_TOKEN) in special_ids

    n_masked = n_maskable = n_draws = 0
    for _ in range(8):                              # many draws
        for lo in range(0, len(feats), 8):
            batch = coll(feats[lo:lo + 8])
            orig = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f["input_ids"]) for f in feats[lo:lo + 8]],
                batch_first=True, padding_value=tokenizer.pad_token_id)
            labels = batch["labels"]
            sup = labels != -100
            # every supervised position was a WORD token (not special/[CUT])
            assert not torch.isin(orig, torch.tensor(sorted(special_ids)))[sup].any(), \
                "a special/[CUT]/[PAR] position got masked"
            # [CUT] inputs are never corrupted either
            assert (batch["input_ids"][orig == cut_id] == cut_id).all(), \
                "a [CUT] input token was altered"
            maskable = (~torch.isin(orig, torch.tensor(sorted(special_ids)))) \
                & (batch["attention_mask"] == 1)
            n_masked += int(sup.sum())
            n_maskable += int(maskable.sum())
            n_draws += 1
    frac = n_masked / max(n_maskable, 1)
    check("(d1) masked positions are ALWAYS word tokens; [CUT] never masked "
          "or corrupted", True, f"{n_draws} batches")
    check("(d2) masked fraction ~15% of WORD tokens",
          0.12 <= frac <= 0.18, f"measured {frac:.4f}")
    check("(d3) something is actually masked", n_masked > 0, f"{n_masked} positions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=os.path.join(
        ROOT, "runs", "cuttoken_smoke_2026-07-10"))
    args = ap.parse_args()

    if os.path.exists(args.out_root):
        raise SystemExit(f"REFUSING: {args.out_root} exists (runs/ is append-only); "
                         f"pass a fresh --out-root")
    os.makedirs(args.out_root)
    print(f"smoke artifacts -> {args.out_root}")
    assert not torch.cuda.is_available(), "CUDA visible — smokes are CPU-only"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [MODEL_PAR_TOKEN, CUT_TOKEN]})

    smoke_a_builder(tokenizer)
    base_dir = smoke_b_train(args.out_root)
    smoke_c_predict(args.out_root, base_dir)
    smoke_d_mlm_masks_words_only(tokenizer)

    print("\n================ SMOKE SUMMARY ================")
    for name, ok in PASS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"ALL {len(PASS)} CHECKS PASSED")


if __name__ == "__main__":
    main()
