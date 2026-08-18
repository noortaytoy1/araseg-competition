"""DECODE-SIDE STACK evaluator (CPU-ONLY, ADDITIVE) — integration-battery companion.

Given ONE trained arm's HF token-classification checkpoint (e.g. an
int_union / int_union_rs arm from runs/integration_battery_2026-07-10, or any
run dir readable by AutoModelForTokenClassification), evaluate the cumulative
DECODE-SIDE stack on dev.  NO training, NO GPU, NO edits to any existing
module — every reused piece is IMPORTED from the frozen code, never re-typed:

  stage (a)  plain predict           predict.predict_doc, window 180 overlap 60
                                     (the canonical baseline decode)
  stage (b)  + center-weighted       multi_stride_decode.collect_stride_views +
             multi-stride            multi_probs_from_views(center_weighted=True)
             accumulation            — the WINNING arm (c) of the multi-stride
                                     probe (strides {30,45,60,90}, triangular
                                     center weights), imported VERBATIM from
                                     src/multi_stride_decode.py, not ported by
                                     hand.
  stage (c)  + U-Net blend           alpha * p_unet + (1 - alpha) * p_model,
                                     alpha in {0.5, 0.8, 0.9}, using the
                                     EXISTING trained U-Net checkpoint
                                     (runs/unet2d_2026-07-10/s42/best.pt).
                                     p_model is the stage-(b) prob vector (the
                                     stack is cumulative); the same blend on the
                                     stage-(a) probs is also printed as a
                                     reference line so the two decode-side
                                     pieces can be attributed separately.

U-NET DEV CACHE REGENERATION (required, per the integration-battery order):
the U-Net's channel 2 is the tagger's single-window boundary probs, so it must
be built from THIS arm's model — the existing cache
unet2d/NoPnx-NP_s42_dev_mid8.npz carries union-s42 probs and would be silently
wrong for any other checkpoint.  This tool therefore regenerates the dev cache
with unet2d_train.cmd_cache (the exact frozen cache code, called in-process)
using --model <this arm>, writing to a NEW cache file that names the arm.  An
existing cache file at the target path is REUSED only if its recorded meta
(model + jsonl) matches exactly; on mismatch we ABORT rather than overwrite
(caches are evidence).  Channel 1 (mid-layer-8 similarity) is regenerated from
the same arm forward pass — that is what cmd_cache does; both channels are
self-consistently from THIS model, exactly as the original cache was
self-consistently from union-s42.

HONESTY NOTE (printed verbatim in every report): the U-Net was trained with
union-s42 probs as channel 2 (and union-s42 mid-layer-8 similarities as
channel 1) — running it on a DIFFERENT model's probs/similarities is a
distribution shift.  The stage-(c) numbers are reported AS MEASURED under that
shift, no hand-waving and no correction; if they are bad, the honest reading is
"the frozen U-Net does not transfer", not "the stack failed".

Scoring: eval_local.compute_metrics (the official-mirror per-doc macro-F1),
via multi_stride_decode.sweep_threshold — the SAME >= operator, SAME metric
path, and the SAME fixed grid 0.30..0.70 step .05 as score_arm.py.  Every
stage line reports F1@0.5 and own-best F1 (dev-selected, optimistic — the
standard caveat).  Dev is EVALUATION/SELECTION only.

Usage (CPU only — the battery owns the GPU):
  CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 python src/decode_stack.py \
      --model runs/integration_battery_2026-07-10/int_union_rs_s42 \
      --dev data/NoPnx-NP_dev.jsonl \
      --unet-ckpt runs/unet2d_2026-07-10/s42/best.pt \
      --cache-out unet2d/decode_stack_int_union_rs_s42_dev_mid8.npz \
      --json-out runs/integration_battery_2026-07-10/decode_stack_int_union_rs_s42.json

Runtime note: stage (b) runs one forward per DISTINCT window start across the
four strides and stage (c) regenerates a full dev cache — on CPU over the full
222-doc dev this is slow (hours, not minutes).  --stages a|ab|abc allows
partial runs; probs computed here are also dumped next to --json-out so a
rerun never has to repeat a finished stage by hand.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl
from predict import predict_doc
# the winning multi-stride arm (c), imported VERBATIM — no re-implementation
from multi_stride_decode import (MULTI_STRIDES, collect_stride_views,
                                 multi_probs_from_views, sweep_threshold)
# the frozen U-Net pieces (cache regeneration, forward, doc stitching)
import unet2d_train

GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]  # same grid as score_arm.py
ALPHAS = [0.5, 0.8, 0.9]

HONESTY_NOTE = (
    "HONESTY NOTE: the U-Net was trained with union-s42 probs as channel 2 "
    "(and union-s42 mid-layer-8 similarities as channel 1) — using it on a "
    "different model's probs is a distribution shift. Stage-(c) numbers are "
    "reported AS MEASURED under that shift; no correction, no hand-waving.")


def assert_cpu_only():
    assert not torch.cuda.is_available(), \
        "CPU-ONLY tool: set CUDA_VISIBLE_DEVICES=-1 (the battery owns the GPU)"


def score_stage(name, docs, gold, prob_by_doc):
    """One stage line: F1@0.5 + own-best over the fixed grid. Returns dict."""
    best_t, best_m, curve = sweep_threshold(docs, gold, prob_by_doc, GRID)
    f1_05 = next(f for t, _, _, f in curve if t == 0.50)
    r = {"f1_05": round(100 * f1_05, 4),
         "best_f1": round(100 * best_m["macro_f1"], 4),
         "best_thr": best_t,
         "P_best": round(100 * best_m["macro_precision"], 4),
         "R_best": round(100 * best_m["macro_recall"], 4)}
    print(f"{name:56s} F1@0.5={r['f1_05']:7.3f}  own-best={r['best_f1']:7.3f} "
          f"@thr={best_t:.2f}  (P={r['P_best']:.2f} R={r['R_best']:.2f})")
    return r


def regen_or_reuse_cache(model_dir, dev_jsonl, cache_out, mid_layer, window,
                         overlap, max_length):
    """Regenerate the U-Net dev cache from THIS model, or reuse an exactly
    matching one.  ABORT (never overwrite) on a meta mismatch."""
    if os.path.exists(cache_out):
        meta = unet2d_train.load_cache(cache_out)["meta"]
        if meta.get("model") == model_dir and meta.get("jsonl") == dev_jsonl:
            print(f"# reusing existing cache (meta matches this model): {cache_out}")
            return
        raise SystemExit(
            f"ABORT: {cache_out} exists but was built from model="
            f"{meta.get('model')!r} jsonl={meta.get('jsonl')!r}, not this arm. "
            "Caches are evidence — pass a fresh --cache-out instead of overwriting.")
    print(f"# regenerating U-Net dev cache from THIS model (channel 2 = this "
          f"model's probs): {cache_out}")
    ns = argparse.Namespace(model=model_dir, jsonl=dev_jsonl, out=cache_out,
                            mid_layer=mid_layer, window=window, overlap=overlap,
                            min_len=4, max_length=max_length)
    unet2d_train.cmd_cache(ns)  # the exact frozen cache code, in-process


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="trained arm's HF checkpoint dir (run dir)")
    ap.add_argument("--dev", required=True, help="dev jsonl (evaluation only)")
    ap.add_argument("--task", default="NoPnx-NP")
    ap.add_argument("--unet-ckpt", default="runs/unet2d_2026-07-10/s42/best.pt",
                    help="EXISTING trained U-Net checkpoint (not retrained here)")
    ap.add_argument("--cache-out", default=None,
                    help="where to write/reuse the regenerated U-Net dev cache "
                         "(default: unet2d/decode_stack_<model-basename>_dev_mid8.npz)")
    ap.add_argument("--stages", choices=["a", "ab", "abc"], default="abc",
                    help="cumulative stages to run (CPU time control)")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--mid-layer", type=int, default=8)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    assert_cpu_only()
    device = "cpu"
    print(f"# DECODE-SIDE STACK  |  model={args.model}")
    print(f"# stages={args.stages}  grid={GRID[0]:.2f}..{GRID[-1]:.2f} step .05  "
          f"alphas={ALPHAS}  unet={args.unet_ckpt}")
    print("# " + HONESTY_NOTE)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()
    docs = load_jsonl(args.dev)
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    print(f"# dev docs={len(docs)}")

    rep = {"model": args.model, "dev": args.dev, "stages": args.stages,
           "grid": GRID, "alphas": ALPHAS, "unet_ckpt": args.unet_ckpt,
           "honesty_note": HONESTY_NOTE, "results": {}}

    # ---------------- stage (a): plain predict (canonical baseline decode) ----
    prob_a = {}
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        prob_a[d["doc_id"]] = predict_doc(words, tok, model, device,
                                          args.window, args.overlap, args.max_length)
    print("\n==== stage (a): plain predict (window=%d overlap=%d) ====" %
          (args.window, args.overlap))
    rep["results"]["a_plain"] = score_stage("(a) plain predict", docs, gold, prob_a)

    prob_b = None
    if "b" in args.stages:
        # ------------ stage (b): + center-weighted multi-stride accumulation --
        # the winning arm (c) of src/multi_stride_decode.py, called verbatim
        prob_b = {}
        for d in docs:
            words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
            views, _ = collect_stride_views(words, tok, model, device,
                                            args.window, MULTI_STRIDES, args.max_length)
            prob_b[d["doc_id"]] = multi_probs_from_views(views, len(words),
                                                         center_weighted=True)
        print("\n==== stage (b): + multi-stride %s center-weighted ====" % MULTI_STRIDES)
        rep["results"]["b_multistride_cw"] = score_stage(
            "(b) + center-weighted multi-stride", docs, gold, prob_b)

    if "c" in args.stages:
        # ------------ stage (c): + U-Net blend --------------------------------
        cache_out = args.cache_out or os.path.join(
            "unet2d", f"decode_stack_{os.path.basename(os.path.normpath(args.model))}"
                      f"_dev_mid8.npz")
        regen_or_reuse_cache(args.model, args.dev, cache_out, args.mid_layer,
                             args.window, args.overlap, args.max_length)
        dv = unet2d_train.load_cache(cache_out)
        assert [str(x) for x in dv["doc_ids"]] == [d["doc_id"] for d in docs], \
            "regenerated cache doc order != dev jsonl doc order"

        blob = torch.load(args.unet_ckpt, map_location="cpu", weights_only=False)
        in_ch = int(blob.get("in_ch", 2))
        assert in_ch == 2, ("stage (c) needs the 2-channel U-Net (channel 2 = "
                            "tagger probs); this ckpt is the in_ch=1 ablation")
        unet = unet2d_train.build_model(base=blob["base"], in_ch=in_ch)
        unet.load_state_dict(blob["state_dict"])
        pw = unet2d_train.predict_windows(unet, dv)
        per_doc, unc = unet2d_train.stitch(dv, pw)
        p_unet = {d["doc_id"]: per_doc[i] for i, d in enumerate(docs)}
        print(f"\n==== stage (c): + U-Net blend (ckpt seed={blob['seed']} "
              f"epoch={blob['epoch']}; uncovered positions -> p=0: {unc}) ====")
        print("# " + HONESTY_NOTE)
        rep["results"]["c_unet_solo_ref"] = score_stage(
            "(ref) U-Net solo (this model's channels)", docs, gold, p_unet)

        base_for_c = prob_b if prob_b is not None else prob_a
        base_name = "b" if prob_b is not None else "a"
        for alpha in ALPHAS:
            blend = {did: alpha * p_unet[did] + (1.0 - alpha) * base_for_c[did]
                     for did in p_unet}
            rep["results"][f"c_blend_a{alpha}_on_{base_name}"] = score_stage(
                f"(c) alpha={alpha}*unet + {round(1-alpha, 2)}*model(stage {base_name})",
                docs, gold, blend)
        if prob_b is not None:
            # reference: same blend on the PLAIN stage-(a) probs, to attribute
            # the two decode-side pieces separately
            for alpha in ALPHAS:
                blend = {did: alpha * p_unet[did] + (1.0 - alpha) * prob_a[did]
                         for did in p_unet}
                rep["results"][f"c_blend_a{alpha}_on_a_ref"] = score_stage(
                    f"(ref) alpha={alpha}*unet + {round(1-alpha, 2)}*model(stage a)",
                    docs, gold, blend)

    print("\n" + HONESTY_NOTE)
    print("CAVEAT: own-best thresholds are dev-selected (optimistic); F1@0.5 is "
          "the honest fixed-operating-point line. Dev is evaluation only.")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, default=float)
        # dump the per-doc probs beside the report so no stage is ever re-run by hand
        probs_out = os.path.splitext(args.json_out)[0] + "_probs.npz"
        arrays = {f"a::{did}": np.asarray(p) for did, p in prob_a.items()}
        if prob_b is not None:
            arrays.update({f"b::{did}": np.asarray(p) for did, p in prob_b.items()})
        if "c" in args.stages:
            arrays.update({f"unet::{did}": np.asarray(p) for did, p in p_unet.items()})
        np.savez_compressed(probs_out, **arrays)
        print(f"# wrote {args.json_out} and {probs_out}")


if __name__ == "__main__":
    main()
