"""BoundRL dev evaluation: greedy generation -> reconstruction -> boundary
labels -> eval_local per-doc macro-F1, plus paper-style metrics.

Inference (paper Sec 5.2): temperature 0 (greedy decoding).  Each dev
document is split into chunks of at most --window-tokens model tokens.
Unlike training, dev chunks CANNOT be sentence-aligned (boundaries are what
we are predicting), so chunks are word-aligned greedy packs.  Seam handling
(documented pilot limitation): the label of the last word of every interior
chunk is set to 0 (the ~8.5% boundary base rate makes 0 the majority-correct
choice; the first segment of the following chunk always starts at its word 0
by construction, which carries no boundary information).  The final word of
each document is labelled 1, matching the gold convention.

Within a chunk: parse the completion, iterative leftmost-match reconstruct,
and every located segment start at chunk-word position p > 0 sets
label[offset + p - 1] = 1.

Outputs a Kaggle-style predictions CSV ("Document ID","Prediction"), scores
it with eval_local.compute_metrics (identical to the official metric), and
reports paper metrics (rho_rec / EM / F1char, macro-averaged over docs,
computed doc-level against gold segment spans).

Usage:
  python src/boundrl_eval.py --model Qwen/Qwen3-1.7B --adapter runs/.../grpo \
    --dev-jsonl data/NoPnx-NP_dev.jsonl --out-dir runs/boundrl_pilot_2026-07-10/eval_dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boundrl_data import (  # noqa: E402
    build_prompt,
    gold_segment_starts,
    parse_target,
    reconstruct,
    spans_from_starts,
)
from boundrl_reward import segmentation_reward  # noqa: E402
from boundrl_sft import load_model_and_tokenizer  # noqa: E402
from data import load_jsonl  # noqa: E402
from eval_local import compute_metrics  # noqa: E402


def word_chunks(words: List[str], count_tokens: Callable[[str], int], budget: int) -> List[dict]:
    """Word-aligned greedy packing into <= budget model tokens (dev has no
    gold sentence boundaries to align to)."""
    chunks, cur, cur_tokens, off = [], [], 0, 0
    for w in words:
        t = count_tokens(w + " ")
        if cur and cur_tokens + t > budget:
            chunks.append({"word_offset": off, "words": cur})
            off += len(cur)
            cur, cur_tokens = [], 0
        cur.append(w)
        cur_tokens += t
    if cur:
        chunks.append({"word_offset": off, "words": cur})
    return chunks


def predict_doc(model, tok, words: List[str], budget: int, max_completion: int, device) -> dict:
    import torch

    labels = [0] * len(words)
    all_spans = []
    n_dropped = 0
    chunks = word_chunks(words, lambda s: len(tok(s, add_special_tokens=False).input_ids), budget)
    for ch in chunks:
        prompt = build_prompt(" ".join(ch["words"]))
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,  # temperature 0 / greedy, per the paper
                max_new_tokens=max_completion,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        completion = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        seqs = parse_target(completion)
        if not seqs:
            continue
        spans, dropped = reconstruct(ch["words"], seqs)
        n_dropped += dropped
        off = ch["word_offset"]
        for sp in spans:
            if sp is None:
                continue
            b, e = sp
            all_spans.append((off + b, off + e))
            if b > 0:
                labels[off + b - 1] = 1
        # seam: last word of interior chunk stays 0 (documented above)
    labels[-1] = 1  # gold convention: doc end is always a boundary
    return {"labels": labels, "pred_spans": all_spans, "n_dropped": n_dropped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--fallback-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (SFT or GRPO output)")
    ap.add_argument("--dev-jsonl", default="data/NoPnx-NP_dev.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-tokens", type=int, default=2048)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--max-docs", type=int, default=None, help="smoke: only first N docs")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = ap.parse_args()

    import pandas as pd
    import torch

    model, tok, used_model = load_model_and_tokenizer(args.model, args.fallback_model, args.dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    device = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()

    docs = load_jsonl(args.dev_jsonl)
    if args.max_docs:
        docs = docs[: args.max_docs]

    os.makedirs(args.out_dir, exist_ok=True)
    rows, paper_metrics = [], []
    for i, doc in enumerate(docs):
        words = doc["tokens"]
        pred = predict_doc(model, tok, words, args.window_tokens, args.max_completion, device)
        rows.append({"Document ID": doc["doc_id"], "Prediction": "".join(map(str, pred["labels"]))})
        gold_spans = spans_from_starts(gold_segment_starts(doc["labels"]), len(words))
        m = segmentation_reward(words, gold_spans, pred["pred_spans"])
        m["n_dropped"] = pred["n_dropped"]
        paper_metrics.append(m)
        if (i + 1) % 10 == 0:
            print(f"[boundrl_eval] {i + 1}/{len(docs)} docs")

    csv_path = os.path.join(args.out_dir, "predictions.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    gold = {d["doc_id"]: list(d["labels"]) for d in docs}
    predm = {r["Document ID"]: [int(c) for c in r["Prediction"]] for r in rows}
    m = compute_metrics(gold, predm)

    n = len(paper_metrics)
    report = {
        "model_used": used_model,
        "adapter": args.adapter,
        "dev_jsonl": args.dev_jsonl,
        "n_docs": n,
        "macro_precision": m["macro_precision"] * 100,
        "macro_recall": m["macro_recall"] * 100,
        "macro_f1": m["macro_f1"] * 100,
        "paper_metrics_macro": {
            "rho_rec": 100 * sum(x["rho_rec"] for x in paper_metrics) / n,
            "em": 100 * sum(x["em"] for x in paper_metrics) / n,
            "f1char": 100 * sum(x["f1char"] for x in paper_metrics) / n,
            "reward": sum(x["reward"] for x in paper_metrics) / n,
            "dropped_segments_total": sum(x["n_dropped"] for x in paper_metrics),
        },
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[boundrl_eval] boundary macro-F1 = {report['macro_f1']:.2f}  (predictions: {csv_path})")


if __name__ == "__main__":
    main()
