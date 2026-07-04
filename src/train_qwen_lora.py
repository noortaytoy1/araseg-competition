"""LLM voter: Qwen2.5-3B + LoRA (bf16, no quantization) as a token classifier.
Architecture family #5 (causal decoder). Closed-legal: pretrained model,
fine-tuned only on AraSeg train. Run in venv27 (torch 2.12 + peft).

Same framing as every voter: label on last subword of each whitespace token,
180-word windows (stride 90), window-averaged probs. Dumps
probs/{task}_{split}_{suffix}.npz + LoRA adapter in open_runs/.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="NoPnx-NP")
ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--suffix", default="qwen")
ARGS = ap.parse_args()

WIN, STRIDE, BATCH = 180, 90, 8
DEV = torch.device("cuda")


def windows(tokens, labels):
    n = len(tokens)
    if n <= WIN:
        return [(0, tokens, labels)]
    out, i = [], 0
    while i < n:
        out.append((i, tokens[i:i + WIN], labels[i:i + WIN] if labels else None))
        if i + WIN >= n:
            break
        i += STRIDE
    return out


def encode(tok, words):
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=1024,
              return_tensors="pt")
    wid = enc.word_ids()
    last = {}
    for pos, w in enumerate(wid):
        if w is not None:
            last[w] = pos
    return enc, last


def doc_probs(model, tok, tokens):
    acc, cnt = np.zeros(len(tokens)), np.zeros(len(tokens))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for off, wt, _ in windows(tokens, None):
            enc, last = encode(tok, wt)
            logits = model(input_ids=enc["input_ids"].to(DEV),
                           attention_mask=enc["attention_mask"].to(DEV)).logits[0, :, 0]
            p = torch.sigmoid(logits.float()).cpu().numpy()
            for w, pos in last.items():
                acc[off + w] += p[pos]
                cnt[off + w] += 1
    return acc / np.maximum(cnt, 1)


def dev_f1(model, tok, docs):
    ys, ps = [], []
    for d in docs:
        pr = doc_probs(model, tok, d["tokens"])
        ys.extend(d["labels"])
        ps.extend((pr >= 0.5).astype(int).tolist())
    return f1_score(ys, ps, zero_division=0) * 100


def main() -> None:
    task = ARGS.task
    train = load_jsonl(f"data/{task}_train.jsonl")
    dev = load_jsonl(f"data/{task}_dev.jsonl")
    tok = AutoTokenizer.from_pretrained(ARGS.model)
    model = AutoModelForTokenClassification.from_pretrained(
        ARGS.model, num_labels=1, torch_dtype=torch.bfloat16,
        pad_token_id=tok.pad_token_id)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      modules_to_save=["score"])
    model = get_peft_model(model, lcfg).to(DEV)
    model.print_trainable_parameters()

    samples = []
    for d in train:
        for _, wt, wl in windows(d["tokens"], d["labels"]):
            samples.append((wt, wl))
    rate = float(np.mean([l for _, wl in samples for l in wl]))
    pos_w = torch.tensor((1 - rate) / max(rate, 1e-4), device=DEV)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=ARGS.lr)
    rng = np.random.default_rng(0)

    best, best_state = -1.0, None
    for ep in range(ARGS.epochs):
        model.train()
        order = rng.permutation(len(samples))
        tot = 0.0
        for bi in range(0, len(order), BATCH):
            opt.zero_grad()
            loss_acc = 0.0
            for si in order[bi:bi + BATCH]:
                wt, wl = samples[si]
                enc, last = encode(tok, wt)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=enc["input_ids"].to(DEV),
                                   attention_mask=enc["attention_mask"].to(DEV)).logits[0, :, 0]
                pos = torch.tensor(sorted(last.values()), device=DEV)
                y = torch.tensor([wl[w] for w in sorted(last)], device=DEV, dtype=torch.float)
                loss = F.binary_cross_entropy_with_logits(
                    logits[pos].float(), y, pos_weight=pos_w) / BATCH
                loss.backward()
                loss_acc += float(loss)
            opt.step()
            tot += loss_acc
        model.eval()
        f1 = dev_f1(model, tok, dev)
        print(f"epoch {ep + 1}: loss {tot:.1f} dev-F1(0.5) {f1:.2f}", flush=True)
        if f1 > best:
            best = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()
                          if "lora" in k or "score" in k or "modules_to_save" in k}

    for k, v in best_state.items():
        model.state_dict()[k].copy_(v)
    model.eval()
    outdir = f"open_runs/{task.lower()}-{ARGS.suffix}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(best_state, f"{outdir}/lora_state.pt")
    for split in ["dev", "test"]:
        docs = load_jsonl(f"data/{task}_{split}.jsonl")
        np.savez(f"probs/{task}_{split}_{ARGS.suffix}.npz",
                 **{d["doc_id"]: doc_probs(model, tok, d["tokens"]) for d in docs})
        print("cached", split, flush=True)
    print(f"QWEN DONE task={task} best-dev-F1(0.5)={best:.2f}")


if __name__ == "__main__":
    main()
