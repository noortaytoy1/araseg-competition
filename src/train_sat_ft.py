"""Fine-tune SaT-12L-sm (wtpsplit, custom xlm-token arch) on AraSeg NoPnx-NP as a
decorrelated ensemble voter. Run in venv27 (torch 2.12 + wtpsplit).

Word-level framing mirrors train_encoder.py: label on the LAST subword of each
whitespace token, overlapping 180-word windows (stride 90), window-averaged
probabilities at inference. Saves per-doc word probs for dev+test:
  probs/NoPnx-NP_dev_satft.npz   probs/NoPnx-NP_test_satft.npz
and the fine-tuned state dict in open_runs/nopnx-np-satft/ (for blind-day use).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import wtpsplit.models  # noqa: F401 -- registers "xlm-token" with transformers
from transformers import AutoModelForTokenClassification, XLMRobertaTokenizerFast

from data import load_jsonl

TASK = "NoPnx-NP"
MODEL = "segment-any-text/sat-12l-sm"
WIN, STRIDE, EPOCHS, LR, BATCH = 180, 90, 6, 1e-5, 8
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
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=510,
              return_tensors="pt")
    wid = enc.word_ids()
    last = {}
    for pos, w in enumerate(wid):
        if w is not None:
            last[w] = pos  # final assignment wins = last subword
    return enc, last


def doc_probs(model, tok, tokens):
    acc, cnt = np.zeros(len(tokens)), np.zeros(len(tokens))
    with torch.no_grad():
        for off, wt, _ in windows(tokens, None):
            enc, last = encode(tok, wt)
            logits = model(input_ids=enc["input_ids"].to(DEV),
                           attention_mask=enc["attention_mask"].to(DEV)).logits[0, :, 0]
            p = torch.sigmoid(logits).float().cpu().numpy()
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
    train = load_jsonl(f"data/{TASK}_train.jsonl")
    dev = load_jsonl(f"data/{TASK}_dev.jsonl")
    try:
        tok = XLMRobertaTokenizerFast.from_pretrained(MODEL)
    except Exception:
        tok = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModelForTokenClassification.from_pretrained(MODEL).to(DEV)

    samples = []
    for d in train:
        for _, wt, wl in windows(d["tokens"], d["labels"]):
            samples.append((wt, wl))
    rate = float(np.mean([l for _, wl in samples for l in wl]))
    pos_w = torch.tensor((1 - rate) / max(rate, 1e-4), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = np.random.default_rng(0)

    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(samples))
        tot = 0.0
        for bi in range(0, len(order), BATCH):
            opt.zero_grad()
            loss = 0.0
            for si in order[bi:bi + BATCH]:
                wt, wl = samples[si]
                enc, last = encode(tok, wt)
                logits = model(input_ids=enc["input_ids"].to(DEV),
                               attention_mask=enc["attention_mask"].to(DEV)).logits[0, :, 0]
                pos = torch.tensor(sorted(last.values()), device=DEV)
                y = torch.tensor([wl[w] for w in sorted(last)], device=DEV, dtype=torch.float)
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits[pos], y, pos_weight=pos_w) / BATCH
            loss.backward()
            opt.step()
            tot += float(loss)
        model.eval()
        f1 = dev_f1(model, tok, dev)
        print(f"epoch {ep + 1}: loss {tot:.1f} dev-F1(0.5) {f1:.2f}", flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    os.makedirs("open_runs/nopnx-np-satft", exist_ok=True)
    torch.save(best_state, "open_runs/nopnx-np-satft/state.pt")
    for split in ["dev", "test"]:
        docs = load_jsonl(f"data/{TASK}_{split}.jsonl")
        np.savez(f"probs/{TASK}_{split}_satft.npz",
                 **{d["doc_id"]: doc_probs(model, tok, d["tokens"]) for d in docs})
        print("cached", split, flush=True)
    print(f"SATFT DONE best-dev-F1(0.5)={best:.2f}")


if __name__ == "__main__":
    main()
