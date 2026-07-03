"""Dump SaT-ft voter probabilities for any (blind) jsonl. Run in venv27.

  venv27/bin/python satft_blind.py --jsonl blind/NoPnx-NP_test.jsonl \
      --out probs/blind_NoPnx-NP_satft.npz

predict_blind.py then merges this npz via the config's `probs_npz` entry.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import wtpsplit.models  # noqa: F401 -- registers "xlm-token"
from transformers import AutoModelForTokenClassification, XLMRobertaTokenizerFast

from data import load_jsonl
from train_sat_ft import MODEL, doc_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--state", default="open_runs/nopnx-np-satft/state.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        tok = XLMRobertaTokenizerFast.from_pretrained(MODEL)
    except Exception:
        tok = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModelForTokenClassification.from_pretrained(MODEL)
    model.load_state_dict(torch.load(args.state, map_location="cpu"))
    model = model.to("cuda").eval()

    docs = load_jsonl(args.jsonl)
    np.savez(args.out, **{d["doc_id"]: doc_probs(model, tok, d["tokens"]) for d in docs})
    print(f"satft probs: {len(docs)} docs -> {args.out}")


if __name__ == "__main__":
    main()
