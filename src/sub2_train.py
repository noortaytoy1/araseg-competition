"""SUB2 — Substructure Substitution (Shi, Livescu & Gimpel, Findings-ACL 2021),
adapted to AraSeg NoPnx-NP boundary tagging. CLOSED-LEGAL, self-contained.

Self-contained — does NOT touch the shared train_encoder/predict/eval pipeline.
The data / window / label / [PAR] / weighted-CE handling MIRRORS
src/consistency_train.py exactly (AraBERTv02 token classifier, window 180 /
stride 90, epochs 8, batch 16, lr 5e-5, weighted CE with pos_weight capped ~8,
labels on the LAST subword of each word, MODEL_PAR_TOKEN for the "\n" paragraph
token). Like char_aug.py / teacher_train.py it ALSO emits an additive HF
token-classification checkpoint at the end of train so predict.py /
eval_local.py / diag_preposition.py can load the run directory.

WHY THIS LEVER (Noor chat order of record, 2026-07-07)
------------------------------------------------------
"Label-preserving augmentation means changing training text without changing the
true boundaries." SUB2 (Shi et al. 2021) is the strongest-evidenced
label-preserving augmentation for span-structured NLP: it substitutes a
substructure with a DIFFERENT real substructure that carries the SAME label,
producing new-yet-in-distribution training examples. Here the "substructure" is a
contiguous TOKEN SPAN and its "label" is the span's BOUNDARY-LABEL PATTERN (the
0/1 sequence over the span). Replacing a span with another real corpus span of
the IDENTICAL length and IDENTICAL 0/1 pattern:
  * is LABEL-PRESERVING by construction — the 0/1 sequence over the doc is
    byte-identical before and after (we never touch a boundary position, we only
    swap in tokens that already carry the same per-position labels);
  * is IN-DISTRIBUTION by construction — every substituted token is copied
    verbatim from the 174 train docs (real corpus Arabic), and every pattern used
    already occurs in the data (we only ever substitute a span for another span
    that shares its exact observed pattern; we never synthesize a new pattern);
  * is CLOSED-LEGAL — the ONLY corpus read is the training split handed to the
    trainer (data/{TASK}_train.jsonl, the 174 train docs); nothing external, no
    dev/test, no model-generated tokens.

This is precisely the axis on which naive recombination (make_recomb.py) died at
-1.56: that concatenated random single sentences to a fixed length, going OOD on
BOTH topical coherence and the per-doc length profile. SUB2 does NOT move to a new
distribution — it re-tiles existing local spans with same-shape local spans, so
the label sequence, the token count, and the pattern inventory are all invariant.

TWO FLAG-GATED MECHANISMS (both OFF by default -> bit-for-bit baseline)
----------------------------------------------------------------------
--sub2-rate  (default 0.0 = OFF):
    Per training document, with probability sub2-rate, pick ONE contiguous span
    in the current doc's window and REPLACE its tokens with a DIFFERENT real span
    drawn from the pattern-bank that has the IDENTICAL length AND IDENTICAL 0/1
    boundary-label pattern. Labels are therefore unchanged; every substituted
    token is real corpus Arabic; no boundary position is ever altered; no pattern
    absent from the data is ever created.

--shuffle-seg-rate  (default 0.0 = OFF):
    Per training document, with probability shuffle-seg-rate, pick ONE SEGMENT
    (the tokens strictly BETWEEN two boundaries — i.e. the non-boundary interior
    of a gold sentence) and permute its token ORDER in place. The boundary
    positions are never moved or crossed, so the 0/1 label sequence is unchanged.

Both are applied IN THE COLLATOR at batch materialization time (no augmented data
file on disk -> no race; fresh randomness per epoch). Eval ALWAYS uses rate-0
clean batches. With both rates 0.0 the collator passes words through untouched,
so the tokenised ids AND the labels are byte-for-byte identical to the
un-augmented weighted-CE baseline (self-test (a)).

Usage:
  # baseline (bit-for-bit identical to plain weighted CE):
  python sub2_train.py train --task NoPnx-NP --out runs/sub2_base \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # SUB2 on:
  python sub2_train.py train --task NoPnx-NP --sub2-rate 0.5 --out runs/sub2 \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # shuffle-within-segment on:
  python sub2_train.py train --task NoPnx-NP --shuffle-seg-rate 0.5 --out runs/sub2_shuf \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  python sub2_train.py predict --task NoPnx-NP --split dev --model runs/sub2 \
      --jsonl data/NoPnx-NP_dev.jsonl --out subs/sub2_dev.csv

  python sub2_train.py selftest                       # rate-0 identity + preservation + in-distribution (no GPU)
  python sub2_train.py selftest --gpu-smoke --task NoPnx-NP   # + a few real train steps at rate 0.2
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer, Trainer, TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))

# Span-length window used to index the pattern bank. SUB2's substructures are
# LOCAL spans; we index every contiguous span up to this length (inclusive).
# Longer spans have exponentially rarer exact-pattern collisions and cost memory,
# so 8 is the operative cap (patterns longer than this are simply never chosen as
# a substitution site — the augmentation still fires on the many shorter spans).
MAX_SPAN = 8


# --------------------------------------------------------------------------- #
# SUB2 pattern bank                                                            #
# --------------------------------------------------------------------------- #
class PatternBank:
    """Index of every real train span keyed by its (length, boundary-pattern).

    A "span" is a contiguous run of REAL words (the [PAR] stand-in never appears
    inside an indexed span — see below). Its "pattern" is the tuple of that
    span's boundary labels (0 = no boundary after this token, 1 = boundary).
    Two spans share structure iff they have identical length AND identical
    pattern; SUB2 only ever substitutes one for another within the same key.

    Built ONCE from the training docs handed to the trainer (closed-legal). The
    stored value for each key is the list of the spans' WORD lists; substitution
    copies those words verbatim (in-distribution).

    [PAR] handling: spans that contain the paragraph stand-in are NOT indexed and
    are never chosen as a substitution site, because a [PAR] position carries a
    -100 (ignored) label, so its "pattern" is not a real 0/1 boundary decision and
    swapping across it could relocate a paragraph marker. Keeping [PAR] out of the
    bank keeps every indexed pattern a genuine boundary pattern.
    """

    def __init__(self):
        self.bank = {}   # (length, pattern_tuple) -> list[list[str]]  (word spans)

    @staticmethod
    def _spans_of_doc(words, labels, max_span):
        """Yield (start, length, pattern_tuple, word_list) for every real-word
        contiguous span of length 1..max_span that contains NO [PAR] token."""
        n = len(words)
        for start in range(n):
            if words[start] == MODEL_PAR_TOKEN:
                continue
            span_words = []
            span_labs = []
            for length in range(1, max_span + 1):
                idx = start + length - 1
                if idx >= n or words[idx] == MODEL_PAR_TOKEN:
                    break
                span_words.append(words[idx])
                span_labs.append(labels[idx])
                yield start, length, tuple(span_labs), list(span_words)

    def add_doc(self, words, labels, max_span=MAX_SPAN):
        for _s, length, pat, span_words in self._spans_of_doc(words, labels, max_span):
            self.bank.setdefault((length, pat), []).append(span_words)

    def build(self, docs, max_span=MAX_SPAN):
        """Populate from a list of docs (each {"tokens","labels"}). [PAR] tokens
        are mapped to the stand-in first, exactly as the windowing does."""
        for d in docs:
            words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
            labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
            self.add_doc(words, labs, max_span)
        return self

    def alternatives(self, length, pattern):
        """Real spans sharing this exact (length, pattern), or [] if the key is
        unseen (never happens for a span drawn from an indexed doc, but callers
        must tolerate an empty list — they then leave the span untouched)."""
        return self.bank.get((length, tuple(pattern)), [])


# --------------------------------------------------------------------------- #
# SUB2 + shuffle-within-segment transforms (operate on ONE window at a time)   #
# --------------------------------------------------------------------------- #
def sub2_replace(words, labels, bank, rng, max_span=MAX_SPAN):
    """Return a copy of `words` with ONE eligible span substituted for a real,
    different, same-pattern span from the bank. Labels are returned unchanged.

    Eligibility of a candidate span [start, start+length):
      * length in 1..max_span, contains NO [PAR] token;
      * the bank holds at least one OTHER real span with the same (length,
        pattern) — i.e. a substitution that actually changes the tokens exists.
    If no eligible span exists, `words` is returned unchanged (a no-op is always
    label-preserving). The chosen replacement is guaranteed token-different from
    the original (we resample within the alternatives until the words differ, and
    if the only alternative equals the original we leave the span untouched).
    """
    n = len(words)
    # Collect eligible spans (those with a token-different same-pattern alternative).
    candidates = []
    for start in range(n):
        if words[start] == MODEL_PAR_TOKEN:
            continue
        pat = []
        ok = True
        for length in range(1, max_span + 1):
            idx = start + length - 1
            if idx >= n or words[idx] == MODEL_PAR_TOKEN:
                break
            pat.append(labels[idx])
            alts = bank.alternatives(length, pat)
            # need at least one alternative whose words differ from THIS span
            orig = words[start:idx + 1]
            if any(a != orig for a in alts):
                candidates.append((start, length))
        # loop naturally ends at [PAR] / doc end / max_span
        _ = ok
    if not candidates:
        return list(words)  # nothing to do -> untouched (label-preserving)

    start, length = rng.choice(candidates)
    orig = words[start:start + length]
    pat = labels[start:start + length]
    alts = [a for a in bank.alternatives(length, pat) if a != orig]
    if not alts:
        return list(words)
    repl = rng.choice(alts)
    out = list(words)
    out[start:start + length] = list(repl)
    return out


def shuffle_within_segment(words, labels, rng):
    """Return a copy of `words` with the token ORDER inside ONE segment permuted.

    A "segment" is the maximal run of consecutive positions whose label is 0
    (the interior of a gold sentence, i.e. the tokens strictly between two
    boundaries). Permuting the order within such a run never moves a boundary
    (all its positions are label 0) and never crosses one (we permute strictly
    inside the run), so the 0/1 label sequence is unchanged. [PAR] positions
    (label -100) act as hard segment walls and are never moved.
    Segments of length < 2, or whose permutation would equal the identity, are
    skipped; if no shuffleable segment exists `words` is returned unchanged.
    """
    n = len(words)
    # Build the list of shuffleable segments: maximal contiguous runs of
    # (label == 0 AND word != [PAR]) with length >= 2.
    segments = []
    i = 0
    while i < n:
        if labels[i] == 0 and words[i] != MODEL_PAR_TOKEN:
            j = i
            while j < n and labels[j] == 0 and words[j] != MODEL_PAR_TOKEN:
                j += 1
            if j - i >= 2:
                segments.append((i, j))
            i = j
        else:
            i += 1
    # keep only segments that admit a non-identity permutation (>=2 distinct tokens)
    segments = [(a, b) for (a, b) in segments if len(set(words[a:b])) >= 2]
    if not segments:
        return list(words)

    a, b = rng.choice(segments)
    seg = words[a:b]
    perm = seg[:]
    # draw a permutation that is not the identity (guaranteed to exist: >=2 distinct)
    for _ in range(16):
        rng.shuffle(perm)
        if perm != seg:
            break
    out = list(words)
    out[a:b] = perm
    return out


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py exactly)               #
# --------------------------------------------------------------------------- #
def make_examples(docs, window, stride):
    """Slide a (window, stride) window over each doc; [PAR] positions -> -100."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            out.append({"words": words[s:e], "word_labels": labs[s:e]})
            if e == n:
                break
            s += stride
    return out


def encode_one(words, word_labels, tok, max_len):
    """Tokenise ONE window and place each word's label on its LAST subword."""
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len)
    wid = enc.word_ids()
    lab = [-100] * len(wid)
    for pos in range(len(wid)):
        w = wid[pos]
        if w is None:
            continue
        nxt = wid[pos + 1] if pos + 1 < len(wid) else None
        if nxt != w:
            lab[pos] = word_labels[w]
    enc["labels"] = lab
    return enc


class Sub2Collator:
    """On-the-fly: (optionally) apply SUB2 and/or shuffle-within-segment to each
    window's words, then tokenise + pad.

    The transforms happen HERE, at batch-materialisation time, so every epoch
    (indeed every batch draw) sees freshly augmented spans and NO augmented file
    is ever written. With both rates 0.0 the words pass through untouched, so the
    tokenisation is byte-for-byte identical to the un-augmented baseline.

    Per-window discipline: each window is augmented with probability = its rate,
    matching the task's "per doc, with prob rate" wording at the window grain
    (windows are the doc units the trainer sees). SUB2 uses the window's OWN 0/1
    pattern to look up same-pattern alternatives in the pre-built bank, so the
    substituted span keeps the window's exact labels.

    Determinism: a base seed + a monotonically increasing counter feed a fresh
    per-call RNG, so a given (seed) reproduces the same stream of augmentations
    while still varying across epochs.
    """

    def __init__(self, tok, max_len, bank, sub2_rate=0.0, shuffle_seg_rate=0.0,
                 seed=42, max_span=MAX_SPAN):
        self.tok = tok
        self.max_len = max_len
        self.bank = bank
        self.sub2_rate = float(sub2_rate)
        self.shuffle_seg_rate = float(shuffle_seg_rate)
        self.seed = seed
        self.max_span = max_span
        self._counter = 0

    def _augment(self, words, labels, rng):
        out = words
        if self.sub2_rate > 0.0 and rng.random() < self.sub2_rate:
            out = sub2_replace(out, labels, self.bank, rng, self.max_span)
        if self.shuffle_seg_rate > 0.0 and rng.random() < self.shuffle_seg_rate:
            out = shuffle_within_segment(out, labels, rng)
        return out

    def __call__(self, feats):
        encoded = []
        for f in feats:
            words = f["words"]
            labels = f["word_labels"]
            if self.sub2_rate > 0.0 or self.shuffle_seg_rate > 0.0:
                rng = random.Random(self.seed * 1_000_003 + self._counter)
                self._counter += 1
                words = self._augment(words, labels, rng)
            encoded.append(encode_one(words, labels, self.tok, self.max_len))
        labs = [e.pop("labels") for e in encoded]
        batch = self.tok.pad(encoded, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        return batch


# --------------------------------------------------------------------------- #
# model (boundary head only — mirrors char_aug.BoundaryModel)                  #
# --------------------------------------------------------------------------- #
class BoundaryModel(nn.Module):
    def __init__(self, model_name, pos_weight=1.0, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.bound = nn.Linear(h, 2)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.bound(self.drop(hs))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(hs.device), ignore_index=-100)
        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# metrics / checkpoint / train / predict                                      #
# --------------------------------------------------------------------------- #
def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def _save_hf_checkpoint(model, tok, out_dir, model_name, n_extra_tok=1):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / eval_local.py / diag_preposition.py load with
    AutoModelForTokenClassification.from_pretrained(out_dir) + AutoTokenizer, which
    the custom .pt cannot satisfy. The inference model here is the bare encoder
    `model.enc` followed by the `model.bound` Linear(hidden, 2) boundary head
    (there is no aux head). We assemble a standard
    AutoModelForTokenClassification(model_name, num_labels=2), resize its embeddings
    to match the trainer (vocab + n_extra_tok for [PAR]), load the TRAINED encoder
    body into `.bert` (strict=False tolerates the encoder's unused pooler) and copy
    the TRAINED `bound` weights into `.classifier`. The result reproduces the
    trainer's inference logits exactly. This mirrors char_aug._save_hf_checkpoint /
    teacher_train._save_hf_checkpoint verbatim; it does not touch training, the
    loss, the flags, or the existing .pt/cfg.json outputs — it only adds
    config.json + weights + tokenizer files so the downstream tools can read the
    run directory.
    """
    hf = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    hf.resize_token_embeddings(hf.config.vocab_size + n_extra_tok)
    hf.bert.load_state_dict(model.enc.state_dict(), strict=False)
    with torch.no_grad():
        hf.classifier.weight.copy_(model.bound.weight)
        hf.classifier.bias.copy_(model.bound.bias)
    hf.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)

    # Build the SUB2 pattern bank ONCE from the (closed-legal) training docs.
    bank = PatternBank().build(tr, max_span=a.max_span)
    print(f"pos_weight={pw:.2f}  sub2_rate={a.sub2_rate}  "
          f"shuffle_seg_rate={a.shuffle_seg_rate}  "
          f"bank_keys={len(bank.bank)}  bank_spans={sum(len(v) for v in bank.bank.values())}")

    model = BoundaryModel(a.model_name, pw, n_extra_tok=1)
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride))
    dvds = Dataset.from_list(make_examples(dv, a.window, a.window))  # clean eval

    # Train collator augments on the fly; eval collator NEVER augments (rate 0).
    train_collator = Sub2Collator(tok, a.max_length, bank, a.sub2_rate,
                                  a.shuffle_seg_rate, seed=a.seed, max_span=a.max_span)
    eval_collator = Sub2Collator(tok, a.max_length, bank, 0.0, 0.0,
                                 seed=a.seed, max_span=a.max_span)

    targs = TrainingArguments(
        output_dir=os.path.join(a.out, "ckpts"), num_train_epochs=a.epochs,
        learning_rate=a.lr, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1",
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=20, save_total_limit=1, report_to="none", seed=a.seed,
        remove_unused_columns=False, label_names=["labels"])

    # Two collators, but Trainer takes one: wrap so eval uses the clean collator.
    class DualCollatorTrainer(Trainer):
        def get_eval_dataloader(self, eval_dataset=None):
            saved = self.data_collator
            self.data_collator = eval_collator
            try:
                return super().get_eval_dataloader(eval_dataset)
            finally:
                self.data_collator = saved

    tr_ = DualCollatorTrainer(model=model, args=targs, train_dataset=trds,
                              eval_dataset=dvds, data_collator=train_collator,
                              compute_metrics=metrics)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name}, open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: also emit an HF checkpoint (encoder + boundary head) so predict.py /
    # eval_local.py / diag_preposition.py can load this run dir. Nothing above is
    # changed.
    _save_hf_checkpoint(model, tok, a.out, a.model_name, n_extra_tok=1)
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = BoundaryModel(cfg["model_name"], n_extra_tok=1)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))
    model.to(dev).eval()
    docs = load_task(a.task, a.split, jsonl_path=a.jsonl)
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        n = len(words); probs = {i: [] for i in range(n)}; s = 0
        while s < n:
            e = min(s + a.window, n)
            enc = tok(words[s:e], is_split_into_words=True, truncation=True,
                      max_length=a.max_length, return_tensors="pt")
            wid = enc.word_ids(0)
            logit = model(input_ids=enc["input_ids"].to(dev),
                          attention_mask=enc["attention_mask"].to(dev))["logits"][0]
            pb = torch.softmax(logit, -1)[:, 1].cpu().numpy()
            last = -1
            for pos, w in enumerate(wid):
                if w is None:
                    continue
                nxt = wid[pos + 1] if pos + 1 < len(wid) else None
                if nxt != w:
                    probs[s + w].append(float(pb[pos]))
                last = max(last, w)
            if s + last >= n - 1:
                break
            s = max(s + last + 1 - a.overlap, s + 1)
        lab = [1 if (np.mean(probs[i]) if probs[i] else 0) >= a.threshold else 0 for i in range(n)]
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        preds.append(lab)
    write_submission(docs, preds, a.out)
    print(f"wrote {len(docs)} -> {a.out}")


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #
def _synthetic_docs():
    """Corpus-independent synthetic docs used when data/ is hidden. Rich enough
    to exercise SUB2 (repeated same-pattern spans) and shuffle (multi-token
    segments), including a [PAR] paragraph marker."""
    return [
        {"doc_id": "syn0",
         "tokens": ["ذهب", "الولد", "الى", "المدرسة", "درس", "الطالب", "في", "الصف",
                    PARAGRAPH_TOKEN, "قرأ", "المعلم", "الكتاب", "كتب", "الطالب", "الدرس"],
         "labels": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]},
        {"doc_id": "syn1",
         "tokens": ["جلس", "الرجل", "على", "الكرسي", "نام", "الطفل", "في", "السرير",
                    "لعب", "الاطفال", "في", "الحديقة"],
         "labels": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
        {"doc_id": "syn2",
         "tokens": ["اكل", "القط", "السمك", "شرب", "الكلب", "الماء", "ركض", "الحصان",
                    "بسرعة", "طار", "العصفور", "عاليا"],
         "labels": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]},
    ]


def _load_docs_for_selftest(task, jsonl_path):
    """Prefer the real train split if present; else fall back to synthetic docs
    so the self-test runs corpus-independently with data/ hidden."""
    try:
        if jsonl_path and os.path.exists(jsonl_path):
            return load_task(task, "train", jsonl_path=jsonl_path), "real-train"
    except Exception:
        pass
    return _synthetic_docs(), "synthetic"


def _selftest_rate0_identity(docs, seed, max_span):
    """(a) rate-0 -> token ids + labels bit-for-bit identical to the un-augmented
    baseline."""
    tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    bank = PatternBank().build(docs, max_span=max_span)
    exs = make_examples(docs, 180, 90)
    coll0 = Sub2Collator(tok, 512, bank, 0.0, 0.0, seed=seed, max_span=max_span)
    n = 0
    for f in exs:
        clean = encode_one(f["words"], f["word_labels"], tok, 512)
        batch = coll0([copy.deepcopy(f)])
        got_ids = batch["input_ids"][0].tolist()
        want_ids = clean["input_ids"] + [tok.pad_token_id] * (len(got_ids) - len(clean["input_ids"]))
        assert got_ids == want_ids, "rate-0 token ids differ from the un-augmented baseline"
        got_lab = batch["labels"][0].tolist()
        want_lab = clean["labels"] + [-100] * (len(got_lab) - len(clean["labels"]))
        assert got_lab == want_lab, "rate-0 labels differ from the un-augmented baseline"
        n += 1
    print(f"  [a: rate-0 identity] {n} windows: token ids AND labels bit-for-bit "
          f"identical to the un-augmented baseline.")


def _selftest_label_preservation_and_indistribution(docs, seed, max_span):
    """(b) at rate>0, for EVERY augmented window the LABEL sequence is
    byte-identical to the original (SUB2 preserves the pattern) AND every
    substituted token exists in the source corpus (in-distribution)."""
    tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    bank = PatternBank().build(docs, max_span=max_span)
    # corpus vocabulary = every real word token in the source docs (for the
    # in-distribution check). [PAR] stand-in is legal too (it never moves).
    corpus_vocab = set()
    for d in docs:
        for w in d["tokens"]:
            corpus_vocab.add(MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w)
    exs = make_examples(docs, 180, 90)

    for rate in (0.5, 1.0):
        n_windows = 0
        n_sub2_changed = 0
        n_shuf_changed = 0
        for f in exs:
            words, labels = f["words"], f["word_labels"]
            # SUB2 many draws
            for k in range(20):
                rng = random.Random((seed + k) * 7919 + n_windows)
                if rate > 0.0 and rng.random() < 1.0:  # force-fire to stress it
                    out = sub2_replace(words, labels, bank, rng, max_span)
                    # label preservation: labels are positional and length is fixed
                    assert len(out) == len(words), "SUB2 changed token count"
                    # every substituted (changed) token must be real corpus Arabic
                    for w in out:
                        assert w in corpus_vocab, f"SUB2 emitted an OOV token: {w!r}"
                    if out != words:
                        n_sub2_changed += 1
                    # shuffle-within-segment
                    out2 = shuffle_within_segment(words, labels, rng)
                    assert len(out2) == len(words), "shuffle changed token count"
                    for w in out2:
                        assert w in corpus_vocab, f"shuffle emitted an OOV token: {w!r}"
                    if out2 != words:
                        n_shuf_changed += 1
            n_windows += 1
        print(f"  [b: preservation+in-distribution] rate={rate}: {n_windows} windows x20 draws; "
              f"labels byte-identical for ALL; every substituted token is real corpus Arabic; "
              f"SUB2 changed content {n_sub2_changed} times, shuffle {n_shuf_changed} times.")

    # Explicit pattern-identity assertion for SUB2 replacements (the core claim):
    n_checked = 0
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        for k in range(50):
            rng = random.Random(4242 + k)
            out = sub2_replace(words, labs, bank, rng, max_span)
            # find the changed span (if any) and confirm its pattern is one that
            # exists in the bank for that exact (length, pattern) key.
            if out != words:
                # locate the contiguous changed region
                lo = next(i for i in range(len(words)) if out[i] != words[i])
                hi = max(i for i in range(len(words)) if out[i] != words[i]) + 1
                length = hi - lo
                pat = tuple(labs[lo:hi])
                alts = bank.alternatives(length, pat)
                assert list(out[lo:hi]) in alts, "substituted span not in the same-pattern bank"
                n_checked += 1
    print(f"  [b: SUB2 pattern-identity] verified {n_checked} substitutions each "
          f"drew a real same-(length,pattern) span from the bank.")


def _selftest_nontrivial(docs, seed, max_span):
    """(c) SUB2 substitutions actually CHANGE token content (non-trivial) while
    keeping the pattern. Uses the synthetic docs' engineered repeated patterns
    so a token-different alternative is guaranteed to exist."""
    bank = PatternBank().build(docs, max_span=max_span)
    changed = 0
    total = 0
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        for k in range(30):
            rng = random.Random(999 + k)
            out = sub2_replace(words, labs, bank, rng, max_span)
            total += 1
            if out != words:
                changed += 1
                # confirm the change is a genuine token swap, not a whitespace/no-op
                assert any(a != b for a, b in zip(out, words)), "no-op flagged as change"
    assert changed > 0, "SUB2 never changed content on docs with repeated patterns"
    print(f"  [c: non-trivial] SUB2 changed token content in {changed}/{total} draws "
          f"on docs with same-pattern spans (substitutions are real, not no-ops).")


def _selftest_gpu_smoke(task, jsonl_path, seed, rate=0.2, steps=6, max_span=MAX_SPAN):
    """(d) tiny GPU smoke: train a few real steps at rate>0 with no OOM /
    no shape error. Kept small on purpose (main GPU busy)."""
    set_seed(seed); random.seed(seed)
    model_name = "aubmindlab/bert-base-arabertv02"
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    docs, src = _load_docs_for_selftest(task, jsonl_path)
    # small slice keeps memory tiny while the GPU is shared
    docs = docs[: min(len(docs), 8)]
    pos = sum(sum(d["labels"]) for d in docs); tot = sum(len(d["labels"]) for d in docs)
    pw = min((tot - pos) / max(pos, 1), 8.0)
    bank = PatternBank().build(docs, max_span=max_span)
    model = BoundaryModel(model_name, pw, n_extra_tok=1)
    trds = Dataset.from_list(make_examples(docs, 180, 90))
    coll = Sub2Collator(tok, 512, bank, rate, 0.0, seed=seed, max_span=max_span)
    targs = TrainingArguments(
        output_dir=os.path.join(HERE, os.pardir, "runs", "_sub2_smoke_ckpts"),
        max_steps=steps, per_device_train_batch_size=min(4, max(1, len(trds))),
        learning_rate=5e-5, fp16=torch.cuda.is_available(), logging_steps=1,
        report_to="none", save_strategy="no", remove_unused_columns=False,
        label_names=["labels"], seed=seed)
    tr_ = Trainer(model=model, args=targs, train_dataset=trds, data_collator=coll)
    out = tr_.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [d: gpu-smoke] device={device}  src={src}  steps={steps}  rate={rate}  "
          f"final_train_loss={out.training_loss:.4f}  (no OOM, no shape error).")


def cmd_selftest(a):
    jsonl = a.train_jsonl or os.path.join(HERE, os.pardir, "data", f"{a.task}_train.jsonl")
    docs, src = _load_docs_for_selftest(a.task, jsonl)
    print("=" * 72)
    print(f"SUB2 self-test  (task={a.task}, seed={a.seed}, source={src}, max_span={a.max_span})")
    print("=" * 72)
    _selftest_rate0_identity(docs, a.seed, a.max_span)
    _selftest_label_preservation_and_indistribution(docs, a.seed, a.max_span)
    _selftest_nontrivial(_synthetic_docs(), a.seed, a.max_span)  # engineered repeats guarantee (c)
    if a.gpu_smoke:
        _selftest_gpu_smoke(a.task, jsonl, a.seed, rate=max(a.sub2_rate, 0.2),
                            steps=a.smoke_steps, max_span=a.max_span)
    print("ALL SELF-TESTS PASSED.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--sub2-rate", type=float, default=0.0,
                   help="per-window SUB2 substitution probability (0.0 = OFF = untouched baseline)")
    t.add_argument("--shuffle-seg-rate", type=float, default=0.0,
                   help="per-window shuffle-within-segment probability (0.0 = OFF)")
    t.add_argument("--max-span", type=int, default=MAX_SPAN,
                   help="max span length indexed / substituted by SUB2")
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--task", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)

    s = sub.add_parser("selftest")
    s.add_argument("--task", default="NoPnx-NP")
    s.add_argument("--train-jsonl"); s.add_argument("--dev-jsonl")
    s.add_argument("--sub2-rate", type=float, default=0.2)
    s.add_argument("--shuffle-seg-rate", type=float, default=0.0)
    s.add_argument("--max-span", type=int, default=MAX_SPAN)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--gpu-smoke", action="store_true", help="also run a few real train steps")
    s.add_argument("--smoke-steps", type=int, default=6)
    s.set_defaults(func=cmd_selftest)

    a = ap.parse_args(); a.func(a)
