"""CUT-TOKEN GROUPING model for AraSeg 2026 (Noor's design, 2026-07-10).

Reframes sentence segmentation as GROUPING-BY-ERASURE instead of tagging:
a new special token [CUT] is inserted between EVERY adjacent word pair,

    w1 [CUT] w2 [CUT] w3 ... [CUT] wW

and the model classifies each [CUT] position's final hidden state:
    1 = KEEP  (a gold sentence boundary follows the LEFT word)
    0 = ERASE (the two words belong to the same sentence).
Labels live ONLY on the [CUT] tokens; every word subtoken is -100. This is a
NEW PROCEDURE relative to the tagger (train_encoder.py): the decision unit is
an explicit boundary SLOT the encoder can attend to, not a word's last subword.

INIT MODES (--init-mode):
  base : aubmindlab/bert-base-arabertv02 (the tagger's encoder)
  tapt : runs/tapt_NoPnx-NP — Noor's stage-1 MLM self-supervised encoder,
         continued-pretrained on our 174 train docs (verified at load; if it
         cannot be loaded the script REPORTS this and falls back to base with
         a loud TAPT_INCOMPATIBLE note).

OPTIONAL STAGE 2a (--cut-mlm-epochs N, default 0 = off): before the supervised
stage, N epochs of MLM on the SAME cut-riddled windows — 15% of WORD tokens
are masked (80/10/10), [CUT] / [PAR] / [CLS] / [SEP] / pad are NEVER masked
and never predicted — so the encoder learns to read text through [CUT]-riddled
input and the fresh [CUT] embedding integrates. Self-supervised on the train
split only: closed-legal (same legality argument as tapt.py).

WINDOW MATH (measured, not assumed): the real subword expansion is measured on
the actual training corpus and W is chosen as the LARGEST window whose p99
cut-sequence length fits --max-length (512). Cut-sequence length for k words =
2 ([CLS]+[SEP]) + sum(subwords) + (k-1) [CUT]s. Measured on the 174
NoPnx-NP train docs (aubmindlab/bert-base-arabertv02): mean subword expansion
1.1858 per word -> W=203 (p99=512, max=596, 0.98% of windows truncate; the
truncated tail [CUT]s are re-covered by the stride-W/2 overlap). The chosen W
and the p99 are printed at train time; `windowmath` prints the full table.

CLASS BALANCE: keeps are ~10.4% of gaps -> same weighted-CE convention as the
tagger: pos_weight = min(neg/pos, 8) (here that saturates to 8.00), computed
on the [CUT]-grid label population (every word except each doc's last carries
a gap). --pos-weight overrides.

STRUCTURAL NOTE (doc-final gap): the last word of a document has no [CUT]
after it, so the cut grid CANNOT express the doc-final boundary. Empirically
gold labels[-1]==1 on 174/174 train and 222/222 dev docs. `predict` therefore
exports P=0.0 at each doc's final word (predict.py's uncovered->0 convention)
and offers --force-doc-end (same flag predict.py has) to set that label 1.
The battery scores BOTH readings (plain row + "<arm>_de" row).

HF-CHECKPOINT SAVE: the battery's `_save_hf_checkpoint` pattern does not fit
directly because the DECISION GRID differs — downstream tools that walk the
word grid via predict.predict_doc would read word last-subword logits, which
is NOT where this model's decisions live. We still save the full model +
tokenizer (AutoModelForTokenClassification-loadable) so the `predict`
subcommand reloads it; diagnostic tools that need word-grid probabilities get
them via the predict path's per-word [CUT] probabilities exported to an npz
(--npz-out; same doc_id->prob-array format as cache_probs.py).

ADDITIVE: new standalone file. No load-bearing file is imported-and-modified;
defaults elsewhere are untouched. Dev is EVALUATION / SELECTION only (repo
convention); training reads the train split only.

CPU smoke:      python src/smoke_cuttoken.py
Battery (GPU, guarded): runs/cuttoken_battery_2026-07-10/run_cuttoken.sh
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task

CUT_TOKEN = "[CUT]"
BASE_MODEL = "aubmindlab/bert-base-arabertv02"
DEFAULT_TAPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "runs", "tapt_NoPnx-NP")
CFG_NAME = "cuttoken_cfg.json"


# --------------------------------------------------------------------------- #
# window math — measure the REAL subword expansion, derive W
# --------------------------------------------------------------------------- #
def subword_counts(docs: List[dict], tokenizer) -> List[np.ndarray]:
    """Per-word subword counts, [PAR]-mapped, memoized per surface form."""
    cache = {}

    def n_sub(w: str) -> int:
        if w not in cache:
            cache[w] = max(len(tokenizer.tokenize(w)), 1)
        return cache[w]

    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        out.append(np.array([n_sub(w) for w in words], dtype=np.int64))
    return out


def cut_seq_lengths(counts: List[np.ndarray], window: int, stride: int) -> np.ndarray:
    """Cut-sequence token length of every window: 2 + subwords + (k-1) [CUT]s."""
    lens = []
    for c in counts:
        n = len(c)
        cs = np.concatenate([[0], np.cumsum(c)])
        start = 0
        while start < n:
            end = min(start + window, n)
            k = end - start
            lens.append(2 + int(cs[end] - cs[start]) + (k - 1))
            if end == n:
                break
            start += stride
    return np.asarray(lens)


def choose_window(counts: List[np.ndarray], max_length: int,
                  w_hi: int = 256, w_lo: int = 32) -> Tuple[int, dict]:
    """Largest W (scanning down from w_hi) whose p99 window length <= max_length
    at stride W//2. Returns (W, stats). Prints the chosen W and p99 (required
    readout)."""
    allc = np.concatenate(counts)
    exp = float(allc.mean())
    for W in range(w_hi, w_lo - 1, -1):
        L = cut_seq_lengths(counts, W, max(W // 2, 1))
        p99 = float(np.percentile(L, 99))
        if p99 <= max_length:
            stats = {"window": W, "stride": max(W // 2, 1),
                     "mean_subwords_per_word": round(exp, 4),
                     "p99_seq_len": p99, "max_seq_len": int(L.max()),
                     "mean_seq_len": round(float(L.mean()), 1),
                     "n_windows": int(len(L)),
                     "frac_windows_over_max_length": round(float((L > max_length).mean()), 4)}
            print(f"[windowmath] measured subword expansion = {exp:.4f} subwords/word "
                  f"({len(allc)} words)")
            print(f"[windowmath] chosen W={W} (stride {stats['stride']}): "
                  f"p99 seq len = {p99:.1f} <= {max_length}, max = {stats['max_seq_len']}, "
                  f"{stats['frac_windows_over_max_length']*100:.2f}% of windows truncate "
                  f"(tail [CUT]s re-covered by the W/2 overlap)")
            return W, stats
    raise SystemExit(f"no window in [{w_lo},{w_hi}] fits max_length={max_length}")


# --------------------------------------------------------------------------- #
# input builder — interleave [CUT] between every adjacent word pair
# --------------------------------------------------------------------------- #
def build_cut_units(words: List[str], word_labels: List[int]) -> Tuple[List[str], List[int]]:
    """Interleave: [w1, [CUT], w2, [CUT], ..., wk]. Labels live ON the [CUT]
    units: the [CUT] after word i carries word i's gold label (1 = KEEP =
    boundary after the left word, 0 = ERASE); word units are -100. A -100 word
    label (e.g. [PAR]) propagates -100 to its [CUT] (excluded from the loss,
    same exclusion the tagger applies at [PAR]).

    Invariant (asserted): #[CUT] units == #words - 1."""
    assert len(words) == len(word_labels)
    units: List[str] = []
    unit_labels: List[int] = []
    for i, (w, l) in enumerate(zip(words, word_labels)):
        units.append(w)
        unit_labels.append(-100)                    # never label a word unit
        if i + 1 < len(words):
            units.append(CUT_TOKEN)
            unit_labels.append(l)                   # gap AFTER the left word
    n_cut = sum(1 for u in units if u == CUT_TOKEN)
    assert n_cut == len(words) - 1, f"[CUT] count {n_cut} != words-1 {len(words)-1}"
    return units, unit_labels


def make_cut_windows(docs: List[dict], window: int, stride: int) -> List[dict]:
    """Overlapping word windows (>=2 words each; a 1-word window has no gap).
    [PAR] mapping + -100-at-[PAR] mirror train_encoder.make_windows."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labels = [-100 if w == MODEL_PAR_TOKEN else int(l)
                  for w, l in zip(words, d["labels"])]
        n = len(words)
        start = 0
        while start < n:
            end = min(start + window, n)
            if end - start >= 2:
                out.append({"words": words[start:end],
                            "word_labels": labels[start:end],
                            "doc_id": d["doc_id"], "start": start})
            if end == n:
                break
            start += stride
    return out


def encode_cut_batch(examples, tokenizer, max_length: int, with_labels: bool = True):
    """Tokenize interleaved units with is_split_into_words=True; align labels
    to the LAST subword of each unit ([CUT] units are single subtokens, so the
    label sits exactly on the [CUT]); everything else -100."""
    unit_seqs, label_seqs = [], []
    for words, wl in zip(examples["words"], examples["word_labels"]):
        u, ul = build_cut_units(words, wl)
        unit_seqs.append(u)
        label_seqs.append(ul)
    enc = tokenizer(unit_seqs, is_split_into_words=True, truncation=True,
                    max_length=max_length)
    if with_labels:
        all_labels = []
        for i, ul in enumerate(label_seqs):
            uids = enc.word_ids(batch_index=i)
            labels = [-100] * len(uids)
            for pos in range(len(uids)):
                uid = uids[pos]
                if uid is None:
                    continue
                nxt = uids[pos + 1] if pos + 1 < len(uids) else None
                if nxt != uid:                     # last subword of this unit
                    labels[pos] = ul[uid]          # -100 for word units
            all_labels.append(labels)
        enc["labels"] = all_labels
    return enc


# --------------------------------------------------------------------------- #
# stage 2a — MLM through cut-riddled input (mask WORD tokens only)
# --------------------------------------------------------------------------- #
class CutMLMCollator:
    """MLM collator for cut-riddled sequences. Candidates for the 15% masking
    pool are ONLY non-special tokens — [CUT], [PAR], [CLS], [SEP], pad (all in
    tokenizer.all_special_ids) are never masked and never predicted. Chosen
    positions follow the standard 80/10/10 rule (mask / random / keep);
    labels = original ids at chosen positions, -100 everywhere else."""

    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_ids = torch.tensor(sorted(set(tokenizer.all_special_ids)),
                                        dtype=torch.long)

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        input_ids = batch["input_ids"].clone()
        labels = input_ids.clone()

        special = torch.isin(input_ids, self.special_ids)
        maskable = (~special) & (batch["attention_mask"] == 1)

        prob = torch.full(labels.shape, self.mlm_probability)
        prob.masked_fill_(~maskable, 0.0)
        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100                     # predict masked WORDS only

        replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        input_ids[replaced] = self.tokenizer.mask_token_id
        rand = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked & ~replaced)
        input_ids[rand] = torch.randint(len(self.tokenizer), labels.shape,
                                        dtype=torch.long)[rand]

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch


def run_cut_mlm(init_ckpt: str, tokenizer, train_windows, args, use_cuda: bool) -> str:
    """Stage 2a: N MLM epochs over the cut-riddled windows; saves the adapted
    encoder to <out-dir>/cutmlm and returns that path (the supervised stage
    then initializes from it)."""
    from datasets import Dataset
    from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments

    mlm_out = os.path.join(args.out_dir, "cutmlm")
    model = AutoModelForMaskedLM.from_pretrained(init_ckpt)
    model.resize_token_embeddings(len(tokenizer))

    ds = Dataset.from_list([{"words": w["words"], "word_labels": w["word_labels"]}
                            for w in train_windows])
    fn = lambda ex: encode_cut_batch(ex, tokenizer, args.max_length, with_labels=False)
    ds = ds.map(fn, batched=True, remove_columns=ds.column_names)

    targs = TrainingArguments(
        output_dir=os.path.join(mlm_out, "ckpts"),
        overwrite_output_dir=False,                # runs/ is append-only
        num_train_epochs=args.cut_mlm_epochs,
        learning_rate=args.cut_mlm_lr,
        per_device_train_batch_size=args.batch_size,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=use_cuda,
        use_cpu=not use_cuda,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=CutMLMCollator(tokenizer, args.cut_mlm_prob))
    out = trainer.train()
    hist = [h for h in trainer.state.log_history if "loss" in h]
    if hist:
        print(f"[cut-mlm] loss first-logged={hist[0]['loss']:.4f} "
              f"final-logged={hist[-1]['loss']:.4f}")
    print(f"[cut-mlm] mean train loss = {out.training_loss:.4f}")
    trainer.save_model(mlm_out)
    tokenizer.save_pretrained(mlm_out)
    print(f"[cut-mlm] saved cut-adapted encoder -> {mlm_out}")
    return mlm_out


# --------------------------------------------------------------------------- #
# supervised stage — weighted CE on [CUT] positions only
# --------------------------------------------------------------------------- #
class WeightedCutTrainer:
    """Deferred import wrapper: constructed in cmd_train (keeps transformers'
    Trainer import out of module import time for light-weight consumers)."""
    def __new__(cls, *a, **k):
        from transformers import Trainer

        class _T(Trainer):
            def __init__(self, *args, pos_weight: float = 1.0, **kwargs):
                super().__init__(*args, **kwargs)
                self._loss = nn.CrossEntropyLoss(
                    weight=torch.tensor([1.0, pos_weight]), ignore_index=-100)

            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = self._loss.to(logits.device)(
                    logits.view(-1, logits.size(-1)), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        return _T(*a, **k)


def cut_f1_metrics(eval_pred):
    """Token-level P/R/F1 on the [CUT] grid (positions with labels != -100) —
    fast model-selection proxy, same shape as train_encoder.token_f1_metrics.
    The true document metric comes from `predict` + eval_local.py."""
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = logits.argmax(-1)
    mask = labels != -100
    p, g = preds[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


def resolve_init(init_mode: str, tapt_dir: str) -> Tuple[str, str]:
    """Resolve the init checkpoint. For 'tapt', VERIFY the stage-1 encoder
    actually loads (config + weights); if incompatible, report loudly and fall
    back to base (per the pre-registered fallback)."""
    if init_mode == "base":
        return BASE_MODEL, "base"
    from transformers import AutoConfig, AutoModelForMaskedLM
    try:
        cfg = AutoConfig.from_pretrained(tapt_dir)
        assert cfg.model_type == "bert", f"unexpected model_type {cfg.model_type!r}"
        m = AutoModelForMaskedLM.from_pretrained(tapt_dir)   # full weight load
        n_emb = m.get_input_embeddings().weight.shape[0]
        assert n_emb == cfg.vocab_size, "embedding/config vocab mismatch"
        del m
        print(f"[init] tapt encoder verified: {tapt_dir} "
              f"(BertForMaskedLM, vocab {cfg.vocab_size})")
        return tapt_dir, "tapt"
    except Exception as e:  # pre-registered fallback, loud
        print(f"[init] TAPT_INCOMPATIBLE: could not load {tapt_dir!r} "
              f"({type(e).__name__}: {e}) -> FALLING BACK TO base "
              f"({BASE_MODEL}). Note this in the report.")
        return BASE_MODEL, "base_fallback_from_tapt"


def cmd_train(args) -> None:
    from datasets import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification,
                              TrainingArguments, set_seed)

    set_seed(args.seed)
    random.seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu

    train_docs = load_task(args.task, "train", jsonl_path=args.train_jsonl)
    dev_docs = load_task(args.task, "dev", jsonl_path=args.dev_jsonl)
    if args.max_docs is not None:
        train_docs = train_docs[: args.max_docs]
    if args.max_dev_docs is not None:
        dev_docs = dev_docs[: args.max_dev_docs]

    # doc-count tripwire (same convention as train_encoder.py --expect-docs)
    print(f"loaded train docs={len(train_docs)}  from={args.train_jsonl or '(default)'}")
    if args.expect_docs is not None and len(train_docs) != args.expect_docs:
        raise SystemExit(f"--expect-docs {args.expect_docs} but loaded "
                         f"{len(train_docs)} train docs -> ABORT")

    init_ckpt, init_note = resolve_init(args.init_mode, args.tapt_dir)

    tokenizer = AutoTokenizer.from_pretrained(init_ckpt)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [MODEL_PAR_TOKEN, CUT_TOKEN]})
    cut_id = tokenizer.convert_tokens_to_ids(CUT_TOKEN)
    print(f"[tok] vocab={len(tokenizer)}  [CUT] id={cut_id}  "
          f"[PAR] id={tokenizer.convert_tokens_to_ids(MODEL_PAR_TOKEN)}")

    # ---- WINDOW MATH: measure real expansion, pick W (unless overridden) ----
    counts = subword_counts(train_docs, tokenizer)
    if args.window is None:
        window, wstats = choose_window(counts, args.max_length)
    else:
        window = args.window
        L = cut_seq_lengths(counts, window, max(args.stride or window // 2, 1))
        wstats = {"window": window, "p99_seq_len": float(np.percentile(L, 99)),
                  "max_seq_len": int(L.max())}
        print(f"[windowmath] OVERRIDE W={window}: p99 seq len = "
              f"{wstats['p99_seq_len']:.1f}, max = {wstats['max_seq_len']}")
    stride = args.stride if args.stride is not None else max(window // 2, 1)

    # ---- class balance on the [CUT]-grid population (gap after every word
    # except each doc's last; [PAR] gaps excluded) — same convention as the
    # tagger: pos_weight = min(neg/pos, 8) ----
    pos = tot = 0
    for d in train_docs:
        toks, labs = d["tokens"], d["labels"]
        for i in range(len(toks) - 1):             # word i has a gap after it
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            tot += 1
            pos += int(labs[i])
    pos_weight = (args.pos_weight if args.pos_weight is not None
                  else min((tot - pos) / max(pos, 1), 8.0))
    print(f"[CUT]-grid keep rate={pos / max(tot, 1):.4f} "
          f"({pos}/{tot})  pos_weight={pos_weight:.2f}")

    train_windows = make_cut_windows(train_docs, window, stride)
    dev_windows = make_cut_windows(dev_docs, window, window)   # no overlap at eval
    print(f"windows: train={len(train_windows)}  dev={len(dev_windows)}")

    # ---- STAGE 2a (optional): MLM through cut-riddled input ----
    sup_init = init_ckpt
    if args.cut_mlm_epochs > 0:
        print(f"[stage 2a] cut-MLM for {args.cut_mlm_epochs} epoch(s) "
              f"(mask {args.cut_mlm_prob:.0%} of WORD tokens only, never [CUT])")
        sup_init = run_cut_mlm(init_ckpt, tokenizer, train_windows, args, use_cuda)

    # ---- supervised stage: binary head on [CUT] final hidden states ----
    model = AutoModelForTokenClassification.from_pretrained(sup_init, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    fn = lambda ex: encode_cut_batch(ex, tokenizer, args.max_length)
    tr_ds = Dataset.from_list([{k: w[k] for k in ("words", "word_labels")}
                               for w in train_windows])
    dv_ds = Dataset.from_list([{k: w[k] for k in ("words", "word_labels")}
                               for w in dev_windows])
    tr_ds = tr_ds.map(fn, batched=True, remove_columns=tr_ds.column_names)
    dv_ds = dv_ds.map(fn, batched=True, remove_columns=dv_ds.column_names)

    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ckpts"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=use_cuda,
        use_cpu=not use_cuda,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = WeightedCutTrainer(
        model=model,
        args=targs,
        train_dataset=tr_ds,
        eval_dataset=dv_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=cut_f1_metrics,
        pos_weight=pos_weight,
    )
    trainer.train()
    print("best [CUT]-grid dev metrics:", trainer.evaluate())

    # full model + tokenizer so `predict` reloads it (see HF-CHECKPOINT SAVE
    # note in the module docstring: the decision grid differs from the tagger,
    # word-grid probs are exported by `predict --npz-out`).
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    cfg = {"design": "cuttoken", "task": args.task, "init_mode": args.init_mode,
           "init_ckpt": init_ckpt, "init_note": init_note,
           "cut_mlm_epochs": args.cut_mlm_epochs,
           "window": window, "stride": stride, "max_length": args.max_length,
           "pos_weight": pos_weight, "seed": args.seed,
           "window_stats": wstats}
    with open(os.path.join(args.out_dir, CFG_NAME), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved model -> {args.out_dir}")
    print(f"next: python src/cuttoken_train.py predict --model {args.out_dir} "
          f"--task {args.task} --split dev --out subs/{args.task}_cut_dev.csv "
          f"--npz-out probs/cut_{args.task}_dev.npz")


# --------------------------------------------------------------------------- #
# predict — same insertion at inference; kept [CUT] = boundary after left word
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_doc_cut(words: List[str], tokenizer, model, device,
                    window: int, overlap: int, max_length: int) -> np.ndarray:
    """P(boundary) per WORD from the [CUT] probabilities, overlapping windows
    averaged (same stitching convention as predict.py: coverage-tracked
    stepping, mean over collected probabilities, uncovered -> 0.0).
    words[i]'s probability = P(the [CUT] after words[i] is KEPT).
    The doc-final word has no [CUT] -> 0.0 (see module docstring)."""
    n = len(words)
    if n < 2:
        return np.zeros(n)
    gaps = defaultdict(list)                        # gap i = after word i
    start = 0
    while start < n - 1:
        chunk = words[start:start + window]
        units, _ = build_cut_units(chunk, [0] * len(chunk))
        enc = tokenizer(units, is_split_into_words=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        uids = enc.word_ids(0)
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
        p_keep = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        last_gap = -1
        for pos, uid in enumerate(uids):
            if uid is None or uid % 2 == 0:         # None/word unit -> skip
                continue
            g = start + (uid - 1) // 2              # [CUT] unit -> its gap
            gaps[g].append(float(p_keep[pos]))
            last_gap = max(last_gap, g)

        if last_gap >= n - 2:                       # all gaps covered
            break
        if last_gap < start:                        # degenerate (fully truncated)
            covered = start                         # step forward minimally
        else:
            covered = last_gap
        # keep `overlap` covered gaps for context; always advance >= 1 word and
        # never past the first uncovered gap's left word (continuity guarantee)
        start = max(min(covered + 1 - overlap, covered + 1), start + 1)

    return np.array([float(np.mean(gaps[i])) if gaps[i] else 0.0
                     for i in range(n)])


def cmd_predict(args) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from baselines import write_submission

    cfg = {}
    cfg_path = os.path.join(args.model, CFG_NAME)
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
    window = args.window or cfg.get("window", 203)
    stride = args.stride or cfg.get("stride", max(window // 2, 1))
    overlap = window - stride
    max_length = args.max_length or cfg.get("max_length", 512)
    print(f"[predict] window={window} stride={stride} overlap={overlap} "
          f"max_length={max_length} (from {'cfg' if cfg else 'defaults'})")

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert CUT_TOKEN in tokenizer.get_vocab(), \
        f"{args.model} tokenizer has no {CUT_TOKEN} — not a cuttoken checkpoint"
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model = model.to(device).eval()

    docs = load_task(args.task, args.split, jsonl_path=args.jsonl)
    if args.max_docs is not None:
        docs = docs[: args.max_docs]

    preds, prob_map = [], {}
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc_cut(words, tokenizer, model, device,
                            window, overlap, max_length)
        prob_map[d["doc_id"]] = p.astype(np.float32)
        lab = (p >= args.threshold).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:               # predict.py convention
                lab[i] = 0
        if args.force_doc_end:                     # predict.py's flag, mirrored
            for i in range(len(lab) - 1, -1, -1):
                if d["tokens"][i] != PARAGRAPH_TOKEN:
                    lab[i] = 1
                    break
        preds.append(lab.tolist())

    write_submission(docs, preds, args.out)
    rate = sum(map(sum, preds)) / max(sum(map(len, preds)), 1)
    print(f"wrote {len(docs)} docs (boundary rate={rate:.3f}, "
          f"threshold={args.threshold}) -> {args.out}")

    if args.npz_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.npz_out)), exist_ok=True)
        np.savez(args.npz_out, **prob_map)
        print(f"word-grid [CUT] probabilities ({len(prob_map)} docs) -> "
              f"{args.npz_out}  (doc-final word carries 0.0 by construction)")

    if args.check_ids:
        import pandas as pd
        ref = pd.read_csv(args.check_ids, dtype={"Prediction": str})
        ref_lens = {str(r["Document ID"]): len(str(r["Prediction"]))
                    for _, r in ref.iterrows()}
        ours = {d["doc_id"]: len(p) for d, p in zip(docs, preds)}
        assert set(ref_lens) == set(ours), "doc ID set differs from example file"
        bad = [k for k in ours if ours[k] != ref_lens[k]]
        assert not bad, f"token-count mismatch for {len(bad)} docs, e.g. {bad[:3]}"
        print(f"check-ids OK: {len(ours)} docs match {args.check_ids}")


def cmd_windowmath(args) -> None:
    from transformers import AutoTokenizer

    docs = load_task(args.task, "train", jsonl_path=args.train_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [MODEL_PAR_TOKEN, CUT_TOKEN]})
    counts = subword_counts(docs, tokenizer)
    W, stats = choose_window(counts, args.max_length)
    print(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="stage 2a (optional) + supervised cut head")
    tr.add_argument("--task", default="NoPnx-NP", choices=sorted(TASKS))
    tr.add_argument("--train-jsonl", default=None)
    tr.add_argument("--dev-jsonl", default=None)
    tr.add_argument("--expect-docs", type=int, default=None,
                    help="assert the train split has exactly this many docs")
    tr.add_argument("--init-mode", choices=["base", "tapt"], default="base",
                    help="base = arabertv02; tapt = Noor's stage-1 MLM encoder")
    tr.add_argument("--tapt-dir", default=DEFAULT_TAPT_DIR,
                    help="stage-1 TAPT encoder dir (default runs/tapt_NoPnx-NP)")
    tr.add_argument("--cut-mlm-epochs", type=int, default=0,
                    help="stage 2a: MLM epochs WITH [CUT]s present (0 = off). "
                         "Masks WORD tokens only, never [CUT]; closed-legal")
    tr.add_argument("--cut-mlm-lr", type=float, default=1e-4,
                    help="stage 2a LR (TAPT convention 1e-4)")
    tr.add_argument("--cut-mlm-prob", type=float, default=0.15)
    tr.add_argument("--window", type=int, default=None,
                    help="words per window (default: AUTO from measured "
                         "subword expansion so p99 cut-seq len fits max-length)")
    tr.add_argument("--stride", type=int, default=None, help="default W//2")
    tr.add_argument("--max-length", type=int, default=512)
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--pos-weight", type=float, default=None,
                    help="KEEP-class CE weight (default min(neg/pos, 8) on the "
                         "[CUT] grid — same convention as the tagger)")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--max-docs", type=int, default=None, help="CPU smoke cap")
    tr.add_argument("--max-dev-docs", type=int, default=None, help="CPU smoke cap")
    tr.add_argument("--cpu", action="store_true", help="force CPU (smokes)")
    tr.set_defaults(fn=cmd_train)

    pr = sub.add_parser("predict", help="insert [CUT]s, keep/erase, emit CSV+npz")
    pr.add_argument("--model", required=True)
    pr.add_argument("--task", default="NoPnx-NP", choices=sorted(TASKS))
    pr.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    pr.add_argument("--jsonl", default=None)
    pr.add_argument("--out", required=True, help="submission CSV")
    pr.add_argument("--npz-out", default=None,
                    help="word-grid P(boundary) npz (cache_probs.py format) for "
                         "diag tools; the doc-final word carries 0.0")
    pr.add_argument("--threshold", type=float, default=0.5)
    pr.add_argument("--window", type=int, default=None, help="default: model cfg")
    pr.add_argument("--stride", type=int, default=None, help="default: model cfg")
    pr.add_argument("--max-length", type=int, default=None)
    pr.add_argument("--force-doc-end", action="store_true",
                    help="force a boundary on each doc's last non-\\n token "
                         "(predict.py's flag; gold ends with 1 on 174/174 "
                         "train docs)")
    pr.add_argument("--check-ids", default=None)
    pr.add_argument("--max-docs", type=int, default=None, help="CPU smoke cap")
    pr.add_argument("--cpu", action="store_true")
    pr.set_defaults(fn=cmd_predict)

    wm = sub.add_parser("windowmath", help="measure expansion, print chosen W")
    wm.add_argument("--task", default="NoPnx-NP", choices=sorted(TASKS))
    wm.add_argument("--train-jsonl", default=None)
    wm.add_argument("--tokenizer", default=BASE_MODEL)
    wm.add_argument("--max-length", type=int, default=512)
    wm.set_defaults(fn=cmd_windowmath)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
