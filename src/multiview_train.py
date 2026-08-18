# -*- coding: utf-8 -*-
"""MULTI-VIEW JOINT trainer for the AraSeg closed track (SHAS/SaT lineage).

ADDITIVE + self-contained: clones src/consistency_train.py's verified pipeline
(AraBERTv02 token classifier, window 180 / stride 90, epochs 8, lr 5e-5,
batch 16, weighted CE with pos_weight capped ~8, labels on the LAST subword,
[PAR] -> -100, load_best on window-level f1, HF-checkpoint save, predict
subcommand compatible with eval_local.py). It does NOT touch the shared
train_encoder / predict / eval / stitching code; it IMPORTS the pieces of
record instead of re-implementing them:

  * window/encode/metrics   : consistency_train.make_examples / .encode / .metrics
  * label fold rule         : xview_align.fold_labels (the ONE verified
                              OR-fold of record, 174/174 exact on train)
  * no-indicator inference  : predict.predict_doc (the shared stitcher)

NOVELTY vs the union runs (S2 arm (i) / union-*): those trained on 2 views
concatenated with NO view indicator. The two UNTRIED sub-variants built here:
  (a) MORE THAN 2 VIEWS in one model (up to all four organizer train views of
      the SAME 174 docs: PA, NP, NoPnx-PA, NoPnx-NP -- all closed-track legal);
  (b) a VIEW-INDICATOR token: one per-view special token ([VIEW_PA],
      [VIEW_NP], [VIEW_NoPnx-PA], [VIEW_NoPnx-NP]; all 4 added to the
      tokenizer whenever the indicator is on, so the vocab is arm-stable)
      prepended to every window so the model knows which convention it is
      reading; its word label is -100; at predict time the TARGET view's
      token is prepended ([VIEW_NoPnx-NP] for the NoPnx-NP submission view).
Plus the SaT-style smooth bridge:
  (c) --punct-dropout RATE: for windows drawn from the PUNCTUATED views
      (PA, NP) each punctuation token (every char in a Unicode P* category;
      [PAR] is never punctuation) is dropped with prob RATE ON-THE-FLY at
      batch time (fresh draw every epoch), and the window's labels are
      re-folded onto the preceding kept token by the SAME verified fold rule
      (xview_align.fold_labels -- imported, not duplicated). This creates a
      continuum between the punctuated and the stripped views. Boundaries in
      a dropped run BEFORE the first kept token, or folding onto a kept
      [PAR] (-100), are lost for that draw -- same edge semantics as the
      verified gold fold, counted in the smoke.

DEFAULTS reproduce the plain baseline: --views NoPnx-NP (single view) with
--view-indicator auto -> none and --punct-dropout 0.0 is BIT-FOR-BIT the
consistency_train.py all-off baseline (same tokenizer with NO extra view
tokens, same RNG order at model construction, same windows, same encoding,
same TrainingArguments field-for-field, same weighted CE). Verified by
src/smoke_multiview.py section (a) with torch.equal.

EVAL/SELECTION is ALWAYS on the TARGET view's dev (--task / --dev-jsonl,
e.g. data/NoPnx-NP_dev.jsonl). The other views are training signal only;
their dev/test files are never touched.

Usage:
  # single-view baseline (bit-for-bit the consistency_train all-off baseline):
  python multiview_train.py train --task NoPnx-NP --out runs/mv_base \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # all four views + indicator tokens (auto) + punct-dropout bridge:
  python multiview_train.py train --task NoPnx-NP --out runs/mv4_pd \
      --views "NoPnx-NP,NoPnx-PA,NP,PA" \
      --train-jsonl "data/NoPnx-NP_train.jsonl,data/NoPnx-PA_train.jsonl,data/NP_train.jsonl,data/PA_train.jsonl" \
      --dev-jsonl data/NoPnx-NP_dev.jsonl --punct-dropout 0.5

  python multiview_train.py predict --task NoPnx-NP --split dev --model runs/mv4_pd \
      --jsonl data/NoPnx-NP_dev.jsonl --out subs/mv4_pd_dev.csv
"""
import argparse
import json
import os
import random
import unicodedata
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission
from predict import predict_doc                     # shared stitcher of record
from xview_align import fold_labels                 # THE verified fold rule
import consistency_train as C                       # pipeline of record

HERE = os.path.dirname(os.path.abspath(__file__))

ALL_VIEWS = ["PA", "NP", "NoPnx-PA", "NoPnx-NP"]    # organizer train views, 174 docs each
VIEW_TOKENS = {v: f"[VIEW_{v}]" for v in ALL_VIEWS}
PUNCTUATED_VIEWS = {"PA", "NP"}                     # punct-dropout applies here only


# --------------------------------------------------------------------------- #
# tokenizer / views                                                            #
# --------------------------------------------------------------------------- #
def resolve_indicator(indicator, views):
    """'auto' -> 'token' when >1 view else 'none' (the pre-registered default)."""
    if indicator == "auto":
        return "token" if len(views) > 1 else "none"
    return indicator


def build_tokenizer(model_name, indicator):
    """[PAR] always (pipeline of record); the 4 view tokens ONLY when the
    indicator is on. Returns (tok, n_extra_tok). With indicator='none' this is
    exactly the consistency_train tokenizer (n_extra_tok=1, no view tokens)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    extra = [MODEL_PAR_TOKEN]
    if indicator == "token":
        extra = extra + [VIEW_TOKENS[v] for v in ALL_VIEWS]
    tok.add_special_tokens({"additional_special_tokens": extra})
    return tok, len(extra)


def parse_views(views_arg):
    views = [v.strip() for v in views_arg.split(",") if v.strip()]
    for v in views:
        if v not in ALL_VIEWS:
            raise ValueError(f"unknown view {v!r}; choose from {ALL_VIEWS}")
    if len(set(views)) != len(views):
        raise ValueError(f"duplicate views in {views_arg!r}")
    return views


def view_train_paths(views, train_jsonl):
    """--train-jsonl is a comma list matching --views order; empty/None ->
    data/{view}_train.jsonl per view."""
    if not train_jsonl:
        return [os.path.join("data", f"{v}_train.jsonl") for v in views]
    paths = [p.strip() for p in train_jsonl.split(",") if p.strip()]
    if len(paths) != len(views):
        raise ValueError(f"--train-jsonl has {len(paths)} paths for {len(views)} views")
    return paths


# --------------------------------------------------------------------------- #
# punct-dropout (the SaT-style smooth bridge)                                  #
# --------------------------------------------------------------------------- #
def is_punct_token(w):
    """Generic punctuation word: non-empty and EVERY char in a Unicode P*
    category (covers . ! ? ؟ ، ؛ « » ( ) - … dot-leaders ...). [PAR] / '\\n'
    are paragraph structure, never punctuation."""
    if not w or w == MODEL_PAR_TOKEN or w == PARAGRAPH_TOKEN:
        return False
    return all(unicodedata.category(c).startswith("P") for c in w)


def punct_dropout(words, word_labels, rate, rng):
    """Drop each punctuation token with prob `rate`; re-fold boundary labels
    onto the preceding kept token via xview_align.fold_labels (the ONE
    verified implementation -- imported, not duplicated).

    `words`/`word_labels` are in post-make_examples form ([PAR] words carry
    label -100; only [PAR] carries -100). Kept [PAR] positions stay -100
    (a boundary folding onto one is lost for this draw); boundaries in a
    dropped run before the first kept token are likewise lost -- identical
    edge semantics to the verified gold fold. Returns
    (kept_words, folded_labels, n_lost_boundaries)."""
    n = len(words)
    keep = [i for i in range(n)
            if not (is_punct_token(words[i]) and rng.random() < rate)]
    if not keep or len(keep) == n:            # all-punct window / nothing dropped
        return list(words), list(word_labels), 0
    bin_labs = [0 if l == -100 else int(l) for l in word_labels]
    folded = fold_labels(bin_labs, keep, n)
    out = [-100 if word_labels[i] == -100 else folded[j]
           for j, i in enumerate(keep)]
    # LOST boundaries = 1s that reach no kept label slot: the dropped prefix
    # before the first kept token, plus 1s in a group headed by a kept [PAR]
    # (-100). Multiple 1s merging into one kept slot are FOLDS, not losses
    # (identical accounting to the verified gold fold).
    n_lost = sum(bin_labs[:keep[0]])
    for j, i in enumerate(keep):
        if word_labels[i] == -100:
            end = keep[j + 1] if j + 1 < len(keep) else n
            n_lost += sum(bin_labs[i + 1:end])
    return [words[i] for i in keep], out, n_lost


# --------------------------------------------------------------------------- #
# windows (the consistency_train loop, plus view metadata / indicator)         #
# --------------------------------------------------------------------------- #
def prepend_indicator(ex, view):
    """The view-indicator token as the window's FIRST word; its label is -100
    (it is an added special token -> single id, last-subword rule applies)."""
    return {"words": [VIEW_TOKENS[view]] + ex["words"],
            "word_labels": [-100] + ex["word_labels"]}


def make_view_examples(docs, window, stride, view, indicator, raw):
    """Windows for ONE view via consistency_train.make_examples (verbatim
    loop of record). raw=True keeps un-encoded words (+view/pd metadata) for
    the on-the-fly punct-dropout collator; raw=False (precompute path)
    prepends the indicator here so the windows can be dataset.map-encoded
    exactly like the baseline."""
    exs = C.make_examples(docs, window, stride)
    out = []
    for ex in exs:
        if raw:
            out.append({"words": ex["words"], "word_labels": ex["word_labels"],
                        "view": view, "pd": 1 if view in PUNCTUATED_VIEWS else 0})
        else:
            out.append(prepend_indicator(ex, view) if indicator == "token" else ex)
    return out


# --------------------------------------------------------------------------- #
# collators                                                                    #
# --------------------------------------------------------------------------- #
class PadCollator:
    """Pad pre-encoded features; labels padded with -100. Identical padding
    mechanics to consistency_train.Collator (which only adds the two prep
    masks this trainer does not use -- they never enter the CE loss)."""
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        return batch


class OnTheFlyCollator:
    """Batch-time pipeline for --punct-dropout > 0: per RAW window
    (1) stochastic punct-dropout + verified re-fold (ONLY where pd==1, i.e.
        train windows of the punctuated views; dev windows carry pd=0),
    (2) view-indicator prepend (when on),
    (3) consistency_train.encode (labels on the LAST subword),
    (4) the same padding as PadCollator.
    Fresh RNG draws each call -> per-epoch stochastic subsets (SaT recipe)."""
    def __init__(self, tok, max_len, rate, indicator, seed):
        self.tok = tok
        self.max_len = max_len
        self.rate = float(rate)
        self.indicator = indicator
        self.rng = random.Random(seed)

    def __call__(self, feats):
        words_l, labs_l = [], []
        for f in feats:
            words, labs = list(f["words"]), list(f["word_labels"])
            if self.rate > 0 and int(f.get("pd", 0)) == 1:
                words, labs, _ = punct_dropout(words, labs, self.rate, self.rng)
            if self.indicator == "token":
                words = [VIEW_TOKENS[f["view"]]] + words
                labs = [-100] + labs
            words_l.append(words)
            labs_l.append(labs)
        enc = C.encode({"words": words_l, "word_labels": labs_l},
                       self.tok, self.max_len)
        labs = enc.pop("labels")
        feats2 = [{k: enc[k][i] for k in enc.keys()}
                  for i in range(len(words_l))]
        batch = self.tok.pad(feats2, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        return batch


# --------------------------------------------------------------------------- #
# model: plain token classification + weighted CE (the baseline loss, exactly) #
# --------------------------------------------------------------------------- #
class MultiViewModel(nn.Module):
    """Construction order mirrors ConsistencyModel(cf=0, rdrop=0) exactly
    (from_pretrained num_labels=2, then resize by n_extra_tok, then the class
    -weight buffer) so that under the same set_seed the initial weights are
    bit-for-bit identical when n_extra_tok matches (=1 in single-view mode)."""
    def __init__(self, model_name, pos_weight=1.0, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        logits = self.enc(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(logits.device), ignore_index=-100)
        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #
def cmd_train(a):
    views = parse_views(a.views)
    indicator = resolve_indicator(a.view_indicator, views)
    paths = view_train_paths(views, a.train_jsonl)
    if not (0.0 <= a.punct_dropout <= 1.0):
        raise ValueError("--punct-dropout must be in [0, 1]")
    if a.punct_dropout > 0 and not (set(views) & PUNCTUATED_VIEWS):
        print("WARNING: --punct-dropout > 0 but no punctuated view (PA/NP) "
              "in --views; nothing will be dropped.")
    if a.task not in views:
        print(f"WARNING: eval view --task {a.task} is not among --views {views}; "
              "the model never trains on the target convention.")

    set_seed(a.seed); random.seed(a.seed)
    tok, n_extra = build_tokenizer(a.model_name, indicator)

    docs_by_view = {v: load_task(v, "train", jsonl_path=p)
                    for v, p in zip(views, paths)}
    if a.expect_docs_per_view:
        for v in views:
            n = len(docs_by_view[v])
            assert n == a.expect_docs_per_view, \
                f"DOC-COUNT GUARD: view {v} has {n} docs != {a.expect_docs_per_view}"
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)

    all_tr = [d for v in views for d in docs_by_view[v]]
    pos = sum(sum(d["labels"]) for d in all_tr)
    tot = sum(len(d["labels"]) for d in all_tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  views={views}  indicator={indicator}  "
          f"punct_dropout={a.punct_dropout}  n_extra_tok={n_extra}")

    model = MultiViewModel(a.model_name, pw, n_extra_tok=n_extra)

    on_the_fly = a.punct_dropout > 0
    tr_exs, per_view = [], {}
    for v in views:
        exs = make_view_examples(docs_by_view[v], a.window, a.stride, v,
                                 indicator, raw=on_the_fly)
        per_view[v] = len(exs)
        tr_exs.extend(exs)
    print("train windows per view: " +
          "  ".join(f"{v}={per_view[v]}" for v in views) +
          f"  total={len(tr_exs)}")

    dv_exs = make_view_examples(dv, a.window, a.window, a.task,
                                indicator, raw=on_the_fly)  # clean eval, stride=window
    if on_the_fly:
        for ex in dv_exs:
            ex["pd"] = 0                       # NEVER drop punctuation at eval
    trds = Dataset.from_list(tr_exs)
    dvds = Dataset.from_list(dv_exs)
    if on_the_fly:
        collator = OnTheFlyCollator(tok, a.max_length, a.punct_dropout,
                                    indicator, a.seed)
    else:
        fn = lambda ex: C.encode(ex, tok, a.max_length)
        trds = trds.map(fn, batched=True, remove_columns=trds.column_names)
        dvds = dvds.map(fn, batched=True, remove_columns=dvds.column_names)
        collator = PadCollator(tok)

    targs = TrainingArguments(
        output_dir=os.path.join(a.out, "ckpts"), num_train_epochs=a.epochs,
        learning_rate=a.lr, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1",
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=20, save_total_limit=1, report_to="none", seed=a.seed,
        remove_unused_columns=False, label_names=["labels"])
    tr_ = Trainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                  data_collator=collator, compute_metrics=C.metrics)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name, "views": views,
               "view_indicator": indicator, "punct_dropout": a.punct_dropout,
               "eval_view": a.task, "seed": a.seed,
               "view_token": VIEW_TOKENS[a.task] if indicator == "token" else None},
              open(os.path.join(a.out, "cfg.json"), "w"))
    # HF checkpoint so predict.py-style loaders can read the run dir; model.enc
    # IS the inference model (no aux parameters).
    model.enc.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved -> {a.out}")


# --------------------------------------------------------------------------- #
# predict                                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_probs(words, tok, model, device, window, overlap, max_length,
                  view_word=None):
    """P(boundary) per word. With view_word=None this IS predict.predict_doc
    (delegated -- one code path, bit-identical to the shared stitcher). With a
    view token it prepends the token to every window (word 0, skipped in the
    read-out; real word w sits at word-id w+1) and keeps the same overlap-mean
    stitching."""
    if view_word is None:
        return predict_doc(words, tok, model, device, window, overlap, max_length)
    n = len(words)
    probs = defaultdict(list)
    start = 0
    while start < n:
        chunk = [view_word] + words[start:start + window]
        enc = tok(chunk, is_split_into_words=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        word_ids = enc.word_ids(0)
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
        p_boundary = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        last_seen = -1
        for pos, wid in enumerate(word_ids):
            if wid is None or wid == 0:        # specials + the view indicator
                continue
            nxt = word_ids[pos + 1] if pos + 1 < len(word_ids) else None
            if nxt != wid:                     # last subword of this word
                probs[start + wid - 1].append(float(p_boundary[pos]))
            last_seen = max(last_seen, wid - 1)
        covered_through = start + last_seen
        if covered_through >= n - 1:
            break
        start = max(covered_through + 1 - overlap, start + 1)
    return np.array([np.mean(probs[i]) if probs[i] else 0.0 for i in range(n)])


def load_view_word(model_dir, override_view=None):
    """The predict-time view token from the run dir's cfg.json (None when the
    model was trained without an indicator)."""
    cfg = json.load(open(os.path.join(model_dir, "cfg.json")))
    if cfg.get("view_indicator") != "token":
        return None
    view = override_view or cfg.get("eval_view")
    return VIEW_TOKENS[view]


@torch.no_grad()
def cmd_predict(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(device).eval()
    view_word = load_view_word(a.model, a.view)
    print(f"view indicator at predict: {view_word!r}")
    docs = load_task(a.task, a.split, jsonl_path=a.jsonl)
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_probs(words, tok, model, device, a.window, a.overlap,
                          a.max_length, view_word=view_word)
        lab = (p >= a.threshold).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        preds.append(lab.tolist())
    write_submission(docs, preds, a.out)
    print(f"wrote {len(docs)} -> {a.out}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True, choices=sorted(TASKS),
                   help="TARGET view: eval/selection runs on ITS dev only")
    t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--views", default="NoPnx-NP",
                   help="comma list of train views (default single NoPnx-NP = "
                        "bit-for-bit plain baseline)")
    t.add_argument("--train-jsonl", default=None,
                   help="comma list of train JSONLs matching --views order "
                        "(default data/{view}_train.jsonl each)")
    t.add_argument("--dev-jsonl", default=None, help="TARGET view dev JSONL")
    t.add_argument("--view-indicator", default="auto", choices=["auto", "none", "token"],
                   help="auto = token when >1 view, none for a single view")
    t.add_argument("--punct-dropout", type=float, default=0.0,
                   help="batch-time punct-token dropout rate on PA/NP windows "
                        "(0 = off; labels re-folded by the verified fold rule)")
    t.add_argument("--expect-docs-per-view", type=int, default=None,
                   help="assert every view train file has exactly this many docs")
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--task", required=True, choices=sorted(TASKS)); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--view", default=None,
                   help="override the predict-time view token (default: the "
                        "run's eval_view from cfg.json)")
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(); a.func(a)
