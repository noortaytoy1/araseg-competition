"""MRT — MINIMUM RISK TRAINING over per-doc macro-F1 (Shen et al., ACL 2016), adapted.

CLOSED-LEGAL, self-contained — does NOT touch the shared train_encoder / predict /
eval / stitching logic; it only IMPORTS and CALLS them. The data / window / label /
[PAR] / weighted-CE handling MIRRORS src/consistency_train.py exactly
(AutoModelForTokenClassification aubmindlab/bert-base-arabertv02 num_labels=2,
window 180 / stride 90, epochs 8, batch 16, lr 5e-5, weighted CE with pos_weight
capped ~8, labels on the LAST subword of each word, MODEL_PAR_TOKEN for the "\n"
paragraph token). Like consistency_train.py / teacher_train.py / char_aug.py /
sub2_train.py it ALSO emits an additive HuggingFace token-classification checkpoint
at the end of train, so predict.py / eval_local.py / diag_preposition.py can load
the run directory.

WHAT MRT IS (adapted from Shen, Cheng, He, He, Wu, Sun, Liu — "Minimum Risk
Training for Neural Machine Translation", ACL 2016)
------------------------------------------------------------------------------
Standard MRT minimizes the *expected task loss* over a subset S of candidate
outputs, sampled from the model, using a smoothed (temperature-annealed)
distribution:

    Q(y | x) = P(y | x)^alpha / sum_{y' in S} P(y' | x)^alpha
    Risk     = sum_{y in S} Q(y | x) * Delta(y)

where Delta(y) is the sequence-level task cost (1 - reward). We adapt it to AraSeg
boundary tagging with these choices (see RECIPE below):

  * UNIT OF RISK = the WHOLE DOCUMENT. The reward is the *per-document* macro-F1
    of the boundary sequence, exactly as eval_local.py scores a submission
    (sklearn f1_score(..., average="binary"), positive class = 1). So
    Delta(y) = 1 - per_doc_F1(y, gold).

  * CANDIDATE SET S per document = k=6 full-document boundary labelings sampled by
    temperature-scaled independent Bernoulli draws on the per-token boundary
    probs, ALWAYS including the greedy/argmax decode (this guarantees the highest-
    probability, usually highest-reward candidate is present and prevents the
    subset distribution from degenerating).

  * PER-TOKEN PROBS and the doc-level boundary sequence are produced by the SAME
    frozen window stitching AraSeg uses at inference (probs averaged across
    overlapping windows, mirroring cmd_predict in consistency_train.py). MRT calls
    that stitcher; it does not re-implement or edit the eval / stitching rules.

  * GRADIENT is score-function REINFORCE **with a baseline** (the subset-expected
    risk Qbar = sum_y Q(y) Delta(y) is the baseline). We build the surrogate loss

        L_mrt = sum_{y in S} Q(y).detach() * (Delta(y) - Qbar) * ( - logP(y) )

    so that autograd returns the correct MRT/REINFORCE gradient. Subtracting the
    baseline Qbar is variance reduction; it does not change the expected gradient
    (sum_y Q(y).detach() * Qbar * grad logP(y) has zero expectation under Q, since
    Qbar is constant w.r.t. y). logP(y) is the model's log-probability of the
    sampled *document* labeling, summed over the per-token boundary decisions of
    the last-subword positions (the positions the tagger actually predicts).

RECIPE (order of record)
------------------------
MRT is a FINE-TUNE stage, warm-started from a weighted-CE checkpoint via
--init-model (NOT trained from scratch). Per document at each step:
  (1) get per-token boundary probs from the current model over the doc's windows;
  (2) sample k=6 full-doc boundary labelings by temperature-scaled Bernoulli,
      ALWAYS including the greedy/argmax decode in the sample set S;
  (3) reconstruct the full-doc boundary sequence from windows via the existing
      frozen stitching (probs averaging) — call only, never edit;
  (4) Delta(y) = 1 - per_doc_macro_F1(y, gold) using the eval_local scorer logic;
  (5) Q(y) = P(y)^alpha / sum_S P(y')^alpha, alpha default 5e-3 (--mrt-alpha);
      loss = sum_y Q(y) * Delta(y), differentiated via the REINFORCE-with-baseline
      surrogate above.

FLAG GATE (the important safety property)
-----------------------------------------
--mrt off  (equivalently --mrt-weight 0.0), the DEFAULT, makes the trainer
reproduce the plain weighted-CE baseline BIT-FOR-BIT: the exact same window CE
loop as consistency_train.py's baseline, with NO document sampling, NO reward, NO
REINFORCE term. The GPU smoke (smoke_mrt.py) verifies loss/logits torch.equal
against an independent weighted-CE computed straight from the encoder logits.
When --mrt on, MRT runs as a fine-tune: base weighted CE on the windows PLUS the
document-level MRT term (weighted by --mrt-weight).

Usage:
  # 0) obtain a weighted-CE warm start (e.g. from consistency_train.py baseline):
  #    python consistency_train.py train --task NoPnx-NP --out runs/ce_base ...
  #
  # 1) MRT fine-tune warm-started from that checkpoint:
  python mrt_train.py train --task NoPnx-NP --mrt on --init-model runs/ce_base \
      --out runs/mrt --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl
  #
  # 2) baseline (bit-for-bit weighted CE; no warm start needed):
  python mrt_train.py train --task NoPnx-NP --mrt off \
      --out runs/mrt_base --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl
  #
  # 3) predict a scorable submission CSV (eval_local.py reads it directly):
  python mrt_train.py predict --task NoPnx-NP --split dev --model runs/mrt \
      --jsonl data/NoPnx-NP_dev.jsonl --out subs/mrt_dev.csv
  python eval_local.py --gold-jsonl data/NoPnx-NP_dev.jsonl --predictions subs/mrt_dev.csv
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py exactly)              #
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


def encode(examples, tok, max_len):
    """Sub-word encode; put each word's label on its LAST subword only."""
    enc = tok(examples["words"], is_split_into_words=True, truncation=True, max_length=max_len)
    all_lab = []
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        lab = [-100] * len(wid)
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:  # last subword of word w
                lab[pos] = wl[w]
        all_lab.append(lab)
    enc["labels"] = all_lab
    return enc


class Collator:
    """Pad the window CE batch. Identical semantics to consistency_train.py's
    collator, minus the counterfactual masks MRT does not use."""
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        return batch


# --------------------------------------------------------------------------- #
# model wrapper: plain token-classification weighted CE                        #
# --------------------------------------------------------------------------- #
class MRTModel(nn.Module):
    """AraBERTv02 token classifier (num_labels=2) with plain weighted CE.

    The MRT document-level term lives in the TRAINER (it needs the frozen
    doc-level stitcher + gold docs), NOT in this forward, because MRT's unit of
    risk is the whole document, not a padded window batch. This forward is the
    baseline window CE that every default reproduces bit-for-bit; `self.enc` is
    exactly the inference model (MRT adds no parameters)."""
    def __init__(self, model_name, pos_weight=1.0, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def _logits(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).logits

    def _ce(self, logits, labels):
        return nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        logits = self._logits(input_ids, attention_mask)
        loss = None
        if labels is not None:
            loss = self._ce(logits, labels)
        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# FROZEN doc-level stitching — a faithful call of the SAME rule as             #
# consistency_train.py:cmd_predict. It produces, per word position, the        #
# average boundary prob across overlapping windows. MRT samples / argmaxes on   #
# TOP of these probs; it never edits the stitching or the eval rules.          #
# --------------------------------------------------------------------------- #
def _doc_word_probs(model, tok, words, window, overlap, max_length, dev,
                    with_grad=False):
    """Return a tensor `probs[n]` = averaged boundary prob per word, plus the
    total log-prob-graph pieces MRT needs.

    Mirrors the window sweep + `probs[i].append(...)` averaging in
    consistency_train.py:cmd_predict verbatim (window / overlap advance,
    last-subword selection, s+w indexing). We additionally keep, for each word,
    the LIST of per-window log-softmax boundary logits so a differentiable
    log P(y) over the stitched sequence can be formed when with_grad=True.

    with_grad=False (sampling pass): runs under no_grad, returns detached probs
    and None for the graph pieces.
    """
    n = len(words)
    prob_lists = [[] for _ in range(n)]      # detached float probs (for sampling)
    logp1_lists = [[] for _ in range(n)]     # log P(boundary=1) tensors  (graph)
    logp0_lists = [[] for _ in range(n)]     # log P(boundary=0) tensors  (graph)
    s = 0
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        while s < n:
            e = min(s + window, n)
            enc = tok(words[s:e], is_split_into_words=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            wid = enc.word_ids(0)
            logit = model(input_ids=enc["input_ids"].to(dev),
                          attention_mask=enc["attention_mask"].to(dev))["logits"][0]
            logsm = torch.log_softmax(logit.float(), -1)   # (T, 2)
            pb = logsm[:, 1].exp()                          # boundary prob per subword
            last = -1
            for pos, w in enumerate(wid):
                if w is None:
                    continue
                nxt = wid[pos + 1] if pos + 1 < len(wid) else None
                if nxt != w:  # last subword of word (s + w)
                    prob_lists[s + w].append(float(pb[pos].detach().cpu()))
                    if with_grad:
                        logp1_lists[s + w].append(logsm[pos, 1])
                        logp0_lists[s + w].append(logsm[pos, 0])
                last = max(last, w)
            if s + last >= n - 1:
                break
            s = max(s + last + 1 - overlap, s + 1)
    # average prob per word (0.0 where no window covered it, matches cmd_predict)
    avg = np.array([float(np.mean(pl)) if pl else 0.0 for pl in prob_lists],
                   dtype=np.float64)
    return avg, logp1_lists, logp0_lists


def _apply_par_zeros(labeling, tokens):
    """[PAR] / paragraph-token positions are never boundaries (mirrors both the
    label pipeline's -100 and cmd_predict's post-hoc zeroing)."""
    out = list(labeling)
    for i, w in enumerate(tokens):
        if w == PARAGRAPH_TOKEN:
            out[i] = 0
    return out


# --------------------------------------------------------------------------- #
# MRT core: sampling, reward, subset distribution, REINFORCE-with-baseline     #
# --------------------------------------------------------------------------- #
def _mrt_doc_loss(model, tok, doc, args, dev):
    """Compute the MRT surrogate loss for ONE document.

    Returns (loss_tensor, stats_dict). loss_tensor is differentiable w.r.t. model
    params through the stitched log-probabilities; stats_dict carries diagnostics
    (n_unique candidates, reward range, |grad-driving weights|) for the smoke.
    """
    tokens = doc["tokens"]
    gold = list(doc["labels"])
    words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in tokens]
    n = len(words)

    # (1) per-word boundary probs via the FROZEN stitcher (no grad, for sampling)
    avg_probs, _, _ = _doc_word_probs(
        model, tok, words, args.window, args.overlap, args.max_length, dev,
        with_grad=False)
    # positions the tagger actually decides on: non-[PAR] words. [PAR] positions
    # are forced 0 and excluded from log P(y) (they carry no boundary decision).
    par_mask = np.array([w == PARAGRAPH_TOKEN for w in tokens], dtype=bool)
    decide = ~par_mask

    # (2) build candidate set S: greedy/argmax ALWAYS included, then k-1 samples
    #     from temperature-scaled Bernoulli on the stitched probs.
    temp = max(args.mrt_temp, 1e-6)
    # temperature on the logit of the prob: p_t = sigmoid(logit(p)/temp)
    eps = 1e-6
    p = np.clip(avg_probs, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    p_temp = 1.0 / (1.0 + np.exp(-logit_p / temp))

    greedy = (avg_probs >= 0.5).astype(int)
    greedy = np.array(_apply_par_zeros(greedy, tokens), dtype=int)

    cands = [tuple(greedy.tolist())]
    seen = {cands[0]}
    rng = np.random  # seeded once per epoch in the caller via set_seed / random
    tries = 0
    while len(cands) < args.mrt_k and tries < args.mrt_k * 20:
        tries += 1
        draw = (rng.random(n) < p_temp).astype(int)
        draw = np.array(_apply_par_zeros(draw, tokens), dtype=int)
        t = tuple(draw.tolist())
        cands.append(t)         # keep even if duplicate: still a valid S member
        seen.add(t)
    cand_arr = [np.array(c, dtype=int) for c in cands]
    n_unique = len(seen)

    # (3+4) reward per candidate: Delta = 1 - per-doc macro-F1 (eval_local logic).
    #       eval_local uses sklearn f1_score(g, p, average="binary", pos=1).
    deltas = []
    for c in cand_arr:
        f1 = f1_score(gold, c.tolist(), average="binary", zero_division=0)
        deltas.append(1.0 - float(f1))
    deltas = np.array(deltas, dtype=np.float64)

    # (5) differentiable log P(y) over the stitched sequence, per candidate.
    #     One grad pass through the doc windows to get per-word log P(b=1)/log P(b=0),
    #     averaged across overlapping windows (same stitching), then summed over
    #     decision positions according to each candidate's 0/1.
    _, logp1_lists, logp0_lists = _doc_word_probs(
        model, tok, words, args.window, args.overlap, args.max_length, dev,
        with_grad=True)
    # per-word averaged log-probs (mean over the windows covering that word)
    logp1 = []
    logp0 = []
    for i in range(n):
        if logp1_lists[i]:
            logp1.append(torch.stack(logp1_lists[i]).mean())
            logp0.append(torch.stack(logp0_lists[i]).mean())
        else:  # uncovered word (shouldn't happen for covered docs); prob->0.5
            z = torch.zeros((), device=dev)
            logp1.append(z + math.log(0.5))
            logp0.append(z + math.log(0.5))

    logP = []  # log P(y) per candidate (sum over decision positions)
    for c in cand_arr:
        terms = []
        for i in range(n):
            if not decide[i]:
                continue
            terms.append(logp1[i] if c[i] == 1 else logp0[i])
        logP.append(torch.stack(terms).sum() if terms else torch.zeros((), device=dev))
    logP = torch.stack(logP)  # (|S|,)

    # subset distribution Q(y) = P(y)^alpha / sum P^alpha, computed stably in
    # log-space and DETACHED (Q is a weighting, not part of the score gradient).
    with torch.no_grad():
        scaled = args.mrt_alpha * logP.detach()
        logQ = scaled - torch.logsumexp(scaled, 0)
        Q = logQ.exp()                                   # (|S|,)  detached
        delta_t = torch.tensor(deltas, device=dev, dtype=Q.dtype)
        Qbar = (Q * delta_t).sum()                       # subset-expected risk = baseline

    # MRT surrogate whose autograd equals the TRUE gradient of the MRT risk
    #   R = sum_y Q(y) Delta(y),  Q(y) = P(y)^alpha / sum P^alpha,
    # which (standard Shen-et-al result) is
    #   grad R = alpha * sum_y Q(y) (Delta(y) - Qbar) * grad logP(y).
    # So the differentiable surrogate to MINIMIZE via gradient DESCENT is
    #   L = alpha * sum_y Q(y).detach() * (Delta(y) - Qbar) * logP(y)   [+logP, not -logP]
    # variance-reduced by the constant baseline Qbar. Using (-logP) here would
    # NEGATE the gradient and make descent ASCEND the risk (drive doc-F1 to 0);
    # omitting alpha would mis-scale the term by 1/alpha (~200x at alpha=5e-3).
    surrogate = (args.mrt_alpha * Q * (delta_t - Qbar) * logP).sum()

    stats = {
        "n_cand": len(cand_arr),
        "n_unique": n_unique,
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "reward_mean_f1": float(1.0 - deltas.mean()),
        "Qbar": float(Qbar.detach().cpu()),
        "surrogate": float(surrogate.detach().cpu()),
        "grad_weight_absmax": float((Q * (delta_t - Qbar)).abs().max().detach().cpu()),
    }
    return surrogate, stats


# --------------------------------------------------------------------------- #
# metrics / warm-start / checkpoint                                            #
# --------------------------------------------------------------------------- #
def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def _load_init_model(model, init_dir):
    """Warm-start: load a prior HF token-classification checkpoint's weights into
    model.enc. `init_dir` is a run directory written by any of the sibling
    trainers' _save_hf_checkpoint (config.json + weights). We resize first so the
    [PAR] special-token row exists, then load with strict=False for robustness to
    the extra head/label buffers. Returns the number of matched tensors."""
    src = AutoModelForTokenClassification.from_pretrained(init_dir, num_labels=2)
    sd = src.state_dict()
    tgt = model.enc.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in tgt and tgt[k].shape == v.shape}
    missing = model.enc.load_state_dict({**tgt, **matched}, strict=False)
    return len(matched), missing


def _save_hf_checkpoint(model, tok, out_dir):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / eval_local.py / fair_threshold.py / diag_preposition.py
    load with AutoModelForTokenClassification.from_pretrained(out_dir) +
    AutoTokenizer, which the custom .pt cannot satisfy. Here `model.enc` IS the
    trained AutoModelForTokenClassification (encoder + boundary classifier); MRT
    adds NO parameters (the reward / REINFORCE terms are training-time only), so
    `model.enc` is exactly the inference model. We write it verbatim. This does
    not touch training, the loss, the flags, or the existing .pt/cfg.json outputs
    — it only adds config.json + weights + tokenizer files so the downstream tools
    can read the run directory. Mirrors consistency_train.py:_save_hf_checkpoint."""
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #
def _train_ce_baseline(a, tok, model, tr, dv):
    """MRT OFF path: the plain weighted-CE window trainer, byte-identical in loss
    math to consistency_train.py's baseline (HF Trainer, same TrainingArguments).
    Guarantees the default reproduces yesterday's weighted CE bit-for-bit."""
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride))
    dvds = Dataset.from_list(make_examples(dv, a.window, a.window))  # clean eval
    fn = lambda ex: encode(ex, tok, a.max_length)
    trds = trds.map(fn, batched=True, remove_columns=trds.column_names)
    dvds = dvds.map(fn, batched=True, remove_columns=dvds.column_names)
    targs = TrainingArguments(
        output_dir=os.path.join(a.out, "ckpts"), num_train_epochs=a.epochs,
        learning_rate=a.lr, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1",
        warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
        logging_steps=20, save_total_limit=1, report_to="none", seed=a.seed,
        remove_unused_columns=False, label_names=["labels"])
    tr_ = Trainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                  data_collator=Collator(tok), compute_metrics=metrics)
    tr_.train()
    print("best window dev:", tr_.evaluate())


def _eval_dev_docf1(model, tok, dv, a, dev):
    """Doc-level macro-F1 over the dev split via the frozen stitcher — the metric
    MRT actually optimizes. Used only for logging during the MRT loop."""
    model.eval()
    f1s = []
    for d in dv:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        avg, _, _ = _doc_word_probs(model, tok, words, a.window, a.overlap,
                                    a.max_length, dev, with_grad=False)
        lab = _apply_par_zeros((avg >= a.threshold).astype(int).tolist(), d["tokens"])
        f1s.append(f1_score(list(d["labels"]), lab, average="binary", zero_division=0))
    model.train()
    return float(np.mean(f1s)) if f1s else 0.0


def _train_mrt(a, tok, model, tr, dv, dev):
    """MRT ON path: fine-tune. Base weighted CE on the doc's windows PLUS the
    document-level MRT surrogate (weighted by --mrt-weight). One optimizer step
    per document (the unit of risk). Warm-started upstream via --init-model."""
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    # CE on the doc's windows keeps the model well-formed while MRT shapes it
    # toward doc-F1 (matches Shen et al.'s practice of MRT as a fine-tune with a
    # standing likelihood term). mrt_ce_weight defaults to 1.0.
    n_docs = len(tr)
    step = 0
    # EARLY STOPPING: MRT optimizes TRAIN doc-F1, so more epochs overfit the 174
    # docs. Keep the epoch with the best DEV doc-macro-F1 (not the final one) and
    # stop if it stalls. This replaces guessing a fixed epoch count.
    best_f1, best_state, no_improve = -1.0, None, 0
    mrt_patience = int(getattr(a, "mrt_patience", 2))
    for ep in range(a.epochs):
        order = list(range(n_docs))
        random.shuffle(order)
        run_ce = run_mrt = 0.0
        for di in order:
            d = tr[di]
            opt.zero_grad()
            # base weighted-CE over this doc's windows
            ce_loss = torch.zeros((), device=dev)
            exs = make_examples([d], a.window, a.stride)
            enc = [encode({"words": [e["words"]], "word_labels": [e["word_labels"]]},
                          tok, a.max_length) for e in exs]
            feats = [{k: e[k][0] for k in ("input_ids", "attention_mask", "labels")}
                     for e in enc]
            coll = Collator(tok)
            for j in range(0, len(feats), a.batch_size):
                batch = coll([dict(f) for f in feats[j:j + a.batch_size]])
                batch = {k: v.to(dev) for k, v in batch.items()}
                out = model(**batch)
                ce_loss = ce_loss + out["loss"] * (batch["labels"] != -100).sum()
            ntok = sum(int((torch.tensor(f["labels"]) != -100).sum()) for f in feats)
            ce_loss = ce_loss / max(ntok, 1)
            # document-level MRT surrogate
            mrt_loss, st = _mrt_doc_loss(model, tok, d, a, dev)
            loss = a.mrt_ce_weight * ce_loss + a.mrt_weight * mrt_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
            opt.step()
            run_ce += float(ce_loss.detach().cpu())
            run_mrt += float(mrt_loss.detach().cpu())
            step += 1
            if step % a.log_every == 0:
                print(f"  ep{ep} step{step} doc={d['doc_id']} "
                      f"ce={float(ce_loss.detach().cpu()):.4f} "
                      f"mrt={float(mrt_loss.detach().cpu()):+.4e} "
                      f"reward_f1={st['reward_mean_f1']:.3f} "
                      f"n_uniq={st['n_unique']}")
        dev_f1 = _eval_dev_docf1(model, tok, dv, a, dev)
        print(f"[MRT ep{ep}] mean_ce={run_ce / n_docs:.4f} "
              f"mean_mrt={run_mrt / n_docs:+.4e} dev_doc_F1={dev_f1:.4f}")
        # keep the best-dev-docF1 epoch; stop if no improvement for `mrt_patience`
        if dev_f1 > best_f1 + 1e-6:
            best_f1, no_improve = dev_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[MRT ep{ep}] new BEST dev_doc_F1={best_f1:.4f} (checkpoint kept)")
        else:
            no_improve += 1
            if no_improve >= mrt_patience:
                print(f"[MRT] early stop @ep{ep}: no dev gain for {mrt_patience} epochs "
                      f"(best={best_f1:.4f})")
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        print(f"[MRT] restored BEST checkpoint: dev_doc_F1={best_f1:.4f}")


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    # Convenience: `--mrt on` with the weight left at its default 0.0 means the
    # user asked for MRT but did not name a weight — promote it to 1.0 so "on"
    # actually turns MRT on. `--mrt off` (the DEFAULT) and an EXPLICIT
    # `--mrt-weight 0` both remain the plain weighted-CE baseline (bit-for-bit).
    if str(a.mrt).lower() == "on" and float(a.mrt_weight) == 0.0:
        a.mrt_weight = 1.0
        print("note: --mrt on with default weight -> --mrt-weight 1.0")
    mrt_on = _mrt_enabled(a)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  mrt={'ON' if mrt_on else 'OFF'}  "
          f"mrt_weight={a.mrt_weight}  mrt_k={a.mrt_k}  mrt_alpha={a.mrt_alpha}  "
          f"mrt_temp={a.mrt_temp}  init_model={a.init_model}")
    model = MRTModel(a.model_name, pw, n_extra_tok=1)
    if a.init_model:
        nmatch, miss = _load_init_model(model, a.init_model)
        print(f"warm-start from {a.init_model}: matched {nmatch} tensors "
              f"(missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)})")
    elif mrt_on:
        print("WARNING: --mrt on without --init-model — MRT is a fine-tune stage; "
              "the recipe warm-starts from a weighted-CE checkpoint.")
    model.to(dev)

    if mrt_on:
        _train_mrt(a, tok, model, tr, dv, dev)
    else:
        _train_ce_baseline(a, tok, model, tr, dv)

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name}, open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: also emit an HF checkpoint so predict.py / eval_local.py /
    # diag_preposition.py can load this run dir. Nothing above is changed.
    _save_hf_checkpoint(model, tok, a.out)
    print(f"saved -> {a.out}")


# --------------------------------------------------------------------------- #
# predict — writes a submission CSV scorable by eval_local.py                   #
# (window sweep / stitching identical to consistency_train.py:cmd_predict)     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = MRTModel(cfg["model_name"], n_extra_tok=1)
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
# flag plumbing                                                                #
# --------------------------------------------------------------------------- #
def _mrt_enabled(a):
    """MRT is ON iff --mrt on AND weight > 0. --mrt off (default) OR weight 0.0 is
    the plain weighted-CE baseline (bit-for-bit)."""
    return (str(a.mrt).lower() == "on") and (float(a.mrt_weight) > 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--init-model", default=None,
                   help="warm-start HF checkpoint dir (MRT is a fine-tune stage)")
    # MRT flags — all OFF by default (bit-for-bit weighted CE)
    t.add_argument("--mrt", choices=["on", "off"], default="off",
                   help="on = minimum risk training fine-tune; off (default) = plain weighted CE")
    t.add_argument("--mrt-weight", type=float, default=0.0,
                   help="weight on the doc-level MRT surrogate (0 = off = bit-for-bit CE)")
    t.add_argument("--mrt-ce-weight", type=float, default=1.0,
                   help="weight on the standing weighted-CE term during MRT fine-tune")
    t.add_argument("--mrt-k", type=int, default=6,
                   help="candidate labelings per doc (greedy + k-1 samples)")
    t.add_argument("--mrt-alpha", type=float, default=5e-3,
                   help="MRT subset-distribution temperature exponent (Shen et al. alpha)")
    t.add_argument("--mrt-temp", type=float, default=1.0,
                   help="Bernoulli sampling temperature on the per-token boundary probs")
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--log-every", type=int, default=25)
    # shared window/CE hyperparams (mirror consistency_train.py)
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--overlap", type=int, default=60)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--threshold", type=float, default=0.5)
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--task", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(); a.func(a)
