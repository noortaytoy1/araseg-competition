"""Hard-negative contrastive boundary tagger for AraSeg NoPnx-NP (closed track).

Noor's idea, sharpened: the model's worst failure mode is CONFIDENT false
boundaries (over-firing). Instead of only re-weighting the CE, we mine the
positions the warm CE baseline is confidently WRONG about on TRAIN, and push
their representations AWAY from the boundary-anchor centroid with a margin
contrastive loss on a small projection head.

Self-contained — does NOT touch the shared train_encoder/predict pipeline. The
data / window / label / [PAR] handling MIRRORS src/consistency_train.py exactly
(window 180, stride 90, labels on the LAST subword of each word,
MODEL_PAR_TOKEN for the "\n" paragraph token, weighted CE with pos_weight
capped ~8, HF Trainer, epochs 8, lr 5e-5, batch 16, HF-checkpoint save so
predict.py / eval_local.py / the battery scorers load the run dir verbatim).

CLOSED-TRACK / EVALUATION HYGIENE: hard negatives are mined on TRAIN docs with
a TRAIN-trained checkpoint. Dev is never read at mining time and never trained
on. The dev false positives (evaluation data) are NOT used anywhere here.

MECHANISM (all gated by --hardneg-weight, default 0.0 = bit-for-bit plain CE:
no mining, no projection head created, single forward, identical loss and
identical RNG stream to consistency_train.py's cons_baseline):

  1. HARD-NEGATIVE MINING (once, at trainer startup, TRAIN-side).
     --init-model points at a trained CE checkpoint (e.g. the frontier
     cons_baseline of the same seed). NOTE: --init-model is used ONLY as the
     MINER — the student below still trains FROM SCRATCH with the exact
     cons_baseline recipe, so the A/B vs cons_baseline isolates the added loss
     term (unlike mrt_train.py, where --init-model warm-starts the weights).
     The miner runs over the TRAIN docs with the same windowed mean-pooled
     word-probability used by predict (window 180, overlap 60).
     --hardneg-source:
       'train-fp'     : hard negatives = gold-0 positions with p >= --hardneg-thresh
                        (default 0.8). The model memorizes train (~99 F1), so
                        these can be FEW — the count is measured and printed.
       'train-margin' : gold-0 positions with the HIGHEST p, top --hardneg-topn
                        (default 2000), regardless of the 0.8 bar.
       'auto' (default): use 'train-fp' if it yields >= --hardneg-min-count
                        (default 200) positions, else FALL BACK to
                        'train-margin'. Which source fired is printed and
                        recorded in cfg.json + mined_negatives.jsonl.

  2. LOSS (per batch). anchors = valid gold-1 token representations (LAST-
     subword hidden states, the same grid the CE uses). negatives = the mined
     hard-negative positions present in the batch, PLUS the --dyn-negs
     (default 64) in-batch valid gold-0 tokens with the highest CURRENT
     boundary probability (dynamic negatives; selection detached). A small
     projection head (supcon_head.ProjectionHead pattern: Linear-ReLU-Linear
     -> L2 norm) maps hidden states to unit vectors; with
     c = normalize(mean(anchor projections))  (the anchor centroid),
     s(x) = cos(c, x)  (dot of unit vectors),
       L_hn = mean over (anchor a, negative n) pairs of
              relu(margin - (s(a) - s(n)))          margin default 0.3.
     --arctanh {on,off} (default off): replace the raw-cosine score with
     s(x) = arctanh(clamp(cos, -0.9999, 0.9999)) per arXiv 2501.17683 — the
     clamp keeps it finite at extreme cosines.

  3. loss = weighted_CE + hardneg_weight * L_hn.
     The projection head is DISCARDED at inference: the saved HF checkpoint is
     model.enc only, and predict uses logits only.

Usage:
  # baseline (bit-for-bit identical to plain weighted CE / cons_baseline):
  python hardneg_contrastive_train.py train --task NoPnx-NP --out runs/hn_base \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # hard-negative contrastive:
  python hardneg_contrastive_train.py train --task NoPnx-NP --hardneg-weight 0.3 \
      --init-model runs/frontier_battery_2026-07-09_v3/cons_baseline_s42 \
      --out runs/hn03 --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # mining only (CPU-OK; measures the train-side confident-FP count):
  python hardneg_contrastive_train.py mine --task NoPnx-NP \
      --train-jsonl data/NoPnx-NP_train.jsonl \
      --init-model runs/frontier_battery_2026-07-09_v3/cons_baseline_s42 \
      --out-jsonl runs/hardneg_battery_2026-07-10/mined_s42.jsonl

  python hardneg_contrastive_train.py predict --task NoPnx-NP --split dev \
      --model runs/hn03 --jsonl data/NoPnx-NP_dev.jsonl --out subs/hn03_dev.csv
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission
from supcon_head import ProjectionHead  # reused pattern; SupConLoss itself unused

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py; + per-word hn flag)   #
# --------------------------------------------------------------------------- #
def make_examples(docs, window, stride, hn_sets=None):
    """Slide a (window, stride) window over each doc; [PAR] positions -> -100.

    hn_sets: optional {doc_index: set(word_index)} of mined hard negatives;
    emitted per window as a parallel 0/1 `word_hn` list (all zeros when None,
    which is the --hardneg-weight 0 / dev-eval case)."""
    out = []
    for di, d in enumerate(docs):
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        hn = [0] * len(words)
        if hn_sets:
            for wi in hn_sets.get(di, ()):
                hn[wi] = 1
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            out.append({"words": words[s:e], "word_labels": labs[s:e],
                        "word_hn": hn[s:e]})
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, max_len):
    """Sub-word encode; put each word's label (and hn flag) on its LAST subword."""
    enc = tok(examples["words"], is_split_into_words=True, truncation=True, max_length=max_len)
    all_lab, all_hn = [], []
    for i, wl in enumerate(examples["word_labels"]):
        wh = examples["word_hn"][i]
        wid = enc.word_ids(batch_index=i)
        lab = [-100] * len(wid)
        hnm = [0] * len(wid)
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:  # last subword of word w
                lab[pos] = wl[w]
                hnm[pos] = wh[w]
        all_lab.append(lab)
        all_hn.append(hnm)
    enc["labels"] = all_lab
    enc["hn_mask"] = all_hn
    return enc


class Collator:
    """Pads input ids/masks via tok.pad; pads labels with -100 and hn_mask with 0."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        hns = [f.pop("hn_mask") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        batch["hn_mask"] = torch.tensor([h + [0] * (m - len(h)) for h in hns])
        return batch


# --------------------------------------------------------------------------- #
# model wrapper: plain token-classification + flag-gated hard-neg margin loss  #
# --------------------------------------------------------------------------- #
class HardNegModel(nn.Module):
    def __init__(self, model_name, pos_weight=1.0, hardneg_weight=0.0, margin=0.3,
                 arctanh=False, proj_dim=128, dyn_negs=64, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.hardneg_weight = float(hardneg_weight)
        self.margin = float(margin)
        self.arctanh = bool(arctanh)
        self.dyn_negs = int(dyn_negs)
        # Projection head ONLY when the mechanism is on: at weight 0 no extra
        # parameters exist and no extra RNG is drawn -> the init (and therefore
        # the whole training trajectory) is bit-for-bit the plain CE baseline.
        if self.hardneg_weight > 0:
            self.proj = ProjectionHead(self.enc.config.hidden_size, proj_dim=proj_dim)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config
        # readouts for smokes/logging (floats, not part of the graph)
        self.last_hn = 0.0
        self.last_n_anchor = 0
        self.last_n_neg = 0

    def _logits(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).logits

    def _ce(self, logits, labels):
        return nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100)

    def _sim(self, cos):
        """Similarity score from a cosine. Default: the raw cosine. --arctanh on:
        arctanh(cos) per arXiv 2501.17683, clamped to +/-0.9999 for stability."""
        if self.arctanh:
            return torch.arctanh(cos.clamp(-0.9999, 0.9999))
        return cos

    def _hardneg_loss(self, hidden, logits, labels, hn_mask):
        """Margin loss pushing negatives below anchors w.r.t. the anchor centroid.

        anchors   : valid gold-1 last-subword hidden states.
        negatives : mined hard negatives present in the batch (hn_mask==1 at a
                    valid gold-0 position) + top --dyn-negs valid gold-0 tokens
                    by CURRENT boundary prob (dynamic; selection detached).
        Returns (loss, n_anchor, n_neg); grad-safe zero when either side empty.
        """
        valid = labels != -100
        anch = valid & (labels == 1)
        n_anch = int(anch.sum())
        if n_anch == 0:
            return hidden.sum() * 0.0, 0, 0
        neg_pool = valid & (labels == 0)
        if hn_mask is not None:
            mined = neg_pool & hn_mask.bool()
        else:
            mined = torch.zeros_like(neg_pool)
        dyn_pool = neg_pool & ~mined
        dyn = torch.zeros_like(neg_pool)
        n_dyn = min(self.dyn_negs, int(dyn_pool.sum()))
        if n_dyn > 0:
            p = torch.softmax(logits.detach().float(), -1)[..., 1]
            scores = p.masked_fill(~dyn_pool, -1.0).reshape(-1)
            top = scores.topk(n_dyn).indices
            dyn.reshape(-1)[top] = True
        negs = mined | dyn
        n_neg = int(negs.sum())
        if n_neg == 0:
            return hidden.sum() * 0.0, n_anch, 0
        h32 = hidden.float()
        z_a = self.proj(h32[anch])                       # [Na, D] unit-norm
        z_n = self.proj(h32[negs])                       # [Nn, D] unit-norm
        c = nn.functional.normalize(z_a.mean(0), dim=-1)  # anchor centroid
        s_a = self._sim((z_a @ c).float())               # [Na]
        s_n = self._sim((z_n @ c).float())               # [Nn]
        l = torch.relu(self.margin - (s_a.unsqueeze(1) - s_n.unsqueeze(0))).mean()
        return l, n_anch, n_neg

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                hn_mask=None, **kw):
        need_hidden = self.hardneg_weight > 0 and labels is not None
        if need_hidden:
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            logits, hidden = out.logits, out.hidden_states[-1]
        else:
            # Gated OFF at weight 0: single plain forward -> loss is exactly
            # plain weighted CE (bit-for-bit).
            logits = self._logits(input_ids, attention_mask)
        loss = None
        if labels is not None:
            loss = self._ce(logits, labels)
            if self.hardneg_weight > 0:
                l_hn, na, nn_ = self._hardneg_loss(hidden, logits, labels, hn_mask)
                self.last_hn = float(l_hn.detach())
                self.last_n_anchor, self.last_n_neg = na, nn_
                loss = loss + self.hardneg_weight * l_hn
        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# hard-negative mining (TRAIN-side; once, at startup or via `mine` subcommand) #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _doc_word_probs(words, tok, model, device, window, overlap, max_length,
                    batch_windows=16):
    """Mean-pooled per-word boundary probability over sliding windows.

    Window stepping and last-subword pooling are IDENTICAL to cmd_predict below
    (and to consistency_train.py's predict); windows are batched for speed."""
    n = len(words)
    probs = {i: [] for i in range(n)}
    encs = []  # (window start word index, BatchEncoding)
    s = 0
    while s < n:
        e = min(s + window, n)
        enc = tok(words[s:e], is_split_into_words=True, truncation=True,
                  max_length=max_length)
        wid = enc.word_ids()
        last = max((w for w in wid if w is not None), default=-1)
        encs.append((s, enc))
        if s + last >= n - 1:
            break
        s = max(s + last + 1 - overlap, s + 1)
    for i in range(0, len(encs), batch_windows):
        chunk = encs[i:i + batch_windows]
        feats = [{"input_ids": c[1]["input_ids"],
                  "attention_mask": c[1]["attention_mask"]} for c in chunk]
        batch = tok.pad(feats, return_tensors="pt")
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        pb = torch.softmax(logits.float(), -1)[..., 1].cpu().numpy()
        for j, (s0, enc) in enumerate(chunk):
            wid = enc.word_ids()
            for pos, w in enumerate(wid):
                if w is None:
                    continue
                nxt = wid[pos + 1] if pos + 1 < len(wid) else None
                if nxt != w:  # last subword of word w
                    probs[s0 + w].append(float(pb[j][pos]))
    return [float(np.mean(probs[i])) if probs[i] else 0.0 for i in range(n)]


def mine_hard_negatives(init_model, docs, tok, device, source="auto",
                        thresh=0.8, topn=2000, min_count=200,
                        window=180, overlap=60, max_length=512):
    """Mine TRAIN-side hard negatives with a trained CE checkpoint.

    Returns (hn_sets {doc_index: set(word_index)}, stats dict, rows list).
    rows: one dict per mined position, sorted by p desc (for the record file).
    """
    miner = AutoModelForTokenClassification.from_pretrained(init_model).to(device).eval()
    cands = []  # (p, doc_index, word_index)
    n_gold0 = 0
    for di, d in enumerate(docs):
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        pw = _doc_word_probs(words, tok, miner, device, window, overlap, max_length)
        for wi, (w, lab) in enumerate(zip(d["tokens"], d["labels"])):
            if w == PARAGRAPH_TOKEN:
                continue
            if lab == 0:
                n_gold0 += 1
                cands.append((pw[wi], di, wi))
    del miner
    cands.sort(key=lambda t: -t[0])
    fp = [c for c in cands if c[0] >= thresh]
    n_fp = len(fp)
    if source == "train-fp":
        fired, chosen = "train-fp", fp
    elif source == "train-margin":
        fired, chosen = "train-margin", cands[:topn]
    else:  # auto: pre-registered fallback — train-fp if it yields enough, else margin
        if n_fp >= min_count:
            fired, chosen = "train-fp", fp
        else:
            fired, chosen = "train-margin", cands[:topn]
    hn_sets = {}
    rows = []
    for p, di, wi in chosen:
        hn_sets.setdefault(di, set()).add(wi)
        toks = docs[di]["tokens"]
        ctx = " ".join(t if t != PARAGRAPH_TOKEN else "[PAR]"
                       for t in toks[max(0, wi - 3):wi + 4])
        rows.append({"doc_id": docs[di].get("doc_id", str(di)), "doc_index": di,
                     "word_index": wi, "p": round(p, 6), "word": toks[wi],
                     "context": ctx})
    stats = {"n_gold0": n_gold0, "n_train_fp_at_thresh": n_fp,
             "thresh": thresh, "topn": topn, "min_count": min_count,
             "source_requested": source, "source_fired": fired,
             "n_mined": len(rows), "init_model": init_model}
    print(f"[mine] gold-0 positions: {n_gold0}  "
          f"confident train FPs (p>={thresh}): {n_fp}  "
          f"source requested: {source}  -> FIRED: {fired}  mined: {len(rows)}")
    for r in rows[:10]:
        print(f"[mine]   p={r['p']:.4f}  doc={r['doc_id']}  w#{r['word_index']}  "
              f"word={r['word']!r}  ctx={r['context']!r}")
    return hn_sets, stats, rows


# --------------------------------------------------------------------------- #
# metrics / train / mine / predict                                             #
# --------------------------------------------------------------------------- #
def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def _save_hf_checkpoint(model, tok, out_dir):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json — so predict.py /
    eval_local.py / fair_threshold.py / the battery scorers can load the run dir
    with AutoModelForTokenClassification.from_pretrained. `model.enc` IS the
    trained classifier; the projection head adds parameters ONLY to model.pt and
    is DISCARDED here (inference never uses it). Mirrors
    consistency_train.py:_save_hf_checkpoint."""
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  hardneg_weight={a.hardneg_weight}  margin={a.margin}  "
          f"arctanh={a.arctanh}  dyn_negs={a.dyn_negs}  source={a.hardneg_source}  "
          f"init_model={a.init_model}")

    hn_sets, mine_stats = None, {}
    if a.hardneg_weight > 0:
        if not a.init_model:
            raise SystemExit("--hardneg-weight > 0 requires --init-model "
                             "(a trained CE checkpoint used ONLY to mine TRAIN-side "
                             "hard negatives; the student still trains from scratch).")
        mdev = "cuda" if torch.cuda.is_available() else "cpu"
        hn_sets, mine_stats, rows = mine_hard_negatives(
            a.init_model, tr, tok, mdev, source=a.hardneg_source,
            thresh=a.hardneg_thresh, topn=a.hardneg_topn,
            min_count=a.hardneg_min_count, window=a.window, overlap=60,
            max_length=a.max_length)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "mined_negatives.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"stats": mine_stats}, ensure_ascii=False) + "\n")
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    model = HardNegModel(a.model_name, pw, a.hardneg_weight, margin=a.margin,
                         arctanh=(a.arctanh == "on"), proj_dim=a.proj_dim,
                         dyn_negs=a.dyn_negs, n_extra_tok=1)
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride, hn_sets))
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
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name, "hardneg_weight": a.hardneg_weight,
               "margin": a.margin, "arctanh": a.arctanh, "proj_dim": a.proj_dim,
               "dyn_negs": a.dyn_negs, "hardneg_source": a.hardneg_source,
               "init_model": a.init_model, "mine_stats": mine_stats},
              open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: plain HF checkpoint (enc only; projection head discarded) so
    # predict.py / eval_local.py / battery scorers can read the run directory.
    _save_hf_checkpoint(model, tok, a.out)
    print(f"saved -> {a.out}")


def cmd_mine(a):
    """Mining only (CPU-OK). Measures the TRAIN-side confident-FP count and
    writes the mined list + stats to --out-jsonl. Trains nothing."""
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mine] device={dev}  docs={len(tr)}")
    hn_sets, stats, rows = mine_hard_negatives(
        a.init_model, tr, tok, dev, source=a.hardneg_source,
        thresh=a.hardneg_thresh, topn=a.hardneg_topn,
        min_count=a.hardneg_min_count, window=a.window, overlap=60,
        max_length=a.max_length)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_jsonl)), exist_ok=True)
    with open(a.out_jsonl, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"stats": stats}, ensure_ascii=False) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mine] wrote stats + {len(rows)} rows -> {a.out_jsonl}")


@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    # Rebuild with the SAVED flags so the state dict matches (head present iff
    # trained with weight > 0); inference uses logits only — head is dead code.
    model = HardNegModel(cfg["model_name"],
                         hardneg_weight=cfg.get("hardneg_weight", 0.0),
                         margin=cfg.get("margin", 0.3),
                         arctanh=(cfg.get("arctanh", "off") == "on"),
                         proj_dim=cfg.get("proj_dim", 128),
                         dyn_negs=cfg.get("dyn_negs", 64), n_extra_tok=1)
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


def _add_mine_flags(sp):
    sp.add_argument("--hardneg-source", choices=["auto", "train-fp", "train-margin"],
                    default="auto",
                    help="auto (default): train-fp if it yields >= --hardneg-min-count, "
                         "else fall back to train-margin (documented in cfg.json)")
    sp.add_argument("--hardneg-thresh", type=float, default=0.8,
                    help="train-fp bar: gold-0 with p >= this (default 0.8)")
    sp.add_argument("--hardneg-topn", type=int, default=2000,
                    help="train-margin: top-N hardest gold-0 by p (default 2000)")
    sp.add_argument("--hardneg-min-count", type=int, default=200,
                    help="auto-mode fallback bar (default 200)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--hardneg-weight", type=float, default=0.0,
                   help="hard-negative margin-contrastive loss weight (0 = off, "
                        "bit-for-bit plain weighted CE)")
    t.add_argument("--init-model", default=None,
                   help="trained CE checkpoint used ONLY to mine TRAIN-side hard "
                        "negatives (NOT a weight warm start; student trains from scratch)")
    _add_mine_flags(t)
    t.add_argument("--margin", type=float, default=0.3)
    t.add_argument("--arctanh", choices=["on", "off"], default="off",
                   help="on: score = arctanh(clamp(cos, +/-0.9999)) per arXiv 2501.17683")
    t.add_argument("--proj-dim", type=int, default=128)
    t.add_argument("--dyn-negs", type=int, default=64,
                   help="in-batch dynamic negatives: top-K gold-0 by current p")
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("mine")
    m.add_argument("--task", required=True)
    m.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    m.add_argument("--train-jsonl"); m.add_argument("--init-model", required=True)
    m.add_argument("--out-jsonl", required=True)
    _add_mine_flags(m)
    m.add_argument("--window", type=int, default=180)
    m.add_argument("--max-length", type=int, default=512)
    m.set_defaults(func=cmd_mine)

    p = sub.add_parser("predict")
    p.add_argument("--task", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(); a.func(a)
