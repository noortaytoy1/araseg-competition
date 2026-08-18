"""Sentence-level paragraph-membership contrastive tagger (SupCon at the
SENTENCE level; Noor's idea). Self-contained — does NOT touch the shared
train_encoder/predict pipeline (mirrors src/contrastive_train.py's structure:
train/predict subcommands, its own model class, its own dataset/collator).

THE IDEA (SENTENCE-level, not token-level)
------------------------------------------
The real task is unchanged: a token-level binary head predicts a sentence
boundary AFTER token i (class-weighted cross-entropy, identical to
contrastive_train.py). On top of that we add ONE auxiliary objective:

  * Pool each SENTENCE's token hidden states into ONE sentence embedding
    (mean-pool over the sentence's subword positions).
  * Supervised-contrastive signal = PARAGRAPH MEMBERSHIP. Two sentences in the
    SAME paragraph are a POSITIVE pair (pull together); two sentences in
    DIFFERENT paragraphs are NEGATIVE (push apart). This is exactly Supervised
    Contrastive learning (Khosla et al. 2020) applied at the SENTENCE level with
    the paragraph-id as the contrastive class label.

  loss = boundary_CE + w * sentence_paragraph_SupCon

We REUSE the vendored SupConLoss (vendor/SupContrast/losses.py, imported via
src/supcon_head.py) — the contrastive math is the authors' code, unchanged. We
also reuse supcon_head.ProjectionHead's spirit but keep the projection head
identical to contrastive_train.py's SimCSE head (Linear -> Tanh) as the prompt
specifies, then L2-normalize (SupConLoss expects unit-norm features).

DATA SEMANTICS (NoPnx-PA, verified on data/NoPnx-PA_train.jsonl)
----------------------------------------------------------------
  * "\n" == data.PARAGRAPH_TOKEN marks a paragraph break; it is fed to the model
    as MODEL_PAR_TOKEN "[PAR]" (an added special token). A [PAR] token is NEVER
    a boundary (label 0 for all 3327 of them) and is NEVER inside a sentence; it
    carries label -100 so it is ignored by the boundary CE and excluded from
    every sentence span.
  * Every paragraph ends on a boundary token (the word just before each [PAR]
    has label 1 in all 3327 cases). ~3.1 sentences/paragraph (10247 boundaries
    / 3327 paragraphs in train).
  * A SENTENCE is a maximal token span ending on a label-1 (boundary) word,
    WITHIN a paragraph. Paragraphs are the chunks between [PAR] tokens.

Paragraph ids are GLOBALLY unique per (doc, paragraph-index-within-doc): the
paragraph counter runs over the WHOLE document (incrementing at each [PAR]) and
is packed with the document index into one int id. Only genuinely-same-paragraph
sentences share an id, so only they count as positives. Windows are cut so a
batch contains sentences from several paragraphs => negatives always exist.

With --para-contrastive-weight 0 (default) the projection head and the SupCon
term are never touched and forward() returns exactly the baseline class-weighted
CE — bit-for-bit the plain boundary baseline. The projection head is discarded
at inference; predict() uses the boundary head only.

TRAIN-ONLY: fits/reads only the train split. dev is used solely by the existing
eval (window-dev for model selection), never for the contrastive labels.

Usage:
  python para_contrastive_train.py train --task NoPnx-PA \
      --para-contrastive-weight 0.5 --out runs/para_cl
  python para_contrastive_train.py predict --task NoPnx-PA --split dev \
      --model runs/para_cl --out subs/x.csv
  python para_contrastive_train.py selftest        # CPU/tiny proof (see below)
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (AutoModel, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

# Vendored HobbitLong/SupContrast SupConLoss (Khosla et al. 2020), imported via
# the existing glue module so the contrastive math is the authors' code verbatim.
from supcon_head import SupConLoss

HERE = os.path.dirname(os.path.abspath(__file__))

# Sentinel word-id column value for "not part of any sentence" (a word that is
# a [PAR] token, or a truncated tail after the last boundary in a window). Kept
# out of int8/long confusion by using a large negative int.
NO_SENT = -1


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class ParaCLModel(nn.Module):
    """Boundary tagger + sentence projection head for paragraph-membership SupCon.

    The boundary head (Linear -> 2) is the real task. The sentence projection
    head (Linear -> proj_dim -> Tanh, then L2-normalize) maps a pooled sentence
    embedding into the contrastive space; it is used ONLY by the SupCon loss and
    is discarded at inference. With para_contrastive_weight == 0 the projection
    head and the SupCon term are never touched and forward() returns exactly the
    baseline class-weighted CE logits/loss.
    """

    def __init__(self, model_name, pos_weight=1.0, para_contrastive_weight=0.0,
                 proj_dim=128, temp=0.1, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        # +1 slot for the added [PAR] special token (matches contrastive_train.py).
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.bound = nn.Linear(h, 2)                       # THE TASK
        # Sentence projection head g(.): Linear -> proj_dim -> Tanh (then L2-norm
        # in forward). Same head design as contrastive_train.py's SimCSE head.
        self.proj = nn.Sequential(nn.Linear(h, proj_dim), nn.Tanh())
        self.para_contrastive_weight = para_contrastive_weight
        self.temp = temp
        self.proj_dim = proj_dim
        # Vendored SupConLoss; base_temperature == temperature so the scale is 1.
        self._supcon = SupConLoss(temperature=temp, base_temperature=temp) \
            if para_contrastive_weight > 0 else None
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    # -- sentence pooling ---------------------------------------------------- #
    def _sentence_embeddings(self, hs, sent_ids, par_ids):
        """Mean-pool token hidden states into per-sentence embeddings.

        hs       : [B, T, H] encoder last hidden states.
        sent_ids : [B, T] long. A GLOBAL sentence index (unique across the whole
                   batch) for every SUBWORD position that belongs to a sentence,
                   or NO_SENT for positions that belong to no sentence ([PAR],
                   padding, truncated tail). Every subword of a word inherits the
                   word's sentence id, so pooling is over ALL of a sentence's
                   subword positions.
        par_ids  : [n_global_sent] long. par_ids[s] = the global paragraph id of
                   global sentence s (built by the collator).

        Returns (sent_emb, labels):
          sent_emb : [n_sent, proj_dim] L2-normalized sentence embeddings, in
                     increasing global-sentence-id order.
          labels   : [n_sent] long paragraph ids aligned to sent_emb.
        Returns (None, None) if fewer than 2 sentences are present.
        """
        B, T, H = hs.shape
        flat_hs = hs.reshape(B * T, H)
        flat_sid = sent_ids.reshape(B * T)
        valid = flat_sid != NO_SENT
        if valid.sum() == 0:
            return None, None
        flat_hs = flat_hs[valid]
        flat_sid = flat_sid[valid]
        # Global sentence ids are dense 0..n_sent-1 (built by the collator).
        n_sent = int(par_ids.numel())
        if n_sent < 2:
            return None, None
        # Sum-pool then divide by count => mean-pool per sentence.
        sums = flat_hs.new_zeros(n_sent, H)
        sums.index_add_(0, flat_sid, flat_hs)
        counts = flat_hs.new_zeros(n_sent)
        counts.index_add_(0, flat_sid, torch.ones_like(flat_sid, dtype=flat_hs.dtype))
        # A sentence with no surviving subword (shouldn't happen, but guard) keeps
        # a zero vector; clamp counts to avoid divide-by-zero.
        pooled = sums / counts.clamp_min(1.0).unsqueeze(1)          # [n_sent, H]
        z = self.proj(pooled)                                      # [n_sent, proj]
        z = nn.functional.normalize(z, dim=-1)                     # unit norm
        return z, par_ids

    def _para_supcon(self, hs, sent_ids, par_ids):
        """Sentence-level paragraph-membership SupCon loss (differentiable 0 if
        there are <2 sentences or <2 distinct paragraphs among them)."""
        z, labels = self._sentence_embeddings(hs, sent_ids, par_ids)
        if z is None:
            return hs.sum() * 0.0
        # Need at least two sentences AND at least two distinct paragraph ids so
        # that both positives (same par) and negatives (different par) can exist.
        if z.shape[0] < 2 or int(labels.unique().numel()) < 2:
            return hs.sum() * 0.0
        feats = z.unsqueeze(1)                                     # [n_sent,1,proj]
        return self._supcon(feats, labels=labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                sent_ids=None, par_ids=None, **kw):
        need_hidden = self.para_contrastive_weight > 0 and self.training \
            and sent_ids is not None and par_ids is not None
        hs = self.enc(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state
        logits = self.bound(self.drop(hs))
        loss = None
        if labels is not None:
            # ---- boundary CE (THE TASK), identical to contrastive_train.py ---
            loss = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(hs.device), ignore_index=-100)
            # ---- sentence paragraph-membership SupCon: only when weight>0 -----
            if need_hidden:
                # par_ids arrives as a padded [B, max_sent] tensor from the
                # collator (Trainer stacks per-example tensors). We flattened
                # global sentence ids so we rebuild the per-batch par_ids vector
                # here (see Collator.__call__ for the packing contract).
                loss = loss + self.para_contrastive_weight * self._para_supcon(
                    hs, sent_ids, par_ids)
        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# Dataset: windows carrying per-word sentence spans + global paragraph ids     #
# --------------------------------------------------------------------------- #
def make_examples(docs, window, stride):
    """Cut docs into overlapping windows. For each window precompute, at WORD
    granularity:
      * words       : window words with "\\n" mapped to [PAR];
      * word_labels : boundary labels, -100 for [PAR] (ignored by CE + pooling);
      * word_sent   : per-word LOCAL sentence index within this window
                      (0,1,2,...) or NO_SENT for a word that belongs to no
                      sentence ([PAR], or a truncated tail after the last
                      boundary of the window); and
      * word_par    : per-word GLOBAL paragraph id (unique per (doc, paragraph
                      index within doc)) or NO_SENT for [PAR]/tail words.

    Sentence rule: within a paragraph (the span between [PAR] tokens), a sentence
    is a maximal run of words ending on a label-1 word. The paragraph counter
    runs over the WHOLE document so a paragraph split across two windows keeps
    the same global id.
    """
    out = []
    for di, d in enumerate(docs):
        words_full = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                      for w in d["tokens"]]
        labs_full = [-100 if w == MODEL_PAR_TOKEN else l
                     for w, l in zip(words_full, d["labels"])]
        # Global paragraph id per WORD, over the whole doc. Paragraph index
        # increments AFTER each [PAR]; [PAR] words themselves get NO_SENT.
        # Pack (doc_index, par_index) into one int: di * 100000 + par_index.
        par_full = []
        par_idx = 0
        for w in words_full:
            if w == MODEL_PAR_TOKEN:
                par_full.append(NO_SENT)
                par_idx += 1
            else:
                par_full.append(di * 100000 + par_idx)

        n = len(words_full)
        s = 0
        while s < n:
            e = min(s + window, n)
            w_ = words_full[s:e]
            l_ = labs_full[s:e]
            gp_ = par_full[s:e]
            # Assign LOCAL sentence ids within this window. A sentence closes on
            # a label-1 word; [PAR] words break sentences and get NO_SENT. Words
            # after the last boundary in a window that never close (truncated
            # tail) get NO_SENT so they are not pooled into a partial sentence.
            word_sent = [NO_SENT] * len(w_)
            word_par = [NO_SENT] * len(w_)
            local_sid = 0
            cur = []                      # indices of the open sentence
            for i, (wi, li) in enumerate(zip(w_, l_)):
                if wi == MODEL_PAR_TOKEN:
                    # [PAR] never belongs to a sentence; drop any open (unclosed)
                    # run — a real sentence always ends on a boundary before [PAR].
                    cur = []
                    continue
                cur.append(i)
                if li == 1:               # boundary => close this sentence
                    for j in cur:
                        word_sent[j] = local_sid
                        word_par[j] = gp_[j]
                    local_sid += 1
                    cur = []
            out.append({"words": w_, "word_labels": l_,
                        "word_sent": word_sent, "word_par": word_par})
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, max_len):
    """Tokenize word-split windows and project word-level labels/sentence-ids/
    paragraph-ids onto SUBWORD positions.

      * label : placed on the LAST subword of each word (same grid as the CE
                boundary loss in contrastive_train.py); -100 elsewhere.
      * sent  : EVERY subword of a word inherits that word's LOCAL sentence id
                (or NO_SENT) so mean-pooling averages over all of a sentence's
                subword positions.
      * par   : EVERY subword inherits the word's global paragraph id (NO_SENT
                for non-sentence words).
    """
    enc = tok(examples["words"], is_split_into_words=True,
              truncation=True, max_length=max_len)
    all_lab, all_sent, all_par = [], [], []
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        ws = examples["word_sent"][i]
        wp = examples["word_par"][i]
        lab = [-100] * len(wid)
        sent = [NO_SENT] * len(wid)
        par = [NO_SENT] * len(wid)
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            sent[pos] = ws[w]             # subword inherits word's sentence id
            par[pos] = wp[w]              # subword inherits word's paragraph id
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:
                lab[pos] = wl[w]          # label on the LAST subword of the word
        all_lab.append(lab)
        all_sent.append(sent)
        all_par.append(par)
    enc["labels"] = all_lab
    enc["sent_local"] = all_sent          # LOCAL (per-window) sentence ids
    enc["par_local"] = all_par            # global paragraph ids per subword
    return enc


class Collator:
    """Pad labels + sentence/paragraph id columns, then renumber every window's
    LOCAL sentence ids into GLOBAL (per-batch) dense sentence ids 0..S-1 and
    build the aligned per-sentence paragraph-id vector the model consumes.

    Contract with ParaCLModel._sentence_embeddings:
      * batch["sent_ids"] : [B, T] long, a GLOBAL (batch-unique, dense) sentence
        index per subword, or NO_SENT. Same sentence within one window shares an
        id; sentences in DIFFERENT windows get DIFFERENT global ids (a sentence
        never straddles a window since windows are cut at word granularity).
      * batch["par_ids"]  : [n_global_sent] long, paragraph id per global
        sentence, indexable by the ids in sent_ids. Padded per Trainer stacking
        rules is avoided by making it a single 1-D tensor via a custom key that
        the Trainer passes straight through (remove_unused_columns=False).
    """

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        sents = [f.pop("sent_local") for f in feats]
        pars = [f.pop("par_local") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]

        batch["labels"] = torch.tensor(
            [l + [-100] * (m - len(l)) for l in labs], dtype=torch.long)

        # Renumber LOCAL per-window sentence ids -> GLOBAL dense ids 0..S-1, and
        # record each global sentence's paragraph id.
        global_sent = []                   # per-example [T] with global ids
        par_of_sent = []                   # paragraph id for each global sentence
        next_gid = 0
        for si, pi in zip(sents, pars):
            # local sentence id -> global sentence id, within THIS window
            local_to_global = {}
            row = [NO_SENT] * m
            for pos in range(len(si)):
                loc = si[pos]
                if loc == NO_SENT:
                    continue
                if loc not in local_to_global:
                    local_to_global[loc] = next_gid
                    par_of_sent.append(pi[pos])   # this subword's paragraph id
                    next_gid += 1
                row[pos] = local_to_global[loc]
            global_sent.append(row)

        batch["sent_ids"] = torch.tensor(global_sent, dtype=torch.long)
        # par_ids: 1-D [n_global_sent]; empty batches (no sentence) -> length 0.
        batch["par_ids"] = torch.tensor(par_of_sent, dtype=torch.long)
        return batch


def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1)
    mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


class ParaCLTrainer(Trainer):
    """Lets the model return its own loss (boundary CE + optional sentence
    SupCon). With para_contrastive_weight=0 in the model this is a plain
    class-weighted CE trainer — identical to the baseline path."""

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        loss = out["loss"]
        return (loss, out) if return_outputs else loss


def cmd_train(a):
    set_seed(a.seed)
    random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)     # TRAIN-ONLY
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)         # eval selection only
    pos = sum(sum(d["labels"]) for d in tr)
    tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  para_contrastive_weight={a.para_contrastive_weight}  "
          f"proj_dim={a.proj_dim}  temp={a.temp}")
    model = ParaCLModel(a.model_name, pw, a.para_contrastive_weight,
                        a.proj_dim, a.temp, n_extra_tok=1)
    model.enc.resize_token_embeddings(len(tok))
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride))
    dvds = Dataset.from_list(make_examples(dv, a.window, a.window))  # clean eval
    fn = lambda ex: encode(ex, tok, a.max_length)
    trds = trds.map(fn, batched=True, remove_columns=trds.column_names)
    dvds = dvds.map(fn, batched=True, remove_columns=dvds.column_names)
    targs = TrainingArguments(
        output_dir=os.path.join(a.out, "ckpts"), num_train_epochs=a.epochs,
        learning_rate=a.lr, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", warmup_ratio=0.1, weight_decay=0.01,
        fp16=torch.cuda.is_available(), logging_steps=20, save_total_limit=1,
        report_to="none", seed=a.seed, remove_unused_columns=False,
        label_names=["labels"])
    tr_ = ParaCLTrainer(model=model, args=targs, train_dataset=trds,
                        eval_dataset=dvds, data_collator=Collator(tok),
                        compute_metrics=metrics)
    tr_.train()
    print("best window dev:", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name, "proj_dim": a.proj_dim, "temp": a.temp},
              open(os.path.join(a.out, "cfg.json"), "w"))
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    # torch.cuda.is_available() can be True while device_count()==0 (e.g.
    # CUDA_VISIBLE_DEVICES=""); gate on an actually-usable device so CPU
    # prediction never tries to deserialize onto a phantom cuda:0.
    dev = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = ParaCLModel(cfg["model_name"], proj_dim=cfg.get("proj_dim", 128),
                        temp=cfg.get("temp", 0.1), n_extra_tok=1)
    model.enc.resize_token_embeddings(len(tok))
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"),
                                     map_location=dev))
    model.to(dev).eval()
    docs = load_task(a.task, a.split, jsonl_path=a.jsonl)
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        n = len(words)
        probs = {i: [] for i in range(n)}
        s = 0
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
        lab = [1 if (np.mean(probs[i]) if probs[i] else 0) >= a.threshold else 0
               for i in range(n)]
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:          # [PAR]/newline is never a boundary
                lab[i] = 0
        preds.append(lab)
    write_submission(docs, preds, a.out)
    print(f"wrote {len(docs)} -> {a.out}")


# --------------------------------------------------------------------------- #
# Self-test: PROOF this does what Noor asked (SENTENCE-level paragraph SupCon)  #
# --------------------------------------------------------------------------- #
def _cosine_report(model, batch, device):
    """On a real batch, mean-pool sentence embeddings and report
    (a) mean cosine of SAME-paragraph sentence pairs,
    (b) mean cosine of DIFFERENT-paragraph sentence pairs, (c) gap (a)-(b).
    Uses the UNIT-NORM projected sentence embeddings (the space the loss acts in).
    """
    model.eval()
    with torch.no_grad():
        hs = model.enc(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)
                       ).last_hidden_state
        z, labels = model._sentence_embeddings(hs, batch["sent_ids"].to(device),
                                               batch["par_ids"].to(device))
    if z is None or z.shape[0] < 2:
        return None
    sim = z @ z.t()                                  # [S, S] cosine (unit norm)
    S = z.shape[0]
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    iu = torch.triu(torch.ones(S, S, dtype=torch.bool, device=z.device), 1)
    same_pairs = same & iu
    diff_pairs = (~same) & iu
    a = float(sim[same_pairs].mean()) if int(same_pairs.sum()) > 0 else float("nan")
    b = float(sim[diff_pairs].mean()) if int(diff_pairs.sum()) > 0 else float("nan")
    return {"same": a, "diff": b, "gap": a - b,
            "n_sent": S, "n_same_pairs": int(same_pairs.sum()),
            "n_diff_pairs": int(diff_pairs.sum())}


def _encoder_grad_norm_from_contrastive(model, batch, device):
    """Backprop the SENTENCE-level SupCon loss ALONE (no boundary CE) and return
    the L2 grad-norm accumulated on the ENCODER parameters. >0 proves the
    contrastive signal trains the encoder, not just the projection head."""
    model.train()
    model.zero_grad(set_to_none=True)
    hs = model.enc(input_ids=batch["input_ids"].to(device),
                   attention_mask=batch["attention_mask"].to(device)
                   ).last_hidden_state
    sc = model._para_supcon(hs, batch["sent_ids"].to(device),
                            batch["par_ids"].to(device))
    sc.backward()
    total = 0.0
    for p in model.enc.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item()) ** 2
    model.zero_grad(set_to_none=True)
    return sc.item(), total ** 0.5


def cmd_selftest(a):
    """CPU/tiny end-to-end proof. Reads ONLY train (NoPnx-PA), builds one real
    batch, measures same/diff paragraph cosine gap BEFORE training, runs ~150
    optimizer steps of boundary_CE + 0.5 * sentence-paragraph-SupCon, then
    measures the gap AFTER. Prints the encoder grad-norm from the contrastive
    loss alone. Kept small so it finishes on CPU."""
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cpu"
    seed = a.seed
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})

    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    # Keep it tiny: take the first few docs, short window so a batch holds
    # sentences from SEVERAL paragraphs (=> both positives and negatives exist).
    tr = tr[:a.n_docs]
    window = a.window
    exs = make_examples(tr, window, window)          # non-overlapping windows
    ds = Dataset.from_list(exs)
    ds = ds.map(lambda ex: encode(ex, tok, a.max_length),
                batched=True, remove_columns=ds.column_names)
    coll = Collator(tok)
    # Take the first `batch_size` windows as the fixed evaluation/training batch.
    feats = [dict(ds[i]) for i in range(min(a.batch_size, len(ds)))]
    eval_batch = coll([dict(f) for f in feats])       # collator pops columns

    pos = sum(sum(d["labels"]) for d in tr)
    tot = sum(len(d["labels"]) for d in tr)
    pw = min((tot - pos) / max(pos, 1), 8.0)

    model = ParaCLModel(a.model_name, pos_weight=pw,
                        para_contrastive_weight=0.5, proj_dim=a.proj_dim,
                        temp=a.temp, n_extra_tok=1)
    model.enc.resize_token_embeddings(len(tok))
    model.to(device)

    print("=" * 72)
    print("SENTENCE-LEVEL PARAGRAPH-MEMBERSHIP SupCon — SELF-TEST (CPU)")
    print("=" * 72)
    print(f"model={a.model_name}  task={a.task}  docs={len(tr)}  window={window}")
    print(f"batch: {len(feats)} windows  |  para_contrastive_weight=0.5  "
          f"proj_dim={a.proj_dim}  temp={a.temp}")

    before = _cosine_report(model, eval_batch, device)
    if before is None or before["n_same_pairs"] == 0 or before["n_diff_pairs"] == 0:
        print("!! batch lacks both same- and different-paragraph pairs; "
              "increase --batch-size or --n-docs.")
        print("   report:", before)
        return
    print(f"\nbatch has {before['n_sent']} sentences, "
          f"{before['n_same_pairs']} same-paragraph pairs, "
          f"{before['n_diff_pairs']} different-paragraph pairs")

    # Encoder grad-norm from the CONTRASTIVE loss ALONE (proves it trains enc).
    sc_val, enc_gn = _encoder_grad_norm_from_contrastive(model, eval_batch, device)
    print(f"\n[encoder grad-norm from SupCon-alone]  supcon={sc_val:.4f}  "
          f"encoder_grad_norm={enc_gn:.4e}   (must be > 0)")

    print("\nBEFORE training:")
    print(f"  (a) same-paragraph mean cosine = {before['same']:+.4f}")
    print(f"  (b) diff-paragraph mean cosine = {before['diff']:+.4f}")
    print(f"  (c) gap (a)-(b)                = {before['gap']:+.4f}")

    # ---- ~150 optimizer steps of boundary_CE + 0.5 * sentence SupCon --------
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    model.train()
    losses = []
    for step in range(a.steps):
        out = model(input_ids=eval_batch["input_ids"].to(device),
                    attention_mask=eval_batch["attention_mask"].to(device),
                    labels=eval_batch["labels"].to(device),
                    sent_ids=eval_batch["sent_ids"].to(device),
                    par_ids=eval_batch["par_ids"].to(device))
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 25 == 0:
            print(f"  step {step + 1:3d}  loss={loss.item():.4f}")

    after = _cosine_report(model, eval_batch, device)
    print("\nAFTER training:")
    print(f"  (a) same-paragraph mean cosine = {after['same']:+.4f}")
    print(f"  (b) diff-paragraph mean cosine = {after['diff']:+.4f}")
    print(f"  (c) gap (a)-(b)                = {after['gap']:+.4f}")

    d_same = after["same"] - before["same"]
    d_diff = after["diff"] - before["diff"]
    d_gap = after["gap"] - before["gap"]
    print("\nDELTAS (after - before):")
    print(f"  same-paragraph cosine  {d_same:+.4f}   "
          f"(want INCREASE, i.e. > 0)")
    print(f"  diff-paragraph cosine  {d_diff:+.4f}   "
          f"(want DECREASE, i.e. < 0)")
    print(f"  gap                    {d_gap:+.4f}   (want WIDER, i.e. > 0)")

    ok_same = d_same > 0
    ok_diff = d_diff < 0
    ok_gap = d_gap > 0
    ok_enc = enc_gn > 0
    verdict = ok_same and ok_diff and ok_gap and ok_enc
    print("\nSUCCESS CRITERIA:")
    print(f"  same-paragraph cosine INCREASED : {ok_same}")
    print(f"  diff-paragraph cosine DECREASED : {ok_diff}")
    print(f"  gap WIDENED                     : {ok_gap}")
    print(f"  encoder grad-norm > 0 (SupCon)  : {ok_enc}")
    print(f"\n{'PASS' if verdict else 'FAIL'}: sentence-level paragraph SupCon "
          f"{'does' if verdict else 'does NOT'} pull same-paragraph sentences "
          f"together and push different-paragraph sentences apart, and trains "
          f"the encoder.")
    return verdict


# --------------------------------------------------------------------------- #
# CLI (mirrors contrastive_train.py)                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--task", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl")
    t.add_argument("--dev-jsonl")
    t.add_argument("--para-contrastive-weight", type=float, default=0.0,
                   help="sentence-level paragraph-membership SupCon weight "
                        "(0=off => plain boundary-CE baseline)")
    t.add_argument("--proj-dim", type=int, default=128,
                   help="sentence projection-head output dim")
    t.add_argument("--temp", type=float, default=0.1,
                   help="SupCon temperature")
    t.add_argument("--window", type=int, default=180)
    t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--pos-weight", type=float, default=None)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True)
    p.add_argument("--jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)

    st = sub.add_parser("selftest",
                        help="CPU/tiny proof: before/after same-vs-diff "
                             "paragraph cosine gap + encoder grad-norm")
    st.add_argument("--task", default="NoPnx-PA")
    st.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    st.add_argument("--train-jsonl", default=os.path.join(
        HERE, os.pardir, "data", "NoPnx-PA_train.jsonl"))
    st.add_argument("--n-docs", type=int, default=6)
    st.add_argument("--window", type=int, default=120)
    st.add_argument("--batch-size", type=int, default=6)
    st.add_argument("--steps", type=int, default=150)
    st.add_argument("--proj-dim", type=int, default=128)
    st.add_argument("--temp", type=float, default=0.1)
    st.add_argument("--max-length", type=int, default=256)
    st.add_argument("--lr", type=float, default=5e-5)
    st.add_argument("--seed", type=int, default=42)
    st.set_defaults(func=cmd_selftest)

    a = ap.parse_args()
    a.func(a)
