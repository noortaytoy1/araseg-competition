"""Cross-view privileged distillation student trainer for AraSeg NoPnx-NP.

SPEC of record: the GO-WITH-CHANGES build order + SPEC v1 2026-07-10
(cross-view distillation, PA teacher -> NoPnx-NP student). Method context:
docs/RESEARCH_PROMPT_xview_distill_2026-07-10.md.

Self-contained -- does NOT touch the shared train_encoder/predict/eval/stitching
pipeline. The data / window / label / [PAR] handling MIRRORS
src/consistency_train.py EXACTLY (window 180, stride 90, labels on the LAST
subword of each word, MODEL_PAR_TOKEN for the "\n" paragraph token, weighted CE
with pos_weight capped ~8, epochs 8 / lr 5e-5 / bs 16 / warmup 0.1 / fp16 /
load_best f1, model.pt + cfg.json + tokenizer + HF-checkpoint save, predict
subcommand verbatim-compatible with eval_local.py). The two-head architecture
and the HF-save mirror src/teacher_train.py (TeacherModel + _save_hf_checkpoint).

THE IDEA (privileged information / generalized distillation, Lopez-Paz et al.
ICLR 2016): a PA teacher (sees punctuation, ~94.4 F1) was run OFFLINE over the
SAME 174 closed-track train docs, its per-token boundary probabilities folded
onto the NoPnx-NP token grid (backward fold onto the nearest preceding kept
token -- the fold reproduces the official NoPnx-NP gold bit-exactly, verified
174/174), and cached by src/xview_teacher_cache.py. This trainer NEVER loads
the teacher model; it only reads that immutable cache. A second model-free
cache carries "what punctuation was stripped right after this token" classes
for an auxiliary punctuation-prediction head (the WtP/SaT hedge).

CLOSED-TRACK / TRAIN-ONLY: the caches exist for the TRAIN split only (the
cache builder refuses non-_train jsonls -- guard g1 -- and refuses *_trdev_*
teachers -- guard g0; this trainer re-checks g0 from the cache meta). Dev is
used solely by the existing eval for checkpoint selection; dev NEVER passes
through the teacher, and the dev dataset here carries NO teacher columns.

TWO FLAG-GATED methods on top of the plain weighted-CE baseline
(ALL defaults 0/off = single forward, no cache I/O, bit-for-bit plain
weighted-CE -- same discipline as consistency_train.py / teacher_train.py):

METHOD KD -- soft-target distillation  (--kd-weight, default 0):
    Per valid label position (last subword of a real word; [PAR]/pad excluded),
    the folded teacher probability q is turned back into a logit
    z_t = clamp(log(q/(1-q)), -8, +8), temperature-softened qT = sigmoid(z_t/T),
    and the student's boundary logit-difference z_s = logit1 - logit0 is trained
    with the STANDARD Hinton direction, forward KL(teacher || student):
        kd_i = qT*log(qT/pT) + (1-qT)*log((1-qT)/(1-pT)),  pT = sigmoid(z_s/T)
        kd   = T^2 * mean(kd_i over selected positions)
        loss = weightedCE + kd_weight * kd
    NAMED DECISION (spec section 3): the distillation direction is the standard
    KL(teacher || student); reverse KL is deliberately not offered.
    Selection (mutually exclusive, spec section 3 / pitfall (b)):
      --kd-gate G          : selected = valid & (teacher_p >= G). One-sided per
                             the spec of record ("certain-zeros excluded");
                             G=0 (default) = all valid positions. Bare
                             `--kd-gate` (no value) means the pre-registered 0.05.
      --kd-entropy-weight 1: instead of gating, weight kd_i by the teacher's
                             binary entropy H2(q)/log2 (in [0,1], max at q=0.5),
                             normalized over valid positions.

METHOD AUX -- punctuation-prediction auxiliary head  (--punct-aux-weight, 0):
    punct = Linear(h, C) on the SAME encoder hidden states (one encoder pass
    total), unweighted CE against the punct-after classes from --punct-cache
    (0=none / 1=strong / 2=comma / 3=other; with --punct-aux-classes 2 the
    classes are binarized to was-punct-stripped). The punct head is ALWAYS
    constructed (init order: enc, bound, punct) so arms share the enc/bound
    init stream, is NEVER exercised at weight 0, and is DISCARDED at inference
    (not copied into the HF checkpoint).

BIT-FOR-BIT DISCIPLINE: with --kd-weight 0 and --punct-aux-weight 0 there is
ONE forward through enc+bound, no cache is opened, no extra step-time RNG; the
loss is exactly the plain weighted CE (torch.equal in the smokes). The punct
head's parameter INIT does consume RNG at construction (like teacher_train's
aux head), so this trainer's all-off arm is its OWN baseline of record in the
battery -- not bit-identical to cons_baseline (same convention as
semicrf/sam/dro).

Cache validation at load (hard aborts): meta.json present next to the cache,
meta["student_task"] == --task, teacher basename matches *_base_* and is not
*_trdev_* (guard g0 re-check), doc_id set == train jsonl doc_id set, per-doc
n_tokens == len(doc["tokens"]). Cache paths + meta sha256 are recorded in the
run's cfg.json (provenance).

Usage:
  # baseline (all-off; bit-for-bit plain weighted CE, no cache I/O):
  python src/xview_distill_train.py train --task NoPnx-NP --out runs/xv_base \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # KD, T=2, all valid positions:
  python src/xview_distill_train.py train --task NoPnx-NP --kd-weight 1.0 \
      --teacher-cache runs/xview_teacher_2026-07-10/teacher_pa_fold_NoPnx-NP_train.jsonl \
      --out runs/xv_kd --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # punctuation-prediction aux head:
  python src/xview_distill_train.py train --task NoPnx-NP --punct-aux-weight 0.5 \
      --punct-aux-classes 4 \
      --punct-cache runs/xview_teacher_2026-07-10/punct_after_NoPnx-NP_train.jsonl \
      --out runs/xv_aux --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  python src/xview_distill_train.py predict --task NoPnx-NP --split dev \
      --model runs/xv_kd --jsonl data/NoPnx-NP_dev.jsonl --out subs/xv_kd_dev.csv
"""
import argparse
import hashlib
import json
import math
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

# Sentinels for the two per-token side channels (mirroring labels' -100):
TEACHER_PAD = -1.0   # teacher_p at [PAR] / non-last-subword / pad -> excluded
PUNCT_PAD = -100     # punct_cls at [PAR] / non-last-subword / pad -> ignore_index


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py exactly; the two       #
# optional per-word parallel lists ride the SAME window/stride loop)           #
# --------------------------------------------------------------------------- #
def make_examples(docs, window, stride, teacher_map=None, punct_map=None):
    """Slide a (window, stride) window over each doc; [PAR] positions -> -100.

    teacher_map / punct_map: optional {doc_id: per-token list} from the caches
    (doc-level, already on the NoPnx grid). They are windowed IDENTICALLY to
    words/word_labels, so teacher targets and gold can never be windowed
    differently (pitfall (c)). [PAR] positions get the sentinels."""
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        tp = pc = None
        if teacher_map is not None:
            tp = [TEACHER_PAD if w == MODEL_PAR_TOKEN else float(p)
                  for w, p in zip(words, teacher_map[d["doc_id"]])]
        if punct_map is not None:
            pc = [PUNCT_PAD if w == MODEL_PAR_TOKEN else int(c)
                  for w, c in zip(words, punct_map[d["doc_id"]])]
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            ex = {"words": words[s:e], "word_labels": labs[s:e]}
            if tp is not None:
                ex["teacher_p"] = tp[s:e]
            if pc is not None:
                ex["punct_cls"] = pc[s:e]
            out.append(ex)
            if e == n:
                break
            s += stride
    return out


def encode(examples, tok, max_len):
    """Sub-word encode; put each word's label on its LAST subword only.
    teacher_p / punct_cls land on EXACTLY the same subword positions as the
    labels (elsewhere: the sentinels), mirroring the labels' placement."""
    enc = tok(examples["words"], is_split_into_words=True, truncation=True, max_length=max_len)
    has_tp = "teacher_p" in examples
    has_pc = "punct_cls" in examples
    all_lab, all_tp, all_pc = [], [], []
    for i, wl in enumerate(examples["word_labels"]):
        wid = enc.word_ids(batch_index=i)
        lab = [-100] * len(wid)
        tp = [TEACHER_PAD] * len(wid) if has_tp else None
        pc = [PUNCT_PAD] * len(wid) if has_pc else None
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:  # last subword of word w
                lab[pos] = wl[w]
                if has_tp:
                    tp[pos] = examples["teacher_p"][i][w]
                if has_pc:
                    pc[pos] = examples["punct_cls"][i][w]
        all_lab.append(lab)
        if has_tp:
            all_tp.append(tp)
        if has_pc:
            all_pc.append(pc)
    enc["labels"] = all_lab
    if has_tp:
        enc["teacher_p"] = all_tp
    if has_pc:
        enc["punct_cls"] = all_pc
    return enc


class Collator:
    """Pads labels with -100 (mirroring consistency_train.Collator) and, when
    present, teacher_p with -1.0 and punct_cls with -100. The dev dataset never
    carries the side channels (dev never passes through the teacher), so eval
    batches simply lack those keys."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        tps = [f.pop("teacher_p") for f in feats] if "teacher_p" in feats[0] else None
        pcs = [f.pop("punct_cls") for f in feats] if "punct_cls" in feats[0] else None
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        if tps is not None:
            batch["teacher_p"] = torch.tensor(
                [t + [TEACHER_PAD] * (m - len(t)) for t in tps], dtype=torch.float32)
        if pcs is not None:
            batch["punct_cls"] = torch.tensor(
                [p + [PUNCT_PAD] * (m - len(p)) for p in pcs], dtype=torch.long)
        return batch


# --------------------------------------------------------------------------- #
# model: encoder + boundary head + (flag-gated) KD term + (flag-gated) punct   #
# aux head. Mirrors teacher_train.TeacherModel's two-head architecture.        #
# --------------------------------------------------------------------------- #
class XViewModel(nn.Module):
    """Boundary tagger with a flag-gated KD term and a flag-gated punct head.

    * bound : the real per-token boundary head (2-way). Always used.
    * punct : punctuation-prediction aux head (C-way). ALWAYS constructed
              (init order: enc, bound, punct -- so the enc/bound init stream is
              identical across arms), NEVER exercised at weight 0, and
              DISCARDED at inference (not copied to the HF checkpoint).

    With kd_weight == 0 and punct_aux_weight == 0, forward() runs ONE encoder
    pass + bound head + weighted CE -- bit-for-bit the plain weighted-CE
    baseline. No cache tensor is read, no second dropout call, no extra RNG.
    """

    def __init__(self, model_name, pos_weight=1.0, kd_weight=0.0, kd_temp=2.0,
                 kd_gate=0.0, kd_entropy_weight=0, punct_aux_weight=0.0,
                 punct_aux_classes=2, n_extra_tok=1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.bound = nn.Linear(h, 2)
        # punct aux head -- constructed LAST so enc/bound init draws are
        # identical whatever C is; discarded at inference.
        self.punct = nn.Linear(h, punct_aux_classes)
        self.kd_weight = float(kd_weight)
        self.kd_temp = float(kd_temp)
        self.kd_gate = float(kd_gate)
        self.kd_entropy_weight = int(kd_entropy_weight)
        self.punct_aux_weight = float(punct_aux_weight)
        self.punct_aux_classes = int(punct_aux_classes)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                teacher_p=None, punct_cls=None, **kw):
        hs = self.enc(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state
        logits = self.bound(self.drop(hs))                       # [B, L, 2]
        loss = None
        if labels is not None:
            ce = nn.functional.cross_entropy(
                logits.view(-1, 2).float(), labels.view(-1),
                weight=self.w.to(hs.device), ignore_index=-100)
            loss = ce

            # ---- METHOD KD: soft-target distillation (only when kd_weight>0)
            # Gated OFF at weight 0: teacher_p is never touched, loss == CE.
            if self.kd_weight > 0 and teacher_p is not None:
                labels_f = labels.view(-1)
                tp = teacher_p.view(-1).float()
                valid = (labels_f != -100) & (tp >= 0)           # sentinel -1.0 out
                T = self.kd_temp
                # teacher prob -> 2-class logit, clamped to +/-8 (spec: the
                # clamp both prevents inf at q in {0,1} and caps the target
                # sharpness of the in-sample teacher).
                q = tp.clamp(1e-6, 1.0 - 1e-6)
                z_t = torch.log(q / (1.0 - q)).clamp(-8.0, 8.0)
                z_s = (logits[..., 1] - logits[..., 0]).view(-1).float()
                qT = torch.sigmoid(z_t / T)
                log_qT = nn.functional.logsigmoid(z_t / T)
                log_1mqT = nn.functional.logsigmoid(-z_t / T)
                log_pT = nn.functional.logsigmoid(z_s / T)
                log_1mpT = nn.functional.logsigmoid(-z_s / T)
                # forward KL(teacher || student), teacher as fixed target
                kd_i = qT * (log_qT - log_pT) + (1.0 - qT) * (log_1mqT - log_1mpT)
                if self.kd_entropy_weight:
                    # weight by teacher binary entropy H2(q)/log2 in [0,1]
                    h2 = -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q)) / math.log(2.0)
                    wv = h2 * valid.float()
                    kd = (T * T) * (kd_i * wv).sum() / wv.sum().clamp(min=1e-9)
                else:
                    sel = valid & (tp >= self.kd_gate)           # gate 0 -> sel = valid
                    if sel.any():
                        kd = (T * T) * kd_i[sel].mean()
                    else:
                        kd = (z_s * 0.0).sum()                   # keeps graph, term 0
                loss = loss + self.kd_weight * kd

            # ---- METHOD AUX: punct-prediction head (only when weight>0) ----
            # SAME hidden states -> one encoder pass total; unweighted CE.
            if self.punct_aux_weight > 0 and punct_cls is not None:
                aux_logits = self.punct(self.drop(hs))           # fresh dropout mask
                aux = nn.functional.cross_entropy(
                    aux_logits.view(-1, self.punct_aux_classes).float(),
                    punct_cls.view(-1), ignore_index=PUNCT_PAD)
                loss = loss + self.punct_aux_weight * aux

        return {"loss": loss, "logits": logits}   # logits = main head only


# --------------------------------------------------------------------------- #
# HF-checkpoint save (clone of teacher_train._save_hf_checkpoint)              #
# --------------------------------------------------------------------------- #
def _save_hf_checkpoint(model, tok, out_dir, model_name, n_extra_tok=1):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / fair_threshold.py / score_arm.py / diag_arm.py load
    with AutoModelForTokenClassification.from_pretrained(out_dir) +
    AutoTokenizer, which the custom .pt cannot satisfy. The inference model
    here is ONLY the main boundary path: the bare encoder `model.enc` followed
    by the `model.bound` Linear(hidden, 2) head. The punct aux head
    (`model.punct`) is discarded at inference, so it is NOT copied.

    We assemble a standard AutoModelForTokenClassification(model_name,
    num_labels=2), resize its embeddings to match the trainer (vocab +
    n_extra_tok for [PAR]), then load the TRAINED encoder body into `.bert`
    (strict=False tolerates the encoder's unused pooler) and copy the TRAINED
    `bound` weights into `.classifier`. The result reproduces the trainer's
    inference logits exactly. This does not touch training, the loss, the
    flags, or the existing .pt/cfg.json outputs."""
    hf = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    hf.resize_token_embeddings(hf.config.vocab_size + n_extra_tok)
    hf.bert.load_state_dict(model.enc.state_dict(), strict=False)
    with torch.no_grad():
        hf.classifier.weight.copy_(model.bound.weight)
        hf.classifier.bias.copy_(model.bound.bias)
    hf.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


# --------------------------------------------------------------------------- #
# cache loading + validation (hard aborts; spec section 3)                     #
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cache_meta(cache_path, task):
    """meta.json must sit next to the cache jsonl; task must match; the teacher
    that built the cache must be a *_base_* (train-only) checkpoint -- re-check
    of guard g0 (a *_trdev_* teacher would launder dev-training into the
    student through the teacher's function; rule-illegal)."""
    d = os.path.dirname(os.path.abspath(cache_path))
    mp = os.path.join(d, "meta.json")
    if not os.path.isfile(mp):
        raise SystemExit(f"CACHE VALIDATION FAILED: no meta.json next to {cache_path}")
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("student_task") != task:
        raise SystemExit(f"CACHE VALIDATION FAILED: meta student_task="
                         f"{meta.get('student_task')!r} != --task {task!r}")
    tb = os.path.basename(str(meta.get("teacher", "")).rstrip("/\\"))
    if "_trdev_" in tb:
        raise SystemExit("CACHE VALIDATION FAILED (guard g0): cache was built from a "
                         f"*_trdev_* (dev-trained) teacher {tb!r} -- rule-illegal, refusing")
    if "_base_" not in tb:
        raise SystemExit(f"CACHE VALIDATION FAILED (guard g0): teacher basename {tb!r} "
                         "does not match *_base_*")
    return meta, mp


def _load_cache_jsonl(path, key, docs, what):
    """doc_id sets must be identical and per-doc n_tokens must match the train
    jsonl exactly; any mismatch is a hard abort (spec section 3)."""
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r["doc_id"]] = r
    train_ids = {d["doc_id"] for d in docs}
    if set(by_id) != train_ids:
        raise SystemExit(f"CACHE VALIDATION FAILED ({what}): doc_id sets differ "
                         f"(cache-only={len(set(by_id) - train_ids)}, "
                         f"train-only={len(train_ids - set(by_id))})")
    out = {}
    for d in docs:
        r = by_id[d["doc_id"]]
        vals = r[key]
        if r.get("n_tokens") != len(d["tokens"]) or len(vals) != len(d["tokens"]):
            raise SystemExit(f"CACHE VALIDATION FAILED ({what}): n_tokens mismatch for "
                             f"doc {d['doc_id']}: cache n_tokens={r.get('n_tokens')} "
                             f"len={len(vals)} vs train jsonl {len(d['tokens'])}")
        if what == "teacher" and any((v < 0.0 or v > 1.0) for v in vals):
            raise SystemExit(f"CACHE VALIDATION FAILED (teacher): prob outside [0,1] "
                             f"in doc {d['doc_id']}")
        if what == "punct" and any(c not in (0, 1, 2, 3) for c in vals):
            raise SystemExit(f"CACHE VALIDATION FAILED (punct): class outside 0..3 "
                             f"in doc {d['doc_id']}")
        out[d["doc_id"]] = vals
    return out


# --------------------------------------------------------------------------- #
# metrics / train / predict (mirror consistency_train.py)                     #
# --------------------------------------------------------------------------- #
def metrics(ep):
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)   # TRAIN-ONLY caches
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)       # eval selection only
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  kd_weight={a.kd_weight}  kd_temp={a.kd_temp}  "
          f"kd_gate={a.kd_gate}  kd_entropy_weight={a.kd_entropy_weight}  "
          f"punct_aux_weight={a.punct_aux_weight}  punct_aux_classes={a.punct_aux_classes}")

    # Caches are opened ONLY when their method is on (spec: "else ignored AND
    # not opened"). Validation hard-aborts before any training step.
    teacher_map = punct_map = None
    cache_meta = None; cache_meta_sha = None
    if a.kd_weight > 0:
        cache_meta, meta_path = _load_cache_meta(a.teacher_cache, a.task)
        cache_meta_sha = _sha256(meta_path)
        teacher_map = _load_cache_jsonl(a.teacher_cache, "teacher_probs", tr, "teacher")
        print(f"teacher cache OK: {a.teacher_cache}  docs={len(teacher_map)}  "
              f"meta_sha256={cache_meta_sha[:16]}...")
    if a.punct_aux_weight > 0:
        p_meta, p_meta_path = _load_cache_meta(a.punct_cache, a.task)
        p_sha = _sha256(p_meta_path)
        if cache_meta_sha is None:
            cache_meta, cache_meta_sha = p_meta, p_sha
        elif p_sha != cache_meta_sha:
            raise SystemExit("CACHE VALIDATION FAILED: --teacher-cache and "
                             "--punct-cache come from different cache builds "
                             "(meta.json sha256 mismatch)")
        punct_map = _load_cache_jsonl(a.punct_cache, "punct_after", tr, "punct")
        if a.punct_aux_classes == 2:   # binarize: was ANY punct stripped after?
            punct_map = {k: [0 if c == 0 else 1 for c in v] for k, v in punct_map.items()}
        print(f"punct cache OK: {a.punct_cache}  docs={len(punct_map)}  "
              f"classes={a.punct_aux_classes}")

    model = XViewModel(a.model_name, pw, a.kd_weight, a.kd_temp, a.kd_gate,
                       a.kd_entropy_weight, a.punct_aux_weight,
                       a.punct_aux_classes, n_extra_tok=1)
    trds = Dataset.from_list(make_examples(tr, a.window, a.stride, teacher_map, punct_map))
    dvds = Dataset.from_list(make_examples(dv, a.window, a.window))  # clean eval, NO teacher cols
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
    cfg = {"model_name": a.model_name, "kd_weight": a.kd_weight, "kd_temp": a.kd_temp,
           "kd_gate": a.kd_gate, "kd_entropy_weight": a.kd_entropy_weight,
           "punct_aux_weight": a.punct_aux_weight, "punct_aux_classes": a.punct_aux_classes,
           "teacher_cache": a.teacher_cache if a.kd_weight > 0 else None,
           "punct_cache": a.punct_cache if a.punct_aux_weight > 0 else None,
           "cache_meta_sha256": cache_meta_sha, "cache_meta": cache_meta}
    with open(os.path.join(a.out, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # ADDITIVE: also emit an HF checkpoint (main boundary path only; punct aux
    # head discarded) so predict.py / score_arm.py / fair_threshold.py /
    # diag_arm.py can load this run dir. Nothing above is changed.
    _save_hf_checkpoint(model, tok, a.out, a.model_name, n_extra_tok=1)
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    # Gate on an actually-usable device so CPU prediction never deserializes
    # onto a phantom cuda:0 (mirrors teacher_train.cmd_predict).
    dev = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json"), encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = XViewModel(cfg["model_name"],
                       punct_aux_classes=cfg.get("punct_aux_classes", 2), n_extra_tok=1)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--kd-weight", type=float, default=0.0,
                   help="METHOD KD soft-target distillation weight (0 = off)")
    t.add_argument("--kd-temp", type=float, default=2.0,
                   help="KD temperature T (T^2-scaled forward KL)")
    t.add_argument("--kd-gate", type=float, nargs="?", const=0.05, default=0.0,
                   help="distill only where folded teacher prob >= gate "
                        "(one-sided per spec: certain-zeros excluded; 0 = all "
                        "valid positions; bare --kd-gate means 0.05)")
    t.add_argument("--kd-entropy-weight", type=int, default=0, choices=[0, 1],
                   help="weight KD by teacher binary entropy instead of gating "
                        "(mutually exclusive with --kd-gate)")
    t.add_argument("--teacher-cache", default=None,
                   help="teacher_pa_fold_*.jsonl from xview_teacher_cache.py; "
                        "REQUIRED iff --kd-weight > 0 (else ignored AND not opened)")
    t.add_argument("--punct-aux-weight", type=float, default=0.0,
                   help="METHOD AUX punct-prediction head CE weight (0 = off)")
    t.add_argument("--punct-aux-classes", type=int, default=2, choices=[2, 4],
                   help="2 = binary was-punct-stripped; 4 = none/strong/comma/other")
    t.add_argument("--punct-cache", default=None,
                   help="punct_after_*.jsonl from xview_teacher_cache.py; "
                        "REQUIRED iff --punct-aux-weight > 0")
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
    a = ap.parse_args()
    if a.cmd == "train":
        # spec section 3: mutual exclusion + required-iff, enforced at argparse.
        if a.kd_entropy_weight == 1 and a.kd_gate > 0:
            ap.error("--kd-entropy-weight and --kd-gate are mutually exclusive")
        if a.kd_weight > 0 and not a.teacher_cache:
            ap.error("--teacher-cache is REQUIRED when --kd-weight > 0")
        if a.punct_aux_weight > 0 and not a.punct_cache:
            ap.error("--punct-cache is REQUIRED when --punct-aux-weight > 0")
    a.func(a)
