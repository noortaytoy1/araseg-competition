"""Character-level Arabic augmentation for sentence segmentation (Şahin 2021).
Self-contained — does NOT touch the shared train_encoder/predict/eval pipeline.

WHY THIS LEVER
--------------
The closed-track wall for AraSeg is DATA: the model fails on tokens (and bigrams)
it never saw at train time. Character-level augmentation is the best-evidenced
low-resource sequence-labelling lever (Şahin, "To Augment or Not to Augment?",
CL 2021): it perturbs the SURFACE FORM of words so the encoder sees many more
spellings of the same underlying token, widening coverage of the unseen-token
tail. It is CLOSED-LEGAL — it touches ONLY the 174 training documents' own
strings, adds no external data and no external model — and it is LABEL-PRESERVING
by construction: every perturbation stays WITHIN a single word, so no token is
split, merged, added, or dropped. Token count and the 0/1 boundary labels are
therefore identical before and after; the label sequence stays perfectly aligned.

TWO FAMILIES OF PERTURBATION (both operate on ONE word string at a time)
------------------------------------------------------------------------
(a) generic character noise at a small rate — one of:
      substitute a random Arabic letter, delete a char, insert a random Arabic
      letter, or swap two adjacent chars.
(b) Arabic-AWARE orthographic variants that mimic REAL spelling variation, so
    the added forms are ones the model will actually meet in the wild:
      * alef forms            ا  <->  أ إ آ
      * hamza carriers        ء  <->  ئ ؤ  (and bare/​seated hamza)
      * taa marbuta vs haa    ة  <->  ه
      * yaa vs alef-maqsura   ي  <->  ى
      * add / remove tashkeel (harakat / diacritics)
    These specifically expand coverage of the orthographic variation that drives
    the unseen-token errors.

The augmentation is ON-THE-FLY and FLAG-GATED:
      --char-aug-rate  (default 0.0 = OFF = the untouched baseline, bit-for-bit)
Each word is independently perturbed with probability `char_aug_rate`; the whole
training set is re-perturbed at the start of EVERY epoch (fresh randomness per
epoch), so there is NO augmented data file on disk and NO race — the corpus never
leaves memory.

The rest of the pipeline MIRRORS src/coherence_train.py exactly: AraBERTv02 token
classifier, window 180 / stride 90, epochs 8, batch 16, lr 5e-5, weighted CE,
labels placed on the LAST subword of each word. The coherence auxiliary head is
dropped (boundary head only); nothing here alters the shared modules.

Usage:
  python char_aug.py train   --task NoPnx-NP --char-aug-rate 0.10 --out runs/charaug
  python char_aug.py predict  --task NoPnx-NP --split dev --model runs/charaug --out subs/x.csv
  python char_aug.py selftest                         # label-preservation + rate-0 identity (no GPU)
  python char_aug.py selftest --gpu-smoke --task NoPnx-NP   # + a few real train steps at rate>0
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
                          AutoTokenizer, Trainer, TrainingArguments,
                          TrainerCallback, set_seed)

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Character-level Arabic augmentation
# ---------------------------------------------------------------------------

# Arabic letters used as the pool for generic substitute / insert.
ARABIC_LETTERS = list(
    "ابتثجحخدذرزسشصضطظعغفقكلمنهويءةى"
    "أإآؤئ"
)
# Arabic combining diacritics (tashkeel / harakat) — used for add/remove.
ARABIC_DIACRITICS = list(
    "ًٌٍَُِّْٰٕٓٔ"
)
_DIAC_SET = set(ARABIC_DIACRITICS)

# Arabic-aware orthographic variant groups. Each key may be rewritten to any
# OTHER member of its group. All members are single characters, so the word
# length is preserved for substitution-style variants; add/remove diacritics are
# handled separately (they change length but never the token count / labels).
_ALEF_FORMS = "اأإآ"
_HAMZA_CARRIERS = "ءئؤأإ"
_ORTHO_GROUPS = [
    _ALEF_FORMS,        # alef forms
    _HAMZA_CARRIERS,    # hamza carriers / seats
    "ةه",              # taa marbuta vs haa
    "يى",              # yaa vs alef-maqsura
]
# Map each char -> the string of its alternative variants (same group, minus itself).
_ORTHO_ALT = {}
for _grp in _ORTHO_GROUPS:
    for _c in _grp:
        _ORTHO_ALT.setdefault(_c, "")
        _ORTHO_ALT[_c] += "".join(ch for ch in _grp if ch != _c)


def _strip_diacritics(word):
    return "".join(ch for ch in word if ch not in _DIAC_SET)


def perturb_word(word, rate, rng):
    """Return a possibly-perturbed copy of a SINGLE word string.

    With probability `rate` the word is perturbed by exactly one operation drawn
    uniformly from the two families (generic noise vs Arabic-aware orthography).
    The perturbation stays strictly within the word — it never introduces or
    removes whitespace — so the surrounding tokenisation, token count, and the
    word's boundary label are all unchanged. Empty or single-token markers (the
    "[PAR]" stand-in has already been mapped upstream) are returned untouched.
    """
    if rate <= 0.0 or not word:
        return word
    if rng.random() >= rate:
        return word

    ops = ["subst", "delete", "insert", "swap", "ortho", "diac"]
    op = rng.choice(ops)
    chars = list(word)

    if op == "ortho":
        # Arabic-aware: rewrite one char to an orthographic variant of itself.
        cand = [i for i, c in enumerate(chars) if c in _ORTHO_ALT and _ORTHO_ALT[c]]
        if cand:
            i = rng.choice(cand)
            chars[i] = rng.choice(list(_ORTHO_ALT[chars[i]]))
            return "".join(chars)
        # no orthographic-variant char present -> fall through to a generic op
        op = rng.choice(["subst", "delete", "insert", "swap"])

    if op == "diac":
        # Arabic-aware: add OR remove tashkeel. Removing changes nothing about
        # token identity; adding inserts a combining mark after a base letter.
        if any(c in _DIAC_SET for c in chars) and rng.random() < 0.5:
            return _strip_diacritics(word)  # remove all diacritics
        base_positions = [i for i, c in enumerate(chars) if c not in _DIAC_SET]
        if base_positions:
            i = rng.choice(base_positions)
            chars.insert(i + 1, rng.choice(ARABIC_DIACRITICS))
            return "".join(chars)
        return word

    if op == "subst":
        i = rng.randrange(len(chars))
        chars[i] = rng.choice(ARABIC_LETTERS)
        return "".join(chars)

    if op == "delete":
        if len(chars) <= 1:
            return word  # never delete a word to empty (would drop a token)
        i = rng.randrange(len(chars))
        del chars[i]
        return "".join(chars)

    if op == "insert":
        i = rng.randrange(len(chars) + 1)
        chars.insert(i, rng.choice(ARABIC_LETTERS))
        return "".join(chars)

    if op == "swap":
        if len(chars) < 2:
            return word
        i = rng.randrange(len(chars) - 1)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    return word


def augment_words(words, rate, rng):
    """Perturb each REAL word independently; leave the [PAR] stand-in untouched.

    Returns a NEW list of the same length (token count preserved). The caller's
    label list therefore stays exactly aligned.
    """
    out = []
    for w in words:
        if w == MODEL_PAR_TOKEN:
            out.append(w)                      # never perturb the paragraph marker
        else:
            out.append(perturb_word(w, rate, rng))
    return out


# ---------------------------------------------------------------------------
# Model (boundary head only — mirrors coherence_train.CohModel minus coherence)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Windowing (mirrors coherence_train.make_examples, minus the coherence shuffle)
# ---------------------------------------------------------------------------

def make_examples(docs, window, stride):
    """Slice each doc into overlapping windows of (words, word_labels).

    [PAR] tokens are label -100 (ignored), exactly as in coherence_train.
    We keep the ORIGINAL (un-augmented) words here; augmentation is applied
    on-the-fly per epoch (see EpochAugment) so nothing is baked in.
    """
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


class AugmentEncodeCollator:
    """On-the-fly: (optionally) perturb each window's words, then tokenise + pad.

    The perturbation happens HERE, at batch-materialisation time, so every epoch
    (indeed every batch draw) sees freshly perturbed surfaces and NO augmented
    file is ever written. With rate 0.0 the words pass through untouched, so the
    tokenisation is byte-for-byte identical to the un-augmented baseline.

    Determinism: a base seed + a monotonically increasing counter feed a fresh
    per-call RNG, so a given (seed) reproduces the same stream of perturbations
    while still varying across epochs.
    """

    def __init__(self, tok, max_len, char_aug_rate=0.0, seed=42):
        self.tok = tok
        self.max_len = max_len
        self.rate = char_aug_rate
        self.seed = seed
        self._counter = 0

    def __call__(self, feats):
        encoded = []
        for f in feats:
            words = f["words"]
            if self.rate > 0.0:
                rng = random.Random(self.seed * 1_000_003 + self._counter)
                self._counter += 1
                words = augment_words(words, self.rate, rng)
            encoded.append(encode_one(words, f["word_labels"], self.tok, self.max_len))
        labs = [e.pop("labels") for e in encoded]
        batch = self.tok.pad(encoded, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        return batch


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

    Purpose: predict.py / fair_threshold.py / diag_preposition.py load with
    AutoModelForTokenClassification.from_pretrained(out_dir) + AutoTokenizer, which
    the custom .pt cannot satisfy. The inference model here is the bare encoder
    `model.enc` followed by the `model.bound` Linear(hidden, 2) boundary head
    (there is no aux head). We assemble a standard
    AutoModelForTokenClassification(model_name, num_labels=2), resize its embeddings
    to match the trainer (vocab + n_extra_tok for [PAR]), load the TRAINED encoder
    body into `.bert` (strict=False tolerates the encoder's unused pooler) and copy
    the TRAINED `bound` weights into `.classifier`. The result reproduces the
    trainer's inference logits exactly. This does not touch training, the loss, the
    flags, or the existing .pt/cfg.json outputs — it only adds config.json + weights
    + tokenizer files so the downstream tools can read the run directory.
    """
    hf = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    hf.resize_token_embeddings(hf.config.vocab_size + n_extra_tok)
    hf.bert.load_state_dict(model.enc.state_dict(), strict=False)
    with torch.no_grad():
        hf.classifier.weight.copy_(model.bound.weight)
        hf.classifier.bias.copy_(model.bound.bias)
    hf.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"pos_weight={pw:.2f}  char_aug_rate={a.char_aug_rate}")
    model = BoundaryModel(a.model_name, pw, n_extra_tok=1)

    trds = Dataset.from_list(make_examples(tr, a.window, a.stride))
    dvds = Dataset.from_list(make_examples(dv, a.window, a.window))  # clean, un-augmented eval

    # Train collator perturbs on the fly; eval collator NEVER perturbs (rate 0).
    train_collator = AugmentEncodeCollator(tok, a.max_length, a.char_aug_rate, seed=a.seed)
    eval_collator = AugmentEncodeCollator(tok, a.max_length, 0.0, seed=a.seed)

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
    # fair_threshold.py / diag_preposition.py can load this run dir. Nothing above
    # is changed.
    _save_hf_checkpoint(model, tok, a.out, a.model_name, n_extra_tok=1)
    print(f"saved -> {a.out}")


# ---------------------------------------------------------------------------
# Predict (identical to coherence_train.cmd_predict; no augmentation at inference)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest_label_preservation(task, jsonl_path, rate, seed):
    """For EVERY doc: after augmentation the token COUNT and the LABEL sequence
    must be identical (labels preserved, still aligned)."""
    docs = load_task(task, "train", jsonl_path=jsonl_path)
    rng = random.Random(seed)
    n_docs = 0
    n_changed_words = 0
    n_words = 0
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        aug = augment_words(words, rate, rng)
        assert len(aug) == len(words), (
            f"TOKEN COUNT changed in {d['doc_id']}: {len(words)} -> {len(aug)}")
        # labels are attached positionally; a preserved count => preserved alignment.
        # Confirm the [PAR] markers are still exactly where the -100 labels are.
        aug_labs = [-100 if w == MODEL_PAR_TOKEN else lb
                    for w, lb in zip(aug, labs)]
        assert aug_labs == labs, f"LABEL sequence changed in {d['doc_id']}"
        n_docs += 1
        n_words += len(words)
        n_changed_words += sum(1 for a_, b_ in zip(words, aug) if a_ != b_)
    frac = n_changed_words / max(n_words, 1)
    print(f"  [label-preservation] {n_docs} docs, {n_words} words, rate={rate}: "
          f"count+labels identical for ALL docs; "
          f"{n_changed_words} words perturbed ({frac:.3f} of words).")
    return frac


def _selftest_rate0_identity(task, jsonl_path, seed):
    """rate 0.0 must reproduce the untouched baseline bit-for-bit (words AND the
    tokenised encoding are identical to feeding the raw words)."""
    docs = load_task(task, "train", jsonl_path=jsonl_path)
    tok = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    exs = make_examples(docs, 180, 90)
    coll0 = AugmentEncodeCollator(tok, 512, 0.0, seed=seed)
    n = 0
    for f in exs:
        # word-level identity
        aug = augment_words(f["words"], 0.0, random.Random(seed * 1_000_003 + n))
        assert aug == f["words"], "rate 0.0 changed a word (must be untouched)"
        # encoding identity vs a clean, augmentation-free encode
        clean = encode_one(f["words"], f["word_labels"], tok, 512)
        batch = coll0([copy.deepcopy(f)])
        got = batch["input_ids"][0].tolist()
        want = clean["input_ids"] + [tok.pad_token_id] * (len(got) - len(clean["input_ids"]))
        assert got == want, "rate 0.0 encoding differs from the un-augmented baseline"
        n += 1
    print(f"  [rate-0 identity] {n} windows: words unchanged AND token ids "
          f"bit-for-bit identical to the un-augmented baseline.")


def _selftest_gpu_smoke(task, jsonl_path, dev_jsonl, rate, seed, steps=6):
    """Train a few real steps at rate>0 to confirm no OOM / no shape errors."""
    set_seed(seed); random.seed(seed)
    model_name = "aubmindlab/bert-base-arabertv02"
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(task, "train", jsonl_path=jsonl_path)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = min((tot - pos) / max(pos, 1), 8.0)
    model = BoundaryModel(model_name, pw, n_extra_tok=1)
    trds = Dataset.from_list(make_examples(tr, 180, 90))
    coll = AugmentEncodeCollator(tok, 512, rate, seed=seed)
    targs = TrainingArguments(
        output_dir=os.path.join(HERE, os.pardir, "runs", "_charaug_smoke_ckpts"),
        max_steps=steps, per_device_train_batch_size=16, learning_rate=5e-5,
        fp16=torch.cuda.is_available(), logging_steps=1, report_to="none",
        save_strategy="no", remove_unused_columns=False, label_names=["labels"],
        seed=seed)
    tr_ = Trainer(model=model, args=targs, train_dataset=trds, data_collator=coll)
    out = tr_.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [gpu-smoke] device={device}  steps={steps}  rate={rate}  "
          f"final_train_loss={out.training_loss:.4f}  (no OOM, no shape error).")


def cmd_selftest(a):
    jsonl = a.train_jsonl or os.path.join(HERE, os.pardir, "data", f"{a.task}_train.jsonl")
    devj = a.dev_jsonl or os.path.join(HERE, os.pardir, "data", f"{a.task}_dev.jsonl")
    print("=" * 70)
    print(f"char_aug self-test  (task={a.task}, seed={a.seed})")
    print("=" * 70)
    _selftest_rate0_identity(a.task, jsonl, a.seed)
    for r in (0.05, 0.10, 0.25, 0.50):
        _selftest_label_preservation(a.task, jsonl, r, a.seed)
    if a.gpu_smoke:
        _selftest_gpu_smoke(a.task, jsonl, devj, max(a.char_aug_rate, 0.10),
                            a.seed, steps=a.smoke_steps)
    print("ALL SELF-TESTS PASSED.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--char-aug-rate", type=float, default=0.0,
                   help="per-word perturbation probability (0.0 = OFF = untouched baseline)")
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
    s.add_argument("--char-aug-rate", type=float, default=0.10)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--gpu-smoke", action="store_true", help="also run a few real train steps")
    s.add_argument("--smoke-steps", type=int, default=6)
    s.set_defaults(func=cmd_selftest)

    a = ap.parse_args(); a.func(a)
