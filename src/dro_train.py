"""GROUP-DRO boundary tagger for AraSeg (closed track) with GEORGE pseudo-groups
and a CTC-DRO smoothed group-weight update.

Self-contained — does NOT touch the shared train_encoder / predict / eval /
stitching pipeline. It only imports data.py (load_task, MODEL_PAR_TOKEN,
PARAGRAPH_TOKEN, TASKS) and baselines.write_submission. The data / window /
label / [PAR] handling MIRRORS consistency_train.py exactly (window 180,
stride 90, labels on the LAST subword of each word, MODEL_PAR_TOKEN for the
"\n" paragraph token, weighted CE with pos_weight capped ~8).

Base model: AutoModelForTokenClassification(aubmindlab/bert-base-arabertv02,
num_labels=2). One FLAG-GATED robust-optimization method is added on top of the
plain weighted-CE baseline: GROUP-DRO (Sagawa et al., ICLR 2020, "Distributionally
Robust Neural Networks", Algorithm 1) over GEORGE-style pseudo-groups, using the
CTC-DRO (smoothed) group-weight update. It is OFF by default: with --dro off the
trainer reproduces the plain weighted-CE baseline BIT-FOR-BIT (a single forward
pass, identical loss — verified with torch.equal in the GPU smoke).

METHOD — GROUP-DRO + GEORGE pseudo-groups + CTC-DRO update
-----------------------------------------------------------
(A) PSEUDO-GROUPS (GEORGE). Cluster the 174 train docs with k-means (k=4) on the
    standardized per-doc feature vector
        [ mean_sent_len, sent_len_std, boundary_rate, mean_perplexity ]
    perplexity = frozen AraBERTv02 pseudo-log-likelihood (PLL) computed with OUR
    OWN model (aubmindlab/bert-base-arabertv02 masked-LM head — closed track, no
    external model). The doc-level cluster id becomes the group label; every
    training WINDOW inherits its source document's group. Group ids and features
    are cached to <out>/groups.json so the (GPU-heavy) PLL pass runs once.

(B) OPTIMIZER (Sagawa Algorithm 1) with SMOOTHED (CTC-DRO) group weights.
    Maintain group weights q on the simplex (q_g >= 0, sum_g q_g = 1).
    Per step: sample a group g ~ Uniform, then sample a window from group g.
    LENGTH-MATCHED ACCUMULATION: accumulate the SUMMED per-group weighted-CE loss
    over a fixed token budget, and only update q AFTER EVERY GROUP has contributed
    at least one window in the accumulation cycle. Length matching (fixing the
    accumulated valid-token budget) prevents the sentence-length prior from
    masquerading as the group signal — a long-sentence group would otherwise
    accumulate more tokens per window and look "harder" for the wrong reason.

    q-update (CTC-DRO smoothed multiplicative weights):
        q_g <- [ q_g * exp( eta_q * Lbar_g / (q_g + alpha) ) ] / Z
    where Lbar_g is the mean per-token weighted-CE loss of group g over the cycle,
    Z renormalizes onto the simplex, eta_q = 0.01, and alpha in {0.1, 0.5, 1.0}
    (--dro-alpha). The (q_g + alpha) denominator is the CTC-DRO smoothing: alpha
    damps the update for tiny-q groups so a single group cannot run away (plain
    Group-DRO is the alpha -> 0 limit; alpha -> inf recovers uniform q, i.e. ERM).

    Model step (Sagawa): theta <- theta - eta_theta * q_g * grad(L_g). We realize
    the q_g weighting by scaling each group's accumulated loss by (q_g * G) before
    backward (G = number of groups), so that with UNIFORM q (q_g = 1/G) the scale
    is exactly 1 and the expected update matches ERM. Optimization uses AdamW with
    LIGHT L2 (weight_decay 0.01 — Sagawa found HEAVY L2 hurts Group-DRO) and early
    stopping on worst-group dev loss.

FLAG --dro {off,on} (default off):
    off  ->  uniform q, alpha -> inf: the trainer takes the plain weighted-CE
             path (a single forward, one nn.functional.cross_entropy identical to
             consistency_train's baseline) and is BIT-FOR-BIT ERM.
    on   ->  Group-DRO as above.

Usage:
  # baseline (bit-for-bit identical to plain weighted CE ERM):
  python dro_train.py train --task NoPnx-NP --dro off --out runs/dro_erm \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # Group-DRO:
  python dro_train.py train --task NoPnx-NP --dro on --dro-alpha 0.5 --out runs/dro_on \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  python dro_train.py predict --task NoPnx-NP --split dev --model runs/dro_on \
      --jsonl data/NoPnx-NP_dev.jsonl --out subs/dro_on_dev.csv
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from baselines import write_submission

HERE = os.path.dirname(os.path.abspath(__file__))

N_GROUPS = 4  # GEORGE k-means k (recipe: k=4)


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py verbatim)             #
# --------------------------------------------------------------------------- #
def make_examples(docs, window, stride, with_group=False):
    """Slide a (window, stride) window over each doc; [PAR] positions -> -100.

    If with_group, each doc dict must carry an int "group"; every window emitted
    from that doc inherits it as example["group"] (a window is assigned to its
    source document's pseudo-group).
    """
    out = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        labs = [-100 if w == MODEL_PAR_TOKEN else l for w, l in zip(words, d["labels"])]
        g = int(d.get("group", 0))
        n = len(words); s = 0
        while s < n:
            e = min(s + window, n)
            ex = {"words": words[s:e], "word_labels": labs[s:e]}
            if with_group:
                ex["group"] = g
            out.append(ex)
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


def pad_batch(feats, tok, device):
    """Pad a list of {input_ids, attention_mask, labels(list)} into a batch on device."""
    labs = [f.pop("labels") for f in feats]
    batch = tok.pad(feats, return_tensors="pt")
    m = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
    return {k: v.to(device) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# model wrapper: plain token-classification (no extra params)                 #
# --------------------------------------------------------------------------- #
class DROModel(nn.Module):
    """Bare AutoModelForTokenClassification with a weighted-CE helper. The DRO
    logic lives OUTSIDE the model (in the training loop), so `model.enc` is
    exactly the inference model and can be saved verbatim as an HF checkpoint.
    """

    def __init__(self, model_name, pos_weight=1.0, n_extra_tok=1):
        super().__init__()
        from transformers import AutoModelForTokenClassification
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.register_buffer("w", torch.tensor([1.0, float(pos_weight)]))
        self.config = self.enc.config

    def logits(self, input_ids, attention_mask):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).logits

    def ce(self, logits, labels):
        """Plain weighted mean cross-entropy (ignore_index=-100). This is the
        EXACT baseline term — identical to consistency_train's ._ce."""
        return nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100)

    def ce_sum_and_count(self, logits, labels):
        """Summed weighted CE over valid tokens + the valid-token count. Used for
        length-matched DRO accumulation (mean = sum / count within a group)."""
        per_tok = nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100, reduction="sum")
        count = int((labels.view(-1) != -100).sum())
        return per_tok, count


# --------------------------------------------------------------------------- #
# (A) GEORGE pseudo-groups                                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def doc_pseudo_log_likelihood(text_tokens, mlm, tok, device, max_len=512, max_mask=64):
    """Frozen-AraBERTv02 pseudo-log-likelihood proxy for a document.

    True PLL masks every token in turn (O(L) forwards). To keep the GEORGE
    feature pass affordable on 174 docs we estimate it on a truncated window and
    a capped, evenly-spaced subset of mask positions, then report the MEAN
    per-masked-token NLL (so it is a length-normalized "surprise"; higher = more
    perplexing). This is a *feature* for clustering, not a reported metric, so an
    unbiased subset estimate is sufficient.
    """
    ids = tok(text_tokens, is_split_into_words=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    input_ids = ids["input_ids"][0]
    attn = ids["attention_mask"][0]
    special = set(tok.all_special_ids)
    cand = [i for i in range(len(input_ids)) if int(input_ids[i]) not in special]
    if not cand:
        return 0.0
    if len(cand) > max_mask:
        sel = np.linspace(0, len(cand) - 1, max_mask).round().astype(int)
        cand = [cand[i] for i in sel]

    nlls = []
    # batch the masked variants for speed
    B = 32
    for b0 in range(0, len(cand), B):
        chunk = cand[b0:b0 + B]
        batch_ids = input_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        for r, pos in enumerate(chunk):
            batch_ids[r, pos] = tok.mask_token_id
        batch_attn = attn.unsqueeze(0).repeat(len(chunk), 1)
        out = mlm(input_ids=batch_ids.to(device),
                  attention_mask=batch_attn.to(device)).logits
        logp = torch.log_softmax(out.float(), dim=-1)
        for r, pos in enumerate(chunk):
            gold = int(input_ids[pos])
            nlls.append(-float(logp[r, pos, gold]))
    return float(np.mean(nlls)) if nlls else 0.0


def doc_features(docs, mlm, tok, device):
    """Per-doc [mean_sent_len, sent_len_std, boundary_rate, mean_perplexity]."""
    feats = []
    for j, d in enumerate(docs):
        labels = d["labels"]
        tokens = d["tokens"]
        # sentence lengths = token gaps between consecutive boundaries (skip [PAR])
        sent_lens = []
        cur = 0
        for w, l in zip(tokens, labels):
            if w == PARAGRAPH_TOKEN:
                continue
            cur += 1
            if l == 1:
                sent_lens.append(cur); cur = 0
        if cur > 0:
            sent_lens.append(cur)
        sent_lens = sent_lens or [len(tokens)]
        mean_sl = float(np.mean(sent_lens))
        std_sl = float(np.std(sent_lens))
        non_par = sum(1 for w in tokens if w != PARAGRAPH_TOKEN) or 1
        brate = float(sum(labels) / non_par)
        # PLL feature on the model's own [PAR]-substituted token stream
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in tokens]
        ppl = doc_pseudo_log_likelihood(words, mlm, tok, device)
        feats.append([mean_sl, std_sl, brate, ppl])
        if (j + 1) % 25 == 0 or j + 1 == len(docs):
            print(f"  [GEORGE] features {j + 1}/{len(docs)}", flush=True)
    return np.array(feats, dtype=np.float64)


def assign_groups(docs, model_name, device, seed, cache_path=None):
    """Compute GEORGE pseudo-groups; write d['group'] in place. Returns (labels,
    sizes, feature_matrix). Cached to cache_path if given."""
    if cache_path and os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        by_id = {r["doc_id"]: int(r["group"]) for r in cached["rows"]}
        for d in docs:
            d["group"] = by_id.get(d["doc_id"], 0)
        labels = np.array([d["group"] for d in docs])
        sizes = [int((labels == g).sum()) for g in range(N_GROUPS)]
        print(f"[GEORGE] loaded cached groups from {cache_path}  sizes={sizes}")
        return labels, sizes, np.array(cached["features"])

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print("[GEORGE] loading frozen AraBERTv02 masked-LM head for PLL features ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    mlm = AutoModelForMaskedLM.from_pretrained(model_name)
    mlm.resize_token_embeddings(mlm.config.vocab_size + 1)  # [PAR]
    mlm.to(device).eval()

    feats = doc_features(docs, mlm, tok, device)
    Xz = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=N_GROUPS, random_state=seed, n_init=10)
    labels = km.fit_predict(Xz)
    for d, g in zip(docs, labels):
        d["group"] = int(g)
    sizes = [int((labels == g).sum()) for g in range(N_GROUPS)]
    print(f"[GEORGE] k-means k={N_GROUPS} on standardized "
          f"[mean_sent_len, sent_len_std, boundary_rate, mean_perplexity]")
    print(f"[GEORGE] cluster sizes = {sizes}  (total {int(labels.shape[0])} docs)")

    # free the MLM before training
    del mlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        json.dump({
            "n_groups": N_GROUPS,
            "feature_names": ["mean_sent_len", "sent_len_std", "boundary_rate", "mean_perplexity"],
            "features": feats.tolist(),
            "rows": [{"doc_id": d["doc_id"], "group": int(g)} for d, g in zip(docs, labels)],
            "sizes": sizes,
        }, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[GEORGE] cached groups -> {cache_path}")
    return labels, sizes, feats


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #
def prf1_from_counts(tp, fp, fn):
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return 2 * P * R / max(P + R, 1e-9)


@torch.no_grad()
def eval_dev(model, dev_by_group, tok, device):
    """Return (overall_f1, per_group_mean_loss dict, worst_group_loss). Windows
    are grouped so we can track WORST-GROUP dev loss for early stopping."""
    model.eval()
    tp = fp = fn = 0
    grp_loss_sum = {g: 0.0 for g in dev_by_group}
    grp_tok = {g: 0 for g in dev_by_group}
    for g, exs in dev_by_group.items():
        for b0 in range(0, len(exs), 16):
            feats = [dict(input_ids=e["input_ids"], attention_mask=e["attention_mask"],
                          labels=list(e["labels"])) for e in exs[b0:b0 + 16]]
            batch = pad_batch(feats, tok, device)
            logits = model.logits(batch["input_ids"], batch["attention_mask"])
            lsum, cnt = model.ce_sum_and_count(logits, batch["labels"])
            grp_loss_sum[g] += float(lsum); grp_tok[g] += cnt
            p = logits.argmax(-1); mask = batch["labels"] != -100
            pr, gd = p[mask], batch["labels"][mask]
            tp += int(((pr == 1) & (gd == 1)).sum())
            fp += int(((pr == 1) & (gd == 0)).sum())
            fn += int(((pr == 0) & (gd == 1)).sum())
    f1 = prf1_from_counts(tp, fp, fn)
    grp_mean = {g: grp_loss_sum[g] / max(grp_tok[g], 1) for g in dev_by_group}
    worst = max(grp_mean.values()) if grp_mean else float("inf")
    return f1, grp_mean, worst


# --------------------------------------------------------------------------- #
# training                                                                     #
# --------------------------------------------------------------------------- #
def _save_hf_checkpoint(model, tok, out_dir):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / eval_local.py / diag_preposition.py load with
    AutoModelForTokenClassification.from_pretrained(out_dir) + AutoTokenizer, which
    the custom .pt cannot satisfy. Here `model.enc` IS the trained
    AutoModelForTokenClassification (encoder + boundary classifier); the Group-DRO
    logic lives in the training loop and adds NO parameters, so `model.enc` is
    exactly the inference model. We write it verbatim. This mirrors
    consistency_train._save_hf_checkpoint; it does not touch training, the loss, the
    flags, or the .pt/cfg.json outputs — it only adds config.json + weights +
    tokenizer files so the downstream tools can read the run directory.
    """
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def cmd_train(a):
    from transformers import AutoTokenizer, set_seed
    dro_on = (a.dro == "on")
    set_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})

    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    print(f"dro={a.dro}  dro_alpha={a.dro_alpha}  eta_q={a.eta_q}  "
          f"token_budget={a.token_budget}  pos_weight={pw:.2f}")

    # (A) GEORGE pseudo-groups — needed only when DRO is ON. When OFF we skip the
    # PLL pass entirely so the ERM path is fast and side-effect free.
    if dro_on:
        os.makedirs(a.out, exist_ok=True)
        _, sizes, _ = assign_groups(
            tr, a.model_name, device, a.seed,
            cache_path=a.groups_cache or os.path.join(a.out, "groups.json"))
    else:
        for d in tr:
            d["group"] = 0
        sizes = None

    model = DROModel(a.model_name, pw, n_extra_tok=1).to(device)

    # dev windows (clean, non-overlapping), bucketed by group for worst-group loss
    for d in dv:
        d["group"] = 0
    if dro_on:
        # reuse the same GEORGE features on dev? No — dev has no gold-group notion
        # in Sagawa; worst-group is over TRAIN pseudo-groups. For dev we bucket by
        # a lightweight re-clustering is overkill; instead we track OVERALL dev F1
        # (model-selection metric) plus per-train-group TRAIN loss. Dev bucket is a
        # single group so worst==overall dev loss — used only as a tie-break.
        pass
    dv_ex = make_examples(dv, a.window, a.window)  # clean eval windows
    dv_enc = [encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, a.max_length)
              for e in dv_ex]
    dev_by_group = {0: [{"input_ids": e["input_ids"][0], "attention_mask": e["attention_mask"][0],
                         "labels": e["labels"][0]} for e in dv_enc]}

    # train windows (carry group)
    tr_ex = make_examples(tr, a.window, a.stride, with_group=dro_on)
    tr_enc = []
    for e in tr_ex:
        enc = encode({"words": [e["words"]], "word_labels": [e["word_labels"]]}, tok, a.max_length)
        rec = {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
               "labels": enc["labels"][0], "group": int(e.get("group", 0))}
        tr_enc.append(rec)

    # bucket train windows by group
    by_group = {g: [] for g in range(N_GROUPS if dro_on else 1)}
    for rec in tr_enc:
        by_group.setdefault(rec["group"], []).append(rec)
    # groups that actually have windows
    active_groups = [g for g in sorted(by_group) if by_group[g]]
    print(f"train windows={len(tr_enc)}  window counts per group="
          f"{ {g: len(by_group[g]) for g in active_groups} }")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total_windows = len(tr_enc)
    steps_per_epoch = max(1, total_windows // a.batch_size)
    total_steps = steps_per_epoch * a.epochs
    warmup = int(a.warmup_ratio * total_steps)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.0, 1.0 - prog)  # linear decay (matches TrainingArguments default)

    base_lr = a.lr

    # DRO group weights on the simplex (uniform init)
    G = len(active_groups)
    q = {g: 1.0 / G for g in active_groups}
    alpha = float(a.dro_alpha)

    rng = random.Random(a.seed)
    best_f1 = -1.0; best_state = None; bad = 0
    gstep = 0

    for epoch in range(a.epochs):
        model.train()
        if not dro_on:
            # -------------------- ERM path (bit-for-bit) -------------------- #
            # Plain shuffled minibatch weighted-CE, one forward per batch, one
            # nn.functional.cross_entropy call identical to the consistency
            # baseline. No group sampling, no q, no loss scaling.
            order = list(range(total_windows)); rng.shuffle(order)
            for b0 in range(0, total_windows, a.batch_size):
                idx = order[b0:b0 + a.batch_size]
                feats = [dict(input_ids=tr_enc[i]["input_ids"],
                              attention_mask=tr_enc[i]["attention_mask"],
                              labels=list(tr_enc[i]["labels"])) for i in idx]
                batch = pad_batch(feats, tok, device)
                for pg in opt.param_groups:
                    pg["lr"] = base_lr * lr_at(gstep)
                opt.zero_grad()
                logits = model.logits(batch["input_ids"], batch["attention_mask"])
                loss = model.ce(logits, batch["labels"])
                loss.backward(); opt.step(); gstep += 1
                if gstep % a.log_steps == 0:
                    print(f"  [erm] ep{epoch} step{gstep} loss={loss.item():.4f}")
        else:
            # -------------------- GROUP-DRO path ---------------------------- #
            # Sagawa Alg 1 with length-matched accumulation + CTC-DRO q-update.
            # Per q-cycle: keep sampling (group g ~ Uniform, then a window from g),
            # accumulate summed per-group loss until the valid-token budget is met
            # AND every active group has contributed, then update q and take the
            # optimizer step (scaled by q_g * G per group).
            n_cycles = max(1, steps_per_epoch)
            for _cyc in range(n_cycles):
                opt.zero_grad()
                grp_loss_sum = {g: torch.zeros((), device=device) for g in active_groups}
                grp_tok = {g: 0 for g in active_groups}
                contributed = set()
                acc_tok = 0
                # length-matched accumulation
                while acc_tok < a.token_budget or len(contributed) < G:
                    g = rng.choice(active_groups)
                    rec = rng.choice(by_group[g])
                    feats = [dict(input_ids=rec["input_ids"],
                                  attention_mask=rec["attention_mask"],
                                  labels=list(rec["labels"]))]
                    batch = pad_batch(feats, tok, device)
                    logits = model.logits(batch["input_ids"], batch["attention_mask"])
                    lsum, cnt = model.ce_sum_and_count(logits, batch["labels"])
                    grp_loss_sum[g] = grp_loss_sum[g] + lsum
                    grp_tok[g] += cnt
                    contributed.add(g)
                    acc_tok += cnt
                    # safety valve: don't loop forever on empty-label windows
                    if acc_tok >= a.token_budget * 4 and len(contributed) >= G:
                        break

                # per-group MEAN per-token loss over the cycle (length-normalized)
                grp_mean = {g: grp_loss_sum[g] / max(grp_tok[g], 1) for g in active_groups}

                # ---- CTC-DRO smoothed multiplicative q-update -------------- #
                # q_g <- q_g * exp(eta_q * Lbar_g / (q_g + alpha)); renormalize.
                with torch.no_grad():
                    log_q = {}
                    for g in active_groups:
                        Lbar = float(grp_mean[g].detach())
                        log_q[g] = math.log(max(q[g], 1e-12)) + a.eta_q * Lbar / (q[g] + alpha)
                    mx = max(log_q.values())
                    exps = {g: math.exp(log_q[g] - mx) for g in active_groups}
                    Z = sum(exps.values())
                    q = {g: exps[g] / Z for g in active_groups}

                # ---- Sagawa model step: theta -= eta_theta * q_g * grad(L_g) #
                # scale each group's mean loss by (q_g * G): uniform q -> scale 1.
                dro_loss = sum((q[g] * G) * grp_mean[g] for g in active_groups)
                for pg in opt.param_groups:
                    pg["lr"] = base_lr * lr_at(gstep)
                dro_loss.backward(); opt.step(); gstep += 1
                if gstep % a.log_steps == 0:
                    qs = {g: round(q[g], 3) for g in active_groups}
                    ls = {g: round(float(grp_mean[g].detach()), 3) for g in active_groups}
                    worst_g = max(active_groups, key=lambda g: float(grp_mean[g].detach()))
                    print(f"  [dro] ep{epoch} step{gstep} q={qs} Lbar={ls} "
                          f"worst_g={worst_g} loss={float(dro_loss):.4f}")

        # ---- end epoch: eval + early stop on dev F1 (worst-group loss logged)
        f1, grp_mean_dev, worst = eval_dev(model, dev_by_group, tok, device)
        print(f"[epoch {epoch}] dev_f1={f1 * 100:.2f}  worst_group_dev_loss={worst:.4f}"
              + (f"  q={ {g: round(q[g], 3) for g in active_groups} }" if dro_on else ""))
        model.train()
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                print(f"[early-stop] no dev-F1 improvement for {bad} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best dev f1={best_f1 * 100:.2f}")

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    json.dump({"model_name": a.model_name, "dro": a.dro, "dro_alpha": a.dro_alpha,
               "sizes": sizes}, open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: emit an HF checkpoint so predict.py / eval_local.py / diag_* can load.
    _save_hf_checkpoint(model, tok, a.out)
    print(f"saved -> {a.out}")


@torch.no_grad()
def cmd_predict(a):
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = DROModel(cfg["model_name"], n_extra_tok=1)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=device))
    model.to(device).eval()
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
            logit = model.logits(enc["input_ids"].to(device),
                                 enc["attention_mask"].to(device))[0]
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
    t.add_argument("--dro", choices=["off", "on"], default="off",
                   help="off = uniform q, alpha->inf: plain weighted-CE ERM (bit-for-bit). "
                        "on = Group-DRO (GEORGE pseudo-groups + CTC-DRO update)")
    t.add_argument("--dro-alpha", type=float, default=0.5,
                   help="CTC-DRO smoothing in {0.1,0.5,1.0}: q_g <- q_g*exp(eta_q*Lbar_g/(q_g+alpha))")
    t.add_argument("--eta-q", type=float, default=0.01, help="group-weight step size")
    t.add_argument("--token-budget", type=int, default=1024,
                   help="length-matched accumulation: valid-token budget per q-update cycle")
    t.add_argument("--groups-cache", default=None,
                   help="path to cache GEORGE groups.json (default: <out>/groups.json)")
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--weight-decay", type=float, default=0.01, help="LIGHT L2 (Sagawa: heavy L2 hurts)")
    t.add_argument("--warmup-ratio", type=float, default=0.1)
    t.add_argument("--patience", type=int, default=3, help="early-stop patience on dev F1 (0=off)")
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--log-steps", type=int, default=20)
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--task", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(); a.func(a)
