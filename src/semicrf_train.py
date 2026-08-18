"""Filtered Semi-Markov CRF boundary segmenter for AraSeg NoPnx-NP (closed track).

Self-contained — does NOT touch the shared train_encoder / predict / eval /
stitching pipeline. It only imports/calls them. The data / window / label /
[PAR] handling MIRRORS src/consistency_train.py EXACTLY (window 180, stride 90,
labels on the LAST subword of each word, MODEL_PAR_TOKEN for the "\n" paragraph
token, weighted CE with pos_weight capped ~8, epochs 8, batch 16, lr 5e-5).

Base model: AutoModelForTokenClassification(aubmindlab/bert-base-arabertv02,
num_labels=2). A FLAG-GATED Filtered Semi-Markov CRF is added on top of the plain
weighted-CE baseline (Sarawagi & Cohen NIPS 2004; Filtered variant Zaratiana et
al. Findings-EMNLP 2023; pre-blessed by DIRECTIVES v13 item 106 — the
"learned segment-length distribution replacing the hand prior" stretch of the
learned-length-decode line).

  --semicrf off  (default)  =>  plain weighted-CE token head, BIT-FOR-BIT
                                identical to consistency_train.py's baseline
                                (single forward pass; loss == weighted CE;
                                verified with torch.equal in smoke_semicrf.py).
  --semicrf on              =>  joint loss L = L_local + L_global on top of the
                                same token logits; inference uses the semi-CRF's
                                OWN Viterbi decode (NOT token-argmax).

RECIPE (reframes binary boundary-after-token as SEGMENT labeling)
-----------------------------------------------------------------
A segmentation of a length-T window is a sequence of non-overlapping spans
[0,e0], [e0+1,e1], ... covering every position. The boundary label 1 lands on
each span's LAST token (a boundary FOLLOWS it); every interior token is 0. That
is exactly equivalent to the binary boundary-after-token task. Every chosen span
is a "sentence" (1 real label 'sentence' + a 'null' label used only by the local
filter). Max segment length is CAPPED at L (default 32).

  * SPAN SCORER  phi(i,j) = MLP( concat(h_i, h_j, length_embedding(j-i+1)) )
    -> 2 logits [null, sentence]. h are AraBERTv02 token vectors at the LAST
    subword of words i and j.
  * LEARNED DURATION POTENTIAL: a bias table over span length (one scalar per
    length 1..L), added to the 'sentence' score in the GLOBAL layer. This is the
    mechanism that replaces the hand-tuned ~9-10 token length prior, and it is
    inspectable (dump_duration()).
  * FILTERED pruning: the local span classifier scores EVERY candidate span; a
    span SURVIVES iff argmax label != null. Null-argmax spans are dropped before
    the global layer. Gold spans are always KEPT in the lattice (train >= decode:
    the gold path must be reachable so logZ >= gold-path score by construction).
  * JOINT LOSS L = L_local + L_global:
      L_local  = weighted cross-entropy of the local classifier over all
                 candidate spans (gold span -> 'sentence', every other candidate
                 span -> 'null'); the -log p_local(i,j,l*) of the gold spans is
                 its 'sentence'-class term.
      L_global = -S(y|x) + logZ(x), forward algorithm over the FILTERED lattice,
                 S_span(i,j) = phi_sentence(i,j) + duration_bias(j-i+1).
    Trained WITH the layer (train >= decode constraint). Plain PyTorch DP only —
    no Triton / Flash-SemiCRF (dependency violation).

Complexity O(T*L*C^2) with C=1 real label -> O(T*L); instant at T=180, L=32.
Per-window semi-CRF; the existing stitching (predict overlap-average / merge) is
NOT touched — the predict subcommand decodes each window with Viterbi and writes
an eval_local-compatible submission CSV via baselines.write_submission.

RESCUE 2026-07-10 (forensic verdict of record: lr starvation + window-end decode
defect; both fixes ADDITIVE and flag-gated, defaults bit-for-bit):
  train   --head-lr / --dur-lr : dedicated AdamW param groups for the from-
          scratch head (scorer.mlp / scorer.len_emb at --head-lr, the
          scorer.duration_bias table at --dur-lr with weight decay FORCED to 0).
          Default None = the legacy single-lr HF optimizer, untouched.
  predict --mask-window-end on : skip the FORCED boundary vote at each window's
          final label position (Viterbi must end every window with a boundary —
          a tiling constraint, not a model decision). Default off = legacy merge.

ADDITIVE HF CHECKPOINT: at the end of train we also emit a plain HuggingFace
AutoModelForTokenClassification checkpoint into the run dir (same
_save_hf_checkpoint pattern as consistency / teacher / char_aug / sub2), so
predict.py / eval_local.py / diag_preposition.py can still load the run dir with
token-argmax. The semi-CRF's own decode is exposed through THIS file's `predict`
subcommand (per the pre-registered exception).

Usage:
  # baseline (bit-for-bit identical to plain weighted CE):
  python semicrf_train.py train --task NoPnx-NP --out runs/semicrf_off \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # filtered semi-Markov CRF:
  python semicrf_train.py train --task NoPnx-NP --semicrf on --out runs/semicrf_on \
      --train-jsonl data/NoPnx-NP_train.jsonl --dev-jsonl data/NoPnx-NP_dev.jsonl

  # predict with the semi-CRF's OWN Viterbi decode (writes submission CSV):
  python semicrf_train.py predict --task NoPnx-NP --split dev --model runs/semicrf_on \
      --jsonl data/NoPnx-NP_dev.jsonl --out subs/semicrf_on_dev.csv
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

HERE = os.path.dirname(os.path.abspath(__file__))

NEG_INF = -1e9  # additive mask for pruned / impossible spans in the DP


def _gold_spans_capped(ends, L):
    """Turn a sorted list of gold segment END indices into (start,end) spans that
    tile 0..ends[-1] with EVERY span length <= L.

    A gold segment [prev..e] longer than the hard cap L cannot be a single span
    in a length-capped semi-Markov CRF (Sarawagi & Cohen NIPS2004). Standard
    handling: chunk it into consecutive spans of length <= L. The chunk that
    lands on the TRUE boundary `e` is the segment's real end; the interior chunk
    joints are forced-continuation splits (an accepted, documented approximation
    of the hard cap). This guarantees the gold path is representable in the
    L-capped lattice, so the forward algorithm's logZ >= gold-path score.
    """
    spans = []
    prev = 0
    for e in ends:
        i = prev
        while e - i + 1 > L:          # too long: emit a maximal L-length chunk
            spans.append((i, i + L - 1))
            i += L
        spans.append((i, e))          # final chunk carries the true boundary e
        prev = e + 1
    return spans


# --------------------------------------------------------------------------- #
# window / label pipeline (mirrors consistency_train.py EXACTLY)              #
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
    """Pads input_ids/attention_mask/labels — identical to consistency_train's
    padding (labels padded with -100). The semi-CRF consumes the SAME `labels`
    tensor: it reads the last-subword label positions (label != -100) to recover
    the per-window boundary sequence and hence the gold segmentation.
    """
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labs = [f.pop("labels") for f in feats]
        batch = self.tok.pad(feats, return_tensors="pt")
        m = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([l + [-100] * (m - len(l)) for l in labs])
        return batch


# --------------------------------------------------------------------------- #
# Filtered Semi-Markov CRF head                                               #
# --------------------------------------------------------------------------- #
class SpanScorer(nn.Module):
    """phi(i,j) = MLP( concat(h_i, h_j, length_embedding(j-i+1)) ) -> [null, sentence].

    Also owns the LEARNED DURATION bias table (one scalar per length 1..L),
    added to the 'sentence' score in the GLOBAL layer only.
    """
    def __init__(self, hidden, max_span, len_emb_dim=32, mlp_dim=256):
        super().__init__()
        self.max_span = max_span
        # length embedding indexed by (d-1) for d in 1..max_span
        self.len_emb = nn.Embedding(max_span, len_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden + len_emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 2),   # [null, sentence]
        )
        # LEARNED DURATION POTENTIAL: bias table over span length (inspectable).
        self.duration_bias = nn.Parameter(torch.zeros(max_span))

    def span_logits(self, h_last, starts, ends):
        """Local 2-class logits for a batch of candidate spans.

        h_last : (P, hidden)  hidden vector at each label position (last subword)
        starts : (S,) long    start label-index of each candidate span
        ends   : (S,) long    end   label-index of each candidate span (inclusive)
        returns (S, 2) local logits [null, sentence].
        """
        hi = h_last[starts]                       # (S, hidden)
        hj = h_last[ends]                         # (S, hidden)
        d = (ends - starts)                       # length-1 in [0, max_span-1]
        le = self.len_emb(d.clamp(max=self.max_span - 1))
        feat = torch.cat([hi, hj, le], dim=-1)
        return self.mlp(feat)                     # (S, 2)


class SemiCRFModel(nn.Module):
    """AutoModelForTokenClassification base + a flag-gated Filtered Semi-CRF head.

    OFF (semicrf=False): forward returns exactly weighted-CE over the base
      classifier logits — bit-for-bit identical to consistency_train's baseline
      (the semi-CRF submodules are never touched, so they add no compute/graph).

    ON  (semicrf=True):  forward computes L_local + L_global on top of the SAME
      base hidden states. The base token-classifier head is still trained (its
      weighted CE is folded into L_local as an auxiliary term at ce_weight, and
      the additive HF checkpoint reuses it); inference uses the semi-CRF decode.
    """
    def __init__(self, model_name, pos_weight=1.0, semicrf=False, max_span=32,
                 len_emb_dim=32, mlp_dim=256, ce_weight=1.0, keep_per_end=8,
                 n_extra_tok=1):
        super().__init__()
        self.enc = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
        self.enc.resize_token_embeddings(self.enc.config.vocab_size + n_extra_tok)
        self.semicrf = bool(semicrf)
        self.max_span = int(max_span)
        self.ce_weight = float(ce_weight)
        self.keep_per_end = int(keep_per_end)
        self.register_buffer("w", torch.tensor([1.0, pos_weight]))
        self.config = self.enc.config
        if self.semicrf:
            h = self.enc.config.hidden_size
            self.scorer = SpanScorer(h, self.max_span, len_emb_dim, mlp_dim)

    # -- base token path (identical to the plain weighted-CE baseline) ------- #
    def _base(self, input_ids, attention_mask, output_hidden_states=False):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=output_hidden_states)

    def _ce(self, logits, labels):
        return nn.functional.cross_entropy(
            logits.view(-1, 2).float(), labels.view(-1),
            weight=self.w.to(logits.device), ignore_index=-100)

    def _candidates(self, P, device):
        """All spans [i,j] with 1 <= j-i+1 <= L, plus an index map (i,j)->row.
        Shared by the training loss and the decode so train >= decode holds."""
        L = self.max_span
        starts_list, ends_list = [], []
        for i in range(P):
            for j in range(i, min(i + L, P)):
                starts_list.append(i); ends_list.append(j)
        starts = torch.tensor(starts_list, device=device, dtype=torch.long)
        ends_t = torch.tensor(ends_list, device=device, dtype=torch.long)
        idx = {}
        for r, (i, j) in enumerate(zip(starts_list, ends_list)):
            idx[(i, j)] = r
        return starts, ends_t, ends_list, idx

    def _survive_mask(self, span_local, ends_list, device):
        """FILTERED pruning, IDENTICAL in train and decode.

        A span SURVIVES iff its local argmax label != null (the recipe's rule).
        Floor: additionally keep the top-`keep_per_end` 'sentence'-scoring spans
        ENDING at each position, so the lattice is never starved and the learned
        duration potential stays able to compete (Zaratiana et al. keep a top-k
        floor rather than a bare argmax threshold — otherwise an untrained/over-
        confident null head can prune every multi-token span and force the
        degenerate every-token-a-boundary tiling). Gold spans are added on top of
        this by the caller (train only)."""
        survive = (span_local.argmax(-1) != 0)          # argmax != null
        if self.keep_per_end > 0:
            sent = span_local[:, 1]                       # 'sentence' score
            by_end = {}
            for r, e in enumerate(ends_list):
                by_end.setdefault(e, []).append(r)
            for e, rows in by_end.items():
                if len(rows) <= self.keep_per_end:
                    for r in rows:
                        survive[r] = True
                else:
                    scores = torch.tensor([float(sent[r]) for r in rows])
                    top = torch.topk(scores, self.keep_per_end).indices.tolist()
                    for t in top:
                        survive[rows[t]] = True
        return survive

    # ------------------------------------------------------------------ #
    #  Per-example semi-CRF loss over ONE window                          #
    # ------------------------------------------------------------------ #
    def _window_losses(self, h_last, gold_bound):
        """Compute (L_local, L_global) for one window.

        h_last     : (P, hidden)  hidden vectors at the P label positions.
        gold_bound : (P,) long    gold boundary label (0/1) at each label pos;
                     position P-1 is forced to a boundary (window/doc segment end).

        Returns (l_local, l_global, decode_ok, n_span) all as python-usable
        tensors/ints. Uses only positions 0..P-1 (the real label positions).
        """
        P = h_last.shape[0]
        L = self.max_span
        device = h_last.device

        # -- gold segmentation: spans END at every gold boundary. The final
        #    label position is always a segment end. -----------------------
        gb = gold_bound.clone()
        gb[P - 1] = 1
        ends = torch.nonzero(gb == 1, as_tuple=False).flatten().tolist()
        gold_spans = _gold_spans_capped(ends, L)
        # gold_spans tile 0..P-1 exactly, and every span has length <= L
        # (a gold segment longer than the hard cap L is chunked into consecutive
        #  <=L spans — the standard hard-cap handling for a length-capped
        #  semi-Markov CRF; see _gold_spans_capped. This keeps the gold path
        #  REACHABLE inside the L-capped lattice, so logZ >= gold-path score by
        #  construction (train >= decode). Under L=32 this touches ~2.7% of
        #  NoPnx-NP gold segments — the near-empty-boundary docs.)

        # -- enumerate ALL candidate spans [i,j], j-i+1 <= L ---------------- #
        starts, ends_t, ends_list, idx = self._candidates(P, device)
        span_local = self.scorer.span_logits(h_last, starts, ends_t)  # (S,2)

        # ---- L_local: CE over every candidate span (gold->sentence else null)
        tgt = torch.zeros(span_local.shape[0], dtype=torch.long, device=device)
        for (i, j) in gold_spans:
            tgt[idx[(i, j)]] = 1
        # weight the rare 'sentence' class like the token head (pos_weight)
        l_local = nn.functional.cross_entropy(
            span_local.float(), tgt, weight=self.w.to(device))

        # ---- FILTERED lattice: survive iff local argmax != null (+top-k floor).
        #      Gold spans are ALWAYS kept (train >= decode). ---------------- #
        survive = self._survive_mask(span_local, ends_list, device)
        for (i, j) in gold_spans:
            survive[idx[(i, j)]] = True

        # ---- global span score S_span = phi_sentence + duration_bias(len) - #
        dur = self.scorer.duration_bias  # (L,)
        d_all = (ends_t - starts)        # 0..L-1
        s_span = span_local[:, 1] + dur[d_all.clamp(max=L - 1)]   # (S,)
        # pruned spans -> NEG_INF so the DP never uses them
        s_masked = torch.where(survive, s_span, torch.full_like(s_span, NEG_INF))

        # ---- forward algorithm over the filtered lattice (logZ) ----------- #
        # alpha[t] = logsumexp over spans ending at t-1 of alpha[start] + s_span
        # positions are 0..P; a segmentation tiles [0..P-1].
        alpha = torch.full((P + 1,), NEG_INF, device=device)
        alpha[0] = 0.0
        for t in range(1, P + 1):
            # spans ending at label index t-1, of length d = t - i, 1..min(L,t)
            terms = []
            j = t - 1
            for i in range(max(0, t - L), t):
                r = idx[(i, j)]
                terms.append(alpha[i] + s_masked[r])
            stacked = torch.stack(terms)
            alpha[t] = torch.logsumexp(stacked, dim=0)
        logZ = alpha[P]

        # ---- gold-path score S(y|x) -------------------------------------- #
        gold_score = torch.zeros((), device=device)
        for (i, j) in gold_spans:
            gold_score = gold_score + s_span[idx[(i, j)]]   # gold uses unmasked

        l_global = logZ - gold_score

        return l_local, l_global, logZ, gold_score, len(gold_spans)

    # ------------------------------------------------------------------ #
    #  Viterbi decode over ONE window (used by predict, NOT token-argmax) #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode_window(self, h_last):
        """Max-scoring segmentation over the filtered lattice.

        Returns a 0/1 boundary label per label position (last position forced 1).
        """
        P = h_last.shape[0]
        L = self.max_span
        device = h_last.device
        if P == 0:
            return []

        starts, ends_t, ends_list, idx = self._candidates(P, device)
        span_local = self.scorer.span_logits(h_last, starts, ends_t)

        # SAME filtered pruning as training (argmax != null + top-k floor):
        survive = self._survive_mask(span_local, ends_list, device)
        dur = self.scorer.duration_bias
        d_all = (ends_t - starts)
        s_span = span_local[:, 1] + dur[d_all.clamp(max=L - 1)]
        s_masked = torch.where(survive, s_span, torch.full_like(s_span, NEG_INF))

        # Viterbi over span lengths. best[t] = max score to tile 0..t-1.
        best = torch.full((P + 1,), NEG_INF, device=device)
        back = [-1] * (P + 1)
        best[0] = 0.0
        for t in range(1, P + 1):
            j = t - 1
            bi, bv = -1, NEG_INF
            for i in range(max(0, t - L), t):
                v = best[i] + s_masked[idx[(i, j)]]
                if float(v) > float(bv):
                    bv = v; bi = i
            best[t] = bv; back[t] = bi
        # If the whole path is NEG_INF (pathological: everything pruned yet gold
        # not present at decode), fall back to the length-1 tiling (every token a
        # boundary is still a VALID segmentation). Guaranteed valid output.
        if not torch.isfinite(best[P]):
            return [1] * P

        # backtrace -> segment ends -> boundary labels
        labels = [0] * P
        t = P
        while t > 0:
            i = back[t]
            if i < 0:
                i = t - 1  # safety: length-1 span
            labels[t - 1] = 1  # end of this span is a boundary
            t = i
        return labels

    # ------------------------------------------------------------------ #
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        # -- OFF: bit-for-bit plain weighted CE (single forward, no semi-CRF) -
        if not self.semicrf:
            logits = self._base(input_ids, attention_mask).logits
            loss = self._ce(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits}

        # -- ON: joint semi-CRF loss on top of the same base hidden states ---
        out = self._base(input_ids, attention_mask, output_hidden_states=True)
        logits = out.logits                            # (B,T,2) token head
        h = out.hidden_states[-1]                       # (B,T,hidden)
        loss = None
        if labels is not None:
            device = logits.device
            ce = self._ce(logits, labels)               # auxiliary token CE
            l_local_sum = torch.zeros((), device=device)
            l_global_sum = torch.zeros((), device=device)
            n = 0
            B = input_ids.shape[0]
            for b in range(B):
                lab = labels[b]
                keep = (lab != -100)
                if int(keep.sum()) == 0:
                    continue
                pos = torch.nonzero(keep, as_tuple=False).flatten()  # label positions
                h_last = h[b][pos]                       # (P, hidden)
                gold_bound = lab[pos].clamp(min=0)       # (P,) 0/1
                l_loc, l_glob, _lz, _gs, _ns = self._window_losses(h_last, gold_bound)
                l_local_sum = l_local_sum + l_loc
                l_global_sum = l_global_sum + l_glob
                n += 1
            n = max(n, 1)
            l_local = l_local_sum / n
            l_global = l_global_sum / n
            loss = l_local + l_global + self.ce_weight * ce
        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# metrics / helpers                                                           #
# --------------------------------------------------------------------------- #
def metrics(ep):
    """Window-level token-argmax F1 on the base head — a training monitor only.
    (The reported / submitted numbers come from the semi-CRF `predict` decode.)"""
    logits, labels = ep
    p = logits.argmax(-1); mask = labels != -100
    p, g = p[mask], labels[mask]
    tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": 2 * P * R / max(P + R, 1e-9)}


def _save_hf_checkpoint(model, tok, out_dir):
    """ADDITIVE: also write a plain HuggingFace token-classification checkpoint
    into `out_dir`, alongside the existing model.pt / cfg.json.

    Purpose: predict.py / eval_local.py / diag_preposition.py load with
    AutoModelForTokenClassification.from_pretrained(out_dir) + AutoTokenizer, which
    the custom .pt cannot satisfy. Here `model.enc` IS the trained
    AutoModelForTokenClassification (encoder + boundary classifier); the semi-CRF
    head adds parameters but does NOT alter `model.enc`, so `model.enc` is exactly
    the token-argmax inference model those tools expect. We write it verbatim. This
    mirrors consistency / teacher / char_aug / sub2 _save_hf_checkpoint; it does
    not touch training, the loss, the flags, or the existing .pt/cfg.json outputs
    — it only adds config.json + weights + tokenizer files.

    NB per the pre-registered exception: the semi-CRF's OWN Viterbi decode is
    reached through THIS file's `predict` subcommand (which loads model.pt with
    the scorer weights), not through this token-argmax HF checkpoint.
    """
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


# --------------------------------------------------------------------------- #
# RESCUE 2026-07-10: per-module learning-rate param groups (flag-gated)        #
# --------------------------------------------------------------------------- #
def rescue_param_groups(model, base_lr, base_wd, head_lr, dur_lr):
    """AdamW param groups for the semi-CRF rescue (lr-starvation fix).

    Forensic verdict of record (2026-07-10): with a single lr=5e-5 over 512
    steps the from-scratch duration table's Adam travel ceiling (sum lr_t =
    0.0128) is ~4% of the median Viterbi decision margin (0.316) — the learned
    length prior physically could not act. Groups:

      enc.*                  lr=base_lr  wd=base_wd  (bias/LayerNorm no-decay,
                                                      HF Trainer convention)
      scorer.mlp.* weights   lr=head_lr  wd=base_wd
      scorer.mlp.* biases    lr=head_lr  wd=0
      scorer.len_emb.weight  lr=head_lr  wd=0
      scorer.duration_bias   lr=dur_lr   wd=0   (MUST be 0 — weight decay pulls
                                                 the learned length prior back
                                                 toward zero)

    Every trainable parameter must land in exactly one bucket; an unrouted name
    raises instead of silently training at some default lr.
    """
    def _no_decay(n):
        return n.endswith(".bias") or "LayerNorm" in n  # HF Trainer convention

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    keys = ["enc.decay", "enc.no_decay", "scorer.mlp.weights",
            "scorer.mlp.biases", "scorer.len_emb.weight", "scorer.duration_bias"]
    buckets = {k: [] for k in keys}
    names = {k: [] for k in keys}
    for n, p in named:
        if n == "scorer.duration_bias":
            k = "scorer.duration_bias"
        elif n == "scorer.len_emb.weight":
            k = "scorer.len_emb.weight"
        elif n.startswith("scorer.mlp."):
            k = "scorer.mlp.biases" if n.endswith(".bias") else "scorer.mlp.weights"
        elif n.startswith("enc."):
            k = "enc.no_decay" if _no_decay(n) else "enc.decay"
        else:
            raise ValueError(f"rescue_param_groups: unrouted parameter {n!r} — "
                             "refusing to silently train it at a default lr")
        buckets[k].append(p)
        names[k].append(n)
    spec = [("enc.decay", base_lr, base_wd),
            ("enc.no_decay", base_lr, 0.0),
            ("scorer.mlp.weights", head_lr, base_wd),
            ("scorer.mlp.biases", head_lr, 0.0),
            ("scorer.len_emb.weight", head_lr, 0.0),
            ("scorer.duration_bias", dur_lr, 0.0)]
    groups = []
    for k, lr, wd in spec:
        if not buckets[k]:
            continue
        groups.append({"params": buckets[k], "lr": lr, "weight_decay": wd,
                       "group_name": k, "param_names": names[k]})
    n_grouped = sum(len(g["params"]) for g in groups)
    assert n_grouped == len(named), \
        f"rescue_param_groups coverage: {n_grouped} grouped != {len(named)} trainable"
    return groups


class ParamGroupTrainer(Trainer):
    """ADDITIVE, flag-gated rescue Trainer (mirrors src/train_encoder.py's
    create_optimizer override, lines ~399-419).

    head_lr is None -> defer to HuggingFace's optimizer so the baseline is
    bit-for-bit (cmd_train additionally never instantiates this class unless
    --head-lr is set). Otherwise build the rescue param groups with the SAME
    optimizer class/kwargs the Trainer would use, so only per-group
    lr/weight_decay differ. The Trainer still owns the scheduler, and the HF
    LambdaLR scales every group's lr proportionally through warmup/decay."""

    def __init__(self, *args, head_lr=None, dur_lr=None, **kw):
        super().__init__(*args, **kw)
        self._head_lr = head_lr
        self._dur_lr = dur_lr if dur_lr is not None else head_lr

    def create_optimizer(self):
        if self._head_lr is None or self.optimizer is not None:
            return super().create_optimizer()
        opt_model = self.model
        grouped = rescue_param_groups(opt_model, self.args.learning_rate,
                                      self.args.weight_decay,
                                      self._head_lr, self._dur_lr)
        for g in grouped:
            print(f"[rescue-optim] {g['group_name']:<22} lr={g['lr']:g}  "
                  f"wd={g['weight_decay']:g}  n_params={len(g['params'])}")
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model)
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        # per-group lr/weight_decay come from `grouped`; drop the scalar
        # weight_decay default so it can't override our per-group values.
        optimizer_kwargs.pop("weight_decay", None)
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        return self.optimizer


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #
def cmd_train(a):
    set_seed(a.seed); random.seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model_name)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    tr = load_task(a.task, "train", jsonl_path=a.train_jsonl)
    dv = load_task(a.task, "dev", jsonl_path=a.dev_jsonl)
    pos = sum(sum(d["labels"]) for d in tr); tot = sum(len(d["labels"]) for d in tr)
    pw = a.pos_weight if a.pos_weight else min((tot - pos) / max(pos, 1), 8.0)
    semicrf = (a.semicrf == "on")
    print(f"pos_weight={pw:.2f}  semicrf={a.semicrf}  max_span={a.max_span}  "
          f"ce_weight={a.ce_weight}")
    model = SemiCRFModel(a.model_name, pw, semicrf=semicrf, max_span=a.max_span,
                         len_emb_dim=a.len_emb_dim, mlp_dim=a.mlp_dim,
                         ce_weight=a.ce_weight, keep_per_end=a.keep_per_end,
                         n_extra_tok=1)
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
    if a.head_lr is None:
        # legacy path — the SAME class + construction as always (bit-for-bit)
        tr_ = Trainer(model=model, args=targs, train_dataset=trds, eval_dataset=dvds,
                      data_collator=Collator(tok), compute_metrics=metrics)
    else:
        if not semicrf:
            raise SystemExit("--head-lr targets the from-scratch semi-CRF head: "
                             "it requires --semicrf on")
        dur_lr = a.dur_lr if a.dur_lr is not None else a.head_lr
        print(f"RESCUE param groups ON: head_lr={a.head_lr:g}  dur_lr={dur_lr:g}  "
              f"(enc stays at lr={a.lr:g})")
        tr_ = ParamGroupTrainer(model=model, args=targs, train_dataset=trds,
                                eval_dataset=dvds, data_collator=Collator(tok),
                                compute_metrics=metrics,
                                head_lr=a.head_lr, dur_lr=a.dur_lr)
    tr_.train()
    print("best window dev (token-argmax monitor):", tr_.evaluate())
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    tok.save_pretrained(a.out)
    cfg_out = {"model_name": a.model_name, "semicrf": a.semicrf,
               "max_span": a.max_span, "len_emb_dim": a.len_emb_dim,
               "mlp_dim": a.mlp_dim, "ce_weight": a.ce_weight,
               "keep_per_end": a.keep_per_end}
    if a.head_lr is not None:
        # RESCUE provenance — recorded ONLY when the flag is used, so the
        # default cfg.json stays byte-identical to yesterday's.
        cfg_out["head_lr"] = a.head_lr
        cfg_out["dur_lr"] = a.dur_lr if a.dur_lr is not None else a.head_lr
    json.dump(cfg_out, open(os.path.join(a.out, "cfg.json"), "w"))
    # ADDITIVE: also emit an HF checkpoint so predict.py / eval_local.py /
    # diag_preposition.py can load this run dir (token-argmax). Nothing above is
    # changed.
    _save_hf_checkpoint(model, tok, a.out)
    if semicrf:
        np.savetxt(os.path.join(a.out, "duration_bias.txt"),
                   model.scorer.duration_bias.detach().cpu().numpy())
    print(f"saved -> {a.out}")


# --------------------------------------------------------------------------- #
# predict — semi-CRF's OWN Viterbi decode (per the pre-registered exception)   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cmd_predict(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(a.model, "cfg.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    semicrf = (cfg.get("semicrf", "off") == "on")
    model = SemiCRFModel(cfg["model_name"], semicrf=semicrf,
                         max_span=cfg.get("max_span", 32),
                         len_emb_dim=cfg.get("len_emb_dim", 32),
                         mlp_dim=cfg.get("mlp_dim", 256),
                         ce_weight=cfg.get("ce_weight", 1.0),
                         keep_per_end=cfg.get("keep_per_end", 8), n_extra_tok=1)
    model.load_state_dict(torch.load(os.path.join(a.model, "model.pt"), map_location=dev))
    model.to(dev).eval()
    docs = load_task(a.task, a.split, jsonl_path=a.jsonl)
    preds = []
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        n = len(words)
        # per-word boundary votes across overlapping windows (segment ends).
        votes = {i: [] for i in range(n)}
        s = 0
        while s < n:
            e = min(s + a.window, n)
            enc = tok(words[s:e], is_split_into_words=True, truncation=True,
                      max_length=a.max_length, return_tensors="pt")
            wid = enc.word_ids(0)
            if semicrf:
                out = model.enc(input_ids=enc["input_ids"].to(dev),
                                attention_mask=enc["attention_mask"].to(dev),
                                output_hidden_states=True)
                h = out.hidden_states[-1][0]                # (T, hidden)
                # last-subword positions -> label indices, and their word ids
                lab_pos, lab_word = [], []
                for pos, w in enumerate(wid):
                    if w is None:
                        continue
                    nxt = wid[pos + 1] if pos + 1 < len(wid) else None
                    if nxt != w:
                        lab_pos.append(pos); lab_word.append(w)
                if lab_pos:
                    h_last = h[torch.tensor(lab_pos, device=dev)]
                    win_lab = model.decode_window(h_last)   # 0/1 per label pos
                    mask_wend = getattr(a, "mask_window_end", "off") == "on"
                    for k, w in enumerate(lab_word):
                        if mask_wend and k == len(lab_word) - 1 and (s + w) < n - 1:
                            # RESCUE 2026-07-10 decode fix: decode_window FORCES
                            # a boundary at the window's final label position
                            # (the tiling must end there) — under the mean>=0.5
                            # merge one forced 1 among two windows always wins.
                            # Skip that vote UNLESS this is the document-final
                            # position (a real segment end, kept).
                            continue
                        votes[s + w].append(win_lab[k])
            else:
                # OFF checkpoint: token-argmax fallback (kept for completeness;
                # the ON path is the one this file exists for).
                logit = model.enc(input_ids=enc["input_ids"].to(dev),
                                  attention_mask=enc["attention_mask"].to(dev)).logits[0]
                pr = logit.argmax(-1).cpu().numpy()
                for pos, w in enumerate(wid):
                    if w is None:
                        continue
                    nxt = wid[pos + 1] if pos + 1 < len(wid) else None
                    if nxt != w:
                        votes[s + w].append(int(pr[pos]))
            # advance window (overlap-average / merge stitching, mirrors predict.py)
            last = max([w for w in wid if w is not None], default=-1)
            if s + last >= n - 1:
                break
            s = max(s + last + 1 - a.overlap, s + 1)
        # merge overlapping-window votes: majority (>=0.5) => boundary.
        lab = [1 if (np.mean(votes[i]) if votes[i] else 0) >= 0.5 else 0 for i in range(n)]
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        preds.append(lab)
    write_submission(docs, preds, a.out)
    rate = sum(map(sum, preds)) / max(sum(map(len, preds)), 1)
    print(f"wrote {len(docs)} docs (semicrf={semicrf}, boundary rate={rate:.3f}) -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--task", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="aubmindlab/bert-base-arabertv02")
    t.add_argument("--train-jsonl"); t.add_argument("--dev-jsonl")
    t.add_argument("--semicrf", choices=["on", "off"], default="off",
                   help="off (default) = plain weighted-CE token head, BIT-FOR-BIT; "
                        "on = Filtered Semi-Markov CRF joint loss + Viterbi decode")
    t.add_argument("--max-span", type=int, default=32, help="CAP L on segment length")
    t.add_argument("--len-emb-dim", type=int, default=32)
    t.add_argument("--mlp-dim", type=int, default=256)
    t.add_argument("--ce-weight", type=float, default=1.0,
                   help="weight on the auxiliary token-CE term when semicrf=on "
                        "(keeps the base classifier — used by the HF checkpoint — trained)")
    t.add_argument("--keep-per-end", type=int, default=8,
                   help="filtered-lattice floor: also keep the top-k 'sentence'-"
                        "scoring spans ending at each position (0 = pure argmax filter)")
    t.add_argument("--window", type=int, default=180); t.add_argument("--stride", type=int, default=90)
    t.add_argument("--max-length", type=int, default=512); t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5); t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--head-lr", type=float, default=None,
                   help="RESCUE 2026-07-10: dedicated lr for the from-scratch semi-CRF "
                        "head (scorer.mlp.* and scorer.len_emb.weight). Default None = "
                        "legacy single-lr HF AdamW, bit-for-bit identical to before. "
                        "Requires --semicrf on when set.")
    t.add_argument("--dur-lr", type=float, default=None,
                   help="RESCUE 2026-07-10: lr for scorer.duration_bias only (default: "
                        "the --head-lr value). Its weight decay is FORCED to 0 — decay "
                        "would pull the learned length prior back toward zero.")
    t.add_argument("--pos-weight", type=float, default=None); t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--task", required=True); p.add_argument("--split", default="dev")
    p.add_argument("--model", required=True); p.add_argument("--jsonl"); p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=180)
    p.add_argument("--overlap", type=int, default=60); p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--mask-window-end", choices=["on", "off"], default="off",
                   help="RESCUE 2026-07-10 decode fix: skip the FORCED boundary vote at "
                        "each window's final label position (Viterbi must end every "
                        "window with a boundary — a tiling constraint, not a model "
                        "decision). The document-final position keeps its vote. "
                        "off (default) = legacy vote merge, bit-for-bit.")
    p.set_defaults(func=cmd_predict)
    a = ap.parse_args(); a.func(a)
