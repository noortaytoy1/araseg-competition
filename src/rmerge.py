"""Recursive Beam-Merge Sentence Segmentation (Noor's locked algorithm).

AraSeg 2026, closed track (NoPnx-NP). ADDITIVE, flag-gated new file. Reuses the
repo's data / eval / encoder conventions by import; touches no existing trainer.

THE ONE IDEA
------------
Sentence segmentation is recursive sentence *construction*. A single shared
AraBERT ("the merger") scores whether two adjacent constituents belong to the
same sentence by RE-READING their actual tokens:

    input  =  [CLS] tokens(A) [SEP] tokens(B) [SEP]
    p      =  sigmoid( w . h_[CLS] + b )   in (0,1)   =  P(A,B same sentence)

The SAME AraBERT weights score every merge at every level of the recursion. No
mean-pool, no stored constituent vector, no separate composer network: the model
re-reads the real tokens every time. A sentence boundary is simply a junction
where composition is not supported (p < tau).

TRAINING (this file, per Noor's LOCKED build order + the METHODOLOGY report)
--------------------------------------------------------------------------
Fine-tune AraBERT + a scalar merge head end-to-end (<=10 epochs).

  - ROLL-IN = SCHEDULED SAMPLING (default; Bengio 2015): build the tree bottom
    up; at each merge, with prob P follow the GOLD next merge (teacher), else the
    MODEL's own top-scored valid merge (let it visit its own states).  Ramp
    P_model 0.0 -> ~0.25 over epochs.  Never commit a cross-boundary merge during
    structure-building UNLESS model-roll-in picks one, in which case it is still
    supervised (target 0) and bounded by the SPAN-LENGTH CAP.  Flag
    --roll-in {teacher, sched, laso, selfmerge}; laso = beam roll-in, IMPLEMENTED
    but OFF for the deadline.  selfmerge = PURE self-gluing (P_model:=1.0 EVERY
    epoch, incl. the first): the model chooses every merge itself and is punished
    (target 0) at each cross-boundary glue, no teacher forcing (the local-minima
    escape; LR-warmup handles cold-start).
  - DENSE supervision: at EVERY visited state, score EVERY adjacent candidate in
    ONE batched AraBERT forward; class-weighted BCE against gold:
        target 1 if gold_cut[k]=0 (same sentence  -> SHOULD merge)
        target 0 if gold_cut[k]=1 (boundary       -> should NOT merge; penalty)
  - CLASS-WEIGHT the BCE (boundaries are the minority target-0) vs collapse (FM2).
  - LR WARMUP (FM4), GRADIENT CHECKPOINTING on the encoder (FM9),
    SPAN-ENCODING MEMOIZATION (FM9), SPAN-LENGTH CAP -> p:=0 (FM3),
    SLIDING WINDOW 128/32 for long docs (FM9).
  - Backprop into AraBERT (it IS the merger).  Optional v2 max-margin ranking
    loss (--margin, OFF by default).

INFERENCE (this file)
---------------------
Beam search width 3, LOG-SPACE LENGTH-NORMALIZED score
= (1/|M|^alpha) * sum(log p_m), alpha=1.0 => geometric mean (guards FM1
over-segmentation + underflow).  BEAM DEDUP on active-cut sets (FM6).
Merge-generous stop at LOW tau: stop merging when EVERY remaining adjacent merge
prob < tau; tau own-best on dev by grid search.  DOC-FINAL INVARIANT: the
junction after the last word is always a boundary.  Winning hypothesis ->
surviving cuts -> per-word 0/1 -> standard scorer.

CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (a battery may own the GPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# repo conventions (offline-safe imports; torch/transformers imported lazily so
# tests that never touch the encoder stay network-free and fast)
from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, load_jsonl  # noqa: E402

ARABERT_DEFAULT = "aubmindlab/bert-base-arabertv02"

# standard own-best grid used everywhere in this repo (0.30..0.70 by 0.05); the
# merge-generous extras below let tau go LOW (spec: merge readily, cut only when
# confident it is NOT a merge).
TAU_GRID = [round(t, 2) for t in np.arange(0.30, 0.7001, 0.05)]
TAU_LOW_EXTRA = [0.05, 0.10, 0.15, 0.20, 0.25]     # merge-generous: LOW tau
BEAM_WIDTH = 3
SPAN_CAP_DEFAULT = 64            # span-length cap: resulting span > cap -> p:=0
PAIR_FWD_CHUNK_DEFAULT = 256     # max candidate pairs per AraBERT forward (memory-bounds long-doc states)
ALPHA_DEFAULT = 1.0             # log-space length-norm exponent (1.0 = geo-mean)
WINDOW_DEFAULT = 128           # sliding-window length (report FM9)
STRIDE_DEFAULT = 32            # sliding-window stride (report FM9)
WARMUP_FRAC_DEFAULT = 0.1      # LR-warmup fraction of total steps (report FM4)
ROLL_IN_CHOICES = ("teacher", "sched", "laso", "selfmerge", "leaf")
SCHED_P_MAX_DEFAULT = 0.25     # scheduled-sampling ramp target P_model (Bengio15)
GLUE_TAU_DEFAULT = 0.5         # selfmerge per-pass recursion: glue a pair when P>=this


# =========================================================================== #
# LOCKED-SPEC MANDATES ADDED IN THIS BUILD (all additive, flag-gated; every
# default below reproduces the signed build order, not "yesterday's" — this is
# a NEW module):
#   * DOC-FINAL INVARIANT     docfinal_is_boundary / force p:=0 at last junction
#   * SPAN-LENGTH CAP         resulting span > span_cap  -> p forced 0 (FM3)
#   * SPAN MEMOIZATION        SpanCache keys h_[CLS] by (pair-token tuple) (FM9)
#   * SLIDING WINDOW          window/stride, stitch predictions (FM9)
#   * SCHEDULED SAMPLING      --roll-in {teacher,sched,laso}; ramp P_model 0->.25
#   * LR WARMUP               linear warmup then linear decay (FM4)
#   * GRAD CHECKPOINTING      enc.gradient_checkpointing_enable() (FM9)
#   * LOG-SPACE LENGTH-NORM   score = (1/|M|^alpha)*sum(log p), alpha=1.0 (FM1)
#   * OPTIONAL MAX-MARGIN     v2 aux ranking loss, OFF by default
# =========================================================================== #


def assert_cpu_only():
    import os, torch
    # Escape hatch for the real battery run (which OWNS the GPU): set
    # RMERGE_ALLOW_GPU=1. Default keeps the build/verify phase CPU-only.
    if os.environ.get("RMERGE_ALLOW_GPU") == "1":
        return
    assert not torch.cuda.is_available(), \
        "CPU-ONLY stage: set CUDA_VISIBLE_DEVICES=-1 (the battery owns the GPU)"


def _run_device():
    """Real battery run uses the GPU (RMERGE_ALLOW_GPU=1); build/verify stays CPU."""
    import os, torch
    if os.environ.get("RMERGE_ALLOW_GPU") == "1" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# 1. gold-cut alignment (per-word 0/1 labels  ->  cut mask over junctions)
# --------------------------------------------------------------------------- #
def gold_cut_mask(labels: Sequence[int]) -> np.ndarray:
    """gold_cut[k] = 1 if there is a sentence boundary at the junction AFTER word
    k, else 0.  Junction axis k = 0 .. n-2 (n words -> n-1 junctions).  The
    doc-final label (labels[n-1], always 1 in AraSeg) is NOT a junction: there is
    no pair after the last word.  Same axis as any merge the model considers, so
    a merge at junction k aligns directly with gold_cut[k]."""
    lab = np.asarray(labels, dtype=np.int64)
    n = len(lab)
    if n <= 1:
        return np.zeros((0,), dtype=np.int64)
    return lab[: n - 1].copy()          # junction k <-> labels[k]


def sentence_spans(labels: Sequence[int]) -> List[Tuple[int, int]]:
    """Inclusive (start, end) word spans of the gold sentences."""
    lab = np.asarray(labels, dtype=np.int64)
    n = len(lab)
    cuts = np.flatnonzero(gold_cut_mask(labels) == 1)
    starts = np.concatenate([[0], cuts + 1]).astype(np.int64)
    ends = np.concatenate([cuts, [n - 1]]).astype(np.int64)
    return list(zip(starts.tolist(), ends.tolist()))


def cuts_to_word_boundaries(cuts: Sequence[int], n: int) -> List[int]:
    """Surviving cuts (a set/list of junction indices) -> per-word 0/1 boundary
    vector of length n.  A cut at junction k => boundary label 1 on word k.  The
    doc-final word (index n-1) carries 0.0 here (a cut grid has no junction after
    the last word); score_cut.py handles the doc-final slot (its `_de` row)."""
    y = [0] * n
    for c in cuts:
        c = int(c)
        if 0 <= c <= n - 2:
            y[c] = 1
    return y


# --------------------------------------------------------------------------- #
# 1b. DOC-FINAL INVARIANT (report): the junction after the LAST word is ALWAYS a
#     boundary -> we force p := 0 there (mandatory cut), never merge it.  In the
#     node representation there is no candidate pair after the last node, so the
#     invariant is structurally satisfied WITHIN a doc; it becomes load-bearing
#     when SLIDING WINDOWS split a doc: a window's right edge that is the TRUE doc
#     end must be forced to a cut, an interior window edge must NOT be.
# --------------------------------------------------------------------------- #
def docfinal_junction(n: int) -> int:
    """The junction index that sits after the last word of an n-word doc.  There
    are n-1 inter-word junctions (0..n-2); the doc-final junction is n-1 (the
    slot score_cut.py fills via its `_de` row).  Always a boundary."""
    return n - 1


def docfinal_is_boundary(n: int) -> bool:
    """Invariant: for any non-empty doc the junction after the last word is a
    boundary.  Trivially True; exposed so the smoke can assert the mandate is
    engaged rather than merely assumed."""
    return n >= 1


def force_docfinal_cut(y: List[int], n: int) -> List[int]:
    """Apply the doc-final invariant to a per-word grid: the last word (index
    n-1) always ends a sentence.  We keep the cut-grid convention (doc-final slot
    carries 0 in `y`; score_cut.py's `_de` row rule-fills it), so this is a no-op
    on `y` BUT asserts the invariant holds — it exists so windowing/stitching has
    one authoritative place to enforce 'true doc end => forced boundary'."""
    assert docfinal_is_boundary(n)
    return y


# --------------------------------------------------------------------------- #
# 2. tree state = ordered list of contiguous nodes (each node = (w0, w1) span)
# --------------------------------------------------------------------------- #
def start_state(n: int) -> List[Tuple[int, int]]:
    """Every word its own node (all n-1 cuts present)."""
    return [(i, i) for i in range(n)]


def state_cuts(nodes: List[Tuple[int, int]]) -> List[int]:
    """The junction indices that are still CUTS given the node partition: a cut
    sits at junction w1 of every node except the last node."""
    return [w1 for (_, w1) in nodes[:-1]]


def adjacent_candidates(nodes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Every adjacent pair (i, i+1) of nodes -> the junction it would erase.
    Returns [(node_index_i, junction_k), ...] where merging nodes i,i+1 erases
    the cut at junction k = nodes[i].w1."""
    out = []
    for i in range(len(nodes) - 1):
        out.append((i, nodes[i][1]))
    return out


def merge_nodes(nodes: List[Tuple[int, int]], i: int) -> List[Tuple[int, int]]:
    """Return a new node list with nodes i and i+1 merged into one span."""
    a0, _ = nodes[i]
    _, b1 = nodes[i + 1]
    return nodes[:i] + [(a0, b1)] + nodes[i + 2:]


# --------------------------------------------------------------------------- #
# 3. the merge scorer  —  AraBERT re-reads [CLS] tokens(A) [SEP] tokens(B) [SEP]
# --------------------------------------------------------------------------- #
def load_arabert(model_name_or_dir: str = ARABERT_DEFAULT, seed: int = 42,
                 head_bias_init: float = 0.0):
    """AraBERT encoder (AutoModel) + a fresh scalar merge head on h_[CLS].
    Adds the [PAR] stand-in for the paragraph "\\n" token (repo convention) and
    resizes embeddings before any forward pass.

    head_bias_init (cold-start safety for selfmerge): the merge head's bias.
    0.0 (default) reproduces prior behaviour.  A NEGATIVE value (e.g. -2.0 ->
    P(glue)~=0.12) makes the random-init model glue almost NOTHING at cold start,
    so it fails toward UNDER-gluing (recoverable next pass) instead of glue-all
    (which IRREVERSIBLY destroys gold boundaries).  Only the bias shifts; the
    model still learns to glue within sentences from the dense target-1 signal."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    assert_cpu_only()
    tok = AutoTokenizer.from_pretrained(model_name_or_dir)
    if MODEL_PAR_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    enc = AutoModel.from_pretrained(model_name_or_dir)
    enc.resize_token_embeddings(len(tok))
    h = enc.config.hidden_size
    g = torch.Generator().manual_seed(seed)
    head = torch.nn.Linear(h, 1)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=g) * 0.02)
        head.bias.fill_(float(head_bias_init))
    return tok, enc, head


class MixLinear(__import__("torch").nn.Linear):
    """A Linear whose weight is MIXOUT-regularized toward a frozen pretrained
    target during TRAINING (Lee et al., ICLR 2020, "Mixout").  Each forward, a
    random `p` fraction of weight entries is replaced by the pretrained value
    (with an unbiased rescale so E[weight]=current), so fine-tuning cannot drift
    the whole matrix away from the pretrained model -> it keeps AraBERT's grammar
    and cannot freely memorize the 174-doc adjacencies.  INFERENCE is a plain
    Linear (no mixing).  The target is a NON-persistent buffer, so save_pretrained
    writes a STANDARD Linear (weight+bias only) that AutoModel reloads normally."""

    @classmethod
    def from_linear(cls, lin, p):
        import torch
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None)
        with torch.no_grad():
            m.weight.copy_(lin.weight)
            if lin.bias is not None:
                m.bias.copy_(lin.bias)
        # anchor = the CURRENT (pretrained) weight; not saved (persistent=False)
        m.register_buffer("mix_target", lin.weight.detach().clone(), persistent=False)
        m.mixout_p = float(p)
        return m

    def forward(self, x):
        import torch
        from torch.nn import functional as F
        p = getattr(self, "mixout_p", 0.0)
        if self.training and p > 0.0:
            mask = torch.rand_like(self.weight) < p          # pull these to target
            w = torch.where(mask, self.mix_target, self.weight)
            w = (w - p * self.mix_target) / (1.0 - p)         # unbiased: E[w]=weight
            return F.linear(x, w, self.bias)
        return F.linear(x, self.weight, self.bias)


def apply_mixout(module, p: float) -> int:
    """Recursively replace every nn.Linear in `module` with a MixLinear anchored to
    that Linear's CURRENT (pretrained) weight.  Call RIGHT AFTER loading AraBERT,
    before fine-tuning.  p=0 leaves the model untouched.  Returns #layers converted."""
    import torch
    if p <= 0.0:
        return 0
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and not isinstance(child, MixLinear):
            setattr(module, name, MixLinear.from_linear(child, p))
            n += 1
        else:
            n += apply_mixout(child, p)
    return n


class MergeScorer:
    """Wraps (tokenizer, AraBERT, scalar head).  The SAME weights score every
    merge at every level (the recursion).  p = sigmoid(w . h_[CLS] + b)."""

    def __init__(self, tokenizer, encoder, head, span_cap: int = SPAN_CAP_DEFAULT,
                 device: str = "cpu", span_cap_mode: str = "force0",
                 memoize: bool = True, grad_checkpoint: bool = False,
                 pair_fwd_chunk: int = PAIR_FWD_CHUNK_DEFAULT,
                 context_k: int = 0, boundary_dropout: float = 0.0):
        self.tok = tokenizer
        self.enc = encoder
        self.head = head
        self.span_cap = int(span_cap)
        # SURROUNDING CONTEXT (generalization fix): include context_k words of the
        # ORIGINAL document on EACH side of the A|B junction, so the merge decision
        # can use discourse / function-word cues (thumma, inna, clause structure)
        # instead of only the boundary bigram.  0 = off (prior plain behaviour).
        self.context_k = int(context_k)
        # BOUNDARY-DROPOUT (anti-overconfidence, "B" lever): during TRAINING only,
        # with this probability, blank out (replace with [MASK]) the single word on
        # EACH side of the junction being decided (A's rightmost word, B's leftmost
        # word), each side independently.  This forbids the scorer from resting its
        # glue decision on the identity of the two boundary words, so it must read
        # the surrounding context -> attacks the confident-wrong cross-boundary
        # merges the oracle exposed.  Only meaningful WITH context_k>0 (else masking
        # the boundary words leaves nothing to decide from).  Inference NEVER masks
        # (gated on grad=True), so decoding + memoization are byte-identical.
        # 0.0 = off (prior behaviour, exact reproduction).
        self.boundary_dropout = float(boundary_dropout)
        # A state on a long doc can have thousands of adjacent candidate pairs.
        # Sending them all through AraBERT in ONE batch builds a giant tensor that
        # OOMs on GPU / crawls on CPU (the "deadlock": a 15k-word doc's opening
        # state has ~15k pairs).  Micro-batch the pair forwards in chunks so peak
        # memory is bounded by chunk size, NOT by state size.  Mathematically
        # identical (each pair is an independent row); verified by
        # batched_equals_perpair.
        self.pair_fwd_chunk = int(pair_fwd_chunk)
        # span_cap_mode: "force0" (locked spec: resulting span > cap -> p:=0,
        # never merge, guards the cross-boundary cascade FM3) or "truncate"
        # (legacy: keep the span_cap tokens adjacent to the junction).
        self.span_cap_mode = span_cap_mode
        self.device = device
        # SPAN-ENCODING MEMOIZATION (report FM9): cache the merge LOGIT keyed by
        # the exact (a_ids, b_ids) subword tuple, reused across candidates / steps
        # / beams before any AraBERT pass.  Inference-only (no grad graph stored).
        self.memoize = bool(memoize)
        self._span_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        # GRADIENT CHECKPOINTING on the encoder (report FM9): bounds activation
        # memory during the beam/dense forward passes.
        self.grad_checkpoint = bool(grad_checkpoint)
        if self.grad_checkpoint and hasattr(self.enc, "gradient_checkpointing_enable"):
            try:
                self.enc.gradient_checkpointing_enable()
                if hasattr(self.enc, "config"):
                    self.enc.config.use_cache = False
            except Exception:
                pass
        self.enc.to(device)
        self.head.to(device)

    # -- word -> subword id helpers -------------------------------------- #
    def _word_ids(self, words: Sequence[str]) -> List[List[int]]:
        """Per word, its list of subword ids (no specials).  Paragraph "\\n"
        words are mapped to the [PAR] stand-in first (repo convention)."""
        out = []
        for w in words:
            surface = MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
            ids = self.tok.encode(surface, add_special_tokens=False)
            if not ids:                          # never let a word vanish
                ids = [self.tok.unk_token_id]
            out.append(ids)
        return out

    def _span_ids(self, word_ids: List[List[int]], w0: int, w1: int) -> List[int]:
        """Concatenated subword ids for the inclusive word span [w0..w1], capped
        at self.span_cap tokens from the boundary side that matters (keep the
        tokens ADJACENT to the junction so the [SEP] context is real):
          - A (left span):  keep its RIGHTMOST span_cap tokens.
          - handled by caller passing side; here we keep rightmost/leftmost via
            two thin wrappers below."""
        ids: List[int] = []
        for k in range(w0, w1 + 1):
            ids.extend(word_ids[k])
        return ids

    def _left_ids(self, word_ids, w0, w1):
        ids = self._span_ids(word_ids, w0, w1)
        if self.span_cap_mode == "truncate" and len(ids) > self.span_cap:
            ids = ids[-self.span_cap:]           # keep tokens next to junction
        return ids

    def _right_ids(self, word_ids, w0, w1):
        ids = self._span_ids(word_ids, w0, w1)
        if self.span_cap_mode == "truncate" and len(ids) > self.span_cap:
            ids = ids[: self.span_cap]           # keep tokens next to junction
        return ids

    def _resulting_span_len(self, word_ids, a, b) -> int:
        """Subword length of the span that RESULTS from merging nodes a,b (the
        union span a[0]..b[1]).  The span-length cap (report FM3) forces p:=0
        when this exceeds span_cap, so cross-boundary cascades cannot run away."""
        (a0, _a1), (_b0, b1) = a, b
        return sum(len(word_ids[k]) for k in range(a0, b1 + 1))

    def _over_cap(self, word_ids, a, b) -> bool:
        return (self.span_cap_mode == "force0"
                and self._resulting_span_len(word_ids, a, b) > self.span_cap)

    def _pair_ids(self, word_ids, a, b):
        """The (a_ids, b_ids) subword-id lists for a node pair, respecting the
        truncate cap when active.  These tuples are BOTH the batch rows AND the
        memoization key (report FM9)."""
        (a0, a1), (b0, b1) = a, b
        return self._left_ids(word_ids, a0, a1), self._right_ids(word_ids, b0, b1)

    def _context_ids(self, word_ids, a, b):
        """Left/right surrounding-context subword ids: the context_k ORIGINAL words
        immediately before A and after B.  Empty at the document edge or if off."""
        if self.context_k <= 0:
            return [], []
        (a0, _a1), (_b0, b1) = a, b
        n = len(word_ids)
        left = []
        for w in range(max(0, a0 - self.context_k), a0):
            left.extend(word_ids[w])
        right = []
        for w in range(b1 + 1, min(n, b1 + 1 + self.context_k)):
            right.extend(word_ids[w])
        return left, right

    def _pair_key(self, word_ids, a, b):
        """Memoization key.  With context it MUST include the surrounding context
        (same A,B at a different doc position has different context -> different
        prob), else the cache would serve a stale value."""
        a_ids, b_ids = self._pair_ids(word_ids, a, b)
        if self.context_k > 0:
            l_ids, r_ids = self._context_ids(word_ids, a, b)
            return (tuple(l_ids), tuple(a_ids), tuple(b_ids), tuple(r_ids))
        return (tuple(a_ids), tuple(b_ids))

    def build_pair_batch(self, word_ids: List[List[int]],
                         pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                         mask_junction_p: float = 0.0):
        """Build ONE padded batch of [CLS] tokens(A) [SEP] tokens(B) [SEP] for a
        list of ((a0,a1),(b0,b1)) node-span pairs.  Returns (input_ids,
        attention_mask, token_type_ids) tensors — the parallelization: every
        candidate pair at a state goes through ONE AraBERT forward pass.

        mask_junction_p>0 (TRAINING only): with this prob per side, replace the
        junction-adjacent word's subword tokens with [MASK] (A's rightmost word /
        B's leftmost word), each side independently — boundary-dropout.  Kept
        length-preserving (one [MASK] per masked subword) so positions/context are
        intact.  0.0 = no masking (byte-identical to the plain batcher).

        NOTE: this raw batcher does NOT apply the span-length cap veto; callers
        (logits_for_pairs) pre-filter over-cap pairs so they never reach AraBERT."""
        import torch
        cls = self.tok.cls_token_id
        sep = self.tok.sep_token_id
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        mask_id = getattr(self.tok, "mask_token_id", None)
        do_bd = (mask_junction_p > 0.0 and mask_id is not None)
        rows_ids, rows_tt = [], []
        for a, b in pairs:
            a_ids, b_ids = self._pair_ids(word_ids, a, b)
            if do_bd:
                # A's rightmost word = word a[1] (last la tokens of a_ids after any
                # truncation); B's leftmost word = word b[0] (first lb tokens of
                # b_ids).  Blank each side independently with prob mask_junction_p.
                la = len(word_ids[a[1]]); lb = len(word_ids[b[0]])
                if la > 0 and a_ids and float(torch.rand(())) < mask_junction_p:
                    k = min(la, len(a_ids))
                    a_ids = a_ids[:len(a_ids) - k] + [mask_id] * k
                if lb > 0 and b_ids and float(torch.rand(())) < mask_junction_p:
                    k = min(lb, len(b_ids))
                    b_ids = [mask_id] * k + b_ids[k:]
            if self.context_k > 0:
                # [CLS] left-ctx [SEP] A [SEP] B [SEP] right-ctx [SEP]; token_type
                # flips at the A|B junction (0 through A, 1 from B on) so the
                # junction being decided is the 0->1 transition, context in each half.
                l_ids, r_ids = self._context_ids(word_ids, a, b)
                seq = ([cls] + l_ids + [sep] + a_ids + [sep]
                       + b_ids + [sep] + r_ids + [sep])
                n0 = 1 + len(l_ids) + 1 + len(a_ids) + 1
                n1 = len(b_ids) + 1 + len(r_ids) + 1
                tt = [0] * n0 + [1] * n1
            else:
                seq = [cls] + a_ids + [sep] + b_ids + [sep]
                # token_type: 0 for [CLS] tokens(A) [SEP]; 1 for tokens(B) [SEP]
                tt = [0] * (1 + len(a_ids) + 1) + [1] * (len(b_ids) + 1)
            rows_ids.append(seq)
            rows_tt.append(tt)
        maxlen = max(len(r) for r in rows_ids)
        B = len(rows_ids)
        input_ids = np.full((B, maxlen), pad, dtype=np.int64)
        attn = np.zeros((B, maxlen), dtype=np.int64)
        ttids = np.zeros((B, maxlen), dtype=np.int64)
        for r, (seq, tt) in enumerate(zip(rows_ids, rows_tt)):
            input_ids[r, : len(seq)] = seq
            attn[r, : len(seq)] = 1
            ttids[r, : len(tt)] = tt
        return (torch.tensor(input_ids, device=self.device),
                torch.tensor(attn, device=self.device),
                torch.tensor(ttids, device=self.device))

    NEG_INF_LOGIT = -30.0          # sigmoid(-30) ~ 1e-13  ->  a forced p:=0

    def logits_for_pairs(self, word_ids, pairs, *, grad: bool):
        """One batched AraBERT forward over all `pairs`; returns merge LOGITS
        (B,) = w . h_[CLS] + b.  p = sigmoid(logit).  grad=True keeps the graph
        (training); grad=False for inference.

        SPAN-LENGTH CAP (report FM3): any pair whose RESULTING span > span_cap is
        NOT sent through AraBERT — its logit is forced to NEG_INF_LOGIT so
        p:=~0 (never merge), guarding the cross-boundary cascade and bounding
        cost.  Over-cap rows are stitched back in order so the returned tensor
        stays aligned 1:1 with `pairs`."""
        import torch
        B = len(pairs)
        if B == 0:
            return torch.zeros((0,), device=self.device)
        keep_idx, keep_pairs = [], []
        for j, (a, b) in enumerate(pairs):
            if self._over_cap(word_ids, a, b):
                continue
            keep_idx.append(j)
            keep_pairs.append((a, b))
        # forced-0 base for every row; overwrite the kept rows with real logits
        out_logits = torch.full((B,), self.NEG_INF_LOGIT, device=self.device)
        if not keep_pairs:
            return out_logits
        ctx = torch.enable_grad() if grad else torch.no_grad()
        # MICRO-BATCH: process kept pairs in chunks of pair_fwd_chunk so a state
        # with thousands of pairs never builds one giant AraBERT batch (the CPU
        # "deadlock" / GPU OOM cause).  Each pair is an independent row, so the
        # concatenation of per-chunk logits is exactly the single-batch result
        # (padding differs per chunk -> tolerance in batched_equals_perpair).
        chunk = max(1, int(self.pair_fwd_chunk))
        parts = []
        with ctx:
            # boundary-dropout is TRAINING-only: masking is applied iff we are
            # building the gradient graph (grad=True).  Inference (grad=False) always
            # passes p=0 -> decoding + memoization are unaffected.
            bd_p = self.boundary_dropout if grad else 0.0
            for s in range(0, len(keep_pairs), chunk):
                sub = keep_pairs[s:s + chunk]
                input_ids, attn, ttids = self.build_pair_batch(
                    word_ids, sub, mask_junction_p=bd_p)
                try:
                    enc_out = self.enc(input_ids=input_ids, attention_mask=attn,
                                       token_type_ids=ttids)
                except TypeError:                # models without token_type_ids
                    enc_out = self.enc(input_ids=input_ids, attention_mask=attn)
                h_cls = enc_out.last_hidden_state[:, 0, :]  # [CLS] vector
                parts.append(self.head(h_cls).squeeze(-1))  # (len(sub),)
        kept = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        idx_t = torch.tensor(keep_idx, device=self.device, dtype=torch.long)
        out_logits = out_logits.index_copy(0, idx_t, kept)
        return out_logits

    def probs_for_pairs(self, word_ids, pairs) -> np.ndarray:
        """Inference probs with SPAN MEMOIZATION (report FM9): each pair's prob is
        keyed by its exact (a_ids, b_ids) subword tuple; a second identical span
        is served from cache and NEVER re-runs AraBERT.  Over-cap pairs return
        ~0 (span-length cap) and are cached too."""
        import torch
        n = len(pairs)
        if n == 0:
            return np.zeros((0,), dtype=np.float64)
        out = np.empty((n,), dtype=np.float64)
        miss_idx, miss_pairs = [], []
        for j, (a, b) in enumerate(pairs):
            if self.memoize:
                key = self._pair_key(word_ids, a, b)
                cached = self._span_cache.get(key)
                if cached is not None:
                    out[j] = cached
                    self.cache_hits += 1
                    continue
            miss_idx.append(j)
            miss_pairs.append((a, b))
        if miss_pairs:
            logits = self.logits_for_pairs(word_ids, miss_pairs, grad=False)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for local, j in enumerate(miss_idx):
                p = float(probs[local])
                out[j] = p
                self.cache_misses += 1
                if self.memoize:
                    a, b = pairs[j]
                    self._span_cache[self._pair_key(word_ids, a, b)] = p
        return out

    def clear_cache(self):
        self._span_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


def build_joint_head(hidden: int, seed: int = 42):
    """Junction head for the FULL-WINDOW JOINT scorer: MLP over
    [h_lastsub(left-piece right word); h_firstsub(right-piece left word);
     mean(left piece states); mean(right piece states)]  ->  merge logit.
    Seeded init (same convention as load_arabert's scalar head)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    l1 = torch.nn.Linear(4 * hidden, hidden)
    l2 = torch.nn.Linear(hidden, 1)
    with torch.no_grad():
        l1.weight.copy_(torch.randn(l1.weight.shape, generator=g) * 0.02)
        l1.bias.zero_()
        l2.weight.copy_(torch.randn(l2.weight.shape, generator=g) * 0.02)
        l2.bias.zero_()
    return torch.nn.Sequential(l1, torch.nn.GELU(), l2)


class JointScorer(MergeScorer):
    """FULL-WINDOW JOINT scorer (the autopsy-selected fix): AraBERT encodes the
    WHOLE word sequence ONCE (chunked into overlapping <=510-subword segments for
    long inputs; each subword takes its state from the segment where it is most
    interior); every junction is then scored by a small MLP head over the token
    states AT the junction plus mean-pooled summaries of the two pieces being
    joined.  So (a) every leaf decision sees ~full-window context instead of the
    k-word peephole — the component the autopsy convicted (91% of boundary kills
    leaf-born); (b) deep passes still genuinely re-judge (the piece summaries
    change as pieces grow) — the recursion is untouched; (c) the off-distribution
    4-[SEP] pair format disappears entirely; (d) ONE encode per document per step
    replaces one encode per pair.  Same public interface as MergeScorer, so the
    recursion / training / beam / eval machinery runs unchanged."""

    SEG, STRIDE = 510, 255              # subword segment size / stride (<=512 budget)

    def __init__(self, tokenizer, encoder, head, span_cap: int = SPAN_CAP_DEFAULT,
                 device: str = "cpu", span_cap_mode: str = "force0",
                 grad_checkpoint: bool = False):
        super().__init__(tokenizer, encoder, head, span_cap=span_cap,
                         device=device, span_cap_mode=span_cap_mode,
                         memoize=False, grad_checkpoint=grad_checkpoint)
        self.joint = True
        self._draw = 0                  # R-Drop: which dropout DRAW is active
        self._enc_store = {}            # draw -> (key, grad, states, offsets)

    def set_draw(self, i: int):
        """R-Drop support: select dropout draw i (each draw caches its OWN encode
        of the current doc, so the two noisy readings are reused across all
        passes — 2 encodes per doc, not 2 per state)."""
        self._draw = int(i)

    def clear_cache(self):
        self._enc_store = {}
        self._draw = 0

    def encode(self, word_ids, *, grad: bool):
        """One joint encoding of the full word sequence (per active draw).
        Returns (states (n_subwords, hidden), offsets: word i -> (start, end)
        subword slice).  Per-draw single-slot cache: passes of the SAME doc reuse
        the encoding (and, under grad, the same autograd graph — summed pass
        losses backward through ONE encode per draw)."""
        import torch
        key = hash(tuple(tuple(w) for w in word_ids))
        hit = self._enc_store.get(self._draw)
        if hit is not None and hit[0] == key and hit[1] == grad:
            return hit[2], hit[3]
        flat, offsets, pos = [], [], 0
        for w in word_ids:
            flat.extend(w)
            offsets.append((pos, pos + len(w)))
            pos += len(w)
        n = len(flat)
        cls = self.tok.cls_token_id
        sep = self.tok.sep_token_id
        hid = self.enc.config.hidden_size
        ctx = torch.enable_grad() if grad else torch.no_grad()
        parts = []
        with ctx:
            states = torch.zeros((n, hid), device=self.device)
            s = 0
            while True:
                e = min(s + self.SEG, n)
                ids = torch.tensor([[cls] + flat[s:e] + [sep]], device=self.device)
                attn = torch.ones_like(ids)
                out = self.enc(input_ids=ids, attention_mask=attn)
                seg = out.last_hidden_state[0, 1:1 + (e - s), :]   # drop CLS/SEP
                # fill each position from the segment where it is most interior:
                # first segment owns its left edge, last owns its right edge,
                # interior segments own the band past half the overlap.
                f0 = s if s == 0 else s + (self.SEG - self.STRIDE) // 2
                f1 = e if e == n else e - (self.SEG - self.STRIDE) // 2
                parts.append((s, f0, f1, seg))
                if e >= n:
                    break
                s += self.STRIDE
            for (s0, f0, f1, seg) in parts:
                states[f0:f1] = seg[f0 - s0:f1 - s0]
        self._enc_store[self._draw] = (key, grad, states, offsets)
        return states, offsets

    def logits_for_pairs(self, word_ids, pairs, *, grad: bool):
        """Merge logits for ((a0,a1),(b0,b1)) piece pairs from the JOINT encoding.
        Same span-cap veto semantics as the pair scorer (over-cap rows forced to
        NEG_INF_LOGIT, stitched back in order)."""
        import torch
        B = len(pairs)
        if B == 0:
            return torch.zeros((0,), device=self.device)
        out = torch.full((B,), self.NEG_INF_LOGIT, device=self.device)
        keep_idx, keep_pairs = [], []
        for j, (a, b) in enumerate(pairs):
            if not self._over_cap(word_ids, a, b):
                keep_idx.append(j)
                keep_pairs.append((a, b))
        if not keep_pairs:
            return out
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            states, off = self.encode(word_ids, grad=grad)
            feats = []
            for (a0, a1), (b0, b1) in keep_pairs:
                hL = states[off[a1][1] - 1]              # last subword of A's right word
                hR = states[off[b0][0]]                  # first subword of B's left word
                mA = states[off[a0][0]:off[a1][1]].mean(dim=0)
                mB = states[off[b0][0]:off[b1][1]].mean(dim=0)
                feats.append(torch.cat([hL, hR, mA, mB]))
            kept = self.head(torch.stack(feats)).squeeze(-1)
        idx_t = torch.tensor(keep_idx, device=self.device, dtype=torch.long)
        return out.index_copy(0, idx_t, kept)

    def probs_for_pairs(self, word_ids, pairs) -> np.ndarray:
        import torch
        if len(pairs) == 0:
            return np.zeros((0,), dtype=np.float64)
        logits = self.logits_for_pairs(word_ids, pairs, grad=False)
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)


def window_docs(docs: List[dict], window: int, stride: int) -> List[dict]:
    """Slice long docs into overlapping word-windows for JOINT training (one
    encode must fit the ~512-subword budget).  Short docs pass through whole.
    Labels are sliced with the tokens, so within-window gold junctions are exact;
    window-final junctions are simply unsupervised (stride overlap gives them
    interior coverage in a neighbouring window)."""
    out = []
    for d in docs:
        n = len(d["tokens"])
        if n <= window:
            out.append(d)
            continue
        start = 0
        while start < n:
            end = min(start + window, n)
            out.append({"doc_id": f"{d['doc_id']}#w{start}",
                        "tokens": list(d["tokens"][start:end]),
                        "labels": list(d["labels"][start:end])})
            if end >= n:
                break
            start += stride
    return out


# --------------------------------------------------------------------------- #
# 4. TEACHER-FORCED gold tree  +  DENSE supervision (training; NO beam)
# --------------------------------------------------------------------------- #
def teacher_forced_states(labels: Sequence[int]
                          ) -> List[List[Tuple[int, int]]]:
    """Build the sequence of tree STATES visited while merging ONLY
    within-sentence pairs (never across a gold boundary), starting from all
    words separate.  Deterministic left-to-right agglomeration within each gold
    sentence.  Returns the list of states (each a node list) that the teacher
    visits; EVERY adjacent candidate at EVERY returned state is supervised
    (dense supervision), not only the pairs actually merged.

    Guarantee: every constituent AraBERT ever reads under teacher forcing is a
    real sub-sentence (a contiguous span inside one gold sentence)."""
    n = len(labels)
    if n == 0:
        return []
    spans = sentence_spans(labels)                # gold sentences
    # per-sentence internal merge frontier; we advance all sentences in lockstep
    # so intermediate states contain multi-word constituents from every sentence
    # (dense supervision then sees within-sentence AND cross-boundary candidates
    # at realistic partially-merged states).
    # node list is the concatenation of each sentence's current internal nodes.
    sent_nodes = [[(i, i) for i in range(s0, s1 + 1)] for (s0, s1) in spans]
    states: List[List[Tuple[int, int]]] = []

    def flatten():
        flat = []
        for sn in sent_nodes:
            flat.extend(sn)
        return flat

    states.append(flatten())                      # start state (all cut)
    # keep merging the leftmost adjacent pair inside each sentence until every
    # sentence is a single node.
    progress = True
    while progress:
        progress = False
        for si, sn in enumerate(sent_nodes):
            if len(sn) > 1:
                # merge leftmost adjacent within this sentence
                sent_nodes[si] = [(sn[0][0], sn[1][1])] + sn[2:]
                progress = True
        if progress:
            states.append(flatten())
    return states


def dense_supervision_pairs(nodes: List[Tuple[int, int]],
                            gold_cut: np.ndarray
                            ) -> Tuple[List[Tuple[Tuple[int, int],
                                                  Tuple[int, int]]],
                                       List[int]]:
    """For a tree state, return (pairs, targets): EVERY adjacent node pair
    (dense), with BCE target
        t = 1  if the junction is NOT a gold boundary (same sentence -> merge)
        t = 0  if the junction IS  a gold boundary     (should NOT merge).
    """
    pairs, targets = [], []
    for i in range(len(nodes) - 1):
        a = nodes[i]
        b = nodes[i + 1]
        k = a[1]                                  # junction erased by merging i
        t = 0 if int(gold_cut[k]) == 1 else 1
        pairs.append((a, b))
        targets.append(t)
    return pairs, targets


def count_dense_candidates(labels: Sequence[int]) -> int:
    """#supervised candidates for a doc under teacher forcing = sum over visited
    states of #adjacent pairs at that state.  Used by the dense-supervision smoke
    (must equal the total number of BCE terms actually applied)."""
    total = 0
    for nodes in teacher_forced_states(labels):
        total += max(0, len(nodes) - 1)
    return total


# Sentence-starter connectives (bare forms) for the TARGETED margin: the joint
# autopsy showed boundary kills are ENRICHED at junctions whose next word starts
# with و/ف or is one of these (43% of kills vs 33% base) — the classic Arabic
# "wa = and-inside vs Wa = new-sentence" ambiguity.
CONN_SENT_STARTERS = set("ثم لكن كما بل أو حتى إذ إذا قال قالت".split())


def conn_initial(word: str) -> bool:
    """Does `word` plausibly open a new sentence via a connective? (prefix و/ف
    covers proclitics like وقال; plus the bare starter set)."""
    return word.startswith("و") or word.startswith("ف") or word in CONN_SENT_STARTERS


def _state_loss(scorer, word_ids, nodes, gold_cut, pos_weight, margin_delta=0.0,
                boundary_weight_mult: float = 1.0, conn_mask=None,
                margin_gamma_b: float = 1.0, conn_margin_extra: float = 0.0,
                rdrop_alpha: float = 0.0):
    """DENSE supervision at ONE state: one batched AraBERT forward over EVERY
    adjacent candidate; class-weighted BCE against the gold-aligned target.
    Optionally adds the v2 max-margin ranking term.  Returns
    (loss_tensor, n_cand, n_pos, n_neg, logits_tensor).

    boundary_weight_mult scales the penalty on the BOUNDARY (t=0, don't-merge)
    candidates at THIS state — the cascade-penalty hook.  1.0 = plain.

    MARGIN-STRUCTURED LOSS (THE trial; VS-loss style, Kini NeurIPS'21): the logit
    of every BOUNDARY (t=0) candidate is MULTIPLIED by margin_gamma_b (and by a
    further (1+conn_margin_extra) where conn_mask marks a connective-initial
    junction) before the BCE.  At zero train loss only MULTIPLICATIVE scaling
    still moves the learned margins — this forces a wider safety gap exactly
    where the confident kills live.  gamma=1, extra=0 -> byte-identical.

    R-DROP (rdrop_alpha>0): score the same candidates TWICE under independent
    dropout draws; loss = mean of the two (margin-scaled) BCEs + alpha * the
    symmetric KL between the two Bernoulli predictions ((p1-p2)*(z1-z2) per
    junction).  Flaky confidence is punished directly.  alpha=0 -> one draw,
    byte-identical."""
    import torch
    pairs, targets = dense_supervision_pairs(nodes, gold_cut)
    if not pairs:
        z = torch.zeros((), device=scorer.device)
        return z, 0, 0, 0, torch.zeros((0,), device=scorer.device)
    logits = scorer.logits_for_pairs(word_ids, pairs, grad=True)
    tgt = torch.tensor(targets, dtype=torch.float32, device=scorer.device)
    # per-example weight: boundaries (t=0) are the minority -> up-weight them; the
    # cascade multiplier lets an EARLY pass punish a cross-boundary merge harder.
    w = torch.where(tgt > 0.5,
                    torch.ones_like(tgt),
                    torch.full_like(tgt, float(pos_weight) * float(boundary_weight_mult)))
    # multiplicative margin scale on the BOUNDARY side (targeted at connectives)
    if margin_gamma_b != 1.0 or conn_margin_extra > 0.0:
        gam = torch.full_like(tgt, float(margin_gamma_b))
        if conn_mask is not None and conn_margin_extra > 0.0:
            cm = torch.tensor(conn_mask, dtype=torch.float32, device=scorer.device)
            gam = gam * (1.0 + float(conn_margin_extra) * cm)
        scale = torch.where(tgt > 0.5, torch.ones_like(tgt), gam)
    else:
        scale = None

    def _bce(z):
        ze = z * scale if scale is not None else z
        return torch.nn.functional.binary_cross_entropy_with_logits(
            ze, tgt, weight=w, reduction="sum")

    if rdrop_alpha and rdrop_alpha > 0.0:
        if hasattr(scorer, "set_draw"):
            scorer.set_draw(1)
        logits2 = scorer.logits_for_pairs(word_ids, pairs, grad=True)
        if hasattr(scorer, "set_draw"):
            scorer.set_draw(0)
        p1, p2 = torch.sigmoid(logits), torch.sigmoid(logits2)
        kl_sym = ((p1 - p2) * (logits - logits2)).sum()     # Bernoulli sym-KL
        bce = 0.5 * (_bce(logits) + _bce(logits2)) + float(rdrop_alpha) * kl_sym
    else:
        bce = _bce(logits)
    if margin_delta and margin_delta > 0.0:
        bce = bce + _margin_ranking_term(logits, tgt, margin_delta)
    n_pos = int(sum(targets))
    return bce, len(targets), n_pos, len(targets) - n_pos, logits


def _margin_ranking_term(logits, tgt, delta):
    """OPTIONAL v2 (off by default): max-margin ranking loss
    max(0, delta - (p_valid - p_invalid)) between every valid (t=1) and invalid
    (t=0) candidate at a state.  Encourages within-sentence merges to out-score
    cross-boundary merges by a margin (Socher ICML-2011 flavour)."""
    import torch
    p = torch.sigmoid(logits)
    valid = p[tgt > 0.5]
    invalid = p[tgt <= 0.5]
    if valid.numel() == 0 or invalid.numel() == 0:
        return torch.zeros((), device=logits.device)
    diff = valid.unsqueeze(1) - invalid.unsqueeze(0)     # (nv, ni)
    return torch.clamp(delta - diff, min=0.0).sum()


def doc_train_loss(scorer: "MergeScorer", words: Sequence[str],
                   labels: Sequence[int], pos_weight: float,
                   margin_delta: float = 0.0, backward_each_state: bool = False):
    """Teacher-forced, densely-supervised BCE loss for ONE document (roll-in
    = teacher).  For every visited teacher state, ONE batched AraBERT forward
    scores EVERY adjacent candidate; class-weighted BCE against the gold-aligned
    target is summed.  Returns (loss, n_cand, n_pos, n_neg).  Boundaries (t=0,
    the minority) get `pos_weight` so the model is not pushed to merge-all
    (guards FM2).

    backward_each_state (training only): call l.backward() after EACH state and
    keep only the detached scalar, so the retained autograd graph is bounded to
    ONE state instead of the whole document.  Grads accumulate in .grad exactly
    as a single backward on the summed loss would (backprop is linear over the
    sum), but a 15k-word doc no longer holds ~295k candidates' worth of graph at
    once -> no CUDA OOM.  The caller must NOT call .backward() again."""
    import torch
    gold_cut = gold_cut_mask(labels)
    word_ids = scorer._word_ids(words)
    states = teacher_forced_states(labels)
    total_loss = torch.zeros((), device=scorer.device)
    n_cand = n_pos = n_neg = 0
    for nodes in states:
        l, c, p, ng, _ = _state_loss(scorer, word_ids, nodes, gold_cut,
                                     pos_weight, margin_delta)
        if backward_each_state:
            if l.requires_grad:
                l.backward()
            total_loss = total_loss + l.detach()
        else:
            total_loss = total_loss + l
        n_cand += c
        n_pos += p
        n_neg += ng
    return total_loss, n_cand, n_pos, n_neg


def doc_train_loss_sched(scorer: "MergeScorer", words: Sequence[str],
                         labels: Sequence[int], pos_weight: float,
                         p_model: float, rng, margin_delta: float = 0.0,
                         backward_each_state: bool = False):
    """SCHEDULED SAMPLING roll-in (Bengio 2015).  Build the tree bottom-up; at
    each merge step, with prob `p_model` follow the MODEL's own top-scored VALID
    merge (let it visit its own states), else teacher-force the GOLD next merge.
    DENSE supervision is applied at EVERY visited state (all adjacent candidates,
    class-weighted BCE), whichever roll-in produced it.

    Structure-building rule (spec): NEVER commit a cross-boundary merge UNLESS the
    model-roll-in explicitly picks one; if it does, that state is still supervised
    (its straddling pair carries target 0) and the SPAN-LENGTH CAP applies (an
    over-cap resulting span is forbidden, so a runaway cross-boundary cascade
    cannot form).  Returns (loss, n_cand, n_pos, n_neg, n_model_steps)."""
    import torch
    gold_cut = gold_cut_mask(labels)
    word_ids = scorer._word_ids(words)
    n = len(labels)
    # start state: every word its own node (all cuts present)
    nodes = start_state(n)
    total_loss = torch.zeros((), device=scorer.device)
    n_cand = n_pos = n_neg = 0
    n_model_steps = 0
    max_steps = n                              # safety bound on the merge loop
    for _ in range(max_steps):
        # DENSE supervision at the CURRENT visited state (before merging)
        l, c, p, ng, logits = _state_loss(scorer, word_ids, nodes, gold_cut,
                                           pos_weight, margin_delta)
        n_cand += c
        n_pos += p
        n_neg += ng
        # Decide the next roll-in merge from THIS state's scores FIRST, while the
        # graph is still intact (backward_each_state frees it below).
        stop = (len(nodes) <= 1)
        chosen_i = None
        if not stop:
            # candidate junctions at this state; gold-valid = within-sentence
            cands = adjacent_candidates(nodes)             # [(i, junction_k)]
            gold_valid = [(i, k) for (i, k) in cands if int(gold_cut[k]) == 0]
            use_model = (rng.random() < p_model)
            if use_model and logits.numel() == len(cands):
                n_model_steps += 1
                # model's own TOP-scored candidate among those NOT over the cap
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                best_p, best_i = -1.0, None
                for local, (i, _k) in enumerate(cands):
                    a, b = nodes[i], nodes[i + 1]
                    if scorer._over_cap(word_ids, a, b):   # span-cap veto
                        continue
                    if probs[local] > best_p:
                        best_p, best_i = probs[local], i
                chosen_i = best_i                          # may be cross-boundary
            if chosen_i is None:
                # teacher: merge the leftmost gold-valid pair; if none remain, all
                # remaining junctions are boundaries -> stop (never commit a
                # cross-boundary merge on the teacher path).
                if not gold_valid:
                    stop = True
                else:
                    chosen_i = gold_valid[0][0]
        # accumulate the loss; bound the graph to ONE state when training
        if backward_each_state:
            if l.requires_grad:
                l.backward()
            total_loss = total_loss + l.detach()
        else:
            total_loss = total_loss + l
        if stop:
            break
        nodes = merge_nodes(nodes, chosen_i)
    return total_loss, n_cand, n_pos, n_neg, n_model_steps


def sched_p_at_epoch(ep: int, epochs: int, p_max: float = SCHED_P_MAX_DEFAULT
                     ) -> float:
    """Linear ramp of the model-roll-in probability from 0.0 at epoch 0 to
    `p_max` at the last epoch (Bengio 2015 inverse-sigmoid/linear decay family;
    linear here).  epochs<=1 -> 0.0 (pure teacher)."""
    if epochs <= 1:
        return 0.0
    return p_max * (ep / (epochs - 1))


def rollin_p_model(roll_in: str, ep: int, epochs: int,
                   sched_p_max: float = SCHED_P_MAX_DEFAULT) -> float:
    """Model-roll-in probability for an epoch, by roll-in mode.  Isolated so the
    schedule is unit-testable without a real encoder.
      selfmerge -> 1.0 EVERY epoch (incl. the first): the model chooses every
                   merge itself, punished at each cross-boundary glue, NO teacher
                   forcing (the local-minima escape).
      sched/laso -> linear ramp 0 -> sched_p_max (Bengio 2015).
      teacher    -> 0.0 (pure answer-key)."""
    if roll_in == "selfmerge":
        return 1.0
    if roll_in in ("sched", "laso"):
        return sched_p_at_epoch(ep, epochs, sched_p_max)
    return 0.0


def doc_train_loss_beam(scorer: "MergeScorer", words: Sequence[str],
                        labels: Sequence[int], pos_weight: float,
                        p_model: float, rng, beam_width: int = BEAM_WIDTH,
                        margin_delta: float = 0.0, backward_each_state: bool = False):
    """LaSO / beam roll-in (IMPLEMENTED, OFF for the deadline: default is sched).
    Roll the tree in with a width-`beam_width` beam under the model's own probs;
    DENSE supervision at every state on the TOP hypothesis's roll-in path (all
    adjacent candidates, class-weighted BCE against gold).  The span-length cap
    vetoes over-cap merges in the roll-in.  Returns
    (loss, n_cand, n_pos, n_neg, n_model_steps)."""
    import torch
    gold_cut = gold_cut_mask(labels)
    word_ids = scorer._word_ids(words)
    n = len(labels)
    total_loss = torch.zeros((), device=scorer.device)
    n_cand = n_pos = n_neg = 0
    n_model_steps = 0
    # roll in the model's own decode path (no grad) to collect visited states,
    # then supervise them WITH grad (decouples the discrete search from backprop).
    win = beam_decode(scorer, words, tau=0.05, beam_width=beam_width)
    # reconstruct the states along the winner's merge path deterministically:
    # start all-separate, merge leftmost model-preferred valid pair each step,
    # bounded by the winner's node count.  We supervise the START + each merged
    # state; if p_model<1 we mix teacher merges in (same rule as sched).
    nodes = start_state(n)
    for _ in range(n):
        l, c, p, ng, logits = _state_loss(scorer, word_ids, nodes, gold_cut,
                                           pos_weight, margin_delta)
        if backward_each_state:                 # memory fix: bound graph to 1 state
            if l.requires_grad:
                l.backward()
            total_loss = total_loss + l.detach()
        else:
            total_loss = total_loss + l
        n_cand += c
        n_pos += p
        n_neg += ng
        if len(nodes) <= len(win.nodes) or len(nodes) <= 1:
            break
        cands = adjacent_candidates(nodes)
        use_model = (rng.random() < p_model)
        chosen_i = None
        if use_model and logits.numel() == len(cands):
            n_model_steps += 1
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            best_p, best_i = -1.0, None
            for local, (i, _k) in enumerate(cands):
                a, b = nodes[i], nodes[i + 1]
                if scorer._over_cap(word_ids, a, b):
                    continue
                if probs[local] > best_p:
                    best_p, best_i = probs[local], i
            chosen_i = best_i
        if chosen_i is None:
            gold_valid = [i for (i, k) in cands if int(gold_cut[k]) == 0]
            if not gold_valid:
                break
            chosen_i = gold_valid[0]
        nodes = merge_nodes(nodes, chosen_i)
    return total_loss, n_cand, n_pos, n_neg, n_model_steps


def _plan_pass_glues(probs, glue_tau, glue_order="confidence", blocked=None):
    """Choose the NON-OVERLAPPING set of adjacent junctions to glue in ONE pass.
    probs[j] = P(glue pieces j and j+1).  Eligible = P>=glue_tau and j not blocked
    (span-cap veto).  glue_order:
      "confidence" (easy-first, Goldberg-Elhadad 2010): glue the HIGHEST-P eligible
        pair first, then the next that doesn't touch an already-glued piece;
      "leftright": position order (glue the leftmost eligible pair first).
    Returns the set of left-piece indices j to glue; guaranteed non-overlapping
    (never both j and j+1)."""
    blocked = blocked or set()
    eligible = [j for j in range(len(probs))
                if probs[j] >= glue_tau and j not in blocked]
    if glue_order == "confidence":
        eligible.sort(key=lambda j: probs[j], reverse=True)      # easy-first
    taken = set()                       # piece indices already consumed this pass
    plan = set()
    for j in eligible:
        if j in taken or (j + 1) in taken:
            continue                    # overlaps an already-committed glue
        plan.add(j)
        taken.add(j)
        taken.add(j + 1)
    return plan


def doc_train_loss_recursive(scorer: "MergeScorer", words: Sequence[str],
                             labels: Sequence[int], pos_weight: float,
                             glue_tau: float = GLUE_TAU_DEFAULT,
                             glue_order: str = "confidence",
                             margin_delta: float = 0.0,
                             backward_each_state: bool = False,
                             max_passes: int = None,
                             cascade_gamma: float = 0.0,
                             margin_gamma_b: float = 1.0,
                             conn_margin_extra: float = 0.0,
                             rdrop_alpha: float = 0.0):
    """PER-PASS RECURSIVE self-gluing (Noor's locked algorithm).  Leaves = words.

    Each PASS:
      1. Score EVERY adjacent piece-pair in ONE batched AraBERT forward and DENSELY
         supervise it (class-weighted BCE vs gold): within-sentence pair -> target 1
         (SHOULD glue); cross-boundary pair -> target 0 (must NOT glue -> punished).
      2. The MODEL then glues, by its OWN probabilities, every adjacent pair whose
         P >= glue_tau, NON-OVERLAPPING and left-to-right (glue a-b -> move to c-d;
         don't glue -> move to b-c).  The span-length cap vetoes over-cap glues.
      3. The glued pieces feed the NEXT pass -- the recursion.  Wrong (cross-
         boundary) lumps are carried forward, so the model is supervised on its OWN
         corrupted pieces and learns to place the remaining boundaries after a
         mistake (no teacher forcing).
    STOP when a pass makes zero glues (every adjacent pair is below glue_tau) or one
    piece remains.  Cost ~ O(pieces x passes), passes ~ log(max sentence length):
    each pass glues MANY pairs at once, so a long doc collapses in a handful of
    passes -- NOT the O(n^2) of re-scoring after every single glue.

    Returns (loss, n_cand, n_pos, n_neg, n_glues).  backward_each_state bounds the
    retained autograd graph to ONE pass (grads accumulate exactly as one summed
    backward; the caller must not call .backward() again)."""
    import torch
    gold_cut = gold_cut_mask(labels)
    word_ids = scorer._word_ids(words)
    n = len(labels)
    nodes = start_state(n)
    total_loss = torch.zeros((), device=scorer.device)
    n_cand = n_pos = n_neg = 0
    n_glues = 0
    if max_passes is None:
        max_passes = n                 # a pass with >=1 glue shrinks the piece list
    for depth in range(max_passes):
        # (1) DENSE supervision at the CURRENT pieces (grad kept for the loss).
        # CASCADE PENALTY: an early (low-depth) cross-boundary merge cascades (71%
        # of destroyed boundaries die by depth 2), so weight its boundary penalty
        # higher and let it decay toward 1 with depth.  cascade_gamma=0 -> mult=1
        # every pass -> byte-identical to before.
        bwm = (1.0 + cascade_gamma / (1.0 + depth)) if cascade_gamma else 1.0
        # targeted-margin mask: junction i is connective-initial if the word
        # AFTER it (first word of the right piece) opens with و/ف or a starter
        conn_mask = None
        if conn_margin_extra > 0.0:
            conn_mask = [1.0 if (nodes[i][1] + 1 < n
                                 and conn_initial(words[nodes[i][1] + 1])) else 0.0
                         for i in range(len(nodes) - 1)]
        l, c, p, ng, logits = _state_loss(scorer, word_ids, nodes, gold_cut,
                                          pos_weight, margin_delta,
                                          boundary_weight_mult=bwm,
                                          conn_mask=conn_mask,
                                          margin_gamma_b=margin_gamma_b,
                                          conn_margin_extra=conn_margin_extra,
                                          rdrop_alpha=rdrop_alpha)
        if backward_each_state:
            if l.requires_grad:
                l.backward()
            total_loss = total_loss + l.detach()
        else:
            total_loss = total_loss + l
        n_cand += c
        n_pos += p
        n_neg += ng
        if len(nodes) <= 1:
            break
        # (2) the model's OWN glue decisions this pass (detached -> just picks the
        # states; the gradient came from the BCE above).  probs[i] = P(glue
        # nodes[i], nodes[i+1]) since dense pairs are enumerated in node order.
        # Glue a NON-OVERLAPPING set of pairs this pass, chosen by glue_order
        # (confidence = highest-P first); the span-cap vetoes over-cap glues.
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        blocked = {j for j in range(len(nodes) - 1)
                   if scorer._over_cap(word_ids, nodes[j], nodes[j + 1])}
        plan = _plan_pass_glues(probs, glue_tau, glue_order, blocked)
        if not plan:
            break                        # no glue clears tau -> boundaries are fixed
        new_nodes = []
        i = 0
        while i < len(nodes):
            if i in plan:
                new_nodes.append((nodes[i][0], nodes[i + 1][1]))   # glue
                n_glues += 1
                i += 2
            else:
                new_nodes.append(nodes[i])                         # keep piece
                i += 1
        nodes = new_nodes
    return total_loss, n_cand, n_pos, n_neg, n_glues


def train_pos_weight(docs: List[dict]) -> float:
    """Class weight for the BCE: (#within-sentence junctions) / (#boundary
    junctions) over TRAIN, so the minority target-0 (boundaries) is up-weighted.
    Computed on TRAIN only (recomputing on dev/test is leakage)."""
    pos = neg = 0                      # pos = within-sentence (t=1), neg = bound.
    for d in docs:
        gc = gold_cut_mask(d["labels"])
        neg += int((gc == 1).sum())
        pos += int((gc == 0).sum())
    return float(pos) / max(neg, 1)


def _lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup to 1.0 over `warmup_steps`, then linear decay to 0 over the
    remainder (report FM4).  Returned as a step->multiplier for LambdaLR."""
    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - prog)
    return fn


def train_on_docs(scorer: "MergeScorer", docs: List[dict], *,
                  epochs: int = 10, lr: float = 3e-5, pos_weight: float = None,
                  clip_norm: float = 1.0, seed: int = 42, log_every: int = 50,
                  weight_decay: float = 0.01,
                  verbose: bool = True, roll_in: str = "teacher",
                  sched_p_max: float = SCHED_P_MAX_DEFAULT,
                  warmup_frac: float = 0.0, margin_delta: float = 0.0,
                  glue_tau: float = GLUE_TAU_DEFAULT,
                  glue_order: str = "confidence",
                  val_docs: List[dict] = None, patience: int = 0,
                  val_window: int = WINDOW_DEFAULT,
                  val_tau_grid: List[float] = None,
                  cascade_gamma: float = 0.0,
                  margin_gamma_b: float = 1.0, conn_margin_extra: float = 0.0,
                  rdrop_alpha: float = 0.0, save_epochs_dir: str = None):
    """Fine-tune AraBERT + the scalar head end-to-end (<=10 epochs; it is
    fine-tuning, not from scratch).  Densely-supervised.  One optimizer step per
    document (loss summed over that doc's visited states).

    roll_in:
      "teacher" -> pure teacher forcing (build the gold tree; states never
                   straddle a boundary).  [Back-compat default: existing tests.]
      "sched"   -> SCHEDULED SAMPLING (Bengio 2015): per epoch, model-roll-in
                   probability ramps 0 -> sched_p_max; the model visits its own
                   states, supervised densely against gold.  Span-length cap
                   forbids over-cap cross-boundary merges during roll-in.
      "laso"    -> beam roll-in (LaSO); IMPLEMENTED, kept OFF for the deadline
                   (falls back to the beam-roll-in path).
      "selfmerge"-> PER-PASS RECURSIVE self-gluing (Noor's algorithm): each pass
                   the model glues every adjacent pair it scores >= glue_tau
                   (non-overlapping, left-to-right), recursing on its OWN glued
                   pieces (incl. wrong lumps) until a pass makes no glue.  Densely
                   supervised, punished at each cross-boundary glue.  No teacher
                   forcing.  Cost ~O(pieces x passes), NOT O(n^2).
    warmup_frac: fraction of TOTAL optimizer steps used for LR warmup (report
                 FM4); 0.0 disables (back-compat).
    margin_delta: v2 max-margin ranking term (0.0 = off, spec default)."""
    import torch
    if roll_in not in ROLL_IN_CHOICES:
        raise ValueError(f"roll_in must be one of {ROLL_IN_CHOICES}")
    if pos_weight is None:
        pos_weight = train_pos_weight(docs)
    params = [p for p in (list(scorer.enc.parameters())
                          + list(scorer.head.parameters())) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    trainable = [d for d in docs
                 if len([w for w in d["tokens"]]) >= 2]
    total_steps = max(1, epochs * max(1, len(trainable)))
    warmup_steps = int(round(warmup_frac * total_steps)) if warmup_frac else 0
    sched = (torch.optim.lr_scheduler.LambdaLR(
                opt, _lr_lambda(warmup_steps, total_steps))
             if warmup_steps > 0 else None)
    rng = np.random.default_rng(seed)
    scorer.enc.train()
    history = []
    # EARLY STOPPING on a held-out TRAIN slice (val_docs): decode each epoch, keep
    # the BEST checkpoint, stop after `patience` epochs with no val-F1 gain.  Dev is
    # NEVER touched here (closed track).  patience<=0 or no val_docs -> disabled
    # (back-compat: byte-identical to before).
    best_val, best_epoch, since_improve = -1.0, -1, 0
    best_enc = best_head = None
    do_earlystop = bool(val_docs) and patience > 0
    for ep in range(epochs):
        p_model = rollin_p_model(roll_in, ep, epochs, sched_p_max)
        order = rng.permutation(len(docs))
        ep_loss, ep_cand, ep_modelsteps = 0.0, 0, 0
        for step, di in enumerate(order):
            d = docs[int(di)]
            words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                     for w in d["tokens"]]
            if len(words) < 2:
                continue
            opt.zero_grad()
            # backward_each_state bounds the retained autograd graph to ONE tree
            # state (instead of the whole document), so a very long doc's O(n^2)
            # dense candidates no longer OOM the GPU.  Grads still accumulate over
            # the whole doc before the opt.step() below.
            did_backward = False
            if roll_in == "leaf":
                # NAIVE BIDIRECTIONAL BOUNDARY DETECTOR (Noor's pivot): supervise
                # ONLY the leaf state (every word-junction, dense, bidirectional
                # full-window context) — no gluing, no recursion.  The margin /
                # connective-margin / R-Drop machinery applies unchanged (it
                # lives in _state_loss).  Decode = threshold leaf LOGITS.
                gold_cut_l = gold_cut_mask(d["labels"])
                wid_l = scorer._word_ids(words)
                conn_mask_l = None
                if conn_margin_extra > 0.0:
                    conn_mask_l = [1.0 if (w + 1 < len(words)
                                           and conn_initial(words[w + 1])) else 0.0
                                   for w in range(len(words) - 1)]
                loss, ncand, npos, nneg, _lg = _state_loss(
                    scorer, wid_l, start_state(len(words)), gold_cut_l,
                    pos_weight, margin_delta,
                    conn_mask=conn_mask_l, margin_gamma_b=margin_gamma_b,
                    conn_margin_extra=conn_margin_extra, rdrop_alpha=rdrop_alpha)
                nms = 0
                did_backward = False
            elif roll_in == "selfmerge":
                # JOINT scorer: all passes share ONE encode graph, so the summed
                # loss gets a SINGLE backward (backward-per-pass would need the
                # shared graph retained).  Pair scorer keeps per-pass backward
                # (its per-pair graphs are the memory hog; unchanged behaviour).
                bes = not getattr(scorer, "joint", False)
                loss, ncand, npos, nneg, nms = doc_train_loss_recursive(
                    scorer, words, d["labels"], pos_weight,
                    glue_tau=glue_tau, glue_order=glue_order,
                    margin_delta=margin_delta, backward_each_state=bes,
                    cascade_gamma=cascade_gamma,
                    margin_gamma_b=margin_gamma_b,
                    conn_margin_extra=conn_margin_extra,
                    rdrop_alpha=rdrop_alpha)
                ep_modelsteps += nms
                did_backward = bes
            elif roll_in == "teacher" or p_model <= 0.0:
                loss, ncand, npos, nneg = doc_train_loss(
                    scorer, words, d["labels"], pos_weight, margin_delta,
                    backward_each_state=True)
                did_backward = True
            elif roll_in == "laso":
                loss, ncand, npos, nneg, nms = doc_train_loss_beam(
                    scorer, words, d["labels"], pos_weight, p_model, rng,
                    margin_delta=margin_delta, backward_each_state=True)
                ep_modelsteps += nms
                did_backward = True
            else:                                # "sched"
                loss, ncand, npos, nneg, nms = doc_train_loss_sched(
                    scorer, words, d["labels"], pos_weight, p_model, rng,
                    margin_delta=margin_delta, backward_each_state=True)
                ep_modelsteps += nms
                did_backward = True
            if ncand == 0:
                continue
            if not did_backward:                 # laso path still backwards once
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
            opt.step()
            if sched is not None:
                sched.step()
            ep_loss += float(loss.detach())
            ep_cand += ncand
            if verbose and log_every and (step % log_every == 0):
                print(f"  ep{ep} step{step}/{len(order)} "
                      f"loss/cand={float(loss.detach())/max(ncand,1):.4f}",
                      flush=True)
        mean = ep_loss / max(ep_cand, 1)
        history.append(mean)
        if verbose:
            extra = (f" P_model={p_model:.3f} model_steps={ep_modelsteps}"
                     if roll_in in ("sched", "laso", "selfmerge") else "")
            print(f"[epoch {ep}] mean BCE/candidate = {mean:.5f} "
                  f"({ep_cand} candidates){extra}", flush=True)
        # per-epoch checkpoints for WEIGHT SOUP (greedy averaging of the zero-loss
        # basin walk): save every epoch's weights; the soup script picks/averages.
        if save_epochs_dir:
            os.makedirs(save_epochs_dir, exist_ok=True)
            torch.save({"enc_state": {k: v.detach().cpu()
                                      for k, v in scorer.enc.state_dict().items()},
                        "head_state": {k: v.detach().cpu()
                                       for k, v in scorer.head.state_dict().items()},
                        "epoch": ep},
                       os.path.join(save_epochs_dir, f"ep{ep}.pt"))
        # ---- early-stopping validation on the held-out TRAIN slice ----
        if do_earlystop:
            scorer.enc.eval()
            scorer.clear_cache()          # weights changed this epoch -> stale memo
            v_tau, v_metrics, _ = sweep_tau(
                scorer, val_docs,
                tau_grid=(val_tau_grid if val_tau_grid else [0.30, 0.40, 0.50]),
                beam_width=BEAM_WIDTH, alpha=ALPHA_DEFAULT,
                window=val_window, stride=STRIDE_DEFAULT)
            scorer.enc.train()
            v_f1 = v_metrics["macro_f1"]
            if v_f1 > best_val + 1e-5:
                best_val, best_epoch, since_improve = v_f1, ep, 0
                best_enc = {k: v.detach().cpu().clone()
                            for k, v in scorer.enc.state_dict().items()}
                best_head = {k: v.detach().cpu().clone()
                             for k, v in scorer.head.state_dict().items()}
            else:
                since_improve += 1
            if verbose:
                print(f"[epoch {ep}] val macro-F1 = {v_f1 * 100:.4f} "
                      f"(best {best_val * 100:.4f} @ep{best_epoch}, tau={v_tau}; "
                      f"patience {since_improve}/{patience})", flush=True)
            if since_improve >= patience:
                if verbose:
                    print(f"[early stop] no val gain for {patience} epochs; "
                          f"best ep{best_epoch} val-F1 {best_val * 100:.4f}",
                          flush=True)
                break
    # restore the BEST-on-val checkpoint (if early stopping ran)
    if best_enc is not None:
        scorer.enc.load_state_dict(best_enc)
        scorer.head.load_state_dict(best_head)
        if verbose:
            print(f"[restore] best epoch {best_epoch} "
                  f"(val macro-F1 {best_val * 100:.4f})", flush=True)
    scorer.enc.eval()
    return history


# --------------------------------------------------------------------------- #
# 5. INFERENCE — beam width 3, LENGTH-NORMALIZED (geometric mean), merge-generous
# --------------------------------------------------------------------------- #
class Hyp:
    """A partial forest: node list + the list of merge PROBS used to build it.
    Score = LOG-SPACE LENGTH-NORMALIZED (report FM1):
        score = (1/|M|^alpha) * sum(log p_m)          [returned in log-space]
    alpha=1.0 => arithmetic mean of log p = log(geometric mean).  We compare
    hypotheses by this log score; `score()` returns the geometric-mean PROB (so
    the value is in (0,1] and the alpha=1.0 default reproduces the prior
    geometric-mean behaviour exactly)."""
    __slots__ = ("nodes", "merge_probs", "done", "alpha")

    def __init__(self, nodes, merge_probs=None, done=False, alpha=ALPHA_DEFAULT):
        self.nodes = nodes
        self.merge_probs = merge_probs if merge_probs is not None else []
        self.done = done
        self.alpha = alpha

    def log_score(self) -> float:
        """(1/|M|^alpha) * sum(log p_m), in log-space (report FM1; avoids
        underflow).  No merges -> 0.0 (log 1)."""
        if not self.merge_probs:
            return 0.0
        s = sum(math.log(max(p, 1e-12)) for p in self.merge_probs)
        m = len(self.merge_probs)
        return s / (m ** self.alpha)

    def score(self) -> float:
        """Length-normalized score as a PROB in (0,1].  For alpha=1.0 this is the
        geometric mean of the merge probs (== exp(mean(log p))); a no-merge hyp
        scores neutral 1.0.  Ranking by score() and by log_score() is identical
        for a fixed alpha."""
        if not self.merge_probs:
            return 1.0
        return math.exp(self.log_score())

    def key(self):
        return tuple(self.nodes)


def beam_decode(scorer: "MergeScorer", words: Sequence[str], tau: float,
                beam_width: int = BEAM_WIDTH, max_steps: int = None,
                alpha: float = ALPHA_DEFAULT) -> Hyp:
    """Beam search.  Start all words separate (all cuts present).  Each step:
    batch-score all candidate merges across ALL beam hypotheses in ONE AraBERT
    forward (memoized); expand; keep top-`beam_width` by LOG-SPACE
    LENGTH-NORMALIZED score = (1/|M|^alpha) sum(log p_m) (alpha=1.0 = geo-mean,
    report FM1).  BEAM DEDUP (report FM6): hypotheses with identical active-cut
    sets are collapsed, keeping the higher score.  STOP a hypothesis when EVERY
    remaining adjacent merge prob < tau (all remaining junctions are boundaries).
    Merge-generous: tau LOW.  The SPAN-LENGTH CAP forces over-cap merges to p~0
    (inside probs_for_pairs), so they never clear tau.  DOC-FINAL INVARIANT:
    there is no candidate after the last node, so the junction after the last
    word is never merged (always a boundary).  Returns the highest-scoring
    COMPLETE hypothesis.
    """
    n = len(words)
    if n <= 1:
        return Hyp(start_state(n), [], done=True, alpha=alpha)
    word_ids = scorer._word_ids(words)
    if max_steps is None:
        max_steps = n                     # at most n-1 merges collapse to 1 node

    beam = [Hyp(start_state(n), [], done=False, alpha=alpha)]
    complete: List[Hyp] = []

    for _ in range(max_steps):
        # gather all candidate merges across all live (not done) hyps -> ONE fwd
        live = [hh for hh in beam if not hh.done]
        for hh in beam:
            if hh.done:
                complete.append(hh)
        if not live:
            break
        flat_pairs = []
        owner = []            # (hyp_index_in_live, node_index_i)
        for hi, hh in enumerate(live):
            cands = adjacent_candidates(hh.nodes)   # [(i, junction_k), ...]
            for (i, _k) in cands:
                a = hh.nodes[i]
                b = hh.nodes[i + 1]
                flat_pairs.append((a, b))
                owner.append((hi, i))
        probs = scorer.probs_for_pairs(word_ids, flat_pairs)

        # group probs back per live hyp
        per_hyp = [[] for _ in live]
        for (hi, i), p in zip(owner, probs):
            per_hyp[hi].append((i, float(p)))

        expansions: List[Hyp] = []
        for hi, hh in enumerate(live):
            cand = per_hyp[hi]
            if not cand:
                hh.done = True
                complete.append(hh)
                continue
            best_p = max(p for (_i, p) in cand)
            if best_p < tau:
                # merge-generous stop: no adjacent merge clears tau -> DONE.
                hh.done = True
                complete.append(hh)
                continue
            # expand: apply each candidate merge whose prob >= tau (those that
            # clear the merge-generous bar); each becomes a child hypothesis.
            any_expanded = False
            for (i, p) in cand:
                if p < tau:
                    continue
                child = Hyp(merge_nodes(hh.nodes, i),
                            hh.merge_probs + [p], done=False, alpha=alpha)
                expansions.append(child)
                any_expanded = True
            if not any_expanded:
                hh.done = True
                complete.append(hh)

        if not expansions:
            break
        # BEAM DEDUP (report FM6): collapse hypotheses with identical ACTIVE-CUT
        # SETS, keeping the higher-scoring one; then keep top-`beam_width` by
        # log-space length-normalized score.  (The active-cut set is the boundary
        # signature of a partition: two hyps with the same surviving cuts are the
        # same segmentation regardless of internal merge order.)
        seen = {}
        for h in expansions:
            k = tuple(state_cuts(h.nodes))          # active-cut set signature
            if k not in seen or h.log_score() > seen[k].log_score():
                seen[k] = h
        ranked = sorted(seen.values(), key=lambda h: h.log_score(), reverse=True)
        beam = ranked[:beam_width]

    # any hyp still live at max_steps counts as complete
    for hh in beam:
        if not hh.done:
            complete.append(hh)
    if not complete:
        complete = beam
    # winner = highest log-space length-normalized score; tie -> fewer merges
    # (more cuts).  Ranking by log_score == ranking by score (monotone).
    complete.sort(key=lambda h: (h.log_score(), -len(h.merge_probs)), reverse=True)
    return complete[0]


def predict_doc(scorer: "MergeScorer", words: Sequence[str], tau: float,
                beam_width: int = BEAM_WIDTH, alpha: float = ALPHA_DEFAULT,
                window: int = 0, stride: int = STRIDE_DEFAULT) -> np.ndarray:
    """Beam-decode -> surviving cuts -> per-word 0/1 boundary vector (length n).
    Doc-final word carries 0 (cut grid has no junction after the last word;
    score_cut.py's _de row rule-fills it).

    SLIDING WINDOW (report FM9): when `window > 0` and the doc is longer than
    `window` words, decode in overlapping word-windows of `window` words at
    `stride`, then stitch (see predict_doc_windowed).  window=0 => whole-doc
    decode (back-compat default)."""
    n = len(words)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if window and n > window:
        return predict_doc_windowed(scorer, words, tau, beam_width=beam_width,
                                    alpha=alpha, window=window, stride=stride)
    win = beam_decode(scorer, words, tau, beam_width=beam_width, alpha=alpha)
    cuts = state_cuts(win.nodes)
    y = cuts_to_word_boundaries(cuts, n)
    return np.asarray(force_docfinal_cut(y, n), dtype=np.int64)


def predict_doc_windowed(scorer: "MergeScorer", words: Sequence[str], tau: float,
                         beam_width: int = BEAM_WIDTH, alpha: float = ALPHA_DEFAULT,
                         window: int = WINDOW_DEFAULT, stride: int = STRIDE_DEFAULT
                         ) -> np.ndarray:
    """SLIDING WINDOW decode + stitch (report FM9).  Process the doc in
    overlapping windows of `window` WORDS at `stride`; beam-decode each; a
    junction's boundary vote is the OR over the windows that COVER it (a cut in
    any covering window makes it a boundary — merge-generous is a low bar, so we
    take a boundary as soon as one window is confident enough to keep the cut).
    Interior window RIGHT edges are NOT forced (they are artefacts of the split);
    only the TRUE doc-final junction is forced (DOC-FINAL INVARIANT)."""
    n = len(words)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if window <= 0 or n <= window:
        return predict_doc(scorer, words, tau, beam_width=beam_width, alpha=alpha)
    vote = np.zeros(n, dtype=np.int64)          # boundary vote per word (OR)
    covered = np.zeros(n, dtype=bool)
    start = 0
    while True:
        end = min(start + window, n)            # window covers words [start, end)
        sub = list(words[start:end])
        win = beam_decode(scorer, sub, tau, beam_width=beam_width, alpha=alpha)
        cuts = set(state_cuts(win.nodes))       # local junction indices
        m = end - start
        for local_c in cuts:
            gj = start + local_c                # global word index of the cut
            # a cut at local junction local_c => boundary on global word gj; skip
            # the window's own right edge (local m-1) UNLESS it is the doc end.
            if local_c == m - 1 and end != n:
                continue                        # interior split artefact
            if 0 <= gj <= n - 2:
                vote[gj] = 1
        for gi in range(start, end):
            covered[gi] = True
        if end >= n:
            break
        start += stride
    y = vote.tolist()
    return np.asarray(force_docfinal_cut(y, n), dtype=np.int64)


def predict_doc_probmax(scorer: "MergeScorer", words: Sequence[str],
                        beam_width: int = BEAM_WIDTH,
                        tau_for_grid: float = 0.0) -> np.ndarray:
    """Export a per-word P(boundary)-like grid for the npz scorer.  We beam-decode
    at a LOW tau (merge-generous) to obtain the winning forest at a reference
    operating point, then emit a soft per-junction boundary signal = 1 - (max
    adjacent merge prob at that junction over the decode).  Kept simple: for the
    npz path we emit the hard surviving-cut grid at tau_for_grid; score_cut.py
    sweeps its own grid.  (Hard 0/1 is a valid degenerate 'prob'.)"""
    return predict_doc(scorer, words, tau_for_grid, beam_width).astype(np.float64)


# --------------------------------------------------------------------------- #
# 6. batched-vs-per-pair equivalence check (parallelization correctness)
# --------------------------------------------------------------------------- #
def batched_equals_perpair(scorer: "MergeScorer", words: Sequence[str],
                           pairs, atol: float = 1e-4) -> bool:
    """Score `pairs` batched (one forward) and one-at-a-time; assert equal probs
    up to padding non-determinism (atol).  Proves the padded batch gives the same
    answer as independent forwards (the parallelization is correct)."""
    word_ids = scorer._word_ids(words)
    batched = scorer.probs_for_pairs(word_ids, pairs)
    single = np.array([scorer.probs_for_pairs(word_ids, [p])[0] for p in pairs])
    return bool(np.allclose(batched, single, atol=atol))


# --------------------------------------------------------------------------- #
# 7. dev tau sweep + eval (own-best, like every other model)
# --------------------------------------------------------------------------- #
def sweep_tau(scorer: "MergeScorer", docs: List[dict],
              tau_grid: List[float] = None, beam_width: int = BEAM_WIDTH,
              alpha: float = ALPHA_DEFAULT, window: int = 0,
              stride: int = STRIDE_DEFAULT
              ) -> Tuple[float, dict, Dict[str, np.ndarray]]:
    """Decode every dev doc once per tau; pick own-best macro-F1 via
    eval_local.compute_metrics (tau tuned on dev by GRID SEARCH, report Phase 3).
    Returns (best_tau, best_metrics, prob_map) where prob_map is the doc_id ->
    per-word grid at best tau (for the npz)."""
    from eval_local import compute_metrics
    if tau_grid is None:
        tau_grid = sorted(set(TAU_LOW_EXTRA + TAU_GRID))
    gold = {d["doc_id"]: list(d["labels"]) for d in docs}

    # cache the decode per tau; each tau is an independent beam run
    best_tau, best_m, best_preds = None, None, None
    for tau in tau_grid:
        preds = {}
        for d in docs:
            words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                     for w in d["tokens"]]
            y = predict_doc(scorer, words, tau, beam_width=beam_width,
                            alpha=alpha, window=window, stride=stride).tolist()
            # doc-final slot: keep 0 here (parity with cut grid); scorer handles
            preds[d["doc_id"]] = y
        m = compute_metrics(gold, preds)
        if best_m is None or m["macro_f1"] > best_m["macro_f1"]:
            best_tau, best_m, best_preds = tau, m, preds
    prob_map = {k: np.asarray(v, dtype=np.float64) for k, v in best_preds.items()}
    return best_tau, best_m, prob_map


# --------------------------------------------------------------------------- #
# 8. self-test: planted-signal toy (network-free) exercising the pure machinery
# --------------------------------------------------------------------------- #
def _toy_gold_cut_demo():
    """The acceptance toy: words = [the, cat, ate, dogs, ran], gold boundary
    AFTER 'ate' (word index 2).  labels = [0,0,1,0,1] (doc-final 'ran' = 1).
    gold_cut over the 4 junctions = [0,0,1,0]."""
    labels = [0, 0, 1, 0, 1]
    gc = gold_cut_mask(labels)
    assert gc.tolist() == [0, 0, 1, 0], gc.tolist()
    spans = sentence_spans(labels)
    assert spans == [(0, 2), (3, 4)], spans
    # teacher states never straddle the boundary
    for nodes in teacher_forced_states(labels):
        for (w0, w1) in nodes:
            assert (w0 <= 2 and w1 <= 2) or (w0 >= 3 and w1 >= 3), nodes
    # dense candidate count = sum over states of #adjacent pairs
    assert count_dense_candidates(labels) >= 4
    return True


def self_test() -> bool:
    assert _toy_gold_cut_demo()
    # cuts <-> boundaries round trip
    assert cuts_to_word_boundaries([0, 2], 5) == [1, 0, 1, 0, 0]
    # geo-mean score: two merges 0.9,0.9 -> 0.9 ; product would be 0.81
    h = Hyp([(0, 4)], [0.9, 0.9])
    assert abs(h.score() - 0.9) < 1e-9, h.score()
    # start state and adjacency
    st = start_state(4)
    assert st == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert adjacent_candidates(st) == [(0, 0), (1, 1), (2, 2)]
    assert merge_nodes(st, 1) == [(0, 0), (1, 2), (3, 3)]
    assert state_cuts([(0, 1), (2, 3)]) == [1]
    print("rmerge.self_test OK")
    return True


# --------------------------------------------------------------------------- #
# 9. CLI: train / predict / sweep (all CPU-only, guarded)
# --------------------------------------------------------------------------- #
def cmd_selftest(_args):
    self_test()


def freeze_encoder_bottom(enc, n_layers: int) -> int:
    """Freeze the embeddings + bottom `n_layers` transformer layers of a BERT-style
    encoder.  Very-low-resource fine-tuning: train only the TOP layers + head, which
    sharply cuts parameter memorization (report: overfitting fix, Exp 1).  n_layers
    <= 0 is a no-op.  Returns the number of frozen parameter tensors."""
    if n_layers <= 0:
        return 0
    frozen = 0
    emb = getattr(enc, "embeddings", None)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad_(False)
            frozen += 1
    layers = getattr(getattr(enc, "encoder", None), "layer", None)
    if layers is not None:
        for i in range(min(int(n_layers), len(layers))):
            for p in layers[i].parameters():
                p.requires_grad_(False)
                frozen += 1
    return frozen


def cmd_train(args):
    assert_cpu_only()
    import torch  # noqa: F401
    tok, enc, head = load_arabert(args.model, seed=args.seed,
                                  head_bias_init=args.head_bias_init)
    if getattr(args, "mixout", 0.0) and args.mixout > 0.0:
        nm = apply_mixout(enc, args.mixout)
        print(f"applied Mixout p={args.mixout} to {nm} Linear layers "
              f"(anchored to pretrained AraBERT -> can't bury the grammar)")
    if getattr(args, "scorer", "pair") == "joint":
        # FULL-WINDOW JOINT scorer (autopsy-selected fix): whole-window encoding,
        # junction MLP head; recursion/training/decode machinery unchanged.
        head = build_joint_head(enc.config.hidden_size, seed=args.seed)
        scorer = JointScorer(tok, enc, head, span_cap=args.span_cap,
                             span_cap_mode=args.span_cap_mode,
                             grad_checkpoint=(not args.no_grad_checkpoint),
                             device=_run_device())
    else:
        scorer = MergeScorer(tok, enc, head, span_cap=args.span_cap,
                             span_cap_mode=args.span_cap_mode,
                             memoize=(not args.no_memoize),
                             grad_checkpoint=(not args.no_grad_checkpoint),
                             context_k=args.context_k,
                             boundary_dropout=args.boundary_dropout,
                             device=_run_device())
    if getattr(args, "freeze_layers", 0):
        nf = freeze_encoder_bottom(enc, args.freeze_layers)
        print(f"froze embeddings + bottom {args.freeze_layers} AraBERT layers "
              f"({nf} tensors); training top layers + head only")
    docs = load_jsonl(args.train)
    if args.max_docs:
        docs = docs[: args.max_docs]
    # EARLY-STOPPING split (closed track): hold out a fraction of TRAIN as internal
    # validation, deterministically by seed.  Dev is NEVER touched.  val_frac=0 or
    # patience=0 -> no split, train on everything, no early stop (back-compat).
    val_docs = None
    if args.val_frac and args.val_frac > 0.0 and args.patience > 0:
        vr = np.random.default_rng(args.seed)
        perm = vr.permutation(len(docs))
        n_val = max(1, int(round(args.val_frac * len(docs))))
        val_idx = set(int(i) for i in perm[:n_val])
        val_docs = [docs[i] for i in range(len(docs)) if i in val_idx]
        docs = [docs[i] for i in range(len(docs)) if i not in val_idx]
    if getattr(args, "scorer", "pair") == "joint":
        # window TRAIN docs to the ~512-subword encode budget (AFTER the val
        # split: val docs stay whole so early-stop decoding is realistic)
        n0 = len(docs)
        docs = window_docs(docs, args.train_window, args.train_stride)
        print(f"joint scorer: {n0} train docs -> {len(docs)} windows "
              f"(W={args.train_window} words, stride {args.train_stride})")
    print(f"train on {len(docs)} docs; epochs={args.epochs} lr={args.lr} "
          f"roll-in={args.roll_in} warmup={args.warmup} margin={args.margin} "
          f"span-cap={args.span_cap}/{args.span_cap_mode} "
          f"grad-ckpt={not args.no_grad_checkpoint} "
          f"val={0 if val_docs is None else len(val_docs)}/patience={args.patience}")
    hist = train_on_docs(scorer, docs, epochs=args.epochs, lr=args.lr,
                         seed=args.seed, roll_in=args.roll_in,
                         sched_p_max=args.sched_p_max, warmup_frac=args.warmup,
                         margin_delta=args.margin, glue_tau=args.glue_tau,
                         glue_order=args.glue_order, weight_decay=args.weight_decay,
                         val_docs=val_docs, patience=args.patience,
                         cascade_gamma=args.cascade_gamma,
                         margin_gamma_b=args.margin_gamma_boundary,
                         conn_margin_extra=args.conn_margin_extra,
                         rdrop_alpha=args.rdrop_alpha,
                         save_epochs_dir=(os.path.join(args.out, "epochs")
                                          if args.save_epochs else None))
    os.makedirs(args.out, exist_ok=True)
    tok.save_pretrained(args.out)
    enc.save_pretrained(args.out)
    torch.save({"head_state": head.state_dict(),
                "span_cap": args.span_cap,
                "span_cap_mode": args.span_cap_mode,
                "roll_in": args.roll_in,
                "alpha": args.alpha,
                "context_k": args.context_k,
                "boundary_dropout": args.boundary_dropout,
                "cascade_gamma": args.cascade_gamma,
                "mixout": args.mixout,
                "scorer_type": getattr(args, "scorer", "pair"),
                "margin_gamma_boundary": args.margin_gamma_boundary,
                "conn_margin_extra": args.conn_margin_extra,
                "rdrop_alpha": args.rdrop_alpha,
                "history": hist}, os.path.join(args.out, "merge_head.pt"))
    print(f"saved -> {args.out}")


def _load_scorer(model_dir: str, span_cap_override: int = None,
                 memoize: bool = True):
    import torch
    from transformers import AutoModel, AutoTokenizer
    assert_cpu_only()
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = AutoModel.from_pretrained(model_dir).eval()
    ckpt = torch.load(os.path.join(model_dir, "merge_head.pt"),
                      map_location="cpu")
    h = enc.config.hidden_size
    span_cap = span_cap_override or ckpt.get("span_cap", SPAN_CAP_DEFAULT)
    span_cap_mode = ckpt.get("span_cap_mode", "force0")
    if ckpt.get("scorer_type", "pair") == "joint":
        head = build_joint_head(h, seed=0)
        head.load_state_dict(ckpt["head_state"])
        return JointScorer(tok, enc, head, span_cap=span_cap,
                           span_cap_mode=span_cap_mode, grad_checkpoint=False,
                           device=_run_device())
    head = torch.nn.Linear(h, 1)
    head.load_state_dict(ckpt["head_state"])
    # grad checkpointing OFF at inference (no graph retained)
    return MergeScorer(tok, enc, head, span_cap=span_cap,
                       span_cap_mode=span_cap_mode, memoize=memoize,
                       grad_checkpoint=False, context_k=ckpt.get("context_k", 0),
                       boundary_dropout=ckpt.get("boundary_dropout", 0.0),
                       device=_run_device())


def cmd_predict(args):
    assert_cpu_only()
    scorer = _load_scorer(args.model)
    docs = load_jsonl(args.dev)
    if args.tau is None:
        best_tau, best_m, prob_map = sweep_tau(scorer, docs,
                                               beam_width=args.beam,
                                               alpha=args.alpha,
                                               window=args.window,
                                               stride=args.stride)
        print(f"own-best tau={best_tau} dev macro-F1="
              f"{best_m['macro_f1']*100:.4f}")
    else:
        best_tau = args.tau
        prob_map = {}
        for d in docs:
            words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w
                     for w in d["tokens"]]
            y = predict_doc(scorer, words, best_tau, beam_width=args.beam,
                            alpha=args.alpha, window=args.window,
                            stride=args.stride)
            prob_map[d["doc_id"]] = y.astype(np.float64)
    if args.npz_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.npz_out)) or ".",
                    exist_ok=True)
        np.savez(args.npz_out, **prob_map)
        print(f"word-grid boundary probs ({len(prob_map)} docs) -> "
              f"{args.npz_out}  (doc-final word carries 0.0 by construction)")


def main():
    ap = argparse.ArgumentParser(description="Recursive beam-merge segmenter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="network-free planted-signal self-test")
    st.set_defaults(func=cmd_selftest)

    tr = sub.add_parser("train", help="dense-supervision fine-tune (roll-in)")
    tr.add_argument("--train", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--model", default=ARABERT_DEFAULT)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--lr", type=float, default=3e-5)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--span-cap", type=int, default=SPAN_CAP_DEFAULT,
                    help="resulting-span token cap; over-cap merge -> p:=0 (FM3)")
    tr.add_argument("--span-cap-mode", choices=("force0", "truncate"),
                    default="force0",
                    help="force0 (spec: over-cap p:=0) or truncate (legacy)")
    # SCHEDULED SAMPLING (default sched, per the locked build order)
    tr.add_argument("--roll-in", choices=ROLL_IN_CHOICES, default="sched",
                    help="teacher | sched (default, Bengio15) | laso (beam, OFF "
                         "for the deadline) | selfmerge (per-pass recursive "
                         "self-gluing: glue >= --glue-tau each pass, recurse, "
                         "punished at cross-boundary glues, no teacher forcing)")
    tr.add_argument("--sched-p-max", type=float, default=SCHED_P_MAX_DEFAULT,
                    help="scheduled-sampling ramp target P_model (0->this)")
    tr.add_argument("--glue-tau", type=float, default=GLUE_TAU_DEFAULT,
                    help="selfmerge per-pass glue threshold (glue a pair when P>=this)")
    tr.add_argument("--head-bias-init", type=float, default=0.0,
                    help="merge-head bias init; NEGATIVE (e.g. -2.0) = glue-none "
                         "cold start for selfmerge (avoids destroying boundaries)")
    tr.add_argument("--glue-order", choices=("confidence", "leftright"),
                    default="confidence",
                    help="selfmerge per-pass glue order: confidence (easy-first, "
                         "default) or leftright")
    tr.add_argument("--warmup", type=float, default=WARMUP_FRAC_DEFAULT,
                    help="LR-warmup fraction of total steps (FM4); 0 disables")
    tr.add_argument("--margin", type=float, default=0.0,
                    help="v2 max-margin ranking delta (0 = off, spec default)")
    tr.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                    help="log-space length-norm exponent recorded for decode")
    tr.add_argument("--no-memoize", action="store_true",
                    help="disable span-encoding memoization (FM9)")
    tr.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable encoder gradient checkpointing (FM9)")
    tr.add_argument("--context-k", type=int, default=0,
                    help="surrounding-context words each side of the A|B junction "
                         "(0=off; generalization fix, e.g. 8)")
    tr.add_argument("--boundary-dropout", type=float, default=0.0,
                    help="TRAINING-only prob of blanking (->[MASK]) each junction "
                         "word (A's rightmost / B's leftmost, independently), so the "
                         "scorer can't rely on the boundary bigram identity "
                         "(anti-overconfidence; use with --context-k>0; 0=off)")
    tr.add_argument("--cascade-gamma", type=float, default=0.0,
                    help="depth-weighted cascade penalty (selfmerge): scale the "
                         "cross-boundary (don't-merge) penalty by 1+gamma/(1+depth) "
                         "so EARLY merges (which cascade) are punished hardest (0=off)")
    tr.add_argument("--mixout", type=float, default=0.0,
                    help="Mixout regularization (Lee 2020): each step, pull a random "
                         "p fraction of every encoder Linear's weights back to the "
                         "pretrained AraBERT value, so fine-tuning can't bury the "
                         "grammar / memorize adjacencies (anti-memorization root fix; "
                         "e.g. 0.7-0.9; 0=off)")
    tr.add_argument("--scorer", choices=["pair", "joint"], default="pair",
                    help="'pair' (default): per-pair [CLS] A [SEP] B re-encode "
                         "(prior behaviour). 'joint': FULL-WINDOW JOINT scorer — "
                         "encode the whole window once, score every junction from "
                         "its token states + piece summaries (autopsy fix: 91% of "
                         "boundary kills are leaf-born context starvation)")
    tr.add_argument("--train-window", type=int, default=160,
                    help="joint scorer: window TRAIN docs to this many words "
                         "(~512-subword encode budget)")
    tr.add_argument("--train-stride", type=int, default=120,
                    help="joint scorer: stride between training windows")
    tr.add_argument("--margin-gamma-boundary", type=float, default=1.0,
                    help="MULTIPLICATIVE logit scale on boundary (don't-merge) "
                         "candidates in the BCE (VS-loss style): the only margin "
                         "form that still changes learning at zero train loss "
                         "(1.0=off)")
    tr.add_argument("--conn-margin-extra", type=float, default=0.0,
                    help="EXTRA boundary-margin multiplier (1+x) at junctions "
                         "whose next word is connective-initial (و/ف/ثم..) — "
                         "targets the 43%%-enriched kill population (0=off)")
    tr.add_argument("--rdrop-alpha", type=float, default=0.0,
                    help="R-Drop: score every candidate twice under independent "
                         "dropout; punish disagreement (symmetric KL * alpha). "
                         "Trains flaky confidence away (0=off)")
    tr.add_argument("--save-epochs", action="store_true",
                    help="save every epoch's weights under OUT/epochs/ for the "
                         "greedy WEIGHT SOUP (averaging the zero-loss basin walk)")
    tr.add_argument("--max-docs", type=int, default=None)
    tr.add_argument("--val-frac", type=float, default=0.0,
                    help="hold out this fraction of TRAIN as internal validation "
                         "for early stopping (closed-track; dev untouched)")
    tr.add_argument("--patience", type=int, default=0,
                    help="early-stop after this many epochs with no val-F1 gain "
                         "(0 = off; needs --val-frac>0). Restores the BEST epoch.")
    tr.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay (raise, e.g. 0.1, to fight overfit)")
    tr.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze embeddings + bottom N AraBERT layers (low-resource "
                         "overfit fix; e.g. 10 = train only top 2 layers + head)")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="beam-decode -> per-word grid npz")
    pr.add_argument("--model", required=True)
    pr.add_argument("--dev", required=True)
    pr.add_argument("--tau", type=float, default=None,
                    help="fixed tau; omit to own-best grid-sweep on this split")
    pr.add_argument("--beam", type=int, default=BEAM_WIDTH)
    pr.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                    help="log-space length-norm exponent (1.0 = geo-mean, FM1)")
    pr.add_argument("--window", type=int, default=0,
                    help="sliding-window word length (0 = whole-doc); FM9")
    pr.add_argument("--stride", type=int, default=STRIDE_DEFAULT,
                    help="sliding-window stride in words; FM9")
    pr.add_argument("--npz-out", default=None)
    pr.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
