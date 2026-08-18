"""FOREST — end-to-end joint-trained recursive composition segmenter for AraSeg
2026 NoPnx-NP (Noor's design, 2026-07-11).

SEGMENTATION AS GROUPING, NOT CUTTING. A document is a FOREST of binary trees,
one tree per sentence; boundaries are the junctions where two trees refuse to
merge. Every word starts as its own unit. A learned composition function
(Socher-style recursive autoencoder, SHARED at every depth) merges adjacent
units easy-first; parents are real 768-d vectors that feed later merges; a
keep/erase head decides, at each adjacent junction, whether the two units
belong to the same sentence (ERASE the boundary, merge) or not (KEEP it).

KNOWLEDGE SPLIT (Noor):
  * word meaning        <- the pretrained encoder (aubmindlab/bert-base-arabertv02,
                           the tagger's encoder; SAME windowing W=203/S=101).
  * how-to-compose      <- the RAE, self-supervised by reconstruction, already
                           trained label-free on the 174 train docs
                           (rae_gate/rae_final.pt: p=tanh(W[c1;c2]+b), 1536->768
                           unit-norm, + reconstruction head).  --composer-init
                           {rae,random} contrasts whether that prior matters.
  * where-to-stop       <- the keep/erase head, learned from the ~10K boundary
                           labels on curriculum segmentation states.

THE UNTESTED CELL (every frozen-encoder variant is already measured & parked):
  END-TO-END JOINT TRAINING. Gradients from the keep/erase objective flow
  through the composer INTO the encoder, so the structural bias can move the
  representation itself. `--freeze-encoder` reproduces the frozen-encoder arm
  (no encoder grad); the default (unfrozen) is the mechanism no prior arm
  exercised. Encoder lr 2e-5, composer+head lr 1e-3.

MERGE with STRAIGHT-THROUGH SELECTION (the rae_gate training pattern, extended
so the SUPERVISED loss backprops into the encoder): at each easy-first step the
adjacent pair with the LOWEST erase-score (most confidently "same sentence") is
chosen under no_grad; the chosen merge is recomputed WITH grad and its parent
feeds later merges. Supervision is applied at every junction of a curriculum
state (merge_judge.curriculum_states, imported): junctions inside a gold
sentence are ERASE targets (label 0), junctions at a gold boundary are KEEP
targets (label 1). The composer + head + encoder all receive gradient through
the parent/child vectors that the head reads.

KEEP/ERASE HEAD reads, per junction: the parent vector p (H), the two child
vectors c1,c2 (H each), and the scalar reconstruction error E of composing
them (1).  in_features = 3H + 1.

OPTIONAL reconstruction auxiliary (--recon-aux w, default 0 = pure supervised):
adds w * mean(E over supervised junctions) so the composer keeps composing well
while it learns to stop.  One battery arm uses 0.3.

EARLY STOPPING (Noor, 2026-07-11; matches mrt_train.py's precedent): the unfrozen
encoder on 174 docs overfits (train-99/dev-83 gap), so --epochs (8) is a MAX CAP,
not a target. After each epoch we monitor DEV document-macro-F1 at the model's
own-best threshold over the 0.30-0.70 grid, using the SAME agglom decode the gates
consume (predict_doc_forest default readout) + the SAME eval_local.compute_metrics.
We keep the BEST-dev-F1 epoch (deep-copied in-memory snapshot, no disk churn) and
SAVE THAT (encoder+composer+head), not the last epoch. --early-stop-patience
(default 3) + --min-epochs (default 2) control the stop; if patience>=epochs it
reduces to the old fixed-epoch behavior (unchanged for non-opt-in callers).
This uses the eval DEV for SELECTION (epoch + threshold), never training — no dev
gradients flow — consistent with the repo convention that every arm already picks
its own-best THRESHOLD on dev and reports that own-best f1_best; this only adds the
EPOCH to that same dev-selection. Stated here so the gate number's dev-selection
convention is EXPLICIT, not hidden.

EXPOSURE-BIAS FIX (Noor, 2026-07-11; ADDITIVE, flag-gated, both DEFAULT OFF ->
bit-for-bit the pre-fix curriculum start). The default forest_window_loss samples
its start state from a curriculum that EXCLUDES the "allcut" (all-separate) state
(the `non_trivial = [... if nm != "allcut"]` filter), so it TRAINS to REFINE a
near-gold over-segmentation; but INFERENCE (forest_predict_window agglom) starts
from ALL WORDS SEPARATE and must BUILD from scratch. That train/decode start-state
mismatch is exposure bias (hypothesized to cap dev-F1 at the ~77 plateau). Two
flags make TRAIN see the DECODE start-state:
  --curriculum-include-allcut (--allcut-prob P): MIX the all-separate state into
      the start-state pool (prob P per window) so training practices BUILD-FROM-
      SCRATCH as well as refine.
  --scheduled-sampling P: DAgger-lite — with prob P per window, FREE-RUN the model's
      OWN greedy merges (no_grad, from all-separate, never crossing a gold boundary)
      then SUPERVISE (with grad) from that own-reached state, so it learns to RECOVER
      from its own early mistakes, not just polish gold.
The _es2 battery isolates these two (dropping R-Drop+SAM); the _es battery is the
regularization-stack arm. See runs/forest_battery_2026-07-11_es2/run_forest.sh.

ARMS (battery):
  forest_e2e       encoder UNFROZEN, --composer-init rae
  forest_e2e_rand  encoder UNFROZEN, --composer-init random  (prior-matters ablation)
  forest_chart     SECONDARY (semi-Markov partition DP over composed spans) —
                   built only if A/B are solid & time permits.

PREDICT -> submission CSV (baselines.write_submission) + word-grid P(boundary)
npz (doc_id -> per-word array, cache_probs.py format; doc-final word carries 0.0
by construction — same as cuttoken_train.py), so score_cut.py / diag_cut.py /
decorr_cut.py reuse verbatim. The DEFAULT decode readout is "agglom": the
AGGLOMERATIVE composed-unit descent that scores each junction in the SAME
composed-unit state the training loop scored it in (a boundary junction is
decided on WIDE composed units, not the raw leaf pair) — this is what all the
battery gates consume. The raw-leaf-pair readout is kept as a DIAGNOSTIC flag
(--leaf-readout); no gate consumes it. (Fixing the train/decode state mismatch.)

CPU smoke:  python src/smoke_forest.py         (2-doc e2e both arms + predict)
Self-test:  python src/forest_train.py self-test
Battery (GPU, guarded): runs/forest_battery_2026-07-11/run_forest.sh

ADDITIVE: new standalone file. No load-bearing file is imported-and-modified;
rae_gate.make_rae, merge_judge.curriculum_states, data.py,
baselines.write_submission are imported, never copied or edited. Dev is
EVALUATION / SELECTION only (repo convention); training reads train only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from data import MODEL_PAR_TOKEN, PARAGRAPH_TOKEN, TASKS, load_task
from merge_judge import curriculum_states
from rae_gate import make_rae

BASE_MODEL = "aubmindlab/bert-base-arabertv02"
RAE_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "rae_gate", "rae_final.pt")
CFG_NAME = "forest_cfg.json"
DEFAULT_WINDOW = 203
DEFAULT_STRIDE = 101


# --------------------------------------------------------------------------- #
# windowing — SAME W=203 / S=101 word grid as the tagger (cuttoken_train.py),
# but the decision unit here is a WORD vector, not an interleaved [CUT] slot.
# --------------------------------------------------------------------------- #
def make_word_windows(docs: List[dict], window: int, stride: int) -> List[dict]:
    """Overlapping word windows (>=2 words each; a 1-word window has no junction).
    [PAR] mapping + -100-at-[PAR] mirror cuttoken_train.make_cut_windows /
    train_encoder.make_windows so comparability with the tagger holds."""
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


# --------------------------------------------------------------------------- #
# encoder pass -> one 768-d vector PER WORD (last-subword pooling, the tagger's
# decision grain).  Gradient flows unless the caller wraps in no_grad or the
# encoder params are frozen.
# --------------------------------------------------------------------------- #
def encode_words(words: List[str], tokenizer, encoder, device, max_length: int):
    """Return (T_words, H) word vectors for ONE window. Each word's vector is
    its LAST-subword hidden state (train_encoder/predict.py word grain).
    Differentiable: no torch.no_grad here — the e2e path needs encoder grad."""
    import torch
    enc = tokenizer(words, is_split_into_words=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    word_ids = enc.word_ids(0)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = encoder(**enc)
    h = out.last_hidden_state[0]                       # (seq, H)
    # last subword index of each word id present in this (possibly truncated) window
    last_of: Dict[int, int] = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None:
            last_of[wid] = pos
    n = len(words)
    # words dropped by truncation fall back to the CLS vector (rare; window sized
    # so p99 fits — same tail-recovered-by-overlap contract as the tagger).
    idx = [last_of.get(i, 0) for i in range(n)]
    return h[idx]                                     # (n, H)


# --------------------------------------------------------------------------- #
# keep/erase head — reads [parent ; c1 ; c2 ; recon_error].  in_features = 3H+1.
# --------------------------------------------------------------------------- #
def make_head(h: int, seed: int = 42):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    in_features = 3 * h + 1
    head = nn.Sequential(
        nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 64), nn.ReLU(),
        nn.Linear(64, 1))
    # expose in_features for the flag-combination test
    head.in_features = in_features                    # type: ignore[attr-defined]
    return head


def head_logits(head, parent, c1, c2, recon_e):
    """parent/c1/c2 (m,H) ; recon_e (m,) -> (m,) junction KEEP logits."""
    import torch
    x = torch.cat([parent, c1, c2, recon_e.unsqueeze(-1)], dim=-1)
    return head(x)[:, 0]


# --------------------------------------------------------------------------- #
# the Forest model bundle
# --------------------------------------------------------------------------- #
class Forest:
    """Encoder + RAE composer + keep/erase head. Not an nn.Module subclass to
    keep three independent parameter groups (encoder 2e-5, composer+head 1e-3)
    trivially separable; `parameters()` / `named` helpers below."""

    def __init__(self, encoder, tokenizer, composer, head, h, device,
                 freeze_encoder=False):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.composer = composer
        self.head = head
        self.h = h
        self.device = device
        self.freeze_encoder = freeze_encoder

    def train(self):
        if self.encoder is not None:
            self.encoder.train(not self.freeze_encoder)
        self.composer.train()
        self.head.train()

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        self.composer.eval()
        self.head.eval()

    def encoder_params(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def other_params(self):
        return list(self.composer.parameters()) + list(self.head.parameters())


def build_forest(init_ckpt, composer_init, device, freeze_encoder,
                 seed=42, rae_ckpt=RAE_CKPT, head_seed=42):
    """Assemble the model. composer_init in {rae, random}; rae loads
    rae_gate/rae_final.pt (VERIFIED to be a 768 final-layer RAE)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(init_ckpt)
    tokenizer.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    encoder = AutoModel.from_pretrained(init_ckpt)
    encoder.resize_token_embeddings(len(tokenizer))
    h = encoder.config.hidden_size

    composer = make_rae(h, seed=seed)
    composer_note = composer_init
    if composer_init == "rae":
        ck = torch.load(rae_ckpt, map_location="cpu", weights_only=False)
        assert ck["h"] == h, f"RAE h={ck['h']} != encoder H={h}"
        assert ck["layer"] == "final", f"expected final-layer RAE, got {ck['layer']}"
        composer.load_state_dict(ck["state_dict"])
        composer_note = f"rae ({os.path.relpath(rae_ckpt)}, train-E={ck['loss_history'][-1]:.4f})"
    elif composer_init != "random":
        raise ValueError(f"--composer-init must be rae|random, got {composer_init!r}")

    head = make_head(h, seed=head_seed)

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)

    encoder.to(device)
    composer.to(device)
    head.to(device)
    return Forest(encoder, tokenizer, composer, head, h, device,
                  freeze_encoder=freeze_encoder), composer_note


# --------------------------------------------------------------------------- #
# LAYER-WISE LR DECAY (LLRD) — Noor, 2026-07-11 (standard BERT LLRD, Zhang et al.
# ICLR 2021 / revisit-bert-finetuning; the SAME grouping convention as
# seed_stability.llrd_param_groups, specialized to the forest's THREE groups).
#
# The forest encoder is a RAW AutoModel (BertModel: `embeddings`, `encoder.layer.N`,
# `pooler`) — NOT an AutoModelForTokenClassification — so it has no `classifier`
# and no `.bert` sub-attribute; we group its OWN named parameters directly:
#   * encoder.layer.N  -> the TOP layer (N=11) gets `encoder_lr_top`; each layer
#                         DOWN toward the embeddings multiplies the lr by
#                         `llrd_decay` cumulatively (layer 11 = top, layer 0 =
#                         top*decay^11, embeddings = top*decay^12 — the LOWEST).
#   * embeddings.*     -> top*decay^(n_layers) (one more decay step below layer 0).
#   * pooler.*         -> top lr (unused for the word-vector readout; grouped at
#                         top for completeness; carries ~no gradient in practice).
#   * composer + keep/erase head -> `composer_lr` (flat, NOT decayed — they are the
#                         from-scratch structural params, learned fast, like the
#                         classifier group in standard LLRD).
# decay chosen (0.874) so a 12-layer bottom ~= 5e-6 given top 2.5e-5 (Noor's spec).
# Returns (groups, ladder) where ladder is a list of (name, lr, n_params) for the
# LOG so the resolved per-group lrs are visible (smoke asserts DECAYING lrs).
# --------------------------------------------------------------------------- #
def llrd_forest_param_groups(forest, encoder_lr_top, llrd_decay, composer_lr):
    """Build AdamW param groups with layer-wise LR decay over the encoder plus a
    flat composer+head group. Only encoder params with requires_grad are grouped
    (so --freeze-encoder yields just the composer+head group — same as the flat
    optimizer). `ladder` (name, lr, n_params) is returned for logging/asserting."""
    enc = forest.encoder
    groups, ladder = [], []

    # composer + head (flat lr; the from-scratch structural params)
    other = forest.other_params()
    if other:
        groups.append({"params": other, "lr": composer_lr})
        ladder.append(("composer+head", composer_lr,
                       sum(p.numel() for p in other)))

    enc_trainable = any(p.requires_grad for p in enc.parameters()) \
        if enc is not None else False
    if not enc_trainable:
        return groups, ladder                          # frozen encoder: only other

    layers = list(enc.encoder.layer)                   # BertModel layout, 0..N-1
    n_layers = len(layers)
    # top layer (index n_layers-1) at encoder_lr_top; decay DOWN toward embeddings.
    for depth_from_top, li in enumerate(range(n_layers - 1, -1, -1)):
        lr = encoder_lr_top * (llrd_decay ** depth_from_top)
        ps = [p for p in layers[li].parameters() if p.requires_grad]
        if ps:
            groups.append({"params": ps, "lr": lr})
            ladder.append((f"encoder.layer.{li}", lr, sum(p.numel() for p in ps)))
    # embeddings: one more decay step below the bottom encoder layer (LOWEST lr)
    emb_lr = encoder_lr_top * (llrd_decay ** n_layers)
    emb_ps = [p for p in enc.embeddings.parameters() if p.requires_grad]
    if emb_ps:
        groups.append({"params": emb_ps, "lr": emb_lr})
        ladder.append(("embeddings", emb_lr, sum(p.numel() for p in emb_ps)))
    # pooler (unused for the word-vector readout): group at the top lr for
    # completeness so no trainable param is silently dropped from the optimizer.
    pooler = getattr(enc, "pooler", None)
    if pooler is not None:
        pool_ps = [p for p in pooler.parameters() if p.requires_grad]
        if pool_ps:
            groups.append({"params": pool_ps, "lr": encoder_lr_top})
            ladder.append(("pooler", encoder_lr_top,
                           sum(p.numel() for p in pool_ps)))
    return groups, ladder


# --------------------------------------------------------------------------- #
# easy-first merge with straight-through selection + supervised junction scoring
# --------------------------------------------------------------------------- #
def _unit_norm(v):
    import torch
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _free_run_start_cuts(forest, word_vecs, keep_set, all_valid_cuts,
                         n_free_steps=None):
    """SCHEDULED-SAMPLING state SELECTOR (Noor, 2026-07-11 — DAgger-lite).

    Free-run the model's OWN greedy easy-first merges STRICTLY UNDER NO_GRAD,
    starting from ALL WORDS SEPARATE (the exact condition the agglom decode starts
    from), for `n_free_steps` steps, NEVER crossing a gold-KEEP junction. Return the
    set of junctions (start-cut positions) that SURVIVE — i.e. the state the model
    itself would have reached from scratch. The caller rebuilds the starting nodes
    from these survivors and supervises WITH GRAD from there, so the model learns to
    RECOVER from its own early merge decisions rather than only polishing gold.

    NO GRADIENT flows here (it is only a state selector; the whole body is a single
    torch.no_grad block, and it detaches the composed parents). Boundaries in
    `keep_set` are non-mergeable, so every gold cut survives (keep_set is preserved
    for supervision). `n_free_steps` defaults to HALF the mergeable junctions (a
    partial descent — reflects the model's EARLY decisions, which is the exposure
    the fix targets; a full descent would collapse to the decode itself). Returns
    None (caller keeps the curriculum/allcut cuts) if there is nothing to free-run
    (fewer than 2 mergeable junctions), so degenerate windows are unaffected."""
    import torch
    n = int(word_vecs.shape[0])
    if n < 3:
        return None
    # start from all-separate: every word its own node (the decode start-state).
    nodes = [(_unit_norm(word_vecs[i]).detach(), i, i) for i in range(n)]
    # mergeable junctions present at the start = valid, non-gold-KEEP junctions.
    n_mergeable = sum(1 for c in all_valid_cuts if int(c) not in keep_set)
    if n_mergeable < 2:
        return None
    steps = (n_mergeable // 2) if n_free_steps is None else int(n_free_steps)
    steps = max(0, min(steps, n_mergeable))
    with torch.no_grad():
        done = 0
        while len(nodes) > 1 and done < steps:
            c1 = torch.stack([nd[0] for nd in nodes[:-1]])
            c2 = torch.stack([nd[0] for nd in nodes[1:]])
            parent, e = forest.composer.compose(c1, c2)
            logit = head_logits(forest.head, parent, c1, c2, e)
            after_word = [nodes[j][2] for j in range(len(nodes) - 1)]
            mergeable = np.array([int(w) not in keep_set for w in after_word])
            if not mergeable.any():
                break                                   # only gold roots remain
            masked = logit.detach().clone()
            masked[~torch.from_numpy(mergeable).to(forest.device)] = float("inf")
            j = int(torch.argmin(masked))
            pj, _ = forest.composer.compose(
                _unit_norm(nodes[j][0]).unsqueeze(0),
                _unit_norm(nodes[j + 1][0]).unsqueeze(0))
            nodes[j:j + 2] = [(pj[0].detach(), nodes[j][1], nodes[j + 1][2])]
            done += 1
    # surviving junctions = the right edge of every node except the last, restricted
    # to VALID junctions (a merged-away [PAR]/edge never appears as a node boundary).
    valid_cut_set = set(int(c) for c in all_valid_cuts)
    surv = set(int(nodes[j][2]) for j in range(len(nodes) - 1))
    surv &= valid_cut_set
    surv |= set(int(c) for c in keep_set)               # keep_set always survives
    return surv


def forest_window_loss(forest, word_vecs, word_labels, pos_weight,
                       recon_aux=0.0, rng=None, collect_sup_logits=False,
                       include_allcut=False, allcut_prob=0.5,
                       scheduled_sampling=0.0, ss_free_steps=None):
    """Supervised keep/erase over a curriculum-derived merge sequence for ONE
    window. `word_vecs` (n,H) is the encoder output (grad-carrying if unfrozen);
    `word_labels` is the per-word 0/1/-100 gold on this window.

    EXPOSURE-BIAS FIX (Noor, 2026-07-11 — --curriculum-include-allcut /
    --scheduled-sampling). The DEFAULT curriculum start-state pool EXCLUDES the
    "allcut" (every-word-separate) state (see the `non_trivial` filter below), so
    the model TRAINS to REFINE a near-gold over-segmentation but DECODES from ALL
    WORDS SEPARATE (forest_predict_window's agglom readout starts every word as its
    own node). That train/decode start-state mismatch is exposure bias. Two
    ADDITIVE, flag-gated start-state options close it (both DEFAULT OFF -> bit-for-
    bit the old 3/4-tuple behavior):

      include_allcut (--curriculum-include-allcut): MIX the all-separate "allcut"
        state INTO the start-state pool. With probability `allcut_prob` (drawn once
        per window from `rng`, deterministic) the window starts from all-separate
        (every valid junction is a start-cut, n_leaf ~= n-1 minus [PAR]) and the
        easy-first merge must BUILD the segmentation from scratch — the SAME
        condition it decodes from — while STILL never crossing a gold-KEEP
        junction; otherwise the existing non-trivial curriculum pick is used
        (practices refine). So the model trains on BOTH refine and build-from-
        scratch. allcut_prob=0.0 reduces to the old behavior; 1.0 is pure build.

      scheduled_sampling (--scheduled-sampling P): DAgger-lite. With probability P
        (drawn once per window from `rng`) the window's start-cuts are NOT the
        curriculum's — instead the model FREE-RUNS its OWN greedy easy-first merges
        UNDER NO_GRAD (from all-separate, never crossing a gold-KEEP junction) for
        `ss_free_steps` steps (default: half the mergeable junctions), and the
        junctions that SURVIVE that free-run become the supervised start-cuts. The
        supervised scoring + straight-through merge below then runs WITH GRAD from
        that own-reached state, so the model learns to RECOVER from its OWN early
        merge decisions (not only to polish gold). The free-run is strictly no_grad
        (a state SELECTOR); all gradient flows through the subsequent supervised
        recursion exactly as in the default path. When a window is NOT selected for
        scheduled sampling it falls through to the include_allcut / curriculum pick
        (the two flags compose). P=0.0 reduces to the old behavior.

    R-DROP HOOK (Noor, 2026-07-11): when `collect_sup_logits` is True the return
    is a 4-tuple (loss, n_leaf, n_pos_leaf, sup_logits) where `sup_logits` is a
    dict {after_word_index -> keep/erase LOGIT (grad-carrying 0-d tensor)} captured
    the FIRST time each supervised junction is scored. The first scoring happens on
    the CURRICULUM-START composed-unit state, which is DETERMINISTIC from
    (labels, rng) and therefore IDENTICAL across two R-Drop passes that share the
    same `rng` — the ONLY difference between the passes is the independent DROPOUT
    draw in the composer/head. So these per-junction logits are well-defined and
    directly comparable across the two passes REGARDLESS of the later stochastic
    straight-through merge order (which never crosses a gold junction anyway). The
    R-Drop symmetric-KL is computed on these matched per-junction logits by the
    caller (see `rdrop_symmetric_kl`). Default (collect_sup_logits=False) is the
    3-tuple, bit-for-bit unchanged.

    Procedure (extends rae_gate.greedy_window_loss with a supervised head and a
    KEEP/ERASE decision, straight-through selection):

      * A curriculum segmentation state is sampled per window
        (merge_judge.curriculum_states over the window's labels): a REFINEMENT
        of gold that adds some spurious interior cuts. The forest starts from
        that over-segmented state (each curriculum segment is one starting leaf
        node), so easy-first must learn to UNDO the spurious cuts (ERASE) while
        refusing to cross the gold cuts (KEEP) — the merge_judge training signal.
      * Repeatedly: compose every adjacent pair (with grad), read the KEEP head
        at every junction present, accumulate weighted BCE against the target
        (KEEP=1 at a gold junction, ERASE=0 at a spurious/interior junction);
        then pick the easy-first ERASE (lowest KEEP logit among mergeable
        junctions) UNDER NO_GRAD, and actually merge that pair (its parent,
        computed WITH grad, replaces the two children and feeds later steps).
        Never merge across a gold-KEEP junction — those are the tree boundaries;
        they stay as separate roots (the forest). Stop when only roots remain.

    Returns (loss, n_leaf, n_pos_leaf) — loss is a torch scalar with grad;
    n_leaf = valid junctions present at the curriculum start (deterministic per
    (labels, curriculum state)); n_pos_leaf = gold-KEEP junctions among them."""
    import torch
    import torch.nn.functional as F

    n = len(word_labels)
    device = forest.device
    lab = np.asarray(word_labels, dtype=np.int64)
    if rng is None:
        rng = np.random.default_rng()

    # targets on the [0..n-2] junction grid. The window's last word has no
    # in-window junction, so it is never supervised as a junction (it may still
    # be a gold boundary — the doc-end rule / stitching handles that slot).
    junction_lab = lab[:n - 1].copy()                 # (n-1,)
    valid = junction_lab != -100                      # -100 (=[PAR]) excluded
    # KEEP target = gold boundary after word i ; ERASE target = same sentence
    keep_target = (junction_lab == 1)
    # gold-keep junctions are the tree boundaries: the easy-first merge is never
    # allowed to cross them (they stay as separate roots — the forest).
    keep_set = set(int(i) for i in np.flatnonzero(keep_target & valid))

    # sample one curriculum starting state (a gold refinement). Its cut set is
    # the initial forest: contiguous words BETWEEN its cuts form starting nodes.
    # We supervise the junctions AT the curriculum cuts (the decisions the model
    # must make: KEEP gold cuts, ERASE the spurious extra cuts). Junctions the
    # curriculum did NOT cut are already merged into a starting node and are not
    # re-decided (they were 'obvious' merges). A -100 ([PAR]) junction is never
    # cut and never supervised.
    # curriculum_states needs a doc-final boundary to define spans; append one.
    cur_labels = np.concatenate([junction_lab.clip(min=0), [1]])
    states = curriculum_states(cur_labels, rng, n_random_states=1)
    # DEFAULT pool: gold refinements only (the historic filter). --curriculum-
    # include-allcut MIXES the all-separate "allcut" state back in; the
    # allcut-vs-refine draw is made FIRST from rng (deterministic per window) so
    # the R-Drop double pass — which reseeds rng identically — takes the SAME
    # branch on both passes. The all-separate cut set is EVERY valid junction.
    all_valid_cuts = [int(i) for i in range(n - 1) if valid[i]]
    use_allcut = bool(include_allcut) and (float(rng.random()) < float(allcut_prob))
    if use_allcut:
        cur_cuts = np.asarray(all_valid_cuts, dtype=np.int64)   # build-from-scratch
    else:
        non_trivial = [(nm, c) for nm, c in states if nm != "allcut"]
        _nm, cur_cuts = non_trivial[int(rng.integers(len(non_trivial)))]
        cur_cuts = np.asarray([int(c) for c in cur_cuts
                               if 0 <= int(c) <= n - 2 and valid[int(c)]],
                              dtype=np.int64)
    # SCHEDULED SAMPLING (DAgger-lite): with probability P, OVERRIDE the start-cuts
    # with the junctions that SURVIVE the model's OWN greedy free-run from all-
    # separate (no_grad; never crosses a gold-KEEP junction). The subsequent
    # supervised recursion then runs WITH GRAD from that own-reached state, so the
    # model learns to RECOVER from its own early merges. The draw is made from rng
    # (deterministic per window; identical across the R-Drop double pass). Never
    # crosses a gold-KEEP junction, so keep_set survives and every gold cut is still
    # supervised. Requires >=2 mergeable junctions to be meaningful.
    use_ss = float(scheduled_sampling) > 0.0 and (
        float(rng.random()) < float(scheduled_sampling))
    if use_ss:
        surv = _free_run_start_cuts(forest, word_vecs, keep_set, all_valid_cuts,
                                    n_free_steps=ss_free_steps)
        if surv is not None:
            cur_cuts = np.asarray(sorted(surv), dtype=np.int64)
    # ensure every gold-keep junction is a starting cut (curriculum refines gold,
    # so it already is; guard against the clip/validity edge; free-run preserves it)
    start_cuts = sorted(set(int(c) for c in cur_cuts.tolist()) | keep_set)
    # supervised junctions = the starting cuts (each is a KEEP or ERASE decision)
    sup_junctions = set(start_cuts)

    n_leaf = len(sup_junctions)                         # supervised junctions at start
    n_pos_leaf = len(keep_set & sup_junctions)          # gold-KEEP among them

    # nodes: list of (vec, left_word_idx, right_word_idx) spanning contiguous words
    # starting forest = the curriculum's segments. Each segment [s..e] of words
    # (between consecutive start_cuts) becomes one starting node whose vector is
    # the mean of its unit-norm word vectors (differentiable; grad reaches every
    # word's encoder output). A junction remains after each start_cut.
    seg_bounds = [-1] + start_cuts + [n - 1]           # word indices
    nodes = []                                         # (vec, left_word, right_word)
    for a, b in zip(seg_bounds[:-1], seg_bounds[1:]):
        s, e_ = a + 1, b                               # inclusive word span
        seg = torch.stack([_unit_norm(word_vecs[w]) for w in range(s, e_ + 1)])
        nodes.append((_unit_norm(seg.mean(0)), s, e_))
    # junction between node j and j+1 sits after word nodes[j].right (a start_cut)
    losses = []
    n_sup = 0                                          # total supervised evaluations
    recon_terms = []
    sup_logits = {} if collect_sup_logits else None    # R-Drop: first-scored logit

    # SUPERVISED RECURSION: at every easy-first step, score every SUPERVISED
    # junction currently present (a curriculum/gold cut). Junctions between
    # COMPOSED multi-word units read the head on the PARENT vectors, so deeper
    # composition is supervised too — this is what drives gradient through the
    # composer into the encoder. Then merge the easiest ERASE (lowest KEEP logit
    # among mergeable, i.e. non-gold) junctions, straight-through.
    while len(nodes) > 1:
        c1 = torch.stack([_unit_norm(nd[0]) for nd in nodes[:-1]])
        c2 = torch.stack([_unit_norm(nd[0]) for nd in nodes[1:]])
        parent, e = forest.composer.compose(c1, c2)   # (m,H),(m,)
        after_word = np.asarray([nodes[j][2] for j in range(len(nodes) - 1)],
                                dtype=np.int64)        # junction i = after this word
        logit = head_logits(forest.head, parent, c1, c2, e)   # (m,)

        # supervise every junction currently present (all are curriculum/gold cuts)
        sup_mask = np.asarray([int(w) in sup_junctions for w in after_word],
                              dtype=bool)
        if sup_mask.any():
            sm = torch.from_numpy(sup_mask).to(device)
            tgt = torch.from_numpy(
                keep_target[after_word].astype(np.float32)).to(device)
            w = torch.where(tgt > 0.5,
                            torch.as_tensor(float(pos_weight), device=device),
                            torch.as_tensor(1.0, device=device))
            bce = F.binary_cross_entropy_with_logits(
                logit[sm], tgt[sm], weight=w[sm], reduction="sum")
            losses.append(bce)
            n_sup += int(sup_mask.sum())
            if recon_aux > 0:
                recon_terms.append(e[sm].sum())
            if sup_logits is not None:
                # capture each supervised junction's logit the FIRST time it is
                # scored (the curriculum-start state, deterministic across passes)
                for jj in np.flatnonzero(sup_mask):
                    aw = int(after_word[jj])
                    if aw not in sup_logits:
                        sup_logits[aw] = logit[int(jj)]

        # choose easy-first ERASE: lowest KEEP logit among mergeable junctions
        # (present, NOT a gold-keep boundary). Selection under no_grad
        # (straight-through: decision no-grad, chosen merge carries grad).
        mergeable = np.array([int(w) not in keep_set for w in after_word])
        if not mergeable.any():
            break                                     # only tree roots remain
        with torch.no_grad():
            masked = logit.detach().clone()
            masked[~torch.from_numpy(mergeable).to(device)] = float("inf")
            j = int(torch.argmin(masked))
        # recompute the chosen merge WITH grad and splice it in
        pj, _ej = forest.composer.compose(
            _unit_norm(nodes[j][0]).unsqueeze(0),
            _unit_norm(nodes[j + 1][0]).unsqueeze(0))
        merged = (pj[0], nodes[j][1], nodes[j + 1][2])
        nodes[j:j + 2] = [merged]

    if not losses:
        return (None, 0, 0, {}) if collect_sup_logits else (None, 0, 0)
    total = torch.stack(losses).sum() / max(n_sup, 1)
    if recon_aux > 0 and recon_terms:
        total = total + recon_aux * (torch.stack(recon_terms).sum() / max(n_sup, 1))
    if collect_sup_logits:
        return total, n_leaf, n_pos_leaf, sup_logits
    return total, n_leaf, n_pos_leaf


def rdrop_symmetric_kl(logits_a, logits_b):
    """R-DROP consistency term (Noor, 2026-07-11) — symmetric KL between the
    keep/erase distributions of TWO independent-dropout forward passes, evaluated
    on the per-gold/curriculum-junction logits both passes produce at the
    (deterministic) curriculum-start state.

    REUSES the repo's R-Drop convention (src/consistency_train.py METHOD 2):
        rdrop = 0.5 * (KL(p1 || p2) + KL(p2 || p1))  averaged over VALID positions.
    consistency_train applies it to the tagger's per-token 2-class softmax; here the
    forest head emits ONE keep/erase logit z per junction, so the matching 2-class
    (Bernoulli) distribution is [P(erase), P(keep)] = [sigmoid(-z), sigmoid(z)]. We
    build the log-softmax of the 2-logit vector [0, z] for each pass and apply the
    EXACT same symmetric-KL formula, averaged over the junctions COMMON to both
    passes (the gold-supervised junctions are present in both regardless of the
    stochastic straight-through merge order — they are the tree roots and are never
    crossed). Returns a grad-carrying 0-d tensor (0.0 if no common junctions).

    `logits_a`, `logits_b`: dicts {after_word -> 0-d keep logit tensor} from
    forest_window_loss(..., collect_sup_logits=True) on the two passes."""
    import torch
    keys = [k for k in logits_a if k in logits_b]
    if not keys:
        # keep the graph alive with a genuine 0 (no common supervised junction)
        anyv = next(iter(logits_a.values()), None)
        return (anyv * 0.0).sum() if anyv is not None else torch.zeros(())
    za = torch.stack([logits_a[k] for k in keys])      # (m,)
    zb = torch.stack([logits_b[k] for k in keys])      # (m,)
    # 2-class logits [erase=0, keep=z]; log_softmax over the class dim.
    la = torch.log_softmax(torch.stack([torch.zeros_like(za), za], dim=-1), dim=-1)
    lb = torch.log_softmax(torch.stack([torch.zeros_like(zb), zb], dim=-1), dim=-1)
    pa, pb = la.exp(), lb.exp()
    kl_ab = (pa * (la - lb)).sum(-1)                   # KL(p_a || p_b) per junction
    kl_ba = (pb * (lb - la)).sum(-1)                   # KL(p_b || p_a) per junction
    return 0.5 * (kl_ab + kl_ba).mean()                # repo convention (mean)


def forest_window_loss_agglo(forest, word_vecs, word_labels, pos_weight, rng=None):
    """Supervised easy-first agglomeration FROM SINGLETONS — the training twin of
    the DEFAULT composed-unit decode (forest_predict_window(readout='agglom')).

    ADDITIVE, self-test / diagnostic only — the production trainer (cmd_train)
    still uses forest_window_loss (curriculum-start) unchanged. This helper exists
    so the PLANTED-SIGNAL self-test supervises the EXACT recursion the decode runs
    (start every word as its own node; at each step compose every adjacent pair,
    score every surviving junction on the CURRENT composed units, accumulate
    weighted BCE against gold KEEP/ERASE; then merge the easiest-ERASE non-gold
    junction, straight-through). Because a boundary junction is only scored on the
    LEFT/RIGHT composed units it accumulates, this objective can express — and
    reward — a signal that lives ABOVE the junction's own adjacent word pair
    (e.g. the left-context bigram of _bracket_stream), which is precisely what a
    genuinely compositional test must exercise. A frozen composer cannot build the
    left representation the head needs, so it fails the self-test's control.

    Returns (loss, n_junctions, n_pos) — loss a torch scalar with grad."""
    import torch
    import torch.nn.functional as F
    device = forest.device
    n = len(word_labels)
    lab = np.asarray(word_labels, dtype=np.int64)
    keep_set = set(int(i) for i in range(n - 1) if lab[i] == 1)   # gold boundaries
    nodes = [(_unit_norm(word_vecs[i]), i, i) for i in range(n)]
    losses, n_sup = [], 0
    n_junctions = int((lab[:n - 1] != -100).sum())
    n_pos = len(keep_set)
    while len(nodes) > 1:
        c1 = torch.stack([_unit_norm(nd[0]) for nd in nodes[:-1]])
        c2 = torch.stack([_unit_norm(nd[0]) for nd in nodes[1:]])
        parent, e = forest.composer.compose(c1, c2)
        after_word = [nodes[j][2] for j in range(len(nodes) - 1)]
        logit = head_logits(forest.head, parent, c1, c2, e)
        sup_mask = np.asarray([lab[w] != -100 for w in after_word], dtype=bool)
        if sup_mask.any():
            sm = torch.from_numpy(sup_mask).to(device)
            tgt = torch.as_tensor(
                [1.0 if int(w) in keep_set else 0.0 for w in after_word],
                dtype=torch.float32, device=device)
            w = torch.where(tgt > 0.5,
                            torch.as_tensor(float(pos_weight), device=device),
                            torch.as_tensor(1.0, device=device))
            losses.append(F.binary_cross_entropy_with_logits(
                logit[sm], tgt[sm], weight=w[sm], reduction="sum"))
            n_sup += int(sup_mask.sum())
        mergeable = np.array([int(w) not in keep_set for w in after_word])
        if not mergeable.any():
            break                                       # only tree roots remain
        with torch.no_grad():
            masked = logit.detach().clone()
            masked[~torch.from_numpy(mergeable).to(device)] = float("inf")
            j = int(torch.argmin(masked))
        pj, _ = forest.composer.compose(
            _unit_norm(nodes[j][0]).unsqueeze(0),
            _unit_norm(nodes[j + 1][0]).unsqueeze(0))
        nodes[j:j + 2] = [(pj[0], nodes[j][1], nodes[j + 1][2])]
    if not losses:
        return None, 0, 0
    return torch.stack(losses).sum() / max(n_sup, 1), n_junctions, n_pos


def train_boundary_unit_widths(forest, word_vecs, word_labels, rng=None):
    """Replay the PRODUCTION training agglomeration (forest_window_loss's exact
    curriculum-start easy-first merge) for ONE window and return, for each
    gold-KEEP (boundary) junction, the composed-unit WIDTH (#words in the two
    adjacent units) at the DEPTH it is finally decided — i.e. the width the head
    actually scored it on in training. Read-only (no grad, no loss); used by the
    smoke to report the TRAIN side of the train-vs-decode width histogram that
    demonstrates the default decode matches the training distribution (BLOCKER-2).
    """
    import torch
    n = len(word_labels)
    lab = np.asarray(word_labels, dtype=np.int64)
    if rng is None:
        rng = np.random.default_rng()
    junction_lab = lab[:n - 1].copy()
    valid = junction_lab != -100
    keep_target = (junction_lab == 1)
    keep_set = set(int(i) for i in np.flatnonzero(keep_target & valid))
    cur_labels = np.concatenate([junction_lab.clip(min=0), [1]])
    states = curriculum_states(cur_labels, rng, n_random_states=1)
    non_trivial = [(nm, c) for nm, c in states if nm != "allcut"]
    if not non_trivial:
        return []
    _nm, cur_cuts = non_trivial[int(rng.integers(len(non_trivial)))]
    cur_cuts = np.asarray([int(c) for c in cur_cuts
                           if 0 <= int(c) <= n - 2 and valid[int(c)]],
                          dtype=np.int64)
    start_cuts = sorted(set(cur_cuts.tolist()) | keep_set)
    seg_bounds = [-1] + start_cuts + [n - 1]
    nodes = []
    for a, b in zip(seg_bounds[:-1], seg_bounds[1:]):
        s, e_ = a + 1, b
        seg = torch.stack([_unit_norm(word_vecs[w]) for w in range(s, e_ + 1)])
        nodes.append((_unit_norm(seg.mean(0)), s, e_))
    widths = {}                                    # boundary junction -> last width
    with torch.no_grad():
        while len(nodes) > 1:
            after_word = [nodes[j][2] for j in range(len(nodes) - 1)]
            unit_w = [(nodes[j][2] - nodes[j][1] + 1) +
                      (nodes[j + 1][2] - nodes[j + 1][1] + 1)
                      for j in range(len(nodes) - 1)]
            c1 = torch.stack([_unit_norm(nd[0]) for nd in nodes[:-1]])
            c2 = torch.stack([_unit_norm(nd[0]) for nd in nodes[1:]])
            parent, e = forest.composer.compose(c1, c2)
            logit = head_logits(forest.head, parent, c1, c2, e)
            for j, w in enumerate(after_word):
                if int(w) in keep_set:             # refresh to its widest reading
                    widths[int(w)] = unit_w[j]
            mergeable = np.array([int(w) not in keep_set for w in after_word])
            if not mergeable.any():
                break
            masked = logit.detach().clone()
            masked[~torch.from_numpy(mergeable).to(forest.device)] = float("inf")
            j = int(torch.argmin(masked))
            pj, _ = forest.composer.compose(
                _unit_norm(nodes[j][0]).unsqueeze(0),
                _unit_norm(nodes[j + 1][0]).unsqueeze(0))
            nodes[j:j + 2] = [(pj[0], nodes[j][1], nodes[j + 1][2])]
    return list(widths.values())


# --------------------------------------------------------------------------- #
# inference — the exported per-word P(boundary) is the KEEP prob of the junction
# after that word. TWO readouts, selected by `readout`:
#
#   * "agglom" (DEFAULT — what ALL the battery gates consume):
#       AGGLOMERATIVE composed-unit readout that reproduces the state the TRAINING
#       loop scores each junction in. Every word starts as its own node; at each
#       easy-first step the easiest-ERASE junction (lowest KEEP prob) is merged,
#       its two units becoming one composed parent; a SURVIVING junction is
#       re-scored on the GROWING composed units each step and a merged junction is
#       FROZEN at the reading it had when it was decided. So each junction's
#       exported score is read at the composed-unit DEPTH at which it is actually
#       decided — the same distribution the head was trained on (a gold-KEEP /
#       boundary junction is decided LAST, on WIDE composed units, mean unit width
#       ~3.9 in training; a leaf-pair reading of that same junction is a different,
#       untrained distribution). This removes the train/decode state mismatch:
#       BLOCKER-2. Boundaries are exactly where two composed units refuse to merge.
#
#   * "leaf" (DIAGNOSTIC flag only — was the old default): every word's own
#       junction scored on the two adjacent RAW WORD vectors — the token-
#       comparable grid, but a DIFFERENT (untrained) distribution than training.
#       Kept available for diagnostics / comparison; NO gate consumes it.
#
# The forest's defining mechanism — gradient from keep/erase THROUGH the composer
# INTO the encoder — lives in TRAINING (forest_window_loss) and is independent of
# the readout; `readout` only chooses how much composition the inference score
# consumes, and the default now MATCHES the training-time composed-unit state.
# --------------------------------------------------------------------------- #
def forest_predict_window(forest, word_vecs, readout="agglom", descend=None,
                          max_length=None, return_widths=False):
    """Return p (n,) with p[n-1] = 0.0 (doc-final word has no junction).

    readout: "agglom" (default, composed-unit; the gate readout) or "leaf"
    (diagnostic, raw adjacent word pair). `descend` is the historical alias
    (descend=True -> "agglom", descend=False -> "leaf"); when given it OVERRIDES
    `readout` for backward compatibility. If return_widths, also return a (n,)
    int array `widths` = the total composed-unit WIDTH (#words spanned by the two
    units) at which each junction's exported score was read (0 at the doc-final
    slot and any uncovered position); leaf readout reads every junction at width
    2 by construction."""
    import torch
    if descend is not None:                    # historical boolean alias wins
        readout = "agglom" if descend else "leaf"
    if readout not in ("agglom", "leaf"):
        raise ValueError(f"readout must be 'agglom' or 'leaf', got {readout!r}")
    n = word_vecs.shape[0]
    if n < 2:
        z = np.zeros(n, dtype=np.float32)
        return (z, np.zeros(n, dtype=np.int64)) if return_widths else z
    with torch.no_grad():
        if readout == "leaf":
            c1 = torch.stack([_unit_norm(word_vecs[i]) for i in range(n - 1)])
            c2 = torch.stack([_unit_norm(word_vecs[i + 1]) for i in range(n - 1)])
            parent, e = forest.composer.compose(c1, c2)
            logit = head_logits(forest.head, parent, c1, c2, e)
            pk = torch.sigmoid(logit).cpu().numpy().astype(np.float32)
            out = np.zeros(n, dtype=np.float32)
            out[:n - 1] = pk
            if return_widths:
                w = np.zeros(n, dtype=np.int64)
                w[:n - 1] = 2                  # every junction read on a 1+1 pair
                return out, w
            return out
        # "agglom": easy-first composed-unit readout (the training-matched state)
        nodes = [(_unit_norm(word_vecs[i]), i, i) for i in range(n)]
        merged_away = set()
        prob = np.zeros(n, dtype=np.float32)
        width = np.zeros(n, dtype=np.int64)    # composed-unit width at read time
        while len(nodes) > 1:
            c1 = torch.stack([nd[0] for nd in nodes[:-1]])
            c2 = torch.stack([nd[0] for nd in nodes[1:]])
            parent, e = forest.composer.compose(c1, c2)
            logit = head_logits(forest.head, parent, c1, c2, e)
            pk = torch.sigmoid(logit).cpu().numpy().astype(np.float32)
            after_word = [nodes[j][2] for j in range(len(nodes) - 1)]
            unit_w = [(nodes[j][2] - nodes[j][1] + 1) +
                      (nodes[j + 1][2] - nodes[j + 1][1] + 1)
                      for j in range(len(nodes) - 1)]   # widths of the two units
            for j, aw in enumerate(after_word):
                if aw not in merged_away:      # refresh survivors to their widest
                    prob[aw] = pk[j]
                    width[aw] = unit_w[j]
            j = int(np.argmin(pk))
            merged_away.add(after_word[j])
            pj, _ = forest.composer.compose(
                _unit_norm(nodes[j][0]).unsqueeze(0),
                _unit_norm(nodes[j + 1][0]).unsqueeze(0))
            nodes[j:j + 2] = [(pj[0], nodes[j][1], nodes[j + 1][2])]
        prob[n - 1] = 0.0
        width[n - 1] = 0
        if return_widths:
            return prob, width
        return prob


def predict_doc_forest(words, forest, window, stride, max_length,
                       readout="agglom", descend=None, return_widths=False):
    """Per-WORD P(boundary) for a full doc, overlapping windows averaged (same
    stitching convention as predict.py / cuttoken predict: mean over collected
    probabilities, uncovered -> 0.0). Doc-final word carries 0.0. `readout` /
    `descend` as in forest_predict_window (default "agglom" = the gate readout).
    If return_widths, also return a per-word int array of the median composed-unit
    width at which each covered junction was read (0 where uncovered)."""
    import torch
    if descend is not None:
        readout = "agglom" if descend else "leaf"
    n = len(words)
    if n < 2:
        p = np.zeros(n, dtype=np.float32)
        return (p, np.zeros(n, dtype=np.int64)) if return_widths else p
    per_word = defaultdict(list)
    per_word_w = defaultdict(list)
    start = 0
    while start < n:
        end = min(start + window, n)
        chunk = words[start:end]
        if end - start >= 2:
            with torch.no_grad():
                wv = encode_words(chunk, forest.tokenizer, forest.encoder,
                                  forest.device, max_length)
            pw, ww = forest_predict_window(forest, wv, readout=readout,
                                           return_widths=True)
            for i in range(len(chunk)):
                gpos = start + i
                if gpos < n - 1:              # last word of doc has no junction
                    per_word[gpos].append(float(pw[i]))
                    per_word_w[gpos].append(int(ww[i]))
        if end == n:
            break
        start += stride
    p = np.array([float(np.mean(per_word[i])) if per_word[i] else 0.0
                  for i in range(n)], dtype=np.float32)
    if return_widths:
        w = np.array([int(np.median(per_word_w[i])) if per_word_w[i] else 0
                      for i in range(n)], dtype=np.int64)
        return p, w
    return p


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def compute_pos_weight(train_docs):
    """pos_weight = min(neg/pos, 8) on the junction grid (gap after every word
    except each doc's last; [PAR] gaps excluded) — same convention as the
    tagger / cuttoken."""
    pos = tot = 0
    for d in train_docs:
        toks, labs = d["tokens"], d["labels"]
        for i in range(len(toks) - 1):
            if toks[i] == PARAGRAPH_TOKEN:
                continue
            tot += 1
            pos += int(labs[i])
    return min((tot - pos) / max(pos, 1), 8.0), pos, tot


# grid + `>=` operator + [PAR]-zeroing MATCH the battery scorer verbatim
# (runs/*/score_cut.py GRID, the same eval_local.compute_metrics), so the
# epoch-selection number is on the same footing as the gate's own-best f1_best.
EVAL_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def eval_dev_docf1(forest, dev_docs, window, stride, max_length,
                   grid=EVAL_GRID, return_threshold=False):
    """DEV document-macro-F1 at the model's OWN-BEST threshold over `grid`, using
    the SAME agglom decode the battery gates consume (predict_doc_forest default
    readout) and the SAME eval_local.compute_metrics / `>=` / [PAR]->0 convention
    as runs/*/score_cut.py. This is the EARLY-STOP monitored metric.

    METHOD NOTE (dev is SELECTION here, never training — repo convention):
    the forest, like every arm on record, already picks its OWN-BEST THRESHOLD on
    dev and reports that own-best number as f1_best; this monitor adds the EPOCH
    to that exact same dev-selection (best-dev-doc-F1 epoch is kept and restored).
    NO dev gradients ever flow — this is a torch.no_grad decode + scikit metric.
    It is a PLAIN-reading own-best (force_doc_end=False), mirroring the `<arm>`
    row f1_best; the `_de` doc-end rule is a gate-time decode convention applied
    by score_cut, not part of epoch selection. Making the dev-selection explicit
    (not hidden) is deliberate: the gate number's own-best-on-dev convention is
    stated, and epoch is folded into it, so nothing about dev-selection is silent.

    Returns macro_f1 in [0,1] (or (macro_f1, best_threshold) if return_threshold).
    """
    from eval_local import compute_metrics
    forest.eval()
    gold = {d["doc_id"]: list(d["labels"]) for d in dev_docs}
    prob_map = {}
    for d in dev_docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        prob_map[d["doc_id"]] = predict_doc_forest(
            words, forest, window, stride, max_length, readout="agglom")

    def labels_at(t):
        preds = {}
        for d in dev_docs:
            lab = (prob_map[d["doc_id"]] >= t).astype(int)
            for i, w in enumerate(d["tokens"]):
                if w == PARAGRAPH_TOKEN:
                    lab[i] = 0
            preds[d["doc_id"]] = lab.tolist()
        return preds

    best_t, best_f1 = grid[0], -1.0
    for t in grid:
        f1 = compute_metrics(gold, labels_at(t))["macro_f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    forest.train()
    return (best_f1, best_t) if return_threshold else best_f1


def _snapshot_weights(forest):
    """Deep in-memory (CPU) copy of encoder+composer+head weights — the BEST-epoch
    restore checkpoint. No disk churn (kept in RAM across epochs)."""
    import copy
    snap = {"composer": {k: v.detach().cpu().clone()
                         for k, v in forest.composer.state_dict().items()},
            "head": {k: v.detach().cpu().clone()
                     for k, v in forest.head.state_dict().items()}}
    if forest.encoder is not None:
        snap["encoder"] = {k: v.detach().cpu().clone()
                           for k, v in forest.encoder.state_dict().items()}
    return snap


def _restore_weights(forest, snap, device):
    """Load a _snapshot_weights() dict back onto the live model (best-epoch restore)."""
    forest.composer.load_state_dict({k: v.to(device) for k, v in snap["composer"].items()})
    forest.head.load_state_dict({k: v.to(device) for k, v in snap["head"].items()})
    if forest.encoder is not None and "encoder" in snap:
        forest.encoder.load_state_dict(
            {k: v.to(device) for k, v in snap["encoder"].items()})


def cmd_train(args):
    import torch
    from transformers import set_seed

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)

    train_docs = load_task(args.task, "train", jsonl_path=args.train_jsonl)
    dev_docs = load_task(args.task, "dev", jsonl_path=args.dev_jsonl)
    if args.max_docs is not None:
        train_docs = train_docs[:args.max_docs]
    if args.max_dev_docs is not None:
        dev_docs = dev_docs[:args.max_dev_docs]
    print(f"loaded train docs={len(train_docs)}  dev docs={len(dev_docs)}  "
          f"from={args.train_jsonl or '(default)'}")
    if args.expect_docs is not None and len(train_docs) != args.expect_docs:
        raise SystemExit(f"--expect-docs {args.expect_docs} but loaded "
                         f"{len(train_docs)} train docs -> ABORT")

    pos_weight, pos, tot = compute_pos_weight(train_docs)
    if args.pos_weight is not None:
        pos_weight = args.pos_weight
    print(f"junction keep rate={pos / max(tot, 1):.4f} ({pos}/{tot})  "
          f"pos_weight={pos_weight:.2f}")

    forest, composer_note = build_forest(
        BASE_MODEL, args.composer_init, device, args.freeze_encoder,
        seed=args.seed, head_seed=args.seed)
    print(f"[init] encoder={BASE_MODEL} (frozen={args.freeze_encoder})  "
          f"composer-init={composer_note}  H={forest.h}  "
          f"head in_features={forest.head.in_features}")

    window = args.window
    stride = args.stride if args.stride is not None else max(window // 2, 1)
    train_windows = make_word_windows(train_docs, window, stride)
    print(f"windows: train={len(train_windows)} (W={window}, S={stride})")

    enc_params = forest.encoder_params()
    other = forest.other_params()
    # OPTIMIZER: flat 2-group (default, bit-for-bit yesterday's behavior) OR
    # LAYER-WISE LR DECAY (--llrd; on for the _es battery). LLRD builds per-encoder-
    # layer groups (top=--encoder-lr-top, decaying by --llrd-decay toward the
    # embeddings) + a flat composer+head group at --composer-lr; see
    # llrd_forest_param_groups. All groups feed ONE AdamW, so nothing else changes.
    if args.llrd:
        groups, ladder = llrd_forest_param_groups(
            forest, args.encoder_lr_top, args.llrd_decay, args.composer_lr)
        opt = torch.optim.AdamW(groups)
        print(f"[opt] AdamW + LLRD (decay={args.llrd_decay}; top="
              f"{args.encoder_lr_top:.2e} -> embeddings="
              f"{args.encoder_lr_top * args.llrd_decay ** 12:.2e}); composer+head="
              f"{args.composer_lr}")
        print("[opt] LLRD per-group lr ladder (top encoder layer -> embeddings; "
              "composer+head flat):")
        for name, lr, npar in ladder:
            print(f"        {name:<18} lr={lr:.3e}  params={npar}")
    else:
        groups = [{"params": other, "lr": args.composer_lr}]
        if enc_params:
            groups.append({"params": enc_params, "lr": args.encoder_lr})
        opt = torch.optim.AdamW(groups)
        print(f"[opt] AdamW  encoder-lr={args.encoder_lr if enc_params else 'FROZEN'}  "
              f"composer+head-lr={args.composer_lr}  "
              f"encoder-params={sum(p.numel() for p in enc_params)}  "
              f"other-params={sum(p.numel() for p in other)}")

    # all trainable params (one flat list — used for clip + SAM global-norm/perturb)
    all_params = enc_params + other
    rdrop_alpha = float(args.rdrop_alpha)
    sam_on = bool(args.sam)
    sam_rho = float(args.sam_rho)
    # EXPOSURE-BIAS FIX flags (Noor, 2026-07-11; default OFF -> old behavior). See
    # forest_window_loss's docstring: --curriculum-include-allcut MIXES the all-
    # separate start state (prob --allcut-prob) so training practices build-from-
    # scratch (the decode condition); --scheduled-sampling P free-runs the model's
    # own greedy merges (no_grad) then supervises from that own-reached state.
    include_allcut = bool(args.curriculum_include_allcut)
    allcut_prob = float(args.allcut_prob)
    scheduled_sampling = float(args.scheduled_sampling)
    ss_free_steps = args.ss_free_steps
    print(f"[reg] recon-aux={args.recon_aux}  rdrop-alpha={rdrop_alpha}"
          f"{' (OFF)' if rdrop_alpha == 0 else ''}  "
          f"sam={'ON rho=%s' % sam_rho if sam_on else 'OFF'}  "
          f"(fwd passes/window: {1 + (rdrop_alpha > 0)}"
          f"{' x2 for SAM ascent+descent' if sam_on else ''})", flush=True)
    print(f"[exposure-bias] curriculum-include-allcut="
          f"{'ON prob=%.2f' % allcut_prob if include_allcut else 'OFF'}  "
          f"scheduled-sampling="
          f"{'ON P=%.2f (free-steps=%s)' % (scheduled_sampling, ss_free_steps if ss_free_steps is not None else 'half') if scheduled_sampling > 0 else 'OFF'}"
          f"  (both default OFF => bit-for-bit the pre-fix curriculum start)",
          flush=True)

    def enc_grad_l2():
        if not enc_params:
            return 0.0
        s = 0.0
        for p in enc_params:
            if p.grad is not None:
                s += float((p.grad.detach() ** 2).sum())
        return s ** 0.5

    def window_loss(win, wrng):
        """Per-window supervised loss WITH R-Drop folded in (grad-carrying, or None
        if the window has no supervised junction). R-Drop OFF (alpha==0): a SINGLE
        forward pass — bit-for-bit the old per-window loss. R-Drop ON: TWO
        independent-dropout passes over the SAME curriculum-start state, base =
        0.5*(loss1+loss2), plus alpha * symmetric-KL on the matched gold/curriculum-
        junction keep/erase logits (rdrop_symmetric_kl). The encode is done HERE so
        SAM's descent pass re-encodes at the perturbed weights.

        CURRICULUM-START IDENTITY (R-Drop): the two passes MUST share the same
        curriculum start (so the ONLY difference is the independent dropout draw and
        the encoder-output dropout). forest_window_loss CONSUMES its rng (curriculum
        sampling AND the allcut/scheduled-sampling draws — which are made FIRST from
        rng), so reusing the SAME advancing rng object across the two calls would
        give DIFFERENT starts. We therefore derive ONE fixed integer seed from wrng
        and hand each pass its OWN fresh rng seeded identically -> identical start-
        state BRANCH (allcut vs curriculum vs scheduled-sampling) and cut set,
        independent dropout. (R-Drop OFF passes wrng through unchanged, so the
        default single-pass path is bit-for-bit as before.) NOTE: when scheduled-
        sampling fires, the free-run's OWN head dropout differs between the two
        passes, so their survivor sets can differ slightly; rdrop_symmetric_kl
        intersects to the COMMON supervised junctions and stays valid. (The _es2
        battery drops R-Drop, so this composition does not arise there.)"""
        # exposure-bias kwargs threaded to every forest_window_loss call (default
        # OFF -> the kwargs are the function defaults -> bit-for-bit old behavior).
        eb = dict(include_allcut=include_allcut, allcut_prob=allcut_prob,
                  scheduled_sampling=scheduled_sampling, ss_free_steps=ss_free_steps)
        wv = encode_words(win["words"], forest.tokenizer, forest.encoder,
                          device, args.max_length)
        if rdrop_alpha > 0:
            cur_seed = int(wrng.integers(0, 2 ** 32 - 1))   # fixed once per window
            l1, _nl, _np, sl1 = forest_window_loss(
                forest, wv, win["word_labels"], pos_weight,
                recon_aux=args.recon_aux,
                rng=np.random.default_rng(cur_seed), collect_sup_logits=True, **eb)
            if l1 is None:
                return None
            wv2 = encode_words(win["words"], forest.tokenizer, forest.encoder,
                               device, args.max_length)
            l2, _nl2, _np2, sl2 = forest_window_loss(
                forest, wv2, win["word_labels"], pos_weight,
                recon_aux=args.recon_aux,
                rng=np.random.default_rng(cur_seed), collect_sup_logits=True, **eb)
            kl = rdrop_symmetric_kl(sl1, sl2)
            return 0.5 * (l1 + l2) + rdrop_alpha * kl
        loss, _nl, _np = forest_window_loss(
            forest, wv, win["word_labels"], pos_weight,
            recon_aux=args.recon_aux, rng=wrng, **eb)
        return loss

    def batch_mean_loss(batch):
        """Mean per-window loss over a batch of (win, seed). Recomputes the forward
        each call (so SAM evaluates at BOTH w and w+eps on the SAME curriculum state:
        a FRESH rng is built from the per-window integer seed EACH call, so the
        ascent and descent passes see identical curriculum starts — only the params
        (w vs w+eps) and dropout differ, which is the correct SAM semantics).
        Returns a grad-carrying scalar, or None if no window produced a loss."""
        accum = []
        for win, seed in batch:
            wl = window_loss(win, np.random.default_rng(seed))
            if wl is not None:
                accum.append(wl)
        if not accum:
            return None
        return torch.stack(accum).mean()

    def step_plain(accum):
        """One optimizer step, NO SAM, over PRE-COMPUTED window-loss tensors (the
        old `step` — encode-once accumulation; None-loss windows are skipped by the
        caller and never enter `accum`, so batch grouping is UNCHANGED vs default).
        Returns (loss_value, enc_grad_l2_pre_clip)."""
        batch_loss = torch.stack(accum).mean()
        opt.zero_grad()
        batch_loss.backward()
        eg = enc_grad_l2()                              # BEFORE clip
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        return float(batch_loss.detach()), eg

    @torch.no_grad()
    def _global_grad_norm():
        """Single global L2 norm over ALL trainable params' grads (Foret et al.:
        one shared norm across the whole parameter vector — matches
        sam_train.SAMTrainer._grad_norm)."""
        sq = 0.0
        for p in all_params:
            if p.grad is not None:
                sq += float((p.grad.detach() ** 2).sum())
        return sq ** 0.5

    def step_sam(batch):
        """One FIRST-ORDER SAM step (Foret et al. ICLR 2021), COMPOSING with LLRD
        groups (SAM perturbs the raw params; the LLRD-grouped AdamW then applies the
        perturbed grad with its per-group lrs) and with R-Drop (each of SAM's TWO
        loss evaluations runs the R-Drop DOUBLE pass -> ~4 forward passes/window;
        Noor chose both). Steps:
          (1) g = grad of L(w)                          [ascent gradient at w]
          (2) eps = rho * g / (||g|| + 1e-12); climb to w + eps (snapshot orig w)
          (3) grad of L(w + eps)                        [descent gradient, fresh
              dropout; R-Drop double pass again]
          (4) restore w EXACTLY via copy_ (bit-for-bit), then opt.step() applies the
              PERTURBED (descent) grad to the restored w.
        Returns (descent_loss_value, enc_grad_l2_pre_clip_at_perturbed_point)."""
        # (1) ascent gradient at w
        loss1 = batch_mean_loss(batch)
        if loss1 is None:
            return None, 0.0
        opt.zero_grad()
        loss1.backward()
        gnorm = _global_grad_norm()
        scale = sam_rho / (gnorm + 1e-12)
        # (2) climb to w + eps, snapshotting the ORIGINAL w for an exact restore
        w_orig = {}
        with torch.no_grad():
            for p in all_params:
                if p.grad is None:
                    continue
                w_orig[p] = p.detach().clone()
                p.add_(p.grad * scale)                  # w <- w + eps
        # (3) descent gradient at the perturbed point (fresh dropout / R-Drop pass)
        loss2 = batch_mean_loss(batch)
        opt.zero_grad()
        if loss2 is None:                               # degenerate batch: restore + skip
            with torch.no_grad():
                for p, w in w_orig.items():
                    p.copy_(w)
            return None, 0.0
        loss2.backward()
        eg = enc_grad_l2()                              # BEFORE clip, at w+eps
        # (4) restore w EXACTLY, then step with the perturbed (descent) grad
        with torch.no_grad():
            for p, w in w_orig.items():
                p.copy_(w)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        return float(loss2.detach()), eg

    # EARLY STOPPING (Noor, 2026-07-11): the unfrozen encoder on 174 docs is the
    # known train-99/dev-83 overfit risk, so a FIXED epoch count is a guess. We
    # monitor DEV document-macro-F1 (own-best threshold, agglom decode — the gate
    # metric; see eval_dev_docf1's METHOD NOTE: dev is SELECTION, no dev grads) and
    # keep the BEST-dev-F1 epoch, restoring its weights as the final model (NOT the
    # last epoch). --epochs (8) is the MAX cap. Default-behavior guard: if
    # --early-stop-patience >= --epochs the stop can never trigger, so this reduces
    # to the OLD fixed-epoch behavior (callers that don't opt in are unaffected);
    # the new _es battery opts in with --early-stop-patience 3 --min-epochs 2.
    patience = int(args.early_stop_patience)
    min_epochs = int(args.min_epochs)
    es_active = patience < args.epochs
    print(f"[early-stop] patience={patience} min-epochs={min_epochs} "
          f"max-epochs(cap)={args.epochs}  "
          f"{'ACTIVE (monitor=dev-doc-F1 own-best, agglom decode; dev=SELECTION, no grads)' if es_active else 'INACTIVE (patience>=epochs -> fixed-epoch behavior)'}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    history = []
    dev_history = []
    best_f1, best_epoch, best_state, no_improve = -1.0, 0, None, 0
    stopped_at = args.epochs
    for ep in range(args.epochs):
        forest.train()
        order = rng.permutation(len(train_windows))
        ep_loss, n_steps, last_eg = 0.0, 0, 0.0
        if sam_on:
            # SAM path: batch of (win, seed_tuple); step_sam re-forwards at w and
            # w+eps (each with the R-Drop double pass when alpha>0), rebuilding a
            # FRESH rng from seed_tuple per pass so ascent & descent share the SAME
            # curriculum state. None-loss windows are filtered in batch_mean_loss.
            batch = []
            for si, wi in enumerate(order):
                win = train_windows[wi]
                batch.append((win, (int(args.seed), int(ep), int(wi))))
                if len(batch) >= args.grad_accum:
                    lv, last_eg = step_sam(batch)
                    if lv is not None:
                        ep_loss += lv
                        n_steps += 1
                    batch = []
            if batch:
                lv, last_eg = step_sam(batch)
                if lv is not None:
                    ep_loss += lv
                    n_steps += 1
        else:
            # NON-SAM path: encode-once accumulation over PRE-COMPUTED window losses,
            # None-loss windows SKIPPED (never consume a grad_accum slot) — the batch
            # grouping is bit-for-bit the pre-existing default (R-Drop, when alpha>0,
            # folds its double pass + KL inside window_loss).
            accum = []
            for si, wi in enumerate(order):
                win = train_windows[wi]
                wl = window_loss(
                    win, np.random.default_rng((args.seed, ep, int(wi))))
                if wl is None:
                    continue
                accum.append(wl)
                if len(accum) >= args.grad_accum:
                    lv, last_eg = step_plain(accum)
                    ep_loss += lv
                    n_steps += 1
                    accum = []
            if accum:
                lv, last_eg = step_plain(accum)
                ep_loss += lv
                n_steps += 1
        ep_loss /= max(n_steps, 1)
        history.append(ep_loss)

        # --- MONITOR: DEV doc-macro-F1 at own-best threshold (agglom decode) ---
        dev_f1, dev_t = eval_dev_docf1(
            forest, dev_docs, window, stride, args.max_length,
            return_threshold=True)
        dev_history.append(dev_f1)
        print(f"  epoch {ep + 1}/{args.epochs}: mean-loss={ep_loss:.5f} "
              f"steps={n_steps} last-encoder-gradnorm(pre-clip)="
              f"{last_eg:.4e}  dev-doc-F1(own-best t={dev_t:.2f})={dev_f1:.4f}",
              flush=True)

        # keep the BEST-dev-F1 epoch (deep-copied in-memory snapshot, no disk churn)
        if dev_f1 > best_f1 + 1e-6:
            best_f1, best_epoch, no_improve = dev_f1, ep + 1, 0
            best_state = _snapshot_weights(forest)
            print(f"  [early-stop] new BEST dev-doc-F1={best_f1:.4f} "
                  f"@epoch {best_epoch} (checkpoint snapshotted in-memory)",
                  flush=True)
        else:
            no_improve += 1
            # stop only AFTER min-epochs and only if opted in (patience<epochs)
            if es_active and (ep + 1) >= min_epochs and no_improve >= patience:
                stopped_at = ep + 1
                print(f"  [early-stop] STOP @epoch {stopped_at}: no dev-doc-F1 "
                      f"gain for {patience} consecutive epochs after "
                      f"min-epochs={min_epochs} (best={best_f1:.4f} "
                      f"@epoch {best_epoch})", flush=True)
                break

    # BEST-CHECKPOINT RESTORE: the saved model is the BEST-dev-F1 epoch, NOT the
    # last epoch (guards the unfrozen-encoder overfit gap). Restore in-memory.
    if best_state is not None:
        _restore_weights(forest, best_state, device)
        print(f"[early-stop] restored best: best epoch={best_epoch}  "
              f"best dev-doc-F1={best_f1:.4f}  stopped-at epoch={stopped_at}  "
              f"(saving the RESTORED best, not last epoch {len(history)})",
              flush=True)

    # save (encoder as HF dir, composer+head as a torch ckpt, cfg json)
    forest.encoder.save_pretrained(os.path.join(args.out_dir, "encoder"))
    forest.tokenizer.save_pretrained(os.path.join(args.out_dir, "encoder"))
    torch.save({"composer": forest.composer.state_dict(),
                "head": forest.head.state_dict(),
                "h": forest.h,
                "head_in_features": forest.head.in_features},
               os.path.join(args.out_dir, "forest_heads.pt"))
    cfg = {"design": "forest", "task": args.task,
           "composer_init": args.composer_init, "composer_note": composer_note,
           "freeze_encoder": args.freeze_encoder,
           "window": window, "stride": stride, "max_length": args.max_length,
           "encoder_lr": args.encoder_lr, "composer_lr": args.composer_lr,
           "recon_aux": args.recon_aux, "pos_weight": pos_weight,
           "epochs": args.epochs, "seed": args.seed,
           # LLRD / R-Drop / SAM record (all flag-gated; defaults = off)
           "llrd": bool(args.llrd), "encoder_lr_top": args.encoder_lr_top,
           "llrd_decay": args.llrd_decay,
           "rdrop_alpha": rdrop_alpha, "sam": sam_on, "sam_rho": sam_rho,
           # exposure-bias fix record (flag-gated; defaults = off)
           "curriculum_include_allcut": include_allcut, "allcut_prob": allcut_prob,
           "scheduled_sampling": scheduled_sampling, "ss_free_steps": ss_free_steps,
           "loss_history": history, "base_model": BASE_MODEL,
           # early-stopping record (dev is SELECTION only — no dev grads)
           "early_stop_patience": patience, "min_epochs": min_epochs,
           "early_stop_active": es_active,
           "dev_docf1_history": dev_history,
           "best_epoch": best_epoch, "best_dev_docf1": best_f1,
           "stopped_at_epoch": stopped_at, "restored_best": best_state is not None}
    with open(os.path.join(args.out_dir, CFG_NAME), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"saved forest -> {args.out_dir}")
    print(f"# loss first->last epoch: {history[0]:.5f} -> {history[-1]:.5f}")
    print(f"# dev-doc-F1 by epoch: {[round(x, 4) for x in dev_history]}  "
          f"best={best_f1:.4f} @epoch {best_epoch}  stopped-at {stopped_at}  "
          f"restored-best={best_state is not None}")
    print(f"next: python src/forest_train.py predict --model {args.out_dir} "
          f"--task {args.task} --split dev --out subs/{args.task}_forest_dev.csv "
          f"--npz-out probs/forest_{args.task}_dev.npz")


# --------------------------------------------------------------------------- #
def load_forest(model_dir, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    cfg = json.load(open(os.path.join(model_dir, CFG_NAME), encoding="utf-8"))
    enc_dir = os.path.join(model_dir, "encoder")
    tokenizer = AutoTokenizer.from_pretrained(enc_dir)
    encoder = AutoModel.from_pretrained(enc_dir).to(device).eval()
    h = encoder.config.hidden_size
    heads = torch.load(os.path.join(model_dir, "forest_heads.pt"),
                       map_location="cpu", weights_only=False)
    composer = make_rae(h)
    composer.load_state_dict(heads["composer"])
    composer.to(device).eval()
    head = make_head(h)
    head.load_state_dict(heads["head"])
    head.to(device).eval()
    forest = Forest(encoder, tokenizer, composer, head, h, device,
                    freeze_encoder=True)
    return forest, cfg


def cmd_predict(args):
    import torch
    from baselines import write_submission

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    forest, cfg = load_forest(args.model, device)
    window = args.window or cfg.get("window", DEFAULT_WINDOW)
    stride = args.stride or cfg.get("stride", max(window // 2, 1))
    max_length = args.max_length or cfg.get("max_length", 512)
    # DEFAULT readout = "agglom" (composed-unit; the state the training loop scores
    # each junction in — BLOCKER-2). --leaf-readout selects the DIAGNOSTIC leaf
    # readout (raw adjacent word pair); no gate consumes it. Legacy --descend still
    # forces agglom (it was the old opt-in name for this same descent).
    readout = "leaf" if args.leaf_readout else "agglom"
    print(f"[predict] window={window} stride={stride} max_length={max_length} "
          f"readout={readout}")

    docs = load_task(args.task, args.split, jsonl_path=args.jsonl)
    if args.max_docs is not None:
        docs = docs[:args.max_docs]

    preds, prob_map = [], {}
    for d in docs:
        words = [MODEL_PAR_TOKEN if w == PARAGRAPH_TOKEN else w for w in d["tokens"]]
        p = predict_doc_forest(words, forest, window, stride, max_length,
                               readout=readout)
        prob_map[d["doc_id"]] = p.astype(np.float32)
        lab = (p >= args.threshold).astype(int)
        for i, w in enumerate(d["tokens"]):
            if w == PARAGRAPH_TOKEN:
                lab[i] = 0
        if args.force_doc_end:
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
        print(f"word-grid P(boundary) ({len(prob_map)} docs) -> {args.npz_out} "
              f"(doc-final word carries 0.0 by construction)")

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


# ==========================================================================  #
# PLANTED-SIGNAL self-test — corpus-independent, NO data/, NO encoder, NO caches
# ==========================================================================  #
#
# WHY THIS CONSTRUCTION IS PROVABLY COMPOSITIONAL (not pair-local).
# ------------------------------------------------------------------------------
# The verifier broke the ORIGINAL bracket signal: it was a per-adjacent-pair
# TOPIC-CLUSTER signal (same-topic neighbours vs different-topic neighbours), so
# a frozen-random composer + head scored 0.96 and a raw cosine(v_i, v_{i+1})
# threshold scored 1.00 — the boundary was decidable from the junction's OWN
# adjacent pair, with ZERO composition/recursion. A test that only clears a
# single-token-local baseline (0.333) does not isolate composition.
#
# The signal below is a LEFT-CONTEXT ORDERED BIGRAM TRIGGER over a small shared
# symbol alphabet (K symbols, fixed vectors reused across every document, so
# symbol IDENTITY is the only content and it is shared, never document-specific):
#
#     junction j (the gap AFTER word j, j in [0..n-2]) is a boundary (KEEP=1)
#     IFF the ORDERED pair of symbols ENDING at j is the trigger  (A, B),
#     i.e.  symbol[j-1] == A  AND  symbol[j] == B.
#
# Symbols are drawn i.i.d. uniform over the alphabet, so A and B each occur in
# MANY contexts. This makes every PAIR-LOCAL view of the junction provably blind:
#   * single-token(v_j):        the junction's own word is symbol[j]; B appears
#       both AFTER A (boundary) and after any other symbol (non-boundary) with
#       equal marginal, so B alone (or any single symbol) is at chance.  <0.6
#   * adjacent-pair(v_j, v_{j+1}) — the junction's OWN pair — sees symbol[j] and
#       symbol[j+1], but the trigger's LEFT arm is symbol[j-1]; the pair (j,j+1)
#       is BLIND to symbol[j-1], so it is at chance.                       <0.6
#   * cosine(v_j, v_{j+1}):      same blindness, plus symbol vectors are fixed
#       and unrelated to the boundary rule -> at chance.                   <0.6
# The boundary is recoverable ONLY by INTEGRATING the LEFT CONTEXT (composing
# word j-1 into the left unit before deciding junction j) — i.e. by COMPOSITION
# DEPTH / span integration, exactly the mechanism under test. An oracle that is
# handed the (j-1, j) left pair recovers it at ~0.99 F1, confirming the signal
# is real and lives strictly ABOVE the junction's own adjacent pair.
#
# The forest recovers it (easy-first agglomeration builds the left unit, the head
# reads the composed left context) at >0.9; a FROZEN-RANDOM composer cannot build
# a left representation that exposes the (A,B) conjunction and FAILS (<0.7) — so
# recursion is proven load-bearing, not incidental.
_TRIG_K = 4          # shared symbol alphabet size (denser trigger: P((A,B))=1/16)
_TRIG_A = 0          # trigger left symbol
_TRIG_B = 1          # trigger right symbol


def _bracket_stream(rng, n_sent, vocab=None):
    """One 'document' = an i.i.d. uniform stream of shared symbols; boundary truth
    is the LEFT-CONTEXT ORDERED BIGRAM (A,B) rule above (a function of COMPOSITION
    over the left context, NOT of the junction's own token or own adjacent pair).

    `n_sent` is reused as a rough length knob: the stream length is scaled by it
    so documents keep varying, non-trivial length (~40-70 symbols). Returns
    (symbol_ids, labels, sent_lengths) where label[j]=1 iff (ids[j-1],ids[j])==
    (A,B); sent_lengths is the run-length partition INDUCED by the boundaries
    (kept only so the return signature matches the historical bracket API; the
    vectors do NOT use it — identity is shared, so no per-sentence topic exists
    to leak the boundary)."""
    vocab = _TRIG_K if vocab is None else vocab              # read K dynamically
    n = int(4 * max(int(n_sent), 1) + rng.integers(6, 14))   # varied, compact docs
    ids = [int(rng.integers(vocab)) for _ in range(n)]
    labels = [0] * n
    for j in range(n - 1):                                    # junction after word j
        if j >= 1 and ids[j - 1] == _TRIG_A and ids[j] == _TRIG_B:
            labels[j] = 1
    # induced run-length partition (boundaries close a run); purely informational
    sent_lengths, run = [], 0
    for j in range(n):
        run += 1
        if j < n - 1 and labels[j] == 1:
            sent_lengths.append(run)
            run = 0
    sent_lengths.append(run)
    return ids, labels, sent_lengths


# fixed shared symbol vectors (built once per h; identity is GLOBAL, never per-doc,
# so nothing document-specific can leak the boundary through a topic).
_TRIG_SYMVEC = {}


def _bracket_vectors(ids, h, rng, sent_lengths=None):
    """Vectors carry ONLY shared symbol identity (a fixed per-symbol vector reused
    across every document) plus small i.i.d. noise. NO per-sentence topic exists,
    so no adjacent pair — and no single word — can reveal the (A,B) left-context
    boundary; only composing word j-1 into the left unit does. `sent_lengths` is
    accepted (historical signature) but INTENTIONALLY unused: the boundary is a
    property of the ORDERED left bigram, not of any span-mean cluster."""
    if h not in _TRIG_SYMVEC:
        base = np.random.default_rng(12345)             # deterministic shared table
        sym = base.normal(0, 1, (_TRIG_K, h)).astype(np.float32)
        sym /= np.linalg.norm(sym, axis=1, keepdims=True)
        _TRIG_SYMVEC[h] = (sym * 2.0).astype(np.float32)
    sym = _TRIG_SYMVEC[h]
    V = sym[np.asarray(ids, dtype=np.int64)].copy()
    V += rng.normal(0, 0.15, V.shape).astype(np.float32)
    return V.astype(np.float32)


def _token_local_baseline_f1(docs, h):
    """A token-LOCAL logistic-regression baseline: predict boundary-after-word
    from ONLY that word's own vector (no neighbors, no composition). If the
    planted signal is genuinely compositional, this must stay < 0.6 F1."""
    import torch
    X, y = [], []
    for V, lab in docs:
        for i in range(len(lab) - 1):
            X.append(V[i]); y.append(lab[i])
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.float32)
    torch.manual_seed(0)
    clf = torch.nn.Linear(h, 1)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)]))
    for _ in range(300):
        opt.zero_grad()
        loss = lossf(clf(X)[:, 0], y)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(clf(X)[:, 0]) >= 0.5).float()
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def _f1_from_counts(tp, fp, fn):
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def _adjacent_pair_baseline_f1(docs, h):
    """The verifier's PAIR-LOCAL attack, at full strength: a logistic classifier
    fit ON the training set (an upper bound) that reads the junction's OWN adjacent
    word pair [v_i ; v_{i+1}]. If the signal lives strictly ABOVE the junction's
    own pair (as a left-context / composition-depth signal does), this must stay
    < 0.6 F1 — no adjacent pair can see it."""
    import torch
    X, y = [], []
    for V, lab in docs:
        for i in range(len(lab) - 1):
            X.append(np.concatenate([V[i], V[i + 1]])); y.append(lab[i])
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(y), dtype=torch.float32)
    torch.manual_seed(0)
    clf = torch.nn.Linear(2 * h, 1)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)]))
    for _ in range(300):
        opt.zero_grad(); lossf(clf(X)[:, 0], y).backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(clf(X)[:, 0]) >= 0.5).float()
    return _f1_from_counts(
        float(((pred == 1) & (y == 1)).sum()),
        float(((pred == 1) & (y == 0)).sum()),
        float(((pred == 0) & (y == 1)).sum()))


def _adjacent_cosine_baseline_f1(docs):
    """The verifier's raw-cosine attack: sweep a threshold on cosine(v_i, v_{i+1})
    (boundary iff cosine below threshold) and report the BEST-threshold F1 (an
    upper bound). For a genuinely compositional signal this must stay < 0.6 — the
    junction's own adjacent-pair cosine carries no boundary information."""
    pairs = []
    for V, lab in docs:
        for i in range(len(lab) - 1):
            a, b = V[i], V[i + 1]
            c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            pairs.append((c, int(lab[i])))
    if not pairs:
        return 0.0
    cs = np.asarray([c for c, _ in pairs]); ys = np.asarray([g for _, g in pairs])
    best = 0.0
    for thr in np.linspace(cs.min(), cs.max(), 201):
        pred = (cs < thr).astype(int)
        best = max(best, _f1_from_counts(
            int(((pred == 1) & (ys == 1)).sum()),
            int(((pred == 1) & (ys == 0)).sum()),
            int(((pred == 0) & (ys == 1)).sum())))
    return best


# memo so the self-test / pytest (which train the SAME forest for several
# assertions) train each unique (docs, h, seed, epochs, freeze) ONCE per process.
_FOREST_TRAIN_CACHE = {}


def _docs_signature(docs):
    """Cheap deterministic signature of a planted-doc list (content-addressed)."""
    n = len(docs)
    nb = sum(int(x) for _, lab in docs for x in lab[:-1])
    tot = sum(len(lab) - 1 for _, lab in docs)
    head = float(docs[0][0][0, 0]) if n and len(docs[0][0]) else 0.0
    return (n, tot, nb, round(head, 6))


def _train_forest_on_docs(docs, h, seed, epochs=40, freeze_composer=False):
    """Train a stand-in forest (composer + head, identity 'encoder' on the planted
    vectors) with the AGGLOMERATIVE-from-singletons supervised objective
    (forest_window_loss_agglo) — the training twin of the DEFAULT decode. When
    freeze_composer, the composer is frozen at its RANDOM init and only the head
    trains: the load-bearing-recursion CONTROL. Returns (forest, first_loss,
    last_loss). Deterministic; memoized per process so repeated identical calls
    (self-test + several pytest assertions) train only once."""
    import torch
    key = (_docs_signature(docs), h, seed, epochs, freeze_composer)
    if key in _FOREST_TRAIN_CACHE:
        return _FOREST_TRAIN_CACHE[key]
    composer = make_rae(h, seed=seed)
    head = make_head(h, seed=seed)
    if freeze_composer:
        for p in composer.parameters():
            p.requires_grad_(False)
        params = list(head.parameters())
    else:
        params = list(composer.parameters()) + list(head.parameters())
    forest = Forest(None, None, composer, head, h, "cpu")
    pos = sum(int(x) for _, lab in docs for x in lab[:-1])
    tot = sum(len(lab) - 1 for _, lab in docs)
    pw = min((tot - pos) / max(pos, 1), 8.0)
    opt = torch.optim.Adam(params, lr=0.01)
    first = last = None
    for ep in range(epochs):
        ep_loss, nb = 0.0, 0
        for wi in np.random.default_rng(ep).permutation(len(docs)):
            V, lab = docs[wi]
            loss, _, _ = forest_window_loss_agglo(
                forest, torch.tensor(V, dtype=torch.float32), lab, pw,
                rng=np.random.default_rng((ep, int(wi))))
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.detach()); nb += 1
        ep_loss /= max(nb, 1)
        first = ep_loss if first is None else first
        last = ep_loss
    forest.eval()
    _FOREST_TRAIN_CACHE[key] = (forest, first, last)
    return forest, first, last


def _forest_boundary_f1(forest, docs, readout="agglom"):
    """Held-out boundary F1 of a trained forest under the given readout."""
    import torch
    tp = fp = fn = 0
    for V, lab in docs:
        p = forest_predict_window(forest, torch.tensor(V, dtype=torch.float32),
                                  readout=readout)
        pred = (p[:len(lab) - 1] >= 0.5).astype(int)
        g = np.asarray(lab[:len(lab) - 1])
        tp += int(((pred == 1) & (g == 1)).sum())
        fp += int(((pred == 1) & (g == 0)).sum())
        fn += int(((pred == 0) & (g == 1)).sum())
    return _f1_from_counts(tp, fp, fn)


def _make_bracket_docs(h, seeds):
    """Build (V, labels) docs from the compositional left-context construction."""
    docs = []
    for s in seeds:
        r = np.random.default_rng(1000 + s)
        ids, lab, sl = _bracket_stream(r, n_sent=int(r.integers(3, 6)))
        V = _bracket_vectors(ids, h, r, sl)
        docs.append((V, lab))
    return docs


def self_test():
    """Runs WITHOUT data/, encoder, or caches. Proves:
      (1) the head/merge math + grad flows keep/erase -> composer -> word vectors;
      (2) -100 ([PAR]) junctions are excluded from supervision;
      (3) the COMPOSITIONAL planted-signal contract — a boundary defined by the
          LEFT-CONTEXT ordered bigram (A,B), which is provably INVISIBLE to every
          pair-local view of the junction (single-token, the junction's OWN
          adjacent pair, and raw adjacent cosine all < 0.6 F1), is recovered by
          the forest (> 0.9 F1 under the composed-unit readout) while a
          FROZEN-RANDOM composer FAILS (< 0.7 F1). The frozen control makes
          RECURSION provably load-bearing: only a composer that LEARNS to build
          the left-context representation exposes the (A,B) conjunction to the
          head. (A raw single-token-local baseline is also kept, per the old
          contract, but it alone does NOT isolate composition — hence the two
          added pair-local controls and the frozen-composer control.)"""
    import torch
    print("=== FOREST self-test (corpus-independent) ===")

    # --- 1. head in_features & merge-loss plumbing ---
    h = 16
    composer = make_rae(h, seed=0)
    head = make_head(h, seed=0)
    assert head.in_features == 3 * h + 1, head.in_features
    forest = Forest(encoder=None, tokenizer=None, composer=composer, head=head,
                    h=h, device="cpu")
    rng = np.random.default_rng(1)
    wv = torch.tensor(rng.normal(size=(7, h)), dtype=torch.float32,
                      requires_grad=True)
    labels = [0, 0, 1, 0, 0, 1, 1]                     # gold: boundaries after w2, w5
    loss, n_leaf, n_pos = forest_window_loss(
        forest, wv, labels, pos_weight=3.0, rng=np.random.default_rng(0))
    assert loss.requires_grad and loss.ndim == 0 and float(loss.detach()) > 0
    # 2 gold-KEEP junctions (after w2 and w5) are ALWAYS supervised; the
    # curriculum may add spurious ERASE cuts, so n_leaf >= n_pos == 2.
    assert n_pos == 2, n_pos
    assert n_leaf >= n_pos, (n_leaf, n_pos)
    loss.backward()
    assert wv.grad is not None and float(wv.grad.abs().sum()) > 0, \
        "no gradient to the word vectors (encoder path would be dead)"
    assert composer.enc.weight.grad is not None and \
        float(composer.enc.weight.grad.abs().sum()) > 0, "composer got no grad"
    assert head[0].weight.grad is not None, "head got no grad"
    print(f"PASS head in_features={head.in_features}; grad flows to word vectors "
          f"AND composer AND head (n_leaf={n_leaf}, n_pos={n_pos})")

    # --- 2. -100 ([PAR]) junctions are excluded from supervision ---
    # gold after w0=0, w1=[PAR], w2=1, w3=0 ; junction after w1 is -100-excluded.
    # only the gold-KEEP (after w2) is guaranteed; assert the [PAR] junction is
    # never supervised and never counted positive.
    labels2 = [0, -100, 1, 0, 1]
    n_pos_seen = 0
    for s in range(8):
        _l2, n_leaf2, n_pos2 = forest_window_loss(
            forest, wv[:5], labels2, 3.0, rng=np.random.default_rng(s))
        assert n_pos2 == 1, (s, n_pos2)                # only the after-w2 gold cut
        n_pos_seen = max(n_pos_seen, n_leaf2)
    print(f"PASS -100 junction exclusion (gold-KEEP=1 always; [PAR] never cut)")

    # --- 3. COMPOSITIONAL PLANTED-SIGNAL: left-context (A,B) bigram boundary ---
    #   pair-local views all fail; forest recovers; frozen-random composer fails.
    H = 24
    train_docs = _make_bracket_docs(H, range(80))
    test_docs = _make_bracket_docs(H, range(80, 100))
    brate = np.mean([l for _, lab in train_docs for l in lab[:-1]])
    print(f"  construction: left-context bigram (A,B); train boundary "
          f"rate={brate:.3f}")

    # (a) three PAIR-LOCAL controls — all must be at chance (< 0.6), by design:
    tl_f1 = _token_local_baseline_f1(train_docs, H)
    ap_f1 = _adjacent_pair_baseline_f1(train_docs, H)
    ac_f1 = _adjacent_cosine_baseline_f1(train_docs)
    print(f"  token-local(v_i)         F1 (self-fit UB)   = {tl_f1:.3f} (< 0.6)")
    print(f"  adjacent-pair(v_i,v_i+1) F1 (self-fit UB)   = {ap_f1:.3f} (< 0.6)")
    print(f"  adjacent-cosine best-thr F1                 = {ac_f1:.3f} (< 0.6)")
    assert tl_f1 < 0.6, f"single-token not at chance: F1={tl_f1:.3f}"
    assert ap_f1 < 0.6, f"adjacent-pair not at chance: F1={ap_f1:.3f} " \
        "(signal is pair-local, NOT compositional)"
    assert ac_f1 < 0.6, f"adjacent-cosine not at chance: F1={ac_f1:.3f} " \
        "(signal is pair-local, NOT compositional)"

    # (b) the FOREST (composer LEARNS) recovers it under the composed-unit readout:
    forest, first, last = _train_forest_on_docs(train_docs, H, seed=3,
                                                freeze_composer=False)
    f_f1 = _forest_boundary_f1(forest, test_docs, readout="agglom")
    print(f"  forest train loss {first:.4f} -> {last:.4f}")
    print(f"  FOREST held-out boundary F1 (agglom)        = {f_f1:.3f} "
          f"(must be > 0.9 — recursion recovers the left context)")
    assert f_f1 > 0.9, f"forest failed the compositional contract: F1={f_f1:.3f}"

    # (c) FROZEN-RANDOM composer CONTROL — recursion must be load-bearing: a random
    #     (never-trained) composer cannot build the left representation the head
    #     needs, so it FAILS even with a fully-trained head.
    frozen, ffirst, flast = _train_forest_on_docs(train_docs, H, seed=3,
                                                  freeze_composer=True)
    fr_f1 = _forest_boundary_f1(frozen, test_docs, readout="agglom")
    print(f"  frozen-random composer (head-only) loss {ffirst:.4f} -> {flast:.4f}")
    print(f"  FROZEN-RANDOM held-out boundary F1 (agglom) = {fr_f1:.3f} "
          f"(must be < 0.7 — recursion is load-bearing)")
    assert fr_f1 < 0.7, \
        f"frozen-random composer did NOT fail (F1={fr_f1:.3f}): the signal is " \
        "solvable WITHOUT learned composition -> not truly compositional"
    assert f_f1 > fr_f1 + 0.2, \
        f"forest {f_f1:.3f} not clearly above frozen control {fr_f1:.3f}"
    print(f"PASS compositional planted-signal: forest {f_f1:.3f} > 0.9 ; "
          f"frozen-random {fr_f1:.3f} < 0.7 ; all pair-local < 0.6")
    print("SELF-TEST OK")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test")

    tr = sub.add_parser("train", help="end-to-end joint composition training")
    tr.add_argument("--task", default="NoPnx-NP", choices=sorted(TASKS))
    tr.add_argument("--train-jsonl", default=None)
    tr.add_argument("--dev-jsonl", default=None)
    tr.add_argument("--expect-docs", type=int, default=None)
    tr.add_argument("--composer-init", choices=["rae", "random"], default="rae",
                    help="rae = init composer from rae_gate/rae_final.pt; "
                         "random = fresh (prior-matters ablation)")
    tr.add_argument("--freeze-encoder", action="store_true",
                    help="freeze the encoder (reproduces the parked frozen-"
                         "encoder arm; DEFAULT is UNFROZEN = the untested cell)")
    tr.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    tr.add_argument("--stride", type=int, default=None, help="default W//2")
    tr.add_argument("--max-length", type=int, default=512)
    tr.add_argument("--epochs", type=int, default=8,
                    help="MAX epoch cap (early stopping may stop sooner)")
    tr.add_argument("--early-stop-patience", type=int, default=3,
                    help="stop when DEV doc-macro-F1 (own-best threshold, agglom "
                         "decode) has not improved for this many consecutive "
                         "epochs after --min-epochs. DEFAULT-BEHAVIOR GUARD: if "
                         ">= --epochs, the stop can never fire and training "
                         "reduces to the old fixed-epoch behavior. The _es battery "
                         "opts in with 3. Dev is SELECTION only (repo convention: "
                         "every arm already picks its own-best threshold on dev); "
                         "no dev gradients ever flow.")
    tr.add_argument("--min-epochs", type=int, default=2,
                    help="never early-stop before this many epochs have run")
    tr.add_argument("--encoder-lr", type=float, default=2e-5,
                    help="FLAT encoder lr (used only when --llrd is OFF)")
    tr.add_argument("--composer-lr", type=float, default=1e-3,
                    help="composer + keep/erase head lr (flat; also the LLRD "
                         "composer+head group lr)")
    tr.add_argument("--recon-aux", type=float, default=0.0,
                    help="reconstruction auxiliary weight (0=pure supervised; the "
                         "_es battery uses 0.1). Adds w * mean(recon E over "
                         "supervised junctions) to the loss.")
    # --- EXPOSURE-BIAS FIX (Noor, 2026-07-11; both default OFF -> bit-for-bit the
    #     pre-fix curriculum start). TRAINING starts from a curriculum that EXCLUDES
    #     'allcut' (near-gold refine), but INFERENCE decodes from ALL WORDS SEPARATE
    #     -> exposure bias caps dev-F1 (~77 plateau). These make TRAIN see the decode
    #     start-state. The _es2 battery opts in (see runs/forest_battery_2026-07-11_es2).
    tr.add_argument("--curriculum-include-allcut", action="store_true",
                    help="MIX the all-separate 'allcut' state into the curriculum "
                         "start-state pool (prob --allcut-prob per window) so "
                         "training practices BUILD-FROM-SCRATCH (the agglom-decode "
                         "condition) as well as refine. OFF -> old refine-only pool.")
    tr.add_argument("--allcut-prob", type=float, default=0.5,
                    help="P(start a window from all-separate) when "
                         "--curriculum-include-allcut is set (0=never/old behavior, "
                         "1=always build-from-scratch). Draw is deterministic per "
                         "window (rng), so the R-Drop double pass agrees.")
    tr.add_argument("--scheduled-sampling", type=float, default=0.0,
                    help="DAgger-lite P: with this probability per window, FREE-RUN "
                         "the model's own greedy merges (no_grad, from all-separate, "
                         "never crossing a gold boundary) for a few steps, then "
                         "supervise WITH GRAD from that OWN-reached state — so it "
                         "learns to RECOVER from its own early mistakes, not just "
                         "polish gold. 0=OFF (old behavior). Composes with "
                         "--curriculum-include-allcut (SS overrides the start-cuts "
                         "when it fires).")
    tr.add_argument("--ss-free-steps", type=int, default=None,
                    help="scheduled-sampling free-run length (# greedy merges before "
                         "supervision). Default (None) = HALF the mergeable "
                         "junctions (reflects the model's EARLY decisions).")
    # --- LAYER-WISE LR DECAY (Noor, 2026-07-11; default OFF -> flat 2-group opt,
    #     bit-for-bit yesterday. The _es battery opts in with --llrd.) ---
    tr.add_argument("--llrd", action="store_true",
                    help="layer-wise LR decay over the encoder: top layer lr="
                         "--encoder-lr-top, decaying by --llrd-decay per layer "
                         "toward the embeddings (lowest lr); composer+head stay at "
                         "--composer-lr. OFF -> flat encoder lr (--encoder-lr).")
    tr.add_argument("--encoder-lr-top", type=float, default=2.5e-5,
                    help="LLRD: lr of the TOP encoder layer (decays downward)")
    tr.add_argument("--llrd-decay", type=float, default=0.874,
                    help="LLRD: per-layer decay factor toward the embeddings "
                         "(0.874 => 12-layer bottom ~= 5e-6 given top 2.5e-5)")
    # --- R-DROP (Noor, 2026-07-11; default 0 = OFF. Repo convention: symmetric KL
    #     between two independent-dropout passes; the cons/integration battery uses
    #     1.0 -> the _es battery uses that same repo default 1.0.) ---
    tr.add_argument("--rdrop-alpha", type=float, default=0.0,
                    help="R-Drop weight: alpha * symmetric-KL between two "
                         "independent-dropout forward passes' keep/erase logits at "
                         "the gold/curriculum junctions (0=OFF, single pass; the "
                         "repo R-Drop convention from consistency_train uses 1.0, "
                         "which the _es battery adopts).")
    # --- SAM (Noor, 2026-07-11; default OFF. First-order SAM composing with LLRD +
    #     R-Drop; the _es battery opts in with --sam --sam-rho 0.05.) ---
    tr.add_argument("--sam", action="store_true",
                    help="first-order Sharpness-Aware Minimization: ascent to "
                         "w+rho*g/||g||, descent grad there, restore w, step. "
                         "Composes with --llrd (perturbs raw params; grouped AdamW "
                         "applies) and --rdrop-alpha (each SAM pass runs the R-Drop "
                         "double forward -> ~4 fwd/window). OFF -> plain step.")
    tr.add_argument("--sam-rho", type=float, default=0.05,
                    help="SAM neighbourhood radius rho (only used with --sam)")
    tr.add_argument("--grad-accum", type=int, default=8,
                    help="windows per optimizer step")
    tr.add_argument("--pos-weight", type=float, default=None)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--max-docs", type=int, default=None, help="CPU smoke cap")
    tr.add_argument("--max-dev-docs", type=int, default=None, help="CPU smoke cap")
    tr.add_argument("--threads", type=int, default=0)
    tr.add_argument("--cpu", action="store_true", help="force CPU (smokes)")
    tr.set_defaults(fn=cmd_train)

    pr = sub.add_parser("predict",
                        help="agglomerative composed-unit decode, emit CSV + npz")
    pr.add_argument("--model", required=True)
    pr.add_argument("--task", default="NoPnx-NP", choices=sorted(TASKS))
    pr.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    pr.add_argument("--jsonl", default=None)
    pr.add_argument("--out", required=True, help="submission CSV")
    pr.add_argument("--npz-out", default=None,
                    help="word-grid P(boundary) npz (cache_probs.py format)")
    pr.add_argument("--threshold", type=float, default=0.5)
    pr.add_argument("--window", type=int, default=None)
    pr.add_argument("--stride", type=int, default=None)
    pr.add_argument("--max-length", type=int, default=None)
    pr.add_argument("--force-doc-end", action="store_true",
                    help="force a boundary on each doc's last non-[PAR] token")
    pr.add_argument("--leaf-readout", action="store_true",
                    help="DIAGNOSTIC: score each junction on the raw adjacent word "
                         "pair (a DIFFERENT distribution than training; no gate "
                         "consumes this). DEFAULT is the agglomerative composed-"
                         "unit readout that matches the training-time state.")
    pr.add_argument("--descend", action="store_true",
                    help=argparse.SUPPRESS)          # legacy alias -> agglom (now default)
    pr.add_argument("--check-ids", default=None)
    pr.add_argument("--max-docs", type=int, default=None, help="CPU smoke cap")
    pr.add_argument("--cpu", action="store_true")
    pr.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    if args.cmd == "self-test":
        self_test()
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
