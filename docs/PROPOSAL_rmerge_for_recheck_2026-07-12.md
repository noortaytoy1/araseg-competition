# Recursive Beam-Merge Segmenter — proposal restated for a code recheck

For a fresh Claude: verify `src/rmerge.py` implements the proposal below. Source of
truth = `docs/METHODOLOGY_recursive_beam_merge_2026-07-11.md` (§ refs) and the
`rmerge.py` module docstring. Do NOT trust the previous session; check the code
against each numbered item and report every mismatch, most severe first.

## The proposal (what the code MUST implement)

1. **One idea (§1).** Segmentation = recursive sentence *construction*, not per-token
   labeling. A boundary is a junction where composition is not supported.

2. **Merge scorer — AraBERT IS the merger (§2.2).** For adjacent nodes A, B:
   `input = [CLS] tokens(A) [SEP] tokens(B) [SEP]`, `p = sigmoid(w·h_[CLS] + b)`.
   SAME AraBERT weights at every recursion level. No mean-pool, no stored constituent
   vector, no separate composer — it re-reads the real tokens every merge.
   → check `MergeScorer.logits_for_pairs`, `build_pair_batch`, `load_arabert`.

3. **Beam-3 search, TRAINING and inference (§2.3, §2.5).** Hypothesis = node list +
   score = product of the merge probs that built it. Each step: for every hypothesis
   in the beam, score all adjacent candidate merges in ONE batched pass; expand; keep
   top-3 by score. Stop a hypothesis when its best merge `< τ`.
   → check `beam_decode`; confirm width-3, batched scoring, top-3 retention.

4. **Loss — dense, gold-aligned, NO TEACHER FORCING (§2.4, FM7).** For EVERY candidate
   merge at junction k: `t=1 if gold_cut[k]=0` (within-sentence, should merge),
   `t=0 if gold_cut[k]=1` (boundary, should NOT merge). `L = Σ BCE(p_k, t_k)` over
   **all candidates, all hypotheses, all steps**. The beam picks merges by the model's
   OWN probabilities (it will wrongly merge across a boundary; the gold loss punishes
   exactly that). Class-weight the BCE (boundaries are the minority → guard FM2 collapse).
   → check `_state_loss`, `doc_train_loss_beam`, `train_pos_weight`.

5. **Training (§2.5).** Fine-tune AraBERT + scalar head end-to-end, ≤10 epochs. Beam-3
   runs DURING training. All candidate pairs in a beam step batched into ONE forward.
   Dense supervision at EVERY visited state (decouples learning from the search path).

6. **Inference (§2.3, §2.6).** Same beam decode. Score is LOG-SPACE LENGTH-NORMALIZED:
   `(1/|M|^α)·Σ log p_m`, α=1 → geometric mean (guards FM1 over-segmentation + underflow).
   Beam dedup on active-cut sets (FM6). Stop at low `τ`. Doc-final junction ALWAYS a
   boundary. Own-best `τ` on dev by grid. Surviving cuts → per-word 0/1 → standard scorer.
   → check `beam_decode` length-normalization + dedup; `sweep_tau`; `cmd_predict`.

7. **Acceptance / eval (§6).** dev doc-macro-F1 vs baseline **83.61** / gate **84.06**;
   predicted boundary rate ∈ **0.05–0.15** (NOT ≈0/1 → FM2); decorrelation **φ < 0.65**
   vs the AraBERT tagger cache. The model's closed-track value is as a decorrelated
   voter (FM8), not a solo leaderboard jump.

## Prime suspect the previous session flagged — CHECK THIS HARD

**Does `doc_train_loss_beam` (the `--roll-in laso` path) actually supervise ALL beam
hypotheses (item 4/§2.4), or only ONE path?** Reading it: it calls `beam_decode(...)`
once (no grad) to get the WINNING hypothesis, then rebuilds and supervises states along
the **winner's merge path only** — not all 3 beam hypotheses at every step. If so, that
is a **simplification of §2.4** ("all candidates, **all hypotheses**, all steps") and
should be reported as a spec/code divergence. Decide whether it's faithful enough or
must be fixed to score every hypothesis in the beam at every step.

## Also verify (previous session's changes)
- **Memory fix** `backward_each_state` in `doc_train_loss{,_sched,_beam}`: is per-state
  `.backward()` bit-equivalent to one backward on the summed loss? (should be; confirm
  no dropped/detached grad). `pytest tests/test_rmerge.py` must be green (30 passed).
- **Run config must be spec-exact:** `--roll-in laso` (NOT teacher/sched); decode via
  `predict` DEFAULTS (`--window 0` full-doc — NO windowing; `--tau` unset → sweep →
  own-best F1). Windowing at 128/32 exists only as a `predict` arg / training cost note
  (§2.5, FM9); the previous session wrongly windowed a decode and must not repeat it.
- **Ignore** `src/rmerge_voter_eval.py` — it re-decoded the model's output through
  Omar's DP decoder, which is NOT the spec's scorer. Score only via `rmerge.py predict`.
