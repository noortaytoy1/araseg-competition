# Audit prompt — verify the recursive beam-merge segmenter run against the spec

Copy everything below into a fresh Claude session (with this repo mounted) to
independently audit the previous session's work. The previous session was told
its judgment had drifted from the spec; your job is to catch any remaining slop.

---

You are auditing an implementation + run of a **recursive beam-merge sentence
segmenter** for AraSeg 2026 (closed track, NoPnx-NP). Trust nothing the previous
session claims; verify against the source of truth.

## Source of truth (the ONLY spec)
- `docs/METHODOLOGY_recursive_beam_merge_2026-07-11.md` (the algorithm, §1–6)
- `src/rmerge.py` module docstring (restates the locked algorithm)
- `CLAUDE.md` (standing rules), `DIRECTIVES.md`

## The spec's non-negotiables (from the methodology, quote and check each)
1. **Beam-3 search runs during TRAINING, not just inference** (§2.3, §2.5).
2. **NO TEACHER FORCING** — the beam picks merges by the model's own probabilities;
   dense gold-aligned BCE punishes wrong cross-boundary merges (§2.4). This is the
   whole "explore-your-own-mistakes" mechanism.
3. **Dense supervision:** at every visited state, score EVERY adjacent candidate in
   one batched AraBERT pass; class-weighted BCE, target 1 = within-sentence (should
   merge), 0 = boundary (§2.4, FM7).
4. **Inference:** beam-3, LOG-SPACE LENGTH-NORMALIZED score (geometric mean — guards
   FM1 over-segmentation), stop at low `τ`, own-best `τ` on dev grid, doc-final
   always a boundary (§2.3, §2.6).
5. **Sanity gates (§6):** predicted boundary rate ∈ **0.05–0.15** (NOT 0/1 → FM2);
   dev doc-macro-F1 vs baseline 83.61 / gate 84.06; decorrelation **φ < 0.65** vs the
   tagger cache (the model's actual closed-track value is as a decorrelated voter, FM8).

## What the previous session did (verify each claim; assume nothing)
- Found and fixed a real OOM/hang: the three per-doc loss functions
  (`doc_train_loss`, `doc_train_loss_sched`, `doc_train_loss_beam`) accumulated the
  whole document's autograd graph → OOM on the 15k-word train doc / 18k-word dev doc.
  Fix = a `backward_each_state` flag doing per-state `.backward()`. **VERIFY this is
  result-preserving** (per-state backprop sum == backprop of the summed loss; check
  it does not double-count, drop, or detach a needed grad). Run `pytest tests/test_rmerge.py`.
- **Ran `--roll-in teacher` first — a spec VIOLATION (spec = no teacher forcing).**
  Result: dev F1 ≈ 34, predicted boundary rate ≈ **0.32** (fails the 0.05–0.15 gate;
  hit FM1+FM2). Confirm this is the expected consequence of teacher-only training.
- Built `src/rmerge_voter_eval.py` that scored the model by **re-decoding its output
  through Omar's DP decoder** (`src/dp_decode.py`) — this is NOT the spec's scorer and
  should be **disregarded**. The spec scorer is `rmerge.py predict` (beam decode +
  own-best `τ` grid). Confirm the voter-eval is irrelevant and was discarded.
- Now running/queued **`--roll-in laso`** (beam roll-in, the spec's no-teacher-forcing
  training) with the memory fix. Decode = `rmerge.py predict` **spec defaults**
  (`--window 0` full-doc unless the doc needs the §2.5 sliding window, `--tau` unset →
  sweep → own-best F1). Windowing at 128/32 IS in the spec (§2.5, FM9) — not an add-on.

## Your audit tasks
1. Read the methodology + `rmerge.py`. Does `doc_train_loss_beam` (the laso path)
   actually implement §2.3–2.5 (beam-3, no teacher forcing, dense supervision, length
   normalization at DECODE)? Flag any place it diverges (e.g., it supervises only the
   winner's path, not all beam hypotheses — is that faithful to §2.4/FM7?).
2. Confirm the memory fix is bit-equivalent (not a silent behavior change).
3. Confirm the run config is spec-exact: `--roll-in laso`, decode via `predict`
   defaults, own-best `τ`, and that scoring uses the standard scorer — NOT any
   custom voter/DP re-decode.
4. When results exist: check boundary rate ∈ 0.05–0.15, report dev macro-F1 vs
   83.61/84.06, and φ vs the AraBERT tagger cache (`probs/union-NoPnx-NP-*_dev.npz`).
5. Report any remaining deviation from the spec as a ranked list, most severe first.
   Do not fix anything without sign-off; per CLAUDE.md, STOP and ask on any ambiguity.
