# Brief for the next Claude — fix rmerge TRAINING to actually implement §2.4

READ FIRST: `docs/METHODOLOGY_recursive_beam_merge_2026-07-11.md` (the spec) and
`docs/PROPOSAL_rmerge_for_recheck_2026-07-12.md`. This brief tells you the ONE thing
to understand and the exact fix. Do not run anything on the GPU without Noor's explicit
yes (GPU spend is a hard stop). Do not touch `runs/` (append-only evidence).

## The one thing to understand

`src/rmerge.py`'s **inference** beam, **scorer**, and **loss form** are faithful and
verified clean (C5/C6/C7 in the audit). The problem is 100% in **TRAINING**: **no code
path implements the §2.4 roll-in.** All three `--roll-in` options are wrong:
- `teacher` = pure teacher forcing (spec says NO teacher forcing).
- `sched` = scheduled sampling, p_model ramps 0→0.25 (still ≥75% teacher-forced).
- `laso` = the beam is **decorative**: `doc_train_loss_beam` (rmerge.py:714) calls
  `beam_decode(...)` once (no grad) and uses the result **only** as `len(win.nodes)`,
  a stop-depth counter (rmerge.py:751); it then supervises a **single** teacher-heavy
  roll-in path. **The 3 beam hypotheses are never scored into the loss.**

Net effect: what any run trains is "AraBERT as a pairwise merge-scorer via dense,
mostly-teacher-forced supervision on clean within-sentence sub-spans." The load-bearing
mechanism of the idea — **the model explores its OWN cross-boundary mistakes during
training and gets punished for them** (FM3, the local-minimum escape) — is never
exercised. **A bad dev-F1 from any current path does NOT kill the recursive idea**
([[no-slop-verify-all-code]], [[diverse-encoder-ensemble-closed]]).

## What §2.4 actually requires (build THIS)

A true width-3 beam during training where the scored probabilities that drive the beam
ARE the probabilities fed into the loss, with NO teacher forcing:

1. Maintain a beam of ≤3 hypotheses; each = a node list + cumulative log-product score.
   Start = all words separate (all cuts present).
2. **Each step:** enumerate every adjacent candidate pair across **all** live hypotheses;
   score them in ONE batched AraBERT forward **WITH grad** → logits `p` (§2.3).
3. **Loss (§2.4):** add `BCE(p_k, t_k)` for **every** candidate just scored (all
   hypotheses, all steps), `t=1` if within-sentence (`gold_cut[k]=0`), `t=0` if boundary.
   Class-weight the minority (boundary) target. This is dense supervision on all hyps.
4. **Expand + prune (NO teacher forcing):** generate children by applying candidate
   merges (respect span-cap force-0); keep top-3 by the model's OWN product score — the
   beam may merge across a boundary when overconfident; step 3's `BCE(p,0)` punishes it.
5. **Stop** a hypothesis when its best candidate `p < τ_train` (e.g. 0.05).
6. Backprop through the summed loss (the discrete top-3 pick needs no gradient, §2.7).

The difference from today's code: today `beam_decode` runs no-grad and its logits are
discarded; you must run the beam so the **same** forward that ranks hypotheses also
produces the graded logits for the loss, and supervise **all 3**, not one path.

## Also fix
- **C4 stale cache:** `_span_cache` is keyed on token-ids only and never cleared across
  epochs → epoch>1 reads stale probs while weights changed. In the training beam, either
  disable memoization or `scorer.clear_cache()` every epoch. (Teacher/sched bypass it.)
- **Memory:** the per-step candidate graph across 3 hypotheses accumulates → OOM on long
  docs (same class as the bug already fixed). Backward per step (accumulate grads), like
  `backward_each_state`, so retained graph is bounded to one step.

## How to prove it's right (before asking Noor for a GPU run)
- Add a TEST that guards the mechanism (the suite currently has none): assert that on a
  toy doc the training step (a) scores >1 hypothesis, (b) can take a cross-boundary merge
  under the model's own probs (no teacher forcing), (c) that merge contributes a `BCE(p,0)`
  term. Keep the existing CAN-FAIL acceptance test green.
- `pytest tests/test_rmerge.py` must stay 30/30 (CPU, `CUDA_VISIBLE_DEVICES=-1`).
- Sanity gates on any eventual run (§6): predicted boundary rate ∈ **0.05–0.15** (not 0/1);
  own-best `τ` dev grid; report `φ` vs the AraBERT tagger (value is decorrelation, FM8).

## Do NOT
- Do NOT window the decode (`predict --window 0`, full-doc). Windowing is a training
  cost note only; the previous session wrongly windowed a decode.
- Do NOT teacher-force the training roll-in.
- Do NOT score the model via `rmerge_voter_eval.py` / Omar's DP re-decode — use
  `rmerge.py predict` (beam + own-best τ) only.
- Do NOT start any GPU run without Noor's explicit yes. Fixing code first; run is a
  separate approval.
