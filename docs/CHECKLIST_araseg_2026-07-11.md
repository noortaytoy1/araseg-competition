# AraSeg 2026 — Running Checklist (tried · running · to-try)

**Standing:** ~4th, closed track. Submission scores: PA **94.43** · NP **92.88** · NoPnx-PA **87.38** · NoPnx-NP **84.97**.
**The wall:** data, not architecture — ~88% of NoPnx errors are on token-bigrams unseen in the 174 train docs. Caps everyone.
*Updated 2026-07-11.*

---

## ✅ TRIED — dead or parked (with the number that killed it)

| idea | verdict |
|---|---|
| Filtered Semi-CRF | **−4 F1** (79.6 vs 83.5). Dead. |
| Minimum-Risk Training (MRT) | **−1.34**. Dead. |
| Group-DRO / GEORGE | **−0.92**. Dead. |
| kNN boundary voter | flat (83.94 < 83.96). Parked. |
| Instability filter (as hard cut) | AUC 0.711 real, but net-negative ΔF1 → kept only as a stacking feature. |
| Char-level augmentation | sub-floor. Parked. |
| Fire-penalty (Noor's flagship) | +1.0 precision / −1.2 recall = null. Parked. |
| Label smoothing · SUB2 · conf-penalty | all sub-floor. Parked. |
| Audio two-expert | closed on published evidence (gate 0.04; SLAM/Talman/LUPI). |
| **Cut-token grouping** (Noor's [CUT] idea) | +0.22 sub-floor, φ 0.72 = tagger twin. Parked. |
| **Merge-eraser** (Noor's erase-policy) | +0.05, both gates failed. Parked. |
| **RAE composability** (Socher hint) | boundary signal real (0.86) but *inverted* on the error slice (0.31). Salvage = stacker feature. |
| Composer pipeline (pretrain→finetune) | hit the 0.72 ceiling, no value beyond the tagger. Parked. |
| **forest_train / forest_rlm** (recursive, teacher-forced) | exposure-bias plateau **~78**. Superseded by rmerge. |
| **BoundRL** (RLVR generative) | dead-end on this box: eval intractable (>1h/30 docs), RL never engaged (swaps=0). Parked. |
| Recursive re-reading | killed by its battery. |
| Surprise / OOV / perplexity gates | chance (~0.50). Dead. |
| Multiview / union training | +0.9 solo → **absorbed** to a test-tie by the ensemble. |

## 🟡 ALIVE — real signals, banked, not yet cashed
- **SAM** +0.39 solo · **R-Drop** +0.37 solo (they interfere: +0.29 combined).
- **S-only U-Net** (2D similarity matrices) — **φ 0.53**, the *one* genuinely decorrelated voter; vetoes 29–37% of the tagger's confident false cuts.
- **Instability std** — AUC 0.711, usable as a stacking feature.

## 🔵 RUNNING NOW
- **rmerge** — recursive AraBERT beam-merge segmenter (Noor's full spec + all report mandates, scheduled sampling). Building → 1-seed run tonight. Value = *decorrelated voter* (target φ<0.65), not a solo win.

## ⬜ TO TRY — remaining legitimate levers (the road left)
- [ ] **Stacking meta-combiner** — built + audited; blocked on disk (needs runs/ cleanup). Trained referee over the pool + U-Net + instability.
- [ ] **Hard-negative contrastive** — built (HARDNEG_GO). Attacks the representation-locus (drags confident-FP contexts out of the boundary region) — the only lever aimed at the actual disease.
- [ ] **Ensemble retrain with R-Drop** — the +0.37 ingredient applied to the 6-voter submission. Absorption risk. Touches the live submission = Noor's go.
- [ ] **Comma-convention disambiguator (PA/NP)** — the *one measured* lever on your two highest tracks; a learnable annotation rule, not the data wall. Best fresh shot at real points.
- [ ] **Morphology normalization** — measure-first: does lemmatizing shrink the unseen-bigram wall? If yes, a genuinely new lever; if no, killed with a number.
- [ ] **rmerge → ensemble blend** — if φ<0.65, blend it in (Krogh-Vedelsby ambiguity dividend).

## 🚫 OFF THE TABLE (with reason)
- **Cross-view fold** (+9.86 dev) — *leakage*; Noor's own ethics call. Off.
- **Dev-set training** — organizer-illegal (Bashar's ruling).
- **LM generation / external data** — closed-track illegal (open-track only).
- **Audio** — closed on evidence.

## 📋 ON NOOR'S DESK (decisions that unblock work)
- [ ] **runs/ cleanup** (289GB dead weights) → unblocks the stacker + frees disk for the ensemble retrain.
- [ ] **Ensemble retrain go/no-go** — touches the live submission.
