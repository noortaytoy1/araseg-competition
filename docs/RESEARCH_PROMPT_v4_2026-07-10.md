# Research prompt v4: composing sub-threshold gains into a real improvement (low-resource Arabic segmentation)

You are a skeptical NLP researcher. Everything below is measured and independently verified. We are past the idea-lottery phase: we hold a pile of REAL but individually sub-threshold improvements and are about to stack them. **We need evidence-based guidance on (a) making the stack compose instead of overlap, (b) partial fine-tuning to attack a +16 memorization gap, (c) beating ensemble absorption, and (d) anything that moves a poisoned representation.** Cite only real papers (venue+year, effect sizes). The graveyard at the bottom is non-negotiable — do not re-suggest it.

## 1. Task, constraint, standing
AraSeg 2026 closed track: **174 labeled Arabic documents** (same docs provided in 4 views: ±punctuation × ±paragraph marks), binary boundary-after-token, per-document macro-F1. No external data, no generated text, no dev training (organizer rulings). Pretrained models allowed. Hardest track (NoPnx-NP: fully stripped): single encoder ≈83.6, 6-voter ensemble + semi-Markov decode ≈85.0 dev / 84.97 test. ~4th place; top-3 on identical data. Single-model noise floor ±0.45; ensemble-level ≈±0.2.

## 2. Verified diagnosis (compressed)
- **Memorization:** train 99.02 / dev 83.01 F1 (+16 gap). 97.9% of outputs at extremes; median confidence on a WRONG cut = 0.992.
- **Errors are systematic** (79.5% shared across seeds) and **representation-resident**: at the model's confident false cuts, a kNN over training-token embeddings votes them UP (mean 0.976 — indistinguishable from real boundaries' 0.995). The encoder maps these wrong contexts INTO the boundary region. OOV/familiarity signals: AUC≈0.5. MC-dropout instability: AUC 0.711 (real but net-negative as a hard filter; kept as a feature). 2D similarity-matrix block-contrast at tagger-blind positions: AUC 0.62.
- Bidirectional genre failure with a ~9-token length prior: over-cuts cloze (1.39×)/legal/long sentences/after-numbers; under-cuts genealogy/isnad/verse chains and connective-initial sentences (P≈0.000).

## 3. The POSITIVE pile (all single-encoder NoPnx-NP, 3 seeds, dev; all sub-floor individually except union)
| lever | Δ | mechanism level |
|---|---|---|
| union/multi-view training (doc + punctuated twin, 348 examples) | **+0.9** | data |
| SAM (rho 0.05) | +0.39 | optimizer (weight-space flatness) |
| R-Drop (w=1.0) | +0.37 | loss (output consistency; precision +1.9, over-firing down in all genres) |
| class-asymmetric SAM (alpha 2) | +0.32 | optimizer |
| 2D U-Net refiner (similarity-matrix + prob channel) | +0.24 | decode-side refiner |
| center-weighted multi-stride decode | +0.12 | decode |
| MC-dropout instability / 2D block-contrast | AUC 0.71 / 0.62 | features (for a stacking combiner) |
Historical absorption warning: union's +0.9 → pool +0.46 → **test tie** ("ensemble absorbs per-voter gains"); smooth+FGM +0.68 solo → test −0.06. Voter error correlation 0.74.

## 4. The integration plan (running) — critique it
One recipe: union data + R-Drop + SAM in one trainer (different mechanism levels), then decode-side multi-stride + U-Net blend, 3 seeds vs proper baselines with an arm isolating union-alone. If the stack lands ≥+1 solo, retrain all 6 ensemble voters with the winning recipe.
**Ask 4a:** evidence on COMPOSITION — do SAM and R-Drop (weight-space flatness + output-consistency) compose or overlap in practice? Any published SAM+consistency-regularization stacking results? What third regularizer composes best with both (EMA/SWA? mixup? stochastic depth?)?
**Ask 4b:** evidence on making solo gains SURVIVE ensembling — is uniform-recipe retraining of all voters the right move, or does it re-correlate errors? Alternatives with evidence?

## 5. Partial fine-tuning (the user's directive — assess and give the best recipe)
The +16 memorization gap suggests the 135M-param encoder has far too much capacity for 174 docs. Untried here: freezing most layers, LoRA/low-rank updates on the closed-track encoder, Child-Tuning (gradient masking), BitFit, layer-wise lr decay (exists but unused). **Ask 5:** for token-level tagging at n≈174 docs with a strong pretrained encoder, what does the evidence say is the best parameter-efficient/partial fine-tuning recipe to REDUCE the generalization gap (not just match full fine-tuning)? Concrete configs (which layers, ranks, lr ratios) with measured low-resource gains. Note: full fine-tuning of LARGER encoders already failed here (AraBERT-large < base on all tracks).

## 6. Moving the representation (the deepest problem)
Since the errors are representation-resident (kNN votes them up), we built a hard-negative contrastive arm: anchors = true-boundary token states, hard negatives = the model's highest-confidence wrong-cut positions mined from TRAIN (train-side confident FPs = only 156 due to memorization → fallback to top-2000 hardest gold-0), margin loss on a projection head, optional arctanh-InfoNCE (arXiv 2501.17683). **Ask 6:** critique this design; known failure modes of hard-negative mining when the miner IS the model being fixed (self-referential negatives)? Better-evidenced ways to surgically move specific error clusters out of a decision region at n=174 (e.g., prototype/metric methods, targeted feature editing, model editing literature — ROME/MEMIT-style for encoders)?

## 7. One idea to assess honestly: audio foundation models
The user proposes leveraging audio foundation models (prosody knows where utterances end), fine-tuning only parts. Recorded local evidence AGAINST the direct form: a fused CamelBERT + Arabic-TTS-text-encoder two-expert model scored 51.4 F1 audio-only, and the learned fusion gate weighted the audio expert 0.04 (the model ignored it). **Ask 7:** is there ANY published evidence of audio/prosody foundation-model representations helping segmentation of WRITTEN text (no audio input at test time) — e.g., prosody-aware text embeddings, TTS-intermediate supervision, punctuation restoration via prosody transfer? If the evidence is thin, say so plainly so this stays closed.

## 8. Graveyard — measured, verified, do NOT re-suggest
Preposition-targeted consistency (cf/MT/CVT): null. Char-aug: worsened over-firing. SUB2/recomb/TAPT/DAGA: dead. Token-SupCon/SimCSE/para-contrastive/coherence-shuffle: null. Fire-side + symmetric confidence penalties, label smoothing, soft-F1: null. OOV abstention: AUC 0.5. kNN voter: votes the errors UP (rejected). Instability hard filter: net-negative at every cutoff. Multi-stride alone: sub-floor (+0.12, kept only as stack layer). Larger encoders (AraBERT-large, XLM-R-large): < base. Recursive re-reading: killed. Punctuation-dropout: lost to union. Cross-view KD (PA teacher): teacher memorized train (0.40% soft mass) — no dark knowledge. Group-DRO: −0.92. MRT: heavy negative at epoch 0 (time-boxed rerun pending). Semi-CRF end-to-end: −4.1 = decode bug (+1.9 recovered) + lr starvation of duration table + encoder drag (rescue with param groups pending — R1/R2/R3 pre-registered). Direct audio-expert fusion: gate weight 0.04, dead. 2-channel U-Net with tagger-prob input: copies the tagger (phi 0.96) — only the matrix-only version remains under test.
