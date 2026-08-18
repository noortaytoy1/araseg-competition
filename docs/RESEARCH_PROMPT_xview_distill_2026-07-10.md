# Research prompt: critique and extend "privileged cross-view distillation" for punctuation-free Arabic sentence segmentation

You are a skeptical NLP researcher. Below is a fully-diagnosed low-resource problem and ONE new proposed method. Your job: (1) stress-test the proposal against real literature, (2) tell us the design choices that decide success vs a null, (3) propose anything BETTER that exploits the same unique data structure. Cite only real papers with venue+year; give measured effect sizes where they exist.

## The problem (compressed — full null-list at bottom)
- AraSeg 2026 closed track: **174 labeled Arabic documents**, binary boundary-after-token, per-document macro-F1. No external data, no generated text, no dev-set training (explicit organizer ruling). Pretrained encoders allowed.
- Four tracks = the **same 174 documents in four views**: with/without punctuation × with/without paragraph marks. Verified: identical doc IDs across views.
- Scores (fine-tuned AraBERTv02, single encoder): **PA (punctuated) ≈ 94.4** vs **NoPnx-NP (stripped) ≈ 83.6**. Submission ensembles: ~95.2 / ~85.0.
- NoPnx-NP model diagnosis (measured, verified): confidently over-segments (false cuts : misses ≈ 1.8:1 at P≈0.99), imposes a ~9–10-token sentence-length prior (gold mean 9.88 → predicted 9.04), over-cuts long/legal sentences and document ends, under-cuts (P≈0.00) genealogy chains, hadith isnad, Quranic verse openings, connective-initial sentences. ~15 generic methods nulled against a ±0.45 seed-noise floor; only R-Drop moved (+0.37, sub-floor).

## The proposed method (critique THIS)
**Privileged cross-view distillation (LUPI / generalized distillation, Lopez-Paz et al. 2016):**
1. Teacher = the PA-finetuned model, run once over the **punctuated view of the same 174 training docs** (~94.4 F1 on this distribution — it sees the periods).
2. Fold the teacher's per-token boundary probabilities onto the stripped-token grid (punctuation-token probability mass folds onto the preceding kept token, mirroring how the gold labels were derived).
3. Student = the NoPnx model, trained with weighted-CE on gold **+ λ·KL(student ∥ teacher soft targets, temperature T)** on aligned tokens.
4. Variant/sibling: an **auxiliary punctuation-prediction head** on the student ("was a punctuation mark stripped after this token, and which class"), supervised from the PA view, discarded at inference.

Why we think it can work: the teacher's soft targets carry exactly what the student lacks — *calibrated* boundary judgment (the teacher is right where the student is confidently wrong), knowledge that genealogy chains DO break and long legal clauses DON'T, i.e. the length/genre structure. It is closed-legal: same 174 docs, organizer-provided views, teacher never touches dev/test.

## What we need from you
1. **Failure-mode analysis with citations:** when does privileged/cross-view distillation fail? Specifically: the teacher's decisions are *caused by* punctuation the student cannot see — is there evidence (LUPI theory or empirical) about distilling from features the student cannot recover? Does the student just learn an unreachable target, or does the dark knowledge still transfer? What do ASR / speech-segmentation / punctuation-restoration literatures say (they face exactly this: text/audio without punctuation, training signal derived from punctuated text)?
2. **Design knobs that decide the outcome:** temperature; λ schedule; distill on all tokens vs only teacher-uncertain tokens vs only boundary-plausible tokens; KD on logits vs probabilities; teacher = single model vs the PA *ensemble* (95.2); should the teacher read the NoPnx-PA view (paragraphs but no punctuation — a "middle" teacher at ~87) instead of / in addition to PA, as a curriculum?
3. **Anything better that uses the same unique structure** (same docs in four aligned views)? E.g. multi-view co-training, view-consistency regularization (predictions must agree across views), training ONE model on all four views jointly with view embeddings, curriculum from punctuated → stripped. Real evidence only.
4. **Prior art check:** has punctuation-as-privileged-signal for sentence segmentation been published (SaT "corruption" recipe, WtP, ersatz lineage, "punctuation restoration improves X")? If someone already did this, tell us what worked and what numbers they got.

## Already tried — do NOT re-suggest
Preposition-targeted consistency (counterfactual/Mean-Teacher/CVT): null. Char-level augmentation: worsened over-firing. SUB2 span-swap, recombination (−1.56), TAPT (−0.61), DAGA (−3.5): dead. SupCon/SimCSE/paragraph-contrastive: null. Fire-side & symmetric confidence penalties, label smoothing, soft-F1: null. OOV/perplexity abstention gate: provably useless (AUC≈0.5 — true boundaries are as "unfamiliar" as false cuts). Multi-stride/center-weighted decode: +0.117 sub-floor. Larger encoders (AraBERT-large, XLM-R-large): lose to base on 174 docs. R-Drop: +0.37 (real, sub-floor — being folded into the ensemble). Currently running: MRT over macro-F1 (early-stopped), end-to-end Filtered Semi-Markov CRF with learned duration potential, SAM/class-SAM, Group-DRO with length pseudo-groups.
