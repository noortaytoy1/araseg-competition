# Research request: cures for seven measured failure modes of a low-resource Arabic sentence-segmentation model

Self-contained — no prior context needed. **Structure: Section A gives the task in five lines. Section B is the core — the SEVEN FAILURE MODES of our model, each with real examples and what already failed against it. Section C gives the three cure families we believe in (contrastive representation repair, regularization, ensemble combination). Your job (Section E): map each failure mode to the best-evidenced cure from those families, with citations and effect sizes.**

## A. Task in five lines
Split unpunctuated Arabic text into sentences: after each word, answer "does a sentence end here?" Trained on exactly **174 labeled documents** (hard rule; no external/generated/validation data — organizer rulings). Pretrained encoders allowed. Score = boundary F1 per document, averaged (macro), on 222 held-out dev docs. Our single model: ~83.6. Our 6-model ensemble submission: ~85.0 dev / 84.97 test, rank ~4th — the top 3 beat us on identical data, so a path exists. Random-seed noise floor: ±0.45 for one model (gains below that are luck), ≈±0.2 for the ensemble.

## B. THE SEVEN FAILURE MODES (each measured and independently verified — reason from these)

**FM1 — Confident false cuts on unfamiliar material (the biggest: ~66% of confident errors).**
The model's reflex is *strange context → cut here*, delivered at probability ≈0.99. Example (▮=model's cut, gold says no): `...بقشيش الكازينو ▮(P=1.00) هل نذهب...` — it slams a boundary next to a rare loanword (casino).
*Crucial finding:* these errors are **inside the learned representation** — a nearest-neighbor lookup over the training set's own embeddings votes these false cuts UP (0.976, same as real boundaries at 0.995): the encoder has genuinely mapped these contexts into "boundary-shaped" vectors. Familiarity/OOV signals score AUC 0.50 (useless — 95% of dev word-pairs are unseen; true boundaries are exactly as "unfamiliar" as false cuts). Failed against it: character-level augmentation (made it worse), all confidence penalties (traded precision for recall 1:1).

**FM2 — Exercise/quiz text gets shredded (worst genre).**
School-exercise documents (blanks written as فراغ) have a rhythm the model never learned (genre appears 2× in train, 142× in dev). It cuts them to pieces: predicted sentence length 3.7 words vs true 5.1; the blank-token فراغ alone sits before **99 false cuts**; **44% of its most-certain cuts (P≥0.99) in this genre are wrong.** Example: `...الصحيحة بدلا من فراغ ؟ ▮(P=1.000) فراغ . العطلة...` Fixing just 4 such documents is worth +0.54 macro-F1 (measured ceiling).

**FM3 — Chain-style text gets fused (the opposite error, same model).**
Genealogies ("X son-of Y ▮ son-of Z..." where each link is a sentence), hadith transmission chains, Quranic verse lists: the model refuses to cut at probability ≈0.000. Example: `...بن حسلي بن نجاي ▮(MISSED, P≈0.000) بن مآث بن متاثيا...` Most damning: one random seed under-cuts the genealogy document (misses 17/27 boundaries) while its identically-trained sibling seed OVER-cuts the same document by 85% — **the model has no learned convention at all here; it guesses.**

**FM4 — Sentences starting with connectives are missed.** Arabic freely starts sentences with و ("and"), ثم ("then"), كما ("also"). The model reads them as continuations, quietly (P<0.05): `...من جميع الجهات ▮(MISSED, P=0.012) ثم قاموا بتفجيرها...` ~10% of all misses.

**FM5 — A memorized length prior distorts everything.** Gold sentences average 9.88 words; the model predicts 9.04 average and *narrower spread* — it chops long legal clauses (~2.5× more false cuts right after bare numbers) and merges short sentences (1–6-word sentences lose their endings at 13–14%). An end-to-end semi-Markov CRF with a learned length model was our structural fix — its length table turned out learning-rate-starved (repaired version queued).

**FM6 — The engine underneath: memorization + overconfidence.** Train score 99.02 vs dev 83.01 (+16 gap; 135M parameters vs 174 documents). 98% of outputs at probability extremes; median confidence on a WRONG cut = 0.992; 80% of errors repeat across random seeds (systematic, not luck). Post-hoc calibration can't help (it preserves rankings); only training-time changes moved anything.

**FM7 — Ensemble absorption (the meta-failure).** Our 6 voters make the same mistakes (error correlation 0.74), so single-model gains vanish in the ensemble: a +0.9 single-model gain → +0.46 in-pool → **tie** at test; +0.68 → −0.06. The submission only improves via (a) very large single-model gains, or (b) voters/combiners with *different* errors.

## C. The three cure families we believe in (our bet — assess and strengthen them)

**C1 — Contrastive representation repair (aimed at FM1/FM6).** Since the errors live in representation space (FM1), we built: anchors = true-boundary contexts, hard negatives = the model's own highest-confidence wrong-cut positions mined from training data (train-side confident errors are rare — 156 — so fallback: the 2,000 hardest non-boundary positions), margin loss on a projection head. Status: built, queued for GPU.

**C2 — Regularizer stack (aimed at FM6, indirectly FM1/FM5).** Individually measured, all real, all sub-floor: SAM (flat-minima optimizer) +0.39; R-Drop (dropout-consistency loss) +0.37 with precision +1.9; class-asymmetric SAM +0.32; plus the one big data lever: training on each document *and its punctuated twin* (organizer-provided view of the same 174 docs) +0.9. An integration run stacking twin-data + SAM + R-Drop is queued. Also untried and promising for FM6: partial/parameter-efficient fine-tuning (freeze layers / LoRA / Child-Tuning) to cut capacity below memorization level.

**C3 — Ensemble combination (aimed at FM7).** Replace the ensemble's plain probability averaging with a trained meta-combiner ("stacking") over out-of-fold member predictions + two weak auxiliary signals we measured (dropout-instability of a prediction: AUC 0.71 at flagging the model's confident errors; a 2D similarity-matrix block signal: AUC 0.62 at the same). Spec ready (5-fold out-of-fold design to avoid the memorized-predictions trap), not yet run.

## D. Banned (measured failures — do not re-suggest)
Char/span/recombination/generative augmentation; continued-MLM pretraining; token-SupCon, SimCSE, paragraph/coherence contrastive (generic semantic variants — distinct from C1's hard-negative design); confidence penalties, label smoothing, soft-F1; OOV-abstention gates; nearest-neighbor voting at inference; hard filtering by instability; bigger encoders (large models LOSE to base at n=174); reinforcement/minimum-risk training; group-robust optimization (−0.92); audio/TTS expert fusion (learned fusion gate weighted it 0.04 — dead); recursive re-reading.

## E. YOUR JOB — four steps, in order
**Step 1 — Failure-mode → cure map.** For EACH of FM1–FM7: name the best-evidenced published technique from families C1/C2/C3 (or an unbanned fourth option if the evidence is strong) that attacks it, with citation + measured effect size in a comparable low-resource setting. If no evidenced cure exists for a failure mode, write "no evidenced cure — park" (that is a valid answer).
**Step 2 — Strengthen C1 (the contrastive bet).** Failure modes of hard-negative contrastive when the negative miner IS the model being repaired; the concrete design changes with the best evidence (negative sampling scheme, margin vs InfoNCE, projection depth, joint vs sequential with cross-entropy); and the strongest published precedent of contrastive repair moving a *token-level* error cluster at <500 training docs.
**Step 3 — Strengthen C2 + C3.** (a) Do SAM and R-Drop compose or overlap (evidence)? The one additional regularizer with the best evidence of composing with both. (b) The best-evidenced partial fine-tuning config for ~200-doc token tagging that *reduces the train-dev gap*. (c) For stacking: the strongest small-data sequence-labeling precedent of a trained combiner beating probability averaging, and the features that mattered.
**Step 4 — Blind spot.** The ONE technique with real published evidence for this exact regime that neither Section D bans nor Sections C covers. If none, certify "coverage complete."

**Format:** answer as Step 1–4. Step 1 as a table (FM → cure → citation → effect size). Every claim cited (venue+year); speculation flagged as speculation; max ~2 pages per step.
