# ENSAR at AraSeg 2026 — verified fact sheet for the paper writer
Every number below is recomputed from source by `python paper/verify_paper.py` (154 checks) unless marked
PENDING. Do not invent numbers. Do not soften or inflate. PENDING items are being measured now; leave a
visible placeholder until Noor supplies the value.

## A. The task (Abstract / Intro Q1 / Background)
- AraSeg 2026 shared task (ArabicNLP 2026). Cite: elkholy-etal-2026-araseg-shared-task (organizers' BibTeX,
  already in custom.bib) and the corpus paper elkholy2026arabicsentencesegmentationgenres.
- Label every whitespace token sentence-final or not. Four tracks = 2x2: punctuation present/removed x
  paragraph structure kept/removed. Tracks: PA, NP, NoPnx-PA, NoPnx-NP.
- Paragraph-aware tracks: paragraph breaks are newline tokens; the token before a break always ends a
  sentence; the break token never carries a label. The final token of every document ends a sentence.
- Corpus: Modern Standard Arabic, 8 genres (scripture, classical narrative, hadith, children's readers,
  translated monographs, exam/MCQ sheets, ...). Boundaries are 8.3%–10.4% of tokens by track.
- Splits per track: 174 train / 222 dev / 262 test; blind 212 (NP tracks) / 100 (PA tracks).
- Why hard for Arabic (cite corpus paper, REWORD it): punctuation ambiguous and inconsistently used; many
  texts carry no reliable boundary markers; cross-genre generalization reported as still challenging. Also:
  the conjunction wa opens sentences as readily as it joins clauses inside one. In the NoPnx tracks a
  boundary has no surface mark at all.

## B. Official result (Abstract / Intro Q3 / Results 6.1) — AUTHORITATIVE
- Ranked FIRST on Pnx-PA, Pnx-NP, NoPnx-NP, in BOTH closed and open settings.
- Blind F1: Pnx-PA 95.30, Pnx-NP 92.0, NoPnx-NP 89.1.
- NoPnx-PA: 90.0, did NOT place first. Reason to state: those juries were rebuilt from scratch in the final
  hours before the deadline and had been trained on only 60 of the 174 training documents.
- NEVER write "first on all four". Write "first on three of the four tracks".

## C. The approach in one paragraph (Intro Q2 / System Overview)
Stage 1: an ensemble of five fine-tuned encoders proposes boundaries (semi-Markov DP decode).
Stage 2: two LLM juries (Claude Opus 5, maximum reasoning effort). Each jury holds a policy file it wrote
ITSELF during training from its own graded errors on the TRAIN split. At inference the two juries debate
each document from their own policies; only edits BOTH endorse are applied to the ensemble's draft.
Motivating number (Appendix B): 48.3% of the ensemble's dev errors are sites where all five voters agree
with each other and disagree with the annotator (1,811 of 3,749). No threshold, calibration or extra voter
can reach those; a statement of which convention governs the document can.

## D. Encoder ensemble (System 4.1) — exact
- Encoders: AraBERTv0.2-base, AraBERTv0.2-large, Arabic-BERT-large (Safaya), XLM-RoBERTa-large.
  Bibs: antoun-etal-2020-arabert, safaya-etal-2020-kuisail, conneau-etal-2020-unsupervised.
- Five voters: AraBERTv0.2-large twice (FGM eps=1.0 [miyato2017adversarial]; supervised contrastive
  weight 0.1 [khosla2020supervised]) + the other three at defaults.
- Training: 10 epochs; windows of 180 words, stride 90; max seq len 512; cross-entropy with the boundary
  class weighted min(neg/pos, 8) = 8.00 on every track; lr 5e-5, batch 16 default; xlmr-2e5 = lr 2e-5,
  batch 4. Probabilities averaged across voters and across overlapping windows. Checkpoints selected on dev.
- Decode: semi-Markov DP [sarawagi2004semimarkov], lambda=0.4, beta=0.75; length prior refit on the
  first-pass output, blended with the corpus prior at 0.7, decode again; structurally certain boundaries
  forced. Pnx-PA at submission: tuned 3-voter subset with beta=0.6.
- Plain English for the equation: a boundary is kept when its probability and the lengths of the segments
  it creates are jointly more believable than leaving it out.
- Solo voters (NoPnx-NP dev, threshold 0.5): every voter alone scores below the ensemble; solo macro F1
  80.74–84.56 vs ensemble 85.62.
- Resources beyond train: NONE (no lexicons, no extra data, no translation, no augmentation). FGM and
  SupCon are training-time regularizers on train only. Only external resource: the pretrained LLM (permitted).

## E. Jury training loop (System 4.2) — exact
Per batch of FIVE training documents:
1. ATTEMPT: the jury reads its current policy, segments each doc from scratch, records boundaries as token
   index + the token itself.
2. REWARD: scored against TRAIN gold. (The grader allowed a 2-token alignment window; it fired on 1 of
   19,854 boundaries written during training — effectively exact. Say "effectively exact", give the count.)
3. DIAGNOSIS: every false cut and missed boundary returned in context with its answer beside the gold.
4. POLICY UPDATE: the jury writes the rule behind each error into its own file, its failure as the worked
   example; contradicted entries revised in place.
5. RESET: documents discarded; the next batch = updated policy + nothing else.
- Two juries on disjoint halves: 87 docs each. NoPnx-PA doctrine files: 1,398 and 2,098 lines.
- Superseded rules struck through, never deleted; every revision names the errors that forced it.
- Figure 1 (LAW 5, batch 2) is a verbatim abridged entry: a rule induced in batch 1, retired in batch 2.

## F. Adjudication (System 4.3) — exact
- MUST-STATE (Noor flagged 2026-08-20): both juries are simulated within a SINGLE model context per
  sitting ("You hold both juries" in the prompt); each is constrained to argue strictly from its own
  separately-trained policy file. The independence is in the POLICIES (disjoint training halves,
  different files), not in separate processes. This is the original competition-day design and produced
  every reported number. Do not let the paper imply two independent model instances.
- Ensemble marks presented as "an unreliable cheat sheet: consult, never trust".
- Each jury argues from its own doctrine; a rule earned from a graded error in this register outranks
  unsupported intuition; only unanimous edits apply; unresolved -> draft stands; "unchanged" is admissible.
- STRICT application: an edit applies only if the stated index holds exactly the stated token.
  3,258 of 3,258 evaluation edits met this (zero snapped, zero rejected). All F1 from the organizers' script.
- Train-from-scratch vs edit-at-inference asymmetry is deliberate: a jury that only ever corrected drafts
  could learn to defer to them; the learned object (where sentences end in each kind of text) applies to
  writing and judging alike; transfer is empirical (Table 1 shows it transfers).
- Optional anecdote: on one scripture document one jury argued from verse structure, the other from rhyme,
  and they endorsed the same edit by different routes.

## G. Experimental setup (Section 5) — answer the organizers' questions literally
- Final models trained on TRAIN ONLY. Dev = checkpoint + hyperparameter selection, never training data.
  Juries learned from train docs + train gold only. Released TEST split used for the controlled
  measurements (Table 1). Entered all four tracks, closed and open.
- Organizer ruling: post-editing an encoder system with a pretrained LM is permitted in both closed and
  open; dev/test may be used for selection but never as training data. PENDING: date + channel of the
  ruling (Noor must supply; nothing on disk records it).
- Preprocessing (VERIFIED in code): NO normalization of any kind (alef/ya, ta marbuta, diacritics, tatweel
  all preserved; the token sequence is the label space). Each encoder's own subword tokenizer over the
  organizer whitespace tokens; the word label is carried by the LAST subword of each word.
- Dialect: corpus is MSA; no dialect, Arabizi, or code-switching; all documents treated identically.
- Prompt language: all jury prompts in English over Arabic documents; Arabic prompts NOT tried (say so).
- Tools: PyTorch 2.5.1, Hugging Face Transformers 4.57.3 (pinned); no Farasa/CAMeL Tools; organizers'
  evaluation script for all reported F1.
- Hardware: PENDING — GPU model and per-voter training time (Noor/Omar). Jury adjudication: provider API,
  maximum reasoning effort, ~6 min wall-clock per document per lane, 4 lanes in parallel.
- Runs: every reported number is a SINGLE run, no averaging. Variability measured directly (section I).

## H. Ablation on released test (Table 1) — exact, strict, verified
Macro F1, n=262 per track, released test split, organizers' script, strict exact-index edits:

| track    | ensemble | + doctrine juries | delta | docs +/-/= | bounds +/- | paired t p | 95% CI     |
|----------|----------|-------------------|-------|------------|------------|------------|------------|
| NoPnx-NP | 84.64    | 89.49             | +4.85 | 186/28/48  | 923/417    | 1.5e-13    | [3.6, 6.1] |
| NoPnx-PA | 87.05    | 92.59             | +5.54 | 166/35/61  | 547/722    | 8.2e-13    | [4.1, 7.0] |
| NP       | 92.83    | 93.61             | +0.77 | 80/38/144  | 192/203    | 1.4e-04    | [0.4, 1.2] |
| PA       | 94.56    | 94.92             | +0.36 | 58/44/160  | 141/113    | 0.024      | [.05, .67] |

Average delta = +2.88. Total edits: 1,340 / 1,269 / 395 / 254.
Caveats to state: NoPnx-PA doctrine row = juries retrained on the full 174 after the deadline (the
submitted pair had 60/174); the ablation baseline = a thresholded average, ~0.5 below the submitted
length-aware decode on the NoPnx tracks; neither affects sign or significance.
Which juries (IMPORTANT, per-track provenance -- the paper must state this):
  - Pnx-NP and Pnx-PA SHARE ONE jury pair: np_train_work/doctrine_np_juror{0,1}.md, the submitted
    competition juries. One doctrine set covers both punctuated formats: the files carry general laws plus
    PA-specific sections (279 and 308 PA-numbered law mentions in j0/j1), and the exam prompt's FORMAT
    paragraph selects the track. That pair trained during competition week on punctuated TRAIN documents in
    both formats (banked attempts: 81+78 NP-format, 76+75 PA-format per juror pair; train split only,
    leakage-scrubbed before submission).
  - NoPnx-NP: its own pair, retrain3/nonp/doctrine_j{0,1}.md, the submitted juries, trained on a curated
    39-document curriculum (20 + 19 banked attempts).
  - NoPnx-PA: its own pair, retrain4/nopa/doctrine_j{0,1}.md, post-deadline retrain on the FULL 174
    (87 + 87 attempts, disjoint halves) -- the row marked unofficial.
  Blind-set structure (verified): NP-family tracks share one 212-doc blind set, PA-family tracks share one
  100-doc blind set, and the two families are DISJOINT (0 shared documents). Hence the shared pnx jury pair
  never saw any Pnx-NP blind document in PA form or vice versa; only its frozen train-learned rules crossed.
  CONSEQUENCE for the System section: the sentence "two juries trained on disjoint halves, 87 documents
  each" describes the NoPnx-PA retrain exactly; for the other tracks state the actual extents above. The
  "1,398 and 2,098 lines" figure is the NoPnx-PA pair. A nice side-fact the sharing gives the paper: the
  SAME doctrine pair produces +0.77 on NP and +0.36 on PA under different format laws, i.e. one learned
  policy serving two input formats.
The baseline row is purely neural: the ensemble's blind predictions reproduce exactly from cached voter
probabilities; the SUBMITTED systems are ensemble + doctrine juries.
Micro F1 (token-pooled) for the ensemble: 86.23 / 89.30 / 93.76 / 95.76 (higher than macro because errors
concentrate in long docs). The paper reports macro = the official metric.

## I. The three control experiments — status
- Exp 1, ZERO-SHOT juries (no doctrine step at all; told they are untrained; own Arabic + the text only;
  same prompt structure/packets/draft/unanimity/model): all four tracks, 262 each.
  NoPnx-NP DONE (strict scoring): ensemble 84.64 -> zero-shot 87.59 (+2.96) -> doctrine 89.49 (+4.85).
  Doctrine - zero-shot = +1.89 F1, paired t p=9.5e-04, Wilcoxon p=0.011; doctrine better/worse/same on
  92/73/97 docs; zero-shot applied 1,104 edits (0 non-exact) vs doctrine 1,340. The gap is ~4x the
  run-to-run wobble (0.48). Decomposition sentence: of the +4.85, about +2.96 is the model-plus-debate
  scaffolding and +1.89 is the learned policy on top, so the same model without the doctrine cannot
  produce the submitted result. Memorization pre-empt now has data: zero-shot lands 1.89 BELOW doctrine,
  so the model is not recalling the source punctuation from pretraining.
  ALL FOUR TRACKS DONE (strict scoring, n=262 each):
    | track    | ensemble | zero-shot        | doctrine | doctrine-zeroshot (paired t / Wilcoxon) |
    | NoPnx-NP | 84.64    | 87.59 (+2.96)    | 89.49    | +1.89 (p=9.5e-04 / 1.1e-02) |
    | NoPnx-PA | 87.05    | 88.71 (+1.66)    | 92.59    | +3.88 (p=2.4e-07 / 7.0e-04) |
    | NP       | 92.83    | 91.42 (-1.42)    | 93.61    | +2.19 (p=2.3e-06 / 2.7e-06) |
    | PA       | 94.56    | 92.18 (-2.38)    | 94.92    | +2.73 (p=8.3e-08 / 1.0e-08) |
  Zero-shot edits: 1104/867/780/760 (all exact, 0 rejected). HEADLINE: zero-shot juries HURT the two
  punctuated tracks (-1.42, -2.38) by over-editing strong drafts (780 and 760 edits vs the doctrine's 395
  and 254); they help only where the draft is weak. Averages: zero-shot nets +0.20 across tracks; doctrine
  nets +2.80. Doctrine beats zero-shot on ALL FOUR tracks (avg +2.60, every p<=0.011). The learned policy
  teaches restraint as much as action: it tells the juries when the draft is already right.
  Decomposition (NoPnx-NP): of +4.85, +2.96 is model-plus-debate scaffolding, +1.89 the learned policy.
  Memorization pre-empt: zero-shot lands below doctrine on all four tracks and below the ENSEMBLE on two,
  so the model is not recalling source punctuation from pretraining.
  Access audit: 275 transcripts; only packets, own verdicts, own temp analysis files; 0 forbidden paths;
  0 gold vectors (339 checked); the two "LOCKED"/"doctrine" string hits are agents' own variable names and
  document content (verified in context). Outputs scratch_exo/papereval/<track>/exam_out_zeroshot.
  DISCLOSURE (verified 2026-08-20): the agent platform auto-recalls brief session-memory notes into some
  agent contexts. Audit of all transcripts: the ONLY recalled note that ever appeared (16 zero-shot
  transcripts, and some doctrine/training agents) is a one-line curriculum ruling whose relevant content is
  "scripture stays [in the curriculum]" - it names the existence of a training curriculum and nothing else.
  Probes confirm NO note containing results, conventions, labels or numbers ever appeared in any agent
  (0 hits for the standings/ablation/scoring notes). The string "89.49" in doctrine-reading transcripts is
  a line of doctrine_j0.md itself ("BATCH 3 (mean F1 89.49)", a training-batch mean written during
  competition week) and is a numeric coincidence with the later ablation figure. Isolation paragraph should
  add one clause: "the platform may inject brief session notes into agent context; audits show the only
  such note named the existence of a training curriculum and carried no corpus content, labels,
  conventions, or results."
- Exp 2, GOLD EXAMPLES instead of doctrine: KILLED by Noor (2026-08-20), do not run or report. Reason: the
  comparison is contestable under either fairness framing (information-matched needs ~87 docs ~350k chars of
  context, impractical; budget-matched gives the examples arm less information than the doctrine distilled
  from 87 docs). 60/262 partial verdicts exist in exam_out_examples but MUST NOT be scored or cited.
  The paper instead carries one Limitations sentence: "We did not test an alternative that places raw
  gold-labelled training documents in the juries' context instead of the learned policy. At a matched
  context budget such an arm carries less information than a policy distilled from 87 documents; matching
  the information would require roughly 150-180k tokens of raw solved text per jury and would convert the
  policy's offline distillation into per-decision search, and the policy additionally encodes each jury's
  own graded errors, which raw documents cannot carry at any context length. We therefore leave the
  comparison open."
  Remove the gold-examples row from Table 1 in the scaffold.
- Exp 3, RERUN of the real doctrine setup on the first 50 NoPnx-NP test docs: DONE.
  Run 1 (published) 91.09 vs run 2 90.61 on the same 50 docs -> difference 0.48 F1 (paired t p=0.09,
  n.s.); 25/50 docs scored identically; both +5.9 / +5.4 over the ensemble (85.17 on those 50). Edits
  254 / 263, 0 non-exact. A third independent run by another account is in progress (kit sent).
  Sentence for Limitations: "A second full run on 50 NoPnx-NP test documents differed from the first by
  0.48 F1; the effect is an order of magnitude larger than the run-to-run variability."
- The arms are ONE-variable: identical docs, gold, draft, prompts, unanimity rule, model; only the jury
  context differs (nothing / 8 gold docs / learned doctrine). Caveat as treatment: without two distinct
  doctrines the control arms are closer to two copies of one policy, so their unanimity filter is weaker.
- "Is it the policy or the model?" paragraph: WRITE ONLY AFTER the Exp 1/2 numbers exist. Pre-committed
  branches: (i) doctrine > examples => the loop extracted rules raw examples do not carry; (ii) examples ~
  doctrine => accuracy matches; the doctrine's advantage is an auditable, revisable stated policy.
  Memorization pre-empt: NoPnx gold = the source punctuation deleted; recall would live in weights, not in
  the doctrine, and would show in the zero-shot arm; degraded documents disprove recall by construction.
  Model-constant corroboration available NOW: blind training F1 rises 85.1->90.9 (J0) and 88.5->94.5 (J1)
  as the files grow (Fig 2, 10-batch moving average); the same model reverses its default between batch 2
  and batch 13 on written rules (Appendix C).

## J. Edit-direction analysis (6.3) — exact
- NoPnx-NP: +923 added / -417 removed. NoPnx-PA: +547 / -722. The ensemble under-segments with nothing and
  over-segments with paragraph tokens; the juries correct each. No threshold/temperature/penalty change
  produces opposite corrections on two tracks from one system -> the layer conditions on what the document
  is.
- Punctuated tracks: 395 (NP) / 254 (PA) edits vs 1,340 / 1,269; more than half of docs untouched
  (144/262, 160/262); gain ~5 -> <1. The policy substitutes for missing punctuation rather than
  duplicating it.
- Failure mode: degradations concentrate where BOTH juries shared a wrong reading; unanimity filters
  disagreement, not shared error.

## K. Error analysis (Appendix B) — exact (NoPnx-NP, threshold 0.5)
- Dev: 3,749 error sites; 1,811 (48.3%) all-five-wrong. Median ensemble prob at a FALSE POSITIVE = 0.86;
  22.7% of TRUE boundaries fall below 0.86 -> a threshold that removes the typical FP forfeits ~1/4 of the
  real boundaries; calibration is monotone, cannot reorder. The worst fifth of dev docs holds 51.4% of
  error sites.
- Test: of 3,413 ensemble error sites the juries corrected 1,110 and introduced 230; 299 of the corrected
  (26.9%) were all-five-wrong sites (unreachable by reweighting/recalibration/promotion); the rest are
  mixed sites where the juries sided with the minority reading when the convention licensed it.

## L. What the doctrines contain (6.4) — exact, from the files
1. Count-first rule: measure words between paragraph breaks; short lines -> the line is the sentence, cut
   at the breaks and almost nowhere else. Written after scoring 33.3 on an MCQ exam bank where prose rules
   had been misapplied.
2. Genre-conditioned lengths: children's picture book ~10.6 words/sentence; children's magazine ~14.3;
   classical folk narrative ~11; translated monograph ~20.3.
3. The Fig 1 arc: induced, refuted, retired in writing; misses sat 1–2 clauses from a nearby false cut.
4. Both juries independently made "what kind of text is this" their explicit first step near the end of
   training.
Close: none of this is expressible by the encoders; none of it was written by the authors.

## M. Appendix C reading note (MUST keep)
NoPnx-PA input has ZERO punctuation (101,784 train tokens, not one mark). When a jury says "comma" it is
inferring the source's deleted marks from where the gold boundaries fall. That inference is the point.
Keep the inline editorial note in the Jury 0 batch-2 quote.

## N. Limitations / Ethics facts
- Cost: tens of thousands of tokens per document; quality over throughput.
- Proprietary model via API; nondeterministic decoding (provider defaults, no temperature/top-p set);
  reproduction depends on model availability; doctrines + prompts released.
- Run-to-run: 0.48 F1 on 50 docs (section I).
- Shared wrong readings uncaught; every degraded doc is one. One corpus, one language.
- Ethics: low-risk preprocessing; read-only over scripture/classical literature; MSA only, dialects
  unrepresented, no transfer claim; the organizers' corpus license (name it); no personal data; API cost ->
  access asymmetry; the encoder-only ensemble is the cheap alternative and is released.

## O. Bibliography
All keys below exist in paper/custom.bib (verified venues): elkholy-etal-2026-araseg-shared-task,
elkholy2026arabicsentencesegmentationgenres, antoun-etal-2020-arabert, safaya-etal-2020-kuisail,
conneau-etal-2020-unsupervised, miyato2017adversarial, khosla2020supervised, sarawagi2004semimarkov,
christiano2017deep, ouyang2022training, shinn2023reflexion, madaan2023selfrefine, yang2024optimizers,
yuksekgonul2024textgrad, read2012sentence, wicks2021unified, tilk2016bidirectional, minixhofer2023wheres,
frohmann2024segment, zhao2024expel (AAAI 2024, 38(17):19632–19642), wang2024awm (arXiv 2409.07429).
Citation-claim audit (what each paper actually supports): Read 2012 = "too narrow", "overly optimistic",
"degradation, drastic for some systems, away from newswire" (NOT "collapses"); Tilk 2016 = punctuation
restoration, boundaries follow from the predicted marks (no separate segmentation step); WtP =
punctuation-agnostic char classifier, 85 languages, newline self-supervision; SaT = pretraining for missing
punctuation, beats all baselines incl. strong LLMs esp. on poorly formatted text; ersatz = punctuated text,
candidate punctuation contexts, 23 languages; Reflexion = verbal RL from task feedback, episodic memory;
Self-Refine = own critique, no training; OPRO = prompt from scored previous candidates; TextGrad = textual
feedback through compound systems; ExpeL = insights from comparing successful/failed trajectories; AWM =
induced reusable workflows.

## P2. Execution environment (state in the paper)
Juries executed through the Claude Code agent framework (public CLI, version 2.1.234 recorded in
transcripts), model string claude-opus-5, maximum reasoning effort, provider-default sampling. Replication
as-performed requires Claude Code; jury/REPLICATION.md in the release documents the exact procedure, and
jury/run_sitting_reference.py provides an untested framework-free SDK approximation. Reproducibility is
bounded empirically, not procedurally: the 50-document rerun differed by 0.48 F1.

## P. Code / release
https://github.com/noortaytoy1/araseg-competition (jury/ = prompts, all six doctrines, grader, training
attempts, verdicts, ablation results, scorer; paper/ = tex + verifier). No data, gold, test text or weights.

## Q. Things that must NOT be in the paper
- "first on all four"; any number not in this sheet; the word "significantly" without its test;
  "collapses" for Read 2012; any claim that zero-shot is weak or strong before Exp 1 exists; haiku anywhere
  (there is none); any ±2 tolerance language for the evaluation (it was strict: 3,258/3,258 exact).


## R. Adjudication traces (Noor's request: show the juries solving a document step by step)
Two complete recorded traces, ready to paste as a new appendix section ("How the juries work a
document"). Both are the juries' own recorded arguments, verbatim except abridgement marked [...];
both are from the variance rerun (same doctrines, same protocol as the submitted system; the rerun is
already reported in Limitations) because that run's records are in pure transliteration, which the
pdflatex template can typeset. Both documents are wins: Surat an-Nas draft 28.6 -> jury 100.0
(20 tokens); the Doctorow translation draft 77.1 -> jury 87.5 (746 tokens). Body pointer sentence for
Section 4.3 or 6.4: "Appendix D reproduces two complete adjudication traces, one verse and one prose."

LATEX (paste as an appendix section):

\section{How the juries work a document}
\label{sec:traces}

Two complete adjudication records, in the juries' own words as written at decision time,
abridged where marked. The first document rose from 28.6 to 100.0 $F_1$; the second from
77.1 to 87.5.

\paragraph{A 20-token sura (Surat an-Nas, Q114).}
\begin{tcolorbox}[colback=white, boxrule=0.4pt, left=3pt, right=3pt, top=3pt, bottom=3pt]
\scriptsize
``REGISTER: the whole document is Surat an-Nas (Q114) entire, 20 tokens, standing alone,
not a quotation inside prose. J0 S21: for the Quran the RECORD is the AYA, every
\textit{wa}/\textit{fa} rule suspended, and the text carries its own hard external test
(known verse count plus known fasila list). J1 S16/S12c: verse cuts at the rhyme and
nowhere else. Both juries: 6 ayat = 6 units. J1's objection from S2b (quoted verses are
ONE unit) was withdrawn: that datapoint was a Quranic fragment embedded in Ibn Tufayl
prose, and the material J1 called verse 1 and verse 2 is in fact two clauses of a single
aya (Q8:17), so it is no evidence against per-aya segmentation of a standalone sura. J0
S21 is evidence from THIS register and outranks it. [Five adds follow, each the end of an
aya, each endorsed by both.] @10 al-waswas ARGUED AND REJECTED: J1's mechanical rhyme
scan (cut at every token carrying -as) fires here. J0 rebutted with S21's sharpest
lesson: a fasila can repeat INSIDE its own verse, so take the LAST occurrence before the
next ending, not the first (the an-Nahl trap that cost J0 a double penalty). al-waswas is
verse-internal; cutting it would give 7 units for a 6-aya sura. J1 conceded. NO CUT.
DENSITY: 20/6 = 3.3 tokens per unit, below J1's verse band of 4 to 6, but an exact
external invariant (the sura has exactly 6 ayat) outranks a density heuristic; J0 S24.2
orders verification against the invariant rather than the ear.''
\end{tcolorbox}

\paragraph{A 746-token translated novel chapter (Doctorow, \textit{Little Brother}).}
\begin{tcolorbox}[colback=white, boxrule=0.4pt, left=3pt, right=3pt, top=3pt, bottom=3pt]
\scriptsize
``REGISTER: Hindawi Arabic translation of Cory Doctorow, Little Brother, chapter 1:
first-person YA novel, running prose, not layout. J1 bucket (d) translated prose, cut at
the source-language period, 13 to 17 tokens; J0 S8 translated literary fiction about 14.
[...] @188: J1 wanted the mark removed (English has a comma before the but), J0 S1.3
holds that \textit{lakin} opens a sentence. UNRESOLVED, draft stands. @205: the 36-token
unit is the worst density outlier in the document. J1 first objected (parallel negations
do not cut), then conceded that this is an affirmative-then-negative contrast, not a
parallel negation, and that bucket (d) points at the source period; J0: \textit{wa} plus
finite verb in narrative. ADD. [...] @624 to @647, THE ONE STRUCTURAL ERROR IN THIS
DOCUMENT: the draft read four data lines as display-line records and split them off,
leaving a unit that ends on a dangling \textit{kana} with no predicate and a next unit
opening on a bare predicate with no subject. Correct parse: the quote-plus-tag sentence
ends at @624, and the identity block is a dash parenthetical between \textit{kana}'s
subject and its predicate, one sentence. J0's false-heading trap (does the verb after the
candidate agree with it and lack any other subject) and J1's rule (parse the verb's
subject before deciding a heading's extent) are the same law, and both fire. ADD 624;
REMOVE 627, 633, 638, 644, 647. @667 and @708: English joins these to the following
clause, but the Arabic translator plausibly broke here; no jury evidence strong enough
either way. NO CHANGE.''
\end{tcolorbox}

Six behaviours are visible across the two traces and none was scripted: register
identification before any edit; suspension of general rules when a register-specific law
applies; verification against an external invariant in place of intuition; an objection
withdrawn after the jury re-audited its own evidence; disagreements left unresolved with
the draft standing; and a structural correction argued from syntax rather than surface
pattern.


## S. RAW reasoning traces (chain of thought) — Noor's request, the model's internal reasoning verbatim
Distinct from section R (the juries' RECORDED arguments): these are the model's internal reasoning as
logged by the platform during the variance rerun, abridged where marked, Arabic transliterated. They show
the step-by-step solving: hypothesis at a site, law lookup, self-correction, relocation, verification.
Ready-to-paste LaTeX (a subsection inside the traces appendix, or its own):

\paragraph{Internal reasoning, a hadith document (abridged; Arabic transliterated).}
\begin{tcolorbox}[colback=white, boxrule=0.4pt, left=3pt, right=3pt, top=3pt, bottom=3pt]
\scriptsize
``I am looking at this hadith document from Sahih al-Bukhari, checking my current mark
placements against the isnad-breaking doctrine: the chain of narration splits at each
\textit{qala}, since these are separate transmission records rather than one continuous
sentence. Jury 1's notes add nuance: the final \textit{qala} still welds to the quoted
speech regardless of isnad length. [...] Now I am testing whether the chain breaks after
Mudrik. It should not, since the internal \textit{`an}-links within one transmitter's
report stay fused, so the whole stretch from `\textit{qala akhbarani Ali ibn Mudrik}'
through the Prophet's quote forms one long 23-token unit; I am cross-checking this
against a similar gold example with nested transmitters. [...] Now for \textit{faqala}
at token 28: this opens a new speech act. J0 \S22 names \textit{faqala} a boundary head
in hadith, and J1's \S9d and \S18 both support cutting before a narrative
\textit{faqala}, so a cut looks endorsed by both juries. But J0's gold example has the
boundary after a complete narrative clause, whereas here it falls after a quoted
imperative, and per \S9c an opening imperative right after the introducer welds to it
rather than breaking away; so `\textit{qala lahu fi hijjat al-wada` istansit al-nas}'
stays one unit and that cut should not happen. Both juries converge on adding the
boundary at token 27 (\textit{al-nas}) instead, once the next full clause
(`\textit{faqala la tarji`u...}') begins. Checking density: 37 tokens and 3 boundaries
is about 12.3 tokens per unit against J0's hadith benchmark of roughly 9.3; adding this
one boundary brings it to 9.25.''
\end{tcolorbox}

\paragraph{Internal reasoning, a translated novel (abridged).}
\begin{tcolorbox}[colback=white, boxrule=0.4pt, left=3pt, right=3pt, top=3pt, bottom=3pt]
\scriptsize
``[The model reconstructs the chapter unit by unit, confirming each draft mark before
moving on.] ...the narrator as a senior at Cesar Chavez High School, named Marcus
Yallow: that confirms a new sentence boundary there. I am now checking the next segment
around `w1n5t0n' and the explanation of how Winston is pronounced versus the letters and
numbers. [...] `Marcus Yallow' is the name called over the PA, and then a new sentence
begins noting that the PA system itself is not very good to begin with. [...] [It then
verifies the remaining units one at a time, quoting each in Arabic with its token range
and ticking it off: (473--485), (486--499), (500--513), (514--531)...]''
\end{tcolorbox}

Framing sentence for the paper: "The platform records the model's internal reasoning; we reproduce two
abridged excerpts from the rerun. The first shows the full pattern of a decision: a candidate edit
proposed, endorsed by two laws, rejected against a third, relocated one token, and verified by density."
Note for the writer: these are internal reasoning, so label them as such and keep the RECORDED arguments
of section R as the primary exhibit; the raw excerpts corroborate that the recorded arguments reflect the
actual process rather than post-hoc summaries.

## T. NoPnx-PA provenance repair (2026-08-20)
26 of 262 NoPnx-PA released-test verdicts had been written on Aug 5 by launchers pointing at the
pre-retrain (retrain3) pair, before the final retrain4 doctrines existed; the Aug 9-10 exam's RESUME
clause kept them. Those 26 were voided (exam_out_void_retrain3pair/) and re-adjudicated with the
released retrain4 pair (launcher released: exam_doctrine_nopa_redo26.js). Corrected row:
87.05 -> 92.59, delta +5.54 (was 92.28/+5.23); docs 166/35/61; bounds +547/-722; edits 1,269
(0 rejected); t p=8.2e-13, CI [4.1, 7.0]; doc-0shot +3.88 (t 2.4e-07, Wilcoxon 7.0e-04).
Grand strict-edit total across tracks: 3,258/3,258 exact. All other tracks verified clean
(hash + timestamp + script-path evidence). Under the old pair the 26 docs scored +4.27; under the
released pair the full-262 delta ROSE. Every number above is enforced by verify_paper.py (200 checks).
