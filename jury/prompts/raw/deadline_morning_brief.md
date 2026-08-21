# FINAL PROMPT FOR THE NEW SESSION — paste this ENTIRE file as the first message
# [Released as-run, Aug 4 2026. See INDEX.md for the correction of record regarding
#  the "census-balanced from TRAIN metadata only" clause.]

Nothing here is left to interpretation: every document list is literal, every script is complete code you write to disk VERBATIM, every agent prompt is fully written out. If you are about to do ANYTHING not written here, stop and ask Noor first.

Repository: C:\Users\pc\Downloads\evolving-vlm-46\arabic-sentence-segmentation — work only there.

## DEFINITIONS
- ENSEMBLE = already-trained neural encoder models. You never touch, run, or retrain them. Their saved blind outputs are the DRAFT rows.
- JURY = an LLM judge persona created by prompting; its entire knowledge is one doctrine text file it writes itself during training. There are exactly 2 juries per track, never more.
- The word 'voters' is BANNED. Say ENSEMBLE or JURY.
- Track Pnx-PA is DONE and submitted — never touch anything of it.

## THE JOB
Three tracks, in this order: nonp (NoPnx-NP), then pnxnp (Pnx-NP, data files prefixed 'NP'), then nopa (NoPnx-PA). Per track: Phase 1 train 2 fresh juries on TRAIN documents only; Phase 2 the trained juries edit the DRAFT rows on the blind exam packet; Phase 3 gate on sealed probes, assemble, zip. Noor uploads every zip — deadline 3:00 PM today.

## ABSOLUTE RULES
- R1: Juries learn ONLY from train documents + train gold. Dev/test/blind-derived ANYTHING must never appear in any agent's context: no documents, no gold, no exemplars, no statistics, no census/composition numbers, no evaluation results, no per-doc hints, no 'the blind contains X'. Gate results die with you — never fed back to juries.
- R2: The ONLY copy of dev/test gold is C:\Users\pc\araseg_heldout_LOCKED.zip. Open it IN MEMORY via python zipfile, from the main loop only, in Phase 3 only. Never extract it, never open it in an agent, never quote its contents.
- R3: Every jury/exam agent: model opus, effort max, prompts say 'no limit on your reasoning'. Never instruct brevity or depth limits.
- R4: Every answer/verdict is written to its own file the moment it is finished. Resume after any crash = skip files that already exist.
- R5: Never write into subs_blind/ or data/. All new files go under scratch_exo/retrain3/.
- R6: Report every grading result and every gate result to Noor verbatim. Announce the estimated token cost BEFORE every launch. Announce every action before taking it.

## SCHEDULE AND EXACT AGENT COUNTS (deadline 15:00 today)
- Phase 1: EXACTLY 2 training agents per track (one per jury), launched in parallel, background. Each processes its 19-20 documents in batches of five (final batch may be short), sequentially within the agent. ~1.5-2h, ~1.5-2M tokens per track.
- Phase 2: EXACTLY 6 concurrent sittings for nonp and pnxnp, EXACTLY 4 for nopa. Never run two tracks' Phase 2 at the same time. ~40-45k tokens/doc => nonp ~9-10M, pnxnp ~9-10M, nopa ~4-4.5M; ~2-2.5h each at those lane counts.
- PIPELINE OVERLAP IS MANDATORY: while a track's Phase 2 runs, launch the NEXT track's Phase 1.
- Target timeline: start ~02:00 -> nonp P1 done ~04:00 -> nonp P2 + pnxnp P1 -> nonp ZIP ~06:30 -> pnxnp P2 + nopa P1 -> pnxnp ZIP ~09:00 -> nopa P2 -> nopa ZIP ~11:30. That leaves 3+ hours of slack; one full restart per track is absorbable.

## WHAT ALREADY EXISTS (never rebuild, never modify)
- data/NoPnx-NP_train.jsonl, data/NoPnx-PA_train.jsonl, data/NP_train.jsonl — train docs, lines of {doc_id, tokens, labels}. In NoPnx-PA, paragraph breaks are literal newline tokens in 'tokens'.
- data/NoPnx-NP_blind.jsonl, data/NoPnx-PA_blind.jsonl, data/NP_blind.jsonl — blind docs {doc_id, tokens}, no labels.
- subs_blind/prediction_NoPnx-NP, prediction_NoPnx-PA, prediction_NP — the DRAFT rows (header 'Document ID,Prediction', rows 'doc_id,0101...'). READ-ONLY.
- Sealed exam packets (docs/{doc_id}.json = {doc_id, text} with ¶ draft marks and ⟨i⟩ anchors; manifest.json = {order, probes, sittings}; verified: no labels anywhere): scratch_exo/np_rebuild — 222 (212 blind + 10 probes) docs, output dir scratch_exo/retrain3/nonp/exam_out/; scratch_exo/npa_rebuild — 108 (100 blind + 8 probes) docs, output dir scratch_exo/retrain3/nopa/exam_out/; scratch_exo/np2_rebuild — 221 (212 blind + 9 probes) docs, output dir scratch_exo/retrain3/pnxnp/exam_out/. Probes are sealed test docs mixed in for the engineer-side gate — AGENTS MUST NEVER BE TOLD PROBES EXIST.
- C:\Users\pc\araseg_heldout_LOCKED.zip — dev/test gold, gate only (R2). Inner paths used: heldout_data/NoPnx-NP_test.jsonl, heldout_data/NoPnx-PA_test.jsonl, heldout_data/NP_test.jsonl.
- Old doctrines/answers from previous sessions exist in scratch_exo/np_train_work and elsewhere — do NOT reuse or read them; the whole point is from-scratch juries.

## PHASE 0 — write these three files to disk EXACTLY as printed, then run the first once
Write with LF newlines. Do not edit a character.

### File 1: scratch_exo/retrain3_setup.py — then run python scratch_exo/retrain3_setup.py

### File 2: scratch_exo/retrain3/grade3.py

### File 3: scratch_exo/retrain3/gate3.py

[The three script bodies are on disk at exactly those paths, byte-verified against
this brief's originals on Aug 4; they are not duplicated here.]

## PHASE 1 — TRAINING (per track: exactly 2 agents, parallel, opus, effort max)
The complete agent prompt is below. Make EXACTLY these substitutions and no other change:
{TRACK} -> nonp | nopa | pnxnp;  {J} -> 0 | 1;  {FORMAT} -> that track's paragraph from the FORMAT TABLE;  {LIST} -> that track+jury's literal list from DOCUMENT LISTS.

### AGENT PROMPT (verbatim except the four substitutions)
"""
You are Jury {J}, a brand-new Arabic sentence-segmentation judge. You start with zero knowledge of this corpus's conventions — everything you will ever know comes from your own graded attempts on TRAINING documents. Work in C:\Users\pc\Downloads\evolving-vlm-46\arabic-sentence-segmentation.
The ONLY files you may touch: the practice documents listed below, your own answer files, the grader's mistakes file named below, and your own doctrine file scratch_exo/retrain3/{TRACK}/doctrine_j{J}.md. Touching ANY other file voids the training — answer keys exist elsewhere in this repository.
FORMAT: {FORMAT}
YOUR DOCUMENTS, in batches of five, in this order: {LIST}
PER BATCH:
1. SEGMENT strictly one document at a time from scratch_exo/retrain3/{TRACK}/train_docs/<doc_id>.json. No limit on your reasoning — first work out what the text IS (genre, register, structure) and what its natural sentence unit is, then walk it and argue every uncertain site in writing. Write your answer to scratch_exo/retrain3/{TRACK}/ans/j{J}/<doc_id>.json as {"doc_id":"...","identity":"what this text is","law":"its sentence law as you read it","reasoning":"your argument at the hard sites","boundaries":[{"i":<index of the LAST token of a sentence>,"w":"<that token copied from the text>","c":"hi|med|lo"}]} the moment it is done, before opening the next document.
2. GRADE: run  python scratch_exo/retrain3/grade3.py --track {TRACK} --juror {J} --only <the five ids comma-separated>  and read scratch_exo/retrain3/{TRACK}/mistakes_j{J}.txt in full.
3. LEARN: for every FALSE CUT and every MISSED, find the RULE behind it — what do these annotators actually do — and write it into your doctrine file in your own words, filed by text type, with your own error as the example. When a new correction contradicts an earlier entry, fix the earlier entry and say so. The doctrine is cumulative — never delete knowledge.
4. Take your updated doctrine into the next batch.
If you hit a usage limit or crash, everything is already saved; on restart, skip documents whose answer file exists. When all four batches are graded and absorbed, return your per-batch mean F1 and 4 sentences on your doctrine's core laws. No web. No PowerShell except the grader command given above.
"""

### FORMAT TABLE
- nonp: Punctuation REMOVED and paragraph marks REMOVED — plain words only. The final token of a document always ends a sentence. ⟨i⟩ means the next token is token number i (an index anchor, not part of the text).
- nopa: Punctuation REMOVED; paragraph structure KEPT as literal <NL> tokens (each <NL> has its own index). The ¶ on the token immediately before every <NL> and on the final token is ALWAYS correct and must never be removed; an <NL> token itself NEVER carries a boundary. Everything inside paragraphs is yours to judge. ⟨i⟩ = index anchor, not part of the text.
- pnxnp: Punctuation KEPT — punctuation marks are ordinary tokens with their own indices, and a sentence usually ends ON its punctuation mark when one is present. NO <NL> tokens. The final token always ends a sentence. ⟨i⟩ = index anchor, not part of the text.

### DOCUMENT LISTS (literal — use exactly these, in this order)
- nonp Jury 0 (20 docs): doc_3f6c5f1e305a, doc_44ee993e046e, doc_66c98646b9a9, doc_1d513dc2c22a, doc_c567a2257203, doc_47764cf032d7, doc_226fc860e550, doc_eb278f2980be, doc_54f919517225, doc_405d5fba81e0, doc_5b0751f40176, doc_7bc0656a48ce, doc_32908af53b40, doc_501500835788, doc_70d28c9dfebb, doc_c188bbc9702e, doc_86d3b22a1be5, doc_1fa12697d487, doc_e1435023282b, doc_9e0419ed749d
- nonp Jury 1 (19 docs): doc_8b2d52902b9b, doc_201b37ae8a22, doc_de12d5da4854, doc_16f2b3eccb91, doc_3272f207e8c9, doc_32b866eb5dc4, doc_5e12ebe2976f, doc_9d15d9d5d637, doc_8921b794ce19, doc_56781db1019f, doc_0a957c052364, doc_099def456da2, doc_51fccce927eb, doc_20dcf69f9905, doc_25ed373f8aeb, doc_cb57e21e8421, doc_b9371c2fbb78, doc_27a114d4e376, doc_361233315d24
- nopa Jury 0 (19 docs): doc_3f6c5f1e305a, doc_44ee993e046e, doc_201b37ae8a22, doc_1d513dc2c22a, doc_c567a2257203, doc_47764cf032d7, doc_226fc860e550, doc_eb278f2980be, doc_54f919517225, doc_56781db1019f, doc_5b0751f40176, doc_0a957c052364, doc_32908af53b40, doc_501500835788, doc_70d28c9dfebb, doc_c188bbc9702e, doc_86d3b22a1be5, doc_b9371c2fbb78, doc_e1435023282b
- nopa Jury 1 (19 docs): doc_8b2d52902b9b, doc_66c98646b9a9, doc_de12d5da4854, doc_16f2b3eccb91, doc_3272f207e8c9, doc_32b866eb5dc4, doc_5e12ebe2976f, doc_9d15d9d5d637, doc_8921b794ce19, doc_405d5fba81e0, doc_7bc0656a48ce, doc_78b2ed132e2f, doc_51fccce927eb, doc_20dcf69f9905, doc_25ed373f8aeb, doc_cb57e21e8421, doc_1fa12697d487, doc_27a114d4e376, doc_361233315d24
- pnxnp Jury 0 (20 docs): doc_3f6c5f1e305a, doc_44ee993e046e, doc_de12d5da4854, doc_c567a2257203, doc_46717e5891d4, doc_3272f207e8c9, doc_47764cf032d7, doc_54f919517225, doc_9d15d9d5d637, doc_8921b794ce19, doc_5b0751f40176, doc_0a957c052364, doc_32908af53b40, doc_501500835788, doc_1c49614cd6bc, doc_c188bbc9702e, doc_86d3b22a1be5, doc_1fa12697d487, doc_e1435023282b, doc_9e0419ed749d
- pnxnp Jury 1 (19 docs): doc_8b2d52902b9b, doc_201b37ae8a22, doc_66c98646b9a9, doc_1d513dc2c22a, doc_16f2b3eccb91, doc_226fc860e550, doc_5e12ebe2976f, doc_eb278f2980be, doc_56781db1019f, doc_405d5fba81e0, doc_7bc0656a48ce, doc_a045e64631c7, doc_51fccce927eb, doc_25ed373f8aeb, doc_153372a232d6, doc_cb57e21e8421, doc_b9371c2fbb78, doc_27a114d4e376, doc_361233315d24

These lists were census-balanced from TRAIN metadata only (expository/narrative/exercise/conversational/poetry/hadith/legal/quran), split disjointly between the two juries. Do not swap, drop, or add documents.

## PHASE 2 — BLIND PREDICTION (one Workflow per track, only after BOTH its doctrines exist)
[Workflow driver as released in this directory's exam scripts; PACKET TABLE: nonp = scratch_exo/np_rebuild LANES 6; nopa = scratch_exo/npa_rebuild LANES 4; pnxnp = scratch_exo/np2_rebuild LANES 6. NEVER-TOUCH per the FORMAT TABLE's track laws.]

## PHASE 3 — GATE + ASSEMBLE + ZIP (main loop ONLY, never inside an agent)
1. python scratch_exo/retrain3/gate3.py --track <track>  — paste FULL output to Noor.
2. SHIP RULE: jury probe mean must BEAT the draft probe mean, else stop and tell Noor.
3. If SHIP=YES: same command with --write; NOOR uploads; you never submit anything.

## HONESTY
Report gate results verbatim. Ship nothing that loses to the draft. Disclose every anomaly, every skipped document, every deviation, at the moment it happens. Every agent's file access is logged in its transcript; Noor audits.
