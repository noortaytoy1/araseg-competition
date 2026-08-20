# Experiments in progress for the ENSAR paper (AraSeg 2026) — handoff for the reviewing session
Written 2026-08-19 ~18:00 EDT. Camera-ready deadline: 2026-08-22. Repo: this directory.
The paper is `paper/ensar_araseg.tex`; every number in it is recomputed by `python paper/verify_paper.py`.

## 0. Why these experiments exist
The paper's claim: two LLM "juries" (Claude Opus 5, maximum reasoning effort) whose segmentation
policy was LEARNED AS TEXT from graded errors on the TRAIN split add +2.80 F1 on average over a
5-encoder ensemble (released test split, n=262/track):
NoPnx-NP 84.64->89.49 (+4.85) | NoPnx-PA 87.05->92.28 (+5.23) | Pnx-NP 92.83->93.61 (+0.77) | Pnx-PA 94.56->94.92 (+0.36)
Two reviewer attacks the current draft cannot yet answer with DATA:
 (A) "The gain is just Opus 5 being a big model, not your training loop."
 (B) "Even if annotator conventions matter, you didn't need the loop; just show the model gold examples."
Plus (C): "Your model is non-deterministic; would you get the same result again?"
The three experiments below answer A, B, C respectively. All use the SAME test packets, the SAME
ensemble draft rows, the SAME two-jury debate prompt structure, the SAME unanimity rule (only edits both
juries endorse are applied), and the SAME model: Claude Opus 5 at maximum effort. NO haiku anywhere.

## 1. Experiment 1 — ZERO-SHOT control (answers A)
What: the two juries with NO doctrine step at all. Prompt tells them plainly: "Neither has been trained
on this corpus and neither holds any written policy or examples from it; each reasons only from its own
knowledge of Arabic and from the text in front of it." They see the ensemble's draft marks and edit them.
Tracks: all four (NoPnx-NP first, then NoPnx-PA, NP, PA), 262 docs each, 4 docs per sitting, 4 lanes.
Outputs: scratch_exo/papereval/<track>/exam_out_zeroshot/<doc_id>.json
Driver: workflow task wduuwehxw (run wf_45ec1845-2ac); script papereval-zeroshot-control-wf_45ec1845-2ac.js
Status at handoff: NoPnx-NP ~80/262 banked; other tracks queued behind it automatically.
Expected reading: if zero-shot lands near the ensemble (84.64 on NoPnx-NP) while doctrine gives 89.49,
the gap is the learned policy's contribution; it also shows the model is NOT recalling these public
texts' original punctuation from pretraining (if it were, it would not need the doctrine to score high).

## 2. Experiment 2 — GOLD EXAMPLES instead of doctrine (answers B) — NoPnx-NP only
What: same exam, but each jury's file is 8 TRAINING documents with the gold boundaries drawn in (‖ after
each sentence-final token, punctuation/paragraphs removed exactly as the jury will see them), instead of
its learned doctrine. Jury 0 gets the first 8 docs of ITS OWN training curriculum, Jury 1 the first 8 of
its own (fixed rule, no cherry-picking; train split only; leak-checked: 0 dev/test ids).
Example files: scratch_exo/papereval/examples_nonp/examples_j0.md (59,098 chars), examples_j1.md (51,205)
 vs the real doctrines retrain3/nonp/doctrine_j0.md (38,763) and doctrine_j1.md (46,139): the examples are
 LONGER, so volume of gold is not the doctrine's advantage.
Outputs: scratch_exo/papereval/NoPnx-NP/exam_out_examples/<doc_id>.json (262 docs)
Driver: workflow task wm3s30re2 (run wf_22036d85-2d5); script nonp-exp3-exp2-now-wf_22036d85-2d5.js
 (runs Exp 3 first, then Exp 2, in parallel with Exp 1).
Expected reading: if doctrine > examples at matched length, the loop extracted rules raw examples don't
carry (the paper's thesis). If examples ~= doctrine, say so honestly.

## 3. Experiment 3 — RERUN of the real doctrine setup (answers C) — NoPnx-NP, first 50 docs
What: identical to the published ablation (retrain3/nonp doctrines, same prompt), run a SECOND time on the
first 50 documents of the manifest order. Compare to the published verdicts on those same 50 docs.
Outputs: scratch_exo/papereval/NoPnx-NP/exam_out_rerun/<doc_id>.json (50 docs)
Reading: report "two runs differ by X F1 on 50 documents" in one sentence.

## 4. Scoring rules (binding; Noor's order 2026-08-19)
- STRICT edit application: an edit is applied ONLY if the jury's index holds exactly the token it named.
  No ±2 word-snap, ever. (The published ablation was re-verified strict: 3,245/3,245 edits exact, numbers
  unchanged to the second decimal. Training attempts: 19,854 boundaries, 1 non-exact.)
- Gold is read IN MEMORY from C:/Users/pc/araseg_heldout_LOCKED.zip by the main loop only; never by agents.
- Scorer for the new experiments: write a strict variant of scratch_exo/papereval/score_ablation.py
  (same normalization: final token = 1; NL rules for PA tracks; different exam_out_* dir per experiment).
- Same-doc comparisons for Exp 3: score both runs on exactly the 50 docs, report both means and the diff.

## 5. Access guard (installed 2026-08-19)
.claude/settings.json + .claude/hooks/jury_guard.py: a PreToolUse hook that hard-denies any jury agent
(detected by "exam sitting" + "UNRELIABLE CHEAT SHEET" in its transcript) file access outside: the test
packets (papereval/<track>/docs/), its output dir (exam_out_*), the Exp-2 example files, the three real
doctrine pairs, and its own temp scratchpad. Shell access to zip/data/doctrine/git/python is denied.
Main-loop (non-jury) sessions are unaffected. Self-tested: blocks zip, train/dev/test jsonl, repo files;
allows packets, verdicts, own notes.
Audit of the zero-shot run at handoff: 21 agents, files touched = packets, own verdicts, own scratch notes
ONLY; 0 doctrine/examples/train/gold paths in any transcript.

## 6. What to do when results land (for the reviewing session)
1. Re-run the access audit on each finished exam_out_* dir's transcripts (paths above) before scoring.
2. Score strictly (rule 4). Add rows to Table 2 of the paper: "+ juries, zero-shot" (four tracks),
   "+ juries, gold examples" (NoPnx-NP), and a one-sentence rerun-variance note (Exp 3).
3. Add the new checks to paper/verify_paper.py so every new number is recomputed on build.
4. Update the "Is it the policy or the model?" paragraph and the Limitations sentence with the data;
   remove the "we did not run the juries without their learned policies" admission once Exp 1 is in.
5. If anything could not finish by Aug 22, say so explicitly in the paper (name the missing experiment).

## 7. Things that are NOT part of the paper (do not use)
- scratch_exo/papereval/NP/exam_out_void_retrain3doctrines: voided NP run with incomplete doctrines.
- Any earlier "stub doctrine" no-policy run: deleted; superseded by the pure zero-shot design above.

## 8. Note for any FUTURE jury run (added 2026-08-20)
The platform auto-recalls session-memory notes into agent contexts (verified: a curriculum one-liner
reached 16 zero-shot agents; no note with numbers/conventions ever appeared). Before launching any future
jury run, move the session memory directory aside temporarily, or audit recalled content in transcripts
afterward as done here.
