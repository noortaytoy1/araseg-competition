# Jury layer: learned segmentation policies, prompts, and verdicts

This folder is the release of the reasoning stage described in *ENSAR at AraSeg 2026: An
Ensemble of Encoders With Agentic Error Refinement*. The encoder ensemble lives in `../src`.
Everything here is text: policies the juries wrote, the prompts that produced them, their
blind training attempts, and the edits they applied on the released test split.

## Layout

| Path | What it is |
|---|---|
| `prompts/train_nopa_rolling.js` | The exact training prompt (rolling batches of five, grade, absorb) as run for the NoPnx-PA juries. Model: Claude Opus 5, maximum reasoning effort. |
| `prompts/exam_*.js` | The exact adjudication prompts per track. Two juries argue each document; only edits both endorse are applied. |
| `doctrines/<track>/doctrine_jury{0,1}.md` | The learned policies, verbatim. Every entry cites the batch and the graded errors that produced it; superseded rules are struck through, not deleted. `Pnx-NP-PA` holds the pair used for both punctuated tracks. |
| `graders/grade_training_batch.py` | The training grader: scores a jury's attempt against TRAIN gold only, writes the mistakes report the jury reads. Boundaries match within a two-token tolerance against an identical surface form. |
| `training_answers/NoPnx-PA/` | All 87 + 87 blind training attempts of the NoPnx-PA juries, plus `curriculum_split.json` (the disjoint halves and their order). Re-scoring these in order reproduces the learning curve in the paper. |
| `verdicts_test/<track>/` | The juries' edits on the 262-document released test split (`add`/`remove` token positions with surface forms) and the per-track `ablation_result.json`. |
| `eval/score_ablation.py` | Applies verdicts to the ensemble draft and scores both arms against gold. |

## What is deliberately not here

- No dev, test, or blind documents or labels. Test gold is read by `score_ablation.py` from a
  local archive that is not distributed; obtain the released splits from the task organizers.
- Verdict files have had their `argued` field removed for release, because juries quote the
  test passage they are arguing about. The `add`/`remove` edits, which are what the scores
  depend on, are untouched.
- Exam packets (test text with the ensemble's draft marks) are not distributed for the same
  reason; `score_ablation.py` rebuilds the draft from cached voter probabilities.

## Provenance

- NoPnx-NP: `doctrines/NoPnx-NP` are the submitted juries.
- NoPnx-PA: `doctrines/NoPnx-PA` are the post-deadline full-corpus retrain (the submitted pair
  had reached 60 of 174 training documents at the deadline). Their complete training run is
  in `training_answers/`.
- Pnx-NP and Pnx-PA: `doctrines/Pnx-NP-PA` are the submitted juries.

Local paths in the scripts are replaced by `<REPO>`, `<HELDOUT_ZIP>` and `<WORKFLOW_DIR>`.
