# REPRODUCTION HANDOFF — paste this file as the first message of a fresh Claude session

You are a fresh session. Your ONLY job: independently re-derive the paper's numbers
from this repository and report each one PASS/FAIL against the expected values below.
You change nothing. You trust nothing you "remember" — every claim you make must come
from a file you opened or a command you ran in THIS session.

Repository: C:\Users\pc\Downloads\evolving-vlm-46\arabic-sentence-segmentation
Rules: read-only except your own scratch. Do not modify the repo, the paper, or
anything under scratch_exo/. Do not "fix" anything you think is wrong — report it.
Report results as a table: check | expected | got | PASS/FAIL.

## Step 1 — the full verifier (everything at once)
    python paper/verify_paper.py
Expected: **224/224 checks PASSED**. This recomputes, from data: corpus statistics,
training hyperparameters (from code), the complete 2x2 ablation for all four tracks,
error analysis, prompt-record integrity, jury training purity, and released==counted
verdict identity. If ANY check fails, stop and report the exact FAIL lines.

## Step 2 — the ablation rows, independently (do not reuse the verifier's code)
Write your own scorer from this spec and run it per track:
- Gold: heldout_data/<track>_test.jsonl inside C:\Users\pc\araseg_heldout_LOCKED.zip
  (open with python zipfile, in memory).
- Draft rows: scratch_exo/papereval/<track>/draft_rows.json ("rows": doc_id -> "0101...").
- Verdicts: jury/verdicts_test/<track>/<doc_id>.json ({doc_id, add, remove}).
- STRICT application: an edit applies ONLY if tokens[i] == w exactly. add sets label
  1 at i (never on a "\n" token); remove sets 0 (never the final token; in PA-format
  tracks never a token adjacent to "\n"). PA-format normalization (NoPnx-PA, PA):
  final token = 1, "\n" tokens = 0, token before "\n" = 1, applied to draft and result.
- Score: macro F1 over documents (boundary class), organizers' definition.
Expected (ensemble -> jury):
- NoPnx-NP: 84.64 -> 89.49 (delta +4.85)
- NoPnx-PA: 87.05 -> 92.59 (delta +5.54)
- NP:       92.83 -> 93.61 (delta +0.77)
- PA:       94.56 -> 94.92 (delta +0.36)

## Step 3 — packet integrity
    python jury/build_packets.py --track NoPnx-NP --split test --data <fetch per README> --draft jury/draft_rows/NoPnx-NP.json --out <scratch>/kit
Then byte-compare a sample of <scratch>/kit/docs/*.json against
scratch_exo/papereval/NoPnx-NP/docs/. Expected: byte-identical; no packet contains labels.

## Step 4 — provenance spot-checks (hashes)
- md5(jury/doctrines/NoPnx-NP/doctrine_jury{0,1}.md) == md5(scratch_exo/retrain3/nonp/doctrine_j{0,1}.md)
- md5(jury/doctrines/NoPnx-PA/*) == md5(scratch_exo/retrain4/nopa/*)
- md5(jury/doctrines/Pnx-NP-PA/*) == md5(scratch_exo/np_train_work/doctrine_np_juror{0,1}.md)
Expected: all equal.

## Step 5 — read CONTAMINATION_MAP.md and confirm its §4 purity claims yourself
Re-run at least: (a) every id in scratch_exo/retrain3/nonp/ans and
scratch_exo/retrain4/nopa/ans is in data/NoPnx-NP_train.jsonl's id set; (b) no doc id
cited in any jury/doctrines/**.md is outside that train id set.

## Known confusions that burned previous sessions — read before concluding anything
1. Two different pairs have "NoPnx-NP format" history: the RELEASED NoPnx-NP pair
   (curated 39, retrain3) and the wave-era pair (full-corpus de-punctuated, released
   as Pnx-NP-PA). Do not conflate them.
2. The blind Pnx-NP submission was the ensemble alone; the NP ablation row is a
   released-test experiment with the shared pair.
3. "trdev"/"devheld" files in the locked zip are historical experiment splits — the
   official splits are heldout_data/<track>_{dev,test}.jsonl.
4. scratch_exo/ is unreleased working history (append-only evidence). jury/ is the
   release. The paper's numbers must reproduce from the release + the locked zip.
5. Session memories and summaries are NOT evidence. Files are evidence.

Deliverable: the PASS/FAIL table, any discrepancy verbatim, and one sentence:
"every paper number re-derived" or exactly what didn't.
