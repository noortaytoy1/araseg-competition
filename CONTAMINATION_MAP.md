# ENSAR Contamination Map — final audit of record

Complete data-lineage and contamination accounting for every submitted and reported
number. Compiled 2026-08-21 from primary sources only: file hashes, filesystem
timestamps, preserved workflow/agent run records, and session transcripts. Every
mechanical claim below is either enforced by `paper/verify_paper.py` (224 checks, run
on every build) or carries its evidence pointer.

## 1. The contamination event, and its remediation (the record, honestly)

- **~Jul 27 (wave 2):** during the first jury campaign, wave-2 jury training material
  included documents from dev/test. This is the original contamination.
- **Aug 3, afternoon:** discovered and confronted. Adversarial audits launched
  (`audit_compliance`, `nopnx-np-purity-audit`, `pnx-pa-purity-audit` — released in
  `jury/prompts/raw/`).
- **Aug 3, 18:41–18:46:** the purge. Contaminated doctrine entries struck and
  quarantined (`scratch_exo/np_train_work/quarantine/` preserves pre-cleanse
  snapshots and the removed text verbatim); cleansed doctrines written 18:46.
  Machine-checked today: **0 of the 44 quarantined lines appear in any released
  doctrine.**
- **Aug 3, evening → Aug 4:** clean-room protocol by user order: dev/test data files
  deleted from the working tree; the only gold copy sealed in
  `araseg_heldout_LOCKED.zip` (opened in-memory, gate-only, never in any agent);
  fresh training/output directories; sealed test probes hidden from agents;
  juries retrained from scratch for the final submissions.
- **Aug 20 (post-hoc, paper phase):** one further defect found and fixed — 26 of 262
  NoPnx-PA released-test ablation verdicts had been produced by a pre-retrain
  launcher (RESUME leftover). Voided (`exam_out_void_retrain3pair/`), re-adjudicated
  with the released pair, all numbers recomputed and republished. Drift guard added
  (verifier §10: released == counted, byte-projected, all four tracks).

## 2. Blind submissions — what was actually uploaded, per track

| Track | Official | Submission artifact | System | Jury training data |
|---|---|---|---|---|
| Pnx-PA | **95.30, 1st** | shipped Aug 2 (pa-group-exam sittings) | ensemble draft + shared jury pair | de-punctuated train corpus (wave era, cleansed Aug 3) + punctuated bridge rounds; train split only |
| Pnx-NP | **92.0, 1st** | `subs_blind/prediction_NP.zip` (Jul 21) | **ensemble draft alone — no jury ever touched this submission** (the deadline pnxnp jury exam produced 0 gated verdicts) | n/a |
| NoPnx-NP | **89.1, 1st** | `prediction_NoPnx-NP.retrain3.zip` (Aug 4 12:50) | ensemble draft + clean-room jury pair | curated 39 train docs (20+19), census matched to blind input genres; train split only |
| NoPnx-PA | 90.0, not 1st | retrain3/nopa zips (Aug 4 14:54) | ensemble draft + clean-room jury pair | ~60 train docs, rebuilt in the final hours; train split only |

Earlier superseded uploads (jury1/2/3 era, Jul 21–31, including the pre-cleanse
NoPnx-NP 89.5) were replaced by the entries above and count for nothing.

## 3. Paper ablation (released test, 262 docs/track) — pair lineage

| Track | Doctrine pair (released) | Trained on | Blind-set relation |
|---|---|---|---|
| NoPnx-NP | `jury/doctrines/NoPnx-NP` (= retrain3/nonp, hash-verified) | 39 train docs | same pair as the 89.1 submission |
| NoPnx-PA | `jury/doctrines/NoPnx-PA` (= retrain4/nopa) | all 174 train docs (87+87), post-deadline | NOT the submitted pair (disclosed in paper) |
| Pnx-NP | `jury/doctrines/Pnx-NP-PA` (= np_train_work, cleansed) | wave-era de-punctuated corpus + punctuated rounds | its blind submission was ensemble-only; jury evaluated on released test only |
| Pnx-PA | same shared pair | same | same pair as the 95.30 submission |

The sharing of one pair across both punctuated tracks was a session-level decision
under deadline pressure, not an explicit user order; the paper's Limitations disclose
it and treat the two rows as correlated measurements.

## 4. Jury training purity — machine-checked on every build (verifier §11)

- Every graded training attempt on disk, every generation including superseded ones
  (442 answer files): **train split only. 0 dev, 0 test.**
- Every doc id cited in any released doctrine (271 citations): **train only.**
- Every Arabic 8-gram quoted in any released doctrine (~20k shingles vs 274k
  dev/test shingles): **zero passages that exist only in dev/test.**
- Training packets contain no ensemble marks; every attempt is a from-scratch
  boundary list (no edit-format answers). The draft appears only at adjudication.
- Doctrines froze before their evaluations (timestamps verified); dev gated
  selection only, never fed content back.

## 5. Evaluation isolation

- Exam packets: text only, byte-verified builder released; **0 packets contain labels.**
- File-access guard (`.claude/hooks/jury_guard.py`) released; transcript audits (275
  zero-shot transcripts alone): no access beyond permitted paths, no gold vector in
  any agent context.
- The blind leaderboard labels were never in our possession at any time.
- Curriculum design note (disclosed in Limitations): the NoPnx-NP curriculum's genre
  mix was matched to a classification of the **blind documents' input texts**
  (`blind_genre.json`, Jul 21). Inputs only; no labels of any split involved; the
  juries were never told the blind composition.

## 6. Standing corrections still owed to the paper (author's hand)

1. RESOLVED 2026-08-21: §4.2 rewritten by the author (per-pair training counts, the
   despawn design, the Reset wording); the 39-document curriculum is disclosed in the
   Official-results paragraph.
2. Ablation-setup caveat: "the NP, PA, and NoPnx-NP doctrine rows use the submitted
   juries" → NP's blind submission carried no jury; its row evaluates the shared pair
   on released test only.
3. Limitations: one sentence on the blind-input-genre-matched curriculum (§5 above).

## 7. What was lost, for completeness

Nothing that was ever completed. Three full-corpus NoPnx-NP training attempts on
deadline night (npnpR ~7 min; retrain3-all 55 s; one curated relaunch) died before
writing any doctrine; their run records are preserved and released. One temp
scratchpad (npnpR session) was cleaned by the platform, empty of finished artifacts.
The wave-era full-corpus training's products are intact, cleansed, and released as
the punctuated pair.

## 8. Post-release audit findings (2026-08-21, independent fresh-session audit)

- **Exp-2 examples file colophon overlap (confirmed, zero impact on reported numbers):**
  the killed gold-examples arm's `examples_j0.md` contains the Hindawi publisher
  colophon with gold boundaries; the identical span (43 Arabic tokens, 10 boundaries)
  recurs verbatim in test doc `doc_92e8c7876890`. The arm was killed 2026-08-19 and
  reports nothing; no verdict for the affected doc exists; the files were unreleased
  scratch; the entire arm (examples files and its 60 abandoned verdicts) was DELETED
  at the user's order on 2026-08-21, after this finding was recorded. Build-time leak checks verified doc ids, not
  shared boilerplate content — future example-based arms must also shingle-check text.
- **Design weaknesses acknowledged** (real, none crossing a gold wall): agent
  containment was honor-system prompts plus a late-added guard hook rather than the
  API runner's structural allowlist from day one; the containment clause "answer keys
  exist elsewhere in this repository" advertises the keys' existence; some
  gold-derived score files landed mid-run in directories agents could name; the
  `exam_out_shipped/` -> `exam_out/` rename left a provenance hole (reconstructed in
  this map); drafts are dev-tuned (disclosed in the paper).
- One fresh-session claim was wrong: the paper's Limitations DOES disclose the shared
  punctuated-track jury pair ("one shared jury pair ... correlated measurements").

## 9. The naming trap (root cause of repeated confusion; resolved 2026-08-22)

The shared punctuated pair's doctrines carry internal headers naming "NoPnx-NP" and
cumulative counts "10 batches, 141 docs" / "123 graded documents": that pair's first
phase trained on the DE-PUNCTUATED form and the jurors titled their own files by the
format, not the track. Two different pairs therefore answer to the name "NoPnx-NP":
the wave-trained shared pair (internal header; 133/121 distinct docs on disk, 85
overlapping — NOT disjoint halves) and the track-folder pair (curated 39, the 89.1
submission). Consequences applied: the paper's 4.2 counts now state 141/123 with
partial overlap for the shared pair (disjointness holds only for the NoPnx pairs);
the supplementary README carries the naming note; the lessons records the shared
pair's doctrines cite are included in the supplementary zip. Doctrine files remain
byte-untouched — they are the juries' writing and the release's evidence.
