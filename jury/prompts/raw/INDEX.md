# Raw launcher scripts of record — every prompt, verbatim

These are the exact orchestration scripts, byte-for-byte as invoked (local absolute
paths included), recovered from the run records. Each embeds the full prompt text its
agents received. Sittings ran on `claude-opus-5` at maximum reasoning effort; no
sampling parameter is set anywhere (provider defaults). Cleaned path-parameterized
templates of the five main forms live one directory up in `jury/prompts/`; the
one-paste replication orchestrator is `jury/orchestrator_prompt.txt`.

Successive launchers of one stage (initial / resume / finish) rephrase connective
wording without changing any rule; every variant is here.

## Training — NoPnx-NP pair (curated 39-doc curriculum → `jury/doctrines/NoPnx-NP`)
| file | role |
|---|---|
| train_nonp_retrain3.js | the curriculum trainer (batches of five, graded per batch) |
| train_nonp_retrain3_learn_final.js | final consolidation pass |
| train_nonp_retrain3_repair.js | doctrine repair cycle |

## Training — NoPnx-PA pair (full 174 docs, 87 per jury → `jury/doctrines/NoPnx-PA`)
| file | role |
|---|---|
| train_nopa_retrain4.js / _gentle.js / _rolling.js | rolling trainer generations |
| train_nopa_retrain4_finish.js | final batches |

## Training — shared punctuated pair (→ `jury/doctrines/Pnx-NP-PA`, adjudicates Pnx-NP and Pnx-PA)
Phase 1: attempt-from-scratch on the full de-punctuated corpus; phase 2: graded
punctuated paragraph-aware rounds (the TRACK NOTE bridge law).
| file | role |
|---|---|
| train_pnx_01_np_pipeline.js | initial pipeline |
| train_pnx_02_np_juror_wide.js | per-document solo attempts (canonical phase-1 prompt) |
| train_pnx_03_np_juror_rlhf.js / 04_np_sliding_window.js | later phase-1 generations |
| train_pnx_05_np_group_exam.js | graded group rounds |
| train_pnx_06_np_reflect.js | reflection/consolidation |
| train_pnx_08_pa_arbitration.js | phase-2 arbitration |
| train_pnx_09_pa_strike.js / 10_npa_strike.js | contamination strikes (quarantined entries) |

## Evaluation — the counted ablation verdicts (released test split, 262 docs/track)
| file | track |
|---|---|
| exam_doctrine_nonp.js / _b.js | NoPnx-NP |
| exam_doctrine_nopa_2lane.js / _serial.js / _jury.js | NoPnx-PA (retrain4 pair) |
| exam_doctrine_nopa_redo26.js | NoPnx-PA: re-adjudication of 26 docs first judged by a pre-retrain launcher |
| exam_doctrine_nopa_c.js / _d.js | VOIDED: launched Aug 5 against the earlier retrain3 pair, before retrain4 existed; their 26 verdicts sit in `exam_out_void_retrain3pair/` and feed no reported number |
| exam_doctrine_np_shipped.js / _finish.js | Pnx-NP (shipped pair) |
| exam_doctrine_pa_resume.js | Pnx-PA |
| exam_doctrine_chain_superseded_np.js / _b.js | SUPERSEDED Pnx-NP run with the wrong (retrained) pair; verdicts voided on disk, feeds no reported number |

## Competition day
| file | role |
|---|---|
| exam_blind_pa_competition.js | the blind Pnx-PA exam sittings (contains the TRACK NOTE) |

## Controls, rerun, audits
| file | role |
|---|---|
| exam_zeroshot_all_tracks.js / _resume.js | zero-shot control, all four tracks (no doctrine step) |
| exam_rerun_variance_nonp.js | 50-doc doctrine rerun (variance) + the killed gold-examples arm's chain (that arm reported nothing) |
| audit_compliance.js / audit_nonp_purity.js / audit_pnxpa_purity.js | file-access & purity audit prompts |

## Deadline-morning brief (the submission-day orchestrator)
| file | role |
|---|---|
| deadline_morning_brief.md | the verbatim Aug-4 pipeline brief (three clean-room tracks, sealed probes, ship gates). CORRECTION OF RECORD: its clause "census-balanced from TRAIN metadata only" is inaccurate — genre labels came from train metadata, but the census targets came from a classification of the BLIND documents' input texts (no labels involved); see the paper's Limitations and CONTAMINATION_MAP.md §5. Its per-jury lists (19–20 docs) are the shipped curricula; the user's standing order that night was full-corpus training — the full-corpus launches died (see exam/train records above). |
