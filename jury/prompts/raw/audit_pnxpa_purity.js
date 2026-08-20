export const meta = {
  name: 'pnx-pa-purity-audit',
  description: 'Forensic audit of the Pnx-PA 95.30 submission, mirroring the NoPnx-NP audit: three adversarial verifiers on (1) the exact prompts in its lineage, (2) doctrine content at its ship time, (3) the base rows (plain ensemble) lineage — hunting for any dev/test instruction material.',
  phases: [{ title: 'Verify' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const SESS = 'C:/Users/pc/.claude/projects/C--Users-pc-Downloads/09419303-a8fa-420e-8295-f82b217c8b29'
const OUT = { type:'object', required:['verdict','evidence'], properties:{ verdict:{enum:['CLEAN','VIOLATION','SUSPICIOUS']}, evidence:{type:'string'} } }
const COMMON = `Adversarial auditor for the Pnx-PA submission (board 95.30, shipped 2026-08-02 ~17:42 as prediction_PA.jury2.zip, assembled from scratch_exo/pa_exam/out verdicts applied to subs_blind/prediction_PA rows). NOTE: dev/test gold data files have been moved into C:/Users/pc/araseg_heldout_LOCKED.zip — read them from inside that zip via python zipfile when you need split membership or gold; do NOT extract them into the repo. Known and already-disclosed: one behavioral sentence ("your first appendix made you TIMID on exercise grids") derived from a test-probe gate appeared in two training prompts BEFORE this exam — confirm its exact occurrences but it is a KNOWN finding; your job is to find anything BEYOND it. The later "3-document arbitration lesson" post-dates this ship (verify the timeline yourself). Return CLEAN only if nothing beyond the known sentence exists; quote paths and text for anything found.`
const DIMS = [
  { k:'prompts', p:`${COMMON}
DIMENSION — EXACT PROMPTS in the Pnx-PA lineage: every training prompt that shaped the jurors' PA doctrine BEFORE the 95.30 exam, and the exam prompt itself. Sources: the session transcript ${SESS}.jsonl (Agent tool calls 2026-08-01..08-02: "PA adaptation training", "PA training tranche 2"/census-matched, "exercise tranche", "full-coverage final tranche") and the exam script (pa_group_exam.js in C:/Users/pc/AppData/Local/Temp/claude/C--Users-pc-Downloads/09419303-a8fa-420e-8295-f82b217c8b29/scratchpad/). Read each in full. Hunt for dev/test docs, ids, statistics, outcomes, or steering derived from non-train evaluation — anything beyond the known TIMID sentence. Also enumerate every doc id in juror-facing inputs of that period and classify against splits (read splits from the LOCKED zip).` },
  { k:'doctrine', p:`${COMMON}
DIMENSION — DOCTRINE AT SHIP TIME (2026-08-02 ~afternoon): the jurors' doctrines when the 95.30 exam ran. The doctrines evolved after; pre-cleanse copies are in scratch_exo/np_train_work/quarantine/doctrine_np_juror{0,1}.pre_cleanse.md — use the appendix batch structure and the transcript timeline to identify what existed at ship time. Audit that content for dev/test-derived material: quoted Arabic traceable to dev/test docs (grep candidates against the split files inside the LOCKED zip), probe references, non-train statistics. The TIMID sentence's doctrinal echoes count as the known finding; anything else is new.` },
  { k:'baserows', p:`${COMMON}
DIMENSION — BASE ROWS: the exam's draft rows were subs_blind/prediction_PA (blind) and prediction_PnxPA (test probes) — both "plain ensemble" outputs. Audit their lineage: how were these generated (search scratch_exo and src for the generating scripts around 2026-07-20..21; check runs/pnxpa_* training configs/logs and scratch_exo/train_pnx_voters.sh)? Confirm the voters were fine-tuned on PA train only, and that no jury, exemplar, or dev/test instruction step touched these rows. Decode-parameter selection on dev is ALLOWED (tuning); any dev/test document used as training data points for the voters, or any LLM step with dev/test exemplars in this lineage = VIOLATION.` },
]
phase('Verify')
const res = await parallel(DIMS.map(d => () => agent(d.p,
  { label:`pa-audit:${d.k}`, phase:'Verify', schema: OUT, model:'opus', effort:'high', agentType:'general-purpose' })
  .then(v => ({ dim:d.k, ...v }))))
log(res.filter(Boolean).map(r=>`${r.dim}: ${r.verdict}`).join(' | '))
return { findings: res.filter(Boolean) }
