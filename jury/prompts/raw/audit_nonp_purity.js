export const meta = {
  name: 'nopnx-np-purity-audit',
  description: 'Focused adversarial audit of the NoPnx-NP 89.5 submission: three verifiers hunt for any dev/test influence in (1) the exact prompts used, (2) the doctrine content at ship time, (3) the inherited base rows lineage.',
  phases: [{ title: 'Verify' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const SESS = 'C:/Users/pc/.claude/projects/C--Users-pc-Downloads/09419303-a8fa-420e-8295-f82b217c8b29'
const OUT = { type:'object', required:['verdict','evidence'], properties:{ verdict:{enum:['CLEAN','VIOLATION','SUSPICIOUS']}, evidence:{type:'string'} } }
const COMMON = `Adversarial auditor: try to PROVE that dev/test influence reached the NoPnx-NP submission (board 89.5, shipped 2026-07-31 as prediction_NoPnx-NP.jury3.zip). Ground truth splits: ${REPO}/data/NoPnx-NP_{train,dev,test,blind}.jsonl. Return CLEAN only if you genuinely failed to find a violation; quote exact text and paths for anything found.`
const DIMS = [
  { k:'prompts', p:`${COMMON}
DIMENSION — THE EXACT PROMPTS. The jurors' training prompts and exam prompts are preserved verbatim in the workflow scripts under ${SESS}/workflows/scripts/ (files matching jury-train-np*, np-juror-rlhf*, np_juror_wide-like names in the scratchpad too: C:/Users/pc/AppData/Local/Temp/claude/C--Users-pc-Downloads/09419303-a8fa-420e-8295-f82b217c8b29/scratchpad/np_juror_wide.js) and the NoPnx-NP exam script (np-group-exam-*.js). Read every one of these scripts IN FULL. Hunt for: any mention of dev/test documents, any statistic, count, tendency, or claim that could only derive from dev or test gold (e.g. error counts on non-train data, probe outcomes, per-genre performance numbers from non-train sources), any instruction steering jurors using evaluation results. Statements about the DRAFT being unreliable in general terms are fine; numbers or verdicts sourced from dev/test gold are not. Also check the practice/exam packet builders (scratch_exo/build_train_practice_np.py, np_build_exam.py) for what entered packet text.` },
  { k:'doctrine', p:`${COMMON}
DIMENSION — DOCTRINE CONTENT AT SHIP TIME. The exam that produced the submission ran 2026-07-30/31 using ${REPO}/scratch_exo/np_train_work/lessons_np_juror{0,1}.md as they existed then (append-only; the batches are dated/numbered, so content added later is identifiable — anything referencing PA-track practice, punctuation appendices, arbitration or probes came AFTER). Read both archives and both doctrine files. Identify which content existed by Jul 31 and audit THAT for any dev/test-derived material: quoted phrases (grep suspicious quoted Arabic against dev/test jsonl), doc references, statistics not derivable from train grading. Also verify the graded corpus behind those lessons: scratch_exo/practice_np_manifest.json + np_train_work answer files -> all train.` },
  { k:'baserows', p:`${COMMON}
DIMENSION — THE INHERITED BASE ROWS. The submission edited the prior 88.30 rows (subs_blind/prediction_NoPnx-NP.jury2), produced by the PREVIOUS session's UNTRAINED jury waves. Audit that lineage: read scratch_exo/jury_fleet.js, jury2_fleet.js, jury2_pipeline.py, build_blind_packets.py, build_refine_packets.py, build_wholedoc_packets.py and any wave scripts. Questions: (a) did those wave jurors receive any dev/test documents or gold as instruction/practice material (exemplars, few-shot examples, error statistics from dev/test gold)? (b) or were they untrained editors judging blind text with train-sourced exemplars only, selected/gated via test probes (which is allowed tuning)? Quote the packet-builder docstrings and any exemplar sources. Any dev/test data-point used as instruction material in that lineage = VIOLATION; gating/selection on probes = allowed.` },
]
phase('Verify')
const res = await parallel(DIMS.map(d => () => agent(d.p,
  { label:`np-audit:${d.k}`, phase:'Verify', schema: OUT, model:'opus', effort:'high', agentType:'general-purpose' })
  .then(v => ({ dim:d.k, ...v }))))
log(res.filter(Boolean).map(r=>`${r.dim}: ${r.verdict}`).join(' | '))
return { findings: res.filter(Boolean) }
