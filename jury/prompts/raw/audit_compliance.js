export const meta = {
  name: 'compliance-audit',
  description: 'Adversarial dev/test-training audit across all four AraSeg tracks: six independent verifiers each try to PROVE a violation occurred; findings compiled into a verdict of record.',
  phases: [{ title: 'Audit' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const OUT = { type:'object', required:['verdict','evidence'], properties:{
  verdict:{enum:['CLEAN','VIOLATION','SUSPICIOUS']}, evidence:{type:'string'}, details:{type:'string'} } }

const COMMON = `You are an adversarial compliance auditor for the AraSeg 2026 competition. The claim under attack: "the two AI jurors were trained EXCLUSIVELY on train splits; dev was never used in any form; test appeared only as unlabeled probe text in exams." Your job is to BREAK this claim — hunt for a violation. If you cannot find one after genuinely trying, return CLEAN with the evidence you checked. If you find one, return VIOLATION with exact file paths and content. If something is ambiguous, return SUSPICIOUS and quote it. Work from ${REPO}. Split membership ground truth: data/<TRACK>_{train,dev,test,blind}.jsonl for TRACK in {NoPnx-NP, NoPnx-PA, NP, PA}. Be precise; do not pad.`

const DIMS = [
  { k:'answers', p:`${COMMON}
DIMENSION: every juror answer file. Enumerate ALL doc ids in scratch_exo/np_train_work/ans/j*/, ans_pa/j*/, ans_np2/j*/, and all docs arrays inside scratch_exo/np_train_work/practice_np*juror*.json. Independently re-derive each id's split membership per its track family (ans->NoPnx-NP, ans_pa->PA, ans_np2->NP) by loading the split files with python. Report exact counts per split. Any answered doc in dev or test = VIOLATION.` },
  { k:'packets', p:`${COMMON}
DIMENSION: every packet ever shown to a juror. Dirs: scratch_exo/{np_docs,pa_train_docs,np2_train_docs}/, scratch_exo/practice_np_*.json, np_batch2_j*.json, np_sit_j*.json, and exam packets scratch_exo/{np_exam,pa_exam,pa_arb,pa_strike,np2_exam,npa_strike}/docs/. For TRAINING packet sources: verify membership is 100% train. For EXAM packets: verify membership is only blind+test, and specifically ZERO dev ids. For ALL packet files: scan raw text for gold leakage — the strings "labels", "label_str", any long 0/1 boundary string, and any field other than doc_id/topic/text/prior_arguments. Report per-dir counts and any hits.` },
  { k:'graders', p:`${COMMON}
DIMENSION: the feedback channel. Read scratch_exo/grade_practice_np.py, grade_practice_pa.py, grade_practice_np2.py IN FULL: verify the only gold ever loaded is a *_train.jsonl and that mistake reports could only ever contain train-doc content. Then hunt for OTHER writers: grep scratch_exo for any script writing to np_train_work/mistakes_* or lessons_* or doctrine_* and audit each hit's data sources. Any path by which dev/test gold could have reached a juror-readable file = VIOLATION.` },
  { k:'doctrines', p:`${COMMON}
DIMENSION: doctrine contents. Read scratch_exo/np_train_work/doctrine_np_juror0.md and doctrine_np_juror1.md in full (and skim lessons_np_juror*.md archives). Hunt for: (a) any quoted example traceable to a dev or test document — cross-check suspicious quoted Arabic phrases by grepping the dev/test jsonl files for them; (b) any reference to probe doc ids; (c) any per-site gold information about non-train documents. Aggregate gate outcomes (e.g. "the arbitration scored 95.88") are DISCLOSED and allowed — note them but they are not violations. Per-site or quoted gold from dev/test = VIOLATION.` },
  { k:'transcripts', p:`${COMMON}
DIMENSION: what juror agents actually did. The juror agents' full transcripts live under C:/Users/pc/.claude/projects/C--Users-pc-Downloads/09419303-a8fa-420e-8295-f82b217c8b29/subagents/ (workflows/*/agent-*.jsonl and task transcripts). Grep them for reads of forbidden files: occurrences of "_dev.jsonl" and "_test.jsonl" inside tool_use inputs (Read/Bash commands) in JUROR transcripts. Legitimate exceptions you must distinguish: the orchestrator's own gate/assembly scripts (run in the main session, not by jurors) read test gold locally; grade_practice scripts read only train. Any juror agent transcript showing it opened a dev or test jsonl, or an answer-key-bearing file = VIOLATION with the transcript path quoted.` },
  { k:'history', p:`${COMMON}
DIMENSION: inherited artifacts and the old protocol. Read coordination/STATE.md and coordination/ORCHESTRATOR_LESSONS.md. Document precisely: which SHIPPED submissions (by track and zip) derive from the PRE-campaign jury whose practice used dev documents under the old protocol, versus this campaign's train-only jurors. Also check scratch_exo/build_train_practice.py (the OLD PA builder) vs build_train_practice_np.py and the pa/np2 builders — confirm which were used for which shipped result. Output a clean lineage table: submission -> jury generation -> training data provenance. This dimension is about honest characterization, not gotchas.` },
]

phase('Audit')
const res = await parallel(DIMS.map(d => () => agent(d.p,
  { label:`audit:${d.k}`, phase:'Audit', schema: OUT, model:'opus', effort:'high', agentType:'general-purpose' })
  .then(v => ({ dim: d.k, ...v }))))
const ok = res.filter(Boolean)
log(ok.map(r => `${r.dim}: ${r.verdict}`).join(' | '))
return { findings: ok }
