export const meta = {
  name: 'np-training-pipeline',
  description: 'NoPnx-NP training pipeline, end to end: rounds of 20 documents, each document answered in its own clean memory, then graded against train gold, then each juror rewrites its own lessons — repeated until both jurors have covered all 174 training documents.',
  phases: [
    { title: 'Round 1' }, { title: 'Grade 1' }, { title: 'Lessons 1' },
    { title: 'Round 2' }, { title: 'Grade 2' }, { title: 'Lessons 2' },
    { title: 'Round 3' }, { title: 'Grade 3' }, { title: 'Lessons 3' },
    { title: 'Round 4' }, { title: 'Grade 4' }, { title: 'Lessons 4' },
    { title: 'Round 5' }, { title: 'Grade 5' }, { title: 'Lessons 5' },
    { title: 'Round 6' }, { title: 'Grade 6' }, { title: 'Lessons 6' },
  ],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const WORK = REPO + '/scratch_exo/np_train_work'
const CHUNK = 20

const OUT_A = { type:'object', required:['doc_id','identity','law','boundaries'], properties:{
  doc_id:{type:'string'}, identity:{type:'string'}, law:{type:'string'}, reasoning:{type:'string'},
  boundaries:{ type:'array', items:{ type:'object', required:['i','w','c'], properties:{
    i:{type:'integer'}, w:{type:'string'}, c:{enum:['hi','med','lo']}, why:{type:'string'} } } } } }
const OUT_G = { type:'object', required:['report'], properties:{ report:{type:'string'} } }
const OUT_R = { type:'object', required:['lessons'], properties:{ lessons:{type:'string'} } }

const ANSWER = (j, did, round) => `You are Juror ${j}, in TRAINING. A team of Arabic annotators segmented thousands of documents into sentences under a private, internally consistent policy; punctuation stripped. Your task: mark the sentence boundaries in ONE document exactly as you believe the annotators' answer key marks them. YOUR ANSWER WILL BE GRADED against the real answer key and every mistake returned to you, with context, for study — so commit an honest, reasoned answer.

${round > 1 ? `FIRST ACTION: read YOUR OWN doctrine: ${WORK}/lessons_np_juror${j}.md — the lessons YOU wrote after your own graded mistakes. Hold them in front of you while you read this document and apply them. If this document's evidence contradicts a lesson, follow the evidence and say so in your reasoning; the lessons are yours to refine, not a cage.

` : ''}One structural fact you may rely on (it is a property of the format, not advice): the final word of a document always ends a sentence. Everything else is yours to judge. Derive the rest from the text in front of you and your own knowledge of Arabic.

Read exactly ONE file, exactly ONCE, and nothing else: ${REPO}/scratch_exo/np_docs/${did}.json — a single practice document with its "topic" tag and its "text".

This is your only document. Think as long and as hard as it deserves — there is no limit on your reasoning and depth is the entire point. A thin answer is the failure mode; there is no such thing as too much thinking. Work explicitly:
1. RECOGNIZE: what IS this text? Name it precisely ("identity").
2. DERIVE ITS LAW: what is the natural sentence unit of THIS document? ("law").
3. REASON IN WRITING ("reasoning"): the evidence for your identification; the law and WHY, quoting short phrases as evidence; and at every uncertain site, the argument in full — the reading you chose, the reading you rejected, and why the annotators would agree with you.
4. DRAW: the complete boundary list — every sentence-final word as {i, w, c, why}, copying w from the text at the moment you decide. hi = would bet on it; med = uncertain; lo = lean against but flag.

Markup: ⟨i⟩ = next word is number i; boundary = index of the LAST word of a sentence; the final word always ends a sentence.
OUTPUT: Write your complete answer as a single JSON object to ${WORK}/ans/j${j}/${did}.json with exactly the keys doc_id, identity, law, reasoning, boundaries. Then return the same object as the structured result. No Bash, no PowerShell, no other files (answer keys exist elsewhere in this repo — touching anything beyond your one document and your one output file voids your training), no web.`

const GRADE = (round) => `Run the grader for NoPnx-NP training round ${round} and report what it prints.
Run exactly this, from ${REPO}:
  python scratch_exo/grade_practice_np.py round${round}
Return the complete printed output verbatim as "report" — the per-juror document counts and mean F1/P/R lines, plus the per-topic summary blocks from ${WORK}/mistakes_np_juror0.txt and ${WORK}/mistakes_np_juror1.txt (read only the SUMMARY BY TOPIC section at the end of each). Change nothing, interpret nothing, add no commentary.`

const REFLECT = (j, round) => `You are Juror ${j}. Your practice answers have been GRADED against the annotators' real answer keys. Your complete mistake report is in: ${WORK}/mistakes_np_juror${j}.txt — every boundary you wrongly added and every boundary you missed, each with the surrounding text and what the key actually says, organized by document and topic.

Read it exactly once and study it the way a student studies a returned exam: for each error, articulate what you were thinking versus what the annotators' policy actually is. THINK DEEPLY — the errors are the curriculum; patterns across errors are the syllabus. Take as long as you need; there is no limit on your reasoning and no reward for brevity.

Pay particular attention to topics where your score collapsed. The summary at the end gives your F1 per topic; where recall sits far below precision you are systematically failing to cut where the annotators cut, and the report shows you exactly where. Find the rule behind it, not the individual sites.

${round > 1 ? `Your previous doctrine is at ${WORK}/lessons_np_juror${j}.md — read it, then rewrite it whole. Keep what the new mistakes confirm, correct what they contradict, add what is missing, and delete what proved wrong. Say in one line at the top what changed since the last version and why.

` : ''}Write YOUR OWN LESSONS: a doctrine, in your own words, that would have prevented your mistakes — organized by topic (expository, narrative, exercise/worksheet, hadith, quran, bible, poetry, conversational, legal), concrete enough that you could hand it to yourself tomorrow and segment these genres correctly. State each lesson as what the ANNOTATORS do, with a short example from your own errors, not as a vague maxim. Where you were already correct, say so in one line. Write as many lessons as your mistakes justify.

Write the doctrine to ${WORK}/lessons_np_juror${j}.md (overwriting the previous version) — that file is the record and it is what you carry into the exam — then RETURN it as the structured result: {"lessons": "<your full doctrine>"}. No Bash, no PowerShell, no web, and do not open any data file or answer key.`

const A = typeof args === 'string' ? JSON.parse(args) : args
const L0 = A?.['0'] ?? [], L1 = A?.['1'] ?? []
if (L0.length < 100 || L1.length < 100) throw new Error(`job lists did not parse: J0=${L0.length} J1=${L1.length} — refusing to spend tokens`)
const nRounds = Math.ceil(Math.max(L0.length, L1.length) / CHUNK)
log(`pipeline: J0 ${L0.length} docs, J1 ${L1.length} docs, ${CHUNK} per round -> ${nRounds} rounds of answer/grade/lessons`)

const history = []
for (let r = 1; r <= nRounds; r++) {
  const slice0 = L0.slice((r - 1) * CHUNK, r * CHUNK)
  const slice1 = L1.slice((r - 1) * CHUNK, r * CHUNK)
  const jobs = [...slice0.map(d => ['0', d]), ...slice1.map(d => ['1', d])]
  if (!jobs.length) break

  phase(`Round ${r}`)
  const answers = await parallel(jobs.map(([j, d]) => () => agent(ANSWER(j, d, r),
    { label:`r${r}:j${j}:${d.replace('doc_','')}`, phase:`Round ${r}`, schema: OUT_A, model:'opus', effort:'max' })))
  const got = answers.filter(Boolean).length
  log(`round ${r}: ${got}/${jobs.length} documents answered, spend ${budget.spent()}`)

  phase(`Grade ${r}`)
  const g = await agent(GRADE(r), { label:`r${r}:grade`, phase:`Grade ${r}`, schema: OUT_G, model:'opus', effort:'low' })
  log(`round ${r} grading:\n${g?.report ?? '(no report)'}`)

  phase(`Lessons ${r}`)
  const les = await parallel([0, 1].map(j => () => agent(REFLECT(j, r),
    { label:`r${r}:lessons:j${j}`, phase:`Lessons ${r}`, schema: OUT_R, model:'opus', effort:'max' })
    .then(v => ({ juror: j, chars: (v?.lessons ?? '').length }))))
  log(`round ${r} lessons rewritten: ${les.filter(Boolean).map(x=>`J${x.juror}=${x.chars}ch`).join(' ')}, spend ${budget.spent()}`)

  history.push({ round: r, answered: got, of: jobs.length, grading: g?.report ?? null,
                 lessons: les.filter(Boolean) })
}
return { rounds: history.length, history, spent: budget.spent() }
