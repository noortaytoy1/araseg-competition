export const meta = {
  name: 'np-training-sliding-window',
  description: 'NoPnx-NP training, sliding window: each juror works documents one after another inside a link capped at 8000 words, hands off to a fresh link before it can fill up, and after every round of 20 documents the answers are graded against train gold and the juror rewrites its own lessons.',
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

const OUT_L = { type:'object', required:['done'], properties:{ done:{type:'integer'}, note:{type:'string'} } }
const OUT_G = { type:'object', required:['report'], properties:{ report:{type:'string'} } }
const OUT_R = { type:'object', required:['lessons'], properties:{ lessons:{type:'string'} } }

const LINK = (j, docs, round) => `You are Juror ${j}, in TRAINING. A team of Arabic annotators segmented thousands of documents into sentences under a private, internally consistent policy; punctuation stripped. Your task: mark the sentence boundaries as you believe the annotators' answer key marks them. YOUR ANSWERS WILL BE GRADED against the real answer keys and every mistake returned to you, with context, for study — so commit honest, reasoned answers.

${round > 1 ? `FIRST ACTION: read YOUR OWN doctrine: ${WORK}/lessons_np_juror${j}.md — the lessons YOU wrote from your own graded mistakes. Hold them in front of you for every document and apply them. If a document's evidence contradicts a lesson, follow the evidence and say so; the lessons are yours to refine, not a cage.

` : ''}One structural fact you may rely on (it is a property of the format, not advice): the final word of a document always ends a sentence. Everything else is yours to judge. Derive the rest from the text in front of you and your own knowledge of Arabic.

YOUR DOCUMENTS, in this order (${docs.length} of them, ${docs.length === 1 ? 'a single long one' : 'each short'}):
${docs.map((d, n) => `${n + 1}. ${REPO}/scratch_exo/np_docs/${d}.json`).join('\n')}

WORK STRICTLY ONE DOCUMENT AT A TIME. Open document 1, read it once, think it through completely, write your answer file for it, and only then open document 2. Never read ahead. Never hold two documents in mind at once — each document is judged on its own evidence, and nothing you learned from an earlier document in this list is evidence about a later one.

Think as long and as hard as each document deserves — there is no limit on your reasoning and depth is the entire point. A thin answer is the failure mode. For EVERY document:
1. RECOGNIZE: what IS this text? Name it precisely.
2. DERIVE ITS LAW: what is the natural sentence unit of THIS document?
3. REASON IN WRITING: the evidence for your identification; the law and WHY, quoting short phrases as evidence; and at every uncertain site, the argument in full — the reading you chose, the reading you rejected, and why the annotators would agree with you.
4. DRAW: the complete boundary list — every sentence-final word as {i, w, c, why}, copying w from the text at the moment you decide. hi = would bet on it; med = uncertain; lo = lean against but flag.

Markup: ⟨i⟩ = next word is number i; boundary = index of the LAST word of a sentence; the final word always ends a sentence.

OUTPUT, per document, immediately on finishing it: Write a single JSON object to ${WORK}/ans/j${j}/<doc_id>.json with exactly the keys doc_id, identity, law, reasoning, boundaries. One file per document, written the moment that document is done — never batched, never held.
RESUME AWARENESS: before starting each document, if its answer file already exists, skip it and move to the next.
If a document is very long, work it in sections across several responses, extending that one document's answer file as you go — never try to emit a whole long document in one response.
When every document in your list has its file, return {"done": <count>, "note": "<anything that went wrong, or empty>"}. No PowerShell, no other files (answer keys exist elsewhere in this repo — touching anything beyond your listed documents and your own answer files voids your training), no web.`

const GRADE = (round) => `Run the NoPnx-NP grader for training round ${round} and report exactly what it prints.
From ${REPO}, run: python scratch_exo/grade_practice_np.py round${round}
Then read ONLY the "SUMMARY BY TOPIC" section at the end of ${WORK}/mistakes_np_juror0.txt and ${WORK}/mistakes_np_juror1.txt.
Return as "report": the grader's printed output verbatim, followed by both per-topic summaries. Change nothing, interpret nothing, add no commentary.`

const REFLECT = (j, round) => `You are Juror ${j}. Your practice answers have been GRADED against the annotators' real answer keys. Your complete mistake report is in: ${WORK}/mistakes_np_juror${j}.txt — every boundary you wrongly added and every boundary you missed, each with the surrounding text and what the key actually says, organized by document and topic.

Read it and study it the way a student studies a returned exam: for each error, articulate what you were thinking versus what the annotators' policy actually is. THINK DEEPLY — the errors are the curriculum; patterns across errors are the syllabus. Take as long as you need; there is no limit on your reasoning and no reward for brevity.

Pay particular attention to topics where your score collapsed. The summary at the end gives your F1 per topic; where recall sits far below precision you are systematically failing to cut where the annotators cut, and the report shows you exactly where. Find the rule behind it, not the individual sites.

${round > 1 ? `Your previous doctrine is at ${WORK}/lessons_np_juror${j}.md — read it, then rewrite it WHOLE. Keep what the new mistakes confirm, correct what they contradict, add what is missing, delete what proved wrong. Open with one line saying what changed since the last version and why.

` : ''}Write YOUR OWN LESSONS: a doctrine, in your own words, that would have prevented your mistakes — organized by topic (expository, narrative, exercise/worksheet, hadith, quran, bible, poetry, conversational, legal), concrete enough that you could hand it to yourself tomorrow and segment these genres correctly. State each lesson as what the ANNOTATORS do, with a short example from your own errors, not as a vague maxim. Where you were already correct, say so in one line. Write as many lessons as your mistakes justify.

Write the doctrine to ${WORK}/lessons_np_juror${j}.md (overwriting the previous version) — that file is the record and what you carry into the exam — then RETURN it as {"lessons": "<your full doctrine>"}. No PowerShell, no web, and do not open any data file or answer key.`

const A = typeof args === 'string' ? JSON.parse(args) : args
const R0 = A?.['0'] ?? [], R1 = A?.['1'] ?? []
if (R0.length < 5 || R1.length < 5) throw new Error(`window plan did not parse: J0=${R0.length} rounds J1=${R1.length} rounds — refusing to spend tokens`)
const nRounds = Math.max(R0.length, R1.length)
const totalLinks = R0.reduce((a, r) => a + r.length, 0) + R1.reduce((a, r) => a + r.length, 0)
log(`sliding window: ${nRounds} rounds, ${totalLinks} links total (J0 ${R0.reduce((a,r)=>a+r.length,0)}, J1 ${R1.reduce((a,r)=>a+r.length,0)}), grading + lessons after every round`)

const history = []
for (let r = 0; r < nRounds; r++) {
  const links = [
    ...(R0[r] ?? []).map(docs => ['0', docs]),
    ...(R1[r] ?? []).map(docs => ['1', docs]),
  ]
  if (!links.length) continue

  phase(`Round ${r + 1}`)
  const done = await parallel(links.map(([j, docs], n) => () => agent(LINK(j, docs, r + 1),
    { label:`r${r + 1}:j${j}:link${n}(${docs.length}d)`, phase:`Round ${r + 1}`, schema: OUT_L, model:'opus', effort:'max' })
    .then(v => ({ juror: j, want: docs.length, got: v?.done ?? 0, note: v?.note || '' }))))
  const ok = done.filter(Boolean)
  log(`round ${r + 1}: ${ok.reduce((a, x) => a + x.got, 0)}/${links.reduce((a, l) => a + l[1].length, 0)} documents answered` +
      (ok.filter(x => x.note).map(x => ` | J${x.juror}: ${x.note}`).join('') || '') + `, spend ${budget.spent()}`)

  phase(`Grade ${r + 1}`)
  const g = await agent(GRADE(r + 1), { label:`r${r + 1}:grade`, phase:`Grade ${r + 1}`, schema: OUT_G, model:'opus', effort:'low' })
  log(`round ${r + 1} grading:\n${g?.report ?? '(no report)'}`)

  phase(`Lessons ${r + 1}`)
  const les = await parallel([0, 1].map(j => () => agent(REFLECT(j, r + 1),
    { label:`r${r + 1}:lessons:j${j}`, phase:`Lessons ${r + 1}`, schema: OUT_R, model:'opus', effort:'max' })
    .then(v => ({ juror: j, chars: (v?.lessons ?? '').length }))))
  log(`round ${r + 1} doctrine rewritten: ${les.filter(Boolean).map(x => `J${x.juror}=${x.chars}ch`).join(' ')}, spend ${budget.spent()}`)

  history.push({ round: r + 1, links: ok, grading: g?.report ?? null, lessons: les.filter(Boolean) })
}
return { rounds: history.length, history, spent: budget.spent() }
