export const meta = {
  name: 'np-juror-wide',
  description: 'NoPnx-NP juror training by gold feedback, widened: one juror reads a whole batch of documents at once in separate clean contexts, is corrected against the gold keys, and appends every correction to its cumulative record before the next batch.',
  phases: [{ title: 'Training' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const WORK = REPO + '/scratch_exo/np_train_work'
const CHUNK = 16

const OUT_A = { type:'object', required:['doc_id','identity','law','boundaries'], properties:{
  doc_id:{type:'string'}, identity:{type:'string'}, law:{type:'string'}, reasoning:{type:'string'},
  boundaries:{ type:'array', items:{ type:'object', required:['i','w','c'], properties:{
    i:{type:'integer'}, w:{type:'string'}, c:{enum:['hi','med','lo']}, why:{type:'string'} } } } } }
const OUT_L = { type:'object', required:['learned'], properties:{
  learned:{type:'string'}, f1:{type:'string'}, errors:{type:'integer'}, note:{type:'string'} } }

const SEGMENT = (j, did, n, total) => `You are Juror ${j}. You are learning to segment Arabic text the way a specific team of annotators does, by being corrected against their answer keys. This is batch ${n} of ${total}, and this is ONE document from that batch.

FIRST: read ${WORK}/lessons_np_juror${j}.md — your own accumulated record of every correction you have absorbed so far, from every document you have already been graded on. Hold it in front of you while you read this document and apply it. If a rule in your record is contradicted by the evidence in this document, follow the evidence and say so in your reasoning.

YOUR DOCUMENT — read this one file, once, and nothing else:
${REPO}/scratch_exo/np_docs/${did}.json

One structural fact you may rely on: the final word of a document always ends a sentence. Everything else is your judgment, informed by your record.

Think as long and as hard as this document deserves — there is no limit on your reasoning and depth is the entire point. A thin answer is the failure mode; there is no such thing as too much thinking here. Work out:
1. RECOGNIZE — what IS this text? Name it precisely ("identity").
2. DERIVE ITS LAW — what is the natural sentence unit of THIS document? ("law")
3. REASON IN WRITING ("reasoning") — the evidence for your identification; the law and WHY, quoting short phrases from the document as evidence; which rules from your record you applied and where; and at every uncertain site the argument in full: the reading you chose, the reading you rejected, and why these annotators would agree with you. Where a passage is genuinely hard, re-read it and say what changed on the second reading.
4. DRAW — the complete boundary list: every sentence-final word as {i, w, c, why}, copying w from the text at the moment you decide. hi = would bet on it; med = uncertain; lo = lean against but flagging.

Markup: ⟨i⟩ = next word is number i; boundary = index of the LAST word of a sentence.

OUTPUT: Write your complete answer as one JSON object to ${WORK}/ans/j${j}/${did}.json with exactly the keys doc_id, identity, law, reasoning, boundaries — then return the same object as the structured result. If this document is very long, work it in sections across several responses, extending its answer file as you go; never try to emit a long document in one response.
Do NOT write to the lessons file — you are one of many readers working this batch at the same time, and the record is written once, after all of you are graded together.
No other files (answer keys exist elsewhere in this repo — touching anything beyond your one document and your one answer file voids your training), no web.`

const LEARN = (j, docs, n, total) => `You are Juror ${j}. You have just segmented ${docs.length} documents (batch ${n} of ${total}) and now you find out what the annotators' answer keys actually say. This is the step where you learn.

STEP 1 — BE CORRECTED. Run exactly this from ${REPO}:
  python scratch_exo/grade_practice_np.py --juror ${j} --only ${docs.join(',')}
Then read ${WORK}/mistakes_np_juror${j}_chunk.txt — the key's verdict on exactly those documents: every boundary you invented that the key does not have, and every boundary the key has that you missed, each shown in context beside what the key actually says.

STEP 2 — UNDERSTAND. Study it the way a student studies a returned exam. For each error, work out what you were thinking versus what the annotators' policy actually is. Take as long as you need — there is no limit on your reasoning here and no reward for brevity. Look for the RULE behind each error, not the individual site. Then check each rule against your existing record at ${WORK}/lessons_np_juror${j}.md: does it confirm something already written, contradict it, or is it new?

STEP 3 — RECORD IT. Append what you learned to ${WORK}/lessons_np_juror${j}.md. This file is CUMULATIVE and APPEND-ONLY — the full record of every correction you have ever absorbed, and the only thing you carry into the real exam.
  * Never delete an entry. Never compress away an earlier lesson to make room.
  * Add each new correction as: the rule, stated as what the ANNOTATORS do, plus the short example from your own error that produced it, under its topic heading (expository, narrative, exercise, hadith, quran, bible, poetry, conversational, legal).
  * Sharpen or merge an existing entry ONLY when this batch proves it wrong or exactly duplicates it — and say in one line what changed and why.
  * Where you made no errors, record in one line the law that worked, so confirmed rules accumulate too.
  * Keep a short INDEX at the very top of the file: one line per rule, grouped by topic, so the whole record can be scanned quickly before the full entries below are consulted. Update the index as you add entries. The index is a addition, never a replacement — every full entry stays.
Use Edit to append (Write only if the file does not exist yet). The file must be on disk before you finish — it is the record.

Return {"learned": "<one sentence on what this batch taught you>", "f1": "<the grader's printed F1 line for you>", "errors": <number of errors on these documents>, "note": "<anything wrong, else empty>"}.
Do not open any data file or answer key beyond what the grader produces. No web.`

const A = typeof args === 'string' ? JSON.parse(args) : args
const J = String(A?.juror ?? '')
const docs = A?.docs ?? []
if (!J || docs.length < 10) throw new Error(`bad args: juror="${J}" docs=${docs.length} — refusing to spend tokens`)
const total = Math.ceil(docs.length / CHUNK)
log(`juror ${J}: ${docs.length} documents left, ${CHUNK} read at once per batch, ${total} batches, corrected and recorded after each`)

phase('Training')
const catchup = A?.catchup ?? []
if (catchup.length) log(`juror ${J}: folding ${catchup.length} already-answered-but-not-yet-corrected documents into batch 1`)
const trace = []
for (let n = 0; n < total; n++) {
  const slice = docs.slice(n * CHUNK, n * CHUNK + CHUNK)
  const answers = await parallel(slice.map(d => () => agent(SEGMENT(J, d, n + 1, total),
    { label: `J${J} b${n + 1}/${total} ${d.replace('doc_', '')}`, phase: 'Training', schema: OUT_A, model: 'opus', effort: 'max' })))
  const got = answers.filter(Boolean).length
  const gradeSet = n === 0 ? [...catchup, ...slice] : slice
  const v = await agent(LEARN(J, gradeSet, n + 1, total),
    { label: `J${J} b${n + 1}/${total} LEARN`, phase: 'Training', schema: OUT_L, model: 'opus', effort: 'max' })
  trace.push({ batch: n + 1, answered: got, of: slice.length, errors: v?.errors ?? null, f1: v?.f1 ?? '', learned: v?.learned ?? '', note: v?.note ?? '' })
  log(`J${J} batch ${n + 1}/${total}: ${got}/${slice.length} answered, ${v?.errors ?? '?'} errors | ${v?.f1 ?? ''} | ${v?.learned ?? ''}${v?.note ? ' | NOTE: ' + v.note : ''}`)
}
log(`juror ${J} training complete, spend ${budget.spent()}`)
return { juror: J, batches: trace, spent: budget.spent() }
