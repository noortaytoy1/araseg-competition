export const meta = {
  name: 'np-juror-rlhf',
  description: 'NoPnx-NP juror training by gold feedback: each juror segments 3 documents, is corrected against the gold boundaries, appends every correction to its own cumulative record, refreshes, and moves to the next 3 — until all 174 training documents are covered. Two jurors, one chunk each at a time.',
  phases: [{ title: 'Juror 0' }, { title: 'Juror 1' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const WORK = REPO + '/scratch_exo/np_train_work'
const CHUNK = 3
const OUT = { type:'object', required:['answered'], properties:{
  answered:{type:'integer'}, f1:{type:'string'}, learned:{type:'string'}, note:{type:'string'} } }

const STEP = (j, docs, n, total) => `You are Juror ${j}. You are learning to segment Arabic text the way a specific team of annotators does, by being corrected against their answer keys — one small batch of documents at a time. This is batch ${n} of ${total}.

Everything you have learned so far from being corrected is written in ONE file, and it is the only thing that survives between batches: ${WORK}/lessons_np_juror${j}.md

=== STEP 1 — RECALL ===
Read ${WORK}/lessons_np_juror${j}.md if it exists. That is your own accumulated record of every correction you have absorbed so far. Hold all of it in front of you while you work. If it does not exist yet, this is your first batch and you work from your own knowledge of Arabic alone.

=== STEP 2 — SEGMENT ===
Your ${docs.length} documents for this batch:
${docs.map((d, i) => `${i + 1}. ${REPO}/scratch_exo/np_docs/${d}.json`).join('\n')}

Work them STRICTLY ONE AT A TIME. Open one, read it once, think it through completely, write its answer file, and only then open the next. Never read ahead; nothing you learn from one of these documents is evidence about another.

One structural fact you may rely on: the final word of a document always ends a sentence. Everything else is your judgment, informed by your accumulated record.

Think as long and as hard as each document deserves — there is no limit on your reasoning and depth is the entire point. A thin answer is the failure mode. For each document work out: what IS this text (identity); what is its natural sentence unit (law); the evidence for both, quoting short phrases; and at every uncertain site the full argument — the reading you chose, the reading you rejected, and why these annotators would agree with you.

Write each answer the moment that document is done, as a single JSON object to ${WORK}/ans/j${j}/<doc_id>.json with exactly the keys doc_id, identity, law, reasoning, boundaries. Each boundary is {i, w, c, why} where i is the index of the LAST word of a sentence, w is that word copied from the text at the moment you decide, c is hi (would bet on it) / med (uncertain) / lo (lean against but flagging). If a document is long, work it in sections across several responses, extending its answer file as you go — never try to emit a long document in one response.

=== STEP 3 — BE CORRECTED ===
When all ${docs.length} answer files exist, run exactly this from ${REPO}:
  python scratch_exo/grade_practice_np.py --juror ${j} --only ${docs.join(',')}
Then read ${WORK}/mistakes_np_juror${j}_chunk.txt — the answer key's verdict on the documents you just did. Every boundary you invented that the key does not have, and every boundary the key has that you missed, each shown in context beside what the key actually says.

Study it. For each error, work out what you were thinking versus what the annotators' policy actually is. The errors are the curriculum. Look for the RULE behind an error, not the individual site — and check it against your accumulated record: does it confirm something you already wrote, contradict it, or is it new?

=== STEP 4 — RECORD WHAT YOU LEARNED ===
Append what you learned to ${WORK}/lessons_np_juror${j}.md. This file is CUMULATIVE and APPEND-ONLY — it is the full record of every correction you have ever absorbed, and it is what you will carry into the real exam. Rules:
  * Never delete an entry. Never summarise away an earlier lesson to make room.
  * Add each new correction as: the rule, stated as what the ANNOTATORS do, plus the short example from your own error that produced it, filed under its topic heading (expository, narrative, exercise, hadith, quran, bible, poetry, conversational, legal).
  * You may sharpen or merge an existing entry ONLY when this batch proves it wrong or exactly duplicates it — and when you do, say in one line what changed and why.
  * If you made no errors on a document, note in one line what you got right and what law worked, so the record keeps the confirmed rules too.
Use Edit to append to the file (or Write if it does not exist yet). The file is the record — it must be on disk before you finish.

Return {"answered": <count of answer files you wrote>, "f1": "<the grader's printed F1 line for you>", "learned": "<one sentence on what this batch taught you>", "note": "<anything that went wrong, else empty>"}.
Do not open any data file, answer key, or any document outside your ${docs.length} listed above. No web.`

const A = typeof args === 'string' ? JSON.parse(args) : args
const L = { '0': A?.['0'] ?? [], '1': A?.['1'] ?? [] }
if (L['0'].length < 100 || L['1'].length < 100) throw new Error(`lists did not parse: J0=${L['0'].length} J1=${L['1'].length} — refusing to spend tokens`)
log(`gold-feedback training: J0 ${L['0'].length} docs, J1 ${L['1'].length} docs, ${CHUNK} per batch, correction + record after every batch`)

async function juror(j) {
  const docs = L[j]
  const total = Math.ceil(docs.length / CHUNK)
  const trace = []
  for (let n = 0; n < total; n++) {
    const slice = docs.slice(n * CHUNK, n * CHUNK + CHUNK)
    const v = await agent(STEP(j, slice, n + 1, total),
      { label: `J${j} batch ${n + 1}/${total}`, phase: `Juror ${j}`, schema: OUT, model: 'opus', effort: 'max' })
    trace.push({ batch: n + 1, answered: v?.answered ?? 0, f1: v?.f1 ?? '', learned: v?.learned ?? '', note: v?.note ?? '' })
    log(`J${j} batch ${n + 1}/${total}: ${v?.answered ?? 0}/${slice.length} answered | ${v?.f1 ?? ''} | ${v?.learned ?? ''}${v?.note ? ' | NOTE: ' + v.note : ''}`)
  }
  return { juror: j, batches: trace }
}

const out = await parallel([() => juror('0'), () => juror('1')])
log(`training complete, spend ${budget.spent()}`)
return { result: out.filter(Boolean), spent: budget.spent() }
