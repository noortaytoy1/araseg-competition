export const meta = {
  name: 'np-group-exam',
  description: 'NoPnx-NP GROUP EXAM: per document, one sitting holds both trained jurors — Juror 0 takes a stance from its doctrine, Juror 1 from its own, they argue the disagreements, and only corrections both endorse are recorded. 2 sittings run at a time. Probes first. Hard budget halt.',
  phases: [{ title: 'Exam' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const WORK = REPO + '/scratch_exo/np_train_work'
const EXAM = REPO + '/scratch_exo/np_exam'
const CHUNK = 6
const HALT_AT = 12_000_000   // Noor chose full coverage after price disclosure (~11M at measured burn)

const OUT = { type:'object', required:['done'], properties:{ done:{type:'integer'}, note:{type:'string'} } }

const SITTING = (docs, n, total) => `This is sitting ${n} of ${total} of the AraSeg NoPnx-NP blind exam. Two TRAINED jurors sit together in this room: Juror 0 and Juror 1. Each trained separately on ~140 graded documents of this annotation team's corpus and wrote their own doctrine. You hold both jurors: when Juror 0 speaks, reason strictly from doctrine 0; when Juror 1 speaks, strictly from doctrine 1. They are colleagues with different training histories, not the same voice twice.

FIRST, read both doctrines ONCE:
  Juror 0: ${WORK}/doctrine_np_juror0.md
  Juror 1: ${WORK}/doctrine_np_juror1.md

YOUR ${docs.length} DOCUMENTS, one file each, worked STRICTLY ONE AT A TIME (finish and record one before opening the next; nothing from one document is evidence about another):
${docs.map((d, i) => `${i + 1}. ${EXAM}/docs/${d}.json`).join('\n')}

Each document has an automatic system's boundary marks (¶) drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust. It makes both kinds of errors — boundaries the answer key does not have, and missing ones the key has. It is evidence of where a boundary is plausible, never evidence that one is correct. A document whose marks already satisfy its law is FINISHED — unchanged is a common correct verdict. The one structural fact you may rely on: the final word of a document always ends a sentence. Markup: ⟨i⟩ = next word is number i; ¶ = the draft claims the preceding word ends a sentence.

PER DOCUMENT, on the record, with no limit on reasoning depth:
1. JUROR 0 reads the document and takes a full stance from doctrine 0: identity, the document's law, and every place the draft is wrong (a mark to remove, a missing mark to add), argued from the text.
2. JUROR 1 then takes their own stance from doctrine 1, independently argued — agreeing or disagreeing with both the draft and Juror 0, site by site.
3. THEY ARGUE every disagreement: concede when the colleague's reading matches the annotators' demonstrated policy better; rebut precisely when yours does. Changing your mind on good evidence is strength; stubbornness and capitulation are both failures.
4. RECORD the verdict: ONLY changes BOTH jurors endorse after argument survive. An unresolved argument means the draft stands at that site. Write one JSON object to ${EXAM}/out/<doc_id>.json IMMEDIATELY on finishing the document, before opening the next:
   {"doc_id":"...","identity":"<what this text is>","law":"<its sentence law>","add":[{"i":<index>,"w":"<word at i>"}...],"remove":[{"i":<index>,"w":"<word at i>"}...],"argued":"<one line per argued site: who conceded and why>"}
   add = word positions that END a sentence but carry no ¶ in the draft; remove = positions carrying ¶ that do not end a sentence; copy w from the text at the moment you decide. Empty add and remove = unchanged, a respectable verdict.

RESUME AWARENESS: if ${EXAM}/out/<doc_id>.json already exists, skip that document.
When all ${docs.length} are recorded, return {"done": <count>, "note": "<anything wrong, else empty>"}. No web, no PowerShell. Touch nothing beyond the two doctrine files, your listed document files, and your own output files — answer keys exist elsewhere in this repo and touching anything else voids the exam.`

const A = typeof args === 'string' ? JSON.parse(args) : args
let order = A?.order ?? []
if (A?.use_remaining) {
  const m = await agent(`Read the file ${EXAM}/remaining_order.json (a JSON array of document ids) and return it verbatim as {"order": <the array>}. Do nothing else. No other tools.`,
    { label: 'read-remaining', phase: 'Exam', schema: { type:'object', required:['order'], properties:{ order:{ type:'array', items:{type:'string'} } } }, model: 'haiku', effort: 'low' })
  if (!m?.order || m.order.length < 20) throw new Error(`remaining_order read failed: ${m?.order?.length ?? 0}`)
  order = m.order
} else if (A?.use_manifest) {
  // the full 222-doc order lives in the manifest; args carries only the probe head as a checksum
  const m = await agent(`Read the file ${EXAM}/manifest.json and return its "order" array verbatim as your structured result. Do nothing else. No other tools.`,
    { label: 'read-manifest', phase: 'Exam', schema: { type:'object', required:['order'], properties:{ order:{ type:'array', items:{type:'string'} } } }, model: 'haiku', effort: 'low' })
  if (!m?.order || m.order.length < 200) throw new Error(`manifest read failed: got ${m?.order?.length ?? 0} docs — refusing to spend`)
  for (let i = 0; i < A.order.length; i++) if (m.order[i] !== A.order[i]) throw new Error('manifest/probe-head mismatch — refusing to spend')
  order = m.order
}
if (order.length < 20) throw new Error(`order did not parse: ${order.length} — refusing to spend`)

const sittings = []
for (let i = 0; i < order.length; i += CHUNK) sittings.push(order.slice(i, i + CHUNK))
log(`group exam: ${order.length} documents in ${sittings.length} sittings of up to ${CHUNK}, 2 sittings at a time, halt at ${HALT_AT.toLocaleString()} tokens`)

phase('Exam')
const START = budget.spent()   // spent() is turn-wide across all workflows; measure only THIS exam's spend
let doneDocs = 0, failStreak = 0
for (let p = 0; p < sittings.length; p += 2) {
  if (budget.spent() - START > HALT_AT) {
    log(`BUDGET HALT: this exam has spent ${(budget.spent() - START).toLocaleString()} — ${doneDocs}/${order.length} documents juried; the rest keep the standing 88.30 rows. Probes were first, so the gate is computable.`)
    break
  }
  const pair = sittings.slice(p, p + 2)
  const res = await parallel(pair.map((docs, k) => () => agent(SITTING(docs, p + k + 1, sittings.length),
    { label: `sitting ${p + k + 1}/${sittings.length} (${docs.length}d)`, phase: 'Exam', schema: OUT, model: 'opus', effort: 'max' })
    .then(v => v?.done ?? 0)))
  const got = res.filter(Boolean).reduce((a, b) => a + b, 0)
  doneDocs += got
  failStreak = got === 0 ? failStreak + 1 : 0
  if (failStreak >= 1) {
    log(`CIRCUIT BREAKER: a sitting-pair produced nothing (likely a 529 wave) — halting immediately to stop paying into it. ${doneDocs}/${order.length} juried; completed verdicts are on disk.`)
    break
  }
  log(`sittings ${p + 1}-${p + pair.length}/${sittings.length}: ${got} documents recorded (total ${doneDocs}/${order.length}), spend ${budget.spent().toLocaleString()}`)
}
return { juried: doneDocs, of: order.length, spent: budget.spent() }
