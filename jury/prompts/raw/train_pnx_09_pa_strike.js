export const meta = {
  name: 'pa-strike',
  description: 'Pnx-PA STRIKE — completed-doctrine re-exam of the genres whose law changed: per document, one sitting holds both trained jurors — each takes a stance from its own doctrine, they argue disagreements, only 2/2-endorsed corrections recorded. Punctuated, paragraph-aware track. 2 sittings at a time, probes first, budget halt, breaker on first dead pair.',
  phases: [{ title: 'Exam' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const WORK = REPO + '/scratch_exo/np_train_work'
const EXAM = REPO + '/scratch_exo/pa_strike'
const HALT_AT = 6_000_000

const OUT = { type:'object', required:['done'], properties:{ done:{type:'integer'}, note:{type:'string'} } }

const SITTING = (docs, n, total) => `This is sitting ${n} of ${total} of the AraSeg Pnx-PA blind exam (the PUNCTUATED, PARAGRAPH-AWARE track). Two TRAINED jurors sit together: Juror 0 and Juror 1. Each trained separately on ~140 graded documents of this annotation team's corpus and wrote their own doctrine. You hold both jurors: when Juror 0 speaks, reason strictly from doctrine 0; when Juror 1 speaks, strictly from doctrine 1. They are colleagues with different training histories, not the same voice twice.

FIRST, read both doctrines ONCE:
  Juror 0: ${WORK}/doctrine_np_juror0.md
  Juror 1: ${WORK}/doctrine_np_juror1.md

TRACK NOTE, for both jurors: you trained on the DE-PUNCTUATED form of this corpus. This track keeps the original punctuation and paragraph structure. Everything you learned about the annotators' units — canonical-unit sources, grids and printed layouts, prose registers and their densities — applies unchanged; but here the punctuation is REAL EVIDENCE (a period, question mark, or ؟ you see is the source page's own mark), and paragraph breaks appear as <NL> tokens. Two format laws of this track: (1) a ¶ immediately before an <NL>, and the ¶ on the final word, are paragraph ends — ALWAYS correct, never remove them, and never needed elsewhere; (2) an <NL> token itself never carries a boundary. Everything inside paragraphs is yours to judge.

YOUR ${docs.length} DOCUMENT${docs.length > 1 ? 'S' : ''}, one file each, worked STRICTLY ONE AT A TIME (finish and record one before opening the next):
${docs.map((d, i) => `${i + 1}. ${EXAM}/docs/${d}.json`).join('\n')}

Each document was already judged once by YOU; the marks drawn in include your own accepted first-pass edits — they are the current #1 submission. Since that exam your doctrines were COMPLETED: full graded law for exercise grids, conversational line-lists, narrative, poetry-in-exercise, and hadith, plus the register-scoping standing order. Re-judge under the completed law. Each document has an automatic system's boundary marks (¶) drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust. It makes both kinds of errors — marks the answer key does not have, and missing marks the key has. A document whose marks already satisfy its law is FINISHED — unchanged is a common correct verdict. Markup: ⟨i⟩ = next word is number i; <NL> = paragraph-break token with its own index; ¶ = the draft claims the preceding token ends a sentence; the final word always ends a sentence.

PER DOCUMENT, on the record, with no limit on reasoning depth:
1. JUROR 0 takes a full stance from doctrine 0: identity, the document's law, and every place the draft is wrong, argued from the text and its punctuation.
2. JUROR 1 takes their own stance from doctrine 1, independently, site by site.
3. THEY ARGUE every disagreement: concede when the colleague's reading matches the annotators' demonstrated policy better; rebut precisely when yours does. Changing your mind on good evidence is strength; stubbornness and capitulation are both failures.
4. RECORD: ONLY changes BOTH jurors endorse survive; an unresolved argument means the draft stands. Write one JSON object to ${EXAM}/out/<doc_id>.json IMMEDIATELY on finishing the document:
   {"doc_id":"...","identity":"...","law":"...","add":[{"i":<index>,"w":"<token at i>"}...],"remove":[{"i":<index>,"w":"<token at i>"}...],"argued":"<one line per argued site>"}
   add = token positions that END a sentence but carry no ¶; remove = positions carrying ¶ that do not end a sentence; never add or remove on an <NL> token and never remove a ¶ that sits immediately before an <NL> or on the final word. Copy w from the text at the moment you decide. Empty add and remove = unchanged, a respectable verdict.
If a document is very long, work it in SECTIONS across several responses, extending its output file as you go — never hold a long document for one write.

RESUME AWARENESS: if ${EXAM}/out/<doc_id>.json already exists, skip that document.
When all are recorded, return {"done": <count>, "note": "<anything wrong, else empty>"}. No web, no PowerShell. Touch nothing beyond the two doctrine files, your listed documents, and your own output files — answer keys exist elsewhere in this repo and touching anything else voids the exam.`

const A = typeof args === 'string' ? JSON.parse(args) : args
const m = await agent(`Read the file ${EXAM}/manifest.json and return {"sittings": <its "sittings" array>} verbatim. Do nothing else. No other tools.`,
  { label: 'read-manifest', phase: 'Exam', schema: { type:'object', required:['sittings'], properties:{ sittings:{ type:'array', items:{ type:'array', items:{type:'string'} } } } }, model: 'haiku', effort: 'low' })
let sittings = m?.sittings ?? []
if (!sittings.length) throw new Error('manifest read failed — refusing to spend')
if (A?.skip_done) {
  // caller signals some docs are done; sittings shrink via in-prompt resume-awareness anyway
}
log(`Pnx-PA group exam: ${sittings.reduce((a, s) => a + s.length, 0)} documents in ${sittings.length} sittings, 2 at a time, halt at ${HALT_AT.toLocaleString()}`)

phase('Exam')
const START = budget.spent()
let doneDocs = 0, failStreak = 0
for (let p = 0; p < sittings.length; p += 2) {
  if (budget.spent() - START > HALT_AT) {
    log(`BUDGET HALT: this exam spent ${(budget.spent() - START).toLocaleString()} — ${doneDocs} documents juried; the rest keep the standing rows. Probes were first; the gate is computable.`)
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
    log(`CIRCUIT BREAKER: a sitting-pair produced nothing (likely a 529 wave) — halting. ${doneDocs} juried this run; verdicts on disk.`)
    break
  }
  log(`sittings ${p + 1}-${p + pair.length}/${sittings.length}: ${got} documents recorded (run total ${doneDocs}), spend ${(budget.spent() - START).toLocaleString()}`)
}
return { juried: doneDocs, spent: budget.spent() - START }
