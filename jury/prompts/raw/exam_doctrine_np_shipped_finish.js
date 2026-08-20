export const meta = {
  name: 'papereval-np-shipped-finish',
  description: 'Finish the last Pnx-NP ablation docs with the SHIPPED juries: rebuild queue from manifest minus exam_out_shipped, protocol verbatim from the np-shipped run',
  phases: [{ title: 'Pnx-NP shipped' }],
}

const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const P = REPO + '/scratch_exo/papereval/NP'
const D0 = REPO + '/scratch_exo/np_train_work/doctrine_np_juror0.md'
const D1 = REPO + '/scratch_exo/np_train_work/doctrine_np_juror1.md'
const LANES = 4
const OUT = { type: 'object', required: ['done'], properties: { done: { type: 'integer' }, note: { type: 'string' } } }

const SITTING = (docs, n, total) => `This is exam sitting ${n} of ${total}. Two TRAINED juries sit together: Jury 0 and Jury 1. Each trained separately on this annotation team's training documents and wrote its own doctrine from its own graded mistakes. You hold both: when Jury 0 speaks, reason strictly from doctrine 0; when Jury 1 speaks, strictly from doctrine 1. They are colleagues with different training histories, not one voice twice.

FIRST read both doctrines ONCE:
  Jury 0: ${D0}
  Jury 1: ${D1}

FORMAT: Punctuation KEPT. Punctuation marks are ordinary tokens with their own indices, and a sentence usually ends ON its punctuation mark when one is present. There are NO <NL> tokens in this track. The final token always ends a sentence. \u27e8i\u27e9 is an index anchor marking that the next token is token number i; it is not part of the text.

YOUR ${docs.length} DOCUMENTS, one file each, STRICTLY ONE AT A TIME (finish and record one before opening the next):
${docs.map((d, i) => `${i + 1}. ${P}/docs/${d}.json`).join('\n')}

Each document arrives with an automatic system's boundary marks (\u00b6) already drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust. A document whose marks already satisfy its law is FINISHED; unchanged is a common and correct verdict.

PER DOCUMENT, no limit on your reasoning depth:
1. Jury 0 gives its full stance from doctrine 0: what the text is, which kind of text it belongs to, its sentence law, and every place the draft marks are wrong, argued from the text.
2. Jury 1 gives its own independent stance from doctrine 1, site by site.
3. They ARGUE every disagreement. Concede when the colleague's reading matches the annotators' policy better; rebut precisely when it does not. A rule earned from a graded error in this kind of text outranks an unsupported intuition.
4. RECORD: only changes BOTH juries endorse survive. Unresolved means the draft stands. Write {"doc_id":"...","add":[{"i":<index>,"w":"<token>"}],"remove":[{"i":<index>,"w":"<token>"}],"argued":"<one line per argued site>"} to ${P}/exam_out_shipped/<doc_id>.json IMMEDIATELY on finishing that document. add = token positions that end a sentence but carry no \u00b6; remove = positions carrying \u00b6 that do not end a sentence; copy w from the text. Never remove the \u00b6 on the final token.

RESUME: if ${P}/exam_out_shipped/<doc_id>.json already exists, skip that document.
When all are recorded return {"done": <count>, "note": ""}. No web, no PowerShell. Touch NOTHING beyond the two doctrine files, your listed documents, and your own output files.`

const m = await agent(`Read ${P}/manifest.json, list the directory ${P}/exam_out_shipped/, take every doc_id in "order" that has no <doc_id>.json in exam_out_shipped, split them into consecutive groups of 4, and return them as {"sittings": [[...],[...]]}. Return the JSON only.`,
  { label: 'build queue', phase: 'Pnx-NP shipped', schema: { type: 'object', required: ['sittings'], properties: { sittings: { type: 'array', items: { type: 'array', items: { type: 'string' } } } } }, model: 'haiku', effort: 'low' })
const sittings = (m?.sittings ?? []).filter(s => s && s.length)
if (!sittings.length) { log('Pnx-NP shipped: already complete'); return { juried: 0, note: 'already complete' } }
log(`Pnx-NP SHIPPED finish: ${sittings.reduce((a, s) => a + s.length, 0)} documents in ${sittings.length} sittings, ${LANES} lanes`)

phase('Pnx-NP shipped')
const START = budget.spent()
let done = 0, dead = 0
for (let p = 0; p < sittings.length; p += LANES) {
  if (budget.spent() - START > 14_000_000) { log(`BUDGET HALT: ${done} done`); break }
  const round = sittings.slice(p, p + LANES)
  const res = await parallel(round.map((docs, k) => () =>
    agent(SITTING(docs, p + k + 1, sittings.length), { label: `NP-shipped sit ${p + k + 1}/${sittings.length}`, phase: 'Pnx-NP shipped', schema: OUT, model: 'opus', effort: 'max' })
      .then(v => v?.done ?? 0)))
  const got = res.filter(Boolean).reduce((a, b) => a + b, 0)
  done += got
  dead = got === 0 ? dead + 1 : 0
  if (dead >= 2) { log(`BREAKER after two dead rounds: ${done} done, banked`); break }
}
log(`Pnx-NP shipped COMPLETE: ${done} more documents juried`)
return { juried: done, spent: budget.spent() - START }