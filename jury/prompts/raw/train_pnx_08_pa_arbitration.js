export const meta = {
  name: 'pa-arbitration',
  description: 'Pnx-PA ARBITRATION: the two trained jurors reconvene on the documents whose first-exam arguments ended unresolved, with the prior argument transcript on the table, and talk each contested site to a final ruling. 2 sittings at a time, probes first.',
  phases: [{ title: 'Arbitrate' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const WORK = REPO + '/scratch_exo/np_train_work'
const ARB = REPO + '/scratch_exo/pa_arb'
const HALT_AT = 2_500_000

const OUT = { type:'object', required:['done'], properties:{ done:{type:'integer'}, note:{type:'string'} } }

const SITTING = (docs, n, total) => `This is arbitration sitting ${n} of ${total} for the AraSeg Pnx-PA blind exam. The two TRAINED jurors — Juror 0 and Juror 1 — reconvene on documents they have already examined once. In the first exam their rule was conservative: where their argument ended without consensus, the draft stood. Today those contested sites get final rulings. Both doctrines have since been EXTENDED with newly graded law — that new law is often exactly what settles an old argument.

FIRST, read both doctrines ONCE (including their PA APPENDIX sections, newly extended):
  Juror 0: ${WORK}/doctrine_np_juror0.md
  Juror 1: ${WORK}/doctrine_np_juror1.md

YOUR ${docs.length} DOCUMENT${docs.length > 1 ? 'S' : ''}, one file each, STRICTLY ONE AT A TIME:
${docs.map((d, i) => `${i + 1}. ${ARB}/docs/${d}.json`).join('\n')}

Each file holds: the document with its CURRENT boundary marks (¶) — these already include the first exam's accepted edits — and "prior_arguments": the record of what the two jurors argued last time, who conceded where, and which disputes died unresolved.

Track format: ⟨i⟩ = next token is number i; <NL> = paragraph token with its own index (never a boundary; the ¶ before an <NL> and on the final token are untouchable); punctuation marks are tokens; density measured in WORDS.

PER DOCUMENT, on the record, unlimited reasoning:
1. Read the document and the prior_arguments record. Identify every site that was argued — especially those that ended unresolved or in a grudging concession.
2. THE JURORS TALK: for each such site, Juror 0 argues from doctrine 0 (appendix included), Juror 1 from doctrine 1. Where either doctrine now carries GRADED law covering the site — a rule earned from a scored mistake, not an instinct — that law carries the ruling. Concede to graded law; rebut instinct with evidence. They may also flag a NEW site if the extended appendices reveal a clear error the first exam missed — but only with graded law behind it.
3. RULE: every argued site ends in a ruling — keep the current mark, add a boundary, or remove one. "Unresolved" is not a verdict today: where neither doctrine has graded law for the site and the text itself does not decide, the ruling is KEEP CURRENT, stated as such.
4. RECORD to ${ARB}/out/<doc_id>.json IMMEDIATELY on finishing the document:
   {"doc_id":"...","add":[{"i":<index>,"w":"<token>"}...],"remove":[{"i":<index>,"w":"<token>"}...],"rulings":"<one line per argued site: the ruling and which doctrine's law carried it>"}
   add/remove are relative to the CURRENT marks shown in the document. Never touch an <NL> token, the pre-<NL> ¶, or the final token. Empty add and remove = every ruling kept current, a legitimate outcome.

RESUME AWARENESS: if ${ARB}/out/<doc_id>.json exists, skip that document.
When all are recorded, return {"done": <count>, "note": ""}. No web, no PowerShell. Touch nothing beyond the two doctrine files, your listed documents, and your own output files.`

const m = await agent(`Read the file ${ARB}/manifest.json and return {"sittings": <its "sittings" array>} verbatim. Do nothing else. No other tools.`,
  { label: 'read-manifest', phase: 'Arbitrate', schema: { type:'object', required:['sittings'], properties:{ sittings:{ type:'array', items:{ type:'array', items:{type:'string'} } } } }, model: 'haiku', effort: 'low' })
const sittings = m?.sittings ?? []
if (!sittings.length) throw new Error('manifest read failed — refusing to spend')
log(`arbitration: ${sittings.reduce((a, s) => a + s.length, 0)} documents in ${sittings.length} sittings`)

phase('Arbitrate')
const START = budget.spent()
let done = 0, failStreak = 0
for (let p = 0; p < sittings.length; p += 2) {
  if (budget.spent() - START > HALT_AT) { log(`BUDGET HALT at ${(budget.spent() - START).toLocaleString()}: ${done} done`); break }
  const pair = sittings.slice(p, p + 2)
  const res = await parallel(pair.map((docs, k) => () => agent(SITTING(docs, p + k + 1, sittings.length),
    { label: `arb ${p + k + 1}/${sittings.length} (${docs.length}d)`, phase: 'Arbitrate', schema: OUT, model: 'opus', effort: 'max' })
    .then(v => v?.done ?? 0)))
  const got = res.filter(Boolean).reduce((a, b) => a + b, 0)
  done += got
  failStreak = got === 0 ? failStreak + 1 : 0
  if (failStreak >= 1) { log(`CIRCUIT BREAKER after dead pair — ${done} done, verdicts banked`); break }
  log(`arb sittings ${p + 1}-${p + pair.length}/${sittings.length}: ${got} docs (total ${done}), spend ${(budget.spent() - START).toLocaleString()}`)
}
return { arbitrated: done, spent: budget.spent() - START }
