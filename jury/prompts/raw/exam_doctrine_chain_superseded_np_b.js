export const meta = {
  name: 'papereval-chain-nonp-pnxnp-pnxpa',
  description: 'Paper ablation chain: NoPnx-NP, then Pnx-NP, then Pnx-PA — one track at a time, 3 sittings at a time',
  phases: [{ title: 'NoPnx-NP' }, { title: 'Pnx-NP' }, { title: 'Pnx-PA' }],
}

const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const LANES = 3
const OUT = { type: 'object', required: ['done'], properties: { done: { type: 'integer' }, note: { type: 'string' } } }

const TRACKS = [
  {
    phase: 'NoPnx-NP', packet: 'papereval/NoPnx-NP', d0: 'scratch_exo/retrain3/nonp/doctrine_j0.md', d1: 'scratch_exo/retrain3/nonp/doctrine_j1.md',
    format: 'Punctuation REMOVED and paragraph marks REMOVED — plain words only. The final token of a document always ends a sentence. \u27e8i\u27e9 means the next token is token number i (an index anchor, not part of the text).',
    never: 'Never remove the \u00b6 on the final token.',
  },
  {
    phase: 'Pnx-NP', packet: 'papereval/NP', d0: 'scratch_exo/retrain3/pnxnp/doctrine_j0.md', d1: 'scratch_exo/retrain3/pnxnp/doctrine_j1.md',
    format: 'Punctuation KEPT — punctuation marks are ordinary tokens with their own indices, and a sentence usually ends ON its punctuation mark when one is present. NO <NL> tokens. The final token always ends a sentence. \u27e8i\u27e9 = index anchor, not part of the text.',
    never: 'Never remove the \u00b6 on the final token.',
  },
  {
    phase: 'Pnx-PA', packet: 'papereval/PA', d0: 'scratch_exo/np_train_work/doctrine_np_juror0.md', d1: 'scratch_exo/np_train_work/doctrine_np_juror1.md',
    format: 'Punctuation KEPT (marks are ordinary tokens, and a sentence usually ends ON its mark) AND paragraph structure KEPT as literal <NL> tokens. The \u00b6 on the token immediately before every <NL>, and on the final token, is ALWAYS correct and must never be removed; an <NL> token itself NEVER carries a boundary. \u27e8i\u27e9 = index anchor, not part of the text.',
    never: 'Never add or remove on an <NL> token; never remove a \u00b6 that sits immediately before an <NL> or on the final token.',
  },
]

const SITTING = (tr, docs, n, total) => {
  const P = REPO + '/scratch_exo/' + tr.packet
  return `This is exam sitting ${n} of ${total}. Two TRAINED juries sit together: Jury 0 and Jury 1. Each trained separately on this annotation team's training documents and wrote its own doctrine from its own graded mistakes. You hold both: when Jury 0 speaks, reason strictly from doctrine 0; when Jury 1 speaks, strictly from doctrine 1. They are colleagues with different training histories, not one voice twice.

FIRST read both doctrines ONCE:
  Jury 0: ${REPO}/${tr.d0}
  Jury 1: ${REPO}/${tr.d1}

FORMAT: ${tr.format}

YOUR ${docs.length} DOCUMENTS, one file each, STRICTLY ONE AT A TIME (finish and record one before opening the next):
${docs.map((d, i) => `${i + 1}. ${P}/docs/${d}.json`).join('\n')}

Each document arrives with an automatic system's boundary marks (\u00b6) already drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust — it makes both kinds of errors. A document whose marks already satisfy its law is FINISHED; unchanged is a common and correct verdict.

PER DOCUMENT, no limit on your reasoning depth:
1. Jury 0 gives its full stance from doctrine 0 — what the text is, which register it belongs to, its sentence law, and every place the draft marks are wrong, argued from the text.
2. Jury 1 gives its own independent stance from doctrine 1, site by site.
3. They ARGUE every disagreement. Concede when the colleague's reading matches the annotators' policy better; rebut precisely when it does not. Where a doctrine carries a law earned from a graded mistake in THIS register, that law outranks instinct.
4. RECORD: only changes BOTH juries endorse survive. Unresolved means the draft stands. Write {"doc_id":"...","add":[{"i":<index>,"w":"<token>"}],"remove":[{"i":<index>,"w":"<token>"}],"argued":"<one line per argued site>"} to ${P}/exam_out/<doc_id>.json IMMEDIATELY on finishing that document. add = token positions that end a sentence but carry no \u00b6; remove = positions carrying \u00b6 that do not end a sentence; copy w from the text at the moment you decide. ${tr.never}

RESUME: if ${P}/exam_out/<doc_id>.json already exists, skip that document.
When all are recorded return {"done": <count>, "note": ""}. No web, no PowerShell. Touch NOTHING beyond the two doctrine files, your listed documents, and your own output files — answer keys for this corpus exist elsewhere in this repository and opening anything else voids the evaluation.`
}

const summary = []
for (const tr of TRACKS) {
  const P = REPO + '/scratch_exo/' + tr.packet
  phase(tr.phase)
  const m = await agent(`Read ${P}/resume_manifest.json if it exists and return {"sittings": <its "sittings" array>}. If that file does NOT exist, instead read ${P}/manifest.json, list the directory ${P}/exam_out/, take every doc_id in "order" that has no <doc_id>.json in exam_out, split them into consecutive groups of 4, and return them as {"sittings": [[...],[...]]}. Return the JSON only.`,
    { label: `${tr.phase}: build queue`, phase: tr.phase, schema: { type: 'object', required: ['sittings'], properties: { sittings: { type: 'array', items: { type: 'array', items: { type: 'string' } } } } }, model: 'haiku', effort: 'low' })
  const sittings = (m?.sittings ?? []).filter(s => s && s.length)
  if (!sittings.length) { log(`${tr.phase}: nothing to do (already complete)`); summary.push({ track: tr.phase, juried: 0, note: 'already complete' }); continue }
  log(`${tr.phase}: ${sittings.reduce((a, s) => a + s.length, 0)} documents in ${sittings.length} sittings, ${LANES} at a time`)

  const START = budget.spent()
  let done = 0, dead = 0
  for (let p = 0; p < sittings.length; p += LANES) {
    if (budget.spent() - START > 14_000_000) { log(`${tr.phase}: BUDGET HALT at ${(budget.spent() - START).toLocaleString()} — ${done} done, banked`); break }
    const round = sittings.slice(p, p + LANES)
    const res = await parallel(round.map((docs, k) => () =>
      agent(SITTING(tr, docs, p + k + 1, sittings.length), { label: `${tr.phase} sit ${p + k + 1}/${sittings.length}`, phase: tr.phase, schema: OUT, model: 'opus', effort: 'max' })
        .then(v => v?.done ?? 0)))
    const got = res.filter(Boolean).reduce((a, b) => a + b, 0)
    done += got
    dead = got === 0 ? dead + 1 : 0
    if (dead >= 2) { log(`${tr.phase}: CIRCUIT BREAKER after two dead rounds — ${done} done, banked`); break }
    if ((p / LANES) % 4 === 0 || got === 0) log(`${tr.phase} ${p + 1}-${p + round.length}/${sittings.length}: ${got} docs (total ${done}), spend ${(budget.spent() - START).toLocaleString()}`)
  }
  summary.push({ track: tr.phase, juried: done, spent: budget.spent() - START })
  log(`${tr.phase} COMPLETE: ${done} documents juried`)
}
return summary