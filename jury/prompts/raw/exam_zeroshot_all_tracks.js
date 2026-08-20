export const meta = {
  name: 'papereval-zeroshot-control',
  description: 'Pure zero-shot control: two untrained juries, NO doctrine step, same prompt structure/packets/drafts/unanimity rule as the ablation; all four tracks, Opus 5 max only',
  phases: [{ title: 'NoPnx-NP' }, { title: 'NoPnx-PA' }, { title: 'NP' }, { title: 'PA' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const LANES = 4
const OUT = { type: 'object', required: ['done'], properties: { done: { type: 'integer' }, note: { type: 'string' } } }
const TRACKS = [
  { phase: 'NoPnx-NP', packet: 'papereval/NoPnx-NP',
    format: 'Punctuation REMOVED and paragraph marks REMOVED, plain words only. The final token of a document always ends a sentence. \u27e8i\u27e9 means the next token is token number i (an index anchor, not part of the text).',
    never: 'Never remove the \u00b6 on the final token.' },
  { phase: 'NoPnx-PA', packet: 'papereval/NoPnx-PA',
    format: 'Punctuation REMOVED and paragraph structure KEPT as literal <NL> tokens. The \u00b6 on the token immediately before every <NL>, and on the final token, is ALWAYS correct and must never be removed; an <NL> token itself NEVER carries a boundary. \u27e8i\u27e9 = index anchor, not part of the text.',
    never: 'Never add or remove on an <NL> token; never remove a \u00b6 that sits immediately before an <NL> or on the final token.' },
  { phase: 'NP', packet: 'papereval/NP',
    format: 'Punctuation KEPT: punctuation marks are ordinary tokens with their own indices, and a sentence usually ends ON its punctuation mark when one is present. NO <NL> tokens. The final token always ends a sentence. \u27e8i\u27e9 = index anchor, not part of the text.',
    never: 'Never remove the \u00b6 on the final token.' },
  { phase: 'PA', packet: 'papereval/PA',
    format: 'Punctuation KEPT (marks are ordinary tokens, and a sentence usually ends ON its mark) AND paragraph structure KEPT as literal <NL> tokens. The \u00b6 on the token immediately before every <NL>, and on the final token, is ALWAYS correct and must never be removed; an <NL> token itself NEVER carries a boundary. \u27e8i\u27e9 = index anchor, not part of the text.',
    never: 'Never add or remove on an <NL> token; never remove a \u00b6 that sits immediately before an <NL> or on the final token.' },
]
const ORDERS = args.orders
const SITTING = (tr, docs, n, total) => {
  const P = REPO + '/scratch_exo/' + tr.packet
  return `This is exam sitting ${n} of ${total}. Two juries sit together: Jury 0 and Jury 1. Neither has been trained on this corpus and neither holds any written policy or examples from it; each reasons only from its own knowledge of Arabic and from the text in front of it. You hold both: when Jury 0 speaks, reason as Jury 0; when Jury 1 speaks, reason independently as Jury 1. They are colleagues who reach their own readings, not one voice twice.

FORMAT: ${tr.format}

YOUR ${docs.length} DOCUMENTS, one file each, STRICTLY ONE AT A TIME (finish and record one before opening the next):
${docs.map((d, i) => `${i + 1}. ${P}/docs/${d}.json`).join('\n')}

Each document arrives with an automatic system's boundary marks (\u00b6) already drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust, it makes both kinds of errors. A document whose marks already satisfy its law is FINISHED; unchanged is a common and correct verdict.

PER DOCUMENT, no limit on your reasoning depth:
1. Jury 0 gives its full stance, what the text is, which register it belongs to, its sentence law, and every place the draft marks are wrong, argued from the text.
2. Jury 1 gives its own independent stance, site by site.
3. They ARGUE every disagreement. Concede when the colleague's reading is better supported by the text; rebut precisely when it is not.
4. RECORD: only changes BOTH juries endorse survive. Unresolved means the draft stands. Write {"doc_id":"...","add":[{"i":<index>,"w":"<token>"}],"remove":[{"i":<index>,"w":"<token>"}],"argued":"<one line per argued site>"} to ${P}/exam_out_zeroshot/<doc_id>.json IMMEDIATELY on finishing that document. add = token positions that end a sentence but carry no \u00b6; remove = positions carrying \u00b6 that do not end a sentence; copy w from the text at the moment you decide. ${tr.never}

RESUME: if ${P}/exam_out_zeroshot/<doc_id>.json already exists, skip that document.
When all are recorded return {"done": <count>, "note": ""}. No web, no PowerShell. Touch NOTHING beyond your listed documents and your own output files, answer keys for this corpus exist elsewhere in this repository and opening anything else voids the evaluation.`
}
const summary = []
for (const tr of TRACKS) {
  phase(tr.phase)
  const order = ORDERS[tr.phase]
  const sittings = []
  for (let i = 0; i < order.length; i += 4) sittings.push(order.slice(i, i + 4))
  log(`${tr.phase} zero-shot: ${order.length} docs in ${sittings.length} sittings, ${LANES} lanes, Opus 5 max`)
  const START = budget.spent()
  let done = 0, dead = 0
  for (let p = 0; p < sittings.length; p += LANES) {
    if (budget.spent() - START > 16_000_000) { log(`${tr.phase}: BUDGET HALT, ${done} done, banked`); break }
    const round = sittings.slice(p, p + LANES)
    const res = await parallel(round.map((docs, k) => () =>
      agent(SITTING(tr, docs, p + k + 1, sittings.length), { label: `${tr.phase} zs sit ${p + k + 1}/${sittings.length}`, phase: tr.phase, schema: OUT, model: 'opus', effort: 'max' })
        .then(v => v?.done ?? 0)))
    const got = res.filter(Boolean).reduce((a, b) => a + b, 0)
    done += got
    dead = got === 0 ? dead + 1 : 0
    if (dead >= 2) { log(`${tr.phase}: BREAKER after two dead rounds, ${done} done, banked`); break }
    if ((p / LANES) % 4 === 0 || got === 0) log(`${tr.phase} ${p + 1}-${p + round.length}/${sittings.length}: ${got} docs (total ${done}), spend ${(budget.spent() - START).toLocaleString()}`)
  }
  summary.push({ track: tr.phase, juried: done, spent: budget.spent() - START })
  log(`${tr.phase} zero-shot COMPLETE: ${done} documents`)
}
return summary