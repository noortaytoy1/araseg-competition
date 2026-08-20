export const meta = {
  name: 'nonp-exp3-exp2-now',
  description: 'Exp 3 (doctrine rerun, first 50 NoPnx-NP docs) then Exp 2 (gold-example juries, all 262), Opus 5 max, running now in parallel with the zero-shot control',
  phases: [{ title: 'Exp3 rerun' }, { title: 'Exp2 examples' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const P = REPO + '/scratch_exo/papereval/NoPnx-NP'
const LANES = 4
const OUT = { type: 'object', required: ['done'], properties: { done: { type: 'integer' }, note: { type: 'string' } } }
const FORMAT = 'Punctuation REMOVED and paragraph marks REMOVED, plain words only. The final token of a document always ends a sentence. \u27e8i\u27e9 means the next token is token number i (an index anchor, not part of the text).'
const NEVER = 'Never remove the \u00b6 on the final token.'
const ORDER = args.order
const SITTING = (d0, d1, outdir, docs, n, total, intro) => `This is exam sitting ${n} of ${total}. Two TRAINED juries sit together: Jury 0 and Jury 1. ${intro} You hold both: when Jury 0 speaks, reason strictly from file 0; when Jury 1 speaks, strictly from file 1. They are colleagues with different training histories, not one voice twice.

FIRST read both files ONCE:
  Jury 0: ${d0}
  Jury 1: ${d1}

FORMAT: ${FORMAT}

YOUR ${docs.length} DOCUMENTS, one file each, STRICTLY ONE AT A TIME (finish and record one before opening the next):
${docs.map((d, i) => `${i + 1}. ${P}/docs/${d}.json`).join('\n')}

Each document arrives with an automatic system's boundary marks (\u00b6) already drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust, it makes both kinds of errors. A document whose marks already satisfy its law is FINISHED; unchanged is a common and correct verdict.

PER DOCUMENT, no limit on your reasoning depth:
1. Jury 0 gives its full stance from file 0, what the text is, which register it belongs to, its sentence law, and every place the draft marks are wrong, argued from the text.
2. Jury 1 gives its own independent stance from file 1, site by site.
3. They ARGUE every disagreement. Concede when the colleague's reading matches the annotators' policy better; rebut precisely when it does not. Where a file carries evidence from THIS register, that evidence outranks instinct.
4. RECORD: only changes BOTH juries endorse survive. Unresolved means the draft stands. Write {"doc_id":"...","add":[{"i":<index>,"w":"<token>"}],"remove":[{"i":<index>,"w":"<token>"}],"argued":"<one line per argued site>"} to ${P}/${outdir}/<doc_id>.json IMMEDIATELY on finishing that document. add = token positions that end a sentence but carry no \u00b6; remove = positions carrying \u00b6 that do not end a sentence; copy w from the text at the moment you decide. ${NEVER}

RESUME: if ${P}/${outdir}/<doc_id>.json already exists, skip that document.
When all are recorded return {"done": <count>, "note": ""}. No web, no PowerShell. Touch NOTHING beyond the two files named above, your listed documents, and your own output files, answer keys for this corpus exist elsewhere in this repository and opening anything else voids the evaluation.`

async function runExam(label, phaseName, d0, d1, outdir, docs, intro, budgetCap) {
  const sittings = []
  for (let i = 0; i < docs.length; i += 4) sittings.push(docs.slice(i, i + 4))
  log(`${label}: ${docs.length} docs in ${sittings.length} sittings, ${LANES} lanes, Opus 5 max`)
  const START = budget.spent()
  let done = 0, dead = 0
  for (let p = 0; p < sittings.length; p += LANES) {
    if (budget.spent() - START > budgetCap) { log(`${label}: BUDGET HALT, ${done} done, banked`); break }
    const round = sittings.slice(p, p + LANES)
    const res = await parallel(round.map((ds, k) => () =>
      agent(SITTING(d0, d1, outdir, ds, p + k + 1, sittings.length, intro), { label: `${label} sit ${p + k + 1}/${sittings.length}`, phase: phaseName, schema: OUT, model: 'opus', effort: 'max' })
        .then(v => v?.done ?? 0)))
    const got = res.filter(Boolean).reduce((a, b) => a + b, 0)
    done += got
    dead = got === 0 ? dead + 1 : 0
    if (dead >= 2) { log(`${label}: BREAKER after two dead rounds, ${done} done, banked`); break }
    if ((p / LANES) % 4 === 0 || got === 0) log(`${label} ${p + 1}-${p + round.length}/${sittings.length}: ${got} docs (total ${done})`)
  }
  log(`${label} COMPLETE: ${done} documents`)
  return { label, juried: done, spent: budget.spent() - START }
}

phase('Exp3 rerun')
const r3 = await runExam('Exp3 doctrine rerun', 'Exp3 rerun',
  REPO + '/scratch_exo/retrain3/nonp/doctrine_j0.md', REPO + '/scratch_exo/retrain3/nonp/doctrine_j1.md',
  'exam_out_rerun', ORDER.slice(0, 50),
  'Each trained separately on this annotation team\'s training documents and wrote its own doctrine from its own graded mistakes.', 5_000_000)

phase('Exp2 examples')
const r2 = await runExam('Exp2 gold-examples', 'Exp2 examples',
  REPO + '/scratch_exo/papereval/examples_nonp/examples_j0.md', REPO + '/scratch_exo/papereval/examples_nonp/examples_j1.md',
  'exam_out_examples', ORDER,
  'Each was shown a different set of this annotation team\'s training documents with the correct sentence boundaries drawn in, and has no other training.', 16_000_000)
return { exp3: r3, exp2: r2 }