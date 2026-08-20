export const meta = {
  name: 'papereval-nopa-redo26',
  description: 'Re-adjudicate the 26 NoPnx-PA released-test docs that a pre-retrain launcher had judged: retrain4 doctrines, exact counted-run sitting prompt, 2 lanes',
  phases: [{ title: 'Redo' }],
}

const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const D = REPO + '/scratch_exo/retrain4/nopa'
const P = REPO + '/scratch_exo/papereval/NoPnx-PA'
const LANES = 2
const OUT = { type: 'object', required: ['done'], properties: { done: { type: 'integer' }, note: { type: 'string' } } }

const ORDER = ["doc_00f2233edc0d","doc_00f88da2b078","doc_017a890a1353","doc_0320579c9b44","doc_03f245c53dbb","doc_0543259bfa3d","doc_068a21144901","doc_08492bb4e2e0","doc_095943cddbd5","doc_09ce68ed38ae","doc_0bc8763ff800","doc_0bd6abf67d39","doc_0bf9145e7a1e","doc_0c3f03cb7410","doc_0c9cfacdacca","doc_0cef21e58564","doc_0d6258e8011b","doc_0f6feafed945","doc_0fd3dd0e1c7f","doc_0fddaa619a23","doc_100dc983c8cf","doc_10f00a5be838","doc_1356da2f70fb","doc_1ba40b096385","doc_1e9d85470458","doc_23d6a0b477dc"]

const SITTING = (docs, n, total) => `This is exam sitting ${n} of ${total}. Two TRAINED juries sit together: Jury 0 and Jury 1. Each trained separately, on its own half of this annotation team's training corpus, and wrote its own doctrine from its own graded mistakes. You hold both: when Jury 0 speaks, reason strictly from doctrine 0; when Jury 1 speaks, strictly from doctrine 1. They are colleagues with different training histories, not one voice twice.

FIRST read both doctrines ONCE:
  Jury 0: ${D}/doctrine_j0.md
  Jury 1: ${D}/doctrine_j1.md

FORMAT: Punctuation REMOVED; paragraph structure KEPT as literal <NL> tokens, each with its own index. The ¶ on the token immediately before every <NL>, and on the final token, is ALWAYS correct and must never be removed; an <NL> token itself NEVER carries a boundary. Everything inside paragraphs is yours to judge. ⟨i⟩ = index anchor, not part of the text.

YOUR ${docs.length} DOCUMENTS, one file each, STRICTLY ONE AT A TIME (finish and record one before opening the next):
${docs.map((d, i) => `${i + 1}. ${P}/docs/${d}.json`).join('\n')}

Each document arrives with an automatic system's boundary marks (¶) already drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust — it makes both kinds of errors. A document whose marks already satisfy its law is FINISHED; unchanged is a common and correct verdict.

PER DOCUMENT, no limit on your reasoning depth:
1. Jury 0 gives its full stance from doctrine 0 — what the text is, which register it belongs to, its sentence law, and every place the draft marks are wrong, argued from the text.
2. Jury 1 gives its own independent stance from doctrine 1, site by site.
3. They ARGUE every disagreement. Concede when the colleague's reading matches the annotators' policy better; rebut precisely when it does not. Where a doctrine carries a law earned from a graded mistake in THIS register, that law outranks instinct.
4. RECORD: only changes BOTH juries endorse survive. Unresolved means the draft stands. Write {"doc_id":"...","add":[{"i":<index>,"w":"<token>"}],"remove":[{"i":<index>,"w":"<token>"}],"argued":"<one line per argued site>"} to ${P}/exam_out/<doc_id>.json IMMEDIATELY on finishing that document. add = token positions that end a sentence but carry no ¶; remove = positions carrying ¶ that do not end a sentence; copy w from the text at the moment you decide. Never add or remove on an <NL> token; never remove a ¶ that sits immediately before an <NL> or on the final token.

RESUME: if ${P}/exam_out/<doc_id>.json already exists, skip that document.
When all are recorded return {"done": <count>, "note": ""}. No web, no PowerShell. Touch NOTHING beyond the two doctrine files, your listed documents, and your own output files — answer keys for this corpus exist elsewhere in this repository and opening anything else voids the evaluation.`

const sittings = []
for (let i = 0; i < ORDER.length; i += 4) sittings.push(ORDER.slice(i, i + 4))
log(`redo: ${ORDER.length} docs in ${sittings.length} sittings, ${LANES} lanes, retrain4 doctrines, Opus max`)

phase('Redo')
const START = budget.spent()
let done = 0, dead = 0
for (let p = 0; p < sittings.length; p += LANES) {
  if (budget.spent() - START > 6_000_000) { log(`BUDGET HALT at ${(budget.spent() - START).toLocaleString()}: ${done} done, banked`); break }
  const round = sittings.slice(p, p + LANES)
  const res = await parallel(round.map((docs, k) => () =>
    agent(SITTING(docs, p + k + 1, sittings.length), { label: `redo sit ${p + k + 1}/${sittings.length} (${docs.length}d)`, phase: 'Redo', schema: OUT, model: 'opus', effort: 'max' })
      .then(v => v?.done ?? 0)))
  const got = res.filter(Boolean).reduce((a, b) => a + b, 0)
  done += got
  dead = got === 0 ? dead + 1 : 0
  if (dead >= 2) { log(`CIRCUIT BREAKER after two dead rounds — ${done} done, banked`); break }
  log(`sittings ${p + 1}-${p + round.length}/${sittings.length}: ${got} docs (total ${done}), spend ${(budget.spent() - START).toLocaleString()}`)
}
return { juried: done, of: ORDER.length, spent: budget.spent() - START }