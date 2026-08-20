export const meta = {
  name: 'retrain4-nopa-finish-training',
  description: 'Finish NoPnx-PA juror training: 2 long-running agents, rolling-origin, batches of five, grade + absorb between batches',
  phases: [{ title: 'Train' }],
}

const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const D = REPO + '/scratch_exo/retrain4/nopa'
const OUT = { type: 'object', required: ['done', 'remaining'], properties: { done: { type: 'integer' }, remaining: { type: 'integer' }, note: { type: 'string' } } }

const JUROR = (j) => `You are Jury ${j}, an Arabic sentence-segmentation judge continuing your own training. You have ALREADY trained on part of the corpus and written your doctrine. Work in ${REPO}.

FIRST: read your existing doctrine in full — ${D}/doctrine_j${j}.md. That is everything you have learned so far. You will keep extending it.

YOUR REMAINING DOCUMENTS: read ${D}/remaining.json and take the list under key "${j}". Before starting, list ${D}/ans/j${j}/ and SKIP any document whose answer file already exists (you may have been interrupted mid-run; never redo a banked document).

FORMAT (NoPnx-PA): Punctuation REMOVED; paragraph structure KEPT as literal <NL> tokens, each with its own index. The boundary on the token immediately before every <NL>, and on the final token, is ALWAYS correct. An <NL> token itself NEVER carries a boundary. Everything inside paragraphs is yours to judge. ⟨i⟩ marks that the next token is token number i — an index anchor, not part of the text.

WORK IN BATCHES OF FIVE, strictly in list order. Per batch:
1. SEGMENT each document one at a time from ${D}/../../retrain3/nopa/train_docs/<doc_id>.json (path: ${REPO}/scratch_exo/retrain3/nopa/train_docs/<doc_id>.json). No limit on your reasoning — work out what the text IS, what its natural sentence unit is, and argue every uncertain site in writing. The moment a document is finished, write ${D}/ans/j${j}/<doc_id>.json as {"doc_id":"...","identity":"what this text is","law":"its sentence law as you read it","reasoning":"your argument at the hard sites","boundaries":[{"i":<index of the LAST token of a sentence>,"w":"<that token copied from the text>","c":"hi|med|lo"}]}. Write it BEFORE opening the next document.
2. GRADE the batch: python scratch_exo/retrain4/grade4.py --track nopa --juror ${j} --only <the five ids comma-separated>
3. READ ${D}/mistakes_j${j}.txt in full. For every FALSE CUT and every MISSED, work out the RULE behind it — what do these annotators actually do — and write it into ${D}/doctrine_j${j}.md in your own words, filed by text type, with your own error as the example. When a new correction contradicts an earlier entry, fix that entry and say so. The doctrine is cumulative: never delete earned knowledge.
4. Move to the next batch of five with the updated doctrine in hand.

Keep going until your remaining list is exhausted, or until you sense you are running out of room — everything is banked per document, so stopping cleanly costs nothing.

STRICT RULES: You may touch ONLY your listed training documents, your own answer files, your mistakes file, and your own doctrine. Training gold lives in data/NoPnx-PA_train.jsonl and is read ONLY by the grader — never open it yourself. Never open, search for, or reference any dev, test, or blind data; answer keys for other splits exist in this repository and touching anything outside your allowed files voids the training. No web.

Return {"done": <documents you completed this run>, "remaining": <documents still unanswered on your list>, "note": "<per-batch mean F1 values the grader printed>"}.`

phase('Train')
let round = 0, lastTotal = -1
const results = []
while (round < 4) {
  round++
  const r = await parallel([0, 1].map(j => () =>
    agent(JUROR(j), { label: `juror ${j} (round ${round})`, phase: 'Train', schema: OUT, model: 'opus', effort: 'max' })
  ))
  const ok = r.filter(Boolean)
  const done = ok.reduce((a, b) => a + (b.done ?? 0), 0)
  const remaining = ok.reduce((a, b) => a + (b.remaining ?? 0), 0)
  results.push({ round, done, remaining, notes: ok.map(x => x.note) })
  log(`round ${round}: ${done} documents completed, ${remaining} still outstanding`)
  if (remaining === 0) { log('training complete — every document answered and absorbed'); break }
  if (done === 0) { log('no progress this round — stopping so nothing is burned in a loop'); break }
  if (remaining === lastTotal) { log('stalled at the same count — stopping'); break }
  lastTotal = remaining
}
return results