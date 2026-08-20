export const meta = {
  name: 'retrain3-repair-cycle1',
  description: 'Repair outage damage: segment 30 missing curated docs, grade all four juries, absorb into doctrines. Max 7 concurrent.',
  phases: [{ title: 'Repair' }],
}
const REPO = 'C:\\Users\\pc\\Downloads\\evolving-vlm-46\\arabic-sentence-segmentation'
const FMT = {
  pnxnp: 'Punctuation KEPT — punctuation marks are ordinary tokens with their own indices, and a sentence usually ends ON its punctuation mark when one is present. NO <NL> tokens. The final token always ends a sentence. ⟨i⟩ = index anchor, not part of the text.',
  nopa: 'Punctuation REMOVED; paragraph structure KEPT as literal <NL> tokens (each <NL> has its own index). The ¶ on the token immediately before every <NL> and on the final token is ALWAYS correct and must never be removed; an <NL> token itself NEVER carries a boundary. Everything inside paragraphs is yours to judge. ⟨i⟩ = index anchor, not part of the text.',
}
const PRE = {
  'pnxnp|0': ["doc_3f6c5f1e305a","doc_44ee993e046e","doc_de12d5da4854","doc_c567a2257203","doc_46717e5891d4","doc_3272f207e8c9","doc_47764cf032d7","doc_54f919517225","doc_9d15d9d5d637","doc_8921b794ce19","doc_5b0751f40176","doc_0a957c052364","doc_32908af53b40","doc_501500835788","doc_1c49614cd6bc","doc_c188bbc9702e","doc_86d3b22a1be5","doc_1fa12697d487","doc_e1435023282b","doc_9e0419ed749d"],
  'pnxnp|1': ["doc_8b2d52902b9b","doc_201b37ae8a22","doc_66c98646b9a9","doc_1d513dc2c22a","doc_16f2b3eccb91","doc_226fc860e550","doc_5e12ebe2976f","doc_eb278f2980be"],
  'nopa|0': ["doc_3f6c5f1e305a","doc_44ee993e046e","doc_201b37ae8a22","doc_1d513dc2c22a","doc_c567a2257203","doc_47764cf032d7","doc_226fc860e550","doc_eb278f2980be","doc_54f919517225","doc_56781db1019f","doc_5b0751f40176","doc_0a957c052364","doc_32908af53b40","doc_501500835788","doc_70d28c9dfebb","doc_c188bbc9702e","doc_86d3b22a1be5","doc_e1435023282b"],
  'nopa|1': ["doc_66c98646b9a9"],
}
const MISS = {
  'pnxnp|0': [],
  'pnxnp|1': ["doc_56781db1019f","doc_405d5fba81e0","doc_7bc0656a48ce","doc_a045e64631c7","doc_51fccce927eb","doc_25ed373f8aeb","doc_153372a232d6","doc_cb57e21e8421","doc_b9371c2fbb78","doc_27a114d4e376","doc_361233315d24"],
  'nopa|0': ["doc_b9371c2fbb78"],
  'nopa|1': ["doc_8b2d52902b9b","doc_de12d5da4854","doc_16f2b3eccb91","doc_3272f207e8c9","doc_32b866eb5dc4","doc_5e12ebe2976f","doc_9d15d9d5d637","doc_8921b794ce19","doc_405d5fba81e0","doc_7bc0656a48ce","doc_78b2ed132e2f","doc_51fccce927eb","doc_20dcf69f9905","doc_25ed373f8aeb","doc_cb57e21e8421","doc_1fa12697d487","doc_27a114d4e376","doc_361233315d24"],
}
const segP = (tr,j,d) => `You are Jury ${j}, an Arabic sentence-segmentation judge in training on this corpus. You know ONLY what your own graded practice has taught you — that knowledge lives in your doctrine file. Work in ${REPO}.
STEP 0: if scratch_exo/retrain3/${tr}/ans/j${j}/${d}.json already exists, return the word SKIPPED and stop immediately.
STEP 1: read your doctrine file scratch_exo/retrain3/${tr}/doctrine_j${j}.md if it exists — it is your own accumulated law from previously graded batches; apply it faithfully. If it does not exist, this is your first batch: you start with zero knowledge of this corpus's conventions and must reason from the text alone.
FORMAT: ${FMT[tr]}
STEP 2: SEGMENT the document scratch_exo/retrain3/${tr}/train_docs/${d}.json. No limit on your reasoning — first work out what the text IS (genre, register, structure) and what its natural sentence unit is, then walk it and argue every uncertain site in writing. Write your answer to scratch_exo/retrain3/${tr}/ans/j${j}/${d}.json as {"doc_id":"...","identity":"what this text is","law":"its sentence law as you read it","reasoning":"your argument at the hard sites","boundaries":[{"i":<index of the LAST token of a sentence>,"w":"<that token copied from the text>","c":"hi|med|lo"}]}. Use ONLY properly escaped JSON — no raw newline characters inside strings.
The ONLY files you may touch: this practice document, your doctrine file (read), your answer file (write). Touching ANY other file voids the training — answer keys exist elsewhere in this repository. Do not run the grader. No web.`
const gradeP = (tr,j,ids) => `Work in ${REPO}. Run exactly this one command from that directory and return its COMPLETE stdout verbatim as your final answer. Do nothing else; open no files.
python scratch_exo/retrain3/grade3.py --track ${tr} --juror ${j} --only ${ids.join(',')}`
const learnP = (tr,j) => `You are Jury ${j}, an Arabic sentence-segmentation judge in training on this corpus. Work in ${REPO}. Your practice batch has just been graded.
Read scratch_exo/retrain3/${tr}/mistakes_j${j}.txt IN FULL. Then update your doctrine file scratch_exo/retrain3/${tr}/doctrine_j${j}.md (create it if this is your first batch).
LEARN: for every FALSE CUT and every MISSED in the mistakes file, find the RULE behind it — what do these annotators actually do — and write it into your doctrine file in your own words, filed by text type, with your own error as the example. When a new correction contradicts an earlier entry, fix the earlier entry and say so. The doctrine is cumulative — never delete knowledge. No limit on your reasoning.
The ONLY files you may touch: the mistakes file (read) and your doctrine file (read/write). Touching ANY other file voids the training. No web, no commands.
Return 2 sentences: the biggest law you learned and its text type.`
function chunks(a,n){const o=[];for(let i=0;i<a.length;i+=n)o.push(a.slice(i,i+n));return o}
phase('Repair')
const KEYS = ['pnxnp|0','pnxnp|1','nopa|0','nopa|1']
const jobs = []
for(const k of KEYS){ const tr=k.split('|')[0], j=k.split('|')[1]; for(const d of MISS[k]) jobs.push([tr,j,d,k]) }
const got = {}
for(const k of KEYS) got[k] = []
let pending = jobs
for(let attempt=0; attempt<2 && pending.length; attempt++){
  const failed = []
  for(const grp of chunks(pending,7)){
    const res = await parallel(grp.map(x => () => agent(segP(x[0],x[1],x[2]),{label:`seg ${x[0]} j${x[1]} ${x[2]}`,phase:'Repair',model:'opus',effort:'max'}).then(r=>({x,ok:r!==null})).catch(()=>({x,ok:false}))))
    for(const r of res){ if(r&&r.ok) got[r.x[3]].push(r.x[2]); else if(r) failed.push(r.x) }
  }
  pending = failed
  if(pending.length) log(`retry pass: ${pending.length} segs still missing`)
}
if(pending.length) log(`UNRECOVERED after retries: ${pending.map(x=>x[0]+'/'+x[2]).join(', ')}`)
const gr = await parallel(KEYS.map(k => () => {
  const tr=k.split('|')[0], j=k.split('|')[1]
  const ids = PRE[k].concat(got[k])
  return ids.length ? agent(gradeP(tr,j,ids),{label:`grade ${k}`,phase:'Repair',model:'haiku',effort:'low'}) : Promise.resolve('')
}))
KEYS.forEach((k,i) => {
  const mean = String(gr[i]||'').split('\n').filter(l=>l.indexOf('BATCH MEAN')>=0).join('')
  log(`${k}: graded ${PRE[k].length+got[k].length} docs — ${mean||'output banked'}`)
})
const ln = await parallel(KEYS.map(k => () => agent(learnP(k.split('|')[0],k.split('|')[1]),{label:`learn ${k}`,phase:'Repair',model:'opus',effort:'max'})))
return { graded: gr, learned: ln, unrecovered: pending.map(x=>x[0]+'/'+x[2]) }