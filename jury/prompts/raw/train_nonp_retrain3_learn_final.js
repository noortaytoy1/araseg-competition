export const meta = {
  name: 'retrain3-learn-final',
  description: 'Final doctrine absorption: 4 learn agents (one per track per jury), parallel.',
  phases: [{ title: 'Learn' }],
}
const REPO = 'C:\\Users\\pc\\Downloads\\evolving-vlm-46\\arabic-sentence-segmentation'
const learnP = (tr,j) => `You are Jury ${j}, an Arabic sentence-segmentation judge in training on this corpus. Work in ${REPO}. Your practice batch has just been graded.
Read scratch_exo/retrain3/${tr}/mistakes_j${j}.txt IN FULL. Then update your doctrine file scratch_exo/retrain3/${tr}/doctrine_j${j}.md (create it if this is your first batch).
LEARN: for every FALSE CUT and every MISSED in the mistakes file, find the RULE behind it — what do these annotators actually do — and write it into your doctrine file in your own words, filed by text type, with your own error as the example. When a new correction contradicts an earlier entry, fix the earlier entry and say so. The doctrine is cumulative — never delete knowledge. No limit on your reasoning.
The ONLY files you may touch: the mistakes file (read) and your doctrine file (read/write). Touching ANY other file voids the training. No web, no commands.
Return 2 sentences: the biggest law you learned and its text type.`
phase('Learn')
const KEYS = [['pnxnp','0'],['pnxnp','1'],['nopa','0'],['nopa','1']]
const res = await parallel(KEYS.map(k => () => agent(learnP(k[0],k[1]),{label:`learn ${k[0]} j${k[1]}`,phase:'Learn',model:'opus',effort:'max'})))
const missing = KEYS.filter((k,i)=>!res[i]).map(k=>k.join('/'))
if(missing.length) log(`LEARN FAILED for: ${missing.join(', ')} — relaunch needed`)
return { learned: res }