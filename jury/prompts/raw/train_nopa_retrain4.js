export const meta = {
  name: 'retrain4-train-nopa',
  description: 'From-scratch full-174 retraining, track nopa: fresh juries, 4 graded learning cycles, 7-agent cap.',
  phases: [{ title: 'Train' }],
}
const REPO = 'C:\\Users\\pc\\Downloads\\evolving-vlm-46\\arabic-sentence-segmentation'
const FMT = 'Punctuation REMOVED; paragraph structure KEPT as literal <NL> tokens (each <NL> has its own index). The ¶ on the token immediately before every <NL> and on the final token is ALWAYS correct and must never be removed; an <NL> token itself NEVER carries a boundary. Everything inside paragraphs is yours to judge. ⟨i⟩ = index anchor, not part of the text.'
const SP = {"0":[["doc_3f6c5f1e305a","doc_44ee993e046e","doc_201b37ae8a22","doc_1d513dc2c22a","doc_c567a2257203","doc_47764cf032d7","doc_226fc860e550","doc_eb278f2980be","doc_54f919517225","doc_56781db1019f","doc_5b0751f40176","doc_0a957c052364","doc_32908af53b40","doc_501500835788","doc_70d28c9dfebb","doc_c188bbc9702e","doc_86d3b22a1be5","doc_b9371c2fbb78","doc_e1435023282b"],["doc_00b450a96684","doc_056d9b714661","doc_0623277721a2","doc_085c979b884f","doc_099def456da2","doc_0c92fc467c35","doc_0de3fe1bcb6c","doc_142afcb751a9","doc_1522cf2ee149","doc_1c49614cd6bc","doc_1d08b507b151","doc_27b80bb5ed9c","doc_2baf8d2ac98d","doc_2fabc1ccf342","doc_3484c02411e9","doc_3762b5d5da2f","doc_3ccbb7784a8d","doc_4114b13808e9","doc_4259e5d2a449","doc_46717e5891d4","doc_4f0612ce35d1","doc_51f6f776f3c7","doc_566d3db2b5c9"],["doc_5a3a244c59f1","doc_5ec52c5c5c0b","doc_60d9a7d0d1ca","doc_617f4fad2251","doc_66816d986f39","doc_684c09874e72","doc_6abdaa7e4427","doc_723871f7197e","doc_74177f9e1015","doc_75aefb3d0bfd","doc_7a7737eabf04","doc_7b090cf754e8","doc_7f28cab0c923","doc_83e3dc997e52","doc_87ac286b6785","doc_8d062781c4a6","doc_9a890accffc8","doc_9af6e47014d6","doc_9f8cf954304a","doc_a045e64631c7","doc_a4029b2c8e1b","doc_a74a9e93d97a","doc_a9c7d1c861a3"],["doc_ade6351c3ca0","doc_afd0d91483b4","doc_b1d7ec94fa9e","doc_b4edd9186ca6","doc_b592e620494c","doc_bb10be35e0fa","doc_c0adf987ac2e","doc_c4cb5672fc6c","doc_ca06bbd92e01","doc_cddb10bd6b23","doc_d2cedca907a0","doc_d38d53501296","doc_d715df9a8c78","doc_daa6620988df","doc_dc967c8520e9","doc_dfc470ebeb16","doc_e438f8ad8abe","doc_eb9893e8f98b","doc_f2767b56c842","doc_f41918b2700e","doc_f7c344e48e78","doc_fb2e077fe69e"]],"1":[["doc_8b2d52902b9b","doc_66c98646b9a9","doc_de12d5da4854","doc_16f2b3eccb91","doc_3272f207e8c9","doc_32b866eb5dc4","doc_5e12ebe2976f","doc_9d15d9d5d637","doc_8921b794ce19","doc_405d5fba81e0","doc_7bc0656a48ce","doc_78b2ed132e2f","doc_51fccce927eb","doc_20dcf69f9905","doc_25ed373f8aeb","doc_cb57e21e8421","doc_1fa12697d487","doc_27a114d4e376","doc_361233315d24"],["doc_02c1581a0f82","doc_056f7138837b","doc_0696314023ab","doc_091dca4b6b3c","doc_0ad7fe19f9bb","doc_0d81543aa818","doc_127c1535d7d9","doc_149339985bf9","doc_153372a232d6","doc_1c815410c046","doc_26ebb09caa38","doc_281bb07b48c9","doc_2f73d2ab2ca3","doc_32d5d0cbc877","doc_35734be96b2b","doc_39e47ea428f8","doc_40e700c4f806","doc_4161e96c985e","doc_449f221bdbd2","doc_4ef3bc70a59e","doc_51816712b97c","doc_528da1fb596e","doc_59a45e1e029a"],["doc_5eb0f458c625","doc_5f0d00706642","doc_616eaaa0cc9d","doc_666e6f739000","doc_6806936b7535","doc_690f8f5c21aa","doc_6d2684ec8595","doc_72b8decfdb8c","doc_741bfb3bda19","doc_7825ac72ea71","doc_7aa0f72706ad","doc_7f0f8c191265","doc_7f77a81d1a7a","doc_850a7d56d8e0","doc_886c7e9fc8e5","doc_91f6135ba53a","doc_9ad9ba0d12d4","doc_9e0419ed749d","doc_9f9992d71084","doc_a17df83313f0","doc_a404631139d3","doc_a83273e31ee3","doc_ab452f741db0"],["doc_ae1331c52539","doc_b15bce7039d2","doc_b3d76332711b","doc_b58385fe50df","doc_b8fa69b09c88","doc_bd40f62e73c2","doc_c2a9fc26a488","doc_c626796ad5b3","doc_cc92bd9a59d3","doc_d1b08b16526d","doc_d30b2fb502db","doc_d5e46166411c","doc_d86f75defe49","doc_db3824c69464","doc_df532033acd3","doc_e2e50de1ad8e","doc_ea9c5c63ae76","doc_ede30da2ee29","doc_f336dc46f626","doc_f60bd3a48b58","doc_f876524dd98e","doc_fe3b2556a241"]]}
const segP = (j,d) => `You are Jury ${j}, a brand-new Arabic sentence-segmentation judge in training on this corpus. You start with zero knowledge of this corpus's conventions — everything you will ever know comes from your own graded practice, accumulated in your doctrine file. Work in ${REPO}.
STEP 0: if scratch_exo/retrain4/nopa/ans/j${j}/${d}.json already exists, return the word SKIPPED and stop immediately.
STEP 1: read your doctrine file scratch_exo/retrain4/nopa/doctrine_j${j}.md if it exists — it is your own accumulated law from previously graded batches; apply it faithfully. If it does not exist, this is your first batch: reason from the text alone.
FORMAT: ${FMT}
STEP 2: SEGMENT the document scratch_exo/retrain3/nopa/train_docs/${d}.json. No limit on your reasoning — first work out what the text IS (genre, register, structure) and what its natural sentence unit is, then walk it and argue every uncertain site in writing. Write your answer to scratch_exo/retrain4/nopa/ans/j${j}/${d}.json as {"doc_id":"...","identity":"what this text is","law":"its sentence law as you read it","reasoning":"your argument at the hard sites","boundaries":[{"i":<index of the LAST token of a sentence>,"w":"<that token copied from the text>","c":"hi|med|lo"}]}. Use ONLY properly escaped JSON — no raw newline characters inside strings.
The ONLY files you may touch: this practice document, your doctrine file (read), your answer file (write). Touching ANY other file voids the training — answer keys exist elsewhere in this repository. Do not run the grader; grading happens after your whole batch. No web.`
const gradeP = (j,ids) => `Work in ${REPO}. Run exactly this one command from that directory and return its COMPLETE stdout verbatim as your final answer. Do nothing else; open no files.
python scratch_exo/retrain4/grade4.py --track nopa --juror ${j} --only ${ids.join(',')}`
const learnP = (j,b) => `You are Jury ${j}, an Arabic sentence-segmentation judge in training on this corpus. Work in ${REPO}. Your practice batch ${b} has just been graded.
Read scratch_exo/retrain4/nopa/mistakes_j${j}.txt IN FULL. Then update your doctrine file scratch_exo/retrain4/nopa/doctrine_j${j}.md (create it if this is your first batch).
LEARN: for every FALSE CUT and every MISSED in the mistakes file, find the RULE behind it — what do these annotators actually do — and write it into your doctrine file in your own words, filed by text type, with your own error as the example. When a new correction contradicts an earlier entry, fix the earlier entry and say so. The doctrine is cumulative — never delete knowledge. No limit on your reasoning.
The ONLY files you may touch: the mistakes file (read) and your doctrine file (read/write). Touching ANY other file voids the training. No web, no commands.
Return 2 sentences: the biggest law you learned this batch and what you will change next batch.`
function chunks(a,n){const o=[];for(let i=0;i<a.length;i+=n)o.push(a.slice(i,i+n));return o}
phase('Train')
const report = {0:[],1:[]}
for(let b=0;b<4;b++){
  const jobs = [].concat(SP["0"][b].map(d=>[0,d]), SP["1"][b].map(d=>[1,d]))
  const ok = {0:[],1:[]}
  const fail = []
  for(const grp of chunks(jobs,7)){
    const res = await parallel(grp.map(jd => () => agent(segP(jd[0],jd[1]),{label:`seg j${jd[0]} b${b+1} ${jd[1]}`,phase:'Train',model:'opus',effort:'max'}).then(r=>({j:jd[0],d:jd[1],ok:r!==null})).catch(()=>({j:jd[0],d:jd[1],ok:false}))))
    for(const x of res){ if(x&&x.ok) ok[x.j].push(x.d); else if(x) fail.push([x.j,x.d]) }
  }
  for(const grp of chunks(fail,7)){
    const res = await parallel(grp.map(jd => () => agent(segP(jd[0],jd[1]),{label:`retry j${jd[0]} ${jd[1]}`,phase:'Train',model:'opus',effort:'max'}).then(r=>({j:jd[0],d:jd[1],ok:r!==null})).catch(()=>({j:jd[0],d:jd[1],ok:false}))))
    for(const x of res){ if(x&&x.ok) ok[x.j].push(x.d) }
  }
  for(const j of [0,1]){
    const dropped = SP[String(j)][b].filter(d=>!ok[j].includes(d))
    if(dropped.length) log(`jury ${j} batch ${b+1}: DROPPED from grading (no answer after retry): ${dropped.join(', ')}`)
  }
  const gr = await parallel([0,1].map(j => () => ok[j].length ? agent(gradeP(j,ok[j]),{label:`grade j${j} b${b+1}`,phase:'Train',model:'haiku',effort:'low'}) : Promise.resolve('')))
  for(const j of [0,1]){
    const mean = String(gr[j]||'').split('\n').filter(l=>l.indexOf('BATCH MEAN')>=0).join('')
    log(`jury ${j} batch ${b+1}/4 graded (${ok[j].length} docs): ${mean||'grader output banked'}`)
    report[j].push({batch:b+1,docs:ok[j].length,grader:gr[j]})
  }
  const ln = await parallel([0,1].map(j => () => ok[j].length ? agent(learnP(j,b+1),{label:`learn j${j} b${b+1}`,phase:'Train',model:'opus',effort:'max'}) : Promise.resolve('')))
  for(const j of [0,1]) report[j].push({batch:b+1,learned:ln[j]})
}
return { jury0: report[0], jury1: report[1] }