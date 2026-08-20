export const meta = {
  name: 'jury-np-reflect-1',
  description: 'NoPnx-NP reflection round 1: each juror studies its own graded mistake report from the 126 cold practice documents and writes its own lessons file in its own words.',
  phases: [{ title: 'Reflect', detail: 'juror 0 and juror 1 study their graded mistakes and write their doctrine' }],
}
const REPO = 'C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const WORK = REPO + '/scratch_exo/np_train_work'
const OUT_R = { type:'object', required:['lessons'], properties:{ lessons:{type:'string'} } }

const REFLECT = (j) => `You are Juror ${j}. Your practice answers have been GRADED against the annotators' real answer keys. Your complete mistake report is in: ${WORK}/mistakes_np_juror${j}.txt — every boundary you wrongly added and every boundary you missed, each with the surrounding text and what the key actually says, organized by document and topic, with the law you derived for each document quoted.

Read it exactly once, and study it the way a student studies a returned exam: for each error, articulate what you were thinking versus what the annotators' policy actually is. THINK DEEPLY — the errors are the curriculum; patterns across errors are the syllabus. Take as long as you need; there is no limit on your reasoning and no reward for brevity here.

Pay particular attention to the topics where your score collapsed. The summary at the end of the report gives your F1 per topic — where recall is far below precision you are systematically failing to cut somewhere the annotators do cut, and the report shows you exactly where. Find the rule behind it, not the individual sites.

Then write YOUR OWN LESSONS: a doctrine, in your own words, that would have prevented your mistakes — organized by topic (expository, narrative, exercise/worksheet, hadith, quran, bible, poetry, conversational, legal), concrete enough that you could hand it to yourself tomorrow and segment these genres correctly. State each lesson as what the ANNOTATORS do, with a short example drawn from your own errors, not as a vague maxim. Where you were already correct, say so in one line — do not invent corrections for things you got right. Write as many lessons as your mistakes justify.

You may also, ONCE each, re-read your own answers in ${WORK} for context. Nothing else. No Bash, no PowerShell, no web, and do not open any data file or answer key.
RETURN the lessons as the structured result: {"lessons": "<your full doctrine>"}. Also Write a copy to ${WORK}/lessons_np_juror${j}.md first — that file is the record, and it is what you will carry into the exam.`

phase('Reflect')
const r = await parallel([0, 1].map(j => () => agent(REFLECT(j),
  { label:`np-reflect${j}`, phase:'Reflect', schema: OUT_R, model:'opus', effort:'max' })
  .then(v => ({ juror: j, chars: (v?.lessons ?? '').length }))))
log(`NP reflection 1 done: ${r.filter(Boolean).map(x=>`J${x.juror}=${x.chars}ch`).join(' ')}, spend ${budget.spent()}`)
return { reflect: r.filter(Boolean), spent: budget.spent() }
