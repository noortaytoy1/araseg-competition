export const meta = { name: 'papereval-exam-nonp', description: 'Paper eval, NoPnx-NP test sample: frozen juries revise ensemble drafts, 2/2 edits only.', phases: [{ title: 'Exam' }] }
const REPO='C:/Users/pc/Downloads/evolving-vlm-46/arabic-sentence-segmentation'
const D=REPO+'/scratch_exo/retrain3/nonp'
const PKT=REPO+'/scratch_exo/papereval/NoPnx-NP'
const OUT={type:'object',required:['done'],properties:{done:{type:'integer'},note:{type:'string'}}}
const SITTING=(docs,n,total)=>`This is exam sitting ${n} of ${total}. Two TRAINED juries sit together: Jury 0 and Jury 1. Each trained separately on this annotation team's training documents and wrote its own doctrine. You hold both: when Jury 0 speaks, reason strictly from doctrine 0; when Jury 1 speaks, strictly from doctrine 1. They are colleagues with different training histories, not one voice twice.
FIRST read both doctrines ONCE:
  Jury 0: ${D}/doctrine_j0.md
  Jury 1: ${D}/doctrine_j1.md
FORMAT: Punctuation REMOVED and paragraph marks REMOVED — plain words only. The final token of a document always ends a sentence. ⟨i⟩ means the next token is token number i (an index anchor, not part of the text).
YOUR ${docs.length} DOCUMENTS, one file each, STRICTLY ONE AT A TIME (finish and record one before opening the next):
${docs.map((d,i)=>`${i+1}. ${PKT}/docs/${d}.json`).join('\n')}
Each document arrives with an automatic system's boundary marks (\u00b6) already drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust — it makes both kinds of errors. A document whose marks already satisfy its law is FINISHED — unchanged is a common correct verdict.
PER DOCUMENT, no limit on your reasoning depth:
1. Jury 0 full stance from doctrine 0 — what the text is, its sentence law, every place the draft marks are wrong, argued from the text.
2. Jury 1 its own independent stance from doctrine 1, site by site.
3. They ARGUE every disagreement — concede when the colleague's reading matches the annotators' policy better, rebut precisely when not.
4. RECORD: only changes BOTH juries endorse survive; unresolved = the draft stands. Write {"doc_id":"...","add":[{"i":<index>,"w":"<token>"}],"remove":[{"i":<index>,"w":"<token>"}],"argued":"<one line per argued site>"} to ${PKT}/exam_out/<doc_id>.json IMMEDIATELY on finishing that document. add = token positions that end a sentence but carry no \u00b6; remove = positions carrying \u00b6 that do not end a sentence; copy w from the text. Never remove the ¶ on the final token. Long documents: work in sections, extending the same record. Use ONLY properly escaped JSON — no raw newline characters inside strings.
RESUME: if ${PKT}/exam_out/<doc_id>.json already exists, skip that document.
When all recorded return {"done": <count>, "note": ""}. No web, no PowerShell. Touch NOTHING beyond the two doctrine files, your listed documents, and your own output files — answer keys exist elsewhere in this repo and touching anything else voids the exam.`
const m=await agent(`Read the file ${PKT}/manifest.json and return {"sittings": <its "sittings" array>} verbatim. Do nothing else. No other tools.`,{label:'read-manifest',phase:'Exam',schema:{type:'object',required:['sittings'],properties:{sittings:{type:'array',items:{type:'array',items:{type:'string'}}}}},model:'haiku',effort:'low'})
const sittings=m?.sittings??[]
if(!sittings.length) throw new Error('manifest read failed — refusing to spend')
log(`papereval nonp: ${sittings.reduce((a,s)=>a+s.length,0)} docs in ${sittings.length} sittings, 7 at a time`)
phase('Exam')
const START=budget.spent()
let done=0,failStreak=0
for(let p=0;p<sittings.length;p+=7){
  if(budget.spent()-START>6_000_000){log(`BUDGET HALT at ${(budget.spent()-START).toLocaleString()}: ${done} done`);break}
  const round=sittings.slice(p,p+7)
  const res=await parallel(round.map((docs,k)=>()=>agent(SITTING(docs,p+k+1,sittings.length),{label:`sit ${p+k+1}/${sittings.length} (${docs.length}d)`,phase:'Exam',schema:OUT,model:'opus',effort:'max'}).then(v=>v?.done??0)))
  const got=res.filter(Boolean).reduce((a,b)=>a+b,0)
  done+=got
  failStreak=got===0?failStreak+1:0
  if(failStreak>=1){log(`CIRCUIT BREAKER after dead round — ${done} done, verdicts banked`);break}
  log(`sittings ${p+1}-${p+round.length}/${sittings.length}: ${got} docs (total ${done})`)
}
return {juried:done,spent:budget.spent()-START}