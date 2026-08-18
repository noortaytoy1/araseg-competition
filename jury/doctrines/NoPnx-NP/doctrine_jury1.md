# Doctrine — Jury 1, track nonp (no punctuation, no paragraph marks)

Cumulative. Every entry is paid for with one of my own graded errors.

---

## 0. THE MASTER LAW — **OVERTURNED IN BATCH 2. READ §0-BIS FIRST.**

*(kept for the record; it is TRUE ONLY for literary narrative and journalism)*

**I cut far too little. These annotators segment at the CLAUSE, not at the rhetorical period.**

The single dominant error across all five documents of batch 1 was MISSED at
`… <end of predication> ‖ و/ف/ثم + new finite verb …`.

> **Default rule: a coordinating particle (و ، ف ، ثم) followed by a finite verb that
> starts a NEW predication is a SENTENCE BOUNDARY, not an internal link.**
> I must cut *before* the particle (i.e. label the token immediately preceding و/ف/ثم).

Do NOT cut when the material after the connective is syntactically dependent:
relative clause (الذي/التي/ما), complementiser (أن/إن/بأن), causal لأن/إذ, protasis
إذا/لو/لما, purpose لِـ/حتى, ḥāl participle (محققا، متصدرا، معلقا), or a bare NP that
continues a list started earlier.

### Density check — my safety net
Gold density measured on batch 1:
| doc | genre | tokens/sentence in GOLD |
|---|---|---|
| doc_16f2b3eccb91 | classical philosophy (Ibn Tufayl) | **14.1** |
| doc_8b2d52902b9b | sports magazine | **15.7** |
| doc_201b37ae8a22 | Wikipedia | **18.6** |
| doc_3272f207e8c9 | authorial preface | **22.4** |
| doc_de12d5da4854 | literary essay | **23.8** |

Before writing an answer: divide token count by my boundary count. **If the result is
above ~25 I am certainly under-cutting.** Target 15–20 for news/classical, 20–25 for
essay/memoir.

---

## 1. Journalistic prose (newspaper / magazine feature / sports chronicle)
Example doc: `doc_8b2d52902b9b` — F1 81.4, recall 73.9.

**Law: cut at essentially EVERY و + finite verb.** This writer put a full stop before
almost every wa-clause, even when the subject is unchanged.

My misses, all of the same shape:
- `… تقام البطولة كل 4 سنوات ‖ وكانت قد بدأت أولى بطولاتها …` (same subject — still cut)
- `… في هونج كونج ‖ واستمرت من 1 سبتمبر …`
- `… في إيران ‖ وتمكن المنتخب الإيراني …`
- `… و 20 سبتمبر ‖ وشارك فيها 10 منتخبات ‖ واستطاع المنتخب الكويتي …` (two in a row)
- `… تأهل منتخب الإمارات إلى النهائيات ‖ ووقع في مجموعة صعبة …` (same subject — still cut)
- `… من الفوز باللقب ‖ غير أن ركلات الترجيح انحازت …`
- `… من هوس التشجيع ‖ فإذا أخطأ الحكم خطأ ما …` (فإذا opens a new sentence)
- `… قبل أن تحتاج للفكر الرياضي ‖ وألا يسهو أثناء المباراة …` (even a coordinated وألا)

**Sub-law 1a — enumerative wa-list of clauses.** When a sentence lists parallel members
each introduced by و and each carrying its own locative/predicate, EACH MEMBER IS ITS OWN
SENTENCE:
`… في المجموعة الأولى ‖ ومنتخبات فيتنام … في المجموعة الثانية ‖ ومنتخبات ماليزيا … الثالثة ‖ …`
I ran the whole four-group list as one sentence and lost 3 boundaries.

**Sub-law 1b — وذلك بسبب also cuts.** `… من البطولة الثالثة عشرة ‖ وذلك بسبب وجود …`
I had reasoned it was an "anaphoric causal tail". Wrong for journalism.

**Sub-law 1c — do NOT cut before كذلك + bare NP.** FALSE CUT of mine:
`… بعد ظهوره بمستوى غير متوقع [no cut] كذلك منتخب البحرين الذي ظهر بمستوى جيد …`
كذلك + NP-with-relative continues the list of "أبرز النتائج"; it has no finite main verb.
Cut before كذلك only when a full finite clause follows.

**Sub-law 1d — headlines end where the SYNTAX ends, not where the topic ends.**
My worst structural error: I read `ضرب الحكام ظاهرة خطيرة تزايدت في ملاعب كرة القدم العالمية`
as one headline. Gold: `ضرب الحكام ظاهرة خطيرة ‖` is the headline, and the body then reads
`تزايدت في ملاعب كرة القدم العالمية مناظرُ مؤذيةٌ بدأت تنتشر…` — مناظر مؤذية is the POSTPOSED
SUBJECT of تزايدت. Always parse the verb's subject before deciding a heading's extent.

**Sub-law 1e — a numbered label (أولا/ثانيا/…) ends a unit only if what follows is
syntactically independent of it.**
- correct cuts: `أولا اللاعب ‖` (next: `أول من يتحمل المسئولية … هو اللاعب نفسه`),
  `ثانيا حكم المباراة ‖`, `خامسا لجنة التنظيم ‖`
- my FALSE CUTS: `ثالثا مدرب الفريق [no cut] هو الذي يجب عليه …` and
  `رابعا الجمهور [no cut] مشجعو الفريق الذين يحبون …`
  — here the label IS the subject/topic of the following predicate, so it cannot be cut off.

---

## 2. Classical Arabic philosophical narrative (Ibn Tufayl and kin)
Example doc: `doc_16f2b3eccb91` — F1 65.3, recall 50.8. **My worst document.**

I wrote in my answer that "the fa- of consequence is INSIDE the sentence". **That was
exactly backwards and is hereby overturned.**

> **Law: in edited classical prose, EVERY fa- of consequence carrying a new perception /
> cognition verb is a full stop.** فعلم ، فتبين ، فرأى ، فطلب ، فأخر ، فنظر ، فلم يجد ،
> فليست ، فزالت ، فارتسم ، فتقهقر ، فذلك — all of them opened a new gold sentence.

Missed sites (a representative third):
`… متميزا عنها ‖ فعلم أن ذلك صادر …` · `… من معرفة الأكثر ‖ فطلب أولا …` ·
`… لتفنن أفعالهما ‖ فأخر التفكير …` · `… عن الآخر ‖ فلا يمكن أن يتحرك …` ·
`… لا يعم جميع الأجسام ‖ فليست إذن للجسم بما هو جسم ‖` ·
`… حيها وجمادها ‖ فلم يجد شيئا …` · `… بطل حكم الصورة ‖ فزالت الصورة المائية …` ·
`… لا بد له من محدث ‖ فارتسم في نفسه …` · `… وصلح لها ‖ فذلك الاستعداد هو صورته …`

The same applies to و/ثم/لكن/إذ in this register:
`… على الامتداد المذكور ‖ ويكون بالجملة خلوا …` · `… من حيث هو جسم ‖ لكنه لم يتأت له …` ·
`… وترك الجسم على الإطلاق ‖ إذ هذا الأمر لا يدركه الحس …` (even إذ!) ·
`… إلى الأقطار الثلاثة ‖ هل هو معنى الجسم بعينه …` (an embedded question is its own unit)

**Sub-law 2a — paired أحدهما / والآخر are separate sentences.**
`… مركب على الحقيقة من معنيين ‖ أحدهما يقوم مقام الطين … ‖ والآخر يقوم مقام طول الكرة … ‖ وأنه لا يفهم الجسم …`
I ran all four as one 52-token period. Wrong.
(Contrast §3's essay, where `الأولى هي … والثانية هي …` was correctly ONE sentence. The
difference is genre density, not syntax — see §0 table.)

**Sub-law 2b — a quotation formula is its own sentence; the quoted verses are ONE unit.**
Gold: `… وفي محكم التنزيل ‖ فلم تقتلوهم ولكن الله قتلهم وما رميت إذ رميت ولكن الله رمى ‖`
I did the opposite (joined the formula to verse 1, split verse 1 from verse 2). Two errors.

**Sub-law 2c — do not cut a bare exemplification.** FALSE CUT:
`… ذوات الصور كالطين مثلا [no cut] كان له طول وعرض وعمق على قدر ما ‖`
`كالطين مثلا` is not a standalone unit; the following verb belongs to it.

---

## 3. Literary / philosophical essay (Nahda register)
Example doc: `doc_de12d5da4854` — F1 90.0. Longest gold sentences of batch 1 (23.8 tok).

Cuts I missed:
- `… مغالاة شديدة ‖ فإنهم لا يكتفون …` — **فإن + pronoun + verb cuts.**
- `… على جميع مظاهر الحياة ‖ وصارت كل الأمور تتطور …`
- `… من عناصر الحياة ‖ وهما متلازمان وضروريان …` — **و + nominal sentence cuts.**
- `… وبنائها القديم ‖ وتصوروا ماذا سيكون مصير تلك العضوية …` — **و + imperative cuts.**

Connectives that do **NOT** cut in this register (my FALSE CUTS — corrections to my
batch-1 reasoning, where I had claimed ولهذا/عندئذ "reset the argument"):
- **ولهذا** — `… تستوجب العقاب الصارم [no cut] ولهذا يطلبون معاقبة …` (my §"argument reset"
  claim was wrong for ولهذا specifically; عندئذ at token 214 *was* right, so keep عندئذ.)
- **وحتى** — `… أن يغير شيئا منها [no cut] وحتى التفكير كان أخذ يسير …`
- **وقد** — `… تغيرا كليا [no cut] وقد فقدت روح المحافظة قوتها …`
- **لا شك** after an imperative — `… مصير تلك العضوية [no cut] لا شك في أن هذا المصير …`

---

## 4. Authorial preface / memoir
Example doc: `doc_3272f207e8c9` — F1 86.2.

- **فـ of narrative consequence cuts here too:** `… التي في حوزتي ‖ فرحت أبحث في منافذ بيع الكتب …`
  (I had explicitly argued this 48-token fa-chain was one period. Wrong.)
- **ثم cuts:** `… في أوائل الثمانينيات ‖ ثم تأسست شبكة الإنترنت …` and
  `… خلال ثمانية أعوام ‖ ثم كتبته في عامين …`
- **و + new verb cuts:** `… عامة القراء والباحثين ‖ واخترت مؤسسة هنداوي …` ·
  `… فتفرغت للكتابة بشكل كامل ‖ ولم أفعل شيئا آخر …` · `… من مراجع ‖ وكانت مهمة شاقة …`
- **Asyndetic verb after a long NP cuts:** `… في أسفل الكومة ‖ أسحبه وأتأمله ‖`
- **فقد / وقد does NOT cut** when it explains the sentence just made:
  `… فاق النجاح الأول [no cut] فقد نفدت طبعته الأولى …`

---

## 5. Wikipedia / encyclopedic
Example doc: `doc_201b37ae8a22` — F1 96.4, my best. Only 4 errors, so the batch-1 reading
mostly holds. Refinements:

- Encyclopedic register is the ONE place where **same-subject و-coordination stays inside**
  the sentence: `يستخدم اللاعبون … أرجلهم … ويمكنهم أيضا استخدام رأسهم` (no cut, correct);
  `تشكلت الفيفا … عام 1904 وأعلنت أنها …` (no cut, correct).
  But **different-subject / new-fact و still cuts**:
  `… في جميع أنحاء العالم ‖ وكثيرا ما يذهب الملايين …` (missed).
- **A fronted PP belongs to the sentence that FOLLOWS it, not the one before.** My worst
  encyclopedic error: `… بأعلى نسبة مشاهدة تلفزيونية من بين جميع الرياضات ‖ في العديد من
  الأنحاء من العالم تستحضر كرة القدم حماسا …`. I attached `في العديد من الأنحاء من العالم`
  backwards. **When a sentence-medial PP could go either way, attach it forward** if the
  following clause is verb-initial and would otherwise start bare.
- **Paired الأولى هي … والثانية هي …** enumerations stay in ONE sentence here (FALSE CUT of
  mine at `… في الاجتماعات السابقة [no cut] الأولى هي حمل الكرة باليد …`). Compare §2a: in
  the denser classical register the pair splits. Genre decides.

---

## 6. Universal checklist — **SUPERSEDED by §11. Kept for the record.**
1. Identify genre → pick the target density from §0.
2. Walk the token list; at every و / ف / ثم ask: *is a new finite predication starting?*
   If yes → cut, unless §1c/§3/§4 exception list says otherwise.
3. Never let a fa- of consequence pass uncut in classical or narrative prose.
4. Parse postposed subjects before ending a heading (§1d).
5. Attach ambiguous fronted PPs forward (§5).
6. Divide tokens by boundary count. Over 25? Go back and cut more.

---
---

# BATCH 2 (mean F1 87.01) — THE CORRECTION OF THE MASTER LAW

## 0-BIS. THE REAL MASTER LAW: **و AND ف ARE NOT A RULE. THEY ARE A GENRE VARIABLE.**

Batch 2 scores tell the whole story:

| doc | genre | my P | my R | what went wrong |
|---|---|---|---|---|
| doc_5e12ebe2976f | popular-science history | **71.7** | 97.1 | I cut 14 times too often, all at و/ف |
| doc_32b866eb5dc4 | encyclopedic institutional history | **80.0** | 97.3 | 8 false cuts, all at و |
| doc_56781db1019f | graded reader (Little Women) | 85.6 | 94.3 | 12 false cuts |
| doc_9d15d9d5d637 | literary novel (al-Aqqad, Sara) | 89.7 | **74.5** | 30 misses, mostly speaker labels |
| doc_8921b794ce19 | translated popular psychology | 94.3 | 93.0 | balanced |

> **In batch 1 I under-cut. In batch 2 I over-cut in four documents out of five.**
> The batch-1 order "cut at essentially every و + finite verb" is now restricted to
> **journalism (§1) and original literary narrative (§9)**. Everywhere else it is a
> precision disaster.

**The two-question test I now run at every و / ف:**
1. *Is this text ORIGINALLY-ARABIC EXPOSITORY prose (encyclopedia, institutional history,
   popular science, essay)?* → و and ف are **commas**. Default NO CUT. Only cut when a
   genuinely new topic, a new dated event chain, or an asyndetic verb-initial statement begins.
2. *Is this text NARRATIVE (novel, memoir, chronicle, sports report)?* → و + a finite verb
   with a **new subject or a new event** is a cut; ف is usually NOT (see §9).

---

## 7. Encyclopedic / institutional-history Arabic (the King Faisal Prize document)
Example: `doc_32b866eb5dc4` — F1 87.8, precision only 80.

**Law: this register runs 20+ tokens per sentence and glues its facts with و.**

### Sub-law 7a — THE TRAILING SPECIFICATION NEVER CUTS. (my single most repeated error)
A clause that merely fixes the *time, place, manner or agent* of the event just narrated is
the **TAIL** of that sentence, never a new one. Never cut before:
- **وكان ذلك …** — `… اسم مؤسسة الملك فيصل الخيرية [no cut] وكان ذلك سنة 1976 م ‖`
  and `… للجائزة بنسختها الأولى [no cut] وكان ذلك في شهر ربيع الأول …`
- **ويكون ذلك …** — `… أو من يمثله [no cut] ويكون ذلك في مقر مؤسسة …`
- **وتم فيه …** — `… 15 أكتوبر 1977 م [no cut] وتم فيه إعلان موضوعات الجائزة ‖`
- **وتوفي بعدها …** — `… لظروفه الصحية [no cut] وتوفي بعدها ببضعة أشهر …`
- **وذلك بسبب/لأن …** — *(this OVERTURNS sub-law 1b, which said وذلك بسبب cuts. It cuts in
  a sports chronicle; it does NOT cut in encyclopedic prose. Genre decides, as always.)*

### Sub-law 7b — subordinators that look like new sentences but are not
**بينما**, **كما**, **مع أن** are internal. My false cuts:
`… وابنه حسين فاروق المودودي [no cut] بينما تسلم فؤاد سزكين جائزته بنفسه ‖` ·
`… وعاش فيصل حياته العظيمة من أجلها [no cut] كما تأتي تجسيدا لآماله …`

### Sub-law 7c — **حيث DOES cut.** MISSED:
`… كأول سنة تمنح فيها الجائزة ‖ حيث عقد الاجتماع الأول لأعضاء اللجان …`
(Contrast إذ, which does NOT cut — see §8b. حيث opens a new statement, إذ explains the old one.)

### Sub-law 7d — a Qur'an/verse reference number stays inside the quoted verse.
`… وأولئك هم المفلحون [no cut] ١٠٤ آل عمران 104 ‖`
*(this corrects §2b's implication that the citation formula is separable — the trailing
sura/aya reference is part of the quotation's own unit.)*

### Sub-law 7e — enumerative و + BARE NP continues the list.
`… أول جائزة سنة 1402 ه 1982 م [no cut] والآخر في العلوم على أن تمنح …`
Same shape as §1c (كذلك + NP). A و introducing a second list member with no finite verb
of its own is internal, always, in every register.

---

## 8. Popular-science / intellectual-history narrative (originally Arabic)
Example: `doc_5e12ebe2976f` — F1 82.5, **precision 71.7, my worst precision ever.**

> **Law: in this register the fa- of consequence is INSIDE the sentence.**
> This is the exact opposite of §2 (classical philosophy), where every fa- cut. I have now
> been burned in both directions; the discriminator is REGISTER, and I must decide it first.

My false cuts, all the same shape:
`… كان زاده في ذلك البحث خياله [no cut] فمضى يتخيل كيانات خارقة … [no cut] فوجدت آلهة الأنهار …`
(one 60-token sentence, three fa-clauses, I split it in three) ·
`… إلا أن ثريتنا استثنائية هذه المرة [no cut] فقد استرعت انتباه …` ·
`… مسارا جديدا تماما [no cut] فقد بدأ الفتى دراسة حركة الثريا …` ·
`… من سدوده القوية [no cut] فقد أثبت فاعلية عظيمة …`

**Sub-law 8a — the dated-event chain is ONE sentence.**
`… وفي عام ١٥٨٩ م أنهى دراسته الجامعية [no cut] وفي نفس العام استخدم الرياضيات في حساب
عجلة الجسم [no cut] والبعض يؤرخ لبدايات العلم التجريبي من تلك السنة ‖`
I cut this three times. Zero of them were right.
Also `… في عام ١٥٨١ م [no cut] وكان فتانا هو جاليليو جاليلي …` — the §7a tail again.

**Sub-law 8b — إذ does NOT cut** (also confirmed in §10):
`… من سنة إلى أخرى [no cut] إذ كان يتضمن أسئلة مثل …`
*(this OVERTURNS the batch-1 note in §2 that "even إذ!" cuts. It cut once, in Ibn Tufayl.
In modern prose إذ is a comma. Register decides.)*

**Sub-law 8c — a bare verb after a proper name / title is a relative clause with no الذي.**
`… أطلق عليه لاحقا الخيال العلمي Science Fiction [no cut] وصل لدرجة كبيرة من النضج …`
`… وهو مزود بأزرار ولوالب [no cut]` — ḥāl, never cuts.

**Sub-law 8d — منها/منه resuming a list DOES cut.** MISSED:
`… لعصور أقدم بقرون من مؤلفات ويلز ‖ منها مثلا بعض قصص ألف ليلة وليلة …`

---

## 9. Play scripts and dialogue with SPEAKER LABELS — the highest-yield rule I own
Example: `doc_9d15d9d5d637` (al-Aqqad, *Sara*) — F1 81.4, **recall 74.5**. 20 of my 30 misses
were the same single mistake.

> **LAW (NARROWED IN BATCH 3 - see §13a): in a STAGE PLAY laid out with the label on its own line, a bare proper name standing as a speaker label is ITS OWN ONE-TOKEN SENTENCE. In every other kind of dialogue the label WELDS to its speech.**
> `… زي الحداد ‖ سارة ‖ على رسلكما أيتها الصديقتان ‖ لا تتخاصما …`

I had reasoned that "the label is welded to its speech because a colon is not a sentence
end". **Wrong, and expensively.** The label sits on its own display line; paragraph marks are
stripped; therefore it surfaces as a one-token unit. Gold even produced
`سارة ‖ دعوت سارة و ‖ سارة ‖ سارة ‖ أخشى أن تكون …` — three consecutive one-token units.

**Sub-law 9a — distinguish LABEL from VOCATIVE.** A name that is the *addressee* of the
utterance it sits in is not a label. My false cut: `سارة ‖ أهلا بك سارة ‖ أخشى …` — I cut
after بك and missed the label; gold reads "أهلا بك يا سارة" as one greeting.
Test: if removing the name leaves the utterance complete and the name has no vocative يا and
no syntactic role → LABEL → cut it off on both sides.

**Sub-law 9b — inside a label-led turn, every ? and ! is a boundary.**
`ماذا تقولين ‖ صدقت ‖ يا للعار ‖ هذا كلام العجائز ‖ هذا حديث خرافة ‖ هذا مذهب عتيق …`
Parallel asyndetic nominal clauses (هذا … هذا … هذا …) each cut. I ran them together and lost 2.

**Sub-law 9c — but after a NARRATIVE SPEECH VERB the head of the quote is WELDED.**
This is the mirror image of 9b and it cost me 6 false cuts across two documents:
- `قال همام كفاية [no cut] لقد ظفرنا بتصفيق الممثلة الوحيدة للرواية ‖`
- `قال همام أسعد الله الصباح [no cut] أين زاهر يا مدام ‖`
- `قال هدئي من روعك [no cut] إنما ثناء أردت لا ملامة ‖`
- `قالت جو أسرعا [no cut] لقد حضرت أمنا [no cut] خبئا السلة ‖`
- `وصاحت لا بد أنه خطاب [no cut] يعيش أبي ‖`
- `قالت مارمي كم أنا سعيدة لأنكن تحظين بوقت ممتع [no cut] هل أمضيتن يوما سعيدا ‖`
- `صاحت الفتيات شكرا لك على هدية عيد الميلاد يا مارمي [no cut] لقد قرأنا جزءا منها …`
**Practical form: after قال/قالت/صاحت/أضافت/أومأت, an opening INTERJECTION, GREETING,
VOCATIVE, IMPERATIVE or كم-exclamative never breaks away. Cut only once the quote has
delivered a complete DECLARATIVE statement and another full clause follows** —
e.g. `صاحت جو سأشتري لها خفا من الساتان ‖ أفضل خف يمكنني الحصول عليه ‖` (this one cut),
`قالت ميج في إصرار سيكون هذا الشتاء قاسيا علينا جميعا ‖ لا ينبغي لنا شراء الهدايا … ‖`.

**Sub-law 9d — cut BEFORE a narrative وقال / وقالت.** MISSED twice:
`… نعيم ذلك اليوم الرائق الصافي الجميل ‖ وقالت ماذا تعني ‖` ·
`… أم راثية لحاله ‖ وقالت ضيف ولكن لا أظنه طويل المقام ‖`
But NOT when the speech verb follows an asyndetic narrative clause in a graded reader:
`… رأت الفتيات أن هذه الفكرة رائعة [no cut] قالت ميج إنها ستشتري لها قفازا ‖` (false cut of mine).

**Sub-law 9e — when the introducer is a FULL NARRATIVE CLAUSE (not a bare speech verb),
the quotation DOES break off.** MISSED:
`… بكلمات لا مقدمة لها ولا سابقة لتفسيرها ‖ كم لك من وجوه يا سارة ‖`
Also `وأشار إلى أن خوفه يتعارض تماما مع المنطق بالتأكيد ‖ في كل الأحوال لا أعرف …` —
note the adverb بالتأكيد attaches BACKWARD to the introducer, not forward to the quote.

**Sub-law 9f — in literary narrative، و + finite verb still cuts, ف does NOT.** Confirmed
misses: `… ومشفق تارة أخرى ‖ ويعزو تقلبها …` · `… كما تصلح للخامسة والعشرين ‖ وتسمى آنسة …` ·
`… لم توافق هواها ‖ وسمعها تجيب …` · `… طروبا متهللة ‖ وهو يرى فيما يرى …` ·
`… في ثياب الشرطة ‖ ويصيح أين المشاجرة وأين المتشاجرات ‖`.
Confirmed false fa-cuts: `… ثم ينساه [no cut] فأسفرت عن الغضب …` ·
`… إنما خلقنا للسرور نأخذه ونعطيه [no cut] فمن نذر المرأة للعذاب …` ·
`… واندفعت ضاحكة [no cut] وما زالت حتى أجبرت هماما …`
And **ثم + a bare NP does not cut**: `… اقتراب الشفاه بداهة وطواعية [no cut] ثم نكتة من نكاتها …`
And **parallel negations do not cut**: `… أن يلتقي بسارة [no cut] ولم تقصد سارة أن تلتقي بهمام …`

**Sub-law 9g — إنما CAN cut.** I had written "إنما is never cut off from the negation it
corrects". Half wrong. `… يومض في عينيها ‖ إنما عز عليها أنه جعلها شيئا مهملا …` cut, and so
did my 498. But `لم نقل عنك شيئا وإنما أردنا تعريفك` did not. **وإنما (with و) = internal;
bare إنما after a completed statement = cut.**

---

## 10. Translated modern non-fiction (popular psychology, trade books)
Example: `doc_8921b794ce19` — F1 93.6, my best of batch 2. The reading in that answer file
mostly holds. Corrections:

- **بل cuts.** MISSED: `… ليس شيئا تافها كما يبدو ‖ بل يخفي بين طياته ثقلا مزعجا جدا ‖`
- **لذا does NOT cut** inside a first-person block quotation:
  `… في العربة معرفة شخصية [no cut] لذا ليس لدي شيء أخسره …` ·
  `… كما يتجاهلهم الآخرون جميعا أيضا [no cut] لذا فمن شبه المؤكد …`
  (Contrast §1: in journalism ولهذا/لذا behave differently. In an argument-chain the لذا is a
  comma; only cut before لذا when the preceding sentence is long and self-standing.)
- **وبذلك cuts.** MISSED: `… أنني سأعلن أسماء المحطات ‖ وبذلك يمكنكم القول إنني أؤدي خدمة عامة ‖`
- **و + finite verb with the SAME subject still cuts here more often than I assumed.** MISSED:
  `… دزينة أو نحو ذلك من الكلمات ‖ ويكشف لنا نحن القراء …` ·
  `… بحفاظنا على قيمة اجتماعية عالية ‖ ويعد الإحراج علامة …`
  *(this refines §5's claim that same-subject و-coordination stays inside in encyclopedic
  register — it holds for Wikipedia, not for translated trade non-fiction.)*
- **My cut before ولهذا السبب at 1095 was CORRECT.** So §3's "ولهذا never cuts" is now
  narrowed to: *ولهذا with no following noun, mid-argument, in a Nahda essay* — not a
  general law.
- Section headings that are bare NPs (حقل الألغام الاجتماعي، الألم الاجتماعي) are their own
  units — confirmed right, both times.

---

## 10-BIS. Graded readers / simplified abridgements (children's Little Women)
Example: `doc_56781db1019f` — F1 89.7, precision 85.6.

- **Asyndetic narrative clauses do NOT all cut.** False cut:
  `دقت الساعة السادسة تماما [no cut] وضعت بيث خف والدتها أمام المدفأة لتدفئته ‖`
  and `أخذت جو تسير في غرفة المعيشة [no cut] وبعد دقائق قليلة انفجرت الفتيات …`
  and `وأدت كل واحدة منهن دورها بطريقتها [no cut] قلدت ميج صوت الفلوت …`
  In a graded reader the units are ~13 tokens but they are built by *pairing* two short
  events, not by isolating each one.
- **فنحن / فـ + pronoun DOES cut.** MISSED:
  `… شراء أشياء لنا والاستمتاع قليلا ‖ فنحن نكد في عملنا ‖`
- **A trailing temporal clause belongs to the sentence BEFORE it, not after.** My worst
  structural error here: `… في قضاء أمسيتهن في الخياطة بعد أن حملت هانا الخادمة الصحون إلى
  المطبخ ‖ ذكرت السيدة مارمي بناتها …`. I put the boundary before بعد أن. **Fronted
  عندما/بينما attaches forward; POST-posed بعد أن/عندما attaches backward.** Read the linear
  position, not the conjunction.
- **أما … فـ … + a following و-clause is one sentence:**
  `أما بيث فقد اشتاقت … إلى العزف على البيانو [no cut] واشتاقت إيمي إلى أقلام التلوين …`
- **و + imperative does NOT cut here** (overturning §3 for this register):
  `… سنذهب للتسوق غدا [no cut] وتذكرن أن علينا التمرن …`

---

## 11. THE CHECKLIST I ACTUALLY USE NOW (replaces §6)
1. **Name the register in one sentence before reading further.** Everything else follows
   from it. The four buckets: (a) journalism/chronicle → cut freely at و;
   (b) original expository (encyclopedia, science, institutional) → و and ف are commas;
   (c) narrative fiction/memoir → cut at و + finite verb, never at ف;
   (d) translated/simplified prose → cut at the source-language period, ~13–17 tokens.
2. **Scan for display lines first**: chapter numbers, titles, speaker labels, headings. Each
   is its own unit (§9, §10). Speaker labels are one token each.
3. Then walk the tokens. At every و/ف apply the two-question test of §0-BIS.
4. Never cut before a trailing specification (§7a) or a postposed temporal clause (§10-BIS).
5. Inside a quotation introduced by a bare speech verb, weld the opening interjection (§9c).
   Inside a label-led play turn, cut at every ? and ! (§9b).
6. Density sanity check by bucket: journalism 15–18 · encyclopedic 20–24 · science 20 ·
   novel 10–12 (labels!) · translated non-fiction 17 · graded reader 13.
   **If I am above the bucket I under-cut; if below, I over-cut.**


---
---

# BATCH 3 (mean F1 89.49) — THE DOCUMENT-TYPE LAW

Scores: novel 85.1 (P 80.8) · fairy tale 86.1 (R 81.0) · grade-1 textbook 88.2 · poem
textbook 88.1 (P 80.4) · **MCQ question bank 100.0**.

## 12. FIRST: IS THIS RUNNING PROSE AT ALL?
Two of the five documents in this batch were not prose. Before anything else I now ask
whether the file is a **layout artefact** — a textbook page, a worksheet, a question bank,
a table, a front-matter block. If it is, syntax is irrelevant and only layout matters.

**12a — The MCQ question bank (`doc_25ed373f8aeb`, F1 100).** Stem + four options, twenty
times. Every stem is a unit, every option is a unit, including one-token options
(`أريحا`, `يافا`, `القدس`, `الملك`). Recognise the type by: trailing incomplete stems
(`… في`, `… هو`, `… لـ`), the frame `واحدة من الآتية ليست من …`, and near-identical
distractors clustered in fours. **This is the only document I have ever scored 100 on.
When the layout is regular, trust the layout absolutely.**

**12b — Front matter is one unit per line.** `المكتبة الخضراء للأطفال ‖ الليمون العجيب ‖
بقلم عادل الغضبان ‖ الطبعة الثالثة ‖` — all four correct.

**12c — Verse is one unit per printed line, and the line is SHORT.** In the poem textbook
my hemistich reading scored recall 97.4. The two I still missed were even shorter than a
hemistich: `حمض سموم غبار ‖ كونت مطرا ‖` and `الأرض تصرخ ‖ من يأتي يساندها ‖`. **When in
doubt inside verse, split smaller.** Same in the grade-1 songs:
`ماجد صفي جميل ‖ فسيح ومرتب ‖` and `يا مدرستي ‖ يا مدرستي يا مدرستي ‖ يا لحن الحب على شفتي ‖`
— I ran all of these too long.

---

## 13. THE LABEL LAW — **A LABEL WELDS TO ITS VALUE.** (my biggest batch-3 lesson)
I cut labels off from what they introduce, everywhere, and it cost me ~35 false cuts across
the two textbooks. The gold does the opposite in almost every case:

- speaker label + speech: `ماجد ما أجمل مدرستي ‖ الأب ماذا تعلمت اليوم يا ماجد ‖
  الأم المدرسة تبني جيلا يخدم وطنه ‖ نورة أنا أحب معلمتي كثيرا ‖`
- field label + value: `اسم الشاعر د معتز علي القطب ‖` · `حياته من مواليد مدينة القدس
  وسكانها حاصل على درجة الدكتوراة … للبحث العلمي ‖` · `كتاباته ديوان أخر صورة لمولاتي وديوان
  رسالة إلى مولاتي ‖`
- item letter + item: `أ أختار عنوانا بديلا … ‖` · `ب أحدد من أبيات القصيدة ما يأتي ‖` ·
  `ت أكتب رقم البيت … ‖` · `ث أمنيات ودعوات للإصلاح ‖`
- table row-header + cell: `الأول المواقع التي ينتشر فيها التلوث ‖` · `الثالث مصدر قلق الناس ‖`
  · `الأكاسيد تنتج من اتحاد الأكسجين … ‖` · `الغازات هي إحدى حالات المادة … ‖`
- exemplifier + example: `المثال أتعلم في صف واسع ‖` · `القصة الرافدة المكتبة قصصي الممتعة ‖`
  · `القصة الرافدة أصنع كتابي من كتاب قصصي الممتعة ‖`
- rubric + its word list: `عن معنى المفردات الآتية البيئة الطقس الأوزون ‖` ·
  `أضع حول المقطع الممدود بالياء في الكلمات الآتية معلمي الكتب غصني أمي ‖` · `في قائمة أ ب ‖`
- adjacent header cells run together: `رقم البيت المطلوب ‖ الإجابة ‖` ·
  `العمود الثاني المعنى ‖` · `الكلمة المعنى المعجمي المعنى السياقي ‖`

**13a — This NARROWS §9.** The Sara playlet remains the exception, not the rule: there the
label is a *stage-play display line* and stands alone. Everywhere else — textbook dialogue,
field cards, tables — **the label is the head of its own unit and welds forward.**
Test: is this a printed مسرحية with the same names cycling as turn markers? Then split.
Otherwise weld.

**13b — What DOES stand alone:** section headers with nothing on their line —
`أستوعب ‖ أفكر ‖ أتذوق ‖ أتحدث ‖ أقرأ ‖ أنشد ‖ بأداء معبر ‖ أتعرف وأنطق ‖ بطاقة الشاعر ‖
الدرس الثاني ‖ صفي ‖`. A header is a header when the next token starts a *new kind of thing*
(a rubric, a title, a poem), not when it starts the header's own content.

**13c — Word lists are genuinely inconsistent and I should stop trying to be clever.**
Gold gave `ممرضة ‖ ممرضات ‖ معلمون ‖ معلم ‖` (all split, I paired them) but
`معلمي الكتب غصني أمي ‖` (all joined, I split them) and
`معلمي أبي جدي الباحة أمي الكتب ‖ صفي ‖ مدرستي ‖ جدتي ‖ عائلتي ‖` (six joined then four split)
and `هذا وردة قمر ‖ هذه كتاب قصة ‖ أنا مجتهد مبتكرون ‖ نحن أذكياء مبدع ‖` (threes and twos).
This is two-column board layout leaking through and it is not recoverable. **Splitting every
item still won the trade** (recall 92.7 on that document), so keep splitting — but mark them
`lo` and expect to lose precision there.

---

## 14. Prose corrections from batch 3

### 14a — Staccato first-person novel (`doc_0a957c052364`, P 80.8): فأنا NEVER CUTS
Every one of `فأنا أحب عمتي` (30), `فمعنى وجود القروش` (62), `فأنا شخصيا كإنسان` (649) was a
FALSE CUT, and so were `ولكنني لا أعرف` (616) and `ولكن من يدري لعل` (727).
**In a first-person argumentative voice, فـ + pronoun/noun and ولكن + pronoun are commas.**
Also no cut at `أنا لا أنسى هذا اليوم أنا فرح والآخرون في حزن` (275) — asyndetic
first-person repetition is ONE breath — and none at `كان السائق صالح يقود السيارة وكانت معه
الخادمة` (298) or `كل ما تصبو إليه نفسي وكنت لا أشعر بالسرور` (93).
What DID cut that I missed: `حتى مرضت أمي ‖ وأصبحت دادة عاجزة` (351),
`فقد كنت أنا سعيدا ‖ وما دمت سعيدا فأنا أحب أن أتكلم بل العجيب …` (412 — and note **بل did
NOT cut here**, contradicting §10; بل cuts only when it heads a finite clause reversing a
negation), `أنا لا أدري لماذا أكتب هذا الكلام ‖ هذا الحديث جميعا لماذا أسوقه ‖` (565).

### 14b — §9c CONFIRMED and refined: the introducer swallows the first interjection
`وتسارع بالإجابة لا أبدا ‖ وماذا يمكن أن يحدث ‖` — exactly as §9c predicted; my cut at
`بالإجابة` was false and the real boundary was two tokens later.
**Refinement (from the fairy tale): the introducer + the opening VOCATIVE form one unit, and
then it cuts.** `وقال يخاطب والده مولاي ‖ إن لم أجد …` and
`يا ولدي ويا عصا شيخوختي ودم قلبي ‖ أي فكر غريب جال في خاطرك ‖`.
And **§9d confirmed**: cut BEFORE a narrative وقال — `وتغير تفكيره فجأة ‖ وقال يخاطب والده …`.

### 14c — Children's fairy tale (`doc_099def456da2`, R 81.0): I under-cut the fa- chain
I applied §8 (fa- stays inside) too hard. Gold cut at
`غابت عن ناظريه ‖ فاعتمد رأسه بكفيه …` · `ويحدقن هن إليه ‖ فما أجدى بحثه …` ·
`في مكان من الأمكنة ‖ فأنا أحبها بل أذوب بها غراما …`
but did NOT cut at `في قلب الملك [no cut] فقد خيل إليه …`.
> **Working rule for فـ: فقد + an explanation of what was just said → NO cut. Any other fa-
> opening a fresh event or a fresh assertion after a CLOSED clause → CUT.**
Also missed: `ذهبت هذه الكلمات وغيرها ضياعا ‖ وبقي الأمير …` and
`بنصائح أبيه ‖ وأرسل معه خادمين أمينين وضم … مودعا وصعد …` (the cut is at the FIRST wa- of
the chain, not the last). And **حتى إذا never cuts**: `الرأي الذي يرضيه [no cut] حتى إذا تعب …`

---

## 15. CHECKLIST v3 (replaces §11)
1. **Is it prose?** If it is a worksheet, textbook page, question bank, table or front
   matter → §12. Segment by layout, split small, ignore syntax. Expect 4–6 tokens/unit.
2. If it is prose, name the register (§0-BIS four buckets) and set the density target.
3. **Labels weld forward (§13).** Headers with nothing of their own stand alone.
4. Verse: one unit per printed line, and shorter than feels right (§12c).
5. Speech: bare introducer + first interjection/vocative = one unit, then cut (§14b);
   cut before a narrative وقال.
6. فـ: `فقد`-explanation no; fresh assertion yes (§14c). `فأنا` in a first-person essayistic
   voice: no (§14a).
7. Density check: quiz 6 · textbook 4–6 · verse 4–6 · graded reader 13 · novel 10–13 ·
   journalism 15–18 · translated non-fiction 17 · fairy tale 17 · encyclopedic 20–24.


---
---

# BATCH 4 (mean F1 83.65) — THE TWO LAWS I ALREADY OWNED AND FAILED TO APPLY

Scores: comic strip 97.2 · hadith 85.7 · **Yemeni draft constitution 84.5 (P 74.4)** ·
**Shawqi's children's poems 67.2 (P 51.2)**. Both failures were precision collapses, and
both were caused by ignoring a law already written above.

## 16. VERSE: **THE LINE ENDS AT THE RHYME.** (§12c is hereby corrected)
I split Shawqi at the hemistich and produced 254 units where gold has ~130. Every second cut
was false: `هرتي جد أليفة وهي للبيت حليفة ‖ هي ما لم تتحرك دمية البيت الظريفة ‖`.

> **Method, and it is mechanical: read the last word of the first few chunks, find the
> RHYME LETTER, then cut at every token that carries it — and nowhere else.**

This also explains why my hemistich cuts scored 100% inside the poem-textbook qasida last
batch: there the rhyme -لا fell on EVERY chunk (an urjuza/muzdawija where both halves rhyme),
so bayt and hemistich coincided and I could not tell the readings apart. Shawqi's
`الهرة والنظافة` rhymes -يفة on every SECOND chunk, and gold followed the rhyme.
Checks that would have caught me: the مطلع rhymes twice (تصريع) and then the rhyme spacing
doubles — that doubling IS the signal that the line is a full bayt.
Recall was 97.7, so I was finding the right places and simply cutting twice as often;
**in verse, over-splitting is the only failure mode I have, and the rhyme is the fix.**

## 17. LEGAL INSTRUMENTS: **THE ARTICLE IS THE UNIT, NOT THE CLAUSE.**
`doc_361233315d24` (Yemeni draft constitution, 15 472 tokens): recall 97.8, precision 74.4.
354 false cuts, and essentially every one of them was me splitting a wa-clause out of an
article that gold keeps whole:
`… وإسقاطها محظور [no cut] وينظم القانون حالات اكتساب الجنسية اليمنية …` ·
`… ملك للشعب [no cut] وتكفل الدولة الحفاظ عليها …` ·
`… عادل ومنصف [no cut] وتكون الضريبة على الدخل تصاعدية [no cut] والتهرب الضريبي جريمة …` ·
`… والنظام المصرفي [no cut] ويحدد المقاييس والمكاييل والموازين ‖`
This is nothing but §0-BIS bucket (b) — *in original expository Arabic و and ف are commas* —
which I wrote down two batches ago and then abandoned the moment the document got long.

**17a — §13 applies to statutes too: the SIDE-HEADING WELDS TO ITS ARTICLE.**
`الملكية الخاصة مصونة والتمتع والتصرف بها مكفول …‖` · `الاقتصاد الوطني اقتصاد حر اجتماعي …‖`
· `الشريعة الإسلامية مصدر التشريع والاجتهاد في تقنين أحكام الشريعة مكفول حصرا للسلطة التشريعية ‖`
**And so does the section number:** `الباب الأول الأسس العامة ‖ الفصل الأول الأسس السياسية ‖`
— ONE unit each, not two. (This overturns the graded-reader precedent
`الفصل الثاني ‖ عيد ميلاد سعيد ‖`: a children's book prints the number and the title on two
lines, a statute prints `الباب الأول: الأسس العامة` on one. When the title is a bare topic NP
rather than a narrative title, expect welding.)

**17b — What DOES cut in a statute:** the article boundary, and every item of an enumerated
list introduced by a colon phrase — and the colon phrase keeps its own trailing words:
`يقوم النظام السياسي على أساس ‖ الفصل بين السلطات …` (the boundary is after `أساس`, not before
it) and `… أو مذهبي كما يحظر عليها ‖ المساس بالنظام الجمهوري الديمقراطي ‖ الحصول على تمويل
خارجي ‖ استغلال الدين لأغراض سياسية ‖`. I had both of those a token or two off.

## 18. HADITH: the isnad breaks at every transmitter, the matn at every ḥāl
`doc_27a114d4e376`, F1 85.7. Corrections:
- **`حدثنا فلان ‖ قال حدثني فلان …` — the isnad is NOT one unit.** Gold cut at token 1.
- **`يقول` welds to the matn** — my cut at 21 was false. So §9c beats §9e here: however long
  the isnad, the final `qāla / yaqūlu` still marries the quotation.
- **The ḥāl chain of the matn splits:** `… ثائر الرأس يسمع دوي صوته ‖ ولا يفقه ما يقول حتى دنا ‖
  فإذا هو يسأل عن الإسلام ‖` — three units where I had one.
- Everything else held: `قال لا إلا أن تطوع ‖` stayed whole eight times, exactly as §9c predicts.

## 19. Comic strip (`doc_cb57e21e8421`, F1 97.2 — my second-best document ever)
Title / episode / scenarist / illustrator as four lines, then one unit per speech bubble,
splitting a bubble only at an internal full stop. One false cut and one miss in 36 boundaries.
**Layout documents are where I am strongest; prose is where I lose.**

## 20. FINAL CHECKLIST (replaces §15)
1. **Layout or prose?** Layout (quiz, worksheet, textbook, comic, front matter, table) →
   segment by display line, expect 4–6 tokens/unit, and trust the layout absolutely.
2. **Verse?** → find the rhyme letter, cut at every rhyme, nowhere else (§16).
3. **Statute/constitution?** → the ARTICLE is the unit; headings and section numbers weld;
   only list items after a colon phrase split (§17). Expect 20–30 tokens/unit.
4. **Prose?** → name the bucket (§0-BIS). Originally-Arabic expository: و and ف are commas.
   Narrative fiction: و + finite verb cuts, ف does not. Translated/graded: cut at the source
   period. Journalism: cut at nearly every و.
5. **Labels weld forward** in every register (§13, §17a). Only a header with nothing of its
   own on its line stands alone.
6. **Speech**: bare `قال X` + the opening interjection/vocative = one unit, then cut (§9c,
   §14b); cut before a narrative `وقال`; a label-led PLAY turn instead cuts at every ? and !.
7. **My standing bias by document type:** I over-cut verse, statutes and expository prose;
   I under-cut literary narrative and fairy tale. Set the knob before writing, not after.
