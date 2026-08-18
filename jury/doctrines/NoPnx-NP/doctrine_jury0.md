# Doctrine — Jury 0 — track `nonp`

Cumulative. Never delete; when a later batch contradicts an entry I rewrite it in place and say so.

---

## 0. THE MASTER CALIBRATION (learned batch 1)

**Batch 1 result: F1 85.34. Precision 94–100 on every document, recall 60–86.
Every single point I lost, I lost by NOT CUTTING.** 46 of my 51 errors were MISSED; only 5 were
FALSE CUT. My instinct is far too conservative.

### 0.1 The density law
Gold sentence length, back-computed from my recall on all four genres of batch 1:

| document | genre | gold sentences | tokens/sentence |
|---|---|---|---|
| doc_3f6c5f1e305a | modern translated book (Hindawi) | ~131 | ~16 |
| doc_44ee993e046e | Arabic Wikipedia | ~83 | ~17 |
| doc_66c98646b9a9 | classical (Ibn Tufayl) | ~58 | ~16 |
| doc_1d513dc2c22a | classical (Ibn Tufayl) | ~57 | ~16 |
| doc_c567a2257203 | children's magazine | ~58 | ~14 |

**The unit is 14–17 tokens in EVERY genre.** Classical philosophy is not longer-sentenced than
a kids' magazine. This corpus's sentence is a CLAUSE, not a thought.
**Self-check before submitting any document: ntokens / nboundaries. If it is above ~18, I have
missed boundaries and must go back and cut more.**

### 0.2 The bias rule
Since P ≈ 1.0 and R ≈ 0.6–0.9, a coin-flip site is worth cutting. At a 50% hit rate an added cut
is roughly F1-neutral; at anything above 50% it is a gain. **When genuinely torn about a
clause-initial connective: CUT.** I skipped ب `و` at Majid @590 (`تقسم بين الفريقين ‖ وعلى خط
الوسط يقف الحكم`) after explicitly considering it. It was gold. Do not do that again.

---

## 1. THE CONNECTIVE LAW (applies to ALL text types — the single biggest lesson)

I wrote in batch 1 that in classical Arabic "the chained connectives ف and و are INTERNAL
punctuation — they carry the argument forward inside one period." **THIS WAS COMPLETELY WRONG
AND COST ME ~40 POINTS OF RECALL ON TWO DOCUMENTS. It is hereby reversed.**

> **و and ف at the head of an independent clause are BOUNDARIES, not commas.**

Even when the resulting sentence is two or three words long.

### 1.1 Proof from my own misses — `و`
- `doc_1d513dc2c22a` @157–162: gold = `... والثالث إلى رابع ‖ ويتسلسل ذلك إلى غير نهاية ‖ وهو باطل ‖ فإذن لا بد للعالم من فاعل`
  — **`وهو باطل` is a two-token sentence.** I ran all of this into one period.
- `doc_66c98646b9a9` @320–337, the list of the five senses. Gold:
  `... عند تصادم الأجسام ‖ والبصر إنما يدرك الألوان ‖ والشم يدرك الروائح ‖ والذوق يدرك الطعوم ‖ واللمس يدرك الأمزجة والصلابة واللين والخشونة والملاسة ‖ وكذلك القوة الخيالية ...`
  Five boundaries in twenty tokens. I gave zero.
- Same doc @103 `دونه ‖ وتتبع`, @163 `العلم ‖ وهو هو`, @203 `فيه ‖ وذهل`, @469 `بذاته ‖ ورسخت`,
  @504 `ذاته ‖ وإنما`, @533 `جسمه ‖ وجعل`, @638 `عنه ‖ وقد كان`, @925 `منها ‖ وتبين`.
- `doc_1d513dc2c22a` @129 `الأجسام ‖ ولو كان`, @143 `محدث ‖ ولو كان`, @232 `والعمق ‖ وهو منزه`,
  @579 `الأول ‖ ولم يضره`, @589 `حدوثه ‖ وصح له`, @718 `قديمة ‖ وهو في ذاته غني`.
- Modern book: @494 `الخارج ‖ والتهديد`, @1199 `فارقا ‖ ويمكن`, @1297 `المتخصص ‖ ونرسم`,
  @1414 `جديد ‖ وسوف تحدث`, @1708 `باستمرار ‖ وتظهر`.
- Wikipedia: @948 `التنوير ‖ وتشمل`, @1134 `قوية ‖ وتشمل`.
- Majid: @360 `اللوح ‖ ويحاول`, @402 `بريطانيا ‖ وانتشرت`, @535 `الخشنة ‖ ويمكن`,
  @590 `الفريقين ‖ وعلى خط الوسط يقف الحكم`.

### 1.2 Proof from my own misses — `ف`
`doc_66c98646b9a9` @63 `فعله ‖ فعلم`, @229 `حينه ‖ فينتقل`, @284 `الموجود ‖ فتصفح`,
@370 `بانقسامها ‖ فهي لذلك`, @428 `الاتجاهات ‖ فإذن لا سبيل`.
`doc_1d513dc2c22a` @71 `الاعتقادين ‖ فلعل`, @357 `أسفل ‖ فإنه إن قسم`, @455 `له ‖ فالواجب`,
@520 `يدرك ‖ فإن وجود العالم`, @668 `به ‖ فهو إذن علة`, @694 `قط ‖ فإنها`, @856 `علمه ‖ فتبين`.

### 1.3 Subordinators that ALSO open a new sentence
I had filed these as internal. All wrong:
- **`إذ`** — @815 `شمه ‖ إذ الأشياء`, @610 `عنه ‖ إذ الاتصال`, @292 `الابتداء ‖ إذ لم يسبقها سكون`.
- **`كما أن` / `كما أنك`** — @883 `لها ‖ كما أن من كان`, @775 `بالزمان ‖ كما أنك إذا أخذت`.
- **`حيث`** — Majid @27 `لدى استقبال سموه لهم ‖ حيث أهدى نجوم الفريق`.
- **`ولكن` / `ولكننا`** — @1338, @2019, @928. (But see the exception in 1.5.)
- **Asyndeton** (a finite verb or fresh topic NP with NO connective after a complete
  predication) remains what it always was: a certain boundary. That rule I had right.
- **Embedded Qur'anic tags** get their own unit: @256 `وعالم به ‖ ألا يعلم من خلق وهو اللطيف الخبير`,
  @890 `وفوق الكمال ‖ لا يعزب عنه مثقال ذرة في السماوات`.

### 1.4 What stays INTERNAL after `و` — the narrow exceptions
- **`و` + a SUBORDINATING conjunction** (`وعندما`, `وإذا`, `ولما`, `وحين`, `وهنا`): the
  و-clause is only a protasis, its apodosis follows, so the whole thing is one sentence.
  FALSE CUT @504 `ما نرى أنه الآخر ⟨no cut⟩ وعندما يتشكل هذا التوجه العقلي الجديد ...`
  FALSE CUT (Majid) @617 `ملعب الخصم الهول ⟨no cut⟩ وهنا يحاول اللاعب الآخر ...`
- **`إلا أن`** — never a boundary. FALSE CUT @468 `بتطلعات البشر ⟨no cut⟩ إلا أن السيناريو يتغير`.
- **A tight parallel copular chain sharing ONE subject** stays internal:
  `فهو الوجود وهو الكمال وهو التمام وهو الحسن وهو البهاء وهو القدرة وهو العلم ‖ وهو هو`.
  Contrast the five-senses list, where each و-clause has a DIFFERENT subject and its own verb —
  that splits. **Operational test: و + new subject + its own predicate = CUT.
  و + another predicate hung on the same subject = keep.**
- **`و` coordinating two verbs under one governing particle** stays internal — e.g. Majid
  `ليتحقق الفوز للعيناويين ... وتفرح جماهير العين`, both under the same subjunctive `لـ`.
  I got this one right; keep doing it.

### 1.5 Irreducible noise, do not over-fit
The gold is mechanical transcription of the source's real punctuation, so the same connective
sometimes goes either way and no rule recovers it. `ف` was a boundary at @1093 (`فنحن نعتقد`)
but internal at @965 (`فظهر توافق طيب`); `ولكن` was internal at @1093 but a boundary at @2019.
Accept a few losses here rather than distorting the main rule. **The main rule is CUT.**

---

## 2. TEXT TYPE: front matter, colophon, title page, dedication, signature block
*(learned from `doc_3f6c5f1e305a`, the Hindawi translation of Rothstein & Perlmutter)*

The annotation follows **physical display lines and raw punctuation characters, mechanically**.
This is much finer than I assumed.

- **A field LABEL and its VALUE are two separate units.** I lost @3, @10, @13:
  gold = `تأليف ‖ دانيال دي بيرلمتر وروبرت إل روثستاين ‖ ترجمة ‖ أحمد شكل ‖ مراجعة ‖ ضياء وراد`.
  I had glued `تأليف` to the author names. Same for `ترجمة`, `مراجعة`.
- **A LATIN INITIAL is followed by a boundary**, because the source's `L.` / `D.` dots are taken
  literally: gold = `Robert L ‖ Rothstein and Daniel D ‖ Perlmutter`. A lone capital letter token
  in a Latin name = cut after it.
- **Dots inside URLs and e-mail addresses are boundaries.** gold =
  `البريد الإلكتروني hindawi hindawi ‖ org` (from `hindawi@hindawi.org`) and
  `الموقع الإلكتروني https www ‖ hindawi ‖ org` (from `https://www.hindawi.org`).
  So: after every domain-name component in a stripped URL, CUT.
- **Publisher block splits at its internal period**: `الناشر مؤسسة هنداوي ‖ المشهرة برقم ... بتاريخ ...`.
- **Dedications split into address-line + wish-line + signer**:
  `إلى زوجتي فيليس وأحفادي شيمس ونوا وليف ‖ مع أمل في مستقبل أفضل لجميع الأحفاد ‖ دانيال دي بيرلمتر`
  and `إلى زوجتي الحبيبة جين ‖ أنت مصدر إلهام لجميع دعاة حماية البيئة ‖ روبرت إل روثستاين`.
- **A signature block splits name from affiliation**:
  `دانيال دي بيرلمتر ‖ جامعة بنسلفانيا ‖ روبرت إل روثستاين ‖ جامعة كولجيت`.
- Headings (`تمهيد`, `الفصل الأول`, `البداية`) are their own units — I already had this right.

**Rule of thumb for any display region: cut at every place a typesetter would have started a new
line, and at every literal `.` the stripper removed — including abbreviation dots and URL dots.**

---

## 3. TEXT TYPE: modern expository / translated non-fiction (Hindawi book)
Body prose, ~16 tokens per sentence.
- Apply the connective law of §1 in full.
- **A sentence-initial hedge or frame adverbial attaches FORWARD, to the sentence it opens, not
  backward.** FALSE CUT @1871: I cut after `على ما يبدو`; gold cuts BEFORE it —
  `تم اختراقها على نحو غير قانوني ‖ على ما يبدو كشف عدد قليل من الرسائل ...`.
  So `على ما يبدو`, `في الواقع`, `بالطبع` etc. begin their sentence.
- **Appositive restatement is its own sentence**: @907
  `يقدم شيئا ذا قيمة ‖ شيئا يختلف عن التحليلات ...` — the repeated-head apposition splits.
- **A fragmentary afterthought splits too**: @853
  `وما نحتاج كلنا ‖ أفرادا ودولا ومؤسسات دولية إلى القيام به ...`; @928–933
  `كيف يمكن للمرء ألا يكون مهتما بها ‖ ولكن بخلفيتين علميتين مختلفتين تماما ‖ فأحدنا أستاذ ...`
  — note `ولكن بخلفيتين علميتين مختلفتين تماما` is a verbless three-phrase fragment standing as a
  full unit. **Verblessness is NOT evidence against a boundary anywhere in this corpus.**
- `وكذلك` opens a new sentence: @267 `على الفصل الحادي عشر ‖ وكذلك نشكر المراجعين المجهولين`.

---

## 4. TEXT TYPE: Arabic Wikipedia article
My strongest document (F1 98.2). What worked: short one-claim sentences; section headings as
their own units; new sentence at a fronted adverbial frame or a bare initial verb.
- Remaining leak: `و + تشمل` is a boundary (@948, @1134). Consistent with §1.
- **One real correction — the false-heading trap.** FALSE CUT @243: I cut
  `الحقول الفرعية الرئيسية ‖ تشمل الفلسفة الأكاديمية ...` treating the NP as a section heading.
  It is the grammatical SUBJECT of `تشمل`. **Test before declaring a heading: does the finite verb
  immediately after the candidate agree with it and lack any other subject? Then it is a subject,
  not a heading.** A true heading is followed by a clause that has its OWN independent subject.

---

## 5. TEXT TYPE: classical Arabic philosophical narrative (Ibn Tufayl, حي بن يقظان)
My two worst documents (75.3, 74.7), both purely from the reversed connective law.
**Superseded entry:** batch 1 I wrote "a period is a whole reasoning step, typically 25–60 tokens,
built as [فلما/فإن premise] ... [فـ conclusion]". **Deleted as false.** Replacement:

- The unit is ~16 tokens — one clause, one inferential move, not one whole syllogism.
- `ف` opening a consequence (`فعلم`, `فرأى`, `فتبين`, `فينتقل`, `فإذن`, `فهو`, `فإنه`, `فالواجب`)
  = **new sentence**, every time.
- `و` opening a next step (`وتتبع`, `وذهل`, `ورسخت`, `وجعل`, `وصح`, `ولو كان`, `ويتسلسل`) =
  **new sentence**, every time, even at two tokens (`وهو باطل`).
- `إذ`, `كما أن`, `ألا يعلم`, Qur'anic tags = **new sentence**.
- Still internal: `لأن`, `حتى`, `إلا أن`, `سواء`, relative `الذي/التي`, and a copular chain on one
  subject (`وهو الوجود وهو الكمال ...`).
- What I already had right and keep: `فلما`-protasis stays attached to its apodosis;
  rhetorical questions close a unit; `فإذن`-conclusions open one.

---

## 6. TEXT TYPE: children's / popular magazine page (Majid)
Layout-order extraction, mixed captions + headlines + factoid blocks. F1 91.7.
- **Display lines break finer than I thought.** Misses:
  @44 `مبروك ل العين ‖ كأس الإمارات لكرة القدم` (banner = two stacked lines);
  @766 `فريق نادي العين ‖ بطل كأس صاحب السمو رئيس دولة الإمارات لعام 1999` (caption splits
  subject from predicate — they were two typeset lines);
  @805 `وتفرح جماهير العين فرحة ما بعدها فرحة ‖ بهذا الفوز بالكأس الغالية ...`.
  **In caption/headline regions, suspect a break at every 4–8 token chunk, not one per caption.**
- `حيث` opens a new sentence (@27) — do not treat it as a relative link.
- Section headings (`رفع أثقال`, `أبطال فوق العادة`, `دراجات نارية`, `ألعاب شعبية`, `الهوسة`)
  as their own units: all confirmed correct. The doubled-word signal (`الهوسة الهوسة`) is a
  reliable heading detector.
- Only false cut was @617 `الهول ⟨no cut⟩ وهنا يحاول` — covered by §1.4 (و + `هنا`).

---

## 7. CHECKLIST TO RUN ON EVERY DOCUMENT BEFORE WRITING THE ANSWER
1. Identify display regions (front matter, headings, captions, signature blocks) vs body prose.
2. In display regions: one unit per typeset line; split label from value; split at every
   removed `.` including initials and URL dots.
3. In body prose: walk every clause-initial `و` / `ف` / `ثم` / `إذ` / `حيث` / `كما أن` / `ولكن`
   and CUT unless it falls under §1.4 (و+subordinator, `إلا أن`, same-subject copular chain,
   two verbs under one particle).
4. Cut at every asyndetic finite verb or fresh topic NP.
5. Attach sentence-initial frame adverbials FORWARD.
6. Compute ntokens / nboundaries. **If > 18, go back to step 3 and cut more.**
   *(Superseded by §15 after batch 2.)*

---
---

# BATCH 2 (F1 86.43) — the corrections

Batch 2: 84.1 / 88.9 / 87.1 / 82.4 / 89.7. **The error profile INVERTED.** Batch 1 was
P 94-100 / R 60-86 (under-cutting). Batch 2 was P 74-89 / R 76-98 — 65 FALSE CUTS against 52
MISSED. I applied §1 too hard and too uniformly. Most of what follows is correction, not addition.

## 8. THE DENSITY LAW WAS WRONG — it is per-document, not universal

**Supersedes §0.1.** There is no 14-17 band. Gold sentence length, back-computed from both batches:

| document | register | tokens/sentence |
|---|---|---|
| doc_54f919517225 | graded-reader translation (Tolstoy for young readers) | **9.9** |
| doc_405d5fba81e0 | literary translation (Byatt) | 13.9 |
| doc_c567a2257203 | children's magazine page | 14.0 |
| doc_226fc860e550 | Wikipedia, plain concrete topic (air well) | 15.3 |
| doc_3f6c5f1e305a, doc_66c98646b9a9, doc_1d513dc2c22a | modern book / classical philosophy | 16 |
| doc_eb278f2980be | literary novel (al-Aqqad) | 16.6 |
| doc_44ee993e046e | Wikipedia, philosophy | 17.0 |
| doc_47764cf032d7 | Wikipedia, academic topic (history of scientific method) | **20.5** |

**The range is 10 to 20.5 — a factor of two.** Register predicts it, and I must fix a target
BEFORE walking the text:
- simplified / graded narrative for learners or children -> ~10
- ordinary narrative, magazine page, translated literary fiction -> ~14
- classical Arabic argument, literary essay, general book prose -> ~16
- encyclopedic Wikipedia -> ~15 for a plain concrete topic, 18-20 for an academic/abstract one

The tell for the long end is abstract subject matter with stacked relative clauses; the tell for
the short end is one concrete event per clause and a small vocabulary. I under-cut
doc_54f919517225 by 30% (gold 147, I gave 125) while over-cutting doc_47764cf032d7 by 30%
(gold 38, I gave 50) **in the same batch**. One global setting cannot serve both.

## 9. حيث IS NEVER A BOUNDARY — this reverses §1.3

My single biggest systematic error in batch 2: **seven false cuts**, all before حيث.
doc_47764cf032d7 @116 وأعماله في المنطق / حيث رفض; @394 علم الفلك العلمي / حيث كان;
@515 النظريات العلمية العقلانية / حيث بدأت.
doc_226fc860e550 @165 يختلف الندى عن الضباب / حيث يتكون; @254 على درجة الحرارة / حيث يكون;
@372 في جزيرة لانزاروت / حيث يوجد.
**حيث attaches backward. Always. It is a comma.** My one batch-1 datapoint
(Majid @27 استقبال سموه لهم ‖ حيث أهدى) was a caption LINE break, not a حيث rule.
**بينما likewise internal:** FALSE CUT @195 خارج نطاق الإدراك / بينما في منتصف القرن.
But **في حين DOES cut** (@84 والمدارس البوذية ‖ في حين رفضت — true positive), and
**كما DOES cut** (@613 كما انتهجت, @623 كما تشككت, @708 كما كانت — all true positives).

## 10. THE TWO DOT EXPERIMENTS — both came back NO

- **ق.م does NOT split.** FALSE CUT @319, @368, @530: سنة 1600 ق / م على العناصر.
- **A bare credit label does NOT split from its value.** FALSE CUT doc_54f919517225 @8:
  gold is قصة ليو تولستوي ‖ ترجمة أماني سالم ‖ — one unit per credit LINE, label attached.

**Revised §2:** the mechanical dot-splitting in doc_3f6c5f1e305a (Robert L ‖ Rothstein,
hindawi ‖ org, https www ‖ hindawi ‖ org) is NOT a general rule about dots. It is the
display-line law applied to a PDF colophon whose fields were typeset one per line. Outside such a
page, abbreviation dots, decimal dots and URL dots are all invisible. Do not cut on them.

## 11. WHAT ACTUALLY GOVERNS و + FINITE VERB — refining §1.1

§1 said "و + finite verb = CUT, regardless of subject continuity." Right in narrative, wrong in
expository prose. The rule is register-dependent.

**Wikipedia / expository: و + verb continuing the SAME subject is INTERNAL.**
False cuts, all same-subject: @92 الشارفاكا رفضت ... وفضلت; @121 أرسطو رفض ... وفضل;
@462 and @466 أسسوا ... وكذلك أسسوا ... ووضعوا; @766 وزن إيراسيستراتوس ... وأشار;
@559 استخلص زيبولد ... وحسب; @58 and @82 وقد استخدمت / وقد حققت.
Cut only when the SUBJECT or TOPIC changes — @21 وكان المنهج, @31 وتجادل الفلاسفة,
@254 وظهر أثر ذلك, @406 ويزعم المؤرخ, @491 ولكنهم were all true positives.
**This corrects §1.4:** subject continuity DOES block the cut in expository prose. What batch 1
actually proved is only that it fails to block it in NARRATIVE.

**Graded-reader narrative: cut at essentially every connective.** doc_54f919517225 gold has 147
boundaries in 1453 tokens and I still missed 35: ف + verb (@134 فجمعا, @698 فأبيع, @922 فرآها,
@968 فجلس), و + verb (@150 وذهب, @163 وكان, @221 وأخبره, @305 وأصبح, @397 وحصلت, @798 وعلق,
@863 ونظر, @941 وتابع, @1172 وأصبح, @1233 وأصبحت, @1361 ورأى, @1374 فوقع), ثم + verb (@490 ثم أشار,
@623 ثم تسير, @814 ثم وقف, @834 ثم توقف, @938 ثم انحرف, @1049 ثم استدار, @1416 ثم تركوه) and
لكن (@18, @1124, @1167, @1335 وكاد يتوقف ‖ لكنه رأى).
**In this register لكن and ثم are periods, not commas** — the opposite of what §3 and §6 assumed.

## 12. FRONTED ADVERBIALS ATTACH FORWARD — the off-by-one is a DOUBLE penalty

§3 had this; batch 2 shows it is worth much more than I gave it. A cut placed one token late
scores a FALSE CUT *and* a MISSED — two errors from one mistake. Six of them this batch:
- @68/@69 باءت المحاولات بالفشل ‖ لاحقا ومنذ أواخر القرن العشرين فصاعدا استخدمت — لاحقا leads the
  NEXT sentence even though it reads as the tail of the previous one.
- @1175/@1172 لكنه كان لا يزال بعيدا ‖ وأصبح السير مرهقا فرمى سترته
- @1229/@1233 وفقد السيطرة على قدميه ‖ وأصبحت دقات قلبه
- @1303/@1310 مادا جسمه إلى الأمام ‖ حتى أصبح من العسير
- @1328/@1323 وصل أخيرا إلى أسفل التل ‖ وفي نفس اللحظة أظلمت الدنيا فصاح
- @943/@947 قالت ستبكي ولا شك لا أسألك في ذلك ‖ ولكن كم عبرة
**Procedure: once I decide a boundary exists in a stretch, scan LEFT to the start of the
adverbial phrase and put the cut in front of it.**

## 13. Subordinators that DO open sentences — extending §1.3
- **لأن** — MISSED @1086 مديرة للإضاءة في مسرح تمثيل ‖ لأنها تعلم مواقع الرؤية. §5 had لأن as
  always-internal. Wrong; it can carry a period.
- **حتى / حتى أن** — MISSED @1310 إلى الأمام ‖ حتى أصبح, @240 بارتفاع ظهر الحصان ‖ حتى أن فلاحا.
  Also wrongly filed as internal in §5.
- **إلا أن** — MISSED @470 قوائم للأنواع كلها ‖ إلا أن البابليين. §1.4 said never; batch 1 gave the
  opposite case. Genuinely ~50/50 — decide by sentence length, not by the word.
- **وأن + coordinate complement clause** — MISSED @58 لا تفقه ما تقول ‖ وأنها بمحاكاة المعترفات.

## 14. Appositives, one-word units, and long object lists
- @282 وكأنه إله ‖ شجرة المران الرمادية إجدراسيل — appositive splits.
- @414 كانت تلتهم القصص بشراهة ‖ سطور من العلامات السوداء — a COLON can split after all, when what
  follows is a full appositive restatement. Narrows my batch-2 "colons never split" ruling.
- @21 ورهبة الصوت ‖ ماذا ‖ فيما دون العاشرة — a ONE-WORD question ماذا is its own unit.
- @790/@819/@828/@841: my 69-token صورت إحدى الرسومات sentence was wrong to leave whole. Every
  و-coordinated object that carries its OWN finite verb (ونهرا يجري, وقمم غابات ... تغطي,
  وأناسا ... يحدقون, وأطياف غيوم ... تعلقت) is a separate sentence; only وجذوعا stayed attached.
  **Corrects my batch-2 claim that a long و-object list is necessarily one predication.**
- @376 نهاية العالم ‖ ١ البداية ‖ — a chapter NUMBER stays with its chapter title; do not isolate
  the numeral. And @780: of my three-beat split only the first was real —
  نهاية حقيقية نهاية العالم is ONE unit.

## 15. REVISED CHECKLIST — replaces §7
1. Name the register and FIX A TARGET tokens/sentence from §8 before cutting anything.
2. Display regions: one unit per typeset line. No dot-splitting outside a PDF colophon.
3. Never cut before حيث or بينما. Do cut before في حين and كما; consider لأن, حتى, إلا أن.
4. و / ف / ثم / لكن + clause: in NARRATIVE cut almost always; in EXPOSITORY cut only when the
   subject or the topic changes.
5. Cut on asyndeton, on every question and exclamation, on appositive restatement.
6. Place every cut BEFORE the fronted adverbial, never after it.
7. Count. If tokens/boundary is more than ~20% off the §8 target for that register, revisit.

---
---

# BATCH 3 (F1 92.04) — display-line law confirmed, connective law still leaking

Batch 3: 88.8 / 86.9 / 96.2 / 97.8 / 90.5. The two school-textbook pages scored 96-98, which
confirms §2/§6 and §8. The residue is 37 FALSE CUTS against 27 MISSED — still slightly
over-cutting, and now almost entirely on و/ف coin-flips.

## 16. DENSITY TABLE, extended (adds to §8)
| doc_32908af53b40 | school textbook page, list-dominated | **8.6** |
| doc_501500835788 | primary workbook spread | **8.1** |
| doc_70d28c9dfebb | Islamic-education lesson, mixed display + exegesis | 9.2 |
| doc_5b0751f40176 | children's magazine story | ~11 |
| doc_7bc0656a48ce | popular-science magazine miscellany | ~15 |
A page whose bulk is headings, bullets, rubrics and table cells lands at 8-9 tokens per unit
because a display line is not a sentence. Recognising that the document IS a layout, before
reading it as prose, is worth more than any connective rule.

## 17. ASYNDETIC VERB PILES DO SPLIT — I got this backwards
I wrote in batch 3 that a breathless run of bare verbs is one clause. Wrong:
- @598/@600 gold: `كان ذيله يكبر ‖ يزداد طولا ‖ يتضخم ‖` — three units where I gave one.
- @947 gold: `وكان جسمه كله يصغر ‖ يصغر حتى عاد إلى حجمه الطبيعي`.
**A finite verb arriving with no connective is a new sentence even when it repeats the previous
one.** The exception, confirmed at @603 (a FALSE CUT of mine): a bare verb that is a RELATIVE
modifier of the immediately preceding indefinite noun stays attached —
`يتحول إلى هيليوم تنطلق منه الطاقة` is one clause because تنطلق describes هيليوم.

## 18. حيث AND بينما ARE LENGTH-GOVERNED, not absolute — softening §9
§9 said حيث is never a boundary, on 7/7 evidence from batch 2. Batch 3 gives 2/2 the other way:
- MISSED @485 `أغرب علاقة بين الاثنين ‖ حيث نرى تمساحا صغيرا`
- MISSED @709 `المنزل الذي نراه في الصورة ‖ حيث تنتشر على سطحه مئات الخلايا`
And بينما, which §9 also banned, was a MISSED at @465: gold `على النباتات المائية ‖ بينما التماسيح
مشهورة بأكل اللحوم تنقض على ضحاياها`.
**Revised rule: حيث / بينما attach backward by default, BUT cut before them when leaving them
attached would produce a sentence far longer than the register's target.** In batch 2's Wikipedia
the joined sentences were already 12-18 tokens, so they stayed joined; here they would have been
30-40, so they split. And when I do cut, cut BEFORE the word — at @470 I cut one token late and
paid the double penalty again.

## 19. TEXTBOOK / WORKBOOK LAYOUT — the fine print
- **Title + reference are ONE line.** FALSE CUT @2: `الاقتداء برسول الله سورة الأحزاب 27 21 ‖`.
- **A statement and the instruction that acts on it can be one line.** FALSE CUT @259:
  `تتميز دولتنا الحبيبة الإمارات بموقع استراتيجي هام أدلل على صحة العبارة ‖`.
- **But an activity RUBRIC VERB can be its own line, separate from its content** — and this varies
  by book. In doc_70d28c9dfebb: MISSED @71 `أتوقع ‖ ما يمكن أن يحدث لو كان الرسول ملاكا`,
  @84 `أتلو وأحفظ ‖ سورة الأحزاب Q`, @342 `أستنتج ‖ مما سبق أثر الأسوة الحسنة`. In
  doc_501500835788 the same shape (`أكتب اسم أول من رفع علم دولتي الحبيبة`) stayed joined and
  scored 97.8. So: check whether the rubric verb stands alone ELSEWHERE in the same book before
  deciding, and follow that book's habit.
- **Table cells are separate units, one per cell.** MISSED @92/@94:
  `المفردة ‖ تفسيرها ‖ وفى ‖ عهده ‖ حالفوهم وناصروهم ‖ حصونهم ‖`. I had merged the header pair and
  read `وفى عهده` as a single gloss; in fact وفى is the head-word and عهده its gloss.
- **A stripped fill-in blank creates a boundary.** MISSED @477:
  `تقع دولة الإمارات العربية المتحدة إلى جهة ‖ من شبه الجزيرة العربية ‖`.
- **Timeline rows can split date from event.** MISSED @658/@665: `الألفية الرابعة ق م ‖ عصور ما قبل
  التاريخ ‖ القرن السابع الميلادي ‖ دخول أهل الإمارات والخليج في الإسلام طواعية`. The numeric rows
  (1805 م الحملة الإنجليزية الأولى على القواسم) stayed whole, so this is layout, not grammar.
- **A boxed quotation separates from its attribution.** MISSED @446 `عن أهمية التضامن العربي بقوله
  ‖ إن الحل الوحيد`, @890 `اعترف به العالم أجمع ‖ فقال إننا نعيش`. This narrows my batch-2 ruling
  that a speech tag always runs into its speech: that holds for prose dialogue, not for a
  textbook's display quotation.
- **A dialogue turn in a workbook is ONE unit, question included.** FALSE CUT @29:
  `ماجد رأيت اليوم بيوتا كثيرة وقد رفع عليها علم الإمارات لماذا يا أبي ‖`. So the question rule
  does NOT override the display-line rule.

## 20. THE و/ف COIN-FLIP — trim about a fifth
The remaining loss is almost all here, and it is now symmetric noise rather than a rule I lack.
FALSE CUTS on و/ف/ثم I should not have made: @80 وقد لا تجد, @164 وأخذ ينبح, @210 ثم انطلق,
@394 وكان الصوت, @528 وتقدم, @567 ثم دس, @604 وكانت عيناه, @779 وسوف أجعل, @828 وأحس, @862/@870
ولم يكن, @943 وكان جسمه, @290 واتجهت, @684 وتنطلق, @729 وتنير, @399 ويطل, @540 وقد ساعدت,
@712 ومارسوا, @499 فأخرجهم, @514 وفتح, @581 ثم إنه, @587 والله, @616 ويكون, @1164 فعملوا.
MISSED on the same shapes: @762 وقالت, @915 وأنها, @535 وتهرب, @405 ومد, @901 وإنه, @961 فجعلها,
@172 فما زادهم.
Operational fix: **cut at و/ف only where a genuinely NEW event or NEW topic begins; keep it where
the clause only elaborates, negates in parallel, or completes the same action.** Parallel negatives
(`ولم يكن هناك ضوء ... ولم يكن في العين أي بريق`) and paired verbs on one act (`وتنير وتشغل`)
stay joined. When I cannot tell, prefer JOINING now — I am currently over-cutting 37:27.
Also: enumerated NOMINAL items do split (MISSED @80/@86 `استلزمت مراقبتها ‖ ثم اكتشاف قفزات بعضها
من الأحواض ‖ وسيرها على أقدام غريبة`), so a ثم/و list of gerunds behaves like bullets, not like a
noun coordination.

---
---

# BATCH 4 (F1 83.59) — two perfect scores and two instructive failures

Batch 4: 81.2 / **100.0** / **100.0** / **37.5** / 99.2. Only 5 FALSE CUTS against 31 MISSED —
I over-corrected §20's "prefer joining" instruction into a new round of under-cutting.

## 21. VERSE AND SCRIPTURE — the two documents I got exactly right
The comic strip (100.0), Antara's Mu'allaqa (100.0) and Surat an-Nahl (99.2) all came from the
same insight: **when the source is set one record per line, find the record and stop reasoning
about grammar.** For a qasida the record is the BAYT; for the Qur'an it is the AYA; for a comic
it is the balloon. Every و/ف rule in this doctrine is suspended for such texts.
**And each such text carries its own hard verification test — use it instead of my ear:**
- Antara: the qafiya is mim, so every bayt-final token MUST end in م or مي. My first pass failed
  at exactly one index (299 `هر`), which exposed that the line
  `وكأنما تنأى بجانب دفها ال * وحشي من هزج العشي مؤوم` splits the word الوحشي across the caesura
  and the next bayt OPENS with `هر جنيب`. Corrected to 298. Result: 89/89.
- an-Nahl: 128 ayat, and every fasila is known. I listed all 128 verse-final words and matched
  them forward through the token stream. Result 127/128.
- **The one miss is the sharpest lesson of the batch: a fasila can repeat INSIDE its own verse.**
  Aya 20 is `والذين يدعون من دون الله لا يخلقون شيا وهم يخلقون` — يخلقون twice — and my
  first-match scan stopped at the first one, giving a FALSE CUT at 212 and a MISSED at 215, the
  double penalty. **When matching a known ending list, take the LAST occurrence before the next
  ending, not the first.**

## 22. HADITH — my worst document of the whole training (37.5) and why
I read the isnad as one long sentence with colons and keyed the boundaries off the question marks.
Both halves of that were wrong. Gold for doc_e1435023282b:
`حدثنا عبد الله بن محمد ‖ قال حدثنا أبو عامر ‖ قال حدثنا سليمان بن بلال المديني عن ربيعة ... سأله
رجل عن اللقطة ‖ فقال اعرف وكاءها أو قال وعاءها وعفاصها ‖ ثم عرفها سنة ثم استمتع بها ‖ فإن جاء
ربها فأدها إليه ‖ قال فضالة الإبل فغضب ... أو قال احمر وجهه ‖ فقال وما لك ولها معها سقاؤها
وحذاؤها ترد الماء وترعى الشجر ‖ فذرها حتى يلقاها ربها ‖ قال فضالة الغنم قال لك أو لأخيك أو للذئب`
- **The isnad DOES break at each قال**: `حدثنا فلان ‖ قال حدثنا فلان ‖ قال حدثنا ...`. It is a
  chain of records, not one sentence. The عن-links inside one transmitter's report do NOT break.
- **ف and ثم at the head of a new ruling step ARE boundaries** (فقال, ثم عرفها, فإن جاء, فذرها).
- **The questions are NOT boundaries.** `قال فضالة الإبل فغضب حتى احمرت وجنتاه` is ONE unit and
  `قال فضالة الغنم قال لك أو لأخيك أو للذئب` is ONE unit. My three "certain" question cuts at
  @60, @72 and @86 were all false. So the terminator rule I built in batch 2 from the al-Aqqad
  novel does not transfer to classical hadith, where the printed text simply has no question marks.
- Density ~9.3 tokens per unit, not the 15.5 I assumed.
**General principle behind the failure: I applied a rule (questions terminate) learned in one
register to a register whose punctuation conventions are entirely different. Before importing any
terminator rule, ask whether that mark exists in this kind of printed text at all.**

## 23. MAGAZINE LETTERS COLUMN — narrow columns split at every physical line
doc_c188bbc9702e: P 98.1 but R 69.3. Gold has ~75 units in 420 tokens — **5.6 tokens per unit**,
the densest prose document I have seen, because a narrow magazine column breaks constantly.
- **A reader's signature is at least TWO lines: NAME, then PLACE.**
  `صالح محسن القرين ‖ كريتر اليمن الديمقراطية`, `شادي محمد إسماعيل ‖ الرياض السعودية`,
  `إبراهيم العبدان ‖ الرياض السعودية`, `محمد حسني متولي ‖ الجيزة مصر`, `نائل سليم خليف ‖ عمان الأردن`,
  `عبد الله سعيد علي ‖ السويق سلطنة عمان`, `أحمد بالقاسم خليفة ‖ المدينة المنورة السعودية`,
  `عثمان سالم حسن ‖ الدوحة قطر`, `أحمد خميس محمد ‖ بيادر وادي السير الأردن ص ‖ ب 141242`.
  **This restores the batch-1 Hindawi rule (`دانيال دي بيرلمتر ‖ جامعة بنسلفانيا`) that I wrongly
  retired in §10.** The correct statement is narrower than either of my previous versions:
  a LABEL stays with its VALUE (`ترجمة أماني سالم`, `الرسام نبيل تاج`), but a PERSON stays apart
  from their PLACE or INSTITUTION.
- And note `الأردن ص ‖ ب 141242` — the dot inside ص.ب **does** split. §10 said abbreviation dots
  never split, on the ق.م evidence. Both are true: it is layout, not orthography. Do not
  generalise either way; ق.م sits mid-line, ص.ب sat at a line break.
- Quoted titles split: `بكتابي كسلان جدا ‖ حول العالم ‖ وكسلان جدا والأسد`.
- بمعنى opens a unit: `حياد إيجابي ‖ بمعنى أن الدول غير المنحازة`.
- حيث opened one here (`يعنى الحياد ‖ حيث تلتزم`) even though joining would only have made
  20 tokens — so §18's length threshold for حيث is lower in a narrow column than I set it.
- Asyndeton I declined to cut was real: `هنا يلتقي ماجد كل أسبوع بأصدقائه ‖ يرد على أسئلتهم`.
  My §17 rule was right and my hal-clause exception was wrong.

## 24. FINAL CHECKLIST
1. **First ask what kind of RECORD the source stores**: bayt, aya, hadith link, balloon, bullet,
   table cell, column line, or prose sentence. Get this right and the rest is bookkeeping.
2. If the answer is verse/scripture/enumerated, find an external invariant (rhyme letter, known
   verse count, known fasila list) and verify against it; take the LAST match, not the first.
3. Otherwise set a density target from §8/§16/§23 — 5.6 for a narrow letters column, 8-9 for
   textbook layout and hadith, ~10 for graded narrative, 11-14 for stories and magazines,
   15-20 for encyclopedic prose.
4. Display regions: one unit per line; person apart from place; label with value.
5. Prose: cut on asyndeton, on و/ف/ثم that open a NEW event or topic, on appositive restatement.
   Keep حيث/بينما unless the joined sentence would run long for the register. Keep إلا أن، غير أن,
   لأن, relatives, و+subordinator, and و-chains that spell out one action.
6. Import a terminator rule only if that punctuation mark plausibly exists in this print tradition.
7. Place every cut BEFORE the fronted adverbial. An off-by-one costs twice.
