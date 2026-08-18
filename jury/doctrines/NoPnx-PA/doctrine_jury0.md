# Doctrine — Jury 0 — track `nopa`, retrain4

**Status:** cumulative. Nothing here is ever deleted. When a later batch contradicts an entry,
I edit that entry in place, tag it `REVISED (batch N)`, and keep the superseded wording visible
underneath as `~~old~~` so the reasoning trail survives.

---

## Batch log

| batch | docs | errors | MISSED | FALSE CUT | ratio |
|---|---|---|---|---|---|
| 1 | 18 (2 clean at F1 100) | 279 | 231 | 48 | 4.8 : 1 **under**-cutting |
| 2 | 23 | 370 | 16 | 354 | 22 : 1 **OVER**-cutting |

Worst docs, batch 1: `doc_e1435023282b` (hadith, F1 37.5), `doc_54f919517225` (graded Tolstoy, 62.0),
`doc_1d513dc2c22a` (Ibn Tufayl, 80.0), `doc_0a957c052364` (first-person memoir, 84.9).

Worst docs, batch 2: `doc_1d08b507b151` (Muwaṭṭaʾ hadith, **57.1**), `doc_0de3fe1bcb6c` (Tulunid art
history, 61.0), `doc_1522cf2ee149` (Exodus 8, 61.0), `doc_099def456da2` (literary folk tale, 69.6),
`doc_2fabc1ccf342` (financial news, 70.0), `doc_085c979b884f` (encyclopedia, 70.6),
`doc_4f0612ce35d1` (Matthew 20, 72.2), `doc_4259e5d2a449` (belletristic essay on music, 73.1).

---

# LAW 0 — REVISED (batch 2). THE UNIT IS TYPOGRAPHIC, AND THE TYPOGRAPHY IS DOCUMENT-SPECIFIC.

> **This is the most important correction in the file. Batch 1's LAW 0 was wrong in its central
> claim and I acted on it, which is why batch 2 inverted my error profile from 4.8:1 under-cutting
> to 22:1 over-cutting. I did not learn a rule in batch 1; I learned a base rate, and a base rate
> is not transferable.**

`~~OLD (batch 1) — SUPERSEDED, kept for the trail:~~`
> ~~"LAW 0 — THE BASE RATE. I under-segment by roughly 5 to 1. … Arabic source texts punctuate far
> more densely than my instinct — a full stop lands roughly every 6–14 tokens in most of these
> registers. **Operational consequence: at a coin-flip site, CUT.** … Every mark in the original that
> ended a unit — `.` `؟` `!` `:` `؛` **and often `،`** and a line/table break — became a `‖`."~~

**The two errors in that paragraph:**

1. **`،` (comma) is NOT a boundary.** This one clause is responsible for something like 300 of my
   354 false cuts in batch 2. A comma-separated list, an appositive, a parenthetical gloss, a
   circumstantial ḥāl clause — all of these were commas in the source and all of them **weld**.
2. **"6–14 tokens" was a property of the batch-1 sample, not of the corpus.** Batch 1 happened to be
   loaded with short-unit registers (Bukhārī with punctuated isnād, graded readers, worksheets).
   Batch 2 is loaded with long-unit registers (Bible-by-verse, encyclopedia, literary prose,
   MCQ stems). The invariant is **the source's own unit system**, never a token count.

### LAW 0′ — what a `‖` actually is

A `‖` is a **sentence-final or layout-final mark in the source**, and nothing else:

- `.` `؟` `!` `؛` at the end of a real sentence
- a line, cell, list-item, or balloon break (`<NL>`, table cells, MCQ options)
- **a verse number** in scripture (see §M) — which overrides even internal full stops
- a colon **only** when it is a layout colon (rubric verb, caption label), never a prose colon

Everything below that level — `،` `(...)` `«...»` dashes, decimal points, Latin glosses, honorific
abbreviations — is **inside** the segment.

### LAW 0″ — REVISED AGAIN (batch 13). **The default is CUT at a fresh finite predication.**

`~~OLD (batch 2) — SUPERSEDED:~~` ~~"the operational default flips back to WELD. At a coin-flip site,
DO NOT CUT. Cut only when I can name the typographic reason."~~

**Batches 11, 12 and 13 falsified the WELD default as hard as batch 2 falsified the CUT default.
Precision ran 96–100% while recall ran 57–77%: ~50 misses against ~8 false cuts, three batches
running. "Default weld" is the single most expensive sentence I have ever written in this file.**

The corrected posture is neither "cut" nor "weld" but a **protected-list** procedure:

> **CUT at every clause boundary where a fresh finite predication begins after a complete
> preceding predication — UNLESS the site is on the protected list in LAW 0‴.**

This is not a return to batch 1. Batch 1 cut at *connectives*; this cuts at *predications*, and it
carries an explicit exemption list built from the 354 false cuts of batch 2. The connective (و ف ثم
لكن كما حيث إذ) is neither evidence for nor against — **what matters is whether a new finite clause
with its own predicate starts there, and whether the site is protected.**

### LAW 0‴ — NEW (batch 13). THE PROTECTED LIST — the only places that never take a `‖`

1. **و joining NPs, adjectives or list items** (not clauses). `والشمعدانات والأضرحة` welds;
   `‖ وكلها موشاة` cuts. *The test is whether a finite verb follows the و.*
   > **NARROWED (batch 14) — this covers only و INSIDE ONE NP. A و that introduces a fresh
   > NOMINAL CLAUSE (new topic NP + its own predicate) CUTS.** Two misses proved it in one batch:
   > `doc_c0adf987ac2e` @330 gold `…تشهد لها الحقبة الإمبراطورية المتأخرة ‖ والنتيجة الأخرى هي التحول…`;
   > `doc_ca06bbd92e01` @123 gold `…ثم أعود إليك ‖ ولك علي عهد وميثاق أني أعود إليك…`.
   > The batch-2 weld I was leaning on (`…ليس سوى مضخة والعقل ليس أكثر من آلة دقيقة ‖`) is now the
   > MINORITY case, not the rule. Ask: is the و adding an item to a list, or starting a new
   > predication? Only the first is protected. See §Y1.
2. **Gapped coordination** — a second predicate with no new subject and no new verb
   (`ورثت بشرتي من أبي ولون عيني من عينيه`), or two verbs sharing everything
   (`يحبونها ويقدرونها` welds, `‖ وكان حديث الناس` cuts).
   > **NARROWED (batch 14) — a DISCOURSE connective is not gapped coordination even when the
   > subject is unchanged.** `doc_c0adf987ac2e` @408 (MISSED) gold
   > `…أهمية الخلفية الثقافية لهذه الظاهرة ‖ ومن ثم وجهوا الأنظار إلى النزعة الإنسانية…` — same
   > subject, same agent, still a new sentence. **ومن ثم / ومن هنا / وبالتالي / ولذلك-as-opener cut.**
3. **Apposition, parenthetical gloss, Latin/Cyrillic transliteration, OCR artifact.**
4. **Inside a number**; **name + honorific/title**.
5. **Between a speech verb and its quotation.**
   > **WIDENED (batch 15) — an ATTRIBUTION SWALLOWS ITS WHOLE REPORT, 30–36 tokens and more, in
   > encyclopedic prose as much as in dialogue.** `doc_dc967c8520e9` cost me 5 false cuts inside the
   > Mariti / Robinson / Bauerer reports. The boundary goes in FRONT of the attribution verb and
   > nowhere after it until the NEXT SOURCE is named; a second verb of saying by the same source
   > (`وأشار`, `وذكر`) is inside the report, and even a brand-new subject inside it (`ويصدر قسم منها`)
   > welds. This is the most reliable protected site in the corpus after `<NL>`.
6. **Inside an MCQ stem** (including embedded quotations and example lists).
7. **Inside a scripture verse** — but the verses themselves cut, and Qur'anic verses are SHORT (§W).
8. **A subordinate clause trailing its main clause**: الذي / لأن / ~~حتى~~ / عندما / لكي / أن / رغم أن /
   منها / وذلك / خاصة / وفقا لـ. **NOT** ولكن/لكن/بل/أما — those are coordinate and they cut.
   > **حتى STRUCK OFF THE LIST (batch 14).** `doc_c0adf987ac2e` @447 (MISSED) gold
   > `…قوانين تهدف إلى إلغاء العبودية ‖ حتى أن البابوات أنفسهم كانوا يمتلكون عبيدا…`.
   > Split the word in two: **`حتى` + subjunctive verb ("until / in order to") welds**
   > (`لا أبرح من عندك حتى أنظر`, `فمشى إلى أن وصل`); **`حتى أن` + a full clause ("to the point
   > that") CUTS.** This also reconciles §V's `المعاملات ‖ حتى قال المؤرخون`.
9. **A fronted adverbial** (`قبل الآن`, `في عام 1913`, `وحينها`, `بل أكثر من ذلك`) — it attaches
   FORWARD, so the cut goes *before* it, never after. Confirmed again in batch 14 at four sites
   (`وفي هذا السياق`, `ومن هنا فإننا`, `ومن الناحية السياسية`, `أما من الناحية الاقتصادية فقد`),
   all correct.
10. **A two-part heading on one layout line** (`السنوات الأولى في فلورنسا التعليم والتكوين ‖`).
    > **EXCEPTION (batch 14) — a leading SECTION NUMBER is its own segment.**
    > `doc_c0adf987ac2e` @29 (MISSED) gold `<NL> 1 ‖ العبيد والخدم بين سقوط الإمبراطورية الرومانية والعصور الوسطى ‖`.
    > I welded the numeral to its heading on the strength of §S3 (an MCQ option LETTER welds to its
    > option text) and that transfer was wrong. **List/section numerals in body text are their own
    > segment; option letters inside an MCQ are not.** The discriminator is which register you are in.
    > Contrast, same document, and I got this one right: a FOOTNOTE marker at @160 stays inside the
    > sentence it annotates (`…سواء في الريف أم في المدينة 1 ‖`). Trailing marker → welds back;
    > leading numeral → cuts forward.

### LAW 0⁗ — NEW (batch 13). THE DENSITY IS ~10–16 TOKENS, NOT 25–45.

Measured from batch-13 recall arithmetic: `doc_a045e64631c7` ≈ 10 tok/sentence,
`doc_a74a9e93d97a` ≈ 12.5, `doc_a4029b2c8e1b` ≈ 15, `doc_7b090cf754e8` ≈ 11,
`doc_83e3dc997e52` ≈ 7. **The LAW 6 table below is inflated for almost every register** and I have
been pricing sentences at roughly double their true length. Long-unit registers (scripture verse,
encyclopedic art history, MCQ stems, classical hadith) are the *exception*, not the norm.
**If I finish a document and my mean segment is over ~18 tokens, I have almost certainly under-cut.**

### Confirmed prior — still 100%, two batches running

`<NL>` is always immediately preceded by a `‖`. A line break is a hard boundary, no exceptions in
41 documents. These are free. Take all of them.

---

## THE GENERAL LAWS (batch 1 numbering kept; all now qualified by LAW 0′)

### LAW 1 — REVISED (batch 2). و is NOT a boundary signal. The "new argument" test is dead.

`~~OLD (batch 1):~~` ~~"و + a fresh finite clause is a NEW SENTENCE (~120 of my 231 misses). If و is
followed by a clause that has its own predicate and its own new argument (new subject, new object,
new topic NP), cut before the و. … The test is: does a new argument appear after it?"~~

**What batch 2 shows.** The new-argument test fires on almost every و in Arabic prose and is
therefore useless as a cut criterion. It produced the largest single class of my false cuts, in
every register, including brand-new subjects:

- `doc_085c979b884f` @489 (FALSE CUT): I wrote `…القلب البشري ليس سوى مضخة ‖ والعقل ليس أكثر من آلة دقيقة…`
  gold: `…ليس سوى مضخة والعقل ليس أكثر من آلة دقيقة ‖ ولكن بالرغم من…`
  `والعقل` is a brand-new subject with its own nominal predicate — the strongest possible reading of
  LAW 1 — and it **welds**. The source had a comma. The cut is 8 tokens later, at the full stop.
- `doc_085c979b884f` @28 (FALSE CUT): gold `…بعد الحرب العالمية الأولى وضمت عدة اتجاهات فنية ظهرت آنذاك ‖ <NL>`
- `doc_3762b5d5da2f` @95 (FALSE CUT): gold `…وقد اختفت هذه العملة ولم يبق منها سوى ست قطع في المتاحف ‖`
- `doc_099def456da2` @183 (FALSE CUT): gold `…ولون القشدة واهتزت نفسه وتغير تفكيره فجأة ‖` — three و-clauses,
  three new predicates, ONE segment.
- `doc_566d3db2b5c9` @217 (FALSE CUT): gold `…ويرافق ذلك شعور باليأس وينخفض شعورهم بتقدير الذات ‖`
- `doc_3484c02411e9` @240/@245 (FALSE CUT): gold `…ورثت بشرتي البيضاء من أبي ولون عيني من عينيه الخضراوين وقصر نظري من قصر نظره…`

**What survives from LAW 1.** The *phrase-level* exemption from batch 1 was right and is now the
general case, not the exception. What has to be found is a positive reason to cut, and و by itself
is never one.

> ### LAW 1 — RE-REVISED (batch 13). و + FINITE VERB CUTS. و + NOUN WELDS.
>
> **The batch-2 wording above is over-broad and cost me ~40 misses in batches 11–13.** The correct
> statement is a *syntactic* one, not a semantic "new argument" test and not a blanket weld:
>
> - **و followed by a finite verb or a new predication → CUT** (the common case, most registers):
>   `doc_a74a9e93d97a`: `صناعة الطوب ‖ وبدأ حياته المهنية صائغا متدربا ‖ وما لبث أن طور مهاراته ‖`;
>   `الموجود في البندقية ‖ وكان قد شرع في تصميمه`; `ومستحضراتها المختلفة ‖ وكان أول من جرب`;
>   `لمريم العذراء madonnas ‖ وتظهر بقوة في الكهوف`.
>   `doc_9a890accffc8`: `الأكبر في ألمانيا ‖ وهو صاحب أكبر معدل`; `في حانة محلية ‖ وتم أخذ اسم النادي`;
>   `مصنع بوروسيا للجعة ‖ وتأخذ بوروسيا من اللغة اللاتينية`; `1 8 متر ‖ وتم بناء غرف`.
>   `doc_a045e64631c7`: `وأيقظت زوجها ‖ وحكت له ما رأت`; `وكبرت المولودة ‖ وأصبحت طفلة ‖ ولاحظ الصياد`.
>   `doc_7b090cf754e8`: `عند بعد المسافات ‖ وتساعد الناس على معرفة عادات`.
> - **و followed by a noun/adjective inside a list → WELD**: `والشمعدانات والأضرحة` (then
>   `‖ وكلها موشاة` cuts, one و later — the discriminator is visible in a single token).
> - **و with no new material (gapped, or two verbs sharing one subject and one object) → WELD**:
>   `يحبونها ويقدرونها ‖`, `ونسكن هذا البيت المتواضع`, `إياك نعبد وإياك نستعين`.
>
> The batch-2 weld examples that made me write "و welds" were all of the second and third kinds, plus
> scripture verses and comma-lists. **They were exemptions, and I promoted an exemption to a law.**

**The one و-shaped cue that does carry weight: `وقد` + perfect verb.** This marks a fresh assertion
and cuts more often than not:
- `doc_566d3db2b5c9` @189 (MISSED — my error): I wrote `…لا يستطيع المريض مغادرة فراشه وقد يصل إلى حالة تشبه الغيبوبة…`
  gold: `…مغادرة فراشه ‖ وقد يصل إلى حالة تشبه الغيبوبة ‖`
- gold also cuts before `وقد ظهرت أول وثيقة`, `وقد تمحور`, `وقد كانت هذه التوجهات` (`doc_085c979b884f`),
  `وقد اختفت` (`doc_3762b5d5da2f`), `وقد بلغت نسبة` (`doc_2fabc1ccf342`).
- **Counter-case, so this is a cue and not a law:** `doc_3484c02411e9` @289 gold welds
  `…لاحظت أنها تلهث وريقها جاف وقد لاحظت حالتها قبل أن تلحظها هي ‖`. Call it ~80%.

### LAW 2 — REVISED (batch 2). ف- and ثم- are NOT boundaries either.

`~~OLD (batch 1):~~` ~~"ف- and ثم- resumptive openers are sentence starts (~50 misses). `فكان، فقال،
فرأى، … فإنه، فكذلك، فقد` — when ف opens a new narrative step or a new explanatory move, cut before it."~~

Batch 2 falsified this across the board, including two forms the old entry named explicitly
(`فإنه`, `فقال`):

- `doc_0de3fe1bcb6c` @618 (FALSE CUT): I wrote `…لنتفهم أصول الزخارف الطولونية ‖ فإنه يجدر بنا أن نبدأ…`
  gold: `…أصول الزخارف الطولونية فإنه يجدر بنا أن نبدأ بدراسة مفصلة لتاريخها…`
- `doc_0623277721a2` @150 (FALSE CUT): gold `…فعلم يسوع أفكارهم فقال لماذا تفكرون بالشر في قلوبكم ‖`
- `doc_099def456da2` @113 (FALSE CUT): gold `…نخبة من رجالات الدولة فما أثرت فيه بلاغتهم ولا رجعته دموع أبيه عن عناده ‖`
- `doc_4259e5d2a449` @244 (FALSE CUT): gold `…كثيرا ما تجفل وقت شربها فيصفر لها راعيها فتطمئن وتشرب ‖`
- `doc_0c92fc467c35` @195 (MISSED, mirror error): gold `…وهو صبي ثم التحق بالكلية البحرية ‖ لعله يقابل…` —
  I cut before `ثم` and missed the real boundary before `لعله`. **ثم welded; the cut was one clause later.**

> ### LAW 2 — RE-REVISED (batch 13). ف/ثم/فقد/فهي CUT in exposition and in modern narrative.
> They weld only inside a *tight event chain* or a *conditional apodosis*.
>
> `~~batch-2 wording, now scoped:~~` ~~"Their presence should now count slightly AGAINST cutting."~~
> That is true **only** for classical literary narrative (§O) and scripture (§M). Everywhere else it
> is backwards, and it produced ~20 misses in batches 11–13:
> - `doc_a74a9e93d97a`: `معلم ليوناردو الأول ‖ فقد أمضى حياته ساعيا`; `محترف أعمال معدنية ‖ فقد كان مبرزا`;
>   `ثيمة متكررة في أعمال ليوناردو ‖ فهي تظهر تصويره الجبال`
> - `doc_75aefb3d0bfd`: `هذه القيمة الأخلاقية ‖ فهي توفر العديد من الأشياء`; `قادتنا الأعزاء ‖ فنتمنى أن نصبح`
> - `doc_7b090cf754e8`: `عن الحياة السعيدة ‖ فتؤدي بالفرد إلى الاختباء`; `بما تفعله بوقتك ‖ فالوقت كالسيف ‖`
> - `doc_a045e64631c7`: `هذا البيت المتواضع ‖ فكيف تصبح بنتنا ملكة ‖`; `أصبحت صيادة ماهرة ‖ فكانت تمسك بسنارة ‖`
> - `doc_6abdaa7e4427` (ثم): `من موت محتمل ‖ ثم نجوت من موت آخر ‖ ثم همت في البحر`
>
> **Rule of thumb: `فقد` / `فهي` / `فكان` / `فتؤدي` after a completed statement is a resumptive
> *new sentence*, not a continuation.** ف welds when it is the second beat of one action
> (`فنسي أنه يحمل سكينا وأتى بحركة فجرح إصبعا`) or the apodosis of a fronted condition.

ف and ثم are *continuation* markers in Arabic. They are evidence the writer was still inside a
sentence. ~~**Their presence should now count slightly AGAINST cutting.**~~ (scoped to §M/§O by
batch 13 — see the box above). Where ف does open a segment
(`doc_0de3fe1bcb6c` @8 gold `…فن ملكي بطبيعته ونقصد بذلك أنه مدين بكل شيء للسلطان ‖ فالمثالون والمصورون…`)
the reason is that a full stop was there, not that ف was there.

### LAW 3 — REVISED (batch 2). Comma-lists and appositions WELD. Only a run of complete printed sentences cuts.

`~~OLD (batch 1):~~` ~~"ASYNDETIC REPETITION is always cut … Two adjacent clauses with no connective,
where the second repeats the shape of the first (same verb, same `أنا X` frame, appositive
`شيئا … شيئا`), are separate segments."~~ ~~And B5: "A comma-set parenthetical list becomes two
boundaries, not zero."~~

This cost me my two worst runs of batch 2. Lists are ONE segment:

- `doc_0de3fe1bcb6c` @325–@350, **twelve consecutive false cuts**, my single most expensive run:
  I wrote `…نذكر منهم كوربت CORBETT ‖ وفان برشم VAN BERCHEM ‖ وسلمون SALMON ‖ …`
  gold: `…نذكر منهم كوربت CORBETT وفان برشم VAN BERCHEM وسلمون SALMON وهرتز باشا HERZ PACHA وسلادان SALADIN وزره SARRE وهرتزفلد HERZFELD وشتريجوفسكي STRZYGOWSKI وفلوري FLURY وكريزول CRESWELL وعكوش وهوتكير HAUTECOEUR وفييت WIET ‖`
  **Thirteen names with parenthesised Latin transliterations = ONE segment.**
- `doc_3484c02411e9` @164–@173 (FALSE CUT ×5): gold `…فوق الكباري القديمة والجديدة التي تجتاز النيل كوبري الجيزة كوبري الجامعة كوبري الجلاء كوبري قصر النيل كوبري ٦ أكتوبر ‖`
- `doc_3484c02411e9` @366–@379 (FALSE CUT ×7): gold `…هؤلاء الملايين النساء الشباب الشابات الرجال الأطفال العجائز هذه الوجوه من كل الأعمار والأشكال والبقاع ملايين الملايين تملأ الشوارع والميادين ‖`
- `doc_099def456da2` @46–@50 (FALSE CUT ×4): gold `…ويختار له عروسا نبيلة غنية جميلة رقيقة الحاشية طيبة القلب ‖`
- `doc_0de3fe1bcb6c` @86–@93 (FALSE CUT ×4): gold `…إلى الأسرات الحاكمة فيقال فن أموي وفن عباسي وفن طولوني وفن فاطمي إلخ ‖`
  (note: the prose colon after `فيقال` is also not a boundary — see §A revision)
- Appositions weld too: `doc_27b80bb5ed9c` @195 gold `…به آلاف الكريات الكرات الصغيرة جدا المصنوعة من البلاستيك والمدمجة فيه ‖`;
  @216 gold `…وتنقسم كل كرية إلى نصفين نصف أبيض والنصف الآخر أسود ‖`;
  `doc_085c979b884f` @224 gold `…للفن اللاتشبيهي … في روسيا آنذاك السوبرماتية والبنائية ‖`;
  `doc_566d3db2b5c9` @404 gold `…الطبيب النفسي الألماني إيميل كريبلن أحد الرواد في دراسة هذا المرض…`
- Asyndetic clause chains weld: `doc_3484c02411e9` @450–@457 gold
  `…ينشق الكون عن عساكر جيش وبوليس ملابسهم عادية بعضهم بالجلابيب والشوارب واللحى الطويلة يحملون البنادق والسياط والنبال ‖`

**What survives.** Batch 1's cutting cases (`أنا أقوى الأقوياء ‖ أنا عظيم العظماء ‖`,
`كان ذيله يكبر ‖ يزداد طولا ‖ يتضخم ‖`) were **exclamatory full sentences**, each closed with `.`/`!`
in the source. The corrected discriminator is therefore NOT parallelism (T3 hypothesis: **dead**) but:
> **Was each element a complete printed sentence, or a comma-separated item inside one?**
> Bare NPs, adjectives, names, appositives, ḥāl clauses → one segment. Independent exclamations
> with their own end-stop → separate segments. **When unsure, weld** — in batch 2 the weld was right
> every single time.

### LAW 4 — HOLDS, and generalises (batch 2).

A subordinate clause that FOLLOWS its main clause belongs to it. Batch 2 confirms this and widens the
list of trailing adjuncts that must not be split off:

- `منه / منها` (elaboration): `doc_3ccbb7784a8d` @4 (FALSE CUT) gold `التسامح شيء مهم في الحياة منه نتعلم كيف أن نصبح أكثر تعاطفا ‖`;
  `doc_142afcb751a9` @79 (FALSE CUT) gold `…للحصول على الشهرة والمال بمختلف الطرق منها تصميم ونشر تصاميمهم…`
- `وذلك`: `doc_2fabc1ccf342` @20 (FALSE CUT) gold `…عن مستوى ال 70 دولارا وذلك في آخر تعاملات الأسبوع…`
- `خاصة`: `doc_2fabc1ccf342` @46 (FALSE CUT) gold `…مع إمكانية مواصلة الانكماش خاصة مع ضعف مستويات الاستهلاك ‖`
- `وفقا لـ` (attribution tail): `doc_2fabc1ccf342` @72 (FALSE CUT) gold `…منذ نحو 15 عاما وفقا لبيانات وزارة التجارة الأمريكية ‖`
- `رغم أن`: `doc_3762b5d5da2f` @47 (FALSE CUT) gold `…فيظهر منيرا في سماء الليل رغم أنه لا يعكس من ضوء الشمس إلا حوالي…`
- `لأن`: `doc_4f0612ce35d1` @271 (FALSE CUT) gold `…هكذا يكون الآخرون أولين والأولون آخرين لأن كثيرين يدعون وقليلين ينتخبون ‖`

**The one counter-example, and it is a scripture/verse effect:** `doc_0623277721a2` @401 (MISSED)
gold `…ومست هدب ثوبه ‖ لأنها قالت في نفسها إن مسست ثوبه فقط شفيت ‖` — a trailing `لأن` that DOES cut,
because a verse number falls there (Matt 9:20|21). See §M: **the verse index overrides LAW 4.**

Mirror case from batch 1 (short forward-attaching adverbials: `على ما يبدو`, `في الواقع`, `أخيرا`)
still stands; batch 2 adds `مع أن` as forward-attaching: `doc_566d3db2b5c9` @589 (MISSED) gold
`…وكأنها مسيرة المسيح إلى صليبه ‖ مع أنه يعرف جيدا أنه استحم آلاف المرات…`.
Note the tension with `رغم أن` above — logged as T7.

### LAW 5 — HOLDS and is now a bigger share of my residual (batch 2).

Off-by-one-clause remains a real, separate failure mode. In batch 2 nearly every one of my 16 misses
was paired with a false cut 1–2 clauses away — I had the right idea and the wrong site:

- `doc_0c92fc467c35` @191/@195: mine `…وألف العديد من القطع الموسيقية وهو صبي ‖ ثم التحق بالكلية البحرية لعله يقابل…`
  gold `…وألف العديد من القطع الموسيقية وهو صبي ثم التحق بالكلية البحرية ‖ لعله يقابل في البعيد شهرزاد أو السندباد…`
- `doc_4f0612ce35d1` @300/@295: mine `…وقال لهم ها نحن صاعدون إلى أورشليم ‖ وابن الإنسان يسلم…`
  gold `…وقال لهم ‖ ها نحن صاعدون إلى أورشليم وابن الإنسان يسلم…`
- `doc_566d3db2b5c9` @584/@589, `doc_2baf8d2ac98d` @197/@201, `doc_4259e5d2a449` @146/@156: same shape.

**REVISION to the batch-1 tactic.** Batch 1 said "the annotators cut at the *first* available و/ف in
a run, not the second." Batch 2 shows the opposite as often as not (`ثم التحق … ‖ لعله`), so the
first-joint heuristic is retired. The replacement is not a direction but a question:
**which of these joints had an end-stop?** Speech verbs, verse breaks and topic shifts win; و/ف/ثم lose.

### LAW 6 — HOLDS, with a corrected and much wider table (batch 2).

The register sets the unit; measure it in the first 200 tokens. This law is right and it is now the
load-bearing one, because LAW 0 has been demoted from a base rate to a "find the unit system" step.

| register | gold unit ≈ | batch-1 bias | batch-2 bias |
|---|---|---|---|
| Bible / scripture by verse (§M) | 15–35 tok = one **verse** | — | 3–8× too many cuts |
| classical hadith, `أن…أخبره` chain (§D) | whole isnād+matn to first stop | 20+ too few | 6× too many |
| encyclopedic / expository (§B, §L) | **25–45 tok** | ~25, mild under | 2–3× too many |
| translated nonfiction (§N) | 20–35 tok | — | 2–3× too many |
| literary narrative / folk tale (§O) | 15–30 tok | — | 3× too many |
| literary reportage, asyndetic (§P) | 10–20 tok | — | 3× too many |
| belletristic essay (§I) | 20–40 tok | mild under | 3× too many |
| financial / news wire (§Q) | 25–40 tok | — | 3× too many |
| student essay, unedited (§R) | 20–35 tok | — | 2× too many |
| MCQ / quiz (§S) | stem = 1 seg (up to 55 tok); each option = 1 seg | — | cut inside stems |
| worksheet / textbook body (§J) | 3–10 tok (layout-driven) | 15+ too few | ok |
| graded reader / folk tale, abridged (§E) | 5–10 tok | 25+ too few | — |
| dialogue-heavy children's classic (§G) | 10–16 tok | over-cut in quotes | — |
| comic / speech balloon (§T) | one balloon = 1 seg | — | mild over |
| first-person confessional memoir (§H) | 20–35 tok | over-cut | over-cut |

**Calibrate first, segment second.** Two batches now say the same thing and it is the only law that
has never been falsified.

---

## FILED BY TEXT TYPE

### §A. Publisher front matter, colophons, addresses, credit lines, bibliographies

Source (batch 1): `doc_3f6c5f1e305a` (Hindawi colophon), `doc_c188bbc9702e` (magazine letters page),
`doc_5b0751f40176` (byline block). Batch 2 adds `doc_51f6f776f3c7`, `doc_566d3db2b5c9` (title pages).

**A1. REVISED (batch 2) — a period after a Latin *middle initial inside a personal name* is a
boundary; an *honorific abbreviation* before a name is NOT.**
- batch 1, still correct: gold `Robert L ‖ Rothstein and Daniel D ‖ Perlmutter ‖` (source `Robert L. Rothstein`)
- batch 2, my error `doc_51f6f776f3c7` @9 (FALSE CUT): I wrote `<NL> ترجمة د ‖ دانا شاكر ‖`
  gold: `<NL> ترجمة د دانا شاكر ‖` — the `د.` of `د. دانا شاكر` welds.
- confirmed `doc_566d3db2b5c9` @14 (FALSE CUT): gold `<NL> مراجعة د أحمد خريس ‖`
- same for a spelled-out title: `doc_46717e5891d4` @34 (FALSE CUT): gold `<NL> رفض السير المندوب السامي لمصر السماح…` —
  `السير` ("Sir") never splits from what it modifies.
> Reconciliation: a title/honorific is *bound to* the following name; a middle initial *interrupts*
> one. Rule of thumb: **if removing the dot leaves a well-formed name, weld.**

**A2. REVISED (batch 2) — every dot inside a URL, e-mail or abbreviation is a boundary; a DECIMAL
point is not.**
- batch 1, still correct: `hindawi hindawi ‖ org ‖`, `https www ‖ hindawi ‖ org ‖`, `الأردن ص ‖ ب 141242 ‖`
- batch 2, my error `doc_2fabc1ccf342` @90 (FALSE CUT): I wrote `…بحوالي 0 ‖ 2 ‖ وفي بورصة…`
  gold: `…بحوالي 0 2 ‖ وفي بورصة وول ستريت…` (source `0.2%`)
- same @114, @122; and `doc_3762b5d5da2f` @237 gold `…ستحتاج إلى 4 3 سنة بسرعة الضوء ‖`, @258 gold `…سيستغرق 8 6 سنة ضوئية ‖`
> **A number is one token-group. Never cut inside digits.** The `‖` still lands at the end of the
> sentence containing the number — I was cutting in the right *sentence*, at the wrong *site* (LAW 5).

**A2b. NEW (batch 2) — a Latin-script gloss or transliteration in parentheses is never a boundary.**
- `doc_085c979b884f` @3/@6 (FALSE CUT): I wrote `…العمارة البنائية 1 بالإنجليزية ‖ Constructivist Architecture بالروسية ‖ конструктивистской архитектуры…`
  gold: `…العمارة البنائية 1 بالإنجليزية Constructivist Architecture بالروسية конструктивистской архитектуры هي حركة معمارية معاصرة…` — one segment, three scripts.
- `doc_4259e5d2a449` @299/@302 (FALSE CUT): gold `…أناس يعرفون بحواة الثعابين Charmeurs de serpents يعزفون بمزمار…`
- the 13-name run in `doc_0de3fe1bcb6c` (LAW 3 above) is the same phenomenon at scale.
- OCR/markup artifacts likewise weld: `doc_46717e5891d4` @113/@114 gold `… عسكريه و اجتماعية ‖ SEG د اقتصاد يه و عسكرية ‖`

**A3. Credit lines, imprint fields and dedication adjuncts are one segment each.** (batch 1, holds)
Batch 2 adds: a place-line welds its parts — `doc_51f6f776f3c7` @20 (FALSE CUT) gold `<NL> نويبع سيناء ‖`.

**A4. REVISED (batch 2) — a colon is a boundary only in LAYOUT position (title/rubric/caption),
never in running prose.**
- batch 1, still correct: `doc_c188bbc9702e` @38/@40 gold `…بكتابي كسلان جدا ‖ حول العالم ‖` (`كسلان جدًا: حول العالم`)
- batch 2, my error `doc_0de3fe1bcb6c` @86 (FALSE CUT): I wrote `…إلى الأسرات الحاكمة ‖ فيقال فن أموي ‖ …`
  gold: `…إلى الأسرات الحاكمة فيقال فن أموي وفن عباسي وفن طولوني وفن فاطمي إلخ ‖` — `فيقال:` is a prose colon.
- `doc_0623277721a2` @266 (FALSE CUT): gold `…فاذهبوا وتعلموا ما هو إني أريد رحمة لا ذبيحة لأني لم آت لأدعو…` (`ما هو:` welds)
- `doc_1522cf2ee149` @20 (FALSE CUT): gold `…وقل له هكذا يقول الرب أطلق شعبي ليعبدوني ‖` (two prose colons, zero cuts)

### §B. Modern expository / argumentative nonfiction, translated

Source (batch 1): `doc_3f6c5f1e305a`, `doc_47764cf032d7`, `doc_44ee993e046e`, `doc_201b37ae8a22`,
`doc_226fc860e550`. Batch 2 adds `doc_27b80bb5ed9c`, `doc_3762b5d5da2f`, `doc_0c92fc467c35`.

**B1. REVISED (batch 2) — LAW 1 does NOT apply at full strength here.** The batch-1 entry said
"و + new clause = cut. That was ~30 misses here." In batch 2 the same move in the same register was
~40 false cuts. See LAW 1. Target 25–45 tokens and cut at the full stop, not at the connective.
- `doc_27b80bb5ed9c` @232 (FALSE CUT): gold `…شحنة كهربائية معاكسة لشحنة النصف الآخر فإذا كانت شحنة أحد النصفين موجبة…`
- `doc_3762b5d5da2f` @21 (FALSE CUT): gold `…وتبرز فيها أربع قمم جبلية أعلاها قمة اسمها أوروهينا يصل ارتفاعها إلى 2332 مترا ‖`
- `doc_0c92fc467c35` @226/@231 (FALSE CUT): gold `…ولم يتوقف كورساكوف عند حدود الهواية أو الموهبة لكنه واظب على الدروس الموسيقية وتعلم على أيدي كبار الموسيقيين الروس ‖`

**B2. REVISED (batch 2) — `حيث`, `في حين`, `إذ`, `غير أن` DO NOT open a segment. They weld.**
This reverses the batch-1 entry, which was built on three examples from one document.
- `doc_566d3db2b5c9` @353 (FALSE CUT): I wrote `…في كتابها دانيل ديروندا ‖ حيث تقول لقد وجدها في حالة اكتئاب عميق ‖`
  gold: `…في كتابها دانيل ديروندا حيث تقول لقد وجدها في حالة اكتئاب عميق ‖`
- `doc_566d3db2b5c9` @551 (FALSE CUT): gold `…في مقال لمجلة ذا نيو يوركر حيث يصف نفسه ممددا في سريره…`
- `doc_085c979b884f` @304 (FALSE CUT): gold `…على غرار شروحات وادعاءات أبولينير في حين كان تأثير السوبرماتية أكبر وأوسع انتشارا…`
- `doc_27b80bb5ed9c` @38 (FALSE CUT): gold `…لكن الحقيقة أن العكس هو الذي حدث إذ أن استخدام الورق في تزايد مستمر…`
- `doc_4259e5d2a449` @182 (FALSE CUT): gold `…لا ينكر فضلها أحد إذ هي غذاء الأفئدة ومهذبة النفوس…`; @233 gold `…بل هيمن على الحيوان إذ نشاهد أن الخيل…`
- `doc_566d3db2b5c9` @505 (FALSE CUT): gold `…دون أن يعلم السبب غير أن كل ما يعلمه أن شيئا ما قد…`
- also `كما`: `doc_1c49614cd6bc` @219 (FALSE CUT) gold `…عندما كان المسلمون يخرجون كما حصل في غزوة أحد وأمره أن يصلي بالناس…`
  (this also retires the parenthetical note in batch-1 J7 that "`كما` opens a segment")
> **T1 is now RESOLVED and the resolution is that I had the question backwards.** `غير أن` and
> `إلا أن` behave the same — both weld. Subordinating adversatives are inside the sentence. Only a
> *sentence-initial* `ولكن` after a real full stop cuts.

> ### B2 — RE-REVISED (batch 13). `حيث`, `إذ`, `كما` CUT when they carry a NEW FACT.
> The batch-2 blanket "they weld" cost me 6 misses in batches 11–13. The discriminator is
> **attachment, not the word**:
> - **WELD** when the connective hangs off the immediately preceding NP as a locative/comparative
>   adjunct: `في كتابها دانيل ديروندا حيث تقول`, `في مقال لمجلة ذا نيو يوركر حيث يصف نفسه`,
>   `عندما كان المسلمون يخرجون كما حصل في غزوة أحد`.
> - **CUT** when the preceding predication is already complete and the connective opens a fresh
>   fact-bearing clause with its own subject:
>   `doc_83e3dc997e52`: `وتصدر هذه الترشيحات فيلم جاذبية ‖ إذ حصل على 11 ترشيحا ‖ وجاء بعده فيلم…`
>   `doc_a4029b2c8e1b`: `إيران مع القوى العظمى ‖ حيث ورد في بيان صادر عن الحكومة السعودية…`;
>   `لم يتم التوصل إلى أي تسوية نهائية ‖ كما لمح…`
>   `doc_7b090cf754e8`: `يعتمد هذا الجيل على مواقع التواصل الاجتماعي ‖ حيث يقومون بتقليد الموضات…`
>   `doc_a74a9e93d97a`: `ويستخدمها كعناصر زخرفية ‖ كما اقترب من تصوير الشخصيات…`
> **`كما` meaning "likewise / moreover" is a sentence-opener and cuts.** `كما` meaning "just as"
> is a comparative adjunct and welds. Same for `بيد أن`: `سعى ليوناردو إلى تجاوزه ‖ بيد أنه أخفق`.

**B3. HOLDS and strengthens — `إلا أن`, fronted `وعندما`, `ولا تزال` do NOT open a segment.**
Batch 2 confirms: `doc_1d08b507b151` @48 (FALSE CUT) gold `…والسيئة بمثلها إلا أن يتجاوز الله عنها ‖`.

**B4. REVISED (batch 2) — a ولكن-clause welds by default, regardless of length.**
The batch-1 hypothesis was "under ~6 tokens = tail, over ~6 with new content = head" (T2).
Batch 2 kills the length test:
- `doc_3762b5d5da2f` @36 (FALSE CUT): gold `…القمر جسم بارد غير منير ولكنه يعكس أشعة الشمس الواقعة عليه فيظهر منيرا في سماء الليل…` (long, welds)
- `doc_0de3fe1bcb6c` @67 (FALSE CUT): gold `…مقصور على الفن الإسلامي ولكنا نقول إنه أصبح فيه ظاهرة كبيرة…` (long, welds)
- `doc_099def456da2` @81 (FALSE CUT): gold `…يتحلى بفضائل كثيرة ولكنه كان إذا فوتح بأمر الزواج جمح كالفرس المتوحشة…` (long, welds)
- `doc_566d3db2b5c9` @604 (FALSE CUT): gold `…دون أن يواجه أي صعوبة لكنه يأمل هذه اللحظة…`
- it cuts only where a full stop precedes it: `doc_085c979b884f` @489 gold `…ليس أكثر من آلة دقيقة ‖ ولكن بالرغم من…`
> **T2 RESOLVED: the frame was wrong. Length is not the variable; the preceding punctuation is.**

**B5. RETIRED (batch 2) — "a comma-set parenthetical list becomes two boundaries, not zero."**
Superseded by LAW 3 as revised: it becomes **zero** boundaries. See `doc_27b80bb5ed9c` @216,
`doc_566d3db2b5c9` @454 gold `…وكل ما حوله مصدر للقلق الصحبة والموسيقى والسفر والعمل ‖`.

### §C. Classical Arabic philosophical / theological prose

Source: `doc_1d513dc2c22a` (Ibn Tufayl). No batch-2 document in this register, so C1/C2/C4 stand
unverified but unchallenged.

**C3. REVISED (batch 2) — `إذ` welds.** The batch-1 entry ("`إذ` (causal 'since') opens a segment",
two examples from Ibn Tufayl) is contradicted by four batch-2 examples in three documents (see B2).
Provisional reconciliation: in Ibn Tufayl the `إذ`-clauses followed a completed syllogistic step that
the edition closed with a full stop. **Default is now weld; require an end-stop to cut.**

### §D. Hadith — isnād and matn

Source: `doc_e1435023282b` (Bukhārī, batch 1, F1 37.5); `doc_1d08b507b151` (Mālik, *Muwaṭṭaʾ*,
batch 2, **F1 57.1 — 6 false cuts, 0 misses**). These two documents disagree, and the disagreement
is informative.

**D1. REVISED (batch 2) — cut an isnād link only at `قال حدثنا` / `قال أخبرنا` (an explicit `قال` +
direct transmission verb). A chain subordinated with `أن … أخبره` is ONE segment.**
- batch 1, still correct: gold `حدثنا عبد الله بن محمد ‖ قال حدثنا أبو عامر ‖ قال حدثنا سليمان بن بلال المديني عن ربيعة…`
- batch 2, my error `doc_1d08b507b151` @5–@24 (FALSE CUT ×4): I wrote
  `قال مالك أخبرني زيد بن أسلم ‖ أن عطاء بن يسار أخبره ‖ أن أبا سعيد الخدري أخبره ‖ أنه سمع رسول الله صلى الله عليه وسلم يقول ‖ إذا أسلم العبد…`
  gold: `قال مالك أخبرني زيد بن أسلم أن عطاء بن يسار أخبره أن أبا سعيد الخدري أخبره أنه سمع رسول الله صلى الله عليه وسلم يقول إذا أسلم العبد فحسن إسلامه يكفر الله عنه كل سيئة كان زلفها ‖`
  **The entire isnād plus the first sentence of the matn is ONE segment.**
> The discriminator: `قال حدثنا X. قال حدثنا Y.` is a sequence of *printed sentences* (each `قال` takes
> a colon). `أخبرني X أن Y أخبره أن Z أخبره` is a single *syntactic* sentence with no internal stop.
> Same law as everywhere else — follow the ink, not the transmission structure.

**D2. REVISED (batch 2) — "in the matn, cut before ف + verb" is retired.** See LAW 2. The matn is cut
at its printed stops only: `doc_1d08b507b151` gold `…كل سيئة كان زلفها ‖ وكان بعد ذلك القصاص ‖ الحسنة بعشر أمثالها إلى سبعمائة ضعف ‖ والسيئة بمثلها إلا أن يتجاوز الله عنها ‖`
(note: `والسيئة` — a و + new subject — **does** cut here, because the source has a stop; four words
earlier `والسيئة بمثلها إلا أن…` welds its own subordinate. This is LAW 0′, not a و-rule.)

**D3. HOLDS and generalises — `قال` + a bare quoted question does NOT separate from what follows.**
Confirmed in batch 2 across scripture as well (§M). Strongest single confirmation:
`doc_1522cf2ee149` @138–@141 (FALSE CUT ×4): I wrote `…فقال ‖ غدا ‖ فقال ‖ كقولك ‖ لكي تعرف…`
gold: `…ولكنها تبقى في النهر ‖ فقال غدا فقال كقولك لكي تعرف أن ليس مثل الرب إلهنا ‖`
**A whole four-turn exchange is one segment.**

### §E. Graded readers, folk tales, abridged moral narrative

Source: `doc_54f919517225` (Tolstoy), batch 1, zero false cuts. No batch-2 document at this exact
grade level. E1/E2/E3 stand **but are now explicitly scoped**: this register's *printed sentences* are
5–10 tokens, which is why every connector coincided with a stop. Do not export the E1 reflex to
un-abridged narrative — see §O, where the same reflex cost me 33 false cuts.

### §F. Children's magazine story
Unchanged from batch 1 (F1/F2/F3), but F1 is now scoped by LAW 3 as revised.

### §G. Translated children's classic, dialogue-heavy

**G1 HOLDS and is now one of the most confirmed rules in the file** — a whole quoted utterance is ONE
segment with its speech verb, however many clauses or questions it contains. Batch 2 confirms in
five more registers (§M scripture, §D hadith, §S quiz stems, §T balloons, §I essay).

**G2. REVISED (batch 2) — an appositive inside a quote does NOT reliably cut.** The batch-1 example
(`صاحت جو سأشتري لها خفا من الساتان ‖ أفضل خف يمكنني الحصول عليه ‖`) is now outweighed by the
apposition-welds evidence in LAW 3. **T5 partially RESOLVED: apposition welds; what actually cuts
inside speech is a vocative or a fresh question mark (see §O).**

### §H. First-person literary / confessional memoir

**H1 HOLDS.** ف, و, ولكن weld to the preceding clause. Batch 2 extends this to travel-diary memoir
(`doc_51f6f776f3c7`, §P) and — importantly — batch 2 shows H1 is not an inversion at all. It is the
**general case**, and batch 1's §H only looked exceptional because LAW 1 was wrong.

**H3 HOLDS, confirmed four times in batch 2 — a bare interjection, vocative or one-word turn is its
own segment.** This is my most reliable positive cut cue outside `<NL>`:
- `doc_4114b13808e9` @111 (MISSED): I wrote `<NL> حسنا خذ المزيد منه ‖` → gold `<NL> حسنا ‖ خذ المزيد منه ‖`
- `doc_4259e5d2a449` @146 (MISSED): gold `…وتقول له الله الله ‖ بنغمة أخرى وفتح هاء الأولى…`; @163 (MISSED) gold `…قلت الله ‖ بنغمة مغايرة…`
- `doc_099def456da2` @190 (MISSED): gold `…وقال يخاطب والده مولاي ‖ إن لم أجد في القريب العاجل عروسا…`
- `doc_099def456da2` @297 (MISSED): gold `…يا ولدي ويا عصا شيخوختي ودم قلبي ‖ أي فكر غريب جال في خاطرك ‖ هل فقدت رشدك ‖`
- `doc_2baf8d2ac98d` @209 (MISSED): gold `…لا يا مولاي ‖ يا أمير المؤمنين ‖ أين ننام أنا وزوجتي والخادم والحصان والكلب ‖`

### §I. Belles-lettres psychological essay

Source: `doc_eb278f2980be` (batch 1); `doc_4259e5d2a449` (essay on music, batch 2, F1 73.1, 23 false
cuts, 2 misses).

**I1. REVISED (batch 2) — "و + new clause cuts" is retired here too.** My batch-2 errors:
- @565/@571 (FALSE CUT): gold `…ترتبط الموسيقى بأجمل العلوم ويتسنى طرق أبوابها من جهات مختلفة وفضلها لا يلبث أن يظهر من جميع الوجوه ‖`
- @546/@554 (FALSE CUT): gold `…بثلاثين قرنا قبل الميلاد ولا مشاحة في أن الموسيقى أقدم من الشعر وهي التي نفحته بقوانينه الخاضع لها ‖`
- @311/@315 (FALSE CUT): gold `…وهو خاص بالثعابين ويوقعون عليه ألحانا خاصة فتهرع إليهم الثعابين من جحورها وتقف بين أيديهم ذليلة صاغرة…`

**I4. NEW (batch 2) — an attributive parenthesis around a quoted authority welds.**
- @357/@360 (FALSE CUT): I wrote `قال لوثير المصلح الديني الشهير ‖ وكان موسيقيا فاضلا ‖ إنني أضع بعد الإلهيات مباشرة الموسيقى…`
  gold: `قال لوثير المصلح الديني الشهير وكان موسيقيا فاضلا إنني أضع بعد الإلهيات مباشرة الموسيقى وأمنحها الشرف الثاني ‖`
  — attribution + parenthetical + the whole quotation = ONE segment.

**I2, I3 unchanged.**

### §J. School textbooks, workbooks, worksheets

**J1–J5 HOLD** (rubric verb + colon, table cells, source attributions, fill-in blanks, date labels).
Batch 2 keeps me near-perfect on the true/false worksheets: `doc_00b450a96684` F1 99.3,
`doc_056d9b714661` F1 99.2, `doc_46717e5891d4` F1 98.0.

**J6. REINFORCED (batch 2) — the statement and everything that qualifies it stay together; only the
answer options split.** My two residual errors are both inside the stem:
- `doc_00b450a96684` @18 (FALSE CUT): I wrote `عبرت هبة الشارع و الإشارة الضوئية حمراء ‖ هذا السلوك ‖ غيرصحيح ‖ صحيح ‖`
  gold: `عبرت هبة الشارع و الإشارة الضوئية حمراء هذا السلوك ‖ غيرصحيح ‖ صحيح ‖`
  — the و-clause AND the asyndetic `هذا السلوك` are both part of the prompt.
- `doc_056d9b714661` @10 (FALSE CUT): gold `القمباز من التراث وترتديه النساء ‖ صح ‖ خطأ ‖`

**J7. REVISED (batch 2) — "textbook body prose still obeys LAW 1" is retired**, along with the note
that `كما` opens a segment. See B2 and `doc_1c49614cd6bc` @214/@219/@229/@246.

### §K. Sports / news compendium, captions
**K2 REVISED (batch 2): `حيث` does NOT open a segment** — see B2. K1 is subject to the LAW 1 revision.
K3, K4, K5 unchanged.

### §L. Encyclopedic / reference prose
Rules = §B as revised. Batch 2's `doc_085c979b884f` (F1 70.6) and `doc_0de3fe1bcb6c` (F1 61.0) show
this register has the **longest units in the corpus, 25–45 tokens**. The batch-1 note that I "keep
absorbing `وتشمل` / `وهي` / `والفائدة` appositive-continuations which the gold cuts" is **retired**:
in batch 2 the gold welds exactly those — `doc_085c979b884f` @521/@529 gold
`…أصبح السمة المميزة لمجموعته المجموعة المعمارية المستقلة OSA التي أسسها عام 1924 والتي أصبحت النواة الرئيسية لما يعرف باسم الحركة البنائية ‖`.

---

## NEW SECTIONS FROM BATCH 2

### §M. SCRIPTURE — Bible narrative. **THE UNIT IS THE VERSE.** (my biggest new law)

Source: `doc_0623277721a2` (Matthew 8–9, F1 77.5, 23 false cuts), `doc_1522cf2ee149` (Exodus 8,
F1 61.0, **40 false cuts — my worst document of the batch**), `doc_4f0612ce35d1` (Matthew 19–20,
F1 72.2, 24 false cuts).

**M1. A `‖` in scripture marks a verse number, not a sentence.** Verse numbers were stripped from the
text; the annotators segmented on them. **Internal full stops inside a verse are ignored.** This is
the one place where a *stronger* index than punctuation exists, and it beats LAW 0′.

- `doc_0623277721a2` @65/@70 (FALSE CUT ×2): I wrote
  `…فقال لهم امضوا ‖ فخرجوا ومضوا إلى قطيع الخنازير ‖ وإذا قطيع الخنازير كله قد اندفع…`
  gold: `…فقال لهم امضوا فخرجوا ومضوا إلى قطيع الخنازير وإذا قطيع الخنازير كله قد اندفع من على الجرف…`
  That is Matt 8:32 — **three printed sentences, one verse, one segment.**
- `doc_4f0612ce35d1` @295/@300/@307/@310/@318 — a five-error cluster where I had a boundary in every
  wrong place. gold: `…وقال لهم ‖ ها نحن صاعدون إلى أورشليم وابن الإنسان يسلم إلى رؤساء الكهنة والكتبة فيحكمون عليه بالموت ‖ ويسلمونه إلى الأمم لكي يهزأوا به ويجلدوه ويصلبوه وفي اليوم الثالث يقوم ‖`
  = verses 17 | 18 | 19 exactly. My cuts were all at connectives inside verses.

**M2. Consequences that override earlier laws inside scripture:**
- **`قال / فقال + quotation` never separates** (`doc_1522cf2ee149` @12/@17/@20 FALSE CUT: gold
  `<NL> قال الرب لموسى ادخل إلى فرعون وقل له هكذا يقول الرب أطلق شعبي ليعبدوني ‖`)
- **questions never separate** (`doc_0623277721a2` @35: gold `…يا يسوع ابن الله أجئت إلى هنا قبل الوقت لتعذبنا ‖`)
- **ف, ثم, و never separate** (`doc_1522cf2ee149` @182: gold `…ففعل الرب كقول موسى فماتت الضفادع من البيوت والدور والحقول ‖`)
- **but a trailing `لأن` CAN separate, if the verse breaks there** — LAW 4 is overridden
  (`doc_0623277721a2` @401 MISSED: gold `…ومست هدب ثوبه ‖ لأنها قالت في نفسها إن مسست ثوبه فقط شفيت ‖`)
- **a speech verb can end a verse with the quote starting the next** (`doc_4f0612ce35d1` @295 MISSED)

**M3. Operationally, with no verse numbers visible:** look for a **completed narrative proposition of
15–35 tokens**, prefer boundaries at scene/actor changes (`وكان بعيدا منهم`, `ولما جاء يسوع إلى بيت الرئيس`),
and **never** at a connective, a question, or a speech verb *inside* an ongoing event. When torn,
weld — in this register I was over-cutting 12:1 (87 false cuts vs 6 misses across three documents).

### §N. Translated modern nonfiction (clinical / academic)

Source: `doc_566d3db2b5c9` (*Anatomy of Depression*, F1 78.6, 22 false cuts, 2 misses).
Behaves like §B: 20–35 token sentences, و/ف/حيث/غير أن all weld, `وقد` cuts, `مع أن` cuts.
- @268/@273/@278 (FALSE CUT ×3): gold `…وعدم الاستمتاع بأي شيء وثمة خاصية أخرى تميز الاكتئاب وهي شدته في الصباح الباكر ولذلك يرتبط عادة بالاستيقاظ مبكرا ‖ <NL>`
  — `وثمة` (existential, new topic), `وهي` (appositive), `ولذلك` (result) all weld. I cut all three.
- @373 (FALSE CUT): gold `…ما يصفه وليام ستايرون ببراعة قائلا الاكتئاب كلمة انزلقت في اللغة مثل البزاقة…` — `قائلا` + quote welds (G1).

### §O. Un-abridged literary narrative / literary folk tale

Source: `doc_099def456da2` (the prince and the rose-coloured bride, F1 69.6, **33 false cuts**),
`doc_0c92fc467c35` (Rimsky-Korsakov, F1 83.1).

**O1. This is NOT §E.** The graded-reader reflex (`cut at every ف/ثم/و + new subject`) is catastrophic
here. Units are 15–30 tokens and long ف/و event-chains are single sentences:
- @149–@159 (FALSE CUT ×3): I wrote `…فنسي أنه يحمل سكينا في يده ‖ وأتى بحركة تدل على قلة الصبر ‖ فجرح إصبعا من أصابعه ‖ وتدفق منها الدم…`
  gold: `…فنسي أنه يحمل سكينا في يده وأتى بحركة تدل على قلة الصبر فجرح إصبعا من أصابعه وتدفق منها الدم واستقر في صحن من القشدة ‖`

**O2. What DOES cut here (my two misses tell me exactly what to look for):**
- a **speech verb introducing direct address** — gold `…وتغير تفكيره فجأة ‖ وقال يخاطب والده مولاي ‖ …`
- the **vocative that opens the speech** (@190, @297 — see H3)
- **each rhetorical question inside the speech** — gold `…أي فكر غريب جال في خاطرك ‖ هل فقدت رشدك ‖`
- `<NL>` paragraph breaks (free)
> Note the asymmetry with §P and §H: questions cut inside *dialogue* but weld inside *interior
> monologue* (`doc_51f6f776f3c7` @466/@473 FALSE CUT: gold `…هل ذعرت من وجه نادرا ما تراه أنا الرجل لا أشاهد مرآة في المنزل أم أنها تريد…`). Logged as T8.

### §P. Literary reportage / travel diary — asyndetic, present-tense

Source: `doc_3484c02411e9` (Tahrir Square, F1 77.4, 38 false cuts), `doc_51f6f776f3c7` (Sinai diary,
F1 81.2, 24 false cuts).

**P1. Short asyndetic clauses are chained inside ONE sentence.** This register *looks* like batch-1
LAW 3 (parallel repetition → cut) and is the exact opposite.
- `doc_3484c02411e9` @64/@69 (FALSE CUT ×2): I wrote `…بجوار خيمتها ‖ كانت تلهث قليلا ‖ الزحام شديد ‖ لا مكان لقدم ‖`
  gold: `…جلست فوق حجر بجوار خيمتها كانت تلهث قليلا ‖ الزحام شديد لا مكان لقدم ‖`
  **Note the gold groups them 2-and-2, not 1-1-1-1.** The units are ~8–12 tokens made of 2–3 clauses.
- `doc_51f6f776f3c7` @405–@409 (FALSE CUT ×3): gold `<NL> يدخل رجال إلى المنزل ثلاث مرات أحدهم مبارك الدليل الآخر فتحجب صالحة وجهها بسرعة البرق ‖`

**P2. Descriptive appositions and place/name glosses weld** — `doc_51f6f776f3c7` @202–@211 (FALSE CUT ×3):
gold `<NL> يتكلم أحد البدو وهو شاب نحيل اسمه لافي القليل القليل من الإنجليزية ويصطحبني بسيارته بعد الظهر…`

### §Q. Financial / news wire

Source: `doc_2fabc1ccf342` (F1 70.0, 6 false cuts).
Q1. Attribution and circumstantial tails weld: `وذلك`, `خاصة`, `وفقا لـ` (see LAW 4).
Q2. Decimals never split (see A2).
Q3. The paragraph is close to the sentence here; most true boundaries coincide with `<NL>`.

### §R. Unedited student essay / free composition

Source: `doc_142afcb751a9` (F1 76.9), `doc_3ccbb7784a8d` (F1 84.2). Only 6 errors total, all the same
shape: **`منه / منها` elaboration welds** (see LAW 4). These texts are under-punctuated by their
authors, which means **fewer** boundaries than the topic structure suggests, not more.

### §S. Multiple-choice / quiz items — **the stem is one segment no matter how long**

Source: `doc_1c49614cd6bc` (F1 93.5, 14 false cuts), `doc_46717e5891d4` (F1 98.0).

**S1. Structure: `[stem, including every embedded example, quotation and و-clause] ‖ [option] ‖ [option] ‖ …`**
- @118–@127 (FALSE CUT ×5), my worst stem: I wrote
  `…التي تدل عليها الأمثلة الآتية ‖ الفقه وأصول الفقه ‖ علوم التربية والاجتماع ‖ النفس والاقتصاد ‖ التاريخ ‖ علوم اللغة العربية هو ‖ العلوم الإنسانية والاجتماعية ‖`
  gold: `<NL> من مجالات البحث العلمي التي تدل عليها الأمثلة الآتية الفقه وأصول الفقه علوم التربية والاجتماع النفس والاقتصاد التاريخ علوم اللغة العربية هو ‖ العلوم الإنسانية والاجتماعية ‖ العلوم التطبيقية ‖`
  **The examples listed inside the stem are part of the stem. The segment ends at `هو`.**
- @199–@246 (FALSE CUT ×5): a 55-token stem, one segment, ending right before the first option
  `‖ مراعاة مشاعرهم ‖ تبشيرهم بالأجر والثواب ‖`.
- @316 (FALSE CUT): a Qur'anic quotation inside the stem welds — gold `<NL> علام يدل قوله تعالى ولا تقربا هذه الشجرة ‖ عدم القرب من الشجرة…`; @443 gold `<NL> قال تعالى وشاورهم في الأمر تعود على واحدة من القيم السياسية ‖`

**S2. An option is one segment even when it contains و.**
- @407 (FALSE CUT): gold `<NL> تكون عقوبة الجناة ‖ بفرض القانون على الجميع وتحقيق الأمن على النفس والعرض والممتلكات ‖ تعزيز الشعور بالعدالة ‖`

**S3. An option LETTER welds to its option text.**
- `doc_46717e5891d4` @114 (FALSE CUT): gold `… عسكريه و اجتماعية ‖ SEG د اقتصاد يه و عسكرية ‖`

### §T. Comic strips / speech balloons

Source: `doc_4114b13808e9` (F1 94.3).
**T-1. One balloon = one segment**, vocative + statement together: @127 (FALSE CUT) gold
`<NL> يا إلهي لا أستطيع منح الجائزة لأي من هؤلاء ‖`; @32 gold `<NL> مسابقة الطفل الجميل كوب حليب مجانا لكل الأطفال ‖`.
**T-2. Except a bare interjection, which is its own segment** — @111 (MISSED) gold `<NL> حسنا ‖ خذ المزيد منه ‖`. (H3)

### §U. Stage-play / scripted dialogue

Source: `doc_2baf8d2ac98d` (Abu Dulāma, F1 91.2, 8 false cuts, 2 misses).
Mostly one segment per `<NL>` turn, with و welded (@105 FALSE CUT: gold `<NL> يا أمير المؤمنين أنت سألتني وأنا أجبت ‖ أنا أحتاج إلى كلب ‖`).
**Irreducible residue:** the same vocative welds at @28 (`<NL> نعم يا سيدي هذا هو قصر أمير المؤمنين…`) and
splits at @209 (`لا يا مولاي ‖ يا أمير المؤمنين ‖ أين ننام…`); and @197 shows a و that *does* cut
(`سنزوجك من بنت حسين العطار ‖ وندفع لك مهر الزواج هل استرحت ‖`) while the following `هل` welds.
This is the source's own `!` / `،` choices and I should not expect to recover it. **Accept the loss;
do not build a rule on it.**

---

## TENSION LOG

**Resolved by batch 2:**
- ~~**T1.** `غير أن` cuts but `إلا أن` does not.~~ **RESOLVED:** both weld. The question was malformed —
  neither subordinator is a boundary; the preceding full stop is. (§B2)
- ~~**T2.** ولكن cuts and welds in the same document; hypothesis = clause length.~~ **RESOLVED, hypothesis
  dead:** ولكن welds by default at any length; it cuts only after an end-stop. (§B4)
- ~~**T3.** Asyndeton cuts under repetition, welds under mere sequence.~~ **RESOLVED, discriminator
  replaced:** the variable is "was each element a complete printed sentence?", not parallelism.
  Comma-lists, appositives and asyndetic clause chains **weld**. (LAW 3, §P1)
- ~~**T4.** `فقد` welds after a negated clause, cuts as a reason-for-a-state.~~ **RESOLVED as a
  non-question:** `فقد` follows the punctuation like everything else — welds in `doc_0de3fe1bcb6c` @466,
  cuts in `doc_566d3db2b5c9` @616 after a full stop.
- **T5.** Quoted speech welds but apposition inside a quote cuts. **MOSTLY RESOLVED:** apposition
  welds (LAW 3). What cuts inside speech is a **vocative** or a **question in dialogue** (§O2, H3).

**Open, carry to batch 3:**
- **T6.** `وقد` + perfect cuts ~80% (§B, §N) but welds in `doc_3484c02411e9` @289. Is the discriminator
  paragraph position, or narrative vs. descriptive? Log every instance.
- **T7.** `مع أن` cuts (`doc_566d3db2b5c9` @589) while `رغم أن` welds (`doc_3762b5d5da2f` @47). Same
  meaning, n=1 each. Exactly the shape of the old T1, so my prior is that this too will dissolve into
  "the one with a full stop in front of it cuts."
- **T8.** Questions cut inside *dialogue* (§O2, §H3) and weld inside *interior monologue* (§P,
  `doc_51f6f776f3c7` @466) and inside *scripture* (§M2). Working hypothesis: a question mark becomes a
  `‖` only when the question is a complete conversational turn.
- **T9.** How do I detect §M (verse-indexed) from the text alone, before I place a boundary? Cues seen:
  `<NL>`-delimited chapter blocks, `قال الرب لموسى`, `فقال لهم يسوع`, `الحق أقول لكم`, archaic
  Van Dyck vocabulary. This detection is worth ~90 boundary decisions per batch — get it right first.

---

## CHANGES FOR BATCH 3 — the checklist I will actually run

**The meta-lesson first: I converted batch 1's *sample statistic* into a *global operating rule* and
it cost me 354 false cuts. A base rate is never a law. From now on, an entry earns "LAW" status only
if it names a mark in the source, not a frequency.**

1. **Step zero: identify the UNIT SYSTEM, not the sentence length.** Ask in order —
   (a) Is this scripture? → verse (§M). (b) Is this a quiz/worksheet? → stem/option/cell (§S, §J).
   (c) Is this dialogue/comic? → turn/balloon (§T, §U). (d) Otherwise → the printed full stop.
2. **Default = WELD. Cut only when I can name the source mark.** If my reason for cutting is a
   connective (و ف ثم أو كما حيث إذ في حين غير أن إلا أن ولكن رغم أن منها وذلك خاصة وفقا لـ), that is
   not a reason. Batch 2: every one of those produced false cuts and none produced a miss.
3. **Never cut inside:** a comma-list, an apposition, a parenthetical gloss (Arabic or Latin script),
   a number, a name + honorific, a quotation and its speech verb, an MCQ stem, a verse.
4. **Take the free boundaries:** every `<NL>` (100% over 41 docs), every MCQ option, every table cell,
   every bare interjection/vocative turn (H3 — 5 of my 16 misses were exactly these).
5. **Positive cut cues worth acting on, in descending confidence:** `<NL>`; layout slot change;
   a bare interjection/vocative; a speech verb starting a *new* turn or verse; `وقد` + perfect;
   a fresh question that is a complete conversational turn; a topic sentence after a completed argument.
6. **In scripture: count to ~15–35 tokens of completed narrative and cut at scene/actor change only.**
   Never at ف/و/ثم, never between a speech verb and its quote, never at a question. This single change
   is worth ~87 false cuts.
7. **In hadith: cut the isnād only at explicit `قال حدثنا/أخبرنا`; weld an `أن … أخبره` chain entirely.**
8. **Before committing a boundary, check the neighbouring joints** (LAW 5) — but with the corrected
   question: *which joint had the end-stop?*, not *which joint came first?*
9. **Watch the swing.** Batch 1 = 4.8:1 under, batch 2 = 22:1 over. If batch 3 comes back over 3:1 in
   either direction, the failure is again a global prior overriding a local unit system, not a missing rule.
10. Log every instance bearing on T6–T9 so batch 3 can settle them.

## RECOVERED ENTRIES — batches 9–10 (authored by Jury 0 in its session report when ENOSPC blocked the doctrine write; transferred verbatim by the engineer, 2026-08-06)

1. **The `؛` semicolon is a hard boundary and it surfaces in the stripped text as a bare و/ف.** All 8 misses in `doc_684c09874e72` were semicolon sites I welded (`السماء ‖ فكان المشهد`, `الهائل ‖ والمشهد الأقصر`, `كئيب ‖ وكان المدخل`, `مميتة ‖ وكان قدر كبير`).
2. **B4/T2 finally resolved, and the old entry is wrong:** full-form `ولكن` + independent clause **CUTS** (`الصوت ‖ ولكن بدلا من`, `يوما ‖ ولكن في لحظة الخطر`); clitic `ولكنه/لكنه/ولكنا` continuing the same subject **WELDS**. Every prior welding example was a clitic — the reconciliation is exact.
3. **`أما … ف-` topic-fronting opens a segment** (`السحيق ‖ أما أنا فكنت فوقه`).
4. **NEW register §V — original-Arabic nahḍa essay** (`doc_5a3a244c59f1`, R 62.5): dense stops; و/ف/حتى/فقد all CUT when they advance an argument (`ولسانه ‖ فلم يعللها`, `المعاملات ‖ حتى قال المؤرخون`, `المقريزي ‖ فقد ثبت`, `الثروة ‖ ولا شك`, `معاملاته ‖ ومن المحقق`). LAW 2 must be scoped to narrative, not exposition.
5. **`ثم` CUTS in short-sentence memoir** (`محتمل ‖ ثم نجوت`, `النجاة ‖ ثم همت`) — contradicts the batch-2 blanket rule.
6. **Off-by-one fix: fronted adverbials attach FORWARD, they never end the previous segment** — `قبل الآن`, `بل أكثر من ذلك`, `وحينها`, `وهنا` all begin their segment.
7. MCQ: `لا شيء مما ذكر صحيح` is one option, not two (`doc_5ec52c5c5c0b` @265).

---

# BATCHES 11–13 — THE UNDER-CUTTING CORRECTION (graded together on restart, 2026-08-06)

| batch | mean F1 | MISSED | FALSE CUT | ratio |
|---|---|---|---|---|
| 11 | 90.88 | 8 | 2 | 4 : 1 **under** |
| 12 | 89.14 | 12 | 3 | 4 : 1 **under** |
| 13 | 83.22 | ~50 | 2 | 25 : 1 **UNDER** |

Answers for these three batches were all written under the batch-10 doctrine, so they are three
independent tests of the same posture, and all three say the same thing. Precision was 96–100% in
almost every document; **recall was the entire problem** (57.4, 61.4, 64.3, 69.2, 75.0, 77.3…).
Worst: `doc_a045e64631c7` (folk tale, 72.1), `doc_a74a9e93d97a` (Leonardo biography, 75.0),
`doc_7b090cf754e8` (student essay, 78.3), `doc_a4029b2c8e1b` (news, 81.8), `doc_83e3dc997e52` (news brief, 66.7).

**Meta-lesson, and it is the same shape as batch 2's:** I again converted a *sample-specific
exemption set* into a *global default*. Batch 2's false cuts came from comma-lists, verse-internal
punctuation, appositions and MCQ stems — all structural exemptions. I generalised them into "weld by
default" and lost 50 boundaries. **The stable object is the protected list (LAW 0‴), not a default.**
Both defaults are wrong; only the list transfers.

### §W. NEW — QUR'ANIC TEXT. **One āya = one segment, and āyas are 2–8 tokens.**

Source: `doc_a9c7d1c861a3` (Fātiḥa + classical Ḥanafī scholarship, F1 87.2),
`doc_7f28cab0c923` (Sūrat al-Qadr, F1 90.9).

- **W1. Cut at every āya end**, even at 2–3 tokens. My worst run: `doc_a9c7d1c861a3` @10–@22,
  five consecutive misses — I wrote the whole Fātiḥa as one segment.
  gold: `الحمد لله رب العالمين ‖ الرحمن الرحيم ‖ مالك يوم الدين ‖ إياك نعبد وإياك نستعين ‖ اهدنا الصراط المستقيم ‖ صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين ‖`
  Note `إياك نعبد وإياك نستعين` is ONE segment — the و is intra-āya (LAW 0‴ item 2).
- **W2. The basmala.** It is its own segment **only when a `<NL>` follows it**
  (`doc_a9c7d1c861a3`: `<NL> بسم الله الرحمن الرحيم ‖ <NL>`). Inside a sūra's running text it welds to
  āya 1: `doc_7f28cab0c923` @3 (FALSE CUT) gold `بسم الله الرحمن الرحيم إنا أنزلنه فى ليلة القدر ‖`.
- **W3.** Qur'anic quotation *inside an MCQ stem* still welds (§S1) — `doc_74177f9e1015` @308
  (FALSE CUT) gold `…فأصبحوا في ديارهم جاثمين من هم القوم الذين أهلكهم الله في هذه الآية ‖`.
- **W4.** Contrast with §M (Bible): Bible verses run 15–35 tokens, Qur'anic āyas 2–8. Both are
  "one verse = one segment"; only the length differs. Detect the Qur'an by orthography
  (`أنزلنه`, `أدريك` — alif maqṣūra/defective spelling) and by `قال تعالى` framing.

### §R — REVISED (batch 13). Student / free-composition essay: SHORT units, and ف/و/حيث all CUT.

`~~OLD (batch 2):~~` ~~"These texts are under-punctuated by their authors, which means **fewer**
boundaries than the topic structure suggests, not more."~~ **Exactly backwards.**
`doc_7b090cf754e8` ran ~11 tokens/sentence and I missed 5 of 14 boundaries; `doc_75aefb3d0bfd`
missed 2 of 8. The author's own ف/و resumptives ARE the sentence boundaries:
`الأخلاقية ‖ فهي توفر`, `الأعزاء ‖ فنتمنى`, `الاجتماعي ‖ حيث يقومون`, `السعيدة ‖ فتؤدي`,
`قناع ويتنكر ‖ وتزيد الكراهية`, `المسافات ‖ وتساعد الناس`, `بوقتك ‖ فالوقت كالسيف ‖`.
What survives from the old entry is only the narrow `منه/منها` elaboration weld.

### §O — REVISED (batch 13). Folk tale / children's narrative is a SHORT-unit register too.

`doc_a045e64631c7` (the fisherman's three daughters, F1 72.1, **~25 misses, 1 false cut**) is my
worst recall of the whole run. ~10 tokens/sentence, and و + verb cuts relentlessly:
`وأيقظت زوجها ‖ وحكت له`, `وكبرت المولودة ‖ وأصبحت طفلة ‖ ولاحظ الصياد`, `يستيقظ قبل الفجر ‖ وكانت هناء`,
`إلى البحر ‖ وتحمل معه أدوات الصيد`, `أولاد ملوك ‖ ولكني صياد فقير`, `البعيد المنال ‖ بل إن من المستحيل`,
`سعيدة حالمة ‖ أما مبروك فأخذ`.
> **The §O batch-2 entry ("long ف/و event-chains are single sentences") is now scoped to
> *un-abridged adult literary prose* only** (`doc_099def456da2`). Anything written for children —
> including folk tales that look literary — is §E-like: short printed sentences, cut at the connective.
> One weld to remember: `أتوقظيني… لتحكي لي حلما هيهات أن يتحقق ‖` — `هيهات` predicates the preceding
> object and welds (my only false cut there).

### §S — REFINED (batch 12). The stem absorbs everything the options SHARE.

`doc_87ac286b6785` @132 (FALSE CUT + MISSED, off-by-one): I wrote `يصنع الصابون من ‖ زيت الزيتون ‖ …`
gold: `يصنع الصابون من زيت ‖ الزيتون ‖ دوار الشمس ‖ السمسم ‖`
> **Options must be parallel to each other.** The shared head noun (`زيت`) belongs to the stem;
> only the contrasting tails are options. Before cutting a stem, read all the options and factor out
> their common prefix.

### §Q / news wire — REVISED (batch 12/13). This is a SHORT-unit register (~7–15 tokens).

`doc_83e3dc997e52` (BAFTA nominations brief, F1 66.7, ~7 tok/sentence) and `doc_a4029b2c8e1b`
(Iran/Geneva, F1 81.8). Every miss was a fresh fact:
`لحفلها السابع والستين ‖ وتصدر هذه الترشيحات فيلم جاذبية ‖ إذ حصل على 11 ترشيحا ‖ وجاء بعده فيلم…`;
`إنتاج أسلحة نووية ‖ هذا أمر واضح كل الوضوح ‖ ولكنه لم يكشف عن…`; `تحتاج إلى تسوية ‖ لم يتم التوصل…`.
Note the **asyndetic** cuts (`هذا أمر واضح`, `لم يتم التوصل`) — an asyndetic clause with its own
finite verb after a complete statement is a boundary. Q1 (trailing `وذلك`/`خاصة`/`وفقا لـ` weld) still holds.

### §L / Wikipedia-style article prose — REVISED (batch 12). و + verb cuts ~5 : 1.

`doc_9a890accffc8` (Borussia Dortmund, F1 87.3): 5 misses all at `و` + fresh predication, against
1 weld (`مما يحول دون تعرضها للسرقة ويفترض أن اسم المكان…`). Wikipedia Arabic is fact-listing prose:
**each و-clause is normally a new fact and a new segment.** Also confirmed: a fronted temporal
adverbial takes NO boundary after it and its cut goes before it — @332 (FALSE CUT) gold
`…بشريط أحمر وسروال أسود في عام 1913 قام بوروسيا دورتموند بتغيير ألوانه ‖`.

### §X. NEW — art-historical / scholarly biography (translated).

Source: `doc_a74a9e93d97a` (Leonardo, F1 75.0, **18 misses, 1 false cut**), ~12.5 tokens/sentence.
Reads like §L but is *denser in stops than any other expository register*. Everything cuts:
و + verb, `فقد`, `فهي`, `لكن`, `كما`, `بيد أن`, `وقد`, and asyndetic imperatives
(`عذراء الصخور ‖ انظر الصورتين رقم 27 و 28 ‖ وقد اتسعت معرفة…`).
Only structural welds survive: the two-part chapter heading, and و joining list NPs.

### §U / §T — response particles vs. vocatives (batch 12 refinement of H3 and T-1)

`doc_9af6e47014d6` @49 (MISSED): gold `<NL> نعم يا مولاي ‖ الجميع في انتظار تشريفكم ‖`;
`doc_a9c7d1c861a3` @567 (MISSED): gold `…بلا ريب ‖ نعم ‖ لقد رأى أبو حنيفة النور…`.
> **A response/assent particle (`نعم`, `لا`, `حسنا`, `بلى`) closes its own segment even when a
> vocative is attached to it.** A bare exclamatory vocative that is *part of* the following sentence
> (`يا إلهي لا أستطيع منح الجائزة ‖`) welds forward. The residual `نعم يا سيدي هذا هو قصر…`
> counter-case in `doc_2baf8d2ac98d` stays logged as irreducible.

---

## TENSION LOG — updates

- **T6 (`وقد`) RESOLVED as CUT.** `doc_a74a9e93d97a` `من الدرجة الثانية ‖ وقد هجر الرسم`;
  `doc_a9c7d1c861a3` and §L agree. Treat `وقد`/`فقد` + perfect as a boundary unless inside a protected site.
- **T7 (`مع أن` vs `رغم أن`) — dissolved as predicted.** Both are subordinators; the one after a
  completed predication cuts, and the trailing one welds. No separate rule needed.
- **T8 (questions) — refined.** A question that is a complete conversational turn cuts
  (`فكيف تصبح بنتنا ملكة ‖`); a question inside a verse or inside an MCQ stem welds.
- **T10 NEW.** `حيث`/`إذ` attachment test (B2 re-revised) is n≈10 and clean, but I have no case yet of
  a *long* حيث-clause that welds. Log every instance.

---

## CHANGES FOR BATCH 14 — the checklist I will actually run

**Supersedes the batch-3 checklist above wherever they disagree.**

1. **Step zero unchanged: identify the UNIT SYSTEM.** (a) Qur'an → āya, 2–8 tokens (§W).
   (b) Bible → verse, 15–35 (§M). (c) quiz/worksheet → stem/option, factor out the shared prefix (§S).
   (d) dialogue/comic → turn, with response particles split off (§U). (e) otherwise → prose, and
   **prose means ~10–16 tokens per sentence** (LAW 0⁗).
2. **Default = CUT at a fresh finite predication.** Walk the text clause by clause; at every و / ف /
   ثم / لكن / بل / أما / كما / حيث / إذ / فقد / وقد / asyndetic juncture, ask only:
   *does a finite verb with its own predication start here, and is the preceding predication complete?*
   If yes → cut, unless protected.
3. **Run the LAW 0‴ protected list before every cut.** That list, not a default, is what transfers.
4. **The و-test is one token wide:** و + verb → cut; و + noun (list) → weld.
5. **Take the free boundaries:** every `<NL>`; every MCQ option; every table cell; every response
   particle (`نعم`/`لا`/`حسنا`); every āya end.
6. **Fronted adverbials attach forward** — put the `‖` before `في عام X`, `قبل الآن`, `وحينها`, `بل`,
   `أما`, never after.
7. **Sanity check before writing the file:** divide token count by boundary count. If the mean
   segment exceeds ~18 tokens outside §M/§S/§D, go back and find the missing cuts.
8. **Watch the swing.** Batch 1 = 4.8:1 under, batch 2 = 22:1 over, batches 11–13 = 4–25:1 under.
   I have now over-corrected twice in the same way. If batch 14 comes back lopsided again, the fault
   is a *default* overriding the *protected list*, which is the only durable thing in this file.

---

# BATCH 14 — mean F1 93.32 (2026-08-09)

| doc | register | F1 | P | R |
|---|---|---|---|---|
| `doc_bb10be35e0fa` | song lyric (Digimon theme) | **100.0** | 100 | 100 |
| `doc_c4cb5672fc6c` | history MCQ bank | **100.0** | 100 | 100 |
| `doc_cddb10bd6b23` | Abu Dhabi driving-theory MCQ | **100.0** | 100 | 100 |
| `doc_c0adf987ac2e` | translated scholarly monograph | 89.7 | 92.9 | 86.7 |
| `doc_ca06bbd92e01` | **Alf Layla wa-Layla** | **76.9** | 96.2 | **64.1** |

18 errors: **14 MISSED, 4 FALSE CUT — 3.5 : 1 UNDER.** Same direction as batches 11–13, and thirteen
of the fourteen misses sit in ONE document. The layout registers (lyric, two MCQ banks) are now
solved: three clean 100s, 212 boundaries placed without a single error. **All remaining loss is in
running prose, and all of it is recall.**

## §Y. NEW — CLASSICAL POPULAR NARRATIVE (Alf Layla wa-Layla). ~11 TOKENS. A SHORT-UNIT REGISTER.

Source: `doc_ca06bbd92e01` (حكاية التاجر مع العفريت, الليلة الأولى), F1 76.9, **14 misses / 1 false cut**.
Gold: 39 boundaries over 427 tokens; body 419 tokens / 37 segments = **11.3 tokens per segment.**
I priced it at 17.5 and was wrong by 60%.

**Y0. THE ERROR OF RECORD — I applied §O's batch-2 rule to the wrong text.** §O batch-2 says
"long ف/و event-chains are single sentences," earned on `doc_099def456da2`, and batch 13 scoped it to
"un-abridged adult literary prose." I read Alf Layla as un-abridged adult literary prose — it *is* —
and welded chain after chain. **That scope line is wrong. The split is not adult-vs-children and not
abridged-vs-un-abridged. It is MODERN LITERARY PROSE vs CLASSICAL / ORAL-DERIVED NARRATIVE.**
`doc_099def456da2` is a nahḍa-era literary tale carrying a modern editor's long sentences.
Alf Layla is ف-chained oral narrative and its editions punctuate at almost every beat.
> **§O batch-2 is hereby narrowed to modern literary prose. Classical popular narrative is §Y and it
> is SHORT.** The tell is the connective density itself: if nearly every clause opens with ف or و and
> the vocabulary runs `فدنا / فاستوثق / فغشي / وإذا بـ / ثم أنه / فلم يستقر به الجلوس`, you are in §Y.

**Y1. At a ف/و + finite-verb juncture the base rate is ~72% CUT (36 cuts against 14 welds).**
All 14 of my misses were ف/و junctures I welded:
`فجلس تحت شجرة ‖ وحط يده` · `إلى أهلها ‖ وأعلم زوجته` · `ونساءه وأولاده ‖ وأوصى وقعد` ·
`رغما عن أنفه ‖ وأقيم عليه العياط` · `ذلك البستان ‖ وكان ذلك اليوم` · `وحياه ‖ وقال له ما سبب` ·
`صاحب الغزالة ‖ وقال والله يا أخي` · `يتحدث معه ‖ فغشي على ذلك التاجر` · `الكلاب السود ‖ فسألهما` ·
`بغلة زرزورية ‖ فسلم عليهم` · `تلك البرية ‖ فانكشفت الغبرة` · `من بينهم ‖ وقال له قم` ·
`التاجر وبكى ‖ وأعلن الثلاثة شيوخ` · `ثم أعود إليك ‖ ولك علي عهد وميثاق`.

**Y2. The weld condition, stated positively: ONE segment = ONE verb, or a TIGHT PAIR of verbs naming
a single act, and it runs out at about 6 tokens.** Every weld in the gold is either the first
continuation after a segment-initial verb or an idiomatic doublet:
`فسلم على هذا التاجر وحياه ‖` (5) · `فسلم عليهم وسألهم ‖` (9) · `فأتاهم وجذب ذلك التاجر من بينهم ‖` (6) ·
`فانتحب ذلك التاجر وبكى ‖` (4) · `وأوصى وقعد عندهم إلى تمام السنة ‖` (6) · `فاشتد عليه الحر فجلس تحت شجرة ‖` (6).
> **Operationally: allow 2, at most 3, coordinated verbs per segment. When a THIRD ف/و verb arrives,
> cut.** The longest weld run in the whole document is
> `فاستوثق منه الجني وأطلقه فرجع إلى بلده وقضى جميع تعلقاته وأوصل الحقوق إلى أهلها ‖` — 5 verbs,
> 14 tokens — and it is the exception that proves the ceiling.

**Y3. My ONE false cut names the counter-shape: a fronted `فلما` clause welds BACKWARD.**
@45 I wrote `…وأكل كسرة كانت معه وتمرة ‖ فلما فرغ من أكل التمرة رمى النواة…`; gold reads
`…وحط يده في خرجه وأكل كسرة كانت معه وتمرة فلما فرغ من أكل التمرة رمى النواة وإذا هو بعفريت طويل القامة وبيده سيف ‖`
— 23 tokens, the longest segment in the body.
**LAW 0‴ item 9 (fronted adverbial attaches forward) does NOT cover `فلما` + apodosis inside a
narrative chain: the whole لما-structure joins the clause BEFORE it.** Note the contrast in the same
document — I cut before `فبينما هو جالس` @203 and that was right. **`بينما` opens a scene, `لما`
closes one.** Same shape, opposite direction, one letter apart.

**Y4. `وقال` after a completed act opens a segment; `وقال` as the second beat of an approach welds.**
CUT: `فسلم على هذا التاجر وحياه ‖ وقال له ما سبب جلوسك…` · `فتعجب الشيخ صاحب الغزالة ‖ وقال والله يا أخي…` ·
`فأتاهم وجذب ذلك التاجر من بينهم ‖ وقال له قم أقتلك…`
WELD: `فدنا من ذلك التاجر وقال له قم حتى أقتلك مثل ما قتلت ولدي ‖` — the approach and the speech are
one motion.
G1 is untouched: the speech verb never separates from its own quotation, so the boundary always lands
on the last token BEFORE the verb, never between verb and quote.

**Y5. Free confirmations in §Y — dialogue structure is 100% reliable here.** Every end-of-turn cut
was right (@72 `ما قتلت ولدي`, @78 the question `كيف قتلت ولدك`, @95 `ومات من ساعته`,
@139 `والله على ما أقول وكيل`, @238 `وهو مأوى الجن`, @416 `وحشاشة كبدي`), and so was every
`ثم أنه` / `فبينما` / `فإذا بشيخ ثان` scene-shift. **The dialogue skeleton I can read; the ف/و chain
is the only thing I misprice.**

## §Z. Translated scholarly monograph (book-length, European original) — the LONG unit survives.

Source: `doc_c0adf987ac2e` (Delpiano, *Slavery in the Modern Age*, tr. Ḥabashī), F1 89.7.
Gold: 30 boundaries over 609 tokens = **20.3 tokens/segment**; my estimate of 21.75 was close, and
the four long segments I defended (a fronted concessive + apodosis, a ليس…وإنما correlative,
a 37-token relative chain) were all correct.
**LAW 0⁗'s ~10–16 does NOT hold here.** LAW 6's 20–35 band for §N/§L stands for book-length
scholarly translation. Read batch 13's "the table is inflated for almost every register" as:
inflated for news, biography, student essays, folk tales — **accurate for translated monographs.**

**Z1. Openers that cut, all confirmed in this document:** `ذلك أن` · `بل` · `إن ما نسميه` ·
`ومن هنا فإننا` · `لقد` · `أما … فقد` · full-form `ولكن` · `إلا أن` · every fronted `من الناحية X`
and `وفي هذا السياق` · an asyndetic finite clause after a completed statement (`لم تفعل الكنيسة في الواقع…`).
**`إلا أن` cutting is a real result.** It overrides the batch-2 B3 blanket "إلا أن welds" and files
إلا أن / غير أن / بيد أن together under the batch-13 finding: adversative + completed preceding
predication + fresh subject = CUT.

**Z2. `إذ` — B2 re-revised was OVER-extended. Settled version:**
`doc_c0adf987ac2e` @229 (FALSE CUT): I wrote `…غزو الأوروبيين للعالم الجديد ‖ إذ لم يحدث أن تم إلغاء العبودية قانونيا ‖`;
gold welds it.
> **`إذ` = "since / for", giving the REASON for the claim just made → WELDS.**
> **`إذ` = "in that / whereby", reporting a NEW FACT with its own subject → CUTS**
> (`فيلم جاذبية ‖ إذ حصل على 11 ترشيحا`).
> Test: substitute "because" — if it reads, weld. Substitute "specifically, X happened" — if that
> reads better, cut.

**Z3. `كما` is REGISTER-CONDITIONED, not lexically decidable. (T11)**
@422 (FALSE CUT): gold `…للتيار الرواقي الذي تطور في القرن الرابع كما أشاروا بصفة خاصة إلى الديانة المسيحية…`
welds — and this is the "moreover" كما with an unchanged subject, exactly the shape that CUT in
`doc_a74a9e93d97a` (`كعناصر زخرفية ‖ كما اقترب`) and `doc_a4029b2c8e1b` (`تسوية نهائية ‖ كما لمح`).
> The cutting cases are **short-unit registers** (art-historical biography ~12.5 tok, news ~7–15).
> The welding case is a **long-unit register** (~20 tok). **Decide `كما` by the document's measured
> unit length, not by what the word means.**

## TENSION LOG — batch 14

- **T11 NEW.** `كما` cuts in short-unit registers and welds in long-unit ones (n=3, clean so far).
  If this holds, the same conditioning probably governs `حيث` and `إذ` too, and B2 stops being a
  lexical rule at all.
- **T12 NEW.** In §Y the ف/و juncture is ~72% cut and I cannot yet name the 28%. Working hypothesis
  from Y2: it is a *counter*, not a meaning — the first continuation welds, the second or third cuts.
  Log the running verb-count at every §Y juncture in batch 15 and see whether the counter holds.

## CHECKLIST DELTA FOR BATCH 15

Applied in this order, on top of the batch-14 checklist:

a. **Step zero grows a branch.** After "scripture / quiz / dialogue?", ask **"is this prose CLASSICAL
   ف-chained narrative (§Y, ~11 tok) or MODERN / TRANSLATED expository (§N/§Z, ~20–27 tok)?"** The two
   differ by a factor of two and I have now been burned on both sides of that line.
b. **In §Y: cut at the 2nd or 3rd coordinated verb, every time.** If a segment has reached ~12 tokens
   and another ف/و + finite verb arrives, cut — no further justification needed.
c. **و + a NEW TOPIC NP with its own predicate CUTS** (`ولك علي عهد`, `والنتيجة الأخرى هي`). The
   و-welds are list items only. That was 2 of my 14 misses and it is a one-token test.
d. **حتى أن cuts · ومن ثم cuts · a leading section numeral cuts.** Three cheap mechanical boundaries.
e. **إذ meaning "because" welds; إذ meaning "in that" cuts. كما follows the register's unit length.**
f. **Layout registers are solved — do not overthink them.** Lyric = line. MCQ = stem + options, with
   the shared-prefix test (printed once → stem; printed once per option → option). Three 100s.
g. **The swing check.** batch 1 = 4.8:1 under · batch 2 = 22:1 over · 11–13 = 4–25:1 under ·
   14 = 3.5:1 under. I have stopped oscillating and settled into a *steady mild under-cut of running
   prose*. That is a bias, not a hole in the protected list. **At a coin-flip site in prose, CUT.**

---

# BATCH 15 — mean F1 96.19 (2026-08-09)

| doc | register | F1 | P | R |
|---|---|---|---|---|
| `doc_d715df9a8c78` | geography MCQ bank | **100.0** | 100 | 100 |
| `doc_daa6620988df` | primary-school verse + rubric | **100.0** | 100 | 100 |
| `doc_d38d53501296` | biology MCQ bank | 99.0 | 99.0 | 99.0 |
| `doc_d2cedca907a0` | primary maths MCQ | 97.8 | 96.7 | 98.9 |
| `doc_dc967c8520e9` | **Arabic Wikipedia article** | **84.2** | **72.7** | 100 |

18 errors: **2 MISSED, 16 FALSE CUT — 8 : 1 OVER.** The direction has flipped from batch 14 and
twelve of the sixteen false cuts are in one document. **I acted on the batch-14 instruction "at a
coin-flip site in prose, CUT" and it was worth −16 precision points in a single article.** That
instruction is hereby scoped: it applies to §Y classical narrative and to registers I have
*measured*, never as a prior on an unfamiliar expository document.

## §L — RE-REVISED (batch 15). WIKIPEDIA IS NOT A UNIT LENGTH. MEASURE THE ARTICLE, NOT THE SITE.

Source: `doc_dc967c8520e9` (زجاج الخليل / Hebron glass), F1 84.2, **12 false cuts, 0 misses**.
Gold: **32 boundaries over 669 tokens = 20.9 tokens/segment.** I predicted 15.2 and over-cut by 37%.
By block: lead 19.4 · history-1 **30.0** · history-2 21.6 · history-3 24.6.

`~~batch 12 (doc_9a890accffc8, Borussia Dortmund):~~` ~~"Wikipedia Arabic is fact-listing prose:
each و-clause is normally a new fact and a new segment."~~
> **NARROWED. That was one author's punctuation habit, not the encyclopedia's.** Borussia Dortmund
> ran short; Hebron glass runs long and shaggy. "Arabic Wikipedia" is not a register with a unit —
> it is a container. The unit belongs to the individual contributor. **Do not carry a density prior
> into an unfamiliar article. Place only strongly-cued boundaries on the first pass, then measure.**

### L-CUT — the cues that actually fired here (all 32 gold boundaries fit these)
- **و / ف + finite verb WITH A NEW EXPLICIT SUBJECT**: `‖ ولا تزال … منطقة` · `‖ ويقدر الإنتاج السنوي` ·
  `‖ وتتوزع أسواق` · `‖ وتم الحفاظ` · `‖ وساعد أيضا العلاقات` · `‖ ووصف روبينسون` · `‖ فقد ذكر المؤرخ ماريتي`
- **an asyndetic fresh finite clause**: `‖ تشكل صناعة الزجاج` · `‖ تعتمد هذه الصناعة` · `‖ يستعمل زجاج الخليل` ·
  `‖ كانت تتمثل` · `‖ تعتمد عديد من العائلات` · `‖ تتنوع المنتوجات`
- **a NEW ATTRIBUTION**: `‖ قال المؤرخ بوجي بونسي` · `‖ فقد ذكر المؤرخ ماريتي` · `‖ ووصف روبينسون` ·
  `‖ بينما يصف بورير`
- **a fronted topic phrase after a completed statement**: `‖ ومن أهم متطلبات العمل` · `‖ في الوقت الحاضر` ·
  `‖ حسب الحفريات` · `‖ مثال آخر على` · `‖ من أهم الصادرات` · `‖ أما بالنسبة للتلوين`
- **openers**: `‖ لقد عانت` · `‖ إلا أن البعض الآخر` · `‖ ولكن يعتقد` · `‖ ويعتقد أن`

### L-WELD — my twelve false cuts, and the rule behind each
1. **`لذا` / `ولذلك` WELD. They are trailing result adjuncts, not openers.** @169 (FALSE CUT):
   gold `…لاكتساب مهارات التشكيل لذا لا يوجد الإقبال الكافي لتعلم تلك الحرفة…`.
   **This was already in the file** — §N recorded `ولذلك يرتبط عادة بالاستيقاظ مبكرا` welding in
   `doc_566d3db2b5c9` — and I cut anyway because "لذا" *looks* like a discourse opener. It is not.
2. **`حيث` WELDS in a long-unit article.** @197 (FALSE CUT): gold `…في فلسطين حيث يتوجه البعض…`.
   B2's attachment test was available (حيث hanging off في فلسطين) and I overrode it with Z3's
   register test — after mis-measuring the register. **Z3 stands, but it can only be applied to a
   MEASURED unit length, never to a guessed one.** When in doubt about the register, حيث welds.
3. **`و` + finite verb with an UNCHANGED subject WELDS**, even with fresh adverbial material.
   @345 (FALSE CUT): gold `…التي قامت على أراضيها وتطورت بشكل واضح بعد دخول الإسلام حيث ابتكرت…`.
   The batch-12 §L rule is really a NEW-SUBJECT rule. Contrast the gold cuts above: every one of
   them introduces a named subject (منطقة، الإنتاج، أسواق، العلاقات، روبينسون). Test in one glance:
   **is there a new noun phrase doing the verb? If not, weld.**
4. **Bare `لكن` WELDS; `ولكن` CUTS.** @380 (FALSE CUT): gold `…التي حملت اسم هذه الصناعة لكن ذلك لا يعني…`;
   @466 (CORRECT CUT): gold `…اكتسبوها من العرب ‖ ولكن يعتقد أن…`. Recovered entry 2 said full-form
   ولكن cuts and the clitic ولكنه welds; **the و is doing the work, not the form of لكن.**
5. **`أما … ف-` WELDS here.** @400, @410, @428 (FALSE CUT ×3): gold reads
   `اشتهرت بلاد الشرق وخاصة سوريا بصناعة الزجاج أما فلسطين بحدودها الحالية فقد كانت جزء من سوريا الجنوبية فمنذ القدم عمل الفينيقيون بصناعة وتجارة الزجاج ‖`
   — 24 tokens, ONE segment, containing both أما…فقد and فمنذ. And
   `قال المؤرخ بوجي بونسي … وبالقدس أما الرحالة فلكس فابري حرفة الزجاج … أبناء المدينة ‖` — 26 tokens, one segment.
   `~~recovered entry 3:~~` ~~"أما … ف- topic-fronting opens a segment."~~ **SCOPED: it opens a segment
   only where the author is punctuating densely** (it was right in `doc_c0adf987ac2e` @563, inside a
   من الناحية X enumeration in a monograph). In a loosely-punctuated article أما is just a topic shift
   inside the sentence. Same for `فمنذ`.

## G1 — GENERALISED (batch 15). AN ATTRIBUTION SWALLOWS ITS ENTIRE REPORT, 30–36 TOKENS AND MORE.

Five of my twelve false cuts (@566, @575, @596, @606, @636) were inside a historian's report, and
I made them deliberately, reasoning that "encyclopedic paraphrase is not a quoted utterance."
**That carve-out is dead.** Gold segments the three travellers' reports whole:
- `فقد ذكر المؤرخ ماريتي عام 1767 يتم تصنيع الأساور والأطواق وبعض أدوات الزينة للنساء ويصدر قسم منها إلى مصر وسوريا عن طريق يافا ويتم استخدام تربة تصنيع الزجاج والتي يجلبها البدو من مناطق مجاورة للخليل ‖` — **35 tokens**
- `ووصف روبينسون عام 1834 يوجد مصنع في ممر ضيق وهناك كميات كبيرة توضع في أقفاص وتحمل على الجمال للشحن وأشار أن طريقة التصنيع في الخليل نفسها المعروفة في العالم ‖` — **29 tokens**
- `بينما يصف بورير عام 1843 في رحلته … وذكر من بين ما ينتجون الأقراط والأساور الملونة والقناديل التي كانت تصدر إلى مصر بكميات كبيرة ‖` — **36 tokens**
> **The boundary goes IN FRONT of the attribution verb and nowhere after it, until the next
> attribution.** Note that `ويصدر قسم منها` has a brand-new subject and still welds, and that a
> *second* verb of saying belonging to the same source (`وأشار`, `وذكر`) welds too — only a NEW
> SOURCE (روبينسون → بورير) breaks the report. This makes reported content the single most reliable
> protected site in the whole corpus after `<NL>`, and it now covers dialogue (G1), scripture (§M2),
> hadith matn (§D), MCQ stems (§S1) and encyclopedic attribution alike.

## §S — TWO MECHANICAL REFINEMENTS THAT COST ME 4 ERRORS ACROSS TWO CLEAN-LOOKING QUIZZES

**S4. NEW — a candidate shared head that REAPPEARS INSIDE ANY OPTION is not a shared head.**
`doc_d38d53501296` @44/@45 (MISSED + FALSE CUT): I wrote `نبات الفول توجد به حركة ‖ اللمس ‖ نوم ويقظة ‖ حركة إنتحاء ‖`;
gold `نبات الفول توجد به ‖ حركة اللمس ‖ نوم ويقظة ‖ حركة إنتحاء ‖ أخر إجابتين ‖`.
I *saw* that حركة recurs at @49 and overrode the evidence with the doc_87ac286b6785 precedent.
> **The test is now binary and I will run it before every stem-end call: scan the option region for
> the candidate head word. If it occurs again anywhere, it is NOT factored — it opens option 1.**
> Verified against everything that graded clean: `الاستعمار` (doc_c4cb5672fc6c) once → factored;
> `رياح` and `التخطيط` ×4 items (doc_d715df9a8c78) once each → factored; `الضغط`، `تيار`، `الوقود`،
> `وزن`، `السير بسرعة أقل من`، `7 شباط عام` all recur → not factored.

**S5. NEW — options are UNIFORM IN SHAPE. Prefer the split that makes them so; the residue is stem.**
`doc_d2cedca907a0` @28/@31 (FALSE CUT + MISSED): I wrote `…تسمى ‖ الدائرة ‖ وتر ‖ مركز دائرة ‖ قطر ‖`;
gold `…تسمى الدائرة ‖ وتر ‖ مركز ‖ دائرة ‖ قطر ‖` — four ONE-TOKEN options, with الدائرة pushed into
the stem. Both readings give four options; gold picked the one where all four options have the same
shape. **When two splits both yield the modal option count, take the one with uniform option length.**

**S6. The maths-expression case, decided.** `doc_d2cedca907a0` @183/@195 (FALSE CUT ×2): gold
`ما ناتج العملية الحسابية 80 20 10 ‖ 100 ‖ 50 ‖ 30 ‖ 40 ‖` and `… 20 60 30 ‖ 50 ‖ 80 ‖ 90 ‖ 70 ‖`.
I argued this at length and chose the two-operand / five-option reading; the **three-operand /
four-option** reading was right.
> **Rule: fix the option count from the document's mode FIRST, then let the stem absorb whatever
> numbers are left over.** Do not reason from the arithmetic — both readings were arithmetically
> valid (80−20−10=50 and 80+20=100 both land on a printed option). The option count is structural
> evidence; the arithmetic is not.

## TENSION LOG — batch 15

- **T11 (كما / حيث by register) — CONFIRMED but made conditional.** حيث welded in a 21-token article
  exactly as كما welded in a 20-token monograph. The rule is sound; the *input* to it (measured unit
  length) is the thing I get wrong. **Never apply T11 to an unmeasured document.**
- **T13 NEW.** أما…ف cuts in a monograph (doc_c0adf987ac2e) and welds in a Wikipedia article
  (doc_dc967c8520e9), both long-unit. So unit length alone does not predict it. Working hypothesis:
  أما cuts when it heads a member of an explicit ENUMERATION the author has already begun
  (من الناحية العسكرية / السياسية / الاقتصادية) and welds when it is a one-off topic shift.
- **T12 (the §Y ف/و counter) — untested this batch**, no classical narrative appeared.

## CHECKLIST DELTA FOR BATCH 16

1. **SCOPE the batch-14 rule "at a coin-flip in prose, CUT".** It holds for §Y and for a register
   whose density I have measured. On an unfamiliar expository document the default is the opposite:
   **place only strongly-cued boundaries, then check the density.** Batch 14 = 3.5:1 under,
   batch 15 = 8:1 over, and both came from believing a density prior instead of a cue.
2. **The one-glance و-test is now a SUBJECT test, not a verb test.** و + verb + a new noun phrase
   doing the verb → cut. و + verb continuing the same subject → weld.
3. **Never cut inside a report.** From the attribution verb to the next attribution verb is ONE
   segment, at any length. Second verbs of saying by the same source (وأشار، وذكر) are inside it.
4. **These weld, and I have now paid for each: لذا · ولذلك · حيث (unmeasured) · bare لكن · أما…ف
   (outside an enumeration) · فمنذ.** These cut: ولكن · ويعتقد · لقد · إن · إلا أن · a new attribution ·
   a fronted topic phrase · an asyndetic finite clause.
5. **MCQ stem-end, in order: (a) count the options and take the document's mode; (b) if a candidate
   head word recurs inside any option, do not factor it (S4); (c) if two splits tie, take the one
   with uniform option shape (S5).** Never reason from content — arithmetic, semantics and
   plausibility all failed me this batch while the structural tests would have succeeded.

---

# BATCH 16 — mean F1 91.64 (2026-08-09)

| doc | register | F1 | P | R |
|---|---|---|---|---|
| `doc_e438f8ad8abe` | primary true/false + MCQ worksheet | **100.0** | 100 | 100 |
| `doc_eb9893e8f98b` | Jordanian social-studies MCQ | **100.0** | 100 | 100 |
| `doc_f2767b56c842` | **Luke 12:54–13:35, Van Dyck** | **100.0** | 100 | 100 |
| `doc_f41918b2700e` | children's book (المكتبة الخضراء) | 83.2 | 88.7 | **78.3** |
| `doc_dfc470ebeb16` | single Bukhārī ḥadīth | **75.0** | 75 | 75 |

21 errors: **14 MISSED, 7 FALSE CUT — 2 : 1 under.** Three perfect documents, including my first
clean scripture document ever: **41 verses of Luke identified and placed without a single error.**

## §M — SOLVED. The verse method works and I should trust it completely.

`doc_f2767b56c842` is the first §M document to grade 100. What made it work was doing the *reference
work* first — reading the passage as Luke 12:54–13:35, enumerating the 41 verses, and placing one
boundary per verse — instead of looking for sentence cues. Every §M2 trap was present and refusing all
of them was right: speech verbs (فأجاب يسوع وقال لهم), questions (أقليل هم الذين يخلصون), vocatives
(يا مرائي، يا أورشليم), ف/ثم/و chains (ووضع عليها يديه ففي الحال استقامت ومجدت الله), and
verse-internal full stops (13:25 runs 28 tokens across three printed sentences).
> **Method of record for §M: identify book and chapter, count out the verses, boundary on the last
> token of each. Do not segment scripture by cue. The unit is 14–35 tokens and my one uncertainty —
> @404, where Luke 13:23 ends on the bare speech verb فقال لهم — was right too.**

## §E / §O — RE-CONFIRMED THE HARD WAY. In a CHILDREN'S BOOK, و + VERB CUTS EVERY TIME.

Source: `doc_f41918b2700e` (الحصان الطيار في بلاد الأسرار, Ahmed Naguib), F1 83.2, **13 misses**.
Gold: **60 boundaries over 634 tokens = 10.6 tokens/segment.** I priced it at 12.3 and still under-cut.

**E4. THE SAME-SUBJECT VERB CHAIN IS NOT ONE SENTENCE. IT IS THREE.**
@462 and @466 (MISSED ×2), the single most instructive site in the batch:
gold `فخرج الضابط من الباب ‖ ونزل من أعلى الجبل ‖ وسار في طريقه إلى المدينة ‖`
— three segments of 4, 4 and 5 tokens, same subject, no new argument anywhere.
I welded them under LAW 0‴ item 2 (gapped coordination) and under §Y2's "allow 2–3 coordinated verbs
per segment."
> **NEITHER RULE TRANSFERS HERE.** LAW 0‴ item 2 was earned on adult prose; §Y2's verb-counter was
> earned on Alf Layla. **A children's book gives one clause one sentence.** Same finding, same
> direction, at @517 (`ثم نادى النعمان قائد جيشه ‖ وطلب منه`), @561
> (`حتى ابتعدوا عن بلادهم ‖ وغابت بيوتهم عن العيون`), @426 (`وسكت الساحر ‖ ففتح الضابط فمه`) — the
> last of these a TWO-TOKEN segment.

**E5. و + a new topic NP with its own predicate cuts here too** (the batch-14 rule, confirmed twice):
@213 gold `فرأى الجبل عاليا عاليا ‖ وطريق الصعود إليه صعبا ‖`; @270 gold
`عليها نقوش غريبة ‖ وحولها كراسي أشكالها عجيبة ‖`. **And @357: `وإليك الجواب ‖` is its own segment.**

**E6. `وإنما` CUTS.** @290 (MISSED): gold `فنظر الضابط ولكنه لم ير أحدا ‖ وإنما شعر بالسجادة الصغيرة…`.
I welded it reasoning that `لم … وإنما` is a single correlative. It is not — the negation closes and
وإنما opens. File it with بل and أما as a coordinate corrective that opens a segment.

**E7. `حتى` + perfect verb can OPEN a segment in narrative.** @227 (MISSED) + @232 (FALSE CUT), an
off-by-one of exactly the LAW 5 kind: I wrote `…ثم سار في طريق طويل ملتو حتى وصل إلى بيت الساحر ‖ فوقف أمامه…`;
gold `…ثم سار في طريق طويل ملتو ‖ حتى وصل إلى بيت الساحر فوقف أمامه ورفع يده ليدق الباب ‖`.
**The boundary is BEFORE حتى and the ف-clause after it welds.** This corroborates the batch-14 removal
of حتى from the protected list and §V's `المعاملات ‖ حتى قال المؤرخون`.

**E8. ولكن / ولكنه — THE FORM IS NOT THE DISCRIMINATOR, AND I SHOULD STOP TRYING.**
`~~recovered entry 2 + batch 15:~~` ~~"full-form ولكن CUTS; clitic ولكنه/ولكني WELDS."~~
This document has nine of them and they split BOTH ways in BOTH forms:
- clitic CUTS: `على البال ‖ ولكنه لم يكن سعيدا` (@60) · `عندي كل شيء ‖ ولكني أريد` (@90)
- clitic WELDS: `لم ير أحدا ولكنه دخل` · `فنظر الضابط ولكنه لم ير أحدا` · `فمه ليتكلم ولكنه أحس السجادة` (@430)
- full form CUTS: `ليدق الباب ‖ ولكن قبل أن يفعل هذا` · `في الفضاء ‖ ولكن أحدا لا يستطيع`
- full form WELDS: `فتح الضابط فمه ليتكلم ولكن قبل أن ينطق بحرف واحد` (@341)
> The only regularity I can see is LENGTH OF THE PRECEDING CLAUSE: the four welds all follow clauses
> of 2–6 tokens, the four cuts follow clauses of 6–11. **Treat ولكن as a cut when the running segment
> has already reached ~7 tokens and as a weld below that**, and accept the residue. The rule I had
> was clean, memorable and wrong, which is exactly the shape of error this file keeps recording.

**E9. My false cut @605 names the counter-shape.** gold
`والصحراء لا تريد أن تنتهي والنهر الذي بعدها لا يريد أن يظهر ‖` — two parallel و + topic-NP clauses in
ONE segment, though the cut BEFORE the pair (@600) was right. Same at @402:
`…هلك لأن طريقها صعب وبيننا وبينها صحراء واسعة ونهر كبير وبحر وثلاثة جبال عالية ‖`.
> **A و-clause that ELABORATES the same picture welds; a و-clause that ADVANCES the action cuts.**
> Rough, but it separates `وبيننا وبينها صحراء` (elaboration of "the road is hard") from
> `ونزل من أعلى الجبل` (next event). When both readings are open, the presence of a VERB OF MOTION OR
> EVENT favours the cut; a stative/existential favours the weld.

## §D — TWO CORRECTIONS FROM A FOUR-BOUNDARY DOCUMENT

Source: `doc_dfc470ebeb16` (Bukhārī, the dream of the shirts), F1 75.0 — 1 miss, 1 false cut out of
four gold boundaries. Gold: `حدثنا محمد بن عبيد الله ‖ قال حدثنا إبراهيم بن سعد … ومنها ما دون ذلك ‖ وعرض على عمر بن الخطاب وعليه قميص يجره ‖ قالوا فما أولت ذلك يا رسول الله قال الدين ‖`

**D4. THE MATN IS CUT AT ITS SCENE CHANGES — I over-extended G1 into it.** @49 (MISSED): gold cuts
before `وعرض على عمر بن الخطاب`. I argued explicitly that the whole dream was one report and that
batch 15 had taught me a new subject inside a report welds. **Wrong: batch 15's rule is about an
ENCYCLOPEDIC ATTRIBUTION quoting a source, not about a prophetic matn.** D2 already said "the matn is
cut at its printed stops," and the two panels of the dream (the people in shirts / ʿUmar with his
trailing shirt) are two printed sentences. **Inside a matn, و + verb + a new participant CUTS.**

**D3 — CONFIRMED, AND I VIOLATED MY OWN ENTRY.** @64 (FALSE CUT): gold
`قالوا فما أولت ذلك يا رسول الله قال الدين ‖` — the companions' question and the Prophet's
one-word answer are **ONE segment**. D3 has said since batch 2 that "قال + a bare quoted question does
NOT separate from what follows," with `doc_1522cf2ee149`'s four-turn exchange as the example, and I
cut anyway because T8 says a question that is a conversational turn cuts.
> **Reconciliation, and this settles T8: a question cuts when it stands as a turn IN AN EXTENDED
> DIALOGUE (§O, §U, §T). A short closing Q&A tag on a ḥadīth or a scripture verse — question plus
> its immediate answer — is ONE unit.** The test is whether the exchange occupies more than one
> printed sentence, not whether it changes speaker.

## TENSION LOG — batch 16

- **T8 (questions) — RESOLVED**, see D3 above. Turn-in-dialogue cuts; question-plus-answer tag welds.
- **T12 (the §Y verb counter) — SCOPED, NOT GENERAL.** It held for Alf Layla (11 tok) and fails
  outright for a children's book (10.6 tok), where every coordinated verb takes its own segment.
  Two registers of almost identical density with opposite coordination behaviour: **density does not
  determine coordination.** The classical ف-chain really is a syntactic unit; the children's و-chain
  is not.
- **T14 NEW.** ولكن/ولكنه is form-independent (E8). Provisional length rule logged; needs n>9.

## CHECKLIST DELTA FOR BATCH 17

a. **Scripture: do the reference work.** Name the book and chapter, enumerate the verses, one boundary
   per verse end. That produced my first §M 100.
b. **Children's book (§E/§O): every finite verb is a sentence.** فخرج ‖ ونزل ‖ وسار. Do not import
   §Y's verb-counter or LAW 0‴ item 2 into it. Target ~10 tokens and expect two-token segments.
c. **In a ḥadīth matn, cut at scene changes** (و + verb + new participant); do NOT extend the batch-15
   attribution rule into prophetic speech. But keep a closing question-and-answer tag WHOLE (D3).
d. **وإنما cuts. حتى + perfect can open. و + a new topic NP cuts — unless it elaborates rather than
   advances (E9).**
e. **Stop looking for a form-based rule for ولكن.** Use the running-segment length instead.
f. **MCQ is now solved: five consecutive 100s** (doc_c4cb5672fc6c, doc_cddb10bd6b23, doc_d715df9a8c78,
   doc_e438f8ad8abe, doc_eb9893e8f98b) plus two near-misses that S4/S5 would have caught. Run the
   three tests in order — modal option count, head-recurrence, uniform option shape — and do not
   reason from content.

---

# BATCH 17 (partial, 2 docs — list exhausted) — mean F1 91.33 (2026-08-09)

| doc | register | F1 | P | R |
|---|---|---|---|---|
| `doc_f7c344e48e78` | Qatari geography MCQ | **100.0** | 100 | 100 |
| `doc_fb2e077fe69e` | children's-magazine biography | 82.7 | **70.5** | 100 |

13 errors, **all FALSE CUTS, zero misses.** Precision 70.5 on one document. **This is the same failure
as batch 15 and it has the same cause: I carried a density figure from one document into another that
merely looked like it.**

## §F — NEW ENTRY. A CHILDREN'S MAGAZINE FEATURE IS NOT A CHILDREN'S PICTURE BOOK.

Source: `doc_fb2e077fe69e` (رحالة ومغامرون — a life of Ibn Khaldūn), F1 82.7, **13 false cuts**.
Gold: **31 boundaries over 444 tokens = 14.3 tokens/segment.** I priced it at 10.1, straight off
batch 16's `doc_f41918b2700e` (10.6), and over-cut by 42%.

Both are "written for children." One runs 10.6 tokens and cuts at every coordinated verb; the other
runs 14.3 and welds them. **The signal that separates them is not the audience, it is the SYNTAX
DENSITY of the prose**: the picture book is a chain of 3–5-token clauses with a named actor in almost
every one (`فخرج الضابط من الباب ونزل من أعلى الجبل`); the magazine feature is continuous adult-shaped
narrative with subordination, fronted adverbials and relative clauses. Read the first 60 tokens and
ask which one you are in.

### F4. THE DECISIVE FINDING: `ف` IS A CONTINUATION PARTICLE HERE AND NEVER TAKES A BOUNDARY.
Six of my thirteen false cuts were a ف I treated as a new event:
- @128 gold `…كما انتشر بمناطق عديدة من العالم فتوفي والده وفقد عددا كبيرا من معلميه وأساتذته ‖`
- @182 gold `…عمل كاتبا وحاجبا للعديد من الوزراء والسلاطين فراحت تلاحقه الاتهامات حتى ذهبت به مرات إلى السجن والأسر وكادت تصل به مرات إلى الموت ‖` (23 tokens, one segment, three ف/و verbs)
- @229 gold `…كما كانت تلاحقه منذ بداية شبابه فقرر الابتعاد عن بلاد المغرب والأندلس نهائيا ‖`
- @271 gold `…أنها حضرة الدنيا وبستان العالم فلم يغادرها إلا للحج بمكة وزيارة المسجد الأقصى بفلسطين ‖`
- @348 gold `…وخشي ابن خلدون أن يفتك بهم تيمورلنك فذهب إليه يفاوضه ويطلب الأمان لرفقائه ‖`
- @371 gold `…تظاهر بطاعته ثم عاد إلى مصر بحجة إحضار كتبه ‖` (ثم also welds)
> **This is LAW 2 in its ORIGINAL batch-2 form, and it is alive.** ف marks the consequence of what
> just happened and stays inside the sentence. `ثم` welds here too. The batch-13 scoping ("ف/ثم cut
> in exposition and modern narrative") does not reach this register.

### F5. What DOES cut — and all 31 gold boundaries fit it exactly:
1. **A FRONTED ADVERBIAL or subordinate clause**, boundary placed before it: `ولما رزق بابنه` ·
   `وبعد أن تعلم` · `وبعد ما يزيد عن العشرين عاما` · `ولما أرسل لأسرته` · `وقبل أن يهدأ حزنه` ·
   `ودون أن يحارب` · `ولإعجاب الطاغية بعلمه` · `وحتى يتجنب ابن خلدون غضبه` · `برغم كثرة ما عانى` ·
   `وفي عام 1406 م` · `غير أن وشايات زملائه`.
2. **و + a NEW ACTOR taking the stage**: `وأخذ ابن خلدون` · `واضطر للعمل` · `وكانت آخر نكبات ابن خلدون` ·
   `وخشي ابن خلدون`.
3. **An asyndetic clause that opens a new SCENE** (not merely a new fact): `هاجر من أشبيلية` ·
   `انتشر وباء الطاعون` · `عمل كاتبا وحاجبا` · `اصطحب الناصر لجنوده` · `كتب في التاريخ كتابا ضخما`.
4. **`كما` in its "moreover" sense** at @430 — the one connective call I got right here, and Z3's
   register test held even though I mis-measured the register.

### F6. Three welds that contradict rules I had just written down:
- **@39 `أما … فقد` WELDS**: gold `كان من الخلدونيين العديد من الوزراء والأمراء أما والده فقد انصرف عن وظائف الدولة وتفرغ للعلم ‖`. This is the SECOND document in three batches where أما welds (see T13). **Retire "أما opens a segment" as a default.** It cuts only inside an author's explicit enumeration (`من الناحية العسكرية / السياسية / الاقتصادية`), which is now 1 case for and 2 against.
- **@107 clitic `لكنه` WELDS**: gold `كان قوي الفهم وشديد الإقبال على العلم لكنه وهو في السابعة عشرة من عمره واجه أولى نكباته ‖`. My E8 length rule (cut once the running clause reaches ~7 tokens) fired here and was wrong. **Lower the threshold or drop the rule; the clitic weld is the better bet.**
- **@416 an ASYNDETIC NEW FACT WELDS**: gold `كتب في التاريخ كتابا ضخما من سبعة أجزاء شهد عدد من العلماء الأوربيين أنه أعظم عمل أنجزه الفكر البشري في كل الأزمان ‖` — 22 tokens. The §Q rule "an asyndetic finite clause after a complete statement is a boundary" is a NEWS-WIRE rule. In narrative, an asyndetic clause that COMMENTS on what precedes welds; only one that opens a new scene cuts (F5.3).

## THE META-LESSON OF THIS WHOLE RUN — write it at the top of the next session

Four batches, four different directions:

| batch | ratio | what I did |
|---|---|---|
| 14 | 3.5 : 1 **under** | trusted §O's "long ف/و chains weld" in Alf Layla |
| 15 | 8 : 1 **over** | applied batch-14's "at a coin-flip, CUT" to a Wikipedia article |
| 16 | 2 : 1 **under** | applied batch-15's caution to a children's picture book |
| 17 | 13 : 0 **over** | applied batch-16's picture-book density to a magazine feature |

**Every single one of these is the same error: I transferred the LAST document's unit length to the
NEXT document because the two shared a surface label** (literary prose / encyclopedia / "for
children"). The label is never the register. **The register is a measured token-per-sentence figure
plus a connective profile, and the two are independent** — Alf Layla at 11 tokens welds its ف-chains
while a picture book at 10.6 cuts every و, and a magazine at 14.3 welds every ف while a news brief
at 7–15 cuts asyndetically.

> **PROCEDURE FOR THE NEXT SESSION, and it replaces the density step in every earlier checklist:**
> 1. Segment the first ~80 tokens using ONLY boundaries I can name a structural reason for
>    (`<NL>`, fronted adverbial, new actor, new scene, verse, option).
> 2. Read off the implied tokens-per-segment. That is the document's unit — not a memory.
> 3. Then, and only then, decide the connective profile: **does ف weld or cut IN THIS DOCUMENT?**
>    Find one clear instance of each polarity before committing. ف and و can go opposite ways
>    (this document: ف welds, و + new actor cuts).
> 4. Carry NOTHING forward from the previous document except the protected list (LAW 0‴), the layout
>    rules (§S, §J, §M, §W) and the free `<NL>` boundaries. Those are the only things that transfer.

**What is genuinely solved after this run and should be treated as done:**
- **MCQ / worksheets: six consecutive 100s** (`doc_c4cb5672fc6c`, `doc_cddb10bd6b23`,
  `doc_d715df9a8c78`, `doc_e438f8ad8abe`, `doc_eb9893e8f98b`, `doc_f7c344e48e78`) plus two 97–99s that
  S4/S5 now cover. Run the three tests in order and never reason from content.
- **Layout registers (song lyric, verse, poem): 100s.** The line is the unit; do not look for prose cues.
- **Scripture by verse: `doc_f2767b56c842` at 100.** Identify book and chapter, enumerate verses, one
  boundary each. Do the reference work instead of segmenting by cue.
- **Everything still losing points is running prose, and the loss is always the unit length.**
