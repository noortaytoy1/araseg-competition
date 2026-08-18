# Doctrine — Jury 1 — Arabic sentence segmentation (track: nopa)

Cumulative. Nothing here is ever deleted. When a later batch overturns an entry I rewrite the
entry in place, mark it **[REVISED batch N]**, and keep the old wording underneath as
**[SUPERSEDED]** so the reasoning stays on the record.

---

## STATUS LOG

**Batch 1** — 19 docs, F1 range 33.3 → 100.0. Median ≈ 88. Two catastrophes: `doc_25ed373f8aeb`
(MCQ exam bank, 33.3) and `doc_16f2b3eccb91` (classical philosophy, 75.7).
Error profile: ~250 MISSED vs ~35 FALSE CUT. **I under-segment by roughly 7 to 1.**
That imbalance is the single most important fact about my calibration. Everything below is
written to correct it, with the false-cut rules as the brakes.

**Batch 2** — the graded file I was handed (`mistakes_j1.txt`) is, as far as I can verify,
**the same grading output as batch 1**. Same 19 doc IDs, same F1s to one decimal (33.3 → 100.0,
median 88.9), and the same token offsets and quoted spans: I checked ~30 items against the
citations already written into this file — @811, @701, @594, @272, @951, @863, @766, @15997,
@639/@647/@655, @801–@821, @2822, @11118 — and every one matches character for character.
So batch 2 contains **no new evidence**. I am recording that rather than pretending to a second
sample. I used the re-read the only way it can honestly be used: as a **full audit of batch 1's
doctrine against the complete mistake list**. The audit found four entries that are wrong or
misfiled (**0b, 0c, C5, D2**) and about a dozen mistakes I never filed at all. Corrections and
additions are below, marked `[REVISED batch 2]` / `(batch 2)`.
Net effect this round: the doctrine gets *more correct*, not more confident.

**Batch 4** — first genuinely NEW evidence since batch 1. Mean F1 **93.50**
(98.6 / 100.0 / 88.9 / 96.0 / 84.0). Four of the five were already in the batch-1 grading; the new
document is `doc_02c1581a0f82` at **84.0 — and it is the first document in my life where I
OVER-segmented: 8 FALSE CUTS, 0 MISSED, precision 72.4 / recall 100.0.** My entire doctrine was
built on a 7:1 under-segmentation prior. That prior is genre-conditional, not global, and this
document is the proof. It forced §0g, the biggest single addition since §0e.

---

## 0g. THE `<NL>`-DENSITY LAW (batch 4) — the printed line, finally observable

Batch 2 filed "the annotator follows the printed line" as an **unobservable** variable (0e
corollary, H5, J4, J7, open contradiction #4). That was wrong. **The `<NL>` tokens ARE the printed
lines.** I had been treating them purely as a constraint on where boundaries are forced; they are
also the strongest available *evidence about granularity*.

> **THE LINE-DENSITY TEST. Before segmenting, measure the mean number of real tokens between
> `<NL>` tokens.**
> - **Short lines (mean ≲ 8–10 tokens)** → the document is laid out *as lines* — a song, a poem, a
>   vocabulary page, a drill, a credits block, a table. **The line IS the unit. Cut at the `<NL>`
>   boundaries (already forced) and almost nowhere else.** Internal cuts must now be *justified*,
>   not defaulted to.
> - **Long lines (mean ≳ 25 tokens)** → running prose in paragraphs. The clause-chunk is the unit;
>   the whole of §0–0d and PASS 2 applies inside each paragraph.
> - **In between** → read the content; expect a mixed document (prose body + heading lines).

`doc_02c1581a0f82` is 96 tokens over ~20 lines — **mean ≈ 4**. Every one of my 8 false cuts was a
cut *inside* a printed line:

```
gold: <NL> أصغينا حدثنا الحجر ‖ <NL>
mine: <NL> أصغينا ‖ حدثنا الحجر ‖ <NL>        (asyndetic clause pair — I applied D3)

gold: <NL> غائر عميق مجوف نافر بارز مغاور حمع مغارة الوسم والوشم الرسم على الجلد
      النقش الرسم على الحجر الرقش التزيين ‖ <NL>
mine: …مجوف ‖ نافر بارز ‖ مغاور حمع مغارة ‖ …على الجلد ‖ …على الحجر ‖ الرقش التزيين ‖
                                              (6 false cuts in one line — I applied I4/J1)

gold: <NL> الرسام يرسم لوحة والشاعر يكتب قصيدة فماذا يفعل النحات ‖
mine: …لوحة ‖ والشاعر يكتب قصيدة ‖ فماذا يفعل النحات ‖    (I applied A1 و-cut and B1 ف-cut)
```

**Three of my most reliable laws — asyndetic-clause-cuts (D3), parallel-siblings-are-a-list (I4),
و/ف-opens-a-unit (A1/B1) — all fired, and all three were wrong, because they are PASS-2 laws and
this document never reaches PASS 2.** The gloss line is the sharpest case: `غائر = عميق مجوف`,
`نافر = بارز`, `مغاور = جمع مغارة`, `الوسم والوشم = الرسم على الجلد`, `النقش = الرسم على الحجر`,
`الرقش = التزيين` — six term+gloss pairs, textbook I4 "parallel siblings with no connective", and
gold keeps all 25 tokens as **one unit** because they occupy **one printed line**. J4 (table rows
merge) was right and I4 (options split) was wrong *for this document*, and the `<NL>` layout says
which — no syntactic test ever could.

**This retires open contradiction #1 as a linguistic problem.** Split-vs-merge for parallel
siblings is decided by layout: siblings on **separate lines** (or in a document with no line
structure at all, like the MCQ bank) split; siblings **sharing one line** merge. `doc_25ed373f8aeb`
scored 33.3 precisely because it is an MCQ bank with **no `<NL>` at all** — nothing to ride on, so
the options must be recovered syntactically (I4). Where lines exist, trust the lines over I4.

**Guard against over-correcting.** This law only *restrains* cutting in short-line documents; it
never licenses merging across a `<NL>`, and it does nothing in prose. My 7:1 under-segmentation
prior still governs every long-line document, which is most of the corpus. Read the density first,
then choose which prior I am operating under.

### 0g-bis. Refinement after batch 5 — it is LINE LENGTH, per line, not document mean

Batch 5 (mean F1 **94.46**) hit the same law from three directions and sharpened it. The variable
is not a document-level average; it is **the length of the individual line I am standing in.**
The chance that a line contains an internal boundary scales with its length:

| line length | policy | evidence |
|---|---|---|
| **≲ 8 tokens** | **never cut inside.** The line is the unit. | `doc_0696314023ab` (6 FC), `doc_091dca4b6b3c` @114, `doc_02c1581a0f82` |
| **~9–20 tokens** | cut only for a genuinely **new proposition**, and note the inversion below | `doc_0ad7fe19f9bb` @113 cuts, @45/@179 do not |
| **≳ 25 tokens** | full PASS 2. Multiple internal cuts are normal. | all prose docs; `doc_0ad7fe19f9bb` @77–@103 is the exception that proves it |

**`doc_0696314023ab`** — the Arabic Batman title song, 6 FALSE CUTS / 0 MISSED, F1 80.0. Every
line is one verse line and gold cut **only** at `<NL>`:
```
gold: <NL> عبر الطريق نحو الحقيقة فلا مكان للسر ‖ <NL> لبطلنا روح امينة تكتشف اليقينا ‖
      <NL> ضوء لمع وسط المدينة رسم ندائا لمنادينا ‖ <NL> حينا يبدو ويختفي حينا تلك اشارة بات مان ‖
mine: …نحو الحقيقة ‖ فلا مكان للسر ‖ … روح امينة ‖ تكتشف اليقينا ‖ … وسط المدينة ‖ رسم ندائا… ‖
```
I fired B1 (clause-initial ف) and D3 (asyndetic clause, even 2 words). **Both are PASS-2 laws and
neither applies inside a verse line.**

**K1 IS WRONG — [REVISED batch 5].** See §K below. Batch 1 wrote "each hemistich is a unit" from
two MISSED items in `doc_20dcf69f9905`. Reading that document's layout now: the poem is printed
**one hemistich per `<NL>` line** (`في البر والبحر لا فرق نرى العللا`, `أصابها الضر باتت تدمع
المقلا`, …), so the hemistich boundary was *already* being handed to me by `<NL>`. Gold cut inside
a verse line only **twice in ~25 lines** (@143 `حمض سموم غبار ‖ كونت مطرا`, @159 `الأرض تصرخ ‖ من
يأتي يساندها`). My batch-1 behaviour — treat the line as the unit — was **right**, and the rule I
wrote down from those two misses would have produced ~23 false cuts had I ever applied it.

> **META-LESSON, and it applies to this whole doctrine: a rule induced from MISSED items alone,
> without asking how often the *opposite* case occurs in the same document, can be net-negative.**
> A miss tells me gold cut *there*; it says nothing about the 23 places gold did not. Before
> promoting any "always cut at X", I must count the X's gold left alone. K1 failed this test.
> Entries I now suspect of the same defect, to be checked when evidence arrives: A3/G1 (`حيث`),
> F4 (`وقد`), D3 (asyndetic 2-word clause), I4 (parallel siblings).

**The medium-line inversion (`doc_0ad7fe19f9bb`, interview in a children's magazine, 93.3).**
Inside 9–20-token lines this document does the *opposite* of the prose laws — the **asyndetic**
junction cuts and the **connective** junctions do not:
- MISSED @113 — `السفارة هي رابط بين دولتين ‖ تعمل على تطوير العلاقات بينهما ‖`. Asyndetic
  definition-then-new-predication. This is **A6**, and A6 survives inside a medium line.
- FALSE CUT @45 — `كان لي شرف المشاركة في هذا المخيم حيث حصلت على وسام وعدد من الجوائز الأخرى ‖`.
  **`حيث` did not cut.** A3/G1 say it always does. It yields to **0d**: the حيث-clause elaborates
  the participation just reported → comment → attach.
- FALSE CUT @179 — `السفير المقيم هو الذي يقيم في نفس البلد وغير المقيم هو الذي يتردد حسب الضرورة ‖`.
  One question asked for a *difference*; the two halves are a **matched pair answering it** — one
  record (M3), not two units. Same family as `أما X فـ Y` (F5).
- FALSE CUT @96 — `وقد` did not cut, because the unit runs on to a **cataphoric announcer**:
  `…لإجراء حوار صحفي لمجلة ماجد وقد أجاب مشكورا على الأسئلة التالية ‖`. H1 governs: the unit ends
  **at** `التالية`. `وقد` was mid-stem. **F4 yields when the وقد-clause completes an announcer stem.**

**Consolidated reading:** `<NL>` gives me the annotator's own units for free in laid-out documents.
Spend my judgement on long lines, where they are not free.

---

## 0. THE MASTER LAW (batch 1)

These annotators do **not** segment on grammatical sentences. They segment on **information
chunks — roughly one finite predication each**. A coordinating particle (و، ف، ثم، لكن، بل)
does not "join a sentence"; in this corpus it *opens a new unit* whenever what follows carries
its own predication.

The operational test I got wrong all batch, and will now apply at every و / ف:

> **THE CONJUNCT-WEIGHT TEST.** Look at what follows the connective. Does it carry its own
> predication — a finite verb, a participle, a relative clause, its own PP/adverbial frame,
> its own subject? **If yes → CUT.** If it is a *bare* noun or adjective hanging off the same
> predicate as the previous word → **DO NOT CUT.**

The cleanest proof of this in batch 1, my own MISSED @639/@647/@655 in `doc_8b2d52902b9b`:

```
gold: جاءت منتخبات تايلاند والعراق وأستراليا وسلطنة عمان في المجموعة الأولى ‖
      ومنتخبات فيتنام واليابان والإمارات وقطر في المجموعة الثانية ‖
      ومنتخبات ماليزيا وإيران والصين وأوزبكستان في المجموعة الثالثة ‖
      ومنتخبات إندونيسيا وكوريا الجنوبية والسعودية والبحرين في المجموعة الرابعة ‖
mine: ...one single unit, no cuts at all.
```

Inside a group, `والعراق وأستراليا وسلطنة عمان` are bare nouns → **no cut**. Between groups,
each `ومنتخبات … في المجموعة N` carries its own PP frame → **cut**. Same و, opposite verdicts,
decided purely by the weight of the conjunct. Confirmed again by `doc_66c98646b9a9` @298 where
`السمع والبصر والشم والذوق واللمس` stays whole, and by `doc_405d5fba81e0` @830–@852 where
`ونهرا يجري عبر صدع…` / `وقمم غابات مستدقة الطرف… تغطي أحد المنحدرات` / `وأناسا ضئيلي الحجم
يشبهون النمل` each become a unit **even though they are grammatically all direct objects of one
verb**. Weight beats syntax. Always.

**[AMENDED batch 2] — the weight test needs a ceiling.** The audit turned up the counter-case I
never filed: `doc_66c98646b9a9` @152–@166, where **seven** و-conjuncts run without a single
boundary —

```
gold: فهو الوجود وهو الكمال وهو التمام وهو الحسن وهو البهاء وهو القدرة وهو العلم ‖
      وهو هو و كل شيء هالك إلا وجهه ‖
```

Every `وهو X` is technically a predication (copula + predicate), so a naive weight test says
"cut seven times". Gold cuts **none** of them, and then cuts where the frame *breaks*.
So: **a chain of identical short frames over one unchanged subject is ONE unit — an isocolon,
not an enumeration.** Compare `doc_de12d5da4854` @711/@729
(`…بدون استثناء ‖ وفريق يعتقد بإفلاس الحضارة الغربية… ‖ وفريق يقف موقفا بين بين…`), which *does*
cut at every repetition, because there each conjunct has its own verb, its own complement and a
different referent. **Refined test: does the conjunct bring a new predicate *only* (→ merge,
isocolon), or a new predicate *plus material of its own* (→ cut, enumeration)?**
I merged the `وهو` chain correctly in batch 1 — but only because I was merging everything.
Now I know why it is right, which means I will not over-correct it away next batch.

### 0b. The brake: THE SHORT-TAIL RULE — **[REVISED batch 2: word count was the wrong variable]**

**Current statement.** A **connective-headed** span that *comments on* the clause before it is
absorbed by that clause. Length is only a symptom — comments happen to be short. Two corrections
to the batch-1 wording:

1. **It applies only to spans that begin with a connective** (و / ف / أما / لذا / بل). Every one
   of the eight fragments in the table below is connective-headed. An **asyndetic** short span is
   *not* protected: `doc_3272f207e8c9` @397 reads
   `…يبدو لي في أسفل الكومة ‖ أسحبه وأتأمله ‖ إنه في غلاف طبعته الحادية عشرة…` — a **two-word**
   asyndetic clause gets its own unit, and I merged all three. Short ≠ absorbed.
   Short **and connective-headed and commenting** = absorbed.
2. **The real test is §0d (advance or comment), not the word count.** `doc_5e12ebe2976f` @245 is
   the proof: my false cut stranded `والبعض يؤرخ لبدايات العلم التجريبي من تلك السنة` — **eight**
   words, well over the "≤5" threshold, full predication, its own subject — and gold still
   absorbs it, because it is an attribution *comment* on the fact just stated, not a new fact.

**[SUPERSEDED batch-1 wording]** — *"In running prose only, a cut that would strand a ≤5-word,
non-parallel, dependent-feeling fragment is not taken — the fragment is absorbed by the neighbour
that governs it."* Right about the direction, wrong about the variable. Kept because the evidence
table it produced is still good, and because word count remains a usable tiebreak when I cannot
tell advance from comment:

| my false cut | the stranded fragment | words |
|---|---|---|
| @811 `doc_8b2d52902b9b` | وتتواصل مرحلة التفوق الإماراتي آسيويا | 5 |
| @701 `doc_3272f207e8c9` | فتفرغت للكتابة بشكل كامل | 4 |
| @594 `doc_405d5fba81e0` | وحيرتها فكرة الألمان | 3 |
| @272 `doc_5e12ebe2976f` | ومضى النهر يسقي أرض البشرية العطشى | 6 |
| @951 `doc_405d5fba81e0` | وهذا ما فعلته الطفلة مرارا وتكرارا | 6 |
| @863 `doc_51fccce927eb` | وفي السلة أضع الفضلات | 4 |
| @766 `doc_405d5fba81e0` | أما أسجارد والآلهة فلم تكن كذلك | 5 |
| @15997 `doc_361233315d24` | والله الموفق | 2 |

**The short-tail rule does NOT apply to lists, tables, MCQ options, verse, credits, or
word-drills.** There a single word is a legitimate unit. Genre decides whether the brake exists.

### 0c. The two-clause budget — **[REVISED batch 2: there is no numeric budget]**

**Current statement.** There is no clause count. Units end where the text stops commenting and
starts advancing (§0d). The audit found a three-clause unit that the budget flatly forbids —
`doc_de12d5da4854`, my FALSE CUT @229:

```
gold: عندئذ يفقد القديم كل ما كان له من اعتبار بل يصبح مكروها ومنفورا منه
      وتعتبر روح المحافظة من الأمور الشائنة التي تسيء إلى… ‖
```

Three finite clauses, two connectives (بل، و), one unit. The budget predicts a boundary before
`وتعتبر`; gold has none, because all three clauses are the *same evaluation of the same topic*.
Conversely `doc_3272f207e8c9` @116/@123 keeps two long وكان-clauses together for exactly the same
reason (`وكان عمل الباحث في تلك الأيام…` restates "it was hard work"). Length and count predict
neither case; advance-vs-comment predicts both.

**[SUPERSEDED batch-1 wording]** below. Its reading of @631 is still a correct *observation* —
`ثم` opens, `ودفعته` and `فصدر` ride along — but the "budget" was the wrong explanation of it,
and I also used it in C3 to explain away `وبذلك`, which I am now withdrawing.

A unit tolerates about **one to two finite clauses**. A short second clause gets absorbed
(0b); a *third* connective is almost always a boundary. Watch `doc_3272f207e8c9` @631:
`ثم كتبته في عامين ودفعته إلى المطبعة فصدر عام…` — gold cuts before `ثم` (new unit) and then
lets `ودفعته` and `فصدر` ride along inside it, because they are short same-subject verbs
filling out the same unit's budget. And `doc_3272f207e8c9` @116/@123: gold cuts before
`وكانت مهمة شاقة…` (opening clause 1) but **not** before `وكان عمل الباحث…` (clause 2, budget
still open). I did the exact opposite — I skipped the real boundary and cut at the fake one.

### 0d. THE ADVANCE-OR-COMMENT TEST (batch 2) — the brake I actually needed

This is the biggest thing the audit produced. It replaces **both** the word-count brake (0b) and
the "take the earlier candidate" heuristic (D2), which batch 1 got wrong in opposite directions.

> At every و / ف / بل / أما / لذا boundary in **running prose**, ask what follows:
> does it **ADVANCE** — a new event, a new agent, a new time or place frame, a new topic, a new
> addressee — or does it **COMMENT** on the proposition just stated: evaluate it, restate it,
> conclude from it, refer back to it anaphorically, or answer the question it just asked?
> **ADVANCE → CUT. COMMENT → ATTACH.**

Batch 1 wrote "when two candidates are close, take the EARLIER one" (D2). The full file says that
is a coin flip — four pairs each way, and I lost points on all eight:

| candidate pair | gold takes | why, by advance-or-comment |
|---|---|---|
| @116 / @123 `doc_3272f207e8c9` | earlier | `وكان عمل الباحث…` restates "it was hard work" → comment |
| @931 / @937 `doc_de12d5da4854` | earlier | `لا شك في أن…` answers the question just posed → comment |
| @438 / @443 `doc_16f2b3eccb91` | earlier | `غير أنها لتعاقبها عليه…` qualifies the same claim → comment |
| @229 (no pair) `doc_de12d5da4854` | no cut | `وتعتبر روح المحافظة…` evaluates the same topic → comment |
| @701 / @705 `doc_3272f207e8c9` | **later** | `ولم أفعل شيئا آخر خلال السنوات الثلاثين` = new time frame → advance |
| @594 / @597 `doc_405d5fba81e0` | **later** | `وراودتها أحلام بوجود ألمان…` = new event → advance |
| @272 / @278 `doc_5e12ebe2976f` | **later** | `فجاء المحرك البخاري في عام ١٧٦٩ م` = new event + new date → advance |
| @951 / @957 `doc_405d5fba81e0` | **later** | `وفي كل مرة كانت تجربة جديدة` = new frame → advance |

In every one, the span that got **absorbed** was a comment — `وتتواصل مرحلة التفوق الإماراتي
آسيويا`, `فتفرغت للكتابة بشكل كامل`, `وحيرتها فكرة الألمان`, `ومضى النهر يسقي أرض البشرية العطشى`,
`وهذا ما فعلته الطفلة مرارا وتكرارا`, `والبعض يؤرخ لبدايات العلم التجريبي`, `وتعتبر روح المحافظة
من الأمور الشائنة`, `والله الموفق` — and the span that got the **cut** opened a new event or
frame. Position never mattered. Meaning did.

### 0e. THE NATIVE-UNIT LAW (batch 2) — granularity before syntax

Batch 1 stated this only locally, buried in J3 ("the annotator segments at the granularity of the
task"). It is general, and it outranks everything in §0–0d:

> **Each genre declares its own unit, and the annotators segment to that unit. The clause is only
> the default when the document has no unit of its own.**

- constitution / statute → **the article** (H4: three asyndetic finite clauses, one unit)
- exam paper → **the stem, and each option** (I2/I3)
- workbook drill → **the printed word** (J1)
- workbook reading passage → **the passage** (J3)
- table → **the row** (J4); reference entry / bio stub → **the record** (M3)
- verse → **the hemistich** (K1)
- isnād → **the transmission link** (L1)
- credits, headings, ordinal labels → **the line** (M1, M2, H2)
- everything else — encyclopedic, essay, memoir, narrative, popular science, classical prose →
  **the clause-sized information chunk**, arbitrated by §0 and §0d.

The cost of getting this layer wrong is an order of magnitude larger than any connective mistake:
`doc_25ed373f8aeb` scored **33.3** purely because I read an exam paper as prose, while my worst
*prose* score in the same batch was 75.7. **Classify first. Always.**

**Corollary I have to live with.** In every visually laid-out genre the true unit is a *printed
line* that the tokenised input does not show me. That is the hidden variable behind the residue I
filed as "annotator noise" in batch 1: H5 (`العملة سك النقود السياسة النقدية` kept as one item),
J4's rows, `doc_51fccce927eb` @233 (`ماجد صفي جميل ‖ فسيح ومرتب ‖` — three bare adjectives, cut
between them), and the @936-vs-@938 inconsistency. It is not noise; it is layout I cannot see.
Infer it from parallelism and from announcers, and accept a few points of irreducible loss there
rather than inventing syntax to explain it.

### 0f. ASYNDETON, resolved (batch 2) — open contradiction #2 closed

Batch 1 left this open with a *syntactic* guess ("does the second clause introduce a new subject
→ cut"). The file kills that guess. `doc_8b2d52902b9b` @87 **cuts** a chain of verb-initial
asyndetic clauses:

```
gold: كما يوجد دور أول في نهائيات البطولة ‖ يقسم المنتخبات المتأهلة على مجموعات أيضا ‖
      يتأهل الأفضل منها…
```

while `doc_361233315d24` @9707/@9715 **merges** a chain of verb-initial asyndetic clauses:

```
gold: تتكون الهيئات المستقلة من عدد مناسب من الأعضاء من مختلف الاقاليم يشترط في عضويتهم
      توافر معايير الكفاءة والنزاهة والاستقلالية ينتخبهم مجلس الاتحاد…
```

Same syntax, opposite verdicts. **The variable is genre (0e), not syntax.** Working statement:

- **Asyndetic finite clauses CUT** everywhere *except* where the genre declares a unit bigger
  than the clause — inside a legal article (H4), a workbook reading passage (J3), a table row or
  reference record (J4/M3).
- **Asyndetic sub-clausal material MERGES** by default — term + gloss (J4), cell + cell (J4),
  noun + intensifying appositive (F5, `نهاية ‖ نهاية حقيقية نهاية العالم`), bare terms riding
  inside an instruction (J5) — **unless** the items are exchangeable siblings, i.e. answers,
  options or drill targets, in which case every one is its own unit (I3, J1).
- Clean line between the two halves: **an appositive that grows its own finite verb stops being
  an appositive and cuts.** `doc_405d5fba81e0` @421:
  `كانت تلتهم القصص بشراهة ‖ سطور من العلامات السوداء على صفحات بيضاء تشكل جبالا…` — I merged it.

---

## 1. LAWS BY TEXT TYPE

### A. Encyclopedic / Wikipedia factual prose
*(`doc_8b2d52902b9b` sports, `doc_32b866eb5dc4` prize, `doc_78b2ed132e2f` gazetteer)*

**A1. و + new finite predication = CUT. Non-negotiable. This is where I bled most.**
My misses: `…عام 1956 في هونج كونج ‖ واستمرت من 1 سبتمبر…` (@119);
`…في إيران ‖ وتمكن المنتخب الإيراني…` (@211); `…آل سعود ‖ وتمنح للعلماء…` (@27);
`…19 أكتوبر 1954 ‖ وبمقتضاها تم جلاء…` (@601). In this genre a paragraph is a *chain* of
و-clauses and **each link is a unit**. Three consecutive و-cuts in a row is normal, not
suspicious (`doc_32b866eb5dc4` @18/@27 back to back).

**A2. Date/place/score adjuncts do not protect the following و.** `…بين 15 سبتمبر و 20 سبتمبر ‖
وشارك فيها 10 منتخبات قسمت على مجموعتين ‖ واستطاع المنتخب الكويتي…` (@235/@242). Note the
internal `و` in `15 سبتمبر و 20 سبتمبر` is a bare-noun و → no cut. Conjunct-weight test again.

**A3. `حيث` = CUT.** `…كأول سنة تمنح فيها الجائزة ‖ حيث عقد الاجتماع الأول…` (@355).

**A4. `إلا أن` = CUT.** `…بجائزة الدراسات الإسلامية ‖ إلا أن جائزة الأدب قد حجبت…` (@654).

**A5. Stray numerals / reference marks are their own unit.**
`…منتخبات عمان والأردن وباكستان ‖ 3 3 ‖` (@934). I swallowed the `3 3` into the sentence.

**A6. Asyndetic descriptive continuation = CUT in this genre.**
`…على أعضاء كهربائية ‖ تتكون من خلايا خاصة…` (`doc_78b2ed132e2f` @183). No connective at all,
still a boundary, because the second clause is substantive description with its own predicate.
(Contrast H4 below, where asyndeton *merges*. Genre decides.)

**A7. The one brake in this genre:** a short evaluative/summary و-coda attaches.
My FALSE CUT @811: `…احتل المركز الرابع في النهاية وتتواصل مرحلة التفوق الإماراتي آسيويا ‖
وفي نهائيات كأس الأمم…` — the 5-word present-tense coda rides along; the boundary belongs at
the next و, the one that opens a new temporal frame (`وفي نهائيات…`).
*[AMENDED batch 2]* Generalise this: it is not "short coda", it is **comment vs advance** (0d).
The coda attaches because it evaluates the result just reported; the next و cuts because it opens
a new tournament. Length was a coincidence.

**A8. `فإذا` = CUT — contrast `أما إذا` = NO CUT.** `…وتصل درجة حبهم لفريقهم لدرجة كبيرة من هوس
التشجيع ‖ فإذا أخطأ الحكم خطأ ما ولو كان خطأ صغيرا…` (@1230). Same particle, same verdict, in
hadith narrative: `…ولا يفقه ما يقول حتى دنا ‖ فإذا هو يسأل عن الإسلام ‖` (`doc_27a114d4e376`
@44). The ف does the work; the conditional إذا rides *inside* the new unit. I had `أما إذا` in
the no-cut column and never noticed that bare `فإذا` is its opposite.

**A9. Asyndetic verb-initial clauses chain-cut in this genre.** `…حسب نظام المجموعات ‖ كما يوجد
دور أول في نهائيات البطولة ‖ يقسم المنتخبات المتأهلة على مجموعات أيضا ‖ يتأهل الأفضل منها…`
(@87). Two boundaries, zero connectives; I took neither. This is A6 generalised from description
to any predication, and it is exactly what a legal article does *not* do (H4) — see 0f.

**A10. `وهو / وهي` + a nominal predicate = CUT; `وهي أن` (complementiser) = NO CUT.**
Cut: `…على بعد 160 كيلو مترا جنوب شرقي العاصمة أبو ظبي ‖ وهي المدينة الثانية في إمارة أبو ظبي ‖`
(`doc_78b2ed132e2f` @386); `…من مسكن صاحبه الأستاذ زاهر ‖ وهو رجل ظريف طيب النحيزة من أولئك
الذين يرضون…` (`doc_9d15d9d5d637` @991). No cut: `…برزت المفاجأة الثانية وهي أن عناصرها الوراثية
قادتها إليه ‖ إذ كانت…` (`doc_7bc0656a48ce` @354) — there `وهي أن` is the *complement* of
المفاجأة, i.e. inside the clause, not a new one. Note the boundary with B9/§0: a **single**
`وهو + NP` describing a newly introduced referent advances and cuts; a **chain** of `وهو + NP`
tags on one unchanged subject is isocolon and merges.

**A11. Quoted speech is segmented like prose, not swallowed whole.** `…قال الحكم أتمنى ألا يحدث
مثل هذا الاعتداء مستقبلا ‖ فإننا عندما نرى عنفا من هذا النوع…` (@1322). The speech-verb rule (E3)
attaches the *opening* of a quotation to its `قال`; it does not exempt the rest of the quotation
from the ف / و laws. I treated the whole quote as one block.

**A12. `وأن / وألا` coordinated obligation clauses cut.** `…التي تحتاج للصدق والأمانة والعدل قبل
أن تحتاج للفكر الرياضي ‖ وألا يسهو أثناء المباراة لأنه لو سها للحظة واحدة…` (@1166). Same shape
in legal text (H3, @6299 `ويجب أن تتوافر…`).

---

### B. Classical philosophical / scientific prose
*(`doc_66c98646b9a9`, `doc_16f2b3eccb91` — Ibn Tufayl; my two worst prose scores, 81.6 and 75.7)*

**B1. Clause-initial ف = CUT. Essentially always.** This alone would have fixed ~20 misses in
one document. The inferential/narrative ف of classical prose is a *full stop*, not a comma:
`فرأى، فعلم، فتبين، فطلب، فنظر، فأخر، فلم يجد، فزالت، فارتسم، فينتقل، فتقهقر، فهو، فهي، فإنها،
فليست، فلا`. My misses: `…معرفة الأكثر ‖ فطلب أولا الوقوف…` (@76);
`…لتفنن أفعالهما ‖ فأخر التفكير في صورهما ‖` (@102); `…فلا وجود إلا هو ‖ فهو الوجود…` (@152);
`…والذوق واللمس ‖ فرأى أنها لا تدرك…` (@298); `…بطل حكم الصورة ‖ فزالت الصورة المائية…` (@702).

**B2. Parallel enumerative clauses each get their own unit — even short ones.**
`…عند تصادم الأجسام ‖ والبصر إنما يدرك الألوان ‖ والشم يدرك الروائح ‖ والذوق يدرك الطعوم ‖
واللمس يدرك الأمزجة والصلابة واللين والخشونة والملاسة ‖` (@325–@335). I merged the whole
sequence into one. **Parallelism defeats the short-tail rule (0b).** If N chunks share the same
shape, cut all N. Note the tail `الأمزجة والصلابة واللين والخشونة والملاسة` — bare nouns,
no cuts. Same rule, both directions, in one line.

**B3. Chained و + new predication = CUT, same as A1.** `…صورة أخرى ‖ وحدثت له صورة أخرى بعد أن
لم تكن ‖ وصدر عنه بها أفعال…` (@719/@727). Repetition of vocabulary is not a reason to merge.

**B4. `إذ` = CUT.** `…من يفقد شمه ‖ إذ الأشياء التي يدركها البصر…` (@828);
`…وترك الجسم على الإطلاق ‖ إذ هذا الأمر لا يدركه الحس…` (@607).

**B5. `كما أن` = CUT.** `…في آلام لا نهاية لها ‖ كما أن من كان مدركا له على الدوام…` (@896).

**B6. `لكن / لكنه` = CUT.** `…هو للجسم من حيث هو جسم ‖ لكنه لم يتأت له بالحس…` (@294).

**B7. An embedded polar question is its own unit.**
`…ثم تفكر في هذا الامتداد إلى الأقطار الثلاثة ‖ هل هو معنى الجسم بعينه وليس ثم معنى آخر أو
ليس الأمر كذلك ‖ فرأى أن…` (@328/@341). The whole double question is ONE unit — I neither cut
before `هل` nor recognised that `أو ليس الأمر كذلك` belongs inside it.

**B8. `غير أن` = CUT — but yields to the short-tail rule.** Cuts at `doc_16f2b3eccb91` @420
(`…لم يتبدل ‖ غير أنه لا بد له من طول…`). Does **not** cut at @443, where the preceding clause
`ولا يمكن أن يعرى عنها` was only 5 words: gold put the boundary *before* `ولا يمكن` and let
`غير أنها…` continue.
**[SUPERSEDED lesson, batch 1]** *"When two candidate boundaries are 5 words apart, the annotator
takes the earlier one and skips the later."* — false in general (see 0d's table: four pairs go the
other way). **[REVISED batch 2]** The right reading of @420 vs @443: at @420 `غير أنه لا بد له من
طول وعرض وعمق` introduces a new requirement → advance → cut; at @443 `غير أنها لتعاقبها عليه تبين
له أنها معنى على حدة` qualifies the claim already made → comment → attach, and the boundary
belongs at the genuinely new clause `ولا يمكن أن يعرى عنها`. I took the later one and lost both.

**B9. ISOCOLON: a chain of identical short copular frames over one unchanged subject is ONE
unit.** `فهو الوجود وهو الكمال وهو التمام وهو الحسن وهو البهاء وهو القدرة وهو العلم ‖ وهو هو و كل
شيء هالك إلا وجهه ‖` (@152/@166). See the §0 amendment. **This is the ceiling of B2:** B2 cuts
parallel clauses that each carry a full new predication (`والبصر إنما يدرك الألوان ‖ والشم يدرك
الروائح`); B9 merges parallel *tags* that carry only a predicate word. The boundary falls where
the frame breaks — here at `وهو هو` plus the scriptural clause. My batch-1 answer was right by
accident; recording the reason so I do not "fix" it into seven false cuts next batch.

**B10. A quantity announcement + `أحدهما … والآخر …` is the colon cut (H1), in prose.**
`…أن الجسم بما هو جسم مركب على الحقيقة من معنيين ‖ أحدهما يقوم منه مقام الطين للكرة في هذا
المثال ‖ والآخر يقوم مقام طول الكرة وعرضها وعمقها…` (@478/@487). **The announcer family is not a
legal-text quirk.** Any stem that promises N items ends its unit right there, whatever the genre.

**B11. `ومعنى X أنه…` (definitional gloss) = CUT.** `…فإنها تكون مدركة بالقوة ‖ ومعنى مدركة
بالقوة أنها لا تدرك الآن وتدرك في المستقبل ‖ وفي حال فتحها واستقبالها للمبصر تكون مدركة بالفعل ‖
ومعنى مدركة بالفعل أنها الآن تدرك ‖` (@692/@702/@710). Four units; I produced one. Note this is
the exact opposite of J4, where a term and its gloss share a unit — in a table the *row* is the
unit (0e); in classical prose the gloss is a new predication.

**B12. `وإنما` = CUT** — E2 confirmed outside literary prose. `…فإنها ليست حقيقة ذاته ‖ وإنما
حقيقة ذاته ذلك الشيء الذي أدرك به الموجود…` (@512).

---

### C. Modern essay / op-ed
*(`doc_de12d5da4854`, `doc_5e12ebe2976f`, and the football-violence half of `doc_8b2d52902b9b`)*

**C1. و / ف + new predication = CUT** (A1, B1 carry over).
`…قطيعة تامة ‖ كما يغالي بعض المحافظين…` (@93); `…مسيطرة على جميع مظاهر الحياة ‖ وصارت كل
الأمور تتطور…` (@606); `…اختص بها متميزا عن سائر مملكة الحياة الأرضية ‖ فهو لا يكتفي…` (@15).

**C2. `كما` = CUT. `في حين` = CUT. `بينما` = CUT.**
`…أو يقول بها ‖ في حين أننا نجد في بعض الأدوار…` (@200).

**C3. Result connectives `ولهذا` / `لذا` ATTACH to the premise that licenses them.**
My FALSE CUT @187: gold keeps `…تستوجب العقاب الصارم ولهذا يطلبون معاقبة كل من يقدم عليها…`
as one unit. Same in `doc_8921b794ce19` @1024 with `لذا فمن شبه المؤكد…`.
*Caveat:* `doc_8921b794ce19` @1053 **does** cut before `وبذلك يمكنكم القول…`.
**[REVISED batch 2]** My batch-1 reconciliation appealed to the two-clause budget (0c), which I
have now withdrawn as a counting rule, so that explanation goes with it.
**[SUPERSEDED]** *"there the preceding unit already held two clauses … so the budget was spent."*
Better explanation, by 0d: `ولهذا يطلبون…` and `لذا فمن شبه المؤكد…` draw a conclusion *about the
same proposition* → comment → attach; `وبذلك يمكنكم القول إنني أؤدي خدمة عامة` turns to address
the audience — **new addressee, new speech act** → advance → cut. Confidence: medium.
**Still carried forward** — this is the one connective whose behaviour I cannot yet predict
without seeing the person/addressee shift.

**C4. Rhetorical question + its own answer = ONE unit.** My FALSE CUT @937 `doc_de12d5da4854`:
gold is `…وبنائها القديم ‖ وتصوروا ماذا سيكون مصير تلك العضوية لا شك في أن هذا المصير لن يكون
سوى…`. The boundary goes **before the imperative** (`وتصوروا`, which I missed at @931), and the
answer `لا شك في أن…` stays glued to the question. I placed the cut exactly one clause late.

**C5. `بل` = CUT when the correction is substantive (≥ ~6 words), ATTACH when it is short.**
Cuts: `…ليس شيئا تافها كما يبدو ‖ بل يخفي بين طياته ثقلا مزعجا جدا ‖` (`doc_8921b794ce19` @623).

**[REVISED batch 2 — the @229 half of this entry was misfiled.]** @229 is not a `بل` boundary at
all. My false cut fell *after* the بل-clause, before `وتعتبر`:

```
mine: …بل يصبح مكروها ومنفورا منه ‖ وتعتبر روح المحافظة من الأمور الشائنة التي تسيء إلى…
gold: …بل يصبح مكروها ومنفورا منه وتعتبر روح المحافظة من الأمور الشائنة التي تسيء إلى…
```

So the real lesson is **`بل X و Y` is one corrective package: a بل-correction absorbs its own
و-continuation.** Nine words of new predication with its own subject, and gold still merges,
because the whole run is one evaluation of one topic (0d: comment). Confidence: medium.
**[SUPERSEDED]** *"Attaches: …بل يصبح مكروها ومنفورا منه… (@229, 4 words)."* — the word count was
measured on the wrong span, which is how a real 3-clause unit got recorded as evidence for a
2-clause budget. Two wrong entries propping each other up; both now corrected.

**C6. `أي إن` (gloss/restatement) = CUT.** `…يتعرض للفشل عادة ‖ أي إنه يدق جرس إنذار ‖`
(`doc_8921b794ce19` @667).

**C7. `منها مثلا` = CUT.** `…من مؤلفات ويلز ‖ منها مثلا بعض قصص ألف ليلة وليلة…` (@460).

**C8. `عندئذ` = CUT (batch 2).** `…نجد في بعض الأدوار من التاريخ تتغلب نزعة التجديد على روح
المحافظة ‖ عندئذ يفقد القديم كل ما كان له من اعتبار…` (@215). A fronted temporal deictic opens a
unit — same family as `وهناك` (`doc_405d5fba81e0` @1066, `doc_3272f207e8c9` @751) and `وهنا` (G4).

**C9. A و-clause with a vague-quantifier subject that attributes an opinion is a comment →
ATTACH (batch 2).** My FALSE CUT @245 `doc_5e12ebe2976f`: gold `…وفي نفس العام استخدم الرياضيات
في حساب عجلة الجسم والبعض يؤرخ لبدايات العلم التجريبي من تلك السنة ‖`. Eight words, full
predication, new subject — and still absorbed. **This is the single case that broke the ≤5-word
brake and forced 0d.** Watch for `والبعض / والكثير / والبعض الآخر / ويرى بعضهم`.

---

### D. Memoir / first-person literary
*(`doc_3272f207e8c9`)*

**D1. Same و/ف/ثم laws.** `…التي في حوزتي ‖ فرحت أبحث في منافذ بيع الكتب…` (@34);
`…في أوائل الثمانينيات ‖ ثم تأسست شبكة الإنترنت…` (@149); `…تحت تصرف عامة القراء والباحثين ‖
واخترت مؤسسة هنداوي…` (@243).

**D2. [REVISED batch 2 — "take the earlier candidate" is a coin flip; use 0d instead.]**
Batch 1 called @116/@123 and @701/@705 "twin failures". They are not twins, they are **mirror
images**, and I described the second pair backwards: at @116/@123 gold takes the **earlier**
candidate; at @701/@705 gold takes the **later** one (it merges `فتفرغت للكتابة بشكل كامل` and
cuts before `ولم أفعل شيئا آخر`). The full file gives four pairs each way — table in §0d — and
the deciding variable is advance-vs-comment, never position.
**[SUPERSEDED]** *"When in doubt between two nearby candidates, take the EARLIER one. The
annotators open a unit at the first strong connective and then let it run to its budget."*
The two gold lines below are still exactly the right lines to memorise; only my explanation of
them was wrong. In both, the absorbed clause comments (`وكان عمل الباحث…` restates the hardship;
`فتفرغت للكتابة` restates the shift to professionalism) and the cut clause advances (`وكانت مهمة
شاقة…` opens the evaluation; `ولم أفعل شيئا آخر خلال السنوات الثلاثين` opens a new time frame).
Gold:
```
…بما يلزمني من مراجع ‖ وكانت مهمة شاقة وطويلة تستنفد المال والجهد وكان عمل الباحث…
…من طور الهواية إلى طور التخصص فتفرغت للكتابة بشكل كامل ‖ ولم أفعل شيئا آخر…
```
**D3. Asyndetic narrative clauses cut even when tiny.** `…العقل الأولى دراسة في الأسطورة يبدو لي
في أسفل الكومة ‖ أسحبه وأتأمله ‖ إنه في غلاف طبعته الحادية عشرة الصادرة…` (@397). **Two words.**
In present-tense cinematic memoir every beat is a unit. This is the case that proves the
short-tail brake (0b) applies only to *connective-headed* tails — nothing protects a bare
asyndetic clause, however short.

---

### E. Literary narrative & dramatic dialogue
*(`doc_9d15d9d5d637` — Sāra; 92.0, so my instincts are least bad here)*

**E1. Asyndetic anaphoric parallel clauses each get a unit.**
`…هذا كلام العجائز ‖ هذا حديث خرافة ‖ هذا مذهب عتيق أقدم من حواء والحية ‖ إنما خلقنا للسرور…`
(@467–@477). I ran all three `هذا` clauses together. **Repeated opening word = repeated
boundary.** This is the same principle as B2: parallelism forces cuts.

**E2. `إنما` = CUT.** (@477, @564).

**E3. Speech verbs open a unit.** `…اليوم الرائق الصافي الجميل ‖ وقالت ماذا تعني ‖` (@693);
`…مساعد الحكم ‖ وقال معلقا لا أفهم لماذا…` (`doc_8b2d52902b9b` @1431). The quoted content stays
attached to its `قال`.

**E4. A question is its own unit even when و-joined to what precedes.**
`…ولكن لا أظنه طويل المقام ‖ ألا تراه يتعثر بقدميه ‖` (@1416/@1423). Note `ولكن لا أظنه طويل
المقام` did **not** get its own cut off `وقالت ضيف` — short-tail rule inside dialogue.

**E5. In a play/script, the speaker name is its own unit** and the line after it is another:
`<NL> سارة ‖ <NL> على رسلكما أيتها الصديقتان ‖ لا تتخاصما ولا تشرعا…` (@104/@114). I merged the
vocative address into the following command.

**E6. An imperative opens a unit — in every genre (batch 2, consolidation).** Batch 1 had this
scattered across C4, E5 and J2 without naming it; it is one rule and it cost me four separate
misses:
`…وأولاد الأعمام وأولاد الأخوال ‖ وانظري كيف يضيق بيتك عن الطالبين والطالبات…` (@1527);
`…وهي تتناءى عنه مرحة ضاحكة احمد ربك ‖ عندك من سارة المظلومة حريم كامل…` (@770);
`…على رسلكما أيتها الصديقتان ‖ لا تتخاصما ولا تشرعا في تمزيق ما عليكما من ثياب ‖ إنها تستركما…`
(@104/@114); `…وبنائها القديم ‖ وتصوروا ماذا سيكون مصير تلك العضوية…` (`doc_de12d5da4854` @931);
`…بم تشبه المعلمة الأم ‖ وبم يشبه المعلم الأب ‖` (`doc_51fccce927eb` @417).
The pairing with 0d is exact: the imperative *advances* (new speech act) so it cuts, and the
answer or comment that follows it does not (C4).

---

### F. Translated non-fiction & literary translation
*(`doc_405d5fba81e0` Byatt/Ragnarok, `doc_8921b794ce19`)*

**F1. Long و-chained descriptive conjuncts each become a unit** — the strongest instance of the
conjunct-weight law. See §0. `صورت إحدى الرسومات الصخور في جبال ريزينجيبيرجا العملاقة ‖ ونهرا
يجري عبر صدع… ‖ …وقمم غابات… ‖ وأناسا ضئيلي الحجم… ‖ وأطياف غيوم حاجبة…` (@801–@852).
Grammatically one sentence; six units.

**F2. Same for narrative sequence adverbs.** `…تطابق هذه الكائنات العجيبة ‖ ثم أصبحت الصخور
والصدوع مساكن لها ‖ وأخيرا اعتبرت شخصيات مميزة…` (@887/@893).

**F3. `فـ` of explanation = CUT.** `…مع الطفلة النحيلة ‖ فهي تعلم بداخلها…` (@345);
`…ذات أصول اسكندنافية ‖ فقد هاجرت أسرتها…` (@663); `…إنها تتذكره ‖ فهو ذو شعر ذهبي…` (@272).

**F4. `وقد` = CUT.** `…كان الكتاب باللغة الألمانية ‖ وقد اقتبس من أعمال الدكتور دبليو فاجنر ‖`
(@565).

**F5. Brakes specific to this genre — all three of my false cuts here are the same mistake:**
- **`أما X فـ Y` contrast attaches to the statement it contrasts with.** @766: gold
  `…وقد حملت قصة بنيان رسالة ومعنى واضحين أما أسجارد والآلهة فلم تكن كذلك ‖ فهذا الكتاب هو…`
- **Appositive escalation is one unit.** @790: `…ثم وصوله إلى نهاية ‖ نهاية حقيقية نهاية
  العالم ‖` — `نهاية حقيقية` + `نهاية العالم` are ONE unit, not two. Repetition-with-intensification
  ≠ parallel enumeration. (Distinguish from E1: there the repeated element opens *full clauses*.)
- **Short و-aside absorbed** (@594, @951). Short-tail rule.

**F6. `إلى أن` = NO CUT** (subordinator). My FALSE CUT @719 `doc_5e12ebe2976f`:
`…جزءا أساسيا من حياتهم إلى أن تعافى مجددا في نهاية الستينيات…`.

**F7. An asyndetic appositive that grows its own finite verb becomes a unit.**
`…كانت تلتهم القصص بشراهة ‖ سطور من العلامات السوداء على صفحات بيضاء تشكل جبالا…` (@421).
Contrast F5's `نهاية حقيقية نهاية العالم` — bare NPs, no verb, one unit. **A verb is the line
between apposition (merge) and enumeration (cut)** — the cleanest discriminator I have for the
sub-clausal half of 0f.

**F8. Conjunct weight, both directions inside one sentence.** `…كان مجلدا متينا ومغلفا باللون
الأخضر ‖ وعلى غلافه صورة مثيرة ورائعة ل أودين أثناء صيده…` (@483): `ومغلفا` is a bare participle
sharing the same `كان` → no cut; `وعلى غلافه صورة…` fronts its own PP and predicates something
new → cut. Same و, same sentence, opposite verdicts — the §0 test in miniature.

---

### G. Popular science / magazine
*(`doc_7bc0656a48ce`)*

**G1. `حيث` = CUT** (three times in this doc alone: @505, @735, and @85 environment).
**`إذ` = CUT** (@354). **`بينما` = CUT** (@484).

**G2. Coordinated *verbal nouns* in an enumeration each get a unit** when they carry their own
complements: `…استلزمت مراقبتها ‖ ثم اكتشاف قفزات بعضها من الأحواض ‖ وسيرها على أقدام غريبة
لفترة تصل إلى…` (@85/@91). Not clauses at all — still units. Conjunct weight, again.

**G3. و + new finite predication = CUT** (@331, @556, @587, @760). Nothing special here; I just
did not apply A1 outside encyclopedias.

**G4. Narrative announcers take the colon cut too (batch 2).** `…بعد التغيير الكامل لمعالمها
الجغرافية القديمة ‖ وهنا جاءت المفاجأة ‖ الطيور الكبيرة الخبيرة قامت بتصحيح مسارها…` (@296).
`وهنا جاءت المفاجأة` is a **cataphoric announcement** — it ends its own unit exactly the way
`الآتية` does in H1. The announcer family now has four dialects and one law:

| dialect | trigger | example |
|---|---|---|
| legal | `الآتية / الآتي / بالآتي / ما يلي / على النحو الآتي / ومنها / وتشمل / يتكون … من` | H1 |
| exam | dangling `في / هو / هي / ل / من / بتاريخ` at the end of a stem | I2 |
| classical prose | quantity + `أحدهما … والآخر` | B10 |
| journalism / narrative | `وهنا جاءت المفاجأة` | G4 |

**One law: a stem that promises something ends its unit at the promise.**

---

### H. Legal / constitutional
*(`doc_361233315d24` — Yemen draft constitution, 96.0 but with the most instructive errors)*

**H1. THE COLON CUT — my single most mechanical, most repeated miss (~35 instances).**
When a stem ends in a **cataphoric announcer** and a list follows, the announcer **ends its own
unit**. I was already cutting *between* the list items; I simply never cut *before item 1*.
Announcers seen: `الآتية، الآتي، الاتية، الاتي، بالآتي، ما يلي، فيما يلي، على النحو الآتي،
ومنها، وتشمل، تضمن، بما يضمن، بما في ذلك، يحظر عليها، يحظر عليهم، يتكون … من`.
```
mine: …كما يحظر عليها المساس بالنظام الجمهوري الديمقراطي ‖ الحصول على تمويل خارجي ‖ …
gold: …كما يحظر عليها ‖ المساس بالنظام الجمهوري الديمقراطي ‖ الحصول على تمويل خارجي ‖ …
```
Also triggered by a **bare dangling function word**: `…ويتضمن الحق في ‖ اللجوء إلى القضاء…`
(@2822); `…ينشأ بقانون ‖ جهاز الشرطة الاتحادي…` (@11118); `…تكفل الدولة ‖ العناية باللغة
العربية…` (@1660); `…يتكون مجلس صندوق الإيرادات الوطني من ‖ وزير المالية الاتحادي رئيسا…`
(@12922). **If the stem ends mid-construction and parallel items follow, cut right there.**

**H2. Headings stack, and each level is its own unit.**
`…ثالثا السلطة القضائية ‖ مبادئ عامة ‖`  (@7364);
`…الباب السابع المالية العامة ‖ مبادئ عامة ‖` (@12534);
`…الباب العاشر الأحكام الانتقالية ‖ ترتيبات السلطات ‖` (@15074);
and even the bare ordinal: `… ‖ ثانيا ‖ مدينة عدن ‖` (@9361).
I glued the section label to the sub-label every time.

**H3. و + new obligation/predication = CUT.** `…والخلوة به ‖ وتخضع السجون لإشراف القضاء…`
(@2935); `…القانون الدولي الإنساني ‖ ويجب أن تتوافر…` (@6299). **`كما` = CUT** (@7906).

**H4. THE BRAKE, and it is the opposite of A6: inside one article, asyndetic clause chains
MERGE.** All of @9707, @9715, @9871, @10172, @14773 are the same false cut:
```
gold: تتكون الهيئات المستقلة من عدد مناسب من الأعضاء من مختلف الاقاليم يشترط في عضويتهم
      توافر معايير الكفاءة والنزاهة والاستقلالية ينتخبهم مجلس الاتحاد بأغلبية…
```
Three finite clauses, no connectives, ONE unit. **In legal prose an article is the unit; only a
connective (و/ف/كما) or a list-announcer breaks it.** No connective → no cut, however long.
Corollary: `أما إذا` = NO CUT (@5842). Closing formula attaches (@15997 `والله الموفق`).

**H5. Do not over-shred list items.** @11558/@11560 (`العملة سك النقود السياسة النقدية` = one
item) and @14024 (`المحميات الطبيعية والأنواع النادرة المناطق الرطبة والطيور المهاجرة…` = one
item). Confidence: low — probably source-line artefacts. **Bias: make the colon cut, then be
conservative about splitting *within* a run of bare NPs.**
**[UPGRADED batch 2]** "Probably source-line artefacts" is now the working explanation, not a
hedge — see 0e's corollary. The hidden variable across H5, J4, J5 and `doc_51fccce927eb` @233 is
**the printed line, which the tokenised input does not show me.** That reframes it from "noise I
should ignore" to "a real signal I cannot observe", which changes my behaviour: I stop trying to
find syntax for these cases, and I stop letting them make me timid about the colon cut, which is
observable and worth ~35 boundaries in this one document.

**H6. The article really is the unit — restated as the genre's native unit (0e).** All of @9707,
@9715, @9871, @10172, @14773, @1483, @5842 are the same false cut, and H4 already had the rule.
What batch 2 adds is *why* it does not generalise: `doc_8b2d52902b9b` @87 cuts an identical
asyndetic verb-initial chain in encyclopedic prose. Nothing syntactic separates them. Only genre
does. See 0f.

---

### I. Exam / MCQ question banks — MY WORST GENRE (33.3)
*(`doc_25ed373f8aeb`)*

**I1. Recognise the format first, segment second.** I read this document as running prose. It is
not prose at all. It is: *stem, option, option, option, option, stem, option…* with **no
punctuation, no lettering, no `<NL>` to help.**

**I2. THE STEM ENDS WHERE THE OPTIONS BEGIN — even mid-phrase, even after a dangling
preposition or copula.** Every one of these is a boundary I did not take:
```
…وصل الأمير عبدالله بن الحسين إلى مدينة معان في ‖ 21 تشرين الثاني عام 1920 م ‖ 21 تشرين الأول عام 1921 م ‖ …
…العام الذي ولد فيه الملك عبدالله الأول ابن الحسين في مكة المكرمة هو ‖ 1882 م ‖ 1893 م ‖ 1881 م ‖ 1880 م ‖
…يقدم ديوان المحاسبة تقريره السنوي … بحق المال العام ل ‖ مجلسي الأعيان والنواب ‖ الملك ‖ مجلس الوزارة ‖ المجلس القضائي ‖
…التمهيدية التي عقدت لمشروع وحدة الضفتين عام 1948 م هو ‖ بيت لحم ‖ أريحا ‖ يافا ‖ القدس ‖
```
Trailing `في / هو / هي / ل / بتاريخ / من` at the end of a question is a **colon in disguise** —
identical to H1.

**I3. EVERY option is its own unit**, whether it is one word (`أريحا`), two (`1882 م`), three
(`140 مندوبا`), or a full phrase (`دعم حق الشعب الفلسطيني في العودة إلى أرضه`). The short-tail
rule (0b) is **suspended** here.

**I4. The detection signal: N consecutive constituents of the same syntactic type and similar
length, with no connective between them.** Four dates. Four numbers+`مندوبا`. Four place names.
Four `م`-suffixed years. Four verb-initial achievement phrases
(`الإسهام في… ‖ إقامة… ‖ دعم… ‖ التوقيع على…`). Once I see three or four parallel siblings with
zero connectives, that is an option list, not a sentence — **cut before each one.**

**I5. Options may themselves contain internal `و`** (`مجلسي الأعيان والنواب`,
`وحدة الأردن وفلسطين هي مقدمة للوحدة العربية`) — that internal و is bare-noun و, no cut.
Boundaries go *between* siblings, not inside them.

---

### J. School textbooks & workbooks
*(`doc_51fccce927eb` grade-1 literacy, `doc_20dcf69f9905` poem + exercises)*

**J1. Word drills: ONE WORD = ONE UNIT.** My longest single run of misses in the batch
(@801–@821, ~20 consecutive):
```
gold: حافلة ‖ خلود ‖ <NL> سحور ‖ خليل ‖ أخي ‖ خميس ‖ حروف ‖ عجمان ‖ نخيل ‖ حضور ‖ خرز ‖
      <NL> دجاج ‖ جموع ‖ مساجد ‖ حصير ‖ خروف ‖ حكيم ‖ خالق ‖ نجوم ‖ جميل ‖
mine: one undifferentiated blob.
```
Also `ممرضة ‖ ممرضات ‖ … معلمون ‖ معلم ‖` (@563/@566), `معلمتي ‖ الغصن ‖ ألعاب ‖ الباحة ‖`
(@673–@675), `أبي ‖ الغالي ‖ … مدرستي ‖ الباحة ‖` (@726/@729), `صفي ‖ مدرستي ‖ جدتي ‖ عائلتي ‖`
(@936–@938). **Under an instruction like `أقرأ الكلمات الآتية` / `أصل بين الصورة والكلمة`,
every listed word is a unit.**

**J2. Questions in an exercise are units, even و-joined.**
`…بم تشبه المعلمة الأم ‖ وبم يشبه المعلم الأب ‖` (@417).

**J3. THE BRAKE — a short reading passage is NOT shredded.** My false cuts @140, @355, @863 are
all the same error: two or three short first-person sentences that gold keeps as one unit:
```
gold: …أحب مدرستي وأحافظ على نظافتها وفي السلة أضع الفضلات ‖
gold: ماجد وأنا أحترم معلمتي أنتبه للدرس وأتكلم بصوت واضح ‖
gold: …تستحق الآن أطيب طعام هيا بنا لتناول طعام الغداء ‖
```
**Diagnosis: within one *task block*, the annotator segments at the granularity of the task.**
Word-matching drill → word granularity. Reading passage → passage granularity. Read the
instruction line above the content and let it set the granularity.

**J4. Table rows are units — a header row of N cells is ONE unit, and a term+definition row is
ONE unit.** All twelve of my false cuts in `doc_20dcf69f9905` are this:
```
gold: …رقم البيت المطلوب ‖ <NL> الإجابة ‖ الأول المواقع التي ينتشر فيها التلوث ‖
      <NL> الثالث مصدر قلق الناس ‖ الخامس مصدرين من مصادر التلوث ‖
gold: …العمود الثاني المعنى ‖ الأكاسيد تنتج من اتحاد الأكسجين مع عنصر آخر مثل ‖
gold: …نفايات المواد التي تسبب اضطرابات للكائنات الحية ‖ <NL> سموم مخلفات البيئة أو الفضلات…
gold: …الكلمة المعنى المعجمي المعنى السياقي ‖
```
**Do not cut between a term and its gloss, or between the cells of one row.** Cut between rows.
This is genuinely the *opposite* of the MCQ rule (I3) and of the constitution colon rule (H1),
and telling them apart is my central open problem — see §3.

**J5. A short inline list inside an instruction is NOT split.**
`…ونبحث عن معنى المفردات الآتية البيئة الطقس الأوزون ‖` (@419–@421) — three bare terms after
`الآتية`, all inside the instruction, **no cuts**. Contrast H1 where `الآتية` forces the colon
cut. **Working discriminator: in H the items are on their own lines / are full clauses; in J5
they are three bare words riding inside a single instruction sentence.** Confidence: medium.

**J6. `…توجيهات لإنقاذ الأرض أحددها ‖`** (@559) — instruction + its verb = one unit.

**J7. The `doc_51fccce927eb` @233 anomaly, re-read (batch 2).** Gold: `…ماجد صفي جميل ‖ فسيح
ومرتب ‖ في مكتبة الصف كتب مفيدة وقصص ممتعة ‖`. Three bare adjectives describing one classroom,
and gold puts a boundary between `جميل` and `فسيح ومرتب` — which **no syntactic rule I have
predicts**, and which sits a few lines from @863, where gold *merges* three whole clauses
(`أحب مدرستي وأحافظ على نظافتها وفي السلة أضع الفضلات ‖`). Both make sense only if the annotator
is following the **printed lines of the textbook page** (0e corollary). Practical consequence:
do not build a syntactic rule on either, keep making the J1 word-drill cuts — those are
unambiguous and worth far more points than this residue costs.

---

### K. Verse / poetry
*(the قصيدة inside `doc_20dcf69f9905`)*

**K1. [REVISED batch 5 — the batch-1 wording was wrong and would have been net-negative.]**
**Current statement: in verse, THE PRINTED LINE IS THE UNIT. Do not cut inside a verse line.**
`<NL>` already marks the hemistich in a one-hemistich-per-line layout, so the boundary is handed to
me for free. Gold cut *inside* a verse line only twice in ~25 lines of `doc_20dcf69f9905`
(@143 `حمض سموم غبار ‖ كونت مطرا`, @159 `الأرض تصرخ ‖ من يأتي يساندها` — in both, the split
separates two distinct propositions or speech acts), and **zero times in ~14 lines** of
`doc_0696314023ab`, where I took 6 false cuts and lost 20 F1 doing it. Treat those two as residue,
not as a rule. In verse the short-tail brake is not "suspended" — nothing inside the line is a
candidate at all.

**[SUPERSEDED batch-1 wording]** — *"Each hemistich (شطر) is a unit. `…حمض سموم غبار ‖ كونت مطرا ‖`
(@143); `…الأرض تصرخ ‖ من يأتي يساندها ‖` (@159). I ran the two halves of the line together. In
verse, metrical structure — not syntax — sets the boundary, and the short-tail rule is suspended."*
Induced from two MISSED items without counting the ~23 line-internal junctions gold left alone. My
actual batch-1 *behaviour* (line = unit) was correct and scored 92.6; only the lesson I drew from
it was wrong. This is the case that produced the META-LESSON in §0g-bis.

---

### L. Hadith / isnād
*(`doc_27a114d4e376`)*

**L1. The isnād chain breaks after each transmission verb.**
`حدثنا إسماعيل ‖ قال حدثني مالك بن أنس عن عمه أبي سهيل…` (@1). Asyndetic, and still a cut.

**L2. Narrative و / ف inside the matn cut normally.**
`…يسمع دوي صوته ‖ ولا يفقه ما يقول حتى دنا ‖ فإذا هو يسأل عن الإسلام ‖` (@38/@44).
Note `حتى دنا` stays inside its unit — `حتى` is a tight subordinator, never a boundary.

---

### M. Credits, headings, front matter
*(`doc_cb57e21e8421`, and heading lines throughout)*

**M1. Each credit line is a unit.** `…سيناريو محمد قنديل ‖ رسوم هانئ دراج ‖` (@7).

**M2. `أولا/ثانيا/ثالثا/خامسا + topic label` = its own unit; the body starts fresh.**
`<NL> ثانيا حكم المباراة ‖ وعليه أن يراعي دوره…` (`doc_8b2d52902b9b` @1148);
`<NL> خامسا لجنة التنظيم ‖ منظمو البطولات لهم دور بارز…` (@1252). I glued the label to the body.

**M3. A reference record — biographical stub, entry, caption — is ONE unit (batch 2).**
My FALSE CUT @198 `doc_20dcf69f9905`: gold `…علي القطب ‖ حياته ‖ من مواليد مدينة القدس وسكانها
حاصل على درجة الدكتوراة في العلوم ويعمل في جامعة…`. The heading `حياته` gets its own unit (M2),
and then the whole stub after it — asyndetic clause **plus** a و-clause — stays together. Record
granularity (0e), same as a table row. I cut it into two.

---

## 2. CONNECTIVE TABLE (batch 1 evidence; ★ = added or corrected in batch 2)

| connective | default | notes / brake |
|---|---|---|
| **و + own predication** | **CUT** | the biggest law; see §0 conjunct-weight test |
| و + bare noun/adj | no cut | `والعراق وأستراليا`, `السمع والبصر والشم` |
| **ف (clause-initial)** | **CUT** | near-absolute in classical prose (B1); yields to short-tail (@701) |
| **ثم** | **CUT** | @149, @631, @85, @887 |
| **حيث** | **CUT** | A3, G1 |
| **إذ** | **CUT** | B4, G1 |
| **كما / كما أن** | **CUT** | B5, C2, H3 |
| **بينما / في حين** | **CUT** | C2, G1 |
| **لكن / ولكن** | **CUT** | B6, F (@880); absorbed if very short inside dialogue (@1416) |
| **إلا أن / غير أن** | **CUT** | A4, B8; yields to short-tail when the previous unit is ≤5 words |
| **بل** | CUT if ≥~6 words | attaches if short (@229) |
| **إنما** | **CUT** | E2 |
| **وقد** | **CUT** | F4 |
| **أي إن** | **CUT** | C6 |
| **وأخيرا / منها مثلا / وكذلك** | **CUT** | F2, C7, B (@335) |
| ولهذا / لذا | **no cut** | C3; ★ `وبذلك` @1053 cuts — reread as an addressee shift, not a budget |
| بعدما / إلى أن / حتى / لأن / أما إذا / الذي | **no cut** | tight subordinators, never open a unit |
| أما X فـ Y | **no cut** | attaches to its antecedent (F5) |
| asyndeton | **genre-dependent** | ★ resolved in 0f: clausal asyndeton CUTS except in a legal article (H4) / reading passage (J3) / table row (J4); sub-clausal asyndeton MERGES unless the items are answer-siblings (I3, J1) |
| ★ **فإذا** | **CUT** | A8; note `أما إذا` above is its opposite |
| ★ **عندئذ / وهنا / وهناك** | **CUT** | C8, G4, @1066, @751 |
| ★ **وإنما** | **CUT** | B12 (= E2 outside literary prose) |
| ★ **وهو / وهي + nominal predicate** | **CUT** | A10 (@386, @991) |
| ★ وهي أن (complementiser) | no cut | A10 (@354) — inside the clause, not a new one |
| ★ **وهو X وهو Y وهو Z** (isocolon chain) | **no cut** | B9 (@152–@166); the cut goes where the frame breaks |
| ★ **أحدهما … والآخر …** | **CUT before أحدهما** | B10 — the announcer/colon cut, in prose |
| ★ **ومعنى X أنه…** | **CUT** | B11 (@692/@702/@710) |
| ★ **وأن / وألا** (obligation) | **CUT** | A12 (@1166), H3 (@6299) |
| ★ بل X **و** Y | no second cut | C5 revised — the correction absorbs its own و-continuation (@229) |
| ★ imperative (any) | **CUT before it** | E6 (@1527, @770, @114, @931, @417) |
| ★ و + evaluative / anaphoric / attributive comment | **no cut** | 0d, C9 — `والبعض يؤرخ…` (@245), `وهذا ما فعلته…` (@951), `وتتواصل…` (@811) |
| ★ asyndetic clause, even 2 words | **CUT** | D3 (@397) — nothing protects a bare asyndetic clause |

---

## 3. OPEN CONTRADICTIONS — status after the batch-2 audit

Note: batch 2 re-served batch 1's grading, so none of these got *new* evidence. What they got was
a careful second reading of the evidence already in hand. Three of the four move; none is closed
by new data.

1. **List-splitting vs row-merging.** MCQ options split to the single word (I3) and workbook
   drills split to the single word (J1), but table rows merge (J4) and inline instruction lists
   merge (J5) and constitutional bare-NP runs merge (H5). Current working discriminator:
   *does each sibling stand alone as an answer/target (→ split), or is it a cell paired with a
   partner cell in the same row (→ merge)?* Needs testing.
   → **[batch 2: PARTIALLY RESOLVED]** The discriminator survives re-reading and now sits inside
   0f as the sub-clausal clause. What changed is the *reason*: this is not two competing rules,
   it is one rule (0e — the genre's native unit) applied to two genres whose native units differ.
   The remaining unknown is not linguistic but typographic: I cannot see the printed lines.
2. **Asyndeton.** Cuts in A6/E1/L1, merges in H4. Working discriminator: *does the second clause
   introduce a new subject/topic (→ cut) or continue predicating about the article's existing
   subject (→ merge)?* Needs testing.
   → **[batch 2: CLOSED — and the batch-1 discriminator was wrong.]** `doc_8b2d52902b9b` @87 cuts
   a verb-initial asyndetic chain; `doc_361233315d24` @9707/@9715 merges one. Identical syntax,
   opposite verdicts, so no syntactic test can separate them. Genre does. Restated as 0f.
3. **`وبذلك` @1053 vs `ولهذا` @187 / `لذا` @1024.** Provisionally resolved by the two-clause
   budget (0c). Low confidence.
   → **[batch 2: STILL OPEN, and its old explanation is withdrawn]** — the two-clause budget is
   gone (0c revised). New provisional reading: `وبذلك` shifts **addressee**, which 0d counts as
   an advance. Medium confidence. This is now my *only* fully unexplained connective.
4. **`doc_51fccce927eb` internal inconsistency:** `معلمي أبي جدي الباحة أمي الكتب` is NOT split
   (@936 context) while `صفي ‖ مدرستي ‖ جدتي ‖ عائلتي ‖` on the very next line IS. I currently
   read this as annotator noise. Do not build a rule on it; do not let it scare me off J1.
   → **[batch 2: REFRAMED, not noise]** Same cause as @233 and H5: the source's printed lines
   (0e corollary). An unobservable variable, not a random one. Behaviour is unchanged — do not
   build a rule, do not get timid about J1 — but I should stop calling it noise, because
   "unobservable" tells me to look for parallelism and announcers, whereas "noise" told me to
   look for nothing.

**5. [NEW, batch 2] Where does the comment/advance test stop?** 0d works on every prose false cut
in the file, but it is a *semantic* judgement and I have no worked case where it and the old
word-count brake disagree in the CUT direction — @245 is the only case where they disagree at
all, and it went the ATTACH way. Risk: I now have a brake that can rationalise absorbing almost
anything, which is exactly the wrong tool for a judge whose prior is 7:1 toward over-merging.
**Mitigation, binding for next batch: 0d may only ever be used to *stop* a cut I was already
inclined to make, never as a reason to skip a boundary I had not considered.** The default at
every و / ف stays CUT (checklist step 4).

---

## 4. CHECKLIST v1 — written after batch 1

**Superseded by v2 in §5; kept intact.** Steps 1–4 and 6–8 are still correct. Step 5 is the one
that was wrong (word count + "take the earlier one"), and it is the one that cost me the most.

Run this in order on every document.

1. **Classify the genre before reading for content.** Prose / list / table / MCQ / drill / verse
   / legal / isnād. The wrong granularity costs 50 F1 points (`doc_25ed373f8aeb`: 33.3).
2. **Scan for cataphoric announcers and dangling stems** (`الآتية، ما يلي، بالآتي، ومنها، وتشمل`,
   or a trailing `في / هو / هي / ل / من / بتاريخ`). Make the colon cut at every one. Free points.
3. **Scan for runs of 3+ parallel siblings with no connective.** That is a list. Cut before each.
4. **At every و and every ف, apply the conjunct-weight test** (§0). Default is CUT. I must
   actively justify *not* cutting, not the reverse — my prior is 7:1 wrong in the merge direction.
5. **Before committing a cut in running prose, count the tail.** ≤5 words, non-parallel,
   dependent → absorb it (0b). Then check whether a *stronger* boundary sits 1–5 words earlier;
   if so, take the earlier one instead (B8, D2).
6. **Never cut after** بعدما / إلى أن / حتى / لأن / أما إذا, and never cut inside `أما X فـ Y`.
7. **Headings, ordinals, speaker names, credit lines, hemistiches:** always their own unit.
8. **In legal text, no connective = no cut**, however long the article runs.

---

## 5. CHECKLIST v2 — for the next batch (batch 2 revision)

Two passes, in this order. The first pass is about *granularity* and is worth more points than
everything else combined; the second is about *connectives*.

**PASS 1 — granularity (0e). Do this before reading a single sentence for meaning.**

1. **Name the genre and name its unit.** Article / stem+option / printed word / passage / row /
   hemistich / isnād link / line / clause-chunk. Write it down before segmenting. Getting this
   wrong cost 42 F1 points on one document (`doc_25ed373f8aeb`: 33.3 vs a 75.7 prose floor).
2. **Scan for announcers and dangling stems** — `الآتية / الآتي / بالآتي / ما يلي / فيما يلي /
   على النحو الآتي / ومنها / وتشمل / يتكون … من`, a quantity + `أحدهما`, a narrative
   `وهنا جاءت المفاجأة`, or a trailing `في / هو / هي / ل / من / بتاريخ`. **Cut at every one.**
   One law, four dialects (G4). This is the single most mechanical source of free points in the
   corpus and the largest single block of misses in the file.
3. **Scan for runs of 3+ parallel siblings with no connective.** That is a list — cut before each.
   Then ask 0f's sub-clausal question: are they exchangeable answers/targets (→ split to the
   single word) or complementary cells of one record (→ merge the row)?
4. **Expect an unobservable printed line** in tables, workbooks and statutes. Do not invent
   syntax for the residue; do not let the residue make me timid about step 2.

**PASS 2 — connectives, inside prose only.**

5. **Default at every و and every ف is CUT.** My prior is 7:1 wrong in the merge direction; I
   must actively justify *not* cutting, never the reverse.
6. **Apply the conjunct-weight test** (§0): new predicate *plus its own material* → cut; bare
   noun/adjective/participle → no cut; a chain of identical one-word copular tags → isocolon,
   no cut, and the boundary goes where the frame breaks (B9).
7. **Then, and only then, apply the advance-or-comment test (0d) as a brake.** Does what follows
   open a new event / agent / time-place frame / topic / addressee (→ keep the cut), or does it
   evaluate, restate, conclude, refer back, or answer (→ absorb it)?
   **Binding restriction (§3.5): 0d may only cancel a cut I already decided to make. It is never
   a reason to skip a boundary I had not considered.**
8. **Do not use word count, and do not use "take the earlier candidate".** Both were wrong.
   If 0d genuinely will not decide, *then* fall back on the ≤5-word heuristic, and only for
   connective-headed tails — an asyndetic clause is never absorbed, even at two words (D3).
9. **Never cut after** بعدما / إلى أن / حتى / لأن / أما إذا / الذي, and never inside `أما X فـ Y`
   or `بل X و Y`. **Always cut before** an imperative (E6), `فإذا` (A8), `عندئذ / وهنا / وهناك`
   (C8), `إذ / حيث / كما / بينما / في حين / إلا أن / إنما / وقد / أي إن`, and a definitional
   `ومعنى X` (B11).
10. **Headings, ordinals, speaker names, credit lines, hemistiches, isnād links:** always their
    own unit. **In legal text, no connective = no cut**, however long the article runs.

---

## 6. WHAT I WOULD NEED TO ACTUALLY IMPROVE FROM HERE

Recording this because batch 2 gave me no new evidence and I do not want that fact to get lost
in the volume of writing above.

The doctrine is now well fitted to these 19 documents — arguably over-fitted. Almost every rule
in it is supported by exactly one document, several by exactly one sentence. The rules I trust
are the ones that recur across genres (the announcer law, و + own predication, ف-initial, the
native-unit law). The ones I distrust are the single-instance brakes: `وبذلك`, `بل X و Y`,
`وهي أن`, C9's vague-quantifier subjects.
**What would move me: a genuinely new batch, especially more exam papers, more tables, and more
workbook pages** — the three genres where my errors are structural rather than local, and where
one misclassification costs more than fifty connective calls.

## 0h STEM-TO-ANNOUNCER (batch 6, F1 98.90) [COMPRESSED: disk full, see note]
Stem = ONE unit up to where the parallel items begin, however long, however many
connectives, even if it contains a quotation. Internal f/w do NOT cut inside a stem.
Find where the N siblings START; a dangling function word is a hint, not the address.
Test: sibling+stem must be well-formed; take the LATEST point where that holds.
Ev: doc_153372a232d6 @216 @290 @325 @353 = I cut inside stems; @300/@304 = cut too
early (at hu); doc_149339985bf9 @316/@319 = too late, sliced item 1. Isocolon
generalises past wa-huwa: @281/@284 three "wa-la aqbal X" clauses = ONE unit.

---

# ===================== BATCH 7 — THE GREAT REVERSAL (mean F1 87.01) =====================
docs: 281bb07b48c9 100.0 | 2f73d2ab2ca3 83.3 | 32d5d0cbc877 83.2 | 35734be96b2b 93.5 | 39e47ea428f8 75.0

## 0i. MY CALIBRATION HAS FLIPPED. I NOW OVER-SEGMENT. [THE most important entry since 0e]

Precision / recall this batch: 73.5/96.2, 72.3/97.9, 90.6/96.7, 61.2/96.8.
**~66 FALSE CUTS vs ~3 MISSED. That is 22:1 in the OVER-segmentation direction.**
Batch 1 said "I under-segment 7:1" and every law in sections A-M was written to push me to cut
more. Those laws worked, then kept firing, and have now overshot badly.

> **THE OLD PRIOR IS DEAD. Do not carry "default at every w/f is CUT" (checklist v2 step 5) into
> another document.** Recall is essentially saturated (96-98%): gold almost never has a boundary
> I failed to find. Every remaining point is in PRECISION. From now on a cut must be EARNED.

This does not delete sections A-M; it demotes them. They describe where boundaries CAN fall, not
where they DO. The gate below now sits in front of all of them.

## 0j. THE TOPIC-SPAN LAW — the unit is bigger than the clause [replaces the conjunct-weight default]

Section 0's conjunct-weight test ("does the conjunct carry its own predication? -> CUT") is the
single biggest source of my false cuts. It is too generous: **a new predication is NOT a new unit.**

> **THE REAL QUESTION AT EVERY w / f / hayth / fi hin: is this still the SAME topic, event,
> definition or description — or a genuinely NEW one? Elaboration of one topic stays in one unit
> no matter how many finite clauses and connectives it contains. Cut only on TOPIC CHANGE:
> a new agent, the next event in a sequence, a new time/place frame, a new subject of discussion.**

The proof, `doc_39e47ea428f8` @468-@490, a FIVE-clause w-chain gold keeps as ONE unit:

    gold: ويعد أوجست كونت الذي صك هذا المصطلح عام 1830 وربط فيه بين الكلمة اللاتينية Socius
          وتعني شعبا أو قبيلة أو مدينة متحالفة مع روما وأصبحت تعني فيما بعد كلمة المجتمع Society
          والكلمة اليونانية Logos وتعني العقل أو المعرفة ‖ وسرعان ما انتشر هذا المصطلح بشكل واسع
          وأصبح يستخدم فعليا في جميع اللغات ‖
    mine: five extra cuts inside the first unit (@468 @474 @483 @490) and one inside the second (@504)

Every w there introduces a full new predication -> section 0 says cut five times. Gold cuts ZERO,
because the whole run is **one topic: the etymology of the word "sociology"**. Then
`وسرعان ما انتشر` changes the topic from etymology to diffusion -> that w DOES cut. Same particle,
six times, one verdict change, decided purely by topic identity.

Same shape at @386 (`ابن خلدون ... 1332 1406 م ولد في تونس وشب فيها وتخرج ...` = one bio stub,
M3 record granularity) and @618-@625 (`... تحت عنوان فصل علم الاجتماع فصل علم الاجتماع المستمر
الأقدم في أمريكا وقسم التاريخ وعلم الاجتماع أسسا في 1891 ‖` = one topic, the Kansas department).

**Reconciling with batch 1, so I do not destroy what was right:** the batch-1 wins were all
places where the parallel members were *different topics* — four different tournament groups
(`ومنتخبات فيتنام ... في المجموعة الثانية`), five different sense organs (`والشم يدرك الروائح`).
Enumeration of DISTINCT items still cuts at every item. Elaboration of ONE item does not.
**Ask "different thing, or same thing described further?" before every connective cut.**

## 0k. SCRIPTURE: THE VERSE IS THE UNIT [new genre, new native unit under 0e]

`doc_32d5d0cbc877` is the Gospel of Luke in Arabic (9:57-10:42). 18 false cuts, 1 missed, 83.2.
Gold's units are **Bible verses** and nothing else. Reconstructed count ~48 units / 685 tokens
~= 14 tokens each ~= the verse count of that passage.

MISSED @210 is the proof, and it is a clean verse boundary:

    gold: ... فاخرجوا إلى شوارعها وقولوا ‖ حتى الغبار الذي لصق بنا من مدينتكم ننفضه لكم
          ولكن اعلموا هذا إنه قد اقترب منكم ملكوت الله ‖

That is Luke 10:10 ending at `وقولوا` and 10:11 running whole. `وقولوا` is a **speech announcer**
(H1) and ends its unit; then the entire quoted verse — including its `ولكن` — is ONE unit.

**Inside a verse NOTHING cuts.** All of these fired and all were wrong:
- **f of consequence** @115 `إن الحصاد كثير ولكن الفعلة قليلون فاطلبوا من رب الحصاد` (B1/F3)
- **imperative** @125 `اذهبوا ها أنا أرسلكم مثل حملان بين ذئاب`, @491 `بالصواب أجبت افعل هذا فتحيا`
  -> **E6 IS WRONG IN SCRIPTURE. An imperative does not open a unit here.**
- **bal** @342 `لا تفرحوا بهذا أن الأرواح تخضع لكم بل افرحوا بالحري أن أسماءكم كتبت في السماوات` (C5)
- **parallel vocatives** @247/@252 `ويل لك يا كورزين ويل لك يا بيت صيدا لأنه لو صنعت ...` (E1)
- **parallel relative clauses** @293/@296 `الذي يسمع منكم يسمع مني والذي يرذلكم يرذلني والذي
  يرذلني يرذل الذي أرسلني ‖` = ONE unit (E1/B2)
- **twin questions** @465 `ما هو مكتوب في الناموس كيف تقرأ ‖` = ONE unit
- **w + new predication** @92 `عين الرب سبعين آخرين أيضا وأرسلهم اثنين اثنين` (A1)

Detect the genre by: يسوع / الرب / ملكوت الله / التلاميذ / ويل لك / biblical place names, and by
units that read as complete self-standing sayings of 12-25 tokens.

## 0l. NARRATIVE: THE EPISODE IS THE UNIT, AND thumma IS ITS OPENER

`doc_2f73d2ab2ca3` (folk tale of the men of Jeddah and the fly) and `doc_35734be96b2b`
(classical akhbar anecdote about al-Khalil ibn Ahmad). 9 and 3 false cuts.

> **In narrative, a sequence marker (thumma / amma / a fronted time phrase) opens a beat, and the
> f- and w-verbs that carry that same beat to its end RIDE INSIDE IT.**

    gold: فوقفت ذبابة على أنف القاضي ‖ ثم وقفت أخرى فوق رأس القاضي فضربوه ضربة على رأسه وقالوا
          وهذه ذبابة ‖ ثم وقفت أخرى فوق ركبة القاضي فضربوه ضربة فوق ركبته وقالوا هذه ذبابة ‖ أما القاضي ...
    mine: cut at every f and every w inside both beats (@154 @158 @167 @171)

One beat = fly lands + they hit him + they say "that is a fly". Three verbs, two connectives, ONE
unit. This vindicates the batch-1 observation at `doc_3272f207e8c9` @631 (`ثم كتبته في عامين
ودفعته إلى المطبعة فصدر عام...`) that I recorded and then never used. Same in the anecdote:
`أرسل والي الأهواز ... يستدعيه ليعلم أولاده وبعث مع الرسول بمال كثير فأبى الخليل وخرج إلى الرسول
بخبز يابس ... في البصرة ‖ وقال له ‖` — four clauses, three connectives, one unit (@16 @21 @23).
Also asyndetic merge @11 `كانوا يعيشون في جدة اشتروا حزمة ملوخية ‖` -> **D3/A9 (asyndetic clause
always cuts) is FALSE in folk narrative when the two clauses are one beat.**

**MISSED @35 `... يسكنه في البصرة ‖ وقال له ‖ <NL>`** — a bare speech announcer with the speech
on the following line is **its own unit**. H1's announcer law, in narrative. Free points.

## 0m. CONNECTIVES DEMOTED THIS BATCH (they are candidates, not triggers)
- **hayth** — A3/G1 said "always CUT". FALSE CUT @161 `كما توسع نطاق الأساليب العلمية الاجتماعية
  حيث يعتمد الباحثون الاجتماعيون على مجموعة متنوعة من التقنيات` -> hayth elaborating the same
  statement is a COMMENT -> attach. This is exactly the batch-5 finding at `doc_0ad7fe19f9bb` @45
  that I filed and failed to generalise. **hayth now defaults to NO CUT unless it opens a new topic.**
- **fi hin** — C2 said CUT. FALSE CUT @343 `فالعالم كان يتحول إلى كل متكامل ومترابط أكثر فأكثر
  في حين أصبحت حياة الأفراد أكثر فردية وانعزالا ‖` -> a matched contrast pair is ONE record (M3 /
  F5's `أما X فـ Y`). Demoted to no-cut when the two halves answer one question.
- **f of explanation** @334 `كرد أكاديمي على تحدي الحداثة فالعالم كان يتحول ...` -> attaches.
- **amma X fa Y** confirmed again: @645 `وأما مرثا فكانت مرتبكة في خدمة كثيرة فوقفت وقالت ...`
  — and note the following f rides too.
- **asyndetic contrast** @67 `... على السياسة الاجتماعية والرعاية الاجتماعية يركز آخرون في المقام
  الأول على تحسين الفهم النظري ...` -> merges. "some do X, others do Y" is ONE record.

## 0n. THE BOUNDARY GOES BEFORE A FRONTED ADVERBIAL, NOT AT THE NEXT w
MISSED @598 + FALSE CUT @601, one error in two halves:

    gold: ... من قبل الفيلسوف الإنجليزي هيربرت سبينسر ‖ في الولايات المتحدة وعلم هذا التخصص باسمه
          للمرة الأولى في جامعة كانساس ...
    mine: ... هيربرت سبينسر في الولايات المتحدة ‖ وعلم هذا التخصص ...

A fronted place/time frame **opens** its unit. I attached it backwards and then cut at the w after
it. Same family as C8 (`عندئذ / وهنا / وهناك`). **When a fronted PP is followed by a w-clause,
the cut goes BEFORE the PP.**

## 0o. LINE-LENGTH RECALIBRATION (measured, batch 7)
median real tokens per printed line, vs my error direction:
  doc_02c1581a0f82 median 2  -> I over-cut (batch 4)
  doc_35734be96b2b median 4  -> 93.5, my best of the prose four
  doc_361233315d24 median 21 -> under-cut (batch 1, legal)
  doc_8b2d52902b9b median 28 -> under-cut (batch 1, encyclopedic)
  doc_2f73d2ab2ca3 median 44 -> OVER-cut
  doc_32d5d0cbc877 median 85 -> OVER-cut
  doc_39e47ea428f8 median 92 -> OVER-cut badly (75.0)
**Long printed lines do NOT license many internal cuts.** I had assumed line length scales with
number of internal boundaries; it does not. A 90-token line is usually one paragraph containing
2-5 units, not 9. Estimated gold unit sizes this batch: ~8.5 (folk tale), ~14 (Luke verses),
~21 (encyclopedia). **Sanity check before submitting: divide tokens by my boundary count. If the
mean unit is under ~8 tokens in a prose document, I am over-cutting.**

---

# ===================== BATCH 8 — THE PERIOD-RECONSTRUCTION FRAME (mean F1 87.37) =====================
docs: 40e700c4f806 66.7 | 4161e96c985e 100.0 | 449f221bdbd2 100.0 | 4ef3bc70a59e 82.2 | 51816712b97c 88.0

**26 FALSE CUTS, ZERO MISSED. Recall was 100.0 on ALL FIVE documents.**
Two batches running, gold has not had a single boundary I failed to find. 0i is confirmed beyond
argument: **my only remaining problem is precision.** Two documents scored 100.0 — both cases where
I cut sparingly. The three failures are all the same failure: I cut at a subordinator.

## 0p. THE UNIFYING THEORY — I AM RECONSTRUCTING DELETED PERIODS

The format note says punctuation was REMOVED. It follows, and batches 7-8 prove it, that **a gold
boundary is simply where a full stop used to be.** That single idea explains every demotion below
and retires most of my connective table:

> **An Arabic orthographic sentence does not end at a subordinator. hayth, baynama, kama,
> illa anna, wa-lakin, ay anna, alladhi, li-anna, hatta all live INSIDE a sentence. They can
> never be a boundary, because the writer did not put a period in front of them.**
> A period falls where a statement is COMPLETE and the next statement STARTS — typically at
> asyndeton, or at a w that opens a genuinely new event/agent/frame.

I had been treating "this connective introduces a clause" as evidence for a boundary. It is
evidence AGAINST one: a subordinate clause proves the sentence is still running.

## 0q. THE MASS DEMOTION — connectives that are now NO CUT by default
`doc_4ef3bc70a59e` (Aleppo Citadel, Wikipedia) is the execution document: 13 false cuts, 0 missed,
and essentially every one is a subordinator I had listed as "CUT".

| connective | old rule | new rule | evidence |
|---|---|---|---|
| **hayth** | A3/G1 "always CUT" | **NO CUT** | 6 merges in doc_4ef3bc70a59e alone (@28 @163 @235 @275 @309 @462), +@108 doc_40e700c4f806, +@161 doc_39e47ea428f8, +@45 doc_0ad7fe19f9bb. **9 merges vs the 1 MISSED that created A3.** |
| **baynama** | C2 CUT | **NO CUT** | @42 doc_4ef3bc70a59e, @23 doc_40e700c4f806 |
| **kama / kama anna** | B5/C2/H3 CUT | **NO CUT** | @117 doc_4ef3bc70a59e `كما أشير إليه في النصوص المسمارية` |
| **illa anna** | A4 CUT | **NO CUT** | @439 `لا يعرف الكثير عن قلعة حلب في عصر صدر الإسلام إلا أن حلب كانت مدينة حدودية` |
| **wa-lakin** | B6 CUT | **NO CUT** | @483 `قصر رائع على ضفاف نهر قويق الذي يمر في المدينة ولكن نقل إلى القلعة` |
| **ay anna** | C6 CUT | **NO CUT** | @17 `منذ 200 مليون سنة أي أنها عاصرت الديناصورات` |
| **wa-dhalika / wa-tilka hiya** | (new) | **NO CUT** | @77/@84 — anaphoric back-reference is a COMMENT (0d) |

**This is the META-LESSON of section 0g-bis collecting its debt.** Every one of these rules was
induced from MISSED items in batch 1 without ever counting how many times gold left the same
particle alone. I flagged A3 as suspect in that very passage and then went on firing it for six
more batches. When a rule rests on one MISSED item, it is a hypothesis, not a law.

`doc_40e700c4f806` (magazine science briefs, 66.7, precision 50.0) shows the scale of the unit
these connectives live inside — ONE gold unit, ~40 tokens, three subordinators:

    gold: أن السلاحف ظهرت على كوكب الأرض منذ 200 مليون سنة أي أنها عاصرت الديناصورات التي انقرضت
          بينما ظلت السلاحف تواصل حياتها ويخرج صغارها من البيض بطريقة تشبه خروج صغار الديناصورات ‖
    mine: four units.

and:

    gold: ... وصل إلى أمريكا اللاتينية منذ عام 1956 واصل حوادثه العدوانية حيث هاجم مدرسة مكسيكية
          وأصاب العشرات من مدرسيها وتلاميذها نقل 12 منهم إلى المستشفى ‖

Note the last clause `نقل 12 منهم إلى المستشفى` is **asyndetic and still merges** — more evidence
against D3/A9 as universal laws (0l already dented them). Asyndeton merges when it continues the
same event; it cuts when it starts a new one.

## 0r. WHAT STILL DOES CUT — so I do not over-correct into silence
In the same Aleppo document gold DOES cut at:
- `... حيث ارتقت فيما بعد إلى نهضة سياسية واقتصادية ‖ وقام ...` — w + **new agent, new action**
- `... عرضها مترين ‖ ويظهر شارع معمد يصل إلى تل القلعة ‖` — w + **new subject** (walls -> a street)
- `... كما أشير إليه في النصوص المسمارية في إبلا وماري ‖` — statement complete, next one starts
- `... لفترة قصيرة الأمد ‖ قيل ...` — asyndetic **new event**
and does NOT cut at `وتلاهم الفرس` or `وتحول حلب` — both continue the same succession/instability
topic. **Same particle w, both verdicts, decided by 0j topic-change. The test is sound; it was my
subordinator rules that were wrong.**

## 0s. I BROKE THE LINE LAW AGAIN — song refrains (`doc_51816712b97c`, 88.0)
Six false cuts, all the SAME refrain line repeated six times:

    gold: <NL> لعب وإثارة في الملعب فانتظروا الشارة ‖ <NL>
    mine: <NL> لعب وإثارة في الملعب ‖ فانتظروا الشارة ‖ <NL>

A 5-token verse line, and I fired B1 (clause-initial f) inside it — the identical error to
`doc_0696314023ab` (Batman song, 6 false cuts) and `doc_02c1581a0f82`, both already written up in
0g and K1. Knowing the law did not stop me applying a PASS-2 rule in a PASS-1 document.
> **PROCEDURAL FIX, binding: before segmenting, if median line length is under ~10 tokens, I am
> forbidden from cutting inside a line at all. Decide this from the statistics FIRST, then read.**
> **And: if a line repeats verbatim (a refrain), my treatment of every copy must be identical —
> a repeated line means a repeated error, six times over in this document.**

---

# ===================== BATCH 9 — MASDARS, thumma, AND THE POSITIVE LAW (mean F1 82.75) =====================
docs: 528da1fb596e 74.7 | 59a45e1e029a 100.0 | 5eb0f458c625 83.0 | 5f0d00706642 74.1 | 616eaaa0cc9d 81.9

~60 FALSE CUTS vs 3 MISSED. Recall 100/100/100/90.9/95.6. Same story, third batch running.
`doc_59a45e1e029a` (the chant) scored **100.0** — the one document where I applied the line law
and refused every internal candidate. That is the template.

## 0t. G2 IS DEAD — A MASDAR CHAIN IS SUB-CLAUSAL AND MERGES, HOWEVER LONG

G2 said "coordinated verbal nouns in an enumeration each get a unit". It is wrong, and it was
costing me whole clusters of cuts at a time.

    doc_5eb0f458c625 gold: واستطاع التحكم بأجهزة التكييف لتغيير حرارة الجو ضمنها وإشعال وإطفاء
      الأنوار الجانبية والتعرف على موقعها وتغيير مسارها وذلك أثناء تشغيل الطائرة بنظام الملاحة الآلية ‖
    mine: cuts at @128 @132 @135 — four units where gold has one.

    doc_5f0d00706642 gold: لذلك أرى أنه يجب نشر حملات عن التوعية في التسامح ونشرها بين مشاهير
      الإنترنت لينشروه في صفحاتهم ليصل لأكبر عدد ممكن ونشر إعلانات عنها في كل المواقع ونشرها
      بطريقة خفيفة لا ثقيلة كي يتقرب منها الناس ‖
    mine: cuts at @52 @63 @69 — four units where gold has one.

> **REVISED CONJUNCT-WEIGHT TEST (supersedes section 0's version, and 0j's too):
> the question is not "does the conjunct predicate something" but "does the conjunct contain a
> FINITE VERB of its own". A conjunct headed by a masdar / verbal noun / infinitive
> (نشر، إشعال، التعرف، تغيير، ورفعها، وصنع) is SUB-CLAUSAL — it is still an object of the one
> governing verb — and it MERGES no matter how many members the chain has.**

The same document proves the other half. MISSED @165:
`كي لا تموت مشاعر التسامح بين البشر ‖ وينشر الحب والأمان في المجتمع ‖` — `وينشر` is a **finite**
verb opening a new proposition, and there gold DOES cut. Masdar merges, finite verb may cut.
This also re-explains F1 correctly: `ونهرا يجري عبر صدع` cut because of its finite `يجري`.

## 0u. thumma IS DEMOTED — it continues a repeated action as often as it opens an episode
My table had `ثم = CUT` with four citations. `doc_528da1fb596e` (Nasser biography) merges it three
times:
- @87 `أزيح ... من منصبه في الجيش وتولى رئاسة الوزراء ثم اختير رئيسا للجمهورية في 25 يونيو 1956`
- @613 `ففي سنة 1921 انتقلوا إلى أسيوط ثم انتقلوا سنة 1923 إلى الخطاطبة ‖`
- @627 `التحق عبد الناصر بروضة الأطفال بمحرم بك بالإسكندرية ثم التحق بالمدرسة الابتدائية بالخطاطبة`

> **Discriminator: when thumma repeats the SAME verb / the same kind of act about the same subject
> (انتقلوا ... ثم انتقلوا, التحق ... ثم التحق), it is one topic -> MERGE. When it opens a distinct
> new episode in a story (`ثم وقفت أخرى فوق رأس القاضي`, 0l), it CUTS.**

This does not overturn 0l; it bounds it. Biography = career/childhood spans; folk tale = beats.

## 0v. MORE DEMOTIONS CONFIRMED THIS BATCH
- **wa-qad** (F4 "CUT") — @440 `نسبت لأحد أعضاء جماعة الإخوان المسلمين وقد نفت الجماعة علاقتها
  بالحادثة ‖` merges. (It did cut elsewhere in the same doc when it opened a new topic; so wa-qad
  is now a plain candidate governed by 0j, not a trigger.)
- **wa-huwa + nominal** (A10 "CUT") — @276 `قدم ناصر دستورا جديدا في سنة 1964 وهو العام نفسه الذي
  أصبح فيه رئيسا لحركة عدم الانحياز` merges. A10 now needs a NEW referent to fire.
- **wa-anna complementiser** (A12 "CUT") — @302 `الشعب السوري يعلم أنكم صادقون وأن الإعلام الرسمي
  دجال ‖` merges; `وأن` continuing a verb of knowing/saying is sub-clausal.
- **wa-lakin / lakinnahu** — @321 `استقال ... بسبب هذه الهزيمة ولكنه تراجع عن استقالته` merges;
  @34 `السلامة في أرامكوا عالية لكن مع الأسف تجاهل السلامة ...` merges. (0q confirmed twice more.)
- **hayth** @510, **baynama** @405 — merge again. That is now 11 hayth merges on record.
- **idh** @304 `إذ أثارت اهتمام المسؤولين ولم يحاولوا نفي وجود الثغرات` — the w inside merges too.

## 0w. THE "ONE CLAUSE EARLY" ERROR — I keep cutting at the complementiser instead of the real break
`doc_616eaaa0cc9d` FALSE CUT @302 + MISSED @306 are a single mistake in two halves:

    gold: الشعب السوري يعلم أنكم صادقون وأن الإعلام الرسمي دجال ‖ ويكفيكم هذا ‖
    mine: الشعب السوري يعلم أنكم صادقون ‖ وأن الإعلام الرسمي دجال ويكفيكم هذا ‖

I cut one clause too early and then merged across the real boundary. Identical shape to 0n
(@598/@601) and to B8's @420-vs-@443. **When a w-clause is a complement of the preceding verb,
walk PAST it and look for the next independent clause; the boundary is there.**

## 0x. 0n CONFIRMED — the fronted adverbial opens its unit (second independent instance)
`doc_616eaaa0cc9d` MISSED @170 + FALSE CUT @173, again one error in two halves:

    gold: أخي الكريم إن العين لتدمع على القتلى ‖ في كل يوم كل شهيد يشيعه شهيد ‖
    mine: ... إن العين لتدمع على القتلى في كل يوم ‖ كل شهيد يشيعه شهيد ‖

`في كل يوم` belongs to what FOLLOWS. Also `ففي سنة 1921 انتقلوا` (doc_528da1fb596e) — gold cuts
before `ففي`. **Rule: a fronted time/place PP attaches FORWARD. Never let it end a unit.**

## 0y. NEW GENRE — reader-comment threads (`doc_616eaaa0cc9d`, Al Jazeera comments)
Each `<NL>` separates one reader's comment; inside a comment the register is loose speech and the
unit is COARSE — typically the whole comment is 1-3 units. Colloquial run-ons, apostrophes to the
editor, and even a trailing `شكرا` stay inside the unit (@66 `وأهم شيء عند الشركات كم يدخل في
حسابها شكرا ‖`). Do not punish loose grammar with boundaries.

## 0z. THE POSITIVE LAW — written because I have now demoted ~12 connectives and must not go mute
Three batches of demotions create a real risk of swinging into under-segmentation (recall already
slipped to 90.9 / 95.6 this batch). So, explicitly, **what still earns a cut:**
1. **A new finite clause that starts a NEW event, agent, or topic** — asyndetic or w-headed.
   `‖ أمر ناصر بعد ذلك`, `‖ قدم ناصر دستورا جديدا`, `‖ تعرض عبد الناصر لعدة محاولات اغتيال`,
   `‖ ويكفيكم هذا`, `‖ وينشر الحب والأمان`.
2. **A fronted time/place frame** — cut BEFORE it (0n/0x).
3. **Every `<NL>` line end** — free, forced, never wrong.
4. **An announcer/dangling stem before its list** (H1, G4, 0k's `وقولوا`).
5. **Distinct enumerated items that are each a separate topic** — different groups, different
   sense organs, MCQ options, drill words. NOT masdar chains (0t), NOT isocolon (B9), NOT glosses.
6. **A heading, ordinal label, speaker name, credit line.**
Everything else is now presumed sentence-internal.

---

# ===================== BATCH 10 — THE BEAT, AND PLACEMENT ERRORS (mean F1 91.70) =====================
docs: 666e6f739000 93.4 | 6806936b7535 80.3 | 690f8f5c21aa 99.1 | 6d2684ec8595 85.7 | 72b8decfdb8c 100.0

**Best batch since the reversal — the demotions are working.** 91.70 vs 82.75 / 87.37 / 87.01.
One 100.0, one 99.1. The remaining damage is concentrated in one colloquial narrative (80.3),
and there my errors run in BOTH directions for the first time (24 FC / 6 MISSED) — which means
in that genre my problem is no longer *how many* cuts but *where* they go.

## 1a. THE BEAT: in first-person / colloquial narrative the unit is EVENT + ITS IMMEDIATE CONSEQUENCE
`doc_6806936b7535` (a young man's account of his engagement) is 2-4 clauses per unit, and the
clauses that belong together are the ones that form one happening:

    gold: ثم أخبرت أهلي بأني سوف أتقدم إليها ‖ فوافقوا وتقدمت إليها ‖
    mine: ثم أخبرت أهلي بأني سوف أتقدم إليها فوافقوا ‖ وتقدمت إليها ‖   (MISSED @266 + FC @267)

    gold: جائني الخبر بالموافقة فلم أصدق ‖ وفرحت ولم أفرح لأنني لم أصدق ‖
    mine: جائني الخبر بالموافقة ‖ فلم أصدق وفرحت ولم أفرح ... ‖          (FC @464 + MISSED @466)

    gold: فصبرت يوما ولم أستطع وصبرت يومين ‖ أريد إجابة ولا إجابة ‖
    mine: فصبرت يوما ولم أستطع ‖ وصبرت يومين أريد إجابة ولا إجابة ‖      (FC @295 + MISSED @297)

Note the two ف-cases point OPPOSITE ways — `فوافقوا` groups forward, `فلم أصدق` groups backward.
So there is no mechanical "the connective opens the unit" rule. **The grouping is semantic: one
beat = one thing that happened plus its immediate result.** News arrives + I disbelieve = one beat.
They agree + I proposed = one beat. I must read the events, not the particles.

## 1b. PIOUS AND FORMULAIC TAGS ATTACH BACKWARD — the boundary falls AFTER them
FALSE CUT @129 + MISSED @131, one error in two halves:

    gold: كنت في يوم أحلم به وأريده أن يكون في يدي والحمد لله ‖ كنا في بداية الطريق وازداد حبنا ‖
    mine: ... أن يكون في يدي ‖ والحمد لله كنا في بداية الطريق ‖ ...

`والحمد لله` is a closing formula on the sentence it follows, not an opener for the next.
Same family as H4's `والله الموفق` (@15997) and `فمدينتنا والحمد لله مدينة آمنة` (@579,
where it is parenthetical mid-clause). **Watch: والحمد لله، سبحان الله، إن شاء الله، ما شاء الله،
بحمد الله. Never start a unit with one.**

## 1c. `hatta anna` CUTS — bare `hatta` does not [refines the connective table]
MISSED @334 `doc_666e6f739000`:

    gold: ... وأحكامه الواعية التي تدل على الحيلة وحدة الذكاء ‖ حتى أن الوزير كالدهار اكتسب عن
          جدارة لقب الوزير الحكيم ‖

My table has `حتى` in the never-cut column (L2: "a tight subordinator"). That is right for bare
`حتى` ("until/so that") but wrong for **`حتى أن` = "so much so that"**, which introduces a fresh
consequential statement. Small rule, cleanly evidenced, and it is one of the very few MISSED items
I have left.

## 1d. PARALLEL DESCRIPTIVE CLAUSES ABOUT ONE ENTITY MERGE — B2 bounded again
FALSE CUT @287: `على وجهه ترك الزمان آثاره بوضوح ‖ فاللحية بيضاء كثيفة والبشرة بيضاء تعكس الصفاء
والحب الذي يحمله في قلبه ‖` — beard and complexion are two parallel predications, exactly B2's
shape (`والبصر يدرك الألوان ‖ والشم يدرك الروائح`), and gold MERGES them.
**Discriminator, consistent with 0j: B2 cuts when the parallel members are DIFFERENT TOPICS
(five separate sense organs, four separate tournament groups). It merges when they are facets of
ONE thing being described (one face, one man, one place).** Description merges; enumeration cuts.

## 1e. A SPEECH TURN IS ONE UNIT — and a reason-enumeration is sub-clausal
- @371/@373 `فقال لي لا تخف أنا معك وسوف أدافع عنك وأضحي بكل شيء لأجلك ‖` — a whole reassurance,
  imperative included, in ONE unit. (E6 dies here too, as it did in scripture, 0k.)
- @421 `قال الملك كلا بل إنني لم أنم طوال الليلة الماضية ‖` — `كلا بل` merges.
- @338 `و أسأل نفسي هل انتهى كل شيء بيننا ‖` — the question rides with its asking verb (E3).
- @68-@77 `لأن بداية الطريق تبدو صعبة نوعا ما أولا لكوني مازلت طالبا ثم لأنني لا أملك وظيفة
  ثم أني غير مستعد أبدا ‖` — an `أولا ... ثم ... ثم` REASON list, all `لأن/لكون` complements =
  sub-clausal (0t), ONE unit. Ordinal markers do NOT make a list of units here; contrast M2, where
  `أولا/ثانيا + topic label` is a HEADING. Heading ordinals cut; reason ordinals do not.

## 1f. RESIDUE I WILL NOT CHASE (`doc_690f8f5c21aa`, 99.1 — my only error)
`فإن مصادر الصراع هنا تكون ‖ ايديولوجية ‖ سكانية ‖ فردية نفسية ‖ جغرافية ‖` — gold merges
`فردية نفسية` and splits the rest. Pure printed-line artefact (H5 / 0e corollary). The announcer
cut before item 1 was right and worth far more than this one merge costs. **Do not get timid
about the announcer cut because of residue like this.**

---

# ===================== BATCH 11 — THE CORRECTION LANDS (mean F1 97.34) =====================
docs: 741bfb3bda19 100.0 | 7825ac72ea71 98.4 | 7aa0f72706ad 93.5 | 7f0f8c191265 100.0 | 7f77a81d1a7a 94.7

**97.34. Two 100.0s. The over-segmentation is corrected.** Trajectory since the reversal:
87.01 -> 87.37 -> 82.75 -> 91.70 -> **97.34**. Total this batch: 6 FALSE CUTS, 4 MISSED — the
first roughly balanced batch of my life.

## 1g. THE ONE GENRE WHERE I NOW UNDER-CUT: mastheads, covers, bibliographic fields
`doc_7aa0f72706ad` (Majid magazine cover) — **precision 100.0, recall 87.9, 4 MISSED, 0 FC.**
The only document in five batches where gold wanted MORE boundaries than I gave.

    gold: <NL> MAJED ‖ NO 836 ‖ March 1 1995 ‖ <NL>
    mine: <NL> MAJED NO 836 March 1 1995 ‖ <NL>

    gold: <NL> عيد الفطر المبارك ‖ وعيد ماجد السعيد ‖ <NL>      (I merged)
    gold: اليوم يبدأ عام جديد ‖ العام السابع عشر ‖              (I merged; MISSED @138)

> **In a masthead / cover / credits / colophon, EVERY FIELD IS ITS OWN UNIT even when several
> share one printed line** — title, issue number, date, price, strapline, each greeting.
> This is M1/M2 extended, and it is the standing exception to the line law: short lines normally
> forbid internal cuts, but a line made of *distinct catalogue fields* splits at every field.

Note @138 `اليوم يبدأ عام جديد ‖ العام السابع عشر ‖` — an appositive RESTATEMENT gets its own unit
here, the opposite of F5 (`نهاية حقيقية نهاية العالم` = one unit in translated literary prose).
Headline register splits appositives; literary prose merges them. Confidence: medium.

## 1h. I BROKE THE VERSE-LINE LAW FOR THE THIRD BATCH RUNNING
`doc_7f77a81d1a7a` (children's poem about alif) FALSE CUT @23:

    gold: <NL> أين تراني أين مكاني ‖ <NL>
    mine: <NL> أين تراني ‖ أين مكاني ‖ <NL>

Two parallel questions inside one 4-token verse line. E1/parallelism fired inside a PASS-1
document — exactly the error of doc_0696314023ab (batch 5), doc_02c1581a0f82 (batch 4),
doc_51816712b97c (batch 8, six times). **Four documents, one error.** The law is written in three
places already; what fails is the procedure, not the knowledge. Hence the hard gate in v3 below.
Also @153/@165: the instruction paragraph following the poem (`في القصيدة ألفات مقصورة نطلب منكم
أن تجدوها ...`) is ONE unit — J6/J3, an instruction block is not shredded.

---

# ===================== CHECKLIST v3 — supersedes v1 and v2 =====================
v2's step 5 ("default at every w and f is CUT") is REVOKED. It was right for batch 1 and has been
wrong for five batches. Run this instead.

**PASS 0 — MEASURE BEFORE READING. Non-negotiable, because knowing the line law has three times
failed to stop me applying it.**
1. Compute median real tokens per printed line.
   - **median < 10 -> LINE MODE. I am FORBIDDEN to cut inside a line.** The answer is the set of
     line ends, full stop. Exceptions, and only these: a masthead/credits line of distinct
     catalogue fields (1g), and a word-drill/MCQ list (J1/I3).
   - median 10-25 -> mixed; expect 1-2 units per line.
   - median > 25 -> prose mode, PASS 1 and 2 apply.
2. If any line repeats verbatim (a refrain), segment every copy IDENTICALLY.

**PASS 1 — NAME THE GENRE AND ITS NATIVE UNIT (0e). Still worth more than everything else.**
   article / verse (=line) / Bible verse (0k) / stem+option / printed word / reading passage /
   table row / record / isnad link / masthead field / narrative beat (0l, 1a) / clause-chunk.
   Scan for announcers and dangling stems and cut at every one (H1, G4, `وقولوا` 0k) — this is
   still the largest block of free points in the corpus and it survived the reversal intact.

**PASS 2 — IN PROSE ONLY. The default is now NO CUT. A cut must be earned.**
3. **A subordinate clause proves the sentence is still running (0p).** Never cut at
   حيث / بينما / كما / إلا أن / ولكن / أي أن / لأن / حتى / الذي / إذ / أما إذا / إلى أن /
   وأن(complementiser) / وهي أن. (`حتى أن` = "so much so that" IS a cut — 1c.)
4. **Never cut a sub-clausal conjunct (0t).** A masdar / verbal-noun chain, a bare noun or
   adjective or participle, a gloss, an appositive without its own finite verb — all merge,
   however many members.
5. **Ask 0j: same topic or new one?** Elaboration, description of one entity (1d), an etymology,
   a bio stub, a definition, an isocolon chain, an anaphoric comment (`وذلك، وتلك هي، وهذا`)
   -> MERGE. A new agent, a new event in a sequence, a new time/place frame, a new subject of
   discussion -> CUT.
6. **Fronted time/place PPs attach FORWARD — cut BEFORE them, never after (0n/0x).**
7. **Pious formulas attach BACKWARD — cut AFTER them (1b).**
8. **In narrative, group into beats: one event plus its immediate consequence (1a).** Do not cut
   between a happening and its result. thumma merges when it repeats the same act (0u).
9. **Speech: the announcer + its quotation is one unit; a long quotation then segments as prose,
   but a single reassurance/answer/curse stays whole (0k, 1e).**
10. **SANITY CHECK BEFORE WRITING.** tokens / boundary count = mean unit size. Prose should land
    ~15-25 tokens. **Under ~8 in prose means I am over-cutting — go back and merge.** Line-mode
    documents are exempt.


# ===================== BATCH 12 — THE SECOND REVERSAL (mean F1 90.87) =====================
docs: 9ad9ba0d12d4 75.7 | 9e0419ed749d 100.0 | 9f9992d71084 85.7 | a17df83313f0 100.0 | a404631139d3 93.0

**37 MISSED vs 2 FALSE CUTS. Precision 96.6 / 100 / 96.8 / 100 / 100. Recall 62.2 / 100 / 76.9 / 100 / 86.8.**

## 1i. MY CALIBRATION HAS FLIPPED BACK. I UNDER-SEGMENT AGAIN. [as important as 0i was]

0i (batch 7) said "the old prior is dead, a cut must be EARNED". That correction ran from batch 7
to batch 11 and it worked (87.0 -> 97.3). **It has now overshot in the other direction.** Precision
is saturated at 96-100; every remaining point is in RECALL. Two documents scored 100.0 and both were
PASS-1 documents (Qur'an by verse, MCQ by option) where the granularity law did all the work. All
three prose/dialogue documents lost points **only** by merging.

> **THE CURRENT PRIOR: in running prose, dialogue and comment threads I MERGE too much.
> Restore the coordinating-particle cuts. Keep the subordinator demotions. They are different things
> and batch 7-11 wrongly collapsed them.**

## 1j. THE DEMOTIONS WERE OVER-GENERALISED — SUBORDINATOR vs COORDINATOR

0p/0q are right about what they actually observed and wrong about how far it reaches. The rule that
survives is narrow:

> **A SUBORDINATOR proves the sentence is still running: hayth, baynama, kama, alladhi, li-anna,
> hatta, idh, ay anna, ila an, amma idha, wa-anna(complementiser). These stay demoted.**
> **A COORDINATOR heading a NEW FINITE VERB is a boundary again: w+verb, f+verb, thumma, lakin,
> li-dhalik, fa-qad. Batch 7-11 demoted these by contagion, not by evidence.**

Evidence, all from this batch, all MISSED (i.e. gold cut and I did not):

| site | gold | the rule I had wrongly demoted |
|---|---|---|
| `doc_9ad9ba0d12d4` @181 | `...يحارب أشياء لا يعرفها ‖ فهو يحارب عزة وكرامة...` | 0m "f of explanation attaches" |
| `doc_9f9992d71084` @32 | `...لا تفوقها فتاة أخرى في جمالها ‖ فقد كانت جميلة حقا...` | same |
| `doc_9f9992d71084` @249 | `...استهزاء واستهانة به ‖ فضحك الحاضرون جميعا...` | 1a "event + immediate consequence = one beat" |
| `doc_9ad9ba0d12d4` @201 | `...قاموس من يكتبون تأييدا له ‖ لذلك فهم يبحثون عن الأوهام ‖` | C3 "li-dhalik attaches to its premise" |
| `doc_9ad9ba0d12d4` @215 / @347, `doc_9f9992d71084` @41 | `...ويغتصب النساء ‖ ولكنه لا يستطيع...` / `الجيش صمام الأمان لمصر ‖ لكن إذا حاول...` / `...كل من رآها ‖ ولكنها كانت متكبرة...` | 0q "wa-lakin NO CUT" |
| `doc_9ad9ba0d12d4` @384 | `...ومختلف الجمعيات ‖ ثم تقع إقامة الحجة...` | 0u "thumma merges" |
| `doc_9f9992d71084` @80 @142 @322 @501 @582 | `...عيبا من العيوب ‖ وأخذت تضحك...`, `...لغطرستها وتكبرها ‖ وأعطت كلا منهم...`, `...وإهانتها لهم ‖ وغضب منها غضبا شديدا...`, `...وأحسنت الغناء ‖ وكان غناؤك جميلا...`, `...ووعدت هذا الوعد ‖ ويجب أن أفي بنذري...` | 0j/1d "elaboration / parallel description of one topic merges" |
| `doc_9f9992d71084` @433 | `...فسمحوا له وأدخلوه ‖ وهو شاب زمار معه مزماره...` | 0v "wa-huwa needs a NEW referent" |
| `doc_9ad9ba0d12d4` @189 | `...وإيمان الشعب السوري ‖ وكل هذه المعاني ليست في قاموسه...` | 0q "anaphoric back-reference is a comment" |
| `doc_9ad9ba0d12d4` @423 | `...من ناحية الأمن ‖ ولولا شعب بوتلاند وحكومته...` | 0j same-topic merge |
| `doc_a404631139d3` @31 | `هذا الموضوع صعب جدا ‖ ولا أدري ماذا أكتب فيه ‖` | line law + 0j |

**Note what the surviving demotions look like in the same documents: NOT ONE of my 2 false cuts is
a subordinator.** hayth/baynama/kama/alladhi never came up as a miss. The demotions were correct
for the particles they were measured on and I extended them to particles they were never measured on.

## 1k. THE UNIT IS SMALLER THAN I THOUGHT — recalibrated sanity band [supersedes v3 step 10]
Measured gold unit sizes this batch:
- Qur'an by ayah: **14.4** tokens (confirmed exactly, 100.0)
- children's story prose (`doc_9f9992d71084`): 613 tokens / 39 gold units = **15.7** — I had aimed at 22
- reader-comment thread (`doc_9ad9ba0d12d4`): 493 / 46 = **10.7** — I produced 17
- comic speech balloons (`doc_a404631139d3`): 169 / 38 = **4.4** — I produced 5.1
- MCQ options: 2.4 (confirmed, 100.0)

> **New band: PROSE 14-17, DIALOGUE/COMMENT 9-12, COMIC/BALLOON 4-5.** v3 step 10's "prose should
> land ~15-25, under ~8 means over-cutting" is revised: **over 20 in prose means UNDER-cutting.**

## 1l. 0y IS WRONG — A COMMENT THREAD IS FINE-GRAINED, NOT COARSE
`doc_9ad9ba0d12d4` (Al Jazeera-style comments, 75.7, recall 62.2) is my worst document since the MCQ
bank. 0y said "the unit is COARSE - typically the whole comment is 1-3 units. Do not punish loose
grammar with boundaries." **That is backwards.** Loose colloquial speech is a chain of SHORT
sentences and gold cuts at every one:

```
gold: <NL> لا حول ولا قوة إلا بالله ‖ يعني مكتوب علينا ما نرتاح من شبح الحرب ‖
      لازم نظل نتذكر ويلات الحرب بين الفينة والأخرى ‖ والله شيء يقهر ‖
      المشاهد التي شفناها رجعتنا 30 سنة للوراء ‖ كانوا عم يقولوا لنا ما تفرحوا كثيرا
      لأن سيجيء لكم الأعظم ‖ طيب فهمنا الرسالة منيح ‖ علم ‖
mine: one cut where gold has seven.
```
`علم` — **one token, its own unit.** A colloquial discourse particle standing alone (`علم`, `طبعا`,
`كفى`, `خلاص`) is a unit. [REVISED 0y: keep only its observation that the comment boundary is the
`<NL>`; delete "coarse, 1-3 units". A 50-token comment is typically 4-6 units.]

## 1m. PIOUS FORMULAS — 1b WAS TOO STRONG. A DUA SENTENCE IS ITS OWN UNIT
1b said "pious formulas attach BACKWARD - never start a unit with one". Three misses kill the strong
form: @130 `لا حول ولا قوة إلا بالله ‖`, @477 `...يا رجل أرواح ‖ حسبي الله ونعم الوكيل ‖`,
@336 `...في الدنيا والآخرة ‖ وفق الله الجميع لما يحبه ويرضاه ‖`. Also @281, where two supplications
that I merged as "one closing prayer block" are **two units**
(`وكان الله في عون أهلنا في الضفة والقطاع ‖ ولعن الله من أيقظ الفتنة وأشعلها لعنة ما بعدها لعنة ‖`).
> **Revised: a formula that is a full optative CLAUSE (لا حول ولا قوة إلا بالله، حسبي الله ونعم
> الوكيل، وفق الله الجميع، اللهم احفظ X، ولعن الله من...) is its own unit. Only a 2-3 token tag
> welded into a running clause (`والحمد لله`, `فمدينتنا والحمد لله مدينة آمنة`, `والله الموفق`)
> attaches.** 1b's own evidence was of the second kind; I generalised it to the first.

## 1n. VOCATIVES ATTACH FORWARD — my only false cut in the comment thread
```
gold: ...ينعمون براحة ويتكلمون بما لا يرون ‖ يا رجل إنها أنفس ليست ألعاب الأطفال الصغار ‖
mine: ...بما لا يرون يا رجل ‖ إنها أنفس ...
```
Same shape as 0n/0x (fronted adverbial) and as `أخي الكريم` in `doc_0ad7fe19f9bb`. **A vocative
opens the unit it belongs to. Never let one end a unit.** (Note the earlier `يا رجل` at @475 sits
*inside* `إنها أرواح أزهقت يا رجل أرواح` — mid-clause, so it does not open anything. Position matters:
utterance-initial vocative opens; mid-clause vocative is parenthetical.)

## 1o. ADDRESS HEADERS: NOUN-HEADED SPLITS, PP-HEADED MERGES [my worst reasoning error this batch]
Three comments open with an address. Gold treats them **differently**, and I forced them to be the
same on a bogus "0s consistency" argument:
- `ردا على الأخ فيصل اليامي هذا الحادث قضاء وقدر ‖` — **merges** (bare PP adjunct)
- `إلى الأخ رقم 1 من هم قبيلة ورسنكلي ‖` — **merges** (bare PP adjunct)
- `كلمة لكل من يتلاعب ويتاجر بأرواح الناس ‖` — **CUTS** (headed by the noun كلمة = a nominal
  sentence "this is a word to...")
> **Discriminator: a header with its own head NOUN is a unit; a bare prepositional address adjunct
> merges into the sentence it introduces.** I had this distinction and argued myself out of it.
> **0s (treat repeated patterns identically) applies to VERBATIM repeats — a refrain, a stem, an
> option list. It is not a licence to flatten three different constructions into one.**

## 1p. QUESTION + ANSWER SPLITS OUTSIDE LITERARY PROSE — C4 bounded
@404 `إلى الأخ رقم 1 من هم قبيلة ورسنكلي ‖ إنهم من أصغر القبائل التي تسكن إقليم بوتلاند ‖`.
C4 ("rhetorical question + its own answer = ONE unit") was induced from one literary essay
(`doc_de12d5da4854` @937). In a comment thread a self-posed question and its answer are **two
sentences**. Keep C4 only where the question is genuinely rhetorical (no answer intended as new
information); a real Q-then-A pair splits.

## 1q. THE wa-lakin PAIR — and why my discriminator was exactly backwards
`doc_9f9992d71084` gives both verdicts in one document:
- @41 **CUT**: `...يعجب بجمالها كل من رآها ‖ ولكنها كانت متكبرة مغشوشة في نفسها...` — SAME subject.
- @611 **MERGE** (my only false cut there): `...وترجوه ألا يزوجها هذا السائل الفقير ولكن أباها لم
  يتأثر ببكائها ورجائها وصمم على تنفيذ ما نذره ‖` — NEW subject.
I had written the opposite rule ("same subject merges, new agent cuts") and it cost me both.
> **Correct reading: lakin CUTS when it opens a new descriptive/topical block (her character, a new
> argument); it MERGES when it is the immediate counter-outcome of the action just narrated —
> she begged BUT he was unmoved — because that pair is one narrative beat.** Subject identity is
> irrelevant. What matters is whether the lakin-clause answers the clause before it (merge) or
> starts a new subject of discussion (cut).

## 1r. COMIC BALLOONS ARE FINE-GRAINED — the line law does NOT protect a speech balloon
`doc_a404631139d3`, 5 MISSED / 0 FC, all inside balloons I refused to touch because the median line
is 6 tokens:
```
gold: أريد من كل منكم أن يكتب في المنزل موضوع إنشاء ‖ عنوانه كيف تندلع الحرب ‖
gold: هذا الموضوع صعب جدا ‖ ولا أدري ماذا أكتب فيه ‖
gold: لا يمكن أن نحاربها ‖ إن بيننا معاهدة صداقة وحسن جوار ‖
gold: <NL> طبعا ‖ ولكن هذا افتراض فقط ‖
gold: <NL> كفى ‖ خلاص ‖ أعتقد أنني فهمت جيدا كيف تندلع الحرب ‖
```
> **LINE MODE (PASS 0) is about TYPOGRAPHIC lines — verse, song, gloss, drill, table. A speech
> balloon or a caption is a TURN, and a turn is segmented by sentence like any prose, down to
> one-word interjections (`طبعا ‖`, `كفى ‖`, `خلاص ‖`).** The four documents that taught me the line
> law (Batman song, the sculpture page, the football refrain, the alif poem) are all verse or gloss.
> Do not apply it to dialogue. My cuts at `كلام فارغ ‖` were right; I simply did not make enough of
> them, and `كفى خلاص` is TWO units, not one.

## 1s. WHAT WENT RIGHT — hold these fixed
- **Qur'an: 100.0.** 1844 tokens, no `<NL>`, identified as Surat al-Nahl, cut at all 128 ayah ends
  and NOWHERE else. Scripture (0k) is now confirmed twice. Detect by: rhyming clause finals
  (-un/-im/-in), no `<NL>`, formulaic refrains (`إن في ذلك لءاية لقوم يتفكرون`), Uthmani spelling
  (السموت، الملئكة، لءاية). **Segment by verse and refuse every internal candidate — including
  fa-idha, imperatives, bal, twin questions, and wa + new predication.**
- **MCQ worksheet: 100.0.** Median line 8 but LINE MODE was correctly overridden by the MCQ
  exception. Stem ends at the dangling function word (`في`, `هم`, `صفات`); every option is a unit;
  a two-token option (`5 صلوات`, `لا يعجبني`, `6 سنوات`) stays whole. 2.4 tokens/unit.

## CHECKLIST v4 — supersedes v3. Only PASS 2 changes; PASS 0 and PASS 1 stand.
**PASS 0** unchanged (measure line length first) **with one correction: LINE MODE applies to
typographic lines only — verse, song, gloss, drill, table, catalogue. A speech balloon, a caption,
a comment or any dialogue turn is NOT protected (1r).**
**PASS 1** unchanged: name the genre and its native unit; scripture -> verse; MCQ -> stem+options;
announcers and dangling stems always cut.
**PASS 2 — prose, dialogue, comment. The default is CUT AGAIN at a coordinator.**
1. **CUT at w / f / thumma / lakin / li-dhalik / fa-qad whenever a NEW FINITE VERB follows** — even
   if it elaborates the same topic, even if it is the consequence of what precedes, even if the
   subject is unchanged (1j). This is the single biggest source of my remaining loss.
2. **DO NOT CUT at a subordinator**: hayth, baynama, kama, alladhi, li-anna, hatta, idh, ay anna,
   ila an, amma idha, wa-anna(complementiser), in (conditional). These stay demoted (0p/0q).
3. **DO NOT CUT inside a masdar / verbal-noun chain or any conjunct without its own finite verb** (0t).
   This survived batch 12 untouched — @305-@322 `لسوء أخلاق ابنته وقلة أدبها وقلة ذوقها...` merged
   correctly, and the cut came at the next finite verb `وغضب`.
4. **Fronted adverbials and vocatives attach FORWARD; cut BEFORE them** (0n/0x/1n).
5. **A full optative/du'a clause is its own unit; only a short welded tag attaches** (1m).
6. **One-word interjections and discourse particles are units** in dialogue and comment (1l/1r).
7. **SANITY CHECK: tokens / boundaries. Prose 14-17, dialogue 9-12, balloons 4-5, scripture ~14,
   MCQ ~2.5. Over 20 in prose means I am UNDER-cutting — go back and split.** (1k)


# ===================== BATCH 13 — 1j BOUNDED BY REGISTER (mean F1 92.33) =====================
docs: a83273e31ee3 100.0 | ab452f741db0 87.0 | ae1331c52539 87.5 | b15bce7039d2 87.2 | b3d76332711b 100.0

**11 FALSE CUTS, 1 MISSED. Recall 100/100 on both essays; precision 77.8 / 77.3.**
Batch 12 said "restore the coordinator cuts" and I restored them everywhere. **The restoration is
right for edited narrative and wrong for unedited expository writing.** Both MCQ banks scored 100.0
using the section-I law unchanged — that law is now confirmed on four documents and I stop worrying
about it.

## 1t. REGISTER DECIDES WHETHER A COORDINATOR CUTS [the missing variable behind 1j]
Two documents, same gold unit size (~16 tokens), opposite verdicts at the same particle:

| | `doc_9f9992d71084` — edited children's book | `doc_ae1331c52539` / `doc_b15bce7039d2` — student essay |
|---|---|---|
| w + new finite verb | **CUTS** (@80 وأخذت، @142 وأعطت، @322 وغضب، @501 وكان، @582 ويجب) | **MERGES** (@39 وبدأ، @21 ودولة الإمارات لم تكمل، @203 وأنا فخورة) |
| explanatory f | **CUTS** (@32 فقد كانت، @249 فضحك) | **MERGES** (@139 فإنها تستقبل) |
| lakin | **CUTS** (@41 ولكنها) | **MERGES** (@38 ولكن ما هو أحدها) |

Twelve of the essay's fourteen gold boundaries are **ASYNDETIC** — a new sentence simply starts with
no connective at all. Only two are w-headed (`وهذا فخر كبير لكل مواطن`, `ودولة الإمارات لن تتوقف هنا`).

> **THE RULE. In EDITED prose (published books, magazines, encyclopedias, translations) a period
> really did stand before many w/f clauses, so restore them — 1j holds. In UNEDITED writing
> (school essays, forum posts, anything with no original punctuation and run-on syntax) the
> annotator segments at the joints that are unmistakable, i.e. ASYNDETON and fronted frames.
> There, DEFAULT TO MERGE at w / f / lakin and cut where the connective disappears.**
> Detect unedited writing by: spelling slips (إنتفاخ، طيبه الله، ب as a separate token), first-person
> intrusions, repetition of the same phrase, no paragraph structure.

I cannot yet predict the two w-cuts the essay does make (`وهذا فخر كبير`, `ودولة الإمارات لن تتوقف`);
both open a **new speaker stance** (this is a source of pride / it will not stop here) after a run of
factual statements. Low confidence, carried forward.

## 1u. A FRAGMENT WITH NO FINITE VERB OF ITS OWN MERGES — EVEN WHEN ASYNDETIC [0t extended]
Four of my five false cuts in `doc_b15bce7039d2` are one 26-token gold unit that I shattered:
```
gold: فهنا يكون لها تأثير سلبي على الفرد وعلى المجتمع كذلك إضاعة الوقت بما هو ليس مفيدا
      فالوقت كالسيف إن لم تقطعه قطعك الهلاك والضياع والضياع الفكري ‖
mine: four units (@199 @205 @211 @215)
```
`إضاعة الوقت بما هو ليس مفيدا` is a **masdar phrase**; `الهلاك والضياع والضياع الفكري` is a **bare NP
list**; neither has a finite verb, so neither can be a sentence, so neither can carry a boundary —
**however asyndetic it looks.** Same at @226: `انتشار النزاعات والجرائم بسرعة فائقة لأن...` is a
masdar phrase and merges backwards into the أما-clause. And the inserted proverb
`فالوقت كالسيف إن لم تقطعه قطعك` merges too: a quoted maxim supporting the claim is not a new
statement.
> **Before cutting at an asyndetic joint, check that what FOLLOWS has a finite verb (or is a full
> nominal sentence with subject + predicate). A masdar phrase, a bare NP, an appositive or an
> adverbial has no boundary in front of it.** This is checklist v4 step 3, which I had, and applied
> only to *conjuncts* — it applies to asyndetic fragments as well.

## 1v. THE SPEECH ANNOUNCER MERGES IN NARRATIVE — 0k's announcer cut is scripture-only
FALSE CUT `doc_ab452f741db0` @46: gold
`تحول الولد إلى طير أخضر فطار فوق الشجرة وقال أنا الطير الأخضر في قبة العسكر ‖`.
I cut after `وقال` on the strength of 0k (`فاخرجوا إلى شوارعها وقولوا ‖`) and 0l (`‖ وقال له ‖ <NL>`).
Both of those are special: in 0k the quotation is a whole Bible verse, in 0l the quotation begins on
the NEXT PRINTED LINE. **When the quotation continues on the same line, E3 governs: the speech verb
keeps the first chunk of what is said.** The announcer/colon cut stays alive for lists, exam stems,
legal `الآتية`, and cross-line speech — not for `قال` mid-narrative.

## 1w. thumma MERGES INSIDE A FOLK-TALE BEAT — 0u confirmed, 0l bounded
FALSE CUT @20: gold `كانت زوجة الأب تكره الولد فأخذته وحبسته في غرفة ثم قالت للأب ابنك مات ‖`.
0l said "thumma opens a beat"; 0u said "thumma merges when it repeats the same kind of act". Here it
introduces a *different* act (she spoke) and still merges, because hating → seizing → imprisoning →
lying to the father is **one scheme, one beat**. The beat is a purpose, not a verb count.
Gold for this tale: 11 units / 92 tokens = **8.4 tokens per unit**, matching 0o's 8.5 for the Jeddah
folk tale. Folk narrative is the one prose genre whose unit is genuinely small.

## 1x. RHYMED CHANT: THE RHYME-LINE IS THE UNIT, AND IT IS SHORTER THAN I GUESSED
MISSED @57 — the bird's song is segmented as verse even though it sits inside a prose line:
```
gold: وقال أنا الطير الأخضر في قبة العسكر ‖ زوجة أبي حبستني وأختي خلصتني ‖ وأبي ما يدري عني ‖
      أختي خبأتني خلف الزير اليماني وأنا صرت طيرا وبنادي ‖
```
I merged `زوجة أبي حبستني وأختي خلصتني وأبي ما يدري عني` as a B9 isocolon. It is not an isocolon,
it is **two rhyme lines** (-tni / -anni). **Inside an embedded chant, look for the RHYME, not for
the syntax: each change of rhyme is a line and each line is a unit.** Units here run 6 / 5 / 4 / 9.

## CHECKLIST v5 — patches v4 only
- v4 step 1 (CUT at w/f/thumma/lakin + new finite verb) now reads: **CUT in edited prose;
  in unedited student/forum writing DEFAULT TO MERGE and cut at asyndeton instead (1t).**
- v4 step 3 gains: **the no-finite-verb test applies to asyndetic fragments too, not just to
  conjuncts (1u).**
- New: **a speech verb keeps the first chunk of its quotation unless the quote starts on a new
  printed line or is a whole scripture verse (1v).**
- New: **in an embedded chant or song, segment by RHYME (1x).**
- Sanity band, updated with measured gold: edited prose 15-16, student essay 16, folk tale 8.4,
  comment thread 10.7, comic balloon 4.4, scripture 14.4, MCQ 2.4 (school) / 4.6 (biology).


# ===================== BATCH 14 — FIVE PERFECT SCORES (mean F1 100.00) =====================
docs: b58385fe50df 100.0 | b8fa69b09c88 100.0 | bd40f62e73c2 100.0 | c2a9fc26a488 100.0 | c626796ad5b3 100.0

**ZERO errors. The mistakes file is empty for the first time.** Every document in this batch was a
PASS-1 document — one where naming the genre and its native unit settles the whole answer and PASS 2
never runs. That is the entire lesson, and it is worth stating plainly:

> **My score is decided almost entirely by whether the document has a native unit I can recognise.
> When it does (scripture, MCQ, isnad, table, drill, verse) I score 100. When it does not (essay,
> comment thread, narrative prose) I score 75-93 and oscillate between over- and under-cutting.**
> 0e was right in batch 2 and it is still the most valuable thing in this file.

## 1y. SCRIPTURE, THIRD AND FOURTH CONFIRMATIONS — and the verse beats the printed sentence
- `doc_b58385fe50df` = **Genesis 4:25-5:32** (Van Dyck), 34 verses, 11.0 tokens each. The genealogy
  repeats one three-verse formula six times (`وعاش X سنة وولد Y ‖ وعاش X بعد ما ولد Y ... وولد بنين
  وبنات ‖ فكانت كل أيام X ... ومات ‖`), so the verse ends are mechanical once the formula is seen.
  Enoch and Lamech take FOUR verses because their paragraphs break the formula
  (`وسار أخنوخ مع الله` twice; the naming of Noah with its quotation).
- `doc_bd40f62e73c2` = **Luke 14:25-15:32** (Van Dyck), 43 verses, 12.9 tokens each — same book and
  translation as `doc_32d5d0cbc877`, the document that produced 0k with 18 false cuts.

**Two decisions inside these that are now gold-confirmed:**
1. **The verse beats the printed sentence.** Gen 4:25 has a full stop before `لأن قايين كان قد قتله`
   in every printed Van Dyck edition, and gold still has no boundary there. **In scripture the unit
   is the verse even when the translation punctuates a sentence inside it.**
2. **A four-token verse is a real unit.** Luke 15:3 `فكلمهم بهذا المثل قائلا ‖` stands alone. Do not
   let a size heuristic merge a short verse into the next one.
3. The prodigal's speech is drafted at 15:18-19 and repeated **verbatim** at 15:21, and the two
   copies are segmented **differently** (the draft splits after `وقدامك`, the delivered version runs
   whole). **This is a hard limit on 0s: identical wording does not mean identical segmentation when
   an external unit — the verse — decides. 0s applies to refrains and option lists, not to scripture.**

## 1z. THE MCQ LAW IS CLOSED — six documents, six 100.0s
`doc_c2a9fc26a488` (physics) and `doc_c626796ad5b3` (primary science) join `doc_a17df83313f0`,
`doc_a83273e31ee3`, `doc_b3d76332711b`. **I2 + I3 + I5 + 0h have now scored 100.0 on five MCQ banks
in two batches. I stop deliberating about this genre.** Consolidated procedure:
1. **Count the options per line and treat that count as a per-document CONSTANT** (3 in the primary
   worksheets, 4 in the physics and biology banks). Any stem boundary that yields the wrong count is
   wrong. This single check resolved every hard line, including two where the option strings are
   *identical* after OCR stripped the slashes (`كولوم ث | كولوم ث | نيوتن كولوم | نيوتن كولوم`;
   `تسال م امبير` twice).
2. **New dialects of the I2 dangling stem collected this batch:** a bare coordinating **و**
   (`تمدنا الشمس بالضوء و ‖ الحرارة ‖ البرودة ‖ الطعام`), a dangling **finite verb**
   (`يبدأ القمر هلالا وينتهي ‖ بدرا ‖ محاق ‖ هلال`), a stranded counted noun
   (`عدد شهور السنة القمرية شهرا ‖ 11 ‖ 12 ‖ 13`), the copulas `هو / هي / هم`, `بسبب`, `عن طريق`,
   `عندما`, and a dangling idafa head (`صفات`, `الغدة`, `شهر`, `جهة`).
3. **A multi-token option stays whole**: `قنطرة ويتستون`, `5 صلوات`, `لا يعجبني`, `10 نيوتن ث ألعلى`,
   `معامل النفاذية المغناطيسية النسبية لمادة`, and OCR-split words (`نسبي ا`) are never cut inside.
4. **Never cut inside a stem** (0h) — the physics bank's 28-token stem
   (`أسقطت كرة ... فوصلت الأرض ... وارتدت رأسيا ... إن دفع الأرض على الكرة بوحدة يساوي ‖`) contains a
   fa- and a wa- clause and both ride inside it.

## 2a. ISNAD CONFIRMED (L1) — and the announcer rule 1v held
`doc_b8fa69b09c88`, one Bukhari hadith, 5 units / 44 tokens. The chain breaks on the **narrator's
name, before the `قال` that opens the next link** (`حدثنا سعيد بن يحيى بن سعيد القرشي ‖ قال حدثنا أبي ‖
قال حدثنا أبو بردة ... عن أبي بردة عن أبي موسى رضى الله عنه ‖`). The `عن`-links do **not** cut —
L1 says the break follows a transmission VERB and `عن` is a preposition. In the matn, 1v (batch 13)
was decisive and correct: `قال قالوا يا رسول الله أي الإسلام أفضل ‖ قال من سلم المسلمون من لسانه ويده ‖`
— the speech verb keeps its quotation, and I did **not** strand a bare one-token `قال` as its own
unit. **1v earns promotion from hypothesis to rule.**


# ===================== BATCH 15 — ONE ERROR IN FIVE DOCUMENTS (mean F1 99.90) =====================
docs: cc92bd9a59d3 100.0 | d1b08b16526d 100.0 | d30b2fb502db 99.5 | d5e46166411c 100.0 | d86f75defe49 100.0

**1 MISSED, 0 FALSE CUTS, across 308 boundaries.** Second all-PASS-1 batch running, and it confirms
1y/1z completely: two more Qur'anic suras, a third Luke passage, and two more MCQ banks.

## 2b. THE OPTION-COUNT CONSTANT IS BINDING — my one error, and it was a self-inflicted one
`doc_d30b2fb502db` @277. Nineteen lines of that bank have **four** options; on line 16 I could only
find three, wrote in my own answer that I was recording the anomaly rather than forcing the constant,
and was wrong:
```
gold: واحدة من التالية يعتبر سلوك صحيح ‖ استخدام أدوات توفير المياه لوقف هدرها ‖
      استخدام السخان الكهربائي بدلا من السخان الشمسي ‖ في تسخين الماء ‖ تمزيق القصص بعد قراءتها ‖
mine: ...بدلا من السخان الشمسي في تسخين الماء ‖ (one long option instead of two)
```
The fourth "option" is a stray PP, `في تسخين الماء` — a fragment the typist split off, semantically
part of the option before it. **Gold segments the LAYOUT, not the sense.**
> **RULE, upgraded from a regularity to a procedure: count the options on every line; the modal
> count is the document's constant; when one line comes up short, go back and split its LONGEST span
> at the most plausible typographic seam (a fronted PP, a trailing qualifier), even when that leaves
> a semantically incomplete option.** I was one span short and I knew it. Do not do that again.

## 2c. SCRIPTURE — the identification procedure, now worth writing down explicitly
Four scripture documents, four 100.0s. The whole method:
1. **Detect.** No `<NL>` at all, or `<NL>` only at paragraph blocks; Uthmani/Van Dyck spelling
   (`السموت، الملئكة، لءاية، ءالهة، حيوة`; `يسوع، الرب، ملكوت الله`); heavy end-rhyme; formulaic
   refrains. **Then identify the actual passage** — this is the whole job.
2. **For the Qur'an, read the RHYME.** `doc_d5e46166411c` is Surat al-Furqan, 77 verses rhyming
   almost throughout in -īrā / -ūrā / -īlā / -āmā. Locating the rhyme word is faster and safer than
   any syntactic reasoning, and it disambiguates repeated verse-finals (six verses end `سبيلا`,
   three `نشورا`, three `كثيرا`, three `كبيرا`).
3. **Check the arithmetic against the identified passage, not a corpus average.** Verse length varies
   a lot by sura: al-Nahl 14.4, al-Furqan 11.6, al-Baqara 261-273 **23.8**, Genesis genealogy 11.0,
   Luke 12.9 (twice, independently). A 24-token unit is normal in al-Baqara and would be a blunder
   in al-Furqan.
4. **Verify the last verse end lands on the last token of the document.** If it does, the alignment
   never drifted. This check has now passed on four documents.
5. **Suppress everything in PASS 2.** The running list of triggers gold ignores inside a verse:
   imperatives (Luke 17:3 has three in one verse; al-Furqan has `قل` four times), `فإذا`, `بل`,
   `ولكن`, `حتى إن`, `لأن`, speech announcers, twin questions, parallel vocatives, parallel relative
   clauses, `وهو الذي` frames (five separate verses in al-Furqan — the verse law and B9's isocolon
   would give opposite answers, and the verse law wins), and `و` + a new finite verb.
6. **Do not reason from repeated wording.** Luke 17:3 and 17:4 both end on the identical two tokens
   `فاغفر له`; Luke 15:18-19 splits a speech that 15:21 keeps whole. **0s does not apply to scripture.**


# ===================== BATCH 16 — 1j BOUNDED BY 0j (mean F1 94.25) =====================
docs: db3824c69464 83.6 | df532033acd3 100.0 | e2e50de1ad8e 89.7 | ea9c5c63ae76 100.0 | ede30da2ee29 97.8

**19 FALSE CUTS, 3 MISSED.** Two more scripture 100.0s (Mark 2-3, Surat al-Mulk). The damage is
entirely in the two Wikipedia articles, and it is all in one direction: **encyclopedic prose is
much COARSER than I have been cutting it.**

## 2d. MEASURED GOLD UNIT SIZE IN ENCYCLOPEDIC PROSE: 19-26 TOKENS, NOT 15-16
- `doc_db3824c69464` (robot): I gave 32 boundaries, 9 were false, 0 missed → gold has **23**.
  603 tokens / 23 = **26.2 tokens per unit.**
- `doc_e2e50de1ad8e` (Manchester City): gold has **36**. 677 / 36 = **18.8.**
The difference between them is style: the Man City article is written in short asyndetic sentences,
the robot article in long chained ones. **So the number to trust is not a genre constant — it is
whether the document's sentences are asyndetic (many boundaries) or w-chained (few).**

## 2e. 1j IS REAL BUT BOUNDED BY 0j — ELABORATION MERGES, NEW EVENT CUTS
Batch 12 restored the coordinator cuts; batch 13 restricted them to edited prose; **this batch
restricts them again, and the restriction is the old topic law (0j).**

> **At a w / f heading a new finite clause, ask 0j FIRST: does the clause open a NEW event, agent,
> time frame or topic (→ CUT), or does it add a further detail about the fact just stated (→ MERGE)?
> Only if it opens something new does 1j fire.**

All nine false cuts in the robot article are elaborations of the fact just stated:
```
gold: الروبوت دخيل دولي أو الربوط أو الروبوط أو الآلي أو العاتول ويمكن أن يسمى بالعربية الإنسان الآلي
      والرجل الآلي والإنسالة لفظ منحوت من إنسان وآلة والجسمال هو آلة مكانيكية قادرة على القيام بأعمال… ‖
mine: three units (@10 @25 @45)
gold: …لا يوافق البعض الآخر على هذا وحجتهم أن تلك الآلات… ‖   (I cut @229)
gold: …علم يهتم ببناء آلات مؤتمتة… ويعرف أيضا بأنه تقاطع لأربعة علوم… ويقصد بها العلم أو المجال… ‖ (I cut @349 @363)
gold: …منها ما يستعمل في القطاع الصناعي وهي تكون عبارة عن أجهزة أوتوماتيكية… ‖   (I cut @384)
gold: …على الحركة والقيادة من تلقاء نفسه ومنها الطائرة بدون طيار… ‖   (I cut @503)
```
and in the Man City article: `وكان مدربي الفريق حينها…` (@97, names the coaches of the period just
described), `ففاز بدوري الدرجة الأولى…` (@119, spells out the golden age just announced),
`واللقب الثاني للسنة كان الدوري الممتاز` (@222), `واللقب كان كأس الاتحاد الإنجليزي` (@424).

**`الأمر الذي` IS a subordinator — never cut before it.** @173 and @463 were both false cuts. I had
this right from 0p and then talked myself out of it because the resulting units were 40 and 46 tokens.
**A 46-token unit is normal in this genre.** The size worry was the error, not the rule.

**`أما X` attaches backwards (F5), even with a fronted PP.** FALSE CUT @438
`…حصل السيتي على المركز الثاني بشق الأنفس أما في الكأس حصل السيتي على المركز الأول ‖`. F5 has now
survived every batch; stop re-testing it.

**Asyndetic elaboration of the noun just mentioned merges.** FALSE CUT @538
`…وصل إلى مباراتين نهائيتين متتاليتين في كأس الاتحاد الإنجليزي الأولى خسرها أمام ايفرتون… ‖` —
`الأولى` resumes `مباراتين` and is a detail of the same fact. This is 1u (no finite verb of its own
in the resumptive head) generalised: **asyndeton cuts when it starts a new event, not when it picks
up the thing just named.**

## 2f. THE FRONTED FRAME, FOURTH INDEPENDENT CONFIRMATION — and I still got one wrong
MISSED @197 + FALSE CUT @200, one error in two halves, the exact shape of 0n and 0x:
```
gold: في عام 2011 تأهل مانشستر سيتي لدوري أبطال أوروبا وفاز بكأس الاتحاد الإنجليزي ‖
      في السنة التالية فاز في الدوري الممتاز أول لقب له في الدوري منذ 44 عاما في الدوري ‖
mine: …وفاز بكأس الاتحاد الإنجليزي في السنة التالية ‖ فاز في الدوري…
```
I applied 0n correctly at eight other sites in the same document and missed it here because
`في السنة التالية` can also be read as an adjunct of the preceding verb. **Procedural fix: at every
bare time/place PP standing between two finite verbs, the default is that it belongs to the SECOND.**

## 2g. 0h's LATEST-WELL-FORMED TEST — I under-applied it twice in one MCQ bank
`doc_ede30da2ee29`, both errors are the same and both are pairs (a false cut one token early plus the
missed cut one token later):
```
gold: تم إنشاء أول أسطول إسلامي في بلاد ‖ الشام ‖ العراق ‖ شمال افريقيا ‖
mine: …أول أسطول إسلامي في ‖ بلاد الشام ‖ العراق ‖ شمال افريقيا ‖
gold: الإنجاز الخالد للأمويين في مدينة القدس هو بناء ‖ المسجد الأقصى ‖ قبة الصخرة ‖ المصلى المرواني ‖
mine: …في مدينة القدس هو ‖ بناء المسجد الأقصى ‖ قبة الصخرة ‖ المصلى المرواني ‖
```
Both times I stopped the stem at the obvious dangling function word (`في`, `هو`) when the stem
actually runs one content word further (`في بلاد`, `هو بناء`). **0h is explicit — take the LATEST
point at which stem+sibling is well formed — and the dangling function word is only a HINT, not the
address. Test the next content word too: does `في بلاد + الشام` work? does `هو بناء + المسجد الأقصى`
work? If yes, the stem extends.** Cost: 4 errors, all avoidable.

## CHECKLIST v6 — the PASS-2 decision order, rewritten after four reversals
The order matters; each step can only *stop* the next.
1. **Is it a subordinator?** (حيث، بينما، كما، الذي، **الأمر الذي**، لأن، حتى، إذ، أي أن، إلى أن،
   قبل أن، عندما، إذا، إن، أما إذا، وأن-complementiser) → **NO CUT, always.**
2. **Is the conjunct sub-clausal?** (masdar chain, bare NP/adj/participle, gloss, appositive, a
   resumptive head like `الأولى` picking up what was just named) → **NO CUT** (0t/1u).
3. **Is it `أما X فـ Y`, `وذلك`, `وهذا`, an anaphoric back-reference, or a pious tag?** → **NO CUT.**
4. **Now 0j: does the clause open a NEW event / agent / time frame / topic?**
   - **No — it elaborates, names, evaluates or spells out the fact just stated → MERGE.** This is the
     step I keep skipping, and it is where 19 of this batch's 22 errors came from.
   - **Yes → CUT** (1j), whatever the particle: و، ف، ثم، لكن، لذلك، فقد.
5. **Fronted time/place PP between two finite verbs → it belongs to the SECOND. Cut BEFORE it** (2f).
6. **Asyndeton:** cuts when a new event starts, merges when it resumes what was just named (2e).
7. **Sanity:** asyndetic-style encyclopedic prose ≈ 19, w-chained encyclopedic prose ≈ 26,
   edited narrative ≈ 15-16, student essay ≈ 16, comment thread ≈ 11, folk tale ≈ 8, comic ≈ 4,
   scripture 11-24 by passage, MCQ 2-5.


# ===================== BATCH 17 — FINAL BATCH OF THE RUN (mean F1 98.02) =====================
docs: f336dc46f626 92.6 | f60bd3a48b58 100.0 | f876524dd98e 100.0 | fe3b2556a241 99.5

**5 errors in 272 boundaries.** Surat al-Tur 100.0 (49 verses, mean 6.4 tokens — the shortest
scripture unit on record, verse 1 is the single word `والطور`), the Islamic-knowledge MCQ bank 100.0.

## 2h. THE FRONTED FRAME COST ME AGAIN — THIRD BATCH RUNNING. MAKE IT MECHANICAL.
`doc_f336dc46f626` MISSED @267 + FALSE CUT @276, one error in two halves, identical in shape to
0n, 0x and 2f:
```
gold: …فترة الحظر البيوريتاني في عام 1660 والذي استمر لمدة 18 عاما ‖
      خلال فترة حكم تشارلز الثاني ملك إنجلترا 1660 1685 قام بعض كتاب المسرحيات… ‖
mine: …لمدة 18 عاما خلال فترة حكم تشارلز الثاني ملك إنجلترا 1660 1685 ‖ قام بعض كتاب…
```
That is now **five independent documents** (`doc_616eaaa0cc9d`, `doc_528da1fb596e`,
`doc_e2e50de1ad8e`, and twice here) where I attached a time/place frame backwards. I apply it
correctly when the frame is long or fronted at a paragraph start, and miss it when the frame *could*
also read as an adjunct of the preceding verb (`خلال فترة حكم…` after `استمر لمدة 18 عاما`;
`في السنة التالية` after `وفاز بكأس الاتحاد`).
> **BINDING PROCEDURE, not a rule to weigh: whenever a bare time/place PP sits between two finite
> verbs, draw the boundary in front of the PP. Do not read it as an adjunct of the verb before it.
> Check every such PP in the document before submitting — this is a mechanical sweep, like the
> announcer sweep in PASS 1.**

## 2i. 2d's COARSE TARGET IS A CEILING, NOT A QUOTA — two missed coordinator cuts
Aiming at 26 tokens/unit made me merge two clauses that do open new facts:
- MISSED @412 `…زكى نفسه بكتابة مسرحيتين كوميديتين ناجحتين ‖ واعترف به كأحد أفراد الدائرة المقربة ‖`
  — w + a new finite verb, and it IS a new fact (he was admitted to the circle), not an elaboration.
- MISSED @191 `…لكونها عمل غير أخلاقي ‖ حيث تم استخدام النسخة الجيدة المنتقاة فتاة القرية… ‖`
  — **`حيث` CUT.** After 11 recorded merges, this is the first `حيث` gold has cut since batch 1.
> **`حيث` is not "never cut". It merges when it explains the clause it hangs on (`حيث يعتمد
> الباحثون…`, `حيث ارتقت…`) and cuts when it introduces a NEW FACT that merely follows on
> (`حيث تم استخدام النسخة المنتقاة…` — a different play, a different period).** Same discriminator
> as everything else in v6 step 4: elaboration merges, new topic cuts. The particle never decides.
> Precision on that document was 96.2 and recall 89.3 — **I aimed at the ceiling and hit under it.**
> Use 2d's number as a check afterwards, never as a target while segmenting.

## 2j. 2b CONFIRMED A SECOND TIME — force the option constant
`doc_fe3b2556a241` FALSE CUT @18: gold reads `…هفاف ‖ سريع ‖ نظيف ‖ رقيق شفاف ‖ عطر ‖` — the two
adjectives `رقيق شفاف` are ONE option, so the line has four like every other vocabulary line. I saw
that the line broke the document's 4-count, wrote down that I was recording the anomaly, and split
anyway. **That is the same mistake as `doc_d30b2fb502db` @277 in batch 15, made in the opposite
direction: there I was one option short and refused to split, here I was one option long and refused
to merge.** In both cases the count was right and my semantics was wrong.
> **The option count is evidence about the LAYOUT and it outranks my reading of the options'
> meaning. When a line's count differs from the document's mode, fix the line.**

---

# ===================== RUN SUMMARY — retrain4 / nopa / juror 1 =====================
29 documents, six batches: **90.87 · 92.33 · 100.00 · 99.90 · 94.25 · 98.02.**
Four documents at 100.0 in one batch, twelve at 100.0 overall.

**What this run actually taught me, in one page:**
1. **Genre identification is still worth more than everything else.** Twelve of my thirteen perfect
   scores were PASS-1 documents (scripture ×6, MCQ ×7, isnad ×1). Every document I lost more than
   5 points on was prose or dialogue.
2. **Scripture is now a solved genre for me** (2c): detect it, identify the passage, read the rhyme
   for the Qur'an, cut at every verse end and nowhere else, and verify that the last verse lands on
   the last token. Six documents, six perfect scores, mean unit 6.4 to 23.8 depending on the sura.
3. **The MCQ bank is a solved genre** (1z, 2b, 2j): stem to the LATEST well-formed point, one unit
   per option, no cut inside a stem or an option, and let the per-document option count arbitrate.
   Seven documents, five perfect, two off by one.
4. **Prose is not solved and the reason is now clear.** My prior flipped twice inside this run —
   under-segmenting in batch 12, over-segmenting in batch 13 and 16 — because I kept looking for the
   answer in the PARTICLE. It is not in the particle. Checklist v6 puts the particle last: rule out
   subordinators, rule out sub-clausal conjuncts, then ask 0j whether a new topic starts. The
   remaining loss is my judgement of "new topic versus elaboration", and that is a real semantic
   judgement that no rule I have written can shortcut.
5. **Placement errors are as expensive as counting errors and much easier to fix.** Every
   fronted-frame miss costs two (a MISSED and a FALSE CUT). The mechanical sweep in 2h would have
   recovered 6 points across this run.
