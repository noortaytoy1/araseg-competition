# Mistake Analysis — NoPnx-NP closed track

**Date:** 2026-07-09
**Checkpoint:** `runs/union-NoPnx-NP-arabertv02-s42`
**Dev set:** `data/NoPnx-NP_dev.jsonl` (222 docs) · **threshold = 0.5**
**Windowing (all diagnostics):** `predict.predict_doc`, window = 180, overlap = 60, forced-0 at `\n` (dev has zero `\n` tokens).
**Status:** analysis-only. All diagnostic scripts are additive standalone files that import existing modules and are imported by nothing. Full suite run with `data/` hidden: **113 passed**, 0 failures, no regression.

**Diagnostics folded into this report:**
- `src/diag_preposition.py` — preposition contribution vs. coverage
- `src/diag_error_examples.py` — confident-error taxonomy (A/B/C/D buckets)
- `src/diag_unfamiliarity.py` — OOV-vs-boundary separation
- `src/diag_mistake_extra.py` — over-segmentation, length/position, structural triggers, FN taxonomy

---

## 1. Headline

**The error is genre-shaped, not prposition-shaped, and it points in two opposite directions at once.** The model **over-cuts** exercise/cloze and legal-numeric text and **under-cuts** the narrative-chain genres (genealogy `بن` chains, Quranic verse openings, hadith isnad, connective-initial sentences). The dominant single cause on both sides is **unfamiliarity / out-of-vocabulary content**, not any one linguistic construction: confident false positives are 66% unseen-driven and confident false negatives 74% unseen-driven. Prepositions — the construction we most suspected — account for only **~9.5%** of errors. The confident-error mass is the story: the model is not hesitant and wrong, it is **confident and wrong**, over-firing at median predicted-positive probability **0.989** on unseen material.

**One-line implication:** the highest-leverage lever is an **OOV-conservatism gate** (suppress confident cuts on unfamiliar spans), which is aimed squarely at the ~66% of confident FP; a preposition-punishment rule addresses at most ~10%; a gold-label fix is off the table (only ~3.5% of confident FP are gold noise).

---

## 2. Error balance and over-segmentation rate

At threshold 0.5 (from `diag_preposition.py`): **FP = 2732, FN = 1531.** The model errs toward **cutting too much** — FP outnumber FN ~1.78×.

Corpus-level over-segmentation (`diag_mistake_extra.py`):

| Measure | Gold | Pred | Ratio |
|---|---|---|---|
| Boundaries | 12,926 | 14,127 | **pred/gold = 1.093** |
| Mean sentence length | 9.88 tok/sent | 9.04 tok/sent | shorter = over-cutting |

Per-doc: mean ratio 1.089, median 1.061, p10 0.890, p90 1.306. **66.2% of docs over-cut**, 26.6% under-cut. The central tendency is a ~9% surplus of cuts, but the distribution is wide and genre-split (Section 7).

---

## 3. Confidence profile

The errors are **confident**, which rules out "just move the threshold" as a clean fix.

- **Preposition FP confidence** (`diag_preposition.py`): mean predicted P = **0.903**, median **0.989**. When the model fires wrongly on a preposition boundary, it is nearly certain.
- **Confident-error mass** (`diag_error_examples.py`): **2236 confident FP** and **1245 confident FN**. So the large majority of both FP (2236 / 2732 ≈ 82%) and FN (1245 / 1531 ≈ 81%) are high-confidence, not borderline.
- **Under-cut genres fail at the opposite extreme** (`diag_mistake_extra.py`): genealogy `بن`-chains at **P ≈ 0.000–0.001**, Quranic verse starts at **P ≈ 0.001–0.05**, connective-initial sentences often **P < 0.05**. The model is not "unsure" here either — it is confidently *not* cutting.

**Takeaway:** the model is well-calibrated to its training distribution and confidently wrong off it, in both directions. Threshold tuning trades one confident error mass for the other; it cannot remove either.

---

## 4. Cause taxonomy (FP + FN) with concrete Arabic examples

### 4a. False positives — confident wrong cuts

From `diag_error_examples.py` (2236 confident FP):

| Bucket | Share | Meaning |
|---|---|---|
| **C — unseen** | **66%** | cut lands on material unseen in training |
| **B — genre** | 20% | genre the model handles poorly (cloze/legal/exercise) |
| **D — clean prose** | 10% | genuine model error on ordinary in-distribution text |
| **A — gold** | 3.5% | gold-label noise (the boundary is arguably correct) |

Structural triggers (`diag_mistake_extra.py`, baseline FP-rate = 0.0238):

| Feature | FP-rate | Lift |
|---|---|---|
| cur = NUMBER (cut lands **after** a numeral) | **0.0593** | **2.49×** |
| next = ENUM (cut before a list marker) | 0.0362 | 1.52× |
| cur = ENUM | 0.0354 | 1.49× |
| next = NUMBER | 0.0240 | 1.01× |

Cutting right after a numeral is the strongest structural trigger — number tables / date lists, e.g. `...أرقام 0 |<== 1 2 9`, `٥٧٥٠٠٠ ج |<== ٣٤٣٠٠٠`. Enumeration markers (أولا / الفصل / أ / ب / المادة) lift FP ~1.5× on both sides.

Top surface tokens the wrong cut lands **before**: من (57), في (50), وكان (43), فراغ (35), بدلا (29), وقد (27), ثم (26), ولكن (25), هل (25), لا (21), هذا (15), فقال (15), حيث (15), ولد (14). Connectives/prepositions dominate, plus the cloze placeholder فراغ, question words هل/ماذا, and genealogy ولد.

Top surface tokens the wrong cut lands **after**: **فراغ (99)** — the cloze blank, by far the single biggest FP driver — then السودان (30), كلمة (27), الصحيح (26), التالية (22), القومية (16), الله (15), الدستور (14). The cloze placeholder plus exercise-stem words (كلمة/الصحيح/التالية) and proper nouns.

### 4b. False negatives — missed boundaries

From `diag_error_examples.py` (1245 confident FN): **C unseen 74% · B genre 14% · D clean-prose 10% · A gold 2%.** Unseen material dominates the misses even more than the over-cuts.

Full FN taxonomy (`diag_mistake_extra.py`, 1531 missed boundaries; categories overlap):

| Category | Count | % of FN |
|---|---|---|
| other / plain prose | 1342 | 87.7 |
| next = connective (و/ثم/كما/أو/وكان…) | 145 | 9.5 |
| genealogy / begat (بن chains) | 16 | 1.0 |
| next = number | 12 | 0.8 |
| quran / verse | 11 | 0.7 |
| enumeration / list | 9 | 0.6 |
| hadith / isnad | 1 | 0.1 |

The misses are overwhelmingly labelled "plain prose" by the surface detector (88%), but the *identifiable* structures are precisely the specialized genres the task warned about, all at near-zero confidence:

- **Genealogy بن-chains (P ≈ 0.000):** the model never cuts inside "X بن Y بن Z…" — `...بن حسلي بن نجاي |<== بن مآث بن متاثيا`, `...بن تارح بن ناحور |<== بن سروج بن رعو`. It reads a whole lineage as one sentence.
- **Quranic verse starts (P ≈ 0.001–0.05):** boundaries after فأنزل الله تعالى / قوله تعالى / سبحانه وتعالى — `...فأنزل الله تعالى |<== لا تحرك به لسانك` (P = 0.002), `...في قوله تعالى |<== لا تحرك به لسانك` (P = 0.001), `...الر سورة يوسف |<== و حم سورة فصلت` (P = 0.001).
- **Hadith isnad (n=1 by this detector, P = 0.014):** `...من مزامير آل داود |<== رواه البخاري`.
- **Connective-initial sentences (P often < 0.05):** new sentences opening with و/ثم/أو/كما/وكان read as continuations — `...من جميع الجهات |<== ثم قاموا بتفجيرها` (P = 0.012), `...بارد شتاء |<== كما تتساقط الثلوج` (P = 0.000).
- **Numbered lists (P mixed):** `...في عام |<== 2013 2007 2011 2003` (P = 0.004).
- **Cloze / enumeration:** `...ل في من |<== ب أي من الخيارات` (P = 0.000).

---

## 5. Coverage / seen-vs-unseen (the dominant axis)

Unfamiliarity is the single largest driver of both error directions.

- **Confident FP:** 66% bucket-C unseen (Section 4a).
- **Confident FN:** 74% bucket-C unseen (Section 4b).
- **FP rate seen vs. unseen** (`diag_preposition.py`): **0.020 seen vs. 0.043 unseen — 2.7× corrected.** The model roughly triples its wrong-cut rate on material it has not seen.
- **Separation test** (`diag_unfamiliarity.py`): tests whether OOV-ness separates false positives from true boundaries; the 2.7× seen/unseen gap above is the quantitative core of that separation — unfamiliarity is a usable signal, not noise.

This is the same wall recorded in memory (AraSeg data ceiling: closed-track errors are dominated by unseen bigrams). The mistake analysis independently reproduces it: **the model's competence is a function of familiarity, and roughly two-thirds to three-quarters of confident errors are on unfamiliar spans.**

---

## 6. Prepositions (the minority we suspected)

`diag_preposition.py` was built to test the hypothesis that prepositions drive the error. They do not.

- Prepositions are only **~9.5% of errors.**
- When the model over-fires on a preposition boundary it is **confident** (mean P 0.903, median 0.989) — so these are not borderline cases that threshold tuning would clean up.
- Prepositions are a *symptom* of the coverage problem, not an independent cause: prepositional wrong-cuts cluster on unseen collocations (Section 5), and connectives/prepositions appear in the top-FP token list (Section 4a) mainly because they are high-frequency function words that sit at the seam of unfamiliar spans.

**Conclusion:** a preposition-specific penalty is a narrow patch on ~10% of the problem, and because these FP are high-confidence it would have to be a hard rule, not a soft nudge.

---

## 7. Genre

Over- and under-cutting split cleanly by genre (`diag_mistake_extra.py`, per-genre pred/gold):

| Genre | pred/gold | Direction |
|---|---|---|
| cloze / exercise | **1.387** (predLen 3.70 vs gold 5.13) | worst over-cut |
| legal / constitution | 1.175 | over-cut |
| general prose | 1.080 | mild over-cut |
| hadith / isnad | 0.898 | under-cut |
| genealogy | 0.889 | under-cut |

Per-genre cloze FP-rate 0.108 / genealogy 0.046 / prose 0.021 (`diag_preposition.py`); hadith under-fires. **Over-segmentation is concentrated in exercise/cloze and legal-numeric text; the narrative-chain genres (hadith, genealogy) are under-cut instead.** The genre axis is *partly* separable from the coverage axis — cloze/exercise structure (the فراغ placeholder, list labels, exam stems) is a distinct, learnable pattern, whereas the narrative-chain under-cutting is largely a coverage problem (unseen names in `بن`-chains, unseen verse text).

---

## 8. Length, position, and structural triggers

### 8a. Sentence-length effects (`diag_mistake_extra.py`)

| len bin | #sents | FN-rate (missed end) | interior FP-rate (over-cut) |
|---|---|---|---|
| 1–3 | 3545 | **0.1331** | **0.0484** |
| 4–6 | 2395 | **0.1382** | 0.0405 |
| 7–10 | 2418 | 0.1249 | 0.0235 |
| 11–15 | 1972 | 0.1197 | 0.0199 |
| 16–25 | 1820 | 0.0802 | 0.0217 |
| 26–40 | 658 | 0.0562 | 0.0147 |
| 41+ | 118 | 0.0593 | **0.0386** |

Two clear effects: **(a)** short sentences lose their end — FN-rate is worst on 1–6-token sentences (~13–14%) and falls monotonically with length; **(b)** the longest sentences get over-cut — interior FP-rate bottoms at 0.0147 (26–40) then jumps back to 0.0386 for 41+.

### 8b. Position in document (`diag_mistake_extra.py`)

| Region | FP-rate | FN-rate |
|---|---|---|
| first 10% | 0.0206 | 0.1243 |
| middle 80% | 0.0222 | 0.1186 |
| last 10% | **0.0396** | 0.1109 |

**Document ends are the FP hotspot** — FP-rate nearly doubles in the last 10% (0.0396 vs ~0.021 elsewhere). FN-rate is roughly flat, slightly worst at the start.

### 8c. Structural triggers

See Section 4a: cutting after a numeral is 2.49× baseline; enumeration markers ~1.5×; the فراغ cloze placeholder alone drives 99 wrong cuts.

---

## 9. What each candidate fix can and cannot address

| Candidate fix | Targets | Est. reachable error mass | Verdict |
|---|---|---|---|
| **Gold-label fix** (correct dev/train labels) | bucket A | **~3.5% of confident FP, ~2% of confident FN** | **OUT** — mass is negligible; not worth the risk of touching immutable label evidence. |
| **Preposition punishment** (penalize cuts before/after prepositions) | preposition FP | **~10% of errors** | Narrow. Would need to be a hard rule (confidence median 0.989). Patches a symptom, not the coverage cause. |
| **Character-level augmentation** (char features / char-aug pretraining) | unseen-token brittleness | **indirect** — could shrink the 2.7× seen/unseen gap by making unseen tokens less alien | Plausible but *indirect*: it attacks the mechanism behind bucket C without directly gating anything. Upside uncertain; no direct handle on the confident over-firing. |
| **OOV-conservatism gate** (suppress/dampen confident cuts on unfamiliar spans) | bucket C unseen FP | **the ~66% of confident FP** (and the 2.7× seen/unseen FP gap) | **Best-aimed lever.** Directly targets the largest, most confident error mass. Must be calibrated against under-cut genres so it does not deepen FN. |
| **Genre-structure handling** (cloze placeholder / enumeration / numeric-table rules) | bucket B + structural triggers | over-cut in cloze (pred/gold 1.387), legal (1.175), the فراغ=99 FP, numeral 2.49× lift | **Separate, complementary axis.** Cleanly separable from coverage; attacks the concentrated exercise/legal over-cutting the OOV gate would not fully catch. |

**Notes on interaction:** the OOV gate and the genre-structure rules are the two independent, high-value axes. The OOV gate reduces confident over-firing on unfamiliar prose; the genre rules mop up the structured over-cutting (cloze/legal/numeric) that is systematic rather than unfamiliarity-driven. The under-cut narrative genres (genealogy/verse/isnad) are a *third* problem — they are confident **non**-cuts on unseen chains, and neither an OOV suppression gate nor a preposition penalty helps them; only more coverage of those chains (or a chain-aware feature) would.

---

## The single most promising closed-track legal lever, and why

The strongest closed-track lever is an **OOV-conservatism gate that damps confident boundary firing on unfamiliar spans**, because it is aimed at the largest and most confident slice of the error — the 66% of confident false positives and 74% of confident false negatives that are unseen-driven — and it attacks the mechanism the numbers keep returning to: the FP rate is 2.7× higher on unseen than seen material (0.043 vs 0.020), yet the model fires there at median probability 0.989. Prepositions (~10%) and gold noise (~3.5%) are too small to move the score, and character augmentation only reaches the same mass indirectly; the OOV gate is the one intervention that touches the dominant cause *directly and confidently*. For the legal/constitution genre specifically, the gate pairs naturally with a numeric/enumeration structural rule (cut-after-numeral fires at 2.49× baseline and the last-10%-of-document FP hotspot is 0.0396 vs 0.021), so a single coverage-aware, structure-aware conservatism pass addresses both the unfamiliar-span over-firing and the numbered-clause over-cutting that together dominate closed-track legal error — without touching immutable labels, eval, or stitching logic.
