# THE MODEL'S PROBLEMS — the definitive deep document

**Date:** 2026-07-10 · **Track:** AraSeg closed track, mainly **NoPnx-NP** (the hardest variant: no punctuation, no paragraph marks — the model must find sentence ends from words alone)
**Status:** analysis-only. Nothing in `runs/` was touched (read-only); the two new diagnostic scripts (`src/diag_memgap.py`, `src/diag_perdoc.py`) are additive and imported by nothing; everything below was computed on CPU with the GPU untouched.
**Verified:** 2026-07-10 independent spot-check (`src/verify_deepdoc_2026_07_10.py` + a fresh CPU re-run of `src/diag_memgap.py`). All reachable headline numbers re-derived exactly from their primary sources; every Arabic example confirmed present in the dev files. Four small corrections applied in place: wrong-cut median 0.989→0.992 (0.989 is the preposition-only median), "88%"→"90% of tokens aren't boundaries", comma-bucket shares 25%/21%→29% PA / 24% NP, per-arm cloze false-cut range 10.2–10.9%→9.9–10.9%; plus a source-path note for the relocated `np_diag.txt`.

**Plain-words glossary (used throughout):**
- **Token** = one word (or number/symbol) of the text. **Boundary** = a token where a sentence ends. The model's whole job: for each token, say "sentence ends here" (a **cut**) or not.
- **False cut (FP)** = the model cut where the answer key says no. **Missed cut (FN)** = the answer key has a sentence end there and the model didn't cut.
- **Precision (P)** = of the cuts the model made, what share were right. **Recall (R)** = of the real sentence ends, what share the model found. **F1** = the balance of the two (100 = perfect).
- **Macro F1** = compute F1 for each document separately, then take the plain average over documents. This is the official score, so a small terrible document counts as much as a huge good one.
- **Train set** = the 174 documents the model learns from. **Dev set** = the 222 documents used for scoring; the model never trains on them (organizer ruling: dev training is forbidden).
- **P(boundary)**, written **P = 0.99** etc. = the model's own stated probability that a token is a sentence end. **Threshold** = the cutoff (usually 0.5) above which we turn that probability into a cut.
- **Calibration** = whether those probabilities are honest: when the model says "99% sure", is it right 99% of the time?
- **Seed** = the random starting point of training. Same recipe + different seed = a sibling model. If all siblings make the same error, the error is **systematic** (baked into recipe+data); if only one makes it, it's **noise** (lottery).
- **Checkpoint** = one saved trained model. **Bigram** = a pair of adjacent words. **Unseen / OOV** = a word or word-pair that never appears in the 174 training documents.
- **Cloze** = fill-in-the-blank exercise text (the blank is written `فراغ`). **Isnad** = the chain of transmitters at the start of a hadith ("A narrated from B from C…").
- **Noise floor ±0.45** = re-training the identical recipe with different seeds moves dev F1 by up to ~0.45 by pure luck; a change must beat +0.45 to count as real.

**Source tags (every number below carries one):**
- **[MA §n]** = `docs/MISTAKE_ANALYSIS_2026-07-09.md`, section n (checkpoint `runs/union-NoPnx-NP-arabertv02-s42`, dev, threshold 0.5).
- **[MEMGAP]** = `src/diag_memgap.py` run of 2026-07-10 (checkpoint `runs/frontier_battery_2026-07-09_v3/cons_baseline_s42`, 30 train docs vs 30 dev docs, seed 42; output preserved in session task log `b7l74gexn.output`).
- **[CALIB]** = calibration re-derivation from the frozen probability cache `probs/union-NoPnx-NP-arabertv02-s42_dev.npz` + `data/NoPnx-NP_dev.jsonl` (script `calib_from_cache.py`, scratchpad, 2026-07-10; 127,722 dev positions).
- **[PERDOC]** = `src/diag_perdoc.py` run of 2026-07-10 on the frozen prediction files in `runs/prep_battery_2026-07-09` (main arm `cons_baseline_s42`, `pred_dev_t050.csv`; cross-seed cut uses seeds 42/1/2 of three trainer baselines).
- **[PREP-TABLE]** = `runs/prep_battery_2026-07-09/FINAL_TABLE.txt` (the 3-seed A/B battery of targeted fixes).
- **[EXP]** = `docs/EXPERIMENTS.md` (the running experiments ledger, including the June residual-error analysis).
- **[MEM-CEIL]** = memory of record `memory/araseg-data-ceiling.md` (verified corrections and rulings).
- **[LOG *name*]** = the corresponding `runs_*.log` A/B verdict block in the repo root.
- **[_x/np_diag.txt, FA#nn]** = the PA/NP confident-error taxonomy dump (numbered FA examples). NOTE (verifier, 2026-07-10): the `_x/` copy no longer exists — the file survives in the analysis-session scratchpad (`…\3be146eb-364e-439f-b9f7-50d0973ffe61\scratchpad\np_diag.txt`, twin `pa_diag.txt`); every FA# example and the 25.4% bucket share quoted below were re-verified verbatim against that copy and against the dev JSONL files.

Three different single-encoder baselines appear (union-s42, prep-battery cons_baseline_s42 at 83.54, frontier-v3 cons_baseline_s42 sampled at 83.01). All are the same recipe (AraBERTv02 fine-tuned on the 174 docs); their numbers agree within the seed-noise floor, and where a number depends on the checkpoint it is tagged. The **deployed submission is an ensemble** (several models voting) and scores higher — NoPnx-NP ≈ 85.0, NoPnx-PA ≈ 87.3, NP ≈ 92.9–93.4, PA ≈ 94.4–95.2 [MEM-CEIL] — but all error analysis below is on single encoders, the "science instrument".

---

## I. THE ONE-PARAGRAPH TRUTH

The model learned its 174 training documents almost by heart — on documents it trained on it scores **99.02 F1**, on documents it has never seen it scores **83.01**, a gap of **+16.02 points** [MEMGAP] — and it carried the confidence of that memorized world into the real one: on dev it still pushes **97.9% of its answers to the extremes** (probability above 0.9 or below 0.1) [CALIB], so when it is wrong it is almost never hesitant, it is *certain* (its median stated probability on a wrong cut is **0.992** [CALIB re-check 2026-07-10 over all 2,732 wrong cuts; the 0.989 in MA §3 is the median for wrong cuts at prepositions specifically]). What it is wrong *about* is mostly **material it never saw**: roughly two-thirds to three-quarters of its confident errors sit on words or word-pairs absent from the 174 training documents (66% of confident false cuts, 74% of confident misses [MA §4]; 88–89% of NoPnx misses sit on never-seen word pairs [EXP, residual analysis]). The failure is **systematic, not luck**: about **80% of one model's errors are repeated by all three sibling models** trained with different random seeds [PERDOC]. It leans toward cutting too often (**9% more cuts than the answer key**, false cuts outnumber misses 1.78 to 1 [MA §2]), worst in fill-in-the-blank exercises and legal lists, while it *under*-cuts chain-style text (genealogies, scripture verses, hadith chains, sentences that start with "and/then"). The score is not hostage to a few disaster documents — the worst 15 of 222 documents carry only 14.8% of the missing points [PERDOC] — the deficit is spread across ordinary documents, which is exactly what a data-coverage wall looks like. Nothing tried so far has beaten the ±0.45 luck floor, because almost every fix attacked the model, and the problem is mostly what the model was given to learn from.

---

## II. THE COMPLETE PROBLEM LIST — ranked by error mass

Dev at threshold 0.5 (union-s42): **2,732 false cuts + 1,531 missed cuts = 4,263 errors** over 127,722 decisions and 12,926 true boundaries [MA §2, CALIB]. "Error mass" below = the share of those errors a problem accounts for. Confident errors (P ≥ 0.8 for false cuts / P ≤ 0.2 for misses) are 2,236 + 1,245 = **82% of all false cuts and 81% of all misses** [MA §3] — so almost the entire error is high-confidence.

### Problem 1 — The unseen-material wall (the dominant problem)
- **What it is (plain):** the model's competence tracks familiarity. On words and word-pairs that exist in its 174 training documents it is nearly perfect; on material it never saw, it guesses — and guesses confidently. It especially has a reflex: **strange word → cut here**.
- **Numbers:** 66% of confident false cuts and 74% of confident misses sit on unseen material [MA §4a/4b]. Its false-cut rate is 0.020 on seen material vs 0.043 on unseen — **2.7× higher** [MA §5]. In the June residual analysis, **88–89% of all NoPnx misses were on word pairs never seen in training** [EXP]. Important honest nuance: ~95% of *all* word pairs in dev are unseen (174 docs is tiny), and true boundaries live on unfamiliar words just as much as false cuts do — so "unfamiliar" tells you where the model *fires*, not where it fires *wrongly* [MEM-CEIL, killed-gate entry].
- **Real examples:** the poem line `... والريح أناشيد والنهر تجاعيد ياغيمة | CUT | يا أم المطر ...` — the model cuts at P = 1.000 right at the never-seen poetic word `ياغيمة` ("O cloud") [_x/np_diag.txt, FA#17]; `... فإنما أنا أخوكم , | CUT | ففرحوا قائلين الله أكبر` — a confident P = 1.000 cut in front of the unseen verb `ففرحوا` ("so they rejoiced") [_x/np_diag.txt, FA#16].
- **Systematic or noise:** systematic — 65.4% of the union of false cuts are made by *all three* seeds; ~79.5% of any one seed's false cuts are shared by all three [PERDOC]. Identical (±0.5%) across three independently trained baselines.

### Problem 2 — Confident over-cutting (the model's standing bias)
- **What it is (plain):** across the whole corpus the model cuts more often than the answer key — it makes sentences shorter than they should be — and its wrong cuts come with near-total certainty, so no threshold move can filter them out.
- **Numbers:** 14,127 predicted vs 12,926 true boundaries = **1.093, a 9% surplus of cuts**; mean predicted sentence 9.04 tokens vs true 9.88; 66.2% of documents are over-cut [MA §2]. False cuts outnumber misses **1.78:1** (2,732 vs 1,531) [MA §2]. When the model says P ≥ 0.9 ("I'm at least 90% sure this is a boundary"), it averages 99.5% claimed certainty but is right only **84.1%** of the time — a **+15.3-point overconfidence gap** on the fire side. The quiet side is honest: when it says "no boundary here" (P ≤ 0.1) it is right 99.0% of the time (+0.9 gap) [CALIB]. One design note: training weighs a boundary 8× a non-boundary (`pos_weight=8.00`, visible in the battery log `runs_battery_final.log`), a deliberate recall-favoring choice that contributes to the fire-side bias.
- **Real example:** `... ودعا الله النور نهارا والظلمة دعاها ليلا |FALSE-CUT| وكان مساء |FALSE-CUT| وكان صباح يوما واحدا |correct|` — Genesis text, two extra cuts inside one verse [PERDOC, burst examples].
- **Systematic or noise:** systematic — the 1.087–1.093 over-cut ratio reproduces across checkpoints [MA §2 vs PERDOC cut 4].

### Problem 3 — Fill-in-the-blank / exercise text gets shredded (worst genre)
- **What it is (plain):** school-exercise documents (multiple-choice questions, blanks written as `فراغ`) have a rhythm the model has almost never seen (the genre appears 2× in train but 142× across 5 dev docs [EXP]). It cuts them to pieces.
- **Numbers:** cloze/exercise pred-vs-gold cut ratio **1.387** (predicted sentence length 3.70 tokens vs true 5.13) [MA §7]; false-cut rate 10.8% of positions vs 2.1% in ordinary prose [MA §7]; the single word `فراغ` (the printed blank) sits directly before **99 false cuts** — the biggest single token trigger in the corpus [MA §4a]. Of the model's *ultra*-confident cuts (P ≥ 0.99) inside this genre, **44.0% are wrong** [CALIB]. Fixing these 4 documents perfectly would be worth **+0.54 macro F1** (84.53 → 85.07 dev, June measurement) [EXP].
- **Real example:** `... الصحيحة بدلا من [ فراغ ] ؟ | CUT P=1.000 | [ فراغ ] . العطلة قبل ...` — the model treats every blank as a sentence end [_x/np_diag.txt, FA#09].
- **Systematic or noise:** systematic (all seeds, all baselines; the per-genre false-cut rate is ~9.9–10.9% for every arm in the battery [PREP-TABLE]).

### Problem 4 — Legal/constitution lists get over-cut
- **What it is (plain):** the one legal document in dev (a constitution) is a giant list of clauses. The model inserts extra cuts inside enumerated items. Note the scale: this *single* document contains **1,161 of the 12,926 dev boundaries (9.0%)** — but under macro scoring it still counts as just 1 of 222 documents.
- **Numbers:** legal pred/gold ratio **1.175** [MA §7]; of ultra-confident cuts (P ≥ 0.99) in legal text, **17.3% are wrong** [CALIB]; a cut landing right after a numeral fires at **2.49× the baseline false-cut rate**, and list markers (أولا / المادة / أ / ب) lift it ~1.5× on either side [MA §4a/8c].
- **Real example:** `... المبادئ والمعايير القومية والإجراءات المدنية والجنائية ؛ | CUT P=0.995 | أراضي الولاية ومواردها الطبيعية ؛ ...` — semicolon-separated clause list, extra cut mid-list [_x/np_diag.txt, FA#12].
- **Systematic or noise:** systematic across seeds [PREP-TABLE per-genre block]; but it is one document, so its contribution to the *macro* score is capped.

### Problem 5 — Chain-style text: the model doesn't know where lineages, verses and isnads end
- **What it is (plain):** genealogies ("X begat Y, Y begat Z…"), scripture verse sequences, and hadith transmitter chains have boundary conventions the model never learned. It fails them badly under *every* seed — but the *direction* of failure flips between checkpoints (one model reads a whole lineage as one endless sentence; a sibling cuts it at every name).
- **Numbers:** the union checkpoint under-cuts genealogy (ratio 0.889) and misses **17 of the 27** gold boundaries in the genealogy document, with a median stated probability of just **0.078** at true boundaries [MA §7, CALIB]. The prep-battery sibling *over*-cuts the same document (ratio **1.852**, F1 59.7) [PERDOC cut 4 + worst-15]. Hadith/isnad is under-cut by both (0.898 / 0.859) [MA §7, PERDOC]. Quranic verse starts are missed at P ≈ 0.001–0.05 [MA §4b].
- **Real examples:** `... بن حسلي بن نجاي | MISSED, P≈0.000 | بن مآث بن متاثيا ...` — a `بن` ("son of") chain read as one sentence [MA §4b]; `... في قوله تعالى | MISSED, P=0.001 | لا تحرك به لسانك ...` — a verse beginning after "in His saying, exalted be He" not cut [MA §4b].
- **Systematic or noise:** the *failure* is systematic (the genealogy document is in the worst-15 of every arm); the *direction* is checkpoint lottery. Both facts matter: the model has no learned convention here at all.

### Problem 6 — Sentences that start with "and / then / also" are missed
- **What it is (plain):** Arabic freely starts new sentences with connectives (و "and", ثم "then", كما "also", وكان "and it was"). The model reads these as the same sentence continuing, and misses the boundary before them — quietly (very low stated probability).
- **Numbers:** 145 misses (9.5% of all misses) have a connective as the next word [MA §4b]; typical stated probabilities under 0.05. Connectives also dominate the tokens right after false cuts (وكان 43, وقد 27, ثم 26, ولكن 25) [MA §4a] — the model knows connectives matter but has the wrong rule for *which* ones start sentences.
- **Real examples:** `... من جميع الجهات | MISSED, P=0.012 | ثم قاموا بتفجيرها` ("…from all sides. Then they blew it up") [MA §4b]; `... بارد شتاء | MISSED, P=0.000 | كما تتساقط الثلوج` ("…cold in winter. Snow also falls") [MA §4b].
- **Systematic or noise:** systematic (falls inside the 61–65% all-three-seeds error core [PERDOC]).

### Problem 7 — Short sentences lose their endings; the longest get chopped
- **What it is (plain):** the model has internalized "a typical sentence is ~10 words". Very short sentences (1–6 words: headlines, list items, dialogue) get merged into their neighbors; very long ones (41+ words) get cut in the middle.
- **Numbers:** miss-rate of the true ending is **13.3–13.8%** for 1–6-word sentences, falling steadily to ~5.6% for 26–40-word ones; the interior false-cut rate bottoms at 1.5% (26–40 words) then jumps back to **3.9%** for 41+ [MA §8a].
- **Real example:** the phonics-drill fragments `م` `س` `ر` (single-letter "sentences") in doc_02284d965de1 are missed outright [PERDOC burst examples].
- **Systematic or noise:** systematic (monotone pattern across bins, reproduced in both diagnostic checkpoints).

### Problem 8 — Document endings are a false-cut hotspot
- **What it is (plain):** in the last tenth of a document the model fires almost twice as often as elsewhere — closing sections are exactly where lists, references, and dates pile up.
- **Numbers:** false-cut rate 0.0396 in the last 10% of a document vs ~0.021 elsewhere [MA §8b].
- **Systematic or noise:** systematic.

### Problem 9 — Numbers and list markers trigger cuts
- **What it is (plain):** right after a numeral the model expects a sentence end (dates, statistics, article numbers).
- **Numbers:** cut-after-numeral fires at **2.49×** the baseline false-cut rate; list-marker contexts ~1.5× [MA §4a]. Example error: `... في عام | CUT | 2013 2007 2011 2003` — a year list shredded [MA §4b].
- **Systematic or noise:** systematic; overlaps heavily with Problems 3–4.

### Problem 10 — Prepositions (the suspect that was cleared)
- **What it is (plain):** we long suspected cuts before prepositions (من "from", في "in") were the core disease. Measured: they are a symptom.
- **Numbers:** preposition contexts account for only **~9.5% of errors** [MA §6]; when the model wrongly cuts at one it is nearly certain (mean P 0.903, median 0.989) [MA §3]; the prepositions cluster on unseen material, i.e. they are Problem 1 wearing a preposition's clothes [MA §6].
- **Systematic or noise:** systematic but small.

### Problem 11 — Double-cuts: errors arrive in small bursts
- **What it is (plain):** false cuts cluster — two wrong cuts 1–2 tokens apart, typically shredding a vocabulary list or repeated phrase, slightly more than chance would place them.
- **Numbers:** 36.4% of false cuts have another false cut within 5 tokens vs 30.6% expected if spread evenly; **288 double-cut pairs** [PERDOC cut 3].
- **Real example:** a clothing-vocabulary list — `اليابان |CUT| الكيمونو … الإمارات |CUT| الكندورة |CUT| … عمان |CUT| دشداشة` — the model cuts between every country and its garment; the answer key cuts only between entries [PERDOC burst examples].
- **Systematic or noise:** mildly systematic (6-point excess over chance), mostly a surface effect of Problems 3 and 7.

### What is NOT a problem (measured, so we stop re-litigating)
- **The answer key (gold labels):** only **~3.5% of confident false cuts and ~2% of confident misses** are cases where the key itself is arguably wrong [MA §4a/4b]. An earlier "44% of gold is broken" claim was **retracted as fabricated padding** [MEM-CEIL, correction of record]. Do not spend effort auditing gold.
- **Seed lottery:** real but bounded — only ~18.5% of the false-cut union is single-seed noise [PERDOC]; fixing it is what ensembles already do (the deployed ensemble is ~+1.1 to +1.5 over single models [EXP #22, MEM-CEIL]).
- **On the punctuated tracks (PA/NP), a separate, smaller story:** there the biggest "defensible" bucket (A ≈ 25.4% of confident false cuts on NP [_x/np_diag.txt]) is mostly the model disagreeing with the key about whether a **comma** ends a sentence — a learnable annotation convention, not damage [MEM-CEIL]. Clean fixable model error there is only 2.5% (PA) / 4.2% (NP) [MEM-CEIL].

---

## III. WHY IT IS OVERCONFIDENT — the causal chain

**The anchor number: the model scores 99.02 F1 on documents it trained on and 83.01 on documents it never saw — a memorization gap of +16.02 points** [MEMGAP: 30-doc samples each side, same decoding, same threshold; on the train sample it made 26 false cuts and 2 misses in 15,363 decisions].

The chain, step by step, in plain words:

1. **The corpus is tiny, so the model memorizes it.** 174 documents is small enough for a 135-million-parameter encoder to nearly memorize in 8 passes. Result: near-perfect training-world performance (99.02 F1; on train it answers at the extremes 99.5% of the time and its typical cut carries probability **1.000**) [MEMGAP].
2. **The training objective rewards certainty.** Cross-entropy loss keeps pushing every answer toward probability 0 or 1 for as long as the label can be fit — and on a memorized corpus *every* label can be fit, so nothing ever pushes back. (The boundary class is additionally weighted 8:1, tilting the push toward "fire".)
3. **The certainty habit transfers to new text unchanged.** On dev, the model still answers at the extremes **97.9%** of the time — 95.6% of all 127,722 decisions are even beyond 0.99/0.01 [CALIB]. The *habit* generalized perfectly.
4. **The accuracy underneath the certainty did not transfer.** On the fire side, "I'm ≥90% sure this is a boundary" is true only **84.1%** of the time (claimed 99.5% — a **+15.3-point** honesty gap); even "≥99% sure" is true only 87.5% of the time [CALIB]. The quiet side stays honest (99.9% claimed vs 99.0% true) because "no boundary" is the easy default — 90% of tokens aren't boundaries (12,926 boundaries in 127,722 tokens).
5. **So every error arrives wearing a mask of certainty.** The median stated probability on a wrong cut is **0.992** [CALIB re-check 2026-07-10; MA §3's 0.989 is the preposition-cut median]; missed boundaries sit at 0.000–0.05 [MA §4b]. Where the certainty is most misplaced is exactly the unfamiliar material of Problem 1: at P ≥ 0.99 the wrong-cut share is 44% in cloze text and 17% in legal text vs 11% in ordinary prose [CALIB].
6. **Two standard rescues therefore fail, measurably.** (a) *Moving the threshold* cannot separate right cuts from wrong ones — both sit above 0.98; the battery's per-arm best thresholds hover at 0.5–0.6 with no real gain [PREP-TABLE]. (b) *Gating on unfamiliarity* fails because true boundaries are just as unfamiliar as false cuts in a 174-doc world (~95% of all word pairs are unseen): every familiarity signal tried separated wrong cuts from right cuts at ROC-AUC 0.48–0.52 — a coin flip — and a hard unseen-bigram rule killed 97% of false cuts at the cost of 95% of true boundaries [MEM-CEIL, killed-gate entry, `src/diag_unfamiliarity.py`].

**The copyable one-liner:** *trained to certainty on 174 memorized documents (train 99.0 vs dev 83.0, gap +16), the model kept the certainty and lost the correctness — 98% of its answers are extreme, its wrong cuts carry probability 0.99, and neither thresholds nor "is this familiar?" gates can tell its confident errors from its confident successes.*

---

## IV. WHERE THE SCORE ACTUALLY LIVES — the per-document tail

All numbers: [PERDOC], arm cons_baseline_s42, dev, threshold 0.5. Reminder: the official score is **macro** — the plain average of each document's own F1 — so 222 documents each own 1/222 of the score, regardless of size.

- **Distribution:** macro 83.54, median 83.84, worst document 49.28, best 100. Only **1 document below 50** and **13 below 70 (5.9%)**. The bulk (106 docs) sits at 80–90; 55 docs at 90–100; 48 at 70–80.
- **How concentrated is the damage?** The missing mass is 16.46 points (100 − 83.54). The worst 5 docs carry 5.9% of it, the worst 15 (6.8% of documents) carry **14.8%**, the worst 30 carry 26.0%. So the tail hurts about **2× its fair share — but 74% of the deficit lives in the ordinary middle** of the distribution. **The score does not live in a few disaster documents; it lives in a broad, shallow bleed across normal prose.** That is the per-document signature of a coverage wall, and it means no per-document rescue (finding and fixing "the bad docs") can recover more than a sliver.
- **A boundary-mass footnote:** the single legal document holds 9.0% of all *boundaries* (1,161/12,926) but only 1/222 of the *score* — boundary mass and score mass are different currencies here; macro protects us from the legal doc and equally caps what fixing it can pay.

**The worst 15 documents and what they share** [PERDOC cut 1, worst-15 table with text snippets]:

| Doc (F1) | What it is |
|---|---|
| doc_8501aa69ca44 (49.3) | children's song/chant, `مدينة النخيل` repeated with nonsense stretch `نخييييييييل` — repetitive, tiny "sentences" |
| doc_0301b9267980 (56.8) | **Genesis** (سفر التكوين) — scripture verse chains, و-connected |
| doc_49e7cd24ac91 (57.1) | **Exodus** priestly-garment passage — scripture, long و-chains |
| doc_d93fbe520d5c (59.7) | **Matthew 1 genealogy** (إنجيل متى… فلان ولد فلان) — begat-chain |
| doc_31ec2f106148 (61.5) | Quranic-grammar exam (parsing questions) — cloze/exercise |
| doc_86fa63ebacef (61.8) | vocabulary MCQ drill with `فراغ` blanks — cloze/exercise |
| doc_82d8017da35c (63.2) | hadith (Ibn Shihab / revelation-pause narration) — isnad chain |
| doc_111cbfecd397 (66.7) | **Luke 2** (Jesus at twelve) — scripture |
| doc_aaec186baef1 (67.6) | **Quran, Surat al-Ma'arij** in study layout — verse boundaries (recall 52.3: misses) |
| doc_7c7f7a114505 (68.6) | *Majid* children's magazine masthead — dates, prices, headings |
| doc_1a87c3256ea8 (69.0) | Kuwaiti TV interview transcript — spoken, name-lists |
| doc_710c373430cb (69.6) | **Exodus 18** (Jethro) — scripture |
| doc_9d090e13673e (69.6) | **Matthew 2** (return from Egypt) — scripture |
| doc_02b21fa0f702 (70.0) | beauty-tips listicle with `ɵ` bullet glyphs — list items |
| doc_7335b1d047dd (70.6) | **Genesis 33** (Jacob at Succoth) — scripture |

What they share, in plain words: **seven of the worst fifteen are Bible passages** (plus one Quran layout and one hadith) — chain-style scripture whose sentences are verses linked by "و/then/begat", a convention absent from the 174 training documents; two are exam/cloze drills; the rest are children's/list/transcript formats with very short, repetitive "sentences". That is Problems 5, 3, and 7 made flesh: **unseen genre conventions + chain or fragment sentence shapes.** Not one of the worst 15 is ordinary essay prose.

---

## V. WHAT THIS MEANS — problem by problem: what was tried, what happened, what could still work

Ground rules first, because they bound everything: the closed track = **the 174 training documents only** — no generated text, no external text (ruling of record, 2026-07-09) [MEM-CEIL]; **dev training is forbidden** (organizer, explicit, 2026-07-10) [MEM-CEIL]; a fix is real only if it beats the **±0.45** seed-noise floor over 3 seeds. And the sobering context: we stand ~#4; three teams beat us on the *same* 174 documents [MEM-CEIL] — so a real lever exists that we have not found.

**Problem 1 — unseen-material wall.**
- *Tried:* character-level noise augmentation to break the "strange word → cut" reflex: **failed** (char010 −0.03, char025 +0.12, both far under +0.45; and per-genre false cuts got *worse* — ordinary-prose FP rate rose from 2.10% to 2.60%) [PREP-TABLE]. Connective-focused augmentation: **failed** (−0.26; the connective over-fire count actually rose 280→296) [LOG connaug_fix]. Self-generated in-genre corpus for domain-adaptive pretraining: **null** (+0.09) [LOG gendapt]. Quran-pretrained encoder backbones (to import scripture familiarity): **catastrophic** (−6.01 and −52.68) [LOG quran]. Sentence-recombination augmentation: **hurt** (−1.56, out-of-distribution) [MEM-CEIL]. Folding dev into training: **works** (+2.10 on NoPnx-NP) **and is illegal** [MEM-CEIL]. An "unfamiliarity gate" (hold fire on unseen material): **killed by measurement** — separates wrong cuts from right cuts at coin-flip AUC 0.48–0.52 [MEM-CEIL].
- *Could still work (honest):* (a) **cross-view distillation** — teach the NoPnx student from the PA teacher (the same 174 documents *with* punctuation, where the teacher scores ~94.4): the alignment is verified exact on all 174 docs (`src/xview_align.py`, label folding reproduces gold verbatim) and the trainer is built (`src/xview_distill_train.py`), battery **pending GPU** — this is legal new *supervision*, not new data, and it is the only untested idea aimed squarely at this problem; (b) the **open track**, where external data is legal — measured value so far is small (+0.11 test via two external-pretrained voters, plateaus at two) [EXP]; (c) whatever the top-3 teams do — unknown, and worth reverse-engineering effort. No overselling: nothing yet measured on the closed track has moved this problem.

**Problem 2 — confident over-cutting.**
- *Tried:* R-Drop consistency training — the **only arm that moved the needle in the right direction**: precision +1.85, false-preposition-cuts −0.35 points, F1 **+0.33** — but +0.33 < +0.45, so by our own pre-registered rule it is *not* adopted as real [PREP-TABLE]. Counterfactual fire-penalty: −0.11 [PREP-TABLE]. Dice loss (trains the score directly): −0.30 [LOG dice]. Soft-F1 loss: null [runs_softf1.log]. Mean-Teacher / CVT: null [PREP-TABLE]. Threshold re-tuning: structurally can't work (Section III step 6a). Boundary label smoothing + adversarial FGM: **+0.68 on a solo model** (June) — the one calibration-flavored idea that ever paid, and it is already banked inside the deployed ensemble story [EXP #43].
- *Could still work:* the frontier battery **running on the GPU tonight** (semi-Markov CRF = scoring whole segmentations instead of independent tokens; minimum-risk training = optimizing the F1 directly; DRO = weighting hard groups; SAM = flatter optima) is precisely the calibration/structure bet — results pending, no promises; R-Drop remains a legitimate ensemble ingredient even below the solo floor. The honest reading of the killed-gate entry stands: **the discriminator between right and wrong cuts is calibration/consistency, not familiarity** [MEM-CEIL].

**Problem 3 — cloze/exercise shredding.**
- *Tried:* exam-stem + enumeration decode rules (the legal decode pack, v16-B1 + v17-2) — built and battery-tested under quarantine; landed inside the noise floor like every decode-side patch [MEM-CEIL]. A dev-tuned "suppress cuts at فراغ" rule was **deliberately rejected** as overfitting to 4 documents [EXP, residual analysis].
- *Could still work:* the ceiling is known and small — perfect repair of all four cloze docs = **+0.54 macro** [EXP]. The principled fix is genre coverage (open track) or the PA-teacher distillation (the teacher sees the exercise punctuation `؟` and `[ ]` and segments these docs far better). Anything dev-tuned here is a trap.

**Problem 4 — legal-list over-cutting.**
- *Tried:* enumeration-mode decoding (same pack, same verdict: inside noise) [MEM-CEIL]. The numeral/list triggers are rule-shaped, but dev contains exactly **one** legal document — any rule tuned on it is tuned on one document.
- *Could still work:* macro caps this at ~1/222 of the score; treat as bounded. The teacher-distillation route again applies (PA view keeps the `؛` semicolons that mark these clause ends).

**Problem 5 — chain genres (genealogy/scripture/isnad).**
- *Tried:* Quran-encoder backbones (**−6 to −53**, see above) [LOG quran]; retrieval-side hadith augmentation is constrained by the standing ruling (retrieve verbatim only, never generate) [MEM-CEIL] and external retrieval is closed-track-illegal anyway.
- *Could still work:* honestly, on the closed track, very little that we've conceived: the conventions simply aren't in the 174 documents (the training set contains 2 exercise docs and near-zero begat-chains). Cross-view distillation helps only if the PA teacher itself knows the convention (it partly does — verse text carries punctuation). Open track: genre-matched scripture text is plentiful and legal there. This is the purest data problem on the list.

**Problems 6–9 (connective starts, length habit, doc-end hotspot, numeral triggers).**
- *Tried:* connective augmentation **failed** (−0.26) [LOG connaug_fix]; the adaptive length prior is **already in the deployed system** (+0.16 NoPnx-NP, June) [EXP #35]; doc-end boundary forcing is already in (+~0.1) [EXP]; multi-stride decoding (averaging several window offsets): best arm **+0.12, inside noise** [scratchpad msd_summary.json].
- *Could still work:* a trained transition structure (the semi-CRF running tonight) is the principled owner of "which connectives start sentences" and "how long sentences run"; decode-time nudges are measured out.

**Problem 10 — prepositions.**
- *Tried and answered:* the counterfactual preposition penalty (−0.11) and the whole battery were built for this hypothesis; the mistake analysis then showed it is ~9.5% of errors and a symptom of Problem 1 [MA §6, PREP-TABLE]. **Closed.** Any future preposition-specific rule must argue against these numbers first.

**Problem 11 — double-cut bursts.**
- *Tried:* nothing directly.
- *Could still work:* burst suppression falls out of any whole-sequence decoder (semi-CRF, tonight) for free; a standalone "no two cuts within 2 tokens" rule would fight the many *correct* adjacent cuts in list docs — measure before trusting.

**Gold labels / comma convention.**
- *Answered:* gold auditing is **out** (~3.5% mass; retraction of the 44% claim on record) [MA §9, MEM-CEIL]. On PA/NP the comma-convention bucket is *learnable headroom* (29% of PA / 24% of NP confident errors, per the record; the FP-only shares are 33% PA / 25% NP), not broken data — a comma-boundary disambiguator is a legitimate open idea for those two tracks [MEM-CEIL].

### The bottom line, without varnish

Every fix that attacked the *model* on the closed track has landed inside the luck floor (twenty-plus attempts since June across two repos [EXP; MEM-CEIL]); the two things that ever moved real points were **more supervision** (dev-folding: +2.10, illegal) and **more models voting** (ensemble: ~+1.1–1.5, already banked). That is exactly what the numbers in Sections I–III predict: a memorized-certainty model at a data-coverage wall. The two live, unspent bets are **tonight's frontier battery** (structure/calibration: semi-CRF, MRT, DRO, SAM — pending) and **cross-view distillation from the punctuation-seeing teacher** (built, verified aligned, pending) — and the standing fact that three teams are ahead of us on the same 174 documents means the wall has a door we haven't found yet.
