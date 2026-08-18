Both binding facts confirmed verbatim: DIRECTIVES.md §207 ("LLM synthesis: not built, closed track (banned/gray)") and the two independent rule sources — `araseg-shared-task-2026/README.md:22` ("trained _exclusively_ on the AraSeg corpus") plus `PLAYBOOK.md:13` (deny-list: "Fine-tuning on anything other than AraSeg **train** in the closed track"). And crucially, DIRECTIVES §42 already has the exact organizer-question channel armed (Q1/Q2). That changes my recommendation: the gate question is not new — it belongs in the existing Q-channel.

Here is the plan.

---

# Plan: Beating the limited-DATA ceiling in AraSeg 2026 (closed track)

## 1. HEADLINE

**The single most promising in-distribution-data path is self-training (noisy-student) with token-level uncertainty filtering, using the 6-model ensemble as teacher over a *real, retrieved, verified* in-genre Arabic corpus — NOT LM-generated text.** Pair it with **TAPT/DAPT continued-MLM adaptation** of the encoders on that same real corpus. Both attack the *named* bottleneck (88–89% of NoPnx errors on unseen bigrams = pure coverage failure) by injecting genuinely new in-genre bigrams, and both keep 100% of the *scored* boundary supervision on AraSeg gold.

**Honest expected gain:** **+1 to +2.5 F1 on the two NoPnx tracks (NoPnx-PA 87, NoPnx-NP 85), ~0 on PA/NP** (95/93 are saturated and not coverage-bound). This lands inside your +2–5 goal *only on the tracks with headroom*, and only if an organizer ruling permits external unlabeled text. Nothing in any surveyed family reaches +2–5 on the saturated punctuated tracks — the wall there is macro-averaging over a 94.5-median corpus plus ~44% unfixable residual (29% ambiguous gold + 15% no-evidence), not a shortage of rows.

**The one lever that is already banked and needs no ruling: fold the 222 dev docs into training (+2.1 F1, organizer-sanctioned, 3-seed held-out).** Do this first, unconditionally. It is larger and safer than anything else on the board.

---

## 2. RANKED TABLE — every surveyed approach

| # | Approach | Adds real coverage? | Closed-track status | Expected F1 (NoPnx) | Build cost | Verdict |
|---|----------|:---:|:---:|:---:|:---:|---|
| 1 | **Dev-fold into train** (organizer-sanctioned) | Yes (real held-out docs) | **Legal** | **+2.1 (measured)** | ~0 (done pattern) | **BUILD NOW** |
| 2 | **TAPT/DAPT** — continued-MLM on AraSeg's own 174+222 docs ([Gururangan ACL2020](https://arxiv.org/abs/2004.10964)) | Partial (own text only) | **Legal** | +0.3–0.8 | Low (reuse `wtp-phase1` harness) | **BUILD NOW** |
| 3 | **Layer-1 legal template-slot grammar** (extends task #18 `legal_parallel_aug.py`, train-tokens only) | No (recombines own vocab) | **Legal** (DIRECTIVES §219) | +0.2–0.8 on المادة-boundary class | Low-Med | **BUILD NOW** |
| 4 | **Self-training / noisy-student** over *real retrieved verified* corpus, ensemble teacher + SeqUST filtering ([Xie CVPR2020](https://arxiv.org/abs/1911.04252), [SeqUST AAAI2023](https://arxiv.org/abs/2302.08659)) | **Yes (new real bigrams)** | **Gray** (external *unlabeled* text) | **+1 to +2.5** | Med | **NEEDS-RULING (Q2)** |
| 5 | **TAPT/DAPT on real retrieved corpus** (Hadith/legal/news as MLM fuel, representation-only) | **Yes** | **Gray** (external unlabeled text) | +0.5–1.0 | Low | **NEEDS-RULING (Q2)** |
| 6 | **CVT / cross-view consistency** on real unlabeled text ([Clark 2018]; Saetia 2019 Thai SBD) | Yes | **Gray** (external unlabeled) | +0.3–1.0 (7–10% rel. err ↓) | Med | **NEEDS-RULING (Q2)** |
| 7 | **DAGA-linearized generator trained AraSeg-only** + ensemble self-verify ([DAGA EMNLP2020](https://aclanthology.org/2020.emnlp-main.488/)) | No (own vocab, recombined) | **Legal** (generator sees only AraSeg) | +0.0–0.3 (dilution wall) | Med-High | PARK (dominated by #2/#3) |
| 8 | **LM-generated in-genre corpus** (Jais/ALLaM/AraGPT2 → train) — *the literal directive* | Yes (but external, laundered) | **Illegal/Gray — §207 banned** | +0.5–2 w/ high variance + DQ tail | High | **NEEDS-RULING (hard stop)** |
| 9 | Multilingual transfer (SaT/WtP weights as voter) | Yes | **Illegal closed** (already VOID §6, task #19) | open-track only | Low | PARK (open-track, done) |
| 10 | Snorkel/skweak weak-supervision, STAr rules → manufacture labels | Yes | **Illegal closed** | ~0 / negative (weaker than ensemble) | High | PARK |
| 11 | Back-translation + label projection | No (rewrites punctuation=label) | **Illegal** (external MT) | negative | — | DO NOT BUILD |
| 12 | Surface aug (LwTR/SiS/EDA/recomb/cutoff/mixup) | No | Legal | **negative — TRIED −0.63 to −0.83** | spent | PARK (exhausted) |
| 13 | Punctuation-restoration / EDU / NSP / topic-seg pretraining | Weak/wrong-granularity | Legal (self-sup) to Illegal (ext. labels) | ~0 to +0.2 | Low-High | PARK |
| 14 | Confident-learning / variation-n-gram gold **audit** | No (measures ceiling) | **Legal** | 0 F1 (diagnostic) | Low | BUILD (diagnostic, separate track) |

---

## 3. THE GATE

**Plain statement:** The LM-generation-then-train path (#8) is **NOT clearly closed-track-legal, and you may not decide it yourself.** Three independent grounds converge:

- **Rule text is a whitelist.** `README.md:22` = "trained _exclusively_ on the AraSeg corpus"; `PLAYBOOK.md:13` explicitly deny-lists "Fine-tuning on anything other than AraSeg **train**." Text a pretrained LM emits is, by your own framing, external corpora (Sahih Bukhari, real legal codes, news) laundered through weights — not the 174 docs.
- **A ruling of record already parks it.** `DIRECTIVES.md:207`: *"LLM synthesis: not built, closed track (banned/gray)."* Under CLAUDE.md §3 ("do not relitigate pre-registered decisions") and §4 ("every pre-registered decision point is a hard stop"), overriding this requires a prompt that *names §207*. None has.
- **Adversarial review split gray↔illegal, never legal.** Both independent legality passes landed at "gray-needs-ruling" or "likely-illegal," and the only distinguisher from the *already-shipped* legal augmentation (task #18) is that #18's tokens are provably resampled from AraSeg train (`legal_parallel_aug.py:7-8`) while LM output provably is not.

**This is the same disqualification asymmetry every survey flagged: you are #1 on all four boards; an illegal-data DQ forfeits a winning position to chase +1–2 on two tracks.**

### The ONE exact question for Bashar (route through the armed DIRECTIVES §42 Q-channel)

Note §42 **Q2 is already the right slot** — it asks whether test/external text may feed unsupervised/self-supervised techniques. Extend it to cover both #4/#5 and #8 in one question:

> **"For the Closed Track ('trained exclusively on the AraSeg corpus'): (a) may the encoders be adapted, or the segmenter self-trained, on *real publicly-available unlabeled Arabic text* outside AraSeg (MLM/DAPT and pseudo-labeling, with all boundary gold still from AraSeg) — or is any external text disqualifying? (b) Separately, may training text be *generated by a pretrained Arabic language model* — given it reproduces in-genre passages the model memorized from non-AraSeg corpora — or is model-generated training text disqualifying regardless of the generator being a permitted pretrained model?"**

### Safe to build REGARDLESS of the ruling (start immediately)

- **#1 Dev-fold** (+2.1, banked, organizer-sanctioned).
- **#2 TAPT/DAPT on AraSeg's OWN 174+222 text** — no external token enters; cleanest legal move in the survey.
- **#3 Layer-1 legal template grammar** — extends the already-adopted task #18, tokens traceable to train (DIRECTIVES §219 "synthesize convention variants from train text only").
- **#14 Gold audit** (confident-learning / variation-n-gram) — quantifies the ~44% unfixable residual so you stop chasing a ceiling that isn't there.

Everything that adds *external* coverage (#4, #5, #6, #8) is blocked until (a)/(b) return YES.

---

## 4. HADITH CORRECTNESS — the algorithm that guarantees no incorrect Hadith is ever used

**Design rule (non-negotiable, `haram` constraint): the LM never authors Hadith. Hadith is RETRIEVED verbatim from a canonical machine-readable source, then exact-match-verified. Anything not matching a canonical record is discarded — never "repaired."** This makes fabricated Hadith structurally impossible, and it makes generation pointless for this genre (if you must verify against the canon, you already hold the real text).

**Oracle:** `AhmedBaset/hadith-json` (50,884 hadith, all 9 canonical books incl. Sahih Bukhari/Muslim + the four Sunan; sourced from Sunnah.com; fully diacritized). Mirror once, offline, read-only, git-ignored like `data/`. Never copied into training text — used only as a pass/fail gate.

**Verification pipeline (plugs in at BUILD ORDER step 5, before any Hadith doc is labeled):**

```
norm(s):                                    # CRITICAL: AraSeg is 100% diacritic-free; oracle is diacritized
  strip tashkeel  [\u064B-\u0652\u0670\u0640]
  أإآ→ا ; ى→ي ; ة→ه ; strip ـ (tatweel) ; collapse whitespace
  (do NOT strip mid-word ء — it distinguishes words)

for each candidate Hadith text h:
  matn = split_off_isnad(h)                 # cut at last isnad connective (قال/عن النبي…قال/سمعت…يقول)
  m = norm(matn)                            # verify MATN only — isnad chains legitimately vary
  cands = char_ngram_index.topk(m, 50)      # TF-IDF char_wb 3–5-gram blocking (sklearn, no new dep)
  if any(m ⊆ norm(c) or norm(c) ⊆ m):       # Stage-2 exact/substring short-circuit
      PASS conf=1.0
  else:
      best = max word_level_edit_sim(m, norm(c)) for c in cands   # word-Levenshtein, not char
      if best ≥ 0.92:            PASS  (log book+id+sim as immutable provenance)
      elif best ≥ 0.85:          QUARANTINE → require BOTH: (i) every isnad narrator ∈ oracle
                                              narrator gazetteer, (ii) cross-generator agreement
                                              → else REJECT
      else:                      REJECT (treat as fabrication; log; never use)
```

**Threshold policy is asymmetric by design: optimize for ZERO false-accepts, accept low recall.** A false-reject costs one training example (cheap; corpus is huge); a false-accept ships fabricated Hadith (unacceptable). **Mandatory human gate:** an Arabic-literate reviewer spot-checks 100 passed Hadith before any bulk use — religious correctness is not a threshold I declare met unilaterally.

Same principle for **legal**: real statute text only (each «المادة N …» a real article), exact-match or human-verified source; the LM never authors operative law.

---

## 5. BUILD ORDER — numbered, immediately actionable

**Phase 0 — banked lever + diagnostics (no ruling needed, do now):**
1. **Dev-fold.** Retrain each ensemble voter on train+dev via the exact held-out protocol that measured +2.1 (3 seeds). New dated `runs/` dir, append-only. This is the floor everything else A/Bs against.
2. **Gold audit** (`#14`): confident-learning + variation-n-gram over OOF ensemble probabilities to quantify ambiguous/broken gold. Sets a realistic ceiling before you spend on coverage.

**Phase 1 — legal, unambiguously legal coverage (no ruling needed):**
3. **TAPT/DAPT on AraSeg's own 174+222 text** (`#2`): continued-MLM on each encoder (AraBERTv02 seeds, AraELECTRA, ARBERTv2) reusing the `configs/wtp-phase1.yaml` harness with the objective swapped to plain MLM; then fine-tune the boundary head on gold. Flag-gated, additive, default reproduces today.
4. **Layer-1 legal template grammar** (`#3`): build `src/legal_lexicon.py` (harvest المادة-clause bank + camel_tools noun-phrase slots from train+dev) and `src/gen_legal.py` (grammar emits synthetic statutes with boundaries **by construction**). Boundaries authored once in PA form → project to all four tracks via the *verified* `augment.py` `deletion_mask`/`project_labels` (p=0 round-trip assertion). Target the NoPnx `(body-final-word, المادة)` unseen-bigram class.

**Phase 2 — external-coverage path (GATED on organizer YES to Q2):**
5. **Build the retrieved+verified corpus.** Hadith via `AhmedBaset/hadith-json` through the §4 verifier; legal via real statutes; news via a clean real source. **Genre mix MUST match the real corpus: ~94% general prose, ≤15% Hadith, ≤10% legal, ≤10% exam** (measured via `genre_buckets.py`). Label boundaries by *concatenating verified complete sentences* (last token of each = gold boundary, by construction — never self-segment).
6. **Self-training + SeqUST filtering** (`#4`): 6-model ensemble pseudo-labels the real corpus; MC-dropout token-level uncertainty keeps only low-variance high-confidence boundaries; noise the student with the existing SaT-corruption curriculum; ≤2 rounds, re-check dev each round.
7. **DAPT on the real corpus** (`#5`): fold the retrieved text into the step-3 MLM phase (representation-only), boundary gold stays 100% AraSeg.

**A/B protocol against the noise floor (every phase):** single AraBERTv02 seed, `train (+dev)` vs `train (+dev) + intervention`, **3 seeds**, evaluate on the held-out slice with the ±0.45 seed floor. **Pre-registered keep/kill:** seed-mean NoPnx gain ≥ +0.5 with worst seed ≥ 0 → adopt standalone; +0.1–0.3 → fold as decorrelated voter behind the bootstrap gate (paired-doc, 20k resamples); < +0.1 → park with the null on record. Zero-PA-regression and constitution-doc F1 reported separately. Admit into the 6-model pool only after clearing the bootstrap gate.

---

## 6. RISKS — how this avoids the recomb −1.56 failure

Random recombination scored **−1.56** because it destroyed **document-level genre structure** — locally-valid sentences, globally incoherent register-salad — teaching boundary statistics that match no real genre. This plan avoids that by four explicit constraints, each load-bearing:

1. **Coherent-block resampling, not global sentence sampling.** Every synthetic/retrieved doc is assembled from **same-source, same-topic** sentence blocks (one Hadith doc = consecutive hadith from one book/chapter; one prose doc = one passage in order). This is the direct fix for register-salad.
2. **Distribution-matched genre mix + length tail.** Match the real 94%-prose mix and the per-genre length/paragraph/boundary-rate histograms from `genre_buckets.py`, **including the heavy right tail** (real max ~19k tokens) that recomb's fixed-median target flattened. Cap Hadith/legal as minority tilt — never flip the majority, or you go OOD exactly like recomb.
3. **Boundaries by construction, projected through verified code.** Author gold once in PA form by concatenating known-complete sentences (last token = boundary), then derive all four tracks with `augment.py`'s p=0-verified `project_labels`. Never self-segment synthetic data (that bakes current errors into gold).
4. **In-distribution quality cascade.** Script-ratio ≥0.9, repetition/degeneracy reject, per-genre length/boundary-rate band reject, held-out-AraGPT2 perplexity reject, punctuation-well-formedness (must survive the projection assertion), **plus an 8-gram tripwire against dev+test inputs** to catch any memorized-source leakage — the single most likely way a #1 system disqualifies itself.

**Additional named risks:** (a) *Distribution drift* — LM/clean text is cleaner than AraSeg's noisy exam/OCR text; cap synthetic fraction ≤30% of train and pipe through `augment.py` corruption to match the noise floor. (b) *Confirmation bias* in self-training (worst at 174 docs) — SeqUST uncertainty filtering is non-negotiable; without it the loop degrades NoPnx. (c) *Gate-A misjudgment = DQ* — the mitigation is the ruling; do not train any external-coverage arm until Q2 returns YES.

**Pre-registered reading, ticked verbatim (Read & Sproat 2012, per SOLUTIONS.md):** *"a moderate interpretation of document structure substantially contributes"* — realized here as legal template structure + verified-Hadith structure as free boundary supervision, gated on convention alignment to AraSeg's sentence definition and on the organizer ruling.

**Files of record:** `src/augment.py` (verified `deletion_mask`/`project_labels`), `src/make_recomb.py` (end-on-boundary check, −1.56 baseline), `src/genre_buckets.py` (mix targets), `src/legal_parallel_aug.py` (task #18 legal precedent), `configs/wtp-phase1.yaml` (reusable TAPT harness), `recurrent-segmenter/DIRECTIVES.md` §42 (armed Q-channel), §207 (the ruling that gates #8), §219 (train-text-only augmentation sanction), `araseg-shared-task-2026/README.md:22` + `docs/PLAYBOOK.md:13` (governing rule).