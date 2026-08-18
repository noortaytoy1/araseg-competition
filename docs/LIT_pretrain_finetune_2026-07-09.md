Guide published: https://claude.ai/code/artifact/dd02f450-1b4c-4508-a093-a0535bcc86a8

Local file: `C:\Users\pc\AppData\Local\Temp\claude\C--Users-pc-Downloads-evolving-vlm-46\3be146eb-364e-439f-b9f7-50d0973ffe61\scratchpad\araseg_guide.html`

## What it contains (the 5 requested sections)

**1. TL;DR** — 6 tagged bullets. The headline: the highest-value NEW moves are optimizer/objective-alignment tricks that are closed-track-legal by construction, NOT more pretraining.

**2. Ranked ledger** — 14 strategies, scrollable table with Strategy | stage | token-labeling evidence | low-resource? | tried-here? | closed-track? | repo | expected value (with visual EV meters). Ranked by expected value for this specific ensemble.

**3. Top 3 to try next**, each with the real repo/paper + minimal flag-gated experiment:
- **#1 Layer-wise LR decay + Mosbach/Zhang stability bundle** (asappresearch/revisit-bert-finetuning) — zero-GPU, doubles as decorrelation lever
- **#2 Child-Tuning-F** (RunxinXu/ChildTuning) — one-line gradient mask, attacks the TAPT −0.61 over-adaptation failure directly
- **#3 FFF-NER objective-aligned voter** (ZihanWangKi/fffner) — flagged as a **GPU hard-stop** per CLAUDE.md §4, with its pre-registered keep/park reading

**4. Skip list** — 10 items with blunt reasons (gradual unfreezing mechanism-mismatch, SpanBERT/SBO wrong info-flow, generative objectives wrong architecture, PET binary-degenerate, LoRA crossover, etc.).

**5. Ceiling verdict** — one honest line: **probably not a beat, most likely a ceiling confirmation.** Realistic upside is ensemble decorrelation + anti-overfit stability from picks #1/#2, not a new pretraining objective.

## Load-bearing calls surfaced
- **Closed-track-illegal flag** raised explicitly on external-labeled STILTs (POS/NLI/chunking/punctuation-restoration corpora all violate the AraSeg-only clause).
- **Punctuation-restoration** flagged legal only via self-derived supervision, and scoped to punctuated tracks (inert on NoPnx, overlaps parked WtP).
- The genuinely-new, never-logged-here items are separated from prior-review repeats: LLRD-as-decorrelator, Child-Tuning, Dodge multi-seed pruning, BitFit voter, FFF-NER's closed-track cell.

One design note: the artifact renders webfont-free (deliberate serif/grotesque/mono system stack) so nothing silently falls back under the CSP, and every wide table scrolls inside its own container so the page body never scrolls sideways.