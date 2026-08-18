# Writing doctrine for the AraSeg paper

Same protocol the juries use, applied to the paper: Noor marks an error, I write down the
rule behind it with the offending example, and every later draft is checked against this
file. Entries are cumulative. When a new correction contradicts an earlier entry I fix the
entry and say so rather than silently dropping it.

Checked mechanically where possible by `python paper/verify_paper.py`.

---

## L1. No em dashes, no en dashes, no tildes anywhere in the source
My error: first draft used them throughout.
Why: Noor's explicit instruction; they read as machine-written.
How to apply: the verifier fails the draft if any appear. Use commas, colons, or a new
sentence. In LaTeX write `Table \ref{...}` without the non-breaking tilde.

## L2. Do not write in the register of a chatbot
My error: balanced hedging, tricolons, "it is worth noting", sentences that qualify
themselves twice.
How to apply: state the claim, then the evidence. One idea per sentence. Cut any clause
that exists only to soften.

## L3. Never write a sentence that describes content instead of delivering it
My error, the worst one so far: "their content is not a list of heuristics over punctuation
but a taxonomy of registers with a characteristic unit length and a set of licensing
conditions for each". Noor: "extremely surface-level wording".
Why: I had nothing concrete to say in that section, so I filled it with vocabulary. The
reader learns nothing about what the system actually knows.
How to apply: if a paragraph characterises an artifact, replace it with the artifact. Quote
the actual rule, give the actual number, name the document that produced it.
Fixed by: the three real rules (line-density test, per-genre sentence lengths, the retired
wa/fa heuristic).

## L4. No jargon the reader has to decode
My error: "credit assignment", "token-localised supervision", "genre taxonomy of the
sentence unit", "licensing conditions". Noor: "extremely jargon heavy and I even can't
understand what u wrote".
How to apply: write the mechanism in plain words. "The jury works out why it went wrong"
beats "performs credit assignment". Keep a term only if it is the standard name of a thing
being cited, and then gloss it once.

## L5. Every claim needs a source: literature or our own measurement
My error: asserting novelty without positioning it, and asserting mechanism without
evidence.
How to apply: each claim is either cited (verified real, with venue) or produced by a run I
can recompute. Citations verified so far: Christiano 2017, Ouyang 2022, Shinn 2023
(Reflexion, NeurIPS), Madaan 2023 (Self-Refine, NeurIPS), Yang 2024 (OPRO, ICLR),
Yuksekgonul 2024 (TextGrad, arXiv 2406.07496). Never invent a citation.

## L6. The paper needs a hook and an idea, not a description of what we built
Noor: the idea is RLHF-shaped. Start from an initial prompt, the model scores, inspects its
own reasoning errors and traces, and uses them as feedback to refine its lessons file.
How to apply: lead with the failure that motivates the method (ensemble is confidently
wrong and wrong together), then the thesis (what is missing is a policy, not calibration),
then the loop. Structure and content first; voice second.

## L7. Show reasoning traces
Noor: the reader needs examples to get hooked.
How to apply: quote real jury output. Currently used: the retired first-joint heuristic
(Figure 1), and the scripture adjudication where one jury argued from verse structure and
the other from rhyme and both endorsed the same edit.

## L8. Every number must survive an independent recomputation
Noor: "I am not very sure of ur numbers, be 1000000% sure".
Why: I produced several wrong numbers by checking claims one at a time as they came up.
How to apply: `paper/verify_paper.py` re-derives all 84 claims from the corpus, the cached
probabilities, the jury verdicts and the training code, and exits non-zero on any mismatch.
Run it after every edit, not only when asked.
Errors it caught or would have caught: "micro is about 1.5 points higher" (true value varies
0.50 to 2.25), "six documents improved for every one harmed" (6.6 on one track, 4.7 on the
other), window size quoted without units or stride, class weighting described without the
formula.

## L9. Do not claim the ablation reconstructs the submitted system
My error: implying the test-split ablation decomposed the leaderboard scores.
Facts: the NoPnx-PA juries in the ablation were retrained on the full corpus after the
deadline; the ablation baseline is a thresholded average, not the length-aware decode used
for submission, and scores about half a point lower.
How to apply: state both, label the NoPnx-PA row an unofficial post-deadline result.

## L10. Name the model exactly
Claude Opus 5, run at maximum reasoning effort, for both training and adjudication.

## L11. Do not raise the rule-based question
Noor: no need to mention it. Verified separately that the submitted predictions reproduce
exactly from cached probabilities (212/212), so the baseline is purely neural and the point
does not need airing in the paper.

## L12. LaTeX hazards seen so far
- `\<w>` is an ArabTeX macro and prints literally as `w>` without that package. Write
  Arabic function words in italic transliteration instead: \textit{wa}, \textit{fa}.
- The numbers interleaved in copied text (024, 034, 037) are ACL review-mode line numbers
  from `\usepackage[review]{acl}`, not corruption. They vanish with `\usepackage{acl}`.
- Arabic script requires XeLaTeX; the draft compiles under pdflatex, so keep examples in
  transliteration.

---

## L16. No flat text
Noor (Aug 13): "stop with the stupid flat texts."
How to apply: every paragraph must move: a fact the reader did not have, a consequence, or
a decision. A sentence that neither surprises nor decides gets cut or merged. Test: read
the paragraph's first and last sentence; if nothing changed between them, it is flat.

## L17. Every claim is proved in the appendix
Noor (Aug 13): "always prove ur claims in the appendix."
How to apply: any number or empirical claim in the body gets an appendix paragraph stating
exactly how it is computed (split, threshold, definition), and a matching check in
verify_paper.py. First applied: the error-analysis paragraph (0.86 median, 48.3\%) with
its appendix derivation and checks 5b.

## L13. The abstract is a hook, not a summary of the method
My error: a 230-word abstract that walked through every step of the loop, then the numbers,
then the significance tests. Noor: "the worst abstract i've ever seen ... THE ABSTRACT
SHOULD SERVE AS THE HOOK".
How to apply: open with the finding that makes someone keep reading (all five models agree
and are all wrong), state the idea in one sentence, give two numbers, close with the
result. Around 150 words. No p-values, no confidence intervals, no per-track breakdowns;
those belong in the results table.
Fixed by: the current 153-word abstract.

## L14. Do not put statistical apparatus where it does not belong
Noor: "DELETE THE P < 10^-12 BS".
How to apply: significance belongs in the results table only. Do not repeat it in the
abstract, the introduction or the conclusion. State the effect, not the test.

## L15. Length is not thoroughness
My error, repeatedly: writing more text about something that does not deserve it, in the
abstract, in the figure brief, in this file.
How to apply: before adding a sentence, ask what the reader does differently for having
read it. If nothing, cut it.
