# Replicating ENSAR end to end

The full pipeline is driven by the repository Makefile:

    make data        # fetch the public AraSeg corpus (four tracks)
    make voters      # fine-tune the five encoder voters (GPU; ~4 min/voter/track on an RTX 5090)
    make probs       # cache per-voter probabilities
    make packets TRACK=NoPnx-NP    # build the sealed jury packets from public text
                                   # + the RELEASED exact draft rows (byte-identical
                                   # to the paper's; verified by build_packets.py)
    make jury  TRACK=NoPnx-NP      # prints the two ways to run the jury stage (below)
    make score TRACK=NoPnx-NP      # strict-scores kit verdicts against public test gold

`make voters`/`make probs` reproduce the encoder METHOD from scratch; to reproduce the
paper's NUMBERS exactly, skip them and use the released draft rows (the default for
`make packets`). The scorer on the paper's own verdicts reproduces 84.64 -> 89.49 on
NoPnx-NP exactly.

## The jury stage itself

The jury stage was executed with **Claude Code** (Anthropic's agent CLI; public product), version 2.1.234,
model `claude-opus-5` at **maximum reasoning effort**, provider-default sampling. Replicating as-performed
therefore means running the sittings inside Claude Code. An approximate framework-free path is described at
the end.

## As-performed path (Claude Code)

1. Install Claude Code (https://claude.com/claude-code) and sign in to an account with Opus 5 access.
2. Lay out a working folder:
   - `doctrines/doctrine_j0.md`, `doctrines/doctrine_j1.md` — from `jury/doctrines/<track>/` in this repo
     (READ-ONLY; never edit).
   - `docs/<doc_id>.json` — the sealed packets: `{doc_id, text}` where the text carries the ensemble's
     draft boundary marks. Packets contain no labels. (Build from your own ensemble output; the format is
     shown in `jury/README.md`.)
   - `out/` — empty; one verdict JSON per document will be written here.
3. For each sitting of 4 documents, give Claude Code the sitting prompt VERBATIM from
   `jury/prompts/adjudication_prompt.txt` (also reproduced in the paper's appendix), with the doctrine
   paths, the 4 packet paths, and the `out/` path filled in. Model: Opus 5, maximum reasoning effort.
   Up to 4 sittings may run in parallel; a sitting that dies is simply relaunched (verdicts bank
   per-document; the prompt's RESUME clause skips banked files).
4. The agent may read ONLY the two doctrine files and its listed packets, and write ONLY its own `out/`
   files. No web, no shell. (We additionally enforced this with a PreToolUse hook denying all other file
   access; see the paper's Isolation paragraph.)
5. Score with `jury/score/score_strict.py` (strict: an edit applies only if the stated index holds exactly
   the stated token; no tolerance).

## Determinism

Decoding is not deterministic and no seed exists. Expect run-to-run variation of the order we measured:
a full second run over 50 documents differed from the first by 0.48 macro F1 (25 of 50 documents scored
identically). Do not expect byte-identical verdicts; expect statistically indistinguishable scores.

## Framework-free approximation

`jury/run_sitting_reference.py` (untested reference) inlines the doctrines and one packet into the same
sitting prompt and calls the API directly via the Anthropic SDK, parsing the verdict JSON from the reply.
This removes the Claude Code dependency at the cost of the file-tool workflow (the agent can no longer
re-read files or keep scratch notes); it is an approximation of the as-performed setup, not a replica.
