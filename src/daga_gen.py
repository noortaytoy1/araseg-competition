"""DAGA learned generator for closed-track data multiplication.

Faithful transfer of DAGA (Ding et al., EMNLP 2020,
"Data Augmentation with a Generation Approach for Low-resource Tagging Tasks",
https://aclanthology.org/2020.emnlp-main.488/). The vendored reference source is
in vendor/daga/ (cloned from https://github.com/ntunlp/daga); this file transfers
their three moving parts:

  1. LINEARIZE a labeled sequence by folding the tag inline and DROPPING the
     majority "O" tag (vendor/daga/tools/preprocess.py::_linearize). DAGA writes
     the tag token *before* its word because a BIOES tag describes the word that
     follows. Our task is BINARY boundary detection where label==1 means "a
     sentence boundary FOLLOWS this token", so the marker describes the PRECEDING
     token and is emitted immediately AFTER it (the scheme named in the build
     order). The no-boundary tag is dropped exactly as DAGA drops "O".
  2. Train a plain LSTM language model on the interleaved token/marker stream
     (vendor/daga/lstm-lm/model.py::LMModel, the num_z_samples==0 branch: a plain
     autoregressive LSTM decoder, no VAE). Re-implemented self-contained in torch
     so we take no torchtext dependency (torchtext is deprecated; torch is already
     a project requirement) -- same architecture: embedding -> LSTM -> linear
     generator -> token-level cross-entropy.
  3. SAMPLE new sequences token-by-token with temperature multinomial sampling
     from <bos> to <eos> (vendor/daga/lstm-lm/model.py::generate), then
     DE-LINEARIZE: every '<BND>' becomes label==1 on the preceding real token
     (vendor/daga/tools/line2cols.py). Text and labels are generated JOINTLY --
     the same <BND> markers the LM learned to place interior to the stream become
     the boundary labels, so labels are never guessed or self-segmented.

CLOSED-TRACK LAW (non-negotiable):
  * The LM is fit ONLY on data/{TASK}_train.jsonl (the 174 train docs).
  * Dev/test jsonl are NEVER read, sampled from, or fit on.
  * No external corpus, no pretrained generative LM world knowledge.
  * The vocabulary is built ONLY from train tokens; a sampled token that would be
    <unk> is rejected, so every token in every synthetic doc is traceable to
    train.
  * Boundary labels are correct BY CONSTRUCTION: the LM emits <BND> and the
    de-linearizer turns each <BND> into a boundary on the preceding token. Labels
    are never guessed or self-segmented.

DISTRIBUTION MATCHING (recomb was scored -1.56 for ignoring this):
  The LM unit is a fixed-length CHUNK of a document's interleaved token/<BND>
  stream (default WIN=64 surface symbols). Training on chunks -- rather than on
  one isolated sentence per sequence -- keeps every <BND> INTERIOR to a running
  stream, so the LM learns realistic inter-boundary spacing from left context
  instead of being forced to emit a boundary-then-EOS after a few tokens. This is
  what makes the generated boundary rate land on the real ~0.10 (an early
  sentence-per-sequence variant over-emitted boundaries at ~0.24; verified). To
  GENERATE, we sample chunk streams and concatenate them into a running document
  until it reaches a target token length drawn (with replacement) from the real
  per-doc token-length profile of train -- reproducing the heavy right tail of the
  document length distribution -- then cut the document at its last <BND> so it
  ends on a boundary by construction.

Output mirrors make_recomb.py exactly: originals kept verbatim, then N x |train|
synthetic docs appended, written to data/daga_{TASK}_train.jsonl with the schema
{"doc_id", "tokens", "labels"}.

Usage:
  python src/daga_gen.py train    <task> [--epochs E --emb-dim D --rnn-size H ...]
  python src/daga_gen.py generate <task> --mult N [--temperature T --seed S ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
MODELS = os.path.join(HERE, os.pardir, "runs", "daga_models")

# Special tokens. <BND> is the inline boundary marker (our single "tag").
PAD, BOS, EOS, UNK, BND = "<pad>", "<bos>", "<eos>", "<unk>", "<BND>"
SPECIALS = [PAD, BOS, EOS, UNK, BND]
# DAGA normalizes any purely-numeric token to "N" (preprocess.py::normalize_tok).
NUM = "N"


def normalize_tok(tok: str) -> str:
    """DAGA's numeric normalization (vendor/daga/tools/preprocess.py)."""
    return NUM if tok.isnumeric() else tok


# --------------------------------------------------------------------------- #
# Data loading (train ONLY) + sentence extraction
# --------------------------------------------------------------------------- #
def _load_train_docs(task: str) -> list[dict]:
    """Load ONLY data/{task}_train.jsonl. The closed-track law lives here."""
    path = os.path.join(DATA, f"{task}_train.jsonl")
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _split_sentences(docs: list[dict]) -> list[list[str]]:
    """Split every doc into its gold sentences (a sentence = tokens up to and
    including a label==1 token). Returns each sentence as a list of RAW tokens.
    Mirrors make_recomb.py::sentences but returns tokens only. Used for vocab
    building and diagnostics; the LM itself trains on document chunks."""
    sents = []
    for d in docs:
        toks, labs = d["tokens"], d["labels"]
        start = 0
        for i, l in enumerate(labs):
            if l == 1:
                sents.append(list(toks[start : i + 1]))
                start = i + 1
        if start < len(toks):  # trailing (rare; all train docs end on boundary)
            sents.append(list(toks[start:]))
    return sents


def _doc_token_lengths(docs: list[dict]) -> list[int]:
    """Per-doc token counts -- the length profile we resample to match the real
    document length distribution (heavy right tail, min/med/max 34/426/15472)."""
    return [len(d["tokens"]) for d in docs]


def linearize_doc(d: dict, vocab: set[str] | None) -> list[str]:
    """Linearize a whole document's token stream inline: emit each (normalized)
    token, and after any boundary token (label==1) emit <BND>. No-boundary is
    dropped (DAGA's O-drop). This is the surface stream the LM is trained on."""
    out = []
    for tok, lab in zip(d["tokens"], d["labels"]):
        t = normalize_tok(tok)
        if vocab is not None and t not in vocab:
            t = UNK
        out.append(t)
        if lab == 1:
            out.append(BND)
    return out


def _doc_chunks(docs: list[dict], vocab_set: set[str], win: int) -> list[list[str]]:
    """Cut every document's linearized surface stream into consecutive windows of
    `win` surface symbols. Each window is one LM training sequence. Windows may
    start/end mid-sentence -- the LM learns the joint stream regardless, exactly
    as DAGA trains on linearized tagged sequences with interior tags."""
    chunks = []
    for d in docs:
        lin = linearize_doc(d, vocab_set)
        for i in range(0, len(lin), win):
            piece = lin[i : i + win]
            if piece:
                chunks.append(piece)
    return chunks


# --------------------------------------------------------------------------- #
# Linearization  (DAGA preprocess.py, adapted: marker AFTER the boundary token,
# "O" / no-boundary dropped)
# --------------------------------------------------------------------------- #
def linearize_sentence(tokens: list[str], vocab: set[str] | None) -> list[str]:
    """Fold labels inline: emit each (normalized) token, and after a boundary
    token emit <BND>. The no-boundary tag is dropped (DAGA's O-drop). If `vocab`
    is given, out-of-vocab tokens map to <UNK> (as in DAGA); pass None to keep raw
    tokens (used only for vocab building)."""
    out = []
    n = len(tokens)
    for i, tok in enumerate(tokens):
        t = normalize_tok(tok)
        if vocab is not None and t not in vocab:
            t = UNK
        out.append(t)
        # By construction the boundary is the LAST token of a gold sentence.
        if i == n - 1:
            out.append(BND)
    return out


def delinearize_stream(stream: list[str]) -> tuple[list[str], list[int]]:
    """Invert linearize_sentence for a sampled marker/token stream.

    Rule (exact inverse): a <BND> sets label==1 on the immediately preceding real
    token; every other real token gets label==0. Consecutive/leading <BND> markers
    and any special tokens are ignored. Returns (tokens, labels).
    """
    tokens: list[str] = []
    labels: list[int] = []
    for sym in stream:
        if sym == BND:
            if tokens:  # a boundary applies to the preceding real token
                labels[-1] = 1
            # a leading/duplicate <BND> with nothing before it is a no-op
        elif sym in (PAD, BOS, EOS, UNK):
            # UNK should never survive (we reject unk during sampling); guard anyway
            continue
        else:
            tokens.append(sym)
            labels.append(0)
    return tokens, labels


# --------------------------------------------------------------------------- #
# Vocab
# --------------------------------------------------------------------------- #
def build_vocab(sents: list[list[str]], min_freq: int = 1) -> dict:
    """Build a token->id map from TRAIN sentences only. Specials first, then
    every normalized train token with freq >= min_freq (DAGA keeps freq>min_freq;
    min_freq=1 default here keeps everything so all tokens stay traceable to train
    and generation is not needlessly starved). <BND> is a first-class vocab item."""
    from collections import Counter

    cnt: Counter = Counter()
    for s in sents:
        for tok in s:
            cnt[normalize_tok(tok)] += 1
    itos = list(SPECIALS)
    seen = set(SPECIALS)
    for tok, freq in cnt.most_common():
        if tok in seen:
            continue
        if freq >= min_freq:
            itos.append(tok)
            seen.add(tok)
    stoi = {t: i for i, t in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}


# --------------------------------------------------------------------------- #
# Model: plain autoregressive LSTM LM (DAGA lstm-lm, num_z_samples==0 branch)
# --------------------------------------------------------------------------- #
class LSTMLM(nn.Module):
    """Embedding -> LSTM -> linear generator. Token-level cross-entropy. This is
    the plain-LM path of vendor/daga/lstm-lm/model.py::LMModel (encoder=None)."""

    def __init__(self, vocab_size: int, emb_dim: int, rnn_size: int,
                 num_layers: int, dropout: float, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim, rnn_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.generator = nn.Linear(rnn_size, vocab_size, bias=False)
        self.rnn_size = rnn_size
        self.num_layers = num_layers

    def forward(self, inp, state=None):
        emb = self.drop(self.embeddings(inp))
        out, state = self.rnn(emb, state)
        out = self.drop(out)
        logit = self.generator(out)
        return logit, state


# --------------------------------------------------------------------------- #
# Batching (bucket by length, pad, BOS/EOS wrap)
# --------------------------------------------------------------------------- #
def _numify(sent_lin: list[str], stoi: dict) -> list[int]:
    unk = stoi[UNK]
    ids = [stoi[BOS]]
    for tok in sent_lin:
        ids.append(stoi.get(tok, unk))
    ids.append(stoi[EOS])
    return ids


def _make_batches(seqs: list[list[int]], batch_size: int, pad_idx: int,
                  rng: random.Random):
    """Sort by length into buckets for efficient padding, shuffle bucket order."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    for idxs in batches:
        maxlen = max(len(seqs[i]) for i in idxs)
        buf = torch.full((len(idxs), maxlen), pad_idx, dtype=torch.long)
        for r, i in enumerate(idxs):
            s = seqs[i]
            buf[r, : len(s)] = torch.tensor(s, dtype=torch.long)
        yield buf


# --------------------------------------------------------------------------- #
# TRAIN step
# --------------------------------------------------------------------------- #
def train_lm(task: str, args) -> str:
    """Fit the LSTM LM on TRAIN document chunks of `task` only. Saves a checkpoint
    to runs/daga_models/daga_{task}.pt and returns the path."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    docs = _load_train_docs(task)
    sents = _split_sentences(docs)  # for vocab only
    vocab = build_vocab(sents, min_freq=args.min_freq)
    stoi = vocab["stoi"]
    pad_idx = stoi[PAD]

    # LM unit = fixed-length window of each doc's linearized token/<BND> stream, so
    # boundaries stay interior and the model learns realistic spacing (see module
    # docstring). OOV -> <unk> (no OOV within train at min_freq=1), BOS/EOS wrap.
    vocab_set = set(vocab["itos"])
    chunks = _doc_chunks(docs, vocab_set, args.win)
    seqs = [_numify(c, stoi) for c in chunks]

    device = torch.device(
        "cuda" if (args.gpuid >= 0 and torch.cuda.is_available()) else "cpu"
    )
    model = LSTMLM(
        vocab_size=len(vocab["itos"]), emb_dim=args.emb_dim, rnn_size=args.rnn_size,
        num_layers=args.num_layers, dropout=args.dropout, pad_idx=pad_idx,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx)

    model.train()
    n_batches = math.ceil(len(seqs) / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        total, ntok, correct = 0.0, 0, 0
        for buf in _make_batches(seqs, args.batch_size, pad_idx, rng):
            buf = buf.to(device)
            inp, tgt = buf[:, :-1], buf[:, 1:]
            logit, _ = model(inp)
            loss = crit(logit.reshape(-1, logit.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            with torch.no_grad():
                mask = tgt != pad_idx
                nt = int(mask.sum().item())
                total += loss.item() * nt
                ntok += nt
                pred = logit.argmax(-1)
                correct += int((pred.eq(tgt) & mask).sum().item())
        nll = total / max(ntok, 1)
        ppl = math.exp(min(nll, 20.0))
        acc = 100.0 * correct / max(ntok, 1)
        if args.verbose:
            print(
                f"[train {task}] epoch {epoch:3d}/{args.epochs} | "
                f"nll {nll:6.3f} | ppl {ppl:8.2f} | acc {acc:5.2f} | "
                f"{len(seqs)} chunks (win {args.win}), {n_batches} batches",
                file=sys.stderr,
            )

    os.makedirs(MODELS, exist_ok=True)
    ckpt_path = os.path.join(MODELS, f"daga_{task}.pt")
    torch.save(
        {
            "vocab": vocab,
            "model_state": model.state_dict(),
            "arch": {
                "emb_dim": args.emb_dim, "rnn_size": args.rnn_size,
                "num_layers": args.num_layers, "dropout": args.dropout,
            },
            "task": task,
            "win": args.win,
            # per-doc token-length profile of TRAIN, for length-matched assembly
            "doc_token_lengths": _doc_token_lengths(docs),
            "n_train_docs": len(docs),
        },
        ckpt_path,
    )
    if args.verbose:
        print(f"[train {task}] saved -> {ckpt_path}", file=sys.stderr)
    return ckpt_path


# --------------------------------------------------------------------------- #
# SAMPLE a continuous linearized stream (temperature multinomial, DAGA-style)
# DAGA lstm-lm/model.py::generate, single-sequence form. We sample until the
# DE-LINEARIZED token count reaches `target_tokens`, restarting a fresh chunk
# (re-seed BOS, reset state) whenever the LM emits <eos> -- so a document is a
# concatenation of sampled chunks, exactly the training unit.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_doc_stream(model: LSTMLM, vocab: dict, device, temperature: float,
                      target_tokens: int, rng_gen: torch.Generator,
                      max_steps: int, reject_unk: bool = True) -> list[str]:
    """Autoregressively sample a linearized surface stream (tokens + <BND>) whose
    de-linearized token count is about `target_tokens`. Returns the surface-symbol
    list (specials stripped). <eos> starts a fresh chunk (BOS + reset state) so the
    per-chunk context matches training; sampling stops once enough real tokens have
    been produced. The caller cuts the stream at its last <BND> to end on a
    boundary."""
    stoi, itos = vocab["stoi"], vocab["itos"]
    bos, eos, unk, pad = stoi[BOS], stoi[EOS], stoi[UNK], stoi[PAD]

    out_syms: list[str] = []
    n_tokens = 0  # real tokens emitted so far (i.e. non-<BND> surface symbols)

    inp = torch.tensor([[bos]], dtype=torch.long, device=device)
    state = None
    steps = 0
    while n_tokens < target_tokens and steps < max_steps:
        steps += 1
        logit, state = model(inp, state)
        logit = logit[:, -1, :] / temperature
        # Ban <pad>/<bos>/<unk>: <pad>/<bos> are structural, and banning <unk>
        # keeps every emitted token traceable to train.
        logit[0, pad] = float("-inf")
        logit[0, bos] = float("-inf")
        if reject_unk:
            logit[0, unk] = float("-inf")
        probs = F.softmax(logit, dim=-1)
        nxt = int(torch.multinomial(probs, 1, generator=rng_gen).item())
        if nxt == eos:
            # End of a chunk: begin a fresh one from BOS with reset hidden state.
            inp = torch.tensor([[bos]], dtype=torch.long, device=device)
            state = None
            continue
        sym = itos[nxt]
        out_syms.append(sym)
        if sym != BND:
            n_tokens += 1
        inp = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return out_syms


def _cut_at_last_boundary(stream: list[str]) -> list[str]:
    """Trim a surface stream so it ends on a <BND> (guaranteeing the delinearized
    doc ends on a boundary). Returns [] if there is no <BND>."""
    last = -1
    for i, s in enumerate(stream):
        if s == BND:
            last = i
    return stream[: last + 1] if last >= 0 else []


# --------------------------------------------------------------------------- #
# GENERATE step: sample sentences, assemble length-matched docs, write file
# --------------------------------------------------------------------------- #
def generate(task: str, args) -> str:
    ckpt_path = os.path.join(MODELS, f"daga_{task}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No trained LM at {ckpt_path}. Run: python src/daga_gen.py train {task}"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    arch = ckpt["arch"]
    doc_token_lengths = ckpt["doc_token_lengths"]
    n_train_docs = ckpt["n_train_docs"]

    device = torch.device(
        "cuda" if (args.gpuid >= 0 and torch.cuda.is_available()) else "cpu"
    )
    model = LSTMLM(
        vocab_size=len(vocab["itos"]), emb_dim=arch["emb_dim"],
        rnn_size=arch["rnn_size"], num_layers=arch["num_layers"],
        dropout=arch["dropout"], pad_idx=vocab["stoi"][PAD],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    py_rng = random.Random(args.seed)

    n_synth = args.mult * n_train_docs

    # Keep originals verbatim (mirrors make_recomb.py).
    originals = _load_train_docs(task)
    out = list(originals)

    made = 0
    attempts = 0
    max_attempts = n_synth * 6 + 100
    while made < n_synth and attempts < max_attempts:
        attempts += 1
        # Draw a target token length from the real per-doc profile -> matches the
        # heavy-tailed document length distribution (min/med/max 34/426/15472).
        target_tokens = py_rng.choice(doc_token_lengths)
        # Budget enough sampling steps: real tokens + boundary markers (~+brate)
        # + <eos> chunk restarts, with generous slack.
        max_steps = int(target_tokens * 2.0) + 200
        syms = sample_doc_stream(
            model, vocab, device, args.temperature,
            target_tokens, gen, max_steps=max_steps, reject_unk=True,
        )
        syms = _cut_at_last_boundary(syms)  # end the doc on a boundary
        if not syms:
            continue
        dt, dl = delinearize_stream(syms)
        if not dt or dl[-1] != 1:
            continue  # must be non-empty and boundary-terminated
        out.append({"doc_id": f"daga_{task}_{made}", "tokens": dt, "labels": dl})
        made += 1

    dst = os.path.join(DATA, f"daga_{task}_train.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Sanity (same spirit as make_recomb.py's final check).
    bad = sum(
        1 for d in out
        if str(d["doc_id"]).startswith("daga_") and d["labels"][-1] != 1
    )
    print(
        f"{len(originals)} orig + {args.mult}x = {len(out)} docs "
        f"({made}/{n_synth} synth made, {attempts} attempts) | "
        f"synth-docs not ending on boundary: {bad} -> {dst}"
    )
    return dst


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(p):
    p.add_argument("task", nargs="?", default="NoPnx-NP")
    p.add_argument("--gpuid", type=int, default=-1, help="-1 = CPU")
    p.add_argument("--seed", type=int, default=3435)  # DAGA's default seed
    p.add_argument("--emb-dim", type=int, default=300)      # DAGA default
    p.add_argument("--rnn-size", type=int, default=512)     # DAGA default
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="fit the LSTM LM on {task}_train.jsonl only")
    _add_common(pt)
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--lr", type=float, default=1e-3)      # DAGA default
    pt.add_argument("--clip", type=float, default=5.0)
    pt.add_argument("--min-freq", type=int, default=1)
    pt.add_argument("--win", type=int, default=64,
                    help="surface symbols per linearized training chunk")
    pt.add_argument("--verbose", action="store_true", default=True)

    pg = sub.add_parser("generate", help="sample N x |train| synthetic docs")
    _add_common(pg)
    pg.add_argument("--mult", type=int, default=5)
    pg.add_argument("--temperature", type=float, default=1.0)  # DAGA default

    args = ap.parse_args()
    if args.cmd == "train":
        train_lm(args.task, args)
    elif args.cmd == "generate":
        generate(args.task, args)


if __name__ == "__main__":
    main()
