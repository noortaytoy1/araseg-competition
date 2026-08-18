"""
Connective-corrective augmentation for Arabic sentence segmentation.
=====================================================================

The model is unsure whether an Arabic connective word (و, ف, ثم, من, في, ...)
starts a NEW sentence or just sits in the MIDDLE of one. This script builds
extra training examples that teach it to decide by MEANING, not by the word:

  POSITIVE example  (there IS a boundary):
      Take two sentences from DIFFERENT documents (so they are unrelated),
      join them with a connective, and put a boundary right before the connective.

          [ ... end of sentence A ]  <BOUNDARY>  <connective>  [ sentence B ... ]
          labels: ...           A_last = 1        connective = 0     ...   B_last = 1

  NEGATIVE example  (there is NO boundary):
      Take ONE real sentence that already has a connective in the middle,
      and keep it exactly as it is (the connective stays label 0).

          [ ...  الرجال   <و>   النساء  ... ]        # one coherent sentence, no cut
          labels: ...      0     ...                 # the connective is NOT a boundary

The two look alike ("...X <connective> Y..."); only the meaning differs, so the
model has to learn coherence. Everything is built from the 174 TRAINING
documents only — nothing external.

Run:   python connective_aug.py NoPnx-NP --n-pos 300 --n-neg 700
Out:   data/connaug_<task>_train.jsonl   (the 174 real docs + the synthetic docs)
"""

import argparse
import json
import os
import random
from collections import Counter

# The Arabic connective / preposition words the model confuses.
CONNECTIVE_WORDS = ["و", "ف", "ثم", "من", "في", "وكان", "وقد", "ولكن",
                    "فقال", "فإن", "كما", "لكن", "أو", "بل", "حتى"]
CONNECTIVE_WORDS_SET = set(CONNECTIVE_WORDS)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")


def read_training_documents(task):
    """Read the training documents. Each one is a dict with keys
    'doc_id', 'tokens' (list of words), and 'labels' (list of 0/1)."""
    path = os.path.join(DATA_DIR, task + "_train.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_document_into_sentences(document):
    """Cut one document into its sentences.
    A sentence is a run of words that ends on a boundary (a label of 1)."""
    tokens = document["tokens"]
    labels = document["labels"]
    sentences = []
    start = 0
    for i, label in enumerate(labels):
        if label == 1:
            sentences.append((tokens[start:i + 1], labels[start:i + 1]))
            start = i + 1
    return sentences


def collect_all_sentences(documents):
    """Return every complete sentence in the corpus, remembering which document
    it came from. Each item is (document_index, sentence_words, sentence_labels)."""
    all_sentences = []
    for document_index, document in enumerate(documents):
        for words, labels in split_document_into_sentences(document):
            if labels and labels[-1] == 1:          # a real, complete sentence
                all_sentences.append((document_index, words, labels))
    return all_sentences


def find_internal_connectives(words, labels):
    """Return the positions of connective words that sit INSIDE the sentence
    (not the first word) and are NOT a boundary (label 0)."""
    return [i for i in range(1, len(words))
            if words[i] in CONNECTIVE_WORDS_SET and labels[i] == 0]


def build_connective_frequency_pool(negative_sentences):
    """Build a list of connectives where each word appears as often as it really
    does inside sentences. We prepend positive examples using this same pool, so
    positives and negatives share the SAME mix of connectives -- that stops the
    model from cheating on the connective word and forces it to use meaning."""
    counts = Counter()
    for words, labels, positions in negative_sentences:
        for i in positions:
            counts[words[i]] += 1
    pool = [word for word, how_many in counts.items() for _ in range(how_many)]
    return pool or list(CONNECTIVE_WORDS)          # fall back to the plain list if empty


def make_positive_example(sentence_a, sentence_b, connective_pool, rng):
    """Join two UNRELATED sentences with a connective, with a boundary between
    them (right before the connective). Returns (words, labels)."""
    a_words, a_labels = sentence_a
    b_words, b_labels = sentence_b

    if b_words and b_words[0] in CONNECTIVE_WORDS_SET:
        # B already begins with a connective -> use that as the joining word.
        right_words = list(b_words)
        right_labels = list(b_labels)
        right_labels[0] = 0                         # a sentence's first word is never the boundary
    else:
        # Otherwise put a connective in front of B, chosen from the frequency pool.
        connective = rng.choice(connective_pool)
        right_words = [connective] + list(b_words)
        right_labels = [0] + list(b_labels)

    words = list(a_words) + right_words
    labels = list(a_labels) + right_labels          # a_labels[-1] is 1 -> boundary before connective
    return words, labels


def make_negative_example(sentence_words, sentence_labels):
    """A negative example is just a real sentence kept whole and unchanged --
    its internal connective already carries label 0 (no boundary)."""
    return list(sentence_words), list(sentence_labels)


def build_examples(all_sentences, number_of_positives, number_of_negatives, rng):
    """Build the positive and negative examples, shuffled together.
    Each example is a (words, labels) pair that ends on a boundary."""
    # Negatives: real sentences that contain an internal connective.
    negative_sentences = []
    for _, words, labels in all_sentences:
        positions = find_internal_connectives(words, labels)
        if positions:
            negative_sentences.append((words, labels, positions))

    connective_pool = build_connective_frequency_pool(negative_sentences)

    examples = []

    for _ in range(number_of_positives):
        document_a, a_words, a_labels = rng.choice(all_sentences)
        document_b, b_words, b_labels = rng.choice(all_sentences)
        while document_b == document_a:             # keep drawing until B is from ANOTHER document
            document_b, b_words, b_labels = rng.choice(all_sentences)
        examples.append(make_positive_example((a_words, a_labels),
                                              (b_words, b_labels),
                                              connective_pool, rng))

    for _ in range(number_of_negatives):
        words, labels, _positions = rng.choice(negative_sentences)
        examples.append(make_negative_example(words, labels))

    rng.shuffle(examples)
    return examples


def pack_examples_into_documents(examples, real_document_lengths, task, rng):
    """Glue several examples together into synthetic documents, each one sized
    like a real document (so the training data keeps its normal shape). Every
    example ends on a boundary, so every synthetic document does too."""
    documents = []
    index = 0
    document_number = 0
    while index < len(examples):
        target_length = rng.choice(real_document_lengths)
        words, labels = [], []
        while index < len(examples) and len(words) < target_length:
            example_words, example_labels = examples[index]
            words += example_words
            labels += example_labels
            index += 1
        documents.append({
            "doc_id": "connaug_%s_%d" % (task, document_number),
            "tokens": words,
            "labels": labels,
            "text": " ".join(words),
            "label_str": "".join(str(label) for label in labels),
        })
        document_number += 1
    return documents


def self_check(synthetic_documents):
    """Verify the labels are exactly what we intended, and print a summary."""
    boundary_before_connective = 0
    broken = 0
    internal_connective = 0
    for document in synthetic_documents:
        words, labels = document["tokens"], document["labels"]
        assert len(words) == len(labels), "words and labels must line up"
        assert labels[-1] == 1, "every document must end on a boundary"
        for i in range(1, len(words)):
            if words[i] in CONNECTIVE_WORDS_SET:
                if labels[i - 1] == 1:               # a boundary sits right before this connective
                    boundary_before_connective += 1
                    if labels[i] != 0:               # the connective itself must NOT be a boundary
                        broken += 1
                else:
                    internal_connective += 1
    print("self-check: %d boundary-before-connective (broken: %d) | %d internal connectives"
          % (boundary_before_connective, broken, internal_connective))
    assert broken == 0, "a connective was wrongly labelled as a boundary"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", default="NoPnx-NP")
    parser.add_argument("--n-pos", type=int, default=300, help="how many POSITIVE examples to build")
    parser.add_argument("--n-neg", type=int, default=700, help="how many NEGATIVE examples to build")
    parser.add_argument("--seed", type=int, default=42, help="random seed (for reproducibility)")
    parser.add_argument("--frozen", action="store_true",
                        help="write to an IMMUTABLE dated path under data/frozen/ "
                             "and REFUSE to overwrite it. Use this for every A/B "
                             "run so seeds cannot train on a mid-run-clobbered "
                             "file (the shared-mutable-file race). The default "
                             "(mutable data/connaug_<task>_train.jsonl) is kept "
                             "for backward compatibility but should NOT be used "
                             "as an A/B training path.")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    real_documents = read_training_documents(args.task)
    all_sentences = collect_all_sentences(real_documents)
    real_document_lengths = [len(document["tokens"]) for document in real_documents]

    examples = build_examples(all_sentences, args.n_pos, args.n_neg, rng)
    synthetic_documents = pack_examples_into_documents(
        examples, real_document_lengths, args.task, rng)
    self_check(synthetic_documents)

    # The final training set = the 174 real documents (unchanged) + the synthetic ones.
    output_documents = real_documents + synthetic_documents
    if args.frozen:
        # IMMUTABLE dated path; refuse to clobber an existing file so two A/B
        # arms / seeds can never train on different data written to the same path.
        import datetime
        frozen_dir = os.path.join(DATA_DIR, "frozen")
        os.makedirs(frozen_dir, exist_ok=True)
        stamp = datetime.date.today().isoformat()
        output_path = os.path.join(
            frozen_dir, "connaug_%s_%s_seed%d.jsonl" % (args.task, stamp, args.seed))
        if os.path.exists(output_path):
            raise SystemExit(
                "REFUSE to overwrite frozen file %s (immutable A/B data). "
                "Delete it by hand or change the seed/date if you truly mean to "
                "regenerate." % output_path)
    else:
        output_path = os.path.join(DATA_DIR, "connaug_%s_train.jsonl" % args.task)
    with open(output_path, "w", encoding="utf-8") as f:
        for document in output_documents:
            f.write(json.dumps(document, ensure_ascii=False) + "\n")

    print("wrote %d documents (%d real + %d synthetic) -> %s"
          % (len(output_documents), len(real_documents), len(synthetic_documents), output_path))
    if args.frozen:
        print("FROZEN: pass this exact path as --train-jsonl and guard every A/B "
              "seed with `[ \"$(wc -l < %s)\" -eq %d ]` and "
              "`--expect-docs %d`."
              % (output_path, len(output_documents), len(output_documents)))


if __name__ == "__main__":
    main()
