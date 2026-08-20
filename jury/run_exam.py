"""Framework-free batch runner for a jury exam (UNTESTED reference).

Needs: pip install anthropic, and ANTHROPIC_API_KEY in the environment.
Loops the manifest, one API call per document (doctrines + packet inlined), banks verdicts
to out/, resumes by skipping banked files, retries once on failure.
COST WARNING: each document is a large maximum-effort request; a 50-doc kit is a
substantial spend. Usage: python run_exam.py <kit_folder>
"""
import json, os, re, sys, time

FORMAT = ("Punctuation REMOVED and paragraph marks REMOVED, plain words only. The final "
          "token of a document always ends a sentence. ⟨i⟩ means the next token "
          "is token number i (an index anchor, not part of the text).")
NEVER = "Never remove the ¶ on the final token."


def prompt_for(d0, d1, packet):
    did = packet["doc_id"]
    return (
        "Two TRAINED juries sit together: Jury 0 and Jury 1. Each trained separately on "
        "this annotation team's training documents and wrote its own doctrine from its own "
        "graded mistakes. You hold both: when Jury 0 speaks, reason strictly from doctrine "
        "0; when Jury 1 speaks, strictly from doctrine 1. They are colleagues with "
        "different training histories, not one voice twice.\n\n"
        "DOCTRINE 0:\n" + d0 + "\n\nDOCTRINE 1:\n" + d1 + "\n\n"
        "FORMAT: " + FORMAT + "\n\n"
        "YOUR DOCUMENT (" + did + "):\n" + packet["text"] + "\n\n"
        "The document arrives with an automatic system's boundary marks (¶) already "
        "drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to "
        "trust, it makes both kinds of errors. A document whose marks already satisfy its "
        "law is FINISHED; unchanged is a common and correct verdict.\n\n"
        "No limit on your reasoning depth:\n"
        "1. Jury 0 gives its full stance from doctrine 0, what the text is, which register "
        "it belongs to, its sentence law, and every place the draft marks are wrong, "
        "argued from the text.\n"
        "2. Jury 1 gives its own independent stance from doctrine 1, site by site.\n"
        "3. They ARGUE every disagreement. Concede when the colleague's reading matches "
        "the annotators' policy better; rebut precisely when it does not. Where a doctrine "
        "carries a law earned from a graded mistake in THIS register, that law outranks "
        "instinct.\n"
        "4. RECORD: only changes BOTH juries endorse survive. Unresolved means the draft "
        "stands. " + NEVER + "\n\n"
        "End your reply with exactly one JSON object on its own lines:\n"
        '{"doc_id":"' + did + '","add":[{"i":<index>,"w":"<token>"}],'
        '"remove":[{"i":<index>,"w":"<token>"}],"argued":"<one line per argued site>"}\n'
        "add = token positions that end a sentence but carry no ¶; remove = positions "
        "carrying ¶ that do not end a sentence; copy w from the text at the moment "
        "you decide."
    )


def main():
    import anthropic
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    d0 = open(os.path.join(root, "doctrines", "doctrine_j0.md"), encoding="utf-8").read()
    d1 = open(os.path.join(root, "doctrines", "doctrine_j1.md"), encoding="utf-8").read()
    man = json.load(open(os.path.join(root, "manifest.json"), encoding="utf-8"))
    outdir = os.path.join(root, "out")
    os.makedirs(outdir, exist_ok=True)
    client = anthropic.Anthropic()
    docs = man["docs"]
    for k, did in enumerate(docs, 1):
        outp = os.path.join(outdir, did + ".json")
        if os.path.exists(outp):
            print(f"[{k}/{len(docs)}] {did} banked, skip")
            continue
        packet = json.load(open(os.path.join(root, "docs", did + ".json"), encoding="utf-8"))
        for attempt in (1, 2):
            try:
                resp = client.messages.create(
                    model="claude-opus-5",
                    max_tokens=32000,
                    thinking={"type": "enabled", "budget_tokens": 30000},
                    messages=[{"role": "user", "content": prompt_for(d0, d1, packet)}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                m = re.search(r'\{"doc_id".*\}', text, re.S)
                if not m:
                    raise ValueError("no verdict JSON in reply")
                v = json.loads(m.group(0))
                json.dump(v, open(outp, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"[{k}/{len(docs)}] {did} done (+{len(v.get('add', []))}/-{len(v.get('remove', []))})")
                break
            except Exception as e:
                print(f"[{k}/{len(docs)}] {did} attempt {attempt} failed: {e}")
                if attempt == 2:
                    print("  giving up on this doc; rerun the script later to retry missing docs")
                time.sleep(20)


if __name__ == "__main__":
    main()
