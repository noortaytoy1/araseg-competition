"""UNTESTED reference implementation of one jury sitting without Claude Code.

Inlines the two doctrines and one packet into the adjudication prompt and calls the API
directly. Approximates the as-performed setup (which ran as a Claude Code agent with file
tools); see REPLICATION.md. Requires: pip install anthropic; ANTHROPIC_API_KEY set.
Cost warning: one document at maximum reasoning effort is a large request.
"""
import json, os, sys, re

def build_prompt(doctrine0: str, doctrine1: str, packet: dict, fmt: str, never: str) -> str:
    return f"""Two TRAINED juries sit together: Jury 0 and Jury 1. Each trained separately on this annotation team's training documents and wrote its own doctrine from its own graded mistakes. You hold both: when Jury 0 speaks, reason strictly from doctrine 0; when Jury 1 speaks, strictly from doctrine 1. They are colleagues with different training histories, not one voice twice.

DOCTRINE 0:
{doctrine0}

DOCTRINE 1:
{doctrine1}

FORMAT: {fmt}

YOUR DOCUMENT ({packet['doc_id']}):
{packet['text']}

The document arrives with an automatic system's boundary marks (¶) already drawn in. Treat them as an UNRELIABLE CHEAT SHEET: allowed to consult, never to trust, it makes both kinds of errors. A document whose marks already satisfy its law is FINISHED; unchanged is a common and correct verdict.

No limit on your reasoning depth:
1. Jury 0 gives its full stance from doctrine 0, what the text is, which register it belongs to, its sentence law, and every place the draft marks are wrong, argued from the text.
2. Jury 1 gives its own independent stance from doctrine 1, site by site.
3. They ARGUE every disagreement. Concede when the colleague's reading matches the annotators' policy better; rebut precisely when it does not. Where a doctrine carries a law earned from a graded mistake in THIS register, that law outranks instinct.
4. RECORD: only changes BOTH juries endorse survive. Unresolved means the draft stands. {never}

End your reply with exactly one JSON object on its own lines:
{{"doc_id":"{packet['doc_id']}","add":[{{"i":<index>,"w":"<token>"}}],"remove":[{{"i":<index>,"w":"<token>"}}],"argued":"<one line per argued site>"}}
add = token positions that end a sentence but carry no ¶; remove = positions carrying ¶ that do not end a sentence; copy w from the text at the moment you decide."""

def main():
    import anthropic  # pip install anthropic
    if len(sys.argv) != 6:
        print("usage: run_sitting_reference.py <doctrine_j0.md> <doctrine_j1.md> <packet.json> <format_txt> <out_dir>")
        sys.exit(1)
    d0 = open(sys.argv[1], encoding="utf-8").read()
    d1 = open(sys.argv[2], encoding="utf-8").read()
    packet = json.load(open(sys.argv[3], encoding="utf-8"))
    fmt = open(sys.argv[4], encoding="utf-8").read().strip()
    never = "Never remove the ¶ on the final token."
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-5",
        max_tokens=32000,
        thinking={"type": "enabled", "budget_tokens": 30000},
        messages=[{"role": "user", "content": build_prompt(d0, d1, packet, fmt, never)}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    m = re.search(r'\{"doc_id".*\}', text, re.S)
    if not m:
        print("no verdict JSON found in reply"); sys.exit(2)
    v = json.loads(m.group(0))
    out = os.path.join(sys.argv[5], v["doc_id"] + ".json")
    json.dump(v, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote", out)

if __name__ == "__main__":
    main()
