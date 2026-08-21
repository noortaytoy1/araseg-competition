"""Build sealed jury exam packets for one track from public corpus text + released draft rows.

The renderer below reproduces the exact packets used in the paper's evaluations
(verified byte-for-byte against the originals on 10 documents across both formats).

Usage:
  python jury/build_packets.py --track NoPnx-NP --split test \
      --data data/NoPnx-NP_test.jsonl --draft jury/draft_rows/NoPnx-NP.json --out kit_NoPnx-NP

Produces: <out>/docs/<doc_id>.json ({"doc_id","text"}; NO labels), <out>/manifest.json
(order + sittings of 4), and empty <out>/out/. Then run the exam per jury/REPLICATION.md.
"""
import argparse, json, os

def render(tokens, row):
    parts = []
    for i, (t, m) in enumerate(zip(tokens, row)):
        if i % 10 == 0:
            parts.append("⟨" + str(i) + "⟩")
        parts.append("<NL>" if t == "\n" else t)
        if m == 1:
            parts.append("¶")
    return " ".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["NoPnx-NP", "NoPnx-PA", "NP", "PA"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", required=True, help="the split's jsonl (fetch with src/data.py)")
    ap.add_argument("--draft", required=True, help="draft rows json {rows: {doc_id: '0101...'}}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hasnl = a.track in ("NoPnx-PA", "PA")
    rows = json.load(open(a.draft, encoding="utf-8"))["rows"]
    os.makedirs(os.path.join(a.out, "docs"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "out"), exist_ok=True)
    # the kit carries its own doctrine pair so the orchestrator prompt is self-contained
    pair_dir = {"NoPnx-NP": "NoPnx-NP", "NoPnx-PA": "NoPnx-PA", "NP": "Pnx-NP-PA", "PA": "Pnx-NP-PA"}[a.track]
    os.makedirs(os.path.join(a.out, "doctrines"), exist_ok=True)
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    for j in (0, 1):
        src = os.path.join(here, "doctrines", pair_dir, f"doctrine_jury{j}.md")
        shutil.copyfile(src, os.path.join(a.out, "doctrines", f"doctrine_j{j}.md"))
    order = []
    for line in open(a.data, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        did = d["doc_id"]
        if did not in rows:
            continue
        T = d["tokens"]
        row = [int(c) for c in rows[did]]
        row[-1] = 1
        if hasnl:
            for i, t in enumerate(T):
                if t == "\n":
                    row[i] = 0
                    if i > 0 and T[i - 1] != "\n":
                        row[i - 1] = 1
        packet = {"doc_id": did, "text": render(T, row)}
        json.dump(packet, open(os.path.join(a.out, "docs", did + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        order.append(did)
    sittings = [order[i:i + 4] for i in range(0, len(order), 4)]
    json.dump({"track": a.track, "docs": order, "sittings": sittings},
              open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"{a.track}/{a.split}: {len(order)} packets, {len(sittings)} sittings -> {a.out}")
    print("Packets contain doc_id + text only; labels never enter a packet.")

if __name__ == "__main__":
    main()
