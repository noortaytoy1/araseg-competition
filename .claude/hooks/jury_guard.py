"""PreToolUse guard for jury exam agents (fail-closed for jury sessions).
A jury agent may read/write ONLY: papereval/<track>/docs/doc_*.json, papereval/<track>/exam_out_*/doc_*.json,
papereval/examples_nonp/examples_j*.md (Exp 2), the three real doctrine pairs (real ablation only), and its own
temp scratchpad. Every other file access is blocked at the tool layer. Non-jury sessions are untouched."""
import json, sys, os, re
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
tool = data.get("tool_name", "")
inp = data.get("tool_input", {}) or {}
tp = data.get("transcript_path", "")
is_jury = False
try:
    if tp and os.path.exists(tp):
        with open(tp, encoding="utf-8", errors="ignore") as f:
            head = f.read(30000)
        is_jury = ("exam sitting" in head) and ("UNRELIABLE CHEAT SHEET" in head)
except Exception:
    is_jury = False
if not is_jury:
    sys.exit(0)

def norm(p):
    return p.replace("\\", "/").lower()

ALLOW = [
    re.compile(r"scratch_exo/papereval/(nopnx-np|nopnx-pa|np|pa)/docs/doc_[0-9a-f]+\.json$"),
    re.compile(r"scratch_exo/papereval/(nopnx-np|nopnx-pa|np|pa)/exam_out[_a-z0-9]*/doc_[0-9a-f]+\.json$"),
    re.compile(r"scratch_exo/papereval/examples_nonp/examples_j[01]\.md$"),
    re.compile(r"scratch_exo/retrain3/nonp/doctrine_j[01]\.md$"),
    re.compile(r"scratch_exo/retrain4/nopa/doctrine_j[01]\.md$"),
    re.compile(r"scratch_exo/np_train_work/doctrine_np_juror[01]\.md$"),
    re.compile(r"/scratchpad/"),
]
def allowed(p):
    q = norm(p)
    return any(a.search(q) for a in ALLOW)

def block(reason):
    print(json.dumps({"decision": "block", "reason": "jury guard: " + reason}))
    sys.exit(0)

if tool in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit"):
    p = inp.get("file_path") or inp.get("path") or inp.get("notebook_path") or ""
    if p and not allowed(p):
        block("file access outside the exam packet/output dirs is denied: " + p)
elif tool in ("Glob", "Grep"):
    p = inp.get("path") or ""
    if not p or not ("papereval" in norm(p) or "/scratchpad/" in norm(p)):
        block("search outside the exam dirs is denied")
elif tool == "Bash":
    cmd = norm(inp.get("command", ""))
    bad = ["heldout", "locked", "_train.jsonl", "_dev", "_test", "subs_blind", "/data/", "data\\", ".zip",
           "retrain", "np_train_work", "doctrine", "curl ", "wget ", "invoke-webrequest", "git ", "python "]
    if any(b in cmd for b in bad):
        block("shell access to protected data or tools is denied")
sys.exit(0)
