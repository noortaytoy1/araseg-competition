# Turnkey replication with Claude Code (no setup beyond install)

1. Install Claude Code (https://claude.com/claude-code) and sign in with an account that has Opus 5 access.
2. Open a terminal IN the kit folder (the one containing `doctrines/`, `docs/`, `manifest.json`, `out/`) and run `claude`.
3. Paste the ENTIRE contents of `orchestrator_prompt.txt` as one message. That is it.
   The session reads the manifest, runs the sittings 4-at-a-time on Opus at maximum effort, banks one
   verdict JSON per document into `out/`, and relaunches anything that dies. If the session itself dies or
   hits a usage limit, start a new session and paste the same message again; it resumes from `out/`.
4. When `out/` has one file per manifest document, zip `out/` and send it back for scoring.

For the API-key path without Claude Code, see `run_exam.py`.
