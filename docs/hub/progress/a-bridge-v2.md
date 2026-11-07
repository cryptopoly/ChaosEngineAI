---
title: Progress A-bridge v2
status: draft
---
# Progress: A-bridge v2 (first wave)

Updated: 2026-11-07. Owner: agent. Branch: `feat/agent-bridge-v2` (off `feat/home-studio`).

## Goal

A1 tool quality, A2 contract tests, A3 persisted jobs, A9 queue view, A11 SKILL refresh, A12 phase 17, A4 scopes.

## Done

- A1: annotations and titles on all 19 tools (`mcp_bridge/scopes.py`), short verb-first descriptions, refreshed INSTRUCTIONS, `upload_image` renamed `upload_file` with alias.
- A2: `tests/test_mcp_contract.py` plus golden `tests/golden/mcp_tools_list.json`.
- A3: job records under `agent_jobs/` (7 days), `interrupted` on restart, `list_jobs`, running cancel for speech and transcripts (kills the voice worker), 20 unfinished jobs per token.
- A9: `system_status.queue.jobs` with positions and estimates.
- A4: scopes read, generate, files, download, clone; migration; read-only token; remote callers lose clone unless ticked; card disclosure.
- A11 SKILL.md, docs/features/agent-bridge.md, A12 phase 17 (passes on a backend at port 8882).

## Not done

- A5, A7, A8, A10 (second wave); A6 tools. `download` and `clone` scopes exist but no tool uses them yet.

## How to resume

1. `cd` to the worktree, `git status`.
2. `.venv/bin/python -m pytest tests/test_mcp_*.py tests/test_job_queue.py tests/test_agent_files.py tests/test_storyboard_tools.py -q`
3. Next: A5 once the approval banner decision is made.

## Decisions to propose

- Voice cancel kills the shared voice worker (a speech started by the Voice tab at the same moment would also fail). Accept, or add a per-request cancel to the voice worker.
