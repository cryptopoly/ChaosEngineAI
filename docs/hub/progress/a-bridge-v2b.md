---
title: Progress A-bridge v2b
status: review
---
# Progress: A-bridge v2 (second wave)

Updated: 2026-11-07. Owner: agent. Branch: `feat/agent-bridge-v2b` (off `feat/home-studio`).

## Goal

Register the clone tools, the V-story tool updates, A6 `music_library`, A8 resources and prompts, A7 progress.
Not in scope: A5 (confirmed download), A10 (remote routing).

## Done

- Clone tools registered (`list_cloned_voices`, `speak_cloned`) behind the `clone` scope: scope check, then
  `clone_allowed`. Trust gap closed: `clone_allowed` now also refuses a caller holding an agent token (main or
  read-only) that lacks `clone`, so the Voice setting alone is not enough.
- Storyboard: `storyboard_captions` (generate, export_srt, set, add, edit, remove), `storyboard_export_otio`,
  `storyboard_render` passes `notify` and returns `extras` (with downloads), `warnings`, and on failure the
  resume hint; `edit_project` export booleans documented. `add_audio` forces `clonedVoice` from the WAV tag.
- `music_library` (list, get, rename, delete).
- Resources (system status, music library and track, recent gallery outputs, storyboard project) and prompts
  (`short_film_from_brief`, `poster_set`, `soundtrack_for_project`), paged with a cursor.
- Progress: `notifications/progress` over a short event stream only when a `progressToken` is sent; only
  `wait_for_job` reports.
- Found and fixed: the app-token middleware answered 401 to every agent token on `/api/agent/uploads` and
  `/api/agent/files` on a real backend (the tests bypass it). Those two paths now go to their own check.
- Tests: `tests/test_mcp_bridge_v2b.py` (21), contract and golden updated, phase 17 extended (9 checks, passes
  live on port 8882 with a throwaway HOME).
- Tool count: 24 (19 before; plus 5: list_cloned_voices, speak_cloned, storyboard_captions,
  storyboard_export_otio, music_library).

## Not done / not verified

- No real cloned speech through an agent (no cloned voice or watermark component on this machine).
- Progress stream tested with the test client and a raw urllib read, not with a real MCP client.
- OTIO and captions tools tested against fake clips, not a real film.
- Gallery resource lists files by modification time only (no prompts or ids).

## How to resume

1. `git switch feat/agent-bridge-v2b`.
2. `.venv/bin/python -m pytest tests/test_mcp_*.py tests/test_job_queue.py tests/test_agent_files.py tests/test_storyboard_tools.py -q`.
3. Next: A5 once the approval banner decision is made; A10.
