---
title: Progress V-story
status: active
---
# Progress: V-story (Storyboard v2)

Updated: 2026-11-07. Owner: agent. Branch: `feat/storyboard-v2` (cut from `feat/home-studio`).

## Goal

Captions (generate, edit, SRT, burn-in), a resumable render, OpenTimelineIO export and lane stems, in the
existing Mixer Export panel and a closed Captions disclosure. Plan: `docs/plans/remaining-and-v2.md` 2.3.

## Done

- Backend: captions model and ops (schema 2, migration 1 to 2), `captions.py`, burn-in plus SRT sidecar, resume
  manifest and leftovers cleanup, `longrun.py` (keep-awake, banner), `otio_export.py`, lane stems.
- Frontend: Captions disclosure, caption and stem switches, resume card, editor export, extras list; en keys
  with null placeholders in the other nine locales.
- Tests: `tests/test_storyboard_v2.py` (real ffmpeg, a real SIGKILL and resume), vitest cases, E2E phase 16.
- Docs: `docs/usage/storyboard.md`, `docs/features/storyboard.md`.

## Not done

- Nobody has watched a whole captioned film or listened to the stems; one burned-in frame was looked at.
- The .otio has not been opened in Resolve or Premiere.
- Captions have no style settings; times cannot be edited by hand. No CSS was added for the new Captions rows
  (`src/styles.css` is not in this stream's files).
- Speech-to-text was exercised end to end on a sine tone only (no real speech).
- Banner and keep-awake are backend helpers (no Tauri notification plugin exists). Not tested on Windows or Linux.
- Bridge: `storyboard_render` should pass the new options (see patch notes in the final report).

## How to resume

1. `cd` to the worktree, `git status`.
2. `.venv/bin/python -m pytest tests/test_storyboard_*.py -q`, `npm test`, `npx tsc --noEmit`, `npm run i18n:validate`.
3. Live check on a spare port: `CHAOSENGINE_PORT=8888 .venv/bin/python scripts/e2e_test_suite.py --phases 16`.

## Decisions to propose

- Keep notify and keep-awake in the backend until the shell gets a notification plugin (then move the banner).
- Stems are dry (no ducking, no loudness); say so in the UI copy if users expect a mixed stem.
