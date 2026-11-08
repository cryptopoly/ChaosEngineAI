---
title: Progress busy-guard
status: draft
---
# Progress: busy-machine guard and gallery provenance

Updated: 2026-11-08. Owner: agent. Branch: `feat/busy-guard` (from `feat/home-studio`).

## Goal

Warn when a heavy job is running while the user chats or loads a model (slower, or not enough memory), and
label Storyboard-made stills and clips in the Image and Video galleries.

## Done

- Provenance: `helpers/provenance.py` (context variable set around the Storyboard's generate calls);
  `projectId`, `projectName`, `shotId` are written into the artifact JSON, the PNG chunk / MP4 comment and the
  gallery rows (file names only). Image and Video galleries show a "From project X" chip and a Project filter.
- `helpers/busy_advice.py`: pure `busy_advice(running_jobs, chat_model_footprint_gb, total_gb, available_gb)`
  returning `ok` / `slower` / `risky`; reads `HEAVY_JOBS.snapshot()` only. `GET /api/busy-advice`,
  `POST /api/models/load` and `POST /api/chat/generate` carry `advice`; the chat stream sends one `advice`
  event before the first token when not ok.
- UI: dismissible notice above the chat composer while a reply is generating; the Model load dialog shows the
  advice for the selected model and asks "Load anyway" when risky; `RunEstimateStrip` takes a `busy` note
  (Image, Video and Music Studios pass the risky verdict).
- `GET /api/storyboard/projects/{id}/films` and `FilmList.tsx` (read-only list of `exports/` films).
- `docs/usage/storyboard.md`: "Where your files go".

## Not done

- The Film list is not mounted in a Storyboard screen (patch note: add
  `<FilmList projectId={...} version={...} revealLabel={props.revealLabel} onReveal={props.onReveal} />` to
  the export panel). The Storyboard files were off limits for this stream.
- Storyboard's own estimate strips (Board, shot drawer, Export) do not show the busy note.
- Job peaks use per-kind defaults (`DEFAULT_JOB_PEAK_GB`) because queue tickets carry no peak figure; a
  `peakGb` field on the ticket in `job_queue.py` would make the estimate exact (patch note for that stream).

## How to resume

1. `cd` to the worktree and run `git status`.
2. `.venv/bin/python -m pytest tests/test_busy_advice.py tests/test_provenance.py tests/test_storyboard_films.py -q`
   and `npm test`.
3. Next step: mount `FilmList`, then a live check with a real video job running.

## Decisions to propose

- Show the busy note in Storyboard's strips too (recommend yes, after the Storyboard stream lands).
- Raise the "risky" share (90%) or keep it; it is `RISKY_SHARE` in `busy_advice.py`.
