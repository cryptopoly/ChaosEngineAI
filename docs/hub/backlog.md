---
title: Backlog and issues
order: 3
summary: Reminders for Dan, then the work in flight and the full follow-up tracker.
---
# Backlog and issues

Priority: P1 urgent, P2 next, P3 later. The complete list of follow-ups (129+ rows with the full reasoning and measurements) is the [tracker](tracker/README.md), generated from `CLAUDE.md`; this page is the short view.

## Reminders for Dan

- [ ] **Before re-tagging `v0.9.9`:** launch the desktop app once on each platform on Tauri 2.12 (never launched there, [FU-121](tracker/open.md)); the release workflow's tauri-action upgrade (Dependabot PR 82) needs a release dry run.
- [ ] **Hugging Face token:** LTX-2.5 and the FLUX 3 Klein rows are gated; the token on this Mac needs gated-read permission ([FU-115](tracker/open.md), [FU-128](tracker/closed.md)).
- [ ] **Native-speaker check** of the German voice strings (`Sprach-` prefix can read as "language"), [FU-119](tracker/open.md).
- [ ] **Listen to a voice call** with a real microphone and speakers (echo is untested), [FU-117](tracker/open.md).
- [ ] **Run the packaged macOS build once:** microphone prompt, `backend/scripts/` in the archive, one Setup install ([FU-104](tracker/closed.md), [FU-117](tracker/open.md)).

## In flight (v0.9.9)

- [ ] **P1, in progress.** PR 119: CI Windows test fixed locally, push held ([decisions](decisions.md) item 1).
- [ ] **P1, todo.** Re-tag `v0.9.9` after merge ([decisions](decisions.md) item 2).
- [ ] **P2, in progress.** DDTree speed on a 27B-class target and block 8 versus 16 on Qwen3.6-27B ([progress](progress/runs-2026-10-29.md)).
- [ ] **P2, in progress.** LTX-2.5 live run, then the default LTX row ([decisions](decisions.md) item 3).
- [ ] **P2, in progress.** Sound in Video Studio: icon, unmuted result window, prompt hint ([FU-130](tracker/open.md)).
- [x] **P2, done.** Project hub scaffolded ([FU-131](tracker/open.md)).
- [ ] **P3, todo.** Models page consolidation after the release ([FU-129](tracker/open.md)).

## Next up (not started)

- [ ] **P2.** Rename the other-app strings in the import-models dialog ([decisions](decisions.md) item 4).
- [ ] **P2.** Backend messages are English only; `backend_service/locales/*.po` are empty ([FU-119](tracker/open.md)).
- [ ] **P2.** DDTree: finish the tree on hybrid and rotating-cache models, or take upstream's own verify (needs the dflash-mlx migration, [FU-057](tracker/open.md)).
- [ ] **P3.** Image-to-video and sound-to-video for the LTX engine (the video request has no image field yet).
- [ ] **P3.** Windows and Linux runs of everything marked "not run" in the tracker (the largest single gap).

## Issues

See [status](status.md#known-breakage) for known breakage; each tracker row lists its own open points.
