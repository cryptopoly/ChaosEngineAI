---
title: Progress files
summary: How per-stream progress notes work.
---
# Progress files

One file per work stream. They exist so a **fresh agent can resume** after an outage, a dropped connection or a credit stall, without asking anyone.

Rules:

- Say what is **done**, what is **not**, and **how to resume** (exact commands, branch, paths).
- The stream's agent writes only its own file. The lead owns [status](../status.md), [decisions](../decisions.md) and the built site.
- Update it at every checkpoint, not at the end.
- Start from [the template](_template.md): copy it to `progress/<stream>.md`. (This project works in the main checkout; there are no per-agent worktrees, see [rules](../rules.md).)

## Streams

| Stream | File | Status | Owner |
|---|---|---|---|
| Release v0.9.9 runs: DDTree on 27B, LTX-2.5, block 8 | [runs-2026-10-29.md](runs-2026-10-29.md) | in progress | Claude (Opus) |
| Models page merge, fit badges, recommended settings, size limits (team, branch `feat/models-fit`) | progress files `fit-backend.md`, `limits.md`, `settings-rec.md`, `map-models-pages.md` and later `ui-core.md`, `ui-video.md`, `ui-chat.md` in this folder (linked once they exist) | in progress | Claude + agents, plan in `docs/plans/models-fit.md` |
| RAM estimates (older stream) | [fix-ram-estimates](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/progress/fix-ram-estimates.md) | done | Claude |
