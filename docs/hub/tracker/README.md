---
title: Follow-up tracker
order: 1
summary: Every FU row from FOLLOW_UPS.md, split into open and closed.
---

# Follow-up tracker

Generated from the Follow-Ups Tracker in `FOLLOW_UPS.md` by `tools/hub_tracker.py`; edit the row there, not here.

- [Open and live follow-ups](open.md): **149** rows.
- [Closed follow-ups](closed.md): **110** rows.

## Newest ten

| Row | Item | Status |
|---|---|---|
| [FU-262](open.md) | Install size labels for the WSL vLLM bridge, DFlash (CUDA) and TriAttention said about 2 G | Done (2026-11-19, test/rtx4090) |
| [FU-261](open.md) | WSL vLLM bridge: use the library folder, serve DFlash, show it in the Boost Pack | Done (2026-11-19, test/rtx4090) |
| [FU-260](open.md) | WSL vLLM bridge: GPU memory share, reasoning split, dead attention variable | Done (2026-11-19, test/rtx4090), live-checked with Qwen3-0.6B |
| [FU-258](open.md) | /v1/chat/completions with stream: true drops tool calls | Done (2026-11-19, test/rtx4090), not run live |
| [FU-257](closed.md) | Tests that make symlinks fail on Windows without Developer Mode | Closed 2026-11-18: the owner turned Developer Mode on (D-96) |
| [FU-256](open.md) | Host-bound tests that fail on Windows by design of the test | Next pass on the Windows or Linux box, or when each spec is touched |
| [FU-255](open.md) | A Windows path written unescaped in a model's tool call gets the wrong advice | Done (2026-11-19, test/rtx4090) |
| [FU-254](open.md) | Native-speaker review of the review's 891 machine translations | Before the release notes claim the nine locales |
| [FU-253](open.md) | npm run i18n:extract cannot be the "zero new keys" check | Before the next release checklist run |
| [FU-252](open.md) | Backend-made text is not translated | When the activity feed or the catalogue labels are next touched (about half a day) |
