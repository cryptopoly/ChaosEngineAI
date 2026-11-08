---
title: Follow-up tracker
order: 1
summary: Every FU row from FOLLOW_UPS.md, split into open and closed.
---

# Follow-up tracker

Generated from the Follow-Ups Tracker in `FOLLOW_UPS.md` by `tools/hub_tracker.py`; edit the row there, not here.

- [Open and live follow-ups](open.md): **62** rows.
- [Closed follow-ups](closed.md): **78** rows.

## Newest ten

| Row | Item | Status |
|---|---|---|
| [FU-142](open.md) | Frame interpolation at export for 48 / 50 / 60 fps (D-43, D-64) | Open (2026-11-08): not built |
| [FU-141](open.md) | Voice rework backend: parts not yet built (D-58) | Open (2026-11-08): backend landed on feat/voice-rework |
| [FU-140](open.md) | Lip sync for storyboard shots (D-57) | Open (2026-11-08): banked on the backlog, not started. |
| [FU-139](open.md) | Music on CUDA (Windows and Linux) and XL on CUDA | Open (2026-11-07): not built |
| [FU-138](closed.md) | pytorch-wavelets dependency in the music pack | Shipped 2026-11-07 (D-37): dropped. |
| [FU-137](open.md) | Linux ARM64 build (NVIDIA DGX Spark class) | Phase 1 coded 2026-11-02 on release/v0.9.10, not yet built in CI |
| [FU-136](open.md) | Boot-time MLX probe that fails once switched MLX off for the whole session | Fixed 2026-10-30 (cause of the one failure not found) |
| [FU-135](open.md) | DFlash on hybrid Qwen3.5 / 3.6 / Next targets produced garbage text on the release MLX sta | Fixed 2026-10-29 (root cause found, patched, self-checked at load) |
| [FU-134](open.md) | Speculative decoding on creative prose is 2 to 4 times slower than plain | Open (2026-10-29), found by the 27B-class check in FU-112 |
| [FU-133](open.md) | "Best fit for your hardware" badges, recommended Studio settings, per-model size limits | Built 2026-10-29 on feat/models-fit (team of agents) |
