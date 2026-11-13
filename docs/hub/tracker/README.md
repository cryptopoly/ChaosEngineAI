---
title: Follow-up tracker
order: 1
summary: Every FU row from FOLLOW_UPS.md, split into open and closed.
---

# Follow-up tracker

Generated from the Follow-Ups Tracker in `FOLLOW_UPS.md` by `tools/hub_tracker.py`; edit the row there, not here.

- [Open and live follow-ups](open.md): **93** rows.
- [Closed follow-ups](closed.md): **80** rows.

## Newest ten

| Row | Item | Status |
|---|---|---|
| [FU-175](open.md) | FLUX.2 Dev: catalog size wrong (64 GB) and the download bar showed the whole repo (154.08) | Open (2026-11-13): size fixed in code (3f534add) |
| [FU-174](open.md) | Idle GPU speech workers slow image and video jobs; the idle-unload setting missed the Qwen | Fixed and verified through the installed app (2026-11-13, installer built at 3f534add). |
| [FU-173](open.md) | Installed app runs LTX-2.5 4-bit and FLUX.2 Klein slower than the dev venv on the same car | Fixed and verified through the installed app (2026-11-13, installer built at a8b0051e). |
| [FU-172](open.md) | Qwen3.8-27B + DFlash2 on the vLLM lane does not fit a 24 GB card; vLLM lane ignores the th | Open (2026-11-13): measured on an RTX 4090 (WSL2, vLLM 0.33). |
| [FU-171](open.md) | TriAttention on the vLLM lane: works only on vLLM 0.18 or older; the app now says so and w | Open: root cause found, compaction working on vLLM 0.18.1, app honest on newer vLLM (2026-11-13). |
| [FU-170](open.md) | cargo test port race in backend::tests::runtime_info_keeps_a_soft_note_while_the_port_is_s | Open; low priority. |
| [FU-169](open.md) | scripts/check-upstreams.py: report every pinned upstream against its newest release (D-78) | Shipped 2026-11-11. |
| [FU-168](open.md) | TensorFold 1.x: build the family table from capabilities --json, add newer families | Open: the 14 families are now in the tables (2026-11-11) |
| [FU-167](open.md) | Image, video and audio speed-ups beyond the diffusion caches | Researched 2026-11-11 |
| [FU-166](open.md) | MiniMax H3 (video and image) and newer image/video models: scan, supersedes the FU-082 / F | Researched 2026-11-10 |
