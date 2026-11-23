---
title: Follow-up tracker
order: 1
summary: Every FU row from FOLLOW_UPS.md, split into open and closed.
---

# Follow-up tracker

Generated from the Follow-Ups Tracker in `FOLLOW_UPS.md` by `tools/hub_tracker.py`; edit the row there, not here.

- [Open and live follow-ups](open.md): **266** rows.
- [Closed follow-ups](closed.md): **123** rows.

## Newest ten

| Row | Item | Status |
|---|---|---|
| [FU-551](open.md) | HyperQwen head-to-head on the 4090; prompt drafting | Next speculative-decoding pass |
| [FU-550](open.md) | MoE expert offload in the llama.cpp engine (big MoE on a 24 GB GPU + system RAM) | Next llama.cpp engine pass |
| [FU-549](open.md) | SenseNova-Loop-9B AWQ on the vLLM lane (Linux) | When vLLM PR #64832 merges |
| [FU-548](open.md) | SenseNova-Loop-9B on MLX | After FU-547, or when an MLX port of the final model appears |
| [FU-547](open.md) | SenseNova-Loop-9B rows (llama.cpp GGUF, all platforms) | When llama.cpp PR #32588 (sensenova_loop, opened 2026-11-19) merges and a tag with prebuilt archives carries i… |
| [FU-546](open.md) | Inkling 2 Mini official NVFP4/MXFP8 on the vLLM lane (Linux, Blackwell) | When a Blackwell box (DGX Spark, RTX 50) is available for testing |
| [FU-545](open.md) | Inkling 2 on MLX | When mlx-lm or mlx-vlm add inkling2 |
| [FU-544](open.md) | First-generation Inkling and Inkling-Small | When llama.cpp PR #25731 (draft) merges |
| [FU-543](open.md) | Inkling 2 Small row gated by fit, untested | Next catalogue pass after FU-540 |
| [FU-542](open.md) | Memory estimate for looped models (KV per logical layer) | With FU-540 or FU-547 |
