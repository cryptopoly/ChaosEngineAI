---
title: Status and handover
order: 1
summary: Read this first in every session.
---
# Status and handover

> **Read first.** ChaosEngineAI is a desktop AI inference app (Tauri shell, React frontend, Python sidecar; MLX, llama.cpp and vLLM engines; image, video and voice studios). It is in **v0.9.9 release prep**: the tag `v0.9.9` is held until [PR 119](https://github.com/cryptopoly/ChaosEngineAI/pull/119) merges. Then read [decisions](decisions.md), [backlog](backlog.md) and the progress file for the active stream in [progress](progress/README.md).

Newest dated update first. Each update has the same sections: Done, Open for Claude, Open for Dan.

## Update 2026-10-29, afternoon

**Done**

- PR 119 open (LTX-2.3 on its own MLX engine, settings guards for chat, image, video and voice, MLX DDTree fix, hardware gating). macOS and Ubuntu checks pass; Windows failed on one test that compared `/pack` with `\pack`. Fixed locally, **not pushed** (push is held, see [rules](rules.md)).
- Linear DFlash on MLX now verifies 8 tokens per step instead of 16 (16 was slower than plain decoding on prose). Tests and `docs/features/dflash.md` updated.
- LTX long-clip memory measured on the M4 Max: 18 GB engine footprint at both 768x512 x 257 frames (352 s) and 1280x704 x 121 frames (363 s); the 20 GB catalog figure holds. See [FU-128](tracker/open.md).
- LTX-2.5 support added to the engine and catalog (own Gemma 4 text encoder, no companion repo); the licence is accepted on Hugging Face. Not yet run live (download queued).
- Sound in Video Studio ([FU-130](tracker/open.md)): sound badge on clips that have it, result window plays them with sound, prompt hint for the LTX rows. Checked in the running Video Gallery (the badge shows); the result window was not opened.
- This hub: [tracker pages](tracker/README.md) are generated from `CLAUDE.md` by `tools/hub_tracker.py`.

- DDTree on a 27B-class target (Qwen3-Coder-30B-A3B 4-bit, the only attention-only model of that size with a drafter): block 8 confirmed (mean 101 tok/s against 91 plain and 74 for block 16); DDTree adds nothing over a chain of the same width; creative prose runs 2 to 4 times slower with any speculative mode ([FU-134](tracker/open.md)).

**Open for Claude**

- Check block 8 against 16 on Qwen3.6-27B (hybrid model, download in progress). See [progress](progress/runs-2026-10-29.md).
- Run LTX-2.5 live, then decide the default LTX row.
- Keep this hub current after every batch.

**Open for Dan**

- Say when to push the Windows test fix and when to re-tag `v0.9.9` ([decisions](decisions.md) items 1 and 2).
- Answer the other open items in [decisions](decisions.md).

## How to run and test

```bash
# backend for manual checks (port 8876)
CHAOSENGINE_PORT=8876 .venv/bin/python -m backend_service.app
# all Python tests / frontend tests / types
.venv/bin/python -m pytest tests/ -q
npm test
npx tsc --noEmit
# the hub: regenerate tracker pages, then build the site
python3 tools/hub_tracker.py && python3 tools/build_hub.py
```

The full list, including the release gate, is in [testing](testing.md).

## Known breakage

| Problem | Since | Workaround | Owner |
|---|---|---|---|
| PR 119 Windows test fails (`\pack` vs `/pack`) | 2026-10-29 | Fixed locally; needs a push | Claude |
| JSON schema on a reasoning model through llama.cpp returns no content (Nemotron 3.5 Bolt) | 2026-10-29 | Reply now explains why; turn the schema off | open ([FU-127](tracker/open.md)) |
| A context overflow answers HTTP 500 with a readable message (400 would be right) | 2026-10-29 | None needed | open ([FU-127](tracker/open.md)) |
| A mflux venv rebuild deletes the old venv first | 2026-10-29 | Re-run the install | open ([FU-123](tracker/open.md)) |
| Desktop shell not launched on Tauri 2.12 | 2026-10-28 | Launch once per platform before release | open ([FU-121](tracker/open.md)) |
