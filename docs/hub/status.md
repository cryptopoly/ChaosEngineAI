---
title: Status and handover
order: 1
summary: Read this first in every session.
---
# Status and handover

> **Read first.** ChaosEngineAI is a desktop AI inference app (Tauri shell, React frontend, Python sidecar; MLX, llama.cpp and vLLM engines; image, video and voice studios). It is in **v0.9.9 release prep**: the tag `v0.9.9` is held until [PR 119](https://github.com/cryptopoly/ChaosEngineAI/pull/119) merges. Then read [decisions](decisions.md), [backlog](backlog.md) and the progress file for the active stream in [progress](progress/README.md).

Newest dated update first. Each update has the same sections: Done, Open for Claude, Open for Dan.

## Update 2026-10-29, release prep for v0.9.9

**Done**

- Release notes, CHANGELOG, README and the mkdocs pages describe v0.9.9 (voice first, then the Models pages, best fit, recommended settings, bigger images, LTX-2.5 with sound). New screenshots are in `Screenshots/v0.9.9/` (3600 x 2260, taken with headless Chromium); the old Discover / My Models / Image Discover screenshots are no longer referenced.
- Website (`ChaosEngineAI-Site`, branch `site-v0.9.9`, not pushed): copy, tour and cards updated for v0.9.9, mobile layout fixed, invented speculative-decoding figures replaced with the release notes' numbers. Its `v0.9.9` images load from the app repo's `main`, so **publish the site only after the app branch is merged**.
- Stale UI labels from the merge fixed (composer placeholder, Studio header buttons, empty-state text, Dashboard hint) in English and all nine locales; the Dashboard's "Backend readiness" line no longer reports MLX as missing at startup (it now waits for the probe).
- Packaging: `npm run pre-build-check` passes (14 gates, 1 warning: the turbo llama-server build is behind upstream, FU-065). `npm run release:macos` built an unsigned `ChaosEngineAI_0.9.9_aarch64.dmg` (513 MB); the staged runtime has `llamaServer`, turbo and `sd`, no warnings, the four Setup installers and `dflash`; the bundled backend starts and reports MLX 0.32.4, mlx-lm 0.32.2, DFlash and GGUF-MTP. The signed and notarised builds for all platforms come from the tag workflow.

**Open for Claude**

- After the merge: re-tag, watch the release workflow, check `llamaServer` in each `runtime-*.manifest.json`, publish the draft release with `RELEASE_NOTES_v0.9.9.md`, then publish the site.

**Open for Dan**

- Say go to push `feat/models-fit` (it contains PR 119 plus everything above) and update the PR.
- The packaged app was not launched as an app (only its backend), and the desktop shell on Tauri 2.12 has not been run on any platform: open the dmg once before the release.

## Update 2026-10-29, evening

**Done**

- **Models page, best-fit badges, recommended settings and bigger image sizes are built on branch `feat/models-fit`** (the main checkout is on it; not pushed, no PR). Four Opus agents built the shared Models page and the Chat, Image and Video pages; Discover and My Models are gone and their old ids redirect ([FU-129](tracker/open.md)). Fit badges and "Best fit" ribbons ([FU-133](tracker/open.md)); the Studios show "Recommended for this machine" with Apply; per-model size limits replace the 2048 px cap (FLUX 3 Klein measured: 3072 px = 41 GB, 110 s; 4096 px refused on 64 GB). FLUX 3 Klein now starts at 8 steps.
- Three read-only reviewers (backend, Models pages, Studios) found and I fixed: the memory gate refused Qwen-Image-2.1 on a 64 GB Mac (1.1 x a measured peak), the best-fit pick for FLUX.1 was the slow sd.cpp row, tables overflowing below 1440 px, a deleted model staying Installed, tracked-only rows that could not be deleted, size limits that broke the aspect ratio, and the recommended panel layout. Details and what is still open are in the FU-129 and FU-133 rows.
- Checks on the merged branch: pytest all pass, vitest 777, `tsc` clean, `i18n:validate` OK (3003 keys in all 10 locales), E2E smoke phases 0, 3-8 pass (includes a live LTX clip). Looked at in the running app: Chat, Image and Video Models pages at 1200 and 1500 px, Image Studio recommended panel.

**Open for Claude**

- Nothing blocking on this branch. Open items are listed in the FU-129 / FU-133 rows (Discover wording in locale strings, menu keyboard handling, CUDA VRAM in the gate).

**Open for Dan**

- Decide whether `feat/models-fit` ships in v0.9.9 (the release gate in `docs/plans/models-fit.md` is met except the native-speaker read of de / ja / ru / zh-TW and Windows / Linux runs) or waits for the next release behind PR 119.
- Real click-throughs of download, pause, resume and delete on the new pages were not run; worth one try with a small model.

## Update 2026-10-29, afternoon

**Done**

- PR 119 open (LTX-2.3 on its own MLX engine, settings guards for chat, image, video and voice, MLX DDTree fix, hardware gating). macOS and Ubuntu checks pass; Windows failed on one test that compared `/pack` with `\pack`. Fixed locally, **not pushed** (push is held, see [rules](rules.md)).
- Linear DFlash on MLX now verifies 8 tokens per step instead of 16 (16 was slower than plain decoding on prose). Tests and `docs/features/dflash.md` updated.
- LTX long-clip memory measured on the M4 Max: 18 GB engine footprint at both 768x512 x 257 frames (352 s) and 1280x704 x 121 frames (363 s); the 20 GB catalog figure holds. See [FU-128](tracker/open.md).
- LTX-2.5 runs live and is the LTX default: 128 s against 155 s for LTX-2.3 on the same clip, 17 GB, clean output with sound ([FU-128](tracker/closed.md)).
- Sound in Video Studio ([FU-130](tracker/open.md)): sound badge on clips that have it, result window plays them with sound, prompt hint for the LTX rows. Checked in the running Video Gallery (the badge shows); the result window was not opened.
- This hub: [tracker pages](tracker/README.md) are generated from `CLAUDE.md` by `tools/hub_tracker.py`.

- **DFlash on hybrid Qwen3.5 / 3.6 models was producing garbage on the release MLX libraries; fixed** (the adapter read the wrong output head, [FU-135](tracker/open.md)) with a load-time self-check. Qwen3.6-27B: 26.9 tok/s plain against 35.1 with DFlash through the app, same text; block 8 beats 16 there too.
- TurboQuant speed measured: decode is 0.18-0.21 of Native at 512 tokens and 0.04 at 16K, so it gets slower exactly where it saves memory ([FU-095](tracker/open.md)); the launch estimate and a long-context warning are updated.
- DDTree on a 27B-class target (Qwen3-Coder-30B-A3B 4-bit, the only attention-only model of that size with a drafter): block 8 confirmed (mean 101 tok/s against 91 plain and 74 for block 16); DDTree adds nothing over a chain of the same width; creative prose runs 2 to 4 times slower with any speculative mode ([FU-134](tracker/open.md)).

**Open for Claude**

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
| Real download / pause / resume / delete clicks on the new Models pages not run | 2026-10-29 | Try one with a small model | open ([FU-129](tracker/open.md)) |
| Desktop shell not launched on Tauri 2.12 | 2026-10-28 | Launch once per platform before release | open ([FU-121](tracker/open.md)) |
