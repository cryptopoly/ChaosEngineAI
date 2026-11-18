# Review stream D: whole-app sweep

Branch `review/d`, worktree `/Users/dan/ChaosEngineAI-review-d`, cut from `release/v0.10.0` at `b31c4ea5`.
Plan: `docs/plans/overnight-e2e-review.md`. Scope: every tab except Benchmarks, Conversion, Fine Tuning, Prompts, Plugins, Logs and Server.
Budget cut by the coordinator to 3.5 hours (start 07:36, 2026-11-18, report due about 11:05), so the sweep is ordered by user impact:
Chat, Models, Studios, first run, backend-down recovery, then the long tail.

## Setup (how to resume)

- Python: `/Users/dan/ChaosEngineAI/.venv/bin/python`. This worktree has no `.venv`, so Playwright needs `E2E_PYTHON` set to it.
- `e2e/node_modules` is a symlink to the main checkout's (git-ignored), Playwright 1.65.1, Chromium headless shell already installed.
- Ports: backend 8903, Vite 8933. One backend at a time: the Playwright run and the hand walk both use 8903, never together.
- Playwright: `cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8933 E2E_API_PORT=8903 npx playwright test --workers=1`
- Hand walk: backend with a throwaway `HOME` in the scratchpad, `CHAOSENGINE_PORT=8903`, a dummy token; Vite with `VITE_CHAOSENGINE_API_BASE=http://127.0.0.1:8903`. The walk is driven by small Playwright scripts in the scratchpad (`review-d/walk.cjs`, `down.cjs`, `pseudo.cjs`, `navreach.cjs`, `firstrun.cjs`), headless, which save a screenshot per page and print what they measure. The desktop app's Browser pane was not used after the first minute: another stream drives the same pane and navigated my tab away.
- Warning for the next agent: `kill <playwright pid>` (SIGTERM) leaves the runner alive after its two servers stop, and it then runs its remaining tests against whatever listens on the same ports. Use `kill -9` on the runner and check `pgrep -fl "playwright test"` before starting anything else on 8903.

## What was checked, and how

Sizes: 1440 x 900, 1400 x 920 (default window), 1100 x 760 (the smallest window `tauri.conf.json` allows), 1024 x 768, 768 x 1024, 375 x 812.
On every page the scripts look for: a crash (error boundary), console errors and failed requests, raw i18n keys and leftover `{placeholders}`, sideways scroll, elements past the right edge, panels that cut their own content with nothing to scroll, sibling panels drawn over each other. Screenshots were looked at for Dashboard, Chat, Chat Models, Image Studio, Video Studio, Voice Studio and Models, Music Studio, Projects, Settings (General, Storage, Diagnostics, Accelerators).

## Log (newest last)

| Time | Check | Result | Evidence |
|---|---|---|---|
| 07:40 | `npx tsc --noEmit`, untouched tip | PASS | exit 0 |
| 07:40 | `npm run i18n:validate`, untouched tip | PASS | en 4951/4951, nine locales 4947/4951, tlh 82.6 % (outside the gate) |
| 07:41 | First Playwright start | FAIL (setup) | `ModuleNotFoundError: No module named 'fastapi'`: no `.venv` beside this worktree. Fixed by `E2E_PYTHON`. |
| 07:42 | Baseline Playwright on the untouched tip | PARTIAL | 78 of 179 passed (specs 01 to 14), 0 failed, then stopped by me to free port 8903 for the hand walk. The rest of the log is not a result (see the warning above). |
| 07:50 | First run on a clean data dir, 1440 x 900 | PASS with one finding | terms dialog shows, Quit and I agree; `GET /api/cache/preview?...num_layers=0...` answers 422 on every launch (D-9) |
| 07:52 | All in-scope tabs, empty state, 1440 x 900 | PASS | no crash, no raw key, no console error beyond D-9 and the updater warning that belongs to the browser build |
| 07:55 | All in-scope tabs, 375 x 812 | FAIL | Image Studio and Image Gallery panels drawn over each other (D-2); Dashboard and Settings General cut (D-3); Storage in two 160 px columns (D-6); Compare and Diagnostics a few px past the edge (L-6) |
| 07:58 | All in-scope tabs, 1100 x 760 | FAIL | Dashboard 309 of 468 px shown, Settings General 280 of 374, Storage 129 of 174 / 213 of 307 / 201 of 288 (D-3); Image Studio overlap 36 px and 272 px (D-2) |
| 08:00 | Backend killed mid-use, 1440 x 900 (`kill` of my backend PID, restart with another token) | PASS with one finding | badge says Offline within 4 s; Voice Studio, Music Studio and Projects say "Waiting for the backend"; Chat, Chat Models, Image Studio, Video Studio, Settings and Dashboard keep their last data with no word about it (D-4). Restart with a different token: one round of 401, then the page reads `/api/auth/session` and every later request passes; badge back to Backend online within 8 s. Done twice (second and third token). |
| 08:03 | Same at 1100 x 760 | FAIL before the fix | the badge is inside the closed drawer, so nothing on screen said the backend was down (D-4). After the fix the notice shows on every tab and clears on restart. |
| 08:10 | Pseudo-locale (accented, +40 %) on Settings General, Storage, Diagnostics, Accelerators, Disk usage, Dashboard, Chat, Chat Models at 1440, 1100 and 375 | PASS for layout | no sideways scroll, no cut button text at 1440 and 1100. Text that never goes through i18n: about 100 strings on Diagnostics (L-2), backend activity lines and folder labels, "Released Nov 2026". |
| 08:14 | Every sidebar button reachable, en / de / ru, four sizes | FAIL before the fix | at 1100 x 760 with the Chat or Tools group open, Settings sits under the status footer (D-5) |
| 08:20 | First launch with a German system language, 1400 x 920 | FAIL before the fix | the language banner takes the sidebar's grid cell: banner 216 x 577, sidebar 1184 wide, workspace 216 x 343 at the bottom left (D-1). Keyboard on the terms dialog: focus starts on the dialog, Escape does not close it, Tab cycles Quit and I agree only. |
| 08:11 | vitest (full) after the first fixes | PASS | 152 files, 1817 tests |
| 08:22 | New specs 32, 33, 34 | see below | |

## Defects fixed on `review/d` (ranked)

| Id | Impact | What a user saw | Fix | Commit | Test |
|---|---|---|---|---|---|
| D-1 | High, first run | On a system set to any of the nine other languages, the first launch at 1180 px and wider showed a broken app: the language banner sat in the sidebar's place, the sidebar filled the page and the workspace was a 216 px column in the corner, until a banner button was pressed. | The banner spans both grid columns and gets its own row. "Keep English" now switches at once (the app had already started in the detected language and stayed in it for up to 5 s). | `fix(first run)` | `34-locales.spec.ts` |
| D-2 | High, smallest window | Image Studio at 1100 px wide and narrower: the Prompt form was drawn over the header and under Recent Outputs, so the prompt box was covered. Image Gallery the same at phone width. | `.image-page-grid` rows are `max-content` (the frame stretched the grid and `auto` rows shared its height). | `ee71969f` | `33-window-sizes.spec.ts` |
| D-3 | High, smallest window | Dashboard and Settings (General, Storage) cut the lower part of their panels at 1100 x 760 with nothing to scroll: UI scale, clock format, the translations link, the storage folders, swap, CPU and power. The Dashboard lost its Power row even at the default 1400 x 920. | Rows never go below their content and the grid scrolls when they do not fit; the activity feed still scrolls inside its panel. | `ee71969f` | `33-window-sizes.spec.ts` |
| D-4 | High, backend stop | A stopped backend showed only as the small badge at the foot of the sidebar; below 1180 px that badge is inside the closed drawer, so nothing said so. Chat, Models, Image and Video Studio, Settings and Dashboard kept showing old data. | A notice at the top of the workspace until the backend answers (`statusBanners.backendOffline.*`, en filled, ten locales `null`). One line in `App.tsx` passes the state. | `ee71969f` | `statusBanners.test.tsx`, `32-backend-offline.spec.ts` |
| D-5 | High, smallest window | In the menu drawer at 1100 x 760 with the Chat or Tools group open (Chat is open whenever you are in Chat), Settings was under the status footer and could not be clicked. | The drawer scrolls; the nav list no longer shrinks under the footer. | `c52f1ea4` | `33-window-sizes.spec.ts` |
| D-6 | Medium | Settings > Storage: the folder path had a third of the row and Browse two thirds, so the path could not be read; below 900 px Storage stayed in two 160 px columns and its buttons were slivers. | Rows that start with the path give it the room; one column below 900 px. | `c52f1ea4` | `33-window-sizes.spec.ts` |
| D-7 | Low | Settings > General: the "Keep all sidebar groups open" tick box sat centred above its own label (same in Integrations). | `.check-row` inside a control stack is a row again. | `ee71969f` | `33-window-sizes.spec.ts` |
| D-8 | Low | Two buttons had a class with no style and showed as raw grey browser buttons: Copy beside "Open en on GitHub", and "Keep English" in the language banner. | `secondary-button`. | `ee71969f` | `33`, `34` |
| D-9 | Low | Every launch sent a cache preview request with an all-zero model shape and logged a 422. | No request until a model shape is known; same fallback value. | `ee71969f` | `cachePreview.test.ts` |

## Left for the owner (ranked, not fixed)

| Id | Impact | Finding | Recommendation |
|---|---|---|---|
| L-1 | Medium | The language banner asks "Switch the app to Deutsch?" in German while the app is already in German (`main.tsx` starts in the detected language). The question makes no sense as worded. | Either start in English until the banner is answered, or reword to "The app is in {language}, your system language" with Keep and Use English. Needs new strings in ten locales. |
| L-2 | Medium | Settings > Diagnostics shows about 100 English strings in every language (Refresh, Copy diagnostics, System, Hardware, CPU logical cores and the rest). The `diagnostics` namespace has 78 keys but the panel's table labels are literals. | Route the labels through `t()`; keep the copied markdown in English for support. |
| L-3 | Medium | Image Studio empty state says the same thing three times (runtime callout, "no image models", "FLUX.1 Schnell is not installed locally"), calls the engine "Stub diffusion pipeline" and the empty gallery "the first saved image artifact for this branch". | One notice with one button; plain words. |
| L-4 | Low | Chat welcome text recommends "a small Qwen3 or Llama 3 variant"; the catalogue is on Qwen 3.6 / 3.8, LFM3, Kolibri. The offline chat fallback text says "Restart the FastAPI sidecar". | Refresh the copy; no model names in the welcome, or read the first recommended row. |
| L-5 | Low | Backend-made text is not translated: activity feed lines ("Settings updated", "0 models discovered..."), model directory labels, "Released Nov 2026" in Chat Models, the DFlash row in Accelerators. | Send keys plus values, format dates with `Intl`. |
| L-6 | Low | 375 px: Chat > Compare's second slot and the Compare button run 33 px past the edge; Diagnostics values run 5 px past. Sideways scroll stays 0. | Stack the slots below 640 px; `overflow-wrap: anywhere` on the values. |
| L-7 | Low | The Language panel in Settings is titled "Settings"; the clock-format and UI-scale button rows have no visible label. | Title "Language and time"; add labels. |
| L-8 | Low | `scripts/i18n-pseudo.mjs` documents `?devLocale=en-XA`; nothing in `src/` reads it, so the checklist item "pseudo-locale smoke-tested on Setup + Diagnostics" could not be done by hand. Spec 34 now does it by answering the German catalogue requests with pseudo text. | Fix the comment or wire the switch. |
| L-9 | Low | There is no light theme (`color-scheme: dark`, no switch). The brief asked for light and dark. | Nothing to do unless a light theme is wanted. |
| L-10 | Low | `e2e/tests/helpers.ts` `openApp` waits 5 s for the terms dialog on every test, accepted or not: about 15 of the suite's 18 minutes. | Ask the backend (`/api/workspace`) whether the terms are accepted before waiting. |
| L-11 | Info | The Dashboard's process table lists `llama-server` and `python` processes of other ChaosEngineAI instances on the machine as "ChaosEngineAI" with Stop buttons (seen: the other review streams' backends). | Mark processes this backend did not start, or leave them out of Stop idle. |

## Not tested

See the final report section at the end of this file (filled in when the run ends).
