# Review stream D: whole-app sweep

Branch `review/d` (pushed), worktree `/Users/dan/ChaosEngineAI-review-d`, cut from `release/v0.10.0` at `b31c4ea5`.
Plan: `docs/plans/overnight-e2e-review.md`. Scope: every tab except Benchmarks, Conversion, Fine Tuning, Prompts, Plugins, Logs and Server.
The coordinator cut the budget to 3.5 hours (start 07:36 on 2026-11-18), so the sweep went by user impact: Chat, Models, Studios, first run, backend stop, then the rest.

## Headline

The app holds up on a clean first run in English at a large window. It did not hold up in four places a user reaches in the first minutes, all fixed on this branch with tests:

1. **A local model that is not a catalogue row could be refused or routed as another model** (D-13). Found with a real 0.5B model; on the owner's own library 9 of 20 chat models were matched to a different model.
2. **First launch on a non-English system showed a broken layout** until the language banner was answered (D-1).
3. **At the smallest window the app allows (1100 x 760)** panels cut their controls, Image Studio drew its panels over each other, and Settings could not be clicked in the menu (D-2, D-3, D-5).
4. **A stopped backend was not announced** where the sidebar is a drawer, and Stop in Chat always raised an error (D-4, D-15).

Final state: tsc clean, vitest 154 files / 1828 tests, `i18n:validate` OK, Playwright 209 tests (179 old, 30 new in specs 32 to 37), result at the end of this file.

## Setup (how to resume)

- Python: `/Users/dan/ChaosEngineAI/.venv/bin/python`. This worktree has no `.venv`, so Playwright needs `E2E_PYTHON` set to it.
- `e2e/node_modules` is a symlink to the main checkout's (git-ignored), Playwright 1.65.1, Chromium headless shell already installed.
- Ports: backend 8903, Vite 8933. One backend at a time: the Playwright run and the hand walk both use 8903, never together.
- Playwright: `cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8933 E2E_API_PORT=8903 npx playwright test --workers=1` (about 12 minutes).
- Hand walk: backend with a throwaway `HOME` in the scratchpad, `CHAOSENGINE_PORT=8903`, a dummy token, `HF_HUB_OFFLINE=1`; Vite with `VITE_CHAOSENGINE_API_BASE=http://127.0.0.1:8903`. The walk is driven by small headless Playwright scripts in the scratchpad folder `review-d/` (`walk.cjs` every tab with layout probes, `down.cjs` and `chatdown.cjs` backend stop and restart, `pseudo.cjs`, `navreach.cjs`, `firstrun.cjs`, `flows.cjs` keyboard, `errors.cjs` forced 500s, `realchat.cjs`, `compare.cjs`, `libs.cjs`). Each saves screenshots and prints what it measures. The desktop app's Browser pane was dropped after the first minute: another stream drives the same pane and navigated my tab away.
- Data used: (a) empty throwaway homes for first run; (b) a home with two tiny local models linked in; (c) a **copy** of the real data dir's chat sessions, images, videos, music, voice and storyboards (1.3 GB, no `settings.json`, so no token) with the Hugging Face cache and the installed image engine linked read-only. Nothing was written to `~/.chaosengine`.
- Model lock (`review-lock/model.lock`) held by this stream 09:04 to 09:08, 09:12 to 09:22 and 09:37 to 09:41. Models loaded: Qwen2.5-0.5B MLX, Qwen3-0.6B MLX, LFM3-1.7B GGUF, and once, by the fault in D-14, Qwen3.8-27B MLX for about a minute.
- Warning for the next agent: `kill <playwright pid>` (SIGTERM) leaves the runner alive after its two servers stop, and it then runs its remaining tests against whatever listens on the same ports. Use `kill -9` on the runner and check `pgrep -fl "playwright test"` first.

## Log (newest last)

| Time | Check | Result | Evidence |
|---|---|---|---|
| 07:40 | `npx tsc --noEmit`, `npm run i18n:validate`, untouched tip | PASS | exit 0; en 4951/4951, nine locales 4947/4951 |
| 07:42 | Baseline Playwright on the untouched tip | PARTIAL | 78 of 179 passed (specs 01 to 14), 0 failed, then stopped by me to free port 8903 |
| 07:50 | First run on a clean data dir, 1440 x 900 | PASS, one finding | terms dialog with Quit and I agree; a 422 on every launch (D-9) |
| 07:52 | Every in-scope tab, empty state, 1440 x 900 | PASS | no crash, no raw key, no console error beyond D-9 |
| 07:55 | Every in-scope tab, 375 x 812 | FAIL | Image Studio and Gallery overlap (D-2), Dashboard and Settings cut (D-3), Storage in two 160 px columns (D-6) |
| 07:58 | Every in-scope tab, 1100 x 760 | FAIL | Dashboard shows 309 of 468 px, Settings General 280 of 374, Storage 129 of 174; Image Studio overlap 36 and 272 px |
| 08:00 | Backend killed mid-use and restarted with another token, 1440 x 900 | PASS, one finding | badge says Offline within 4 s; six pages keep old data with no word (D-4); after restart one round of 401, then the page reads `/api/auth/session` and carries on within 8 s; done twice |
| 08:03 | Same at 1100 x 760 | FAIL before the fix | the badge is inside the closed drawer; nothing on screen said the backend was down |
| 08:10 | Pseudo-locale (accented, +40 %) on Settings General, Storage, Diagnostics, Accelerators, Disk usage, Dashboard, Chat, Chat Models | PASS for layout | no sideways scroll, no cut text at 1440 and 1100; about 100 Diagnostics strings never go through i18n (L-2) |
| 08:14 | Every sidebar button reachable, en / de / ru, four sizes | FAIL before the fix | Settings under the footer at 1100 x 760 with Chat or Tools open (D-5) |
| 08:20 | First launch with a German system language, 1400 x 920 | FAIL before the fix | banner 216 x 577 in the sidebar's cell, sidebar 1184 wide, workspace 216 x 343 (D-1). Terms by keyboard: Escape does not close, Tab cycles Quit and I agree, Enter accepts |
| 08:30 | Every shipped locale boots (zh-CN, zh-TW, ja, de, ru, ko, fr, es, pt-BR, tlh) | PASS | spec 34: no raw key, no leftover placeholder, no sideways scroll on 25 pages each |
| 08:34 | Keyboard flows (Chat, Select Model, Compare, thread search, New thread, Models filter and search, a bad repo, read-only terms) | FAIL before the fix | the model dialogs had no role, no Escape, no focus return (D-11) |
| 08:40 | Error states: 500 on Save settings, clock format, New project, Install voice runtime, image and video Download | PASS | each page shows the backend's sentence; nothing crashes |
| 08:42 | Sample project, seven tabs, 1100 x 760 and 1400 x 920; sidebar style Tabs, every group | PASS | no cut panel, no crash, arrow keys move between sub-tabs |
| 08:56 | Full Playwright (helper no longer waits 5 s per test) | 205 pass, 2 fail | both test faults, fixed (route in flight at test end; a label that changes with the thread) |
| 09:00 | Loaded state on the data copy: 67 models, 200 chat sessions, 315 images, 133 clips, 29 tracks, 11 projects; every in-scope tab at 1400 x 920 and 1100 x 760 | PASS for layout | no crash, no cut or overlapping panel; Video Gallery fetched all 133 clips at once, 205 MB (D-16) |
| 09:05 | Real model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, Chat Models > Chat in a new thread > Start Chat | FAIL | 500 "Qwen3.8 Flash Next runs only on TensorFold"; the request carried another model's repo (D-13) |
| 09:13 | Same after the fix | PASS, then two more faults | loaded in 1.5 s, but no thread opened and Qwen3.8-27B answered (D-14); Stop raised "Streaming timed out" (D-15) |
| 09:21 | Same after those fixes | PASS | new thread, Shift+Enter new line, Enter sends, reply at about 370 tok/s, Stop with no error, next message sends, Unload frees it |
| 09:35 | Full Playwright | PASS | 208 passed, 11.5 min |
| 09:38 | Real model that is a catalogue row: LFM3-1.7B GGUF, Start Chat | PASS after the fix | thread opened and model loaded (before: thread only, "Load a model first") |
| 09:39 | Backend killed mid-chat with that model loaded, messages sent while down, restart | PASS | notice shows; a send in the first second says the backend is not answering (was "Chat error: Failed to fetch"); a later send gets the plain in-thread note (D-12); after restart the page says No model loaded and keeps the typed text; no orphan `llama-server` |
| 09:40 | Compare with Qwen2.5-0.5B and Qwen3-0.6B, Enter in the prompt | PASS | each loads alone (2.9 s, 3.1 s), answers (396 and 404 tok/s) and is unloaded |
| 09:46 | Image Studio with three installed models (form only, no render) | PASS, two findings | estimate strip and Generate state right; a size under the model's minimum said "Above this model's size limit" (D-18); "No models installed" shows for the first seconds after a backend start (L-20) |
| 09:49 | Galleries and libraries on the data copy: search, empty search, expand, load audio, list view, filters, 200-thread search (0.6 s), export menu, HTML Challenge prompt library | PASS | no crash, no console error |
| 09:50 | Full Playwright on the final tree; pytest `test_backend_service`, `test_cache_strategies`, `test_route_contracts`, `test_hub_tracker` | see the end of this file | pytest: 206 passed |

## Defects fixed on `review/d` (ranked)

| Id | Impact | What a user saw | Fix | Commit | Test |
|---|---|---|---|---|---|
| D-13 | Critical, Chat | A local model that is not itself a catalogue row could not be loaded, or was routed as another model. The app joins a library item to a catalogue row by a word score with a threshold of 12; "qwen", "4bit", the same format and the same bits are worth 50 to 70. `mlx-community/Qwen3-0.6B-4bit`, `Qwen2.5-0.5B-Instruct-4bit` and `Qwen3-8B-4bit` all took the repo of Qwen3.8 Flash Next (a TensorFold-only row), that repo went to the backend as the model's own, and the load answered "Qwen3.8 Flash Next runs only on TensorFold". Other wrong pairs on the real library: LFM3-VL to MiniCPM5, a Qwen3-VL GGUF to Granite Switch, Qwen3-0.6B GGUF to Flint, Nemotron Bolt to Nemotron Lightning 30B. The launch dialog also showed the wrong row's context ("256K") and badges. | A Hugging Face cache folder names its own repo: when neither the row's repo nor its model name is in the item's name or path, the score is 0. Same repo, the same model under another uploader, and a conversion of a base repo still match. Checked against the real library: 9 exact, 1 cross-uploader, 10 now unmatched, and every row the backend marks local keeps its item. | `b4cfaf70` | `library.test.ts` (4), `37-chat-local-model.spec.ts`, real load |
| D-1 | High, first run | On a system set to any of the nine other languages, the first launch at 1180 px and wider showed a broken app: the language banner sat in the sidebar's place, the sidebar filled the page and the workspace was a 216 px column in the corner, until a banner button was pressed. | The banner spans both grid columns and gets its own row. "Keep English" switches at once. | `25b20b71` | `34-locales.spec.ts` |
| D-14 | High, Chat | Chat Models > Chat in a new thread on a model with no catalogue row loaded it but opened no thread: the first message went into the last thread and that thread's own model was loaded back in its place. For a catalogue row it was the other way round: thread opened, nothing loaded, "Load a model first". | Both paths open the thread from what the picker names, then load. | `17d2044b`, `cab15024` | `launchActions.test.ts` (3), spec 37, real runs |
| D-15 | High, Chat | Every press of Stop raised "Chat error: Streaming timed out". | Only the 5-minute timer is an error; a stopped reply with no text leaves no empty bubble. | `17d2044b` | `streamPhase.test.ts` (2), real run |
| D-2 | High, smallest window | Image Studio at 1100 px and narrower: the Prompt form was drawn over the header and under Recent Outputs. Image Gallery the same at phone width. | `.image-page-grid` rows are `max-content`. | `ee71969f` | `33-window-sizes.spec.ts` |
| D-3 | High, smallest window | Dashboard and Settings (General, Storage) cut the lower part of their panels at 1100 x 760 with nothing to scroll: UI scale, clock format, storage folders, swap, CPU, power. | Rows never go below their content and the grid scrolls; the activity feed still scrolls inside its panel. | `ee71969f` | spec 33 |
| D-4 | High, backend stop | A stopped backend showed only as the sidebar badge, which is inside the closed drawer below 1180 px. Chat, Models, Image and Video Studio, Settings and Dashboard kept old data. | A notice at the top of the workspace until the backend answers (`statusBanners.backendOffline.*`, en filled, ten locales `null`). One line in `App.tsx`. | `ee71969f` | `statusBanners.test.tsx`, `32-backend-offline.spec.ts` |
| D-5 | High, smallest window | In the menu drawer at 1100 x 760 with the Chat or Tools group open, Settings was under the status footer and could not be clicked. | The drawer scrolls; the nav no longer shrinks under the footer. | `c52f1ea4` | spec 33 |
| D-12 | Medium, backend stop | A chat message sent while the backend is down got a fake reply about "the desktop shell", "this local fallback" and "the FastAPI sidecar"; a send in the first second showed "Chat error: Failed to fetch". | Plain words in both places. | `d621e699`, `cab15024` | `chatErrors.test.ts`, real run |
| D-11 | Medium, keyboard | Select Model (Chat, Compare, HTML Challenge), the model picker and the family details were plain overlays: no dialog role, Escape did nothing, Tab walked into the page behind, focus lost on close. | `useDialogFocus` with a new `active` flag; the opener is remembered before an autoFocus field takes focus. These two components are also used by Benchmarks and Conversion, which gain the same behaviour. | `617d8f80` | `35-keyboard-dialogs.spec.ts` |
| D-16 | Medium, speed | Video Gallery fetched every clip whole when it opened: 133 clips, about 100 MB. | A card fetches its clip when it scrolls into view: 8 requests, 11 MB for the first screen. | `17d2044b` | `36-video-gallery-lazy.spec.ts` |
| D-10 | Medium | Settings > Language: "System default" left the app in the language chosen before, until the next launch. | Applies the system language at once. | `060b0f3e` | spec 34 |
| D-6 | Medium | Settings > Storage: the folder path had a third of the row and Browse two thirds; below 900 px Storage stayed in two 160 px columns. | The path takes the room; one scrolling column below 900 px. | `c52f1ea4`, `060b0f3e` | spec 33 |
| D-17 | Low | The launch dialog's cache preview answered 422 for a model under 0.5B parameters. | Lower bound 0.05. | `806e24a0` | `test_route_contracts.py` |
| D-18 | Low | Image Studio: a size under a model's smallest side (Iris-3B, 768 px) was off with the reason "Above this model's size limit". | Its own reason (`sizePresets.underLimit`, en filled, ten locales `null`). | `6a10c1fd` | none (string) |
| D-7 | Low | Settings: a tick box inside a control stack sat centred above its own label. | A row again. | `ee71969f` | spec 33 |
| D-8 | Low | Two buttons with a class that has no style (Copy beside the translations link, "Keep English" in the banner). | `secondary-button`. | `ee71969f` | specs 33, 34 |
| D-9 | Low | Every launch sent a cache preview request with an all-zero model shape (422). | No request until a shape is known. | `ee71969f` | `cachePreview.test.ts` |
| T-1 | Tests | `openApp` waited 5 s for the terms dialog on every test; a spec route still in flight failed a finished test. | Waits for the startup panel, then looks once; the page fixture drops routes. The suite runs 209 tests in about 12 minutes where 179 took about 18. | `d621e699` | the suite |

## Left for the owner (ranked, not fixed)

| Id | Impact | Finding | Recommendation |
|---|---|---|---|
| L-16 | Medium | The backend trusts the `canonicalRepo` it is sent over the model's own path when it picks an engine (`controller.py` `_tensorfold_match`). D-13 is fixed in the app, but another client (agent bridge, CLI, an older window) can still send a repo that is not the folder's. | When `path` is a Hugging Face cache folder, derive the repo from it and ignore a `canonicalRepo` that disagrees. Inference routing, so it needs the full E2E suite. |
| L-1 | Medium | The language banner asks "Switch the app to Deutsch?" in German while the app is already in German. | Start in English until it is answered, or reword to "The app is in {language}, your system language" with Keep and Use English. New strings in ten locales. |
| L-12 | Medium (demo) | Settings > Integrations prints the API token in clear text inside every snippet. On a screen share or a recording it is readable. | Mask it with a Show switch; Copy still copies the real value. |
| L-2 | Medium | Settings > Diagnostics shows about 100 English strings in every language. | Route the table labels through `t()`; keep the copied markdown in English. |
| L-20 | Medium | For the first seconds after a backend start Image Studio says "You don't have any image models downloaded yet" and disables the model list, then the real list appears. | A loading state in `useImageState` ("Checking your image models"). The same check is worth making for Video. |
| L-3 | Medium | Image Studio's empty state says the same thing three times, calls the engine "Stub diffusion pipeline" and the empty gallery "the first saved image artifact for this branch". | One notice, one button, plain words. |
| L-4 | Low | The Chat welcome recommends "a small Qwen3 or Llama 3 variant". | Refresh the copy, or read the first recommended row. |
| L-5 | Low | Backend-made text is not translated: activity feed lines, folder labels, "Released Nov 2026", the DFlash row, Compare's "Use for Model A". | Send keys plus values; format dates with `Intl`. |
| L-18 | Low | For a second or two after a load the composer says "Type a message" while Send is still disabled; Enter in that window is swallowed. With a library but no model loaded, Enter does nothing and says nothing. | Keep the loading placeholder until the same flag clears; show the Send reason under the composer. |
| L-13 | Low | The generation dialogs (`ImageGenerationModal`, `VideoGenerationModal`), `DataDirRestartModal` and the Mixer's overlay have no dialog role, Escape or focus handling. Escape does not collapse an expanded Video Gallery card or close the thread Export menu. | Same `useDialogFocus` treatment; decide what Escape means while a render runs. |
| L-19 | Low | The word-score match is still loose for model files outside the Hugging Face cache, in both directions. | Require a shared model name there too, tested against real file names. |
| L-6 | Low | 375 px: Compare's second slot and button run 33 px past the edge; Diagnostics values 5 px. | Stack the slots below 640 px; `overflow-wrap: anywhere`. |
| L-14 | Low | Panel header buttons wrap onto two lines ("Save settings", "+ Add Provider", "Inaktive stoppen"). | `white-space: nowrap` plus `flex-wrap` on `.panel-actions`, checked on every panel header. |
| L-7 | Low | The Language panel in Settings is titled "Settings"; the clock-format and UI-scale rows have no visible label. | Title "Language and time"; add labels. |
| L-9 | Info | There is no light theme (`color-scheme: dark`, no switch); the brief asked for light and dark. An OS set to light changes nothing. | Nothing to do unless a light theme is wanted. |
| L-11 | Info | The Dashboard's process table lists `llama-server` and `python` processes of other ChaosEngineAI instances as "ChaosEngineAI" with Stop buttons (seen: the other review streams' backends). | Mark processes this backend did not start, or leave them out of Stop idle. |
| L-21 | Info | The plan gave this stream a `phase_21` in `scripts/e2e_test_suite.py` for the seven features. Streams A, B, C and E each register their own phase 21 part for their features, so none was added here. | The integrator joins the four parts. |

## Per-tab result

PASS means: empty state, loaded state (on the data copy), 1100 x 760, 1400 x 920 and 375 px, backend stopped and back, no crash, after the fixes above. "Loaded" for the studios means the forms with installed models; no image, video, voice or music was rendered.

| Area | Result | Notes |
|---|---|---|
| First run and terms | PASS after D-1 | clean data dir; keyboard on the terms |
| Onboarding language banner | PASS after D-1, D-8 | wording left (L-1) |
| Dashboard | PASS after D-3 | L-11 |
| Chat | PASS after D-13, D-14, D-15, D-12 | real model: load, reply, stop, backend stop, unload |
| Chat > Compare | PASS | real run with two models; L-6 at 375 px |
| Chat > HTML Challenge | PASS (pages and dialogs only) | no challenge was run |
| Chat > Studio tools, Keep going | not run by hand | covered only by the existing specs 15 and 26, which pass |
| Chat Models and library | PASS after D-13, D-11, D-17 | search, filters, bad repo, launch dialog |
| Image Models, Studio, Gallery | PASS after D-2, D-18 | no render; L-20, L-3 |
| Video Models, Studio, Gallery | PASS after D-16 | no render |
| Video tools (FFmpeg row) | PASS (display) | install, download and remove not pressed |
| Voice Models, Studio, Library | PASS | runtime not installed in the test homes, so Studio was only its install prompt; library loaded |
| Music and Audio Models, Studio, Library | PASS | same: pack not installed; library loaded, audio loads |
| Projects (Storyboard) | PASS | sample project and the copy's 11 projects, seven tabs; nothing rendered |
| Settings (8 sections) | PASS after D-3, D-5, D-6, D-7, D-10 | L-2, L-12 |
| Setup (Accelerators) and Diagnostics | PASS | pseudo-locale at two sizes; no installer was run |
| Locales | PASS | ten locales boot; pseudo-locale layout |
| Backend stop and 401 recovery | PASS after D-4 | real kill and restart three times, plus spec 32 |

## Not tested

- Any real render: image, video, voice, speech-to-text, cloning, music, sound effects, storyboard render and export. The Studios were checked as forms only. One Iris-3B image was prepared but the model lock was held by another stream from 09:42 and the run was dropped.
- Chat with Studio tools on, Keep going, a voice call, attachments, documents (RAG), tool calls: only the existing mocked specs.
- Any install, download, delete or update button (voice runtime, music pack, LTX engine, TensorFold, video tools, model downloads, Delete model, Cleanup). The model downloads' error path was checked with a forced 500 only.
- The packaged Tauri app and its window: everything ran in headless Chromium against Vite. WebKit differences (the app's real web view) were not seen. Native dialogs (Choose folder, Reveal) were not opened.
- Windows and Linux.
- Screen readers. Keyboard was checked for the terms, the model dialogs, the sub-tab bar and the Chat composer, not for every control; Tab order was not walked page by page.
- Locales: boot and layout only. No translation was read for sense, and right-to-left was not tried (no such locale ships).
- The seven features owned by other streams, including the pages that share `ModelLaunchModal` and `ModelPicker` with Chat (their dialogs changed with D-11 and were not opened here).
- The full pytest suite and the full `e2e_test_suite.py` (stream G). Run here: four pytest files, 206 passed.
- The hub pages were not rebuilt and no FU rows were added, to keep the streams' branches from colliding in `FOLLOW_UPS.md` and `docs/hub`; this file is the record.

## Final checks (tree at the last code commit, `6a10c1fd`)

| Check | Result |
|---|---|
| `npx tsc --noEmit` | PASS, exit 0 |
| `npx vitest run` | PASS, 154 files, 1828 tests |
| `npm run i18n:validate` | PASS, en 4954/4954, nine locales 4947/4954 (the seven nulls: four from before, three new strings of this branch), tlh outside the gate |
| Playwright, whole suite, one worker (10:04) | PASS, 209 passed, 0 failed, 12.2 min |
| pytest `test_backend_service`, `test_cache_strategies`, `test_route_contracts`, `test_hub_tracker`, `test_model_arch` | PASS, 206 + 45 passed |
| Processes and ports | my backend and Vite stopped, 8903 and 8933 free, model lock released, no model worker of mine left |
