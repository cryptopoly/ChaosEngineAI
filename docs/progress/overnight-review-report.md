# Overnight whole-app review: morning report

Run of 2026-11-18 on the `release/v0.10.0` tip `b31c4ea5` (plan: [overnight-e2e-review.md](../plans/overnight-e2e-review.md), D-87, D-88, FU-220).
Everything is on **`review/integrate`** (pushed). Nothing was merged into `release/v0.10.0`, `staging` or `main`; nothing was deleted.
Integration log with evidence per step: [review-integrate.md](review-integrate.md).

## Headline

**`review/integrate` is green on the Mac gate.** All eleven review branches (A, A2, B, C, D, E, F, GGUF, ROUTING, PLAN, I18NFIX) are merged into it; pytest 4768 passed (17 skipped, 1 xfailed), vitest 1936, tsc and i18n clean, the full `e2e_test_suite.py` 25 of 25 phases with real models, Playwright 254 of 254, `pre-build-check.sh` 13 of 13 with 3 warnings. Fine Tuning and Plugins are hidden behind a new "Show experimental tabs" switch (one commit, `41f5a92a`, reverts alone); the Server page is rebuilt with a model timeline and absorbs Logs; API keys are masked on screen. Nothing is merged into `release/v0.10.0`: that waits for your yes. Not tested: Windows, Linux, the packaged app, real renders on this branch.

## Verdict per feature

| Feature | Works | Honest | Valuable | Verdict | Evidence |
|---|---|---|---|---|---|
| **Benchmarks** | Yes, with real MLX and GGUF models (throughput 323 tok/s on Qwen3-0.6B MLX; perplexity and MMLU after fixes) | After fixes: the 4.0 GB cache figure for a 0.6B model came from a 7B guess (now 0.88 GB from the model's own layout); the constant "Quality 100%" is gone, only measured values show (a2) | Yes (the only place to compare runs) | **Keep** | [review-a](review-a.md) rows 1 to 45, [review-a2](review-a2.md); phase 211 |
| **Conversion to MLX** | Failed at HEAD for any model in the Hugging Face cache (raw traceback); works after the fix: 4-bit and 8-bit, the result loads and chats | After fixes (plain errors for missing repo, missing path, bad dtype, low disk; Cancel added) | Yes on a Mac | **Keep** | review-a rows 8 to 19, 38 to 41; phase 211; a converted folder with "gguf" in its name now loads on MLX ([review-gguf](review-gguf.md), phase 215) |
| **Fine Tuning** | No. Nothing trains, the Start button was dead, an adapter passed at load is ignored | No (now says "not built"; the route answers 501) | Not as built; yes if built (mlx-lm trains a 0.5B model here in under 10 s) | **Hidden** behind "Show experimental tabs"; build later (FU-221, FU-222) | [review-b](review-b.md); phase 212 |
| **Prompts** | Partly; 12 defects fixed (dates, save refusals, search, delete confirm, the suite writing into the real library) | No: "Use in Chat" sets an app-wide system prompt that Chat never shows, lost on reload; presets never applied | Yes, the only way to give Chat a system prompt | **Keep visible**, merge into Chat later (FU-224, proposal `docs/plans/review-c-prompts-into-chat.md`) | [review-c](review-c.md); phase 213 |
| **Plugins** | No. Lists the built-in cache strategies and tools; switches change nothing; no outside code is loaded | No (the page and README promised a plugin system) | No (MCP servers, Skills and the agent bridge cover it) | **Hidden**; remove after v0.10.0 unless a loader is planned (FU-225) | review-c; phase 213 |
| **Logs** | Folded into Server > Activity; old lines still show; `logs` redirects | Yes | Yes, as part of Server | **Merged into Server** (D-88) | [review-e](review-e.md) |
| **Server** | Yes. Rebuilt: Overview and Activity, collapsed connect line, structured model events and a timeline per model, verified with real MLX and GGUF loads | Yes; the API key is now masked on screen | Yes | **Keep (redesigned)**; owner to merge | review-e, screenshots in `docs/progress/review-e/`; phase 214; spec 31 |

The hide is one commit, **`41f5a92a`**, that reverts alone (`git revert 41f5a92a` lists both tabs again). The switch itself is Settings > General > Appearance > "Show experimental tabs" (`7c0556ba`), off by default and remembered.

## Defects fixed (selection, ranked by user impact)

| # | Defect | Stream | Fix |
|---|---|---|---|
| 1 | API key in clear text in the Server samples, the Test dialog, Settings > Integrations and Agent access (on a screen share) | F, integrator | Masked to dots and the last 4 characters until Reveal; Copy copies the real key (`d6aa020b`, spec 39) |
| 2 | The test suite wrote one junk template into the real prompt library per run (505 so far) | C | Tests use their own store (`28ce7b20`); the rows already written are yours to remove |
| 3 | Conversion failed for every model in the Hugging Face cache (raw traceback); a converted folder could not load | A | Snapshot folder, output name, disk check, plain errors, Cancel |
| 4 | Fine-Tuning crashed whenever one adapter was on disk; Start was a dead button; the route said "accepted" for a job that never ran | B | Card fixed, Start off with a sentence, 501 (`b8dab48f`), then hidden |
| 5 | Plugins always said "No plugins found" (broken since April) and pointed to a folder nothing reads | C | Reads the real list; honest text; then hidden |
| 6 | Chat model picker gave a 0.6B model the size and badges of a 39.8B catalogue row | F | Format and quant alone no longer match |
| 7 | Benchmarks: perplexity failed for every model (wrong dataset id); a 4.0 GB cache figure for a 0.6B model; speed deltas shown under accuracy runs | A | Dataset id, model's own layout, mode-aware cards |
| 8 | Dashboard said "70B class @ Native f16" and the activity feed "70B GGUF" on an MLX Mac | F, integrator | "70B class at 4-bit"; activity written after the MLX probe (`59da5ffa`) |
| 9 | Speak started during a video render waited 3 min with a still bar and no word | F | Queue route and a waiting line |
| 10 | Server page opened on a huge connect block; nothing showed a model's load, run and unload | E | Overview and Activity, timeline and event log (D-88) |
| 11 | Prompts: dates printed as epoch seconds, a save with no name did nothing, search ignored the text, delete had no confirm, deleting every template brought the five back, one bad row made every call a 500 | C | Each fixed with a test |
| 12 | Projects opened on "This project is not there any more" after a data-folder change | F | Remembered project checked against the current folder |
| 13 | **Critical:** a local model that is not itself a catalogue row could not load, or loaded as another model (a small Qwen was refused as "Qwen3.8 Flash Next runs only on TensorFold"); 10 of 20 chat models on your library matched a wrong row | D | A cache folder names its own repo; the word score needs the repo or model name (D-13, spec 37-chat-local-model, real load) |
| 14 | First launch on any non-English system at 1180 px and wider: the language banner took the sidebar's place | D | Banner gets its own row; Keep English applies at once (D-1, spec 34) |
| 15 | Chat in a new thread opened no thread (the message went to the last one and its model was reloaded); every Stop raised "Streaming timed out" | D | D-14, D-15 |
| 16 | At the smallest window (1100 x 760): Image Studio's prompt covered, Dashboard and Settings panels cut with nothing to scroll, Settings unreachable in the drawer | D | D-2, D-3, D-5 (spec 33) |
| 17 | A stopped backend showed only as a small badge (inside the closed drawer below 1180 px) | D | Offline notice at the top of the workspace (D-4, spec 32) |
| 18 | Model dialogs: no dialog role, Escape did nothing, Tab walked into the page | D, F | `useDialogFocus` (D-11, spec 35) |
| 19 | Video Gallery fetched every clip whole on open (about 100 MB on your library) | D | Lazy per card (D-16, spec 36-video-gallery-lazy) |
| 20 | Composer said "Type a message" while Send was still off after a load | D, integrator | Says the model is loading (`3a57ca9b`) |
| 21 | Benchmark Cancel only stopped waiting: the backend finished, saved the run and kept the model; progress dialogs showed timed, not real, phases; zero-shot accuracy scored below chance | A2 | Cancel stops the run (200 with `result: null`, a second run at once is 409); real progress for benchmarks and conversion; next-token scoring with `taskScoringMethod` saved per run |
| 22 | Any path or name containing "gguf" was loaded as a GGUF model (an MLX folder named `my-gguf-notes-mlx` failed) | GGUF | Format read from the files (`helpers/weights_format.py`), phase 215 |
| 23 | The backend believed a `canonicalRepo` from any client (CLI, agent, an old window), so D-13 could still happen outside the app | ROUTING | An HF cache path is its own repo; a disagreeing hint is ignored (phase 216, `test_routing_hints.py`) |
| 24 | A process psutil could not read (macOS `SystemError`) broke the system snapshot and app start; seen once in a full pytest run | integrator | Caught in the scan, with a test (`fb29b69a`) |
| 25 | Settings > Diagnostics showed about 100 English strings in every language; the first-launch banner asked a German user, in German, whether to switch to German; Compare and HTML Challenge labels were English | I18NFIX | 80 Diagnostics keys in nine locales; banner states the language in use, "Use English" always in English (D-94); Compare labels through `t()` |


## Defects left (ranked)

| # | Impact | Defect | Where |
|---|---|---|---|
| 1 | High (demo) | Your real settings have LAN access on and "Require API token" off | Owner action, before the podcast |
| 2 | High (feature) | Fine Tuning trains nothing and an adapter passed at load is ignored | FU-221, FU-222 (hidden meanwhile) |
| 3 | Medium | Prompts: the applied system prompt is app-wide, never shown in Chat and lost on reload; presets never applied | FU-224 |
| 4 | Medium | Plugins: switches change nothing, no loader; remove or build | FU-225 (hidden meanwhile) |
| 5 | Medium (demo) | The Studio plan card takes 40 to 60 s with Qwen3.8 27B; the model picks 1280x720 unless told | FU-228, plan FU-231, FU-232, FU-240, FU-241 |
| 6 | Medium (demo) | Keep going is not demo-ready with small fast models (broken calls, loops) | Plan D-89, D-90, FU-230 to FU-239 |
| 7 | Medium | Image Studio says "no image models" for the first seconds after a start, then the list appears; its empty state repeats itself | FU-251 (L-20, L-3) |
| 8 | Low | Fine-Tuning page leftovers: 4 s uncached adapter scan, wrong size, absolute paths in the list, promises in the caption and CLI help | FU-223 |
| 9 | Low | Hard-coded colours in Fine-Tuning, Prompts and Plugins (not theme tokens) | FU-226 |
| 10 | Low | Unreferenced Logs code after the fold into Server; no agent tool reads the model events | FU-227 |
| 11 | Low | Video Studio heads-up says a chat model holds about 30 GB with a 15 GB model loaded | FU-229 |
| 12 | Low | A browser with no backend shows "Extracting embedded runtime"; the Dashboard lists other instances' workers with Stop buttons | FU-247 |
| 13 | Low | Backend-made text (activity feed, folder labels, release dates) is English in every language | FU-252 |
| 14 | Low | 891 new translations have not been read by a native speaker (list below) | FU-254 |
| 15 | Low | `npm run i18n:extract` reports 2977 false missing keys, so the checklist's "zero new keys" check cannot be run | FU-253 |
| 16 | Low | The rest of the sweep's copy, i18n and accessibility items (Image Studio empty state and first seconds, generation dialogs without Escape, 375 px overflow in Compare, panel header buttons wrapping) | FU-251 |
| 17 | Info | The app has one (dark) theme; every stream noted it ignores the OS light setting | Not a defect unless a light theme is wanted |


## Translations that need a native speaker (from review I18NFIX)

All 891 new translations are machine-written. The 23 the stream is least sure of (keys in `src/locales/<lang>/`; detail in [review-i18nfix.md](review-i18nfix.md)):

- **Close to safety:** (1) `diagnostics:panel.reextract.confirm` and `.hint` ("delete the cached embedded runtime", "crashed install"); (2) `diagnostics:panel.cuda.confirm` and `.hint` ("imports as CPU-only", "~2 GB, 1-5 minutes"); (3) `diagnostics:panel.mlxVideo.confirm` and `.hint` ("shadowed the real one"); (4) `diagnostics:panel.reextract.deleted` must quote the Restart Backend button exactly.
- **Terms:** (5) `panel.rows.release`; (6) `panel.rows.machine`; (7) `panel.rows.prefix` and `.basePrefix`; (8) `panel.rows.warmPool` (older "warm pool" terms never reviewed); (9) `panel.sections.environment`; (10) `panel.sections.extras` (long in Russian); (11) `panel.backendOffline` keeps "sidecar".
- **Grammar:** (12) `panel.values.none` gender; (13) `panel.values.installed` and `.missing` after a package name; (14) `panel.cuda.unknownPython` gender; (15) `panel.logTail` Russian plurals, Korean, Chinese and Japanese; (16) `panel.captured` before a date; (17) `common:localeBanner.inUse` and `.keepCurrent` (endonym after a colon in French, Spanish, Portuguese); (18) `common:localeBanner.changeHint` must name each locale's sidebar entry; (19) `chat:compareView.*` slot names in the Russian nominative; (20) `common:statusBanners.backendOffline.detail`; (21) `chat:keepGoing.stop.job_stalled` and `.waitingOn` as lower-case fragments; (22) `chat:emptyBanner.welcomeBody` units.
- **Register:** (23) German mixes "du" (Diagnostics, Disk usage) and "Sie" (Language panel); older than this branch.

## API and behaviour changes to carry into the release notes

- Benchmarks: a cancelled run answers 200 with `result: null` and is not saved; a second run while one is going answers 409; `DELETE /api/benchmarks/{id}` and `GET /api/benchmarks/progress` are new; new runs carry no `quality` field; task accuracy is next-token scoring, saved per run as `taskScoringMethod`.
- Conversion: `POST /api/models/convert/cancel` is new; the result has no `qualityPercent` or `compressionRatio` any more, but `diskRatio`, `bitsPerWeight` and, when asked, a measured perplexity delta (`quality`).
- Fine Tuning: `POST /api/finetuning/start` answers 501 (it answered 200 "accepted" for a job that never ran).
- Server: `GET /api/server/events` and its stream are new; every old log endpoint still answers; `logs` as a saved tab id opens Server > Activity.
- Routing: GGUF or MLX is read from the files, never from "gguf" in a name; the backend takes the repo from the model's own folder and ignores a `canonicalRepo` that disagrees, so a local copy matched to another uploader's catalogue row loads under its own repo and a drafter pinned (`PINNED_DRAFTERS`) to the other uploader's repo id is not used for it; `chaosengine-cli load` prints `cacheBits`.
- UI: Fine Tuning and Plugins are listed only with Settings > General > Appearance > "Show experimental tabs"; API keys are masked on screen until Reveal.

## What the owner must do

1. **Turn LAN access off before the podcast.** Stream F found your real settings have LAN access on and "Require API token" off: anyone on the studio's Wi-Fi could use the server during the show.
2. **505 junk prompt rows in your real library.** `~/.chaosengine/prompt_templates.json` holds 510 templates, 505 of them "Phase 0 test template", written by the test suite before stream C fixed it. The clean-up command (keeps a backup) is in [review-c.md](review-c.md) under "For the owner". It needs your yes; nothing was touched.
3. **Windows and Linux fixes still have to be merged** into this result, then the full suites run again on the merged tree. The next session is on Windows: [windows-handover.md](../plans/windows-handover.md).
4. Decide D-87 removals (Plugins, and whether Prompts moves into Chat), then merge `review/integrate` into `release/v0.10.0`.
5. Rehearse with [demo-path.md](../plans/demo-path.md); the masked key and the Dashboard copy are fixed here, the other on-air risks are listed there.

## Verification on `review/integrate`

| Suite | Result | Tree |
|---|---|---|
| `pytest tests/` (throwaway HOME) | 4767 passed, 17 skipped, 1 xfailed; the one failure in that run was the hub tracker page I had left stale, regenerated and passing (so 4768 passed). No Python changed after it; `pre-build-check.sh`'s pytest step passes on the final tree | `599802fe` |
| `npx vitest run` | 169 files, 1936 tests passed | final (`ba1c5e4d`) |
| `npx tsc --noEmit` | clean | final |
| `npm run i18n:validate` (and `--strict`) | OK; nine locales 96.6 %, en 100 % | final |
| `e2e_test_suite.py --smoke` | 7 phases, 81 checks pass; the same phases pass again inside the full run below | `68e35083` |
| `e2e_test_suite.py` full, real models (Qwen3-0.6B MLX, Qwen2.5-0.5B MLX, Qwen3-0.6B GGUF; offline) | 25 of 25 phases pass (0 to 11, 14 to 20, 211 to 216): 178 checks pass, 53 skip with their reasons, 0 fail, 105 s | final (`e23b8c48`) |
| Playwright full, `--workers=1` | 254 passed in 10.9 min | final (`e23b8c48`) |
| `check-upstreams.py` | report only: llama.cpp pinned b12441, newest b12503 | |
| `audit-dependencies.py` | PASS: nothing blocking (one high, `picomatch`, dev-only) | |
| `pre-build-check.sh` (throwaway HOME) | 13 passed, 0 failed, 3 warnings (turbo binary not under the throwaway HOME; smoke skipped with no backend on 8876) | `06c52231` |
| `cargo test` | not run: no Rust file changed on any review branch | |


## Not tested

- **Windows and Linux:** nothing here ran on either. The next session is on the Windows PC ([windows-handover.md](../plans/windows-handover.md)).
- **The packaged Tauri app:** every browser check ran in headless Chromium against Vite, not in the app's own webview; no installer was built from this branch.
- **Real renders:** no image, video, voice, music or storyboard render on the integrated branch (streams D and F rendered on their own branches; F's demo path rendered clips four times).
- **Real-model coverage on the integrated tree:** only three small models (Qwen3-0.6B MLX, Qwen2.5-0.5B MLX, Qwen3-0.6B GGUF), offline: perplexity and MMLU (datasets not fetched), conversion (no Transformers checkpoint under 10 GB linked), DFlash, MTP, TensorFold, Kolibri, LFM3 and the CUDA lane skipped with their reasons. Streams A and A2 ran perplexity, MMLU and conversion with real models on their own branches.
- **Fine Tuning training:** not built, so not testable through the app (stream B trained a 0.5B model with mlx-lm directly).
- **Translations:** no native speaker read any of the 891 new strings (list above); the other locales were checked for raw keys and layout only.
- **A light theme:** the app has none.
- **Screen readers:** keyboard only, for the dialogs, sub-tab bars and the composer.
