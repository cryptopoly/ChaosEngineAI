---
title: Open and live follow-ups
order: 2
summary: 247 rows with work, decisions or watch items still open, newest first.
---

# Open and live follow-ups

Generated from the Follow-Ups Tracker in `FOLLOW_UPS.md` by `tools/hub_tracker.py`; edit the row there, not here. Rows with open work, a pending decision, a watch-list entry, or a shipped item that still lists open points. Newest first; open a row for the full note.


<details>
<summary>FU-534 · Local paths in music refusals reach agents — Next agent bridge pass</summary>

**Status or trigger:** Next agent bridge pass

**Notes:** Security review 2026-11-20. The weight-copy notice (`music_env._copy_notice_message`, `copy_plan`) names the checkpoints and cache folders; `make_music` and `make_sound` pass `MusicNotReady` text to agents as tool errors, where the bridge otherwise gives no local paths. Strip folder paths from the agent text (keep them for the UI).

</details>

<details>
<summary>FU-533 · WSL bridge PATH prefix and the setup steps' wsl -- — Next WSL bridge change</summary>

**Status or trigger:** Next WSL bridge change

**Notes:** Security review 2026-11-20. Under `wsl --` the `export PATH="~/.chaosengine/vllm-venv/bin:$PATH"` prefix was lost in the default shell's parse; the bridge now runs `wsl --exec` (the injection fix) and keeps it, but with a literal `~` that a child's `execvp` does not expand: use `$HOME`. `routes/setup/vllm_wsl.py` `_run_wsl_step` still uses `wsl --` with constant commands only; move it to `--exec` and recheck the probes' quoting on a WSL host.

</details>

<details>
<summary>FU-532 · WSL vLLM bridge always passes --trust-remote-code — Next WSL bridge change</summary>

**Status or trigger:** Next WSL bridge change

**Notes:** Security review 2026-11-20. Loading any Hub repo through the bridge runs that repo's Python inside WSL, which reaches the Windows profile through `/mnt/c`. Pass it only for catalog rows that need it (Qwen3-VL and the like), as a row flag; check the native vLLM lane the same way.

</details>

<details>
<summary>FU-531 · Hash-pin the music, sound and voice installers — Next installer change or release audit</summary>

**Status or trigger:** Next installer change or release audit

**Notes:** Security review 2026-11-20. `music_requirements.txt`, `sound_requirements.txt` and the voice add-ons install exact `==` pins without `--require-hashes`, with `--extra-index-url` for the PyTorch CUDA index (pip takes the highest version across both indexes). ACE-Step and Stable Audio 3 install from GitHub archives pinned by commit SHA but not by archive hash. Export with hashes (`uv export` without `--no-hashes`) or record a sha256 per archive.

</details>

<details>
<summary>FU-530 · Agent image inputs take any picture path — Next agent bridge pass</summary>

**Status or trigger:** Next agent bridge pass

**Notes:** Security review 2026-11-20 ([docs/review/security-review-2026-11-20.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/review/security-review-2026-11-20.md)). `generate_image` `images`, `generate_video` `start_image`/`end_image` and `make_lipsync_shot` `start_image` go through `resolve_input_image`, which accepts any PNG, JPEG or WebP on the disk; the Storyboard holds agents to `upload:` handles and output files (`_agent_source`). The picture reaches a downloadable output (frame 0 of a clip, an edit of the image). Decide whether agents get the Storyboard rule everywhere; the lip-sync `audio` already has it.

</details>

<details>
<summary>FU-514 · release-constraints.txt is frozen from the Mac test environment (Python 3.11.2) — Never rerun freeze-release-constraints.py on the Windows PC</summary>

**Status or trigger:** Never rerun `freeze-release-constraints.py` on the Windows PC

**Notes:** The Windows venv (Python 3.12.10) has 91 pins against 126 and no mlx; a rerun there would drop the Mac pins and add `+cu128` torch. The CUDA add-on packages (`kaldifst==1.9.0`, transformers 5.2/5.23, torch 2.9.1 cu128) are pinned in code (`voice_env.py`, `music_requirements.txt`), not in the constraints file, so it does not need a rerun for them. Rerun on the Mac after the last dependency change before the tag.

</details>

<details>
<summary>FU-513 · Two non-permissive licences ship or are offered: CUDA runtime DLLs beside sd.exe, FLUX.1 \[dev] Nunchaku weights — Before a Windows CUDA bundle is published</summary>

**Status or trigger:** Before a Windows CUDA bundle is published

**Notes:** `THIRD_PARTY_NOTICES.md` now lists both. The NVIDIA CUDA EULA allows redistributing `cudart`, `cublas` and `cublasLt` with the app (about 790 MB); confirm the CUDA EULA text is shipped beside them. `mit-han-lab/svdq-int4-flux.1-dev` carries the FLUX.1 \[dev] non-commercial licence: the row is user-downloaded, check the row's licence note says so.

</details>

<details>
<summary>FU-512 · Competing desktop AI app names remain in code, comments and docs — Owner decision on the two features</summary>

**Status or trigger:** Owner decision on the two features

**Notes:** Scan 2026-11-20 (`git grep` for the CLAUDE.md Editorial Rules names; the Hugging Face namespace `lmstudio-community` is allowed). User-visible: Settings model-storage "LM Studio models folder" and Library "Share with LM Studio" (`src/locales/*/common.json`, `library.json`, `ModelStoragePanel.tsx`, `ChatModelsPage.tsx`, `ImportModelsPanel.tsx`, `backend_service/routes/storage.py`), "Ollama-compatible apps" (`ConnectPanel.tsx`, `ollama_compat.py`). Not user-visible: `backend_service/helpers/model_import.py`, `routes/models.py`, `helpers/settings.py`, `scripts/e2e_test_suite.py`, `docs/research/m4max-generation-times.md`, `docs/plans/demo-path.md`, `docs/image-generation-mvp.md`. Another agent handles the user-visible copy; the rest needs a decision on whether importing from those folders stays a feature.

</details>

<details>
<summary>FU-511 · TensorFold native is behind: pinned 1.2.9, newest 1.3.1 — Before the tag, or write down why not</summary>

**Status or trigger:** Before the tag, or write down why not

**Notes:** `check-upstreams.py` 2026-11-20. A bump changes `NATIVE_VERSION` in `_tensorfold_native.py` and needs `capabilities --json` compared with the family rows in `_tensorfold.py` (FU-168). Not bumped by the hygiene pass.

</details>

<details>
<summary>FU-510 · llama.cpp is behind: pinned b12441, newest tag with prebuilt archives b12576 — Before the v0.10.0 tag, or write down why not</summary>

**Status or trigger:** Before the v0.10.0 tag, or write down why not

**Notes:** `check-upstreams.py` 2026-11-20. A bump moves `LLAMA_CPP_REF` in `scripts/update-llama-cpp.sh` and `LLAMA_CPP_TAG` in `scripts/stage-runtime.mjs`, changes the Windows CUDA digests, and needs the chat rows re-run on the 4090 (the 4090 speed table was measured on b12441). Not bumped by the hygiene pass.

</details>

<details>
<summary>FU-495 · Partial pruning can remove a partial another download is writing (POSIX) — If a companion download ever fails on its final rename</summary>

**Status or trigger:** If a companion download ever fails on its final rename

**Notes:** `start_download` prunes `.incomplete` files for the repo and its companions; POSIX `unlink` succeeds on an open file. With the resume patch there is one partial per blob, so the snapshot helper is safe; an in-process `hf_hub_download` of a companion next to an older, larger partial would lose its file. Prune only partials older than a minute, or only the repo being started.

</details>

<details>
<summary>FU-494 · Weight-copy advice names Windows Developer Mode on a Mac or Linux — When the music weight copy is next touched</summary>

**Status or trigger:** When the music weight copy is next touched

**Notes:** `music_env._copy_notice_message` and `MusicWeightCopyPanel` (`musicStudio.copy.wayDeveloper`) advise Windows Developer Mode whenever symlinks and hard links both fail; on a Mac or Linux that only happens on exFAT / FAT32 or across file systems, where the advice is wrong. Say "this file system does not allow links" there.

</details>

<details>
<summary>FU-493 · Mac run of the Windows / CUDA pass before it merges into the release — Before test/rtx4090 merges into release/v0.10.0</summary>

**Status or trigger:** Before `test/rtx4090` merges into `release/v0.10.0`

**Notes:** Not provable on the PC (`docs/review/mac-regression-review-2026-11-20.md`): (1) sounds pack reinstall on Apple Silicon (marker schema 2) with the new `stable-audio-3` step and verify import on MPS, then a Stable Audio Open sound; consider skipping the SA3 install on the `mps` lane, since its rows are CUDA-only; (2) lip sync through `ltx-2-mlx a2v --distilled` (never run); (3) a paused and resumed Hub download with xet; (4) ACE-Step on MLX after the CUDA lane changes; (5) Nemotron 3.5 Lightning GGUF with the raised 16K context; (6) Voice > Models collapsed rows and Dictation default; (7) the Image catalog without the Nunchaku rows.

</details>

<details>
<summary>FU-492 · Other app names left in UI strings after the share rewording — Next translation pass</summary>

**Status or trigger:** Next translation pass; D-12 for the Server tab label

**Notes:** The "Share with ..." button, its done / already messages, the folder label and hint, and three backend error details were reworded in English ("Link to shared folder", "Shared models folder", 2026-11-20). Still naming the app: the same five keys in the 9 other shipping locales (re-translate from the new English), `serverTab.connect.ollamaApps` / `ollamaHint` in every locale (protocol label, D-12 open), the API names `lmstudioDir` and `/api/models/share-lmstudio`, code comments and identifiers.

</details>

<details>
<summary>FU-479 · Translations and wording after review B — Open</summary>

**Status or trigger:** Open

**Notes:** English only, null elsewhere: `studio:musicStudio.packMissingCuda`, `library:models.fit.needsAccess`. English changed while the nine locales keep the older sentence: `common:benchmarkTab.noRunsYet`, `common:fineTuning.noAdaptersHint`, `studio:imageStudio.recent.emptyHint`, `studio:imageGallery.empty.firstGeneration`, `studio:storyboard.pathPlaceholder`. The failed-run sentence in `useImageState.ts` / `useVideoState.ts` is still a hard-coded English string. Wording to settle: Projects (sidebar) against Storyboard (buttons, "Send to storyboard" in lower case); "CUDA" chips on music rows; the Fine-Tuning line about Apple Silicon on a PC; dead string `musicUnsupported.soundReason`.

</details>

<details>
<summary>FU-478 · Image Studio: default row whose engine is missing; refusal dialog — Open</summary>

**Status or trigger:** Open

**Notes:** With the Nunchaku pack on disk but the Nunchaku package not installed, Image Studio opened on FLUX.1 Schnell Nunchaku INT4 with Generate live. Do not default to a row whose engine is missing. The refusal dialog repeats its title, adds "Adjust the prompt or runtime settings" whatever the cause, has no Download or Install button, and the same text also shows in a banner; the backend sentence in `helpers/image_validation.py` says "Download it in Image Studio first" (the download is on Image Models). Also: three notices for "not installed" on a first run under a chip that says "Real local generation available". Review M2, M8.

</details>

<details>
<summary>FU-477 · Phone width: H3 licence buttons under the fold, music rows clipped — Open</summary>

**Status or trigger:** Open

**Notes:** At 375 px the MiniMax H3 licence dialog cuts Not now / Confirm in half at its bottom edge since the D-99 text was added (pin the button row); Music and Audio Models rows clip the licence chip, the platform chips and the Accept licence button at the right edge. Review M5; shots `m-B-video-h3-licence-ticked.png`, `m-A-music-models.png`.

</details>

<details>
<summary>FU-476 · Models pages: loading state, icon-only status, "Stub diffusion pipeline", licence "Other" — Open</summary>

**Status or trigger:** Open

**Notes:** Image and Video Models show "No image models in the catalog yet. Check that the backend is running." while the catalogue loads (over 2.5 s on a first open, backend online): add a loading state (`ModelsPage` `emptyCatalog`). Ten image rows carry the runtime name "Stub diffusion pipeline", printed in the Image Studio header; renaming needs a look at `imageOrVideoVariantPlatformGate`, which reads the string. Status and row actions are icons only; music rows use a text pill in the same column. Licence reads "Other" where the note names it (FLUX 3 Klein, LTX-2.5). "Watching upstream" rows have a greyed download and no explanation. Fit tooltip says "Measured on this machine." for catalogue measurements. Review M1, M6, M7, L1, L2.

</details>

<details>
<summary>FU-475 · Video Studio defaults and the way to lip sync (D-109) — Open</summary>

**Status or trigger:** Open

**Notes:** Every row opens at 33 frames (1.4 s at 24 fps) under a chip that says "4.0s clip"; the SD pill is highlighted and labelled 896x512 while the size is 768x512; size pills past the model's limit are disabled but look enabled (no `.preset-pill:disabled` style); Expert offers a "4-bit (NF4)" switch on a row that is already 4-bit. Lip sync is reachable only from a shot's editor in Projects: nothing in Video Studio or on the LTX-2.3 row mentions it (D-109). Review M3, M4.

</details>

<details>
<summary>FU-474 · Projects offers Make with no model installed (High) — Open</summary>

**Status or trigger:** Open

**Notes:** Overview and Storyboard show "Make 3 missing" as the primary button, the shot editor "Make this shot", and Plan "Make them in Storyboard" / "Rework with the model", while the same panels say no image, video or chat model is installed. Offer "Open Image Models" / "Open Video Models" instead. The shot editor shows "Estimate: unknown until the first run" beside an "Estimated" chip. The vocal field accepts a 2 s file although the clip must last 3 to 8 s: say so on attach. Review H4, M4. Not verified: what the click answers.

</details>

<details>
<summary>FU-473 · Benchmarks: Run button out of sight, preview without a model, MLX-only modes on a PC (High) — Open</summary>

**Status or trigger:** Open

**Notes:** At 1280 x 900 `.benchmark-run-config.scrollable-panel-content` is 465 px tall with 774 px inside and the Run benchmark button at y 758, below the box's end at y 590; at 375 px 277 of 1,004 px show. Make the page one scroll or pin the Run bar. With no model picked the Performance preview still shows 113.7 tok/s, Disk 31.2 GB and "Fits easily". Perplexity and Task accuracy are offered where no MLX model can exist. Review H3; measurement in `capture-extra.spec.ts`.

</details>

<details>
<summary>FU-472 · Studio estimates against this month's measurements (High) — Open</summary>

**Status or trigger:** Open

**Notes:** Nunchaku INT4 FLUX.1 Schnell shows "About 2 s, Peak 10.2 GB" in Image Studio and "~10 GB, Fits est." on Models; measured in the app on the 4090: 5.9 s warm, about 18 GB (offload). FLUX 3 Klein opens at 24 steps although its recommended value is 8 (13 s shown, 7.0 s measured at 8); apply the row's recommended steps when it is picked, and hide CFG decay for rows without guidance. The "may struggle on this device" warning shows on rows measured working and marked Tight (FLUX 3 Klein, LTX-2.3): one verdict per row. Review H2. The Klein 2048 px and LTX-2.5 large-size errors of 2026-11-19 are still open under FU-165.

</details>

<details>
<summary>FU-471 · Models pages overstate the download size before download (High, D-108) — Open, waits on D-108</summary>

**Status or trigger:** Open, waits on D-108

**Notes:** The Size column and Video Studio prefer the sum of every weights file in the Hugging Face repo (`coreWeightsGb`) over the catalogue `sizeGb`: LTX-Video 236.4 GB shown for a 2.0 GB download, LTX-2.5 162.4 for 66.2 on disk, Qwen-Image-2.1 Turbo sd.cpp 60.4 for 3.9 on disk (table in the review, H1). `videoPrimarySizeLabel` / `imagePrimarySizeLabel` in `src/utils/format.ts` and the two `*SizeSortKey` helpers. Right fix: the backend sums the files its own download patterns match and sends that as the download size; until then prefer `sizeGb` for rows not on disk.

</details>

<details>
<summary>FU-470 · UI/UX review B for v0.10.0 (Image, Video, Projects, Music, Tools, galleries): small fixes — Done 2026-11-20 (agent branch, not pushed)</summary>

**Status or trigger:** **Done 2026-11-20 (agent branch, not pushed); unit-tested and seen in screenshots, no model run**

**Notes:** Review at [docs/review/ui-ux-v0100-b/README.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/review/ui-ux-v0100-b/README.md): 130 screenshots at 1280 and 375 px against two throwaway backends on the 4090 PC, 4 High, 8 Medium, 9 Low left open. Fixed: Video Studio no longer shows the MLX engine banner for LTX-2 on diffusers (`VideoStudioTab.tsx`, keyed on the ltx-2-mlx runtime, not on `fixedSampler`); the RAM/VRAM column shows the fit verdict's own number (`memoryAgreeingWithFit` in `components/models/fit.ts`; it read "~67 GB" beside "Fits"); the music pack reads 6.6 GB on the CUDA lane (`musicPackGb`, new string `musicStudio.packMissingCuda`); the music estimate strip compares the VRAM peak with the card (`estimate_total_gb` in `routes/music.py`; it read "8.6 GB of 63.8 GB"); a gated video row with no access says "Needs access" (`models.fit.needsAccess`, `Fit.accessBlocked`); accelerator cards lost the tracker-id chip; failed runs point to Server > Activity; five English sentences; "MIT"; e2e spec 04's stale sentence. Checks: tsc, vitest 1,998, i18n:validate, `tests/test_music_routes.py`, Playwright 01, 02, 04, 10, 24, 28, 42.

</details>

<details>
<summary>FU-463 · Performance preview shows numbers with no model chosen (Benchmarks &gt; Run, Select Model) — After D-105 C</summary>

**Status or trigger:** After D-105 C

**Notes:** "113.7 tok/s est., Very fast", "Disk: 31.2 GB", "Quality 100 %", "Fits easily" before any model is picked. Recommendation: hide the preview until a model is chosen.

</details>

<details>
<summary>FU-462 · Accessibility labels, skip link, English left in other languages (review A, P7) — Next accessibility pass</summary>

**Status or trigger:** Next accessibility pass

**Notes:** No accessible label on the Chat composer, Compare prompt, HTML Challenge title and prompt, "Run from Hugging Face", eight Storage path fields; one unnamed button on Integrations; 14 Tab stops before page content. German shows "Released Nov 2026" and catalogue notes in English. Contrast of the small muted text not measured.

</details>

<details>
<summary>FU-461 · Server page: LAN warning, one word per secret, tiles (review A, P6) — Before a release that advertises LAN access</summary>

**Status or trigger:** Before a release that advertises LAN access

**Notes:** With LAN on nothing says anyone on the network with the key can use the API; "API key", "API token" and "token" name one secret and Integrations shows a second (agent) token without saying how they differ; tiles repeat ("Port 8961 / Active"); checkboxes sit above their labels. 0.5 day.

</details>

<details>
<summary>FU-460 · Phone layout of Chat, popovers and the Dashboard table (review A, P5) — With the next compact-layout pass</summary>

**Status or trigger:** With the next compact-layout pass

**Notes:** At 375 px the thread list sits above the thread (about 200 px left for messages), the Voice popover opens at the top of the page, the Dashboard process table is a stack of four header words. About 1 day.

</details>

<details>
<summary>FU-459 · Chat polish: plan card fields, job card titles, header overflow, Prompts wording (review A, P4) — When Chat is next touched</summary>

**Status or trigger:** When Chat is next touched

**Notes:** Expert fields are unstyled browser inputs; "Use for the rest of this chat" does not say what; job cards are titled `generate_image`, `speak`; Ready wraps under a long model name; Prompts panel says prompts and templates for one thing and repeats the category in the tags; "search terms.Clear Filters" has no space; menu "Preferences…" against the app's "Settings". 0.5 day.

</details>

<details>
<summary>FU-458 · Windows wording on Storage, Skills and Accelerators; first-run jargon (review A, P3) — Before or soon after the v0.10.0 tag</summary>

**Status or trigger:** Before or soon after the v0.10.0 tag

**Notes:** "MLX cache" folder listed as MISSING on a PC; `~/.cache/...` beside `E:\...` paths; `/path/to/skill-folder` placeholder; Accelerators "CUDA only" on every row with an empty Status column and "a single pip install away"; the "NATIVE F16" header chip; "Cross-platform mode will prefer llama.cpp GGUF..."; "Response metadata is collapsed by default..."; "Their pages and routes stay either way". 0.5 to 1 day.

</details>

<details>
<summary>FU-457 · First run: "Start here" on the Dashboard, Chat Models opening on a recommended list (review A, P2) — After D-105 D</summary>

**Status or trigger:** After D-105 D

**Notes:** Dashboard card with the GPU named, one recommended model, Download, then Chat. Models: filters folded, a word beside each status icon, one download line whose size matches the row, with speed and time left (today the cell says 19.1 GB and the bar 20.3 GB because the drafter is counted in one). About 1.5 days.

</details>

<details>
<summary>FU-456 · One source for fit numbers (review A, P1) — After D-105 B and C</summary>

**Status or trigger:** After D-105 B and C

**Notes:** The Dashboard names the catalogue's best-fit row for this machine instead of the tier table and constant headroom; the memory column is titled for what it measures; `measuredGpuGb` on the other rows run on the 4090. About 1 day. `docs/review/ui-ux-v0100-a/README.md` section 5.

</details>

<details>
<summary>FU-455 · Keep going's panel scrolled the Chat thread sideways at 1280 px; web search badge read "SEAR"; Models header said "Unknown on disk" — Done (2026-11-20, agent branch)</summary>

**Status or trigger:** **Done (2026-11-20, agent branch)**

**Notes:** The panel was anchored to the left edge of its button, the last one in its group, and ran past the thread panel; focusing its first field scrolled the panel. It now opens leftwards (`right: 0`); inside the window at 1280, 1100, 900 and 700 px. Badge text comes from a four-letter slice, so `web_search` maps to "web". The header says "0 GB on disk" with nothing downloaded. Browser spec `43-ui-review-a`.

</details>

<details>
<summary>FU-454 · Windows and plain-language copy on Settings and the Dashboard — Done (2026-11-20, agent branch)</summary>

**Status or trigger:** **Done (2026-11-20, agent branch)**

**Notes:** Provider keys no longer claim "0600 permissions" (not true on Windows); the Hugging Face token text no longer says "Used by MLX conversion" (it is used for every gated download); the Dashboard subtitle and four offline lines no longer say "sidecar"; the Activity feed on a machine that is not a Mac says "GGUF runtime: yes/no" without three "MLX ...: no" answers (`activity.backendReadiness.detailGguf`). Changed English strings are `null` in the other ten locales.

</details>

<details>
<summary>FU-453 · Chat Models: the memory cell disagreed with the fit badge beside it — Done (2026-11-20, agent branch)</summary>

**Status or trigger:** **Done (2026-11-20, agent branch)**

**Notes:** The cell showed the catalogue's `estimatedMemoryGb` (a 28.6 GB file read "~23.7 GB") while the badge was judged on the backend's `fit.needGb` (33.5 GB). The cell now shows `fit.needGb` when there is a verdict (`adapters/chat.ts`); test in `chatAdapter.test.ts`. Open: the column is titled "RAM" on a PC where the figure is GPU memory (FU-456).

</details>

<details>
<summary>FU-452 · Dashboard recommended "70B GGUF" with "Headroom for 70B: 65 %" on a 64 GB PC with a 24 GB card — Part done (2026-11-20, agent branch)</summary>

**Status or trigger:** **Part done (2026-11-20, agent branch)**

**Notes:** `_best_fit_recommendation` was fed system RAM. A machine whose GPU has its own memory is now judged by the card (24 GB gives "14B"); test in `test_dashboard_recommendation.py`. Still open (FU-456): the tier table itself and the headroom percentage, which is a constant per tier (43 to 68), not a measurement.

</details>

<details>
<summary>FU-451 · Chat rows named "24 GB card" said "Won't fit" on a 24 GB card before download — Done (2026-11-20, agent branch)</summary>

**Status or trigger:** **Done (2026-11-20, agent branch)**

**Notes:** Without the file there is no GGUF layout, so the generic estimate (weights + 10 % + cache) gave 23.0 GB (Nemotron 3.5 Lightning Q4_0) and 24.6 GB (Qwen3.6 35B A3B Q4_K_M). A catalogue row may now carry `measuredGpuGb` (GPU memory on the RTX 4090 at 8,192 context, no drafter): 18.1 and 20.1 GB. `helpers/fit.py` uses it on a discrete NVIDIA card at the default context while no layout is read; basis "measured". Tests in `test_model_fit.py`, one of which checks every "24 GB card" row. Not re-measured at the 16,384 context Nemotron loads with. Open: the same field for the other rows run on the 4090 (Gemma 4 QAT 12B 8.1 GB, 26B-A4B 16.1 GB, Qwen3.6 UD-IQ4_XS 17.2 GB, Bonsai 2 8.2 GB), see FU-456.

</details>

<details>
<summary>FU-450 · Chat first run: green "Ready" badge with nothing loaded; the model picker said "No models match your search" with an empty library; a long unbroken word ran out of the reply bubble — Done (2026-11-20, UI/UX review A, agent branch)</summary>

**Status or trigger:** **Done (2026-11-20, UI/UX review A, agent branch)**

**Notes:** `ChatHeader` compared two undefined refs as equal. Fixed there, in `ModelLaunchModal` (new string `modelLaunchModal.emptyLibrary`) and with `overflow-wrap: anywhere` on `.markdown-content`. Tests: `chatHeaderReady.test.tsx`, browser spec `43-ui-review-a`. Review: `docs/review/ui-ux-v0100-a/README.md`.

</details>

<details>
<summary>FU-441 · E2E assertions for the GGUF draft length and sidecar choice (FU-264, FU-266, FU-268) — Next chat speed pass on the 4090</summary>

**Status or trigger:** Next chat speed pass on the 4090

**Notes:** Unit tests hold the draft length 2 for the Qwen3.6 35B MTP repos and the one-mmproj and `MTP/` folder rules; the phase 1 GGUF checks run real loads but do not assert either. Put the draft length and the chosen sidecar in the load's runtime note or fields and assert them where those files are on disk.

</details>

<details>
<summary>FU-440 · Live check that MiniMax H3 videos carry the machine-generated tag — When H3_RELEASED is turned on</summary>

**Status or trigger:** When `H3_RELEASED` is turned on

**Notes:** `tests/test_licence_gate.py` covers the tag; phase 5 checks the notice, the tick sentence and the live link in process while H3 is unreleased. Once released, make one short H3 clip in phase 5 and read `machine_generated` from the file's settings.

</details>

<details>
<summary>FU-439 · E2E check for the Windows 260-character path answer in Storyboard — Next Windows E2E work</summary>

**Status or trigger:** Next Windows E2E work

**Notes:** `tests/test_storyboard_path_errors.py` maps WinError 206 and 3 to a plain 400. An E2E check needs a storyboard folder deep enough to cross 260 characters: point `CHAOSENGINE_STORYBOARDS_DIR` at a deep temp folder for a throwaway backend, import a take, and expect the 400 sentence. Windows only.

</details>

<details>
<summary>FU-438 · Live check for tool-call repair advice (FU-255) — Next tool-call scorecard run</summary>

**Status or trigger:** Next tool-call scorecard run

**Notes:** The lone-backslash advice and the other repairs in `agent_call_repair.py` are unit-tested only. Add a scorecard case (`scripts/tool-call-scorecard.py`) whose expected call carries a Windows path, so a real model's slip is repaired or advised as the tests say.

</details>

<details>
<summary>FU-437 · Mac-only E2E gaps from the coverage audit — Next Mac session</summary>

**Status or trigger:** Next Mac session

**Notes:** The MLX tool block with allowed values and limits (FU-231 part 1, `tests/test_tool_prompt_limits.py`) has no live check: add one to phase 1 that loads a small MLX model with Studio tools and reads the prompt. Lip sync on the Mac (FU-370) has no run at all. Neither can pass on the Windows PC.

</details>

<details>
<summary>FU-436 · E2E check of the sd.cpp device label (cpu or cuda read from the binary) — When a route exposes the sd probe</summary>

**Status or trigger:** When a route exposes the sd probe

**Notes:** `sd_binary_device` is unit-tested and the Video Studio shows the label, but no route returns the sd probe, so the suite cannot read it. Expose `device` and `deviceName` on the image or video runtime route, then check in phase 4 that a CUDA build reads `cuda` and a CPU build `cpu`.

</details>

<details>
<summary>FU-435 · Live E2E checks for the WSL vLLM bridge (preflight, DFlash over the bridge, reasoning split, GPU share) — On the WSL2 box with vLLM installed through the app</summary>

**Status or trigger:** On the WSL2 box with vLLM installed through the app

**Notes:** Unit tests cover all of it (`test_vllm_wsl_engine.py`, `test_vllm_wsl_prereqs.py`); phase 6 only reads the install status. Add a phase 1 check that loads a small model through the bridge, asserts the engine, the reasoning split and the GPU share, and a DFlash run with its drafter; skip without WSL vLLM.

</details>

<details>
<summary>FU-434 · E2E check that FLUX runs fully on the GPU when it fits and offloads when it does not — When the image runtime reports the offload choice</summary>

**Status or trigger:** When the image runtime reports the offload choice

**Notes:** `tests/test_image_cpu_offload.py` covers the decision; phase 4 generates but cannot see which path ran. Put the choice in the image artifact's runtime note, then assert it in phase 4 for an installed FLUX row on a 24 GB card.

</details>

<details>
<summary>FU-433 · E2E check for the ACE-Step XL refusal on a shared GPU (FU-278) — Next music hardware session on the 4090</summary>

**Status or trigger:** Next music hardware session on the 4090

**Notes:** `tests/test_music_worker.py` holds the plain message; a live check needs another process holding VRAM so XL cannot fit. Add to phase 11 with a skip when the card has room or XL is not downloaded.

</details>

<details>
<summary>FU-432 · E2E check for resuming a paused Hub download from the partial file (FU-275) — When a small, stable test repo is chosen for network checks</summary>

**Status or trigger:** When a small, stable test repo is chosen for network checks

**Notes:** Only `tests/test_download_resume.py` covers the HF resume wrapper. An E2E check needs a real download: start a small repo, pause it part way, resume, and assert the bytes already on disk were not fetched again. Needs network; skip offline.

</details>

<details>
<summary>FU-431 · E2E check for the Windows Job Object (workers end when the backend is force-killed) — Next E2E harness work on Windows</summary>

**Status or trigger:** Next E2E harness work on Windows

**Notes:** `tests/test_windows_job.py` kills a parent with `taskkill /F` and checks the grandchild dies, but no E2E check proves the real backend attaches its job: the suite cannot kill the backend it is testing. Add a phase 7 check that starts a second throwaway backend, loads a small GGUF, force-kills it and asserts llama-server is gone.

</details>

<details>
<summary>FU-421 · Native-speaker review of the 2026-11-20 translations — Before the release notes claim the nine locales</summary>

**Status or trigger:** Before the release notes claim the nine locales; adds to FU-254

**Notes:** 119 flagged strings in `docs/progress/i18n-translation-doubts-2026-11-20.md`, per locale. Common themes: "Kept warm", "Prefill" and "Word timings" wording, the "Voice > Models > Voice runtime > Add-ons" menu paths (must match each locale's real labels), "Share with LM Studio" quoted inside `modelStoragePanel.lmstudioHint`, Windows Settings path names in `musicStudio.copy.wayDeveloper`, the word for shot in `board.editor.vocalHint`, and plural agreement in `serverActivity.paused`.

</details>

<details>
<summary>FU-420 · Final translation pass before the release — Near the release/v0.10.0 merge (D-97: strings still change)</summary>

**Status or trigger:** Near the `release/v0.10.0` merge (D-97: strings still change)

**Notes:** The 2026-11-20 pass filled every null in the nine shipping locales (254 strings each; `npm run i18n:validate` reports 5265 of 5270 per locale). Left on purpose: the 3 `common.pluginsTab.*` keys (another stream removes the Plugins tab) and the 2 `meta.supportedLocales.10.*` fields of the experimental `tlh` entry. Re-run `node` null scan (any `null` leaf in a non-`en` file) for the Chat Prompts keys and whatever else lands, translate those, and rerun the validator. Nothing was pending in `src-tauri/locales/*.yml` or `backend_service/locales/*.po`.

</details>

<details>
<summary>FU-411 · Prompt templates: variables as rows, presets applied or dropped — With FU-410</summary>

**Status or trigger:** With FU-410

**Notes:** The panel's editor keeps the JSON box for variables. `presetModelRef`, `presetSamplers` and `fewShotExamples` are no longer shown but are sent back unchanged on save; nothing applies them. Apply the samplers through `writeSamplerOverrides` on use, or drop the three fields from the store.

</details>

<details>
<summary>FU-410 · Per-thread system prompt and a /prompt slash command in Chat — When Chat is next touched (D-87 follow-up)</summary>

**Status or trigger:** When Chat is next touched (D-87 follow-up)

**Notes:** The Chat Prompts panel (D-87) still sets the one app-wide `systemPrompt` from `useSettings.ts`: every thread gets it and a reload drops it. Plan in `docs/plans/review-c-prompts-into-chat.md` section 1 step 1 and 3: `chat.systemPrompt.<sessionId>` like `samplerOverrides.ts`, a chip in the composer, `/prompt`, `/prompt <name>`, `/prompt off`.

</details>

<details>
<summary>FU-399 · Windows pre-release verification: full bundle build with the CUDA sd.exe, the extra 790 MB against the 1.9 GB installer archive limit, a real Nunchaku image after the 409 fix — When the Windows code is ready (owner asked 2026-11-20), before the release/v0.10.0 merge</summary>

**Status or trigger:** When the Windows code is ready (owner asked 2026-11-20), before the `release/v0.10.0` merge

**Notes:** From the sd.cpp Windows build work (FU-300 to FU-302, FU-301): only the staging block was run against a scratch folder; the Nunchaku ready path was untouched but not re-run. Do these with the full E2E, pytest and Playwright runs, then the hands-on checks (D-99).

</details>

<details>
<summary>FU-373 · Vocal start offset per Storyboard shot — With FU-371</summary>

**Status or trigger:** With FU-371

**Notes:** `audioStartSeconds` exists on the request, but `Shot` has no field for it, so a Storyboard shot always uses its vocal file from 0 s.

</details>

<details>
<summary>FU-372 · Attach a vocal in the Video Studio — When lip sync is used outside the Storyboard</summary>

**Status or trigger:** When lip sync is used outside the Storyboard

**Notes:** The API (`audioPath`) and the agent tool take it; only the Storyboard Shot drawer has a field for it. A Video Studio audio picker, shown for rows with `lipSync`, would show the 3 to 8 s rule and the estimate.

</details>

<details>
<summary>FU-371 · Phase 2: cut a song into lip-sync shots in the Storyboard — Owner's go after phase 1 (D-98)</summary>

**Status or trigger:** Owner's go after phase 1 (D-98)

**Notes:** Not started on purpose. Phase 1 takes one vocal per shot (`Shot.vocalClip`, the whole file from 0 s). Phase 2: slice the song's vocal (stem or mix, FU-331) at the cut plan's shot boundaries with a 0.3-0.5 s pre-roll, attach each slice, two takes per sung shot, rank with FU-334. Phase 1 hint for the pre-roll: a Kokoro line that starts at 0 s and fills the shot scored r 0.27 against chance 0.36-0.40 though the strip shows the mouth talking and shutting after the line; the spike's clip with a 0.4 s lead-in scored 0.50 (chance 0.16).

</details>

<details>
<summary>FU-370 · Run Mac lip sync (ltx-2-mlx a2v --distilled) for the first time — Next M4 Max session</summary>

**Status or trigger:** Next M4 Max session

**Notes:** Wired behind the same `audioPath` API for `dgrauet/ltx-2.3-mlx-q4` (`lipsync_runtime.mlx_a2v_command`, unit tests check the command line against ltx-2-mlx 0.18.0's parser), never run. Check: the pack's `audio_vae.safetensors` holds the encoder half a2v needs (the download list was made for text to video), time and peak memory per 5 s 768x512 shot, sync on the spike's speech clip, that the output carries our prepared vocal after the re-mux. Then fill the row's `lipSync` block (now `basis: untested`) and an estimate.

</details>

<details>
<summary>FU-363 · kaldifst pin for the speech GPU add-on — When kaldifst ships a Windows wheel again</summary>

**Status or trigger:** When kaldifst ships a Windows wheel again

**Notes:** `kaldifst==1.9.0` is pinned in `voice_env.SPEECH_GPU_PACKAGES` because 1.9.1 has only a source archive for Windows and its build fails there. Move with wetext when a newer kaldifst has `win_amd64` wheels.

</details>

<details>
<summary>FU-362 · Listen to the GPU Chatterbox, VoxCPM2 and Orpheus clips and clones — Owner, before the tag</summary>

**Status or trigger:** Owner, before the tag

**Notes:** Nobody has listened to the 2026-11-20 clips (`E:\tmp\vg-out`, `E:\tmp\vg-out3`): the checks were length, level, silence and a Whisper transcript. Likeness of the clones to the Kokoro sample is not measured. Chatterbox Multilingual V3's long plain clip peaks at full scale (1.0): listen for clipping. Orpheus's short line is 45 % quiet frames (trailing pause).

</details>

<details>
<summary>FU-361 · VoxCPM2 on Windows: twice slower than real time, 11 GB of host memory — Next 4090 pass</summary>

**Status or trigger:** Next 4090 pass

**Notes:** Through the routes VoxCPM2 took 19.9 s per 10 s of speech (design) and 19.0 (clone) at about 3.3 steps per second, against 9.5 s per 10 s on a short text in the 2026-11-12 run and 5.2 on the M4 Max; the worker held about 11 GB of RAM (Chatterbox about 4.8 GB). Check whether `voxcpm` compiles or falls back on Windows (no Triton), bf16 vs fp32, and whether the weights stay on the CPU after the move to the GPU.

</details>

<details>
<summary>FU-352 · Dictation and Word timings downloads not tried end to end — Next 4090 session with bandwidth</summary>

**Status or trigger:** Next 4090 session with bandwidth

**Notes:** The new bar, Cancel and resume-after-reload were seen only with mocked `/api/models/download` answers in the review recipe; a real Qwen3-ASR 0.6B (1.58 GB) and aligner (1.84 GB) download, Cancel mid-way, a reload while downloading, Delete of the aligner, and a Transcribe run with words after the aligner is in were not run.

</details>

<details>
<summary>FU-351 · Dictation cards: per-model guidance is unmeasured; Chat popover on default change — Next 4090 session</summary>

**Status or trigger:** Next 4090 session

**Notes:** The one-line guides say what each model is for from the model cards (languages, detection, translation, word timings); none ranks accuracy from a run here. Measure WER on a short multi-language set on the 4090 (Qwen3-ASR 1.7B and 0.6B, Parakeet, Nemotron ASR, Whisper turbo, Granite) and put the numbers in the guides. Also: say in Chat's Voice popover when the dictation model switched to Qwen3-ASR after its download (D-102 C mentions it; the badge and line are in).

</details>

<details>
<summary>FU-350 · Backend text may still say "(GPU)" on a PC — Next voice UI pass</summary>

**Status or trigger:** Next voice UI pass

**Notes:** P1 drops "(GPU)" from speech model names where the catalogue hook hands rows out, so the Models table, Speak and Voices read the same; text the backend writes with a name in it (a voice's `needs` line, notes on a Speak result) was not checked and may still carry the suffix. Drop the suffix in `catalog/voice_models.py` display names once the Mac twins are hidden on a PC, or keep it backend-side only for logs.

</details>

<details>
<summary>FU-340 · MiniMax H3 licence changes built; release still waits for a lawyer (D-99) — Built 2026-11-20, H3_RELEASED still False.</summary>

**Status or trigger:** **Built 2026-11-20, `H3_RELEASED` still False.**

**Notes:** Tick sentence now also accepts the MiniMax H3 Community License and its Acceptable Use Policy (old ticks lapse, audit file keeps them); licence screen and row note carry the hosted-service notice (V.2, V.5) and a link to the live licence page beside the pinned one; H3 videos get `machine_generated: true` and `model: "MiniMax H3"` in the mp4 comment tag through `generation_settings`. **Open:** flip `H3_RELEASED` only after a lawyer answers questions 1 and 3 in `docs/research/minimax-h3-licence-check-2026-11-20.md`; do not ship an `sd` build with H3 support in a worldwide release until question 2 is answered; a Report-misuse route and safeguards page are needed if the answer to question 1 is yes. Not run: no H3 weights downloaded or executed, the H3 command-line tests fail on this PC with WinError 193 (host).

</details>

<details>
<summary>FU-335 · ACE-Step 1.5 turbo ignored the lyrics in 3 of 3 tries — Next music pass</summary>

**Status or trigger:** Next music pass

**Notes:** Through `/api/music/generate`, 30 s, lyrics given, `vocalLanguage=en`: turbo sang other words ("Ooh, ooh", "I'm not alone", "I'm not afraid of the dark", by Whisper and Qwen3-ASR on the Demucs stem); XL turbo sang the lyrics word for word (1 of 2 variations exact). Check whether turbo needs the LM / thinking on, and default lyric songs for music videos to XL.

</details>

<details>
<summary>FU-334 · Automatic lip-sync score for takes — When sung shots get two takes (FU-331)</summary>

**Status or trigger:** When sung shots get two takes (FU-331)

**Notes:** The spike's mouth-opening vs envelope correlation (MediaPipe Face Landmarker, Apache-2.0) separates synced from unsynced speech but is weak for sustained singing and does not check mouth shapes. A SyncNet-style confidence (LatentSync's SyncNet is Apache-2.0) could rank takes.

</details>

<details>
<summary>FU-333 · Lightricks A2V pipeline as the low-VRAM / reference option (route a) — A 12-16 GB card target, or a quality comparison</summary>

**Status or trigger:** A 12-16 GB card target, or a quality comparison

**Notes:** Measured: dev 22B fp8 + Gemma 3 12B with `--quantization fp8-cast --offload cpu` peaks at 9.0 GB GPU, 29.8 GB host RAM, 270-284 s per 5.04 s 768x512 shot; speech sync r 0.45; on the sung vocal stem the mouth stayed mostly closed (closed while voiced 0.86). Needs 62 GB of new weights (fp8 dev 29.15, distilled LoRA 7.61, upscaler 1.0, `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized` 24.4; the `google/` Gemma repo is gated and the saved token has no access), its own venv (transformers < 5.15) and the FU-332 fixes.

</details>

<details>
<summary>FU-332 · Report two Windows bugs to Lightricks' LTX-2 (ltx-core 1.6.2, commit daa6b16) — Before shipping route (a), or when upstream fixes them</summary>

**Status or trigger:** Before shipping route (a), or when upstream fixes them

**Notes:** (1) `ltx_core/loader/buffers.py` `host_bytes()` uses `mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS`, which Windows lacks, so `--offload cpu` crashes at the first weight load; `mmap.mmap(-1, n)` works there (patched in the spike venv, then ran fine). (2) A mono WAV crashes the A2V audio VAE (2 input channels); the pipeline should up-mix. Not filed (no messaging without the owner's go-ahead).

</details>

<details>
<summary>FU-331 · Singing lip sync is partial: takes, pre-roll, stem vs mix — With FU-330</summary>

**Status or trigger:** With FU-330

**Notes:** Both routes open and close the mouth with sung phrases but a held or hummed passage can stay closed while the singer is audible (same passage across seeds 42/7 and both routes). Modality guidance 5 instead of 3 changed nothing. Full mix scored as well as the Demucs vocal stem on route (b) (r 0.41 vs 0.23 against their own envelopes), so the stem default is unproven. Test 10+ sung shots: stem vs mix, 0.3-0.5 s pre-roll, 2 takes, a tighter face framing; decide the default from that.

</details>

<details>
<summary>FU-327 · Voice i18n, accessibility and a Windows browser-test failure — Polish pass</summary>

**Status or trigger:** Polish pass

**Notes:** Backend text stays English in every locale (model summaries, language lists, notes, voice descriptions, "measured on"). Delete confirms use the system `confirm()`; the voice-sample picker is the browser's own file input; `e2e/tests/14-voice-speak.spec.ts` "Voices groups every voice" fails on Windows fixed 2026-11-20: the test was wrong, not the UI (Windows has no Mandarin add-on); `voiceMocks.ts` now pins the runtime to the Apple Silicon lane its lists describe. Dictation download progress is now announced (FU-323). The app has no light theme (app-wide, not Voice).

</details>

<details>
<summary>FU-326 · Voices list length — After FU-321</summary>

**Status or trigger:** After FU-321

**Notes:** 3,500 px at desktop and 7,760 px at 375 px with nine quick languages open. Collapse the quick languages except the UI language's; check the row layout at 375 px with the new "cannot run here" text.

</details>

<details>
<summary>FU-325 · Voice runtime install: sizes and progress — After D-102 D</summary>

**Status or trigger:** After D-102 D

**Notes:** Size copy done 2026-11-20 (agent branch): one source, `src/features/voice/runtimeSize.ts` (base 2.3 GB, Kokoro 0.33, speech GPU 5.7 measured, speech-to-text GPU 2.9 estimated); Models says "About {gb} GB with the add-ons ticked below" and sums the ticks, the add-on labels and the Studio gate ("about 2.6 GB with the Kokoro quick voices") read the same table. Still to do: say what each add-on is for; percent and current step on the bar (it has neither) with `role="progressbar"`; keep the chosen add-ons ticked through a reload while the install runs; Speak's first-run copy when no voice exists at all (Kokoro left out). About 1 day.

</details>

<details>
<summary>FU-324 · Voice numbers measured on a Mac shown on a PC — Next 4090 session</summary>

**Status or trigger:** Next 4090 session

**Notes:** Kokoro's Fit cell and every quick voice's first-audio time and memory come from the M4 Max run although a PC runs Kokoro through kokoro-onnx; measure it on the portable lane or label the figures. Natural voices on the 4090 take about 17 s per 10 s of speech and show a green chip: label a voice that is slower than real time and say so on the install card.

</details>

<details>
<summary>FU-320 · Voice review for v0.10.0: Windows GPU fixes — Done (2026-11-20, agent branch), unit-tested</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), unit-tested; not tried on a real install**

**Notes:** Second D-73 review ([docs/review/voice-v0100/](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/review/voice-v0100/README.md), 18 findings, 48 screenshots against throwaway backends). Fixed: a collapsed speech family shows the first row that runs here (a PC showed an Apple-only row); GPU rows, the voice picker and the greyed New cloned voice button say to tick the speech GPU add-on instead of "Apple Silicon Macs only" (rows carry `lanes`); no Download on Voices or Speak for a model that cannot run here; no second "Download ... for style" button for the same model; a GPU dictation model this GPU cannot run has no Download and cannot be picked; a line says Qwen3-ASR 1.7B becomes the dictation model once downloaded; the speech GPU add-on label names Qwen3-TTS and "about 5 GB". New English strings are `null` in the other locales. tsc, vitest 1,970, i18n validate, voice pytest files, 66 of 67 voice browser tests. Not run: full pytest, full browser suite, backend E2E, a real install or download.

</details>

<details>
<summary>FU-312 · Parakeet's own word timings: early starts and a slow decode — When the aligner is a default download, or a speed complaint</summary>

**Status or trigger:** When the aligner is a default download, or a speed complaint

**Notes:** Parakeet's native word starts are on average 84 ms early and ends 60 ms late (80 ms frames), median error 90 ms against 20 ms for the aligner; the words path costs +0.43 s per 30 s (a Python `DecodeStream` per token; +2.4 s on 155 s). Possible: use the aligner for Parakeet too when it is downloaded and the language is one of its eleven (Parakeet speaks 25), or batch the token decode. Low priority.

</details>

<details>
<summary>FU-311 · Word timings: languages and audio not run — A recording in the language, a Linux box, or an 80 GB card</summary>

**Status or trigger:** A recording in the language, a Linux box, or an 80 GB card

**Notes:** Only clean Kokoro speech in English and French was measured. Not run: Japanese and Korean (the aligner speaks them, but its word splitting needs nagisa and soynlp, which the add-on does not install, so those rows give a note instead), Chinese and Cantonese (characters as words, no clip), Nemotron ASR on auto-detect in a language outside the aligner's eleven (it tries the aligner; not measured), Nemotron 3.5 Transcribe (64.7 GB; Transformers 5.23 has `Nemotron3_5TranscribeWordTimestampModel`, not wired), accented, noisy, overlapping or musical audio, Linux, the CPU.

</details>

<details>
<summary>FU-302 · sd.cpp on Windows: what is not done — Before shipping sd.exe in a Windows bundle</summary>

**Status or trigger:** Before shipping `sd.exe` in a Windows bundle

**Notes:** Not done: a full `stage-runtime.mjs` / `build.sh` bundle with the CUDA `sd.exe` (only the staging block was run, against a scratch folder; the 790 MB of DLLs must still fit the 1.9 GB NSIS archive guard, unmeasured); the default CUDA architecture list (only `SDCPP_CUDA_ARCHITECTURES=89` was built; ggml's default list for other cards is slower to build and untested); a system-wide CUDA Toolkit path; Image Studio's banner still shows the diffusers device for an sd.cpp row until the first image; catalogue `runtimeFootprintGb` 11.0 for the Qwen-Image 2.1 sd.cpp row vs about 14.4 GB measured (unchanged here).

</details>

<details>
<summary>FU-301 · Nunchaku row answered HTTP 200 with a placeholder picture when not downloaded — Done (2026-11-20, agent branch), reproduced and re-checked through the app routes on the RTX 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), reproduced and re-checked through the app routes on the RTX 4090**

**Notes:** Reproduced on a throwaway home with an empty model cache: `POST /api/images/generate` for `black-forest-labs/FLUX.1-schnell-nunchaku-int4` answered **200** with a placeholder SVG saved to outputs, `runtimeLabel` "Placeholder image engine" and a raw Hugging Face "local_files_only" error (with the Hub reachable the same run would have started fetching the whole base repo). Fix: the route refuses a Nunchaku row whose pack or base parts are not on disk with a **409** saying "is not downloaded" (or "still downloading" / "incomplete, retry"), before any engine runs (`_nunchaku_generation_block` in `helpers/image_validation.py`); an empty or dangling pack file no longer counts as downloaded; and the image manager raises instead of returning a placeholder when a Nunchaku row fails to load or the diffusers runtime is missing. After the fix the same request answers 409 with the plain message. Tests: `tests/test_nunchaku_download.py` (block states, route 409 without running, no placeholder). The download itself already ends "failed" with a reason when the pack or base parts are missing (validation after the download).

</details>

<details>
<summary>FU-300 · stable-diffusion.cpp on Windows: CUDA build script, bundle staging, device label — Done (2026-11-20, agent branch), built and run through the app routes on the RTX 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), built and run through the app routes on the RTX 4090**

**Notes:** `scripts/build-sdcpp.ps1` existed but floated on `master` and passed `SD_CUBLAS` (ignored at the pin, so a "CUDA" build came out CPU-only). It now builds the pinned `SDCPP_REF` with `-DSD_CUDA=ON`, loads the MSVC environment itself and builds with the Ninja that ships with the Build Tools, finds a CUDA Toolkit (`CUDAToolkit_ROOT`, `CUDA_PATH`, `nvcc` on PATH) or, with `CHAOSENGINE_SDCPP_CUDA_REDIST=1`, downloads NVIDIA's portable 12.8 archives (sha256 checked, nothing installed system-wide), honours `SDCPP_CUDA_ARCHITECTURES`, and installs `sd.exe` plus `cudart64_12`, `cublas64_12`, `cublasLt64_12` (about 790 MB) beside it; `sd.version` records the backend. `update-sdcpp.ps1` compares against the pin. `stage-runtime.mjs` copies those DLLs with `sd.exe` on Windows and logs when no build is found (release CI still builds no `sd`, so CI bundles keep `sdCpp: null`). `probe()` (image and video) now reads the back end from the binary (`ggml_backend_cuda` inside it, or `ggml-cuda` beside it) and reports `cuda` with the NVIDIA card's name, `cpu` for a CPU build, `mps` on a Mac; every non-Mac host used to read "cuda". A successful sd.cpp image's status carries that device instead of the diffusers one. Measured: script run on a clean throwaway home, CUDA 12.8 redist, arch 89: configure plus build 4 min 34 s (the archive download about 10 min on this line); Qwen-Image-2.1 Q4_K, 1024 px, 40 steps, seed 7 through `/api/images/generate` with the binary found only through `%USERPROFILE%\.chaosengine\bin\sd.exe`: 35.1 s cold, **20.6 s warm** (denoise 16.0 s), peak VRAM 16.0 GB, pixel-identical to the 19 Nov direct run. Tests: `test_sdcpp_image.py`, `test_check_upstreams.py`, `test_stage_runtime_scripts.py`. Open: FU-302.

</details>

<details>
<summary>FU-292 · Draft length for the other MTP and drafter rows — Next 4090 pass</summary>

**Status or trigger:** Next 4090 pass

**Notes:** Pinned by measure now: Qwen3.6 35B-A3B GGUF 2 (FU-268), Gemma 4 12B, 26B-A4B and 31B QAT 2 (FU-264). Not measured on the 4090: Nemotron Lightning (`mtp-` sidecar, default 2, +6 % at 2), the dense Qwen3.6 27B MTP row (3 from the Mac), Qwen3.8 27B with the z-lab DFlash 2 drafter at 15 (only the Mac numbers). A block drafter can slow a MoE (FU-263), so any new DFlash pairing needs the plain run beside it.

</details>

<details>
<summary>FU-291 · Engine takes the largest mmproj on disk — Next engine pass</summary>

**Status or trigger:** Next engine pass

**Notes:** `_resolve_mmproj_path` (`llama_cpp_engine.py`) sorts by size and takes the biggest, so a repo folder that holds F32 (2.3 GB on the unsloth Gemma 31B) loads F32 although F16 or Q8_0 is as good. FU-259 stops new downloads fetching more than one, so this only matters for old folders and hand-made ones. Use the same preference as `_best_mmproj` (F16, Q8_0, BF16) when several are present. Not measured: image quality of Q8_0 against BF16 on these models (the Bonsai 2 Q8_0 read a test picture correctly).

</details>

<details>
<summary>FU-290 · Parallel ranged Hub download when the single stream is starved — Only if owners report slow downloads</summary>

**Status or trigger:** Only if owners report slow downloads

**Notes:** See FU-266: on a contended link the app's one stream got 0.35 to 1.3 MB/s while 16 ranged requests got 7 to 14 MB/s; on a quiet link the app was fine (23 MB/s). A fallback would split a file into 64 MB ranges after the stream's rate stays under a floor for a minute, write into the same `.incomplete` file and reuse `hf_resume_script.py`'s hash check. Weigh against the extra load on the Hub; ask the owner before building.

</details>

<details>
<summary>FU-280 · CUDA sounds lane: what is not tested — Linux NVIDIA box, a card under 12 GB, or a listening pass</summary>

**Status or trigger:** Linux NVIDIA box, a card under 12 GB, or a listening pass

**Notes:** Only Windows 11 + RTX 4090 was run. Not tested: Linux x86_64; a smaller card (the worker's VRAM check needs 3 GB for the small rows, 9 GB for Medium, but Open 1.0 peaks at 13.8 GB in float32 and has no check); flash-attn (would be faster; no Windows wheel assumed); Medium's first clip after a load costs about 9 s more. Nobody listened to the clips: the numbers say full length, -1 dBFS peak, not silent, plausible spectra. Small SFX writes near-identical channels (L minus R under -47 dBFS); the music rows are real stereo. The pack install pulls the 3.3 GB torch wheel through pip in one stream (slow through a VPN).

</details>

<details>
<summary>FU-279 · Stable Audio 3 on Apple Silicon — Next Mac session</summary>

**Status or trigger:** Next Mac session

**Notes:** The sounds pack now installs `stable-audio-3` on every lane (a Mac pack from before asks to be reinstalled once: marker schema 2), but the three Stable Audio 3 rows list only `cuda`: nothing was run on MPS. To do on an M-series Mac: install the pack, run each row (the library loads in float32 off CUDA), measure time and footprint, add `apple-silicon` and Mac `estimates`, and check the Open rows still run after the reinstall.

</details>

<details>
<summary>FU-278 · ACE-Step XL on a shared GPU — Done (2026-11-20, agent branch), unit-tested</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), unit-tested; the live shared-card run was not repeated**

**Notes:** With another program holding about 12 GB, XL loaded but ACE-Step refused every run with a message about batch size and duration. `music_worker_acestep.vram_shortfall_message` now checks free VRAM before each CUDA run (peak table: turbo 8.7 GB, XL 12.5 GB, +2 GB over 300 s, minus what the worker already holds) and says how much is free, how much the model needs, that another program may be using the GPU, and to close it or use the turbo model. The offload decision is unchanged. Peaks are from the 4090 runs in FU-139; other cards and the offload-on peak are estimates.

</details>

<details>
<summary>FU-277 · CUDA music lane: what is not tested — Linux NVIDIA box, a 12 to 16 GB card, or the planner turned on</summary>

**Status or trigger:** Linux NVIDIA box, a 12 to 16 GB card, or the planner turned on

**Notes:** Only Windows 11 + RTX 4090 was run. Not tested: Linux x86_64 (same pins plus the cu128 index), the CPU-offload path under 20 GB VRAM, the planner LM on the `pt` backend, Linux aarch64 (refused; needs the cu130 index), XL at 180 s and above. The same seed is not byte-identical on CUDA (FLAC sizes differ by about 200 bytes between runs; level and spectrum match).

</details>

<details>
<summary>FU-276 · Windows weight mirror may copy 10 to 20 GB — Done (2026-11-20, agent branch), unit-tested with links forced to fail</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), unit-tested with links forced to fail; not run on a PC without Developer Mode**

**Notes:** `music_env._place` still tries symlink, then hard link, but a large file that can be neither is no longer copied silently: `prepare_checkpoints` collects them, and the first load stops (`WeightCopyNotice`, a 409 through `MusicNotReady`) with the file count, the size, the two drives and the fixes (Developer Mode, or the cache on the same drive); the next try copies (via a `.partial` file; the acknowledged file list is kept in `checkpoints/.copy-notice.json`). Pointing ACE-Step at the cache instead is not feasible: it reads one merged checkpoints folder built from several repos and rewrites small files there. Decision D-100 (owner chose a Copy now button, built 2026-11-20): `music_weight_copy.py` runs the copy as a tracked job (`POST /api/music/weights/copy`, `GET .../status`, `POST .../cancel`; bytes done and total, current file, ETA; refuses with a plain message when the target drive lacks the files plus a margin of the larger of 1 GB and 5 %; each file via `.partial`, size-checked, then renamed, so finished files are skipped on a retry and a cancelled partial is removed and restarted; the load is held while it runs; Activity entry when done). The 409 detail is the `{message, needsCopy, fileCount, totalBytes, source, target, sourceDrive, targetDrive}` envelope; `MusicWeightCopyPanel` shows the button, progress, Cancel and Done, loading under the failed load in Music and Sounds studios (strings in `musicStudio.copy.*`, null in other locales). Tests: `tests/test_music_weight_copy.py`, `src/features/music/__tests__/weightCopy.test.tsx`, `src/api/music.test.ts`. Not run on a PC without Developer Mode, no Playwright spec (no music spec forces links to fail). Tests: `tests/test_music_env.py` MirrorTests.

</details>

<details>
<summary>FU-275 · Music download pause restarts from zero — Done (2026-11-20, agent branch), tested through the app routes</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), tested through the app routes**

**Notes:** Cause: huggingface_hub 1.33 writes each file to `<etag>.<random>.incomplete`, never reuses it, and only deletes it on a clean exit, so a paused (killed) download restarted at zero and left the partial behind; the same helper serves `/api/models/download`, so both paths had it. Fix: [hf_resume_script.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/hf_resume_script.py) wraps the library's private `_download_to_tmp_and_move` in the snapshot subprocess (`HF_SNAPSHOT_DOWNLOAD_HELPER`): a partial from an earlier run is kept (largest only), cut back 32 MB and finished with an HTTP range request, then checked against the SHA-256 blob name; no partial uses the original xet fast path. `start_download` calls `hf_cache_paths.prune_stale_partials` first (dead partials of finished blobs, extra copies), and the progress meter skips partials of finished blobs. Real test on SmolLM2-135M through the routes (throwaway HOME): paused at 100 MB, resumed from 71 MB, completed, no partial left. Not tested: a multi-GB music repo, a range-ignoring proxy (the library then restarts that file), a library that renames the private function (the wrapper is then not installed and downloads behave as before).

</details>

<details>
<summary>FU-274 · Sounds on CUDA, and Stable Audio 3 — Done (2026-11-20, agent branch), measured through the app routes on the RTX 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), measured through the app routes on the RTX 4090**

**Notes:** The sounds lane is `cuda` wherever music is (`music_sound_env.lane()`), with the music sentence on hosts with neither lane; the pack installs `torch 2.7.1+cu128` from the PyTorch CUDA index, checks `torch.cuda.is_available()`, and also installs the MIT `stable-audio-3` library at `790793b` (`--no-deps`; marker schema 2 records it and the lane). Three gated rows: Stable Audio 3 Small SFX and Small Music (3.49 GB, 1 to 120 s) and Medium (10.44 GB, 1 to 380 s), Stability Community Licence plus the Gemma Terms of Use for the in-repo T5Gemma encoder; float16, 8 steps, SDPA (no flash-attn on Windows). Warm on the 4090: SFX 5 s 0.41 s, Music 60 s 0.63 s, Medium 60 s 3.0 s and 380 s 15.4 s; VRAM peak (torch reserved) 2.0 / 2.0 / 8.3 GB. The Open rows also list `cuda` (Small 0.27 s, 1.0 15.6 s at 100 steps, 13.8 GB). Worker refuses a run with a plain sentence when free VRAM is short. Not adopted: `stable-audio-3-optimized` (TensorRT plans for sm_90 and sm_120 only, none for the 4090's sm_89; ONNX, TFLite, MLX and CPU-AMX builds need runtimes the app does not ship). Open items in FU-279 and FU-280.

</details>

<details>
<summary>FU-273 · New image and video rows from the 2026-11-19 landscape page — Done (2026-11-19, agent branch), measured on the 4090</summary>

**Status or trigger:** **Done (2026-11-19, agent branch), measured on the 4090**

**Notes:** Rows: FLUX 3 Klein on diffusers (CUDA, text to image, 7.0 s at 1024 px / 8 steps, 20 GB) and on sd.cpp Q8_0 (5.5 s, 11.5 GB); Qwen-Image-2.1 Turbo sd.cpp Q4_K_M (7.5 s, 14.5 GB); MOSS-Image SFT through a vendored, hash-pinned pipeline with a 4-bit text encoder (53 s at 50 steps, 22 GB); Kandinsky 6.0 Lite video with sound (110 s for 864x480 x 121, 17 GB). Open: Krea 2 Turbo (the owner must request gated access; no row), FLUX 3 Klein editing on CUDA (returned the reference unchanged without the prompt upsampler), FLUX 3 Klein Base CUDA row, the LTX-2.3 nf4 fallback (pin `OzzyGT` at `0b92179` or load the official pack in 4-bit), cards below 24 GB. `docs/progress/rtx4090-progress.md`, 2026-11-19 FU-273 section.

</details>

<details>
<summary>FU-272 · Run the stt-gpu add-on install end to end, and Nemotron 3.5 Transcribe on a large GPU — A good network line</summary>

**Status or trigger:** A good network line; an 80 GB card

**Notes:** 2026-11-19 RTX 4090: download.pytorch.org gave about 90 KB/s, so the add-on's torch 2.9.1 step did not run (the venv got torch 2.11.0+cu128 from the pip cache; the asr-site and import-check commands ran as written). Re-run `POST /api/voice/install {"extras":["kokoro","stt-gpu"]}` and the phase 8 check. Nemotron 3.5 Transcribe 30B-A4B (64.7 GB bf16) is listed fit-gated at 72 GB and untested; its processor and model class were checked on the meta device only. NVIDIA mentions FP8 / NVFP4 builds that might fit 32 to 48 GB cards.

</details>

<details>
<summary>FU-269 · Bonsai 2 27B GGUF Q2_0 (7.63 GB) on stock llama.cpp b12441 as a 24 GB row — Done (2026-11-20, agent branch), measured on the 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), measured on the 4090**

**Notes:** `prism-ml/Ternary-Bonsai-2-27B-gguf` `Ternary-Bonsai-2-27B-Q2_0.gguf` loads and decodes on stock b12441 CUDA, no fork: **direct 77.3, 77.0, 77.1 tok/s, prefill about 2,400; through `/api/models/load` 77.6, 75.7, 67.9 (the last while another job was on the GPU)**, load 5.6 to 8.2 s, VRAM 8.2 GB above idle for text and 13.8 GB peak through the app with the projector loaded (other programs may have shared the card). The Q8_0 mmproj (0.63 GB) read an image correctly (a generated picture: red background, the white word, the blue rectangle). Reply read: a sound two-sentence answer. Row added in the Bonsai family (`prism-ml/Ternary-Bonsai-2-27B-gguf`, 7.6 GB, vision); the catalogue comment and readme that said Bonsai 2's GGUFs need the fork now say only PQ2_0 and PTQ1_0 do. Not run: PQ2_0 and PTQ1_0 (fork), a quality comparison with the MLX pack, a DFlash drafter on it (the z-lab Qwen3.8-27B one is not paired).

</details>

<details>
<summary>FU-268 · Draft length for MoE MTP (Qwen3.6 35B-A3B) — Done (2026-11-20, agent branch), measured on the 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), measured on the 4090**

**Notes:** Qwen3.6-35B-A3B UD-IQ4_XS, baked-in MTP, direct on b12441 (same prompt and runs as FU-263): plain 170.4 tok/s; n=1 193.4; **n=2 196.9 and 201.7 (two sweeps)**; n=3 171.0 and 179.3; n=4 167.0. n=2 pinned for the three GGUF MoE repos in `_MTP_GGUF_DRAFT_N` (`inference/_mtp.py`); the canonical `Qwen/Qwen3.6-35B-A3B` value (3, used by the MLX lane) is unchanged, a test pins both. Earlier 'no gain' at 3 (174 vs 170) was the n=3 setting, not the model. The 27B dense GGUF (`ggml-org/Qwen3.6-27B-MTP-GGUF`) keeps 3: not run here.

</details>

<details>
<summary>FU-267 · Tier 2 for the look-up-first models — With tier 2</summary>

**Status or trigger:** With tier 2

**Notes:** Gemma 4 (12B, 26B-A4B) and Nemotron Lightning looked things up first on 14 to 19 of 36 tier-1 replies; tier 2 with real tool answers shows whether the action follows.

</details>

<details>
<summary>FU-266 · Check the app's own Hub download on a slow link — Checked (2026-11-20, agent branch)</summary>

**Status or trigger:** **Checked (2026-11-20, agent branch); no downloader change**

**Notes:** Two checks through `/api/models/download` on a throwaway HOME and HF cache. (1) While 16 ranged requests of mine and another agent's job shared the link, the app took 1.45 GB (`openbmb/MiniCPM5-2B-GGUF`) in 1,162 s, 1.3 MB/s average, 0.35 MB/s at the slowest, while the ranged downloader beside it held 7 to 14 MB/s. (2) On a quiet link 1.08 GB (`prism-ml/Bonsai-8B-gguf`) took 51 s, 23 MB/s. So the app's single xet stream is not broken, it just gets a small share of a contended link; a ranged downloader is only faster there. **Resume works on the slow link:** paused at 94 MB of `LiquidAI/LFM3-1.7B-GGUF`, the partial was cut to 71 MB (the 32 MB safety trim) and grew from there to completion (no restart from zero). Not tested: a proxy that ignores Range, a multi-GB file paused twice. Whether to add parallel ranges for contended links: FU-290.

</details>

<details>
<summary>FU-265 · Nemotron 3.5 Lightning cannot take the Studio tool list at the default 8,192 context — Done (2026-11-20, agent branch), checked live on the 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), checked live on the 4090**

**Notes:** Its template renders the 20 Studio tools as 9,228 tokens, so every request was refused with HTTP 400 (scorecard run 2). **Fixed** by a per-row `minContextTokens` (16,384 on the three Lightning GGUF rows): `state/lifecycle.py` `load_model` raises a smaller requested context to it after the model-max clamp, logs it, and puts one plain sentence in the loaded model's `runtimeNote` ("Context raised from 8,192 to 16,384 tokens: this model's template turns the Studio tool list into about 9,200 tokens..."). `requestedContextTokens` stays as asked, so the launch panel still matches and the next message does not reload; a larger request is left alone. No smaller tool list was built (FU-231 stays the long-term answer). **Live:** `scripts/tool-call-scorecard.py --context 8192 --runs 1` against the Q4_0 row through `/api/models/load` on a throwaway HOME: the load came back at 16,384 and all 12 requests got an answer (12 of 12 calls, 8 right tools, 4 right values, 9,349 prompt tokens), against 12 HTTP 400 before. Tests: `tests/test_backend_service.py::test_a_row_with_a_min_context_loads_larger_and_says_so`, `tests/test_catalog_text_families.py`. Not done: the MLX Lightning rows (no measure; the text block there is smaller) and other models whose templates are large.

</details>

<details>
<summary>FU-264 · Gemma 4 31B on the 4090 and its drafter — Done (2026-11-20, agent branch), measured on the 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), measured on the 4090**

**Notes:** QAT Q4_0 (17.65 GB + 1.2 mmproj), direct on b12441, 8,192 context: **45.0 and 45.0 tok/s plain; with `unsloth/gemma-4-31B-it-GGUF` `mtp-gemma-4-31B-it.gguf` (0.51 GB) as `draft-mtp`: n=2 58.9 and 61.1 (+31 to 36 %, 26 to 32 % accepted), n=1 57.0, n=3 57.1**; VRAM above idle 19.3 GB plain, 19.8 with the drafter; prefill about 1,770 tok/s. Pinned in `PINNED_DRAFTERS` (the engine's default of 2 is the best). Row size corrected from 17.0 to 17.7 GB and the note carries the numbers. Not run: the unsloth Q4_K_M 31B row with its `MTP/` sidecar (now resolved by FU-259) and any Q5 or larger quant.

</details>

<details>
<summary>FU-263 · ggml-org/Qwen3.6-35B-A3B-GGUF Q4_K_M (20.5 GB) as the 24 GB row, and its DFlash drafter — Done (2026-11-20, agent branch), measured on the 4090</summary>

**Status or trigger:** **Done (2026-11-20, agent branch), measured on the 4090**

**Notes:** The row is re-pinned from Q8_0 (36.9 GB) to `Qwen3.6-35B-A3B-Q4_K_M.gguf`; Q8_0 stays reachable as the `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF` row (37.8 GB, baked-in heads). Measured directly on the b12441 CUDA server with the app's flags (8,192 context, f16 KV, `--fit on`, ~500-token prompt, 300 tokens, 3 warm runs, two sweeps): **plain 175.9 and 177.7 tok/s; DFlash drafter (`dflash-...-Q8_0`, 15 per round) 119.0 and 122.1, only 6.5 % of drafted tokens accepted, so it is 32 % slower; `mtp-...-Q4_0` sidecar at two per round 204.3 (+16 %), at three 202.0.** VRAM above idle: 20.1 GB plain, 21.1 with the sidecar, 22.7 with DFlash. Because the app puts block drafters first (FU-178, measured on the dense 27B), this repo would have run DFlash and slowed down: `SIDECAR_KINDS_SKIPPED` in `helpers/text_gguf.py` now keeps `dflash-` out of this repo's download and out of `resolve_spec_sidecar`, so Speculative decoding uses the mtp- sidecar. Through `/api/models/load` (throwaway HOME): see `docs/progress/rtx4090-progress.md` 2026-11-20. Tests in `tests/test_text_gguf.py`, `tests/test_catalog_text_families.py`.

</details>

<details>
<summary>FU-262 · Install size labels for the WSL vLLM bridge, DFlash (CUDA) and TriAttention said about 2 GB — Done (2026-11-19, test/rtx4090)</summary>

**Status or trigger:** **Done (2026-11-19, `test/rtx4090`)**

**Notes:** Measured: the WSL venv is 9.4 GB (torch 2.14 CUDA, flashinfer, triton). TriAttention pulls vLLM and needs triton, so it does not install on native Windows. | `wslBridgePanel.readySummary` (en and the 10 locales; tlh keeps its pattern with the numbers changed), the panel comment and the installer comment say about 9 GB and 10-15 min; the catalogue rows say "~9 GB (vLLM 0.28+; in WSL2 on Windows)" and "~9 GB (pulls vllm; Linux or WSL2 only)".

</details>

<details>
<summary>FU-261 · WSL vLLM bridge: use the library folder, serve DFlash, show it in the Boost Pack — Done (2026-11-19, test/rtx4090)</summary>

**Status or trigger:** **Done (2026-11-19, `test/rtx4090`); DFlash command line tested, DFlash not run live (27B + drafter does not fit 24 GB)**

**Notes:** A repo-id load made vLLM download the model again; the Boost Pack "DFlash (CUDA)" row read only the Windows venv; the bridge dropped speculative decoding. | `state/lifecycle.py` hands a vLLM load the library entry's folder (HF cache resolved to its snapshot) when no path was sent; live: loaded from `/mnt/e/AI Models/...`, no download. The bridge builds `--speculative-config` (DFlash drafter from `dflash.get_draft_model`, `num_speculative_tokens` from the drafter config, `--max-num-seqs 8`) when the WSL vLLM is >= `VLLM_DFLASH_MIN_VERSION`, else runs plain and says why in `runtimeNote`. `dflashCudaAvailable` and `dflashCudaVersion` also count a new enough WSL vLLM (`dflash_cuda_via_wsl`). Tests: `test_vllm_wsl_engine.py`, `test_accelerator_capabilities.py`. A library folder on a Windows drive is read through `/mnt/<drive>` (slower to read than the WSL disk; a Hugging Face cache with real symlinks may not resolve there).

</details>

<details>
<summary>FU-260 · WSL vLLM bridge: GPU memory share, reasoning split, dead attention variable — Done (2026-11-19, test/rtx4090), live-checked with Qwen3-0.6B</summary>

**Status or trigger:** **Done (2026-11-19, `test/rtx4090`), live-checked with Qwen3-0.6B**

**Notes:** vLLM was started with its 0.92 default, so a 1.1 GiB model held 23.6 of 24 GB; the `<think>` block came back inside the reply; `VLLM_ATTENTION_BACKEND=TORCH_SDPA` is not a variable in vLLM 0.34 at all. | The bridge passes `--gpu-memory-utilization` (weights x 1.3 + 4 GiB, at least 6 GiB, cap 0.85, never more than the free memory; `CHAOSENGINE_VLLM_GPU_SHARE` overrides); live: 6.2 GB used instead of about 22. generate and stream_generate split `<think>` like the llama.cpp lane (`ThinkingTokenFilter`, `strip_thinking_tokens`, a server-side `reasoning_content`); the dead variable is gone. Tests in `tests/test_vllm_wsl_engine.py`. Not tried on a card above 24 GB.

</details>

<details>
<summary>FU-259 · Unsloth GGUF repos: MTP/ sidecar lands in a subfolder, and every mmproj precision downloads — Done (2026-11-20, agent branch)</summary>

**Status or trigger:** **Done (2026-11-20, agent branch)**; the Gemma 4 26B-A4B Q8_0 row was not run (does not fit 24 GB)

**Notes:** Found 2026-11-19 on the Gemma 4 26B-A4B Q8_0 row (`unsloth/gemma-4-26B-A4B-it-GGUF`): `_best_sidecar` picked `MTP/mtp-...-Q8_0.gguf`, but `resolve_spec_sidecar` scanned only the model's own folder, so the 0.46 GB file was downloaded and never used; `*mmproj*.gguf` fetched BF16, F16 and F32 (4.7 GB). **Fixed** in `helpers/text_gguf.py`: `resolve_spec_sidecar` also reads a folder named `MTP` beside the model (that one folder only, so a neighbour repo's drafter is still never used), and with the Hub file list the allowlist names one mmproj (`_best_mmproj`: F16, then Q8_0, then BF16, else the first by name) instead of `*mmproj*.gguf`. Tests: `tests/test_text_gguf.py` (subfolder sidecar, one mmproj, the Bonsai 2 pair, Gemma-style three precisions). **Old downloads:** the engine (`_resolve_mmproj_path`) still takes the largest projector on disk when several are there, so a folder that already holds F32 keeps loading F32; only new downloads shrink. See FU-291.

</details>

<details>
<summary>FU-258 · /v1/chat/completions with stream: true drops tool calls — Done (2026-11-19, test/rtx4090), not run live</summary>

**Status or trigger:** **Done (2026-11-19, `test/rtx4090`), not run live**

**Notes:** Found on the 4090 (2026-11-18). The non-streamed reply now returns `tool_calls` (structured from llama-server, or read from the text on the MLX path), but `StreamChunk` has no field for calls and the llama.cpp stream ignores `delta.tool_calls`, so a streaming client gets an empty reply with `finish_reason` stop. Needs a calls field on the final chunk and the OpenAI `delta.tool_calls` chunks. | Fixed: `StreamChunk.tool_calls`, the llama.cpp stream joins `delta.tool_calls` pieces, and with `tools` the reply holds its text until the calls are known, then sends content, an OpenAI `delta.tool_calls` chunk and `finish_reason` tool_calls (`test_openai_stream_*` in `tests/test_backend_service.py`, fake runtime only; no real llama-server stream was run).

</details>

<details>
<summary>FU-256 · Host-bound tests that fail on Windows by design of the test — Next pass on the Windows or Linux box, or when each spec is touched</summary>

**Status or trigger:** Next pass on the Windows or Linux box, or when each spec is touched

**Notes:** From the 2026-11-18 Windows gate (`docs/progress/rtx4090-progress.md`). Playwright, 18: `04-music-sounds-licence` 2 and `14-voice-speak` 1 (Mac-only text), `06-busy-advice` 3 and `37-chat-local-model` 1 (the mocked model is MLX, so Windows will not start it; a GGUF mock runs everywhere), `21-slow-row-labels` 1 (fakes darwin, no Mac row to label), `27-benchmarks-conversion` 10 (Conversion is Apple Silicon only; the perplexity run mocks an MLX model). pytest: `test_licence_gate` H3 2 (a shell script as the `sd` stand-in, WinError 193), `test_music_env` mirror 3, `test_time_estimate` 1 (Mac-only catalogue row), `test_chat_studio_once` 1 (a batch thread still writing `project.json` at tear-down).

</details>

<details>
<summary>FU-255 · A Windows path written unescaped in a model's tool call gets the wrong advice — Done (2026-11-19, test/rtx4090)</summary>

**Status or trigger:** **Done (2026-11-19, `test/rtx4090`)**

**Notes:** Found on the 4090 (2026-11-18): a call holding `C:\Users\...` with single backslashes does not parse as JSON, and the repair says "the JSON is missing 1 closing brace", so the model resends the same path. Either mend lone backslashes inside a string value (only where the result then parses and names a real tool), or say "write each backslash in a path as two". `tests/test_keep_going_recovery.py` now escapes its path, so it no longer shows this. | Fixed by the second way: a lone backslash now gets "a backslash in a path must be written as two" (`_lone_backslash`, `tests/test_agent_call_repair_paths.py`).

</details>

<details>
<summary>FU-254 · Native-speaker review of the review's 891 machine translations — Before the release notes claim the nine locales</summary>

**Status or trigger:** Before the release notes claim the nine locales

**Notes:** `docs/progress/review-i18nfix.md`, "Needs native-speaker review": 23 items (destructive-action confirms in Diagnostics, terms such as release, machine and warm pool, gender and plural agreement, the banner's endonym after a colon, the Russian slot names in Compare, German du and Sie mixed).

</details>

<details>
<summary>FU-253 · npm run i18n:extract cannot be the "zero new keys" check — Before the next release checklist run</summary>

**Status or trigger:** Before the next release checklist run

**Notes:** CLAUDE.md section 6 asks for it, but the script ignores the namespace given to `useTranslation("...")`: on an untouched tree it reports 2977 keys missing and 4086 unreferenced (for example `diskUsage.title`, which exists), and the tracked `.extract-report.json` is stale. Teach it namespaces or replace the checklist line with `i18n:validate --strict`.

</details>

<details>
<summary>FU-252 · Backend-made text is not translated — When the activity feed or the catalogue labels are next touched (about half a day)</summary>

**Status or trigger:** When the activity feed or the catalogue labels are next touched (about half a day)

**Notes:** Review D, L-5, backend part (`docs/progress/review-i18nfix.md`): activity feed lines, model directory labels, "Released Nov 2026" (about 20 hand-written `updatedLabel` strings in `backend_service/catalog/*.py` in three shapes), the DFlash row in Accelerators. The backend sends a key plus values; the frontend formats dates with `Intl`.

</details>

<details>
<summary>FU-251 · Copy, i18n and accessibility leftovers from the whole-app sweep — When each area is next touched</summary>

**Status or trigger:** When each area is next touched

**Notes:** Review D, ranked in `docs/progress/review-d.md`: Image Studio says "no image models" for the first seconds after a start (L-20, needs a loading state in `useImageState`, Video too) and repeats its empty state three times with "Stub diffusion pipeline" (L-3); backend-made text untranslated (L-5, FU-252; the frontend part, the Chat welcome L-4 and the Language panel title L-7 were fixed by `review/i18nfix`); after a load Enter with no model says nothing (L-18, second half; the placeholder half is fixed); generation dialogs, the data-folder restart dialog and the Mixer overlay lack dialog role and Escape (L-13); the catalogue word match is loose outside the HF cache (L-19); 375 px overflow in Compare and Diagnostics (L-6); panel header buttons wrap (L-14). The light theme (L-9) and other instances' workers (L-11, FU-247) are noted elsewhere; L-12 (key in clear text) is fixed on `review/integrate`.

</details>

<details>
<summary>FU-247 · No-backend splash and other backends' workers — When the shell start-up is next touched</summary>

**Status or trigger:** When the shell start-up is next touched

**Notes:** Review B: in a browser with no backend a reload shows "Extracting embedded runtime, first launch only", which is not what happens; the Dashboard process list shows other backends' Python workers with Stop buttons.

</details>

<details>
<summary>FU-246 · Stills-first approval for costly clip plans; first-frame preview for a single clip — Queued (2026-11-18), D-92</summary>

**Status or trigger:** Queued (2026-11-18), D-92. After FU-241

**Notes:** A plan whose clips cost over 5 minutes (proposed) is approved in two steps: the stills, then the clips, so the cheap look comes before the expensive render. Skipped when renders are auto-approved. For a single clip in Chat, an option on the card to make the first frame as a still first, for video models that take a start image.

</details>

<details>
<summary>FU-245 · Follow the draft: the final still starts from the kept draft still — Prototype after FU-244</summary>

**Status or trigger:** Prototype after FU-244; offered only if claim C12 holds

**Notes:** A draft and its final are different pictures today: another still model, another size, a new seed. Use the kept draft still as the reference picture of the final still where the quality model takes one; elsewhere keep the prompt and the locked seed and say on the card that the picture may differ. A clip's first frame already matches, because it starts from the shot's still; the motion is not promised. Stop: if the owner does not call two of three test shots the same picture, ship the seed lock alone.

</details>

<details>
<summary>FU-244 · Final for chosen shots from Chat and Keep going — Queued (2026-11-18), D-92</summary>

**Status or trigger:** Queued (2026-11-18), D-92. After FU-241

**Notes:** The Final card lists the shots with ticks, kept ones ticked. A `keep` action on `storyboard_shot` (the operation `set_kept` exists; only the tool action is missing; the bridge's golden tool list changes once). Keeping a shot locks its seed to the kept take's. Keep going takes a tier in its panel; on Draft the run stops with a new resumable code `review_ready` ("drafts are ready: pick the shots to keep, then Resume"). The upscale tick sits on the Final film card. Open point in D-92: films of more than 8 shots start on Draft.

</details>

<details>
<summary>FU-243 · Whole-project estimate before the first render: time, memory, disk and free space — Queued (2026-11-18), D-92</summary>

**Status or trigger:** Queued (2026-11-18), D-92. After FU-241

**Notes:** `storyboard/estimate.py` prices one batch today and the Chat card drops its disk figure. Add one figure for the film at its tier: both passes, music, voice, captions and the film render, with peak memory, disk and free space. Disk from this film's own files, else from this model's earlier outputs of a similar size, else "not estimated" (never a guess). A warning when the estimate is more than the free space less 5 GB (proposed); no free-disk check exists today. Shown on the first card of a film and in the Keep going preflight; the Storyboard run bar keeps its strip. Checked by claim C14. E2E: the estimate's shape for a new film in the storyboard phase.

</details>

<details>
<summary>FU-242 · Grammar-constrained tool calls on MLX — Parked. Revisit when the demo model is under 54 of 60 valid calls after FU-231 to FU-234, or claim C7 shows th…</summary>

**Status or trigger:** Parked. Revisit when the demo model is under 54 of 60 valid calls after FU-231 to FU-234, or claim C7 shows the llama.cpp path clearly better, or mlx-lm ships structured output

**Notes:** The MLX worker does not read `jsonSchema` today. Needs a new dependency (licence check and `THIRD_PARTY_NOTICES.md` per CLAUDE.md), a logits processor in the worker and care with the prompt cache. Cannot fix a valid but wrong choice, such as a large size.

</details>

<details>
<summary>FU-241 · Quality tiers (Draft, Standard, Final) in Chat: one definition, applied to every render, switch and Expert fields on the card — Backend built for single renders (2026-11-19, test/rtx4090): tools/studio_tiers.py (tier words, sizes, apply_t…</summary>

**Status or trigger:** **Backend built for single renders (2026-11-19, `test/rtx4090`):** `tools/studio_tiers.py` (tier words, sizes, `apply_tier`, `tiers_table`), `GET /api/chat/studio/tiers`, the chat's tier from the message (Standard by default) applied to `generate_image` and `generate_video` in `StudioTool` and shown on the plan card (`tier`, `qualityNotes`); tests `test_studio_tiers.py`, E2E phase 7 check, run live on the 4090 (FLUX.2 Klein 4B and LTX-2.5: Draft 12 s, Standard 55 s, Final 179 s for the clip). **Card buttons built (2026-11-19):** Draft / Standard / Final buttons with each tier's size and time on the plan card, the pick reaches the render (`TierHolder`, decision body `tier`), "Use for the rest of this chat" keeps it for the chat (in memory, resets when the app restarts), `qualityNotes` show what was brought down; vitest, Playwright at 90 % and 125 % UI scale with screenshots. **Expert fields built (2026-11-19):** under "Expert settings" on each image or clip row: width, height, steps, seed, fps (clips) and model, priced live by the backend (`POST /api/chat/studio/price`), checked against the tool's own schema and the installed models when sent (`clean_overrides`, HTTP 400 with a sentence), applied to the approved call and run as typed, above the tier (`TierHolder.exempt`); a declined call keeps what the model asked; tests in `test_studio_tiers.py`, `test_chat_studio_tools.py`, vitest and Playwright. Open: a visible chip to change the kept tier; tiers for `storyboard_generate` and Keep going (3.3 rest, FU-244); Final's size is 1.5x recommended until C10 is measured; claims C10 and C13 not run.

**Notes:** New `tools/studio_tiers.py` returns each tier's size, fps, steps, model per pass and upscale for a model or a film: Draft is half size with the fast still model (24 fps, clips only when asked), Standard the model's recommended size or the film's size, Final the largest that fits or the film's size with the quality still model. The same table maps to `draft: true`, `what: both` and `what: final`, so no new render path. Chat applies the chat's tier (Standard by default; a tier word or a size in the message sets it); a call above the tier runs at the tier with a note in the result and on the job card. The card gets three buttons with time and disk each, Expert fields (size, length, fps, steps, seed, model), validated overrides on the decision, and "use for the rest of this chat" with a chip. External agents are unchanged. `GET /api/chat/studio/tiers` with an E2E check. Layout checked at 90 % and 125 % UI scale. Stop: no Draft tier for single clips if it is under 1.5x faster than Standard.

</details>

<details>
<summary>FU-240 · Plan card: always show tier, size, length, frames, fps, steps, and a time for those settings — Part 1 built (2026-11-19, test/rtx4090): the card states the size, frames, fps and steps that will run (resolv…</summary>

**Status or trigger:** **Part 1 built (2026-11-19, `test/rtx4090`):** the card states the size, frames, fps and steps that will run (resolved the way `generate_image` and `generate_video` do) and a time from `variant_seconds` for exactly those (`_price_settings` in `tools/studio_gate.py`, test in `tests/test_chat_studio_tools.py`); the "at the recommended size" note stays only for a model the catalog cannot price. Tier name and the change-tier click are built (FU-241 card buttons). The 90 % and 125 % scale check is done for the tier card (Playwright, screenshots); at 125 % in a 1280x720 window the Chat thread pane is only a few lines tall, so the test opens its scroll boxes (a Chat layout issue, not the card's). Expert fields are built. Open: disk size (FU-243), a length line. D-91

**Notes:** Was: size, frames, fps and steps show only when the model passed them, and the time is the catalogue figure "at the recommended size" (a 3 min 36 s render was shown against a 1 min 29 s figure). Price with `variant_seconds` for the settings that will run; show disk when known. `tools/studio_gate.plan_item`, `StudioPlanCard.tsx`. Checked at 90 % and 125 % UI scale.

</details>

<details>
<summary>FU-239 · Keep going preflight panel — Queued (2026-11-18), D-89</summary>

**Status or trigger:** Queued (2026-11-18), D-89. After FU-235

**Notes:** Before Turn on: model loaded and its tool verdict, context size (warn under 16K), models and ffmpeg installed for the brief, memory against the largest job, the time and disk estimate for the film at the run's tier against the budget and free space (FU-243 when it lands), render approval chosen. Warns; only "no model loaded" blocks. Checked at 90 % and 125 % UI scale. `GET /api/chat/keepgoing/{id}/preflight`; E2E in `scripts/e2e_phases/keep_going.py`.

</details>

<details>
<summary>FU-238 · Keep going: time budget, wrap-up, step list, status line, resume after a stall — Queued (2026-11-18), D-89</summary>

**Status or trigger:** Queued (2026-11-18), D-89. After FU-237

**Notes:** `budgetMinutes` (off by default), a wrap-up at 80 % that renders the shots that are ready (the render already leaves unfinished shots out), a new resumable stop `time_up`. Status gains steps, elapsed, budget and one sentence ("Making clip 2 of 2, about 1 min 10 s left"), shown in the chip and the thread. After `job_stalled`, Resume offers to cancel the stuck job and redo the step (today it waits on the same job again). Strings in `en`, `null` elsewhere.

</details>

<details>
<summary>FU-237 · Keep going guided run: app-made project, stage tools, demo preset — Queued (2026-11-18), D-89</summary>

**Status or trigger:** Queued (2026-11-18), D-89. After FU-231

**Notes:** The app creates and binds the Storyboard project when it takes the brief; `create` is not offered in that run. Each nudge turn gets the tools of the film's stage (Plan, Make, Sound, Finish) plus job and read tools. Demo preset: two 3 s shots, music, cap 8, budget 10 minutes; it never preselects render approval. Fixes a nudge that names `storyboard_batch`, which Chat does not offer. New `keep_going/guided.py`. Ends with three timed rehearsals: live only if all three finish in the budget, else the recording.

</details>

<details>
<summary>FU-236 · Benchmarks mode "Tool calls" — After D-87 keeps Benchmarks and FU-230 has proven the figures</summary>

**Status or trigger:** After D-87 keeps Benchmarks and FU-230 has proven the figures

**Notes:** Tiers 1 and 2 of the scorecard as a benchmark mode with history: calls read, arguments valid, tasks done. E2E in the benchmarks phase.

</details>

<details>
<summary>FU-235 · Tool capability probe (3 calls, cached per model) and its API — Queued (2026-11-18), D-89 and D-90</summary>

**Status or trigger:** Queued (2026-11-18), D-89 and D-90. After FU-231

**Notes:** Three calls with canned answers on the loaded model: an enum, a nested object with a number, a recovery from an error. Verdict Good, Risky or Not for tools, saved per model, quantisation, engine and tool-prompt version. `GET` and `POST /api/chat/tools/probe`. Also reports a model on the vision runtime, which gets no tool list today. The catalogue's tools flag is then set from measured results. E2E: phase 7 (no model) and the model-loaded phase.

</details>

<details>
<summary>FU-234 · Native tool-call format on MLX for templates that take tools — Only if the scorecard still fails a family after FU-231 and FU-233</summary>

**Status or trigger:** Only if the scorecard still fails a family after FU-231 and FU-233; LFM2.5 first

**Notes:** Pass the tools to the model's own template, keep the text block as the fallback, send results in the template's role. Readers for the three formats exist. As JSON schema the list is 23,540 characters, so this follows the staged list. First check: the LFM2.5 template accepts tools. Stop if tier 1 stays under 54 of 60.

</details>

<details>
<summary>FU-233 · Agent loop: stop repeated calls, repair harmless argument faults — Queued (2026-11-18), D-90</summary>

**Status or trigger:** Queued (2026-11-18), D-90

**Notes:** No guard exists: a Studio turn allows 30 steps and a run 3 idle nudges, about 120 calls. Same tool, arguments and result twice: a corrective message; a third time ends the turn with a reason (`wait_for_job` exempt). Safe repairs only (a number as text, enum letter case, a value one step outside a range moved to the limit with a note); other faults stay errors that name the allowed values. New `agent_guard.py` (`agent.py` is over the soft cap).

</details>

<details>
<summary>FU-232 · Prompt cache: keep the fixed tool prefix across turns on hybrid models — Queued (2026-11-18), D-90</summary>

**Status or trigger:** Queued (2026-11-18), D-90. After FU-230 proves the miss (claim C2)

**Notes:** `mlx_worker_prompt_cache.py` saves the recurrent state only at the last prompt token, so a new user turn or a Keep going nudge re-processes the whole prompt with Qwen3.5, 3.6 and 3.8 (read from the code, not timed). Add a second saved state at the end of the system message and a log line per call (prompt tokens, reused, seconds). Must equal a fresh prefill in tests or it is not merged.

</details>

<details>
<summary>FU-231 · Honest, staged tool prompt on the MLX path — Part 1 built (2026-11-19, test/rtx4090): the text block prints allowed values, ranges, defaults and nested key…</summary>

**Status or trigger:** **Part 1 built (2026-11-19, `test/rtx4090`):** the text block prints allowed values, ranges, defaults and nested keys (`_schema_limits`, `tests/test_tool_prompt_limits.py`; the 20 Studio tools now 15,822 characters). **Part 2 built (2026-11-19):** the seven Storyboard tools reach the model only when the turn is about a film (`wants_storyboard` in `tools/studio.py`: a Keep going run, a film word in the message or the last six messages, or an earlier `storyboard_*` call; `tests/test_studio_staging.py`); the bridge's own list is unchanged. Open: counting the tool block in the history budget, temperature 0.2 for tool steps, the "or the model calls one" case. Not measured on MLX (Mac): the scorecard on the MLX models decides whether part 1 helps. D-90.

**Notes:** The text block prints allowed values, ranges, defaults and nested keys (today `list_models` shows `kind: string (required)` and nothing else; `generate_video` shows no limits). Core Studio tools always; the seven Storyboard tools (7,814 of 14,708 characters) only when the message asks for a film, a project is bound, or the model calls one. The history budget counts the tool block (it reserves 512 tokens against about 4,000). Tool steps at temperature 0.2 if claim C5 holds. The bridge's tool list and its golden test are unchanged.

</details>

<details>
<summary>FU-230 · Tool-call scorecard: a script that scores chat models on calls read, arguments valid and task done — Tier 1 built (2026-11-18, test/rtx4090); tiers 2 and 3 and the Mac runs open</summary>

**Status or trigger:** **Tier 1 built (2026-11-18, `test/rtx4090`)**; tiers 2 and 3 and the Mac runs open. D-90. First: it gates FU-231 to FU-236

**Notes:** Tier 1 is in `scripts/tool-call-scorecard.py` (12 prompts, scored with the bridge's `validate`; `tests/test_tool_call_scorecard.py`). It needed `/v1/chat/completions` to return `tool_calls` at all (fixed the same day; streaming is FU-258). First runs on the 4090 (GGUF, llama.cpp CUDA): Qwen3.8 27B Q4_K_M 50 of 60 right values, all 60 calls valid, the 10 misses all look-ups first as the Studio hint asks; Qwen3-0.6B 37 of 60; the 20 tools cost about 7,200 prompt tokens on this path; details in the research page. Not run: the four MLX models (Mac), so C7 still needs the MLX Qwen3.8 27B. Plan: `scripts/tool-call-scorecard.py`, three tiers (one call through the OpenAI-compatible endpoint with the Studio schemas; a short Studio task with no render; the demo film). Models on disk: Qwen3.8 27B MLX 4-bit, LFM2.5 24B-A2B MLX 4-bit, Qwen3 8B 4-bit, Qwen3 4B; extra: Qwen3.8 27B GGUF, the MTPLX speed build, LFM3 1.7B. Five runs each, counts not percentages, one model loaded at a time. Claims C1 to C11 and the proposed tool bar in [the plan](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/plans/keep-going-and-tools-v2.md); the same run makes the render measurements C10 and C12 to C15 for D-92. Output in `docs/research/tool-call-scorecard-2026-11.md`.

</details>

<details>
<summary>FU-229 · Video Studio heads-up overstates the chat model — When Video Studio is next touched</summary>

**Status or trigger:** When Video Studio is next touched

**Notes:** Says "A loaded chat model holds about 30 GB" with a 15 GB Qwen loaded (review F).

</details>

<details>
<summary>FU-228 · Studio plan latency and video size in Chat — Covered by the Keep going and tools v2 plan (D-89 to D-92)</summary>

**Status or trigger:** Covered by the Keep going and tools v2 plan (D-89 to D-92)

**Notes:** Review F: with Qwen3.8 27B the plan card takes 40 to 60 s (15 to 20 s processing the studio tool list), and the model picks 1280x720 unless told, doubling the render. Work items: FU-231 and FU-232 (tool prompt and its cache), FU-240 and FU-241 (size from the quality tier), FU-237 (Keep going demo run). Close this row when those land.

</details>

<details>
<summary>FU-227 · Logs page code left after the fold into Server — Owner agrees to delete</summary>

**Status or trigger:** Owner agrees to delete

**Notes:** `src/features/logs/LogsTab.tsx`, `LogsIcon`, locale keys `diagnostics.logsTab.*`, `common.tabs.logs`, `common.tabCaptions.logs` are unreferenced since D-88; `src/hooks/useServerLog.ts` was unused before. Also: no agent-bridge tool reads the model events yet (a `server_events` tool would let an agent see why a load failed).

</details>

<details>
<summary>FU-226 · Hard-coded colours in the Fine-Tuning tab — With FU-223</summary>

**Status or trigger:** With FU-223

**Notes:** About a dozen inline hex colours instead of theme tokens, so the stronger-borders switch and any later theme do not reach them. The Prompts and Plugins tabs that had the rest were removed (D-87); the Chat Prompts panel uses theme tokens.

</details>

<details>
<summary>FU-223 · Fine-Tuning page leftovers — When the tab is next touched</summary>

**Status or trigger:** When the tab is next touched

**Notes:** Adapter scan walks the whole HF cache on every open (4.15 s, uncached) and skips the Settings model directories; size counts checkpoints and `lora_rank` is null for mlx-lm adapters; `GET /api/adapters` returns the whole `adapter_config.json` (absolute dataset path); the sidebar caption "LoRA adapters and training", "or via llama.cpp", the empty hint and the CLI help still promise training; inline hex colours, not tokens.

</details>

<details>
<summary>FU-222 · Fine-Tuning: an adapter cannot be applied — With FU-221</summary>

**Status or trigger:** With FU-221

**Notes:** `LoadModelRequest.adapterPath` is accepted and ignored (base-model reply word for word). Wire through `state/lifecycle.py`, `inference/controller.py` (warm-pool key must include the adapter), the MLX worker `load(..., adapter_path=)`, the load reply and the model badge; refuse with a reason on GGUF, TensorFold and speculative paths. Touches inference routing: needs the full E2E suite.

</details>

<details>
<summary>FU-221 · Fine-Tuning: training is not built — Hidden behind "Show experimental tabs" (D-87) until wired</summary>

**Status or trigger:** Hidden behind "Show experimental tabs" (D-87) until wired

**Notes:** From review B (`docs/progress/review-b.md` sections 7 and 8). Needs a training worker subprocess (never in the FastAPI parent, releases the chat model first, goes through busy advice and the job queue), progress, loss and cancel, a dataset path in (no upload today; `prepare_dataset` unused and flattens chat rows), a configure form, adapters under the data directory. mlx-lm trains a 0.5B model here in under 10 s, so the engine is not the gap. `POST /api/finetuning/start` answers 501 meanwhile.

</details>

<details>
<summary>FU-220 · Overnight whole-app review: audit Benchmarks, Conversion, Fine Tuning, Prompts, Plugins, Logs and Server; redesign Server and Logs; demo path for the podcast (D-87, D-88) — Run done 2026-11-18; everything is on review/integrate (not merged to the release branch)</summary>

**Status or trigger:** **Run done 2026-11-18; everything is on `review/integrate` (not merged to the release branch). Owner reads `docs/progress/overnight-review-report.md`, then merges.**

**Notes:** Plan, streams A to G, safety rules, verdict rubric and morning deliverables: [docs/plans/overnight-e2e-review.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/plans/overnight-e2e-review.md). No end-to-end check exists today for the seven features (no `e2e_test_suite.py` phase, no Playwright spec; unit tests only for Plugins and Prompts); the run adds `phase_21` and a Playwright spec per tab touched. Nothing is deleted overnight.

</details>

<details>
<summary>FU-218 · Narration level: the fixed +8 dB (FU-216) suited one file, not the next — Shipped 2026-11-17 (branch fix/voice-level).</summary>

**Status or trigger:** **Shipped 2026-11-17 (branch `fix/voice-level`).**

**Notes:** **Found** in rehearsal 3: the Kokoro narration (voice library `1794914768_8c5f6778.wav`) measures -28.6 dB RMS (-29.2 over the whole file), peak -10.5 dB, so +8 dB left it about 0 dB over the music right outside its window (measured through the real two-pass export: 0.0 dB raw, 0.6 dB normalised, narration window against the bed the 0.3 to 1.8 s after it). **Done:** a new voice lane (`add_audio`, kind `voice`, no `gainDb`) is measured once with the bundled ffmpeg (about 25 ms for 3 s, in the staging step before the project lock, off every job slot) and stored in `gainDb` like any value ([audio_level.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/storyboard/audio_level.py)): the gain that brings the file's RMS to -14 dB, between 0 and +12, never past a +6 dBFS peak, +8 when the file cannot be measured (no ffmpeg, unreadable or silent file). Explicit `gainDb` (0 too) wins, stored lanes keep what they have, nothing is measured at render time. Description, AGENTS.md and MCP golden updated. **Measured** (real files, real `build_mix_graph` export, lead = narration over the ducked bed in its window, FU-216's 0.7 to 3.1 s window and convention): rehearsal 2 narration (-22.1 dB RMS, peak -3.7): +8 gave 11.2 dB, now +8.1 gives the same; rehearsal 3 narration: +8 gave 12.1 dB over the ducked bed and 0.0 dB (raw) / 0.6 dB (normalised) over the bed outside the window, now +12.0 gives 19.6 dB and 3.6 / 2.6 dB. **Not as asked, and why:** the brief said target -18 dB RMS and a -1 dBFS peak ceiling. Both fail its own check on rehearsal 2: -18 and -1 give +2.7 dB there, a lead of 4.5 dB (was 11.2), the bug again; the +8 of FU-216 on its -22.6 dB file landed at -14.6 dB, so the target is -14 and the peak ceiling a +6 dBFS backstop (the export normaliser limits peaks; with it off a hot narration can clip, as FU-216 already noted). The brief's +14 cap is +12: `gainDb` cannot hold more (model, mixer slider). **Open:** rehearsal 3 does not reach 6 dB over the bed outside the window with any gain: +12 gives 2.6 dB normalised (3.6 raw), +14 would give about 4.2 (5.4 raw). What it needs is a quieter bed: a music lane default near -4 to -5 dB (with the voice at +12, normalised: -4 dB gives 5.9, -6 dB gives 6.6; the bed is lowered at creation, order dependent, or the duck depth raised). Not added; the owner's call. Tests: `tests/test_storyboard_audio_defaults.py` (gain maths, real-file measurement, quiet file boosted more than a louder one, loud file left at 0, cap, peak limit, explicit value wins, measured once, failed or silent measurement falls back to +8, the synthetic 10 dB lead holds with the measured gain).

</details>

<details>
<summary>FU-217 · Flaky music test in the full suite: test_unload_while_a_track_is_made_is_refused_and_the_job_finishes — Queued (2026-11-17), test isolation</summary>

**Status or trigger:** Queued (2026-11-17), test isolation

**Notes:** Fails with `200 != 409` in about one full-suite run in three and passes alone and with its file (three runs, then the whole file). `unload()` asks `get_worker()` (the module singleton) while the test holds its own `self.worker`, so a worker left by an earlier test can make the two differ. Not a product fault as far as read. **To do:** make the test use the singleton the route uses (or patch `music_client._WORKER` in setUp and restore it in tearDown), then run the full suite five times.

</details>

<details>
<summary>FU-199 · Live check of the Keep going fixes with a small model — Done 2026-11-17 (branch fix/keep-going-recovery).</summary>

**Status or trigger:** **Done 2026-11-17 (branch `fix/keep-going-recovery`).**

**Notes:** Isolated backend (port 8981, throwaway HOME), LFM2.5-24B-A2B MLX 4-bit (about 14 GB resident, the only model loaded), Chat with Studio tools and Keep going on (cap 2, renders by plan card). Brief: a 2-shot 'Kettle ad' project, create only the project and shots this turn. The model hit the rehearsal's fault on its own: an untagged `storyboard_project create` call one closing brace short. The loop told it twice ('your tool call could not be read, so nothing ran: the JSON is missing 1 closing brace; send it again as <tool_call>...'), then the turn ended on the fragment and nudge 1 read: "Your last tool call (storyboard_project) could not be read, so it did not run: the JSON is missing 1 closing brace. Send it again as a tool call: storyboard_project." followed by "No storyboard project yet" and the create step. The next turn sent it as a proper call, created the project and both shots; the run was stopped by hand at the next plan card. A first attempt with a brief that went straight to `make_music` only showed plan cards (declined, run stopped `plan_declined`), no nudge. Not repaired because that call had no tag at all, as designed. Honest limits: a 24B-A2B model is a weak stand-in for Qwen3.8-27B; the mended path (`repair_call`, call with a close tag) was not seen live, only in tests on the real Qwen replies; assets listing, captions, preview and the FU-198 stop reasons were not reached live. Qwen3.8-27B not loaded (another agent was running).

</details>

<details>
<summary>FU-191 · Keep going for Studio chats (D-86) — In progress</summary>

**Status or trigger:** In progress

**Notes:** Branch `feat/keep-going`. Opt-in per chat, off by default, Studio chats only. A backend run controller (`backend_service/keep_going/`) reads each finished Studio turn and the saved Storyboard project, then stops with a plain reason or sends the next nudge (an app-written user message: what the project lacks, the last error word for word, one next step; the brief only when the history lost it). Stops: goal complete, nudge cap (default 20, 1 to 100), 3 nudges without progress, the same error 3 times, context nearly full, plan declined, edited or unanswered, approval expired, Stop, a new user message, model not loadable, two turn failures, app restart (Resume). Never nudges while a turn, plan card or chat job runs. Runs saved in `keep_going_runs.json` beside the chat sessions; the render grant (auto-approve for up to 12 h, or each plan card, nothing preselected) lives in memory and ends with the run. Progress and completion sit behind an adapter (Storyboard ships; a generic tool-chat adapter exists, not offered). Tests: `tests/test_keep_going_*.py`, `src/features/chat/__tests__/keepGoing.test.tsx`, E2E phase 7. Phase 2, free GPU memory while rendering: a run setting Auto / Always / Never (Auto: swap only when the render's estimated peak does not fit the free GPU memory, or unified memory on a Mac, but fits with the chat model's share freed). An approved render in a run's turn is kept for after the turn; the controller unloads the chat model, runs the renders, waits for their jobs, loads the same model with the same launch settings, then nudges; a model that does not load again stops the run (`gpu_reload_failed`), never retried (`keep_going/gpu_swap.py`, `swap_run.py`, `tests/test_keep_going_swap.py`). The real CUDA path is untested (no NVIDIA card here): the 4090 pass should run one Keep going film with Qwen3.8-27B and LTX on Auto. Not tested: the packaged window, other locales, a real long run with a strong tool model. Neighbour defect found: LFM2.5 tool calls written as `<|tool_call_start|>[name(args)]<|tool_call_end|>` text are not parsed by the agent loop.

</details>

<details>
<summary>FU-190 · Real-model, pause, Music track and touch pass (S-O, S-P) — Done</summary>

**Status or trigger:** Done

**Notes:** S-O (CLI token, unsaved text, Plan tab stops, Mixer probe, length chips) and S-P (Qwen3.8-27B `storyboard_song`, real Music track to cut plan to burnt-in export, pause/resume/cancel on a real Iris-3B Turbo batch, iOS 27 Simulator touch) merged; 12 defects fixed with tests (cut plan now takes `seconds`, 60 s prefill watchdog off for tool turns, lyric overlap, LTX clips 0.3 s short, touch targets 44 px). Notes in `docs/progress/v0100-o.md` and `v0100-p.md`. Open: LFM3-1.7B cannot drive `storyboard_song` (safe, nothing made); Qwen3-4B not tool-capable in the app; fixed by S-Q (`docs/progress/v0100-q.md`): compact sidebar drawer below 1180 px, Export caption count, music test gate. Still open: Plan suggestion reads spelled-out numbers only for Han, not German or Korean; drawer opens over the top bar while the first-launch language banner shows.

</details>

<details>
<summary>FU-189 · Drafts use the fastest installed still model — Done</summary>

**Status or trigger:** Done

**Notes:** S-K found Make drafts still used the project's own (slow) still model at half size, about 6 min per still with Qwen-Image. Drafts now use the fastest installed model when it beats the project's at that size (`_draft_variant` in `routes/storyboard.py`, `Batch.draft_image_model`); the estimate prices the same model; a redo of a draft take uses it too. Final pass unchanged (quality model). Open from S-K: the Overview next-step card still makes Make primary on a slow fresh film; the Length chips can lose a click right after typing the brief (the Plan suggestion link was fixed); suggestion reads length words in English only.

</details>

<details>
<summary>FU-188 · Eight closing improvements (D-84) — In progress</summary>

**Status or trigger:** In progress

**Notes:** Packages S-J (Chat plan card), S-K (Storyboard UX), S-L (clean-up panel), S-M1 (durable queue, song, cut plan, lyric captions), S-M2 (freestyle board), S-N (QA). Plan `docs/plans/v0.10.0-eight-improvements.md`; progress `docs/progress/v0100-{j,k,l,m1,m2,n}.md`.

</details>

<details>
<summary>FU-187 · Watch Qwen 4 — Open</summary>

**Status or trigger:** Open

**Notes:** Announced at Apsara (2026-09-22: Max, Flash, Plus, 27B), no weights or date as of 2026-11-16; Qwen3.8-Flash-Next (125B MoE, 6B active) previews the architecture. On release: catalogue rows, `DRAFT_MODEL_MAP`, vendored MLX architecture if mlx-lm lacks it, llama.cpp support check, matrix run.

</details>

<details>
<summary>FU-186 · Storyboard QA leftovers fixed for 0.10.0 — Done</summary>

**Status or trigger:** Done

**Notes:** Owner chose all four (2026-11-16). (1) A clip made after Make drafts completes the draft still-only take and keeps its pass (draft badge and Export warning stay; Redo clip works; the final pass picks the shot up). (2) MLX non-streaming generate waits 60 s plus 0.5 s per allowed token (cap 1800 s) instead of a fixed 120 s, and a worker that dies mid-request clears the loaded model so the workspace stops saying loaded. (3) Path import of a picture checks the PNG, JPEG or WebP header. (4) Speech and tokenizer repos (chatterbox, orpheus, VoxCPM, S3Tokenizer, any `-tts`) no longer appear in chat model lists. Tests: `tests/test_v0100_closing_fixes.py`, `tests/test_storyboard_passes.py`, `src/utils/__tests__/library.test.ts`. The llama.cpp path had the same fixed 120 s read limit on its non-streaming call (the Windows 4090 lane) and got the same rule. Not tested: a real long reply on a big model.

</details>

<details>
<summary>FU-185 · Matrix run on the 0.10.0 tree: legacy ids and the TensorFold check — Done</summary>

**Status or trigger:** Done

**Notes:** First full matrix on this Mac (2026-11-16, Qwen3-0.6B/4B, Qwen3.5-4B, Qwen3.8-27B): 11 of 11 runnable cells pass (native, TurboQuant, TriAttention, DFlash, DDTree, MTPLX, TensorFold, GGUF native and TurboQuant); skipped: four vLLM cells (no vLLM on a Mac) and GGUF MTP (29 GB model not downloaded). Found and fixed: (1) the dropped `chaosengine` and `rotorquant` ids ran as TurboQuant but reported the old id on the MLX engine, because only the GGUF engine coerced them; now coerced at the load route (`state/lifecycle.py`) and again at the controller, tests in `test_backend_service.py` and `test_inference.py`. (2) The runner expected engine `tensorfold`, but the native lane reports `tensorfold-native`; it accepts both.

</details>

<details>
<summary>FU-184 · Storyboard polish S8 to S13 and S15 — Done (gaps listed)</summary>

**Status or trigger:** Done (gaps listed)

**Notes:** Progress `docs/progress/v0100-{d,e,f}.md`. S8 Plan tab (brief, style pictures, proposed shots, Rework with the model, direction chat, all as ops with Undo); S15 `style` reference kind, local palette, brand kit (logo, caption look), style as pictures or words per shot; S9 rough-cut player (Play film on Storyboard, Mixer, Export; picture follows the audio clock, 22 to 28 ms median); S13 caption look burnt in, .srt beside the file; S10 take compare; S11 Redo clip only and Redo still only (clip knows which still it came from); S12 bundled sample film "The Lighthouse" (1.4 MB) with a Try card on the empty project list. Open: Export does not yet draw the brand logo or a title and end card; a style change does not mark shots stale; agent tools cannot set brief, style or brand; lyric captions not built; the Mixer's own transport still plays a rendered preview only; style as reference pictures unit-tested only (Iris takes none); rough-cut sync not measured in the desktop webview; sample film not opened from a packaged build; LTX 832x512 clips get narrow black bars in a 854x480 film. S14 (long music video on a freestyle board) is the 0.11.0 proposal in the plan.

</details>

<details>
<summary>FU-183 · Storyboard polish S1 to S7 — Done</summary>

**Status or trigger:** Done

**Notes:** Plan `docs/plans/v0.10.0-storyboard-polish.md`; progress `docs/progress/v0100-{a,b,c}.md`. S1: the Chat gate runs the tool's own checks before showing a card, so a call that cannot run gets no card; repeats return the running job (closes FU-179's double approval; proven by tests, not seen on screen). S2: Automatic still model = fastest installed that fits. S3: stale running work reset on every project load and at start-up. S4: Server tab, Dashboard and sidebar footer name a busy engine. S5: first-clip estimate seen in Chat. S6: draft and final passes (Keep for final, Make drafts, Final pass, draft warning on Export). S7: time per shot and total on the plan card; `storyboard_generate` accepts `what: final` and `draft`. Open: Iris draft estimate is low at small sizes (engine raises Iris to 768 px); the film render itself does not warn about draft takes; the Dashboard Current runtime panel still says no model during a studio run. FU-179 stays open only for the slower-default-model part, now covered by S2.

</details>

<details>
<summary>FU-182 · Open button could not open a picture or clip — Done</summary>

**Status or trigger:** Done

**Notes:** Found by the owner on the Mac (2026-11-15): Open on an Image Studio output showed "Could not open file". The backend route behind Open only served HTML Challenge files, the Tauri file URL is outside the opener scope, and `open_path` is not allowed by `opener:default`. New `POST /api/files/open` opens pictures, clips, sound and subtitles from the app's own folders only (data dir, image and video output dirs), never scripts or apps; the Open handler tries it first. Also: with Preview (draft) on, the size block now says the run renders at 512 px and the chosen size is kept (Preview never changed the size setting, but the screen still showed the old size without saying so). Tests: `test_open_file_endpoint_opens_only_media_inside_the_apps_folders`. Not checked in the Tauri app; the owner should click Open on the rebuilt app.

</details>

<details>
<summary>FU-181 · Two bugs found by the owner on the Mac (2026-11-15) — Done</summary>

**Status or trigger:** Done

**Notes:** (1) Iris-3B and FLUX 3 Klein (mflux) failed with `NameError: name 'size_notes' is not defined` whenever the step count was more than twice the model's tuned steps (the High Quality preset): the steps note in `_generate_image_artifacts` used a name from another function; now joined onto `size_note`, test in `test_size_limits.py`. (2) Cancelling a re-render marked the shot Failed with a raw `{'message': 'cancelled', ...}` even though its earlier take was fine, because the studio reports a stopped run as an HTTP error; the batch now treats any error after Cancel as cancelled (shot keeps its last take) and reads the `message` of a dict detail. Test in `test_storyboard_store.py`. Tea Shop Ad 2 shot 4 still shows Failed from before the fix; Retry or New take clears it.

</details>

<details>
<summary>FU-180 · A job started elsewhere shows no progress in the studios — Done</summary>

**Status or trigger:** Done

**Notes:** Found on the Mac (2026-11-15): a Storyboard still ran for minutes while Image Studio and the Server tab looked idle, and the batch bar had no step inside the still. The batch progress now carries the denoise step (`innerStep` / `innerTotal`, from the shared progress tracker) and the bar shows "step 27 of 40"; Image Studio and Video Studio show a "Generating now" card (model, phase, step, elapsed) for a run they did not start, polled every 2 s and hidden during their own run. Checked on screen against a live Qwen-Image 2.1 still. The Server tab still says "No model loaded" for an sd.cpp subprocess; not changed.

</details>

<details>
<summary>FU-179 · Chat storyboard run: two render approvals, slow default still model — Open</summary>

**Status or trigger:** Open

**Notes:** Chat run on the Mac with Qwen3.8-27B MLX 4-bit, Studio tools on (2026-11-15): the model made the project and 4 shots with 6 tool calls, but asked for `storyboard_generate` (stills) twice, so the user had to approve two plan cards; the first approval rendered nothing, the second did. Also the default still model was Qwen-Image 2.1 (sd.cpp Q4_K, 1280x720) at 435 to 637 s per still, so 4 stills took about 30 min; a smaller default still model for storyboards, or a time shown on the plan card, would help. The model also dropped the `<tool_call>` opening tag once; fixed in `agent.py` (commit 136b3097). Not run: the clips, the 4th still (power loss).

</details>

<details>
<summary>FU-178 · llama.cpp 1.0 features not wired — In progress</summary>

**Status or trigger:** In progress

**Notes:** **2026-11-16 (v0100/fu178, merged): done on the Mac:** MTP acceptance shown on the GGUF lane (Qwen3.8-27B MTP 2.2x plain, 2.76 accepted per round), DFlash 2 drafter now outranks MTP and `lmstudio-community/Qwen3.8-27B-GGUF` pairs with the z-lab Q8_0 drafter (3.8x plain), native q8_0/q4_0 KV option with memory estimate and UI slider, FP8 refused off CUDA, MoE design note. **Still open (CUDA, rows 38 to 43 of the 4090 plan):** FP8 KV, MoE expert cache Fit rule, DFlash vs MTP order on CUDA. Earlier notes: b12387 (llama.cpp 1.0) adds: **FP8 KV** (`--cache-type-k/v f8_e4m3`): aborts llama-server on Metal at load (`SET_ROWS` unsupported on MTL0, with or without flash attention); needs a CUDA test, then a native-strategy bit option gated to CUDA and to the help list, plus a Metal refusal, because pre-validation would pass it. Quantized KV in general: the native strategy emits only f16 for llama.cpp; q8_0 / q4_0 run fine on Metal (8B q8_0 about 35 tok/s vs 37-40 f16 with flash attention) and halve/quarter KV memory, so a `cacheBits` 8/4 option on native would be cheap. **MTP speculative decoding** (`--spec-type draft-mtp`): already wired via `_mtp.py` tensor probe and repo aliases, never run on this binary (no MTP GGUF on disk; Qwen3.8-27B is MLX only here); needs a Qwen3.8-27B MTP GGUF run (load time, tok/s, acceptance) and the app to surface acceptance for llama.cpp (`dflashAcceptanceRate` is null on the GGUF lane). **DFlash 2**: llama.cpp now has a separate `draft-dflash2` type, while the app maps every `dflash-` drafter to `draft-dflash` on the assumption that v2 is detected from metadata; check with a DFlash 2 GGUF (z-lab Qwen3.8-27B-DFlash2 is MLX only here). **MoE expert cache** (`--cpu-moe`, `--n-cpu-moe N`, `--moe-cache-mib N`, `--moe-cache-cpu-share F`): lets a large MoE run with experts in host RAM and a GPU-side cache; the app needs a Fit/memory rule (when the model does not fit in VRAM on CUDA), a setting, and an estimate in the memory planner; on Apple unified memory it brings little. Also: `--load-mode` (now used for shared-memory NVIDIA), `--reasoning-effort/-budget` (could replace template switches), and the server's no-API-key warning. Measured on the Mac: b12387 decodes about 7-12% slower than Homebrew b12110 on Llama-3.1-8B Q4_K_M. **Also found 2026-11-14:** b12387 decodes about 7 to 12 % slower than the May build and Homebrew b12110 on Llama-3.1-8B (indicative, another job ran at the same time): re-measured alone 2026-11-14 (Llama-3.1-8B Q4_K_M, 300 tokens, 2 rounds x 3 runs, llama-server direct): the May build 76.9 tok/s, Homebrew b12110 74.4, b12317 69.8, b12387 69.8 (flash attention: 67.7, 66.6, 56.3, 57.8), so on Qwen3.8-27B UD-Q4_K_M (16.6 GB) there is no slowdown (b12387 17.4 tok/s, May build 16.75, b12317 16.05) and **MTP speculative decoding works on Metal: 16.8 tok/s plain, about 28 tok/s with `--spec-type draft-mtp --spec-draft-n-max 3` (1.65x, 63 % of drafts accepted, run through llama-server direct, output text not diffed)**, so wiring MTP for the GGUF lane is worth doing; b12387 is as fast as the previous pin b12317 and the slowdown (about 9 to 12 %) came in upstream before b12317; the bump costs nothing, bisect between b12110 and b12317 only if it matters (no open Metal decode report found; only CUDA ones, issues 32229 and 32245); in a source run the app finds Homebrew's llama-server before `~/.chaosengine/bin` (PATH first). **2026-11-16 TensorFold native 1.2.9 on this Mac (owner asked whether a third-party 65.5 tok/s Qwen3.8-27B code claim holds):** through the app's `TensorFoldNativeEngine`, `Vontra/Qwen3.8-27B-MLX-4bit` + `z-lab/Qwen3.8-27B-DFlash2`, 320 tokens, temperature 0, thinking off, 3 reps; **plain 14.3 to 25.5 tok/s** (noisy: a browser GPU helper ran at about 1100 % CPU, so read ratios), **drafted code 50.9 to 52.6 (7.05 to 7.46 accepted per round), drafted prose 23.9 to 25.0 (1.6 accepted per round)**; load 8 s plain, 54 s drafted. The claim's engine (oMLX 4-bit, MTP k=3) reads 22 to 26 plain and 65.5 / 48 code / prose; our llama.cpp lane has plain about 15 and MTP 33 on the same chip, so the gap is MLX against GGUF base speed (about 1.5x), not the speculative method. Not reproduced: his 114 tok/s code on TensorFold. Re-measure on a quiet machine before drawing a line. **Also fixed:** the MTPLX launch box could not be ticked for an MLX Qwen3.8-27B (a DFlash-availability effect and the load-time sanitiser cleared the flag at once; both now keep it for a model MTPLX serves, test in `runtimeSupport.test.ts`). **2026-11-16 quiet-machine rerun, M4 Max 64 GB, Qwen3.8-27B 4-bit class, 320 tokens, greedy, thinking off, 3 reps (llama.cpp: 2), load average 3 to 6 (browser helper gone):** tok/s code / prose. **TensorFold native 1.2.9** (Vontra MLX 4-bit): plain 25 / 24; DFlash 2 drafts 91-97 / 37 (6.9 and 1.4 accepted per round), so 3.7x / 1.5x; this agrees with the third-party 114 code figure on the same chip. **MTPLX 2.12.3** (`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, native MTP head, depth 3, run through `MtplxEngine` from a scratch venv): 59-63 / 40.5 (acceptance not reported by the engine). **llama.cpp b12441** (`unsloth` UD-Q4_K_M): plain 20-22 / 17.5-18; MTP N=3 40-43 / 29-30 (2.75 and 1.84 accepted per round); DFlash 2 Q8_0 drafter 63-79 / 30-33 (5.4 and 2.1). **Defects found:** (1) the MTPLX box is offered for any Qwen3.8-27B (the support list holds the canonical `Qwen/Qwen3.8-27B`), but `mlx-community/Qwen3.8-27B-4bit` and `Vontra/Qwen3.8-27B-MLX-4bit` carry no MTP tensors (0 of 2180), so MTPLX refuses them (`missing-mtp-weights`) and the load falls back to MLX; the catalog row labelled `(MTP head)` is wrong for Vontra; (2) no catalog row for the checkpoints MTPLX runs (`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, 19 GB, Apache-2.0; the Qwen3.6 ones are in `_mtp.py` only); (3) this machine's MTPLX venv is 0.3.7 while the installer asks for `>=2.12.0`, and 0.3.7 cannot read the 2.x checkpoint metadata (`unknown MTPLX profile 'turbo'`); nothing in the app detects a stale venv. **Fixed 2026-11-16 (all three MTPLX defects):** the library scan now reads each MLX checkpoint's index for MTP tensors (`mtpHeads`), and the launch dialog hides MTPLX for a checkpoint that has none (stock `mlx-community` and Vontra 4-bit); `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` is a catalog row and in `MTP_MODEL_MAP`; the Vontra row is renamed (TensorFold) and no longer claims an MTP head; a venv older than MTPLX 2.12 reports `outdated` (not available), the box shows "Installed version is too old..." with the Install button, which rebuilds it. Seen live in the dev UI against this machine's real 0.3.5 venv and library. `tests/test_mtplx_detect.py`. **MTPLX plain on the same checkpoint (2026-11-16, `quickstart --no-mtp`, 3 reps):** 22.0 code / 21.7 prose tok/s, so MTPLX MTP is 2.7x on code and 1.9x on prose; the engine's runtime note still says MTP is active in that run because it was forced from the harness, not a bug in normal use. **Provenance of the MTPLX checkpoint:** `Youssofal` is an individual account (not an organisation) that also authors the MTPLX engine (`youssofal/MTPLX`, Apache-2.0, 2.7k stars, created 2026-05); the checkpoint is Apache-2.0, safetensors only (no pickle or code files), 21k downloads, 171 likes; Hugging Face reports no security scan result for it. Listed as a plain catalog row with a neutral note, not promoted.

</details>

<details>
<summary>FU-177 · Show the TensorFold family report in Setup and Models — Open (2026-11-14): backend done (FU-168), no UI.</summary>

**Status or trigger:** **Open (2026-11-14): backend done (FU-168), no UI.**

**Notes:** `GET /api/setup/tensorfold-native-status` and `tensorfold-status` return `families` (`modelType`, `servedHere`, `curated`, `offered`, `repos`, `legacy`, `note`), but the Setup and Models pages do not show "served by the installed TensorFold" or why a family is listed and not offered. **To do:** a short list or chip on the TensorFold card and the model rows, with English strings and `null` in the other locales, a vitest case and a Playwright check.

</details>

<details>
<summary>FU-176 · Wan 2.1 via stable-diffusion.cpp: the video path is not wired — Open (2026-11-14): found by v0.10.0 task B, not changed.</summary>

**Status or trigger:** **Open (2026-11-14): found by v0.10.0 task B, not changed.**

**Notes:** `VideoRuntimeManager.generate` never routed anything to `SdCppVideoEngine` before MiniMax H3 (FU-166); the Wan GGUF rows run through diffusers, so "one Wan 2.1 1.3B sd.cpp clip" has no app path to try today (only the CLI), and the Wan branch of `_build_cli_args` in [sdcpp_video_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/sdcpp_video_runtime.py) passes no `-M vid_gen`, VAE or text encoder. **To do:** decide whether Wan on sd.cpp is wanted; if so wire the route and the flags from the pinned sd.cpp docs, then run a real clip. This also blocks the clip half of the sd.cpp pin-bump check in CLAUDE.md (FU-166).

</details>

<details>
<summary>FU-175 · FLUX.2 Dev: catalog size wrong (64 GB) and the download bar showed the whole repo (154.08) — Open (2026-11-13): size fixed in code (3f534add)</summary>

**Status or trigger:** **Open (2026-11-13): size fixed in code (`3f534add`); FLUX.2 Dev NF4 untested on 24 GB.**

**Notes:** Seen on the RTX 4090 through the installed app: a download of `black-forest-labs/FLUX.2-dev` without a token failed at the preflight with the right message (add a read-enabled HF_TOKEN in Settings), but the job reported `totalGb` 154.08 while the row said 64.0 GB. The Hub file list (public, no token) shows a 177.6 GB repo: the diffusers folders the app fetches (transformer 64.45 + text encoder 48.02 + VAE 0.34 GB = **112.8 GB**, 105.1 GiB) plus the single-file `flux2-dev.safetensors` (64.45 GB), which `_IMAGE_DIFFUSERS_ALLOW_PATTERNS` already skips. So the download was already filtered; the size shown was wrong, for two reasons: the row's `sizeGb`, and `_known_repo_size_gb`, which returns the cached Hub total of the whole repo before the catalog size, so the "pinned rows use the catalog size" branch in `state/downloads.py` never applied once the preflight had filled the cache (same for Klein 4B on its restore: bar sized 20.59 GiB, the download was 14.88 GiB, all 20 pinned files present and size-checked). Fixed: `sizeGb` 112.8 on the curated and the tracked FLUX.2 Dev rows; `_known_repo_size_gb(repo, catalog_only=True)` for pinned image and video rows, used on both the normal and the failed-preflight path (bar: 105.05 GiB for FLUX.2 Dev). Tests `tests/test_image_download_size.py`. Shipped in the installer built at `3f534add`; the bar in the installed app not re-checked with a new download. Still open: FLUX.2 Dev NF4 on 24 GB not run (needs the licence accepted on the model page, a token, and 112.8 GB of disk; C: has 73.6 GB free).

</details>

<details>
<summary>FU-174 · Idle GPU speech workers slow image and video jobs; the idle-unload setting missed the Qwen3-TTS worker — Fixed and verified through the installed app (2026-11-13, installer built at 3f534add).</summary>

**Status or trigger:** **Fixed and verified through the installed app (2026-11-13, installer built at `3f534add`).**

**Notes:** `docs/progress/rtx4090-progress.md`, "FU-174: voice workers after the E2E" and "Installer rebuilt at `3f534add`". Installed app: with the Qwen3-TTS worker idle-warm (4.67 GB on the card) an LTX-2.5 768x512x97 clip stopped it 2.9 s into the request, backend shared GPU memory stayed at the idle 0.09 GB, denoise 62.8 s (no 600 s step); the next TTS request reloaded in 33.3 s, then 1.63 s per audio second; `idleUnloadMinutes` 1 stopped the Qwen worker 84 s after its last request; smoke E2E 7 of 7 and full E2E 18 of 18 phases. Measured on the RTX 4090 through the installed app: E2E phase 8 leaves three workers (Qwen3-TTS 1.7B 8.1 GB private / 4.5 GB GPU, Whisper + Kokoro 2.8 / 0.6 GB, turn detector 0.8 / 0 GB), about 5 to 7 GB on the card; by design they stay warm and exit after 10 min idle (turn detector 2 min), on `/api/voice/worker/stop` and on app close (all measured). No leak: 3 Qwen and 3 phase-8 load/stop cycles return to the same GPU (783 MiB) and backend private (3.40 GB). Problem: an LTX-2.5 768x512x97 clip started next to the idle Qwen worker spilled into shared memory (17.9 GB dedicated + 1.5 GB shared, 90 W) and spent about 600 s on step 1; the moment the worker idled out, steps 2 to 8 took about 40 s at 400 W. Fix: `voice_client.release_gpu_for_media()` stops idle speech workers holding a GPU model (not Kokoro, not a busy one, NVIDIA only) before the image and video managers load or generate; the `idleUnloadMinutes` setting now reaches the Qwen3-TTS worker and is read at startup (it was applied only to the main worker and only when saved). Tests `tests/test_voice_gpu_release.py`. The 2.9 s per audio second TTS run was not reproduced: warm runs are 1.61 to 1.66 in every case tried (fresh worker, after a video run, next to the phase-8 workers, started with 19.5 or 21.5 GB of the card taken); a request that loads the model measures 2.7 to 3.5, so the slow figure most likely included a model load.

</details>

<details>
<summary>FU-173 · Installed app runs LTX-2.5 4-bit and FLUX.2 Klein slower than the dev venv on the same card — Fixed and verified through the installed app (2026-11-13, installer built at a8b0051e).</summary>

**Status or trigger:** **Fixed and verified through the installed app (2026-11-13, installer built at `a8b0051e`).**

**Notes:** `docs/progress/rtx4090-progress.md`, "FU-173: installed app slower than dev" and "Installer rebuilt at `a8b0051e`". Installed app, UI open: Klein warm denoise 5.07 / 4.83 s, LTX-2.5 20.90 / 20.74 s, `/api/workspace` 0.10 s median, smoke E2E 7 of 7. Not torch: the installed runtime (torch 2.14.1+cu130) run standalone matches the dev venv (Klein warm denoise 4.72 s, LTX-2.5 512x320x25 20.2 to 20.9 s). Cause: the UI polls `/api/workspace` every few seconds, and its LLM process scan asked psutil for `memory_info` and `cpu_percent` of every process (0.44 s for 290 processes on Windows, GIL held), stalling the Python loop that drives the GPU. With UI-like polling Klein went 4.72 to 5.45 s (+16 %); through the app LTX-2.5 was 22.1 to 23.0 s. The scan now asks only for pid, name and command line (0.01 s; `/api/workspace` 0.56 to 0.15 s); with the same polling Klein is 4.83 s and LTX-2.5 20.6 s. The 33 to 35 s seen earlier did not come back in this session (best guess: more processes or more UI polling then).

</details>

<details>
<summary>FU-172 · Qwen3.8-27B + DFlash2 on the vLLM lane does not fit a 24 GB card; vLLM lane ignores the thinking switch — Open (2026-11-13): measured on an RTX 4090 (WSL2, vLLM 0.33).</summary>

**Status or trigger:** **Open (2026-11-13): measured on an RTX 4090 (WSL2, vLLM 0.33).**

**Notes:** `docs/progress/rtx4090-progress.md`, "Qwen3.8-27B + DFlash2 on the vLLM lane". Plain W4A16 runs through the app (about 53 tok/s, 16K and 32K context fit). With the drafter, weights alone are 20.2 GiB (18.5 with an fp8 drafter) and the recurrent-state blocks need 1.6 to 1.8 GiB even at 1K context, so every load stops at "no memory for the cache blocks". Fixed on the way: the drafter now pairs with quantised packs, `num_speculative_tokens` comes from the drafter's block size, `max_num_seqs` 8, the GPU share follows free memory. **To do:** (1) run the lane on a 32 GB or larger card and record speed-up and acceptance (`LLM` runs with stats off, so acceptance is not shown); (2) `VLLMEngine` renders the chat template without `enable_thinking`, so "thinking off" still thinks; (3) a 24 GB vLLM catalog row needs a 4-bit pack whose embedding loads on stock vLLM (the AutoRound pack with an int8 embedding does not; RedHatAI's INT4 has bf16 embeddings and should). **Update (2026-11-13):** the thinking switch now reaches the vLLM chat template (`VLLMEngine.stream_generate` passes `enable_thinking`; unit tests only, not run live because the Linux test environment was removed). Still needs native Linux to confirm live; the RedHat INT4 load and the DFlash speed-up need a 32 GB card.

</details>

<details>
<summary>FU-171 · TriAttention on the vLLM lane: works only on vLLM 0.18 or older; the app now says so and wires the budget, stats file, eager mode and prefix-cache switch — Open: root cause found, compaction working on vLLM 0.18.1, app honest on newer vLLM (2026-11-13).</summary>

**Status or trigger:** **Open: root cause found, compaction working on vLLM 0.18.1, app honest on newer vLLM (2026-11-13).**

**Notes:** Found 2026-11-12, root cause 2026-11-13 (WSL2, RTX 4090). The plugin at commit `a4bc3c8` copies vLLM <=0.18 model-runner internals. vLLM 0.19 binds each layer's `kv_cache` as a bare tensor (0.18: `[tensor]`), and the plugin's KV group resolver accepts only a list, so it found no KV tensors: every compaction ended `no_compactable_groups` (the 1885 of 1885 seen on 0.19.1). Shimming that made 0.19.1 compact once, then crash in the plugin's copied `_prepare_inputs` (`self.positions.np`; 0.19 made `positions` a GPU-only tensor). 0.33.0 dies on the first request (`log_iteration_details`). On vLLM 0.18.1 (hardlinked scratch venv, +3 GB) it compacts, after one fix in our adapter: the plugin stamped skipped compressions as the last compression step, so after one compaction its one-step dedup guard skipped every later signal (1 applied, 2304 skips); `_install_skip_stamp_fix` keeps the last applied step (20 and 39 compactions after). The adapter now refuses vLLM >=0.19 with the reason (Settings badge Unavailable), needs a stats file `<repo name>.pt` in `~/.chaosengine/triattention/stats` (or `CHAOSENGINE_TRIATTENTION_STATS_DIR`; the pip install ships none), and the engine passes `TRIATTN_RUNTIME_KV_BUDGET` (`CHAOSENGINE_TRIATTENTION_KV_BUDGET`, default 2048), the stats path, protect-prefill, `enforce_eager`, no prefix caching, 2048-token prefill chunks and an in-process engine core. The `[triattention]` extra now pins `vllm>=0.15,<0.19`. Still open: the app installs vLLM >=0.28, so TriAttention on vLLM is unavailable in a stock install until upstream ports to vLLM 0.19+ (or the app adds a separate vLLM 0.18 lane); stats files are per model and made by hand; eager mode costs about 4x decode speed on Qwen3-0.6B (80 vs 340 tok/s). See `docs/progress/rtx4090-progress.md`, section "TriAttention on vLLM: root cause and a working combination".

</details>

<details>
<summary>FU-170 · cargo test port race in backend::tests::runtime_info_keeps_a_soft_note_while_the_port_is_silent — Open; low priority.</summary>

**Status or trigger:** **Open; low priority.**

**Notes:** The test binds port 0, drops the listener, then expects the port to stay silent. Failed once while pip, brew and nvm were opening sockets; passes 3 of 3 when the machine is quiet. Fix: pick a port and hold a non-accepting socket, or retry the probe.

</details>

<details>
<summary>FU-169 · scripts/check-upstreams.py: report every pinned upstream against its newest release (D-78) — Shipped 2026-11-11.</summary>

**Status or trigger:** **Shipped 2026-11-11.**

**Notes:** Reads each pin (llama.cpp, TensorFold, ltx-2-mlx, mflux, dflash-mlx, ACE-Step, TriAttention, Chatterbox, Playwright), asks the upstream for the newest release or commit, prints current, behind or unknown; for ltx-2-mlx it also says whether the commits touch more than CI. Also lists TensorFold model families the installed binary serves that no table row lists. Report only, exit 0; run by `pre-build-check` and the release workflow; `tests/test_check_upstreams.py`. Open: dflash-mlx pinned at v0.1.5.1, newest v0.1.10 (FU-057); llama-server-turbo and DDTree have no pin to compare.

</details>

<details>
<summary>FU-168 · TensorFold 1.x: build the family table from capabilities --json, add newer families — Parts 1 and 2 built 2026-11-14 (v0.10.0 R9)</summary>

**Status or trigger:** **Parts 1 and 2 built 2026-11-14 (v0.10.0 R9); open: UI for the family report (FU-177), offering Flint (1.2.8), Gemma 4 31B on 1.x, D-77 fallback decision, live runs of Spark, Nanbeige, MiniMax and the four CUDA families.**

**Notes:** **Done 2026-11-11 (worktree branch, not merged).** Every model_type the macOS 1.2.7 binary's `capabilities --json` lists now has a row, so `check-upstreams.py` reports no unlisted family. Facts come from the binary (`capabilities --json`, `models --json`, `info`) and its v1.2.7 recipes; every repo was checked on the Hub. *Offered* (rows in `TENSORFOLD_FAMILIES` + catalog): `lfm3` (LFM3-1.7B MLX 4/6/8-bit; tested tier, the bundled port still runs it without the toggle), `lfm2_moe` (LFM2.5-24B-A2B MLX 4/6/8-bit; also a CUDA `NATIVE_MODELS` row for the 4-bit, 15.7 GB = the binary's own floor, `measured=False`), `spark2_5` (XHToken/Spark-X2.6-4B bf16), `llama` (Nanbeige4.3-33B bf16 tested, INT4 exclusive; Nanbeige4.1-3B drafter; TensorFold refuses M1-M4, so the E2E check skips there), `minimax_m3_vl` (BobClawblaw MiniMax-M3.1-Flash mix256k / mix34, vision, 256 GB floor via `minMemoryGb` and `estimatedMemoryGb` 226). *Listed, not offered* (`tested_repos=()` with a `note`): `lfm3_vl` (1.2.7 admits LFM3-VL-2B but refuses to serve it: "LFM3-VL inference requires the stacked vision contribution", checked here), `gemma4_unified` (needs TensorFold's own conversion of google/gemma-4-12B-it), `step5_vl` (FP8 650 GB, needs a local 2-bit conversion; exclusive so a converted dir routes to TensorFold; 256 GB), `glm_moe_dsa` (GLM-5.3 EXLR8 on two 256 GB Macs), `kimi_k3` / `kimi_k31` (groundwork only, never run on real weights; ~1.56 TB), `qwen4` / `qwen4_text` (no released checkpoint), `fish_omni` (Drama 3 speech, not a chat model). The Mac native engine now takes drafter and vision from the family table when no CUDA row matches. Live on the M4 Max through a worktree backend: LFM3-1.7B 4-bit and LFM2.5-24B-A2B 4-bit loaded with `--spec` on `tensorfold-native` (243 and 180 tok/s, short replies); the new E2E check `TensorFold 1.2.7 families (FU-168)` in phase 1 passed. Not run: Spark (8.2 GB), Nanbeige (23 / 66 GB, and M5 only), MiniMax (203+ GB). Still open: build the table from `capabilities --json` at runtime instead of copying docs; 1.x serves Gemma 4 31B (`mlx-community/gemma-4-31B-it-4bit` + z-lab DFlash, M5 for the drafter), not added; decide whether to keep the 0.6.6 fallback for the three `native=False` families (D-77). 2026-11-12: the Windows 1.2.6 source build reports 4 more families no row lists (dory, kolibri1, lfm2_vl, nemotron_h_omni); the Mac 1.2.7 pass covered Metal families only, so run check-upstreams.py on each host and add rows for CUDA-only ones. 2026-11-12 (RTX 4090 pass): the CUDA `NATIVE_MODELS` registry gained Qwen3.6-35B-A3B + MTP (20.9 GB, `qwen3_5_moe`) and Gemma 4 31B (18.44 GB, `gemma4`, vision), both `measured=False` from upstream GB10 receipts and tight on 24 GB; LFM2.5 already had its row. The 26B-A4B stays Python-only. **2026-11-12 (RTX 4090):** `tfgrammar.dll` builds on Windows with `zig build grammar -Dxgrammar=DIR` (xgrammar v0.2.8, 32 s), adopt copies it, structured output verified through the app; Gemma 4 31B (42 tok/s, 21.8 GB, window max 12K on 24 GB) and Qwen3.6-35B-A3B + MTP (186 tok/s, 420 on code, 20.1 GB, 32K fits) measured, rows set `measured`. **2026-11-14 (R9):** (1) four CUDA-only families are rows with `tested_repos=()` and a `note`, listed and not offered: `dory`, `kolibri1_fp8`, `lfm2_vl_d1` (a decision model; TensorFold refuses chat to it), `nemotron_super_vl`; repos checked on the Hub by metadata only. A local Kolibri 1 MLX build still goes to the bundled MLX port. (2) [_tensorfold_native.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_tensorfold_native.py) `cached_capabilities`, `serves_family`, `prefers_python` and `family_report` build the table from the binary's `capabilities --json`; a type no row names is `curated: false` and never offered. New `native_refuses` field keeps Gemma 4 26B-A4B on the Python install although the Metal binary lists `gemma4` (D-77 routing: `native=False` rows go to 0.6.6 while the binary does not list them). `GET /api/setup/tensorfold-native-status` and `tensorfold-status` return `families`; `check-upstreams.py` uses the same report. Mac checks: 20 families, 10 offered, none uncurated; LFM3-1.7B and LFM2.5-24B loaded with `--spec` on the native engine. **Not tested:** the CUDA fixture `tests/fixtures/tensorfold_caps_cuda_1_2_7.json` is assembled from v1.2.7 source and GB10 receipts, not a real dump (replace it with `capabilities --json` from the 4090); no real load of a `native=False` model; no run of the four CUDA families, Spark, Nanbeige (M5 only) or MiniMax. TensorFold 1.2.8 exists; `NATIVE_VERSION` is not bumped (needs the Mac archive sha256 and a re-check of the family table and of `native_refuses`). **2026-11-14 (bump to 1.2.8, branch `v0100/bump-tf`):** `NATIVE_VERSION` 1.2.8 (archive names unchanged; digests are read from the release's `.sha256` files, none pinned). The macOS 1.2.8 `capabilities --json` (verbatim in `tests/fixtures/tensorfold_caps_macos_1_2_8.json`) adds one family, `flint` (Recursive Flint 40B-A2.5B, `RecursiveSI/flint-int4`, M5-checked upstream), and three serve flags (`--max-connections`, `--max-connections-per-address`, `--trusted-proxy`); none of the flags the engine passes went away. Flint is a row listed and not offered: it always reasons and answers 400 to `enable_thinking: false`, which the engine sends whenever thinking is off (every non-streaming turn). To offer it: leave that flag out for always-reasoning families, then add `tested_repos`, `exclusive=True`, a catalog row and the E2E family entry, and run a load. D-77 unchanged: Gemma 4 26B-A4B is still refused (`GemmaMoeNotServed`), the binary still lists neither `deepseek_v4` nor `prism_hadamard_qwen35` (1.2.8's experimental Metal DeepSeek is `deepseek_v41`, FP8 only, not listed), so all three stay on 0.6.6. The CUDA build gains `qwen3` (embedding and reranker) and `dots_ocr` (OCR, text only): both rows listed, not offered. The CUDA fixture is still assembled (now `tensorfold_caps_cuda_1_2_8.json`). LFM2.5's `info` floor is unchanged (14.65 GiB).

</details>

<details>
<summary>FU-167 · Image, video and audio speed-ups beyond the diffusion caches — Researched 2026-11-11</summary>

**Status or trigger:** **Researched 2026-11-11; nothing built.**

**Notes:** See [docs/research/media-speedups-2026-11-11.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/research/media-speedups-2026-11-11.md). Biggest expected win is step-distilled LoRAs / checkpoints (Hyper-SD, Lightning, Wan self-forcing, distilled LTX), then sparse attention for video and SageAttention on CUDA. Notes are from model knowledge, not re-verified online; needs a measured pass on the 4090 and the M4 Max before anything ships. The repo already has fbcache, teacache, taylorseer, magcache, pab and fastercache.

</details>

<details>
<summary>FU-166 · MiniMax H3 (video and image) and newer image/video models: scan, supersedes the FU-082 / FU-125 H3 notes — Built 2026-11-14 (v0.10.0 task B) behind licence_gate.H3_RELEASED = False</summary>

**Status or trigger:** **Built 2026-11-14 (v0.10.0 task B) behind `licence_gate.H3_RELEASED = False`; sd.cpp pinned at `532d2c1` (was `5dcfa79` when H3 was built) (bumped 2026-11-14); H3 video only, no H3-Image row; release bundles carry no `sd`. Owner legal read pending.**

**Notes:** Size is no longer the blocker: pruned Q4_K_M DiT 11.4 GB, stable-diffusion.cpp has `docs/minimax_h3.md` and `docs/minimax_h3_image.md`, community GGUF / nvfp4 / MLX / few-step builds exist, and the official `MiniMaxAI/MiniMax-H3-Image` (2026-11-03, 6-step, about 16 MP) is out. **Licence is the blocker:** rights are granted only outside the EU, UK, Korea and US (same for H3-Image), so it cannot be shipped by default or tested by an owner in the UK. Vimoo-1.0-Lite and Cosmos3-Edge were looked at and **dropped by the owner 2026-11-11**. Still unrun: Motif-Video-2B, LongCat-Video, plus image families stable-diffusion.cpp already documents (Krea 2, Ideogram 4, Qwen-Image 2.1, HiDream-O1). **Open:** check the pinned sd.cpp for H3 code (only if H3 is revived). Write-up and fit table (4090, M4 Max, 512 GB Mac Studio, DGX Spark): [docs/research/minimax-h3-and-new-video-image-2026-11-10.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/research/minimax-h3-and-new-video-image-2026-11-10.md). **2026-11-14 (B):** [licence_gate.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/licence_gate.py) and `/api/licence-gate` (confirm and withdraw accept only the UI), the confirmed sentence and licence URL stored with a record and an audit log; gated rows are hidden from every listing, Storyboard picker, fit best-pick and the agent bridge, and download, preload and generate answer 403 without a tick (`state.start_download` refuses too). `LicenceScreen` takes a `territory`, Video and Image Models show a collapsed "Models with territory limits" notice. Video row `video_minimax_h3.py` (`leejet/MiniMax-H3-GGUF` Q4_K_M DiT 11.4 GB, text encoder 18.2 GB, two VAEs, 35.5 GB pinned files, no time estimate) and an H3 command line in `sdcpp_video_runtime.py`, proven only against a fake `sd`. sd.cpp is pinned (`SDCPP_REF`, scripts, `check-upstreams.py`, CLAUDE.md row); the pin stays at the installed binary because the real FLUX.1 schnell and Wan 2.1 sd.cpp runs that a bump needs could not be done (models not on the Mac). H3-Image needs `32d516c` or later, so it is not built. Release CI never builds or fetches `sd` (`sdCpp` null in CI bundles). H3 was never downloaded or run. **Open (owner):** the legal read (licence II, III, V.2, V.4, V.5, AUP 1) before flipping `H3_RELEASED`; whether to bump sd.cpp for H3-Image (needs about 22 GB of FLUX.1 schnell and Wan 2.1 1.3B downloads first). Related: FU-176. **2026-11-14 (bump):** sd.cpp pin moved to `5dcfa79` (master-1104), which has `docs/models/minimax_h3_image.md`; the H3 video flags are unchanged, so no code change. Verified with the Qwen-Image 2.1 sd.cpp row's own arguments on both builds (same seed, 1024x1024, 40 steps): near-identical sane images, 650 s old vs 510 s new on M-series, peak 10.6 vs 11.1 GB. FLUX.1 schnell and Wan 2.1 runs dropped by the owner. H3-Image row still not added; what it needs is in [docs/progress/v0100-bump-sdcpp.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/progress/v0100-bump-sdcpp.md).

</details>

<details>
<summary>FU-165 · Studio time estimate does not scale with size, frames or steps — Built 2026-11-14 (v0.10.0 R8): measured rows, six comparison runs within 7 %</summary>

**Status or trigger:** **Built 2026-11-14 (v0.10.0 R8): measured rows, six comparison runs within 7 %; CUDA, LTX on CUDA and sd.cpp rows unmeasured.**

**Notes:** Catalog `estimatedGenerationSeconds` is shown flat (Mac and PC). Wan 2.2 TI2V reads 275 s for every setting; 832x480x49 took 34.5 s (8x), Klein 2048² 11 to 13 s against 5.5. Fit from the 4090 runs: `seconds = catalog x (0.04 + 0.96 x steps/ref_steps x w(T)/w(T_ref))`, `w(T) = T + T²/53,300`, `T = W/16 x H/16 x F/4`, within 5 % on four Wan TI2V runs, 15 % low on LTX at 49 frames. **To do:** a reference size, frame count and step count per row (rows lack `defaultSteps`; TI2V's 275 s is at 20 steps), apply in Image and Video Studio and in `storyboard/estimate.py`, check on the Mac (mflux has no offload swap, so its fixed share differs), add tests and an E2E check. **2026-11-14 (R8):** optional `estimateRef` (width, height, frames, steps, `fixed`, `quad`, `ignoresSteps`) on 28 image and 27 video rows; [time_estimate.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/time_estimate.py) and [timeEstimate.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/utils/timeEstimate.ts) share `tests/fixtures/time_estimate_cases.json` (18 cases). Used by Image and Video Studio, `storyboard/estimate.py` and the agent job ETA. Measured on the M4 Max and stored: Iris-3B (catalogue 18 to 10.5 s, ignores steps), FLUX 3 Klein mflux (40 to 34.5 s), LTX-2.5 q4 ltx-2-mlx (128 to 136 s, ignores steps); LTX-2.3 q4 takes the same constants unfitted. Six runs outside the fit grid, all within 7 % of the shown figure (target 20 %). Rows without a trustworthy steps reference stay flat. Detail: `docs/research/time-estimate-fit-2026-11.md`. **Not measured:** every CUDA row with the new constants, LTX on CUDA, the sd.cpp image rows (none installed), other Macs and cards (FU-164). The Klein recommendation text in `recommended_settings.py` still says 40 s. | **CUDA fit done for two rows (2026-11-19, `test/rtx4090`):** FLUX.2 Klein 4B (catalogue 6.9 s, `fixed 0.68`, `quad 1000`) and LTX-2.5 4-bit (62.9 s, `fixed 0.47`, `quad 3162`), three warm runs per case, every median within 9 %; 11 cases added to the shared fixture; tables in `docs/research/time-estimate-fit-2026-11.md`. Open: LTX-2.5 1280x704 swings 145 to 288 s between runs (24 GB card offloading); FLUX.1 schnell did not run the real model; FLUX.2 dev, the bf16 LTX-2.5 row and the other CUDA rows keep their old figures.

</details>

<details>
<summary>FU-164 · Estimates on NVIDIA cards other than a 24 GB RTX 4090, DGX Spark and Linux — Open (2026-11-11): fixes measured and shipped on test/rtx4090 (925f6f2f)</summary>

**Status or trigger:** **Open (2026-11-11): fixes measured and shipped on `test/rtx4090` (`925f6f2f`); other hardware never run.**

**Notes:** `docs/progress/rtx4090-progress.md`, "Estimate audit". Fixed on the 4090: chat VRAM from the GGUF tensor table, shared-KV layers, TurboQuant llama.cpp size and speed, CUDA decode overhead, memory check on the card, offload-aware image/video badges, linear CUDA Studio peaks, mimalloc purge after Unload. **To do (owner):** (1) run the chat and image badges on an 8, 12 or 16 GB card (Klein reads "won't fit" on 12 GB; measured 9.1 GB on 24); (2) DGX Spark: check the usable share (total minus max(8 GB, 10 %)) against what a model can really allocate; (3) Linux CUDA: backend RAM after Unload (glibc, not mimalloc); (4) RTX 5090 / 96 GB: the image runtime always offloads FLUX on CUDA (resident Klein 1.42 s against 6.6 s at 1024² on 24 GB, but spills at 2048²), so give FLUX the video runtime's free-memory placement rule and re-measure; (5) Qwen3.8-27B Q4_K_M + DFlash sidecar against the new chat estimate (20 GB download).

</details>

<details>
<summary>FU-163 · Qwen3.8-27B + DFlash2 on llama.cpp: low acceptance and no 24 GB catalog row — Open (2026-11-11): measured on an RTX 4090, lane works.</summary>

**Status or trigger:** **Open (2026-11-11): measured on an RTX 4090, lane works.**

**Notes:** `docs/progress/rtx4090-progress.md`, "Qwen3.8-27B + DFlash2 through llama.cpp CUDA". Q4_K_M + `dflash-` Q8_0 sidecar: 43 tok/s plain, 78 prose / 93 code / 112 JSON drafted, acceptance 31 to 50 %, card full (24.0 of 24.5 GB) at 4K context; TensorFold on the same card does 150 prose / 300 code. **To do:** (1) find why acceptance is low (try `--spec-draft-n-max` 4 to 15, the BF16/Q4_0 sidecars, a `--spec-draft-p-min`-style cutoff if b11981 has one, and compare with a newer tag); (2) a Q4_K_M catalog row that fits 24 GB (the repo pins one `ggufFile` per row, so it needs a second repo id or per-variant file selection); (3) a long-context pass (8K to 32K) to see where the card runs out; (4) run the new E2E check through the app on this PC.

</details>

<details>
<summary>FU-161 · DDTree on NVIDIA: a torch tree-verification lane (the algorithm is portable, the app's verifier is MLX-only) — Open (2026-11-11): read from the code, nothing built.</summary>

**Status or trigger:** **Open (2026-11-11): read from the code, nothing built.**

**Notes:** `ddtree.py` builds the tree in NumPy; `ddtree_attention.py` (`tree_forward`, `compact_kv_cache`, `prefill_with_capture`) imports `mlx` / `mlx_lm`. llama-server has linear drafters only (`draft-dflash`, `draft-mtp`, ngram), so a CUDA lane means a transformers target with a 4D tree attention mask and the same DFlash drafter, in a worker process. DFlash on the 4090 already gives 1.28 to 1.38x at 16K to 64K (`docs/progress/rtx4090-progress.md`, W4), which is the bar a tree lane has to beat.

</details>

<details>
<summary>FU-160 · Weight quantisation on NVIDIA for FLUX.2 and Wan: torchao int8 (best measured quality), NF4 beyond FLUX.1, Nunchaku on the cu130 runtime — Open (2026-11-11): measured on an RTX 4090, nothing wired.</summary>

**Status or trigger:** **Open (2026-11-11): measured on an RTX 4090, nothing wired.**

**Notes:** `docs/progress/rtx4090-progress.md`, W3. On FLUX.2 Klein 4B: torchao int8 weight-only 33.1 dB, 10.2 s vs 9.2 s, peak 17.6 to 13.8 GB; NF4 22.1 dB, 12.1 GB; FP8 layerwise 28.4 dB (already shipped). bitsandbytes int8 gives noise on FLUX.2, so do not offer it. On Wan 2.2 TI2V 5B NF4 breaks the picture and GGUF Q4 is 4.7x slower, so only FP8 layerwise passes there. To do: a `weightQuant` choice (int8 / NF4 / FP8) that reads the card, loads torchao int8 through the existing `TorchAoConfig` helper for CUDA and FLUX.2, with the torchao version pinned to the torch build by the Setup path. Nunchaku: the `torch2.11` Windows wheel (1.1.0) fails to import against `torch 2.11.0+cu128`; owner test on the cu130 GPU runtime. NVFP4 gate exists (`helpers/gpu_formats.py`), no live run (needs Blackwell).

</details>

<details>
<summary>FU-159 · Image-to-video and Storyboard clips on NVIDIA PCs (Wan 2.2 TI2V 5B), text-only shots — Shipped 2026-11-10 on test/rtx4090</summary>

**Status or trigger:** **Shipped 2026-11-10 on `test/rtx4090`; live-validated on an RTX 4090.**

**Notes:** Closes the Windows/Linux half of D-40 and D-48. `image_input_support(repo)` in `video_runtime/repos.py` is the one place that says who takes a start and an end picture (LTX-2 MLX: both; Wan 2.2 TI2V 5B on diffusers: start only), used by the catalog flags (`acceptsImageInput`, new `acceptsEndImage`), the video route, the agent bridge and Storyboard. A run with a start picture loads `WanImageToVideoPipeline` (same weights; the loaded-pipeline key carries `i2v`), the picture is scaled to cover and centre-cropped. The row is `cudaOnly` (hidden on a Mac: MPS corrupts Wan 2.2, FU-102/125), 34 GB on disk but `loadMemoryGb` 24 (fp32 transformer on disk, bf16 in memory), and placed with model-level CPU offload because 22 GB of pipeline does not fit a 24 GB card with activations. New shot start mode `startFrom: text` (a text-to-video clip on any model; Shot editor: "Text only (no picture)", all locales). Measured (832x480, 49 frames, 20 steps): text 42.5 s, picture 49.9 s, 12-14 GB VRAM; a three-shot Storyboard (still-started, text-only, continued) made and exported in 286 s. Open: LTX-2.3/2.5 and Kandinsky 6 through diffusers on CUDA (newer, audio; research in docs/research/cuda-model-landscape-2026-11.md), Wan 2.2 I2V A14B for an end frame, FU-145 estimator.

</details>

<details>
<summary>FU-158 · Speech GPU add-on as a tick box on Windows and Linux (D-73 M4) — Open (2026-11-09): wired, never run</summary>

**Status or trigger:** **Open (2026-11-09): wired, never run.** Trigger: the RTX 4090 session (FU-155).

**Notes:** The runtime panel lists the speech GPU add-on as a tick box when the backend offers it; unticking runs `pip uninstall` of torch, torchaudio, torchcodec, chatterbox-tts, voxcpm, diffusers, s3tokenizer, resemble-perth and conformer (`voice_runtime_disk._SPEECH_GPU_REMOVE`). Check that nothing the base runtime needs goes with them, that the tick installs through `start_install(['speech-gpu'])`, and the label and size on the panel.

</details>

<details>
<summary>FU-157 · Voice makeover (D-73): what was deferred or not checked — Open (2026-11-09): built on feat/voice-makeover and feat/voice-fu157, mocked UI and throwaway-cache tests only…</summary>

**Status or trigger:** **Open (2026-11-09): built on `feat/voice-makeover` and `feat/voice-fu157`, mocked UI and throwaway-cache tests only.**

**Notes:** Built: delete for dictation and speech models (companions kept when shared, watermark kept while a cloning model remains, refused while voice is busy or a speech model downloads), add-on ticks with removal, Remove voice runtime, runtime parts and In memory now, speech models on the shared Models table with Fit and Kokoro as the first row, Speak voice row and picker drawer, Voices main-voices strip and Use for menu, Chat Voice popover, Library voice-kind filter, copy. Done on `feat/voice-fu157` (2026-11-09): **Cancel** for a speech-model download (child-process download, window-only `POST /api/voice/clone/models/download/cancel`, row progress and Cancel, partials kept for resume), the **Show N sizes** collapse of a family (families of three or more), the Voice Library title shown once (Voice Models panels have distinct titles). Not checked: a real Cancel of a real download, a real delete on the owner's downloaded models and Hugging Face cache, Remove voice runtime and add-on removal on a real venv (pip uninstall of misaki's `zh` requirements), Windows and Linux, and listening.

</details>

<details>
<summary>FU-156 · Any voice in Chat and calls (D-67 W6, D-72): what was not done or not checked — Open (2026-11-09): W6 built on feat/voice-w6</summary>

**Status or trigger:** **Open (2026-11-09): W6 built on `feat/voice-w6`; tests and a worker-only measurement, no real call.**

**Notes:** Built: `speak/stream` plans through `voice_speak` per sentence (`voice_chat.py`), `ttsVoice` / new `callVoice` validated against `list_voices`, cloned voices read-aloud only (watermark, AI tag, audit; not saved), Kokoro fallback with a note, per-voice chat cost in the pickers. Left: (1) a saved designed voice as chat voice waits for W8 (FU-141; bare "Describe a voice" is refused); (2) the owner's real call on the M4 Max with a chat model loaded: listen to each voice, barge-in feel with a 1.3 s first sentence, memory of chat model plus the 1.7B worker; (3) chat numbers for Chatterbox V3 and Orpheus (`--chat --rows chatterbox-mtl orpheus-3b`; their pickers show no time until then); (4) the fallback note text comes from the backend in English; (5) Windows and Linux (Kokoro only there today) not run.

</details>

<details>
<summary>FU-155 · Speech GPU add-on and CUDA speech rows: run them on the RTX 4090 — Open (2026-11-09): written on a Mac, never run.</summary>

**Status or trigger:** **Open (2026-11-09): written on a Mac, never run.**

**Notes:** `voice_env.speech_gpu_commands` (torch 2.9.1 cu128, a hand-picked dependency list, `chatterbox-tts` and `voxcpm` with `--no-deps`), `voice_worker_torch.py` (Chatterbox and VoxCPM2 on PyTorch, Orpheus GGUF through llama-server plus the SNAC ONNX decoder), rows in `catalog/voice_models_cuda.py` (not offered until measured; `CHAOSENGINE_VOICE_UNMEASURED=1` lists them). Checklist in docs/plans/rtx4090-test.md section 2.1: imports, which Turbo decoder file is read, PerTh after speed and pitch, VRAM and speed into `estimates`, Japanese without pykakasi (GPL, left out), no stray llama-server.

</details>

<details>
<summary>FU-154 · Default cloning model: VoxCPM2 or Qwen3-TTS 0.6B Base (owner listening check) — Built 2026-11-14 (v0.10.0 R6): the New voice form lists every downloaded cloning model with its cost</summary>

**Status or trigger:** **Built 2026-11-14 (v0.10.0 R6): the New voice form lists every downloaded cloning model with its cost; owner decision: `qwen3-tts-1.7b` is preselected and Save is enabled. Listening check still open.**

**Notes:** Listen to `~/.chaosengine/voice/wave2-listening/clone-*.wav` (same synthetic Kokoro sample `reference-kokoro-af_heart.wav`, same text, all watermarked). To switch, set `DEFAULT_CLONE_MODEL = "voxcpm2"` in `backend_service/catalog/voice_models.py` and update the default-model tests. Also fill the rows' `rank` from the same check, and say whether Turbo's style tags (`turbo-style-*.wav`) and sound tags (`turbo-tags.wav`, `orpheus-*.wav`) land. **2026-11-14 (R6):** `DEFAULT_CLONE_MODEL = "qwen3-tts-1.7b"` (was 0.6B), one constant in [voice_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/voice_models.py); the rows' `default` flag, the form's "Default" tag and `pick_clone_model("")` follow it. Measured on the M4 Max (s per 10 s speech, peak GB, download GB): 1.7B 3.6 / 6.5 / 4.54, 0.6B 3.2 / 4.3 / 2.52, VoxCPM2 5.2 / 3.8 / 3.23, Chatterbox MTL 3.6 / 3.7 / 3.2, Chatterbox Turbo 2.1 / 1.7 / 1.2. Reason: the 1.7B is the catalogue's likeness model and costs 0.4 s per 10 s and 2.2 GB more than the 0.6B; no row has a listening `rank`, so nothing beats the Qwen pair on evidence. The owner overrode the plan's "none preselected, 422": an empty model means the default (or any downloaded one), an unknown key is a 400, and the chosen key is saved on the voice (`model`). **Side effect:** old voices with no recorded model now speak with the 1.7B when it is downloaded (was 0.6B). To change the default, edit the one constant and the tests that read it. Real clone with all five models on the Mac passed. **Not tested:** listening quality (owner), the CUDA twins on the 4090 (plan row 7), the Download button in the form on a real download.

</details>

<details>
<summary>FU-153 · Chat waits for long jobs, times on messages and tools, job progress with time left — Built 2026-11-09</summary>

**Status or trigger:** **Built 2026-11-09; first-clip estimate code merged 2026-11-14 (R7, wired in `routes/storyboard.py` `start_batch`); the real 4-shot Chat run is still owed.**

**Notes:** Found on the M4 Max: a 4-shot storyboard (about 25 min) made Qwen3.8-27B call `wait_for_job` 24 times (60 s each) and the turn ended on "maximum tool-use iterations" with no summary. **Done:** in Chat `wait_for_job` blocks server-side up to 30 min (the plan card's limit; [studio.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/tools/studio.py) `CHAT_WAIT_CAP_SECONDS`), streams `toolCallProgress` events every 2 s from a worker thread ([agent.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/agent.py) `_run_tool_call_live`), ends within 0.5 s when Stop is pressed (the job keeps running), and a repeat wait on a job already waited on does not use an iteration (`budget_key`, bounded at 48 free replies). The external `/mcp` tool is unchanged (max 60 s). Messages carry `createdAt` and tool calls `startedAt` (epoch ms; old sessions have none and show no time); the thread shows clock time, full date on hover, live elapsed on tool cards, and one card per waited job. `Job.progress()` ([jobs.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mcp_bridge/jobs.py)) gives `phase`, `step`, `totalSteps`, `elapsedSeconds` and `etaSeconds` + `etaBasis` (`clips`, `steps`, `catalog`) only with a basis; storyboard batches price remaining clips from the clips already made in the run ([batch.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/storyboard/batch.py) `progress`), shown on the chat job card, the Storyboard run bar ([BatchBar.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/storyboard/BatchBar.tsx)) and in `job_status` / `wait_for_job`. **Open:** (1) the first clip of a batch shows elapsed only; it could use the catalogue estimate until one clip is measured; (2) the stand-alone Image, Video, Music and Voice studios keep their own progress displays (not changed); (3) a real model run of a long storyboard in Chat has not been done; (4) an assistant message's time is when it finished, so a 25-minute turn shows the end time; (5) `Intl` unit words ("min", "sec") follow the browser's data for the locale, with no hand translation; (6) a chat wait can still end at 30 min with "still running"; the model must call again. **2026-11-14 (R7):** `Batch.use_models` and `_catalog_price` in [batch.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/storyboard/batch.py) price a step whose kind has no finished clip yet from the catalogue (project size and shot frames, FU-165), so `etaSeconds` shows from the first step with `etaBasis` `catalog`; with no catalogue time it stays elapsed-only. Item (1) above is done. 7 unit tests plus one E2E check (in process, no model run). **Still owed:** a real 4-shot storyboard in Chat with a local model, to see one job card, the time left from the first clip and the summary at the end (plan row 9 on the 4090, and once on the Mac).

</details>

<details>
<summary>FU-152 · Flaky timing test test_mflux_worker.py::CliTimingTests::test_the_cli_split_is_read_off_the_bar — Open (2026-11-08).</summary>

**Status or trigger:** **Open (2026-11-08).**

**Notes:** Failed once in a full run (decode 0.12 s against a 0.15 s floor), passed 3 of 3 alone: it measures wall time and is sensitive to machine load. Loosen the bound or fake the clock.

</details>

<details>
<summary>FU-151 · Hard-coded English strings found by the Klingon pass — Open (2026-11-08): not started.</summary>

**Status or trigger:** **Open (2026-11-08): not started.**

**Notes:** Strings in components or backend text that no locale can translate: Idle, Refreshing..., Copy diagnostics, Gathering system info..., Settings updated, Recommended target text, '0 models discovered' text, DFlash row text, yes/no values. Backend `messages.pot` has never been extracted; about 68 call sites pass `str(exc)`. Two English strings name other apps (`serverTab.connect.ollamaApps`, `agentAccess.subtitle`) and need rewording per the editorial rules.

</details>

<details>
<summary>FU-150 · Klingon locale: re-sync and speaker review — Open (2026-11-08)</summary>

**Status or trigger:** **Open (2026-11-08).** Trigger: each time English strings change; before any launch post.

**Notes:** Translate new `null` keys in `src/locales/tlh/` (52 in studio.json now); have a Klingon speaker review the coined terms listed in docs/plans/klingon/glossary.md and qa.md (three words for 'random' to unify).

</details>

<details>
<summary>FU-149 · Better speech models than Kokoro and Qwen3-TTS presets — Mac rows shipped 2026-11-09 on feat/voice-models-wave2 (D-70 P0 to P5)</summary>

**Status or trigger:** **Mac rows shipped 2026-11-09 on `feat/voice-models-wave2` (D-70 P0 to P5); CUDA lane written, unverified (FU-155); listening check open (FU-154).**

**Notes:** Chatterbox-Turbo, VoxCPM2, Chatterbox Multilingual V3 (default voice from the V2 8-bit `conds.safetensors`) and Orpheus 3B (`mlx-community` 8-bit) are catalog rows with pinned companions, measured on the M4 Max with `scripts/measure-voice-models.py`: Turbo 2.1 s per 10 s, 1.72 GB; VoxCPM2 5.22 s, 3.8 GB; Chatterbox V3 3.63 s, 3.67 GB; Orpheus 7.22 s, 4.19 GB. Sound-tag chips (Turbo's ten style tags shown as undocumented), Chatterbox language, Expression and Guidance, VoxCPM2 design and exact or steered cloning, Orpheus presets; every cloning model gets the AudioSeal mark, the AI tag and the permission rules. Turbo's built-in voice is put back after a clone (mlx-audio leaves the cloned voice on the model). `rank` stays empty until the listening check.

</details>

<details>
<summary>FU-147 · Size caps: useChat.ts and generation.py — Open (2026-11-08): not started</summary>

**Status or trigger:** **Open (2026-11-08): not started.** Trigger: next change to either file.

**Notes:** Both are over the soft caps in CLAUDE.md. Extract before adding.

</details>

<details>
<summary>FU-145 · Verify per-model native frame rates — Open (2026-11-08): not started</summary>

**Status or trigger:** **Open (2026-11-08): not started.** Trigger: RTX 4090 test run (docs/plans/rtx4090-test.md).

**Notes:** The values in `backend_service/size_limits.py` came from memory sizes, not model cards. Check each against its model card or a real render, fix the hint text.

</details>

<details>
<summary>FU-144 · Chat thread pane very short at 375 px — Open (2026-11-08): not started.</summary>

**Status or trigger:** **Open (2026-11-08): not started.**

**Notes:** On a phone-width window the app's own scroll box for the thread is short. Give it a minimum height; add a 375 px check to the Chat e2e.

</details>

<details>
<summary>FU-143 · Chat studio tools: batch a whole job into one plan — Open (2026-11-08): not started</summary>

**Status or trigger:** **Open (2026-11-08): not started.** Models make one plan card per render.

**Notes:** Add a line to the studio tool descriptions and the agent guide (D-61) telling the model to put every render for a job in one plan, then check with a real model.

</details>

<details>
<summary>FU-142 · Frame interpolation at export for 48 / 50 / 60 fps (D-43, D-64) — Open (2026-11-08): not built</summary>

**Status or trigger:** **Open (2026-11-08): not built.** Trigger: a user asks for smooth 50 or 60 fps from a model that makes 24 fps.

**Notes:** The project offers 24 to 60 fps (D-64), but a clip from a model with a lower native rate is converted at join by plain ffmpeg `fps` (frames repeated or dropped), which does not add motion. Real conversion needs RIFE-based interpolation at export, the item D-43 already ranks in [docs/plans/speed-quality.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/plans/speed-quality.md); check its licence first. Fit points: the final encode in [assemble.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/storyboard/assemble.py) and the hint in [FormatFields.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/storyboard/FormatFields.tsx).

</details>

<details>
<summary>FU-141 · Voice rework backend: parts not yet built (D-58) — Open (2026-11-08): backend landed on feat/voice-rework</summary>

**Status or trigger:** **Open (2026-11-08): backend landed on `feat/voice-rework`; these parts deferred.**

**Notes:** Not built: a saved designed voice (design, then clone from the clip so the voice stays the same), a `language` field on `POST /api/voice/speak` (done 2026-11-09 with D-70: `language`, `expression`, `guidance`; Chatterbox V3 uses the code, Qwen3-TTS maps its ten), server-side `[pause]` handling (the screen still cuts at it), streaming audio for the new route, the CUDA path for the new Qwen3-TTS rows (`qwen-tts`), and listening checks. **Real run done 2026-11-08 (M4 Max, 64 GB):** all five Qwen3-TTS rows speak through `POST /api/voice/speak` with real weights; timings and memory are in the catalog `estimates` (CustomVoice 0.6B 2.9 GB, 1.7B 3.5 GB, VoiceDesign 3.5 GB, Base clones 4.3 and 6.5 GB; about 2.5 s of wall time per 10 s of speech on the three speak rows, 3.2 to 3.6 s on clones; the 0.6B rows are not quicker than the 1.7B 8-bit rows, so the old Faster/Slower row text is gone). Seed repeats exactly, speed 1.3 gives 1/1.3 of the length, Cancel stops in under 0.1 s and the next job reloads (about 4 s). Fixed: very quiet clone samples gave very quiet speech (peak 0.04), so quiet speech is now lifted. Nobody has listened to the WAVs yet (style, design and clone quality unverified); response `model` is a repo id for clones but a catalog key for other voices.

</details>

<details>
<summary>FU-140 · Lip sync for storyboard shots (D-57) — Open (2026-11-08): banked on the backlog, not started.</summary>

**Status or trigger:** **Open (2026-11-08): banked on the backlog, not started.**

**Notes:** Today voice, effects and music are placed by hand on the storyboard timeline; a video shot's own sound is its own track; nothing matches mouth movement to speech. Route 1: audio-driven video (a speech-to-video model takes the voice clip and a face image). Route 2: a lip-sync pass over an existing clip. Start with route 1, reusing the Voice Studio output. Needs a catalog row, a measured size and memory on the M4 Max, and an E2E check in the matching phase.

</details>

<details>
<summary>FU-139 · Music on CUDA (Windows and Linux) and XL on CUDA — CUDA lane built and measured on the RTX 4090 (Windows 11), 2026-11-19, branch from test/rtx4090</summary>

**Status or trigger:** **CUDA lane built and measured on the RTX 4090 (Windows 11), 2026-11-19, branch from `test/rtx4090`; Linux not tested.**

**Notes:** `music_env.lane()` is `cuda` on Windows or Linux x86_64 with `nvidia-smi` on PATH; the install adds `--extra-index-url .../whl/cu128` to the same uv.lock export (`torch==2.7.1+cu128` on win32 resolves on Python 3.12; no flash-attn or triton there, SDPA is used) and checks `torch.cuda.is_available()`; the marker records the lane. The worker runs the torch DiT in bf16, no offload from 20 GB (ACE-Step's CPU offload below, `CHAOSENGINE_MUSIC_OFFLOAD`), steps and cancel through forward hooks on `handler.model.decoder`, `timings.vram_peak_gb`; Windows RSS via `GetProcessMemoryInfo`. **Bug fixed on the way:** on Windows the worker's first load always hung (main thread blocked in `ReadFile` on stdin stalls numpy's DLL start-up inside `import torch` on the engine thread; 300 s timeout); requests are now read from a duplicate of the pipe and stdin is NUL (`tests/test_music_worker.py` regression). The memory gate reads host RAM (`estimates.cuda.hostPeakGb`) on CUDA, not the 47 GB Mac XL peak. Sounds stay Mac-only with their own sentence and code `music.soundUnsupportedPlatform`; a sound download is refused there (409). The D-80 wording ("Music generation runs on Apple Silicon Macs in this version...") is replaced by "Music generation needs an Apple Silicon Mac or an NVIDIA GPU on Windows or Linux; ..." (en only, other locales null). **Measured** (app routes, warm, 1 variation): turbo 30 s 0.97 s, 60 s 1.3 s, 180 s 3.1 s, 600 s 10.6 s, VRAM 8.6 GB, host 2.7 GB, load 9 to 15 s, cancel answered 409 in 0.5 s and the next run worked; venv 6.6 GB. XL: see `docs/progress/rtx4090-progress.md` 2026-11-19 FU-139. Follow-ups: FU-259 to FU-262.

</details>

<details>
<summary>FU-137 · Linux ARM64 build (NVIDIA DGX Spark class) — Phase 1 coded 2026-11-02 on release/v0.9.10, not yet built in CI</summary>

**Status or trigger:** **Phase 1 coded 2026-11-02 on `release/v0.9.10`, not yet built in CI; experimental until someone tests on real hardware.**

**Notes:** Asked by a prospective user (2026-11-02). **Today:** releases are Linux x86_64 only (the matrix in [release.yml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/github/workflows/release.yml) has one `ubuntu-22.04` x86_64 job; only the amd64 AppImage and deb exist, neither installs on ARM). **From source on ARM Linux** the app starts but GGUF would run on the CPU: [update-llama-cpp.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/update-llama-cpp.sh) configures `-DGGML_METAL=ON` only (no CUDA flag), [stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) knows only `linux-x64` so no `llama-server` is staged, and the `llama-server` lookup in [binaries.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/binaries.py) then needs `CHAOSENGINE_LLAMA_SERVER` or a PATH entry; a user who supplies a CUDA build that way gets the GPU. GPU detection (`nvidia-smi`) is present and would see a Spark. The Setup page's torch CUDA install ([gpu_bundle.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/gpu_bundle.py)) skips only macOS arm64; aarch64 CUDA wheel coverage for torch and the extras (vLLM, bitsandbytes, Nunchaku) is unchecked. **Plan:** (1) release matrix job on `ubuntu-24.04-arm` (free for public repos) producing aarch64 AppImage and deb; (2) `linux-arm64` entry in stage-runtime picking llama.cpp's `ubuntu-cuda-13.4-arm64` binaries (exist in b11981; plus the `cudart` archive) with `ubuntu-arm64` as the CPU fallback; (3) an aarch64 Python runtime for the bundle; (4) a CUDA branch in `update-llama-cpp.sh` for Linux (`-DGGML_CUDA=ON`); (5) check torch / extras wheels on aarch64. **Phase 1 done (unrun in CI):** `ubuntu-22.04-arm` job in [release.yml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/github/workflows/release.yml) and [build.yml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/github/workflows/build.yml); `linux-arm64` in stage-runtime stages llama.cpp's CPU `ubuntu-arm64` build (the CUDA ARM build is not bundled: it fails to start without a CUDA 13 driver); `generate-updater-manifest.mjs` adds optional `linux-aarch64` entries (AppImage `_aarch64`, deb `_arm64`); `update-llama-cpp.sh` builds with CUDA on Linux when `nvcc` exists, Metal on macOS, else CPU; 4 tests in [test_linux_arm64_build.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_linux_arm64_build.py). **CUDA llama-server (2026-11-03, Linux x86_64 and ARM64):** not bundled (it would add 0.7 GB to every Linux download); Settings > Accelerators shows an Install button on Linux hosts where `nvidia-smi` answers ([LlamaCudaPanel.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/settings/LlamaCudaPanel.tsx)). [llama_cuda.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/llama_cuda.py) downloads ggml-org's CUDA build of the pinned tag plus its CUDA runtime archive into `~/.chaosengine/bin/llama-cuda/<tag>/` (CUDA 13.4; 12.8 on x86_64 with a driver older than 580), checks every archive member stays inside the folder, and flattens both into one folder: the binaries carry `RUNPATH $ORIGIN`, so the runtime libraries are found without `LD_LIBRARY_PATH` and only `libcuda.so.1` comes from the driver. `_resolve_llama_server` ([binaries.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/binaries.py)) prefers it only when `llama-server --list-devices` lists a CUDA device (checked once per binary; `CHAOSENGINE_LLAMA_CUDA=0` switches it off). Routes `GET /api/setup/llama-cuda-status`, `POST /api/setup/install-llama-cuda`, `GET /api/setup/install-llama-cuda/status` ([llama_cuda.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/llama_cuda.py)); 14 tests in [test_llama_cuda.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_llama_cuda.py). **Checked on the Ubuntu 26.04 ARM64 VM (no GPU):** the real b11981 archives download in 49 s, 60 files, 0.9 GB on disk, `llama-server --version` runs, `--list-devices` reports none, so the bundled build stays in use. **Not checked:** a run on a real NVIDIA GPU (Spark or x86_64), the panel itself (it renders only on Linux with a driver). **Spark readiness (2026-11-03):** GPU detection: a shared-memory NVIDIA GPU (DGX Spark GB10, GH200) reports `memory.total` as `[N/A]`, which made `_snapshot_nvidia` report no GPU; [gpu.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/gpu.py) `_parse_nvidia_smi` now takes system RAM as its memory and reads only the first GPU line (two GPUs made the old comma split fail); `isCudaHost` ([platform.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/utils/platform.ts)) accepts Linux ARM64 (Windows on ARM stays out), so CUDA rows, boosters and vLLM show on a Spark. The Spark-shaped `nvidia-smi` line is assumed from how shared-memory GPUs report, not seen. [check-platform-wheels.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/check-platform-wheels.py) proves from any machine that the release venv, the portable voice pack, the GPU runtime and vLLM resolve from wheels on Linux aarch64 and x86_64 (all four groups pass; it first failed on vLLM because its tag list missed `manylinux_2_31`); CI runs it on `ubuntu-latest` and gains an advisory `ubuntu-24.04-arm` test job. [linux-gpu-check.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/linux-gpu-check.sh) is the tester script: machine, driver, app status, CUDA llama-server devices and a `llama-bench` run on LFM3 1.7B, written to `~/chaosengine-gpu-report.md`; run on the ARM VM it produced a full report (CPU: 115 tok/s prefill, 53 tok/s decode on 2 threads). **Research pass (2026-11-03, [notes](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/UPSTREAM_RESEARCH_2026-11-03-linux-arm64.md)):** checked that llama.cpp b11981's CUDA build includes `121a-real` (native Spark kernels), that torch 2.11 on PyPI is a CUDA 13 build for aarch64 and that download.pytorch.org has no cu124 aarch64 build, and that vLLM 0.33 pins a CUDA 13 torch. Fixed from it: `--no-mmap` for llama-server on a shared-memory NVIDIA GPU (`nvidia_shared_memory()`; `CHAOSENGINE_LLAMA_MMAP=1` keeps mmap); the GPU runtime installer's torch index order now follows the CPU and driver (`cuda_torch_indexes()`: ARM64 cu130 then cu128, driver 580+ starts at cu130, 570+ at cu128; the old cu124-first order had no ARM64 build and no Blackwell kernels, so RTX 50-series on x86_64 benefits too); the voice pack skips CUDA 12 libraries on ARM64 (CTranslate2's aarch64 wheel is CPU-only). **Phase 2 open:** whether torch 2.11 cu13 runs diffusers correctly on the GB10 (sm_121); aarch64 Python runtime check; torch / extras wheels on aarch64; first CI run of the ARM job; README download rows once an ARM asset exists. **First real ARM64 packages (2026-11-03):** the manual Build workflow now takes `linux_release_mode=true` (release staging on Linux: embedded Python, pinned constraints; artifacts `ChaosEngineAI-linux-aarch64` / `-x86_64`), because its default development staging ships no Python runtime (ARM deb 8 MB). Run 41635776807: ARM deb 77 MB, AppImage 153 MB, manifest `mode: release`, `llamaServer` set, no warnings. On the Ubuntu 26.04 ARM64 VM (4 vCPU, 3 GB RAM, no GPU): `apt install` of the deb pulls its dependencies and the app starts with the backend answering in 4 s; the AppImage runs without `libfuse2` installed (backend in 2 s); in both the web view polls the API, so the UI loads (the VM screen was locked, so no screenshot); LFM3 1.7B Q4_K_M loads through the bundled CPU llama-server in 4.2 s and answers at 29 tok/s. Tests also fixed for the ARM runner (torch index order follows the host) and Windows (bash is the WSL stub there). **Testing:** the Ubuntu 26.04 ARM64 Parallels VM can test the build, install and CPU-only GGUF chat; CUDA, Blackwell and the unified-memory behaviour need a real DGX Spark (none here).

</details>

<details>
<summary>FU-136 · Boot-time MLX probe that fails once switched MLX off for the whole session — Fixed 2026-10-30 (cause of the one failure not found)</summary>

**Status or trigger:** **Fixed 2026-10-30 (cause of the one failure not found); packaged v0.9.9 build looked at on the M4 Max.**

**Notes:** Found from a Dashboard screenshot of the packaged app: Activity Feed said "MLX installed: no / MLX usable: no" and Hardware Fit said "70B GGUF, cross-platform mode" while `/api/health` of the same backend reported MLX 0.32.4 usable. The capability probe at startup is cached (`CAPABILITY_CACHE_TTL_SECONDS` 300 s) and the controller keeps its result, so one failed probe left MLX off until something forced a refresh (the Settings > Accelerators page does on mount). **Measured:** the probe takes 26 s on the first start of a freshly extracted runtime (every new build extracts to a new folder, so each new install is a cold start) and 2-3 s afterwards; four cold boots in a row (fresh extraction each, backend launched from a parent that stays alive) all probed fine and logged "yes", so the failure is intermittent and was **not reproduced on purpose**: it did happen in two runs, the packaged app itself and a backend I launched from a shell that exited at once (so `_watch_parent_and_exit` may have fired), and the probe payload came back empty with exit code 0 and no message in the second. **Fix:** `_capability_probe_worker` ([controller.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/controller.py)) re-asks up to twice, 3 s apart, when the host is Apple Silicon and MLX is not usable, while the placeholder (`probing` true) is still shown, so the readiness line and the recommendation are only written once; an empty probe result now carries the message "The MLX probe returned no readable result." ([capabilities.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/capabilities.py)). 4 tests in `CapabilityProbeRetryTests`. The Dashboard process table (`_list_llm_processes`) no longer lists Claude Code (its install path contains `.openclaw`, which matched the `openclaw` keyword): the keyword is gone and a command-line marker only counts in the program and its first two arguments, never in a word with spaces (a prompt). 2 tests in `RunningProcessListTests`. **Open:** (a) the cause of the single failed probe; (b) the "Hardware profile refreshed" activity line is still written at boot from the placeholder snapshot, so it can say "70B GGUF" on a Mac; the live recommendation is correct; (c) a user whose probe still fails after the retries needs Settings > Accelerators to refresh.

</details>

<details>
<summary>FU-135 · DFlash on hybrid Qwen3.5 / 3.6 / Next targets produced garbage text on the release MLX stack — Fixed 2026-10-29 (root cause found, patched, self-checked at load)</summary>

**Status or trigger:** **Fixed 2026-10-29 (root cause found, patched, self-checked at load); measured on Qwen3.6-27B.**

**Notes:** Found by the 27B check for FU-112: through the app, DFlash on Qwen3.6-27B 4-bit replied with random tokens at 8 tok/s and 0.0 accepted tokens per step (plain: a correct reply at 29 tok/s); the first generated token already differed from plain decoding. **Isolation** (same model and prompt, the pinned dflash-mlx `fada1eb`): mlx 0.31.2 + mlx-lm 0.31.3 works; mlx 0.32.4 + mlx-lm 0.31.3 works; mlx 0.32.4 + mlx-lm 0.32.2 fails, so the break is mlx-lm 0.32. A logits comparison of the adapter's own forward pass against `model(...)` (no draft, no hooks) differed by 26.8, which put the fault in the adapter, not in the speculative hooks. **Cause:** the Qwen adapter's `text_wrapper` returns the model itself when it has a `model` attribute, else its `language_model`; mlx-lm 0.32 gave the top-level Qwen3.5 / 3.6 / Next `Model` a `model` attribute, so `logits_from_hidden` read `tie_word_embeddings` from the wrong object (missing, so it defaulted to tied) and projected the hidden state through the embedding matrix instead of the untied `lm_head`. Ruled out on the way: the q/k normalisation epsilon, cache truthiness, `fa_idx` / `ssm_idx` / `pipeline_layers`. **Fix:** [dflash_compat.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/dflash_compat.py) `patch_text_wrapper` (prefer `language_model` when it owns a `model`; idempotent) is applied before the adapter is resolved, and `self_check` runs 16 fixed tokens through the adapter's forward pass and through the model at load: a different next token or logits more than 1.0 apart turn DFlash off with the reason in the note ("DFLASH is off for this model: ..."), an error inside the check lets the lane through. Logits then agree to 0.000 (0.344 with the rollback hooks armed). Also fixed here: a speculative reply kept the stop token (`<|im_end|>`) in its text. **Measured, M4 Max, Qwen3.6-27B 4-bit + `z-lab/Qwen3.6-27B-DFlash`, 256 tokens, 2 interleaved runs, decode tok/s (code / explain / story / think):** plain 16.5 / 18.1 / 21.9 / 20.1 (mean 19.2; the bench machine was not idle, the app measured 27 to 29 plain); linear DFlash block 16: 21.6 / 12.3 / 8.9 / 15.0 (mean 14.4); **block 8: 25.7 / 21.2 / 12.5 / 23.0 (mean 20.6)**; accepted tokens per step 4.3 / 2.6 / 1.1 / 2.9 at block 8. Through the app (400-token code reply): 26.9 plain against 35.1 with DFlash, 4.06 accepted per step, identical text. Block 8 beats block 16 here too (16 loses 25 % to plain on average); prose loses (FU-134); every difference from plain is a near-tie of the target's top two logits (0 to 0.125 apart). **Upstream claims 2.8 to 3.1 times on this model; we get 1.3 to 1.6 times on code at the pinned v0.1.5.1, so the gap is the v0.1.7+ adaptive block size and verify work (FU-057).** Tests: [test_dflash_compat.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_dflash_compat.py) (the patch against the real adapter class, the self-check on mismatch, drift and an error), two lifecycle tests, the stop-token helper, and an E2E check that loads a hybrid model with DFlash on. **Open:** (a) the plain and DFlash timings above come from a busy machine; (b) DDTree is not available on hybrid models (FU-112), so their tree budget is dropped as before; (c) the self-check covers only the first forward pass: a mismatch that appears only in the armed rollback path or at longer lengths would pass it (the identical-text check in the bench is the cross-check); (d) other Qwen3.5 / Next sizes and the MoE variants (Qwen3.6-35B-A3B) were not run.

</details>

<details>
<summary>FU-134 · Speculative decoding on creative prose is 2 to 4 times slower than plain — Open (2026-10-29), found by the 27B-class check in FU-112</summary>

**Status or trigger:** **Open (2026-10-29), found by the 27B-class check in FU-112; not fixed.**

**Notes:** On Qwen3-Coder-30B-A3B (and the 4B before it) the story prompt runs 21 to 37 tok/s with DFlash or DDTree against 84 plain: the drafter proposes tokens the target rejects (0.4 to 0.8 accepted per step), so each step pays a verify pass for nothing. Code and reasoning prompts gain 1.1 to 1.7 times. A user who turns speculative decoding on for general chat gets a slower reply on prose with no warning. **Options:** (a) an acceptance gate in our own loops: keep a rolling mean of accepted tokens and, below about 1.2 per step, run plain steps for the next N tokens before probing again (our DDTree loop in [ddtree.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/ddtree.py) can do this today; the linear path is upstream's `stream_dflash_generate` and cannot); (b) take upstream's adaptive block size (dflash-mlx 0.1.7 and later, needs the FU-057 migration); (c) only a note in the launch modal ("speeds up code and reasoning, can slow prose"). **Recommendation:** (c) now, (a) for DDTree and a switch of linear DFlash to our own loop with the same gate if that proves clean, (b) with FU-057. Measure on 4 prompt types before and after; the bench is `scratchpad/work/bench.py`-style (plain, linear, DDTree, interleaved, 3 runs).

</details>

<details>
<summary>FU-133 · "Best fit for your hardware" badges, recommended Studio settings, per-model size limits — Built 2026-10-29 on feat/models-fit (team of agents)</summary>

**Status or trigger:** **Built 2026-10-29 on `feat/models-fit` (team of agents); badges show on the Images page now, Video and Chat follow with FU-129; tests pass; Windows, Linux, CUDA and other Mac sizes not run.**

**Notes:** **Fit:** [fit.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/fit.py) rates every chat, image and video variant `fits` (needs up to 70 % of usable memory), `tight` (up to 92 %), `wont-fit`, or `unknown` (the row cannot run on this platform at all), attaches `fit` to each variant and `bestFitVariantId` to each family in the three catalog payloads; usable memory comes from [hardware_class.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/hardware_class.py) (Apple: the real Metal working-set limit read once in a subprocess, 51.8 GB of 64 on this Mac, never importing mlx in the backend; CUDA: VRAM; CPU: 60 % of RAM). Chat need = file size + 10 % + the KV cache at 8K from the model's own layout (a parameter-count table for models not on disk, never a Hub call); image and video use the catalog footprint (`footprintBasis: "measured"` only where a catalog comment records a run), else size x 1.3 estimated. Best pick: newest release that fits, then for chat the larger model, for image and video the faster one. 67 tests. **Recommendations:** [recommendations.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/recommendations.py) with data in [recommended_settings.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/recommended_settings.py): the Studios show "Recommended for this machine" with Apply, each row marked Measured or Estimated; only engine-relevant settings (mflux, sd.cpp and ltx-2-mlx never get cache strategy, CFG decay, preview VAE, sampler or FP8; those controls are now hidden for mflux and sd.cpp rows too). **Measured on this M4 Max (3 prompts each, looked at side by side):** FLUX 3 Klein 8 steps takes 40 s against 70 s for the default 16 with no visible difference; Qwen-Image-2.1 28 steps (187-191 s) looked as good as 40 (258-271 s) on two prompts, stated in the reason but not made the default; Qwen-Image-2.1 Turbo 6 steps saves 18 % but the model is tuned for 8; LTX-2.3 at 97 frames (4 s) takes 135-138 s and 15 GB; Iris-3B 768 px is not better than 1024. Qwen-Image-2.1 peaks at 46-50 GB, far above the 30 GB the catalog carries (**open: correct the catalog footprint**). **Not timed:** every cache strategy (no diffusers model is downloaded), so TeaCache / FBCache / CFG decay recommendations are estimated from model cards and our own notes and say so. **Size limits:** per-model `maxSide` / `maxPixels` / `nativeSide` in [size_limits.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/size_limits.py) replace the global 2048 px cap (FLUX.2 4096 per the maker's 4 MP, Qwen-Image-2.1 4096, SDXL-class 2048; video follows the model cards: Wan and LTX-Video 1280, HunyuanVideo 1280, Mochi 848, CogVideoX 720 to 1360, LTX-2.x 4096), hard ceiling 8192, aspect ratio kept when a request is over; the memory gate charges 0.15 x footprint per megapixel above the model's reference size (the Studio's own estimate, one cross-check: LTX-2 at 1280x704 x 121 = 18 GB); several rows are 4096 only because no card gives a maximum (FLUX.1, Z-Image, Qwen-Image, FLUX 3: unverified). 61 tests, an E2E catalog check, `scripts/measure-image-ceiling.py` (not run). **Review 2026-10-29 (backend and Studio reviewers):** fixed: the image memory gate charged 1.1 x a footprint that is already a measured process peak, so Qwen-Image-2.1 (50 GB) was refused on a 64 GB Mac below 55 GB free (now 1.0 for measured peaks, `headroom` argument); the best-fit pick for FLUX.1 schnell and dev was the sd.cpp row because its release date was the day the row was added and its time estimate said 4.5 s (now the base model's date and 114 s / 230 s; FLUX 3 Klein rows got estimates too); a convert row no longer wins the chat pick while a row that runs as downloaded fits; the first catalog response waits up to 6 s for the Metal limit instead of 1.5 s so it does not use the 0.75 x RAM guess; the LTX-2 guard rounds to 64 like the engine; a size over a model's limit keeps its aspect ratio in the Studios (3000 x 1688 on a 2048 model is 2048 x 1152, not 2048 x 1688); the LTX-2 memory estimate grows with size and frames again (3 GB per 70,000 latent tokens, fitted on two measurements); the recommended panel has separators, a fixed action column and two-line reasons. **Fixed afterwards:** on a CUDA host the size term of the gate now compares the extra memory a large size needs against free GPU memory (`get_device_vram_free_gb`, read through the metrics monitor, kept 2 s) instead of system RAM, and the refusal says "GPU memory"; the model's own load is still judged as before because offload modes let a big model run on a small card. Unit-tested with a stubbed monitor only; **no NVIDIA card was available to run it**. **Still open from those reviews:** frame count is not charged at the reference size for diffusers video; Iris-3B's footprint is measured at 768 px but used as the 1024 px reference; usable memory is in GiB against decimal GB sizes (about 7 % pessimistic); fixed-resolution video models (CogVideoX, Mochi) get an odd size instead of their trained one; `measured` is not tied to this chip or RAM size; `availableLocally` in `model_family_payload.py` is still a name check; the best-fit default for the LTX-2 family is a gated repo; Studio unit tests cover the helpers, not the wiring (mflux and sd.cpp control hiding, the clamp effect). **Also found and fixed:** the Dashboard already had a chat-only "Hardware Fit" panel; the Video Studio warned "about 42 GB" for LTX at 97 frames where 15-18 GB was measured (the diffusers attention heuristic does not apply to ltx-2-mlx, now skipped); distilled video rows (CausVid, lightx2v, LTX distilled) ran 30 steps at guidance 5 because the Studio never sent the row's own `defaultSteps` / `cfgOverride` and the backend substitutes only schema defaults (the Studio now starts from them; the image side sends 6.0 against the backend's 5.5 schema default and still never applies a row's guidance); the startup elapsed-seconds timer re-rendered the app once a second forever (now cleared after load). **Open:** (a) the real image ceiling on this Mac is unmeasured (measured 2026-10-29 with `scripts/measure-image-ceiling.py`, FLUX 3 Klein through mflux, 1 step, M4 Max 64 GB: 1024² 8 s / 12.1 GB peak; 1536² 13 s / 19.1 GB; 2048² 29 s / 24.1 GB; **3072² 110 s / 41.3 GB**; 4096² refused by the memory gate, which estimated 68 GB against 51.8 usable, and a straight-line fit of the four runs gives about 3.3 GB per megapixel and roughly 65 GB for 4096², so the refusal is right; more steps multiply the time, not the memory; the gate's 0.15 x footprint per megapixel is a little low for this model (2.3 GB per megapixel charged against about 3.3 measured) but it lands within 5 % at 4096²; other models and other Macs not run, a 128 GB or 192 GB machine should reach 4096²); (b) catalog footprints and time estimates for rows other than the four measured ones are guesses (Z-Image-Turbo shows `tight` at 42.9 GB from size x 1.3); (c) fit and recommendation reasons are English-only; (d) no cache-strategy timing until a diffusers model is downloaded; (e) the image-side guidance default mismatch above.

</details>

<details>
<summary>FU-132 · oMLX (jundot/omlx) as an MLX engine lane — Open question (2026-10-29)</summary>

**Status or trigger:** **Open question (2026-10-29); recommendation: do not build a lane now; it already works as a remote provider, so measure it first.**

**Notes:** oMLX is an Apache-2.0 MLX inference server for Apple Silicon (macOS 15+, Python 3.11-3.13; about 23k stars on the repo page; no release list shown there, version unverified): continuous batching through mlx-lm's `BatchGenerator`, a two-tier KV cache (RAM plus SSD, so a long chat's prefix survives a restart), several models served at once with LRU eviction, LLM / VLM / OCR / embedding / rerank models, OpenAI and Anthropic compatible endpoints, a menu-bar app. It documents no in-process Python API, only `omlx serve` over HTTP. The owner's tweet link that prompted this (a note that another desktop app now supports oMLX) could not be read (HTTP 402), so what that app exactly supports is unverified. **What we already have:** our own MLX worker with prompt-cache reuse (FU-090), DFlash / DDTree, TurboQuant, and the isolated-venv + HTTP-proxy engine pattern (MTPLX, TensorFold). **Free today:** Settings > Remote providers takes any OpenAI-compatible `apiBase`, so `http://localhost:8000/v1` should work with no code (not tested). **What a lane would add:** the SSD-persistent KV cache (we keep prompt caches in RAM only), concurrent requests (agents, Compare, the API server), embeddings and rerank. **What it would cost:** another MLX stack in its own venv (mlx pins differ), another HTTP hop, and it cannot combine with our speculative decoding or TurboQuant, so it would be a separate choice in the engine picker. **Recommended next step:** with the owner's yes to installing a third-party tool in an isolated venv, run one A/B on this M4 Max against our MLX lane on the same cached model (single-request tok/s, second-turn time to first token, 4 concurrent requests), then decide; build the lane only if the SSD cache or the concurrency win is clear.

</details>

<details>
<summary>FU-131 · Project hub (status, backlog, decisions) for ChaosEngineAI — Shipped 2026-10-29.</summary>

**Status or trigger:** **Shipped 2026-10-29.**

**Notes:** The working agreement wants one navigable HTML hub per project; ChaosEngineAI predated it. Now `docs/hub/` holds `status.md` (read first), `decisions.md`, `backlog.md`, `overview.md`, `rules.md` (generic agreement plus the project rules), `testing.md`, `engineering.md`, `risks.md`, `ideas.md` and `progress/` (one file per stream). The FU rows stay here in CLAUDE.md as the source of truth: [tools/hub_tracker.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tools/hub_tracker.py) reads them and writes `docs/hub/tracker/` (open and closed pages, one collapsible entry per row; repo links become GitHub links so the hub's link check passes), and `tests/test_hub_tracker.py` fails when the committed pages drift from this file. [tools/build_hub.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tools/build_hub.py) (standard library, from the project-kickoff kit) builds `docs/site/` (gitignored); open `docs/site/index.html` from disk. `tools/run_tests.sh` runs the hub checks (`--full` adds pytest, vitest and tsc). **Rule:** rebuild after every batch; log every question or decision from chat on `decisions.md` the same turn. **Open:** the kick-off answers were never written down, so `overview.md` records the project as it stands, not a kick-off; the status and decisions pages are hand-written and must be kept current by hand.

</details>

<details>
<summary>FU-130 · Sound in Video Studio: no toggle, make the sound visible — Shipped 2026-10-29 (decided by the owner on this recommendation).</summary>

**Status or trigger:** **Shipped 2026-10-29 (decided by the owner on this recommendation).**

**Notes:** LTX-2.3 / 2.5 through ltx-2-mlx always write stereo AAC sound (the DiT makes audio latents jointly; `--no-audio` would only skip the decode and mux), every other video row is silent, so the gap was the players: both `<video>` elements were `muted` and nothing said a clip had sound. **Done:** [video_audio.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/video_audio.py) reads the mp4 boxes (`moov` > `trak` > `mdia` > `hdlr` = `soun`, a few KB even from a large file, no ffprobe) and every saved artifact carries `hasAudio` (older clips without the field are read from their file when listed); a monochrome [SoundBadge](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/SoundBadge.tsx) shows on gallery cards and in the result window; the result window starts a clip with sound unmuted (the user just pressed Generate; silent clips stay muted so autoplay is never refused) while the gallery keeps muted autoplay; Video Studio shows a one-line hint under the prompt for ltx-2-mlx rows ("describe the sound too"). Tests: [test_video_audio.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_video_audio.py) (hand-built boxes including the word `soun` inside media data, truncated and garbage input, real ffmpeg files, the older-clip fallback) and an assertion in the E2E LTX check (`hasAudio` must be true). Verified on six LTX clips made this session (all report sound) and in the running app: the Video Gallery shows the Sound badge on LTX clips (checked 2026-10-29, the result window not seen live). **2026-10-30:** gallery cards of clips with sound also carry a speaker button over the preview (top right, crossed out while muted; click to hear the clip; it follows the player's own volume control, and unmuting one clip mutes the others so only one speaks at a time); clips without sound have no button (checked in the dev UI against the 37 real clips: 16 with sound got the button, 21 without did not; the sound itself was not listened to). **Open:** a "Generate sound" checkbox only makes sense once image-to-video or sound-to-video makes audio an input; a WebKit build that refuses unmuted autoplay would leave the result clip paused with controls shown (not seen on macOS); the locale files carry no translations for the three new strings yet (inline English defaults).

</details>

<details>
<summary>FU-129 · One "Models" page per area instead of separate Discover and My Models pages — Built 2026-10-29 on feat/models-fit by four Opus agents (shared components + Images, Video, Chat) from a read-…</summary>

**Status or trigger:** **Built 2026-10-29 on `feat/models-fit` by four Opus agents (shared components + Images, Video, Chat) from a read-only map of the six old tabs; merged, tested, looked at in the running app; not pushed; Windows and Linux not run.**

**Notes:** Shared parts in [src/components/models/](https://github.com/cryptopoly/ChaosEngineAI/tree/staging/src/components/models/): `ModelEntry` + per-area adapters (`adapters/image.ts`, `video.ts`, `chat.ts`), `resolveModelStatus` (buckets Installed / Available / Downloading / Needs attention; `loaded` is a chip), `ModelsPage` shell, `ModelCard`, `ModelFilterBar`, `ModelDownloadControls`, `FitBadge`, `AcceleratorPills` (host-aware), `confirmAndDelete`. Pages: [ImageModelsPage](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/images/ImageModelsPage.tsx), [VideoModelsPage](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/video/VideoModelsPage.tsx), [ChatModelsPage](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/models/ChatModelsPage.tsx) (+ the specialised Hugging Face results card). The old six tabs are deleted; tab ids `image-discover`, `video-discover`, `online-models`, `my-models` are gone from `TabId` and route through [tabRedirects.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/constants/tabRedirects.ts) in one `setActiveTab` wrapper with a preset (old Discover ids open the Available filter, old My Models ids the Installed filter; persisted sidebar ids migrate). **Chat join rule** (R1): every catalog variant is a row; a local model merges into a catalog row only when each is the other's best fuzzy match (score >= 12) AND its identity is confirmed (the HF-cache folder is that repo, or the item is named `org/name`, or the backend says it is installed); other local models are their own rows; a download no row shows gets a `download://` row; each download sits on exactly one row; `availableLocally` alone no longer makes a catalog row installed (it flagged three models that are not on disk: it is a name-substring check). **Decisions applied:** installed chat rows start a new thread with "Use in this chat" and "Load for server" in a menu; image and video model deletes confirm and name every sibling sharing the files; not-installed rows this host cannot run are hidden, installed ones stay with a "Cannot run on this machine" chip; paused rows show Resume (the old My Models tabs showed "Download model"); the 7-value status select became the four buckets (paused versus downloading and failed versus incomplete can no longer be filtered apart, the status sort still orders them); "Installed Models" / "Browse Catalog" buttons dropped (the filter does that job). Tests: 773 vitest cases (adapters incl. the four join cases with three mutation checks, status, filters, download keys incl. video's variant-id keying, redirects, delete confirm text, FitBadge). **Open:** (a) a real download, pause, resume, delete, Reveal or Convert click was not run on any page; (b) the Hub results card "Installed" claim and the Details window still trust the backend's loose `availableLocally`; (c) the delete confirm names more affected rows than the "shares storage +N" badge counts (video); (d) a preset navigation keeps leftover search and task filters (the empty-state message no longer claims "none installed" when a filter hides rows); (e) locale strings elsewhere still say "My Models" / "Discover" (`chat:composer.placeholderNoModel` is visible in Chat); (f) the old library state in `useModels` / `App.tsx` was pruned but `useVideoState` still returns now-unused fields and `VideoPlaceholderTab.tsx` is dead; (g) 40 `library:models.chat.*` and 16 video keys were translated by a machine pass, German, Japanese, Russian and Traditional Chinese want a native read. **Review 2026-10-29 (three read-only reviewers, one per area):** fixed: the tables now fit the window (below 1440 px the narrow template drops Provider and Tasks on Image and Video, Backend, Compressed and Context on Chat, so nothing scrolls sideways down to a 1200 px window; looked at in the app at 1200 and 1500 px, 1100 px was not run because the app then stacks its sidebar), long names wrap to two lines, the fit chip, the incompatible chip and the shared-storage badge wrap instead of spilling into the next column, the sticky header sticks and the row menu is no longer clipped; a tracked-only image row that has files or a transfer keeps pause, cancel and delete (only an untouched one gets the disabled button); deleting a model drops its finished download record in the backend so the row no longer stays Installed (`delete_model_path`); the row accents for active and failed downloads are back. **2026-10-30:** the actions column of all three tables is now a fixed width and the status column is icon-wide, so spare width goes to the data columns (the gap between status and buttons at 2000 px was about 250 px, now about 60 px; checked at 1200, 1500 and 2000 px, no overflow); the Voice sidebar item reads "Models" and the voice-pack hint says "Voice > Models" in all 10 locales. **Still open:** (h) "Discover" and "My Models" wording remains in about 15 locale strings and two inline defaults (`ChatEmptyStateBanner`, `ImageStudioTab`); (i) the "More actions" menu has no Escape or outside-click handling and the sort headers put `aria-sort` on a button; (j) `groups`, `renderRow` and `accelCompat` are rebuilt on every render so the page `useMemo` never hits (125 chat rows); (k) 1100 px and Windows / Linux layouts not looked at.

</details>

<details>
<summary>FU-128 · LTX-2.3 on its own MLX engine (ltx-2-mlx) replaces Wan 2.2 TI2V 5B on the Mac — Shipped 2026-10-29</summary>

**Status or trigger:** **Shipped 2026-10-29; live-validated on the M4 Max.**

**Notes:** Chosen by the owner's rule of thumb (newer and faster, see FU-125). New engine [ltx2_mlx_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/ltx2_mlx_runtime.py): one `ltx-2-mlx generate --distilled` process per clip from its own venv `~/.chaosengine/ltx2-venv` ([install-ltx2-mlx.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/install-ltx2-mlx.sh), pinned commit `895201d` = v0.18.0 because it is not on PyPI; setup route [ltx2.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/ltx2.py), `chaosengine-cli ltx2-install` / `ltx2-status`, a banner in Video Studio, the installer is staged into the release bundle). Catalog row `dgrauet/ltx-2.3-mlx-q4` in the `ltx-2` family (now its default): the repo holds three transformers and two LoRAs (60 GB), the Download button fetches only the 20.4 GB the distilled pipeline reads (`downloadFiles`) plus the 8.1 GB `mlx-community/gemma-3-12b-it-4bit` text encoder (`downloadCompanions`, now also honoured for video rows). Generation is offline (`HF_HUB_OFFLINE`), reports 8+3 passes, and uses the imageio-ffmpeg binary when the system has no `ffmpeg`. Sizes snap to 64, steps / guidance / negative prompt do not exist for the distilled pipeline, so Video Studio hides them (also CFG decay, preview VAE, diffusion cache, auto-enhance). **Measured (M4 Max 64 GB, through the app, model load included):** 768x512 x 97 frames (4 s, with stereo sound) in **155 s**; 512x320 x 25 frames in 40 s; 768x512 x 25 frames in 50 s with a process footprint of 15.0 GB (sampled every 2 s). The same model through mlx-video took 1130 s for 33 frames, Wan 2.2 5B on diffusers 65-393 s for 33 frames at lower quality; output is a coherent tracked running fox at sunrise with correct motion. **Removed:** the four Wan 2.2 TI2V 5B catalog rows (bf16 + three GGUF), its mlx-video convert entry and its sd.cpp entry, the cached folder; the `wan-2-2` family keeps the A14B rows (default: the fp8 distill). The E2E MPS-burn check now uses Wan 2.1 1.3B. The pipeline-class and default-value tables for the 5B repo (`video_runtime/repos.py`, `defaults.py`) are dead lines left in place. **Also deleted** the unusable `prince-canuma/LTX-2.3-distilled` cache (81 GB, no text encoder). **LTX-2.5 (2026-10-29):** `dgrauet/ltx-2.5-mlx-q4` (gated; bundles its own Gemma 4 text encoder, so no companion repo; the 28.7 GB the distilled pipeline reads) runs through the same engine: **128 s against 155 s for LTX-2.3** on the same prompt and seed (768x512, 97 frames, with sound), 17 GB footprint against 18, clean clip (a fox running through snow at sunrise, natural motion); it is now the `ltx-2` family default. The download first failed with 403 until the Hugging Face token had read access to public gated repos (the app now says so). **Open:** (a) LTX-2.5 is gated (done, see above); LTX-2.5 clips are text-to-video only like 2.3, the pack can also predict the clip length and run image-to-video, neither is exposed; (b) text-to-video only (the video request has no image field, the engine supports image-to-video and sound-to-video); (c) the q8 pack is in `SUPPORTED_REPOS` but has no catalog row and a different 20.6 GB transformer; (d) the peak footprint of a long clip is unmeasured (the catalog says 20 GB); (e) no Windows / Linux path (MLX only; CUDA keeps Wan 2.1 / A14B, LTX-Video and Hunyuan); (f) a small red artefact appeared on the fox in the 4 s clip; (g) mlx-video's LTX rows stay for the dev / quality pipelines; (h) the first install needs `git` for the pinned-commit install.

</details>

<details>
<summary>FU-127 · Settings a model cannot use no longer fail or blank a reply (chat, image, video, voice) — Shipped 2026-10-29 on fix/hardware-gating-ddtree</summary>

**Status or trigger:** **Shipped 2026-10-29 on `fix/hardware-gating-ddtree`; probed live on the M4 Max; open items below.**

**Notes:** **Method:** a battery of 27 hostile requests (repeat penalty 0, XTC with threshold 0, Mirostat, top-p 0, temperature 2, max tokens 1 and 32768, a system prompt and a prompt larger than the context, tools, JSON schema, log-probs, an image sent to a text-only model, a context of 256, a bogus cache strategy, speculative decoding on a model with no drafter) against LFM3 1.7B (MLX and GGUF), Qwen3 4B (MLX), Nemotron 3.5 Bolt and Granite Switch (GGUF); 12 TTS and 15 STT cases against the voice worker. **Found and fixed:** (1) repeat penalty 0 made llama-server answer 400 (`penalty_repeat must be finite and greater than 0`), shown as a 500; (2) XTC with a zero threshold turned the reply into noise on both engines (mlx-lm's own default threshold is 0, so XTC sent without one did the same); (3) an image sent to a text-only GGUF was a 500; (4) a prompt over the loaded context was a raw 500 JSON body; (5) a reply with tokens but no visible text (budget spent on thinking, or a JSON grammar producing whitespace) came back empty with no reason; (6) a lone `</think>` leaked into the visible text; (7) MLX silently ignored Mirostat, DRY, log-probs and a JSON schema; (8) diffusers pipelines were called with every argument whether or not they take it, so a pipeline without a negative prompt or batch argument raised `TypeError`; (9) mflux exits on a size that is not a multiple of 16 (checked with the CLI) and nothing snapped the size, and Wan / LTX frame counts were aligned only on one path each; (10) the FLUX steps cap of 8 applied to FLUX.1 dev and FLUX.2 dev as well as the distilled models; (11) the video runtime computed notes (model-tuned defaults, frame alignment) but only logged them, and the image artifact note could read `None Rendered at ...`; (12) STT with an unsupported language code was a 503, with a language that does not match the audio returned an empty transcript, and an unreadable file was a 503 with an ffmpeg message. **Where:** [request_guards.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/request_guards.py) runs in `RuntimeController.generate` / `stream_generate`, so the chat, OpenAI-compatible, Ollama-compatible and compare routes are all covered; [pipeline_kwargs.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/pipeline_kwargs.py) reads the pipeline call signature before every diffusers image / video call and names what it dropped; [generation_limits.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/generation_limits.py) snaps image sizes to 16, video sizes to the model's multiple (LTX 32) and frame counts to the model's rule (LTX 8k+1, Wan / Hunyuan / CogVideoX 4k+1, Mochi 6k+1), rounding down; every correction lands in the artifact's or reply's `runtimeNote`; voice: [voice_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_runtime.py) retries a rejected or empty-result language with auto-detect and returns a `note` the Transcribe queue shows. 60 new tests ([test_request_guards.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_request_guards.py), [test_voice_library.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_voice_library.py)) and an E2E check (phase 1, unusable settings). **TTS was already sound** (unknown voice, a voice that needs an install, empty text: all clean 400s; emoji-only and punctuation-only text still produce audio). **Second pass 2026-10-29:** (a) a JSON schema that leaves a llama.cpp model with no text is now retried once as plain JSON (the schema goes into the system prompt, a lone ```json fence is removed) and the reply says so, in both the plain and the streaming path ([controller.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/controller.py), [request_guards.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/request_guards.py), 4 tests on a fake engine); (b) a context overflow is a `RequestNotUsable` and the chat and OpenAI-compatible routes answer 400 with both sizes (the streaming route reports it as an error event as before); (d) a few-step model (default steps 8 or fewer) asked for more than twice its tuned steps gets a note with the cost (`distilled_steps_note`), the run is not changed. **Open:** (a2) the retry runs without the Thinking switch, so a reasoning model may spend the second reply thinking too; the JSON from the retry is not validated against the schema; (c) only the MLX and llama.cpp engines were probed, not vLLM, MTPLX, TensorFold or remote providers; (d2) steps / guidance / sampler are still not clamped per model (the diffusers path caps distilled FLUX and `turbo` repos at 8; elsewhere the note above only explains the cost); (e) image and video pipelines other than a tiny SD pipeline were not run with hostile values (the large ones are not downloaded), the kwargs filter and size rules are unit-tested; (f) the backend messages are English only.

</details>

<details>
<summary>FU-126 · AI agents driving ChaosEngineAI (game-dev angle): feasibility — Open: assessment only, nothing built.</summary>

**Status or trigger:** **Open: assessment only, nothing built.**

**Notes:** **Already there:** the FastAPI backend (175 routes) covers chat, image, video, voice (speech to text, text to speech, library), model download and an OpenAI-compatible server, and `scripts/chaosengine-cli` wraps the main ones with JSON output, so an agent that can run a shell can already use it. **Missing for a clean agent story:** an MCP server that exposes those routes as tools with schemas and async job handles (image / video / voice jobs take seconds to minutes and today are polled), a stable asset store an agent can address (outputs live per-feature under `~/.chaosengine`), and auth for non-loopback use. About 1-2 days for a thin MCP layer over the existing routes. **What a Mac Studio or an RTX 4090 can really do locally (Oct 2026):** code and planning with a 27-32B local model (Qwen3.8-27B class, 50-110 tok/s with spec-dec) is usable for scaffolding but behind hosted frontier models on large refactors; concept art, sprites, textures and UI mockups are solid (FLUX.2, Qwen-Image-2.1, Iris-3B); voice-over and transcription are solid (Kokoro, Parakeet, Nemotron ASR); music and SFX have open models (ACE-Step 1.5, Stable Audio Open, MusicGen, AudioGen, HunyuanVideo-Foley) but none is wired in the app; image-to-3D props exist (Hunyuan3D 2.1, TRELLIS.2, TripoSG, and `dgrauet/hunyuan3d-2.1-mlx`, 15 GB, for Apple Silicon) but need a texturing, retopology and rigging pass before an engine can use them and none is wired; short video (5 s clips, minutes each) is fine for trailers and cut-scenes, not for gameplay assets. A 24 GB RTX 4090 holds a 27-32B 4-bit model or one big diffusion model at a time, not both; a 64-128 GB Mac holds several. So: feasible for a solo dev to produce a vertical slice of art, voice, music and prototype code on one machine with an agent in the loop; not feasible to ship a full production game hands-off. The honest differentiator is one local app, one model-memory manager (it already unloads the idle model of the other kind) and one API for all of it. **Banked 2026-10-29** at the owner's call: revisit when a local coding model clearly matches a hosted frontier model (the owner's test: a modern Qwen close to Opus 4.5 would make the coding half possible locally).

</details>

<details>
<summary>FU-124 · Hardware-specific UI hidden where the hardware is absent; Iris-3B needs 768 px — Shipped 2026-10-29 on fix/hardware-gating-ddtree</summary>

**Status or trigger:** **Shipped 2026-10-29 on `fix/hardware-gating-ddtree`; verified on the Mac, not on Windows or Linux.**

**Notes:** **Gating:** [platform.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/utils/platform.ts) `hasNvidiaGpu` = Windows / Linux x86_64 with `gpuVramTotalGb` > 0 (the backend reads it from `nvidia-smi`; `null` means no NVIDIA card, `undefined` means an older backend and keeps the old behaviour), and the `cuda` variant gate now uses it. The Image / Video Studio **Performance boosters** ([MediaStudioBoosters.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/MediaStudioBoosters.tsx), pure rule in `getApplicableAcceleratorsForHost`) no longer show on a Mac or on a PC without an NVIDIA GPU, and never for MLX-native (mflux, mlx-video), stable-diffusion.cpp or LongLive variants, which load no diffusers pipeline (they showed Nunchaku and SageAttention for Qwen-Image-2.1 · mflux on a Mac). The FP8 and NF4 toggles follow `hasNvidiaGpu`; LongLive variants are gated to CUDA hosts; Chat > Discover drops rows the host cannot run (`filterFamiliesForHost`: MLX, MTPLX and TensorFold rows on Windows / Linux, vLLM rows on a Mac) and hides the MTPLX and TensorFold chips off Apple Silicon; the Settings > Accelerators list shows only rows the host can run (it listed every row on every host although its header said otherwise). **Voice Models:** the DOWNLOADED badge overflowed narrow cards; the card head now wraps. **Iris-3B Turbo gibberish:** Iris is pixel-space (no VAE): 256 px renders static, 512 px shows 16-32 px blocks, 768 px is clean (4 and 24 steps gave identical images). The test images the user saw came from E2E checks at 256 and 512 px. Variants can set `minSide`; `_apply_min_side` ([app.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/app.py)) raises a smaller request keeping the aspect ratio and the artifact's runtimeNote says so (Iris: 768). The gallery also recorded the requested 24 steps and CFG 5.5 instead of the effective 4 and 1.0; it now records the effective values. **Open:** (a) not run on Windows or Linux (the filters are unit-tested with their host descriptions); (b) the Image Studio banner still says ENGINE: DIFFUSERS for an mflux variant (fixed 2026-10-29: the banner shows mflux / stable-diffusion.cpp, and the Studio hides the guidance and negative-prompt inputs an mflux checkpoint does not read, from `acceptsGuidance` / `acceptsNegativePrompt` in the variant payload); (c) AMD / Intel GPU hosts get no boosters at all (correct) but also no ROCm / Vulkan-specific hints; (d) only the surfaces found by grep were audited (Studios, Discover, Boost Pack, launch modal already hid engine-incompatible cards); (e) the tiny E2E image sizes were raised by the guard, so the suite now renders Iris at 768 px.

</details>

<details>
<summary>FU-123 · Upstream sweep of 2026-10-29: Nemotron 3.5 Bolt, Granite Switch 4.1, Nemotron 3.5 ASR, Iris-3B, mflux 0.24, llama.cpp b11981 — Shipped 2026-10-29 on feat/upstream-oct29</summary>

**Status or trigger:** **Shipped 2026-10-29 on `feat/upstream-oct29`; measured on the M4 Max; open items below.**

**Notes:** **Nemotron 3.5 Bolt E4B** (NVIDIA, 2026-10-23; Gemma-4-E4B-it post-trained, OpenMDW-1.1): new family `nemotron-3-5-bolt` with the official `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` (Q4_K_M, 6.2 GB) and `ggml-org/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` (Q8_0). llama.cpp b11981 runs it as `gemma4`. NVIDIA publishes the DFlash drafter in a repo of its own under a name with no `dflash-` prefix, so the sidecar scan never found it: [text_gguf.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/text_gguf.py) `PINNED_DRAFTERS` pairs a target with its drafter by exact repo id, `_speculative_args` prefers it, and the download worker fetches it after the model through the new [companions.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/companions.py) (a failed companion fetch only logs a warning). **Measured through the app (llama.cpp b11981, M4 Max, 400-token coding prompt, greedy):** 62-63 tok/s plain, **157-166 tok/s with the drafter** (2.6x); a bare llama-server A/B gave 49-54 against 164-165 (3.1x, 61 % of drafted tokens accepted, block size 8 so the draft is clamped to 7). The two outputs are both good but not byte-identical (batched verification), as on every llama.cpp speculative lane. **Granite Switch 4.1** (IBM, Apache 2.0): family `granite-switch-4-1`, official 8B and 30B Q4_K_M GGUF; `graniteswitch` is in llama.cpp b11938+. Chat works at 29 tok/s (8B, M4 Max); the decision and guardian-core adapters and the speech encoder are not driven by the app. **Nemotron 3.5 ASR streaming 0.6B** (NVIDIA, 40 languages, language-ID prompting): new STT backend `nemotron-asr-mlx` in the voice worker through mlx-audio (`mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit`, 0.72 GB), MLX lane only; ISO codes map to Nemotron's locale keys, `auto` detects the language. Measured on macOS `say` clips: English exact, German auto-detected and exact, **0.14 s for a 5.2 s clip warm** (14.6 s on the first call, model load), word timestamps work. **Iris-3B Turbo** (Sperid Labs, Apache 2.0, pixel-space, no VAE) through mflux: family `iris-3b`; **1024x1024 in 17.5 s, peak 7.5 GB** (4 steps), good output. Its repo holds four checkpoints (42 GB), so image variants may now pin `downloadFiles` ([image_validation.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/image_validation.py)) and list `downloadCompanions` (Iris reads its text encoder from `Qwen/Qwen3-VL-4B-Instruct`, fetched with the download instead of silently on first run). **mflux 0.24.0**: pinned in the installer, `MFLUX_REQUIRED_VERSION` moves with it (an install on 0.23 shows as outdated); the installer dropped its `pillow<13` pin because mflux 0.24 needs Pillow 13 and no longer fails at import on older Python 3.12 patch releases. **Release constraints:** mlx-vlm 0.7.11 (needs mlx-audio >=0.6.2, so mlx-audio 0.6.3) and diffusers 0.42.1; the full pytest suite passes on that stack and Wan 2.2 5B renders a clean clip on diffusers 0.42.1 (46.6 s for 9 frames at 640x352). **llama.cpp b11981** (newest tag with prebuilt assets; b11982 had none) adds LFM3 audio and omni models, video input for LFM3 and a realtime endpoint, none used by the app yet. **Not done, and why:** MOSS-Image-SFT (OpenMOSS, Apache 2.0; mflux supports it and a catalog row was written, then dropped: the 25 GB download ran at about 2 MB/s with 32 GB of disk free, so it was never run here; to add it back: family `moss-image`, repo `OpenMOSS-Team/MOSS-Image-SFT`, command `mflux-generate-moss-image`, model `moss-image-sft`, 50 steps at CFG 3); Nemotron 3.5 Transcribe 30B-A4B (NeMo / SGLang only, 34B parameters, no MLX or llama.cpp path); Bolt 74B-A9B (`nemotron_h_omni`, in no shipped runtime); Bolt MLX (`mlx-community/...-OptiQ-4bit` exists but has no MLX drafter, so it would be slower than the GGUF with its drafter; the 7 GB test download was cancelled); mflux Anima (non-commercial licence, a 39 GB repo for a 4 GB model, three variants in one repo) and FIBO 2 (gated, Bria licence, JSON-caption model); Qwen-Image-2.1 `--small-text-encoder` (the download is keyed by repo, so the 17.5 GB encoder would still be fetched); vllm 0.32 (floor stays 0.28, CUDA-only, not testable here); Muse Spark Open 816B, Reflection Beam 501B, DeepSeek V4.1 Flash 552B (too big); LTX-2.5, MiniMax H3 (gated, no runnable path) and Wan 3.0 (no confirmed open weights). **Open:** (a) mlx-vlm 0.7.11 was not run live on an image prompt; (b) Windows and Linux runs of the Bolt drafter (CUDA llama.cpp) and the Nemotron ASR row (MLX only by design) are untested; (c) Nemotron 3.5 ASR streaming (`create_streaming_session`) is not used, the voice call still takes whole utterances; (d) a mflux venv rebuild deletes the old venv first (2026-10-29: [install-mflux.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/install-mflux.sh) now moves the old venv and its version file aside and puts them back when the new install fails; 3 tests drive the real script with a stand-in interpreter); (e) Bolt `conversational_mode` template switch is not exposed.

</details>

<details>
<summary>FU-122 · Wan 2.2 TI2V still rendered burnt video at the app's default guidance — Shipped 2026-10-28 on fix/wan-default-guidance, stacked on FU-121</summary>

**Status or trigger:** **Shipped 2026-10-28 on `fix/wan-default-guidance`, stacked on FU-121; Wan 2.2 TI2V 5B only.**

**Notes:** Found by downloading `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (32 GB on disk) and running the app's own defaults at a realistic size: 832x480, 33 frames, 20 steps came out over-bright (45-69 % of pixels at 250 or more) with stripes and magenta colour blocks. The E2E Wan check (FU-102) passes because it renders 9 frames at 640x352 and counts clipped pixels, which does not see over-contrast. **Bisected** (M4 Max, 640x352, 17 frames, seed 11, flow-match Euler as FU-102 sets it): the VAE is not the cause (decode in bf16 and fp32 gives identical frames), nor is transformer precision (an fp32 transformer gives the same latents: std 1.39, 45.9 % clipped); latent std and clipped share rise with guidance alone: 1.0 -> 0.88 / 0.1 %, 3.0 -> 0.96 / 6 %, 7.5 at 40 steps -> 1.11 / 17 %, 7.5 at 20 steps -> 1.39 / 45 %. The catalog default `_VIDEO_PIPELINE_DEFAULTS` was 7.5 (the Studio sends the schema default 3.0, which the backend replaces with this value); the model card says 5.0 (`guidance_scale = 5.0`, 50 steps). [repos.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/video_runtime/repos.py) now substitutes 5.0 for Wan 2.2 TI2V 5B. **Measured through the app path** (no guidance argument, second backend on the working tree): the same 832x480 / 33 frame / seed 11 run is a clean running golden retriever on a beach with natural colours, 293 s on the M4 Max (the 7.5 run took 393 s). A test pins the value and fails with 7.5. **Open:** (a) a small strip of pink blocks remains at the top edge from mid-clip onward in the fixed run (frames 16, 24 and 32); not yet tried with 30-50 steps (the card uses 50) or UniPC on CPU; (b) Wan 2.2 T2V A14B is also set to 7.5 (card: 4.0 and 3.0 per expert); Wan 2.1 rows use 6.0 against the card's 5.0 (2026-10-29: now 4.0 and 5.0, from the cards, **not run** here: no A14B weights, and 2.1 only through the E2E check); (c) the E2E check needs a stronger signal and at least 17 frames (2026-10-29: it renders 17 frames and reads the last one: fails above 35 % clipped pixels or mean luma 238, thresholds taken from two real clips only, a smoke guard rather than a quality metric); (d) timings: 9 frames 832x480 = 182 s, 33 frames 640x352 = 345 s, 33 frames 832x480 = 293-393 s through the app, so the catalog estimate of 150 s is low for a 2 s clip.

</details>

<details>
<summary>FU-121 · Tauri 2.12 (closes the zip advisories) and Dependabot triage — Shipped 2026-10-28 on chore/tauri-2-12, stacked on FU-120</summary>

**Status or trigger:** **Shipped 2026-10-28 on `chore/tauri-2-12`, stacked on FU-120; desktop shell not launched.**

**Notes:** `tauri` `~2.11.2` -> `~2.12.0` in [Cargo.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src-tauri/Cargo.toml); lockfile: tauri 2.12.5, tauri-build 2.7.4, tauri-plugin-updater 2.14.1 (zip 4.6.1 -> 9.1.3, so RUSTSEC-2026-0367/0368/0369 are gone), plugin-dialog 2.8.4, plugin-opener 2.7.2, plugin-process 2.4.0, wry 0.57, rust-i18n 4.2.4, serde 1.0.229, libc 0.2.191, flate2 1.1.10, serde_json 1.0.154. npm: `@tauri-apps/api` / `cli` `~2.12.4`, plugin-dialog 2.8.4, plugin-opener 2.7.2, plugin-process 2.4.0, plugin-updater 2.14.1 (versions match the crates; `tauri info` shows no mismatch). The three ignore entries in [dependency-audit-ignore.json](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/dependency-audit-ignore.json) are removed and the audit passes with none accepted. **Dependabot triage (PRs #78-#98):** covered by this change and the sweep: #78, #84, #86, #87, #88, #90, #95, #96, #97, #98 (all Rust, lockfile versions at or above the PR targets), #85 and #94 (docs floors in [requirements-docs.txt](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/requirements-docs.txt), `release-constraints.txt` pymdown-extensions 11.0.2); applied: #80 checkout 7, #81 cache 6, #91 setup-node 7, #92 setup-python 7 (we use none of the removed inputs: `pip-install`, EOL Python versions, the dummy `NODE_AUTH_TOKEN`). **Not applied: #82 tauri-action 1.0.0.** It renames or removes inputs [release.yml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/github/workflows/release.yml) uses (`includeUpdaterJson` -> `uploadUpdaterJson`), changes the `.app.tar.gz` / `.sig` file names (they now carry the version, which `generate-updater-manifest.mjs` and the "Upload updater signatures" step may depend on) and fails when `draft: true` meets a non-draft release; it can only be tested with a real tag build, so it waits for a release dry run. **Tested:** `cargo test` (12), `tsc`, vitest 563/563, `tauri info`, `pip install --dry-run -r requirements-docs.txt -c release-constraints.txt`, audit script PASS. **Not tested:** the desktop shell was not launched on 2.12 (the tool permission for `tauri dev` was denied), no bundle was built, and the Windows updater was not run; Windows `cargo test` runs in CI. **Open:** launch the app once on each platform before the release; migrate tauri-action when a release dry run is possible.

</details>

<details>
<summary>FU-120 · Dependency security sweep and the pre-release audit — Shipped 2026-10-28</summary>

**Status or trigger:** **Shipped 2026-10-28; the Tauri follow-up shipped as FU-121.**

**Notes:** New [scripts/audit-dependencies.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/audit-dependencies.py) (standard library only; needs network): reads `release-constraints.txt`, `src-tauri/Cargo.lock` and `package-lock.json`, asks osv.dev about every pinned version, scores CVSS v3 vectors itself (RustSec and PyPA records have no label), and exits 1 on a high or critical advisory in a shipped package (dev-only npm packages warn), 2 when it cannot reach the network. It also lists packages behind the newest release (never blocks). Accepted risks live in [scripts/dependency-audit-ignore.json](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/dependency-audit-ignore.json): one entry per advisory id with a reason and an expiry date; an expired entry blocks again and a new advisory against the same package is never hidden. Wired into `pre-build-check.sh` / `.mjs` (step 7; no network is a warning there) and into [release.yml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/github/workflows/release.yml) before the draft release is created (fails closed). 39 tests in [test_audit_dependencies.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_audit_dependencies.py); breaking the CVSS arithmetic, the dev-only rule, the ignore matching or the batch-response check each fails one. **First run found:** 41 pinned packages with advisories. **Fixed:** 17 Python pins (certifi 2026.10.19 for the compromised emSign roots, pypdf 6.10.2 -> 6.21.2 with 64 advisories, Pillow 12.2 -> 13.0 with 85, python-multipart 0.0.26 -> 0.0.31, aiohttp 3.13.5 -> 3.14.4 with yarl 1.25.1, urllib3 2.8.0, anyio, idna, msgpack, fsspec, multidict, filelock 3 -> 4, datasets 5, accelerate, plus mkdocs-material and pymdown-extensions), the `pypdf>=6.21.2` floor in pyproject / build.sh / both workflows and the runtime gate in `document_text.py` (older pypdf is refused), Rust (tokio 1.54, rustls 0.23.46, rustls-webpki, webpki-root-certs 1.0.11, plist 1.10.3 with quick-xml 0.42, tauri-plugin-updater 2.12, serde_with, anyhow and others, lockfile only), and npm (react-syntax-highlighter 15 -> 16, vite 7.3.7, vitest 4.1.12, postcss, nanoid, browserslist, source-map-js; `npm audit` is down to 4 low KaTeX findings: rehype-katex 7 pins katex 0.16). The old pins were what release builds installed (`-c release-constraints.txt`). **Checked in a separate environment built from the new constraints** (not the dev `.venv`): `pip check` clean, full pytest, vitest, `tsc`, `cargo test`, wheels exist for all 17 on cp312 macOS arm64 / Windows x64 / Linux x86_64, then live on the M4 Max: E2E phases 0-8, FLUX 3 Klein and Qwen-Image-2.1 Turbo through mflux, a PDF uploaded and queried in chat (multipart + pypdf 6.21.2), code and LaTeX rendering in the chat UI, diffusers image post-processing and the app's mp4 encoder with Pillow 13. **The E2E suite had been skipping most chat checks** because it only looked in `~/AI_Models`; it now also scans the Hugging Face cache (`HF_HUB_CACHE` / `HF_HOME`), sorts candidates smallest first, and ignores test fixtures and drafter / companion GGUFs. **Open:** (a) the `zip` advisories were accepted until 2026-12-31 and removed again in FU-121; (b) katex (low) cannot move while rehype-katex 7 pins 0.16; (c) `torch` 2.11 (low: `torch.jit.script` memory corruption) was left, moving to 2.13+ is a separate MPS test; (d) video generation could not be run: no complete video model is installed (LTX-2.3 distilled lacks the shared 49 GB text encoder from `prince-canuma/LTX-2-distilled`; Wan 2.2 TI2V 5B, 24 GB, is the smallest model the E2E suite exercises); (e) the Dependabot PRs touching the same lockfiles were triaged in FU-121.

</details>

<details>
<summary>FU-119 · Language sweep for v0.9.9: the frontend in all 10 locales — Shipped 2026-10-28</summary>

**Status or trigger:** **Shipped 2026-10-28; remaining gaps listed.**

**Notes:** All 10 locales cover the whole frontend catalog (about 2,840 keys; was 96.4 % of 2,212). Three passes: (1) 79 voice strings that were English-only and 156 strings that existed only as inline `defaultValue` English (TensorFold / MTPLX / GGUF speculative-decoding panels, memory-saver labels, Run from Hugging Face, Import models, RAG badge, Server connect, Dashboard Stop controls, mflux banner, Discover badges), the Voice sidebar group and tabs, Settings > Accelerators / Disk usage; (2) Image / Video Studio presets (aspect ratios, quality, samplers, cache strategies), which called keys no catalog had; (3) about 400 strings that were hardcoded English outside `t()`: the old Discover model window and GPU banner in `App.tsx`, the restart prompt, storage / WSL / accelerator panels, runtime and sampler panels, HTML Challenge, chat sidebar, Image Studio runtime banner, image / video generation windows (progress steps now ICU plurals), download progress and VRAM-fit labels, voice microphone errors and engine descriptions. Transcription language names come from `Intl.DisplayNames` in the UI language. Bugs found: the launch modal's capability badges looked up `capability.*` while the catalog has `modelLaunchModal.capability.*` (English everywhere); the first-launch language banner's "Keep English" button rendered a stray "Success"; four English defaults were built with `"…" + "…"` and the extractor kept only the first piece. Translations are machine-made (helpers per language group), checked for placeholder names, ICU syntax (`i18n:validate`), key parity (vitest), leading spaces in spliced fragments, and spot-read in German and Japanese in the app; no existing translation was changed. German uses the existing `Sprach-` prefix for Voice (Sprachstudio, Sprachmodelle), which can read as "language"; worth a native-speaker check. **Open:** (a) backend messages are not localised (`backend_service/locales/*.po` are empty; e.g. the sidebar's "Idle", runtime notes, error details); (b) the Diagnostics snapshot table stays English on purpose (support copy-out) — its unused `sections` / `fields` keys could be wired if wanted; (c) small leftovers: HTML Challenge `slotSubtitle` and prompt titles / summaries, CompareView "Waiting…", `New Provider` default name, exported role names; (d) `scripts/i18n-extract.mjs` files keys under the wrong namespace (1,060 false "missing" in `common`), so its report is not usable; (e) `importModels.ollama` / `importModels.lmstudio` name other apps in UI strings (2026-10-29: the panel labels each group with the folder it was found in and the two keys are gone from all locales; the API field names `ollama` / `lmstudio`, the `ServerTab` "Ollama-compatible apps" API-protocol label and a few code identifiers still name them — rename or keep is open).

</details>

<details>
<summary>FU-118 · LFM3-1.7B (Liquid AI) on MLX and llama.cpp — MLX shipped 2026-10-27, GGUF 2026-10-28</summary>

**Status or trigger:** **MLX shipped 2026-10-27, GGUF 2026-10-28; both live-validated on the M4 Max.**

**Notes:** Released 2026-10-23 (LFM Open License v1.0): 1.7B, 24 layers (13 gated short convolutions, 5 sliding-window attention with a 256-token window, 6 global attention), 256K context, 16 languages, reasoning on by default. `model_type` `lfm3` is in no released mlx-lm (0.32.2) or mlx-vlm (0.7.10; mlx-vlm main has it since 2026-10-27), so [mlx_models/lfm3.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_models/lfm3.py) ports it onto mlx-lm (adapted from Liquid AI's Apache-2.0 `modeling_lfm3.py` and mlx-vlm's port; reuses mlx-lm's LFM2 `ShortConv`) and loads through the Kolibri `model_file` hook. The official MLX builds are saved in mlx-vlm's layout (`language_model.` prefix), which `sanitize` strips. Full attention rotates only the first quarter of the head (zero-tail inverse frequencies); each attention head has a sigmoid output gate; sandwich norms. Tests in [test_mlx_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_mlx_models.py) check RoPE, the head gate and the decoder layer against NumPy written from the reference, cached decoding against a full pass across the window, and loading an mlx-vlm-layout checkpoint; four deliberate breakages each fail a test. A cross-check against the PyTorch reference was not run (it needs `trust_remote_code` on Hub code). Catalog family `lfm-3`: official `LiquidAI/LFM3-1.7B-MLX-{4,8}bit`. **Live (4-bit, through the backend):** loads in 4 s, 250-280 tok/s, correct English and German answers, Thinking Off honoured on plain chat, calculator tool call runs. **Fixed on the way:** (a) the tokenizer of a vendored model asked "run custom code?" on stdin (the worker's request pipe; LFM3's config names an `auto_map`); vendored loads now pass `tokenizer_config={"trust_remote_code": False}`, which falls back to the plain config. (b) LFM3 (like Qwen3-Coder) writes tool calls as `<function=NAME><parameter=KEY>…`; [agent.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/agent.py) parsed only JSON bodies, so the call showed as text and never ran; XML bodies are now parsed and stripped. (c) after a tool result LFM3 reasons even with thinking off and ends with an unopened `</think>`; the non-streaming MLX path (used by tool turns) now treats text before it as reasoning. **GGUF (2026-10-28):** llama.cpp b11938 is the first release with `lfm3` (PR 31252, merge commit `19cce69`); `LLAMA_CPP_TAG` / `LLAMA_CPP_REF` moved b11810 -> b11938 and the official `LiquidAI/LFM3-1.7B-GGUF` Q4_K_M (1.1 GB) row was added. Live through the backend with the b11938 Mac build: 186-244 tok/s, thinking on and off, German. Running tools on it found that [llama_cpp_engine.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/llama_cpp_engine.py) dropped the structured `tool_calls` llama-server returns (with empty `content`), so tool use on **every** GGUF model returned an empty answer and never ran the tool; it also sent an empty user turn after each tool result and passed the call and result without `tool_calls` / `tool_call_id`. `generate` now returns the calls and `_chat_messages` keeps them structured (shared helpers from `_tensorfold_wire`). Checked live on LFM3 and Flint GGUF; tests in `LlamaCppToolCallTests`. **Open:** (1) other GGUF families' tool calls are not re-run live; (2) the streaming path cannot hide a reasoning block that ends in an unopened `</think>` (seen only after tool results, which do not stream); (3) LFM3-VL-2B, LFM3-Audio-2B and LFM3-Omni-2B are not wired (mlx-vlm main only); (4) the model sometimes repeats a tool call once before answering; (5) delete `lfm3.py` when mlx-lm ships `lfm3`.

</details>

<details>
<summary>FU-117 · Voice I/O rebuilt: voice worker, voice in chat, voice calls, cross-platform — P0 + P1 + P2 + P3 (part) shipped 2026-10-27 on feat/voice-io</summary>

**Status or trigger:** **P0 + P1 + P2 + P3 (part) shipped 2026-10-27 on `feat/voice-io`; live-validated on the M4 Max (MLX lane, and the portable lane run on the same Mac); Windows / Linux not run.**

**Notes:** Plan: [docs/plans/voice-io.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/plans/voice-io.md). **Backend:** [voice_env.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_env.py) builds `~/.chaosengine/voice-venv` from the app's own Python (the packaged interpreter is copied to `~/.chaosengine/voice-python/<ver>` first, because its extraction folder changes with every app version), one job with phases, log and cancel; lanes `mlx` (Parakeet, mlx-whisper, Kokoro via mlx-audio, misaki[en] + `en_core_web_sm`) and `portable` (faster-whisper with `nvidia-cublas/cudnn` wheels when `nvidia-smi` exists, kokoro-onnx); `CHAOSENGINE_VOICE_LANE` / `CHAOSENGINE_VOICE_VENV` force a lane / location. [voice_worker.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_worker.py) runs in that venv (no backend imports), decodes with PyAV (no `ffmpeg` command; parakeet-mlx's `load_audio` is swapped for it), keeps the models loaded; [voice_client.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_client.py) starts it, kills it on timeout, restarts after a crash, stops it after idle minutes; Dashboard Stop knows `voice_worker.py`. Routes are sync `def` (the old `async` transcribe blocked the event loop); 500 MB upload / 20 000-char text limits; `/api/voice/speak/stream` (NDJSON, one WAV per sentence after [voice_text.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_text.py) markdown clean-up); `/api/voice/settings`, `/api/voice/install*`, `/api/voice/worker/stop`. **Permissions:** `com.apple.security.device.audio-input` + `src-tauri/Info.plist` `NSMicrophoneUsageDescription`; Linux turns on WebKitGTK media streams and allows mic requests (`allow_microphone` in [lib.rs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src-tauri/src/lib.rs), compiled only on Linux: CI `cargo check` on ubuntu is its only check); Windows uses WebView2's own prompt. **Frontend:** mic in the chat composer (click / hold / Ctrl+Cmd+Shift+M, Esc cancels, auto-send), speaker on every reply + auto-read + player bar, Voice Studio (record or drop audio/video, editable transcript, timestamps, TXT/SRT/VTT, send to chat; speech with download), Voice Models (voice pack, STT model picker, 54 voices with previews, settings); mic audio is captured raw (AudioWorklet, ScriptProcessor fallback) and sent as 16 kHz WAV. **Measured (M4 Max):** install 3 min / 1.7 GB; warm Kokoro 0.17 s for 6 s of speech; Parakeet 0.09 s for a 6 s clip (first call 1.9 s); portable lane on the same Mac: kokoro-onnx 0.47 s, faster-whisper base 0.6 s. **P2 voice calls:** Silero VAD (`onnx-community/silero-vad` pinned, `onnxruntime` in both lanes; marker `features: ["vad"]`, older packs show "Update voice pack") runs in a second small worker (`get_vad_worker`, own log, 2 min idle) so frames never wait behind TTS; `POST /api/voice/vad` takes base64 PCM16 and returns one probability per 32 ms frame with per-session state. Frontend: [conversation/](https://github.com/cryptopoly/ChaosEngineAI/tree/staging/src/features/voice/conversation/) — `StreamResampler` (no drift on 128-sample worklet blocks), `Endpointer` (threshold 0.5 / 0.35, 160 ms min speech, 400 ms pre-roll, `vadSilenceMs`), `SpeechTextTracker` (sentence pieces outside code fences / think blocks), `useVoiceCall`, `VoiceCallOverlay`; speech player queue mode; `sendMessage(text, {voice: true})` sends Thinking Off + a spoken-style system prompt with `voiceTurn` so the thread's thinking setting is not overwritten. Barge-in only sets the backend stop flag (aborting the stream lost the cut-off reply, and the model then re-answered the old question) and the next turn waits for that stream to end; while the model talks the endpointer needs 0.7 / 256 ms and keeps 1 s of look-back. Measured (M4 Max, LFM2.5-24B-A2B, audio fed through a fake microphone in the browser pane): VAD 2 ms per 128-256 ms batch; end of speech to first audio 1.9 s, of which 1.5 s is the model's own reasoning + answer (LFM2.5 reasons even with Thinking off); barge-in recognised 0.7 s after the user starts. The same work found every chat send waiting ~3 s (re-load + workspace refresh) because saved launch settings lack `speculativeDecoding` / `treeBudget` and a clamped context never matched; fixed with `requestedContextTokens` on the loaded model. **P3 Studio:** transcription takes `task` (translate, Whisper Small / Base only: the Large v3 Turbo fine-tune answers in the spoken language when asked to translate, so its row carries `translate: false`), `words` (word timings from all three backends; Parakeet sub-word tokens are joined at leading spaces) and `speakers` / `speakerCount`; speaker labels run sherpa-onnx (pyannote segmentation 3.0 + 3D-Speaker CAM++, ~37 MB, pinned revisions, both lanes, marker feature `speakers`) in the voice worker, and [voice_transcript.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_transcript.py) gives each word the speaker that overlaps it most and splits segments where the speaker changes. `save=true` moves the upload into the Library ([voice_library.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_library.py), `~/.chaosengine/voice/library/`, replaces the Gallery and migrates its items once); `/api/voice/library` (list with `q` / `kind`, get, update title / text / segments / speaker names, delete, bulk delete, audio with `format=mp3` through the worker's PyAV encoder), `/api/voice/encode`. Frontend [studio/](https://github.com/cryptopoly/ChaosEngineAI/tree/staging/src/features/voice/studio/): queued multi-file transcription, transcript editor (click a time or word to seek, the playing word highlighted, double-click to edit a line, speaker names, autosave, TXT / MD / SRT / VTT / JSON, hand-off prompts into the chat composer), Speak (paragraphs and `[pause]` synthesized one by one with progress and Cancel, joined in the page, voice preview, WAV / MP3, save to Library); Voice Library page. Measured (M4 Max): pack update 20 s; a 16 s two-speaker clip with labels in 2.1 s (Parakeet), both speakers right; 8 s of speech in 3 parts; MP3 export 150 KB. **Open:** (a) dictation and calls with a real microphone on speakers (echo) not tried, and the packaged-app microphone prompt; (b) Windows (WebView2, CUDA libs) and Linux (WebKitGTK permission code, LD_LIBRARY_PATH for CUDA) not run; (c) Japanese voices on MLX need misaki[ja] + the full UniDic download (0.8 GB), not offered yet; (d) P3 not done: dialogue scripts (Dia2) and voice cloning (Chatterbox), both need a licence and speed check; the transcribe queue lives in the Studio page (leaving the page stops waiting files; one already running finishes and is saved); no in-file progress for long files; speaker labels cap at the clustering threshold (0.5) with no tuning UI; the CAM++ embedding file's licence is taken from the 3D-Speaker project (Apache-2.0) because the Hugging Face mirror has no card; P4 OpenAI endpoints / CLI; (e) call strings are machine-translated (P1 strings still show English in other locales); a barge-in keeps the whole generated reply in the thread, not only the part that was spoken; no streaming partial captions for the user's side; (f) the voice venv pulls torch (via mlx-whisper), which is most of its size.

</details>

<details>
<summary>FU-116 · Route architectures only mlx-vlm ships through mlx_vlm.load — Shipped 2026-10-22</summary>

**Status or trigger:** **Shipped 2026-10-22; selection unit-tested, no live model of these types fits this Mac.**

**Notes:** [mlx_models](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_models/__init__.py) `needs_vlm_runtime`: `config.json` `model_type` in {bonsai, deepseek_v4, flint, glm5_next, muse_glimmer, muse_spark, prism_hadamard_qwen35, qwen4_exp}, absent from the installed mlx-lm and present in mlx-vlm. The worker then loads with mlx-vlm and the generate paths take the multimodal branch; `multimodal` in the worker's info (and so `visionEnabled`) now follows a `vision_config` in the checkpoint, not the loader, so text-only models do not offer Attach image. On the release stack (mlx-lm 0.32.2) only `qwen4_exp`, `glm5_next`, `deepseek_v4`, `bonsai`, `flint` need it; mlx-lm 0.32 ships the others. mlx-vlm pin 0.7.9 → 0.7.10. Checked: the real Flint INT4 reaches `mlx_vlm.load` (then fails on the embedding layout, FU-115); Gemma 4 still reports vision; Ternary Bonsai 2 loads (via mlx-lm) without it. **Open:** TensorFold-exclusive families still say "install TensorFold" instead of falling back to this lane; DFlash, prompt-cache reuse and the reasoning split do not apply on the vlm path; no end-to-end run of a vlm-only checkpoint (Qwen3.8 Flash Next and GLM-5.3 Flash are 100+ GB).

</details>

<details>
<summary>FU-115 · Flint (Recursive, 39.8B MoE) via llama.cpp — Shipped 2026-10-22</summary>

**Status or trigger:** **Shipped 2026-10-22; live-validated on the M4 Max.**

**Notes:** `flint` architecture merged in llama.cpp 2026-10-21 (PR 30928), so the pin moved b11786 → b11810. Catalog family `flint`: official `RecursiveSI/flint-int4-gguf` (Q4_1, 26 GB) and `bartowski/RecursiveSI_flint-GGUF` Q4_K_M (one pin per repo). Run through llama-server b11810, 32K context, Metal: 112 tok/s decode, 83 tok/s prefill, separate `reasoning_content`, a calculator tool call parsed. **Open:** no MLX row: mlx-lm has no `flint` (ml-explore/mlx-lm#1998 open) and mlx-vlm 0.7.10 loads only the BF16 checkpoint — the official MLX INT4 fails at load (`Expected shape (150144, 256) but received shape (150144, 2048)` for `embed_tokens`: its int8 embeddings use a layout only Recursive's `rec` engine reads; `rec` is Go, Apple-only, M4 Pro or newer, claims 1.4-2.6x llama.cpp, unmeasured here); no CUDA run; BF16 needs Recursive's SGLang kit. FLUX 3 Klein (mflux) also verified: 83 s at 1024², peak 23.5 GB. Gated repos need a token with gated-read permission.

</details>

<details>
<summary>FU-112 · MLX speculative decoding repaired; MLX DDTree tree verification fixed and re-enabled — DFlash shipped 2026-10-21</summary>

**Status or trigger:** **DFlash shipped 2026-10-21; DDTree re-enabled 2026-10-29.**

**Notes:** **DFlash:** dflash-mlx 0.1.5 made `runtime_context` a required argument of `stream_dflash_generate`, so every MLX DFlash reply raised and fell back to standard decoding while the load note still said DFlash was active (FU-075 checked only the note). [mlx_worker_speculative.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_speculative.py) builds `build_offline_runtime_context(target_fa_window=0)` like upstream's own `generate`. Measured on the M4 Max, Qwen3-4B bf16 with `z-lab/Qwen3-4B-DFlash-b16`: **113 tok/s against 53 plain (138 on mlx 0.32)**, identical output, 6.7 tokens accepted per step. A contract test checks the worker passes every required argument of the installed `stream_dflash_generate_impl`. The DFlash and DDTree paths also ignored the Thinking switch and dropped the reasoning they split; both now match the standard path. **Thinking Off:** the MLX worker never set `enable_thinking`, so Qwen3 / 3.5 / 3.6 / 3.8 reasoned with Thinking Off; `chat_template_switches` now passes it. **DDTree (2026-10-29):** the old port verified tree nodes at consecutive RoPE positions (flat index, not depth), so it produced garbage once its crashes were fixed. New [ddtree_attention.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/ddtree_attention.py) runs one verify pass over the whole tree without forking mlx-lm: each attention layer's `rope` is swapped for the duration of the pass for one that rotates every node at `prefix + depth` (mlx's per-batch rope offset, tokens moved onto the batch axis; a call count proves every layer went through it), an explicit ancestor mask is passed, and afterwards the KV cache is compacted to the accepted path (accepted nodes sit at depths 0..n-1, so a gather needs no re-rotation). [ddtree.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/ddtree.py) loop rewritten: chunked prefill with last-position logits only, top-k on the GPU, budget = tokens verified per step including the root, draft depth capped at the node count, budget 0 = the top-1 chain on the same machinery. Only targets built from attention layers over a plain `KVCache` qualify (`ddtree_unsupported_reason`); hybrid GDN (Qwen3.5 / 3.6, Qwen3-Next) and rotating-cache (Gemma 4) targets drop the budget at load with a note, and the new `requestedTreeBudget` on the loaded model keeps that from reloading the model every turn. `availability_info` reports `is_ddtree_available()` again, so the launch slider is back. **Correctness:** with fp32 activations the output equals plain greedy for 300 tokens on 4 prompts, Qwen3-4B bf16 at budgets 0/4/8/16/64 and 4-bit at 0/8/16/32/64; in bf16 every first divergence (upstream linear DFlash shows the same) is a near-tie within one bf16 ulp, often an exact tie. [test_ddtree_attention.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_ddtree_attention.py) (tiny random Qwen3 on real MLX, no weights) compares every node with an unbatched forward over its path and the loop with plain greedy; flat positions, a causal mask, a rope that ignores the tree positions and skipping the cache gather each make it fail. **Speed** (M4 Max, Qwen3-4B + `z-lab/Qwen3-4B-DFlash-b16`, 256 tokens, 3 interleaved runs, decode tok/s, code / explain / story / think prompts): bf16 plain 54 / 53 / 50 / 51; linear DFlash (16-token block, as shipped) 134 / 56 / 34 / 50 at 6.5 / 2.4 / 1.2 / 2.1 accepted; DDTree budget 8: 180 / 104 / 65 / 81 (4.4 / 2.5 / 1.3 / 1.8 accepted); budgets 16 - 64 ran 117 - 130 on code and 35 - 60 on prose. 4-bit: plain 162 / 144 / 104 / 115, linear DFlash 159 / 73 / 40 / 54, DDTree 8 197 / 114 / 63 / 84. Most of the gain comes from verifying 6 - 8 rows instead of 16 (a verify pass costs roughly in proportion to its rows here): linear DFlash with an 8-token block reaches 193 / 103 / 53 / 75, and a tree beats a chain of the same width by about 2 % on average (0 - 15 % per prompt). With the 4-bit target every speculative mode loses to plain on prose (drafter acceptance 1.1 - 2.4). **Open:** a shorter default linear DFlash block (8) on MLX would give most of this speed without a tree budget; the frontend's `loadedModelMatchesRuntimeProfile` ([chatRuntime.ts](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/utils/chatRuntime.ts)) still compares `treeBudget` (fixed 2026-10-29: it compares `requestedTreeBudget ?? treeBudget`, with a vitest case); **re-measured by the owner's session on a quiet machine, same pattern:** DDTree 8 on Qwen3-4B bf16 gives 112 / 62 / 81 tok/s on explain / story / think prompts against 52 / 50 / 48 plain, and linear DFlash with the shipped 16-token block is slower than plain on prose (34 tok/s on the story prompt), faster only on code; **27B-class check (2026-10-29, M4 Max, `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` + `z-lab/Qwen3-Coder-30B-A3B-DFlash`, 256 tokens, 3 interleaved runs, decode tok/s, code / explain / story / think; the only attention-only target of that size with a drafter, since Qwen3.6-27B is hybrid and Gemma 4 31B uses a rotating cache):** plain 101 / 96 / 84 / 83 (mean 91); linear DFlash block 16 (old default) 110 / 59 / 21 / 108 (mean 74); **linear DFlash block 8 (new default) 146 / 91 / 29 / 138 (mean 101)**; DDTree 8 137 / 97 / 37 / 132 (mean 101); DDTree 16 94 / 68 / 26 / 131 (mean 80). So block 8 is right on the big target too (block 16 loses 18 % to plain on average), and the tree adds nothing over a chain of the same width here (101 against 101). Every divergence from plain output is a near-tie of the target's own top two logits (0 to 0.25 apart, several exactly equal). **New finding:** on the story prompt every speculative mode is 2 to 4 times slower than plain (drafter acceptance 0.4 to 0.8 tokens per step; 21 to 37 tok/s against 84) — see FU-134; on the hybrid Qwen3.6-27B (after FU-135) block 8 also beats block 16 (mean 20.6 against 14.4 tok/s, plain 19.2); not run in the app UI; dflash-mlx's own DDTree verify (v0.1.6+, FU-057) not compared.

</details>

<details>
<summary>FU-111 · TensorFold 1.x (native Zig binary) — CUDA lane shipped (Windows + Linux)</summary>

**Status or trigger:** **CUDA lane shipped (Windows + Linux); Mac moved to the same native lane 2026-11-11 (Python 0.6.6 kept as a fallback for three families).**

**Notes:** TensorFold 1.0 (2026-10) replaced the Python engine with a native binary (Homebrew `ashhart/tensorfold/tensorfold` or a release archive; `tensorfold-native serve`), and 1.1.1 adds CUDA families. Its qualified set differs from the Python engine's: it adds LFM2.5-24B-A2B, Qwen3.5-2B, GLM-5.3 (EXLR8) and Nemotron Super VL (CUDA), but Gemma 4, Bonsai 2, Qwen3.6 and DeepSeek-V4 are "still under qualification", and Qwen3.8-27B decodes slower than the Python engine without its drafter. Moving needs a new installer (archive + sha256, no Python venv), new serve flags (`--drafter`, `--vision`, `--context`), the metrics / capabilities JSON, and a registry built from `capabilities --json`. The app stays on the Python engine's last release, v0.6.6 (`cb2ebf0`), until 1.x covers the families we ship. **Evaluated 2026-11-11 on the RTX 4090** ([findings](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/progress/rtx4090-progress.md)): release 1.2.6 has no Windows asset (Linux x86_64 / aarch64 and macOS arm64 only), and the Linux archive's CUDA images are sm_80, 86, 120 and 121, so an sm_89 card needs a source build (`zig build -Dgpu=sm_89 -Dnvcc=...`, Zig 0.17.0, nvcc, MSVC; `docs/WINDOWS.md` still names a removed `native` step). That build served Qwen3.5-2B 4-bit at 457 to 461 tok/s, 1.58 GiB model, loaded in 1.5 s, driver 621.14 accepted (floor R580). 27B + DFlash2 also fits (18.8 GB on the card, idle-card rerun): 47 to 49 tok/s plain, 147 to 151 with the drafter on prose and about 300 on code; the first start stalled 259 s on kernel first-use, so allow a long timeout or warm up. Bolt and a like-for-like llama.cpp 27B run not done. **Design (not built):** (1) installer = release archive + `.sha256` per platform (Linux x86_64 / aarch64, macOS arm64), unpacked into `~/.chaosengine/tensorfold/<version>/`, no Python venv; Windows and any NVIDIA card outside sm_80 / 86 / 120 / 121 get a "build from source" row (or our own pinned build artefact, sha256 recorded) instead of a one-click install; (2) capability gate = compute capability read from `nvidia-smi`, driver at least R580, platform has an archive, and `capabilities --json` `chip` / `backends` / `families`; sm_89 shows "unqualified, may work" until a served run is recorded; (3) registry built from `tensorfold-native capabilities --json` (`families` by weight format) joined with our catalog, never hard-coded; (4) engine spawns `tensorfold-native serve <path> --name ... --context N [--drafter DIR] [--vision] --parallel N --no-thinking` on a free port and talks OpenAI-compatible `/v1`, metrics from its log lines; (5) keep the Python 0.6.6 engine for the Mac until 1.x covers our families. Keep Apple Silicon on 0.6.6 meanwhile. **Built 2026-11-11 (test/rtx4090):** `_tensorfold_native.py` (platform keys, release assets, GPU + `capabilities --json` gate, registry, VRAM estimates; 27B = 18.8 GB measured), `tensorfold_native_engine.py` (subclass of the Mac engine; serve flags only when the binary lists them), `routes/setup/tensorfold_native.py` (download, sha256, safe unpack, adopt a source build; no bash, no venv; the Setup/launch Install button and `GET /api/setup/tensorfold-native-status` use it off the Mac), controller routing (`backend=tensorfold`, or auto for a registry repo when native is installed and the Mac engine is not), `tests/test_tensorfold_native.py` (43), E2E check in phase 1. **Live on the RTX 4090 through the app:** 27B + DFlash2 loaded in 46 s, 49.6 tok/s plain, 132 on prose and 206 on code drafted (sampling not forced greedy, so below the 147-151 / 300 greedy runs). **Gaps:** no Windows release archive exists (the Windows install adopts a source build or a self-hosted archive named by `CHAOSENGINE_TENSORFOLD_WINDOWS_URL` + `_SHA256`); the Linux archive path is unit-tested only (sm_89 has no image in it, the gate says so); `tfgrammar` needs the archive's `lib/` (structured output refuses without it); Gemma 4, Qwen3.6-MoE, LFM2.5 not in the registry yet (capabilities lists them; add rows once a served run is recorded); Mac keeps 0.6.6 until 1.x covers its families. **Mac (2026-11-11, M4 Max):** the native lane now serves Apple Silicon: the same installer (`macos-arm64` archive, sha256, no Python) and `TensorFoldNativeEngine`, selected by `RuntimeController._tensorfold_engine`. The old Python install (0.6.6) is used only for families flagged `native=False` in `_tensorfold.py` (Gemma 4 26B-A4B, which 1.x refuses; DeepSeek-V4-Flash and Ternary Bonsai 2, absent from the Metal `capabilities --json`), and a Mac can still opt into it with `{"legacy": true}` on the install route. Measured through the real app: Qwen3.8-27B + DFlash2 96.7 tok/s drafted against 24.6 on standard MLX (500 tokens, temperature 0), same replies. Fixed on the way: the native installer failed on the python.org Python with `CERTIFICATE_VERIFY_FAILED`; it now uses the app's certifi context (`helpers/hub_http.ssl_context`). Not done: the Mac family table still follows the old docs (FU-168). **2026-11-12 (RTX 4090):** `tfgrammar.dll` builds on Windows with `zig build grammar -Dxgrammar=DIR` (xgrammar v0.2.8, 32 s), adopt copies it, structured output verified through the app; Gemma 4 31B (42 tok/s, 21.8 GB, window max 12K on 24 GB) and Qwen3.6-35B-A3B + MTP (186 tok/s, 420 on code, 20.1 GB, 32K fits) measured, rows set `measured`.

</details>

<details>
<summary>FU-110 · Memory and speed estimates (chat / image / video) rebuilt — Shipped 2026-10-21 on fix/ram-estimates</summary>

**Status or trigger:** **Shipped 2026-10-21 on `fix/ram-estimates`; open items below.**

**Notes:** **Pressure:** [memory_gate.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/memory_gate.py) pressure is now the share of memory not available (macOS-corrected), without swap; the old (used + compressed + swap) formula read 100% on a Mac the OS rated normal and refused every chat turn with a 33 GB model loaded. **Chat cache:** [model_arch.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/model_arch.py) reads the real attention layout from `config.json`, the GGUF header, or for models not on disk the Hub config (a GGUF repo falls back to its card's `base_model`): full vs sliding layers and window, per-layer KV heads, true head dims, MLA latent width, and the MoE active-weight share. Kolibri 1 was 15x too large, Qwen3.8-27B 3.7x, Bonsai 1.7B 2.8x (plus a stale 262K context on a 32K model). The launch modal now follows its selected model, including Discover's catalog keys. **Speed:** this machine's median of recent chat / benchmark runs for the model and strategy, else memory bandwidth ([bandwidth.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/bandwidth.py): Apple chips incl. binned GPUs, NVIDIA cards) / bytes per token + a fixed per-token cost; fitted on eight models on an M4 Max, within about 20%. **Load check:** `/api/models/memory-check` (advisory; the modal asks before a load that may not fit, never refuses). **Images / video:** estimates know the mflux and sd.cpp engines, count a loaded chat model (`residentModelGb` in the runtime status), and video refuses a non-resident model larger than free memory; both gates run after the idle model of the other kind is unloaded. **Warm pool:** loaded models were counted twice against free memory. **Downloads:** shard files no index references are skipped (38 GB on LTX-2.3 distilled). **LTX-2:** measured 37.2 GB (mlx-video) / 40.0 GB (process) peak and 1130 s at 768x512x33 frames x 8 steps on LTX-2.3 distilled; the catalog had 27 GB and 100 s. **Open:** LTX-2 / 2.3 dev times (600 s) are unmeasured and surely low; MoE speed carries a fitted 6 ms per-token cost from two models; the Kolibri "measured" speed is 7.5 tok/s from runs made while the Mac was swapping and will move as it is used; the sd.cpp image footprints are file sums, not measured peaks; CUDA bandwidth figures and the CUDA side of the estimates are untested.

</details>

<details>
<summary>FU-103 · Kolibri 1 (Aleph Alpha, 78B MoE, German + English) on Apple Silicon — Shipped 2026-10-05</summary>

**Status or trigger:** **Shipped 2026-10-05; live-validated with the 3-bit build, speed unsettled.**

**Notes:** Released 2026-10-03 (Apache 2.0; `Aleph-Alpha/Kolibri-1` FP8 and `-BF16`): 50 MoE layers, 384 experts with 6 routed + 1 shared, 3.46B of 78B active, 4:1 sliding-window to global attention, 262K native / 1M validated context, German 23.9 % of 20T tokens, reasoning effort none / low / medium / high through the chat template, Hermes-style tool calls. `model_type` `kolibri1` is in no released mlx-lm (0.31.3 installed, 0.32.0 on PyPI; ml-explore/mlx-lm#1945 is open with no reviews), so [mlx_models](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_models/__init__.py) vendors the port (`kolibri1.py`, unmodified, compared line by line with Aleph Alpha's reference vLLM plugin) and `vendored_model_config` passes it to `mlx_lm.load` through its `model_file` hook — only when the checkpoint's `model_type` is vendored and mlx-lm does not ship it, so every other model loads as before ([mlx_worker_lifecycle.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_lifecycle.py)). [chat_template_switches](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_prompt.py) sends `reasoning_effort` to templates that define it: the composer's Low / Med / High, and `none` for Thinking Off where the template has a none level (the Kolibri template reasons at high effort by default). Catalog family `kolibri-1`: community MLX conversions `velaia/Kolibri-1-MLX-{4,3,2}bit` (44.3 / 34.9 / 25.5 GB; peak memory 48 / 39 / 30 GB at a 4K prompt, from the converter; this Mac's Metal working-set limit is 51.8 GB of 64). 18 tests in [test_mlx_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_mlx_models.py) on a tiny random Kolibri on real MLX: cached decoding equals a full forward pass across the sliding window, the router matches a NumPy reference (pick on the biased scores, weigh with the unbiased sigmoid), the checkpoint loads through the hook; breaking the router weights or the window mask makes them fail. E2E: a catalog gate and a live check that skips without weights. **Live run (velaia 3-bit, 33 GB, M4 Max 64 GB, stock mlx-lm 0.31.3, through the backend):** loads in 12 s via the hook; correct German answers with Thinking Off (no reasoning tokens) and with low / medium / high effort; a calculator tool call runs and is answered (after FU-105). **Speed is unsettled:** one 93-token run did 32 tok/s, then 2-7 tok/s sustained; plain mlx-lm outside the app gave the same 2-3 tok/s with 12 GB of swap in use and other jobs on the Mac, so it is memory pressure on the box, not this code — re-measure on a quiet machine; the 4-bit was not run. **Open:** (a) a clean tok/s number and a 4-bit run; (b) no GGUF row: llama.cpp has no `kolibri1` (the community GGUFs need a patch on commit 836d571); (c) no CUDA row: vLLM needs Aleph Alpha's plugin (vllm#60026 is open) and about 78 GB of GPU memory; (d) TensorFold v0.6.5 has no kolibri family; (e) the sliding-window layers' `RotatingKVCache` cannot be trimmed once it wraps, so chat history is prefilled again every turn; (f) delete `kolibri1.py` and the hook when mlx-lm ships the architecture; (g) the MLX builds are one community account's conversions — the code in those repos was not run, only our copy.

</details>

<details>
<summary>FU-096 · Qwen-Image-2.1 integration — sd.cpp lane (2026-10-01) + mflux lane (2026-10-02) shipped; diffusers lane open — Both lanes live-validated on the M4 Max</summary>

**Status or trigger:** **Both lanes live-validated on the M4 Max; diffusers lane blocked upstream.**

**Notes:** Variant `leejet/Qwen-Image-2.1-GGUF-q4k` in the `qwen-image` family ([image_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/image_models.py)) routes through [sdcpp_image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/sdcpp_image_runtime.py), which now takes companion files and extra arguments from the catalog (`sdcppAux` = `{--vae, --llm}`, `sdcppArgs`; resolved from the HF cache, downloaded on first use). The variant's repo is the GGUF repo, so the Download button fetches only the 4.2 GB GGUF (allow-list + "installed" check in [image_validation.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/image_validation.py) / [images.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/images.py)), not the 33 GB base snapshot; the Qwen3-VL-8B Q4_K_M encoder (5 GB) and the VAE (0.7 GB) arrive on the first run. `sd-cli` had to be rebuilt (master-929; the May build predates Qwen-Image-2.1). Live: 1024² / 40 steps = 525 s on Metal (**~13 s/step, same with and without `--offload-to-cpu`**), 512² / 10 steps = 49 s through the app; output is good. Cross-platform: the same code path runs on CUDA through the existing `build-sdcpp.sh` cuBLAS branch — **not run on Windows / Linux yet**. **mflux lane (2026-10-02):** mflux 0.20 pins mlx >=0.32 / torch >=2.13 / pillow >=12.3 (the app runs mlx 0.31), so it lives in its own venv (`~/.chaosengine/mflux-venv`, 1.3 GB) built by [install-mflux.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/install-mflux.sh) through [routes/setup/mflux.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/mflux.py) (`/api/setup/{mflux-status,install-mflux,install-mflux/status}`, `chaosengine-cli mflux-install|mflux-status`, an "Install mflux engine" banner in Image Studio). [mflux_engine.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime/mflux_engine.py) was rewritten from the dead in-process `mflux.Flux1.from_name` wrapper to a subprocess engine: one `mflux-generate-*` command per image (family table: FLUX.1 schnell / dev, FLUX.2 klein 4B, Qwen-Image-2.1), tqdm progress into `IMAGE_PROGRESS`, cancellable, weights freed on exit. Variants `Qwen/Qwen-Image-2.1-mflux` (bf16) and `-mflux-q8`, `black-forest-labs/FLUX.2-klein-4B-mflux`. **Measured on the M4 Max:** Qwen-Image-2.1 1024² / 40 steps = **375 s** through the app (cold) at ~7.5 s/step steady, peak 28.7-29.9 GB — vs 525 s on sd.cpp and far from upstream's M5 Max figure (~1.5 s/step; the M5's GPU matrix units are what close that gap); FLUX.2 klein 4B 1024² = 26 s (diffusers MPS: 21 s). **mflux 0.23 (2026-10-21):** Qwen-Image-2.1 Turbo (8 steps, no CFG) renders 1024² in **95 s** on the M4 Max (base model 40 steps: 375 s); FLUX 3 Image Klein (distilled + base, gated FLUX Kommunity licence) wired but not run here. Peak physical footprint read with `vmmap` during a run was ~50 GB for both Qwen-Image-2.1 and Turbo, against 29-30 GB measured earlier (likely steady state versus a load-time transient); the catalog keeps 30 GB until that is settled. `setup` no longer lets the Setup page `pip install mflux` into the extras overlay (it would have replaced mlx). A repo with no diffusers pipeline (`_NO_DIFFUSERS_PIPELINE`) now raises its engine's error instead of falling back to the placeholder picture. **Open:** (a) mflux families beyond the four in the table (Z-Image-Turbo, Krea 2 gated, ERNIE-Image, Ming-Image, Lens, Ideogram 4, FIBO) are one table row each plus a catalog variant; the 8-bit variant is unmeasured; reference-image editing and RGBA output are not exposed; (b) **diffusers lane** — `QwenImage21Pipeline` is only on diffusers git main (PyPI latest 0.40.0, 2026-08-20) and needs `transformers>=5.17`; add a `Qwen/Qwen-Image-2.1` variant + pipeline-class mapping when 0.41 ships; (c) a Q8_0 variant (7.7 GB) needs per-variant downloads — the download API is keyed by repo, so both quants would be fetched; (d) the companion-file download is invisible to the progress UI; (e) the older sd.cpp FLUX variants passed no `--vae` / `--clip_l` / `--t5xxl` — fixed 2026-10-05 (FU-099; they also pointed at a GGUF file that does not exist; SD3.5 has no sd.cpp catalog row); (f) editing / RGBA / reference images are not exposed (text-to-image only); (g) **licence:** Qwen Research is non-commercial — the variant note says so, consider an acceptance prompt before download.

</details>

<details>
<summary>FU-094 · TriAttention pin bump (shipped) + MLX path is inert (open) — Pin shipped 2026-09-30</summary>

**Status or trigger:** **Pin shipped 2026-09-30; the MLX finding is open.**

**Notes:** Pin moved `c3744ee` → `a4bc3c8` in both `pyproject.toml` extras and `_INSTALLABLE_PIP_PACKAGES`; [pre-build-check](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/pre-build-check.mjs) (`.mjs` + `.sh`) now fails if the three drift. The old pin crashed on models whose config has no `head_dim` (`args.head_dim is None`); the new pin defaults it. **Finding:** on Apple Silicon the strategy does nothing. `apply_triattention_mlx` only attaches `model._triattention_mlx`; the upstream port expects the caller's decode loop to call `triattention_generate_step`, and mlx-lm's generation never does. A tiny-model smoke at both pins produced identical tokens and identical KV offsets. `apply_triattention_mlx_compressor` ([mlx_worker_lifecycle.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/mlx_worker_lifecycle.py)) now says so in the runtime note instead of claiming "applied", and the launch modal already hides TriAttention on MLX (engine-incompat, FU-034). **Open:** either drive `triattention_generate_step` from the MLX worker's generation loop, or drop the MLX path. If wired, upstream ships a calibrated stats file for Qwen3.6-27B (`triattention/calibration/qwen3.6_27b_stats.npz`); without stats it runs norm-only scoring. The vLLM path is untouched and was not exercised here (CUDA only).

</details>

<details>
<summary>FU-093 · TensorFold engine lane (exact speculative decoding; Apple Silicon) — foundation shipped, awaiting live validation — Shipped 2026-09-30</summary>

**Status or trigger:** **Shipped 2026-09-30; live-validate on an Apple Silicon box.**

**Notes:** Modelled on MTPLX. [`TensorFoldEngine`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/tensorfold_engine.py) spawns `tensorfold serve <snapshot dir>` from an isolated `~/.chaosengine/tensorfold-venv` (TensorFold pins its own `mlx >=0.32.2,<0.32.4` + `mlx-lm >=0.31.3,<0.32`) and proxies `/v1/chat/completions` (reasoning + tool-call channels, `tensorfold{…}` / `speculative{…}` metrics, reply-budget refusals). Registry + routing in [_tensorfold.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_tensorfold.py) / `RuntimeController._select_tensorfold`: explicit `tensorfold` backend always; **TensorFold-only** families (`qwen4_exp`, `glm5_next`, `deepseek_v4`, `prism_hadamard_qwen35` — stock mlx-lm cannot load them) always, with an install prompt when missing; **tested** checkpoints (exact repo ids) only when installed AND speculative decoding is requested, outranking MTPLX / DFlash. Startup failure falls back to standard MLX except for TensorFold-only families. Installer [scripts/install-tensorfold.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/install-tensorfold.sh) + `routes/setup/tensorfold.py` (background job, `PHASE:` markers, `tensorfold-install` / `tensorfold-status` CLI), capability probes (`tensorfoldAvailable` / `Version` / `Models`), launch-modal control ([TensorfoldLaunchControl.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/TensorfoldLaunchControl.tsx)), routing badge ("Exact drafts"), catalog rows (Qwen3.8 27B, Qwen3.8 Flash Next, Nemotron 3.5 Lightning, Gemma 4 26B-A4B, GLM-5.3-Flash, Ternary Bonsai 2), E2E phases 0/1/6 + matrix cell, docs at [docs/features/tensorfold.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/features/tensorfold.md). **Validation so far:** the engine was driven against TensorFold's real HTTP handler on Linux (with a stub model) and a stub-server suite; nothing ran on real Apple Silicon, and the authoring sandbox could not reach Hugging Face or GitHub's archive endpoint. **Open:** (a) live run — install → load `Vontra/Qwen3.8-27B-MLX-4bit` → stream → check `engine == "tensorfold"` and acceptance; (b) verify the catalog repo ids + sizes against the Hub (`Vontra/*`, `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/DeepSeek-V4-Flash-4bit`) — names come from TensorFold's docs, sizes are estimates; still blocked in the authoring sandbox, whose network policy returns 403 to `huggingface.co` and GitHub's archive host (an environment setting: allow-list `huggingface.co`, `*.hf.co` and `codeload.github.com`); (c) **image input + structured output shipped 2026-10-01 as opt-in extras** (see below) — they are unvalidated on a Mac; (d) `--ssd-experts` / `--ple-on-ssd` (the `[ssd]` extra) could let smaller Macs run the big MoEs; (e) Qwen3.6-35B-A3B is CUDA-only upstream and deliberately absent; (f) the MTPLX env-inheritance hazard `_isolated_child_env` fixes here (the packaged app exports `PYTHONHOME` / `PYTHONPATH` / `DYLD_*`, [runtime.rs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src-tauri/src/runtime.rs)) **was fixed for MTPLX on 2026-10-01**: `MtplxEngine`, the MTPLX installer route and `install-mtplx.sh` now scrub the same variables (MTPLX itself is macOS-only, but the behaviour under test is plain subprocess env, covered against the existing stub server in `test_mtplx_engine_integration.py`; the unfixed child crashed on a bogus `PYTHONHOME`); (g) when `TENSORFOLD_REF` moves, re-read its `families/*/__init__.py` (`MODEL_TYPES` / `MODELS` / `DRAFTER`) and update the registry in lockstep. **Optional extras (2026-10-01):** `scripts/install-tensorfold.sh` takes `TENSORFOLD_EXTRAS=vision,grammar` (`POST /api/setup/install-tensorfold {extras}`, `chaosengine-cli tensorfold-install --extras`): extras are merged with what is already installed, the venv is reused when only adding extras to the same pinned ref and rebuilt (keeping them) otherwise, and the result is recorded as `extras=` on the version file's fourth line (read by `read_install_extras` → `tensorfoldExtras` capability / `system.tensorfold.extras`). The script's logic was run end to end on Linux against a local wheelhouse of fake packages behind `uname` / python shims (fresh, add, union, rebuild-preserves, unknown-extra refusal, ref change, legacy three-line file, URL and local-wheel sources). **`vision`** (Pillow + transformers + mlx-vlm): `family.vision` is True only for `Vontra/Qwen3.8-27B-MLX-4bit`; with the extra the engine starts `tensorfold serve --vision`, images go as `image_url` data-URL parts on the final user turn (raw composer base64, TensorFold sniffs the bytes), `LoadedModelInfo.visionEnabled` flips `supportsVision` so the composer offers Attach image (the engine-level demotion for TensorFold was removed), and an image request without it is ignored with a note that says how to add support. Our message shape was run through TensorFold's own `split_images` / `load_images` with real PNG and JPEG data (a GIF is refused with its reason). Upstream still calls vision experimental and publishes no throughput numbers; image requests start from a fresh KV cache; earlier turns' images are not resent. **`grammar`** (xgrammar → torch): with it a `json_schema` request is sent as `response_format` (our payload was run through TensorFold's `request_spec`); without it the engine now **refuses** the request with an in-app install hint instead of letting TensorFold answer 400 with a `pip` command that does not apply to the isolated venv (ignoring the schema would hand a JSON-mode API client free-form text). **Open:** live Mac run of `--vision` with Qwen3.8-27B (answer quality, memory with the tower loaded) and of an xgrammar install (download size is unmeasured — the UI says only that it adds PyTorch).

</details>

<details>
<summary>FU-089 · DFlash 2 + new llama.cpp spec-dec types — GGUF + vLLM lanes shipped 2026-09-30</summary>

**Status or trigger:** **GGUF + vLLM lanes shipped 2026-09-30; MLX lane still gated on FU-057.**

**Notes:** DFlash 2 (z-lab / Inco AI, 2026-08-18; MIT, drafters Apache-2.0): +16–25 % acceptance over DFlash, 2.7–3.4× at batch 1 on Qwen3.8-27B. **llama.cpp:** [scripts/update-llama-cpp.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/update-llama-cpp.sh) now builds pinned tag `b11277` (was branch HEAD; `LLAMA_CPP_REF=master` to track) — DFlash/DFlash 2, DSpark, EAGLE-3, `ngram-mod`, sidecar discovery, Metal-4 prefill. Flags confirmed from the b11277 source (`common/arg.cpp`, `common/speculative.cpp`, `docs/speculative.md`); a Linux CPU build of b11277 passes every `_llama_server_supports` probe and its argument parser accepts both generated command lines (sidecar and ngram-mod) — they fail only at model load on dummy files. `_speculative_args` in [llama_cpp_engine.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/llama_cpp_engine.py): baked-in MTP heads → `draft-mtp`; else a sidecar beside the GGUF, in llama.cpp's own order `mtp-` > `dspark-` > `dflash-` > `eagle3-` (DFlash 2 is a `dflash-` file, detected by llama.cpp from GGUF metadata) → `--model-draft <file> --spec-type <type>` (+`--spec-draft-n-max 15` for block drafters; llama.cpp clamps to the trained block); else draft-free `ngram-mod`. Each type gated on `--help`; old binaries get standard decode + a note. Sidecar resolver + case-sensitive aux-file filter (mirrors `gguf_filename_is_model`, so `…-MTP-Q8_0.gguf` stays a model) in [text_gguf.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/helpers/text_gguf.py); the FU-087 download allowlist now adds the closest-quant sidecar of each kind. **Frontend bug fixed:** `sanitizeSpeculativeSelection` cleared `speculativeDecoding` for every GGUF model at load, so the FU-074 GGUF MTP toggle never reached the backend; GGUF now keeps it, and the toggle shows for every llama.cpp model. **vLLM:** floor ≥0.28 (FU-091); `dflash.DFLASH2_DRAFT_MODEL_MAP` (`Qwen/Qwen3.8-27B` → `z-lab/Qwen3.8-27B-DFlash2`) consulted only by the vLLM engine. **Open:** live Metal run of a `ggml-org/Qwen3.8-27B-GGUF` sidecar load after `update-llama-cpp.sh`; MLX DFlash 2 needs dflash-mlx post-v0.1.10 (FU-057); the `incoai/*` vLLM drafters and `z-lab/Muse-Glimmer-30B-DFlash2` are unmapped (Muse Glimmer isn't in the catalog).

</details>

<details>
<summary>FU-083 · kokoro-onnx TTS lane (Windows/Linux) — foundation shipped, awaiting live CUDA/CPU validation — Foundation shipped 2026-07-03</summary>

**Status or trigger:** **Foundation shipped 2026-07-03; live-validate when the FU-056 Windows runner comes online or a Windows/Linux UAT pass runs.**

**Notes:** Original bug: the non-Apple branch of `synthesize_speech` called `kokoro_onnx.Kokoro()` with no arguments — upstream requires `Kokoro(model_path, voices_path)` pointing at `kokoro-v1.0.onnx` (~310 MB) + `voices-v1.0.bin` from the kokoro-onnx GitHub releases (asset URLs verified live 2026-07-03), so TTS on Windows/Linux failed at first call. Shipped: new [backend_service/voice_kokoro_onnx.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_kokoro_onnx.py) owns the file lifecycle (background-thread download to `~/.chaosengine/voice/kokoro/` with atomic tempfile-rename so a killed download never passes `is_installed`), correct constructor + per-voice espeak lang mapping (a→en-us, b→en-gb, j→ja); [voice_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/voice_runtime.py) delegates the non-Apple branch and reports `ttsModelRepo: null` + kokoro-file presence in capabilities; routes `POST /api/voice/kokoro/download` + `GET /api/voice/kokoro/download/status` in [routes/voice.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/voice.py); frontend `KokoroOnnxCard` in [VoiceModelsTab.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/features/voice/VoiceModelsTab.tsx) (download button + 2s progress poll, shown when `ttsModelRepo` is null). 8 unit tests in [tests/test_voice.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_voice.py) cover install detection, missing-files raise, constructor/lang args, atomic download worker, failure recording, runtime delegation, and capabilities. Cannot exercise the real kokoro-onnx wheel on this Apple Silicon box — the remaining work is a live Windows/Linux smoke (download → synthesize → audible WAV).

</details>

<details>
<summary>FU-082 · Upstream model-catalog scan — Qwen3.7, Wan2.6/2.7/3.0 not yet addable — Watch-closely, re-scan when weights publish.</summary>

**Status or trigger:** **Watch-closely, re-scan when weights publish.**

**Notes:** 2026-06-30 upstream scan (prompted by "are we geared up for the latest repos"): checked top-of-range coverage for every major family in the catalog against real current releases. **Gemma 4, FLUX.2, DeepSeek V4** are already the latest releases and already have catalog families (`gemma-4`, `flux-2`, `deepseek-v4`) — no gap. **Qwen 3.7** (Qwen3.7-Max + Qwen3.7-Plus, announced 2026-05-20/06-02) has **no open-weight Hugging Face repos** — direct probes of every plausible card (`Qwen/Qwen3.7-Max`, `-Max-Preview`, `-Plus-Preview`, `-27B`, `-35B-A3B`) return 401; it's Bailian-API-only today. **Wan 2.6 / 2.7** (Alibaba Tongyi, first/last-frame control + 15s clips) are likewise hosted-only (DashScope / WaveSpeedAI) — no `Wan-AI/...-Diffusers` repo exists yet, only Wan2.1/2.2 do. **Wan 3.0** (60B, native 4K, open weights "expected mid-to-late Q2 2026" per upstream reporting) is the one to watch — that window is closing now; re-check in the next scan for an actual `Wan-AI/Wan3.0-*` HF repo before adding a catalog family. Deliberately did **not** add speculative catalog rows for any of these — pointing a variant at a repo that doesn't exist yet is the exact FU-053 false-positive class of bug. Re-scan trigger: any of `Qwen/Qwen3.7-*`, `Wan-AI/Wan2.6-*`, `Wan-AI/Wan2.7-*`, or `Wan-AI/Wan3.0-*` resolving on Hugging Face.  **2026-10-01 scan (supersedes the Qwen/Wan notes above):** **Qwen-Image-2.1** (`Qwen/Qwen-Image-2.1`, 2026-09-14) is out with open weights — 7B single-stream DiT + Qwen3-VL-8B encoder, text-to-image + multi-reference editing + native RGBA, **Qwen Research licence (non-commercial only)**; shipped as a launchable sd.cpp variant (FU-096). **Wan 2.5 / 2.6 / 2.7 / 3.0** still have no real HF repo (only empty placeholders); the newest official Wan repos are Wan2.2-Animate-2 and Wan-Dancer-14B. Watching, no catalog row (no runnable path here): **LTX-2.5** (`Lightricks/LTX-2.5`, 2026-07-23, gated, 22B, Gemma-4-12B text encoder, LTX-2.x community licence; `Lightricks/LTX-2.5-Diffusers` exists; no MLX conversion and mlx-video supports LTX-2 / 2.3 only — last upstream commit 2026-05-13); **MiniMax-H3** (`MiniMaxAI/MiniMax-H3`, 2026-07-28, video + stereo audio up to 15 s at 768p, 2K via a regenerate stage, 498 GB repo, Qwen3-VL-32B encoder, community licence, prompt-rewriting stage hosted-only — not practical on a 64 GB Mac); **LongCat-Video** (Meituan, MIT, 13.6B; `mlx-community/LongCat-Video-{q4,q8,bf16}` since 2026-06-04, DiT ~9 GB at q4 — Apple-native but needs its own MLX package, not wired). Image models also worth a catalog row when someone has time: Krea 2 Turbo (2026-06, 12B, gated), Ideogram 4 (typography), Ming-Image-0.1-Design (2026-09, MIT, RGBA/typography), ERNIE-Image. The video lane has no "watching upstream" mechanism (FU-061 is image-only); this row is the tracker for it. Research write-up: [docs/UPSTREAM_RESEARCH_2026-10-01.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/docs/UPSTREAM_RESEARCH_2026-10-01.md).

</details>

<details>
<summary>FU-067 · Watch dflash-mlx for v0.1.8+ migration guide (FU-057 is multi-hour, deferred) — Trigger: (a) upstream publishes v0.1.8 with a stability commitment + migration guide, OR (b) we hit a concrete…</summary>

**Status or trigger:** Trigger: (a) upstream publishes v0.1.8 with a stability commitment + migration guide, OR (b) we hit a concrete user-visible bug on the orphan `fada1eb` pin, OR (c) a shipped catalog model needs a v0.1.6+ feature (adaptive verify / Gemma4 backend / Qwen3-Next GDN).

**Notes:** Dup of FU-057's trigger but resurfaced after the v0.9.3 upstream scan confirmed v0.1.7 is now on PyPI (`pip install dflash-mlx==0.1.7` resolves) and tagged at commit `210a0fc1`. Plan-of-record stays FU-057's six-step migration. Re-checking quarterly via `git ls-remote --tags` for `v0.1.8` / `v0.2.0` release tags — if upstream publishes a migration guide alongside, the cost drops dramatically. **2026-06-11 release scan:** **v0.1.9** is now tagged (branch HEAD `7f884380`; tags `v0.1.5.1…v0.1.9`). Still no published migration guide, so FU-057's six-step rewrite stays the plan of record and remains deferred. Newest migration target is now v0.1.9 (was v0.1.7/v0.1.8). **2026-06-15 release scan:** **v0.1.10** now tagged (branch HEAD `9ca00289`). One more release since last scan; migration target advances to v0.1.10. No migration guide published. FU-057 deferred.

</details>

<details>
<summary>FU-066 · Audit cache-strategy-matrix runner against bumped turboquant-mlx-full 0.11.x — When the 0.11.0 floor is installed on the M4 Max box or when a user reports a TurboQuant regression.</summary>

**Status or trigger:** When the 0.11.0 floor is installed on the M4 Max box or when a user reports a TurboQuant regression.

**Notes:** The runner's TurboQuant cell (`mlx-community/Qwen3-0.6B-4bit × cacheStrategy=turboquant cacheBits=3`) passed against 0.3.0 with output hash `b4337bc07457` (FU-051 evidence). 0.4.x expert-streaming + 0.5.x parallel prefetch + 0.8.x Mamba/hybrid arch support are all no-ops for dense 0.6B but may affect MoE models. **2026-06-15:** floor bumped `>=0.6.2` → `>=0.8.0`. **2026-06-26 scan:** floor bumped `>=0.8.0` → `>=0.11.0` ([pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml)) — 0.9.0/0.10.0/0.11.0 add **weight compression** (Hadamard + Lloyd-Max applied to the weights at compile time, not just the KV cache at run time) on top of the existing path, plus broader MoE/hybrid coverage (DeepSeek-V2/V3, Qwen3-235B-A22B, Nemotron-3-Super-120B). Additive — same `TurboQuantKVCache` call surface, loose `>=` so existing installs stay satisfied; the local dev venv still runs 0.3.0, so this is **not yet live-validated** (this row covers the one-time matrix capture once the 0.11.0 wheel is installed). Threshold tracked "0.8.x" → "0.11.x". **2026-09-28:** PyPI is at 0.28.0 (Qwen3.8 / MTP-rollback / VLM additions; `TurboQuantKVCache` constructor unchanged). Live A/B on Qwen3-4B-bf16, 160-token decode, M4 Max: native 25.7–29.7 tok/s, TurboQuant 3-bit **7.4 tok/s on 0.3.0 vs 5.6 on 0.28.0** — output coherent on both, but 0.28 is ~25 % slower at short context and both are 3.5–5× below native. Floor left at 0.11 (fresh installs already resolve 0.28). Worth a long-context run before recommending TurboQuant as a speed option rather than a memory one — `scripts/calibrate-cache-speed.py` (FU-095) now does that run. **2026-10-30, release bundle (v0.9.9: `turboquant-mlx-full` 0.3.0 pinned in `release-constraints.txt`, on mlx 0.32.4 / mlx-lm 0.32.2; run on an isolated backend started from the extracted bundle, greedy, 140 tokens, 4K context, M4 Max):** with the app's default `fp16Layers` 4 the output is coherent at 2, 3 and 4 bits; speed against native: Qwen2.5-0.5B-4bit 124 against 537 tok/s (3-bit), Qwen3-4B-4bit 74 / 66 / 62 against 166 (4 / 3 / 2-bit). With `fp16Layers` 0 the same two small models degrade badly (0.5B at 3-bit and 4B at 2-bit produced repeated tokens, 4B at 4-bit drifted off topic), while Qwen3-Coder-30B-A3B (MoE) at 3-bit gave 33 against 118 tok/s and the hybrid Qwen3.6-27B at 3-bit gave 23 against 30 tok/s, both coherent and both reported `turboquant` as the effective strategy. So 0.3.0 works on dense, MoE and hybrid models on the release stack and is not a known limitation; the quick matrix's TurboQuant cell only checks for non-empty output and would have passed the garbage runs. **Not covered:** a quality metric beyond reading the first lines, long contexts (see FU-095), a bit-for-bit comparison with 0.28, and whether newer versions fix the `fp16Layers` 0 degradation. The `pyproject.toml` floor (`>=0.11.0`) still disagrees with the pinned bundle version; a bump needs this same run on the new version.

</details>

<details>
<summary>FU-065 · Pin llama-cpp-turboquant to a commit hash instead of branch HEAD — Trigger: any user-reported build divergence between two install runs, OR a release-build gate where reproducib…</summary>

**Status or trigger:** Trigger: any user-reported build divergence between two install runs, OR a release-build gate where reproducibility matters more than tracking upstream.

**Notes:** [scripts/build-llama-turbo.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/build-llama-turbo.sh) + [scripts/update-llama-turbo.sh](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/update-llama-turbo.sh) currently clone `TheTom/llama-cpp-turboquant` at branch `feature/turboquant-kv-cache` (`LLAMA_TURBO_BRANCH` env var), then `git reset --hard origin/$TURBO_BRANCH`. Two installs at different times can ship different binaries — the same drift problem FU-033 fixed for `dflash-mlx`. Today's branch HEAD is `2cbfdc62a1a047b01377948dfdede8cb6a744866`. Plan: add `LLAMA_TURBO_COMMIT="${LLAMA_TURBO_COMMIT:-2cbfdc62...}"` to both scripts, `git checkout "$LLAMA_TURBO_COMMIT"` after fetch, surface the hash in `llama-server-turbo.version`, and add a sync-assert to `pre-build-check` that compares the build-script pin to a value in [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml) or a dedicated `UPSTREAM_PINS.md`. Defer because (a) branch is single-purpose with low churn — author is the same TheTom we already trust for `turboquant_plus`; (b) we already have the v0.9.2 → v0.9.3 release with this code path working. **2026-06-11 release scan:** branch HEAD has drifted `2cbfdc62…` → `73eb521daebc85da7c91d37178940b99a5524cf6` — confirms the reproducibility risk this row tracks. Pin still deferred: pinning the *drifted* `73eb521d` is unsafe without a verified test-compile (could ship a broken turbo binary), and reverting-pinning to the known-good `2cbfdc62` drops upstream work. When picked up, pin to a commit that's been build-tested on the M4 Max box. **2026-06-15 release scan:** branch HEAD drifted again → `7985f6b90bf19881ab7c7a8444954e91cae36056`. Reproducibility risk continues to accumulate. Still deferred pending test-compile.

</details>

<details>
<summary>FU-057 · dflash-mlx upgraded from v0.1.5.1 (fada1eb) to v0.1.10 (9ca0028) — Shipped 2026-11-11.</summary>

**Status or trigger:** **Shipped 2026-11-11.**

**Notes:** **Pin:** [pyproject.toml](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml), `_INSTALLABLE_PIP_PACKAGES` in [routes/setup](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/routes/setup/__init__.py), [stage-runtime.mjs](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs) and the install hint in [RuntimeControls.tsx](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/RuntimeControls.tsx); licence still Apache-2.0; still `mlx>=0.31` / `mlx-lm>=0.31`, so the release constraints (mlx 0.32.4, mlx-lm 0.32.2) are unchanged. **API changes handled:** `dflash_mlx.runtime` is a package whose `__all__` is only `VerifyConfig` / `get_stop_token_ids` / `stream_dflash_generate`; `resolve_target_ops` + `bind_draft_to_target` come from `engine.target_ops`, `load_draft_bundle` from `runtime.loading`, `build_offline_runtime_context` from `runtime.context` (was `runtime_context`); `stream_dflash_generate` requires `target_ops` and `draft_backend` (`EagerDraftBackend`) and yields frozen `TokenEvent` / `SummaryEvent` dataclasses instead of dicts; `configure_full_attention_split` is gone (routing is target-owned); `make_draft_backend` is gone (DDTree's draft cache uses `EagerDraftBackend().make_cache`). The worker still reuses its loaded target and now prepares it like upstream's `load_target_bundle`: `install_speculative_hooks`, verify-length quantized linears where the family supports them, and the drafter's registry default quant (`w4` for the Qwen3.5-9B/27B/35B and Qwen3.6 drafters) ([dflash_compat.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/dflash_compat.py) `prepare_target` / `default_draft_quant`). A `SummaryEvent.fallback_ar` (upstream decoded without the draft) is now said in the runtime note. **FU-135:** v0.1.10 ships the same `text_wrapper` fix (`fix: prefer Qwen language_model wrapper`), so `patch_text_wrapper` was removed; the load-time `self_check` stays and now runs after the hooks and verify linears are in. `is_ddtree_available` checks each symbol in the module that defines it (it read only `runtime/__init__.py`, which would have reported DDTree unavailable). [check-upstreams.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/check-upstreams.py) reports the pin current when it is the newest tag's commit. **Measured, M4 Max 64 GB, Qwen3.6-27B 4-bit + `z-lab/Qwen3.6-27B-DFlash`, through the app (backend from the branch on port 8877, CLI `load --spec` / `prompt --metrics`), 256 tokens, temperature 0, Thinking off, 2 runs, decode tok/s code / explain / story / think:** old pin DFlash 53.2 / 25.4 / 13.8 / 32.5 (mean 31.2), plain 22.6 / 21.5 / 20.8 / 22.0 (21.7); **new pin DFlash 58.0 / 39.8 / 26.8 / 43.2 (mean 41.9, +34 %)**, plain 24.4 / 24.1 / 24.6 / 25.3 (24.6; same plain text as before). Prose no longer loses to plain on this model (story 26.8 against 24.6, was 13.8 against 20.8; see FU-134). Output: code and explain now equal plain greedy byte for byte; story differs from plain at the 3rd word exactly as the old pin did; think differs after ~650 characters ("into a format that can be added" vs "into hours and minutes to add"), the usual near-tie of a speculative verify. Upstream's own adaptive block (`block_tokens=None`, 16 shrinking under low acceptance) measured 70.3 / 38.0 / 27.1 / 39.8 (mean 43.8): better on code, slightly worse on prose, within noise on the mean, so the 8-token block stays. Tests: `test_dflash.py`, `test_mlx_models.py`, `test_dflash_compat.py`, `test_mlx_worker.py`, `test_ddtree_attention.py`, `test_check_upstreams.py` pass against v0.1.10 in a scratch venv. **Not tested:** DDTree live (no attention-only model with a drafter on the machine; the tiny-model tree test passes), Gemma 4 targets, other Qwen3.5 / 3.6 sizes, the packaged app and the Setup tab install button. Upstream `verify_mode="ddtree"`, `copyspec_mode=auto` and the larger prefix cache are not wired. **Still open:** MLX DFlash 2 (Qwen3.8 drafts, FU-089) needs untagged `main` after v0.1.10; class-level attention hooks stay installed in the worker process after a DFlash load (plain output was unchanged in the bench). DDTree measured live 2026-11-12 on Qwen3-8B 4-bit + z-lab/Qwen3-8B-DFlash-b16 (M4 Max, 300 tokens, code prompt): DDTree budget 16 accepts 6.9 tokens per round against 2.5 for linear DFlash, but decodes 57-62 tok/s against 64-73 (plain 36), so the tree verify costs more than it saves on this model.

</details>

<details>
<summary>FU-056 · In-app accelerator install UX (Nunchaku / SageAttention / DFlash CUDA / TriAttention / kvpress + vLLM-via-WSL bridge) — Active. Phase 1 shipped on feature/accelerator-install-ux</summary>

**Status or trigger:** Active. Phase 1 shipped on `feature/accelerator-install-ux`; phases 2-9 pending.

**Notes:** The plan: bring every CUDA-side accelerator install in-band so users never need to drop to PowerShell to type `pip install <name>` — the install affordance lives next to the thing it accelerates (FLUX cards in Image Studio Discover, Wan cards in Video Studio Discover, the chat composer for spec-dec, a one-stop "Boost Pack" panel in Diagnostics for the completionist). The backend pipeline (`POST /api/setup/install-package` → background or sync install → capability re-probe → UI refresh) is already proven by FU-008 / FU-016 / FU-019 / FU-023 / FU-025 — what's missing is per-accelerator capability flags + the contextual badges/buttons on each feature surface. **Phase 1 (foundation, shipped 2026-05-17):** new probe module [backend_service/inference/accelerators.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/accelerators.py) with lazy importability + version helpers for nunchaku, sageattention, dflash-mlx, dflash-cuda, triattention, kvpress, plus a Windows-only `wsl2_available()` shell probe for Phase 8. 11 new fields on `BackendCapabilities` ([base.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/base.py)) + matching serialization in `to_dict`. Probes wired into both the cheap placeholder probe and the full `_probe_native_backends` ([capabilities.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/capabilities.py)) so the frontend gets accurate "Install" / "Installed" state on first paint. 25 unit tests in [tests/test_accelerator_capabilities.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_accelerator_capabilities.py) pin the present / absent / broken-install matrix. **Phases 2-9 planned:** (2) reusable `<AcceleratorCard>` component, (3) Image Studio Discover/Models badges + post-generation suggestion toast, (4) Video Studio Discover/Models badges + LongLive bundle rename, (5) Chat composer hint when CUDA + DRAFT_MODEL_MAP hit, (6) Diagnostics "Boost Pack" panel (one-stop view), (7) per-variant `recommendedAccelerators` catalog metadata + i18n, (8) Windows vLLM-via-WSL2 bridge (WSL detector + isolated venv install + remote subprocess engine), (9) cache-strategy-matrix runner + pre-build gate integration. End-state UX: fresh user installs ChaosEngineAI → downloads FLUX → sees "🚀 Nunchaku +3× available [Install]" pill on the catalog card → one click → 90s later first generation runs at SVDQuant speed, no terminal required.

</details>

<details>
<summary>FU-052 · Cache-strategy matrix: vLLM backend cells + MTPLX MLX cell + GGUF MTP cell (comprehensive testing milestone) — Foundation shipped 2026-05-17</summary>

**Status or trigger:** **Foundation shipped 2026-05-17; awaiting live cross-platform validation.**

**Notes:** Matrix expanded from 9 → 15 cells (see [scripts/cache-strategy-matrix.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/cache-strategy-matrix.py)): (1) **MTPLX MLX cell** on `mlx-community/Qwen3.5-4B-bf16` — smallest entry in `MTP_MODEL_MAP` ([backend_service/inference/_mtp.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_mtp.py)); skips with `MTPLX runtime not installed` on boxes without the MTPLX venv. (2) **GGUF MTP cell** on `ggml-org/Qwen3.6-27B-MTP-GGUF` (29 GB Q8_0) — exercises FU-047's `--spec-type draft-mtp` path; skips with `llama-server lacks --spec-type draft-mtp` on stale llama.cpp builds. (3) **4 vLLM cells** (`backend="vllm"`) — native + turboquant + triattention on `Qwen/Qwen3-0.6B` plus DFlash on `Qwen/Qwen3.5-4B`; skips with `vLLM not installed (CUDA-only)` on macOS. `BackendCapabilities` dataclass extended with `mtplx_available`, `gguf_mtp_available`, `vllm_available`, sourced from `/api/health.nativeBackends`. `skip_reason()` handles every new spec-dec value (`mtplx`, `gguf-mtp`) + backend (`vllm`) with explicit, debuggable skip strings. **Live validation status by platform:** Apple Silicon (M4 Max 64 GB) — MLX strategy cells green (6/6), MTPLX + DFlash + DDTree + GGUF-MTP deferred until disk frees + Qwen3.5-4B-bf16 + Qwen3.6-27B-MTP-GGUF land in library (cache disk currently 7.2 GB free / 100% used). CUDA (RTX 4090 / Windows or Linux) — vLLM 4-cell quartet awaits the Windows test runner being driven end-to-end with `vllm` wheel installed + `Qwen/Qwen3-0.6B` + `Qwen/Qwen3.5-4B` in library. **Trip-wires to close this row:** matrix `--full` returns >0 pass cells from each new family (MTPLX, GGUF-MTP, vLLM-native, vLLM-spec-dec) on the appropriate platform, with output hashes + tok/s recorded. Tracker stays open until those numbers exist; tightening the assertion bar comes next (FU-051 effective-strategy field).

</details>

<details>
<summary>FU-051 · /api/models/load echoes legacy cacheStrategy verbatim instead of returning the FU-030-coerced canonical id — When this row blocks an automated check (currently it's surfaced as a known-bad FU-030 regression line in the…</summary>

**Status or trigger:** When this row blocks an automated check (currently it's surfaced as a known-bad FU-030 regression line in the matrix Markdown report, not a hard fail).

**Notes:** The matrix run revealed that `POST /api/models/load` with `cacheStrategy=chaosengine` returns `runtime.loadedModel.cacheStrategy = "chaosengine"` even though the registry coerces the request to `turboquant` for actual execution (FU-030). Evidence the coercion really happens at the engine level: in two back-to-back matrix runs against `mlx-community/Qwen3-0.6B-4bit`, `cacheStrategy=rotorquant cacheBits=3` and `cacheStrategy=turboquant cacheBits=3` produced **identical** SHA-12 output hashes (`b4337bc07457`), proving rotorquant routes through turboquant under the hood. Only the surface response field is stale. Three options, in order of fix surface: (a) cheapest — add an `effectiveCacheStrategy` field next to the requested `cacheStrategy` in the loaded-model payload, and update the matrix runner's FU-030 assertion to read it; (b) coerce at the request boundary in `routes/models.py::load` so `cacheStrategy` itself is normalised before it lands in `loadedModel`; (c) keep the API response stable and document the cosmetic echo (cheapest UI-wise but means the matrix-runner assertion stays cosmetic too). Recommend (a) — preserves the user's literal request for telemetry while exposing the registry-coerced id as the canonical one. **Files touched:** `backend_service/state/__init__.py` `_loaded_model_payload`, `backend_service/inference/controller.py` `RuntimeController._build_loaded_model_state`, `scripts/cache-strategy-matrix.py` `write_markdown` (assertion + `effective_strategy` column).

</details>

<details>
<summary>FU-049 · Python 3.14 support gate — Re-evaluate quarterly</summary>

**Status or trigger:** Re-evaluate quarterly. Trigger to bump `requires-python` floor: torch ≥2.6 publishes stable cp314 wheels for darwin-arm64 + win-amd64 + linux-x86_64 (CUDA + CPU) **AND** mlx-lm + mlx-vlm + mlx-video publish cp314 wheels **AND** Astral `python-build-standalone` ships a 3.14 portable build for the Tauri sidecar.

**Notes:** Today `pyproject.toml` declares `requires-python = ">=3.10"` and the test matrix runs 3.11/3.12 ([scripts/e2e_test_suite.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/e2e_test_suite.py), Windows test guide, CI). Stay on 3.11/3.12 for ship + test until cp314 wheel coverage closes. **Why 3.14 is on radar:** (a) we already renamed `compression/` → `cache_compression/` to avoid shadowing Python 3.14's new stdlib `compression` namespace pkg — pre-emptive fix landed in v0.8.0. (b) 3.14 ships PEP 779 free-threaded build as stable (still opt-in via `python3.14t` builds), interesting for the FastAPI parent process but irrelevant for subprocess-isolated MLX / sd-cli / longlive workers. (c) GIL-default 3.14 still gives modest perf wins from tail-call interpreter (~5% on CPython benchmarks) + new sub-interpreter API. **3.14 blockers as of 2026-05-17:** (1) **PyTorch** — torch 2.5 stable + 2.6 nightly do not yet publish cp314 wheels on the full darwin-arm64 + win-amd64 + linux-x86_64 × {CPU, CUDA 12.4, ROCm 6.2, MPS} matrix; most painful single dep since image + video runtimes pin torch. (2) **MLX stack** — `mlx`, `mlx-lm`, `mlx-vlm`, `mlx-video` currently ship cp310–cp313 wheels; Apple Silicon adoption typically lags CPython release by 1–3 months. (3) **CUDA-compiled deps** — `bitsandbytes`, `flash-attn`, `sageattention`, `nunchaku`, `triattention`, `vllm-swift`, `dflash-mlx` (git+url, builds from source — needs cp314 cython/setuptools chain too). (4) **Tauri sidecar** — desktop release builds embed Python via Astral's `python-build-standalone`; need their 3.14 portable build before bundling. (5) **3.14 stdlib breakage** — deprecation removals (e.g. `typing.io`, `typing.re`, `asyncio.coroutine` shim, `pkg_resources` consequences); needs an audit pass via `python -W error::DeprecationWarning -m pytest tests/` on a 3.14 venv. (6) **3.14 `compression` namespace pkg** — already mitigated, but a regression probe should land in `pre-build-check` once we run CI on 3.14 (assert `from cache_compression import registry` works on cp314). **Plan when gate opens:** (a) bump `requires-python` to `>=3.11` first as an intermediate step (drops 3.10, which has no cp314 wheels anyway and lets us drop a few back-compat code paths); (b) add cp314 to GitHub Actions CI matrix alongside cp311 + cp312; (c) add a `python -X importtime` regression probe to `scripts/perf-baseline.py` (3.14's tail-call interpreter should improve cold-start by ~3–5%, want to measure); (d) bump `requires-python` to `>=3.11` floor + `<3.15` ceiling once green; (e) the Tauri sidecar Python pin advances independently — driven by `python-build-standalone` releases not pyproject.

</details>

<details>
<summary>FU-048 · Prefer-GGUF-MTP routing preference when both MTPLX + MTP-GGUF available — Re-evaluate when MTPLX exposes an in-process Python API OR when MTPLX framework overhead vs upstream mlx-lm dr…</summary>

**Status or trigger:** Re-evaluate when MTPLX exposes an in-process Python API OR when MTPLX framework overhead vs upstream mlx-lm drops below 5% on a public M-series benchmark

**Notes:** Colleague feedback (2026-05-16) after root-cause investigation of "MTPLX appears slower than baseline" benchmark: MTPLX's MTP path is real (+29% over MTPLX-internal AR with 60-66% draft acceptance, `runtime_mtp_enabled: true` confirmed in `mtplx_stats`) but the MTPLX framework wrapper adds ~25% overhead vs raw `mlx_lm.generate`, so MTPLX-MTP vs upstream mlx-lm comes out roughly flat (-4% net). GGUF MTP via llama.cpp PR #22673 (FU-047) delivers a measurable +51% over Q8_0 baseline live-validated 2026-05-16 (20.9 vs 13.8 tok/s on `ggml-org/Qwen3.6-27B-MTP-GGUF`, 60-72% acceptance), no subprocess overhead beyond the existing llama-server we already run. Today `RuntimeController._select_engine` prefers MTPLX whenever `mtplxAvailable && has_mtp_heads_strict(...)` — which means a user with both lanes available routes to the slower path. Proposed change: when `capabilities.ggufMtpAvailable && capabilities.mtplxAvailable && has_mtp_heads_strict(...)`, prefer the GGUF MTP lane unless the user explicitly pinned MTPLX. UX decision deferred — needs Settings → "Speculative decoding backend" preference (`auto` / `prefer-gguf-mtp` / `prefer-mtplx`) so power users on Apple Silicon who want the MTPLX paged-attention / fan-control burst features keep that path, while default users get the fastest measurable lane. Re-check posture if MTPLX ships either: (a) an in-process Python entrypoint that skips the HTTP wrapper, or (b) framework overhead drops via the v0.4+ "performance-hot" profile they're prototyping. Until then, the existing prefer-MTPLX heuristic stays — flipping it without the Settings toggle would regress users on MTPLX-Optimized quants that aren't GGUF-mirrored.

</details>

<details>
<summary>FU-046 · i18n Phase 4 — long tail + RTL prep — Opportunistic / community-driven</summary>

**Status or trigger:** Opportunistic / community-driven

**Notes:** Stretch locales as community PRs land ≥80 % coverage: `pl` (Bielik / SpeakLeash), `it` (Camoscio / Minerva), `uk` (DragoMan), `tr` (Trendyol-LLM), `vi` (VinaLlama / PhoGPT), `nl` / `sv` / `cs` (smaller but loud cohorts). RTL infrastructure prep for future `ar` (Jais / Fanar) + `he`: convert remaining `margin-left` / `padding-right` to logical properties (`margin-inline-start` / `padding-inline-end`); add `dir="auto"` to root + per-locale override; rehearse bidi flow via pseudo-locale `en-XB`. Don't ship `ar` / `he` until logical-property migration verified end-to-end.

</details>

<details>
<summary>FU-045 · i18n Phase 3 — completion tier (ko + pt-BR + zh-TW) — Follows FU-044</summary>

**Status or trigger:** Follows FU-044

**Notes:** Closes the top-10 locale set: Korean (Upstage Solar / LG EXAONE / Kakao crowd, heavy benchmark scene), Brazilian Portuguese (large maker base, separate from `pt-PT` — both vocab + diacritics differ), Traditional Chinese (Taiwan / Hong Kong; TAIDE / Breeze authors, distinct vocab + idiom from `zh-CN` so it's a separate locale not a variant). Opens contributor PR flow with `GLOSSARY.md` lock + README contributor docs. Diagnostics tab gets a coverage panel showing per-locale `%`.

</details>

<details>
<summary>FU-044 · i18n Phase 2 — EU tier (de + fr + es + ru) — Follows FU-043</summary>

**Status or trigger:** Follows FU-043

**Notes:** Adds German (DiscoLM / LeoLM origin, privacy-first crowd), French (Mistral / CroissantLLM homeland), Spanish (LatAm + Iberia, Salamandra / RigoChat), Russian (Saiga / Vikhr / IlyaGusev — strong r/LocalLLaMA presence). Wide-string layout pass critical for `de` (30–50 % wider than `en`, especially in Setup install descriptions). Slavic 4-form plural pass for `ru` via ICU MessageFormat (one / few / many / other — different rules than the en `plural`). Tauri window minimum width may need bump after `de` audit.

</details>

<details>
<summary>FU-043 · i18n Phase 1 — anchor locales (en + zh-CN + ja) — Follows FU-042</summary>

**Status or trigger:** Follows FU-042

**Notes:** First user-visible localization: ships English baseline + Simplified Chinese (Qwen / DeepSeek / GLM author origin) + Japanese (Sakana AI / ELYZA / heavy Apple-Silicon-tinkerer cohort). Highest-leverage trio given Hugging Face + r/LocalLLaMA demographics. AI-translated (Claude) against a pre-locked `GLOSSARY.md` per locale pinning tech terms (`KV cache`, `tokens/sec`, `quantization`, `inference`, `cache strategy`, etc.) so phrasing stays consistent across 200+ keys. Layout audit pass for CJK glyph density (CJK runs ~30 % narrower than en; verify no fixed-width buttons clip). Live QA matrix: 5 tabs × {happy path, error path, empty state} per locale = ~30 cells.

</details>

<details>
<summary>FU-042 · i18n Phase 0 — infra scaffold + IME composition fix — In progress (started 2026-05-11).</summary>

**Status or trigger:** **In progress (started 2026-05-11).**

**Notes:** Foundation work for full-stack localization. Adds `react-i18next` + `i18next-icu` + `i18next-browser-languagedetector` (FE), `Babel` (Py), `rust-i18n` + `fluent-bundle` (Rs). Creates `src/locales/`, `backend_service/locales/`, `src-tauri/locales/` directory trees with namespaced JSON / `.po` / `.ftl` files. Wires `i18n/index.ts` provider in `src/main.tsx`, FastAPI `Accept-Language` middleware in `backend_service/i18n.py`, `rust-i18n!` macro in `src-tauri/src/i18n.rs`. Adds `locale` field to settings (TS + Py + Rs). Builds Settings → Language dropdown rendering endonyms. Builds first-launch banner that detects OS lang ≠ en and prompts user. Adds `<html lang="…">` sync + CJK font-fallback CSS stack. Writes `scripts/i18n-extract.mjs` / `i18n-validate.mjs` / `i18n-pseudo.mjs` (pseudo-locales `en-XA` + `en-XB` dev-only). Hooks `i18n-validate` into `pre-build-check.sh` + `.mjs`. Fixes IME `e.isComposing` guard on prompt textarea Enter-to-send (latent bug breaking JP/CN/KR IME users). Audits `src/utils/format.ts` to thread `locale` arg into `Intl.NumberFormat` / `Intl.DateTimeFormat` helpers. Per CLAUDE.md §6, en-only feature commits ship with `null` placeholders in other locales — runtime falls back to `en`. Adds `THIRD_PARTY_NOTICES.md` rows for all 5 new deps (MIT × 3, Apache-2.0 × 1, BSD-3 × 1). Coverage gate is warn-only v1 with dashboard panel in Diagnostics tab. Wraps with FU-043 Phase 1 translations.

</details>

<details>
<summary>FU-032 · TurboQuant+ (TheTom/turboquant_plus) Apple Silicon Metal kernels (watch-closely) — Re-evaluate when upstream tags v1.0 release or beats turboquant-mlx-full 0.8.0 on a public M-series benchmark</summary>

**Status or trigger:** Re-evaluate when upstream tags v1.0 release or beats `turboquant-mlx-full` 0.8.0 on a public M-series benchmark

**Notes:** Same author as our `llama-cpp-turboquant` fork. Adds Walsh-Hadamard rotation (improvement over base TurboQuant's Hadamard-only path) + a sparse-V optimization on M5 Max that achieves 0.93x of q8_0 decode speed at long context while saving 50–64% of KV memory. Reported numbers: turbo3 4.6× compression at +1.06% PPL, turbo4 3.8× compression at +0.23% PPL — comparable to our existing `turboquant-mlx-full` pin but with newer kernels. **Not on PyPI** (development install via `git clone` + `pip install -e .[dev]`), so adopting it means a vendored or git+url install pattern like dflash-mlx — re-evaluate when upstream publishes a wheel or tags a v1.0. Apple Silicon stays on `turboquant-mlx-full` for now. **2026-06-15 scan:** latest tags are v0.3.2.1–v0.3.2.3 (HEAD `7f601a13`). Still no PyPI wheel, still no v1.0 tag. FU-032 trigger not met; updated comparison baseline from 0.3.0 to 0.8.0 since our floor advanced. **2026-06-26 scan:** still v0.3.2.1–v0.3.2.3 (HEAD `8c4067ff`), no new tags, still no PyPI wheel, still no v1.0. Trigger not met; comparison baseline advanced 0.8.0 → 0.11.0 to track our bumped `turboquant-mlx-full` floor (FU-066).

</details>

<details>
<summary>FU-029 · KVTC (NVIDIA ICLR 2026) KV cache strategy — Deferred 2026-05-10</summary>

**Status or trigger:** **Deferred 2026-05-10 — CUDA-only upstream, awaiting MLX/Metal port + PyPI release.**

**Notes:** Targeting [OnlyTerp/kvtc](https://github.com/OnlyTerp/kvtc) (Apache 2.0). PCA + adaptive quantization + entropy coding — 8–32× compression vs the dropped ChaosEngine's 3.7×, peer-reviewed at ICLR 2026, beats TurboQuant by 37% at comparable quality on long-context. Upstream blockers: (a) CUDA-only — repo's roadmap mentions MLX/Metal as "planned" but not yet implemented, so the Apple Silicon dev box cannot validate end-to-end; (b) not on PyPI — distributed as a `src.*` repo intended for `git clone`; (c) integration shape is a HuggingFace `DynamicCache` wrapper (not a llama.cpp cache type), so the existing GGUF lane has no path. Re-evaluate when either upstream ships MLX support or a Windows/Linux+CUDA development box becomes available. Apple Silicon users continue on TurboQuant-MLX (also ICLR 2026, native today).

</details>

<details>
<summary>FU-028 · MTP (Multi-Token Prediction) speculative decoding — Deferred 2026-05-10</summary>

**Status or trigger:** **Deferred 2026-05-10 — upstream MTP-head loader gap on both runtimes.**

**Notes:** Target: lossless 1.5–2.2× speedup for trained-with-MTP models (Gemma-4 drafters released 2026-05-05, Apache 2.0; DeepSeek V3/R1; Qwen3.5/3.6/Next; Nemotron-3; MiMo-V2-Flash). **Blocker on Apple Silicon:** mlx-lm 0.31.3 ships ``stream_generate(..., draft_model=...)`` for *separately-trained* draft models but has no native MTP-head loader — Gemma-4-style MTP drafters share activations + KV cache with the target and cannot be loaded as a standalone ``mlx.nn.Module``. Confirmed by inspecting the installed `.venv/lib/python3.11/site-packages/mlx_lm/server.py` + `generate.py` — no MTP-specific code paths. **Blocker on llama.cpp:** PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (am17an, ``--spec-type mtp --spec-draft-n-max N``) is still in Draft as of 2026-05-10, awaiting at least 2 approving reviews. **Third-party path considered + rejected for v1:** [MTPLX](https://github.com/youssofal/MTPLX) (221 stars, MIT) wraps native MTP for Apple Silicon but ships as an OpenAI/Anthropic HTTP server — chaining HTTP servers from our FastAPI backend has unwanted latency + retry surface. **Re-evaluate when:** (a) mlx-lm gains a native MTP head loader (track ``ml-explore/mlx-lm`` releases), OR (b) llama.cpp PR #22673 merges, OR (c) MTPLX exposes a programmatic in-process Python API. The user-facing speedup is real (live benchmarks: M4 Pro × Qwen3.5-27B-4bit 15.3 → 23.3 tok/s) so this stays high-priority on the queue.

</details>

<details>
<summary>FU-027 · NVIDIA/kvpress KV cache toolkit (CUDA-side) — Setup install action pre-staged 2026-05-05</summary>

**Status or trigger:** **Setup install action pre-staged 2026-05-05; integration code pending.**

**Notes:** [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) — Apache 2.0, 1.1k stars, `kvpress>=0.5.3` registered in `_INSTALLABLE_PIP_PACKAGES` so the Setup tab can pre-stage the wheel. Integration hooks land separately under `cache_compression/kvpress.py` once the helper picks an adapter shape (the upstream library exposes `presses` per technique — e.g. SnapKV / TOVA / KIVI / pyramid — and a `Pipeline` wrapper that takes a HF transformers model). Apple Silicon stays on TurboQuant-MLX; this is the CUDA-side complement.

</details>

<details>
<summary>FU-024 · FP8 layerwise casting for non-FLUX DiTs — Foundation shipped 2026-05-05</summary>

**Status or trigger:** **Foundation shipped 2026-05-05; awaiting live CUDA SM 8.9+ validation.**

**Notes:** Apple Silicon can't exercise — Windows/Linux CUDA pull validates. Backend: `_maybe_enable_fp8_layerwise` helper in [image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py) calls `transformer.enable_layerwise_casting(storage_dtype=…, compute_dtype=torch.bfloat16)` post-load. Family-correct fp8 dtype: E5M2 for HunyuanVideo (per upstream model card recommendation), E4M3 elsewhere (FLUX / Wan / Qwen-Image / SD3 / LTX). Compute capability gate refuses pre-Ada GPUs (SM <8.9) since hardware fp8 isn't there + the cast slows wall-time vs bf16. Helper degrades gracefully when `pipeline.transformer.enable_layerwise_casting` is missing (UNet pipelines / old diffusers) — runtimeNote surfaced into the load notes. Wired through both ImageGenerationConfig + VideoGenerationConfig + Request models + frontend hooks (`imageFp8LayerwiseCasting` / `videoFp8LayerwiseCasting`) + types. Default off; opt-in.

</details>

<details>
<summary>FU-023 · SVDQuant / Nunchaku CUDA engine — Foundation shipped 2026-05-05</summary>

**Status or trigger:** **Foundation shipped 2026-05-05; awaiting live Windows / Linux CUDA validation.**

**Notes:** Apple Silicon dev box can't exercise the CUDA path live — wiring is in place so a Windows/Linux CUDA pull validates end-to-end. Backend: `_try_load_nunchaku_transformer` helper in [image_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/image_runtime.py) loads via `NunchakuFluxTransformer2dModel` / `NunchakuQwenImageTransformer2DModel` / `NunchakuSD3Transformer2DModel` / `NunchakuSanaTransformer2DModel` / `NunchakuPixArtSigmaTransformer2DModel` — class registry at `_nunchaku_transformer_class_for_repo`. Preferred over NF4/int8wo on CUDA when `nunchakuRepo` pinned + nunchaku importable; falls back cleanly on Apple Silicon / CPU / missing package. Variant key extends with `nunchaku=...` so toggling rebuilds the pipeline. ImageGenerationConfig + ImageGenerationRequest fields: `nunchakuRepo`, `nunchakuFile`. Catalog rows: FLUX.1 Dev × svdq-int4-flux.1-dev, FLUX.1 Schnell × svdq-int4-flux.1-schnell. Setup install: `nunchaku>=1.2.1` via `_INSTALLABLE_PIP_PACKAGES`. Wan / HunyuanVideo / LTX wrappers don't exist in upstream Nunchaku v1.2.1 — adding a future video variant is a catalog-row change.

</details>

<details>
<summary>FU-013 · Vendored STG-enabled LTX pipeline — Phase F or when a user reports that Phase D + E1 + E2 quality remains short of the upstream reference</summary>

**Status or trigger:** Phase F or when a user reports that Phase D + E1 + E2 quality remains short of the upstream reference

**Notes:** Subclass `LTXPipeline` and override `__call__` to add a third forward pass per step with selected transformer block(s) perturbed (skip self-attention or replace with identity). Combine: `pred = uncond + cfg*(text - uncond) + stg_scale*(text - perturbed)`. Reference: Lightricks' upstream LTX-Video repo's `STGSamplingHook`. Estimated ~250 lines of vendored code + tests. Sequence dependency: do this AFTER FU-007 (Wan TeaCache) ships so the cache vs guidance interactions are tested in isolation.

</details>

<details>
<summary>FU-012 · LTX Spatial Temporal Guidance (STG) — diffusers ships LTXPipeline with perturbed_blocks kwarg, or vendor a forward patch</summary>

**Status or trigger:** diffusers ships LTXPipeline with `perturbed_blocks` kwarg, or vendor a forward patch

**Notes:** Upstream reference workflows enable STG by default — perturbs final transformer blocks during sampling to reduce object breakup / chroma drift. Our pinned diffusers' LTXPipeline does not accept `perturbed_blocks`. Phase D landed `frame_rate` + `decode_timestep` + `decode_noise_scale` + `guidance_rescale` for reference parity on the basic kwargs; STG is the remaining gap. Track upstream; if quality remains short of the reference, vendor a forward patch under [cache_compression/_teacache_patches/ltx_video.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/cache_compression/_teacache_patches/ltx_video.py)-style.

</details>

<details>
<summary>FU-011 · LTX-Video 2.3 diffusers variant — Lightricks publishes diffusers-compatible weights (Lightricks/LTX-2.3 gains model_index.json)</summary>

**Status or trigger:** Lightricks publishes diffusers-compatible weights (`Lightricks/LTX-2.3` gains `model_index.json`)

**Notes:** LTX-2.3 currently routes via mlx-video on Apple Silicon (`prince-canuma/LTX-2.3-{distilled,dev}` already in catalog). Lightricks' own model card states "diffusers support coming soon". When the diffusers-shaped weights land, add a `Lightricks/LTX-Video-2.3` entry to [backend_service/catalog/video_models.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/catalog/video_models.py) under the `ltx-video` family so RTX 4090 / Linux users get a non-MLX path. Until then, no LTX-2.3 path exists for CUDA.

</details>

<details>
<summary>FU-010 · vllm-swift Apple Silicon backend (watch-closely) — Re-evaluate end of June 2026</summary>

**Status or trigger:** Re-evaluate end of June 2026

**Notes:** [TheTom/vllm-swift](https://github.com/TheTom/vllm-swift) — Swift/Metal vLLM forward pass, Python orchestration only. 2.4× over mlx_lm on Qwen3-0.6B single-request; matches vLLM at concurrency 64. Fills the macOS vLLM gap. **Posture upgraded 2026-05-03** from watch-only after 76 → 238 stars and 1 → 15 forks in ~10 days; v0.3.0 (2026-04-28) shipped Metal Invalid Resource race fix + ~10% TQ MoE perf, v0.2.2 (2026-04-26) added hybrid model batched decode + paged-attention. Single contributor still. Trip-wires for adoption: ≥3 contributors with merged commits OR public benchmark beating mlx_lm at concurrency >1 on Llama-3.x-8B-class (current 2.4× claim is Qwen3-0.6B single-request only).

</details>

<details>
<summary>FU-004 · TriAttention SGLang backend — When/if we adopt SGLang as an inference backend</summary>

**Status or trigger:** When/if we adopt SGLang as an inference backend

**Notes:** Added upstream 2026-04-22 as v0.2.0. No action unless SGLang lands in our runtime.

</details>

<details>
<summary>FU-003 · LongLive integration for Wan 2.1 T2V 1.3B — CUDA platforms (Windows/Linux) only</summary>

**Status or trigger:** CUDA platforms (Windows/Linux) only

**Notes:** Real-time causal long video gen ([triattention/longlive](https://github.com/WeianMao/triattention/tree/main/longlive)). We ship the target model already. Needs: new video backend branch in [backend_service/video_runtime.py](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/video_runtime.py), LoRA weights download, torchrun orchestration, UI affordance for long-clip mode. Flash Attention dep.

</details>
