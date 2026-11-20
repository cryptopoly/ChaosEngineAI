# UI/UX review B for v0.10.0: Image, Video, Projects, Music, Tools, galleries

Date: 2026-11-20. Branch: agent worktree off `test/rtx4090` (not pushed). Reviewer: Claude (agent).
Scope: Image Studio and Models, Video Studio and Models, Projects (Storyboard), Music and Audio Studio, Models and
Library, Benchmarks, Tools (Conversion, Fine-Tuning), the image, video and music galleries, download and error
surfaces in those pages, and the song to storyboard to film path. Dashboard, Chat, Server, Settings, Library and
Setup belong to the other reviewer. Voice was reviewed in [voice-v0100](../voice-v0100/README.md); only its links
into Projects were looked at here (the voice lane in the Mixer, the cloned-voice line on Export).

## How the screenshots were made

Windows 11, RTX 4090 (24 GB), 64 GB RAM. Vite on the worktree's code against two throwaway backends, each with
`HOME` and `USERPROFILE` on a folder under `E:\tmp\uiux-b`, a dummy token, free ports. Nothing was installed,
downloaded, generated or deleted, and the GPU was not used.

- **World A** (`d-A-*`, `m-A-*`): nothing downloaded or installed. This is the real first run of this PC.
- **World B** (`d-B-*`, `m-B-*`): the same, with `HF_HUB_CACHE` on the PC's model folder and `HF_HUB_OFFLINE=1`, so
  the rows that are on disk read as downloaded. Every request that could change anything outside the throwaway
  home was answered 503 in the browser.
- Mocked in the browser, as the e2e specs do it: the music pack as installed (to reach the Studio forms), the
  Copy now panel, refusals (409), library and gallery items, the MiniMax H3 gate (the backend ships with
  `H3_RELEASED` off, so the screen's text is the spec 24 copy of `licence_gate.py`). Projects were seeded through
  the backend's own API with a 1 x 1 red still and a 2 s tone.

`d-` is 1280 x 900, `m-` is 375 x 812; tall pages were captured by growing the viewport. 130 screenshots in
[shots/](shots/), all looked at (the long ones in crops). Nine screenshots taken before the fixes are kept in
[before/](before/). Recipes: [capture-baseline.spec.ts](capture-baseline.spec.ts) (every page, both worlds),
[capture-states.spec.ts](capture-states.spec.ts) (states), [capture-extra.spec.ts](capture-extra.spec.ts)
(refusal, Copy now, shot editor, two measurements), [playwright.review.config.ts](playwright.review.config.ts).

Numbers were compared with `docs/progress/rtx4090-progress.md` (2026-11-17 to 2026-11-20) and with the catalogue
answers of the two backends (sizes on disk in world B).

## Verdict

The new rows are described honestly where the text was written this month: the row details for FLUX 3 Klein,
MOSS-Image, Kandinsky 6, LTX-2.5 and LTX-2.3 match the measured figures, the music fit numbers match the 4090
table, the licence screens (Stable Audio 3, MiniMax H3) are clear, and Projects is the most finished area.
The weak part is the older shell around the new rows: three places showed a number that contradicted another
number on the same screen, and several pages still describe a Mac. Seven of those were small enough to fix here.
Four High items are left; all four are about numbers a person uses to decide what to download or run.

## Fixed in this pass (FU-470)

| # | What was wrong (seen on the 4090 PC) | Fix | Before / after |
|---|---|---|---|
| 1 | Video Studio said "This model runs on its own MLX engine (ltx-2-mlx) on the Apple GPU", `ENGINE: LTX-2-MLX`, `DEVICE: MLX` for LTX-2.3 and LTX-2.5 on the NVIDIA card. The banner keyed on "fixed sampler", which the distilled diffusers rows also have. | Only the ltx-2-mlx engine gets that banner. | [before](before/d-B-video-studio-mlx-banner.png) / [after](shots/d-B-video-studio-ltx23.png) |
| 2 | Image and Video Models: the RAM/VRAM figure and the fit chip came from two estimators and disagreed: "~67 GB" beside "Fits" for FLUX.1 Dev GGUF Q4 (the verdict is 8.8 GB), "~13 GB" beside "Won't fit" for FLUX.1 Dev (30.9 GB), "~49 GB, Fits" for SD 3.5 Medium. | The column shows the verdict's own number when the backend gave one; the page's heuristic stays for rows without a verdict. | [before](before/d-B-image-models-67gb-fits.png) / [after](shots/d-B-image-models.png) |
| 3 | "Install the music pack. About 1.6 GB" on the NVIDIA lane. Measured on this PC: 6.6 GB (the CUDA build of PyTorch). The sounds pack already said 6.4 GB. | 6.6 GB on the CUDA lane, in the banner and in the Studio sentence (new English string, null elsewhere). | [before](before/d-A-music-models-pack-1.6gb.png) / [after](shots/d-A-music-models.png) |
| 4 | Music and Sounds estimate strip: "Peak 8.6 GB of 63.8 GB". The peak is VRAM, the total was system RAM. Image and Video say "of 24 GB". | The backend sends the card's memory as the total on the CUDA lane (shared-memory GPUs and Macs unchanged). | [after](shots/d-A-music-studio-error.png) ("Peak 8.6 GB of 24 GB"); no before shot was kept |
| 5 | LTX-2.5 rows (gated, no token): chip "Not for this machine" on a model measured at 19.5 GB on this machine. | The chip says "Needs access" for a gated row nobody can download yet; the tooltip keeps the reason. | [before](before/d-A-video-models-not-for-this-machine.png) / [after](shots/d-A-video-models.png) |
| 6 | Every accelerator card showed an internal tracker id as a chip ("FU-016", "FU-023"). | Chip removed (also on Settings > Accelerators, same component). | [before](before/d-A-video-studio-fu-chip.png) |
| 7 | A failed image or video run ended "... (6.7 GB).. Check the Logs tab (filter: images)". There is no Logs tab since D-88, and the full stop was doubled. | "Server > Activity has the backend details.", one full stop. | [before](before/d-B-image-studio-409-logs-tab.png) |
| 8 | Copy: "Configure a profile on the left" (the page is one column at 1280 px); "create the first saved image artifact for this branch" (Studio and Gallery); Fine-Tuning told people to "use the fine-tuning feature below" above a line saying training is not built; `/path/to/file` as the placeholder on Windows; licence shown as "Mit". | Plain English sentences; "Full path to the file"; "MIT". English only: the other locales keep their older wording (FU-479). | [before](before/d-A-bench-run-on-the-left.png), [before](before/d-A-image-gallery-artifact-branch.png), [before](before/d-A-finetuning-feature-below.png) |
| 9 | e2e spec 04 still expected the D-80 sentence "Music generation runs on Apple Silicon Macs in this version". | Spec text updated to the current sentence. | |

Checks: `npx tsc --noEmit` clean; `npx vitest run` 1,998 passed (170 files); `npm run i18n:validate` OK;
`pytest tests/test_music_routes.py -k "EstimateTotal or estimate"` passed; Playwright specs 01, 02, 04, 10, 24, 28,
42 passed (24 tests) on this PC.

## Findings left open

Ranked. "Shot" paths are under [shots/](shots/).

### High

**H1. The Size column overstates most downloads before they start, by up to 100 times.**
The column prefers the sum of every weights file in the Hugging Face repo over the catalogue's own size. For a repo
that holds several versions, that is not what the app downloads. Rows on disk in world B show the catalogue figure
is the honest one:

| Row | Shown before download | Catalogue | On disk after download |
|---|---|---|---|
| LTX-2.5 distilled 4-bit | 162.4 GB | 71.1 GB | 66.2 GB |
| FLUX 3 Image Klein (diffusers) | 40.3 GB | 20.3 GB | 19.2 GB |
| FLUX.2 Klein 4B | 22.1 GB | 16.0 GB | 14.9 GB |
| Qwen-Image-2.1 Turbo sd.cpp Q4_K_M | 60.4 GB | 4.2 GB | 3.9 GB |
| FLUX 3 Klein sd.cpp Q8_0 | 25.4 GB | 6.2 GB | 5.7 GB |
| LTX-Video | 236.4 GB | 2.0 GB | not downloaded |
| Mochi 1 Preview | 124.3 GB | 10.0 GB | not downloaded |
| FLUX.1 Dev GGUF Q4_K_S | 53.9 GB | 6.8 GB | not downloaded |

Video Studio repeats it beside the Download button: "236.4 GB weights" for LTX-Video. The row detail shows
"Catalog size 71.1 GB" under a 162.4 GB column. Shots: `d-A-video-models.png`, `d-A-video-studio.png`,
`d-B-video-models-rows-open.png`. Not fixed here: the order was chosen on purpose ("curated sizeGb, often stale"),
and the right number is the sum of the files the download's own patterns match, which the backend should compute
(D-108, FU-471).

**H2. Studio estimates for rows measured this month disagree with the measurements.**
- FLUX.1 Schnell Nunchaku INT4 in Image Studio: "About 2 s · Peak 10.2 GB of 24 GB". Measured through the app on
  this card: 5.9 s warm, about 18 GB peak (CPU offload on a 24 GB card). The Models row says "~10 GB, Fits est.".
  Shot: `d-B-image-studio.png`.
- FLUX 3 Klein: the Studio opens at 24 steps ("Balanced") while its own Expert panel says "distilled to 8 steps;
  more steps cost time in proportion and rarely look better". The 13 s estimate is for the 24 steps; the measured
  7.0 s is for 8. The same panel offers CFG decay for a model whose note says "No guidance or negative prompt".
  Shot: `d-B-image-studio-flux3.png`.
- The warning "Heads up, may struggle on this device ... Generation may run slowly or fail" sits on FLUX 3 Klein
  and LTX-2.3, both measured working on this card and marked "Tight" on the Models page. Two verdicts for one row.
- Known and unchanged: Klein 2048 px shown 39 % low, LTX-2.5 above 768x512 shown 22 to 33 % high (progress file,
  2026-11-19).
FU-472.

**H3. Benchmarks: the Run benchmark button is out of sight, and the preview shows numbers for no model.**
At 1280 x 900 the settings sit in an inner scroll box 465 px tall holding 774 px; the button is at y 758 and the
box ends at y 590. At 375 px the box shows 277 of 1,004 px. Nothing hints that the box scrolls. With no model
picked the page still shows "113.7 tok/s est.", "Disk: 31.2 GB", "Fits easily". Perplexity and Task accuracy are
offered in the mode list on a PC, where the note says they run on MLX models only. Shots: `d-A-bench-run.png`,
`m-A-bench-run.png`, `d-A-bench-run-viewport.png`. FU-473.

**H4. Projects offers "Make 3 missing" and "Make this shot" with no image or video model installed.**
The same panel says "No image model is installed" and "No video model is installed"; the primary button stays
live on Overview, Storyboard and the shot editor, and Plan offers "Make them in Storyboard" and "Rework with the
model" with no chat model. The shot editor shows "ESTIMATE: UNKNOWN UNTIL THE FIRST RUN" beside an "Estimated"
chip. Shots: `d-A-project-overview.png`, `d-A-project-storyboard.png`, `d-A-project-shot-editor-vocal.png`.
Not verified: what the click answers (nothing was generated in this review). FU-474.

### Medium

**M1. Image and Video Models say "No image models in the catalog yet. Check that the backend is running." while
the catalogue loads.** On a first open the list was empty for more than 2.5 s with the backend online and the
counter at "0 of 0 models". Shot: `d-A-image-models.png` (taken 2.5 s after opening the page on a freshly started
backend; the catalogue answers in 7 ms once warm). A loading state is missing. FU-476.

**M2. The Image Studio default can be a row that cannot run.** In world B the Studio opened on FLUX.1 Schnell
Nunchaku INT4, "Installed", Generate live, while the Nunchaku package is not installed (its Install card is on the
same screen). The run is refused. The refusal dialog says "Image generation failed" twice, adds "Adjust the prompt
or runtime settings, then try again" (not the cause), has no Download or Install button, and the same text is
also in a banner at the top. The backend sentence says "Download it in Image Studio first"; the download is on
Image Models. Shots: `d-B-image-studio.png`, `d-B-image-studio-409.png`. FU-478.

**M3. Video Studio: the first clip is 1.4 s, the header says "4.0s clip".** Frames default to 33 at 24 fps for
every row; the chip row above shows the model's 4.0 s. The SD preset is highlighted and labelled "480p · 896x512"
while the size in Options is 768 x 512. Sizes past the model's limit are disabled but look the same as the
others. For LTX-2.3 4-bit, Expert offers an unticked "4-bit (NF4)" switch on a row that is already 4-bit.
Shot: `d-B-video-studio-ltx23.png`.

**M4. Lip sync can only be reached from the shot editor.** Video Studio has a start image and an end image, no
audio input, and nothing on the LTX-2.3 row or in the Studio says a vocal can drive the mouth. The field in the
shot editor is clear. It took a 2 s WAV without comment although its own text says the clip "lasts 3 to 8 s".
Shots: `d-A-project-shot-editor-vocal.png`, `d-B-video-studio-ltx23.png`. D-109.

**M5. Phone (375 px).** The MiniMax H3 licence screen puts Not now and Confirm under the fold of the dialog since
the D-99 text was added; the dialog scrolls, but the buttons are cut in half with no hint
(`m-B-video-h3-licence-ticked.png`). Music and Audio Models: licence chips, "APPLE SILICON" and the "Accept
licence" button run off the right edge of the card (`m-A-music-models.png`). FU-477.

**M6. Runtime name "Stub diffusion pipeline".** Ten image rows (FLUX.1 Schnell and Dev, SD 3.5, SDXL) carry this
as their runtime; Image Studio prints it in the header: "FLUX.1 Schnell / Stub diffusion pipeline / 0 saved
outputs". It reads as "this is fake". `d-A-image-studio.png`. Left alone because `platform.test.ts` and the gate
logic read the string.

**M7. Status and actions are icons only.** Not downloaded is an amber circle with a dash, downloaded a green tick,
then up to four icon buttons. Music rows use a text pill ("NEEDS LICENCE") in the same column. Tooltips exist;
on a touch screen they do not.

**M8. Three warnings for one fact in Image Studio (world A).** "FLUX.1 Schnell is not installed locally" appears
in the runtime panel and again under the model picker, next to "You don't have any image models downloaded yet",
with the chip "REAL LOCAL GENERATION AVAILABLE" above them. `d-A-image-studio.png`.

### Low

- L1. Words: the sidebar says "Projects", buttons say "Open in Storyboard" and "Send to storyboard" (lower case);
  "CUDA" and "APPLE SILICON" chips on music rows of a PC; "WATCHING UPSTREAM" rows under "Latest Releases" with a
  greyed download and no explanation; licence "Other" on rows whose note names the licence (FLUX 3 Klein, LTX-2.5).
- L2. Fit tooltip says "Measured on this machine." for catalogue figures measured on the project's 4090; the
  reason line above it says "measured on a 24 GB NVIDIA card". Wrong on any other PC. The row detail prints
  "(measured)" twice.
- L3. Gallery headers carry an "Installed" button that opens the Models page. Video Gallery has no search or sort;
  Image Gallery has both. Music Library and Music Studio show two "Open Studio" / "Open Library" buttons each.
- L4. Fine-Tuning (experimental tab): "Fine-tune models with QLoRA on Apple Silicon (MLX) or via llama.cpp" on a
  PC, above "Training is not built yet".
- L5. The Copy now panel keeps a live "Retry" button under a running copy (`d-A-music-copy-running.png`).
- L6. Mixer: "3 shots have no clip yet and are left out of the preview" under a preview that holds shot 1's still
  for 4 s. "Open in Storyboard" there and "Edit the brand kit" on Export are raw underlined links; everything else
  is a button. The shot editor's number fields use the browser's default input style.
- L7. H3 licence screen states the territories twice (bullet, then "Not licensed for use in").
- L8. `musicUnsupported.soundReason` ("Sound effects run on Apple Silicon Macs in this version") is no longer sent
  by the backend; the string and its test are dead.
- L9. Tools > Conversion is hidden off macOS, so Tools is empty on a PC unless experimental tabs are on. Correct,
  but e2e spec 27's Conversion half cannot run on Windows.

## What holds up

- Row details on Image and Video Models: plain sentences, measured figures, licence terms, what a smaller card
  does (`d-B-image-models-rows-open.png`, `d-B-video-models-rows-open.png`).
- Music fit numbers against the 4090 table: turbo 8.6 GB (measured 8.1 to 8.6), XL 14 (14.3), Stable Audio 3
  Small 2.0, Medium 8.4, Open Small 3.6, Open 1.0 14 (13.8).
- Licence screens: Stable Audio 3 with the Gemma terms, MiniMax H3 with the new tick sentence, the hosted-service
  notice and both licence links (`d-A-music-licence-screen.png`, `d-B-video-h3-licence-ticked.png`).
- Copy now panel: files, size, both drives, two ways to avoid the copy, progress with file name and time left.
- Projects: Overview's next step, Plan's song section (the music library is one dropdown away), Export's
  "Before you export" list, the forced AI tag line for a cloned voice.
- Long titles and prompts wrap in the library and gallery cards at both widths; no page scrolled sideways.

## Plan, in order

1. **Before the tag:** H1 (download size from the download's own file list; until then prefer the catalogue size
   for rows not on disk), H3 (one scrolling page or a pinned Run bar; no preview without a model; hide MLX-only
   modes off a Mac), H4 (replace the Make buttons with "Open Image Models" / "Open Video Models" while nothing is
   installed), M1 (loading state), M2 (never default to a row whose engine is missing; refusal dialog with the
   one useful button).
2. **Before the tag if time allows:** H2 (Nunchaku estimate from the measured offload path; selecting a row applies
   its recommended steps; one verdict per row), M3, M5, M6.
3. **After the tag:** M4 (D-109), M7, M8, the Low list, translations of the English strings changed here (FU-479).

## Questions for the owner

- **D-108.** Size before download: show the catalogue size until the backend can sum the real file list?
  Recommended: yes. It was within 10 % of the disk figure on every row that is on this PC.
- **D-109.** Lip sync in v0.10.0: Storyboard only, or also an audio input in Video Studio? Recommended: Storyboard
  only, plus one line on the LTX-2.3 row and in Video Studio that points to it.
- Should a distilled row's recommended steps be applied when the row is picked (FLUX 3 Klein opens at 24, wants
  8)? Recommended: yes; it is the difference between 7 s and 13 s.
- Is "Stub diffusion pipeline" safe to rename to "diffusers" in the catalogue? It needs a look at the platform
  gate that reads the string.

## Not verified

- No model ran. Every time, size and memory claim was compared with the progress file and the catalogue, not
  re-measured.
- The after-state of the music estimate strip was checked by unit test and by the route's answer, not on a card
  with the music pack installed.
- Real downloads, pause and resume, the Activity entries they write, and a failed download row: not exercised
  (spec 10 covers the mocked progress rows; my own mock of that state was wrong and its screenshot was dropped).
- A real MiniMax H3 gate from the backend (it is off), a real Nunchaku refusal text end to end, a real Copy now.
- macOS and Linux, cards under 24 GB, the packaged app, screen readers, keyboard-only use of these pages.
- Tools > Conversion (not listed on Windows), Benchmarks with a real model and History with real runs.
- The other nine locales: three new strings are English only, and the English sentences changed here are older in
  the translations.
