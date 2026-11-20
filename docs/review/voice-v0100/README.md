# Voice review for v0.10.0 (D-73, second pass)

Date: 2026-11-20. Branch: the agent worktree, cut from `test/rtx4090` (`8f02d7ec`). Follows the first review of 2026-11-09 ([docs/plans/voice-makeover/](../../plans/voice-makeover/review.md)), whose 11 work packages are built. This pass looks at what that build and the Windows GPU work left: Models, Studio (Speak, Voices, Transcribe), Library, dictation, cloning, the GPU dictation models and the two GPU add-ons.

**Headline: on a Mac the Voice section now reads well. On a Windows PC with an NVIDIA GPU it did not: the screens were written for the Mac list and showed Apple-only rows first, promised models the add-on does not give, and offered Download buttons that could only fail.** Six of those are fixed here (section 4). The rest is a plan (section 5) with four choices for the owner (section 6).

## 1. How it was looked at

- The app's frontend (Vite) against two throwaway backends, both with `HOME` and `USERPROFILE` on a temp folder under `e2e/.run/` (git-ignored). No real settings, model folder or GPU was touched; nothing was installed, downloaded, spoken or transcribed.
  - Backend A: plain. The voice runtime is not installed, so its answers are this PC's real first-run state (Windows 11, RTX 4090 24 GB).
  - Backend B: the same code with the install checks patched to "installed" ([fake_installed_backend.py](fake_installed_backend.py)), so the rows, fit numbers, GPU limits, add-ons and the dictation default are the backend's own answers for this GPU. Which add-ons and models look present is set per scenario.
- Playwright (the pinned `e2e/` install, headless Chromium) at 1280 x 900 and 375 x 812. Recipe: [capture-shots.spec.ts](capture-shots.spec.ts).
- 48 screenshots in [shots/](shots/) (before the fixes; `d-` desktop, `m-` phone) and 24 in [after/](after/) (the same scenarios after the fixes).
- **Light and dark:** the app has one theme. `src/styles.css` sets `color-scheme: dark` and has no light palette; a light system theme changes nothing (`shots/d-12-light-system-theme.png`). So every screenshot is the dark theme. A light theme would be an app-wide job, not a Voice one (finding 17).
- German was loaded once to see what stays English (`shots/d-11-de-models.png`, `d-11-de-studio.png`).

Not verified: any audio (nobody listened), a real install or download, the microphone, a voice call, speaker labels on a real transcript, macOS, Linux, a GPU smaller than 24 GB (the GPU-too-small text was seen only on the 72 GB Nemotron row). Install-progress, download and error states were mocked.

## 2. Findings, ranked

Severity: **High** = a first-time user is stopped or misled; **Medium** = slows or confuses; **Low** = polish. "Fixed" means fixed in this branch.

| # | Sev | Finding | Evidence | State |
|---|---|---|---|---|
| 1 | High | **On a PC the speech table opened on an Apple-only row.** The Qwen3-TTS family collapses to its first catalogue row, "Qwen3-TTS 0.6B Base: Not for this machine, Apple Silicon Macs only for now", while the five GPU rows that do run were behind "Show 10 sizes". The install card above it recommends a GPU row the table does not show. | `shots/d-03-models-addons-empty.png` | Fixed: the collapsed family shows the first row that runs here (`after/d-03-models-addons-empty.png`). |
| 2 | High | **Without the speech GPU add-on, every speech row said "Apple Silicon Macs only for now"**, including the "(GPU)" rows, on a PC that has the GPU. Same sentence in the voice picker (Designed group) and as the reason "New cloned voice" is greyed. Nothing pointed at the add-on tick box two panels up. | `shots/d-02-models-base-only.png` | Fixed: GPU rows, the picker and the clone button say "Needs the speech GPU add-on. Tick it under Voice > Models > Voice runtime > Add-ons." (`after/d-02-models-base-only.png`). |
| 3 | High | **Voices offered "Download 3.54 GB" on ten voices that cannot run on a PC** (Turbo, Chatterbox, Orpheus's eight). The backend refuses the download, so each button only produced an error. Speak's model line had the same dead Download. | `shots/d-06-voices.png` | Fixed: no button, the reason instead. |
| 4 | High | **The speech GPU add-on promised "Chatterbox, VoxCPM2 and Orpheus on the NVIDIA GPU". With it installed those three are the rows marked Apple-only;** what it gives today is the five Qwen3-TTS GPU rows, which the label did not name. Size said "several GB"; the 4090 run measured about 5 GB. | `shots/d-03-models-addons-empty.png`; `catalog/voice_models.py` lists only measured CUDA rows | English label fixed. The backend's own label (`voice_env.EXTRAS`) and the docs still name all four: FU-322. |
| 5 | High | **A PC lists nine speech models and ten voices it can never run.** "15 of 15 models" is six real ones (Kokoro and five Qwen3-TTS) plus nine Apple rows; Voices lists Turbo, Chatterbox and eight Orpheus voices under "Natural: from the speech models you downloaded". | `shots/d-05-models-all-sizes-detail.png`, `d-06-voices.png` | Open, needs a choice (section 6, A). |
| 6 | Medium | **Dictation offered a 67.5 GB download for a model this GPU cannot run** (Nemotron 3.5 Transcribe, needs 72 GB), and let it be picked for dictation. | `shots/d-03-models-addons-empty.png` | Fixed: no Download, radio greyed; the reason stays. |
| 7 | Medium | **The dictation default changes silently.** Whisper is "Recommended" and selected; once Qwen3-ASR 1.7B is downloaded it becomes the dictation model with no word on screen, and the Recommended badge disappears because "Downloaded" replaces it. | `shots/d-03-...` then `d-05-models-downloaded.png` | Part fixed: one line says it beforehand. Badge and order: plan P2. |
| 8 | Medium | **Nine dictation cards with no guidance.** Nothing says which is fastest, most accurate or best for a language; licences appear as raw ids ("OpenMDW-1.1", "CC-BY-4.0") with no link or meaning; the Whisper cards show none. The footer says "Whisper uses the language; Parakeet and Nemotron find it themselves" and is silent on Qwen3-ASR and Granite. | `shots/d-05-models-downloaded.png` | Open: P2. |
| 9 | Medium | **A dictation download shows only "37%"**: no bar, no size, no Cancel, no live announcement, though a speech model row has all four. On a 4 GB file that looks stuck. | `shots/d-09-models-download-states.png`; `DictationModelsPanel.tsx` | Open: P2. |
| 10 | Medium | **Mac measurements are shown as this PC's.** Kokoro's Fit cell (0.9 GB peak, "measured on Apple M4 Max") and every quick voice's "first audio 0.13 s (5.7 s cold), 0.87 GB" come from the Mac run; on a PC Kokoro runs through another engine. | `shots/d-06-voices.png`, backend `/api/voice/models` | Open: P3. |
| 11 | Medium | **Natural voices on the 4090 are slower than real time and nothing says so.** "17.2 s per 10 s of speech" sits in a green chip; the install card calls the quick voices "fast but flat" without saying the natural ones take about 1.7 times the clip's length here (the Mac is about 0.25). | `shots/d-06-speak.png` | Open: P3. |
| 12 | Medium | **Runtime size copy disagrees.** Models says "About 2 GB", the Studio gate "about 2.3 GB", and with the speech GPU add-on the folder measured 8.0 GB on the 4090 run (about 11 GB with both add-ons, going by the labels; not measured). The install runs for minutes with a bar but no percent, size or time, and the add-on being installed is not ticked after a reload. | `shots/d-01-...`, `d-08-install-running.png` | Open: P4. |
| 13 | Medium | **Speak with no voice at all** (Kokoro left out, nothing downloaded): the card says "The quick voices are fast but flat" when there are none, the style note says "The small model ignores style" when no model exists, and one model had three Download buttons. | `shots/d-04-speak-no-voice.png` | The duplicate "Download ... for style" button is fixed; the copy is P4. |
| 14 | Medium | **Voices is very long.** 3,500 px at desktop and 7,760 px at 375 px with all nine quick languages open; the eight unusable Orpheus rows sit above the quick voices. The layout itself holds at 375 px (no sideways scroll). | `shots/m-06-voices-new-clone.png` | Open: follows from A; then collapse the quick languages (P5). |
| 15 | Low | **Backend text stays English in every language**: model summaries, language lists, notes ("Untested here."), voice descriptions, "measured on ...". The new English strings fall back as designed. | `shots/d-11-de-models.png` | Open: P6 (known pattern, D-97). |
| 16 | Low | **Accessibility gaps.** Install and dictation progress have no `role="progressbar"` or live text; delete confirms are the system `confirm()` box; the voice-sample picker is the browser's own "Choose File / No file chosen" (not translated, not styled); the collapsed-family link says "Show 10 sizes" when it reveals fewer. Keyboard use of the drawer and menus was covered by the existing specs and still passes. | code, `shots/d-06-voices-new-clone.png` | Open: P6. |
| 17 | Low | **No light theme** anywhere in the app. | `shots/d-12-light-system-theme.png` | Open, app-wide; not a Voice task. |
| 18 | Low | `e2e/tests/14-voice-speak.spec.ts` (Voices) fails on Windows: it expects the Mandarin add-on button, which only the Mac lane has. 66 of the 67 voice browser tests pass here. | this run | Open: FU-327. |

What is good and should stay: Speak's first screen (text, one voice row, style, Generate, all above the fold); the Change voice drawer; the main-voices strip and Fallback card on Voices; the runtime panel's parts line and add-on ticks; Delete on every model with a confirm that names sizes; the clone form's permission tick and plain statement; the GPU-too-small sentence with both numbers; the Library card (voice, kind, AI-made label, Speak again); Chat's Voice popover with its Needs list. Error text for a failed install and a failed download is specific and stays on the row (`shots/d-08-install-failed.png`, `d-09-...`).

## 3. What a first-time user would be confused by

On a Windows PC with an NVIDIA GPU, in the order they meet it:

1. **"Which boxes do I tick?"** Three add-ons, two of them named after model families. Nothing says the first gives quick voices only, the second gives natural voices and cloning, the third gives better dictation, or that the total is 8 GB or more, not "about 2 GB".
2. **"Why does it say Apple only? I have a 4090."** (Fixed for the rows that the add-on unlocks. Still true for Chatterbox, VoxCPM2 and Orpheus, which really are Mac-only today and take up half the table.)
3. **"Which dictation model?"** Nine cards, one badge. They cannot know that Qwen3-ASR is the intended pick, that only Whisper Small and Base translate, or what OpenMDW means.
4. **"Is it downloading?"** A bare percentage on dictation cards.
5. **"Why is my voice so slow?"** A 10 s line takes 17 s with a natural voice; the chip is green.
6. **"What is the difference between Base, CustomVoice and VoiceDesign?"** The Makes chips answer it (Clone / Natural voices, Style / Design) but the names lead. The first review raised this; grouping fixed the Mac view, and the PC view doubles the list with "(GPU)" twins.
7. **"Where do I change the dictation model for Chat?"** Chat's Voice popover names it with a Change link (`shots/d-10-chat-voice-popover.png`; the link was not clicked). Not a problem.

On a Mac, from the first review's list, only items 3, 4 and 6 remain, in milder form (five dictation cards, no GPU twins).

## 4. Fixed in this branch

All small, each with a test. English strings only; the other ten locales carry `null` and fall back.

| Fix | Files | Test |
|---|---|---|
| A collapsed family shows the first row that runs here, else the first GPU row, else the first row | `models/voiceModelsLogic.ts` | `voiceModels.test.tsx` |
| GPU rows on a PC without the add-on say to tick the add-on; so do the picker's Designed note and the greyed New cloned voice button. The backend row now carries `lanes` | `voice_clone.py`, `types/voice.ts`, `voiceModelsLogic.ts`, `SpeechModelsPanel.tsx`, `VoicePicker.tsx`, `VoicesScreen.tsx` | `voiceModels.test.tsx`, `test_voice_qwen_cuda.py` |
| No Download on Voices or on Speak's model line for a model that cannot run here; the reason is shown | `VoicesScreen.tsx`, `SpeakScreen.tsx` | `VoicesScreen.test.tsx` |
| Speak no longer shows a second "Download ... for style" button for the model the first button already downloads | `SpeakScreen.tsx` | covered by the existing Speak specs |
| A GPU dictation model this GPU cannot run has no Download and cannot be picked | `DictationModelsPanel.tsx` | `voiceModels.test.tsx` |
| "Qwen3-ASR 1.7B becomes the dictation model once it is downloaded, unless you pick another." while none was picked by hand | `DictationModelsPanel.tsx`, `voiceModelsLogic.ts` | `voiceModels.test.tsx` |
| Add-on label: "Speech GPU add-on: Qwen3-TTS natural voices, voice design and cloning on the NVIDIA GPU (about 5 GB)" | `VoicePackPanel.tsx`, `src/locales/*/studio.json` | i18n validate |

Checks run: `tsc --noEmit` clean; vitest 1,970 passed (169 files); `i18n:validate` OK; pytest for the voice and route-contract files passed (one expected failure that was already marked); the seven voice Playwright specs: 66 passed, 1 failed (finding 18, not caused by these changes: it asserts a Mac-only button). Not run: the full pytest suite, the full browser suite, the backend E2E suite, a real install.

The add-on label is the one fix that could be argued: if the Chatterbox, VoxCPM2 and Orpheus GPU rows are measured and listed before the tag, the label should name them again (FU-322).

## 5. Plan, in priority order

Sizes are rough. P1 depends on choice A.

| # | Work | Size | Why first |
|---|---|---|---|
| P1 | **A PC shows only what a PC can run.** Hide Apple-only speech rows and their voices on the portable lane (or fold them under one closed "Mac only" line), so the table is Kokoro plus the GPU rows and the counts are true. Drop "(GPU)" from names once the twins are gone. Collapse to the recommended natural row, not the first. | 1 day | Removes findings 5 and 14 and half of 6 in section 3. |
| P2 | **Dictation panel.** Order and badge: the default-to-be first with "Recommended for this GPU", then one line per card on what it is for (fast, most languages, translates). Licence as a chip with a link, same as speech rows; add it to Whisper. Download with bar, size and Cancel through the shared download control. Check which models use the Language setting and fix the footer. | 1 to 1.5 days | The newest screen and the least guided (findings 7, 8, 9). |
| P3 | **Honest numbers per machine.** Do not show Mac timings as this PC's: measure Kokoro on the portable lane or label the figures "measured on a Mac". Colour or label a voice that is slower than real time; say so on the install card. | 0.5 day plus a measuring run on the 4090 | Findings 10, 11. |
| P4 | **Runtime install.** One size figure per lane with the add-ons' sizes summed as they are ticked; percent and current step on the bar; keep the chosen add-ons ticked through a reload; one sentence per add-on on what it is for. First-run Speak copy when no voice exists. | 1 day | Findings 12, 13 and item 1 in section 3. |
| P5 | Voices: quick languages collapsed except the UI language; phone density. | 0.5 day | Finding 14. |
| P6 | i18n and accessibility: translate catalogue text or key it, progress roles and live regions, the app's own confirm dialog and file picker, the "Show N sizes" count. | 1 to 2 days | Findings 15, 16. |

Mock-up for P1 and P2 (text only; the layout is the existing table and cards):

```
Speech models                                   6 models · 1 downloaded
  Kokoro            Quick voices            0.33 GB   Fits        Part of the runtime
  Qwen3-TTS  Natural voices, style, design and cloning · Apache 2.0      Show 4 more
    1.7B CustomVoice  Natural voices, Style  4.52 GB   Fits  slower than real time   [Download]
  Runs on Apple Silicon only: Chatterbox, VoxCPM2, Orpheus                    Show

Dictation models                                            Language [Detect automatically]
  Qwen3-ASR 1.7B   RECOMMENDED FOR THIS GPU   30 languages, finds the language
                   4.09 GB · Apache 2.0 (link)             (o) Use for dictation  [Download]
  Whisper Large v3 Turbo   IN USE NOW   99 languages
  ...
  Downloading: [=======     ] 37 %  0.58 of 1.58 GB   [Cancel]
```

## 6. For the owner to decide

| | Question | Recommendation |
|---|---|---|
| A | On a PC, what happens to speech models and voices that only run on a Mac? | Hide them behind one closed line. They are 9 of 15 rows and 10 voices that can only say no. Showing them was useful while the PC had nothing; it now has six rows of its own. |
| B | Chatterbox, VoxCPM2 and Orpheus on the GPU for v0.10.0? The add-on installs their packages (most of its 5 GB) but no row is offered until measured. | Measure and list them on the 4090 before the tag (the packages are already paid for), or leave them out of the add-on. Not both states. |
| C | Should Qwen3-ASR take over as the dictation model the moment it is downloaded? | Keep the rule, but say it (done) and badge it; do not switch a user who is mid-session without a line in Chat's Voice popover. |
| D | Build P1 to P4 before the v0.10.0 tag? | Yes for P1 and P2 (about 2.5 days): they are the Windows first-run. P3 to P6 can follow. |

Logged as D-102 on the Decisions page; follow-ups FU-320 to FU-327.

## 7. Built after the owner's decisions (2026-11-20)

A, C and D decided as recommended; B is open. P1 and P2 are built (FU-321, FU-323), with the "Word timings" row (D-101, FU-310), one source for the runtime sizes and the FU-327 test fix. Before and after shots, same recipe and scenarios: [p1p2/before/](p1p2/before/) and [p1p2/after/](p1p2/after/) (`d-05-models-downloaded`, `d-09-models-download-states`, `d-06-voices`, `d-06-transcribe`, `d-01-*`, phone `m-` versions; `*-05-models-mac-only-open` is after only). Details and checks: `docs/progress/rtx4090-progress.md`, 2026-11-20 "Voice P1 and P2". P3 to P6 are untouched.
