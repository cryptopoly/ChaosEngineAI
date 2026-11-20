# UI/UX review for v0.10.0, part A: Dashboard, Chat, Chat Models, Server, Settings

Date: 2026-11-20. Branch: the agent worktree, cut from `test/rtx4090` (`131f9031`). Part B (another reviewer) covers Images, Video, Projects, Music, Voice and Tools. Format follows the Voice review ([../voice-v0100/](../voice-v0100/README.md)).

**Headline: the pages hold together and the error states are good, but the numbers a first-time PC owner reads first were not honest.** The Dashboard told a 24 GB RTX 4090 PC to aim for a 70B model, the two chat rows named "24 GB card" said "Won't fit" on that card, and the memory column disagreed with the fit badge beside it. Those are fixed here, with eleven other small defects (section 4). What is left is mostly design: first-run guidance, Windows wording, phone layout (section 5), and five choices for the owner (section 6).

## 1. How it was looked at

- The frontend (Vite, port 8960) against one throwaway backend (port 8961) with `HOME` and `USERPROFILE` on `E:\tmp\uxa\home`. No real settings or model folder was touched; nothing was installed, downloaded, loaded or generated. The backend's answers are this PC's real first-run state (Windows 11, RTX 4090 24 GB, 64 GB RAM, no chat model, no llama.cpp build in the dev checkout).
- Playwright (the pinned `e2e/` install, headless Chromium) at 1280 x 900 and 375 x 812. Recipes beside this file: `99-uxa-pass1` (every page as the backend answers), `pass2` (Chat with a model that looks loaded, composer panels, plan card, job cards, Keep going, Models rows downloading, failed and cancelled), `pass3` (Server with a model, LAN on and events, Settings in used states, backend offline, German), `checks` (panel position, accessible names), `after`. To re-run: copy them and `uxaShared.ts` into `e2e/tests/`, `uxa.config.ts` into `e2e/`, start the two servers, `npx playwright test 99-uxa --config uxa.config.ts`.
- 125 screenshots in [shots/](shots/) (before the fixes unless the name says otherwise; `d-` desktop, `m-` phone) and 11 in [after/](after/). I looked at the desktop set for every page and state and at the phone set for Dashboard, Chat, Models, Server and Storage; the other phone shots were captured, not all opened.
- Loaded-model, download, plan, job, event and offline states are mocked at the browser. German was loaded once.
- The throwaway backend still reads three real things, all read-only: the GPU, the WSL install (the Accelerators page showed the real vLLM venv), and the Hugging Face Hub when the Models search box is used.

Not verified: a real model load, chat turn, download, Keep going run or render; the Tauri window, tray and menus (strings read from `src-tauri/locales/en.yml` only); the update prompt; macOS and Linux; a GPU smaller than 24 GB; a machine with no GPU; colour contrast (not measured); screen readers. There is no command palette in the app, so there was nothing to review there.

## 2. Findings, ranked

Severity: **High** = a first-time user is misled or stopped; **Medium** = slows or confuses; **Low** = polish. "Fixed" means fixed in this branch.

| # | Sev | Finding | Evidence | State |
|---|---|---|---|---|
| 1 | High | **Two chat rows named "24 GB card" said "Won't fit" on a 24 GB card.** Nemotron 3.5 Lightning Q4_0 (measured 233 tok/s on this 4090) and Qwen3.6 35B A3B Q4_K_M (177 tok/s). Before download there is no file to read, so the generic estimate (weights + 10 % + cache) put them at 23.0 and 24.6 GB. | `shots/d-14-models-nemotron-search-expanded.png` | Fixed: the two rows carry the GPU memory measured on the 4090 (18.1 and 20.1 GB) and read "Tight, measured" (`after/d-14-models-24gb-rows.png`). FU-451. |
| 2 | High | **The Dashboard recommended "70B GGUF" with "Headroom for 70B: 65 %" on this PC.** It was judged on system RAM; the Models page judges by the graphics card and calls a 27B "Tight". The 65 % is a constant per tier, not a measurement, and it is shown twice. | `shots/d-01-dashboard.png` | Part fixed: a PC with its own graphics card is judged by the card ("14B", `after/d-01-dashboard.png`). The tier table and the constant percentage remain: P1, FU-452. |
| 3 | High | **The memory cell disagreed with the fit badge on the same row.** A 28.6 GB file read "~23.7 GB" next to "Won't fit" (the verdict used 33.5 GB); a 65.8 GB file read "~31.6 GB". | `shots/d-03-chat-models.png` | Fixed: the cell shows the figure the badge was judged on. The column is still titled "RAM" on a PC where the figure is GPU memory: P1, FU-453. |
| 4 | High | **Benchmarks > Run and the Select Model dialog show a performance preview with no model chosen**: "113.7 tok/s est., Very fast", "Disk: 31.2 GB", "Quality 100 %", "Fits easily". | `shots/d-07-benchmarks-run.png`, `shots/d-12-select-model-dialog-empty.png` | Open, needs a choice (section 6, C). FU-463. |
| 5 | High | **Keep going's panel pushed the whole thread sideways at 1280 px.** It opened to the right of its button, past the thread panel; focusing its first field scrolled the panel, hiding the left of the thread. | `shots/d-13-chat-keep-going-panel.png` | Fixed: it opens leftwards; inside the window at 1280, 1100, 900 and 700 px (`shots/d-30-keep-going-panel-after-fix.png`). FU-455. |
| 6 | High | **First run in Chat: a green "Ready" badge with nothing loaded**, and the model picker said "No models match your search" when no model is downloaded and nothing was searched. | `shots/d-02-chat-empty.png`, `shots/d-12-select-model-dialog-empty.png` | Fixed (`after/d-02-chat-empty.png`, `after/d-12-...`). FU-450. |
| 7 | Medium | **No first step on the landing page.** The Dashboard opens on "Runtime engine", "Warm pool", "Inference activity" and an empty process table. The only pointer is a sentence ("Pick a thread model in Chat or browse a newer family in Chat > Models") with no button, and the GPU is not named anywhere on the page. | `shots/d-01-dashboard.png` | Open: P2. |
| 8 | Medium | **Chat Models opens on filters, not models.** At 1280 x 900 the first row starts 780 px down; at 375 px there are two screens of chips first. 54 rows, no "start here"; the status column is icons only (an orange minus is "not downloaded", a red cross is "failed"). | `shots/d-14-models-top.png`, `shots/m-14-models-top.png` | Open: P2. |
| 9 | Medium | **Names of other desktop apps in the UI**: "LM Studio models folder" and "Share with LM Studio" (Storage, Models rows), "Ollama-compatible apps" (Server). `CLAUDE.md`'s editorial rule forbids exactly these. | `shots/d-06-settings-storage.png` | Open, owner's call (section 6, A). |
| 10 | Medium | **Windows wording.** Storage lists an "MLX cache" folder as MISSING on a PC that can never use it, shows `~/.cache/huggingface/hub` beside `E:\...` paths, and three orange MISSING chips on a fresh install read as faults. Skills' placeholder is `/path/to/skill-folder`. Accelerators says "CUDA only" on every row of a CUDA PC, has an empty Status column and "Each is a single pip install away". | `shots/d-06-settings-storage.png`, `shots/d-21-settings-accelerators-settled.png` | Part fixed: the provider-key text ("0600 permissions"), the Hugging Face token text ("Used by MLX conversion") and the Activity feed's three "MLX ...: no" lines. Rest: P3. FU-454, FU-458. |
| 11 | Medium | **Jargon on first-run surfaces.** "Cross-platform mode will prefer llama.cpp GGUF for broad hardware support"; a "NATIVE F16" chip on every page header; Chat's "Response metadata is collapsed by default, but available per agent turn"; "Their pages and routes stay either way"; "UI scale changes the app chrome density"; "Counts blob bytes once ... naive scanners over-count"; offline panels said "start the sidecar". | `shots/d-01-dashboard.png`, `shots/d-06-settings-general.png`, `shots/d-23-offline-settings-accelerators.png` | Part fixed: the Dashboard subtitle and the four "start the sidecar" lines. Rest: P3. |
| 12 | Medium | **Plan card details.** Expert fields are the browser's own small grey inputs, unlike every other field; "Use for the rest of this chat" does not say what is used; "Approve this" on each row beside "Approve all". Job cards are titled with tool names (`generate_image`, `speak`); the web search badge read "SEAR". | `shots/d-13-chat-plan-card-expert.png`, `shots/d-13-chat-job-cards.png` | The badge is fixed ("WEB"). Rest: P4. |
| 13 | Medium | **Phone (375 px).** Chat puts the thread list above the thread, leaving about 200 px for messages; the Voice popover opens at the top of the page over the header; the Dashboard process table collapses into a stack of four header words; Models needs two screens of filters. No sideways scroll was seen. | `shots/m-10-chat-thread.png`, `shots/m-11-chat-voice-popover.png`, `shots/m-01-dashboard.png` | Open: P5 (the short thread pane is a known item from 2026-11-08). |
| 14 | Medium | **Server.** With LAN access on, the page shows the LAN address but no sentence saying anyone on the network who has the key can use the API. One secret is called "API key", "API token" and "token" on the same page, and Integrations shows a second, different agent token without saying how the two differ. Tiles repeat themselves ("Port 8961 / Active", "Active: None / Idle", "Warm pool: 0 / 0 warm"); the three checkboxes sit above their labels. | `shots/d-20-server-loaded-lan-connect.png`, `shots/d-05-server-overview.png`, `shots/d-06-settings-integrations.png` | Open: P6. |
| 15 | Medium | **A long unbroken word ran out of the reply bubble** (a URL, path or hash would do it). | `shots/d-10-chat-thread.png` | Fixed (`after/d-10-chat-thread.png`). |
| 16 | Medium | **Downloads.** The size cell says 19.1 GB while the bar says "7.5 / 20.3 GB" (the drafter is counted in one, not the other); the progress text appears twice; no speed or time left. Whether a "Won't fit" row warns before download was not checked. | `shots/d-14-models-nemotron-search-expanded.png` | Open: P2. |
| 17 | Medium | **Accessibility gaps.** No accessible label (placeholder only, or nothing) on the Chat composer, the Compare prompt, the HTML Challenge title and prompt, "Run from Hugging Face" and eight Storage path fields; one unnamed button on Integrations; no skip link, so 14 Tab stops pass through the sidebar before the page. Dashboard, Server and the other Settings sections had none. | `checks` recipe output | Open: P7. |
| 18 | Low | **Prompts panel wording.** Title "Prompts", count "5 templates", "Search templates..."; tags repeat the category ("Development · coding · engineering · development"). Its keyboard flow is covered by spec 29 and was not re-run here. | `shots/d-11-chat-prompts-panel.png` | Open: P4. |
| 19 | Low | **English left in other languages.** In German: "Released Nov 2026", model notes, capability chips cut to "Argum." and "Werkz."; the header said "Unknown auf der Festplatte". The release cell also repeats its column title ("Released: Released Aug 2026"). | `shots/d-25-de-chat-models.png` | The header is fixed ("0 GB on disk" with nothing downloaded). Rest: P7. |
| 20 | Low | Chat header: with a long model name the Ready badge wraps to its own line; the row carries six controls. "No models match ... search terms.Clear Filters" has no space before the button. Settings' caption is "Directories and defaults" for eight sections; "Open en on GitHub → [Copy]"; "Save settings" exists in two panels while most controls save at once. The menu says "Preferences…", the app says "Settings". | `shots/d-10-chat-thread.png`, `shots/d-14-models-no-results.png`, `shots/d-06-settings-general.png` | Open: P4. |
| 21 | Low | Diagnostics shows another running copy's log file when its own is missing (seen here: the real app's `chaosengine-backend-8876.log` on a backend on 8961). | `shots/d-06-settings-diagnostics.png` | Open; development setups only. |
| 22 | Low | No light theme (known, app-wide). | | Open. |

What is good and should stay: the terms dialog; the backend-offline banner and that every page still opens (`shots/d-23-*`); a failed download's reason on its row with the exact fix; the fit reason sentence in a row's detail; Activity's failure detail with command line and engine output; the plan card's memory warning in plain words with both numbers; key masking with Reveal; the Agent access explanation; Keep going asking how renders are approved with nothing preselected.

## 3. What a first-time user would be confused by

On a Windows PC with an NVIDIA GPU, in the order they meet it:

1. **"What do I do first?"** The Dashboard shows gauges. The answer (download a small chat model) is one sentence at the bottom left.
2. **"Which model?"** 54 rows after two screens of filters. "Best fit" badges exist, on nine rows.
3. **"It says 70B, the list says 27B is tight."** Fixed to the card's size; the target is still a rough class, not a row they can click.
4. **"Is the server on? What is Warm pool?"** Server and Dashboard use runtime terms without a line of explanation.
5. **"Why is MLX missing?"** Storage and, before the fix, the Activity feed.
6. **"Key or token?"** Server, Integrations and Agent access use both for two different secrets.

## 4. Fixed in this branch

Small, each with a test. New or changed English strings carry `null` in the other ten locales and fall back, with one exception: the Diagnostics offline line is written in the nine shipping languages, because a test requires that page fully translated (my translations, not read by a native speaker).

| Fix | Files | Test |
|---|---|---|
| Rows run on the 4090 carry `measuredGpuGb`; fit uses it on a discrete NVIDIA card before the file is downloaded | `helpers/fit.py`, `catalog/text_models.py` | `test_model_fit.py` (2 tests; every "24 GB card" row is not "won't fit" on 24 GB) |
| Dashboard target judged by the graphics card on a PC | `helpers/system.py` | `test_dashboard_recommendation.py` |
| Activity feed on a PC says "GGUF runtime: yes/no" only | `state/__init__.py`, `dashboard.json` | `test_dashboard_recommendation.py` |
| Memory cell shows the fit verdict's figure | `components/models/adapters/chat.ts` | `chatAdapter.test.ts` |
| No Ready badge with nothing loaded | `features/chat/ChatHeader.tsx` | `chatHeaderReady.test.tsx`, spec 43 |
| Model picker names the empty library | `components/ModelLaunchModal.tsx` | spec 43 |
| Long words wrap in replies | `styles.css` | spec 43 |
| Keep going panel opens inside the thread | `styles.css` | spec 43 (1280 and 1100 px) |
| Models header "0 GB on disk" with nothing downloaded | `features/models/ChatModelsPage.tsx` | spec 43 |
| Web search badge "WEB" | `components/ToolCallCard.tsx` | none (four letters) |
| Copy: provider keys, Hugging Face token, Dashboard subtitle, four offline lines | `SettingsTab.tsx`, four Settings panels, `common.json`, `dashboard.json`, `diagnostics.json` | i18n validate |

Checks run: `tsc --noEmit` clean; vitest 1,998 passed (171 files); `i18n:validate` OK; pytest for fit, catalogue, system, dashboard, route contracts, `test_backend_service.py` and the hub passed; browser spec `43-ui-review-a` 4 of 4 against the review servers. Not run: the full pytest suite, the full browser suite (the other 51 spec files), the backend E2E suite.

Two of the fixes could be argued. The Dashboard change keeps the old tier table and only changes what it is fed; if the owner prefers the catalogue's own pick (P1) it is replaced. The `measuredGpuGb` figures are from runs at 8,192 context without the drafter, and the app loads Nemotron Lightning at 16,384 for the Studio tool list, so its real figure is a little higher than 18.1 GB (not re-measured).

## 5. Plan, in priority order

Sizes are rough.

| # | Work | Size | Why |
|---|---|---|---|
| P1 | **One source for fit numbers.** The Dashboard names the catalogue's best-fit row for this machine (a row to click, with its reason) instead of the tier table and the constant headroom. Hide the performance preview until a model is chosen. Title the memory column for what it measures ("GPU memory" on a PC). Add measured figures to the other rows run on the 4090 (Gemma 4 QAT 12B and 26B, Qwen3.6 IQ4_XS, Bonsai 2). | 1 day | Findings 2, 3, 4. |
| P2 | **First run.** A "Start here" card on the Dashboard (GPU named, one recommended model, Download, then Chat). Chat Models opens on "Recommended for this computer" with filters folded; status with a word beside the icon; one download line with size that matches the row, speed and time left. | 1.5 days | Findings 7, 8, 16. |
| P3 | **Windows wording.** No MLX folder or MLX text on a PC; paths shown the way the system writes them; Accelerators columns that say something; the jargon list in finding 11. | 0.5 to 1 day | Findings 10, 11. |
| P4 | **Chat polish.** Plan card fields in the app's style, checkbox wording; job cards titled in words; header overflow; Prompts wording; the small items in finding 20. | 0.5 day | Findings 12, 18, 20. |
| P5 | **Phone.** Thread list as a drawer in Chat; popovers as bottom sheets (as the Prompts panel already is); Dashboard table as cards. | 1 day | Finding 13. |
| P6 | **Server.** A LAN warning line while it is on; one word for each secret and a sentence on the difference; tiles without the repeats. | 0.5 day | Finding 14. |
| P7 | **Accessibility and language.** Labels on the listed fields, a skip link, release labels and "Unknown" through the catalogue; measure contrast of the small muted text. | 0.5 to 1 day | Findings 17, 19. |

## 6. For the owner to decide

| | Question | Recommendation |
|---|---|---|
| A | "LM Studio" and "Ollama" are named in the UI; `CLAUDE.md` says no other desktop app is named. Keep them as a written exception, or rename? | Keep both as a written exception for interoperability features (the user has to recognise the other app's folder and its API), and say so in `CLAUDE.md`. Renaming them makes the features unfindable. |
| B | Dashboard: replace the size-class target with the catalogue's best-fit row (P1)? | Yes. It is the same answer the Models page gives, and it is clickable. |
| C | The performance preview with no model chosen (Benchmarks, Select Model): hide it until a model is picked? | Yes, hide it. Numbers for nothing are the least honest thing left on these pages. |
| D | Chat Models: open on a short "Recommended for this computer" list with the filters folded (P2)? | Yes for a machine with no model downloaded; keep today's full list once one is installed. |
| E | Build P1 to P3 before the v0.10.0 tag? | Yes for P1 and the Dashboard half of P2 (about 2 days): they are what a new Windows user reads first. P3 to P7 can follow. |

Logged as D-105 on the Decisions page; follow-ups FU-450 to FU-463.
