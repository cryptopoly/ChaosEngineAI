# What is left, and v2 of the home studio and the agent bridge

Status: draft 1 (2026-11-07), branch `plan/remaining-and-v2` (cut from `feat/home-studio`). Plan only; no code changed. Decision: D-46.
Read with: [home-studio](home-studio.md), [home-studio-ux](home-studio-ux.md), [speed-quality](speed-quality.md), [agent-bridge](agent-bridge.md), [hub status](../hub/status.md).
Being written on other branches (referenced here, not duplicated): `docs/plans/ffmpeg-download.md` (`research/ffmpeg-lgpl`), `docs/research/voice-cloning.md` and `docs/research/new-video-models.md` (`research/models-cloning`).

**Sizes:** S = under a day, M = 1 to 3 days, L = 4 or more days of agent time. **Who:** Haiku = chores, Sonnet = bounded work and reviews, Opus = design-heavy or risky work (working agreement).

## In one screen

- **v0.10.0 is built but not shippable yet.** Code and tests are green on `feat/home-studio`; the gaps are proof (browser E2E, the packaged app, Windows), packaging (ffmpeg, the wavelets fix), and release chores (translations, docs, notes). About **8 to 12 agent days** plus your test time, mostly parallel.
- **Three things only you can unblock:** say go on the v0.9.10 merge; pick how ffmpeg ships in v0.10.0; test the Windows build on the 4090 once we hand you one.
- **Home studio v2** (ranked in Part 2): keep models warm (faster storyboards), captions in the Mixer, voice cloning with consent, sound effects, then extend / interpolate / upscale.
- **Agent bridge v2** (Part 3): tidy and test the tool list, scopes on the token, a download tool that needs your click in the app, progress and resources, then the 4090 as a remote worker.
- Stream table, tonight's start list and your decisions are at the end.

---

# Part 1. What is left to ship the home studio (v0.10.0)

## 1.1 Where it stands

Built and checked on this Mac: Music (Models, Studio, Library, `make_music`), Storyboard (Projects, Board, Mixer, export, `.chaosfilm`), calmer Image and Video Studios, terms screen (D-41), Chat back on top (D-42). 2826 Python and 1155 vitest tests pass; E2E phases 9 to 11 cover the backend. Music and Storyboard are hidden behind `HOME_STUDIO_ENABLED = false` in `src/constants/features.ts`.

Not proven: the new screens in a scripted browser run, the packaged app, Windows and Linux, anyone listening to the output.

## 1.2 The work list

Ordered by when to start. "Exit test" is how we know it is done.

| # | Item | Why it blocks | Size | Depends on | Who | Exit test |
|---|---|---|---|---|---|---|
| R1 | **Drop `pytorch-wavelets`** (D-37, branch `fix/drop-pytorch-wavelets` exists) and smoke-test a track | Licence question on a shipped package | S | none | Sonnet | Music venv builds without it; a 30 s track renders; FU-138 closed |
| R2 | **CI on Windows and Ubuntu for `feat/home-studio`** (the existing `build.yml` test jobs) | Catches path and shell bugs before you spend an evening on them (PR 119 failed on `\pack` vs `/pack`) | S | none | Sonnet | All three OS test jobs green, or each failure has a fix or an FU row |
| R3 | **Browser E2E for the new screens** (E2E phase 12, reserved): Music Studio make / play / save / delete; Storyboard create project, add 3 shots, import a clip, reorder, lock, undo, Mixer add music lane, preview, export; "Use in storyboard" from a gallery | CLAUDE.md: no user-visible feature without an E2E check; the switch cannot flip before this | M | owner OK on the test tool (decision 2) | Sonnet; Opus reviews the checks | `./scripts/e2e_test_suite.py --phase 12` passes against a dev build with the switch on; runs headless; under 5 minutes with tiny fixtures (no model runs, generation stubbed by importing files) |
| R4 | **Flip the switch on the branch** and run the full suite (phases 0 to 12) | The release is the switch | S | R3 | Sonnet | Full E2E green with `HOME_STUDIO_ENABLED = true`; one live run of the 8-shot film from the UI with no terminal (the definition of done, timed) |
| R5 | **ffmpeg for the packaged app.** The release workflow does not install `imageio-ffmpeg`, so a packaged app has no pinned ffmpeg for the Mixer. Long-term answer: the LGPL download manager in `docs/plans/ffmpeg-download.md` (D-35) | Without it, export fails in a packaged app | M (interim) / L (downloader) | decision 3; the ffmpeg plan | Opus designs, Sonnet builds | Packaged app on macOS and Windows exports a film; the ffmpeg used is the pinned one (logged with its SHA-256); notices match what ships |
| R6 | **Packaged macOS app run** (FU-121): build the dmg, open it, check Tauri dialogs (Choose file, Reveal in Finder), music install from inside the bundle, the staged runtime has `music_requirements.txt`, the music worker and storyboard code | Never run with any of this | S to M | R1, R5 interim | Sonnet builds; you open it once | Checklist in 1.4 ticked on macOS; `runtime-*.manifest.json` has `llamaServer` and no warnings |
| R7 | **Windows build for you** (unsigned installer from the manual Build workflow) plus the checklist in 1.4 | You test on the 4090 PC | S for us, 1 to 2 h for you | R2, R5 | Sonnet | You tick the checklist; failures come back as FU rows |
| R8 | **Translations of the new strings.** About 550 `null` strings in each locale's `studio.json` (counted in `de`), more in `common.json`. CLAUDE.md wants 95 % coverage in all 10 locales | Release gate | M | string freeze (no new keys after it starts) | Sonnet (model translation), native read later | `npm run i18n:validate` clean; every locale at 95 % or more; `en-XA` pseudo-locale screenshot of Board and Mixer shows no clipping |
| R9 | **Accessibility pass** on Music and Storyboard: keyboard only through Board and Mixer, focus order, labels on icon buttons, contrast of lane colours in both themes, reduced motion | The Mixer is the most complex UI we have | M | R3 (so fixes are tested) | Sonnet with the `design:accessibility-review` skill | Every action in the R3 script also works with the keyboard alone; no unlabeled buttons (axe or a manual list); findings fixed or logged |
| R10 | **Perf baseline**: add `music` (60 s with lyrics) and `storyboard-render` (8 clips, preview and final) to `scripts/perf-baseline.py` and floors to `PERF_BASELINE.md` | Without a floor, v2 speed work cannot prove a win or catch a loss | S | none | Sonnet | Median of 3 recorded; `perf-gate.py` passes |
| R11 | **Release workflow and pre-build check**: R5 outcome in `release.yml`, music and storyboard key deps in `pre-build-check`, `THIRD_PARTY_NOTICES.md` (ACE-Step, ffmpeg choice), `release-constraints.txt` re-frozen, dependency audit | Release gate | S to M | R1, R5 | Sonnet | `npm run pre-build-check` green; audit has no high or critical; tag workflow dry-run on a test tag |
| R12 | **Docs and screenshots**: `docs/usage/music.md` and `storyboard.md` checked against the real UI, README, mkdocs nav, new screenshots (`Screenshots/v0.10.0/`) | Users need it; the website reads these | M | R4 | Haiku drafts, Sonnet checks against the app | Every screenshot is from the flipped build; every documented button exists |
| R13 | **Release notes and CHANGELOG** for v0.10.0 (and v0.9.10 for the bridge and media input) | Release gate | S | R4 | Haiku drafts, Sonnet edits | Notes list only what shipped and what was not tested |
| R14 | **Small open items** from the review logs: stale marking for imported takes (no fingerprint), select on arrival after "Use in storyboard", per-job ids in the music progress route, `transcribe` taking upload handles | Rough edges, not blockers | S each | none | Sonnet | Each with a test, or moved to v2 with an FU row |
| R15 | **Listening and viewing** (yours): the 22 music samples, the three lighthouse films | Nobody has judged quality by ear | 30 min | none | you | Your verdict in decisions; any bad sample becomes an FU row |
| R16 | **Hub hygiene**: one progress file per stream (none exist for the home-studio streams), FU rows for every deferred item, status update | Working agreement; lets a fresh agent resume | S | none | Haiku | `tests/test_hub_tracker.py` passes; each stream below has a file in `docs/hub/progress/` |

**Moved out of v0.10.0** (recommended, decision 4): sound effects, voice cloning, captions, resumable renders. The plan's definition of done says "picture, speech, effects and music"; effects are not built. Ship v0.10.0 with picture, speech and music, and put effects in v2.

## 1.3 Merge order

1. **On hold until you say go:** `feat/agent-bridge` and `feat/media-input` into `release/v0.9.10` (D-32). Nothing else moves until then; `feat/home-studio` stacks on both.
2. Then `feat/home-studio` (with R1 to R16 merged into it) into a `release/v0.10.0` branch; tag after the release gate.
3. v2 streams branch from `feat/home-studio` now and rebase onto `release/v0.10.0` once it exists. No v2 stream merges into the release branch.
4. Merge rule for all streams: small PRs into the integration branch, one owner per shared file (section 4), worktree deleted after its merge hash is on the target.

## 1.4 What you need for the Windows 4090 test

**We hand you:** an unsigned `.msi` or `.exe` from the Build workflow, its SHA-256, and this checklist as an issue you can tick. Before that, R2 (tests green on Windows CI) and R5 (ffmpeg present) must be done, so you are not testing a known break.

**Checklist (about 1 to 2 hours):**

1. Install; first launch shows the terms screen once; the app reaches the Dashboard.
2. Chat with a small GGUF model (llama.cpp path still works).
3. Image Studio: one image with an installed CUDA image model.
4. Video Studio: one short clip with a CUDA video model (Wan or LTX on diffusers).
5. Voice: speak one line, transcribe it back.
6. Music: open Music > Models. Expected today: a plain "not available on this computer yet" (FU-139, no CUDA lane). Report what it says.
7. Storyboard: create a project, import two clips and a sound, reorder, export an MP4. Expected limit: making clips **from stills** is skipped on Windows (only the Mac LTX engine takes an image, D-40). Report whether the skip reason reads clearly.
8. Mixer export: check the file plays with sound, and note the time.
9. Agent access: switch on, copy the line, run it from Claude Code on the PC; ask for one image.
10. Remote (optional, D-29): start the backend with `--host 0.0.0.0`, allow the port on the private network, point Claude Code on the Mac at `http://<pc>:<port>/mcp`; make one image and fetch it. The packaged app's shell talks to `127.0.0.1`, so this may need the headless backend; tell us what you had to do.
11. Send `~/.chaosengine/logs` (or the Logs tab export) for anything that failed.

Linux: same list on any Linux box when you have one; not a v0.10.0 blocker unless you say so.

---

# Part 2. Home studio v2

## 2.1 Candidates

Value: what the user gets. Risk: what could go wrong. Licence as last read; re-read at pin time.

| Feature | Value | Size | Risk | Licence | Depends on |
|---|---|---|---|---|---|
| **Keep models warm** (speed-quality ranks 1 to 4: timing split, warm ltx-2-mlx worker, warm mflux worker, keep-warm policy) | Est. 20 to 30 % off a storyboard batch; every later still and clip faster | M + M + M | Holds 10 to 47 GB until idle; ltx-2-mlx may not keep weights (spike first) | MIT (mflux, ltx-2-mlx) | R10 baseline |
| **Captions in the Mixer** (Parakeet / Whisper on the voice lane, SRT sidecar, optional burn-in) | Films need subtitles; the reviewer ranked it high | M | Burn-in needs a font and the `subtitles` filter in our ffmpeg build (check the LGPL build too) | Parakeet CC-BY 4.0 (from memory), Whisper MIT | R5 ffmpeg choice |
| **Voice cloning with consent** (Qwen3-TTS, D-28, D-45); see `docs/research/voice-cloning.md` | Your own narrator voice; consistent character voices | L | Legal (consent, likeness, labelling); MLX port maturity | Apache 2.0 | voice-cloning research |
| **Sound effects** (Stable Audio Open Small, D-27) as a Sounds mode in Music | Completes the definition of done (effects lane) | M | Gated, "Powered by Stability AI", US$1M cap; licence screen needed | Stability Community License | D-27 yes |
| **Resumable renders** (manifest, resume at first missing intermediate; keep-awake and finish notification) | Long films survive a sleep or crash | M | Tauri notification and keep-awake differ per OS | n/a | none |
| **Extend a clip, retake one second** (LTX native `extend` / `retake`, speed-quality V7) | Fix a shot without redoing it; longer shots | M | Upstream marks it experimental | LTX-2 Community (carried) | warm LTX worker helps |
| **Seed variations** (N takes per shot, pick the best; music variation ranking M5) | Choice without retyping | S to M | Disk use | n/a | none |
| **Frame interpolation** (RIFE 4.25 MLX, "fast motion" mode, V3) | Smoother motion; est. 1.5 to 2x faster clips | M | Warping on fast action; LTX sound length must stay | MIT | timing split |
| **Upscale** (SeedVR2 3B in mflux, I3) | Make at 1 MP, upscale to 4 MP | M | int4 degrades; memory | Apache 2.0 weights | warm mflux worker (same venv) |
| **More video models for storyboard clips** (image input beyond LTX); see `docs/research/new-video-models.md` | Lifts the D-40 limit; makes Windows storyboards useful | L | Each model is its own engine work; CUDA lane on the 4090 | per model | new-video-models research; 4090 tests |
| **Video to sound** (LTX Foley LoRA spike, ThinkSound fallback) | Sound for silent clips | M spike | Unclear the LoRA works on existing footage; ThinkSound VAE under Stability terms | mixed | spike |
| **Title cards, templates, beat-synced cuts** | Faster finished look; cuts on the music's beat | M | Fonts and licences for templates | n/a | captions (shared text rendering) |
| **OpenTimelineIO export** | Finish a cut in Resolve or Premiere | S to M | OTIO licence to read (Apache 2.0 expected); media paths in the file | Apache 2.0 (to confirm) | none |
| **Stems** (lane stems on export: easy; source separation with mlx-audio-separator: harder) | Hand audio to an editor; split a song | S (lane stems) / M (separation) | Separation checkpoints often state no licence | code MIT; weights per checkpoint | M7 licence check |
| **Music repaint / extend / cover + lyrics helper** (M2, M4) | Fix 10 s of a song; longer songs | M | sft model lacks cover / repaint | MIT | none |
| **Voice first-chunk and steadiness** (Vo1, Vo2) | First audio sooner; steady long reads | S | none | n/a | none |
| **Music on CUDA** (FU-139) | Music on the 4090 | M | Python and torch wheels on Windows | MIT | 4090 tests |

## 2.2 Ranking

Weighted by value for the "brief to finished film" goal, then by risk and effort.

1. **Keep models warm** (timing split first). Biggest felt gain; backend only; no new licence.
2. **Captions in the Mixer.** High value, small risk, reuses our STT.
3. **Resumable renders.** Cheap insurance for long renders.
4. **Voice cloning with consent.** High value; the legal guards are designed (D-28, D-45); wait for the research note.
5. **Sound effects.** Closes the definition of done; licence screen is the main work.
6. **Extend / retake and seed variations.** Per-shot control, the storyboard's promise.
7. **Interpolation, then upscale.** After warm workers, so we can measure.
8. **OTIO export and lane stems.** Small and self-contained.
9. **Music repaint / extend and voice chunking.** Polish.
10. **More video models, video to sound, music on CUDA, title cards, separation.** Each needs research or the 4090 first.

## 2.3 Proposed v2 scope (first wave)

Items 1 to 5 plus OTIO and lane stems, in four streams whose files do not overlap. Items 6 to 10 are the second wave.

| Stream | Builds | Owns (only this stream edits) | Must not touch |
|---|---|---|---|
| **V-speed** | C1 timing split, V1 warm ltx-2-mlx worker (spike first), I1 warm mflux worker, C2/C5 keep-warm policy | `backend_service/image_runtime/mflux_engine.py`, `backend_service/ltx2_mlx_runtime.py`, new `backend_service/mflux_worker.py`, new `backend_service/ltx2_worker.py`, `scripts/perf-baseline.py`, `PERF_BASELINE.md`, `tests/test_mflux_worker.py`, `tests/test_ltx2_worker.py` | storyboard, voice, music, mcp_bridge |
| **V-story** | Captions (SRT, burn-in), resumable render manifest, keep-awake and notify, OTIO export, lane stems | `backend_service/storyboard/*` (assemble, render, new `captions.py`, `otio_export.py`), `src/features/storyboard/*`, `tests/test_storyboard_*.py` | voice engine code (calls the existing STT route only) |
| **V-voice** | Qwen3-TTS engine, consent record, cloned-voice tag, Voice Studio clone panel | `backend_service/voice_env.py`, `backend_service/voice_worker*.py`, `backend_service/routes/voice.py`, `backend_service/helpers/voice_library*.py`, `src/features/voice/*`, `tests/test_voice*.py`, new `backend_service/mcp_bridge/voice_clone_tool.py` (the tool body; registration by A-bridge) | storyboard, music |
| **V-audio** | Stable Audio Open Small as the Sounds mode, licence screen, `make_sound` tool body | `backend_service/catalog/music_models.py`, `backend_service/music_*.py`, new `backend_service/music_worker_stableaudio.py`, `src/features/music/*`, `tests/test_music_*.py`, new `backend_service/mcp_bridge/sound_tool.py` | voice, storyboard |

Single owners for shared files are in section 4.

---

# Part 3. Agent bridge v2

## 3.1 Gaps found in the code (`backend_service/mcp_bridge/`, `routes/agent.py`)

| Gap | Where | Effect |
|---|---|---|
| Server `INSTRUCTIONS` still say "images, video clips, speech and transcripts" | `protocol.py` | Agents are not told music and storyboard exist until they read the tool list |
| No tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) | `tools.py`, `music_tools.py`, `storyboard_tools.py` | Clients cannot auto-allow read-only tools or warn before delete |
| A running `speak` or `transcribe` cannot be cancelled | `jobs.py` `_TRACKERS` has image, video, music only | `cancel_job` returns false for voice |
| Jobs live in memory only, no `list_jobs` | `jobs.py` `JobStore` | After a restart an agent's job ids are gone; an agent that lost its id cannot find its job |
| No cap on queued agent jobs | `jobs.py` `submit` | One agent loop can queue hundreds of jobs and block the Studios |
| One token can do everything | `access.py`, `routes/agent.py` | No read-only token for a dashboard agent; no "no downloads" or "no cloning" token; remote and local callers are treated the same |
| No progress notifications; `GET /mcp` answers 405 | `routes/agent.py`, `protocol.py` | Agents poll `wait_for_job` every 60 s; no live progress in clients that show it |
| No MCP resources or prompts | `protocol.py` capabilities: tools only | Agents cannot browse projects, the music library or the gallery as resources; no ready-made "short film" prompt |
| `upload_image` also takes sound and video | `tools.py` | Misleading name; agents may not try it for audio |
| `transcribe` takes a local path only | `tools.py` | A remote agent cannot transcribe |
| No music library tool (list, rename, delete tracks) | `music_tools.py` | An agent cannot reuse an earlier track without the HTTP API |
| No download tool | by design (D-18) | v2 item: a confirmed download |
| Schema tests check validation, not quality | `tests/test_mcp_bridge.py` | Nothing stops a vague description or a missing `required` |
| `SKILL.md` has no recipe for remote use, music in films, or error recovery | `docs/features/agent-bridge/SKILL.md` | Agents relearn by trial |

## 3.2 v2 items

| # | Item | Value | Size | Risk | Exit test |
|---|---|---|---|---|---|
| A1 | **Tool quality pass**: refresh `INSTRUCTIONS`; annotations on every tool; descriptions rewritten short (D-30: group by verb); rename `upload_image` to `upload_file` and keep the old name as an alias | Fewer wrong calls, cheaper tokens | S | Renaming breaks old configs (alias covers it) | Contract test: every tool has annotations, a description under N characters, `required` set, examples valid against its own schema |
| A2 | **Schema and contract tests** (no new dependency: a small meta-check of our schema subset) plus a golden `tools/list` snapshot | Catches silent tool changes | S | Snapshot churn | Test fails when a tool's schema changes without the snapshot |
| A3 | **Jobs**: persist the agent job record (small JSON per job, kept 7 days), `list_jobs` tool, running-cancel for voice, a per-token cap on queued jobs (default 20, plain error) | Restart-safe, visible, cannot flood the queue | M | Persisted state vs. restart reset rules (match the storyboard's "running becomes interrupted") | Restart test: job ids survive and read `interrupted`; 21st job refused with the reason |
| A4 | **Scopes**: each token carries scopes `read` (status, list, job status, resources), `generate` (all make tools, storyboard edits), `files` (upload and download), `download` (A5), `clone` (voice cloning). The card makes a full token (read, generate, files) and an optional read-only one. Remote callers (not loopback) never get `clone` unless you tick it (D-28) | Least privilege; meets the cloning guard | M | Settings card UI; migration of the existing token to the full set | Tests per scope; a remote request with `clone` refused by default |
| A5 | **Confirmed model download** (D-18 v2): `request_model_download(model)` returns size, licence, gated flag, free disk and a pending request id. **You approve it in the app** (a banner: model, size, licence, Allow / Deny), not in the agent's chat, which the agent writes itself. Gated models are refused until your Hugging Face account has accepted; non-commercial licences shown in red; disk checked first. Needs the `download` scope | Agents can set up a machine without you opening Models, with you still in control | M to L | A new in-app approval surface (shared `App.tsx` banner area); abuse by repeated requests (rate limit, one pending at a time) | E2E: request, Deny leaves nothing on disk; Allow downloads a tiny test model; gated refused with the reason |
| A6 | **Music and storyboard completeness**: `music_library` (list, get, rename, delete), `storyboard_render` takes the v2 options (captions, stems, OTIO once V-story lands), `make_sound` and `clone_voice` registered when their streams land | Every UI action is a tool (principle 2) | S each | Tool count creeps past 20 (keep verb grouping) | `tools/list` covers every Music and Storyboard button; count logged |
| A7 | **Progress notifications**: answer `tools/call` and `wait_for_job` as a short event stream when the client sends a `progressToken`, sending `notifications/progress`; plain JSON otherwise | Live progress in clients that show it; fewer polls | M | Client support varies; must keep the JSON path | Test client sees at least 3 progress events for a stubbed job; JSON-only client unchanged |
| A8 | **Resources and prompts**: resources for projects (`chaosengine://storyboard/{id}`), the music library, recent gallery outputs, and system status; prompts `short_film_from_brief`, `poster_set`, `soundtrack_for_project` | Agents browse instead of guessing ids; one-click recipes in clients | M | Resource lists can be large (page them) | `resources/list`, `resources/read`, `prompts/list`, `prompts/get` tests; an agent run uses a prompt end to end |
| A9 | **Queue visibility**: `system_status` gains the whole queue (kind, source, label, position, estimate from the run estimate strip numbers) | Agents can tell you how long to wait | S | Estimates wrong on first runs (label them "estimated") | Status shows 3 queued jobs with positions and estimates |
| A10 | **Remote routing to the 4090** (D-29). Step 1 (S): you prove the PC as a plain `/mcp` target (checklist item 10). Step 2 (L): a Mac-app "Remote worker" setting that sends a job kind (video, music) to the PC and pulls the file back through `download` | The Mac drives the PC's GPU from one app | S then L | Two backends' queues; failure mid-job; security on the LAN (plain HTTP plus token) | Step 1: an image made on the PC and fetched on the Mac. Step 2: the mini film's clips made on the PC from the Mac's storyboard |
| A11 | **`SKILL.md` refresh**: remote recipe, film with music, revision-conflict recovery, scope errors, download request flow | Agents work first time | S | none | A fresh Claude Code session with only the skill makes the 3-shot film without help |
| A12 | **E2E**: extend phase 9 or add phase 17 for annotations, scopes, `list_jobs`, the download refusal path, resources | Feature gate | S | none | Phase passes on this Mac |

## 3.3 Proposed v2 scope for the bridge

**First wave (one stream, A-bridge):** A1, A2, A3, A9, A11, A12, then A4. Backend only, low risk, makes the bridge solid before anything new.
**Second wave:** A5 (needs the approval banner; decision 6), A7, A8, A6 as the V-streams land.
**Third:** A10 step 2, after your step 1 result.

A-bridge owns: `backend_service/mcp_bridge/*` (except the tool bodies named in Part 2), `backend_service/routes/agent.py`, `scripts/chaosengine-cli`, `docs/features/agent-bridge.md`, `docs/features/agent-bridge/SKILL.md`, `src/features/settings/AgentAccessPanel.tsx`, `tests/test_mcp_*.py`, `tests/test_job_queue.py`. It is also the **only** stream that edits the tool registry (`HANDLERS` / `TOOLS` in `tools.py`): other streams hand over a tool module and A-bridge registers it.

---

# 4. Streams, shared files and tonight

## 4.1 Stream table

| Stream | Model tier | Files owned | Branch | Exit test |
|---|---|---|---|---|
| R-pack (R1, R2, R5 interim, R6, R7, R11) | Sonnet; Opus for R5 design | `.github/workflows/*`, `scripts/stage-runtime.mjs`, `scripts/pre-build-check.*`, `THIRD_PARTY_NOTICES.md`, `release-constraints.txt`, `pyproject.toml`, `backend_service/music_requirements.txt` | `release/pack-v0.10.0` | pre-build-check green; dmg opened; Windows build handed over |
| R-e2e (R3, R4, R9 checks) | Sonnet; Opus reviews | `scripts/e2e_test_suite.py`, new `scripts/e2e_ui/*`, `docs/E2E_TESTING.md`, `src/constants/features.ts` (the flip, last) | `test/e2e-ui` | Phase 12 green; full suite green with the switch on |
| R-i18n (R8) | Sonnet | `src/locales/<9 non-en locales>/*`, `backend_service/locales/*`, `src-tauri/locales/*` | `i18n/v0.10.0` | `i18n:validate` clean, 95 % in all locales |
| R-a11y (R9 fixes) | Sonnet | `src/features/storyboard/*`, `src/features/music/*`, `src/components/studio/*` (until V-story starts UI work) | `fix/a11y-home-studio` | Keyboard-only run of the R3 script |
| R-docs (R12, R13, R16) | Haiku drafts, Sonnet checks | `docs/usage/*`, `README.md`, `mkdocs.yml`, `RELEASE_NOTES_v0.10.0.md`, `CHANGELOG.md`, `Screenshots/v0.10.0/`, `docs/hub/progress/*` (new files only) | `docs/v0.10.0` | Docs match the flipped build |
| R-ffmpeg (downloader) | Opus design, Sonnet build | per `docs/plans/ffmpeg-download.md` (new ffmpeg helper module; the one ffmpeg lookup used by `output_metadata.py` and storyboard assemble) | `feat/ffmpeg-download` | Per that plan |
| V-speed | Opus spike, Sonnet build | see 2.3 | `feat/speed-warm-workers` | Lighthouse batch re-timed against R10 floor |
| V-story | Sonnet; Opus reviews render resume | see 2.3 | `feat/storyboard-v2` | Captions burned into the lighthouse film; render resumes after a kill |
| V-voice | Opus (consent design), Sonnet | see 2.3 | `feat/voice-cloning` | A cloned voice in a mixed film, consent recorded, tag forced |
| V-audio | Sonnet | see 2.3 | `feat/sound-effects` | Effects lane in a mixed film; licence screen shown |
| A-bridge | Opus for A4 / A5 / A10, Sonnet for the rest | see 3.3 | `feat/agent-bridge-v2` | Phase 17 green; fresh-session skill test passes |

**Shared files with a single owner** (everyone else asks the owner, or hands over a patch):

| File(s) | Owner | Rule for others |
|---|---|---|
| `src/App.tsx`, `src/app/*`, `src/constants/{tabs,sidebarGroups,tabRedirects}.ts`, `types/system.ts`, `SidebarIcons.tsx` | R-e2e until the flip, then the main session | No new tabs in v2 wave 1; the A5 approval banner goes through the main session |
| `src/locales/en/*.json` | each stream adds keys under its own namespace (`storyboard.*`, `music.*`, `voice.clone.*`, `settingsTab.agentAccess.*`) | New keys go into `en` only with `null` stubs added by R-i18n at merge; string freeze for v0.10.0 when R8 starts |
| Non-en locale JSONs | R-i18n | No one else edits them |
| `scripts/e2e_test_suite.py` | R-e2e | Phase numbers reserved: 12 home-studio UI, 13 speed, 14 voice cloning, 15 sound effects, 16 storyboard v2, 17 agent bridge v2; each stream writes its phase function in a new file `scripts/e2e_phases/phase_NN.py` and R-e2e adds the one import line |
| `backend_service/mcp_bridge/tools.py` registry | A-bridge | Others ship a tool module; A-bridge registers it |
| `backend_service/routes/__init__.py`, `catalog/__init__.py`, `progress.py`, `helpers/job_queue.py`, settings files (`helpers/settings.py`, `state/settings_state.py`, `models/__init__.py`) | main session | One-line patches handed over and applied in order |
| `THIRD_PARTY_NOTICES.md`, `pyproject.toml`, `release-constraints.txt` | R-pack | Streams list new deps in their PR text; R-pack adds them |
| `docs/hub/decisions.md`, `status.md`, `FOLLOW_UPS.md`, `CLAUDE.md` | main session | Streams write to their own `docs/hub/progress/<stream>.md`; the main session copies to the hub and rebuilds it |

## 4.2 Ordered start list for tonight

No merge into `release/v0.9.10` or `main`; everything stays on branches off `feat/home-studio`.

1. **R-pack, R1 + R2** (Sonnet): drop pytorch-wavelets and smoke-test; run the Windows and Ubuntu test jobs on `feat/home-studio`. Cheap, unblocks the Windows build.
2. **A-bridge first wave** (Sonnet; A1, A2, A3, A9): backend-only, owns its folder, no shared files beyond the registry it already owns.
3. **V-speed spike** (Opus): timing split and the "does ltx-2-mlx keep weights" spike. Backend-only, disjoint. Needs R10's baseline numbers, so it also records the baseline (R10 moves into this stream tonight).
4. **R-i18n** (Sonnet) after a string freeze on `feat/home-studio`.
5. **R-docs** (Haiku): progress files for each stream (R16) and a first draft of the v0.10.0 notes.
6. **R-e2e** (Sonnet) as soon as you answer decision 2 (test tool).
7. **Hold** until their inputs land: R-ffmpeg (needs `ffmpeg-download.md` and decision 3), V-voice (needs `voice-cloning.md`), V-audio (needs D-27), A5 (needs decision 6), A10 step 2 (needs your 4090 step 1), more video models (needs `new-video-models.md`).

## 4.3 Your decisions (ELI5)

1. **Merge v0.9.10 now?** The agent bridge and picture input are finished and waiting; saying go lets them ship before the studio. (Recommend: go when you have 30 minutes to read the PR.)
2. **Add a browser test robot (Playwright, Apache 2.0) as a test-only tool?** It clicks through Music and Storyboard like a person, every build. (Recommend: yes; it never ships to users.)
3. **ffmpeg in v0.10.0:** wait for the download-on-demand LGPL version (cleaner, a few more days), or ship the GPL ffmpeg inside the app for this one release with its notice? (Recommend: wait if the downloader plan is under 3 days; otherwise ship GPL once, already noticed.)
4. **Ship v0.10.0 without sound effects?** Films get picture, voice and music; effects come in v2. (Recommend: yes.)
5. **Storyboard on Windows makes no clips from stills yet** (only the Mac video engine takes a picture). Ship that limit, clearly labelled? (Recommend: yes, with the reason on screen.)
6. **Agent downloads: you click Allow in the app**, not "yes" in the agent's chat (the agent could type that itself). OK? (Recommend: yes.)
7. **Token scopes:** the normal token can make things and move files but cannot download models or clone voices unless you tick those. OK? (Recommend: yes.)
8. **Translate the ~550 new strings with a model now, native speakers later?** (Recommend: yes; same as previous releases.)
9. **Keep models in memory between jobs** (D-43): faster, but the app holds up to 47 GB until it sits idle for 10 minutes. (Recommend: yes, with idle stop.)
10. **Remote 4090:** test the PC as a plain agent target first, decide the Mac-sends-jobs-to-PC feature after. (Recommend: yes, D-29.)
