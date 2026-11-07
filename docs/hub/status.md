---
title: Status and handover
order: 1
summary: Read this first in every session.
---
# Status and handover

> **Read first.** ChaosEngineAI is a desktop AI inference app (Tauri shell, React frontend, Python sidecar; MLX, llama.cpp and vLLM engines; image, video and voice studios). It is in **v0.9.9 release prep**: the tag `v0.9.9` is held until [PR 119](https://github.com/cryptopoly/ChaosEngineAI/pull/119) merges. Then read [decisions](decisions.md), [backlog](backlog.md) and the progress file for the active stream in [progress](progress/README.md).

Newest dated update first. Each update has the same sections: Done, Open for Claude, Open for Dan.

## Update 2026-11-07 (morning), the home studio is built behind a switch; read this first

**State:** everything below is on `feat/home-studio` (pushed). Nothing is merged into `release/v0.9.10`, `main` or `staging`. Music and Storyboard are built but **hidden** behind `HOME_STUDIO_ENABLED = false` in `src/constants/features.ts`. Flip it to `true` to see them; no other change is needed.

**Built and verified (real runs on this Mac, numbers from the runs):**

- **Music** (ACE-Step 1.5, MIT): Models, Studio, Library, agent tool `make_music`. 30 s instrumental in 3 s warm; 60 s song with vocals in 5 s; 10 s for a 40 s track cold start; peak about 21 to 23 GB; cancel, queue behaviour (one heavy job at a time, music and image tested together), unload and idle stop all checked.
- **Storyboard:** Projects, Board (takes, locks, stale marking, transitions, undo/redo, batches by kind with one cancel), Mixer (lanes, waveforms, ducking, preview, export 16:9 / 9:16 / 1:1, AI label, end card), `.chaosfilm` bundles, "Use in storyboard" from the image and video galleries, voice library and music library.
- **Acceptance run through the HTTP API only:** the 8 lighthouse clips, the voiceover and a generated 40 s music bed (ducked under the voice) became a 32.9 s, 1280x720 film in 7.8 s, loudness -16.0 LUFS, AI tag and end card on. Files: `~/ChaosEngineAI-demos/lighthouse-keeper/lighthouse-keeper-with-music.mp4` (also `lighthouse-keeper.mp4` and `lighthouse-keeper-v2.mp4`). I have not listened to any of them; please do.
- **Calmer existing Studios (visible now, not behind the switch):** Image and Video Studio have Expert and Options disclosures, one run estimate strip, one Models button and a more-actions menu; first-load controls fell from 28 to 33 to 20 to 27 (the two installed image models and the LTX video model already hide most knobs; a diffusers model would show a bigger drop). The sidebar is an accordion with Make and Run sections (Settings switch keeps every group open).
- **Tests:** 2791 Python and 1108 vitest pass; `tsc`, `i18n:validate` and `npm run build` clean.

**Decisions for Dan**

1. **Switch the home studio on?** And for which release (plan: v0.10.0; merge `feat/agent-bridge` and `feat/media-input` as v0.9.10 first, D-32).
2. **Sidebar order:** Chat now sits under Images / Video / Voice / Music / Storyboard (the "Make" and "Run" sections). Keep, or Chat on top?
3. **D-36 loudness of a stand-alone music track:** keep the -1 dBFS peak normalisation (current), or write the raw level and leave loudness to the mixer?
4. **D-37 `pytorch-wavelets` licence:** PyPI says "Free To Use But Restricted" although the parts ACE-Step uses are MIT. Needs your confirmation before shipping.
5. **ffmpeg in the release bundle (D-35 follow-up):** the release workflow does not install `imageio-ffmpeg`, so a packaged app has no pinned ffmpeg for the Mixer. Adding it ships a GPL binary (about 47 MB on macOS).
6. **Listen to the music samples** in `~/ChaosEngineAI-demos/music-spike/` (22 files, including XL) and the three films; nobody has judged quality by ear.

**Not done / not verified**

- Sound effects (Stable Audio Open, gated), voice cloning (Qwen3-TTS), video-to-sound, captions in the Mixer, resumable renders, upload of audio and video for remote agents, the CUDA lane for music, Windows and Linux for all of it.
- No E2E checks for the new screens (they sit behind the switch; CLAUDE.md wants them before it flips).
- The packaged desktop app was never run with any of this (Tauri dialogs: Choose file, Reveal).
- Stale marking for imported takes (no fingerprint), "select on arrival" after "Use in storyboard", per-job ids in the music progress route.

**Open for Claude:** sound effects and voice cloning (P5), the quality pack (P6), remote 4090 (P7), E2E for the UI, then release prep (P8).

## Update 2026-11-07, music backend (P1) on `feat/music-backend`

**Done** (not pushed; backend only, the Music screens are P2)

- **ACE-Step 1.5 wired end to end:** catalog rows (turbo default, XL opt-in for 64 GB Macs) with licence, makes, platforms and the spike's measured estimates; fit for kind `music`; isolated music venv with plain pip and the pinned commit; JSON-lines worker with real step progress, cooperative cancel and a 3 s hard kill; Music Library with FLAC 24-bit tags (`machine_generated`, D-34 switch `musicEmbedTag`); routes on the shared queue with a memory gate; `make_music` agent tool (18 tools); E2E phase 11. See [music](https://github.com/cryptopoly/ChaosEngineAI/blob/feat/music-backend/docs/features/music.md).
- **Real check on this Mac** (spike venv and weights): 30 s instrumental 2.8 s warm (11 s cold), 60 s song with lyrics 5.1 s at 22.9 GB peak, 2 x 60 s in 10.6 s, cancel in diffusion 1.5 s (worker kept) and in decode 3.4 s (worker stopped, next run byte-identical), music and an image job ran one after the other, idle stop freed 15.6 GB.

**Open for Claude:** the Music UI (P2); the CUDA lane (FU-139).

**Open for Dan**

- `pytorch-wavelets` licence (FU-138): its PyPI page says "Free To Use But Restricted" although the parts ACE-Step uses are MIT. Read its LICENSE before a release ships the music pack.
- Keep the library's -1 dBFS peak normalisation for stand-alone tracks (current default), or write the raw level and leave loudness to the mixer?
- Listen to the real-check tracks (lyrics, BPM, key); quality was not judged by ear.

## Update 2026-11-06 (overnight), home studio build on `feat/home-studio`

**Done** (all merged into `feat/home-studio`, pushed, full suite 2693 passed)

- **UX spec:** [home-studio-ux](https://github.com/cryptopoly/ChaosEngineAI/blob/feat/home-studio/docs/plans/home-studio-ux.md): audit (Image Studio about 44 controls at first load, Video Studio about 47; a new user needs 8 or 9), principles, wireframes, component list, copy deck, and nine ranked clean-up changes for the existing Studios (not built).
- **UI kit and scaffold:** shared components (Disclosure, StudioShell, RunEstimateStrip, licence chip and screen, audio player, media library card, audio picker), lane tokens, icons; the sidebar is now an accordion with "Make" and "Run" section labels (Settings switch "Keep all sidebar groups open" restores the old behaviour); Music and Storyboard tabs are registered but hidden behind `HOME_STUDIO_ENABLED = false` (`src/constants/features.ts`). Voice Library now uses the shared card.
- **Storyboard core (backend):** project folder (self-contained, revisioned, undo/redo, takes, locks, stale marking), timeline from measured durations, batches by kind with one cancel, `.chaosfilm` bundles with hostile-input defences, routes, 5 agent tools (17 in total), E2E phase 10.
- **Assemble engine:** cached per-shot intermediates, cuts and crossfades, ducked music under voice, two-pass loudness, end card, cancel; 32 tests. **Real check:** the 8 lighthouse clips plus the voiceover rendered in 4.4 s straight through the engine and in 7.2 s through the API (export with end card), loudness -16.0 LUFS; I looked at a crossfade frame and the end card.
- **Integration bug caught and fixed:** the store sent non-string metadata that the assembler refused; fixed, with a permanent store + routes + real assembler test.

**Under way:** ACE-Step spike on this Mac; Storyboard Projects and Board UI.

**Decision for Dan (new):** the sidebar order now puts Chat below Images / Video / Voice (the UX spec's "Make" and "Run" sections). Keep, or put Chat back on top?

## Update 2026-11-06, decisions answered; embedded file paths removed

**Done** (branch `feat/media-input`)

- **Decided by the owner:** D-25 (ACE-Step first, other engines as options), D-28 (voice cloning with consent), D-31 (project folder plus a single-file bundle), D-32 (order; merge still on hold), D-33 (storyboard beside music), D-35 (keep the ffmpeg build, notice added). D-34 (AI labelling) decided later the same day: tag on by default with a Settings switch, cloned voice always tagged, optional end card, no hidden watermark yet; legal check of the EU rule's timing and scope still to do (owner is UK-based but wants to follow it).
- **Privacy bug fixed:** images and videos made with an input image carried the full local path (including the account name) inside the file's embedded settings. They now carry the file name only; the library's own JSON keeps the full path. Test added.
- **Plan:** music engines table added (ACE-Step, HeartMuLa, Magenta RT 2, DiffRhythm, MiniMax Music 3, Stable Audio Open); `.chaosfilm` bundle and a later OpenTimelineIO export added to the project format.

**Open for Dan**

- A legal check of the EU AI Act Art. 50 timing and scope for a locally-run tool (no lawyer consulted yet).
- Say when to merge `feat/agent-bridge` and `feat/media-input` into `release/v0.9.10`.

## Update 2026-11-06, home studio P-1 started (cancel fix, ffmpeg notice)

**Done** (branch `feat/media-input`)

- **Cancel fix:** a Studio's Cancel now stops only its own running job and drops only its own waiting jobs (`HEAVY_JOBS.cancel_studio`). An agent's running or waiting job is left alone; an agent cancels its own with `cancel_job`. Three new tests.
- **ffmpeg notice:** `THIRD_PARTY_NOTICES.md` now lists imageio-ffmpeg and its GPL ffmpeg build with a source offer; the pre-build licence checks look for it; `imageio-ffmpeg` is declared as the `ffmpeg` extra in `pyproject.toml` (it was only pulled in by `mlx-video`).

**Found:** the release workflow installs a fixed package list (`.github/workflows/release.yml` lines 241 to 255) that does **not** include imageio-ffmpeg, so a packaged app has no pinned ffmpeg unless something installs it. Adding it changes what ships (a GPL binary of about 47 MB on macOS arm64; other platforms not measured), so I did not change the workflow; it needs your decision with D-35.

- **`App.tsx` extract:** 2,269 lines down to 930. Each tab's render block moved into `src/app/*Tabs.tsx` views taking one `ctx` object; load / launch actions, modals, banners and the header moved to their own files. Every hook stays in `App` in the original order; no tab id, prop contract or string changed. Built by a Sonnet agent in a worktree, then re-applied by me onto this branch (the agent started from an older base without the media-input work). Checked: `tsc`, 791 vitest, `i18n:validate`, `npm run build`, and a click-through of the real UI (Image Studio reference picker and size rows, Video Studio start / end pickers and resolutions, dashboard, server, settings, logs). About 60 lines of already-unused code stay in `App.tsx` on purpose (deleting would reach about 870).

**Open for Claude:** the shared registry scaffold (P-1 c: tab, group, kind and tool registries, catalog licence field) and the rest of the plan once the open decisions are answered.

## Update 2026-11-06, the home studio plan

**Done** (branch `feat/media-input`)

- **Plan:** [home-studio](https://github.com/cryptopoly/ChaosEngineAI/blob/feat/media-input/docs/plans/home-studio.md): a Music area (Models, Studio, Library), a Storyboard with takes, locks, a mixer and export, sound effects, voice cloning, a quality pack and the remote 4090. Built from an Opus model and licence sweep, a Sonnet code survey, and an Opus adversarial review whose findings changed the plan (section 12 of the plan).
- **First music model:** ACE-Step 1.5 (MIT), after a Mac speed spike. Sound effects: Stable Audio Open (gated, opt-in). Voice cloning: Qwen3-TTS (Apache 2.0). Rejected on licence: MusicGen, YuE, SongGeneration, MMAudio, F5-TTS and others (table in the plan).
- **Found along the way (exists today):** one Studio Cancel cancels every waiting image or video job including an agent's; the ffmpeg we ship is a GPL build with no notice in `THIRD_PARTY_NOTICES.md`; `App.tsx` is 2,269 lines against a 500-line cap.
- Decisions D-25 to D-35 logged, all open.

**Open for Dan**

- Answer the open decisions, mainly D-25 (go on the spike), D-32 (merge order), D-34 and D-35.
- Say when to start P-1 (cancel fix, `App.tsx` extract, scaffold, ffmpeg notice).

## Update 2026-11-06, remote files for the agent bridge, and the 32 s film

**Done** (branch `feat/media-input`)

- **Film:** `docs/examples/lighthouse-keeper-demo/` makes a 32.4 s, eight-shot film (768x448, 24 fps, with sound) only through `/mcp`: keeper portrait, one still per shot (FLUX 3 Klein, lighthouse and keeper as `reference_images`), one LTX-2.5 clip per still (`start_image`), Kokoro voiceover, ffmpeg join. About 25 minutes on this Mac. I looked at one frame from each shot: same keeper (yellow raincoat, cap, white beard) and same lighthouse in all eight. I have not played the video or listened to it; the audio level read -24 dB mean, so there is sound.
- **Remote files:** `upload_image` tool and `POST /api/agent/uploads` return an `upload:` handle that image and video tools accept as a path; job outputs carry `download` (`url`, `bytes`, `sha256`) and `GET /api/agent/files` serves only files inside the image, video and voice output folders (404 otherwise, symlinks out refused). Inbox files expire after 24 hours. 12 tools now. E2E phase 9 has a check for the round trip.

**Not tested:** a real second machine (the endpoint was exercised on this Mac only); the packaged Windows app listening on the network (not checked: it may bind to loopback only); audio upload for `transcribe`.

**Open for Dan**

- Install on the 4090 PC, start the backend with `--host 0.0.0.0`, allow the port on the private network, and point Claude at `http://<pc-address>:<port>/mcp` with the Agent access token.

## Update 2026-11-06, Enhance removed

**Done** (branch `feat/media-input`, decision D-24)

- Removed the Enhance button (Image and Video Studio), the "Auto-enhance short prompts" switch, the `/api/prompt/enhance` route, `helpers/prompt_enhancer.py`, the `prompts-enhance` CLI command, the `enhancePrompt` request field and its template suffixes, the frontend API call, locale strings, tests and docs. No model download is needed for any prompt feature now.
- Video prompts are sent exactly as typed. Older callers that still send `enhancePrompt` are not rejected (the field is ignored).

## Update 2026-11-06, Studio sizes, friendly errors, end-only image

**Done** (branch `feat/media-input`)

- **Image Studio sizes:** the four fixed pills are replaced by an aspect row (1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9) and a size row (0.5, 1, 2, 4, 8 MP). A button the model cannot take (its `maxSide` / `maxPixels`) is greyed out with a reason. The 2048 x 2048 failure was not a size limit: the model allows 4096 / 16.7 MP. The Steps box held 248 and the request refused anything over 100; Steps now clamps to 1 to 100.
- **Video Studio sizes:** aspect row (16:9, 9:16, 1:1, 4:3, 21:9) plus named resolutions SD 480p, HD 720p, Full HD 1080p, 2K, 4K, 8K (short edge, multiples of 64), greyed out past the model's limits. LTX-2.5 allows up to 9.4 MP, so 8K is greyed out; 4K is allowed by the limit but a 64 GB Mac may refuse it at the memory check.
- **Friendly errors:** a validation error from the backend now reads "Steps must be at most 100 (you entered 248)." instead of raw JSON (`src/api/validationErrors.ts`).
- **End image alone:** LTX-2 takes an end image without a start image (the clip moves toward it); API, agent tool and Studio all allow it. Not yet judged on a real clip.
- Tests: pytest 2614 passed, vitest 791 passed, `tsc` and `i18n:validate` clean.

## Update 2026-11-06, media input slice C: Studio picker

**Done** (branch `feat/media-input`)

- **Image Studio:** a "Reference image(s)" picker appears under the prompt for models that take one (FLUX.1, FLUX.2 Klein, Qwen: one image plus a Strength slider; FLUX 3 Klein: up to four). Hidden for every other model.
- **Video Studio:** "Start image" for LTX-2.3 / 2.5; "End image (optional)" appears once a start image is set. Removing the start image clears the end image.
- **Picker** (`src/components/InputImagePicker.tsx`): "Choose file" (desktop file dialog; png, jpg, webp) or "From gallery" (the 12 newest Image Studio outputs). A pick is a path on this Mac, not an upload.
- **Catalog flags:** image variants carry `maxReferenceImages`, video variants `acceptsImageInput`, so the Studio never guesses.
- **Checked in the running UI** (Vite dev server against a real backend): picked the earlier lighthouse from the gallery, FLUX 3 Klein, prompt "the same lighthouse in a snowstorm at night, blue moonlight". The request carried `referenceImagePaths`; the result keeps the same tower, stripes, rocks, hut and beam in a new scene. The Video Studio picker showed Start image, then End image after a pick.
- Tests: vitest 784 passed plus 2 new; new `CatalogFlagsTests`; `i18n:validate` OK (new `studio.inputImage.*` keys, en text and null elsewhere).

**Not tested:** the "Choose file" button in the packaged desktop app (the browser pane has no file dialog; the code path is the same plugin the folder picker already uses); a video render started from the Studio; strength slider output; layout on a narrow window.

**Open for Claude**

- Nothing for media input v1. Not planned: voice cloning, sound-to-video, ControlNet, several keyframes, remote upload.

**Open for Dan**

- Merge of `feat/agent-bridge` and `feat/media-input` into `release/v0.9.10` (on hold at your request). Clip-length rounding (round down vs nearest) and Windows/Linux checks are still open.

## Update 2026-11-06, media input slice B: reference images for image generation

**Done** (branch `feat/media-input`)

- **Image references.** `POST /api/images/generate` takes `referenceImagePaths` (up to 4) and `imageStrength`. FLUX.1 (dev, schnell), FLUX.2 Klein and Qwen Image 2.1 (and Turbo) pass one image to `mflux-generate* --image PATH STRENGTH` (start image; higher strength stays closer). FLUX 3 Klein (and Base) switch to `mflux-generate-flux3-edit --image-paths ...` and take up to four. Iris-3B, every diffusers / GGUF / sd.cpp model and any non-mflux variant answer 400 with the list of models that work. If mflux fails or is missing, the run errors instead of falling back to an engine that would drop the image.
- **MCP:** `generate_image` gets `reference_images` and `image_strength`; `list_models` marks image models with `acceptsImageInput` and `maxReferenceImages`. With no width and height the image takes the first reference's shape. The tool-schema checker now handles arrays.
- Reference paths are saved in the artifact JSON and in the PNG's embedded settings. The video route now checks the image fields before the "not installed" gate, so the clear reason shows first.
- **Live check on this Mac:** FLUX 3 Klein with the lighthouse poster as reference and the prompt "the same lighthouse at sunrise, warm golden light, calm sea" gave a 512x768 image in 43 s (peak 12.45 GB). Side by side: same tower, stripes, rocks, hut and beam, new light and sky. Not tried live: FLUX.1, FLUX.2 Klein and Qwen start-image mode (command lines are unit-tested only; those models are not installed here).
- E2E phase 9 has a sixth check (wrong-model refusals, image and video). Phase 9: 6 of 6.

**Open for Claude**

- Slice C: the Studio picker (upload or pick from the gallery) in Image Studio and Video Studio, strings and locales.

**Open for Dan**

- Merge of `feat/agent-bridge` and `feat/media-input` into `release/v0.9.10` stays on hold.

## Update 2026-11-06, media input slice A: video start / end image

**Done** (branch `feat/media-input`, cut from `feat/agent-bridge`; plan [media-input](https://github.com/cryptopoly/ChaosEngineAI/blob/feat/media-input/docs/plans/media-input.md), decision D-23)

- **Video from an image.** `POST /api/video/generate` takes `startImagePath`, `endImagePath` and `imageStrength`; LTX-2.3 / 2.5 pass them to `ltx-2-mlx --image PATH FRAME STRENGTH` (start on frame 0, end on the last frame). Every other video model answers 400 naming LTX-2 (it never ignores the image); a missing or non-image file, an end image without a start image, and files over 30 MB are refused with a readable message (`helpers/media_input.py`). The saved clip's metadata records the input image.
- **MCP:** `generate_video` gets `start_image`, `end_image` and `image_strength`; `list_models` marks models with `acceptsImageInput`. With no width and height the clip takes the image's shape at the model's usual pixel count.
- **Live check on this Mac:** the earlier lighthouse poster (768x1152) as `start_image` gave a 512x768, 2.7 s clip with sound in about 90 s. I looked at the first, last and poster frames side by side: same lighthouse, rocks, sky and beam from first to last frame, only the sea and light move.
- Tests: `test_media_input.py` (9), new cases in `test_mcp_bridge.py` and `test_output_metadata.py`.

**Open for Claude**

- Slice B: reference images for image generation (mflux init-image and the `-flux3-edit` / `-flux2-edit` / `-qwen-2.1-edit` commands); slice C: the Studio picker; docs, SKILL.md and an E2E check for each.

**Open for Dan**

- Say whether slice B should come before the Studio picker (my order) or after.
- Sound-to-video, voice cloning and diffusers video models stay out of scope (listed in the plan).

## Update 2026-11-06, agent bridge: real Claude Code run

**Done**

- **Claude Code (2.1.316, Sonnet) used the bridge from one prompt** ("poster of a lighthouse at dusk, a 3 second clip of waves on rocks, a one-line voiceover"), through the HTTP line the card gives (temporary `--mcp-config`, only the `chaosengine` tools allowed). It called `system_status`, then ran image, video and speech one after the other with `wait_for_job` between them, and reported three files that exist: a 768x1152 PNG (1.0 MB), an MP4 (788 KB, 2.7 s) and a WAV (89 KB). 10 turns, $0.19. It told the user the clip is 2.7 s, not 3 s, because of the 8k+1 frame rule.
- Cleanup: Agent access switched back off, the test backend stopped, the temporary config with the token deleted.

**Open for Dan**

- I did not look at the poster or play the clip or the voice line; only that the files exist and the run was clean. Worth a look at `~/.chaosengine/images/outputs/2026-11-06/img-27df9123926f.png`.
- Still open: Windows / Linux check of the relay and endpoint; merge decision (D-21); clip-length rounding.

## Update 2026-11-06, agent bridge step 5 (v1 built, ready for review)

**Done**

- **E2E phase 9 "Agent bridge"** in `scripts/e2e_test_suite.py` (also in `docs/E2E_TESTING.md`): off answer, `tools/list`, `system_status` + `recommend_models`, stdio relay, and a real image job to a PNG with embedded settings. It switches Agent access on and restores the old state. Passed against a live backend on this Mac (5 of 5, 9 s; Iris-3B). The suite found one bug in itself (it looked for the PNG settings chunk at the start of the file; it is written just before the end).
- **Live run through `/mcp` on this Mac** (M4 Max, 64 GB, driven by a script, not by an agent): a poster (Iris-3B, 1024 px, 11 s), a 3 s LTX-2.5 clip with sound (768x512, 65 frames, 2.7 s) and a voice line (Kokoro, 1.85 s) submitted at the same moment. They ran strictly one after the other (at most 1 running; the others showed queue positions 1 and 2). Notes came back (LTX 8k+1 rule: 72 frames became 65). PNG and MP4 both carry prompt, model, seed, steps, size and app version.
- Full pytest and vitest were green at step 4 (2588 / 782); phase 9 and doc edits since touch no app code.

**Open for Claude**

- Nothing left in the v1 rollout table. Merge needs your go (see below).

**Open for Dan**

- ~~Real Claude Code run~~ Done, see the update above.
- **Not done: Windows check** of the relay and endpoint (no Windows machine here). Linux untested too.
- Decide: merge `feat/agent-bridge` into `release/v0.9.10` or hold it for the release after (plan says after; the queue changes how the Studios start jobs). D-21 still reads "open".
- Rounded-down clip length: a 3 s ask gives 2.7 s on LTX (frames round down to 8k+1). Say if it should round to the nearest instead.
- Agent token is shown in the running app's API response; the backend I started for the test is stopped.

## Update 2026-11-06, agent bridge step 4

**Done**

- **CLI staged in the release bundle:** `scripts/stage-runtime.mjs` now copies `chaosengine-cli` to `backend/scripts/` (new `BUNDLED_CLI_SCRIPTS`; test in `test_stage_runtime_scripts.py`). `GET /api/agent/access` returns `stdioCommand` (this Python plus the staged CLI plus `mcp`), and the Agent access card shows a third snippet for clients that only start local programs. The staged runtime archive was **not rebuilt**; the check is the staging test.
- **Docs:** `docs/features/agent-bridge.md` (setup in four steps, the one-job-at-a-time rule, tools, safety, stdio, troubleshooting; in the mkdocs nav) and `docs/features/agent-bridge/SKILL.md` (drop into `~/.claude/skills/chaosengine/`). The CLAUDE.md test table lists the four new test files.
- Strings: one more English string (`settingsTab.agentAccess.stdio`), `null` in the other nine locales. vitest 782, `tsc` clean, `i18n:validate` OK.

**Open for Claude**

- Step 5: E2E phase check (`tools/list`, then `generate_image` with the smallest installed image model and `wait_for_job`), unit gaps, and a live Claude Code run.

**Open for Dan**

- Docs and the skill were not tried by someone who has not seen the project.
- Pre-build checklist: after the next runtime build, confirm `tar -tzf src-tauri/resources/embedded/runtime-*.tar.gz | grep chaosengine-cli`.

## Update 2026-11-06, agent bridge step 3

**Done**

- **Stdio relay:** `scripts/chaosengine-cli mcp` reads one JSON-RPC message per line on stdin and forwards each to the running app's `/mcp`. The token is `CHAOSENGINE_API_TOKEN` if set, else the app's loopback session token (fetched again once on a 401). While the app is not reachable it answers `initialize` and `ping` itself, and `tools/list` and tool calls say "ChaosEngineAI is not running. Start ChaosEngineAI, then try again."
- **Settings > Integrations > Agent access card** (`src/features/settings/AgentAccessPanel.tsx`): on/off switch, copy buttons for the `claude mcp add --transport http chaosengine ... --header "Authorization: Bearer ..."` line and a JSON block for other clients, and "Make a new token". English strings added; the nine other locales hold `null` and fall back to English.
- Tests: `tests/test_mcp_relay.py` runs the real CLI against a live uvicorn server (app closed, access off, access on). `AgentAccessPanel.test.ts` checks the copied line and JSON. Full pytest 2587 passed, vitest 781, `tsc` clean, `i18n:validate` OK.

**Open for Claude**

- Step 4: `SKILL.md`, `docs/features/agent-bridge.md`, locale strings.

**Open for Dan**

- ~~The CLI is not in the release bundle~~ Staged in step 4.
- The card was not looked at in the running app (type-checked and unit-tested only). The card is first in Integrations, above the OpenAI-compatible snippets.
- Not tried with a real Claude Code client yet (step 5).

## Update 2026-11-06, agent bridge step 2

**Done**

- **`POST /mcp`** (`backend_service/routes/agent.py`, package `backend_service/mcp_bridge/`): hand-written JSON-RPC for `initialize`, `ping`, `tools/list` and `tools/call` over streamable HTTP (single JSON replies). The `mcp` SDK is not used, so no new dependency or notice.
- **11 tools:** `system_status`, `recommend_models`, `list_models`, `list_voices`, `generate_image`, `generate_video`, `speak`, `transcribe`, `job_status`, `wait_for_job`, `cancel_job`. Inputs are validated against each tool's schema; errors come back as readable tool errors. Generate tools reuse the Studio routes' own code (size snapping, frame rule, memory gate), so refusals carry the numbers. Every job takes a slot on the step 1 queue as source `agent`; outputs are file paths.
- **Off by default.** While off, `initialize` and `tools/list` answer without a token and every tool call returns "Agent access is off. Turn it on in ChaosEngineAI: Settings > Integrations > Agent access." While on, `/mcp` needs a Bearer token: the new persisted agent token or the app token. The switch and token live in `agent_access.json` beside `settings.json` (not in the settings payload), read and set through `GET/POST /api/agent/access` (`enabled`, `rotateToken`) for the step 3 card.
- `recommend_models` and `list_models` use the catalog fit ratings. Only text-to-image and text-to-video models are offered (image-to-video is out of v1). Recommendations list open-licence ungated models first, then newest; gated and non-commercial flags are read from catalog notes and badges (the catalog has no licence field).
- Tests: `tests/test_mcp_bridge.py` (12: off/on auth, notifications, access round trip, schema errors, job to a path, refusal text, seconds to frames, cancel while queued). Full pytest 2585 passed. Checked against this Mac's real catalog: `list_models` shows Iris-3B, LTX-2.5 and the STT models; `recommend_models` picks T2V / T2I rows only.

**Open for Claude**

- Step 3: stdio relay (`scripts/chaosengine-cli mcp`) and the Settings > Integrations "Agent access" card.

**Open for Dan**

- No real MCP client or real generation through `/mcp` yet (step 5). Tool texts and the licence-hint heuristic are worth a read.
- The agent token is stored in plain JSON (mode 600) in the app data folder, like the HF token in settings.

## Update 2026-11-06, agent bridge step 1

**Done**

- **Shared job queue** (`backend_service/helpers/job_queue.py`, `HEAVY_JOBS`): first come, first served, one heavy job at a time. `POST /api/images/generate`, `/api/video/generate`, `/api/voice/synthesize` and `/api/voice/transcribe` take a slot before the memory gate and the model run. A waiting request can be cancelled through the existing cancel routes (409 `cancelled`). Live speech streaming and voice activity detection are not queued (light, interactive).
- Image and video progress responses carry `queue` (`waiting`, `position`, `agentAhead`); the Studio progress panel shows "Waiting for an agent job to finish" or "Waiting for another job to finish (position N)". New en strings; other nine locales hold `null` placeholders and fall back to en.
- Agent jobs (step 2) will use `HEAVY_JOBS.slot(kind, "agent", label)`.
- Tests: new `tests/test_job_queue.py` (order, peak concurrency 1, positions, cancel while waiting, slot freed after error, progress route). Full pytest 2573 passed, vitest 779, `tsc` clean, `i18n:validate` OK.

**Open for Claude**

- Step 2: `/mcp` endpoint with v1 tools, off by default, `recommend_models`; check the token itself.

**Open for Dan**

- The "waiting" note in the Studio was not looked at in the running app, and no live two-jobs-at-once run yet (planned in step 5).
- Translations for the two new strings in nine locales are still `null`.

## Update 2026-11-06, agent bridge step 0

**Done**

- Branch `feat/agent-bridge` cut from `release/v0.9.10`. Building the [agent bridge](https://github.com/cryptopoly/ChaosEngineAI/blob/feat/agent-bridge/docs/plans/agent-bridge.md) in rollout order (D-18 to D-21).
- **Step 0, auth:** the demo's calls worked without a token because the author's `~/.chaosengine/settings.json` has `requireApiAuth: false` (the Settings toggle). The default is on and the middleware is correct; not a bug. One gap for step 2: the middleware only guards `/api/` and `/v1/`, so `/mcp` must check the token itself.
- **Step 0, settings in files:** new `backend_service/helpers/output_metadata.py`. PNGs get a `tEXt` chunk `chaosengine`, MP4s a `comment` tag (via ffmpeg stream copy), both holding prompt, model, seed, steps, guidance, size, frames, fps and app version. Written at the one save point each in `image_artifacts.py` and `video_artifacts.py`; best effort. New `tests/test_output_metadata.py` (PNG and MP4 round trips). 560 image/video tests and `test_backend_service.py` pass.

**Open for Claude**

- Step 1: one job queue for image, video and voice.

**Open for Dan**

- D-21 still reads "open, after v0.9.10 ships"; you told me to start, so say if it should move to decided.
- MP4 tagging was not run on a real LTX or Wan clip, only on a test clip.

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
- This hub: [tracker pages](tracker/README.md) are generated from `FOLLOW_UPS.md` by `tools/hub_tracker.py`.

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
