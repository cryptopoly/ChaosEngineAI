# Voice feature audit (branch feat/home-studio)

Read-only inventory so the Speak / Voices / Models redesign drops nothing. Paths are relative to the repo root. Line numbers are from the working tree at audit time.

## 0. Headline findings

1. **Voice Models is already a left-sidebar entry.** `voice-models` is a `TabId` (`src/types/system.ts:14`) with its own `TabConfig` (`src/constants/tabs.ts:45`, `group: "voice"`), its own view (`VoiceModelsView`, `src/app/VoiceTabs.tsx:32`, registered at `src/app/tabViews.ts:35`). It is not a tab inside `VoiceStudioTab`. Studio's inner segmented control is only `transcribe | speak | clone` (`src/features/voice/VoiceStudioTab.tsx:19`). What differs from Chat/Image/Video/Music:
   - Voice Models is a bespoke page (`VoiceModelsTab.tsx`), not the shared `ModelsPage` shell fed by an adapter. `ModelsArea` has no `"voice"` (`src/components/models/modelEntry.ts:15`).
   - Voice's group default child is `voice-studio` (`src/constants/sidebarGroups.ts:18`); Images, Video and Music default to their Models page.
   - Voice lists Models second (Studio, Models, Library) rather than first. The sidebar "Models" short label is the same shared key `tabShortLabels.models`.
   - Sidebar entries for Chat/Image/Video/Music/Voice are all **children of a group accordion** (group headers Chat, Images, Video, Voice, Music, Storyboard). There is no top-level "Models" row for any area. `SidebarGroupId` includes `"models"` (`system.ts:34`) but no group or tab uses it.
   If the intent is a top-level (non-grouped) sidebar row, that is a new pattern; today only Dashboard, Server, Logs and Settings are un-grouped (`tabs.ts:20,41,53,54`, `sidebarModel.ts:77-89`).
2. **None of the following exist anywhere in the repo (UI, backend, MCP, CLI):** pitch, a "style" prompt or instruction field, voice design, dialogue / multi-speaker TTS, pronunciation dictionary, a user-facing text clean-up toggle. The redesign's "style box", "pitch" and "design" are net-new backend work. Details in section 6.
3. Custom clean-up exists but is fixed and server-side only (`backend_service/voice_text.py:36`); it applies to read-aloud, not to Studio Speak (see 2.4).

## 1. Voice Studio UI inventory (today)

Navigation shell: sidebar group "Voice" (mic icon, `src/components/icons/SidebarIcons.tsx:171`), children Studio / Models / Library (`tabs.ts:44-46`). Active tab is `activeTab` state in `src/App.tsx:120`; `setActiveTab` is the redirect-aware wrapper (App.tsx ~136-142). Voice state: `voiceRuntime` + `voiceOpenRequest` in `App.tsx:144-145`, fetched once when the backend comes online (`App.tsx:469-484`), exposed to descendants as `VoiceProvider` context (`App.tsx:486-497`, `src/features/voice/VoiceContext.tsx`), and in `AppTabContext` (`src/app/context.ts:145-149`).

### 1.1 Voice Studio screen (`voice-studio`) - `src/features/voice/VoiceStudioTab.tsx`

Gating states (lines 25-60): backend offline or runtime null shows "Waiting for the backend"; `!voiceRuntime.installed` shows a message plus button "Open Voice Models" which calls `onTabChange("voice-models")` (line 53). The message text says "Install it once from Voice > My Models" (stale label; the sidebar label is "Models").

| Control | Behaviour | State | Backend |
|---|---|---|---|
| Mode segmented control: Transcribe / Speak / Clone (lines 63-81) | switches pane; all three stay mounted (hidden by CSS class `voice-studio__pane--hidden`) so queues/generation survive | `mode` in `VoiceStudioTab` (`useState("transcribe")`, line 22); forced to "transcribe" when `openRequest` arrives (line 24-26) | none |

#### Transcribe pane - `src/features/voice/studio/TranscribePanel.tsx`

| Control | Does | State | Backend |
|---|---|---|---|
| Model select (270) | picks STT model; options show "(N GB download)" if not installed | `options.model` (local); default via `pickDefaultSttModel(runtime.sttModels, settings.sttModel)` (35, 88-94) | form field `model` of `POST /api/voice/transcribe` |
| Language select (281) | 17 codes + "Detect automatically" (`VOICE_LANGUAGES` from `VoiceModelsTab.tsx:46-64`) | `options.language`, initial `settings.language` | form `language` |
| Translate to English checkbox (294) | disabled unless model `translate:true` (Whisper Small/Base only) | `options.translate`; snapshot at enqueue (`enqueue`, ~170) | form `task=translate` |
| Label speakers checkbox (303) | disabled unless `runtime.speakersAvailable` | `options.speakers` | form `speakers=true` |
| Number of speakers select (311) | shown when speakers on: Find how many(0),2,3,4,5,6,8 (`SPEAKER_COUNTS`, line 14) | `options.speakerCount` | form `speakerCount` (0-20) |
| "Open Voice Models" link (~325) | shown when speakers not available (pack needs update) | none | none |
| Record button / level meter / timer (~338-360) | mic capture to WAV via `startMicCapture` (`audio.ts`), auto-stops at limit | `recording`, `level`, `seconds`, `captureRef` | enqueued like a file |
| Drag-and-drop area + "Choose files" + hidden multi-file input (~335-380) | accepts audio/video, 500 MB cap (`MAX_FILE_BYTES` line 12); several files queue | `jobs` | same |
| `MicSettingsButton` (~385) | opens OS microphone privacy page on a blocked-mic error (`micSettings.ts`) | none | Tauri opener |
| Queue list: Waiting / Transcribing timer / Done in N s / Cancelled / Failed; per-row Open / Cancel / Remove (~389-410) | one job at a time (effect at ~125-165); cancel aborts the fetch (AbortController) | `jobs`, `abortRef` | `transcribeBlob` always sends `words:true`, `save:true`, `task`, `speakers`, `speakerCount` (`src/api/voice.ts:64`) |
| Transcript panel (second `Panel`, ~415-445) | opens the first finished job, or a Library item sent via `openRequest` | `open` | `GET /api/voice/library/{id}` and `/audio` |

Transcript editor - `studio/TranscriptEditor.tsx`: title input (127), speaker-rename inputs (163), clickable time/word seek (186, 224), double-click line edit with inline input (195-213) and edit button (234), autosave status "Saving / Not saved / Saved to Library" (139-142), hand-off buttons "Summarise in chat / Action items / Meeting notes" (257, prompts sent to Chat via `onSendToChat`, `VoiceTabs.tsx:20-24` appends to chat draft and switches to Chat), Copy (265), Export buttons txt/md/srt/vtt/json (273-276). Saves go through `updateLibraryItem` -> `POST /api/voice/library/{id}` (fields title, text, segments, speakers). Pure helpers: `studio/transcript.ts`.

#### Speak pane - `src/features/voice/studio/SpeakPanel.tsx`

| Control | Does | State | Backend |
|---|---|---|---|
| Text textarea (136) | up to 200,000 chars (`MAX_CHARS`); placeholder says Markdown/code are tidied up. Blank line = new paragraph, `[pause]` = pause (subtitle string) | `text` (local) | n/a |
| Voice select (148), grouped by language, "(M)/(F)", only `available !== false` | Kokoro voice id | `voice` (local, init from `settings.ttsVoice`; not written back) | `voice` field of `/api/voice/synthesize` |
| Listen/Stop preview (161) | plays a fixed line in the voice's language (`PREVIEW_LINES`, 17-27) via `speechPlayer.play` | `speechPlayer` singleton (`speechPlayer.ts`) | `POST /api/voice/speak/stream` |
| Speed slider 0.5-2.0 step 0.1 (176) | | `speed` (local, init from `settings.speed`) | `speed` |
| Generate audio / "Generate audio (N parts)" (202) | `splitSpeech` (speech.ts) cuts into paragraphs, `[pause]` markers and <=4000-char sentence chunks; each chunk is a separate `POST /api/voice/synthesize`; WAVs are joined in the browser with 0.6 s paragraph gap and 1.2 s pause (`speech.ts:9-11`) | `progress`, `cancelRef`, `result` | `POST /api/voice/synthesize` (JSON text, voice, speed) |
| Progress bar + Cancel (187-197) | cancels between parts only | | |
| Result: audio element, duration, WAV download (214), MP3 download (217), Save to Library (220), Send to Storyboard menu | MP3 = `POST /api/voice/encode`; save = `POST /api/voice/library/speech`; storyboard send saves first then reads `audioPath` | `result`, `savedId`, `busyExport` | see section 2 |

#### Clone pane - `src/features/voice/clone/ClonePanel.tsx` (standalone mode) + `NewVoiceForm.tsx`

| Control | Does | State | Backend |
|---|---|---|---|
| Standing statement (`CloneStatement`, 27-37) | fixed text, no opt-out; asserted by e2e and vitest | none | none |
| "Cloning model" select (CloneBody ~165) | Qwen3-TTS 0.6B / 1.7B (name, GB) | `modelKey` (local), preselect `pickCloneModel` (`cloneLogic.ts:47`) | `model` of `/api/voice/clone/speak` |
| Model state: Downloaded badge / Download button / "Downloading..." and "Installing the watermark component..." (176-195) | polls `GET /api/voice/clone/models` every 3 s while downloading (line 85-89) | `models` | `POST /api/voice/clone/models/download {key}` |
| Message when not Apple Silicon ("Voice cloning runs on Apple Silicon Macs for now") and when `allowedHere` false | | `models.available`, `allowedHere` | `GET /api/voice/clone/models` |
| Cloned voices radio list with consent line ("own voice, confirmed DATE" or "NAME gave permission..."), per-voice Delete with `window.confirm` (205-235) | | `voices`, `voiceId` | `GET /api/voice/clone/voices`, `DELETE /api/voice/clone/voices/{id}` |
| "New cloned voice" button (237) | swaps list for `NewVoiceForm` | `adding` | |
| Text textarea (max 5000 chars) + Generate audio (disabled until model installed, voice picked, text) (245-255) | one request, no parts/pause/speed | `text`, `busy`, `result` | `POST /api/voice/clone/speak {voiceId,text,model}` (15 min timeout, `cloneApi.ts:79`) |
| Result: audio, duration, "Labelled as AI-made" badge, WAV (278), MP3 (281), "Saved to Library" (always saved server-side), Send to Storyboard (clonedVoice flag) | MP3 via `GET /api/voice/library/{id}/audio?format=mp3` | | |
| Checkbox "Allow agents and other computers to use cloned voices (off by default)" (297-303) | writes `allowAgentCloning` | `settings.allowAgentCloning` via `saveSettings` -> `POST /api/voice/settings`; backend only accepts it from origin `ui` (`routes/voice.py:126-137`) | |

New voice form (`NewVoiceForm.tsx`): radio basis "My own voice" / "Someone who gave me permission" (~98-106), Speaker's name (max 60), Name for this voice (optional, max 80), file input (WAV/MP3/M4A/FLAC/OGG), Record/Stop/Record again with playback and 3-30 s length check (`cloneLogic.ts:21-35`), "Words spoken in the sample" optional (max 2000; otherwise the backend transcribes it), permission checkbox, Save voice (gated by `canSaveVoice`), Cancel, `MicSettingsButton`. Posts multipart to `POST /api/voice/clone/voices` with `permissionConfirmed=true` always (`cloneApi.ts:55`). An un-standalone collapsed `<details>` mode of `ClonePanel` exists (lines 62-71) but nothing in the app renders it (only tests do, `clone.test.tsx:65`).

### 1.2 Voice Models screen (`voice-models`) - `src/features/voice/VoiceModelsTab.tsx`

Panels (all inside `VoiceModelsBody`, 505-549):

1. **Voice pack panel** (`VoicePackPanel`, 152-224): lane description (MLX / CUDA / NVIDIA / CPU; 78-91), status badge (healthy "Installed, checked OK (time)", installed, not installed), about text, Install / Update / Repair primary button (mode from `voicePackMode`, `voicePackState.ts`), Cancel during install, collapsed "Troubleshooting" with small Reinstall, update-features hint, install log `<details>`. State: `useVoicePackJob` (93-121) polls `GET /api/voice/install/status` every 1.5 s while running and calls `onRefreshVoiceRuntime`. Calls: `POST /api/voice/install {extras}` (installed extras are re-sent so they are kept), `POST /api/voice/install/cancel`.
2. **Speech-to-text models** (only when pack installed, 524-541): grid of `SttModelCard` (231-293) - name, languages, GB, backend, badges Recommended/Downloaded, radio "Use for dictation" (writes `settings.sttModel` = `key ?? id`), Download button with % (uses the *generic* `POST /api/models/download {repo}` and polls `GET /api/models/download/status` every 2 s, matching on repo), note "Downloads on first use if you skip this."
3. **Voices** (`VoicesPanel`, 295-380): Kokoro voices grouped by language, chip per voice: name button (sets `settings.ttsVoice`, i.e. the chat read-aloud/call voice) + preview play/stop button (`PREVIEW_TEXT`, a different set of strings from Speak's `PREVIEW_LINES`); disabled when `available:false`. Mandarin add-on prompt + "Add Mandarin Chinese voices" button calls `pack.start(["chinese"])`; Japanese is permanently unavailable on the MLX lane (comment in `voice_runtime.py:200-206`). Subtitle hardcodes "54 voices in 9 languages" (it matches the 54-voice list today).
4. **Voice settings** (`SettingsPanel`, 399-495); each change calls `saveSettings` -> `POST /api/voice/settings` immediately:
   - Auto-send after dictation (`autoSend`)
   - Read every chat reply aloud (`autoRead`)
   - Dictation language select (`language`; hint "Whisper uses it; Parakeet detects the language itself")
   - Reading speed slider 0.5-2.0 (`speed`)
   - Voice call pause that ends a turn, slider 300-2000 ms (`vadSilenceMs`; backend allows up to 3000)
   - Voice call interrupt by speaking (`bargeIn`)
   - Free voice memory after N minutes idle, number 0-240 (`idleUnloadMinutes`)
   - Worker status line + "Unload now" (`POST /api/voice/worker/stop`)

Settings persist in `~/.chaosengine/voice/settings.json` (`voice_runtime.py:229-278`); `DEFAULT_SETTINGS` lists exactly: sttModel, language, ttsVoice, speed, autoSend, autoRead, vadSilenceMs, bargeIn, idleUnloadMinutes, allowAgentCloning. The frontend has a mirrored fallback `DEFAULT_SETTINGS` in `VoiceContext.tsx:19-29` (lacks `allowAgentCloning`).

### 1.3 Voice Library screen (`voice-gallery`, label "Voice Library") - `src/features/voice/VoiceLibraryTab.tsx`

Search input (debounced 200 ms), filter segmented All / Transcripts / Speech, `ProjectFilter` dropdown + `ProjectHeader` (with "Open in Storyboard" via `writeOpenProjectId`), "Delete selected (N)", select checkbox per card, per-card: Open (transcript -> sets `voiceOpenRequest` and goes to `voice-studio`, `VoiceTabs.tsx:52-60`), Play (lazy audio fetch), Export select (txt/md/srt/vtt/json for transcripts; original audio; MP3), Delete (confirm), Send to Storyboard (speech only). Backend: `GET /api/voice/library?q=&kind=`, `POST /api/voice/library/delete {ids}`, `GET /api/voice/library/{id}`, `/audio?format=mp3`. Note: cloned voices are stored in the same library as `kind:"clone"` records but `list_items` hides them unless `kind=clone` is asked (`voice_library.py:178`); TS `VoiceLibraryKind` is only `"transcript" | "speech"` (`types/voice.ts`).

### 1.4 Voice controls outside the Voice group (Chat)

- `ComposerMicButton` (`ChatComposer.tsx:404`): click or hold to dictate, Ctrl/Cmd+Shift+M; uses `settings.sttModel`, `settings.language`, `autoSend`; calls `transcribeBlob` without `save`. If pack missing it calls `openVoiceModels()`.
- `ReadAloudButton` per assistant message (`ChatThread.tsx:175`) and `SpeechPlayerBar` (`ChatTab.tsx:17`): pause/resume/stop; uses `speakStream` -> `POST /api/voice/speak/stream`.
- Auto-read effect (`ChatTab.tsx:218-230`) uses `settings.autoRead`, `ttsVoice`, `speed`.
- Voice call: Call button in `ChatHeader.tsx:195` (title text) and `ChatTab.tsx:477-509`; blocked reasons list "Voice Models" and click opens `voice-models`. `VoiceCallOverlay` (mute, end, status, "headphones" hint), `useVoiceCall.ts` (VAD loop over `POST /api/voice/vad`, STT, streams the chat reply with `voiceTurn:true`, speaks sentence by sentence; `speechText.ts`; `endpointer.ts`).
- Global open-models hook: `openVoiceModels` -> `setActiveTab("voice-models")` (`App.tsx:495`).

## 2. Backend routes

All plain `def` routes in `backend_service/routes/voice.py` (router has no prefix; clone router is included at the bottom, line 463-466) and `routes/voice_clone.py`. Registered via `routes/__init__.py`.

### 2.1 Runtime / settings / install (`routes/voice.py`)

| Route | Request | Notes |
|---|---|---|
| `GET /api/voice/runtime` (103) | none | capabilities + `sttModels` + `ttsVoices` + `settings`; built from `voice_runtime.get_voice_capabilities` (`voice_runtime.py:280`) |
| `GET /api/voice/settings` (119) | none | |
| `POST /api/voice/settings` (126) | `SettingsRequest` (all optional): sttModel (<=128), language (<=8), ttsVoice (<=32), speed 0.5-2.0, autoSend, autoRead, vadSilenceMs 300-3000, bargeIn, idleUnloadMinutes 0-240, allowAgentCloning | `allowAgentCloning` needs origin `ui` else 403 and is audit-logged |
| `POST /api/voice/install` (140) | `{extras: string[]}` (max 8) | unknown extras 400. Only extra today: `chinese` (`voice_env.py:105-107`) |
| `GET /api/voice/install/status` (150) | | job: phase, percent, log, extras, running |
| `POST /api/voice/install/cancel` (157) | | |
| `POST /api/voice/worker/stop` (164) | | `voice_client.stop_all()` |

### 2.2 Speech-to-text

| Route | Request | Notes |
|---|---|---|
| `POST /api/voice/transcribe` (176) multipart | `audio` file; `model` (""=default), `language`, `task` transcribe/translate, `words`, `speakers`, `speakerCount` 0-20, `save` | 500 MB cap, allowed suffixes list at line 21-23; runs in `HEAVY_JOBS.slot("voice","ui","transcript")`; `save` stores a `transcript` library item and returns `libraryId`. Result shape in `voice_runtime.transcribe_file` (316) |
| `POST /api/voice/vad` (251) | `{session (^[A-Za-z0-9_-]{1,64}$), pcm base64 16 kHz PCM16 (<=90,000 chars), reset}` | Silero VAD probabilities per 32 ms frame, separate VAD worker (`voice_client.get_vad_worker`) |

Models catalog (`voice_runtime.py:36-95`): Parakeet TDT 0.6B v3 (default on MLX), Nemotron 3.5 ASR 0.6B (MLX only, 8-bit), Whisper Large v3 Turbo (default non-MLX; `translate:false`), Whisper Small, Whisper Base (`translate:true`). Aliases: large-v3-turbo, turbo, small, base. Diarization (speaker labels) uses sherpa-onnx files (`voice_env.py:50`).

### 2.3 Text-to-speech

| Route | Request | Notes |
|---|---|---|
| `POST /api/voice/synthesize` (266) | `SynthesizeRequest` {text 1-20000, voice <=32 default `af_heart`, speed 0.5-2.0} | returns `audio/wav` (whole text in one WAV). Used by Speak pane and MCP-free |
| `POST /api/voice/speak/stream` (283) | same body | NDJSON, one line per sentence after `voice_text.speech_sentences` (clean-up + sentence split); lines `{index,total,text,audio(b64 WAV),durationS}` then `{done:true}`; errors in-stream. Used by read-aloud, preview, voice call |
| `POST /api/voice/encode` (433) multipart | `audio`, `format=mp3` only | cloned-voice AI tag preserved into ID3 comment |

Voices: 54 Kokoro voices (`voice_runtime.py:176-182`), 9 languages (`:185-195`), default `af_heart`; availability by lane (Japanese off on MLX; Chinese needs the `chinese` extra).

### 2.4 Text clean-up (not exposed as an option)

`backend_service/voice_text.py`: `clean_for_speech` (36) strips code fences, `<think>`, `<tool_call>`, inline code, images, links, URLs, HTML, headings, list markers, quotes, emphasis, table rules; `split_sentences` (63, 400-char cap) and `speech_sentences` (89, `MAX_SPEAK_CHARS` 20,000). Only the *stream* route uses it. `POST /api/voice/synthesize` sends text as-is to the worker (the Speak pane placeholder claims "Markdown and code are tidied up before speaking", which is not true on the `synthesize` path; verify before relying on it). Pause markers (`[pause]`) are handled client-side only in `studio/speech.ts`.

### 2.5 Library (`routes/voice.py`, store `backend_service/voice_library.py`)

| Route | Request |
|---|---|
| `GET /api/voice/library` (336) | query `q` (<=200), `kind` in transcript/speech/clone |
| `POST /api/voice/library/speech` (345) multipart | `audio`, `text` (<=200,000), `voice` (<=32), `speed`, `title` (<=200); keeps cloned-voice tag flag from WAV |
| `POST /api/voice/library/delete` (374) | `{ids: string[1..500]}` |
| `GET /api/voice/library/{id}` (382) | returns item + `audioPath` |
| `POST /api/voice/library/{id}` (394) | `LibraryUpdateRequest`: title, text, segments (<=50,000, each start/end/text/speaker/words), speakers map |
| `DELETE /api/voice/library/{id}` (409) | |
| `GET /api/voice/library/{id}/audio?format=mp3` (418) | |

Item fields persisted (`voice_library.py:129-138`): language, model, voice, durationS, source, task, speed, clonedVoice, cloneVoiceId, projectId, projectName, shotId (+ provenance stamp). Storage `library_dir()`; legacy gallery migration at `_migrate_gallery` (70).

### 2.6 Clone (`routes/voice_clone.py`, logic `voice_clone.py`, safety `voice_clone_safety.py`, catalog `catalog/voice_models.py`)

Every route except `GET /models` is behind `_guard` (`clone_allowed`: UI window on this machine, or agent only with `allowAgentCloning` + token `clone` scope).

| Route | Request |
|---|---|
| `GET /api/voice/clone/models` (41) | returns models, `available` (MLX lane only), `allowedHere` |
| `POST /api/voice/clone/models/download` (53) | `{key}` (1-64) - downloads pinned-revision weights then the AudioSeal watermark component |
| `GET /api/voice/clone/voices` (64) | |
| `POST /api/voice/clone/voices` (72) multipart | `sample` file, `speakerName` (1-60), `basis` own/permission, `permissionConfirmed`, `name` (<=80), `referenceText` (<=2000); sample must be .wav/.mp3/.m4a/.flac/.ogg and 3-30 s; transcribes if no reference text |
| `DELETE /api/voice/clone/voices/{id}` (121) | |
| `GET /api/voice/clone/watermark/{item_id}` (131) | detects the watermark on a library item - **no UI calls this** |
| `POST /api/voice/clone/speak` (149) | `CloneSpeakRequest` {voiceId 1-40, text 1-5000, model <=64, permissionConfirmed bool (required from agents)}; saves a tagged+watermarked WAV to the library; returns the library item |

Clone catalog: Qwen3-TTS 0.6B Base (default, 2.52 GB) and 1.7B Base (4.54 GB), Apache-2.0, MLX lane only.

### 2.7 Voice runtime / venv

`backend_service/voice_env.py`: separate venv at `~/.chaosengine/voice-venv` (override `CHAOSENGINE_VOICE_VENV`, line 35), data dir `~/.chaosengine/voice`, lanes `mlx` / `portable` (`lane()` 118), packages per lane (84-102), CUDA extras, `FEATURES = ["vad","speakers"]` (74), install job `start_install/cancel_install/job_status` (372-395), verify imports (109), watermark install (276). `voice_worker.py` is the long-lived stdin/stdout JSON worker (ops include synthesize, transcribe, diarize, encode, vad); `voice_client.py` holds the worker handle(s) with idle unload; `warm_worker_client.py`, `voice_kokoro_onnx.py` (portable TTS), `voice_worker_watermark.py`. `routes/setup/__init__.py:151-170` holds pinned requirement notes for voice backends (not a Voice UI).

### 2.8 MCP tools (`backend_service/mcp_bridge/`)

| Tool | Scope | Fields | Source |
|---|---|---|---|
| `list_voices` | read | none; returns available Kokoro voices (id, name, gender, language) | `tools.py:46,212`; `models.py:141` |
| `speak` | generate (job) | text (<=20000), voice, speed 0.5-2.0, project props | `tools.py:92-104,409-435` |
| `transcribe` | generate (job) | path, language only (no model/words/speakers/task) | `tools.py:113-120,452-468` |
| `list_models` / `recommend_models` kinds `voice` and `stt` | read | `KINDS` includes voice/stt (`models.py:7`; `tools.py:23` enum). `recommend` for kind `voice` returns an empty list with a note | `models.py:131-165` |
| `list_cloned_voices` | clone | none | `voice_clone_tool.py:20-26` |
| `speak_cloned` | clone (job) | voice_id, text<=5000, permission_confirmed (must be true), model enum 0.6b/1.7b, project props | `voice_clone_tool.py:27-45,87-107` |
| `storyboard_captions generate` | generate | transcribes voice-lane clips | `storyboard_tools.py:82-87,358` |

Scopes: `scopes.py:10-17` (read tools, JOB_TOOLS incl. speak/transcribe/speak_cloned, `_CLONE_TOOLS`), annotations 75-96. `access.py:21` scopes `read, generate, files, download, clone`; clone is never default. Golden contract file `tests/golden/mcp_tools_list.json` pins the tool list (speak at ~305, transcribe ~358); `INSTRUCTIONS` in `protocol.py:15-24` mentions cloned voices. Job cancel for voice kills the worker (`jobs.py:292`). Agents cannot create or delete a cloned voice, by design.

### 2.9 CLI

`pyproject.toml:14` only declares `chaosengine-api`. `scripts/chaosengine-cli` has no voice, speak or transcribe shortcut (only a generic description string at line 1417); the generic `call <METHOD> <PATH>` dispatcher reaches every `/api/voice/*` route (`docs/cli/overview.md`). `scripts/e2e_test_suite.py` plus `scripts/e2e_phases/phase_14.py` (cloning), `phase_15-18` touch voice through `_voice_http`.

## 3. Other places a voice option is exposed

| Where | What | Refs |
|---|---|---|
| Settings page > Integrations > Agent access | token scope "Allow voice cloning" and "Also allow cloning from other computers" (`remoteClone`) | `src/features/settings/AgentAccessPanel.tsx:140-160` |
| Settings page (general) | no voice section. The `allowAgentCloning` toggle lives in the Clone pane, not Settings. Comments in `voice_clone_tool.py:5` and `models.py:157` still say "Voice > Settings" / "Voice > Setup" - neither exists | quirk |
| Setup page | there is no Voice setup UI. Voice install is in Voice Models. `routes/setup/__init__.py` is a runtime-package check | |
| Terms of use | one line: "Only clone a voice with the speaker's consent." | `src/features/terms/TermsOfUse.tsx:12` |
| Chat | mic button, read aloud, auto-read, voice call (see 1.4); `voiceTurn` request flag | `models/__init__.py:196`, `state/generation.py:177,441` |
| Storyboard | project default `defaults.voice` (free-text, max 200) in `ProjectDefaultsPanel.tsx:54,125-133`, backend `Defaults.voice` (`storyboard/models.py:100`); **nothing reads it** (no backend consumer found, only stored/displayed in the summary line `ProjectDefaultsPanel.tsx:64-69`). Voice lane clips: Add audio panel with Voice Library picker and "This clip uses a cloned voice" checkbox (`mixer/AddAudioPanel.tsx:31,110-113`); clip kind `voice` (`storyboard/models.py:181`); cloned-voice flag forces the export AI tag (`models.py:304-309`, `ExportPanel.tsx:62`); "Make from voice" captions (`CaptionsPanel.tsx`, `routes/storyboard.py:234`); duck-under-voice (`ClipInspector.tsx:205`); "Send to Storyboard" menu on Speak, Clone and Library (`sendToStoryboardLogic.ts:11,24,129-132`); the Voice Library "Project" filter/badge |
| Tauri | microphone permission allowance `src-tauri/src/lib.rs:311-361`; capability file allows exactly the two OS mic-settings URLs (`micSettings.ts:4`) | |
| Music Library | reuses `.voice-library__toolbar` CSS (`MusicLibraryTab.tsx:299`) - shared style to keep | |
| Docs | `docs/usage/voice.md`, `docs/plans/voice-io.md`, `docs/hub/progress/v-voice.md` | |
| Process kill list | `voice_worker.py` named in `inference/controller.py:465` | |

## 4. How Chat / Image / Video / Music expose Models in the left menu, and how to add "Voice Models"

### 4.1 Mechanism (single source of truth)

1. `src/types/system.ts` - `TabId` union (`:6-36`) and `SidebarGroupId` (`:38-47`).
2. `src/constants/tabs.ts` - `tabs: TabConfig[]`; each entry has `id`, `label`/`labelKey`, `caption`/`captionKey`, `group`, `shortLabel`/`shortLabelKey`. Models rows:
   - `chat-models` (line 33, group chat, caption key `tabCaptions.chatModels`)
   - `image-models` (34, group images, `tabCaptions.imageModelsPage`)
   - `video-models` (37, group video, `tabCaptions.videoModelsPage`)
   - `music-models` (47, group music)
   - `voice-models` (45, group voice, `tabCaptions.voiceModels`)
   All use short label key `tabShortLabels.models` ("Models"). Array order = order in the sidebar group (Chat has Models last, Images/Video/Music first, Voice second).
3. `src/constants/sidebarGroups.ts` - group definitions with `defaultChild`, optional `readyChild`, `section: "make" | "run"` (voice at line 18; music uses `defaultChild: music-models`, `readyChild: music-studio`, line 19).
4. `src/components/sidebarModel.ts` - `buildItems` groups tabs by `tab.group`; `arrangeRows` orders sections; `resolveGroupTarget` picks the click target from remembered child, `readyChild` (if in `readyGroups`) or `defaultChild`. `src/components/Sidebar.tsx` renders it (accordion by default, "tabs" mode shows one nav button and `SubtabBar.tsx`). Icons: `src/components/icons/SidebarIcons.tsx` (`groupIcon.voice` at 171). Remembered child: `src/hooks/useSidebarPrefs.ts` (localStorage `chaosengine.sidebar.prefs.v1`, migrates via `tabRedirects.migrateLastChildByGroup`).
5. `src/app/tabViews.ts` - `TAB_VIEWS` map from tab id to view; unknown ids fall back to Dashboard (`:58-61`). Views per area: `ModelsTabs.tsx` (ChatModelsView -> `features/models/ChatModelsPage`), `ImageTabs.tsx` (-> `features/images/ImageModelsPage`), `VideoTabs.tsx` (-> `features/video/VideoModelsPage`), `MusicTabs.tsx` (-> `features/music/MusicModelsPage`), `VoiceTabs.tsx` (-> `features/voice/VoiceModelsTab`).
6. Shared Models shell: `src/components/models/` (`ModelsPage.tsx`, `ModelCard`, `ModelFilterBar`, `modelEntry.ts` with `ModelsArea = chat|image|video|music`, `adapters/{chat,image,video,music}.ts`). Per-area availability filter state is in `App.tsx:124-133` (`Record<ModelsArea,...>`), surfaced as `ctx.modelsAvailability` / `ctx.setAreaAvailability` (`context.ts:103-104`). Voice does not use any of this.
7. `src/constants/tabRedirects.ts` - `MODELS_TAB_BY_AREA: Record<ModelsArea, TabId>` (35-40), `modelsAreaForTab` (59-64), legacy id redirects (Discover/My Models).
8. `src/constants/features.ts` - `HOME_STUDIO_TABS` includes music/storyboard ids (flag-gated); Voice is not flag-gated.

### 4.2 Steps to add (or restyle) a "Voice Models" sidebar entry

It already exists as `voice-models`. If you keep it as a Voice-group child and just make it consistent with the other areas:

1. `tabs.ts:45` - keep id; move the entry before `voice-studio` if Models should lead like Images/Video/Music (order = sidebar order). Update caption if the page changes (key `tabCaptions.voiceModels`).
2. `sidebarGroups.ts:18` - set `defaultChild: "voice-models"` and optionally `readyChild: "voice-studio"` (as Music does) so the group opens on Models until the pack is installed and on Speak after.
3. If the redesign renames Studio to Speak and adds Voices: change `voice-studio`/`voice-gallery` ids or labels in `types/system.ts`, `tabs.ts`, `tabViews.ts`, `VoiceTabs.tsx`, and add redirects for the retired ids in `tabRedirects.ts` (`LegacyTabId`, `LEGACY_TAB_REDIRECTS`, and `migrateLastChildByGroup` handles the stored `lastChildByGroup`, test at `tabRedirects.test.ts:95-96`).
4. To adopt the shared Models shell: add `"voice"` to `ModelsArea` (`modelEntry.ts:15`), `MODELS_TAB_BY_AREA` + `modelsAreaForTab` (`tabRedirects.ts`), the `useState` default object (`App.tsx:124-129`), a new `components/models/adapters/voice.ts` (STT models, Kokoro voices, clone models are three different sources and use three different download mechanisms: `/api/models/download`, voice pack extras, `/api/voice/clone/models/download`), and swap `VoiceModelsView` (`VoiceTabs.tsx:32`) to a `VoiceModelsPage` receiving `availability` like `MusicTabs.tsx:6-17`.
5. If the entry should instead be a top-level sidebar row (not inside the Voice accordion): remove `group` from the `TabConfig`, add the icon to `standaloneTabIcon` in `SidebarIcons.tsx`, and update `arrangeRows` ordering (currently singles are listed last, after the Run section, `sidebarModel.ts:88`). Nothing else in the codebase does this for a Models page, so it is a new layout decision.
6. i18n: `src/locales/<locale>/common.json` for **all 10 locales** (de, en, es, fr, ja, ko, pt-BR, ru, zh-CN, zh-TW; verified each has 9 matching voice tab keys). Keys in `tabs` (en lines 36-38: voiceStudio, voiceModels, voiceLibrary), `tabCaptions` (71-73), `tabShortLabels` (104-107, plus shared `models`), `tabGroups.voice` (121). Panel text lives in `studio.json` under `voice.*` starting at line 650 in all 10 locales. New keys must be added to every locale; `npm run i18n:validate` (`scripts/i18n-validate.mjs`, step 9 of `scripts/pre-build-check.sh:250`) checks parity, orphans and ICU syntax against `en`. The stale key `tabShortLabels.voiceModels` ("My Models") is unused (tabs.ts uses `tabShortLabels.models`).
7. e2e: sidebar helpers are text-based, not id-based - `openTab(page, group, child)` / `goTo(page, group, child)` in `e2e/tests/helpers.ts:19-32` click `.nav-group-header` containing the group label then `.nav-child-button` matching `^child$`. A rename of "Studio", "Models" or "Library" labels, or moving Models out of the group, breaks every `goTo(page,"Voice",...)`.
8. Unit tests that enumerate the sidebar: `src/components/__tests__/Sidebar.test.tsx` (row order at ~64-68 lists `voice` between video and music; music default/ready child at ~48-52), `src/constants/__tests__/sidebarGroups.test.ts` (defaultChild belongs to group; every grouped tab has a shortLabel; Models pages short label "Models" for chat/images/video; defaults assertions 62-71), `src/constants/__tests__/tabRedirects.test.ts:80,95-96`.

## 5. Tests that cover Voice today (and what a redesign will touch)

### pytest (`tests/`)
- `test_voice.py` (9 tests), `test_voice_io.py` (47), `test_voice_library.py` (23), `test_voice_clone.py` (22), `test_voice_watermark.py` (8) - routes, runtime, library store, clone safety. Any change to request models, settings keys, `list_items` filtering or `DEFAULT_SETTINGS` lands here.
- MCP: `test_mcp_contract.py:22,30,50-51` (arg samples for list_voices, list_cloned_voices, speak_cloned; "voice"/"language" in UNCAPPED_OK) with golden `tests/golden/mcp_tools_list.json` - adding fields to `speak`/`transcribe` requires regenerating the golden. Also `test_mcp_bridge.py`, `test_mcp_bridge_v2b.py`, `test_mcp_jobs.py`, `test_mcp_scopes.py`, `test_job_queue.py` (voice job kind).
- Others referencing voice: `test_non_chat_models.py:59`, `test_backend_service.py:1072` (voiceTurn), `test_provenance.py`, storyboard tests (`test_storyboard_*.py`, voice clips / cloned flag).
- Script e2e phases: `scripts/e2e_phases/phase_14.py` (clone: routes, refusal, full run), `phase_15.py`-`phase_18.py` (voice HTTP helper, captions, MCP scope checks at phase_17:149).

### vitest (`src/`)
- `features/voice/VoiceStudioTab.test.ts` (imports `pickDefaultSttModel` **from `VoiceStudioTab`**, which re-exports it; moving that function breaks it, and `App.tsx:49` and `context.ts:15` import the `OpenRequest` type from the same file).
- `features/voice/studio/studio.test.ts` (transcript helpers, `splitSpeech`, `joinWavs`), `clone/clone.test.tsx` (logic + `ClonePanel` default collapsed statement + `NewVoiceForm` first-render Save disabled; imports `ClonePanel`, `CloneStatement`), `voicePackState.test.ts`, `audio.test.ts`, `micSettings.test.ts`, `conversation/conversation.test.ts`, `src/api/voice.test.ts` (API client contract: synthesize body, library query, transcribe fields, stream parsing, settings/install).
- Cross-cutting: `components/__tests__/Sidebar.test.tsx`, `constants/__tests__/sidebarGroups.test.ts`, `constants/__tests__/tabRedirects.test.ts`, storyboard `sendToStoryboard.test.tsx`, `mixer/__tests__/*` (voice lane), `components/__tests__/AudioKit.test.tsx`, `ProjectFilter.test.tsx`, `app/__tests__` (check for tab view map coverage if ids change).

### Playwright (`e2e/tests/`)
- `07-voice-clone.spec.ts` - `goTo(page,"Voice","Studio")` then tab role "Clone" (`openClonePanel`), plus labels "New cloned voice", "Save voice", "Record", "Speaker's name", "Name for this voice (optional)", "Voice sample file", `.voice-clone__statement`, `.voice-clone`, placeholder "Text to say in the cloned voice.", button "Generate audio" inside `.voice-clone`, text "Labelled as AI-made", "Saved to Library", body of POST `/api/voice/clone/speak` expected `{voiceId,text,model}` with no `disclaimer`, a 375 px overflow check. Heaviest rewrite target.
- `11-voice-pack.spec.ts` - `goTo(page,"Voice","Models")`, heading "Voice pack", `.voice-pack`, `.voice-pack__troubleshoot`, `.voice-group` Mandarin add-on buttons, install body `{extras:["chinese"]}`.
- `03-libraries-projects.spec.ts:161-` - `goTo(page,"Voice","Library")`, route mock for `/api/voice/library`.
- `09-mobile.spec.ts:64-70` - `openTab(page,"Voice","Library")` heading "Voice Library"; `05-storyboard.spec.ts`, `helpers.ts:79-99` (`seedProject` with `clonedVoice`) cover the Storyboard voice lane.
- No e2e exists for Transcribe or Speak panes, Chat dictation/read-aloud/call.

## 6. Gaps, quirks and dead options

Not present anywhere (needed for the redesign, so new work):
- **Pitch**: no field in UI, `SynthesizeRequest`, worker, MCP, or settings.
- **Style box / instruction prompt**: none. Kokoro has no style control; Qwen3-TTS Base is reference-audio cloning only (`voice_clone.speak` sends `refAudioPath`/`refText`). A style prompt needs a new model/engine.
- **Voice design / "get more voices"**: none (no design route, no MCP tool, no voice-download beyond the Mandarin extra).
- **Dialogue / multi-speaker TTS**: none. "Speakers" in the product means diarization labels on transcripts only.
- **Pronunciation dictionary**: none.
- **Clean-up toggle**: clean-up is automatic and fixed, and only on the streaming route (2.4).

Quirks and dead code:
1. Two disjoint "speak" paths: Speak pane (Kokoro, `/synthesize`, speed, parts/pauses, 200k chars, client-side stitching) vs Clone pane (`/clone/speak`, no speed, no pauses, 5000 chars, always saved to Library). A merged Speak screen needs one request model; cloned voices are not selectable as the chat read-aloud voice (`settings.ttsVoice` is a Kokoro id, `SynthesizeRequest.voice` max 32 chars and validated against Kokoro only in `voice_runtime.synthesize`).
2. Speak pane's `voice` and `speed` are local copies initialised from settings and never saved back; the Models page "Reading speed" and voice chip write the global settings that only Chat/MCP use.
3. Speak pane placeholder promises Markdown/code tidy-up, but `/api/voice/synthesize` does not call `clean_for_speech` (2.4).
4. `[pause]` and paragraph gaps are a client-only convention; MCP `speak` and `speak/stream` do not honour `[pause]`.
5. Storyboard `defaults.voice` is stored but unused by any generator (3).
6. `GET /api/voice/clone/watermark/{item_id}` has no caller in the UI (the Clone panel only says the watermark exists).
7. `ClonePanel` non-standalone `<details>` mode is only used by tests; `CloneStatement` is asserted by both vitest and e2e text, so wording changes break tests.
8. Library `kind:"clone"` is hidden from the default listing and absent from the TS `VoiceLibraryKind`; a unified Voices list must read `/api/voice/clone/voices` (clone) and `/api/voice/runtime` `ttsVoices` (built-in) separately; there is no combined endpoint. Clone list is empty/forbidden (403) from non-UI origins unless agent cloning is allowed, and the whole clone area is MLX-only (`available_here`).
9. Stale copy: studio gating message says "Voice > My Models" (`VoiceStudioTab.tsx` needsPack default); `voice_clone_tool.py:5` and `models.py:157` say "Voice > Settings"/"Voice > Setup"; `voice_clone.py` errors say "Download it in Clone a voice first". `tabShortLabels.voiceModels` ("My Models") is unused.
10. `vadSilenceMs`: slider max 2000 ms but the API accepts up to 3000 (`SettingsRequest`). `sttModel` setting is the catalog key; Studio's Transcribe model select uses repo ids and can override per job without touching settings.
11. Dictation language is a Voice Models setting but also seeds the Transcribe pane language; per-pane override is not persisted.
12. `SidebarGroupId` contains `"models"` with `tabGroups.models` i18n key in all locales but no group - dead.
13. Mixed model downloads: STT models use the generic `/api/models/download`; clone models use `/api/voice/clone/models/download`; Kokoro weights come with the pack install; the Mandarin voices are a pack extra. A single Models page must reconcile three progress sources (polling intervals 1.5 s, 2 s, 3 s).
14. `MCP transcribe` takes only path and language; the UI offers model, translate, words, speakers. `MCP speak` ignores clean-up and `[pause]`.
15. Japanese voices (`jf_*`, `jm_*`) are listed but permanently `available:false` on the MLX lane (needs a 0.8 GB dictionary, "not offered yet").
16. `VoiceContext.tsx` fallback `DEFAULT_SETTINGS` omits `allowAgentCloning` (typed optional).
17. Voice state is fetched once (no polling) and refreshed explicitly via `refreshVoiceRuntime` / `onRefreshVoiceRuntime`; a new Models page must call it after installs.
