# Voice rework plan (D-58)

Inputs: `research-audit.md` (what exists today), `research-models.md` (what the models can do). Prototype: three screens agreed in chat on 2026-11-08.

## Goal

One **Speak** screen that picks a voice, not a model. One **Voices** list with built-in and cloned voices together. **Voice Models** lists every speech model with what it can do. Every control Voice Studio has today survives.

## Findings that shape the plan

1. Voice Models is already its own left-menu entry (`voice-models`), as a child of the Voice group like every other area. It is a custom page, not the shared `ModelsPage` shell the other areas use.
2. Qwen3-TTS ships in three families. We use only Base (cloning). CustomVoice (9 preset speakers plus a style instruction) and VoiceDesign (a voice from a description) have MLX builds.
3. Style instructions work on the 1.7B models only. On 0.6B the style text is ignored, so the box must be greyed out there.
4. The model has no speed, pitch or seed control. Speed and pitch become post-processing (ffmpeg `atempo`, `asetrate`). Long text is split into chunks by us.
5. Only Ryan and Aiden are native English presets. Kokoro keeps the quick English voices. VoiceDesign covers a natural English female voice.
6. Speak (Kokoro) and Clone (Qwen3) have separate paths and limits today (200k characters and speed, versus 5000 characters and no speed). They need one request shape.
7. Not in the code at all: pitch, style prompt, voice design, "get more voices", multi-speaker speech, pronunciation dictionary, clean-up toggle.

## Must-keep checklist

Taken from the audit sections 1.1 to 1.4. QA ticks each line against the new screens.

- Transcribe: model, language, translate, label speakers, speaker count, record, drag and drop, multi-file queue, cancel, remove, mic settings button, transcript editor, hand-offs to Chat, exports.
- Speak: 200k characters, blank-line paragraphs, `[pause]`, voice preview, speed, part-by-part generation with cancel, WAV, MP3, Save to Library, Send to Storyboard.
- Clone: standing statement, model select and download, voice list with consent line and delete, new-voice form (basis, speaker name, name, file or microphone sample, sample text, permission tick), AI label badge, agent-cloning switch.
- Voice Models: voice pack install, update, repair and cancel, speech-to-text models, voices and Mandarin add-on, all voice settings (auto-send, auto-read, dictation language, reading speed, call pause, interrupt, idle unload, unload now).
- Chat voice: mic button, read-aloud, voice call. These read `settings.ttsVoice` and `speed`, so they keep working.
- Library, MCP tools, CLI commands and the Storyboard voice lane keep their current fields.

## Work packages

**A. Backend: models and modes** (catalog, engine, routes)
- Catalog rows for Qwen3-TTS CustomVoice and VoiceDesign at 0.6B and 1.7B, with repo ids, pinned revisions, sizes and `capabilities` flags (speak, style, clone, design).
- One engine call per mode in the voice worker (preset speaker plus style, voice description, reference clone).
- One `POST /api/voice/speak` request: `text`, `voice`, `style`, `speed`, `pitch`, `seed`; the backend picks the model from the voice. Old routes stay as wrappers.
- Post-processing: speed and pitch by ffmpeg, then the existing AI label and watermark for cloned voices. Chunking for long text on every model.
- Clean-up: apply the text tidy-up the placeholder already promises.

**B. Frontend: screens**
- Speak screen: voice picker cards, style box with chips, speed and pitch sliders, model note, generate. Keeps every Speak and Clone item in the checklist.
- Voices screen: one list from `/api/voice/runtime` and `/api/voice/clone/voices`; New voice: clone, design, get more.
- Transcribe stays as a Voice Studio mode, unchanged.
- Voice Models on the shared `ModelsPage` shell: Qwen3-TTS rows with capability chips, Kokoro, STT models, voice pack, settings.
  - Decided (2026-11-08): the owner accepted keeping the custom Voice Models page; the shared `ModelsPage` shell move is not done.

**C. Navigation and copy**
- Voice group order and default child match the other areas. Fix the stale "Voice > My Models" copy.
- i18n keys in all 10 locales (null placeholders where untranslated); `npm run i18n:validate`.

**D. Tests**
- pytest: catalog, capability flags, request shape, post-processing, old-route wrappers, golden MCP list.
- vitest: new screens, picker logic, style box greyed out on 0.6B.
- e2e: update `07-voice-clone`, `goTo` labels, Sidebar and redirect tests; new Speak, Voices and Models specs including 375 px.
- Real run on the M4 Max: speak, style, design and clone, with timings and memory recorded in the catalog.

## Order and who does what

1. A (one agent, own worktree, branch `feat/voice-rework`).
2. B and C (a second agent after A lands the request shape; same branch, so one writer at a time).
3. QA agent: ticks the checklist, reads the diff, tries the screens in the browser at desktop and 375 px.
4. Test agent: full pytest, tsc, vitest, i18n, e2e, then the real Mac run.
5. I verify key results myself before reporting.

## Open questions for the owner

1. Models as a left-menu entry: keep it as a child of the Voice group (what Chat, Images, Video and Music do), or make a new top-level row? Recommendation: keep as a group child, same as every other area.
2. Which models in the first pass: Qwen3-TTS CustomVoice and VoiceDesign now; Chatterbox and VoxCPM2 later? Recommendation: yes, Qwen3-TTS first.
3. Unmeasured: speed and memory of the 1.7B models on the M4 Max. The first real run decides whether Qwen3-TTS can be the default for everyday speech or stays the "natural" option next to fast Kokoro.
