# Voice makeover (D-73): progress

Branch `feat/voice-makeover`, cut from `release/v0.10.0` at `221f37d9` (W6 included). Spec: `docs/plans/voice-makeover/plan.md`, `review.md`, `mockups.html`. Owner answers (2026-11-09): the review's defaults on all six (M3b shared table, Chat popover, add-on ticks, Remove runtime, "Voice runtime", modes Speak, Voices, Transcribe).

## Done (all work packages: M1, M2, M3b, M4, M5, S1, S2, S3, S4, L1, D1)

- **Backend:** `backend_service/voice_model_files.py` (delete a speech model and the companions only it uses, keep the watermark while another cloning model is downloaded, refuse while a voice job runs, the worker answers or a speech model downloads, stop the worker when it holds the model, path checks on the HF cache); `backend_service/voice_runtime_disk.py` (runtime parts and sizes, what is in memory, untick an add-on, remove the runtime); `backend_service/helpers/voice_fit.py` (Fit from the measured peak); routes in `backend_service/routes/voice_files.py` (window only): `GET /api/voice/clone/models/delete-plan`, `POST /api/voice/clone/models/delete`, `POST /api/voice/stt/delete`, `POST /api/voice/extras/remove`, `POST /api/voice/uninstall`; `/api/voice/runtime` gains `parts` and `inMemory`; `/api/voice/models` gains `fit` per row and `kokoro`; `/api/voice/library?voiceKind=`. `voice_client` tracks what the worker loaded and whether it is busy. Backend messages say "voice runtime". Tests: `tests/test_voice_model_files.py`; E2E suite check "voice delete routes (no delete)" in phase 8.
- **Frontend:** Models page = `VoiceRuntimePanel` (VoicePackPanel.tsx), `models/DictationModelsPanel.tsx`, `models/SpeechModelsPanel.tsx` on `ModelsPage` with `components/models/adapters/voice.ts`; Speak = `VoiceRow` + `components/Drawer.tsx`; Voices = main-voices strip, grouped list, `UseForMenu`; Chat = `VoicePopover.tsx` (replaces the Call (i)); read-aloud bar names the voice; Library voice-kind select and "Send to project"; Studio modes Speak, Voices, Transcribe; workspace header drops its title on Voice Studio. Removed: `TtsModelsPanel.tsx`, `ChatCallsPanel.tsx`, the Quick voices and Dictation and memory panels, the agent-cloning switch on Voices (Settings keeps it), the cloning statement outside the New cloned voice form.
- i18n synced (en values, `null` in the other ten locales incl. `tlh`, 60 retired keys removed).
- Playwright: 07, 11, 14, 18-voice-wave2, 19-chat-voice updated; new `20-voice-makeover.spec.ts` (desktop and 375 px).
- Docs: `docs/usage/voice.md`, README, plan status, FU-157 and FU-158, hub rebuilt; after-screenshots in `docs/plans/voice-makeover/after/` (recipe `capture-shots.spec.ts`).

## Deferred / not checked

See FU-157 and FU-158: no Cancel for a speech-model download, no "Show N sizes" family collapse, Library and Models titles still doubled; real deletes and runtime removal on a real machine, Windows and Linux, listening.
