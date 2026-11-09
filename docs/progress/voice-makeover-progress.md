# Voice makeover (D-73): progress

Branch `feat/voice-makeover`, cut from `release/v0.10.0` at `221f37d9` (W6 included). Spec: `docs/plans/voice-makeover/plan.md`, `review.md`, `mockups.html`. Owner answers (2026-11-09): the review's defaults on all six (M3b shared table, Chat popover, add-on ticks, Remove runtime, "Voice runtime", modes Speak, Voices, Transcribe).

## Done

- **Backend (M1, M2, M4, M5, L1):** `backend_service/voice_model_files.py` (delete a speech model and the companions only it uses, keep the watermark while another cloning model is downloaded, refuse while a voice job runs, the worker answers or a speech model downloads, stop the worker when it holds the model, path checks on the HF cache); `backend_service/voice_runtime_disk.py` (runtime parts and sizes, what is in memory, untick an add-on, remove the runtime); `backend_service/helpers/voice_fit.py` (Fit from the measured peak); routes in `backend_service/routes/voice_files.py` (window only): `GET /api/voice/clone/models/delete-plan`, `POST /api/voice/clone/models/delete`, `POST /api/voice/stt/delete`, `POST /api/voice/extras/remove`, `POST /api/voice/uninstall`; `/api/voice/runtime` gains `parts` and `inMemory`; `/api/voice/models` gains `fit` per row and `kokoro`; `/api/voice/library?voiceKind=`. `voice_client` tracks what the worker loaded and whether it is busy. Tests: `tests/test_voice_model_files.py`.
- **Frontend:** Models page = `VoiceRuntimePanel` (VoicePackPanel.tsx), `models/DictationModelsPanel.tsx`, `models/SpeechModelsPanel.tsx` on `ModelsPage` with `components/models/adapters/voice.ts`; Speak = `VoiceRow` + `components/Drawer.tsx`; Voices = main-voices strip, grouped list, `UseForMenu`; Chat = `VoicePopover.tsx` (replaces the Call (i)); read-aloud bar names the voice; Library voice-kind select and "Send to project"; Studio modes Speak, Voices, Transcribe; the workspace header drops its title on Voice Studio. Removed: `TtsModelsPanel.tsx`, `ChatCallsPanel.tsx`, the Quick voices and Dictation and memory panels, the agent-cloning switch on Voices (Settings keeps it), the cloning statement outside the New cloned voice form.
- i18n synced (en values, `null` in the other ten locales incl. `tlh`, retired keys removed); `npm run i18n:validate` OK.

## Next

1. Backend user-facing "voice pack" strings -> "voice runtime".
2. Playwright: update 07, 11, 14, 18-voice-wave2, 19-chat-voice; new spec for delete + Change-voice drawer + main-voices strip + Chat Voice popover incl. 375 px (ports 8961/8962, `--workers=1`, one run at a time).
3. E2E suite check in `scripts/e2e_test_suite.py` for the delete route contract.
4. After-screenshots with `docs/plans/voice-makeover/capture-shots.spec.ts` into `docs/plans/voice-makeover/after/`; compare with the mock-ups.
5. Docs: `docs/usage/voice.md`, README/usage "voice pack" text, plan work packages marked done, FOLLOW_UPS rows (highest was FU-156), hub rebuild.
6. Full runs: pytest, npm test, tsc, i18n.

## Deferred / notes

- Cancel for a speech-model download (no backend cancel); the row shows "Downloading…" instead of Pause/Cancel.
- Family rows are headings only; the "Show N sizes" collapse in the mock-up is not built (the shared table already makes it one row per model).
