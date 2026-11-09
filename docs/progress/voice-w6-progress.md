# Voice W6: any voice in Chat and calls (D-67 W6, D-72): progress

Branch `feat/voice-w6` (from `release/v0.10.0` at `3f5066e4`). Plan row: [../plans/voice-ux-round2.md](../plans/voice-ux-round2.md) W6 and question 3. Owner decision D-72 (2026-11-09): Kokoro stops being the de facto default; it stays in the pack as the small fallback.

## Done

- Backend: `backend_service/voice_chat.py` plans each read-aloud or call request through `voice_speak` (per-model chunking and sound-tag rules) and speaks per sentence with `voice_speak.render_sentence`. `use` is `read` or `call` on `POST /api/voice/speak/stream` (`StreamRequest`, voice up to 64 characters). Fallback to the Kokoro voice (`af_heart`) with a `{"note"}` line first, both when planning fails and when a sentence fails mid-stream (the failed voice is skipped for 120 s). Kokoro failing is still an error.
- Cloned voices: read-aloud only, through `voice_clone.clone_plan` (the same checks as Speak, extracted from `voice_clone.speak`), each sentence marked with `voice_clone.mark_cloned` (watermark and AI tag), one audit line per request (`use: read-aloud`), not saved to the Library. Calls refuse them (fallback note); agents get 403; an agent's `speak` tool never defaults to a cloned chat voice.
- Settings: `ttsVoice` (64 chars) and new `callVoice` ("" = chat voice) validated against `list_voices` and `chatUse` in `routes/voice.py`. `list_voices` rows gain `chatUse` and `chatCost` (`voice_speak.chat_uses`, `chat_cost`); catalog rows gain `chatFirstAudioS`, `chatFirstAudioColdS`, `chatPeakGb`, `chatMeasuredOn`; Kokoro's numbers are `KOKORO_CHAT_ESTIMATES`; `SLOW_FOR_CALLS_S = 2.0`.
- Frontend: Voices tab Use in Chat for allowed voices, Use for calls, cost line and Slow for calls badge; Chat and calls panel has chat and call voice selects with times; speech bar shows the note; call bar names the call voice with Change in Voices; Call info names the voice. `speakStream(..., { use, onNote })`, `speechPlayer` carries `note`, `callVoiceOf` in `useVoiceCall.ts`.
- Measurement: `scripts/measure-voice-models.py --chat` (two runs on the M4 Max 2026-11-09, slower kept): Kokoro 0.13 s warm / 5.7 s cold / 0.87 GB; Chatterbox-Turbo 0.6 / 3.8 / 1.65; Qwen3-TTS 1.7B CustomVoice 1.3 / 4.1 / 3.51; VoxCPM2 1.7 / 3.8 / 3.63. Raw runs: run 1 Qwen warm 0.68 s, run 2 1.27 s.
- Memory: unchanged design already meets the rule (one voice worker, one speech model, `_load_tts` frees the old one; idle unload). Documented in `voice_chat.py` and the usage page.
- Tests: pytest `ChatStreamTests` in `tests/test_voice_speak.py`, settings and fallback in `tests/test_voice_io.py`, `chat_estimates` in `tests/test_measure_voice_models.py`; vitest `speakLogic.test.ts`, `VoicesScreen.test.tsx`, `src/api/voice.test.ts`; Playwright `e2e/tests/19-chat-voice.spec.ts` (+ spec 14 updated, mocks carry `chatUse`/`chatCost`/`callVoice`); `scripts/e2e_test_suite.py` check "chat voice: any voice, Kokoro fallback".
- Docs: `docs/usage/voice.md` (Chat and calls section, table), plan W6 marked done, FU-156.

## Open

- FU-156: owner's real call (listening, barge-in feel, memory with a chat model loaded), Chatterbox V3 and Orpheus chat numbers, saved designed voices (W8), English-only fallback note, Windows and Linux.
