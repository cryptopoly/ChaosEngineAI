# Voice models wave 2: progress

Branch `feat/voice-models-wave2` (from `release/v0.10.0`). Plan: [../plans/voice-models-wave2.md](../plans/voice-models-wave2.md). Owner answers D-70 (2026-11-09).

## Done

- P0 spike and measurement on the M4 Max (2026-11-09): `scripts/measure-voice-models.py` (+ `tests/test_measure_voice_models.py`). Results below.
- P1 framework: per-row licence, family, companions (pinned, downloaded with the model), controls, sampleRate, minReferenceS, rank, builtinWatermark, downloadGb, `listed` (wave-2 rows only with measured numbers); shard-aware readiness; worker companion redirect and family dispatch (`backend_service/voice_worker_models.py`); sound-tag rules after `clean_for_speech`; VoxCPM2 leading-paren guard; speak fields `language`, `expression`, `guidance` (route, MCP `speak`, golden list, AGENTS.md); `CLONE_MODELS` from capabilities; `DEFAULT_CLONE_MODEL` constant.
- P2 Turbo, P3 VoxCPM2 (design, exact and steered clone, continuation from chunk one), P4 Chatterbox V3 (default voice from the V2 8-bit `conds.safetensors`, language, Expression, Guidance), P5 Orpheus (8 presets, angle tags, 250-char chunks, max_tokens 2400): catalog rows measured, backend wired, Speak screen controls.
- Tests: `tests/test_voice_models_wave2.py`, vitest additions in `speakLogic.test.ts` and `speakScreens.test.tsx`.

## Measured (Apple M4 Max, 64 GB, 2026-11-09)

| Row | s per 10 s | peak GB | idle GB | cold start s | short line s |
|---|---|---|---|---|---|
| chatterbox-turbo | 2.1 | 1.72 | 1.72 | 2.4 | 0.8 |
| voxcpm2 (design) | 5.22 | 3.8 | 3.8 | 2.0 | 1.9 |
| chatterbox-mtl (default voice) | 3.63 | 3.67 | 3.66 | 2.9 | 1.4 |
| orpheus-3b (tara) | 7.22 | 4.19 | 4.19 | 3.6 | 3.3 |

Spike answers: V2 conds works as V3's default voice; Orpheus 400 chars needs about 2400 tokens (1200 cuts at 14.6 s of 25 s); VoxCPM2 continuation keeps median pitch within about 7% across chunks (redesign jumps up to 40%); `mx.random.seed` repeats Turbo, VoxCPM2 and Orpheus exactly, not Chatterbox V3; Turbo's built-in voice is restored after a clone (byte-identical). Listening WAVs: `~/.chaosengine/voice/wave2-listening/`.

## Left (in order)

1. voice_env floor 0.6.4 and FEATURES "speech-wave2"; audit script covers voice and music catalogs.
2. NewVoiceForm minimum-length hint; i18n keys (en + null in 10 locales incl. tlh).
3. E2E: Playwright `14-voice-speak` (chips, Chatterbox fields), `07-voice-clone` (Turbo selectable); `scripts/e2e_test_suite.py` voice checks.
4. P6 CUDA lane (unverified): speech GPU add-on install lists, torch engines in the worker, Orpheus via llama-server + SNAC ONNX, CUDA catalog rows, PerTh flag, rtx4090-test.md section 2.
5. THIRD_PARTY_NOTICES, docs/usage/voice.md, plan phases marked, FOLLOW_UPS rows, hub rebuild.
6. Full test run; report.
