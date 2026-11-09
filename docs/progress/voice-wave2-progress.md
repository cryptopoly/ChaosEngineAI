# Voice models wave 2: progress

Branch `feat/voice-models-wave2` (from `release/v0.10.0`). Plan: [../plans/voice-models-wave2.md](../plans/voice-models-wave2.md) (status table at the top). Owner answers D-70 (2026-11-09). Spike results: [../plans/voice-models-wave2/research.md section 5](../plans/voice-models-wave2/research.md).

## Done

- P0: `scripts/measure-voice-models.py` (own worker, no port; `--spike` writes WAVs to `~/.chaosengine/voice/wave2-listening/`), `tests/test_measure_voice_models.py`.
- P1: catalog schema (per-row licence, family, companions, controls, sampleRate, minReferenceS, rank, builtinWatermark, downloadGb, listed), `DEFAULT_CLONE_MODEL`; shard and companion readiness; companion download; `voice_worker_models.py` (redirect + family calls); tag rules and paren guard (`voice_text.py`); `language`/`expression`/`guidance` on the route, MCP `speak`, golden list, AGENTS.md; mlx-audio floor 0.6.4 with feature `speech-wave2` (mlx lane only); catalog audit covers voice and music at pinned revisions with `token=False`.
- P2 to P5: four MLX rows measured and listed; Speak screen controls (`SpeakControls.tsx`, `speakLogic.ts`); clone form names Turbo's 5 s minimum.
- P6 (unverified): `voice_env.speech_gpu_*`, `voice_worker_torch.py`, `catalog/voice_models_cuda.py`, rtx4090-test.md section 2.1, FU-153.
- P7: usage page, THIRD_PARTY_NOTICES, i18n (en + null), FOLLOW_UPS (FU-149, FU-141, FU-153, FU-154), hub rebuilt.
- Tests: `tests/test_voice_models_wave2.py`, `tests/test_voice_cuda_lane.py`, vitest additions, Playwright `e2e/tests/18-voice-wave2.spec.ts`, `scripts/e2e_test_suite.py` phase 8 checks. `voice_speak.py` audio steps moved to `voice_speak_audio.py` (re-exported).

## Open

- FU-154: owner listening check (default cloning model, `rank`, Turbo style tags).
- FU-153: the 4090 run of the CUDA lane.
- `voice_worker.py` is 681 lines (over the 600 soft cap); next change there should move the speech-to-text half out.
