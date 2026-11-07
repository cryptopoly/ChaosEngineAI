---
title: "Progress: V-voice (voice cloning)"
status: review
---
# Progress: V-voice (voice cloning, D-28 / D-45)

Updated: 2026-11-07 10:10. Owner: agent. Branch: `feat/voice-cloning` (off `feat/home-studio`).

## Goal

Voice cloning with Qwen3-TTS Base through mlx-audio in the voice worker, a spoken consent step checked by our STT,
the consent record stored and deleted with the voice, a forced AI tag on every cloned output, a local audit log, and
cloning off for agents and other machines by default, and an AudioSeal watermark on every cloned output (D-47).

## Done

- Catalog `backend_service/catalog/voice_models.py`: Qwen3-TTS 12Hz 0.6B and 1.7B Base, `mlx-community/*-Base-bf16`,
  revisions pinned (1eccf1c..., a6eb4f6...), checked on the Hub 2026-11-07, Apache 2.0, not gated.
- Worker: `synthesize()` branch with `refAudioPath` + `refText` (`synthesize_cloned`, no `voice` passed);
  `encode` writes the AI tag into the MP3 ID3 comment.
- `voice_clone_consent.py` (code challenge, digit and number-word matching), `voice_clone_safety.py`
  (`clone_allowed(request)`, WAV `LIST/INFO` tag, `apply_watermark(path)` no-op hook, JSONL audit log),
  `voice_clone.py` (pinned download, create, list, delete, speak), library kind `clone` with consent file.
- Routes `/api/voice/clone/*` in `routes/voice_clone.py`, included from `routes/voice.py`.
  Setting `allowAgentCloning` (voice settings, default off, only the UI can change it).
- Agent tool bodies `mcp_bridge/voice_clone_tool.py` (`list_cloned_voices`, `speak_cloned`), not registered.
- UI: collapsed "Clone a voice" section under Voice Studio > Speak (`src/features/voice/clone/`), 33 `voice.clone.*`
  keys (en + null stubs).
- Tests: `tests/test_voice_clone.py` (22), `src/features/voice/clone/clone.test.tsx` (6); e2e phase 14
  (`scripts/e2e_phases/phase_14.py`) passed live on port 8891.
- Real run: Kokoro read the consent sentence, Parakeet heard the code, the wrong code was refused, 0.6B spoke a
  sentence (3.4 s audio in about 4 s), WAV and MP3 carried the tag, STT read the cloned clip back word for word.

- D-47 watermark (owner yes, relayed by the coordinator): AudioSeal 0.2.1 (Meta; code MIT, weights MIT per the
  README "full MIT license (including the license for the model weights)" and Hub card `license: mit`, not gated),
  weights `facebook/audioseal` pinned at 3c19eba (generator_base 59 MB, detector_base 35 MB). Chosen over PerTh
  1.0.1 (MIT wheel with bundled weights, but pulls pydub, pyrubberband, matplotlib, torchaudio and marks presence
  only). `voice_worker_watermark.py` embeds a fixed 16-bit message 0xCE5A at 16 kHz and adds the mark to the
  original-rate audio; `detect_watermark` and `GET /api/voice/clone/watermark/{id}`. Installed on demand with the
  cloning model download; cloned speech is refused without it. Tag records `watermark: audioseal-0.2.1:CE5A`.

## Not done

- Watermark real run: NOT run. Installing `audioseal`, `omegaconf`, `einops` into the voice venv was denied by the
  permission classifier, and neither venv has omegaconf. `tests/test_voice_watermark.py::RealRoundTripTests`
  (embed, MP3 and AAC re-encode, detect, SNR > 20 dB) skips until the component is installed; then run it and
  e2e phase 14. Speed on Apple silicon (MPS / CPU) not measured.

- CUDA / Windows / Linux cloning (`qwen-tts` / transformers): follow-up.
- Optional speaker match between consent recording and a separately uploaded sample (biometric; needs a decision).
- Mic capture records at 16 kHz; Qwen3-TTS works at 24 kHz. A higher-rate capture for samples may improve likeness.
- The disclaimer line is English only.
- Nobody has listened to the cloned output.

## How to resume

1. `git switch feat/voice-cloning`.
2. `.venv/bin/python -m pytest tests/test_voice_clone.py tests/test_voice*.py -q` and `npx vitest run src/features/voice`.
3. Next step: A-bridge registers `CLONE_TOOLS` / `CLONE_HANDLERS`; storyboard reads the WAV tag on import.

## Decisions to propose

- Speaker match (on-device, CAM++ embeddings already installed) between consent and uploaded sample: yes or no.
- Re-show the first-run terms (bump `TERMS_VERSION`) once the terms gain the cloning paragraph from the research doc.
