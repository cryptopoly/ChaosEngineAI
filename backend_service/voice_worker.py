"""Voice worker: speech-to-text and text-to-speech in their own process.

Runs from the voice venv (``~/.chaosengine/voice-venv``, built by
``backend_service.voice_env``), never inside the FastAPI backend, so the
speech models can stay loaded between requests without pulling mlx-audio,
parakeet-mlx or faster-whisper into the backend, and a crash or a stuck model
can be killed on its own.

This file imports nothing from ``backend_service``: the venv has only the
voice packages. Protocol: one JSON object per line on stdin, one JSON reply
per line on stdout, matched by ``id``. Library output goes to stderr so it
cannot corrupt the protocol.

Audio is decoded with PyAV (its own FFmpeg libraries), so no ``ffmpeg``
command is needed on PATH.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr  # anything a library prints goes to stderr

STT_SAMPLE_RATE = 16_000

# Kokoro voice prefix -> language. mlx-audio takes the letter itself;
# kokoro-onnx takes an espeak language tag.
KOKORO_ONNX_LANG = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
    "i": "it", "p": "pt-br", "j": "ja", "z": "cmn",
}

_stt: dict[str, Any] = {"key": None, "model": None}
_tts: dict[str, Any] = {"key": None, "model": None}

# Silero VAD: 512-sample frames at 16 kHz (32 ms), each run with the last 64
# samples of the previous frame in front and the model's recurrent state.
VAD_FRAME = 512
VAD_CONTEXT = 64
_VAD_MAX_SESSIONS = 8
_vad: dict[str, Any] = {"path": None, "session": None, "streams": {}}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def decode_audio(path: str, sample_rate: int = STT_SAMPLE_RATE):
    """Decode any audio or video file to mono float32 at ``sample_rate``."""
    import av  # type: ignore[import-untyped]
    import numpy as np

    chunks = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("The file has no audio track.")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("The file contains no audio.")
    return np.concatenate(chunks).astype(np.float32)


def wav_bytes(samples, sample_rate: int) -> bytes:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Platform fixes
# ---------------------------------------------------------------------------

def _short_espeak_data_path() -> None:
    """espeak-ng drops a data path of about 160+ characters and then exits the
    process; point it at a short copy under ``~/.chaosengine``."""
    try:
        import espeakng_loader  # type: ignore[import-untyped]
    except ImportError:
        return
    source = Path(espeakng_loader.get_data_path())
    if len(str(source)) < 140 or not source.is_dir():
        return
    short = Path.home() / ".chaosengine" / "espeak-ng-data"
    if not (short / "phontab").is_file():
        shutil.copytree(source, short, dirs_exist_ok=True)
    espeakng_loader.get_data_path = lambda: str(short)


def _add_cuda_dll_dirs() -> None:
    """Windows: let ctranslate2 find the CUDA libraries pip installed."""
    if os.name != "nt":
        return
    for base in sys.path:
        nvidia = Path(base) / "nvidia"
        if nvidia.is_dir():
            for bin_dir in nvidia.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------

def _load_stt(backend: str, source: str) -> Any:
    key = (backend, source)
    if _stt["key"] == key:
        return _stt["model"]
    _stt["key"], _stt["model"] = None, None
    if backend == "parakeet-mlx":
        import mlx.core as mx  # type: ignore[import-untyped]
        import parakeet_mlx.parakeet as parakeet_module  # type: ignore[import-untyped]
        from parakeet_mlx import from_pretrained  # type: ignore[import-untyped]

        # parakeet-mlx decodes with the ffmpeg command; use PyAV instead.
        parakeet_module.load_audio = lambda filename, sampling_rate, dtype=None: mx.array(
            decode_audio(str(filename), sampling_rate)
        )
        model = from_pretrained(source)
    elif backend == "nemotron-asr-mlx":
        from mlx_audio.stt import load as load_stt  # type: ignore[import-untyped]

        model = load_stt(source)
    elif backend == "mlx-whisper":
        model = source  # mlx-whisper keeps its own loaded-model cache
    elif backend == "faster-whisper":
        _add_cuda_dll_dirs()
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        model = None
        try:
            import ctranslate2  # type: ignore[import-untyped]

            if ctranslate2.get_cuda_device_count() > 0:
                model = WhisperModel(source, device="cuda", compute_type="float16")
        except Exception as exc:  # noqa: BLE001 - missing CUDA libraries raise many types
            print(f"voice_worker: CUDA unavailable for faster-whisper ({exc}); using CPU", file=sys.stderr)
            model = None
        if model is None:
            model = WhisperModel(source, device="cpu", compute_type="int8")
    else:
        raise ValueError(f"Unknown speech-to-text backend {backend!r}.")
    _stt["key"], _stt["model"] = key, model
    return model


def _word(start: float, end: float, text: str) -> dict[str, Any]:
    # float(): faster-whisper and mlx-whisper give numpy floats, which json cannot write.
    return {"start": round(float(start), 2), "end": round(float(end), 2), "text": text.strip()}


def _parakeet_words(tokens: list[Any]) -> list[dict[str, Any]]:
    """Join subword tokens into words: a token that starts with a space starts a word."""
    words: list[dict[str, Any]] = []
    for token in tokens:
        if not words or token.text.startswith(" "):
            words.append(_word(token.start, token.end, token.text))
        else:
            words[-1]["text"] += token.text
            words[-1]["end"] = round(float(token.end), 2)
    return [w for w in words if w["text"]]


def _nemotron_language(model: Any, language: str | None) -> str:
    """Nemotron's prompt key for an ISO code (``de`` -> ``de-DE``); ``auto`` when unknown."""
    if not language:
        return "auto"
    prompts = getattr(model, "prompt_dictionary", None) or {}
    if language in prompts:
        return language
    base = language.split("-")[0].lower()
    return next((key for key in prompts if key.split("-")[0].lower() == base), "auto")


def transcribe(request: dict[str, Any]) -> dict[str, Any]:
    """Text, segments and (when ``words``) word timings.

    ``task`` "translate" turns the speech into English (Whisper only).
    """
    backend = str(request["backend"])
    source = str(request["source"])
    path = str(request["path"])
    language = request.get("language") or None
    task = "translate" if request.get("task") == "translate" else "transcribe"
    want_words = bool(request.get("words"))
    model = _load_stt(backend, source)

    if backend == "parakeet-mlx":
        if task == "translate":
            raise ValueError("Translation to English needs a Whisper model.")
        duration = len(decode_audio(path)) / STT_SAMPLE_RATE
        chunk = 120.0 if duration > 150 else None
        result = model.transcribe(path, chunk_duration=chunk, overlap_duration=15.0)
        segments = []
        for s in getattr(result, "sentences", []) or []:
            segment = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            if want_words:
                segment["words"] = _parakeet_words(s.tokens)
            segments.append(segment)
        return {"text": result.text.strip(), "language": None, "segments": segments, "durationS": round(duration, 2)}

    audio = decode_audio(path)
    duration = len(audio) / STT_SAMPLE_RATE
    if backend == "nemotron-asr-mlx":
        if task == "translate":
            raise ValueError("Translation to English needs a Whisper model.")
        import mlx.core as mx  # type: ignore[import-untyped]

        # Chunked streaming decode: bounded memory on hour-long recordings.
        result = model.generate(mx.array(audio), language=_nemotron_language(model, language))
        segments = []
        for s in getattr(result, "sentences", []) or []:
            segment = {"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": s.text.strip()}
            if want_words:
                segment["words"] = _parakeet_words(s.tokens)
            segments.append(segment)
        # The sentences come back joined with two spaces.
        return {"text": " ".join(result.text.split()), "language": None, "segments": segments, "durationS": round(duration, 2)}
    if backend == "mlx-whisper":
        import mlx_whisper  # type: ignore[import-untyped]

        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=model, language=language, task=task, word_timestamps=want_words, verbose=None
        )
        segments = []
        for s in result.get("segments", []):
            segment = {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            if want_words:
                segment["words"] = [_word(w["start"], w["end"], w["word"]) for w in s.get("words", []) if w["word"].strip()]
            segments.append(segment)
        return {
            "text": (result.get("text") or "").strip(),
            "language": result.get("language"),
            "segments": segments,
            "durationS": round(duration, 2),
        }

    # Beam search makes the small models copy the speech instead of
    # translating it (Whisper Base on Spanish); greedy decoding translates.
    beam_size = 1 if task == "translate" else 5
    raw_segments, info = model.transcribe(
        audio, language=language, task=task, word_timestamps=want_words, beam_size=beam_size
    )
    segments = []
    for s in raw_segments:
        segment = {"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": s.text.strip()}
        if want_words:
            segment["words"] = [_word(w.start, w.end, w.word) for w in (s.words or []) if w.word.strip()]
        segments.append(segment)
    # Segment texts carry their own leading space (none for CJK).
    text = " ".join(s["text"] for s in segments) if info.language not in ("zh", "ja", "ko") else "".join(
        s["text"] for s in segments
    )
    return {"text": text.strip(), "language": info.language, "segments": segments, "durationS": round(duration, 2)}


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------

def _load_tts(engine: str, source: str) -> Any:
    key = (engine, source)
    if _tts["key"] == key:
        return _tts["model"]
    _tts["key"], _tts["model"] = None, None
    _short_espeak_data_path()
    if engine == "mlx-audio":
        from mlx_audio.tts.utils import load_model  # type: ignore[import-untyped]

        model = load_model(source)
    elif engine == "kokoro-onnx":
        from kokoro_onnx import Kokoro  # type: ignore[import-untyped]

        folder = Path(source)
        model = Kokoro(str(folder / "kokoro-v1.0.onnx"), str(folder / "voices-v1.0.bin"))
    else:
        raise ValueError(f"Unknown text-to-speech engine {engine!r}.")
    _tts["key"], _tts["model"] = key, model
    return model


def synthesize(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    engine = str(request["engine"])
    model = _load_tts(engine, str(request["source"]))
    text = str(request["text"])
    voice = str(request.get("voice") or "af_heart")
    speed = float(request.get("speed") or 1.0)
    prefix = voice[:1] if voice[:1] in KOKORO_ONNX_LANG else "a"

    if engine == "mlx-audio":
        parts = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in model.generate(text, voice=voice, speed=speed, lang_code=prefix)
        ]
        if not parts:
            raise RuntimeError("The speech model produced no audio.")
        samples = np.concatenate(parts)
        sample_rate = int(getattr(model, "sample_rate", 24_000))
    else:
        samples, sample_rate = model.create(text, voice=voice, speed=speed, lang=KOKORO_ONNX_LANG[prefix])
    return {
        "wavB64": base64.b64encode(wav_bytes(samples, sample_rate)).decode("ascii"),
        "sampleRate": sample_rate,
        "durationS": round(len(samples) / sample_rate, 2),
    }


# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------

def vad(request: dict[str, Any]) -> dict[str, Any]:
    """Speech probability for each 32 ms frame of 16 kHz PCM16 audio.

    ``session`` names a stream; its model state carries over between calls
    so a microphone can be fed a few frames at a time. ``reset`` starts the
    stream over. Samples past the last whole frame are dropped.
    """
    import numpy as np
    import onnxruntime as ort  # type: ignore[import-untyped]

    path = str(request["path"])
    if _vad["path"] != path:
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        _vad["session"] = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        _vad["path"], _vad["streams"] = path, {}
    streams: dict[str, Any] = _vad["streams"]
    name = str(request.get("session") or "default")
    if request.get("reset") or name not in streams:
        streams.pop(name, None)
        while len(streams) >= _VAD_MAX_SESSIONS:
            streams.pop(next(iter(streams)))
        streams[name] = {
            "state": np.zeros((2, 1, 128), dtype=np.float32),
            "context": np.zeros((1, VAD_CONTEXT), dtype=np.float32),
        }
    stream = streams[name]

    pcm = np.frombuffer(base64.b64decode(request.get("pcm") or ""), dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0
    sample_rate = np.array(STT_SAMPLE_RATE, dtype=np.int64)
    probs = []
    for start in range(0, len(audio) - VAD_FRAME + 1, VAD_FRAME):
        frame = np.concatenate([stream["context"], audio[start:start + VAD_FRAME][None, :]], axis=1)
        out, stream["state"] = _vad["session"].run(
            None, {"input": frame, "state": stream["state"], "sr": sample_rate}
        )
        stream["context"] = frame[:, -VAD_CONTEXT:]
        probs.append(round(float(out[0][0]), 3))
    return {"probs": probs}


# ---------------------------------------------------------------------------
# Speaker labels and audio export
# ---------------------------------------------------------------------------

_diarizer: dict[str, Any] = {"key": None, "model": None}


def diarize(request: dict[str, Any]) -> dict[str, Any]:
    """Who speaks when: ``turns`` of ``{start, end, speaker}``.

    Speakers are numbered 0, 1, ... in order of first appearance.
    ``speakers`` > 0 fixes the count; otherwise it is worked out from
    ``threshold`` (lower finds more speakers).
    """
    import sherpa_onnx  # type: ignore[import-untyped]

    count = int(request.get("speakers") or 0)
    threshold = float(request.get("threshold") or 0.5)
    key = (str(request["segmentation"]), str(request["embedding"]), count, threshold)
    if _diarizer["key"] != key:
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=key[0]),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=key[1]),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=count if count > 0 else -1, threshold=threshold),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError("The speaker-label models could not be loaded.")
        _diarizer["key"], _diarizer["model"] = key, sherpa_onnx.OfflineSpeakerDiarization(config)
    model = _diarizer["model"]
    audio = decode_audio(str(request["path"]), model.sample_rate)
    numbers: dict[int, int] = {}
    turns = []
    for segment in model.process(audio).sort_by_start_time():
        speaker = numbers.setdefault(segment.speaker, len(numbers))
        turns.append({"start": round(segment.start, 2), "end": round(segment.end, 2), "speaker": speaker})
    return {"turns": turns}


def encode(request: dict[str, Any]) -> dict[str, Any]:
    """Re-encode an audio file (``mp3`` only for now) next to ``out``."""
    import av  # type: ignore[import-untyped]

    if request.get("format") != "mp3":
        raise ValueError("Only MP3 export is supported.")
    with av.open(str(request["path"])) as source:
        if not source.streams.audio:
            raise ValueError("The file has no audio track.")
        rate = source.streams.audio[0].rate or 24_000
        with av.open(str(request["out"]), "w", format="mp3") as target:
            stream = target.add_stream("libmp3lame", rate=rate, layout="mono")
            stream.bit_rate = 128_000
            resampler = av.AudioResampler(format=stream.format.name, layout="mono", rate=rate)
            for frame in source.decode(audio=0):
                for out in resampler.resample(frame):
                    for packet in stream.encode(out):
                        target.mux(packet)
            for out in resampler.resample(None):
                for packet in stream.encode(out):
                    target.mux(packet)
            for packet in stream.encode(None):
                target.mux(packet)
    return {"out": str(request["out"])}


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

def unload(request: dict[str, Any]) -> dict[str, Any]:
    what = request.get("what") or "all"
    if what in ("stt", "all"):
        _stt["key"], _stt["model"] = None, None
    if what in ("tts", "all"):
        _tts["key"], _tts["model"] = None, None
    if what in ("speakers", "all"):
        _diarizer["key"], _diarizer["model"] = None, None
    try:
        import gc

        gc.collect()
        import mlx.core as mx  # type: ignore[import-untyped]

        mx.clear_cache()
    except Exception:  # noqa: BLE001 - mlx absent on Windows / Linux
        pass
    return {}


def status(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "stt": list(_stt["key"]) if _stt["key"] else None,
        "tts": list(_tts["key"]) if _tts["key"] else None,
        "pid": os.getpid(),
    }


_COMMANDS = {
    "ping": lambda _r: {"pong": True},
    "status": status,
    "transcribe": transcribe,
    "synthesize": synthesize,
    "vad": vad,
    "diarize": diarize,
    "encode": encode,
    "unload": unload,
}


def _reply(payload: dict[str, Any]) -> None:
    PROTOCOL_OUT.write(json.dumps(payload) + "\n")
    PROTOCOL_OUT.flush()


def main() -> None:
    _reply({"id": None, "ok": True, "ready": True, "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            command = _COMMANDS.get(request.get("cmd"))
            if command is None:
                raise ValueError(f"Unknown command {request.get('cmd')!r}.")
            started = time.perf_counter()
            result = command(request)
            result.update({"id": request_id, "ok": True, "seconds": round(time.perf_counter() - started, 3)})
            _reply(result)
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the caller
            traceback.print_exc(file=sys.stderr)
            _reply({"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
