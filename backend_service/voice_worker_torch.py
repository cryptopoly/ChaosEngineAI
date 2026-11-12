"""Speech models on an NVIDIA GPU inside the voice worker (D-70 Phase 6, FU-149). NOT VERIFIED: written on a Mac.

Runs in the portable voice venv with the speech GPU add-on (``voice_env.speech_gpu_commands``) and imports nothing
from ``backend_service``. Two engines:

* ``torch``: Chatterbox-Turbo and Chatterbox Multilingual V3 through ``chatterbox-tts``, VoxCPM2 through ``voxcpm``,
  each loaded from the pinned local snapshot the backend downloaded (never by repo id: upstream Turbo's
  ``from_pretrained`` asks for a login). ``chatterbox-tts`` adds Resemble's PerTh watermark to every output itself.
* ``llama-gguf``: Orpheus 3B as a GGUF. The worker starts the bundled ``llama-server`` on a free local port, sends the
  Orpheus prompt as token ids, reads the audio-code tokens back and decodes them with the SNAC 24 kHz decoder in
  ONNX on onnxruntime. No PyTorch.

A person's voice sample arrives only as ``refAudioPath`` (set by ``voice_clone.speak`` alone, which watermarks and
tags the result); ``continueAudioPath`` is VoxCPM2's own first chunk.
"""

from __future__ import annotations

import atexit
import json
import socket
import subprocess
import time
import urllib.request
from typing import Any, Callable

_model: dict[str, Any] = {"key": None, "model": None}
_server: dict[str, Any] = {"gguf": None, "process": None, "port": None}

# Orpheus prompt framing and audio-code tokens (canopyai/Orpheus-TTS): start of human turn, end of text, end of
# human turn, start of AI turn, start of speech; codes are custom tokens from 128266, seven per SNAC frame.
ORPHEUS_START = [128259]
ORPHEUS_END = [128009, 128260, 128261, 128257]
ORPHEUS_STOP = 128258
AUDIO_TOKEN_BASE = 128266
CODEBOOK = 4096


def unload() -> None:
    _model["key"], _model["model"] = None, None
    try:
        _qwen_fast().clear_prompts()
    except ImportError:
        pass
    stop_server()
    try:
        import torch  # type: ignore[import-not-found]

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch absent or no GPU
        pass


def keep_float32_loudness(model: Any) -> None:
    """chatterbox-tts 0.1.7 ``norm_loudness`` multiplies the float32 sample by a gain built from pyloudnorm's
    ``numpy.float64``; under NumPy 2's promotion rules that turns the sample into float64, and the tokenizer's
    float32 mel filters then fail with "expected scalar type Float but found Double" on every cloned voice
    (RTX 4090 run, NumPy 2.5). Cast the result back to float32."""
    original = getattr(model, "norm_loudness", None)
    if original is None or getattr(original, "_keeps_float32", False):
        return

    def norm_loudness(wav: Any, sr: Any, *args: Any, **kwargs: Any) -> Any:
        out = original(wav, sr, *args, **kwargs)
        return out.astype("float32", copy=False) if hasattr(out, "astype") else out

    norm_loudness._keeps_float32 = True  # type: ignore[attr-defined]
    model.norm_loudness = norm_loudness


def load_qwen(source: str, device: str) -> Any:
    """Qwen3-TTS (``qwen-tts``) from the pinned local snapshot, straight onto the GPU in bf16 (fp16 on a card without
    bf16) with PyTorch's own attention (flash-attn is not installed). Runs in the Qwen worker, whose transformers is
    the 4.57 that ``qwen-tts`` needs (``voice_env.qwen_worker_env``)."""
    import torch  # type: ignore[import-not-found]

    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # an add-on installed before Qwen3-TTS joined it
        if exc.name not in ("qwen_tts", "sox", "accelerate"):
            raise
        raise RuntimeError(
            "The speech GPU add-on has no Qwen3-TTS package yet. Untick and tick \"More speech models on the NVIDIA GPU\" "
            "under Voice runtime > Add-ons to update it."
        ) from exc

    cuda = device.startswith("cuda")
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)
    where = device if ":" in device or not cuda else f"{device}:0"
    model = Qwen3TTSModel.from_pretrained(
        source, dtype=dtype, device_map=where, attn_implementation="sdpa", local_files_only=True,
    )
    speed_up_qwen(model)
    return model


_qwen_notes: list[str] = []


def _qwen_fast() -> Any:
    try:  # the worker runs as a script, so a sibling is a top-level module (as for voice_worker_models)
        import qwen_tts_fast  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        from backend_service import qwen_tts_fast  # noqa: PLC0415
    return qwen_tts_fast


def speed_up_qwen(model: Any) -> None:
    """Swap the code predictor's per-step ``generate()`` for the plain loop in ``qwen_tts_fast`` (same codes for a
    seed, about 11 % less time on an RTX 4090). Off with ``CHAOSENGINE_QWEN_FAST=0``; any failure keeps the library's
    own path and is written to the worker log."""
    import os
    import sys

    if os.environ.get("CHAOSENGINE_QWEN_FAST", "1") == "0":
        return
    try:
        if not _qwen_fast().fast_code_predictor(model, notes=_qwen_notes):
            print("Qwen3-TTS: unexpected model layout, the fast code predictor is off.", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - never block loading the model
        print(f"Qwen3-TTS: the fast code predictor is off: {exc}", file=sys.stderr)


def _load(family: str, source: str, device: str) -> Any:
    key = (family, source, device)
    if _model["key"] == key:
        return _model["model"]
    unload()
    if family == "chatterbox-turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # type: ignore[import-not-found]

        model = ChatterboxTurboTTS.from_local(source, device)
        keep_float32_loudness(model)
    elif family == "chatterbox":
        try:  # the worker runs as a script, so a sibling is a top-level module (as for voice_worker_models)
            import voice_ja  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError:
            from backend_service import voice_ja  # noqa: PLC0415

        voice_ja.install_pykakasi_stand_in()  # Japanese readings without the GPL pykakasi
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[import-not-found]

        model = ChatterboxMultilingualTTS.from_local(source, device, t3_model="v3")
    elif family == "qwen3-tts":
        model = load_qwen(source, device)
    elif family == "voxcpm2":
        from voxcpm import VoxCPM  # type: ignore[import-not-found]

        model = VoxCPM(voxcpm_model_path=source, zipenhancer_model_path=None, enable_denoiser=False, device=device)
    else:
        raise ValueError(f"No GPU engine for {family!r}.")
    _model["key"], _model["model"] = key, model
    return model


def _seed(seed: Any) -> None:
    if seed is None:
        return
    import torch  # type: ignore[import-not-found]

    torch.manual_seed(int(seed))


def _numpy(audio: Any) -> Any:
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


_DIRECT_AUDIO = (".wav", ".flac")
_ref_wavs: dict[tuple[str, float], str] = {}


def _decode_to_wav(path: str, out: str, rate: int = 24_000) -> None:
    """Decode any sample PyAV reads (m4a/AAC, mp3, ogg) to 16-bit mono WAV at ``rate``."""
    import wave

    import av  # type: ignore[import-not-found]
    import numpy as np

    chunks: list[Any] = []
    with av.open(path) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(audio=0):
            for piece in resampler.resample(frame):
                chunks.append(piece.to_ndarray().reshape(-1))
        for piece in resampler.resample(None):
            chunks.append(piece.to_ndarray().reshape(-1))
    pcm = np.concatenate(chunks).astype(np.int16) if chunks else np.zeros(0, dtype=np.int16)
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def readable_audio(path: str, decode: Callable[[str, str], None] = _decode_to_wav) -> str:
    """A path chatterbox-tts and voxcpm can read. They load samples with librosa/soundfile, which cannot decode
    m4a/AAC on Windows (no CoreAudio, no ffmpeg), so a cloned voice saved from a phone memo failed with
    audioread's ``NoBackendError``. Other formats are decoded once with PyAV, as the MLX path already does."""
    from pathlib import Path
    import tempfile

    source = Path(path)
    if source.suffix.lower() in _DIRECT_AUDIO:
        return path
    key = (str(source), source.stat().st_mtime)
    cached = _ref_wavs.get(key)
    if cached and Path(cached).is_file():
        return cached
    folder = Path(tempfile.gettempdir()) / "chaosengine-voice-ref"
    folder.mkdir(exist_ok=True)
    out = folder / f"{source.stem}-{abs(hash(key)) & 0xFFFFFFFF:08x}.wav"
    decode(str(source), str(out))
    _ref_wavs[key] = str(out)
    return str(out)


def chatterbox_call(model: Any, request: dict[str, Any]) -> tuple[Any, int]:
    """Chatterbox-Turbo and V3 through chatterbox-tts; the reference clip goes in as a file path."""
    kwargs: dict[str, Any] = {}
    if request.get("refAudioPath"):
        kwargs["audio_prompt_path"] = readable_audio(str(request["refAudioPath"]))
    if request.get("family") == "chatterbox":
        kwargs.update({
            "language_id": str(request.get("language") or "en"),
            "exaggeration": float(request.get("expression", 0.5)),
            "cfg_weight": float(request.get("guidance", 0.5)),
            "temperature": 0.8,
        })
    else:
        kwargs.update({"temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.2})
    audio = model.generate(str(request["text"]), **kwargs)
    return _numpy(audio), int(getattr(model, "sr", 24_000))


def voxcpm_call(model: Any, request: dict[str, Any]) -> tuple[Any, int]:
    """VoxCPM2 through voxcpm: design is "(description)text"; exact clone = reference + same clip as prompt +
    transcript; steered clone = reference + "(style)"; continuation = its own first chunk as the prompt."""
    text = str(request["text"])
    kwargs: dict[str, Any] = {"cfg_value": 2.0, "inference_timesteps": 10}
    if request.get("refAudioPath"):
        reference = readable_audio(str(request["refAudioPath"]))
        kwargs["reference_wav_path"] = reference
        if request.get("mode") == "steered" and request.get("instruct"):
            text = f"({request['instruct']}){text}"
        elif request.get("mode") != "steered":
            kwargs["prompt_wav_path"] = reference
            kwargs["prompt_text"] = str(request.get("refText") or "")
    elif request.get("continueAudioPath"):
        kwargs["prompt_wav_path"] = str(request["continueAudioPath"])
        kwargs["prompt_text"] = str(request.get("continueText") or "")
    elif request.get("instruct"):
        text = f"({request['instruct']}){text}"
    else:
        raise ValueError("VoxCPM2 needs a voice description or a cloned voice.")
    audio = model.generate(text=text, **kwargs)
    rate = int(getattr(getattr(model, "tts_model", None), "sample_rate", 48_000))
    return _numpy(audio), rate


def qwen_call(model: Any, request: dict[str, Any]) -> tuple[Any, int]:
    """Qwen3-TTS: a preset speaker with an optional style (CustomVoice), a described voice (VoiceDesign) or a voice
    sample (Base: with its transcript the whole clip is the prompt, without one only the speaker embedding is used)."""
    text = str(request["text"])
    language = str(request.get("language") or "auto")
    mode = str(request.get("mode") or "")
    if request.get("refAudioPath"):
        transcript = str(request.get("refText") or "").strip()
        reference = readable_audio(str(request["refAudioPath"]))
        try:  # the reference's speaker embedding and codes once per clip and transcript
            prompt = _qwen_fast().cached_clone_prompt(model, reference, transcript or None, not transcript)
        except Exception:  # noqa: BLE001 - no prompt API or an unreadable clip: let the model encode it itself
            prompt = None
        if prompt is not None:
            wavs, rate = model.generate_voice_clone(text=text, language=language, voice_clone_prompt=prompt)
        else:
            wavs, rate = model.generate_voice_clone(
                text=text, language=language, ref_audio=reference, ref_text=transcript or None,
                x_vector_only_mode=not transcript,
            )
    elif mode == "preset":
        wavs, rate = model.generate_custom_voice(
            text=text, speaker=str(request["speaker"]), language=language, instruct=request.get("instruct") or None,
        )
    elif mode == "design":
        wavs, rate = model.generate_voice_design(text=text, instruct=str(request["instruct"]), language=language)
    else:
        raise ValueError("Qwen3-TTS needs a preset voice, a voice description or a voice sample.")
    return _numpy(wavs[0]), int(rate)


# ---------------------------------------------------------------------------
# Orpheus through llama-server and SNAC ONNX
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_server() -> None:
    process = _server.get("process")
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    _server.update({"gguf": None, "process": None, "port": None})


atexit.register(stop_server)


def server_command(binary: str, gguf: str, port: int) -> list[str]:
    """llama-server for Orpheus: local only, every layer on the GPU, room for one chunk's codes."""
    return [binary, "--model", gguf, "--host", "127.0.0.1", "--port", str(port), "--n-gpu-layers", "999",
            "--ctx-size", "4096", "--parallel", "1"]


def _post(port: int, path: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - our own local server
        return json.loads(response.read().decode("utf-8"))


def _ensure_server(binary: str, gguf: str) -> int:
    if _server["gguf"] == gguf and _server["process"] is not None and _server["process"].poll() is None:
        return int(_server["port"])
    stop_server()
    if not binary:
        raise RuntimeError("llama-server was not found; reinstall ChaosEngineAI or set CHAOSENGINE_LLAMA_SERVER.")
    port = _free_port()
    process = subprocess.Popen(server_command(binary, gguf, port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server stopped while loading the Orpheus model.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as reply:  # noqa: S310
                if reply.status == 200:
                    _server.update({"gguf": gguf, "process": process, "port": port})
                    return port
        except Exception:  # noqa: BLE001 - still loading
            time.sleep(0.5)
    process.kill()
    raise RuntimeError("llama-server did not load the Orpheus model in 3 minutes.")


def orpheus_prompt(text_tokens: list[int]) -> list[int]:
    return [*ORPHEUS_START, *text_tokens, *ORPHEUS_END]


def codes_from_tokens(tokens: list[int]) -> list[int]:
    """Audio codes from generated token ids: custom tokens only, each minus its slot's codebook offset, whole frames."""
    audio = [t - AUDIO_TOKEN_BASE for t in tokens if t >= AUDIO_TOKEN_BASE]
    codes = [code - (index % 7) * CODEBOOK for index, code in enumerate(audio)]
    return codes[: len(codes) - len(codes) % 7]


def snac_layers(codes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Split the seven codes of each frame into SNAC's three layers (1, 2 and 4 codes per frame)."""
    layer_1: list[int] = []
    layer_2: list[int] = []
    layer_3: list[int] = []
    for i in range(len(codes) // 7):
        frame = codes[7 * i: 7 * i + 7]
        layer_1.append(frame[0])
        layer_2.extend([frame[1], frame[4]])
        layer_3.extend([frame[2], frame[3], frame[5], frame[6]])
    return layer_1, layer_2, layer_3


_snac: dict[str, Any] = {"path": None, "session": None}


def snac_decode(decoder_path: str, codes: list[int]) -> Any:
    import numpy as np
    import onnxruntime as ort  # type: ignore[import-untyped]

    if not codes:
        raise RuntimeError("Orpheus produced no audio codes.")
    if any(c < 0 or c >= CODEBOOK for c in codes):
        raise RuntimeError("Orpheus produced codes outside the SNAC codebook.")
    if _snac["path"] != decoder_path:
        _snac["session"] = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
        _snac["path"] = decoder_path
    session = _snac["session"]
    layers = [np.asarray([layer], dtype=np.int64) for layer in snac_layers(codes)]
    feeds = {item.name: layers[i] for i, item in enumerate(session.get_inputs())}
    return np.asarray(session.run(None, feeds)[0], dtype=np.float32).reshape(-1)


def orpheus_call(request: dict[str, Any]) -> tuple[Any, int]:
    if request.get("refAudioPath"):
        raise ValueError("Orpheus speaks its eight preset voices only; it does not clone.")
    port = _ensure_server(str(request.get("llamaServerPath") or ""), str(request["gguf"]))
    voice = str(request.get("speaker") or "tara")
    tokens = _post(port, "/tokenize", {"content": f"{voice}: {request['text']}", "add_special": True})["tokens"]
    reply = _post(port, "/completion", {
        "prompt": orpheus_prompt(list(tokens)), "n_predict": int(request.get("maxTokens") or 2400),
        "temperature": 0.6, "top_p": 0.8, "repeat_penalty": 1.3, "stop": [], "return_tokens": True,
        "seed": int(request["seed"]) if request.get("seed") is not None else -1,
    })
    generated = [int(t) for t in reply.get("tokens", [])]
    if ORPHEUS_STOP in generated:
        generated = generated[: generated.index(ORPHEUS_STOP)]
    return snac_decode(str(request["snacDecoderPath"]), codes_from_tokens(generated)), 24_000


def synthesize(request: dict[str, Any], _decode_audio: Callable[..., Any] | None = None) -> tuple[Any, int]:
    engine = str(request.get("engine") or "")
    if engine == "llama-gguf":
        return orpheus_call(request)
    if engine != "torch":
        raise ValueError(f"Unknown GPU speech engine {engine!r}.")
    family = str(request.get("family") or "")
    model = _load(family, str(request["source"]), str(request.get("device") or "cuda"))
    _seed(request.get("seed"))
    if family == "voxcpm2":
        return voxcpm_call(model, request)
    if family == "qwen3-tts":
        return qwen_call(model, request)
    return chatterbox_call(model, request)
