"""Plain, stable error codes for the Voice routes (voice studio UX audit, change 6).

Worker and setup failures reach the routes as raw text (``ValueError: ...``, ``OutOfMemoryError: CUDA out of
memory ...``, "Qwen3-TTS is not downloaded. Download it in Voice > Models."). ``classify`` maps that text to a
stable ``code``, a one-sentence plain ``message`` and an ``action`` hint the screen turns into one button. The raw
text stays as ``detail`` (and is what the screen shows under "Details").

A voice error response keeps its HTTP status and ``detail`` and gains three fields::

    {"detail": "<raw text>", "code": "gpu_memory", "message": "<one plain sentence>",
     "action": "smaller-model", "extra": null}

``extra`` names the voice install extra to tick (``kokoro``, ``chinese``, ``speech-gpu``, ``stt-gpu``) when the
code is ``addon_missing``, else null. The sentence and the button label are English here; the app maps ``code``
to its own translated text (``src/features/voice/voiceErrorText.ts``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# What the screen can offer as the one button. ``none`` means the sentence stands alone.
ACTIONS = (
    "install-voice-runtime",
    "install-addon",
    "download-model",
    "smaller-model",
    "pick-another-model",
    "install-ffmpeg",
    "pick-another-language",
    "change-sample",
    "confirm-permission",
    "add-hf-token",
    "retry",
    "none",
)


@dataclass(frozen=True)
class VoiceErrorInfo:
    code: str
    message: str
    action: str
    extra: str | None = None

    def fields(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "action": self.action, "extra": self.extra}


# code -> (plain sentence, action). The sentence says what happened, never the raw text.
_TABLE: dict[str, tuple[str, str]] = {
    "runtime_missing": ("The voice runtime is not installed yet.", "install-voice-runtime"),
    "runtime_outdated": ("The voice runtime needs an update before it can do this.", "install-voice-runtime"),
    "addon_missing": ("A voice add-on that this needs is not installed.", "install-addon"),
    "model_missing": ("The model for this is not downloaded yet.", "download-model"),
    "watermark_missing": ("The watermark part that cloned voices need is missing.", "download-model"),
    "gpu_memory": ("This model needs more GPU memory than is free.", "smaller-model"),
    "gpu_unsupported": ("This model needs a newer NVIDIA GPU than this computer has.", "pick-another-model"),
    "unsupported_here": ("This model cannot run on this computer.", "pick-another-model"),
    "ffmpeg_missing": ("Changing the speed or pitch needs ffmpeg, which is not installed.", "install-ffmpeg"),
    "reference_too_short": ("The voice sample is too short.", "change-sample"),
    "reference_too_long": ("The voice sample is too long.", "change-sample"),
    "reference_unreadable": ("The voice sample could not be used.", "change-sample"),
    "audio_unreadable": ("This file could not be read as audio.", "change-sample"),
    "language_unsupported": ("This model does not speak the chosen language.", "pick-another-language"),
    "permission_required": ("Permission to clone this voice has not been confirmed.", "confirm-permission"),
    "cloning_not_allowed": ("Cloned voices can only be used from the app window unless agent access is on.", "none"),
    "voice_not_found": ("That voice no longer exists.", "none"),
    "no_backup_voice": ("The chosen voice could not speak and no backup voice is set.", "none"),
    "download_blocked": ("The download was refused; the model may need a Hugging Face sign-in.", "add-hf-token"),
    "download_failed": ("The download did not finish.", "retry"),
    "worker_busy": ("Another voice job is running.", "retry"),
    "worker_timeout": ("The voice worker took too long to answer.", "retry"),
    "worker_stopped": ("The voice worker stopped unexpectedly.", "retry"),
    "unknown": ("Voice failed for a reason the app does not recognise.", "retry"),
}

# Python modules the speech GPU add-on brings; a missing one means the add-on is not (fully) installed.
_SPEECH_GPU_MODULES = {"torch", "torchaudio", "chatterbox", "voxcpm", "qwen_tts", "sox", "accelerate", "snac", "transformers"}

_DURATION = re.compile(r"this one is (\d+(?:\.\d+)?) s")


def _info(code: str, extra: str | None = None) -> VoiceErrorInfo:
    message, action = _TABLE[code]
    return VoiceErrorInfo(code, message, action, extra)


def _addon_extra(text: str) -> str | None:
    if "kokoro" in text:
        return "kokoro"
    if "speech-to-text" in text or "transformers" in text:
        return "stt-gpu"
    if "speech gpu add-on" in text or "nvidia gpu" in text:
        return "speech-gpu"
    if "chinese" in text or "mandarin" in text:
        return "chinese"
    return None


def classify(raw: object, status: int | None = None) -> VoiceErrorInfo:
    """The code, plain sentence and action for a raw error text (or an exception)."""
    text = str(raw or "")
    low = text.lower()

    # The worker itself: a dead, slow or busy process, told apart from what the request asked for.
    if "did not answer within" in low:
        return _info("worker_timeout")
    if "voice worker exited" in low or "voice worker stopped" in low or "voice worker did not start" in low:
        return _info("worker_stopped")
    if "another voice is speaking" in low or "another transcription is running" in low:
        return _info("worker_busy")

    # Setup: the runtime, then add-ons, then models.
    if "voice runtime is not installed" in low:
        return _info("runtime_missing")
    if "voice runtime update" in low:
        return _info("runtime_outdated")
    if ("runs on apple silicon macs only" in low or "runs on computers with an nvidia gpu only" in low
            or "is not offered on this version" in low or "speech gpu add-on is for windows" in low
            or "nvidia-smi was not found" in low or "is not checked on linux arm64" in low):
        return _info("unsupported_here")
    if "compute capability" in low:
        return _info("gpu_unsupported")
    if "gb of gpu memory" in low or "out of memory" in low or "outofmemoryerror" in low:
        return _info("gpu_memory")
    if "watermark component" in low:
        return _info("watermark_missing")
    if ("kokoro quick voices are not installed" in low or "speech gpu add-on" in low
            or "speech-to-text gpu add-on" in low or "speech-to-text models on the nvidia gpu" in low
            or "models on the nvidia gpu" in low or "need an extra install" in low
            or re.search(r"cannot run this model\. untick", low)):
        return _info("addon_missing", _addon_extra(low))
    module = re.search(r"no module named '([\w.]+)'", low)
    if module:
        name = module.group(1).split(".")[0]
        return _info("addon_missing", "speech-gpu" if name in _SPEECH_GPU_MODULES else None)
    if "is not downloaded" in low:
        return _info("model_missing")

    # The audio the user chose or recorded.
    if ("needs ffmpeg" in low or "ffmpeg was not found" in low or "ffmpeg not found" in low
            or "ffmpeg is not installed" in low or ("ffmpeg" in low and "no such file or directory" in low)):
        return _info("ffmpeg_missing")
    if "voice sample of at least" in low:
        return _info("reference_too_short")
    if "voice sample must be 3 to 30" in low:
        length = _DURATION.search(low)
        return _info("reference_too_short" if length and float(length.group(1)) < 3.0 else "reference_too_long")
    if ("voice sample must be a" in low or "no speech was heard in the voice sample" in low
            or "sample is missing" in low):
        return _info("reference_unreadable")
    if ("invalid data found" in low or "invaliddataerror" in low or "no audio track" in low
            or "contains no audio" in low or "could not be read as audio" in low):
        return _info("audio_unreadable")
    if "unsupported language" in low or "is not one of them" in low or "language must be" in low:
        return _info("language_unsupported")

    # Cloning consent and ownership.
    if "no fallback voice is set" in low:
        return _info("no_backup_voice")
    if "cloned voice not found" in low:
        return _info("voice_not_found")
    if "cloned voices can only be used from" in low:
        return _info("cloning_not_allowed")
    if "confirm that this is" in low or "permission_confirmed" in low:
        return _info("permission_required")

    # Downloads: refused (gated, not signed in) versus broken (network).
    if ((re.search(r"\b40[13]\b", low) and "download" in low)
            or "gated" in low or "is restricted" in low or "repository not found" in low
            or "access to model" in low):
        return _info("download_blocked")
    if re.search(r"downloading .* failed|connection|getaddrinfo|name or service not known|network is unreachable", low):
        return _info("download_failed")

    info = _info("unknown")
    if status is not None and status < 500:
        return VoiceErrorInfo(info.code, info.message, "none", None)
    return info


def error_fields(raw: object, status: int | None = None) -> dict[str, Any]:
    """``{code, message, action, extra}`` for a raw error text; merged into error bodies and stream lines."""
    return classify(raw, status).fields()


class VoiceHTTPError(HTTPException):
    """An ``HTTPException`` whose response also carries ``code``, ``message``, ``action`` and ``extra``."""

    def __init__(self, status_code: int, detail: str, info: VoiceErrorInfo) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.info = info


def http_error(status_code: int, raw: object) -> VoiceHTTPError:
    """The voice route error for ``raw`` at ``status_code``: same status and ``detail`` as a plain HTTPException."""
    text = str(raw)
    return VoiceHTTPError(status_code, text, classify(text, status_code))


async def _voice_http_error_handler(request: Request, exc: VoiceHTTPError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, **exc.info.fields()},
        headers=getattr(exc, "headers", None),
    )


def install(app: FastAPI) -> None:
    """Render ``VoiceHTTPError`` with the extra fields (the default handler would send ``detail`` only)."""
    app.add_exception_handler(VoiceHTTPError, _voice_http_error_handler)  # type: ignore[arg-type]
