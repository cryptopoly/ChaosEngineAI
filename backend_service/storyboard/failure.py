"""Why a batch step failed, as a stable code, one plain sentence and the one action that helps (audit change 5).

``batch.py`` stores the code beside the raw error text (``BatchResult.code`` and ``error``); the raw text stays for
Technical details. ``classify`` reads the exception type first, then known phrases of the image and video routes,
the memory gate, the model picker and the video tools check; anything else is ``unknown``. ``describe`` turns a code
into the sentence and the action code the app maps to a button. Retry behaviour does not depend on any of this.
"""

from __future__ import annotations

from typing import Any

# code -> (plain sentence, action code). The app maps the action code to one button.
FAILURES: dict[str, tuple[str, str]] = {
    "out-of-memory": ("There was not enough free memory to make this shot at this size.", "smaller-size"),
    "model-missing": ("The model this shot needs is not installed.", "open-models"),
    "lipsync-model": ("The project's video model cannot lip-sync this shot's vocal.", "open-models"),
    "download-blocked": ("The model could not be downloaded (it may need its licence accepted, or the "
                         "connection is down).", "open-models"),
    "video-tools-missing": ("The video tools are not installed.", "install-video-tools"),
    "size-over-cap": ("The shot's size is larger than the model can make.", "smaller-size"),
    "references-unsupported": ("The model cannot take this shot's pictures (a start, end or reference picture).",
                               "open-models"),
    "no-prompt": ("The shot has nothing to make it from: write its prompt first.", "edit-shot"),
    "disk-full": ("The disk is full, so the shot could not be saved.", "free-disk"),
    "cancelled": ("The shot was stopped before it finished.", "retry"),
    "unknown": ("The shot did not render.", "retry"),
}
CODES = tuple(FAILURES)
ACTIONS = ("smaller-size", "open-models", "install-video-tools", "edit-shot", "free-disk", "retry")

# Lower-case phrases per code, checked in this order (the first match wins).
_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lipsync-model", ("cannot lip-sync", "lip-sync model", "lipsync model")),
    ("references-unsupported", ("cannot start a clip from an image", "follows only a start image",
                                "takes no reference picture", "does not take reference", "reference images are not",
                                "reference pictures are not", "a lip-sync shot starts from a still")),
    ("video-tools-missing", ("video tools are needed", "install video tools", "bundled ffmpeg is not available",
                             "ffmpeg is not available", "no usable ffmpeg")),
    ("disk-full", ("no space left on device", "disk is full", "not enough space on the disk", "errno 28")),
    ("out-of-memory", ("out of memory", "outofmemoryerror", "memory pressure is", "of free memory to",
                       "of ram available", "not enough memory", "not enough free memory", "memory_gate_",
                       "cuda error: out of memory", "failed to allocate")),
    ("download-blocked", ("gated", "accept the licence", "accept the license", "401 client error",
                          "403 client error", "connection error", "could not connect", "offline mode",
                          "failed to resolve", "max retries exceeded")),
    ("model-missing", ("is not installed", "not installed locally", "is not an installed", "no image model is installed",
                       "no video model is installed", "unknown image model", "unknown video model",
                       "model files are missing")),
    ("size-over-cap", ("allows at most", "larger than the model", "exceeds the model", "too large for this model",
                       "maximum size")),
    ("no-prompt", ("has no still prompt", "has no motion prompt")),
    ("cancelled", ("cancelled", "canceled")),
)


def classify(error: str | None, exc: BaseException | None = None) -> str:
    """The failure code for an exception and its raw text (``unknown`` when nothing known matches)."""
    if exc is not None:
        from backend_service.progress import GenerationCancelled
        from backend_service.storyboard.ffmpeg_bin import FfmpegMissing

        if isinstance(exc, FfmpegMissing):
            return "video-tools-missing"
        if isinstance(exc, GenerationCancelled):
            return "cancelled"
        if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
            return "out-of-memory"
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            return "disk-full"
    text = (error or "").lower()
    for code, phrases in _PHRASES:
        if any(phrase in text for phrase in phrases):
            return code
    return "unknown"


def describe(code: str | None) -> dict[str, str]:
    """``{"code", "message", "action"}`` for a stored code (an unknown or missing one reads as ``unknown``)."""
    known = code if code in FAILURES else "unknown"
    message, action = FAILURES[known]
    return {"code": known, "message": message, "action": action}


def result_failure(result: Any) -> dict[str, Any]:
    """A failed ``BatchResult`` for the app: its shot and kind, code, sentence, action and the raw text. Results stored
    before codes existed are classified from their text."""
    code = result.code or classify(result.error)
    return {"shotId": result.shotId, "kind": result.kind, **describe(code), "error": result.error}
