"""Free GPU memory while rendering (FU-191 phase 2): a Keep going run can unload the chat model around a heavy
render and load it again afterwards, for cards where both do not fit (a 24 GB NVIDIA card, for example).

How it works: in a run's turn, an approved render call (image, video, music, sound, speech, storyboard generate,
render or captions) is not started when the swap rule says so. It is kept for after the turn, and the model is
told to end its turn. When the turn has ended the controller unloads the chat model, starts the kept renders,
waits for their jobs, loads the same model again with the same launch settings, then sends the next nudge with
the results. A model that will not load again stops the run with a reason; nothing is retried in a loop.

The rule (``swap_decision``): Never, Always, or Auto. Auto swaps only when the render's estimated peak does not fit
the memory free now but would fit with the chat model's share freed. Free memory is the GPU's on a CUDA card and
the system's on unified memory (Apple Silicon, DGX Spark), where Auto rarely triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("auto", "always", "never")
HEADROOM_GB = 1.5
DEFERRED_TEXT = (
    "Queued: to free GPU memory the app runs this render after your turn, with the chat model unloaded, and "
    "tells you the result in its next message. Do not call wait_for_job for it. End your turn now with one line "
    "saying what you queued."
)
# The kinds the job store uses, for the default peak when the catalogue gives none.
_KIND_FOR_TOOL = {"generate_image": "image", "generate_video": "video", "make_music": "music", "make_sound": "music",
                  "speak": "voice", "storyboard_generate": "video", "storyboard_render": "video",
                  "storyboard_captions": "voice"}


@dataclass
class SwapChoice:
    swap: bool
    why: str  # never | always | fits | unknown | too_big | swap


def swap_decision(mode: str, *, job_gb: float | None, free_gb: float | None, chat_gb: float | None) -> SwapChoice:
    """Whether to unload the chat model for a render. Pure, so every branch is unit-tested."""
    if mode == "never":
        return SwapChoice(False, "never")
    if mode == "always":
        return SwapChoice(True, "always")
    if job_gb is None or free_gb is None:
        return SwapChoice(False, "unknown")
    if job_gb + HEADROOM_GB <= free_gb:
        return SwapChoice(False, "fits")
    if chat_gb is not None and job_gb + HEADROOM_GB > free_gb + chat_gb:
        return SwapChoice(False, "too_big")  # it would not fit without the chat model either: the job's gate says so
    return SwapChoice(True, "swap")


def launch_fields(loaded: Any) -> dict[str, Any] | None:
    """The load request that brings this exact model back with the same launch settings."""
    if loaded is None or not getattr(loaded, "ref", None):
        return None
    return {
        "modelRef": loaded.ref,
        "modelName": loaded.name,
        "canonicalRepo": getattr(loaded, "canonicalRepo", None),
        "source": loaded.source,
        "backend": loaded.backend,
        "path": getattr(loaded, "path", None),
        "cacheStrategy": loaded.cacheStrategy,
        "cacheBits": int(loaded.cacheBits),
        "fp16Layers": int(loaded.fp16Layers),
        "fusedAttention": bool(loaded.fusedAttention),
        "fitModelInMemory": bool(loaded.fitModelInMemory),
        "contextTokens": int(getattr(loaded, "requestedContextTokens", None) or loaded.contextTokens),
        "speculativeDecoding": bool(getattr(loaded, "speculativeDecoding", False)),
        "treeBudget": int(getattr(loaded, "treeBudget", 0) or 0),
    }


def same_model(loaded: Any, fields: dict[str, Any]) -> bool:
    return loaded is not None and loaded.ref == fields.get("modelRef") and (
        not fields.get("path") or getattr(loaded, "path", None) == fields.get("path"))


def files_gb(path: str | None) -> float | None:
    """Size of a model's weights on disk (a fair floor for what it holds in memory), or None."""
    if not path:
        return None
    root = Path(path)
    try:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.suffix in (".safetensors", ".gguf", ".bin", ".npz")]
        total = sum(p.stat().st_size for p in files if p.is_file())
    except OSError:
        return None
    return round(total / 1e9, 2) if total else None


class AppSwap:
    """The swap's view of the running app: memory now, the chat model's share, a render's peak, unload and load."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def free_gb(self) -> float | None:
        from backend_service.helpers.memory_gate import cuda_free_vram_gb

        cuda = cuda_free_vram_gb()
        if cuda is not None:
            return cuda
        try:
            import psutil

            from backend_service.helpers.memory_gate import available_memory_gb

            memory = psutil.virtual_memory()
            return available_memory_gb(memory.total, memory.available)
        except Exception:  # noqa: BLE001 - unknown memory: Auto does not swap
            return None

    def chat_gb(self) -> float | None:
        loaded = self.state.runtime.loaded_model
        return files_gb(getattr(loaded, "path", None)) if loaded is not None else None

    def job_gb(self, name: str, args: dict[str, Any]) -> float | None:
        from backend_service.helpers.busy_advice import DEFAULT_JOB_PEAK_GB
        from backend_service.tools.studio_gate import plan_item

        try:
            peak = plan_item({"id": "x", "function": {"name": name, "arguments": args}}, self.state).get("peakGb")
        except Exception:  # noqa: BLE001
            peak = None
        if isinstance(peak, (int, float)) and peak > 0:
            return float(peak)
        return DEFAULT_JOB_PEAK_GB.get(_KIND_FOR_TOOL.get(name, ""), 8.0)

    def unload(self) -> dict[str, Any] | None:
        fields = launch_fields(self.state.runtime.loaded_model)
        if fields is not None:
            self.state.unload_model()
        return fields

    def reload(self, fields: dict[str, Any]) -> None:
        """Load the same model again; raises with a readable reason when it does not come back."""
        from backend_service.models import LoadModelRequest

        if same_model(self.state.runtime.loaded_model, fields):
            return  # the user (or a turn) loaded it again meanwhile
        self.state.load_model(LoadModelRequest(**fields))
        if not same_model(self.state.runtime.loaded_model, fields):
            raise RuntimeError(f"{fields.get('modelName') or fields.get('modelRef')} did not load again")

    def start_render(self, app_request: Any, name: str, args: dict[str, Any]) -> tuple[str | None, str | None]:
        """Run one kept render call as Chat would (owner chat, same scopes); returns (job id, error)."""
        from backend_service.tools.studio import studio_tools

        tool = next((t for t in studio_tools(app_request) if t.name == name), None)
        if tool is None:
            return None, f"{name} is not a studio tool"
        if name.startswith("storyboard_") and "revision" in args:
            try:  # the model's own later edits in the same turn moved the project on; the render was approved
                from backend_service.routes.storyboard import get_store

                args = {**args, "revision": get_store().load(str(args["project_id"])).revision}
            except Exception:  # noqa: BLE001 - the tool reports a bad project itself
                pass
        output = tool.execute_structured(**args)
        data = output.data if isinstance(output.data, dict) else {}
        if data.get("error"):
            return None, str(data["error"])
        job_id = data.get("jobId")
        return (str(job_id), None) if job_id else (None, None)
