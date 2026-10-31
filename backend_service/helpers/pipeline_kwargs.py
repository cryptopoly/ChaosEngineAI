"""Drop the arguments a diffusers pipeline does not accept.

Image and video pipelines differ in what they take: some have no negative
prompt, some no guidance scale, some no batch argument. Passing an unknown
keyword raises ``TypeError`` and the whole run fails for a setting the model
never offered. ``filter_pipeline_kwargs`` reads the pipeline's call signature
and returns the arguments it takes plus a note for each one it dropped.
"""

from __future__ import annotations

import inspect
from typing import Any

# Plain-language names for the arguments a user can set in the Studios.
_LABELS: dict[str, str] = {
    "negative_prompt": "negative prompt",
    "guidance_scale": "guidance",
    "true_cfg_scale": "guidance",
    "num_images_per_prompt": "batch size",
    "num_videos_per_prompt": "batch size",
    "timesteps": "sampler schedule",
    "width": "width",
    "height": "height",
    "num_frames": "frame count",
    "frame_rate": "frame rate",
}

# Arguments the runtime adds for its own bookkeeping. They are dropped without
# a note: a pipeline without them still runs, only the live progress is lost.
_QUIET = frozenset({"callback_on_step_end", "callback_on_step_end_tensor_inputs"})

# Never dropped: a pipeline that lacks these cannot run the request at all.
_ESSENTIAL = frozenset({"prompt"})


def _call_parameters(pipeline: Any) -> dict[str, inspect.Parameter] | None:
    try:
        return dict(inspect.signature(pipeline.__call__).parameters)
    except (TypeError, ValueError, AttributeError):
        return None


def filter_pipeline_kwargs(pipeline: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return ``(accepted kwargs, notes)`` for ``pipeline``.

    Nothing is dropped when the signature cannot be read or the pipeline takes
    ``**kwargs`` (it decides for itself what to do with extras).
    """
    params = _call_parameters(pipeline)
    if params is None or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []

    accepted: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in kwargs.items():
        # ``__name`` keys are private flags the runtime strips before the call.
        if key in params or key in _ESSENTIAL or key.startswith("__"):
            accepted[key] = value
        elif key not in _QUIET:
            dropped.append(key)

    names: list[str] = []
    for key in dropped:
        label = _LABELS.get(key, key)
        if label not in names:
            names.append(label)
    notes = [f"Not supported by this model, so it was ignored: {', '.join(names)}."] if names else []
    return accepted, notes
