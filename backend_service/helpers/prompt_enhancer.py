"""LLM-based prompt enhancer (FU-022).

Replaces the deterministic per-family suffix template that ``_enhance_prompt``
appends in ``video_runtime.py`` with a small instruction model that
auto-rewrites short prompts into the structured 50-100 word format each
video DiT was trained on. Apple Silicon path uses ``mlx_lm`` directly;
CUDA / Linux fall back to the legacy template suffix until a llama.cpp
GGUF path lands.

Default model: ``mlx-community/Qwen2.5-0.5B-Instruct-4bit`` (~700 MB on
disk, ~2-3s cold load on M-series, sub-second per generation). Picked
over the 1B Llama variant the original FU-022 plan named because:
  * smaller memory footprint when the enhancer shares the FastAPI
    sidecar's process (vs spawning a dedicated worker)
  * already cached on most dev boxes (FU-002 spike used it)
  * 0.5B Qwen2.5-Instruct still produces the structured 50-100 word
    rewrites we need; the enhancer task is constrained enough that the
    extra reasoning headroom of 1B isn't load-bearing.

The helper caches the loaded model in a process-level singleton —
first call pays the load cost, subsequent calls reuse it. Failure
modes (model not cached, mlx_lm missing, generation crash) all return
the original prompt + a runtimeNote so the caller can decide whether
to show the user that enhancement was skipped.
"""

from __future__ import annotations

import logging
import platform
import threading
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


# Per-family system prompt that anchors the model to the DiT's training
# distribution. Keeps the rewrite short (under 100 words) so we don't
# produce verbose paragraphs that overflow the text encoder context
# window. Each suffix mirrors the upstream model card's recommended
# prompt structure.
_FAMILY_SYSTEM_PROMPTS: dict[str, str] = {
    "wan": (
        "You rewrite short user prompts into Wan-AI video model format. "
        "Stay under 80 words. Always include: subject + action + setting + "
        "camera angle + lighting + mood. Do not add cinematic jargon the "
        "user did not ask for. Output only the rewritten prompt — no "
        "preamble, no quotation marks."
    ),
    "ltx": (
        "You rewrite short user prompts into LTX-Video format. Stay under "
        "70 words. Always include: subject + action + setting + camera "
        "movement (e.g. 'tracking shot', 'static wide angle') + lighting "
        "(e.g. 'golden hour', 'overcast'). Output only the rewritten "
        "prompt — no preamble, no quotation marks."
    ),
    "hunyuan": (
        "You rewrite short user prompts into HunyuanVideo format. Stay "
        "under 75 words. Always include: subject + action + setting + "
        "camera shot (close-up / medium / wide) + atmosphere. Avoid "
        "redundant adjectives. Output only the rewritten prompt — no "
        "preamble, no quotation marks."
    ),
    "flux": (
        "You rewrite short user prompts into FLUX image format. Stay "
        "under 60 words. Always include: subject + composition + "
        "lighting + style (e.g. 'photorealistic', 'oil painting', "
        "'cinematic'). Output only the rewritten prompt — no preamble, "
        "no quotation marks."
    ),
    "sdxl": (
        "You rewrite short user prompts into SDXL image format. Stay "
        "under 50 words. Always include: subject + composition + "
        "lighting + comma-separated style tags. Output only the "
        "rewritten prompt — no preamble, no quotation marks."
    ),
    "sd3": (
        "You rewrite short user prompts into Stable Diffusion 3 format. "
        "Stay under 60 words. Always include: subject + setting + "
        "composition + lighting + medium / style. Output only the "
        "rewritten prompt — no preamble, no quotation marks."
    ),
    "default": (
        "You rewrite short user prompts into a richer 50-80 word "
        "description while preserving the user's intent. Always include: "
        "subject + action + setting + lighting + style. Output only the "
        "rewritten prompt — no preamble, no quotation marks."
    ),
}


# Repo-prefix → family id (longest match wins). ``family_for`` walks
# this in declared order, so put more-specific prefixes first.
_FAMILY_MAP: list[tuple[str, str]] = [
    ("Wan-AI/", "wan"),
    ("QuantStack/Wan", "wan"),
    ("Lightricks/LTX", "ltx"),
    ("prince-canuma/LTX", "ltx"),
    ("hunyuanvideo-community/", "hunyuan"),
    ("tencent/HunyuanVideo", "hunyuan"),
    ("black-forest-labs/FLUX", "flux"),
    ("fal/FLUX", "flux"),
    ("stabilityai/stable-diffusion-3", "sd3"),
    ("stabilityai/stable-diffusion-xl", "sdxl"),
    ("stabilityai/sdxl-turbo", "sdxl"),
    ("ByteDance/SDXL-Lightning", "sdxl"),
]


# Default enhancer model. Override via ``CHAOSENGINE_ENHANCER_MODEL``
# env var when a different small instruct model is preferred.
_DEFAULT_ENHANCER_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def family_for(repo: str) -> str:
    """Map a base repo id to a family id used by the system prompt
    table. Falls back to ``"default"`` for unknown repos."""
    for prefix, family in _FAMILY_MAP:
        if repo.startswith(prefix):
            return family
    return "default"


@dataclass(frozen=True)
class EnhancementResult:
    """Output of ``enhance_prompt``. ``enhanced == prompt`` when the
    enhancer was unavailable / errored — the caller still gets a
    runtimeNote so the user sees why."""

    enhanced: str
    note: str | None
    modelUsed: str | None
    family: str


class _EnhancerSingleton:
    """Process-level cache for the loaded mlx_lm model + tokenizer.
    First call into ``ensure_loaded`` pays the ~2-3s load cost;
    subsequent calls reuse the in-memory state under a lock so two
    concurrent enhancement requests don't both try to load."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._unavailable_reason: str | None = None

    def reset(self) -> None:
        """Drop the cached model — caller invokes this when a memory
        pressure event tells us to free up RAM, or in test setUp."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._model_id = None
            self._unavailable_reason = None

    def ensure_loaded(self, model_id: str) -> tuple[bool, str | None]:
        """Idempotent load. Returns ``(loaded, error_reason)``."""
        with self._lock:
            if self._model is not None and self._model_id == model_id:
                return True, None
            # Different model requested — drop the old one before loading
            # the new. Prevents two ~700 MB models stacking in memory.
            self._model = None
            self._tokenizer = None
            self._model_id = None

            if platform.system() != "Darwin":
                self._unavailable_reason = (
                    "Prompt enhancer requires Apple Silicon (mlx_lm). "
                    "Falling back to the deterministic template suffix."
                )
                return False, self._unavailable_reason

            try:
                from mlx_lm import load as mlx_lm_load
            except ImportError as exc:
                self._unavailable_reason = (
                    f"Prompt enhancer requires mlx_lm ({exc}). "
                    "Falling back to the deterministic template suffix."
                )
                return False, self._unavailable_reason

            try:
                model, tokenizer = mlx_lm_load(model_id)
            except Exception as exc:
                self._unavailable_reason = (
                    f"Prompt enhancer failed to load {model_id} "
                    f"({type(exc).__name__}: {exc}). Falling back to the "
                    "deterministic template suffix."
                )
                return False, self._unavailable_reason

            self._model = model
            self._tokenizer = tokenizer
            self._model_id = model_id
            self._unavailable_reason = None
            return True, None

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
        """Render the chat-template messages + run a single generation.
        Caller has already confirmed ``ensure_loaded`` succeeded."""
        with self._lock:
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Prompt enhancer model not loaded.")
            from mlx_lm import generate as mlx_lm_generate

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                rendered = self._tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
            except Exception:
                # Tokenizers without a chat template — concatenate manually.
                rendered = (
                    f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
                )

            return mlx_lm_generate(
                self._model,
                self._tokenizer,
                prompt=rendered,
                max_tokens=max_tokens,
                verbose=False,
            )


_SINGLETON = _EnhancerSingleton()


def reset_singleton_for_test() -> None:
    """Test-only hook: forces the next ``enhance_prompt`` call to
    re-load. Production code never calls this."""
    _SINGLETON.reset()


def enhance_prompt(
    prompt: str,
    *,
    repo: str,
    enabled: bool = True,
    model_id: str = _DEFAULT_ENHANCER_MODEL,
    max_tokens: int = 256,
) -> EnhancementResult:
    """Synchronous entry point for the FastAPI route + the runtime
    callbacks.

    Returns the original prompt + a note when the enhancer can't run
    (disabled, non-Apple, mlx_lm missing, model not cached, generation
    crashes). The caller falls back to the deterministic template
    suffix in that case so the user still gets a usable prompt.
    """
    cleaned = (prompt or "").strip()
    family = family_for(repo)

    if not enabled or not cleaned:
        return EnhancementResult(
            enhanced=cleaned, note=None, modelUsed=None, family=family,
        )

    loaded, reason = _SINGLETON.ensure_loaded(model_id)
    if not loaded:
        return EnhancementResult(
            enhanced=cleaned,
            note=reason or "Prompt enhancer unavailable.",
            modelUsed=None,
            family=family,
        )

    system_prompt = _FAMILY_SYSTEM_PROMPTS.get(family, _FAMILY_SYSTEM_PROMPTS["default"])
    try:
        raw = _SINGLETON.generate(system_prompt, cleaned, max_tokens=max_tokens)
    except Exception as exc:
        LOG.exception("Prompt enhancer generation failed")
        return EnhancementResult(
            enhanced=cleaned,
            note=(
                f"Prompt enhancer crashed ({type(exc).__name__}: {exc}). "
                "Using your original prompt verbatim."
            ),
            modelUsed=model_id,
            family=family,
        )

    enhanced = raw.strip().strip('"').strip("'")
    if not enhanced or len(enhanced.split()) < len(cleaned.split()):
        # Model produced something shorter than input — likely a refusal
        # or empty completion. Fall back to the original.
        return EnhancementResult(
            enhanced=cleaned,
            note="Prompt enhancer returned an empty / shorter rewrite — using the original.",
            modelUsed=model_id,
            family=family,
        )

    note = (
        f"Prompt enhanced via {model_id} (family={family}, "
        f"{len(cleaned.split())} → {len(enhanced.split())} words)."
    )
    return EnhancementResult(
        enhanced=enhanced, note=note, modelUsed=model_id, family=family,
    )
