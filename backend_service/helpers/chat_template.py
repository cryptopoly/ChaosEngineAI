"""Phase 3.8: chat-template inspection + auto-fix detection.

Reasoning models and their tokenisers ship a `chat_template` Jinja
fragment that the runtime calls via `apply_chat_template` to format
multi-turn history. The template encodes:

- Where role markers go (`<|im_start|>`, `<start_of_turn>`, etc.)
- Whether system messages are supported
- Whether the tokeniser accepts `add_generation_prompt` so the
  rendered prompt ends with an assistant-side prefix the model
  treats as "your turn now"

Gemma-family models (Gemma-1 through Gemma-4) reject system role
entirely; ChatML-derived templates sometimes ship without
`add_generation_prompt` handling and produce truncated last-user
turns; a handful of GGUF community quants pin a stale chat template
that doesn't match the model's actual training format.

This helper inspects a tokeniser at load time, returns a structured
report of detected issues and fixes the runtime can apply, and gives
the rest of the codebase a single place to encode "we know about
this template quirk".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatTemplateReport:
    """Outcome of inspecting a tokeniser's chat-template support.

    `issues` lists detected problems; `fixes_applied` lists the
    workarounds the runtime can transparently apply (no user action
    needed). When both are empty, the template is healthy.
    """
    issues: list[str] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    template_present: bool = True
    accepts_system_role: bool = True
    accepts_generation_prompt: bool = True

    @property
    def needs_attention(self) -> bool:
        return bool(self.issues) or bool(self.fixes_applied)

    def to_runtime_note(self) -> str | None:
        """Render a single-line note suitable for `runtime_note` on
        a generation result. Returns None when the template is healthy.
        """
        if not self.needs_attention:
            return None
        parts: list[str] = []
        if self.fixes_applied:
            parts.append("auto-fixed: " + ", ".join(self.fixes_applied))
        if self.issues:
            parts.append("issues: " + ", ".join(self.issues))
        return "Chat template " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# Gemma family lowercased markers — used to identify models whose chat
# template rejects the system role.
_GEMMA_PREFIXES: tuple[str, ...] = (
    "google/gemma-",
    "gemma-",
    "mlx-community/gemma-",
    "lmstudio-community/gemma-",
)

# Multimodal (vision-capable) repo prefixes. Lowercased prefix match.
# Models in this set get loaded via ``mlx_vlm.load`` instead of
# ``mlx_lm.load`` and route through the multimodal generate path
# (which decodes the chat ``images`` field into per-image paths and
# passes them to ``mlx_vlm.generate`` / ``stream_generate``).
#
# Add new prefixes here when adopting a vision-capable family. Text-only
# Gemma variants (e.g. older Gemma 1/2 text-only quants on mlx-community
# would go here NEGATIVELY — but Gemma 4 is multimodal across the entire
# family per Google's release, so all gemma-4 variants qualify).
_MULTIMODAL_PREFIXES: tuple[str, ...] = (
    # Gemma 4 family: every variant is multimodal.
    "google/gemma-4",
    "mlx-community/gemma-4",
    "lmstudio-community/gemma-4",
    # Qwen2.5-VL family: vision-language model, every variant is multimodal.
    "qwen/qwen2.5-vl",
    "mlx-community/qwen2.5-vl",
    # Qwen3-VL family: future-proofing — same naming convention.
    "qwen/qwen3-vl",
    "mlx-community/qwen3-vl",
    # LLaVA-style models running through mlx-vlm.
    "mlx-community/llava-",
    "llava-hf/llava-",
)

# ChatML / Qwen2/3 templates ship `<|im_start|>` markers. When a quant
# ships without `add_generation_prompt` support, the rendered prompt
# stops mid-turn and the model continues the user turn instead of
# replying. Detection: template string contains `<|im_start|>` but
# does NOT reference `add_generation_prompt`.
_CHATML_OPEN = "<|im_start|>"
_GENERATION_PROMPT_MARKER = "add_generation_prompt"


def _model_ref_lower(model_ref: str | None) -> str:
    return (model_ref or "").lower()


def is_gemma_family(model_ref: str | None) -> bool:
    lowered = _model_ref_lower(model_ref)
    return any(lowered.startswith(prefix) for prefix in _GEMMA_PREFIXES)


def is_multimodal_family(model_ref: str | None) -> bool:
    """Return ``True`` when the repo id matches a vision-capable family
    that should be loaded via ``mlx_vlm`` rather than ``mlx_lm``.

    Match is a lowercased prefix scan against ``_MULTIMODAL_PREFIXES``.
    Returns ``False`` for text-only models, including Gemma 1/2 quants
    that share the ``gemma-`` prefix but are not multimodal.
    """
    lowered = _model_ref_lower(model_ref)
    return any(lowered.startswith(prefix) for prefix in _MULTIMODAL_PREFIXES)


def fold_system_into_first_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gemma fix — fold the system message (if any) into the first user
    message so the chat template's system-role rejection doesn't kick in.

    Idempotent on inputs without a system message; preserves order
    otherwise.
    """
    out: list[dict[str, Any]] = []
    pending_system: str | None = None
    for message in messages:
        role = message.get("role")
        content = message.get("content") or message.get("text") or ""
        if role == "system" and not out and not pending_system:
            pending_system = str(content)
            continue
        if role == "user" and pending_system is not None:
            merged = f"{pending_system}\n\n{content}" if content else pending_system
            out.append({**message, "role": "user", "content": merged})
            pending_system = None
            continue
        out.append({**message})
    if pending_system is not None and not out:
        # System with no following user — preserve as-is rather than dropping.
        out.append({"role": "user", "content": pending_system})
    return out


def inspect_chat_template(
    template: str | None,
    model_ref: str | None = None,
) -> ChatTemplateReport:
    """Inspect a tokeniser's `chat_template` source and the model ref.

    Returns a structured report. Callers (mlx_worker, inference.py)
    apply the fix the report recommends and then surface the
    `runtime_note` so the UI can show a banner.
    """
    report = ChatTemplateReport()

    if template is None or not template.strip():
        report.template_present = False
        report.issues.append("no chat_template found on tokeniser")
        return report

    # Gemma family always rejects system role — surface this as an
    # auto-fix ("we'll fold system into first user") rather than an
    # issue the user has to act on.
    if is_gemma_family(model_ref):
        report.accepts_system_role = False
        report.fixes_applied.append("Gemma family — fold system into first user message")

    # ChatML without add_generation_prompt handling.
    if _CHATML_OPEN in template and _GENERATION_PROMPT_MARKER not in template:
        report.accepts_generation_prompt = False
        report.issues.append(
            "ChatML template missing add_generation_prompt handling — "
            "responses may truncate mid-turn"
        )

    # Detect templates that hard-code an assistant prefix in the system
    # branch, which double-prefixes when the runtime adds its own.
    if template.count("<|im_start|>assistant") > 1 and "add_generation_prompt" in template:
        report.issues.append(
            "Template hard-codes assistant prefix even when "
            "add_generation_prompt is True — may emit a doubled marker"
        )

    return report
