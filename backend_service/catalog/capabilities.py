"""Model capability resolver — Phase 2.11.

Maps a loaded model's ref/canonical-repo to a typed capability blob the
UI can use to gate composer features (image attach hidden for text-only
models, tools toggle hidden for non-tool models, etc.) and to render
capability badges next to the model picker.

The resolver consults the curated text-model catalog first (each
variant carries a `capabilities: [...]` string list); when no catalog
entry matches it falls back to ref-name heuristics so freshly downloaded
HF models without a catalog entry still get sensible defaults.

Capabilities are intentionally conservative — when in doubt the
resolver omits the flag rather than promising support that may not
materialise. The frontend treats unknown capabilities as "hide the UI
affordance" so incorrectly omitting a flag degrades gracefully.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from backend_service.catalog.text_models import MODEL_FAMILIES


@dataclass
class ModelCapabilities:
    supportsVision: bool = False
    supportsTools: bool = False
    supportsReasoning: bool = False
    supportsCoding: bool = False
    supportsAgents: bool = False
    supportsAudio: bool = False
    supportsVideo: bool = False
    supportsMultilingual: bool = False
    # Free-form tags from the catalog (or heuristic fallback) preserved
    # so the UI can render badges without re-deriving them.
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["tags"] = list(self.tags)
        return out


# Maps catalog capability strings to fields on ModelCapabilities. Strings
# the catalog uses freely ("multilingual", "thinking", etc.) get folded
# into the closest typed flag.
_CAPABILITY_TO_FLAG: dict[str, str] = {
    "vision": "supportsVision",
    "multimodal": "supportsVision",
    "tool-use": "supportsTools",
    "tools": "supportsTools",
    "function-calling": "supportsTools",
    "reasoning": "supportsReasoning",
    "thinking": "supportsReasoning",
    "coding": "supportsCoding",
    "code": "supportsCoding",
    "agents": "supportsAgents",
    "agent": "supportsAgents",
    "audio": "supportsAudio",
    "video": "supportsVideo",
    "multilingual": "supportsMultilingual",
}


def _normalise_ref(value: str | None) -> str:
    return (value or "").strip().lower()


def _catalog_lookup(model_ref: str | None, canonical_repo: str | None) -> list[str] | None:
    """Find the variant whose `id` or `repo` matches the loaded model.

    Falls back to family-level capabilities when no variant matches but
    the family-level repo is a prefix of the loaded ref. This catches
    community quantised forks (e.g. `mlx-community/Qwen3-Coder-Next-MLX-4bit`)
    whose ref doesn't appear verbatim in the catalog.
    """
    ref = _normalise_ref(model_ref)
    canonical = _normalise_ref(canonical_repo)
    if not ref and not canonical:
        return None

    for family in MODEL_FAMILIES:
        for variant in family.get("variants", []):
            variant_id = _normalise_ref(variant.get("id"))
            variant_repo = _normalise_ref(variant.get("repo"))
            if ref and (ref == variant_id or ref == variant_repo):
                caps = variant.get("capabilities")
                if isinstance(caps, list):
                    return [str(c) for c in caps]
            if canonical and (canonical == variant_id or canonical == variant_repo):
                caps = variant.get("capabilities")
                if isinstance(caps, list):
                    return [str(c) for c in caps]

    # Family-level fallback: match by ref or canonical containing the
    # family id or any of its variant repos as a substring.
    for family in MODEL_FAMILIES:
        family_caps = family.get("capabilities")
        if not isinstance(family_caps, list):
            continue
        family_id = _normalise_ref(family.get("id"))
        if not family_id:
            continue
        for needle in (ref, canonical):
            if not needle:
                continue
            if family_id in needle:
                return [str(c) for c in family_caps]
            for variant in family.get("variants", []):
                variant_repo = _normalise_ref(variant.get("repo"))
                if variant_repo and variant_repo in needle:
                    return [str(c) for c in family_caps]
    return None


def _heuristic_capabilities(model_ref: str | None) -> list[str]:
    """Fallback when the catalog has no entry for the loaded model.

    Pure substring sniff against common repo conventions: vision models
    typically include "vl" / "vision" / "llava" in the ref; coder models
    include "coder" / "code"; reasoning models often advertise "r1" /
    "reasoning" / "think". Conservative — only emit flags backed by a
    well-established naming convention.
    """
    if not model_ref:
        return []
    lower = model_ref.lower()
    out: list[str] = []
    if any(needle in lower for needle in ("-vl-", " vl ", "/vl-", "vision", "llava", "qwen-vl", "moondream")):
        out.append("vision")
    if any(needle in lower for needle in ("coder", "/code-", "starcoder", "deepseek-coder", "code-llama")):
        out.append("coding")
    if any(needle in lower for needle in ("r1", "reasoning", "think", "qwen3", "deepseek-r")):
        out.append("reasoning")
    if "tool" in lower or "function" in lower:
        out.append("tool-use")
    if "instruct" in lower or "-it" in lower or "chat" in lower:
        # Instruction-tuned models almost always support chat-style tool
        # prompts even when the catalog hasn't been updated.
        if "tool-use" not in out:
            out.append("tool-use")
    return out


def resolve_capabilities(
    model_ref: str | None,
    canonical_repo: str | None = None,
) -> ModelCapabilities:
    """Public entry point — returns a typed capability blob for a model.

    Catalog match wins; heuristic fallback applies only when nothing in
    the catalog matched. Always returns a valid `ModelCapabilities` (no
    None) so callers don't need to null-check.
    """
    raw = _catalog_lookup(model_ref, canonical_repo)
    if raw is None:
        raw = _heuristic_capabilities(model_ref)

    caps = ModelCapabilities()
    seen: set[str] = set()
    for tag in raw:
        normalised = tag.strip().lower()
        if not normalised:
            continue
        seen.add(normalised)
        flag = _CAPABILITY_TO_FLAG.get(normalised)
        if flag is not None:
            setattr(caps, flag, True)
    caps.tags = tuple(sorted(seen))
    return caps
