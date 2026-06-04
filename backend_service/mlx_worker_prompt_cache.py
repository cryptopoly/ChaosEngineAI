"""Per-session MLX prompt-cache reuse (tier 4 of the chat-LLM review).

Native-strategy chat turns re-prefill the *entire* conversation every time
(`prompt_cache=None` → mlx-lm builds a fresh cache + processes the whole
prompt). This module keeps one persistent mlx-lm prompt cache on the
worker and reuses the longest matching token prefix across turns: trim the
divergent tail off the cache, prefill only the new suffix, then re-commit
the cache keyed by ``prompt_tokens + generated_tokens``. A single-slot port
of mlx-lm's server reuse logic (``LRUPromptCache.fetch_nearest_cache``).

Correctness invariant: the persisted token list ALWAYS equals the cache's
positional contents (prompt + generated), so the next turn's common-prefix
trim is exact. Any uncertainty — compression strategy active, model
changed, cache not trimmable (SSM/Mamba/rotating-full, mlx-lm #980),
tokenisation failure, no common prefix, partial trim — falls back to a
fresh full prefill, i.e. identical output to the pre-cache path, just
without the speedup. Gated to the ``native`` strategy; compression caches
(turboquant / triattention) keep their existing per-call path untouched.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

# cache:         object passed to stream_generate as prompt_cache
# prompt_feed:   what to pass as the `prompt` arg (suffix token list on a
#                reuse hit, full token list on a fresh native cache, or the
#                original prompt_text string for the compression / fallback path)
# note:          runtime note from _make_cache (compression fallback msgs)
# commit_tokens: full prompt token list to re-key after generation (None when
#                not managing a native cache)
# fields_cache:  value to feed _runtime_fields (None for native, the
#                compression cache otherwise) so the strategy badge stays right
# managed:       True only when we own a native persistent cache to commit
Acquired = namedtuple(
    "Acquired", "cache prompt_feed note commit_tokens fields_cache managed"
)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _native_result(cache: Any | None, full_tokens: list[int], prompt_text: str, note: str | None) -> Acquired:
    """Wrap a fresh-native-cache outcome (or a give-up fallback)."""
    if cache is not None:
        return Acquired(cache, full_tokens, note, full_tokens, None, True)
    # Couldn't build a managed cache → behave exactly like before.
    return Acquired(None, prompt_text, note, None, None, False)


def acquire(state: Any, prompt_text: str) -> Acquired:
    base_cache, note = state._make_cache()
    if base_cache is not None:
        # Compression strategy: unchanged behaviour, no persistence.
        return Acquired(base_cache, prompt_text, note, None, base_cache, False)

    # Native strategy — manage a persistent single-slot cache.
    try:
        from mlx_lm.models.cache import (  # noqa: PLC0415
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )

        full_tokens = list(state.tokenizer.encode(prompt_text))
    except Exception:  # noqa: BLE001 — any failure → safe full-reprocess fallback
        return Acquired(None, prompt_text, note, None, None, False)

    def _fresh() -> Any | None:
        try:
            return make_prompt_cache(state.model)
        except Exception:  # noqa: BLE001
            return None

    model_ref = getattr(state, "_loaded_model_ref", None)
    persist = getattr(state, "_persist_cache", None)
    persist_tokens = getattr(state, "_persist_tokens", None) or []
    persist_ref = getattr(state, "_persist_cache_model_ref", None)

    # Reset conditions: nothing cached, different model, empty history.
    if persist is None or persist_ref != model_ref or not persist_tokens:
        return _native_result(_fresh(), full_tokens, prompt_text, note)

    try:
        if not can_trim_prompt_cache(persist):
            return _native_result(_fresh(), full_tokens, prompt_text, note)
        # Always leave >=1 token to process live (mlx-lm does the same).
        common = min(_common_prefix_len(persist_tokens, full_tokens), len(full_tokens) - 1)
        if common <= 0:
            return _native_result(_fresh(), full_tokens, prompt_text, note)
        num_to_trim = len(persist_tokens) - common
        if num_to_trim > 0:
            trimmed = trim_prompt_cache(persist, num_to_trim)
            if trimmed != num_to_trim:
                # Couldn't roll back cleanly — don't risk a spliced mismatch.
                return _native_result(_fresh(), full_tokens, prompt_text, note)
        # Reuse hit: cache now holds exactly the common prefix; prefill suffix.
        return Acquired(persist, full_tokens[common:], note, full_tokens, None, True)
    except Exception:  # noqa: BLE001
        return _native_result(_fresh(), full_tokens, prompt_text, note)


def commit(state: Any, *, cache: Any, commit_tokens: list[int] | None, generated_ids: list[int], model_ref: str | None) -> None:
    """Persist the cache keyed by prompt + generated tokens (positional truth)."""
    if cache is None or commit_tokens is None:
        return
    state._persist_cache = cache
    state._persist_tokens = list(commit_tokens) + [t for t in generated_ids if isinstance(t, int)]
    state._persist_cache_model_ref = model_ref


def invalidate(state: Any) -> None:
    state._persist_cache = None
    state._persist_tokens = []
    state._persist_cache_model_ref = None
