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
changed, cache not trimmable and not snapshot-restorable (rotating-full,
mlx-lm #980),
tokenisation failure, no common prefix, partial trim — falls back to a
fresh full prefill, i.e. identical output to the pre-cache path, just
without the speedup. Gated to the ``native`` strategy; compression caches
(turboquant / triattention) keep their existing per-call path untouched.

Hybrid models (FU-090) — Qwen3.5 / 3.6 / 3.8, Qwen3-Next / Coder-Next,
Nemotron-H, Mamba — mix attention layers (``KVCache``, trimmable) with
recurrent GDN / SSM layers (``ArraysCache``, not trimmable: the state
after N tokens can't be rewound to N-k). mlx-lm's ``can_trim_prompt_cache``
is False for them, so every turn used to re-prefill the whole history.
For these caches we prefill all but the last prompt token ourselves,
snapshot the recurrent layers' state at that boundary, then let
``stream_generate`` process the final token. Next turn, when the common
prefix covers the boundary (the previous prompt is a prefix of the new
one — the normal chat case), attention layers trim back to it, recurrent
layers restore the snapshot, and only the new suffix is prefilled. The
snapshot is a handful of small per-layer arrays, not a second KV cache.
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


def _native_result(state: Any, cache: Any | None, full_tokens: list[int], prompt_text: str, note: str | None) -> Acquired:
    """Wrap a fresh-native-cache outcome (or a give-up fallback)."""
    if cache is not None:
        return _checkpointed(state, cache, full_tokens, full_tokens, note)
    # Couldn't build a managed cache → behave exactly like before.
    return Acquired(None, prompt_text, note, None, None, False)


_PREFILL_STEP = 2048  # mlx-lm's generate_step default


def _is_recurrent_layer(layer: Any) -> bool:
    try:
        from mlx_lm.models.cache import ArraysCache  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    return isinstance(layer, ArraysCache)


def _snapshot_capable(cache: Any) -> bool:
    """Hybrid/recurrent cache we can checkpoint: every layer is either an
    ``ArraysCache`` (snapshot) or trimmable (roll back), and at least one
    is recurrent (pure-attention caches use the plain trim path)."""
    if not isinstance(cache, list) or not cache:
        return False
    has_recurrent = False
    for layer in cache:
        if _is_recurrent_layer(layer):
            has_recurrent = True
            continue
        try:
            if not layer.is_trimmable():
                return False
        except Exception:  # noqa: BLE001
            return False
    return has_recurrent


def _snapshot(cache: list[Any]) -> list[list[Any] | None]:
    """Copy recurrent layers' state (``None`` for attention layers)."""
    import mlx.core as mx  # noqa: PLC0415

    snap: list[list[Any] | None] = []
    for layer in cache:
        if _is_recurrent_layer(layer):
            # ``mx.array(a)`` copies, so later in-place updates to the live
            # state can't leak into the checkpoint.
            snap.append([None if a is None else mx.array(a) for a in layer.state])
        else:
            snap.append(None)
    mx.eval([a for layer in snap if layer for a in layer if a is not None])
    return snap


def _restore(cache: list[Any], snap: list[list[Any] | None], boundary: int, persist_len: int) -> bool:
    """Roll ``cache`` back to ``boundary`` tokens. False → caller refills."""
    if len(snap) != len(cache):
        return False
    for layer, saved in zip(cache, snap):
        if saved is not None:
            if not _is_recurrent_layer(layer):
                return False
            layer.state = list(saved)
            continue
        offset = getattr(layer, "offset", None)
        current = offset if isinstance(offset, int) else persist_len
        excess = current - boundary
        if excess < 0:
            return False
        if excess and layer.trim(excess) != excess:
            return False
    return True


def _prefill(model: Any, cache: list[Any], tokens: list[int]) -> None:
    """Chunked prefill, as mlx-lm's generate_step does it."""
    import mlx.core as mx  # noqa: PLC0415

    for start in range(0, len(tokens), _PREFILL_STEP):
        chunk = mx.array(tokens[start:start + _PREFILL_STEP])
        model(chunk[None], cache=cache)
        mx.eval([layer.state for layer in cache])
    mx.clear_cache()


def _checkpointed(state: Any, cache: Any, full_tokens: list[int], feed: list[int], note: str | None) -> Acquired:
    """For hybrid caches: prefill ``feed[:-1]`` here, checkpoint the
    recurrent state at ``len(full_tokens) - 1``, hand the last token to
    stream_generate. Pure-attention caches pass straight through."""
    state._persist_snapshot_pending = None
    if len(feed) > 1 and _snapshot_capable(cache):
        _prefill(state.model, cache, feed[:-1])
        state._persist_snapshot_pending = (len(full_tokens) - 1, _snapshot(cache))
        feed = feed[-1:]
    return Acquired(cache, feed, note, full_tokens, None, True)


def acquire(state: Any, prompt_text: str) -> Acquired:
    try:
        return _acquire(state, prompt_text)
    except Exception:  # noqa: BLE001 — a failed hybrid prefill must not break the turn
        invalidate(state)
        base_cache, note = state._make_cache()
        if base_cache is not None:
            return Acquired(base_cache, prompt_text, note, None, base_cache, False)
        return Acquired(None, prompt_text, note, None, None, False)


def _acquire(state: Any, prompt_text: str) -> Acquired:
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
        return _native_result(state, _fresh(), full_tokens, prompt_text, note)

    try:
        # Always leave >=1 token to process live (mlx-lm does the same).
        common = min(_common_prefix_len(persist_tokens, full_tokens), len(full_tokens) - 1)
        if common <= 0:
            return _native_result(state, _fresh(), full_tokens, prompt_text, note)
        if not can_trim_prompt_cache(persist):
            # FU-090: hybrid cache — restore the recurrent checkpoint when
            # the shared prefix reaches it.
            snapshot = getattr(state, "_persist_snapshot", None)
            if snapshot is None or not _snapshot_capable(persist):
                return _native_result(state, _fresh(), full_tokens, prompt_text, note)
            boundary, snap = snapshot
            if boundary <= 0 or boundary > common:
                return _native_result(state, _fresh(), full_tokens, prompt_text, note)
            if not _restore(persist, snap, boundary, len(persist_tokens)):
                return _native_result(state, _fresh(), full_tokens, prompt_text, note)
            return _checkpointed(state, persist, full_tokens, full_tokens[boundary:], note)
        num_to_trim = len(persist_tokens) - common
        if num_to_trim > 0:
            trimmed = trim_prompt_cache(persist, num_to_trim)
            if trimmed != num_to_trim:
                # Couldn't roll back cleanly — don't risk a spliced mismatch.
                return _native_result(state, _fresh(), full_tokens, prompt_text, note)
        # Reuse hit: cache now holds exactly the common prefix; prefill suffix.
        return _checkpointed(state, persist, full_tokens, full_tokens[common:], note)
    except Exception:  # noqa: BLE001
        return _native_result(state, _fresh(), full_tokens, prompt_text, note)


def commit(state: Any, *, cache: Any, commit_tokens: list[int] | None, generated_ids: list[int], model_ref: str | None) -> None:
    """Persist the cache keyed by prompt + generated tokens (positional truth)."""
    if cache is None or commit_tokens is None:
        return
    state._persist_cache = cache
    state._persist_tokens = list(commit_tokens) + [t for t in generated_ids if isinstance(t, int)]
    state._persist_cache_model_ref = model_ref
    state._persist_snapshot = getattr(state, "_persist_snapshot_pending", None)
    state._persist_snapshot_pending = None


def invalidate(state: Any) -> None:
    state._persist_cache = None
    state._persist_tokens = []
    state._persist_cache_model_ref = None
    state._persist_snapshot = None
    state._persist_snapshot_pending = None
