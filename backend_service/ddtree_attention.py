"""Tree-structured verification pass for DDTree on MLX.

Runs a pure-attention mlx-lm text model over a whole draft tree in one
forward pass. Every node is rotated at ``past_length + depth`` (its
position on its own root-to-node path, not its flat index in the batch)
and attends to the committed KV cache plus its own ancestors only, so the
logits at each node equal those of an unbatched forward over that node's
path. Afterwards the KV cache is compacted to the accepted path: accepted
nodes sit at consecutive depths, so their keys were already rotated for
the slots they move into and only need gathering.

The rotation is applied by temporarily swapping each attention layer's
``rope`` module for one that ignores the cache offset and uses per-node
positions; mlx-lm itself is not modified. Only models whose layers are
all plain attention over an unrotated, unquantized ``KVCache`` qualify —
recurrent layers (gated delta nets, Mamba) carry one sequential state and
cannot score sibling branches in a single pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

import numpy as np


def tree_verify_unsupported_reason(text_model: Any, cache: Sequence[Any]) -> Optional[str]:
    """Return why this model/cache cannot run tree verification, or None."""
    from mlx_lm.models.cache import KVCache

    layers = list(getattr(text_model, "layers", None) or [])
    if not layers:
        return "the model exposes no decoder layers"
    if len(cache) != len(layers):
        return "the cache does not have one entry per layer"
    for layer in layers:
        if getattr(layer, "is_linear", False) or hasattr(layer, "linear_attn"):
            return "the model has recurrent (linear-attention) layers"
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "rope", None) is None:
            return "a layer has no rotary self-attention"
    for entry in cache:
        if type(entry) is not KVCache:
            return f"the cache uses {type(entry).__name__}, not a plain KVCache"
    return None


def tree_attention_mask(mx: Any, visibility: np.ndarray, past_length: int) -> Any:
    """Boolean attention mask ``(1, 1, T, past_length + T)``; True = attend.

    Every tree row sees the whole committed prefix; inside the tree block
    a row sees itself and its ancestors (``visibility``).
    """
    tree_len = int(visibility.shape[0])
    block = mx.array(np.ascontiguousarray(visibility, dtype=np.bool_))
    if past_length > 0:
        prefix = mx.ones((tree_len, int(past_length)), dtype=mx.bool_)
        block = mx.concatenate([prefix, block], axis=1)
    return block[None, None]


def tree_positions(past_length: int, node_depths: Sequence[int]) -> np.ndarray:
    """RoPE positions for ``[root, *nodes]``: root at ``past_length``, a node at
    ``past_length + depth``."""
    positions = np.empty(1 + len(node_depths), dtype=np.int32)
    positions[0] = past_length
    if len(node_depths):
        positions[1:] = np.asarray(node_depths, dtype=np.int64) + past_length
    return positions


def _make_tree_rope(mx: Any, nn: Any, rope: Any, positions: Any, past_length: int) -> Any:
    class _TreePositionRope(nn.Module):
        """Applies the wrapped rope at one position per token.

        mlx's rope takes a per-batch offset vector, so the tokens are moved
        onto the batch axis, rotated, and moved back.
        """

        def __init__(self) -> None:
            super().__init__()
            self.inner = rope
            self.positions = positions
            self.calls = 0

        def __call__(self, x: Any, offset: Any = 0) -> Any:
            if not isinstance(offset, int) or offset != past_length:
                raise ValueError(
                    f"tree verify expected the cache at offset {past_length}, got {offset!r}"
                )
            if x.ndim != 4 or int(x.shape[0]) != 1 or int(x.shape[2]) != int(self.positions.shape[0]):
                raise ValueError(f"tree verify cannot rotate an input of shape {tuple(x.shape)}")
            self.calls += 1
            rotated = self.inner(x.transpose(2, 1, 0, 3), offset=self.positions)
            return rotated.transpose(2, 1, 0, 3)

    return _TreePositionRope()


@contextmanager
def _tree_rope_swapped(layers: Sequence[Any], positions: np.ndarray, past_length: int) -> Iterator[list[Any]]:
    import mlx.core as mx
    import mlx.nn as nn

    positions_mx = mx.array(positions, dtype=mx.int32)
    swapped: list[tuple[Any, Any]] = []
    wrappers: list[Any] = []
    try:
        for layer in layers:
            attn = layer.self_attn
            original = attn.rope
            wrapper = _make_tree_rope(mx, nn, original, positions_mx, past_length)
            attn.rope = wrapper
            swapped.append((attn, original))
            wrappers.append(wrapper)
        yield wrappers
    finally:
        for attn, original in swapped:
            attn.rope = original


def tree_forward(
    *,
    text_model: Any,
    logits_fn: Any,
    cache: Sequence[Any],
    input_ids: Any,
    positions: np.ndarray,
    mask: Any,
    capture_layer_ids: set[int],
) -> tuple[Any, dict[int, Any]]:
    """One target forward over ``[root, *nodes]`` with tree positions and mask.

    Appends all ``T`` tree entries to every layer's cache (at slots
    ``past_length .. past_length + T - 1``); call ``compact_kv_cache``
    afterwards. Returns ``(logits (1, T, V), {layer_id: hidden (1, T, H)})``
    where layer id ``i + 1`` is the output of decoder layer ``i`` and ``0``
    the embeddings.
    """
    past_length = int(cache[0].offset)
    hidden = text_model.embed_tokens(input_ids)
    captured: dict[int, Any] = {0: hidden} if 0 in capture_layer_ids else {}
    layers = list(text_model.layers)
    with _tree_rope_swapped(layers, positions, past_length) as wrappers:
        for index, (layer, layer_cache) in enumerate(zip(layers, cache)):
            hidden = layer(hidden, mask, layer_cache)
            if index + 1 in capture_layer_ids:
                captured[index + 1] = hidden
    # Every attention layer must rotate its queries and its keys through the
    # tree positions; anything else means the layer bypassed ``self.rope``.
    if any(wrapper.calls != 2 for wrapper in wrappers):
        raise RuntimeError("an attention layer did not apply its rope through self.rope")
    logits = logits_fn(text_model.norm(hidden))
    return logits, captured


def compact_kv_cache(cache: Sequence[Any], past_length: int, accepted_indices: Sequence[int]) -> None:
    """Keep only the accepted tree entries, at consecutive slots after the prefix.

    ``accepted_indices`` are flat tree indices of the accepted path (root
    first). Their depths are ``0 .. n - 1``, so each kept key was rotated
    for exactly the slot it lands in.
    """
    import mlx.core as mx

    count = len(accepted_indices)
    is_prefix = list(accepted_indices) == list(range(count))
    if not is_prefix:
        keep = mx.array(list(accepted_indices), dtype=mx.int32)
        gathered = []
        for entry in cache:
            tree_keys = entry.keys[..., past_length:entry.offset, :]
            tree_values = entry.values[..., past_length:entry.offset, :]
            gathered.append((mx.take(tree_keys, keep, axis=2), mx.take(tree_values, keep, axis=2)))
        # Evaluate the gathers before writing so the writes below can reuse
        # the cache buffers instead of copying them.
        mx.eval(*[array for pair in gathered for array in pair])
        for entry, (kept_keys, kept_values) in zip(cache, gathered):
            entry.keys[..., past_length:past_length + count, :] = kept_keys
            entry.values[..., past_length:past_length + count, :] = kept_values
    for entry in cache:
        entry.offset = past_length + count


def prefill_with_capture(
    *,
    text_model: Any,
    logits_fn: Any,
    cache: Sequence[Any],
    input_ids: Any,
    capture_layer_ids: set[int],
    step: int = 2048,
) -> tuple[Any, dict[int, Any]]:
    """Causal prefill in chunks. Returns the last position's logits ``(1, V)``
    and the captured hidden states for every prompt position."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    total = int(input_ids.shape[1])
    pieces: dict[int, list[Any]] = {layer_id: [] for layer_id in capture_layer_ids}
    layers = list(text_model.layers)
    hidden = None
    for chunk_start in range(0, total, max(1, int(step))):
        chunk = input_ids[:, chunk_start:chunk_start + step]
        hidden = text_model.embed_tokens(chunk)
        if 0 in pieces:
            pieces[0].append(hidden)
        mask = create_attention_mask(hidden, cache[0])
        for index, (layer, layer_cache) in enumerate(zip(layers, cache)):
            hidden = layer(hidden, mask, layer_cache)
            if index + 1 in pieces:
                pieces[index + 1].append(hidden)
        if chunk_start + step < total:
            mx.eval([entry.keys for entry in cache] + [entry.values for entry in cache])
            mx.eval(*[parts[-1] for parts in pieces.values() if parts])
    logits = logits_fn(text_model.norm(hidden[:, -1:, :]))[:, -1, :]
    captured = {
        layer_id: parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        for layer_id, parts in pieces.items()
    }
    return logits, captured
