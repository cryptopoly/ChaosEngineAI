"""DDTree — Diffusion Draft Tree speculative decoding for MLX.

Port of the tree-building algorithm from
https://github.com/liranringel/ddtree adapted for ChaosEngineAI's MLX
runtime.  Reuses the same DFlash draft model — the innovation is in
how draft tokens are expanded into a *tree* of candidates and verified
in a single forward pass with a tree-structured attention mask.

When ``tree_budget=0`` the module falls back to plain linear DFlash
(identical to the existing ``_generate_dflash`` path).
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Optional

import numpy as np


# ======================================================================
# 1.  Tree building  (framework-agnostic — pure Python / NumPy)
# ======================================================================

def build_ddtree_tree(
    draft_logits_np: np.ndarray,
    budget: int,
) -> tuple[
    np.ndarray,       # node_token_ids  (N,)
    np.ndarray,       # node_depths     (N,)
    list[int],         # parents         (1+N,)
    list[dict[int, int]],  # child_maps (1+N dicts)
    np.ndarray,        # visibility      (1+N, 1+N) bool
]:
    """Build a draft tree from top-k logits using a max-probability heap.

    Parameters
    ----------
    draft_logits_np : (depth, vocab) float32 NumPy array
        Raw draft logits for each future position.
    budget : int
        Maximum number of tree nodes (excluding root).

    Returns
    -------
    node_token_ids, node_depths, parents, child_maps, visibility
    """
    if budget <= 0 or draft_logits_np.shape[0] == 0:
        vis = np.zeros((1, 1), dtype=np.bool_)
        vis[0, 0] = True
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            [-1],
            [dict()],
            vis,
        )

    topk = min(budget, draft_logits_np.shape[-1])
    depth_limit = int(draft_logits_np.shape[0])

    logits = draft_logits_np.astype(np.float32)
    log_z = np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) + logits.max(axis=-1, keepdims=True)

    # top-k per position
    top_indices = np.argpartition(logits, -topk, axis=-1)[:, -topk:]
    # sort within top-k by descending value
    row_idx = np.arange(logits.shape[0])[:, None]
    top_vals = logits[row_idx, top_indices]
    sort_order = np.argsort(-top_vals, axis=-1)
    top_indices = np.take_along_axis(top_indices, sort_order, axis=-1)
    top_vals = np.take_along_axis(top_vals, sort_order, axis=-1)
    top_log_probs = top_vals - log_z

    # Heap-based tree expansion
    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw)
    ]

    node_token_ids = np.empty(budget, dtype=np.int64)
    node_depths = np.empty(budget, dtype=np.int64)
    parents_arr = np.empty(budget + 1, dtype=np.int32)
    parents_arr[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_indices[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids[node_count] = token_id
        node_depths[node_count] = depth
        parents_arr[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        # Sibling (same depth, next rank)
        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs[depth - 1, rank]) + float(top_log_probs[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        # Child (deeper)
        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    # Build visibility matrix (ancestor mask)
    current_length = 1 + node_count
    visibility = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility[0, 0] = True
    for index in range(1, current_length):
        parent = int(parents_arr[index])
        visibility[index, :index] = visibility[parent, :index]
        visibility[index, index] = True

    return (
        node_token_ids[:node_count],
        node_depths[:node_count],
        parents_arr[:current_length].tolist(),
        child_maps,
        visibility,
    )


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior_tokens: list[int],
) -> tuple[list[int], int]:
    """Walk the verified tree to find the longest accepted path.

    Returns (accepted_node_indices, next_token_id).
    """
    accepted = [0]
    current = 0
    next_token = posterior_tokens[current]

    while next_token in child_maps[current]:
        current = child_maps[current][next_token]
        accepted.append(current)
        if current < len(posterior_tokens):
            next_token = posterior_tokens[current]
        else:
            break

    return accepted, next_token


# ======================================================================
# 2.  MLX tree compilation & verification
# ======================================================================

def compile_ddtree_tree_mlx(
    mx: Any,
    root_token_id: int,
    start: int,
    node_token_ids: np.ndarray,
    node_depths: np.ndarray,
    visibility: np.ndarray,
    past_length: int,
) -> tuple[Any, Any, Any]:
    """Compile a draft tree into MLX tensors for batched verification.

    Returns (input_ids, position_ids, tree_mask) as MLX arrays.
    ``tree_mask`` is an additive float mask: 0 where attention is allowed,
    -inf where blocked — matching MLX's scaled_dot_product_attention convention.
    """
    current_length = 1 + len(node_token_ids)

    # input_ids: [root, node_0, node_1, ...]
    ids = np.empty(current_length, dtype=np.int32)
    ids[0] = root_token_id
    if len(node_token_ids) > 0:
        ids[1:] = node_token_ids
    input_ids = mx.array(ids)[None]  # (1, tree_len)

    # position_ids: root=start, children=start+depth
    pos = np.empty(current_length, dtype=np.int32)
    pos[0] = start
    if len(node_depths) > 0:
        pos[1:] = node_depths + start
    position_ids = mx.array(pos)[None]  # (1, tree_len)

    # Attention mask: (1, 1, tree_len, past_length + tree_len)
    # Causal prefix (attend to all past) + tree visibility block
    total_seq = past_length + current_length
    mask_np = np.zeros((current_length, total_seq), dtype=np.float32)
    # Block attention to tree positions based on visibility
    tree_block = np.where(visibility, 0.0, -1e9)
    mask_np[:, past_length:past_length + current_length] = tree_block
    tree_mask = mx.array(mask_np)[None, None]  # (1, 1, tree_len, total_seq)

    return input_ids, position_ids, tree_mask


def compact_cache_entries(
    cache_entries: list[Any],
    past_length: int,
    accepted_indices: list[int],
    trim_fn: Any,
) -> None:
    """Compact the KV cache after tree verification.

    Keeps only the entries corresponding to accepted tree nodes, then trims
    the cache to ``past_length + len(accepted_indices)``.
    """
    import mlx.core as mx

    if not accepted_indices:
        trim_fn(cache_entries, past_length)
        return

    keep = mx.array(accepted_indices, dtype=mx.int32)
    for entry in cache_entries:
        if entry is None:
            continue
        # cache entries in mlx-lm / dflash_mlx are typically objects with
        # .keys and .values attributes (or key_cache/value_cache).
        for attr_name in ("keys", "key_cache"):
            tensor = getattr(entry, attr_name, None)
            if tensor is None or not hasattr(tensor, "shape"):
                continue
            seq_dim = len(tensor.shape) - 2  # (..., seq, head_dim)
            if tensor.shape[seq_dim] <= past_length:
                continue
            # Extract the appended tree window and keep only accepted nodes
            appended = tensor[..., past_length:, :]
            kept = mx.take(appended, keep, axis=seq_dim)
            # Write back: prefix + kept
            prefix = tensor[..., :past_length, :]
            new_tensor = mx.concatenate([prefix, kept], axis=seq_dim)
            setattr(entry, attr_name, new_tensor)

        for attr_name in ("values", "value_cache"):
            tensor = getattr(entry, attr_name, None)
            if tensor is None or not hasattr(tensor, "shape"):
                continue
            seq_dim = len(tensor.shape) - 2
            if tensor.shape[seq_dim] <= past_length:
                continue
            appended = tensor[..., past_length:, :]
            kept = mx.take(appended, keep, axis=seq_dim)
            prefix = tensor[..., :past_length, :]
            new_tensor = mx.concatenate([prefix, kept], axis=seq_dim)
            setattr(entry, attr_name, new_tensor)

    trim_fn(cache_entries, past_length + len(accepted_indices))


# ======================================================================
# 3.  Full DDTree generation loop (MLX)
# ======================================================================

def generate_ddtree_mlx(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: Any,
    prompt_tokens: list[int],
    max_new_tokens: int,
    tree_budget: int,
    block_tokens: int = 16,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    """DDTree generation loop using dflash_mlx primitives.

    Falls back to linear DFlash when tree_budget <= 0.
    """
    import mlx.core as mx
    from dflash_mlx.runtime import (
        target_forward_with_hidden_states,
        extract_context_feature_from_dict,
        make_target_cache,
        ContextOnlyDraftKVCache,
        greedy_tokens_with_mask,
        build_suppress_token_mask,
        trim_cache_to,
    )

    # Private helpers from dflash_mlx
    from dflash_mlx.runtime import (
        _target_embed_tokens,
        _lm_head_logits,
        _target_text_model,
    )

    prompt_len = len(prompt_tokens)
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_ids = list(stop_token_ids or [])
    stop_set = set(stop_ids)

    effective_block = max(1, min(int(block_tokens), int(draft_model.block_size)))
    draft_horizon = effective_block - 1
    effective_budget = max(0, min(tree_budget, 64))

    # Caches
    target_cache = make_target_cache(target_model, enable_speculative_linear_cache=False)
    draft_cache = [
        ContextOnlyDraftKVCache(sink_size=0, window_size=0)
        for _ in range(len(draft_model.layers))
    ]
    capture_layer_ids = {int(lid) + 1 for lid in draft_model.target_layer_ids}

    suppress_mask = build_suppress_token_mask(
        tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 151936,
        suppress_token_ids,
    )

    # ── Prefill ──────────────────────────────────────────────
    t_start = time.perf_counter()
    prefill_logits, prefill_hidden = target_forward_with_hidden_states(
        target_model, input_ids=prompt_array, cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )
    mx.eval(prefill_logits)
    if isinstance(prefill_hidden, dict):
        mx.eval(*prefill_hidden.values())
    else:
        mx.eval(*prefill_hidden)

    first_token = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_mask).reshape(-1)
    target_hidden = extract_context_feature_from_dict(
        prefill_hidden, list(draft_model.target_layer_ids),
    )
    mx.eval(first_token, target_hidden)

    generated_tokens: list[int] = [int(first_token.item())]
    start = prompt_len
    cycles = 0
    accepted_from_draft = 0
    acceptance_history: list[int] = []

    embed_fn = _target_embed_tokens(target_model)
    inner = _target_text_model(target_model)

    # ── Decode loop ──────────────────────────────────────────
    while len(generated_tokens) < max_new_tokens:
        remaining = max_new_tokens - len(generated_tokens)
        block_len = max(1, min(effective_block, remaining + 1))

        # Build the noise block: [last_committed_token, mask, mask, ...]
        block_ids_np = np.full(block_len, draft_model.mask_token_id, dtype=np.int32)
        block_ids_np[0] = generated_tokens[-1]
        block_ids = mx.array(block_ids_np, dtype=mx.uint32)[None]

        # ── Draft ────────────────────────────────────────────
        if block_len > 1:
            noise_embedding = embed_fn(block_ids)
            draft_hidden = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
            mx.eval(draft_logits)
        else:
            draft_logits = None

        # Trim draft cache back to committed length
        trim_cache_to(draft_cache, start)

        if draft_logits is None or effective_budget <= 0:
            # Linear DFlash fallback: greedy single path
            if draft_logits is not None:
                drafted = greedy_tokens_with_mask(draft_logits, suppress_mask).squeeze(0)
                block_ids_np[1:block_len] = np.array(drafted.tolist(), dtype=np.int32)[:block_len - 1]
                block_ids = mx.array(block_ids_np, dtype=mx.uint32)[None]

            verify_logits, verify_hidden = target_forward_with_hidden_states(
                target_model, input_ids=block_ids[:, :block_len],
                cache=target_cache, capture_layer_ids=capture_layer_ids,
            )
            mx.eval(verify_logits)

            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_mask)
            mx.eval(posterior)

            # Linear acceptance
            acceptance_len = 0
            for i in range(1, block_len):
                if int(posterior[i - 1].item()) == block_ids_np[i]:
                    acceptance_len += 1
                else:
                    break

            commit_count = 1 + acceptance_len
            committed = [int(block_ids_np[i]) for i in range(1, commit_count)]
            next_tok = int(posterior[min(acceptance_len, block_len - 1)].item())
            committed.append(next_tok)

            generated_tokens.extend(committed)
            accepted_from_draft += acceptance_len
            acceptance_history.append(acceptance_len)
            start += commit_count

            committed_hidden = extract_context_feature_from_dict(
                verify_hidden, list(draft_model.target_layer_ids),
            )[:, :commit_count, :]
            mx.eval(committed_hidden)
            target_hidden = committed_hidden

            trim_cache_to(target_cache, start)
            cycles += 1
        else:
            # ── DDTree path ──────────────────────────────────
            draft_logits_np = np.array(draft_logits[0].tolist(), dtype=np.float32)

            node_token_ids, node_depths, parents, child_maps, visibility = \
                build_ddtree_tree(draft_logits_np, effective_budget)

            tree_input_ids, tree_position_ids, tree_mask = compile_ddtree_tree_mlx(
                mx,
                root_token_id=generated_tokens[-1],
                start=start,
                node_token_ids=node_token_ids,
                node_depths=node_depths,
                visibility=visibility,
                past_length=start,
            )

            # Tree verification: run target model with tree attention mask
            tree_len = 1 + len(node_token_ids)
            tree_embeddings = inner.embed_tokens(tree_input_ids)

            # Build full attention mask: causal for prefix, tree for new tokens
            h = tree_embeddings
            captured_hidden: dict[int, Any] = {}
            if 0 in capture_layer_ids:
                captured_hidden[0] = h

            # Get the cache's current prefix length for mask construction
            from dflash_mlx.runtime import create_attention_mask
            causal_mask = create_attention_mask(h, target_cache[0] if target_cache else None)

            # Replace the tree portion of the causal mask with our tree mask
            if causal_mask is not None:
                # causal_mask shape: (1, 1, tree_len, past + tree_len)
                # We need to override the tree-to-tree block with visibility
                vis_block = mx.where(
                    mx.array(visibility)[None, None],
                    mx.zeros_like(tree_mask[:, :, :tree_len, -tree_len:]),
                    mx.array(-1e9),
                )
                # Build combined mask: keep causal for prefix, use tree for tree block
                combined = causal_mask.astype(mx.float32)
                # Zero out the tree-to-tree portion and replace with visibility
                prefix_len = combined.shape[-1] - tree_len
                combined_np = np.array(combined.tolist(), dtype=np.float32)
                vis_np = np.where(visibility, 0.0, -1e9).astype(np.float32)
                combined_np[0, 0, :tree_len, prefix_len:prefix_len + tree_len] = vis_np
                full_mask = mx.array(combined_np)
            else:
                full_mask = tree_mask

            # Run through target model layers with tree mask
            for i, layer in enumerate(inner.layers):
                h = layer(h, mask=full_mask, cache=target_cache[i])
                layer_id = i + 1
                if layer_id in capture_layer_ids:
                    captured_hidden[layer_id] = h

            h = inner.norm(h)
            verify_logits = target_model.lm_head(h) if hasattr(target_model, "lm_head") else inner.lm_head(h) if hasattr(inner, "lm_head") else h
            mx.eval(verify_logits)

            # Sample from verified logits
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_mask)
            mx.eval(posterior)
            posterior_list = [int(t.item()) for t in posterior]

            # Walk the tree to find accepted path
            accepted_indices, next_tok = follow_verified_tree(child_maps, posterior_list)
            acceptance_len = len(accepted_indices) - 1  # exclude root
            acceptance_history.append(acceptance_len)
            accepted_from_draft += acceptance_len

            # Extract committed tokens
            tree_ids_list = [generated_tokens[-1]] + [int(node_token_ids[i]) for i in range(len(node_token_ids))]
            committed = [tree_ids_list[idx] for idx in accepted_indices[1:]]  # skip root
            committed.append(next_tok)
            generated_tokens.extend(committed)
            start += len(accepted_indices)

            # Compact cache: keep only accepted nodes
            compact_cache_entries(target_cache, start - len(accepted_indices), accepted_indices, trim_cache_to)

            # Extract hidden states for accepted nodes
            accepted_mx = mx.array(accepted_indices, dtype=mx.int32)
            committed_hidden = extract_context_feature_from_dict(
                captured_hidden, list(draft_model.target_layer_ids),
            )
            committed_hidden = mx.take(committed_hidden, accepted_mx, axis=1)
            mx.eval(committed_hidden)
            target_hidden = committed_hidden

            cycles += 1

        # Stop token check
        if stop_set:
            for tok in generated_tokens[-(len(committed) if 'committed' in dir() else 1):]:
                if tok in stop_set:
                    # Truncate at stop token
                    for si, st in enumerate(generated_tokens):
                        if st in stop_set:
                            generated_tokens = generated_tokens[:si + 1]
                            break
                    break

        if stop_set and any(t in stop_set for t in generated_tokens[-10:]):
            break

    elapsed = time.perf_counter() - t_start
    output_tokens = len(generated_tokens)
    avg_acceptance = float(np.mean(acceptance_history)) if acceptance_history else 0.0

    return {
        "generated_tokens": generated_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "cycles": cycles,
        "accepted_from_draft": accepted_from_draft,
        "avg_acceptance_length": avg_acceptance,
        "tree_budget": effective_budget,
    }
