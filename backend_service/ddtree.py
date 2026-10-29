"""DDTree — Diffusion Draft Tree speculative decoding for MLX.

Port of the tree-building algorithm from
https://github.com/liranringel/ddtree adapted for ChaosEngineAI's MLX
runtime.  Reuses the same DFlash draft model — the innovation is in
how draft tokens are expanded into a *tree* of candidates and verified
in a single forward pass with a tree-structured attention mask (see
``backend_service/ddtree_attention.py``).

Verification is exact: with greedy decoding the output equals plain
greedy decoding of the target. ``tree_budget <= 0`` verifies the draft's
top-1 chain, which is linear DFlash on the same machinery.
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Optional

import numpy as np

from backend_service.ddtree_attention import (
    compact_kv_cache,
    prefill_with_capture,
    tree_attention_mask,
    tree_forward,
    tree_positions,
    tree_verify_unsupported_reason,
)

# Tokens verified per step, root included (the launch slider's maximum).
MAX_TREE_BUDGET = 64


# ======================================================================
# 1.  Tree building  (framework-agnostic — pure Python / NumPy)
# ======================================================================

def build_ddtree_tree_from_topk(
    top_indices: np.ndarray,
    top_log_probs: np.ndarray,
    budget: int,
) -> tuple[
    np.ndarray,       # node_token_ids  (N,)
    np.ndarray,       # node_depths     (N,)
    list[int],         # parents         (1+N,)
    list[dict[int, int]],  # child_maps (1+N dicts)
    np.ndarray,        # visibility      (1+N, 1+N) bool
]:
    """Build a draft tree with a max-probability heap.

    ``top_indices`` / ``top_log_probs`` are ``(depth, k)`` arrays holding the
    k most likely tokens per future position, sorted by descending log
    probability. A node's score is the sum of its path's log
    probabilities; the ``budget`` best-scoring nodes (excluding the root)
    are kept. Node order is pop order, so a parent always precedes its
    children.
    """
    if budget <= 0 or top_indices.shape[0] == 0 or top_indices.shape[1] == 0:
        vis = np.ones((1, 1), dtype=np.bool_)
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            [-1],
            [dict()],
            vis,
        )

    topk = int(top_indices.shape[1])
    depth_limit = int(top_indices.shape[0])

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


def build_ddtree_tree(
    draft_logits_np: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict[int, int]], np.ndarray]:
    """Build a draft tree from raw ``(depth, vocab)`` draft logits (NumPy)."""
    if budget <= 0 or draft_logits_np.shape[0] == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.float32), 0,
        )
    logits = draft_logits_np.astype(np.float32)
    row_max = logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.sum(np.exp(logits - row_max), axis=-1, keepdims=True)) + row_max
    topk = min(budget, logits.shape[-1])
    top_indices = np.argpartition(logits, -topk, axis=-1)[:, -topk:]
    row_idx = np.arange(logits.shape[0])[:, None]
    top_vals = logits[row_idx, top_indices]
    order = np.argsort(-top_vals, axis=-1, kind="stable")
    top_indices = np.take_along_axis(top_indices, order, axis=-1)
    top_log_probs = np.take_along_axis(top_vals, order, axis=-1) - log_z
    return build_ddtree_tree_from_topk(top_indices, top_log_probs, budget)


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior_tokens: list[int],
) -> tuple[list[int], int]:
    """Walk the verified tree to find the longest accepted path.

    ``posterior_tokens[i]`` is the target's greedy token after tree node
    ``i``. Returns (accepted_node_indices, next_token_id).
    """
    accepted = [0]
    current = 0
    next_token = posterior_tokens[current]

    while next_token in child_maps[current]:
        current = child_maps[current][next_token]
        accepted.append(current)
        next_token = posterior_tokens[current]

    return accepted, next_token


# ======================================================================
# 2.  Full DDTree generation loop (MLX)
# ======================================================================

def ddtree_unsupported_reason(target_model: Any, target_ops: Any = None) -> Optional[str]:
    """Why ``target_model`` cannot run DDTree on MLX, or None when it can."""
    try:
        if target_ops is None:
            from dflash_mlx.runtime import resolve_target_ops

            target_ops = resolve_target_ops(target_model)
        cache = target_ops.make_cache(target_model, enable_speculative_linear_cache=False)
        return tree_verify_unsupported_reason(target_ops.text_model(target_model), cache)
    except Exception as exc:
        return f"tree verification could not be prepared: {exc}"


def _draft_top_k(mx: Any, draft_logits: Any, k: int, suppress_mask: Any) -> tuple[np.ndarray, np.ndarray]:
    """Top-k tokens and log probabilities per draft position, sorted.

    Runs on the GPU so only ``(depth, k)`` values cross to the host.
    """
    logits = draft_logits.astype(mx.float32)
    if suppress_mask is not None:
        logits = mx.where(suppress_mask, mx.array(-np.inf, dtype=mx.float32), logits)
    log_z = mx.logsumexp(logits, axis=-1, keepdims=True)
    if k == 1:
        top = mx.argmax(logits, axis=-1, keepdims=True)
    else:
        top = mx.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
    top_log_probs = mx.take_along_axis(logits, top, axis=-1) - log_z
    mx.eval(top, top_log_probs)
    ids = np.array(top).astype(np.int64)
    log_probs = np.array(top_log_probs)
    if k > 1:
        order = np.argsort(-log_probs, axis=-1, kind="stable")
        ids = np.take_along_axis(ids, order, axis=-1)
        log_probs = np.take_along_axis(log_probs, order, axis=-1)
    return ids, log_probs


def _make_draft_cache(draft_model: Any, context_len: int) -> list[Any]:
    """The draft cache upstream's own decode loop uses for this drafter."""
    try:
        from dflash_mlx.draft_backend import make_draft_backend
        from dflash_mlx.engine.config import resolve_draft_window
    except ImportError:
        from dflash_mlx.model import ContextOnlyDraftKVCache

        return [
            ContextOnlyDraftKVCache(sink_size=64, window_size=1024)
            for _ in range(len(draft_model.layers))
        ]
    sink_size, window_size = resolve_draft_window(None, draft_model, context_len=context_len)
    return make_draft_backend().make_cache(
        draft_model=draft_model, sink_size=sink_size, window_size=window_size,
    )


def _accepted_spans(
    tokenizer: Any,
    tokens: list[int],
    accepted: list[bool],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Run-length encode per-token draft acceptance into character spans so
    the frontend overlay can tint draft-accepted ranges."""
    limit = min(len(tokens), len(accepted))
    if limit == 0:
        return [], None
    per_token_text: list[str] = []
    for tok_id in tokens[:limit]:
        try:
            per_token_text.append(tokenizer.decode([int(tok_id)]))
        except Exception:
            per_token_text.append("")
    spans: list[dict[str, Any]] = []
    offset = 0
    run_start = 0
    run_kind = accepted[0]
    for idx in range(limit):
        if accepted[idx] != run_kind:
            spans.append({"start": run_start, "length": offset - run_start, "accepted": run_kind})
            run_start = offset
            run_kind = accepted[idx]
        offset += len(per_token_text[idx])
    spans.append({"start": run_start, "length": offset - run_start, "accepted": run_kind})
    return spans, "".join(per_token_text)


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
    """Greedy DDTree generation with a DFlash draft model.

    Raises ``ValueError`` when the target cannot be tree-verified (hybrid
    recurrent layers, rotating or quantized caches); callers fall back to
    linear DFlash.
    """
    import mlx.core as mx
    # dflash-mlx 0.1.5+ exposes the per-architecture entry points (text
    # model, embeddings, lm head, cache factory, hidden-feature extraction)
    # through a per-family ``target_ops`` adapter.
    from dflash_mlx.runtime import resolve_target_ops

    target_ops = resolve_target_ops(target_model)
    bind_target = getattr(draft_model, "bind_target_model", None)
    if bind_target is not None:
        bind_target(target_model, target_ops=target_ops)
    text_model = target_ops.text_model(target_model)
    target_cache = target_ops.make_cache(target_model, enable_speculative_linear_cache=False)
    unsupported = tree_verify_unsupported_reason(text_model, target_cache)
    if unsupported:
        raise ValueError(f"DDTree tree verification needs a pure-attention target: {unsupported}")

    def logits_fn(hidden: Any) -> Any:
        return target_ops.logits_from_hidden(target_model, hidden)

    embed_fn = target_ops.embed_tokens(target_model)
    target_layer_ids = list(draft_model.target_layer_ids)
    capture_layer_ids = {int(lid) + 1 for lid in target_layer_ids}
    prompt_len = len(prompt_tokens)
    draft_cache = _make_draft_cache(draft_model, context_len=prompt_len + max(0, int(max_new_tokens)))
    stop_set = {int(t) for t in (stop_token_ids or [])}

    # The budget counts every token the target verifies per step, the root
    # included, so a budget of 32 is one 32-row verify pass.
    effective_budget = max(0, min(int(tree_budget), MAX_TREE_BUDGET))
    node_budget = max(1, effective_budget - 1) if effective_budget > 0 else 0
    effective_block = max(1, min(int(block_tokens), int(draft_model.block_size)))
    if node_budget > 0:
        # A tree of N nodes is at most N deep; drafting further is wasted work.
        effective_block = min(effective_block, node_budget + 1)
    mask_tail = mx.full((max(0, effective_block - 1),), int(draft_model.mask_token_id), dtype=mx.uint32)

    # ── Prefill ──────────────────────────────────────────────
    t_start = time.perf_counter()
    prefill_logits, prefill_hidden = prefill_with_capture(
        text_model=text_model,
        logits_fn=logits_fn,
        cache=target_cache,
        input_ids=mx.array(prompt_tokens, dtype=mx.uint32)[None],
        capture_layer_ids=capture_layer_ids,
    )
    vocab_size = int(prefill_logits.shape[-1])
    suppress_ids = sorted({int(t) for t in (suppress_token_ids or []) if 0 <= int(t) < vocab_size})
    suppress_mask = None
    if suppress_ids:
        suppress_np = np.zeros(vocab_size, dtype=np.bool_)
        suppress_np[suppress_ids] = True
        suppress_mask = mx.array(suppress_np)

    def greedy(logits: Any) -> Any:
        if suppress_mask is not None:
            logits = mx.where(suppress_mask, mx.array(-np.inf, dtype=logits.dtype), logits)
        return mx.argmax(logits, axis=-1)

    first_token = greedy(prefill_logits[0])
    target_hidden = target_ops.extract_context_feature(prefill_hidden, target_layer_ids)
    mx.eval(first_token, target_hidden)
    prefill_seconds = time.perf_counter() - t_start

    generated_tokens: list[int] = [int(first_token.item())]
    # Per-token accepted-from-draft flags for the accepted-span overlay. The
    # first token is the prefill posterior, so it is not draft-accepted.
    per_token_accepted: list[bool] = [False]
    start = prompt_len
    cycles = 0
    accepted_from_draft = 0
    acceptance_history: list[int] = []
    stopped = generated_tokens[0] in stop_set

    # ── Decode loop ──────────────────────────────────────────
    while not stopped and len(generated_tokens) < max_new_tokens:
        remaining = max_new_tokens - len(generated_tokens)
        # At most ``block_len - 1`` drafts are accepted plus one posterior
        # token, so a cycle never commits more than ``remaining``.
        block_len = max(1, min(effective_block, remaining))
        root_token = generated_tokens[-1]

        if block_len > 1:
            block_ids = mx.concatenate(
                [mx.array([root_token], dtype=mx.uint32), mask_tail[: block_len - 1]]
            )[None]
            draft_hidden = draft_model(
                noise_embedding=embed_fn(block_ids),
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            draft_logits = logits_fn(draft_hidden[:, 1:, :])[0]
            if node_budget > 0:
                top_ids, top_log_probs = _draft_top_k(
                    mx, draft_logits, min(node_budget, vocab_size), suppress_mask,
                )
                tree = build_ddtree_tree_from_topk(top_ids, top_log_probs, node_budget)
            else:
                top_ids, top_log_probs = _draft_top_k(mx, draft_logits, 1, suppress_mask)
                tree = build_ddtree_tree_from_topk(top_ids, top_log_probs, block_len - 1)
        else:
            tree = build_ddtree_tree_from_topk(
                np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.float32), 0,
            )
        node_token_ids, node_depths, _parents, child_maps, visibility = tree

        tree_tokens = [root_token] + [int(t) for t in node_token_ids]
        verify_logits, verify_hidden = tree_forward(
            text_model=text_model,
            logits_fn=logits_fn,
            cache=target_cache,
            input_ids=mx.array(tree_tokens, dtype=mx.uint32)[None],
            positions=tree_positions(start, node_depths),
            mask=tree_attention_mask(mx, visibility, start),
            capture_layer_ids=capture_layer_ids,
        )
        posterior = greedy(verify_logits[0])
        mx.eval(posterior)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior.tolist())
        acceptance_len = len(accepted_indices) - 1

        compact_kv_cache(target_cache, start, accepted_indices)
        context_feature = target_ops.extract_context_feature(verify_hidden, target_layer_ids)
        if accepted_indices == list(range(len(accepted_indices))):
            target_hidden = context_feature[:, : len(accepted_indices), :]
        else:
            target_hidden = mx.take(
                context_feature, mx.array(accepted_indices, dtype=mx.int32), axis=1,
            )
        mx.async_eval(target_hidden)
        start += len(accepted_indices)
        cycles += 1
        accepted_from_draft += acceptance_len
        acceptance_history.append(acceptance_len)

        committed = [tree_tokens[idx] for idx in accepted_indices[1:]] + [int(next_token)]
        for position, token in enumerate(committed):
            generated_tokens.append(token)
            per_token_accepted.append(position < acceptance_len)
            if token in stop_set:
                stopped = True
                break

    elapsed = time.perf_counter() - t_start
    output_tokens = len(generated_tokens)
    avg_acceptance = float(np.mean(acceptance_history)) if acceptance_history else 0.0
    try:
        accepted_spans, accepted_token_text = _accepted_spans(
            tokenizer, generated_tokens, per_token_accepted,
        )
    except Exception:
        accepted_spans, accepted_token_text = [], None

    return {
        "generated_tokens": generated_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "prefill_seconds": prefill_seconds,
        "cycles": cycles,
        "accepted_from_draft": accepted_from_draft,
        "avg_acceptance_length": avg_acceptance,
        "tree_budget": effective_budget,
        "accepted_spans": accepted_spans,
        "accepted_token_text": accepted_token_text,
    }
