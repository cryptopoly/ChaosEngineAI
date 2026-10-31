# SPDX-License-Identifier: Apache-2.0
"""MLX port of Liquid AI's LFM3 text models (model_type "lfm3").

mlx-lm has no ``lfm3`` yet (Liquid AI's model card says support is in
progress), so ``backend_service/mlx_models`` hands this file to ``mlx_lm.load``
through the ``model_file`` hook and steps aside once mlx-lm ships its own.

Adapted from the Apache-2.0 reference code in the LiquidAI/LFM3-1.7B
checkpoint (``modeling_lfm3.py``) and the mlx-vlm port of it
(``mlx_vlm/models/lfm3``, commit 701456f1). LFM3 extends LFM2 with:
- a third layer type: ``conv`` (LFM2 gated short convolution, width 7),
  ``full_attention`` and ``sliding_attention`` (window ``sliding_window``);
- RoPE per attention type; full attention rotates only the first
  ``partial_rotary_factor * head_dim / 2`` frequency pairs of the full head
  progression and leaves the rest unrotated (zero-tail inverse frequencies);
- a sigmoid output gate per attention head (``gate_proj``);
- sandwich norms (``post_operator_norm`` / ``post_ffn_norm``).

The official MLX checkpoints are saved in mlx-vlm's layout, with every weight
under ``language_model.``; ``sanitize`` strips that prefix.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.models.lfm2 import ShortConv

_LAYER_TYPES = {"conv", "full_attention", "sliding_attention"}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: List[str]
    norm_eps: float = 1e-5
    head_dim: Optional[int] = None
    conv_L_cache: int = 7
    conv_bias: bool = False
    rope_parameters: Optional[dict] = None
    sliding_window: Optional[int] = 256
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 262144

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have one entry per hidden layer")
        if not set(self.layer_types) <= _LAYER_TYPES:
            raise ValueError(f"Unsupported LFM3 layer type in {self.layer_types}")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("sliding_attention layers need a sliding_window")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 1e4},
            }


class RotaryEmbedding(nn.Module):
    """Non-interleaved RoPE whose inverse frequencies end in zeros (unrotated pairs)."""

    def __init__(self, dim: int, parameters: dict):
        super().__init__()
        if parameters.get("rope_type", "default") != "default":
            raise ValueError("LFM3 supports only default rotary embeddings")
        angles = int(dim * parameters.get("partial_rotary_factor", 1.0)) // 2
        if not 0 < angles <= dim // 2:
            raise ValueError("Invalid partial_rotary_factor")
        base = parameters.get("rope_theta", 1e6)
        inv_freq = base ** (-mx.arange(angles, dtype=mx.float32) * 2 / dim)
        self._inv_freq = mx.pad(inv_freq, [(0, dim // 2 - angles)])

    def __call__(self, x: mx.array, offset=0) -> mx.array:
        if isinstance(offset, mx.array) and offset.ndim > 0:
            offset = offset[:, None]
        positions = mx.arange(x.shape[-2]) + offset
        angles = positions[..., None].astype(mx.float32) * self._inv_freq
        angles = mx.concatenate([angles, angles], axis=-1)
        if angles.ndim == 3:
            angles = angles[:, None]
        cos = mx.cos(angles).astype(x.dtype)
        sin = mx.sin(angles).astype(x.dtype)
        a, b = mx.split(x, 2, axis=-1)
        return x * cos + mx.concatenate([-b, a], axis=-1) * sin


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.gate_proj = nn.Linear(dim, self.n_heads, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, args.rope_parameters[layer_type])

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1))
        keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1))
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset)
        keys = self.rope(keys, offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3)
        output = output * mx.sigmoid(self.gate_proj(x))[..., None]
        return self.o_proj(output.reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w3 = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.w2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_attention_layer = self.layer_type != "conv"
        self.use_sandwich_norm = args.use_sandwich_norm
        if self.is_attention_layer:
            self.self_attn = Attention(args, self.layer_type)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = MLP(args)
        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
            self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.operator_norm(x)
        if self.is_attention_layer:
            h = self.self_attn(h, mask=mask, cache=cache)
        else:
            h = self.conv(h, mask=mask, cache=cache)
        if self.use_sandwich_norm:
            h = self.post_operator_norm(h)
        x = x + h
        h = self.feed_forward(self.ffn_norm(x))
        if self.use_sandwich_norm:
            h = self.post_ffn_norm(h)
        return x + h


class Lfm3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            if layer.layer_type == "conv":
                mask = create_ssm_mask(h, c)
            elif layer.layer_type == "sliding_attention":
                mask = create_attention_mask(h, c, window_size=self.args.sliding_window)
            else:
                mask = create_attention_mask(h, c)
            h = layer(h, mask, cache=c)
        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        sanitized = {}
        for name, value in weights.items():
            if name.startswith("language_model."):
                name = name[len("language_model."):]
            if self.args.tie_word_embeddings and name.startswith("lm_head."):
                continue
            # PyTorch Conv1d weights are (out, 1, kernel); MLX wants (out, kernel, 1).
            if name.endswith("conv.conv.weight") and value.shape[-1] > value.shape[1]:
                value = value.transpose(0, 2, 1)
            sanitized[name] = value
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == "conv":
                caches.append(ArraysCache(size=1))
            elif layer.layer_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches
