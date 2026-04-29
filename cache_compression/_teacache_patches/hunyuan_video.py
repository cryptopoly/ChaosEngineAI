"""TeaCache forward patch for ``diffusers.models.HunyuanVideoTransformer3DModel``.

Authored against the upstream diffusers ``HunyuanVideoTransformer3DModel.forward``
shape — ali-vilab/TeaCache only ships ``teacache_forward_hunyuanvideo.py``
for the standalone tencent/HunyuanVideo repo (signature ``forward(self, x, t,
text_states, ...)``), not for diffusers' transformer class. This file adapts
the TeaCache caching block to the diffusers signature ``forward(self,
hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
pooled_projections, ...)``.

The caching idea (timestep-embedding-aware residual reuse) is identical to
upstream — only the surrounding transformer plumbing is rewritten.

Reference: https://github.com/ali-vilab/TeaCache (Apache 2.0).
Rescale coefficients pulled from upstream's HunyuanVideo calibration table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# Polynomial coefficients from upstream's HunyuanVideo calibration.
# Maps raw rel-L1 distance to a rescaled value that thresholds well across
# the canonical 50-step inference schedule. Lifted verbatim from
# ali-vilab/TeaCache/TeaCache4HunyuanVideo/teacache_sample_video.py.
_RESCALE_COEFFICIENTS = [
    7.33226126e02,
    -4.01131952e02,
    6.75869174e01,
    -3.14987800e00,
    9.61237896e-02,
]


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> tuple[torch.Tensor] | Transformer2DModelOutput:
    """TeaCache-augmented forward for ``HunyuanVideoTransformer3DModel``.

    On steps where the rescaled accumulated rel-L1 distance is below
    ``rel_l1_thresh``, the dual + single transformer block stack is skipped
    and the previous step's residual is reused. First and last steps always
    run a full pass to anchor the schedule.
    """
    batch_size, _num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    image_rotary_emb = self.rope(hidden_states)
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    if self.enable_teacache:
        inp_clone = hidden_states.clone()
        temb_clone = temb.clone()
        # First block's norm1 returns (norm_hidden_states, gate_msa, shift_mlp,
        # scale_mlp, gate_mlp) — we use the normalized hidden states as the
        # cache-stability probe, matching upstream's modulated_inp choice.
        modulated_inp = self.transformer_blocks[0].norm1(inp_clone, emb=temb_clone)[0]

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            rescale_func = np.poly1d(_RESCALE_COEFFICIENTS)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                ).cpu().item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

        if not should_calc:
            hidden_states = hidden_states + self.previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )
            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )
        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)
