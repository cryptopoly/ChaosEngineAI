"""TeaCache forward patch for ``diffusers.models.MochiTransformer3DModel``.

Authored against diffusers' Mochi transformer signature ``forward(self,
hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, ...)``.
Upstream ali-vilab/TeaCache/TeaCache4Mochi targets the standalone
genmoai/models repo, so this is a fresh diffusers-shaped port of the
caching block only.

Reference: https://github.com/ali-vilab/TeaCache (Apache 2.0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# Polynomial coefficients from upstream's Mochi calibration.
_RESCALE_COEFFICIENTS = [
    -3.51241319e03,
    8.11675948e02,
    -6.09400215e01,
    2.42429681e00,
    3.05291719e-03,
]


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> torch.Tensor:
    """TeaCache-augmented forward for ``MochiTransformer3DModel``.

    Reuses the previous step's transformer residual when the rescaled
    accumulated rel-L1 distance stays below ``rel_l1_thresh``.
    """
    batch_size, _num_channels, num_frames, height, width = hidden_states.shape
    p = self.config.patch_size

    post_patch_height = height // p
    post_patch_width = width // p

    temb, encoder_hidden_states = self.time_embed(
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        hidden_dtype=hidden_states.dtype,
    )

    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
    hidden_states = self.patch_embed(hidden_states)
    hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

    image_rotary_emb = self.rope(
        self.pos_frequencies,
        num_frames,
        post_patch_height,
        post_patch_width,
        device=hidden_states.device,
        dtype=torch.float32,
    )

    if self.enable_teacache:
        first_block = self.transformer_blocks[0]
        inp_clone = hidden_states.clone()
        temb_clone = temb.clone()
        # Mochi's norm1 returns (norm_hidden_states, gate_msa, scale_mlp, gate_mlp).
        # First element is the modulated input probe.
        modulated_inp = first_block.norm1(inp_clone, temb_clone)[0]

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
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    encoder_attention_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
    hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
    output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
