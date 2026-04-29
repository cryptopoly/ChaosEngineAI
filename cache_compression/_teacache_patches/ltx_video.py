"""TeaCache forward patch for ``diffusers.models.LTXVideoTransformer3DModel``.

Authored against diffusers' LTX transformer signature ``forward(self,
hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, ...)``.
Upstream ali-vilab/TeaCache/TeaCache4LTX-Video targets the standalone
Lightricks/LTX-Video repo, so this is a fresh diffusers-shaped port — only
the caching block (rel-L1 distance probe + cached-residual reuse) is
borrowed from upstream.

Reference: https://github.com/ali-vilab/TeaCache (Apache 2.0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# Polynomial coefficients from upstream's LTX-Video calibration table.
_RESCALE_COEFFICIENTS = [
    2.14700694e01,
    -1.28016453e01,
    4.05578358e00,
    -3.31499127e-02,
    9.65706054e-04,
]


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_attention_mask: torch.Tensor,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    rope_interpolation_scale: tuple[float, float, float] | torch.Tensor | None = None,
    video_coords: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> torch.Tensor:
    """TeaCache-augmented forward for ``LTXVideoTransformer3DModel``.

    Skips the transformer block stack on stable timesteps where the
    rescaled accumulated rel-L1 distance stays below ``rel_l1_thresh``.
    First and last steps always run a full forward to anchor the schedule.
    """
    image_rotary_emb = self.rope(
        hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords
    )

    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.size(0)
    hidden_states = self.proj_in(hidden_states)

    temb, embedded_timestep = self.time_embed(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )
    temb = temb.view(batch_size, -1, temb.size(-1))
    embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

    if self.enable_teacache:
        # LTX block applies a layer-norm followed by ada-modulation derived
        # from a per-block scale_shift_table + temb. We replay those two
        # ops on the first block to get a modulated_inp probe consistent
        # with what the block itself sees on step 1.
        first_block = self.transformer_blocks[0]
        inp_clone = hidden_states.clone()
        temb_clone = temb.clone()
        norm_hidden = first_block.norm1(inp_clone)
        ada_values = first_block.scale_shift_table[None, None].to(temb_clone.device) + temb_clone.reshape(
            batch_size, temb_clone.size(1), first_block.scale_shift_table.shape[0], -1
        )
        shift_msa, scale_msa = ada_values.unbind(dim=2)[:2]
        modulated_inp = norm_hidden * (1 + scale_msa) + shift_msa

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
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )

    scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

    hidden_states = self.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale) + shift
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
