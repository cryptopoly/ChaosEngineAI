"""TeaCache forward patch for ``diffusers.models.CogVideoXTransformer3DModel``.

Authored against diffusers' CogVideoX transformer signature ``forward(self,
hidden_states, encoder_hidden_states, timestep, timestep_cond, ofs,
image_rotary_emb, ...)``. Upstream ali-vilab/TeaCache/TeaCache4CogVideoX
targets the standalone THUDM/CogVideo repo, so this is a fresh
diffusers-shaped port of the caching block only.

Reference: https://github.com/ali-vilab/TeaCache (Apache 2.0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# Polynomial coefficients from upstream's CogVideoX calibration.
_RESCALE_COEFFICIENTS = [
    -3.10658903e01,
    2.54732368e01,
    -5.92380459e00,
    1.02011089e00,
    -2.07730091e-02,
]


def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: int | float | torch.LongTensor,
    timestep_cond: torch.Tensor | None = None,
    ofs: int | float | torch.LongTensor | None = None,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> tuple[torch.Tensor] | Transformer2DModelOutput:
    """TeaCache-augmented forward for ``CogVideoXTransformer3DModel``.

    Skips the transformer block stack on stable timesteps where the
    rescaled accumulated rel-L1 distance stays below ``rel_l1_thresh``.
    """
    batch_size, num_frames, _channels, height, width = hidden_states.shape

    timesteps = timestep
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    if self.ofs_embedding is not None:
        ofs_emb = self.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = self.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    if self.enable_teacache:
        first_block = self.transformer_blocks[0]
        inp_clone = hidden_states.clone()
        enc_clone = encoder_hidden_states.clone()
        emb_clone = emb.clone()
        # CogVideoX's norm1 is a CogVideoXLayerNormZero that returns
        # (norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa).
        # We use norm_hidden_states as the cache-stability probe.
        modulated_inp = first_block.norm1(inp_clone, enc_clone, emb_clone)[0]

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
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )
            self.previous_residual = hidden_states - ori_hidden_states
    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

    hidden_states = self.norm_final(hidden_states)
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    p = self.config.patch_size
    p_t = self.config.patch_size_t

    if p_t is None:
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = hidden_states.reshape(
            batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
