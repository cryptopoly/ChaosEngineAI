# Open video models that start from an image (and end on one)

Internal research, 2026-11-07, for D-40 ("storyboard clips need LTX-2.3 or 2.5").
Checked against the Hugging Face API (licence tag, repo size, dates) and the model cards on the day.
"Verified" = the repo exists and the claim is on its card or licence file. "Unverified" = secondary source only.

## ELI5

A storyboard clip has to start on a still picture, and ideally stop on the next one.
We already do this with LTX-2.3 and LTX-2.5 on the Mac, and they are still the best fit.
Nothing new beats them on a Mac today. Two models are worth adding later:
**Wan 2.2 image-to-video** (already in our catalog, and it can do start *and* end frames) and
**Kandinsky 6.0 Lite** (small, MIT, makes sound too, start frame only).
The two newest headline models (MiniMax H3 and the Pika H3 Flex build on it) and HunyuanVideo 1.5
**forbid use in the UK and EU** in their licences, so we cannot ship them.
Wan 2.5, 2.6 and later are not open at all.

**Recommendation for D-40:** accept the LTX-only limit for the first release. Next step is
Wan 2.2 image input (start + end) through the diffusers path we already have, mainly for NVIDIA users.
Kandinsky 6.0 Lite after that if users want a second Mac option with sound.

## How image input works in our app today

- `VideoGenerationRequest` already carries `startImagePath` / `endImagePath`
  (`backend_service/models/__init__.py`, `video_runtime/types.py`).
- Only `ltx2_mlx_runtime.py` uses them: `--image PATH FRAME_IDX STRENGTH`, once for frame 0 and once for the last frame.
- `routes/video.py::_checked_input_images` and `mcp_bridge/models.py` (`acceptsImageInput`) let only
  `_is_ltx2_mlx_repo` repos through. The storyboard batch skips the rest (`storyboard/generators.py`).
- So adding a model = (1) pass the image(s) into its pipeline call, (2) widen that one gate to a
  per-variant `acceptsImageInput` / `acceptsEndImage` flag in `catalog/video_models.py`.
  The request, storyboard and MCP plumbing does not change.

## Verdict table

| Model | Start / end frame | Licence (commercial?) | Weights | Mac | Audio | Verdict |
|---|---|---|---|---|---|---|
| **LTX-2.3 / LTX-2.5** (Lightricks) | Yes / yes (keyframes at any index) | LTX-2.x Community: free under $10M yearly revenue, paid above | q4 MLX packs about 25 to 50 GB | **Yes, ltx-2-mlx (shipped)** | Yes | **Keep as the storyboard default** |
| **Wan 2.2 I2V A14B** (Alibaba) | Yes / yes (`last_image` in diffusers) | Apache 2.0 | 126 GB full repo; lightx2v 4-step distill files in catalog; MLX q8 42.7 GB | MPS via diffusers (slow); community MLX port | No | **Add next** (small effort, CUDA first) |
| Wan 2.2 TI2V 5B | Yes / no | Apache 2.0 | 34 GB | MPS; MLX ports exist | No | Possible light tier; we retired it for LTX |
| Wan 2.1 FLF2V 14B 720P | Yes / yes (built for it) | Apache 2.0 | 90 GB | MPS only, heavy | No | Superseded by Wan 2.2 `last_image` |
| **Kandinsky 6.0 Lite** (3B) | Yes / no (`image=` = first frame) | MIT | 34 GB repo; MLX port 9.1 GB | Yes, diffusers MPS or community MLX | Yes, 44 kHz | **Add after Wan** (medium effort) |
| Kandinsky 6.0 Pro (29B) | Yes / no | MIT | 160 GB | Too big for 64 GB | Yes | CUDA only, later |
| Kandinsky 5.0 Video Pro / Lite | Yes / no | MIT | Lite about 2B, Pro 19B | Diffusers | No | Superseded by 6.0 |
| LongCat-Video (Meituan, 13.6B) | Yes / no; also video continuation | MIT | 83 GB | No port; own torchrun code | No | CUDA only; continuation is interesting; high effort |
| MAGI-1 (Sand AI, 4.5B / 24B) | Yes / no; chunked continuation | Apache 2.0 | 255 GB repo | No | No | Skip (heavy, CUDA only) |
| SkyReels V2 I2V / V3 (Skywork) | Yes / no; V2 "diffusion forcing" long video | Skywork licence (custom; commercial terms need a lawyer's read) | 82 GB (V2 14B) | No | V3 A2V has audio input | Skip for now |
| Ovi 1.1 (11B) | Yes / no | Apache 2.0 | 70 GB repo | No | Yes | CUDA only, watch |
| MOVA (OpenMOSS) | Yes / no | Apache 2.0 | 78 GB (360p) | No | Yes | CUDA only, watch |
| daVinci-MagiHuman (GAIR, 15B) | Yes (reference image) / no | Apache 2.0 | 216 GB repo | No | Yes, speech | People-focused talking clips; CUDA only |
| Cosmos3 Super Image2Video (NVIDIA) | Yes / no | OpenMDW 1.1 (permissive, commercial OK per card) | 196 GB | No | Yes | Robotics / "physical AI" focus; skip |
| CogVideoX 1.5 5B I2V | Yes / no | CogVideoX licence (custom) | 31 GB | Diffusers MPS | No | Old quality; skip |
| HunyuanVideo 1.5 I2V | Yes / no | Tencent Hunyuan Community: **excludes EU, UK, South Korea** | 372 GB repo | Unverified MLX upload (0 downloads) | No | **Cannot ship (territory)** |
| **MiniMax H3** (33B, FL2VA) | Yes / yes | MiniMax H3 Community: **excludes EU, UK, Korea and US**; must show "MiniMax H3" in UI | 355 GB | No | Yes, stereo | **Cannot ship (territory)** |
| Pika H3 Flex (on H3) | Yes / yes | Same MiniMax H3 licence | 220 GB | No | Yes | **Cannot ship (territory)** |
| Mochi 1 | No (text only) | Apache 2.0 | | | No | Not relevant |
| Self-Forcing / CausVid | Text-to-video Wan 1.3B distills | Apache 2.0 | small | | No | Not relevant to image start |
| Wan 2.5 / 2.6 / 2.7 / 3.0 | API only | Closed | none on HF | | | Not open (unverified claims of open 2.6/2.7 exist; the Wan-AI HF org has nothing past 2.2) |

## Notes per model

**LTX-2.3 / LTX-2.5.** Already wired with start and end frames. LTX-2.5 is the most liked image-to-video
repo on Hugging Face (about 10.9k likes) and our measured 4 s 768x512 clip with sound takes about 128 s
on the M4 Max (`docs/research/m4max-generation-times.md`). Licence: free for entities under $10M annual
revenue (LICENSE-2_x, section 2.1); fine for us and most users. LTX-2.5 repos are click-through gated.
- https://huggingface.co/Lightricks/LTX-2.5 , https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
- https://huggingface.co/dgrauet/ltx-2.5-mlx-q4 , https://huggingface.co/mlx-community/ltx-2.5-mlx-q8

**Wan 2.2 I2V A14B.** Apache 2.0, the best-known open I2V model after LTX. The diffusers
`WanImageToVideoPipeline` takes `image=` and `last_image=` (first-last-frame), so one pipeline gives both.
We already ship the variant rows (lightx2v 4-step distill, bf16 and fp8) but block image input at the route.
Effort: **about 1 to 2 days** (pass `image` / `last_image` in the diffusers call, resize to the
model bucket, flip the gate per variant, one E2E check). On a Mac it runs on MPS and is slow
(the 5B took 293 s for 33 frames at 832x480; the 27B MoE needs offload and is much slower), so it is
mainly for NVIDIA users. Blaizzy/mlx-video has Wan 2.2 I2V 14B after a convert step, and a community
q8 MLX build exists (42.7 GB; its card says the 14B "struggles" on 64 GB). Our `mlx_video_runtime.py`
already wraps mlx-video for LTX, so an MLX Wan route is possible later (+2 to 3 days, needs testing on 64 GB).
- https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers , https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_i2v.py (`last_image`)
- https://github.com/Blaizzy/mlx-video , https://huggingface.co/Anes1032/Wan2.2-I2V-A14B-mlx-q8
- https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers

**Kandinsky 6.0 Lite (3B, MIT, Sept 2026).** Video with synchronised 44 kHz audio, 5 s clips, 10-step
distilled checkpoint. `Kandinsky6TI2VAPipeline(image=...)` conditions the first frame; no end frame.
Official diffusers support plus a community MLX port whose card reports about 935 s for a 5 s clip at
864x480 on an M3 Max 48 GB (about 10 % faster than tuned MPS), so expect roughly 10 minutes on our
M4 Max: several times slower than LTX-2.5. Effort: **about 2 to 4 days** (new diffusers version with the
`kandinsky6` pipeline in the video venv, audio muxing, catalog rows, gate, E2E). Worth it as a second
"with sound" model and as a strong CUDA option; not a storyboard default on a Mac.
- https://huggingface.co/kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers (report: https://arxiv.org/abs/2610.05608)
- https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX (community port)
- Older: https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers (MIT)

**LongCat-Video.** MIT, 13.6B, text, image and video continuation in one model. Its own code
(`run_demo_image_to_video.py`, torchrun), no diffusers or MLX. Meituan's own I2V scores are a little
under Wan 2.2. Video continuation would fit "continue from the previous shot", but it is CUDA-only and
a separate runtime. High effort. https://huggingface.co/meituan-longcat/LongCat-Video

**MAGI-1.** Apache 2.0, autoregressive chunks (I2V and continuation), 255 GB repo, CUDA-only. https://huggingface.co/sand-ai/MAGI-1

**SkyReels V2 / V3.** Wan-based, I2V and long video; custom Skywork licence with use conditions written
mainly for Chinese internet services. Needs a lawyer's read before shipping.
https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-720P , https://huggingface.co/Skywork/SkyReels-V3-A2V-19B

**Audio-video newcomers (all CUDA-only, no Mac port found).** Ovi 1.1 (Apache, 960x960, 5 or 10 s):
https://huggingface.co/chetwinlow1/Ovi . MOVA (Apache): https://huggingface.co/OpenMOSS-Team/MOVA-360p .
daVinci-MagiHuman (Apache, 15B, 5 s 1080p in 38 s on one H100): https://huggingface.co/GAIR/daVinci-MagiHuman .
Cosmos3 (OpenMDW): https://huggingface.co/nvidia/Cosmos3-Super-Image2Video .

**Licence blocks (do not ship).**
- MiniMax H3 licence: "Excluded Territories means the European Union, the United Kingdom, the Republic of
  Korea and the United States of America", and use or display of outputs outside the territory is not
  licensed. The owner is in the UK, so even testing is outside the licence.
  https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE ; Pika build: https://huggingface.co/Pika-Labs/Pika-H3-Flex
- HunyuanVideo 1.5: "does not apply in the European Union, United Kingdom and South Korea".
  https://huggingface.co/tencent/HunyuanVideo-1.5
- Wan 2.5 and later: no weights on the Wan-AI Hugging Face org (checked: `Wan-AI/Wan2.5-I2V`, `Wan-AI/Wan2.7` do not exist).
  Secondary claims otherwise: https://flowith.io/blog/wan-2-6-3-0-open-weight-architecture-democratize-ai-film-production (unverified, contradicted by HF).

## What a lawyer or owner should check

- Skywork and CogVideoX custom licences, if either is ever considered.
- The LTX-2.x $10M threshold wording, if the app or a user business crosses it (it binds the user, but our UI should say so).
- MiniMax says it may license excluded territories on request; not worth pursuing now.

## Suggested follow-ups (not done here)

1. FU row: "Wan 2.2 image input (start + end) via diffusers `image` / `last_image`; per-variant `acceptsImageInput`" (1 to 2 days, Sonnet).
2. FU row: "Kandinsky 6.0 Lite T2AV / TI2AV via diffusers, MLX port evaluation on 64 GB" (2 to 4 days).
3. Re-check in Q1 2027: MLX ports of LongCat-Video, Ovi, MOVA; any Wan open release past 2.2.
