"""AI video upscale, fast tier (Real-ESRGAN family through spandrel; docs/research/ai-video-upscaler-feasibility-2026-11.md).

``engine`` is the model contract (frames in, frames out, tiles for big frames), ``pipeline`` the ffmpeg frame
pipeline with segments and a resume manifest, ``worker`` the warm worker process that holds the model, ``jobs``
the background jobs on the shared heavy-job slot, ``addon`` the on-demand install and ``storyboard_hook`` the
per-shot upscale a Storyboard export uses. Nothing here imports torch or spandrel at module level.
"""
