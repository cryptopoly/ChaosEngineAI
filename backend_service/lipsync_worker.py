"""Warm lip-sync worker: LTX-2.3 audio to video on the 4-bit diffusers row, with the audio held fixed (D-103 route b).

Started by ``lipsync_runtime`` as ``python -m backend_service.lipsync_worker`` from the backend's own Python, so the
31 GB pipeline lives in this process, never in FastAPI. The protocol loop is ``warm_worker_base.serve``. The pipeline
stays loaded between shots (a load is about 35 s); ``unload`` or the idle stop frees it.

What one shot does, as measured in ``docs/research/lip-sync-spike-windows-2026-11.md``:

1. The prepared clip (stereo 48 kHz WAV, exactly the shot's length) becomes a log-mel spectrogram the way Lightricks'
   ``ltx_core`` makes it (16 kHz, n_fft 1024, hop 160, 64 slaney mel bins, ``log(clamp(x, 1e-5))``, stereo) and is
   encoded with the pack's audio VAE (posterior mode).
2. ``LTX2ImageToVideoPipeline`` gets it as ``audio_latents`` with ``noise_scale=0``, and a forward pre-hook on the
   transformer puts the clean audio tokens back on every call with ``audio_timestep`` and ``audio_sigma`` at 0 (what
   ``ltx_core`` does for a frozen modality). Without the hold the mouth stays open through silence.
3. Distilled schedule, 8 steps, no CFG, STG 1, modality guidance 3 (three transformer passes per step). The decoded
   video is written silent; the backend muxes the prepared clip in.
"""

from __future__ import annotations

import gc
import time
from typing import Any

from backend_service import warm_worker_base as BASE

MEL_RATE = 16000
N_FFT = 1024
HOP = 160
N_MELS = 64
AUDIO_DOWNSAMPLE = 4  # audio VAE temporal compression: 16 kHz / 160 / 4 = 25 latent frames a second
STG_SCALE = 1.0
MODALITY_SCALE = 3.0


def audio_latent_frames(seconds: float) -> int:
    return round(seconds * MEL_RATE / HOP / AUDIO_DOWNSAMPLE)


def log_mel(path: str, seconds: float) -> Any:
    """``[1, 2, T, 64]`` log-mel of the clip, made exactly like ``ltx_core`` (stereo, power 1, slaney)."""
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415

    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = wav.T
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    wav = wav[:2]
    if sr != MEL_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=MEL_RATE, res_type="soxr_hq")
    n = int(round(seconds * MEL_RATE))
    wav = wav[:, :n] if wav.shape[1] >= n else np.pad(wav, ((0, 0), (0, n - wav.shape[1])))
    x = torch.from_numpy(np.ascontiguousarray(wav))
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=torch.hann_window(N_FFT), center=True,
                      pad_mode="reflect", return_complex=True).abs()
    bank = librosa.filters.mel(sr=MEL_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=MEL_RATE / 2, htk=False,
                               norm="slaney")
    mel = torch.einsum("mf,cft->cmt", torch.from_numpy(bank).float(), spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.permute(0, 2, 1)[None].contiguous()


class AudioHold:
    """Transformer pre-hook: every call sees the clean audio tokens at timestep 0 and sigma 0."""

    def __init__(self) -> None:
        self.clean: Any = None
        self.calls = 0

    def __call__(self, _module: Any, args: tuple, kwargs: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
        import torch  # noqa: PLC0415

        self.calls += 1
        ah = kwargs["audio_hidden_states"]
        if self.clean is None:
            self.clean = ah[:1].detach().clone()  # noise_scale 0: the first call sees the clean latent
        kwargs["audio_hidden_states"] = self.clean.to(ah.device, ah.dtype).expand_as(ah).contiguous()
        zeros = torch.zeros(ah.shape[0], device=ah.device, dtype=kwargs["timestep"].dtype)
        kwargs["audio_timestep"] = zeros
        kwargs["audio_sigma"] = zeros.clone()
        return args, kwargs


def _host_peak_gb() -> float | None:
    try:
        import psutil  # noqa: PLC0415

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss  # Windows reports the peak working set
        return round(peak / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


class LipSyncEngine:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.snapshot: str | None = None

    def loaded(self) -> dict[str, Any]:
        return {"snapshot": self.snapshot} if self.pipe is not None else {}

    def unload(self) -> None:
        self.pipe, self.snapshot = None, None
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _ensure(self, snapshot: str) -> bool:
        """Load the pipeline for ``snapshot`` (unloading another first); True when it was already loaded."""
        if self.pipe is not None and self.snapshot == snapshot:
            return True
        self.unload()
        import torch  # noqa: PLC0415
        from diffusers import LTX2ImageToVideoPipeline  # noqa: PLC0415

        pipe = LTX2ImageToVideoPipeline.from_pretrained(snapshot, torch_dtype=torch.bfloat16, processor=None,
                                                        local_files_only=True)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        self.pipe, self.snapshot = pipe, snapshot
        return False

    def _encode_audio(self, path: str, seconds: float) -> Any:
        import torch  # noqa: PLC0415

        vae = self.pipe.audio_vae
        mel = log_mel(path, seconds)
        vae.to("cuda")
        try:
            with torch.no_grad():
                lat = vae.encode(mel.to("cuda", torch.bfloat16)).latent_dist.mode()
        finally:
            vae.to("cpu")
        want = audio_latent_frames(seconds)
        if lat.shape[2] < want:
            raise RuntimeError(f"the audio VAE gave {lat.shape[2]} latent frames for {seconds:.2f} s, expected {want}")
        return lat[:, :, :want].float()  # stays on the GPU: the pipeline noises it with a CUDA generator

    def generate(self, request: dict[str, Any], hooks: Any) -> dict[str, Any]:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES  # noqa: PLC0415

        from backend_service.video_runtime.pipeline_helpers import (  # noqa: PLC0415
            LTX2_IMAGE_CRF,
            encode_frames_to_mp4,
            load_start_image,
            recompress_still,
        )

        frames, fps, steps = int(request["frames"]), int(request["fps"]), int(request.get("steps") or 8)
        width, height = int(request["width"]), int(request["height"])
        seconds = frames / fps
        t0 = time.perf_counter()
        hooks.stage("load")
        warm = self._ensure(str(request["snapshot"]))
        t_loaded = time.perf_counter()
        hooks.check_cancel()
        hooks.stage("encode")
        latents = self._encode_audio(str(request["audio"]), seconds)
        image = recompress_still(load_start_image(str(request["image"]), width, height), LTX2_IMAGE_CRF)
        t_encoded = time.perf_counter()
        hold = AudioHold()
        handle = self.pipe.transformer.register_forward_pre_hook(hold, with_kwargs=True)
        marks: dict[str, float] = {}

        def on_step(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            hooks.step(step + 1, steps)
            if step + 1 >= steps:
                marks["denoised"] = time.perf_counter()
                hooks.stage("decode")
            hooks.check_cancel()
            return callback_kwargs

        torch.cuda.reset_peak_memory_stats()
        try:
            out = self.pipe(
                image=image, prompt=str(request["prompt"]), width=width, height=height, num_frames=frames,
                frame_rate=float(fps), sigmas=list(DISTILLED_SIGMA_VALUES), num_inference_steps=steps,
                guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=STG_SCALE, audio_stg_scale=STG_SCALE,
                modality_scale=MODALITY_SCALE, audio_modality_scale=MODALITY_SCALE, audio_latents=latents,
                noise_scale=0.0, generator=torch.Generator("cuda").manual_seed(int(request["seed"])), image_crf=0,
                output_type="np", return_dict=True, callback_on_step_end=on_step,
            )
        finally:
            handle.remove()
        t_decoded = time.perf_counter()
        vram_peak = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        clip = np.clip(out.frames[0], 0.0, 1.0)
        del out
        hooks.stage("write")
        data = encode_frames_to_mp4([frame for frame in clip], fps)
        with open(str(request["output"]), "wb") as handle_out:
            handle_out.write(data)
        torch.cuda.empty_cache()
        denoised = marks.get("denoised", t_decoded)
        return {
            "warm": warm,
            "frames": int(clip.shape[0]),
            "transformer_calls": hold.calls,
            "load_s": round(t_loaded - t0, 2),
            "encode_s": round(t_encoded - t_loaded, 2),
            "denoise_s": round(denoised - t_encoded, 2),
            "decode_s": round(t_decoded - denoised, 2),
            "vram_peak_gb": vram_peak,
            "host_peak_gb": _host_peak_gb(),
        }


def main() -> None:
    BASE.strip_own_dir(__file__)
    from importlib.metadata import PackageNotFoundError, version as package_version  # noqa: PLC0415

    try:
        version = f"diffusers {package_version('diffusers')}"  # read without importing it (seconds)
    except PackageNotFoundError:  # the first generate reports the real import error
        version = "diffusers ?"
    BASE.serve(LipSyncEngine, engine_version=version)


if __name__ == "__main__":
    main()
