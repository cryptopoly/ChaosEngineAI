"""Warm quality-upscale worker: holds the FlashVSR+ Turbo pipeline and upscales one segment per ``generate``.

Runs from the quality add-on's venv (``quality_addon.venv_python()``, ``python -I``), never in the backend, and imports
nothing from ``backend_service``: the protocol loop is ``warm_worker_base.py``, loaded by path. A ``generate`` request is
``quality.worker_request``: the ffmpeg decode command (the segment plus its warm-up frames), the encode command (the
exact target size), the padded model size and the model files. Per segment the worker

1. decodes every frame of the segment to memory (``uint8``, source size),
2. runs the pipeline on them as one tensor (``build_inference_pipeline`` + ``InferenceRequest(lq_video=...)``, not
   the CLI): scaled with bicubic to the target, padded by reflection up to the 128 grid, colour correction on,
3. crops the output back to the target, drops the warm-up frames and pipes the rest to the encoder,

and answers ``{"frames": n, "warmup": w, "vram_peak_gb": ...}``. It sends ``step`` events while it encodes. The model
stays loaded between segments and jobs until ``unload``, the idle stop, or a heavy job of another kind stops it.
``CHAOSENGINE_WARM_ENGINE_MODULE`` swaps in a fake engine (tests); ``QualityEngine(model_factory=...)`` takes a fake
model, so ``process_segment`` is tested with real ffmpeg and no torch.
"""

from __future__ import annotations

import gc
import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent


def _base() -> Any:
    spec = importlib.util.spec_from_file_location("chaosengine_warm_worker_base", _HERE.parent / "warm_worker_base.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODE_BATCH = 8


def _read_exact(stream: Any, size: int) -> bytes:
    chunks, got = [], 0
    while got < size:
        block = stream.read(size - got)
        if not block:
            break
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


def _tail(handle: Any) -> str:
    try:
        handle.seek(0)
        lines = handle.read().decode("utf-8", "replace").strip().splitlines()
    except (OSError, ValueError):
        return ""
    return " | ".join(lines[-3:])[:300]


def _stop(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def process_segment(req: dict[str, Any], model: Callable[..., Any], step: Callable[[int, int], None],
                    check_cancel: Callable[[], None]) -> dict[str, Any]:
    """Decode, upscale and encode one segment (see the module docstring). ``model(frames, size, padded)`` takes
    ``uint8`` frames ``(n, h, w, 3)`` and returns ``uint8`` frames ``(n, height, width, 3)`` at the target size."""
    import numpy as np  # noqa: PLC0415

    src_w, src_h = int(req["src_width"]), int(req["src_height"])
    width, height = int(req["width"]), int(req["height"])
    frame_bytes = src_w * src_h * 3
    output = Path(req["output"])
    warm = int(req.get("warmup") or 0)
    blocks: list[bytes] = []
    began = time.perf_counter()
    with tempfile.TemporaryFile() as dec_err, tempfile.TemporaryFile() as enc_err:
        decoder = subprocess.Popen(req["decode"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=dec_err)
        encoder: subprocess.Popen | None = None
        try:
            while True:
                check_cancel()
                data = _read_exact(decoder.stdout, frame_bytes * 16)
                usable = len(data) - len(data) % frame_bytes
                if usable:
                    blocks.append(data[:usable])
                if len(data) < frame_bytes * 16:
                    break
            decoder.wait(timeout=60)
            if decoder.returncode != 0:
                raise RuntimeError(f"ffmpeg could not decode the video: {_tail(dec_err) or decoder.returncode}")
            joined = b"".join(blocks)
            blocks.clear()
            count = len(joined) // frame_bytes
            kept = max(0, count - warm)
            if kept == 0:
                return {"frames": 0, "warmup": warm}
            frames = np.frombuffer(joined, dtype=np.uint8).reshape(count, src_h, src_w, 3)
            check_cancel()
            started = time.perf_counter()
            decode_s = started - began
            shown = [0]

            def report(done: int) -> None:  # the job's frame count only ever grows
                if done > shown[0]:
                    shown[0] = done
                    step(done, kept)

            def on_model(fraction: float) -> None:
                # The DiT pass is about 60 % of a segment's time; the tiny decoder (no progress of its own) and the
                # encode follow, and the encode's frames carry the count to the end.
                check_cancel()
                report(int(fraction * kept * 0.6))

            out = model(frames, (width, height), (int(req["pad_width"]), int(req["pad_height"])), progress=on_model)
            model_s = time.perf_counter() - started
            del frames, joined
            if tuple(out.shape) != (count, height, width, 3):
                raise RuntimeError(f"The model returned frames shaped {tuple(out.shape)}, expected "
                                   f"{(count, height, width, 3)}.")
            check_cancel()
            encode_from = time.perf_counter()
            encoder = subprocess.Popen(req["encode"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=enc_err)
            made = 0
            for start in range(warm, count, ENCODE_BATCH):
                check_cancel()
                part = np.ascontiguousarray(out[start:start + ENCODE_BATCH], dtype=np.uint8)
                try:
                    encoder.stdin.write(part.tobytes())  # type: ignore[union-attr]
                except (BrokenPipeError, OSError) as exc:
                    raise RuntimeError(f"The encoder stopped: {_tail(enc_err) or exc}") from exc
                made += int(part.shape[0])
                report(made)
            encoder.stdin.close()  # type: ignore[union-attr]
            encoder.wait(timeout=600)
            if encoder.returncode != 0 or not output.is_file():
                raise RuntimeError(f"ffmpeg could not encode the upscaled frames: {_tail(enc_err) or encoder.returncode}")
        except BaseException:
            _stop(decoder)
            _stop(encoder)
            output.unlink(missing_ok=True)
            raise
        finally:
            if decoder.stdout is not None:
                decoder.stdout.close()
    encode_s = time.perf_counter() - encode_from
    print(f"segment {output.name}: {count} frames ({warm} warm-up) to {width}x{height} (model {req['pad_width']}x"
          f"{req['pad_height']}): decode {decode_s:.1f} s, model {model_s:.1f} s ({count / model_s:.2f} fps), "
          f"encode {encode_s:.1f} s", file=sys.stderr, flush=True)
    return {"frames": made, "warmup": warm, "decode_seconds": round(decode_s, 2), "model_seconds": round(model_s, 2),
            "encode_seconds": round(encode_s, 2), "model_fps": round(count / model_s, 2) if model_s > 0 else None}


class FlashVSRModel:
    """The FlashVSR+ Turbo pipeline on CUDA, driven with tensors. Needs the add-on's venv (torch, triton) and its
    ``code`` folder on ``sys.path``."""

    CHUNK = 16  # frames scaled and converted on the GPU at a time

    def __init__(self, code: str, paths: dict[str, str], *, color_correction: bool = True, device: str = "cuda:0"):
        for entry in (str(Path(code) / "operators" / "flashvsr_attn"), str(code)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        import torch  # noqa: PLC0415
        from inference import build_inference_pipeline  # noqa: PLC0415
        from inference.loading import ModelPaths  # noqa: PLC0415
        from inference.presets import stream_config_for_mode  # noqa: PLC0415

        self.torch, self.device = torch, device
        config = stream_config_for_mode("turbo", device=device, attn_kernel="triton_block_sparse",
                                        color_correction=color_correction)
        self.pipe = build_inference_pipeline(config, ModelPaths(**paths))
        torch.cuda.synchronize()
        # Every stream chunk (4 latents, 16 frames) calls the DiT's ``forward_stream`` once (one step). A wrapper there
        # gives progress inside a segment and lets a cancel stop within one chunk instead of after the whole segment.
        self._tick: Callable[[], None] | None = None
        denoiser = self.pipe.bundle.denoiser
        original = denoiser.forward_stream

        def forward_stream(*args: Any, **kwargs: Any) -> Any:
            if self._tick is not None:
                self._tick()
            return original(*args, **kwargs)

        denoiser.forward_stream = forward_stream

    @staticmethod
    def expected_chunks(frames: int) -> int:
        """DiT stream calls for a run of ``frames`` frames: ``(F - 1) // 4 + 1`` latents, four a chunk (125 frames: 8,
        counted on the RTX 4090; one more may come for the padded tail, so the fraction is capped)."""
        return max(1, -(-((frames - 1) // 4 + 1) // 4))

    def __call__(self, frames: Any, size: tuple[int, int], padded: tuple[int, int],
                 progress: Callable[[float], None] | None = None) -> Any:
        import numpy as np  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from inference import InferenceRequest  # noqa: PLC0415

        torch = self.torch
        width, height = size
        pad_w, pad_h = padded
        count = int(frames.shape[0])
        began = time.perf_counter()
        with torch.inference_mode():
            # [1, 3, F, H, W] in [-1, 1] at the padded size (the pipeline's range), on the CPU like its file path.
            lq = torch.empty((1, 3, count, pad_h, pad_w), dtype=torch.float32)
            for start in range(0, count, self.CHUNK):
                x = torch.from_numpy(np.array(frames[start:start + self.CHUNK])).to(self.device)  # a writable copy
                x = x.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
                if (x.shape[-1], x.shape[-2]) != (width, height):
                    x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False).clamp_(-1.0, 1.0)
                if (pad_w, pad_h) != (width, height):
                    x = F.pad(x, (0, pad_w - width, 0, pad_h - height), mode="reflect")
                lq[0, :, start:start + x.shape[0]] = x.permute(1, 0, 2, 3).cpu()
                del x
            t_lq = time.perf_counter()
            ticks, total = [0], self.expected_chunks(count)

            def tick() -> None:
                ticks[0] += 1
                if progress is not None:
                    progress(min(1.0, (ticks[0] - 1) / total))

            self._tick = tick
            try:
                result = self.pipe.infer(InferenceRequest(lq_video=lq))
            finally:
                self._tick = None
            del lq
            t_infer = time.perf_counter()
            video = result.video  # [1, 3, F, H, W] in [-1, 1]
            if video is None or int(video.shape[2]) != count:
                raise RuntimeError(f"FlashVSR+ returned {None if video is None else int(video.shape[2])} frames for "
                                   f"{count}.")
            out = np.empty((count, height, width, 3), dtype=np.uint8)
            for start in range(0, count, self.CHUNK):
                part = video[0, :, start:start + self.CHUNK, :height, :width].float()
                part = part.add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
                out[start:start + part.shape[1]] = part.permute(1, 2, 3, 0).cpu().numpy()
            del video, result
        done = time.perf_counter()
        print(f"  model: frames in {t_lq - began:.1f} s, pipeline {t_infer - t_lq:.1f} s ({ticks[0]} chunks, "
              f"{total} expected), frames out {done - t_infer:.1f} s", file=sys.stderr, flush=True)
        return out

    def vram_peak_gb(self) -> float | None:
        try:
            return round(self.torch.cuda.max_memory_reserved() / 2**30, 2)
        except Exception:  # noqa: BLE001
            return None

    def reset_peak(self) -> None:
        try:
            self.torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass

    def unload(self) -> None:
        self.pipe = None
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class QualityEngine:
    def __init__(self, model_factory: Callable[..., Any] | None = None) -> None:
        self._factory = model_factory
        self.model: Any = None
        self.key: tuple[Any, ...] | None = None

    def loaded(self) -> dict[str, Any]:
        return {"model": "flashvsr-plus-turbo", "colorCorrection": self.key[-1]} if self.key else {}

    def unload(self) -> None:
        if self.model is not None and hasattr(self.model, "unload"):
            self.model.unload()
        self.model, self.key = None, None
        gc.collect()

    def _ensure(self, request: dict[str, Any]) -> bool:
        paths = {k: str(v) for k, v in dict(request["paths"]).items()}
        key = (str(request["code"]), tuple(sorted(paths.items())), bool(request.get("color_correction", True)))
        if self.model is not None and self.key == key:
            return True
        self.unload()
        factory = self._factory or FlashVSRModel
        self.model = factory(str(request["code"]), paths, color_correction=key[-1])
        self.key = key
        return False

    def generate(self, request: dict[str, Any], hooks: Any) -> dict[str, Any]:
        hooks.stage("load")
        warm = self._ensure(request)
        hooks.check_cancel()
        if hasattr(self.model, "reset_peak"):
            self.model.reset_peak()
        hooks.stage("upscale")
        result = process_segment(request, self.model, hooks.step, hooks.check_cancel)
        peak = getattr(self.model, "vram_peak_gb", None)
        return {**result, "warm": warm, "vram_peak_gb": peak() if callable(peak) else None}


def main() -> None:
    base = _base()
    base.strip_own_dir(__file__)
    from importlib.metadata import PackageNotFoundError, version as package_version  # noqa: PLC0415

    try:
        torch_version = package_version("torch")
    except PackageNotFoundError:
        torch_version = "?"
    base.serve(QualityEngine, engine_version=f"flashvsr-plus (torch {torch_version})")


if __name__ == "__main__":
    main()
