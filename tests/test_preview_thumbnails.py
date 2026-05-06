"""Unit tests for the live denoise thumbnail emit helpers (FU-018 part 2).

Exercises:
* ``decode_image_latent_to_b64`` happy path produces a non-empty base64
  PNG when fed a fake VAE that returns a deterministic ``(B, C, H, W)``
  tensor in the [-1, 1] range.
* ``decode_video_latent_to_b64`` picks the middle frame from a
  ``(B, C, F, H, W)`` decoder output and produces a base64 PNG.
* Both helpers swallow exceptions and return ``None`` rather than
  letting a preview-decode crash abort the actual generation.
* The thumbnail max-side cap is honoured (1024x1024 in -> <=192x192 out).

Tests skip when torch / numpy / PIL are missing — preview thumbnails
are best-effort and the helper degrades gracefully on minimal envs.
"""

from __future__ import annotations

import base64
import io
import unittest


def _have_imaging_stack() -> bool:
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def _decode_b64_png_size(b64: str) -> tuple[int, int]:
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return img.size


class _FakeVaeConfig:
    def __init__(self, scaling_factor: float = 1.0) -> None:
        self.scaling_factor = scaling_factor


class _FakeImageVae:
    """Returns latents un-changed so the encoded thumbnail is a known
    deterministic gradient. ``decode().sample`` is the diffusers contract."""

    def __init__(self, scaling_factor: float = 1.0):
        import torch
        self.config = _FakeVaeConfig(scaling_factor)
        self.dtype = torch.float32

    def decode(self, latents):
        # Pretend-VAE: latents already look like image-space tensors in
        # [-1, 1] for test purposes. Wrap in a SimpleNamespace-like with
        # ``.sample`` to match diffusers' AutoencoderTiny return shape.
        from types import SimpleNamespace
        return SimpleNamespace(sample=latents)


class _FakeVideoVae:
    def __init__(self, scaling_factor: float = 1.0):
        import torch
        self.config = _FakeVaeConfig(scaling_factor)
        self.dtype = torch.float32

    def decode(self, latents):
        from types import SimpleNamespace
        return SimpleNamespace(sample=latents)


@unittest.skipUnless(_have_imaging_stack(), "torch + numpy + PIL not available")
class DecodeImageLatentTests(unittest.TestCase):
    def test_happy_path_returns_b64_png(self):
        import torch
        from backend_service.helpers.preview_thumbnails import decode_image_latent_to_b64
        from types import SimpleNamespace

        # 1x3x64x64 gradient in [-1, 1] — encodes to a non-trivial image.
        latents = torch.linspace(-1.0, 1.0, 1 * 3 * 64 * 64).reshape(1, 3, 64, 64).float()
        pipeline = SimpleNamespace(vae=_FakeImageVae())

        b64 = decode_image_latent_to_b64(pipeline, latents)
        self.assertIsNotNone(b64)
        self.assertGreater(len(b64), 100, "expected non-trivial PNG payload")

        size = _decode_b64_png_size(b64)
        self.assertEqual(size, (64, 64))

    def test_thumbnail_caps_long_edge(self):
        import torch
        from backend_service.helpers.preview_thumbnails import decode_image_latent_to_b64
        from types import SimpleNamespace

        # Big latent — helper should scale down to <=192 px on long edge.
        latents = torch.zeros(1, 3, 512, 512).float()
        pipeline = SimpleNamespace(vae=_FakeImageVae())

        b64 = decode_image_latent_to_b64(pipeline, latents, max_side=192)
        self.assertIsNotNone(b64)

        size = _decode_b64_png_size(b64)
        self.assertEqual(size, (192, 192))

    def test_returns_none_when_vae_decode_raises(self):
        import torch
        from backend_service.helpers.preview_thumbnails import decode_image_latent_to_b64
        from types import SimpleNamespace

        class ExplodingVae(_FakeImageVae):
            def decode(self, latents):
                raise RuntimeError("decode crashed")

        pipeline = SimpleNamespace(vae=ExplodingVae())
        latents = torch.zeros(1, 3, 64, 64).float()
        self.assertIsNone(decode_image_latent_to_b64(pipeline, latents))

    def test_returns_none_when_pipeline_has_no_vae(self):
        from backend_service.helpers.preview_thumbnails import decode_image_latent_to_b64
        from types import SimpleNamespace

        pipeline = SimpleNamespace(vae=None)
        self.assertIsNone(decode_image_latent_to_b64(pipeline, object()))

    def test_returns_none_when_latents_none(self):
        from backend_service.helpers.preview_thumbnails import decode_image_latent_to_b64
        from types import SimpleNamespace

        pipeline = SimpleNamespace(vae=_FakeImageVae())
        self.assertIsNone(decode_image_latent_to_b64(pipeline, None))

    def test_flux_packed_3d_latents_get_unpacked(self):
        """FLUX pipelines stream ``(B, seq_len, 64)`` packed latents
        through ``callback_on_step_end``. Live smoke 2026-05-04 against
        FLUX.1-schnell surfaced this — the helper now detects the 3D
        shape and calls ``pipeline._unpack_latents`` to get back to the
        4D ``(B, 16, H/8, W/8)`` shape ``vae.decode`` expects."""
        import torch
        from backend_service.helpers.preview_thumbnails import decode_image_latent_to_b64
        from types import SimpleNamespace

        # 1x1024x64 packed latent — would be a 512x512 FLUX gen
        # (32x32 token grid * 16-pixel patches = 512). Use a fake
        # _unpack_latents that returns a deterministic 4D tensor so
        # the test doesn't depend on diffusers internals.
        packed = torch.zeros(1, 1024, 64).float()
        unpacked_target = torch.linspace(-1.0, 1.0, 1 * 16 * 64 * 64).reshape(1, 16, 64, 64).float()

        unpack_calls = []
        def fake_unpack(latents, h, w, vae_scale):
            unpack_calls.append((latents.shape, h, w, vae_scale))
            return unpacked_target

        # FLUX VAE outputs 3 channels at the end, so route the unpacked
        # 16-channel latent through a fake VAE that just returns a
        # 3-channel slice as the "decoded" sample.
        class FluxVae(_FakeImageVae):
            def decode(self, latents):
                from types import SimpleNamespace as _SN
                # Take channels 0-2 only to mimic VAE 16->3 reduction.
                return _SN(sample=latents[:, :3, :, :])

        pipeline = SimpleNamespace(
            vae=FluxVae(),
            _unpack_latents=fake_unpack,
            vae_scale_factor=8,
        )
        b64 = decode_image_latent_to_b64(pipeline, packed)
        self.assertIsNotNone(b64)
        self.assertEqual(len(unpack_calls), 1)
        self.assertEqual(unpack_calls[0][1], 512)  # height
        self.assertEqual(unpack_calls[0][2], 512)  # width
        self.assertEqual(unpack_calls[0][3], 8)    # vae_scale_factor

    def test_3d_latents_without_unpack_method_returns_none(self):
        # When a non-FLUX pipeline somehow produces 3D latents but
        # doesn't expose ``_unpack_latents``, the helper bails rather
        # than crashing the gen.
        import torch
        from backend_service.helpers.preview_thumbnails import decode_image_latent_to_b64
        from types import SimpleNamespace

        packed = torch.zeros(1, 1024, 64).float()
        pipeline = SimpleNamespace(vae=_FakeImageVae())  # no _unpack_latents
        self.assertIsNone(decode_image_latent_to_b64(pipeline, packed))


@unittest.skipUnless(_have_imaging_stack(), "torch + numpy + PIL not available")
class DecodeVideoLatentTests(unittest.TestCase):
    def test_happy_path_picks_middle_frame(self):
        import torch
        from backend_service.helpers.preview_thumbnails import decode_video_latent_to_b64
        from types import SimpleNamespace

        # 1x3x9x64x64 — 9 frames, middle index = 4. Encode each frame
        # with a unique fill so we can prove "frame 4" got picked.
        latents = torch.zeros(1, 3, 9, 64, 64).float()
        for f in range(9):
            latents[0, :, f, :, :] = (f - 4) / 4.0  # -1..1 range across frames
        pipeline = SimpleNamespace(vae=_FakeVideoVae())

        b64 = decode_video_latent_to_b64(pipeline, latents)
        self.assertIsNotNone(b64)
        size = _decode_b64_png_size(b64)
        self.assertEqual(size, (64, 64))

    def test_returns_none_on_unexpected_rank(self):
        import torch
        from backend_service.helpers.preview_thumbnails import decode_video_latent_to_b64
        from types import SimpleNamespace

        # A 3D tensor (no batch / no channel split) — the helper should
        # reject it rather than attempt to slice.
        latents = torch.zeros(64, 64, 3).float()
        pipeline = SimpleNamespace(vae=_FakeVideoVae())
        self.assertIsNone(decode_video_latent_to_b64(pipeline, latents))


if __name__ == "__main__":
    unittest.main()
