"""Cloned-voice watermark (AudioSeal, D-47): message bits, embed and detect plumbing with stand-in models,
and a real embed -> MP3 / AAC re-encode -> detect round trip in the voice venv when AudioSeal is installed."""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from backend_service import voice_env, voice_worker_watermark as wm

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover - the dev venv has both
    np = None
    torch = None


class MessageTests(unittest.TestCase):
    def test_fixed_message_round_trips(self) -> None:
        bits = wm.message_bits()
        self.assertEqual(len(bits), wm.NBITS)
        self.assertEqual(bits[:4], [1, 1, 0, 0])  # 0xC...
        self.assertEqual(wm.bits_value(bits), 0xCE5A)

    def test_pinned_weights_and_packages(self) -> None:
        self.assertRegex(voice_env.WATERMARK_REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(voice_env.WATERMARK_REPO, "facebook/audioseal")
        self.assertIn("audioseal==0.2.1", voice_env.WATERMARK_PACKAGES)
        self.assertEqual(set(voice_env.WATERMARK_FILES), {"generator", "detector"})


@unittest.skipIf(torch is None, "torch not installed")
class PlumbingTests(unittest.TestCase):
    def test_watermark_keeps_length_and_rate_and_adds_the_signal(self) -> None:
        seen = {}

        class Generator(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def get_watermark(self, x, message=None):
                seen["shape"], seen["message"] = tuple(x.shape), message.tolist()
                return torch.full_like(x, 0.01)

        samples = np.zeros(24_000, dtype=np.float32)
        out = wm.watermark_samples(Generator(), samples, 24_000)
        self.assertEqual(out.shape, samples.shape)
        self.assertEqual(seen["shape"], (1, 1, 16_000))  # the generator sees 16 kHz
        self.assertEqual(seen["message"], [wm.message_bits()])
        self.assertAlmostEqual(float(np.median(out)), 0.01, places=3)

    def test_detect_reports_probability_and_message(self) -> None:
        class Detector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def detect_watermark(self, x):
                return torch.tensor([0.97]), torch.tensor([wm.message_bits()])

        found = wm.detect_samples(Detector(), np.zeros(16_000, dtype=np.float32), 16_000)
        self.assertEqual(found, {"probability": 0.97, "message": "CE5A", "matches": True})


_ROUND_TRIP = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import numpy as np, av
    import voice_worker as w
    import voice_worker_watermark as wm

    folder, gen, det = sys.argv[2], sys.argv[3], sys.argv[4]
    rate = 24000
    t = np.arange(rate * 4) / rate
    rng = np.random.default_rng(0)
    # Speech-like test signal: a pitch sweep with harmonics under a syllable envelope, plus a little noise.
    f0 = 120 + 40 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t)) ** 2
    clip = (0.2 * voice * env + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
    plain = folder + "/plain.wav"
    open(plain, "wb").write(w.wav_bytes(clip, rate))
    marked = folder + "/marked.wav"
    wm.embed({"path": plain, "out": marked, "generator": gen}, w.decode_audio, w.wav_bytes)

    def reencode(src, dst, fmt, codec):
        with av.open(src) as s, av.open(dst, "w", format=fmt) as d:
            st = d.add_stream(codec, rate=rate, layout="mono")
            st.bit_rate = 96000
            rs = av.AudioResampler(format=st.format.name, layout="mono", rate=rate)
            for fr in s.decode(audio=0):
                for o in rs.resample(fr):
                    for p in st.encode(o): d.mux(p)
            for o in rs.resample(None):
                for p in st.encode(o): d.mux(p)
            for p in st.encode(None): d.mux(p)
        return dst

    out = {}
    out["plain"] = wm.detect({"path": plain, "detector": det}, w.decode_audio)
    out["wav"] = wm.detect({"path": marked, "detector": det}, w.decode_audio)
    out["mp3"] = wm.detect({"path": reencode(marked, folder + "/m.mp3", "mp3", "libmp3lame"), "detector": det}, w.decode_audio)
    out["aac"] = wm.detect({"path": reencode(marked, folder + "/m.m4a", "mp4", "aac"), "detector": det}, w.decode_audio)
    a = w.decode_audio(plain, rate); b = w.decode_audio(marked, rate)
    n = min(len(a), len(b))
    out["snrDb"] = float(10 * np.log10(np.sum(a[:n] ** 2) / max(np.sum((a[:n] - b[:n]) ** 2), 1e-12)))
    open(folder + "/result.json", "w").write(json.dumps(out))
    """
)


def _real_watermark_ready() -> str | None:
    if not voice_env.venv_python().exists():
        return "voice venv not installed"
    if voice_env.watermark_model_paths() is None:
        return "AudioSeal weights not downloaded at the pinned revision"
    probe = subprocess.run([str(voice_env.venv_python()), "-c", "import audioseal, librosa, av"], capture_output=True)
    return None if probe.returncode == 0 else "audioseal not installed in the voice venv"


class RealRoundTripTests(unittest.TestCase):
    def test_embed_reencode_detect(self) -> None:
        reason = _real_watermark_ready()
        if reason:
            self.skipTest(reason)
        models = voice_env.watermark_model_paths()
        worker_dir = Path(wm.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "round_trip.py"
            script.write_text(_ROUND_TRIP)
            run = subprocess.run(
                [str(voice_env.venv_python()), str(script), str(worker_dir), folder, str(models["generator"]), str(models["detector"])],
                capture_output=True, text=True, timeout=600, cwd=folder,
            )
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            result = json.loads((Path(folder) / "result.json").read_text())
        self.assertFalse(result["plain"]["matches"] and result["plain"]["probability"] > 0.5, result)
        for name in ("wav", "mp3", "aac"):
            self.assertGreater(result[name]["probability"], 0.8, (name, result))
            self.assertTrue(result[name]["matches"], (name, result))
        self.assertGreater(result["snrDb"], 20, result)


if __name__ == "__main__":
    unittest.main()
