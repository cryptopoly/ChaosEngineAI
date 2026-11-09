#!/usr/bin/env python3
"""Measure speech models the way the catalog reports them (plan docs/plans/voice-models-wave2.md section 6).

Drives its own voice worker (the same ``voice_worker.py`` and ``voice_speak.render`` the backend uses, through a
private ``VoiceWorker`` with its own log), never a running app and never a network port. Weights must already be
on disk; nothing is downloaded. For each row, on a fresh worker:

* ``coldStartSeconds``: wall time of the first short line minus the same line warm (the load cost).
* ``wallSecondsShortLine``: one 12-word line, warm.
* ``secondsPer10sAudio``: the fixed 600-character text below, spoken in the backend's chunks; wall seconds
  divided by output seconds, times 10.
* ``peakGb`` / ``idleGb``: the worker's resident memory (psutil RSS) sampled every 100 ms during the long text,
  and 5 s after it ends.

``--spike`` also answers the plan's open checks and writes WAVs for listening to ``--out`` (default
``~/.chaosengine/voice/wave2-listening``): seed repeatability, Turbo tags, VoxCPM2 design with continuation and
both cloning modes next to Qwen3-TTS Base, Chatterbox V3 with the V2 default voice and three languages, Orpheus
voices and its token budget. Cloned clips are made from a Kokoro voice (synthetic, no person's voice) and get our
AudioSeal watermark, as every cloned clip does in the app.

    .venv/bin/python scripts/measure-voice-models.py --rows chatterbox-turbo voxcpm2 --json out.json
    .venv/bin/python scripts/measure-voice-models.py --spike
    .venv/bin/python scripts/measure-voice-models.py --chat --rows kokoro qwen3-tts-custom-1.7b chatterbox-turbo voxcpm2

``--chat`` measures what Chat read-aloud and voice calls feel (D-67 W6): time to the first audio of one chat
sentence on a fresh worker (``chatFirstAudioColdS``, worker start and model load included) and warm (median of
three, ``chatFirstAudioS``), and the worker's peak resident memory over those runs (``chatPeakGb``). ``kokoro``
is accepted as a row key here (it has no catalog row).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHORT_LINE = "The quick brown fox jumps over the lazy dog near the river bank."
# 600 characters of plain English, the same text for every model.
LONG_TEXT = (
    "Every morning the harbour town wakes slowly. Fishing boats return with the tide, gulls circle the market, "
    "and the bakery on the corner opens its shutters to a short queue of early customers. By nine o'clock the "
    "streets are busy with delivery vans and children walking to school. Visitors arrive later, drawn by the old "
    "lighthouse and the cliff path that winds north along the coast. In the afternoon the wind picks up and the "
    "sailing club launches its small dinghies into the bay. When evening comes, lights glow in the windows above "
    "the shops, and the sound of the sea fills the quiet lanes once again."
)
GB = 1e9


# ---------------------------------------------------------------------------
# Pure helpers (tested in tests/test_measure_voice_models.py)
# ---------------------------------------------------------------------------

def seconds_per_10s(wall_s: float, audio_s: float) -> float:
    """Wall seconds needed for 10 s of speech."""
    if audio_s <= 0:
        raise ValueError("No audio was produced.")
    return round(wall_s / audio_s * 10.0, 2)


def cold_start(first_wall_s: float, warm_wall_s: float) -> float:
    """The load cost: the first request's wall time minus the same request warm (never below zero)."""
    return round(max(0.0, first_wall_s - warm_wall_s), 1)


def estimates(*, measured_on: str, cold: float, per_10s: float, short: float, peak_gb: float, idle_gb: float) -> dict[str, Any]:
    """The catalog's ``estimates`` block."""
    return {
        "basis": "measured",
        "measuredOn": measured_on,
        "coldStartSeconds": round(cold, 1),
        "secondsPer10sAudio": round(per_10s, 2),
        "wallSecondsShortLine": round(short, 1),
        "peakGb": round(peak_gb, 2),
        "idleGb": round(idle_gb, 2),
    }


class RssSampler:
    """Peak resident memory of a process (and its children) sampled every ``interval`` seconds."""

    def __init__(self, pid: int, interval: float = 0.1, reader: Any = None) -> None:
        self.pid = pid
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = reader or rss_bytes

    def sample(self) -> int:
        value = self._reader(self.pid)
        self.peak = max(self.peak, value)
        return value

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "RssSampler":
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()


def rss_bytes(pid: int) -> int:
    import psutil  # noqa: PLC0415

    process = psutil.Process(pid)
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def machine_label(today: str | None = None) -> str:
    """"Apple M4 Max, 64 GB, 2026-11-09" on a Mac; the platform name elsewhere."""
    chip = platform.processor() or platform.machine()
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip() or chip
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        import psutil  # noqa: PLC0415

        memory = f", {round(psutil.virtual_memory().total / 2**30)} GB"
    except ImportError:
        memory = ""
    return f"{chip}{memory}, {today or time.strftime('%Y-%m-%d')}"


# ---------------------------------------------------------------------------
# Driving the worker
# ---------------------------------------------------------------------------

class Bench:
    """A private voice worker that ``voice_speak.render`` talks to instead of the backend's."""

    def __init__(self) -> None:
        from backend_service import voice_client  # noqa: PLC0415

        self.client = voice_client
        self.worker = voice_client.VoiceWorker(log_name="measure-worker.log")
        self.worker.idle_minutes = 0
        voice_client.get_worker = lambda: self.worker  # render() looks this up at call time

    def fresh(self) -> None:
        self.worker.stop()

    def pid(self) -> int:
        if self.worker.pid is None:
            self.worker.request("ping", timeout=120)
        assert self.worker.pid is not None
        return self.worker.pid

    def render(self, plan: Any, text: str, seed: int | None = None) -> tuple[Any, float]:
        from backend_service import voice_speak  # noqa: PLC0415

        started = time.perf_counter()
        rendered = voice_speak.render(plan, text, seed=seed)
        return rendered, time.perf_counter() - started

    def close(self) -> None:
        self.worker.stop()


def measure(bench: Bench, plan: Any, label: str) -> dict[str, Any]:
    bench.fresh()
    pid = bench.pid()
    _r, first = bench.render(plan, SHORT_LINE)
    _r, warm = bench.render(plan, SHORT_LINE)
    with RssSampler(pid) as sampler:
        rendered, wall = bench.render(plan, LONG_TEXT)
    time.sleep(5)
    idle = rss_bytes(pid)
    return estimates(
        measured_on=label, cold=cold_start(first, warm), per_10s=seconds_per_10s(wall, rendered.duration_s),
        short=warm, peak_gb=sampler.peak / GB, idle_gb=idle / GB,
    ) | {"chunks": rendered.chunks, "audioSeconds": rendered.duration_s}


def plan_for_key(key: str, *, reference: Path | None = None, reference_text: str = "") -> Any:
    """The plan the backend would build for a row's typical use (preset, built-in, design or clone)."""
    from backend_service import voice_clone, voice_speak  # noqa: PLC0415
    from backend_service.catalog.voice_models import voice_model  # noqa: PLC0415

    row = voice_model(key)
    if row["family"] == "chatterbox-turbo":
        return voice_speak.plan_for("preset", "chatterbox-turbo", row, {})
    if row["family"] == "orpheus":
        return voice_speak.plan_for("preset", "orpheus:tara", row, {"speaker": "tara"})
    if row["family"] == "voxcpm2" or row["kind"] == "design":
        extra = {"mode": "design", "instruct": "A calm woman in her thirties, clear and warm, at an easy pace."}
        if row["family"] == "qwen3-tts":
            extra["language"] = "auto"
        return voice_speak.plan_for("design", "design", row, extra)
    if row["family"] == "chatterbox" and row["capabilities"]["speak"]:
        return voice_speak.plan_for("preset", "chatterbox", row, voice_speak.chatterbox_fields(row, "en", 0.5, 0.5))
    if row["kind"] == "custom":
        return voice_speak.plan_for("preset", "qwen3:Ryan", row, {"mode": "preset", "speaker": "Ryan", "language": "auto"})
    if reference is None:
        raise ValueError(f"{key} clones only; pass a reference clip.")
    extra, _notes = voice_clone.clone_fields(row, str(reference), reference_text, "", "en", 0.5, 0.5)
    return voice_speak.plan_for("clone", "clone:bench", row, extra)


CHAT_SENTENCES = (
    "Sure, here is a short answer to your question about the weather tomorrow.",
    "It should stay dry in the morning, with some light rain after lunch.",
    "Take a jacket if you plan to be outside in the evening.",
    "Let me know if you want the forecast for the rest of the week.",
)


def chat_estimates(*, measured_on: str, cold_s: float, warm_s: list[float], peak_gb: float) -> dict[str, Any]:
    """The ``chat`` part of a row's ``estimates``: first audio of one chat sentence, cold and warm (median)."""
    ordered = sorted(warm_s)
    median = ordered[len(ordered) // 2]
    return {
        "chatMeasuredOn": measured_on,
        "chatFirstAudioColdS": round(cold_s, 1),
        "chatFirstAudioS": round(median, 2),
        "chatPeakGb": round(peak_gb, 2),
    }


def chat_plan(key: str) -> Any:
    """The plan a chat voice runs: Kokoro's ``af_heart``, or the row's usual voice."""
    from backend_service import voice_speak  # noqa: PLC0415

    if key == "kokoro":
        return voice_speak.Plan("kokoro", "af_heart", "kokoro", "Kokoro", native_speed=True)
    return plan_for_key(key)


def measure_chat(bench: Bench, plan: Any, label: str) -> dict[str, Any]:
    """Fresh worker: the first sentence is the cold time, the next three the warm ones."""
    bench.fresh()
    started = time.perf_counter()
    pid = bench.pid()  # the worker starting is part of the cold time a person waits through
    with RssSampler(pid) as sampler:
        _r, first = bench.render(plan, CHAT_SENTENCES[0])
        cold = time.perf_counter() - started
        warm = [bench.render(plan, sentence)[1] for sentence in CHAT_SENTENCES[1:]]
    del first
    return chat_estimates(measured_on=label, cold_s=cold, warm_s=warm, peak_gb=sampler.peak / GB)


REFERENCE_TEXT = (
    "Good morning. This is a short sample of a calm, clear voice, recorded so that a speech model can learn "
    "how it sounds."
)


def kokoro_reference(bench: Bench, folder: Path) -> Path:
    """A synthetic voice sample (Kokoro af_heart): no person's voice is used for the cloning checks."""
    from backend_service import voice_speak  # noqa: PLC0415

    rendered, _wall = bench.render(voice_speak.build_plan("af_heart"), REFERENCE_TEXT)
    path = folder / "reference-kokoro-af_heart.wav"
    path.write_bytes(rendered.wav)
    return path


def watermark(bench: Bench, path: Path) -> None:
    from backend_service import voice_env  # noqa: PLC0415

    models = voice_env.watermark_model_paths()
    if models is None:
        raise RuntimeError("The watermark component is not installed; cloned clips are not written without it.")
    bench.worker.request("watermark", timeout=600.0, path=str(path), generator=str(models["generator"]),
                         detector=str(models["detector"]))


# ---------------------------------------------------------------------------
# Spike: the plan's open checks (P0)
# ---------------------------------------------------------------------------

COMPARE_TEXT = (
    "I have been meaning to tell you about the garden. The roses came back this year, brighter than ever, "
    "and the old apple tree finally gave us enough fruit for a pie."
)
# Median pitch per file (librosa pYIN, in the voice venv): a steady designed voice keeps it within a few percent from
# chunk to chunk, a voice designed again jumps (2026-11-09: 171-183 Hz continued against 149-243 Hz redesigned).
# Speaker-embedding similarity was tried first and swung between runs on the same files, so it is not used.
_PITCH_CODE = r"""
import json, sys
import librosa, numpy as np
out = {}
for path in json.loads(sys.argv[1]):
    y, sr = librosa.load(path, sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    out[path] = round(float(np.nanmedian(f0))) if np.any(voiced) else None
print(json.dumps(out))
"""


def median_pitch(files: list[Path]) -> dict[str, int | None]:
    from backend_service import voice_env  # noqa: PLC0415

    result = subprocess.run(
        [str(voice_env.venv_python()), "-I", "-c", _PITCH_CODE, json.dumps([str(f) for f in files])],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _wav_seconds(data: bytes) -> float:
    from backend_service import voice_speak  # noqa: PLC0415

    return round(voice_speak.wav_info(data)[3], 2)


def _worker_wav(bench: Bench, plan: Any, text: str, **extra: Any) -> bytes:
    import base64  # noqa: PLC0415

    fields = {**plan.extra, **extra}
    reply = bench.worker.request("synthesize", timeout=900.0, engine=plan.engine, source=plan.source, text=text, **fields)
    return base64.b64decode(reply["wavB64"])


def run_spike(bench: Bench, out: Path, reference: Path, reference_text: str, keys: list[str]) -> dict[str, Any]:
    from backend_service import voice_clone, voice_speak  # noqa: PLC0415
    from backend_service.catalog.voice_models import ORPHEUS_VOICES, voice_model  # noqa: PLC0415

    report: dict[str, Any] = {}

    def save(name: str, data: bytes, cloned: bool = False) -> Path:
        path = out / name
        path.write_bytes(data)
        if cloned:
            watermark(bench, path)
        return path

    def guarded(name: str, call: Any) -> None:
        try:
            report[name] = call()
        except Exception as exc:  # noqa: BLE001 - recorded, the rest goes on
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print("spike", name, json.dumps(report[name])[:600], flush=True)

    def seeds() -> dict[str, Any]:
        same = {}
        for key in keys:
            if voice_model(key)["family"] == "qwen3-tts":
                continue
            plan = plan_for_key(key, reference=reference, reference_text=reference_text)
            first, _ = bench.render(plan, SHORT_LINE, seed=7)
            second, _ = bench.render(plan, SHORT_LINE, seed=7)
            same[key] = first.wav == second.wav
        return same

    def turbo() -> dict[str, Any]:
        plan = plan_for_key("chatterbox-turbo")
        lines = {
            "turbo-tags.wav": "Well [laugh] that is the funniest thing I have heard all week. [sigh] Anyway, back to work.",
            "turbo-no-tags.wav": "Well that is the funniest thing I have heard all week. Anyway, back to work.",
            "turbo-style-whispering.wav": "[whispering] This part is a secret, so keep it between us.",
            "turbo-style-angry.wav": "[angry] I told you three times already, and nobody listened.",
        }
        seconds = {}
        for name, text in lines.items():
            rendered, wall = bench.render(plan, text)
            save(name, rendered.wav)
            seconds[name] = {"audio": rendered.duration_s, "wall": round(wall, 2)}
        return seconds

    def voxcpm_continuation() -> dict[str, Any]:
        plan = plan_for_key("voxcpm2")
        text = LONG_TEXT + " " + LONG_TEXT
        chunks = voice_speak.prepare_text(text, plan)[0]
        first = _worker_wav(bench, plan, chunks[0])
        first_path = save("voxcpm2-design-chunk1.wav", first)
        held, fresh = [], []
        for index, chunk in enumerate(chunks[1:6], start=2):
            cont = _worker_wav(bench, plan, chunk, mode="continue", instruct=None, continueAudioPath=str(first_path),
                               continueText=chunks[0])
            held.append(save(f"voxcpm2-design-chunk{index}-continued.wav", cont))
            again = _worker_wav(bench, plan, chunk)
            fresh.append(save(f"voxcpm2-design-chunk{index}-redesigned.wav", again))
        pitch = median_pitch([first_path, *held, *fresh])
        result: dict[str, Any] = {
            "chunks": len(chunks),
            "medianPitchContinued": [pitch[str(p)] for p in (first_path, *held)],
            "medianPitchRedesigned": [pitch[str(p)] for p in (first_path, *fresh)],
        }
        rendered, wall = bench.render(plan, text)
        save("voxcpm2-design-1200-chars.wav", rendered.wav)
        result.update({"audio": rendered.duration_s, "wall": round(wall, 2)})
        return result

    def clone_comparison() -> dict[str, Any]:
        timings: dict[str, Any] = {}
        cases = [
            ("qwen3-tts-0.6b", "", "clone-qwen3-0.6b-base.wav"),
            ("qwen3-tts-1.7b", "", "clone-qwen3-1.7b-base.wav"),
            ("voxcpm2", "", "clone-voxcpm2-exact.wav"),
            ("voxcpm2", "speaking cheerfully, a little faster", "clone-voxcpm2-steered.wav"),
            ("chatterbox-turbo", "", "clone-chatterbox-turbo.wav"),
            ("chatterbox-mtl", "", "clone-chatterbox-v3.wav"),
        ]
        for key, style, name in cases:
            row = voice_model(key)
            if voice_clone.snapshot_dir(row) is None:
                timings[name] = "not downloaded"
                continue
            try:
                extra, _notes = voice_clone.clone_fields(row, str(reference), reference_text, style, "en", 0.5, 0.5)
                plan = voice_speak.plan_for("clone", "clone:bench", row, extra)
                bench.fresh()
                rendered, wall = bench.render(plan, COMPARE_TEXT, seed=11)
                save(name, rendered.wav, cloned=True)
                timings[name] = {"audio": rendered.duration_s, "wall": round(wall, 2)}
            except Exception as exc:  # noqa: BLE001
                timings[name] = {"error": f"{type(exc).__name__}: {exc}"}
        # After cloning, Turbo's built-in voice must be its own again (the clip's conditionals are put back).
        turbo = plan_for_key("chatterbox-turbo")
        after = save("turbo-builtin-after-clone.wav", _worker_wav(bench, turbo, SHORT_LINE, seed=7))
        before_plan = plan_for_key("chatterbox-turbo")
        bench.fresh()
        before = save("turbo-builtin-fresh.wav", _worker_wav(bench, before_plan, SHORT_LINE, seed=7))
        timings["turboBuiltinRestored"] = before.read_bytes() == after.read_bytes()
        return timings

    def chatterbox_v3() -> dict[str, Any]:
        row = voice_model("chatterbox-mtl")
        conds = Path(voice_clone._revision_dir("mlx-community/chatterbox-8bit", "9617d61b596a03d1bed766a28c341680e993a1b9")) / "conds.safetensors"
        result: dict[str, Any] = {"v2CondsPresent": conds.is_file()}
        plan = voice_speak.plan_for("preset", "chatterbox", row, {
            "language": "en", "expression": 0.5, "guidance": 0.5, "defaultVoicePath": str(conds),
        })
        bench.fresh()
        try:
            rendered, wall = bench.render(plan, SHORT_LINE)
            save("chatterbox-v3-default-voice-en.wav", rendered.wav)
            result["defaultVoiceWorks"] = rendered.duration_s > 1.0
            result["defaultVoice"] = {"audio": rendered.duration_s, "wall": round(wall, 2)}
        except Exception as exc:  # noqa: BLE001
            result["defaultVoiceWorks"] = False
            result["defaultVoiceError"] = f"{type(exc).__name__}: {exc}"
            return result
        lines = {"de": "Guten Morgen, heute scheint die Sonne über der ganzen Stadt.",
                 "ja": "おはようございます。今日は町じゅうに日が差しています。"}
        for code, text in lines.items():
            p = voice_speak.plan_for("preset", "chatterbox", row, {**plan.extra, "language": code})
            rendered, _ = bench.render(p, text)
            save(f"chatterbox-v3-default-voice-{code}.wav", rendered.wav)
            result[code] = rendered.duration_s
        for value in (0.3, 0.5, 0.8):
            p = voice_speak.plan_for("preset", "chatterbox", row, {**plan.extra, "expression": value})
            rendered, _ = bench.render(p, COMPARE_TEXT, seed=5)
            save(f"chatterbox-v3-expression-{value}.wav", rendered.wav)
            result[f"expression{value}"] = rendered.duration_s
        return result

    def orpheus() -> dict[str, Any]:
        row = voice_model("orpheus-3b")
        result: dict[str, Any] = {}
        for voice in ORPHEUS_VOICES:
            plan = voice_speak.plan_for("preset", f"orpheus:{voice['id']}", row, {"speaker": voice["id"]})
            rendered, _ = bench.render(plan, "Hi there [laugh], it is lovely to finally meet you. [sigh] What a week.")
            save(f"orpheus-{voice['id']}.wav", rendered.wav)
            result[voice["id"]] = rendered.duration_s
        chunk = (LONG_TEXT[:400]).rsplit(" ", 1)[0] + "."
        plan = voice_speak.plan_for("preset", "orpheus:tara", row, {"speaker": "tara"})
        budget = {}
        for tokens in (1200, 2400, 4000):
            data = _worker_wav(bench, plan, chunk, maxTokens=tokens)
            save(f"orpheus-400-chars-{tokens}-tokens.wav", data)
            budget[tokens] = _wav_seconds(data)
        result["chars"] = len(chunk)
        result["secondsByMaxTokens"] = budget
        return result

    guarded("seedRepeatable", seeds)
    if "chatterbox-turbo" in keys:
        guarded("turbo", turbo)
    if "voxcpm2" in keys:
        guarded("voxcpm2Continuation", voxcpm_continuation)
    guarded("chatterboxV3", chatterbox_v3)
    if "orpheus-3b" in keys:
        guarded("orpheus", orpheus)
    guarded("cloneComparison", clone_comparison)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="*", help="catalog keys (default: every row whose weights are on disk)")
    parser.add_argument("--json", type=Path, help="write the results here")
    parser.add_argument("--spike", action="store_true", help="answer the open checks and write WAVs to --out")
    parser.add_argument("--out", type=Path, default=Path.home() / ".chaosengine" / "voice" / "wave2-listening")
    parser.add_argument("--chat", action="store_true", help="time to first audio of one chat sentence, cold and warm")
    args = parser.parse_args(argv)
    if args.chat:
        return main_chat(args)

    from backend_service import voice_clone  # noqa: PLC0415
    from backend_service.catalog.voice_models import VOICE_MODELS  # noqa: PLC0415

    keys = args.rows or [row["key"] for row in VOICE_MODELS if voice_clone.snapshot_dir(row) is not None]
    label = machine_label()
    bench = Bench()
    args.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"machine": label, "rows": {}}
    try:
        reference = kokoro_reference(bench, args.out)
        for key in keys:
            try:
                plan = plan_for_key(key, reference=reference, reference_text=REFERENCE_TEXT)
                results["rows"][key] = measure(bench, plan, label)
            except Exception as exc:  # noqa: BLE001 - one failing model must not hide the others
                results["rows"][key] = {"error": f"{type(exc).__name__}: {exc}"}
            print(key, json.dumps(results["rows"][key]), flush=True)
        if args.spike:
            results["spike"] = run_spike(bench, args.out, reference, REFERENCE_TEXT, keys)
    finally:
        bench.close()
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def main_chat(args: argparse.Namespace) -> int:
    from backend_service import voice_clone  # noqa: PLC0415
    from backend_service.catalog.voice_models import VOICE_MODELS  # noqa: PLC0415

    keys = args.rows or ["kokoro", *(row["key"] for row in VOICE_MODELS
                                     if row["capabilities"]["speak"] and voice_clone.snapshot_dir(row) is not None)]
    label = machine_label()
    bench = Bench()
    results: dict[str, Any] = {"machine": label, "chat": {}}
    try:
        for key in keys:
            try:
                results["chat"][key] = measure_chat(bench, chat_plan(key), label)
            except Exception as exc:  # noqa: BLE001 - one failing model must not hide the others
                results["chat"][key] = {"error": f"{type(exc).__name__}: {exc}"}
            print(key, json.dumps(results["chat"][key]), flush=True)
    finally:
        bench.close()
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
