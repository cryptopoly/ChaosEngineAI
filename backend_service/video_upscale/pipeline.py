"""The frame pipeline: decode with ffmpeg to a pipe, upscale in batches, encode segment files, then join and put the
original sound back.

A job is cut into segments of ``SEGMENT_SECONDS``. Each segment is decoded on its own (``-ss`` to just before its
first frame, a whole number of frames), upscaled, scaled with lanczos to the exact target when the model's 4x is not
the size asked for, and encoded to ``seg-NNNNN.mp4`` (H.264 at the Storyboard's final quality, yuv420p, 8-bit, the
source frame rate; the encoder is the one ``ffmpeg_bin`` picks, libx264 when the build has it). A JSON manifest in
the job's work folder lists the finished segments, so a job that was cancelled, crashed or lost power restarts at the
first unfinished segment. The last step joins the segments with the concat demuxer and copies the source's audio,
subtitle and chapter streams back in one mux (audio becomes AAC only when the container refuses a copy).

``process_segment`` runs where the model is (the worker process; tests run it in-process with a fake upscaler) and
needs only numpy and ffmpeg. ``run_job`` runs in the backend and hands each segment to a ``runner``.

Pre-steps: only deinterlacing (``bwdif``, LGPL in FFmpeg, checked in the binary's filter list before a job starts).
``yadif`` is not used (its licence in the pinned build was not confirmed), nor any denoise filter (the GPL-only ones
are absent from the pinned LGPL build and the others need tuning per source).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend_service.video_upscale.engine import UpscaleError, check_target

SEGMENT_SECONDS = 10.0
# A last segment shorter than the encoder's frame reordering has no reorder delay of its own, so the joined file gets
# a timestamp gap at the seam and a player repeats frames to fill it (a 21-frame film came back as 23). Such a tail is
# folded into the segment before it.
MIN_TAIL_FRAMES = 16
MANIFEST_VERSION = 1
PROTOCOLS = "file,pipe"  # same as storyboard.ffmpeg_bin.PROTOCOLS: local files and pipes only
DEINTERLACE_FILTER = "bwdif"
# Frame rates stored as fractions so an hour at 29.97 does not drift against the copied sound.
_NTSC = ((23.976, "24000/1001"), (29.97, "30000/1001"), (47.952, "48000/1001"), (59.94, "60000/1001"),
         (119.88, "120000/1001"))
_SAR = re.compile(r"\[SAR (\d+):(\d+) DAR")
_FIELD_ORDER = re.compile(r"[(,]\s*(progressive|(?:top|bottom)(?: coded)? first|tt|bb|tb|bt)(?=[,)\s])", re.IGNORECASE)
_TBR = re.compile(r"Video: .*?, (\d+(?:\.\d+)?) tbr")
_FRAME_COUNT = re.compile(r"frame=\s*(\d+)")


class Cancelled(Exception):
    """The job was cancelled; finished segments stay on disk for a resume."""


# ---------------------------------------------------------------------------
# the source
# ---------------------------------------------------------------------------


def fps_text(fps: float) -> str:
    for nominal, text in _NTSC:
        if abs(fps - nominal) < 0.01:
            return text
    if abs(fps - round(fps)) < 0.01:
        return str(int(round(fps)))
    return f"{fps:.6g}"


def fps_value(text: str) -> float:
    num, _, den = text.partition("/")
    return float(num) / float(den or 1)


@dataclass
class SourceInfo:
    width: int
    height: int
    fps: str  # a fraction or a whole number, as ffmpeg takes it
    duration: float
    frames: int  # the video stream's own frame count; duration x fps only when that count could not be read
    has_audio: bool
    sar: str = "1/1"
    frames_exact: bool = False  # ``frames`` was counted (the container's duration also covers a longer sound track)


def probe_source(exe: str, path: Path) -> SourceInfo:
    return _probe(exe, path)[0]


def probe_details(exe: str, path: Path) -> tuple[SourceInfo, bool | None]:
    """The source and whether its first video stream is interlaced (``None`` when ffmpeg names no field order)."""
    info, text = _probe(exe, path)
    flagged = interlaced_from_probe(text)
    return info, flagged if flagged is not None else interlaced_from_idet(exe, path)


_IDET = re.compile(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)")
IDET_FRAMES = 150
IDET_SHARE = 0.8  # combed frames needed, out of those it could classify, before the file is called interlaced


def interlaced_from_idet(exe: str, path: Path) -> bool | None:
    """For a file that names no field order: ffmpeg's ``idet`` looks at the first frames (``None`` when it cannot tell)."""
    try:
        run = subprocess.run([exe, "-hide_banner", "-nostdin", "-protocol_whitelist", PROTOCOLS, "-i", str(path), "-an",
                              "-vf", "idet", "-frames:v", str(IDET_FRAMES), "-f", "null", "-"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    found = _IDET.findall(run.stderr)  # the last match is the whole run
    if not found:
        return None
    tff, bff, progressive = (int(n) for n in found[-1])
    seen = tff + bff + progressive
    return None if seen == 0 else (tff + bff) / seen >= IDET_SHARE


def interlaced_from_probe(stderr: str) -> bool | None:
    """Field order from the video stream line of ``ffmpeg -i`` (``top first``, ``bottom coded first``, ``progressive``)."""
    for line in stderr.splitlines():
        if "Video:" not in line or "(attached pic)" in line:
            continue
        match = _FIELD_ORDER.search(line)
        if match:
            return match.group(1).lower() != "progressive"
        return None
    return None


def count_video_frames(exe: str, path: Path) -> int | None:
    """The frames in the first video stream, counted from its packets (``-c copy`` to the null muxer: no decoding, so
    it takes seconds, not the length of the film). ``None`` when ffmpeg gives no count. The app has no ffprobe."""
    try:
        run = subprocess.run([exe, "-hide_banner", "-nostdin", "-protocol_whitelist", PROTOCOLS, "-i", str(path),
                              "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    found = _FRAME_COUNT.findall(run.stderr)  # the progress line repeats; the last one is the whole file
    count = int(found[-1]) if run.returncode == 0 and found else 0
    return count or None


def _probe(exe: str, path: Path) -> tuple[SourceInfo, str]:
    from backend_service.storyboard.ffmpeg_bin import parse_probe  # noqa: PLC0415

    run = subprocess.run([exe, "-hide_banner", "-nostdin", "-protocol_whitelist", PROTOCOLS, "-i", str(path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    try:
        info = parse_probe(run.stderr)
    except ValueError as exc:
        raise UpscaleError(f"Could not read {path.name} as a video.") from exc
    if not info["has_video"] or not info["width"] or not info["fps"]:
        raise UpscaleError(f"{path.name} has no video stream ffmpeg can read (size and frame rate are needed).")
    sar = _SAR.search(run.stderr)
    # The stream's base rate (tbr) over its average (fps): an edit list or a variable-rate phone clip skews the
    # average (a 24 fps Storyboard shot reads "25.04 fps, 24 tbr"), and the fps filter makes the output constant anyway.
    tbr = _TBR.search(run.stderr)
    base = float(tbr.group(1)) if tbr and 1.0 <= float(tbr.group(1)) <= 240.0 else float(info["fps"])
    rate = fps_text(base)
    source = SourceInfo(width=int(info["width"]), height=int(info["height"]), fps=rate, duration=float(info["duration"]),
                        frames=max(1, round(float(info["duration"]) * fps_value(rate))), has_audio=bool(info["has_audio"]),
                        sar=f"{sar.group(1)}/{sar.group(2)}" if sar and sar.group(1) != "0" else "1/1")
    counted = count_video_frames(exe, path)
    if counted:
        source.frames, source.frames_exact = counted, True
    return source, run.stderr


# ---------------------------------------------------------------------------
# a job
# ---------------------------------------------------------------------------


@dataclass
class JobSpec:
    input: str
    output: str
    model_id: str
    weights: str
    tile: int
    batch: int
    width: int  # target
    height: int
    source: dict[str, Any]  # SourceInfo as a dict
    input_bytes: int
    input_mtime: float
    deinterlace: bool = False
    segment_frames: int = 0
    codec_args: list[str] | None = None
    avtb: bool = False

    def key(self) -> str:
        """The job id: a hash of everything that changes the result, so the same request finds its own manifest."""
        payload = {k: v for k, v in asdict(self).items() if k != "codec_args"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobSpec":
        return cls(**data)


def make_spec(exe: str, src: Path, dst: Path, *, model_id: str, weights: Path, size: tuple[int, int], tile: int = 0,
              batch: int = 1, deinterlace: bool = False, codec_args: list[str] | None = None,
              segment_seconds: float = SEGMENT_SECONDS, avtb: bool = False) -> JobSpec:
    check_target(*size)
    info = probe_source(exe, src)
    stat = src.stat()
    return JobSpec(input=str(src), output=str(dst), model_id=model_id, weights=str(weights), tile=int(tile),
                   batch=max(1, int(batch)), width=int(size[0]), height=int(size[1]), source=asdict(info),
                   input_bytes=stat.st_size, input_mtime=stat.st_mtime, deinterlace=bool(deinterlace),
                   segment_frames=max(1, round(segment_seconds * fps_value(info.fps))), codec_args=codec_args,
                   avtb=bool(avtb))


def plan_segments(total_frames: int, segment_frames: int, exact: bool = False) -> list[dict[str, Any]]:
    """``{index, start, frames, last}`` per segment. With an ``exact`` (counted) total the last segment is the
    remainder; otherwise it reads to the end of the file, since the total is only an estimate."""
    count = max(1, -(-total_frames // segment_frames))
    if exact and count > 1 and total_frames - (count - 1) * segment_frames < min(MIN_TAIL_FRAMES, segment_frames):
        count -= 1  # a tail this short joins the segment before it (see MIN_TAIL_FRAMES)
    return [{"index": i, "start": i * segment_frames,
             "frames": max(1, total_frames - i * segment_frames) if exact and i == count - 1 else segment_frames,
             "last": i == count - 1}
            for i in range(count)]


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


class Manifest:
    """``manifest.json`` in a job's work folder: the spec, the segments and which are finished."""

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path, self.data = path, data

    @classmethod
    def open(cls, work: Path, spec: JobSpec) -> "Manifest":
        path = work / "manifest.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") == MANIFEST_VERSION and data.get("key") == spec.key():
                return cls(path, data)
        except (OSError, ValueError, AttributeError):
            pass
        data = {"version": MANIFEST_VERSION, "key": spec.key(), "spec": spec.as_dict(), "state": "running",
                "segments": [{**seg, "done": False, "made": 0}
                             for seg in plan_segments(spec.source["frames"], spec.segment_frames,
                                                      bool(spec.source.get("frames_exact")))]}
        manifest = cls(path, data)
        manifest.save()
        return manifest

    @property
    def segments(self) -> list[dict[str, Any]]:
        return self.data["segments"]

    def segment_file(self, index: int) -> Path:
        return self.path.parent / f"seg-{index:05d}.mp4"

    def finished(self, seg: dict[str, Any]) -> bool:
        if not seg.get("done"):
            return False
        return seg.get("made", 0) == 0 or self.segment_file(seg["index"]).is_file()

    def mark(self, index: int, made: int) -> None:
        self.segments[index].update(done=True, made=int(made))
        self.save()

    def set_state(self, state: str, error: str | None = None) -> None:
        self.data["state"], self.data["error"] = state, error
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        os.replace(tmp, self.path)


def read_manifest(work: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("version") == MANIFEST_VERSION else None


# ---------------------------------------------------------------------------
# one segment (runs where the model is)
# ---------------------------------------------------------------------------


def decode_command(req: dict[str, Any]) -> list[str]:
    fps, start = str(req["fps"]), int(req["first_frame"])
    # The first kept frame starts the clock at 0, so the fps grid (which makes a variable-rate source constant) lands
    # on the frames themselves and never repeats the first one.
    filters = ([f"{DEINTERLACE_FILTER}=mode=send_frame"] if req.get("deinterlace") else []) + [
        "setpts=PTS-STARTPTS", f"fps={fps}"]
    command = [str(req["ffmpeg"]), "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", PROTOCOLS]
    if start > 0:  # half a frame early: accurate seek keeps frame ``start`` and drops the one before it
        command += ["-ss", f"{(start - 0.5) / fps_value(fps):.6f}"]
    command += ["-i", str(req["input"]), "-map", "0:v:0", "-vf", ",".join(filters), "-an", "-sn", "-dn"]
    if not req.get("last"):
        command += ["-frames:v", str(int(req["frames"]))]
    return command + ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]


def encode_command(req: dict[str, Any], model_size: tuple[int, int], output: Path) -> list[str]:
    width, height = int(req["width"]), int(req["height"])
    chain = [] if model_size == (width, height) else [f"scale={width}:{height}:flags=lanczos"]
    chain += [f"setsar={req.get('sar') or '1/1'}", "format=yuv420p"]
    if req.get("avtb"):  # the Storyboard's shot intermediates use this time base; its concat needs them to match
        chain += ["settb=AVTB", "setpts=N/(FRAME_RATE*TB)"]
    codec = list(req.get("codec_args") or ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"])
    return [str(req["ffmpeg"]), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{model_size[0]}x{model_size[1]}", "-framerate", str(req["fps"]),
            "-i", "pipe:0", "-vf", ",".join(chain), *codec, "-an", "-f", "mp4", str(output)]


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
        text = handle.read().decode("utf-8", "replace").strip().splitlines()
    except (OSError, ValueError):
        return ""
    return " | ".join(text[-3:])[:300]


def _stop(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def process_segment(req: dict[str, Any], upscaler: Callable[[np.ndarray], np.ndarray],
                    on_frames: Callable[[int], None] | None = None,
                    check_cancel: Callable[[], None] | None = None) -> dict[str, Any]:
    """Decode, upscale and encode one segment to ``req['output']``. Returns ``{"frames": n}`` (0: nothing left to
    read, no file written). ``check_cancel`` raises to stop; the half-written file is removed."""
    src_w, src_h = int(req["src_width"]), int(req["src_height"])
    frame_bytes = src_w * src_h * 3
    batch = max(1, int(req.get("batch") or 1))
    output = Path(req["output"])
    made = 0
    encoder: subprocess.Popen | None = None
    with tempfile.TemporaryFile() as dec_err, tempfile.TemporaryFile() as enc_err:
        decoder = subprocess.Popen(decode_command(req), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=dec_err)
        try:
            while True:
                if check_cancel is not None:
                    check_cancel()
                data = _read_exact(decoder.stdout, frame_bytes * batch)
                count = len(data) // frame_bytes
                if count == 0:
                    break
                frames = np.frombuffer(data[:count * frame_bytes], dtype=np.uint8).reshape(count, src_h, src_w, 3)
                out = np.ascontiguousarray(upscaler(frames), dtype=np.uint8)
                if encoder is None:
                    size = (int(out.shape[2]), int(out.shape[1]))
                    encoder = subprocess.Popen(encode_command(req, size, output), stdin=subprocess.PIPE,
                                               stdout=subprocess.DEVNULL, stderr=enc_err)
                try:
                    encoder.stdin.write(out.tobytes())  # type: ignore[union-attr]
                except (BrokenPipeError, OSError) as exc:
                    raise RuntimeError(f"The encoder stopped: {_tail(enc_err) or exc}") from exc
                made += count
                if on_frames is not None:
                    on_frames(made)
            decoder.wait(timeout=60)
            if decoder.returncode != 0:
                raise RuntimeError(f"ffmpeg could not decode the video: {_tail(dec_err) or decoder.returncode}")
            if encoder is not None:
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
    return {"frames": made}


def local_runner(upscaler: Callable[[np.ndarray], np.ndarray]) -> Callable[..., dict[str, Any]]:
    """A runner that upscales in this process (tests, or a caller that already holds a model)."""

    def run(req: dict[str, Any], on_frames: Callable[[int], None], is_cancelled: Callable[[], bool]) -> dict[str, Any]:
        def check() -> None:
            if is_cancelled():
                raise Cancelled("cancelled")

        return process_segment(req, upscaler, on_frames, check)

    return run


# ---------------------------------------------------------------------------
# the whole job (runs in the backend)
# ---------------------------------------------------------------------------


def segment_request(spec: JobSpec, exe: str, seg: dict[str, Any], output: Path) -> dict[str, Any]:
    src = spec.source
    # Field names must not collide with ``WarmWorker.request``'s own keywords (``start``, ``timeout``, ``job``).
    # ``last`` means "read to the end of the file": only when the frame count is an estimate. A counted last segment is
    # capped at its remainder, because the fps filter pads a short tail out to the container's (longer) end.
    return {"ffmpeg": exe, "input": spec.input, "first_frame": seg["start"], "frames": seg["frames"],
            "last": seg["last"] and not src.get("frames_exact"),
            "fps": src["fps"], "sar": src.get("sar") or "1/1", "src_width": src["width"], "src_height": src["height"],
            "width": spec.width, "height": spec.height, "deinterlace": spec.deinterlace, "batch": spec.batch,
            "codec_args": spec.codec_args, "output": str(output), "weights": spec.weights, "tile": spec.tile,
            "model_id": spec.model_id, "avtb": spec.avtb}


def concat_list(names: list[str]) -> str:
    for name in names:
        if not re.fullmatch(r"seg-\d{5}\.mp4", name):
            raise ValueError(f"Refused segment name {name!r}")
    return "".join(f"file '{name}'\n" for name in names)


def mux_attempts(exe: str, listing: str, source: str, output: Path) -> list[tuple[list[str], str | None]]:
    """The final mux, most faithful first: (arguments, note when this attempt is the one that worked)."""
    head = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-protocol_whitelist", PROTOCOLS, "-f", "concat", "-safe", "1", "-i", listing,
            "-protocol_whitelist", PROTOCOLS, "-i", source, "-map", "0:v:0", "-map", "1:a?"]
    tail = ["-map_metadata", "1", "-map_chapters", "1"]
    if output.suffix.lower() in (".mp4", ".mov", ".m4v"):
        tail += ["-movflags", "+faststart"]
    return [
        ([*head, "-map", "1:s?", "-c", "copy", *tail, str(output)], None),
        ([*head, "-c", "copy", *tail, str(output)], "subtitle streams were left out (the container refused them)"),
        ([*head, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", *tail, str(output)],
         "the sound was re-encoded to AAC (the container refused a copy); subtitles were left out"),
    ]


def run_job(spec: JobSpec, runner: Callable[..., dict[str, Any]], *, exe: str, work: Path,
            progress: Callable[[int, int, int, int], None] | None = None,
            is_cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Run (or resume) a job; returns ``{output, frames, segments, resumedSegments, notes}``."""
    manifest = Manifest.open(work, spec)
    total = int(spec.source["frames"])
    segments = manifest.segments
    resumed = sum(1 for seg in segments if manifest.finished(seg))
    manifest.set_state("running")
    frames_before = 0
    if progress is not None:  # the resumed frames are the baseline a frames-a-second rate counts from
        progress(sum(int(seg.get("made", 0)) for seg in segments if manifest.finished(seg)), total, 0, len(segments))
    try:
        for seg in segments:
            if manifest.finished(seg):
                frames_before += int(seg.get("made", 0))
                continue
            if is_cancelled():
                raise Cancelled("cancelled")
            part = work / f"seg-{seg['index']:05d}.part.mp4"
            base = frames_before

            def on_frames(made: int, base: int = base, index: int = seg["index"]) -> None:
                if progress is not None:
                    progress(base + made, max(total, base + made), index, len(segments))

            result = runner(segment_request(spec, exe, seg, part), on_frames, is_cancelled)
            made = int(result.get("frames") or 0)
            if made:
                os.replace(part, manifest.segment_file(seg["index"]))
            manifest.mark(seg["index"], made)
            frames_before += made
        if is_cancelled():
            raise Cancelled("cancelled")
        names = [manifest.segment_file(s["index"]).name for s in segments if s.get("made")]
        if not names:
            raise UpscaleError(f"No frames could be read from {Path(spec.input).name}.")
        notes = _mux(spec, exe, work, names)
    except Cancelled:
        manifest.set_state("cancelled")
        raise
    except Exception as exc:
        manifest.set_state("failed", str(exc)[:300])
        raise
    manifest.set_state("done")
    if progress is not None:
        progress(frames_before, frames_before, len(segments) - 1, len(segments))
    return {"output": spec.output, "frames": frames_before, "segments": len(segments), "resumedSegments": resumed,
            "notes": notes}


def _mux(spec: JobSpec, exe: str, work: Path, names: list[str]) -> list[str]:
    listing = work / "segments.txt"
    listing.write_text(concat_list(names), encoding="utf-8")
    output = Path(spec.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = work / f"out{output.suffix.lower()}"
    last = ""
    for args, note in mux_attempts(exe, listing.name, spec.input, tmp):
        run = subprocess.run(args, cwd=str(work), capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=3600)
        if run.returncode == 0 and tmp.is_file():
            os.replace(tmp, output)
            return [note] if note else []
        tmp.unlink(missing_ok=True)
        last = (run.stderr or "").strip().splitlines()[-1:][0] if (run.stderr or "").strip() else str(run.returncode)
    raise RuntimeError(f"ffmpeg could not join the upscaled video with the original sound: {last[:200]}")
