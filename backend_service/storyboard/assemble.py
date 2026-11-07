"""Join and mix a Storyboard film from a ``RenderPlan`` with the pinned ffmpeg.

Two stages (home-studio plan 5.4). (1) Each shot becomes a cached intermediate in ``work_dir``: one
size, fps, ``yuv420p``, timebase and 48 kHz stereo PCM in a ``.mov``, with trims, speed and the
clip's own sound gain applied, named by a hash of everything that changes its bytes, so redoing one
shot re-renders only that shot. (2) Runs of hard cuts are joined losslessly with the concat demuxer;
crossfades are ``xfade`` / ``acrossfade`` between those runs; then one audio graph mixes the film's
own sound with the voice, effect and music lanes (music ducked under voice), normalises loudness and
writes the MP4. Every ffmpeg call is an argument list with its filter graph in a script file and the
process working directory set to ``work_dir``, so no plan path or text is ever spliced into a filter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from backend_service.storyboard import ffmpeg_bin
from backend_service.storyboard.media import replace_file
from backend_service.storyboard.render_plan import AudioInput, RenderPlan, ShotInput, Transition

ENGINE_VERSION = "1"
SAMPLE_RATE = 48000
META_PREFIX = "chaosengine"  # same tag convention as helpers/output_metadata.py

ProgressCb = Callable[[str, float], None]

_BASE_FILTERS = {
    "trim", "setpts", "fps", "scale", "pad", "setsar", "format", "settb", "tpad", "atrim", "asetpts",
    "aresample", "aformat", "volume", "apad", "anullsrc", "amix", "adelay", "afade", "anull",
}


class RenderCancelled(Exception):
    """The caller set the cancel event; ffmpeg was stopped and no output was left behind."""


class RenderError(RuntimeError):
    """ffmpeg failed or lacks a filter; the message is meant for the user."""


@dataclass
class RenderResult:
    path: str
    duration_s: float
    shots_rendered: int
    shots_cached: int
    ffmpeg_version: str
    warnings: list[str] = field(default_factory=list)
    loudness: dict[str, Any] | None = None


@dataclass
class JoinTiming:
    starts: list[float]  # where each shot starts on the final timeline
    total: float


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested without ffmpeg)


def _f(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def atempo_chain(speed: float) -> list[float]:
    """``atempo`` factors whose product is ``speed``, each within atempo's 0.5 to 2.0 range."""
    if not (speed > 0 and math.isfinite(speed)):
        raise ValueError("speed must be a positive number")
    factors: list[float] = []
    rest = speed
    while rest > 2.0:
        factors.append(2.0)
        rest /= 2.0
    while rest < 0.5:
        factors.append(0.5)
        rest /= 0.5
    if abs(rest - 1.0) > 1e-9:
        factors.append(rest)
    return factors


def compute_join_offsets(durations: Sequence[float], transitions: Sequence[float | Transition]) -> JoinTiming:
    """Start time of every shot and the film length when shot i joins shot i-1 by ``transitions[i]``.

    A transition is the crossfade length in seconds (0 for a cut) or a ``Transition``; the first is
    ignored. A crossfade overlaps the two shots, so each one shortens the film by its length, and the
    next ``xfade`` offset is measured on that shortened film: ``offset_i = length_so_far - d_i``.
    """
    if len(durations) != len(transitions):
        raise ValueError("durations and transitions must have the same length")
    fades = [float(t.seconds if isinstance(t, Transition) else t) for t in transitions]
    if not durations:
        return JoinTiming([], 0.0)
    for index, length in enumerate(durations):
        if not length > 0:
            raise ValueError(f"shot {index} has no length")
        fade_in = fades[index] if index > 0 else 0.0
        fade_out = fades[index + 1] if index + 1 < len(fades) else 0.0
        if fade_in < 0 or fade_out < 0:
            raise ValueError(f"shot {index}: transition seconds must not be negative")
        if fade_in + fade_out > length + 1e-9:
            raise ValueError(f"shot {index} ({length:.2f} s) is shorter than its crossfades "
                             f"({fade_in:.2f} s in + {fade_out:.2f} s out)")
    starts = [0.0]
    total = float(durations[0])
    for index in range(1, len(durations)):
        start = total - fades[index]
        starts.append(start)
        total = start + float(durations[index])
    return JoinTiming(starts, total)


def shot_cache_key(shot: ShotInput, plan: RenderPlan) -> str:
    payload = json.dumps([
        ENGINE_VERSION, shot.hash, plan.width, plan.height, plan.fps, plan.preview, shot.trim_in,
        shot.trim_out, shot.speed, "-inf" if shot.native_sound_db == -math.inf else shot.native_sound_db,
    ], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def shot_frames(shot: ShotInput, source_duration: float, fps: int) -> tuple[int, float, float]:
    """(frames, source in, source out) for a shot after trims and speed."""
    src_in = shot.trim_in
    src_out = source_duration if shot.trim_out is None else min(shot.trim_out, source_duration)
    if src_out - src_in <= 1e-6:
        raise ValueError(f"shot {shot.id}: trim leaves nothing of a {source_duration:.2f} s clip")
    frames = max(1, round((src_out - src_in) / shot.speed * fps))
    return frames, src_in, src_out


def _samples(frames: int, fps: int) -> int:
    return round(frames * SAMPLE_RATE / fps)


def _scale_chain(plan: RenderPlan) -> str:
    w, h = plan.width, plan.height
    return (f"fps={plan.fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p,settb=AVTB")


def build_shot_graph(shot: ShotInput, plan: RenderPlan, has_audio: bool, frames: int,
                     src_in: float, src_out: float) -> str:
    samples = _samples(frames, plan.fps)
    trim = f"start={_f(src_in)}:end={_f(src_out)}"
    video = f"[0:v]trim={trim},setpts=PTS-STARTPTS"
    if shot.speed != 1.0:
        video += f",setpts=PTS/{_f(shot.speed)}"
    video += (f",{_scale_chain(plan)},tpad=stop_mode=clone:stop={frames},trim=end_frame={frames},"
              f"setpts=N/(FRAME_RATE*TB)[v]")
    if has_audio:
        audio = f"[0:a]atrim={trim},asetpts=PTS-STARTPTS"
        for factor in atempo_chain(shot.speed):
            audio += f",atempo={_f(factor)}"
        audio += f",aresample={SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=stereo"
        if shot.native_sound_db == -math.inf:
            audio += ",volume=0"
        elif shot.native_sound_db != 0:
            audio += f",volume={_f(shot.native_sound_db)}dB"
    else:
        audio = f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},aformat=sample_fmts=s16"
    audio += f",apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB[a]"
    return f"{video};\n{audio}\n"


def build_card_graph(plan: RenderPlan, frames: int, font_name: str | None, text_name: str | None) -> str:
    seconds = frames / plan.fps
    video = f"color=c=black:s={plan.width}x{plan.height}:r={plan.fps}:d={_f(seconds + 1)},format=yuv420p"
    if font_name and text_name:
        size = max(8, plan.height // 12)
        video += (f",drawtext=fontfile={font_name}:textfile={text_name}:expansion=none:fontcolor=white:"
                  f"fontsize={size}:line_spacing={size // 3}:x=(w-text_w)/2:y=(h-text_h)/2")
    video += f",setsar=1,settb=AVTB,trim=end_frame={frames},setpts=N/(FRAME_RATE*TB)[v]"
    samples = _samples(frames, plan.fps)
    audio = (f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE},aformat=sample_fmts=s16,"
             f"atrim=end_sample={samples},asetpts=N/SR/TB[a]")
    return f"{video};\n{audio}\n"


def build_xfade_graphs(group_frames: Sequence[int], fades: Sequence[int], starts: Sequence[float],
                       fps: int) -> tuple[str, str]:
    """Video and audio graphs that crossfade the inputs pairwise: ``fades[g]`` frames into group g,
    which starts at ``starts[g]`` on the (already shortened) film. They run as two ffmpeg calls:
    with many inputs, one call carrying both ``xfade`` and ``acrossfade`` can end the audio output
    before its first frame (seen with ffmpeg 7.1), while each half on its own is reliable."""
    video_lines = [f"[{g}:v]settb=AVTB,format=yuv420p[v{g}]" for g in range(len(group_frames))]
    audio_lines: list[str] = []
    video, audio = "v0", "0:a"
    for g in range(1, len(group_frames)):
        out_v, out_a = f"x{g}", f"y{g}"
        video_lines.append(f"[{video}][v{g}]xfade=transition=fade:duration={_f(fades[g] / fps)}:"
                           f"offset={_f(starts[g])}[{out_v}]")
        audio_lines.append(f"[{audio}][{g}:a]acrossfade=ns={_samples(fades[g], fps)}:c1=tri:c2=tri[{out_a}]")
        video, audio = out_v, out_a
    video_lines.append(f"[{video}]null[vout]")
    audio_lines.append(f"[{audio}]anull[aout]")
    return ";\n".join(video_lines) + "\n", ";\n".join(audio_lines) + "\n"


@dataclass
class Lane:
    clip: AudioInput
    input_index: int
    source_seconds: float


def build_mix_graph(lanes: Sequence[Lane], total_samples: int, loudnorm: str | None,
                    captions_name: str | None = None) -> tuple[str, list[str]]:
    """The audio graph (plus captions burn-in when asked). Returns (graph, warnings)."""
    total = total_samples / SAMPLE_RATE
    warnings: list[str] = []
    lines = [f"[0:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
             f"apad=whole_len={total_samples},atrim=end_sample={total_samples}[nat]"]
    voice, ducked, plain = [], [], []
    for n, lane in enumerate(lanes):
        clip = lane.clip
        available = lane.source_seconds - clip.trim_in
        if available <= 0.01:
            warnings.append(f"audio {clip.id}: trim_in is past the end of the clip; skipped")
            continue
        room = total - clip.start
        if room <= 0.01:
            warnings.append(f"audio {clip.id}: starts after the film ends; skipped")
            continue
        if clip.loop:
            length = clip.duration if clip.duration is not None else room
        else:
            length = min(clip.duration if clip.duration is not None else available, available)
        length = min(length, room)
        chain = (f"[{lane.input_index}:a]atrim=start={_f(clip.trim_in)},asetpts=PTS-STARTPTS,"
                 f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo")
        if clip.loop:
            chain += f",aloop=loop=-1:size={max(1, round(available * SAMPLE_RATE))}"
        chain += f",atrim=duration={_f(length)},asetpts=N/SR/TB"
        if clip.gain_db != 0:
            chain += f",volume={_f(clip.gain_db)}dB"
        if clip.fade_in > 0:
            chain += f",afade=t=in:st=0:d={_f(min(clip.fade_in, length))}"
        if clip.fade_out > 0:
            fade = min(clip.fade_out, length)
            chain += f",afade=t=out:st={_f(length - fade)}:d={_f(fade)}"
        delay = round(clip.start * SAMPLE_RATE)
        if delay > 0:
            chain += f",adelay=delays={delay}S:all=1"
        chain += f",apad=whole_len={total_samples},atrim=end_sample={total_samples}[l{n}]"
        lines.append(chain)
        label = f"l{n}"
        if clip.kind == "voice":
            voice.append(label)
        elif clip.duck:
            ducked.append(label)
        else:
            plain.append(label)

    def bus(labels: list[str], name: str) -> str:
        if len(labels) == 1:
            lines.append(f"[{labels[0]}]anull[{name}]")
        else:
            joined = "".join(f"[{x}]" for x in labels)
            lines.append(f"{joined}amix=inputs={len(labels)}:normalize=0:duration=longest[{name}]")
        return name

    mix_inputs = ["nat"]
    if voice:
        voice_bus = bus(voice, "vbus")
        if ducked:
            lines.append(f"[{voice_bus}]asplit=2[vmix][vkey]")
            music = bus(ducked, "dbus")
            lines.append(f"[{music}][vkey]sidechaincompress=threshold=0.02:ratio=10:attack=10:"
                         f"release=300:makeup=1[ducked]")
            mix_inputs += ["vmix", "ducked"]
        else:
            mix_inputs.append(voice_bus)
    else:
        plain = ducked + plain
    mix_inputs += plain
    tail = f"aresample={SAMPLE_RATE},atrim=end_sample={total_samples}[aout]"
    if loudnorm:
        tail = f"{loudnorm},{tail}"
    if len(mix_inputs) == 1:
        lines.append(f"[nat]{tail}")
    else:
        joined = "".join(f"[{x}]" for x in mix_inputs)
        lines.append(f"{joined}amix=inputs={len(mix_inputs)}:normalize=0:duration=first,{tail}")
    if captions_name:
        lines.append(f"[0:v]subtitles=filename={captions_name}[vout]")
    return ";\n".join(lines) + "\n", warnings


def anchored_audio(clips: Sequence[AudioInput], shot_ids: Sequence[str], starts: Sequence[float]) -> list[AudioInput]:
    """The clips with each anchored one's ``start`` taken from its shot's rendered start plus the offset.

    The rendered starts come from whole frames, so they can differ from the timeline's seconds by a few
    milliseconds per shot; anchoring here keeps a voice line on its shot however long the film is. A start
    before 0 becomes 0 (the plan's ``trim_in`` already skips the part before the film).
    """
    index = {shot_id: i for i, shot_id in enumerate(shot_ids)}
    out: list[AudioInput] = []
    for clip in clips:
        if clip.anchor_shot is not None and clip.anchor_shot in index:
            clip = dataclasses.replace(clip, start=max(0.0, starts[index[clip.anchor_shot]] + clip.anchor_offset))
        out.append(clip)
    return out


def concat_list(names: Sequence[str]) -> str:
    """A concat-demuxer list of file names relative to the list (safe under ``-safe 1``)."""
    out = []
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
            raise ValueError(f"unsafe concat entry {name!r}")
        out.append(f"file '{name}'")
    return "\n".join(out) + "\n"


_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)


def parse_loudnorm(stderr: str) -> dict[str, str] | None:
    found = _LOUDNORM_JSON.findall(stderr)
    if not found:
        return None
    try:
        data = json.loads(found[-1])
    except ValueError:
        return None
    return {str(k): str(v) for k, v in data.items()}


def find_font() -> Path | None:
    """A plain sans font for the end card, from the operating system's own font folder."""
    if sys.platform == "darwin":
        candidates = [Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                      Path("/System/Library/Fonts/Helvetica.ttc"), Path("/Library/Fonts/Arial.ttf")]
    elif os.name == "nt":
        fonts = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
        candidates = [fonts / "arial.ttf", fonts / "segoeui.ttf", fonts / "calibri.ttf"]
    else:
        candidates = [Path(p) for p in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/noto/NotoSans-Regular.ttf")]
    return next((p for p in candidates if p.is_file()), None)


# ---------------------------------------------------------------------------------------------
# Running ffmpeg


def _run(args: list[str], *, cwd: Path, expected_s: float, report: Callable[[float], None],
         cancel_event: threading.Event | None) -> str:
    """Run ffmpeg, feed ``-progress`` into ``report``, stop it on cancel. Returns stderr."""
    if cancel_event is not None and cancel_event.is_set():
        raise RenderCancelled()
    cmd = [ffmpeg_bin.ffmpeg_exe(), "-nostdin", "-y", "-hide_banner", "-nostats",
           "-progress", "pipe:1", *args]
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    tail: deque[str] = deque(maxlen=400)
    reader = threading.Thread(target=lambda: tail.extend(proc.stderr), daemon=True)  # type: ignore[arg-type]
    reader.start()
    cancelled = threading.Event()

    def watch() -> None:
        while proc.poll() is None:
            if cancel_event.wait(0.05):  # type: ignore[union-attr]
                cancelled.set()
                proc.terminate()
                return

    watcher = None
    if cancel_event is not None:
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    report(0.0)
    assert proc.stdout is not None
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and value.lstrip("-").isdigit() and expected_s > 0:
            report(max(0.0, min(1.0, int(value) / 1e6 / expected_s)))
        elif key == "progress" and value == "end":
            report(1.0)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    reader.join(timeout=5)
    if watcher is not None:
        watcher.join(timeout=1)
    if cancelled.is_set() or (cancel_event is not None and cancel_event.is_set()):
        raise RenderCancelled()
    stderr = "".join(tail)
    if proc.returncode != 0:
        lines = [x.strip() for x in stderr.splitlines() if x.strip()]
        errors = [x for x in lines if re.search(r"error|invalid|no such|failed|not found|unable", x, re.I)]
        raise RenderError("ffmpeg failed: " + " | ".join((errors or lines)[-4:]))
    return stderr


def _codec_args(preview: bool) -> list[str]:
    if preview:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]


def _inp(path: str, *pre: str) -> list[str]:
    return ["-protocol_whitelist", ffmpeg_bin.PROTOCOLS, *pre, "-i", path]


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------
# The render


def _fade_frames(transition: Transition, fps: int) -> int:
    """Crossfade length in whole frames (at least one); 0 for a cut."""
    return max(1, round(transition.seconds * fps)) if transition.type == "crossfade" else 0


def _required_filters(plan: RenderPlan, has_voice_duck: bool) -> set[str]:
    needed = set(_BASE_FILTERS)
    if any(s.speed != 1.0 for s in plan.shots):
        needed.add("atempo")
    if any(s.transition_in.type == "crossfade" for s in plan.shots[1:]):
        needed |= {"xfade", "acrossfade", "null"}
    if plan.loudness_lufs is not None:
        needed.add("loudnorm")
    if has_voice_duck:
        needed |= {"sidechaincompress", "asplit"}
    if any(a.loop for a in plan.audio):
        needed.add("aloop")
    if plan.captions_srt:
        needed.add("subtitles")
    return needed


def render(plan: RenderPlan, progress_cb: ProgressCb | None = None,
           cancel_event: threading.Event | None = None) -> RenderResult:
    plan = plan.validate()
    progress = progress_cb or (lambda stage, fraction: None)
    warnings: list[str] = []
    work = Path(plan.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(plan.out_path)
    inputs = {str(Path(s.video)) for s in plan.shots} | {a.path for a in plan.audio}
    if plan.captions_srt:
        inputs.add(plan.captions_srt)
    if out.exists() and str(out.resolve()) in inputs:
        raise RenderError("out_path must not be one of the input files")

    has_voice = any(a.kind == "voice" for a in plan.audio)
    has_voice_duck = has_voice and any(a.duck and a.kind != "voice" for a in plan.audio)
    missing = ffmpeg_bin.missing_filters(_required_filters(plan, has_voice_duck))
    if missing:
        raise RenderError(f"The bundled ffmpeg lacks the filters this film needs: {', '.join(missing)}")

    run_id = uuid.uuid4().hex[:10]
    scratch: list[Path] = []
    out_tmp = out.with_name(f".{out.stem}.partial-{run_id}.mp4")

    def script(name: str, text: str) -> str:
        path = work / f"{run_id}-{name}"
        path.write_text(text, encoding="utf-8")
        scratch.append(path)
        return path.name

    try:
        # Stage 1: per-shot intermediates.
        infos = [ffmpeg_bin.probe(s.video) for s in plan.shots]
        spans = []
        for shot, info in zip(plan.shots, infos):
            if not info["has_video"]:
                raise RenderError(f"shot {shot.id}: the file has no video stream")
            spans.append(shot_frames(shot, info["duration"], plan.fps))
        # Check the crossfades fit before spending time on any intermediate.
        compute_join_offsets([s[0] / plan.fps for s in spans],
                             [_fade_frames(s.transition_in, plan.fps) / plan.fps for s in plan.shots])
        names, frames_list, rendered, cached = [], [], 0, 0
        for index, (shot, info, span) in enumerate(zip(plan.shots, infos, spans)):
            frames, src_in, src_out = span
            frames_list.append(frames)
            name = f"shot-{shot_cache_key(shot, plan)[:32]}.mov"
            names.append(name)
            target = work / name
            report = lambda f, i=index: progress("shots", (i + f) / len(plan.shots))  # noqa: E731
            if target.is_file() and target.stat().st_size > 0:
                cached += 1
                report(1.0)
                continue
            graph = script(f"shot{index}.txt", build_shot_graph(shot, plan, info["has_audio"], frames,
                                                                 src_in, src_out))
            tmp = work / f"{run_id}-{name}"
            scratch.append(tmp)
            _run([*_inp(shot.video), "-filter_complex_script", graph, "-map", "[v]", "-map", "[a]",
                  *_codec_args(plan.preview), "-c:a", "pcm_s16le", "-map_metadata", "-1", "-f", "mov",
                  tmp.name], cwd=work, expected_s=frames / plan.fps, report=report,
                 cancel_event=cancel_event)
            os.replace(tmp, target)
            rendered += 1
        transitions = [s.transition_in for s in plan.shots]

        # Optional end card, appended as one more hard cut.
        if plan.end_card is not None:
            card = _render_card(plan, work, run_id, script, scratch, warnings, cancel_event)
            if card is not None:
                names.append(card[0])
                frames_list.append(card[1])
                transitions.append(Transition())

        # Stage 2a: group runs of cuts; crossfade between groups.
        fade_frames = [0] + [_fade_frames(t, plan.fps) for t in transitions[1:]]
        timing = compute_join_offsets([f / plan.fps for f in frames_list],
                                      [f / plan.fps for f in fade_frames])
        total_samples = round(timing.total * SAMPLE_RATE)
        groups: list[list[int]] = []
        for i in range(len(names)):
            if i == 0 or fade_frames[i] > 0:
                groups.append([i])
            else:
                groups[-1].append(i)

        def group_input(g: int, members: list[int]) -> list[str]:
            if len(members) == 1:
                return _inp(names[members[0]])
            listing = script(f"group{g}.txt", concat_list([names[i] for i in members]))
            return _inp(listing, "-f", "concat", "-safe", "1")

        if len(groups) == 1:
            film_input = group_input(0, groups[0])
        else:
            group_files = []
            for g, members in enumerate(groups):
                if len(members) == 1:
                    group_files.append(names[members[0]])
                    continue
                gfile = work / f"{run_id}-group{g}.mov"
                scratch.append(gfile)
                _run([*group_input(g, members), "-map", "0", "-c", "copy", "-f", "mov", gfile.name],
                     cwd=work, expected_s=sum(frames_list[i] for i in members) / plan.fps,
                     report=lambda f: progress("join", 0.2 * f), cancel_event=cancel_event)
                group_files.append(gfile.name)
            group_frames = [sum(frames_list[i] for i in members) for members in groups]
            video_graph, audio_graph = build_xfade_graphs(
                group_frames, [fade_frames[m[0]] for m in groups],
                [timing.starts[m[0]] for m in groups], plan.fps)
            args: list[str] = []
            for name in group_files:
                args += _inp(name)
            joined_v, joined_a, joined = (work / f"{run_id}-join{part}.mov" for part in ("v", "a", ""))
            scratch += [joined_v, joined_a, joined]
            _run([*args, "-filter_complex_script", script("xfade-v.txt", video_graph), "-map", "[vout]",
                  *_codec_args(plan.preview), "-f", "mov", joined_v.name],
                 cwd=work, expected_s=timing.total, report=lambda f: progress("join", 0.2 + 0.6 * f),
                 cancel_event=cancel_event)
            _run([*args, "-filter_complex_script", script("xfade-a.txt", audio_graph), "-map", "[aout]",
                  "-c:a", "pcm_s16le", "-f", "mov", joined_a.name],
                 cwd=work, expected_s=timing.total, report=lambda f: progress("join", 0.8 + 0.15 * f),
                 cancel_event=cancel_event)
            _run([*_inp(joined_v.name), *_inp(joined_a.name), "-map", "0:v", "-map", "1:a", "-c", "copy",
                  "-f", "mov", joined.name], cwd=work, expected_s=timing.total,
                 report=lambda f: progress("join", 0.95 + 0.05 * f), cancel_event=cancel_event)
            film_input = _inp(joined.name)
        progress("join", 1.0)

        # Stage 2b: one audio mix graph over the joined film and the lanes.
        lane_args: list[str] = []
        lanes: list[Lane] = []
        placed = anchored_audio(plan.audio, [s.id for s in plan.shots], timing.starts[:len(plan.shots)])
        for n, clip in enumerate(placed):
            info = ffmpeg_bin.probe(clip.path)
            if not info["has_audio"]:
                raise RenderError(f"audio {clip.id}: the file has no audio stream")
            lanes.append(Lane(clip, n + 1, info["duration"]))
            lane_args += _inp(clip.path)
        captions_name = None
        if plan.captions_srt:
            captions_name = f"{run_id}-captions.srt"
            shutil.copyfile(plan.captions_srt, work / captions_name)
            scratch.append(work / captions_name)

        loudness: dict[str, Any] | None = None
        target = plan.loudness_lufs
        loudnorm = None
        if target is not None:
            base = f"loudnorm=I={_f(target)}:TP=-1.5:LRA=11"
            if plan.preview:
                loudnorm = base
                loudness = {"passes": 1, "target_lufs": target}
            else:
                graph_text, _ = build_mix_graph(lanes, total_samples, f"{base}:print_format=json")
                graph = script("measure.txt", graph_text)
                stderr = _run([*film_input, *lane_args, "-filter_complex_script", graph,
                               "-map", "[aout]", "-f", "null", "-"], cwd=work, expected_s=timing.total,
                              report=lambda f: progress("measure", f), cancel_event=cancel_event)
                measured = parse_loudnorm(stderr) or {}
                keys = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
                try:
                    values = [float(measured[k]) for k in keys]
                except (KeyError, ValueError):
                    values = []
                if len(values) == len(keys) and all(math.isfinite(v) for v in values):
                    i, tp, lra, thresh, offset = values
                    loudnorm = (f"{base}:measured_I={_f(i)}:measured_TP={_f(tp)}:measured_LRA={_f(lra)}:"
                                f"measured_thresh={_f(thresh)}:offset={_f(offset)}:linear=true:"
                                f"print_format=json")
                    loudness = {"passes": 2, "target_lufs": target, "measured": measured}
                else:
                    warnings.append("loudness not normalised: the mix is silent or could not be measured")

        graph_text, mix_warnings = build_mix_graph(lanes, total_samples, loudnorm, captions_name)
        warnings += mix_warnings
        graph = script("mix.txt", graph_text)
        video_args = (["-map", "[vout]", *_codec_args(plan.preview)] if captions_name
                      else ["-map", "0:v", "-c:v", "copy"])
        meta_args = []
        if plan.metadata:
            comment = json.dumps(plan.metadata, ensure_ascii=True, separators=(",", ":"))
            meta_args = ["-metadata", f"comment={META_PREFIX}:{comment}"]
        stderr = _run([*film_input, *lane_args, "-filter_complex_script", graph, *video_args,
                       "-map", "[aout]", "-c:a", "aac", "-b:a", "128k" if plan.preview else "192k",
                       "-ar", str(SAMPLE_RATE), "-map_metadata", "-1", *meta_args,
                       "-movflags", "+faststart", "-f", "mp4", str(out_tmp)],
                      cwd=work, expected_s=timing.total, report=lambda f: progress("mix", f),
                      cancel_event=cancel_event)
        if loudness is not None and loudness["passes"] == 2:
            second = parse_loudnorm(stderr)
            loudness["normalization_type"] = (second or {}).get("normalization_type")
        replace_file(out_tmp, out)  # the old film may be being served (Windows): retried briefly
        duration = ffmpeg_bin.probe(out)["duration"]
        return RenderResult(path=str(out), duration_s=duration, shots_rendered=rendered,
                            shots_cached=cached, ffmpeg_version=ffmpeg_bin.ffmpeg_version(),
                            warnings=warnings, loudness=loudness)
    finally:
        _unlink(out_tmp)
        for path in scratch:
            _unlink(path)


def _render_card(plan: RenderPlan, work: Path, run_id: str, script: Callable[[str, str], str],
                 scratch: list[Path], warnings: list[str],
                 cancel_event: threading.Event | None) -> tuple[str, int] | None:
    """Render (or reuse) the end card intermediate. Returns (file name, frames) or None if skipped."""
    card = plan.end_card
    assert card is not None
    frames = max(1, round(card.seconds * plan.fps))
    filters = ffmpeg_bin.ffmpeg_filters()
    font = find_font() if "drawtext" in filters else None
    if "color" not in filters or font is None:
        reason = "drawtext filter missing" if "drawtext" not in filters else "no system font found"
        warnings.append(f"end card skipped: {reason}")
        return None
    key = hashlib.sha256(json.dumps([ENGINE_VERSION, "card", card.text, frames, plan.width, plan.height,
                                     plan.fps, plan.preview, str(font)]).encode()).hexdigest()[:32]
    name = f"card-{key}.mov"
    target = work / name
    if target.is_file() and target.stat().st_size > 0:
        return name, frames
    font_name = f"{run_id}-font{font.suffix.lower()}"
    shutil.copyfile(font, work / font_name)
    scratch.append(work / font_name)
    text_name = script("card.txt", card.text)
    graph = script("cardgraph.txt", build_card_graph(plan, frames, font_name, text_name))
    tmp = work / f"{run_id}-{name}"
    scratch.append(tmp)
    _run(["-filter_complex_script", graph, "-map", "[v]", "-map", "[a]", *_codec_args(plan.preview),
          "-c:a", "pcm_s16le", "-f", "mov", tmp.name], cwd=work, expected_s=frames / plan.fps,
         report=lambda f: None, cancel_event=cancel_event)
    os.replace(tmp, target)
    return name, frames
