"""The pure part of the Storyboard assembler: frame and join maths, cache keys and ffmpeg filter graphs.

No ffmpeg runs here and nothing touches the disk, so every function is unit-tested on its own; the
``assemble`` module runs ffmpeg with what these build and re-exports them, so existing imports keep working.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Sequence

from backend_service.storyboard.render_plan import AudioInput, RenderPlan, ShotInput, Transition

ENGINE_VERSION = "1"
SAMPLE_RATE = 48000


@dataclass
class JoinTiming:
    starts: list[float]  # where each shot starts on the final timeline
    total: float


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
