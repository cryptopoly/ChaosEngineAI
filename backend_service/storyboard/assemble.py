"""Join and mix a Storyboard film from a ``RenderPlan`` with the pinned ffmpeg.

Two stages (home-studio plan 5.4). (1) Each shot becomes a cached intermediate in ``work_dir``: one
size, fps, ``yuv420p``, timebase and 48 kHz stereo PCM in a ``.mov``, with trims, speed and the
clip's own sound gain applied, named by a hash of everything that changes its bytes, so redoing one
shot re-renders only that shot. (2) Runs of hard cuts are joined losslessly with the concat demuxer;
crossfades are ``xfade`` / ``acrossfade`` between those runs; then one audio graph mixes the film's
own sound with the voice, effect and music lanes (music ducked under voice), normalises loudness and
writes the MP4. Every ffmpeg call is an argument list with its filter graph in a script file and the
process working directory set to ``work_dir``, so no plan path or text is ever spliced into a filter.
The pure graph builders and timing maths live in ``assemble_graphs`` (re-exported here).
"""

from __future__ import annotations

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
from typing import Any, Callable

from backend_service.storyboard import ffmpeg_bin
from backend_service.storyboard.assemble_graphs import (  # noqa: F401 - re-exported for callers and tests
    ENGINE_VERSION,
    SAMPLE_RATE,
    JoinTiming,
    Lane,
    _f,
    _samples,
    anchored_audio,
    atempo_chain,
    build_card_graph,
    build_mix_graph,
    build_shot_graph,
    build_xfade_graphs,
    compute_join_offsets,
    concat_list,
    parse_loudnorm,
    shot_cache_key,
    shot_frames,
)
from backend_service.storyboard.media import replace_file
from backend_service.storyboard.render_plan import RenderPlan, Transition

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
