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

import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
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
    stems: list[str] = field(default_factory=list)  # file names written next to the film


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


def _codec_args(plan: RenderPlan) -> list[str]:
    """H.264 arguments for the chosen encoder (libx264 when the binary has it, else a platform encoder)."""
    return ffmpeg_bin.h264_args(plan.preview, width=plan.width, height=plan.height, fps=plan.fps)


def _inp(path: str, *pre: str) -> list[str]:
    return ["-protocol_whitelist", ffmpeg_bin.PROTOCOLS, *pre, "-i", path]


PRUNE_AFTER_S = 7 * 24 * 3600
# Cached intermediates and the scratch files of a run (``<run id>-...``), the only names this engine writes.
_WORK_NAME = re.compile(r"(?:(?:shot|card)-[0-9a-f]{32}\.mov|[0-9a-f]{10}-.+)")


def prune_intermediates(work: Path, keep: set[str], now: float | None = None) -> list[str]:
    """Remove intermediates this plan does not use that were last used more than 7 days ago.

    Every trim, speed or size change makes a new intermediate, so without this ``work_dir`` only grows.
    The age is the file's own time (when it was made); one the current plan uses is always kept.
    """
    cutoff = (now if now is not None else time.time()) - PRUNE_AFTER_S
    removed: list[str] = []
    try:
        entries = list(work.iterdir())
    except OSError:
        return removed
    for path in entries:
        try:
            if (path.name in keep or not _WORK_NAME.fullmatch(path.name) or path.is_symlink()
                    or not path.is_file() or path.stat().st_mtime >= cutoff):
                continue
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass
    return removed


_SCRATCH_NAME = re.compile(r"[0-9a-f]{10}-.+")
_PARTIAL_NAME = re.compile(r"\..+\.partial-[0-9a-f]{6,10}(?:-\w+)?\.(?:mp4|wav)")


def remove_leftovers(work: Path, exports: Path) -> list[str]:
    """Delete what a killed render left behind: its run-id scratch files and half-written ``.partial`` outputs.

    Call it only when no render of this project is running (the render manager runs one at a time). The
    finished per-shot intermediates (``shot-<hash>.mov``) are kept: they are what a resumed render reuses.
    """
    removed: list[str] = []
    for folder, pattern in ((work, _SCRATCH_NAME), (exports, _PARTIAL_NAME)):
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for path in entries:
            try:
                if pattern.fullmatch(path.name) and path.is_file() and not path.is_symlink():
                    path.unlink()
                    removed.append(path.name)
            except OSError:
                pass
    return removed


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


def intermediate_names(plan: RenderPlan) -> list[str]:
    """The cached per-shot file name of every shot, in order (the resume manifest lists these)."""
    return [f"shot-{shot_cache_key(s, plan)[:32]}.mov" for s in plan.shots]


STEMS = (("voice", "voice"), ("music", "music"), ("sfx", "sfx"), ("import", "import"))


def _write_stems(plan: RenderPlan, work: Path, film_input: list[str], lanes: list[Lane], total_samples: int,
                 total: float, progress: ProgressCb, cancel_event: threading.Event | None,
                 warnings: list[str]) -> list[str]:
    """Each audio lane as its own 48 kHz stereo WAV the length of the film, plus the shots' own sound.

    A stem is the lane's clips with their gain, fades, trims, loops and positions, and nothing else: no
    ducking (that depends on the voice lane), no loudness normalisation. A lane with no clips has no file.
    """
    assert plan.stems_base
    base = Path(plan.stems_base)
    jobs: list[tuple[str, list[str]]] = [("shot-sound", [*film_input, "-map", "0:a"])]
    scripts: list[Path] = []
    for kind, name in STEMS:
        mine = [lane for lane in lanes if lane.clip.kind == kind]
        if not mine:
            continue
        renumbered = [Lane(lane.clip, n + 1, lane.source_seconds) for n, lane in enumerate(mine)]
        graph_text, mix_warnings = build_mix_graph(renumbered, total_samples, None)
        warnings += [f"{name} stem: {w}" for w in mix_warnings]
        graph = work / f"{uuid.uuid4().hex[:10]}-stem-{name}.txt"
        graph.write_text(graph_text, encoding="utf-8")
        scripts.append(graph)
        args = ["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={SAMPLE_RATE}"]
        for lane in renumbered:
            args += _inp(lane.clip.path)
        jobs.append((name, [*args, "-filter_complex_script", graph.name, "-map", "[aout]"]))
    written: list[str] = []
    try:
        for index, (name, args) in enumerate(jobs):
            final = base.with_name(f"{base.name}-stem-{name}.wav")
            tmp = base.with_name(f".{base.name}-stem-{name}.partial-{uuid.uuid4().hex[:6]}.wav")
            try:
                _run([*args, "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-map_metadata", "-1",
                      "-t", _f(total), "-f", "wav", str(tmp)], cwd=work, expected_s=total,
                     report=lambda f, i=index: progress("stems", (i + f) / len(jobs)), cancel_event=cancel_event)
                os.replace(tmp, final)
                written.append(final.name)
            finally:
                _unlink(tmp)
    finally:
        for path in scripts:
            _unlink(path)
    return written


def render(plan: RenderPlan, progress_cb: ProgressCb | None = None,
           cancel_event: threading.Event | None = None) -> RenderResult:
    plan = plan.validate()
    progress = progress_cb or (lambda stage, fraction: None)
    warnings: list[str] = []
    work = Path(plan.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    prune_intermediates(work, set(intermediate_names(plan)))
    if plan.captions_srt and "subtitles" not in ffmpeg_bin.ffmpeg_filters():
        warnings.append("Captions were not burned in: this build of ffmpeg has no 'subtitles' filter. "
                        "The .srt file is still written.")
        plan = dataclasses.replace(plan, captions_srt=None)
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
                  *_codec_args(plan), "-c:a", "pcm_s16le", "-map_metadata", "-1", "-f", "mov",
                  tmp.name], cwd=work, expected_s=frames / plan.fps, report=report,
                 cancel_event=cancel_event)
            os.replace(tmp, target)
            rendered += 1
        transitions = [s.transition_in for s in plan.shots]

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
                  *_codec_args(plan), "-f", "mov", joined_v.name],
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

        upscale_to = None if plan.preview else plan.upscale_to
        graph_text, mix_warnings = build_mix_graph(lanes, total_samples, loudnorm, captions_name, upscale_to)
        warnings += mix_warnings
        graph = script("mix.txt", graph_text)
        video_args = (["-map", "[vout]", *_codec_args(plan)] if captions_name or upscale_to
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
        stems: list[str] = []
        if plan.stems_base and not plan.preview:
            try:
                stems = _write_stems(plan, work, film_input, lanes, total_samples, timing.total, progress,
                                     cancel_event, warnings)
            except RenderError as exc:  # the film is done; say the stems are not
                warnings.append(f"Stems were not written: {exc}")
        return RenderResult(path=str(out), duration_s=duration, shots_rendered=rendered,
                            shots_cached=cached, ffmpeg_version=ffmpeg_bin.ffmpeg_version(),
                            warnings=warnings, loudness=loudness, stems=stems)
    finally:
        _unlink(out_tmp)
        for path in scratch:
            _unlink(path)
