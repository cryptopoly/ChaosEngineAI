"""What the Storyboard agent tools tell an agent beyond the project summary (FU-200 to FU-203).

The size, rate, models and caption export a project got, with notes on what they mean (a size LTX resizes, the model
generate will use, captions that will not be drawn); the clips per audio lane after an add; and what a render will
lack (no audio, unburned captions, a preview rather than the film). Small models reported a silent preview as the
finished film without these.
"""

from __future__ import annotations

from typing import Any

from backend_service.mcp_bridge.tools import ToolError


def parse_size(raw: Any) -> tuple[int, int]:
    """``size`` as (width, height): WIDTHxHEIGHT, both even and 64 to 4096 (the project's limits; yuv420p needs even)."""
    import re

    found = re.fullmatch(r"\s*(\d{2,4})\s*[xX\u00d7]\s*(\d{2,4})\s*", str(raw or ""))
    if found is None:
        raise ToolError("size is WIDTHxHEIGHT in pixels, for example 1024x576.")
    width, height = int(found.group(1)), int(found.group(2))
    if not (64 <= width <= 4096 and 64 <= height <= 4096) or width % 2 or height % 2:
        raise ToolError("size needs even numbers from 64 to 4096, for example 1024x576.")
    return width, height


def align64(value: int) -> int:
    """The nearest multiple of 64 (at least 64): the size LTX-2 clips are made at."""
    return max(64, (value + 32) // 64 * 64)


def model_note(ctx: Any, kind: str, chosen: str | None) -> str | None:
    """What generate will use for stills or clips: the default when none is set, or why the set one cannot run."""
    from backend_service.mcp_bridge.tools import _pick_variant

    try:
        picked = _pick_variant(ctx, kind, chosen)
    except ToolError as exc:
        return f"{kind}Model: {exc}"
    except Exception:  # noqa: BLE001 - a hint only; the generate call reports the real problem
        return None
    return None if chosen else f"{kind}Model is unset: generate uses {picked['id']}."


def settings_info(ctx: Any, project: Any, created: bool) -> dict[str, Any]:
    """The project's effective size, rate, models and caption export, with notes on what they mean, so an agent
    sees what it got instead of assuming its request landed."""
    d, export = project.defaults, project.export
    settings = {"size": f"{d.width}x{d.height}", "aspect": d.exportPreset, "resolution": d.resolution, "fps": d.fps,
                "imageModel": d.imageModel, "videoModel": d.videoModel, "voice": d.voice,
                "captionsBurn": export.captionsBurn, "captionsSrt": export.captionsSrt, "upscale": export.upscale}
    notes = [n for n in (model_note(ctx, "image", d.imageModel), model_note(ctx, "video", d.videoModel)) if n]
    if d.width % 64 or d.height % 64:
        notes.append(f"LTX clips are made at multiples of 64 ({align64(d.width)}x{align64(d.height)}) and fitted to "
                     f"{d.width}x{d.height}. Pass size (e.g. 1024x576) for an exact fit.")
    if created:
        if not export.captionsBurn:
            notes.append("Captions are drawn into the picture only with fields.export.captionsBurn true (create or edit).")
        notes.append("Music and narration play only once made (make_music, speak) and added with storyboard_audio add.")
    return {"settings": settings, **({"notes": notes} if notes else {})}


def lanes(project: Any) -> dict[str, int]:
    """Clips per audio lane (music, voice, sfx, import), so the agent sees the clip landed."""
    counts: dict[str, int] = {}
    for clip in project.audio:
        counts[clip.kind] = counts.get(clip.kind, 0) + 1
    return counts


def render_warnings(project: Any, preview: bool) -> list[str]:
    """What the film will lack, said before the render so an agent does not report a silent or uncaptioned film."""
    out = []
    if not project.audio:
        out.append("The project has no audio lanes, so the film is silent apart from the clips' own sound. Add speak or "
                   "make_music output with storyboard_audio add.")
    if project.export.captionsBurn and not project.captions:
        out.append("Captions burn-in is on but the project has no captions: add them with storyboard_captions first "
                   "(generate from the voice lane, or add), or the film has no captions.")
    if project.captions and not project.export.captionsBurn:
        out.append("The project has captions but captionsBurn is off, so they are not drawn into the picture. Set it with "
                   "storyboard_project edit fields.export.captionsBurn true.")
    if preview:
        out.append("This is a preview: a small fast draft, not the finished film. Call again with preview: false for the final MP4.")
    return out
