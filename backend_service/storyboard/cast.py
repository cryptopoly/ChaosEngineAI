"""Who is in a shot (B1309): the one place a shot's effective cast is worked out, and how it reaches a still.

A character marked ``inEveryShot`` is in every shot except the shots that list it in ``castOut``; a shot also
has the references it ticks itself (``referenceImages``: characters, places or styles). So a shot's effective
references are its ticked ones, in their order, followed by every ``inEveryShot`` character it neither ticks
nor leaves out. Nothing else decides this: the still request, the stale fingerprint and the planner all call
``effective_references`` / ``shot_cast``.

A still always gets the cast's names and looks written into its prompt in the same words in every shot, so
identity comes from words even when no picture can go in. The cast's pictures go in too when the still model
takes reference pictures (``style.picture_slots``, the storyboard's capability check), characters first.
"""

from __future__ import annotations

from typing import Any, Literal

from backend_service.storyboard.models import Project, Reference, Shot

CastMode = Literal["pictures", "words", "none"]
MAX_LOOK_CHARS = 300  # one character's look in the still prompt
MAX_CAST_CHARS = 1200  # the whole cast sentence


def effective_references(project: Project, shot: Shot) -> list[str]:
    """The reference ids a shot's still uses: its ticked references, then the every-shot characters it keeps."""
    ids = [i for i in shot.referenceImages if project.reference(i) is not None]
    for ref in project.references.characters:
        if ref.inEveryShot and ref.id not in ids and ref.id not in shot.castOut:
            ids.append(ref.id)
    return ids


def shot_cast(project: Project, shot: Shot) -> list[Reference]:
    """The characters in a shot, in the order of ``effective_references``."""
    characters = {r.id: r for r in project.references.characters}
    return [characters[i] for i in effective_references(project, shot) if i in characters]


def tidy(shot: Shot, cast_out_wins: bool) -> None:
    """Keep ``referenceImages`` and ``castOut`` apart: the list the caller just set wins an id named in both."""
    if cast_out_wins:
        shot.referenceImages = [i for i in shot.referenceImages if i not in shot.castOut]
    else:
        shot.castOut = [i for i in shot.castOut if i not in shot.referenceImages]


def cast_words(cast: list[Reference]) -> str:
    """'Characters: Mia, red-haired woman in a yellow raincoat; Tom.' written the same way in every shot, or ''."""
    parts = []
    for ref in cast:
        name = ref.name.strip() or "unnamed character"
        look = " ".join(ref.description.split())[:MAX_LOOK_CHARS].strip().rstrip(".")
        parts.append(f"{name}, {look}" if look else name)
    if not parts:
        return ""
    return f"Characters: {'; '.join(parts)}."[:MAX_CAST_CHARS]


def apply_to_still(project: Project, shot: Shot, paths: list[str], slots: int,
                   prompt: str) -> tuple[list[str], str, CastMode, str]:
    """(reference paths, prompt, how the cast went in, a note for the take) for one still.

    ``paths`` are the pictures of the shot's effective references, characters first (``generators.reference_paths``);
    ``slots`` is how many the still model takes. The cast's words are added to the prompt whatever the model."""
    cast = shot_cast(project, shot)
    paths = paths[:max(0, slots)]
    words = cast_words(cast)
    if words and words not in prompt:
        prompt = f"{prompt.rstrip()} {words}".strip()
    if not cast:
        return paths, prompt, "none", ""
    with_pictures = [r for r in cast if r.images]
    if with_pictures and slots > 0:
        missing = ", ".join(r.name.strip() or "unnamed" for r in cast if not r.images)
        return paths, prompt, "pictures", f"No pictures of {missing} yet, so they went in as words only." if missing else ""
    names = ", ".join(r.name.strip() or "unnamed" for r in cast)
    if with_pictures:
        note = f"The still model takes no reference pictures, so {names} went in as words only."
    else:
        note = f"No pictures of {names} yet, so they went in as words only."
    return paths, prompt, "words", note


def cast_mode(project: Project, variant: dict[str, Any] | None) -> CastMode:
    """How the every-shot characters reach this still model: as pictures, as words only, or not at all."""
    from backend_service.storyboard.style import picture_slots

    every = [r for r in project.references.characters if r.inEveryShot]
    if not every:
        return "none"
    return "pictures" if picture_slots(variant) > 0 and any(r.images for r in every) else "words"
