"""Freestyle board ops (S-M2): where the picture cards sit, and a picture dropped on the board as a new shot.

Split out of ``ops`` like ``plan_ops`` and ``song_ops``; ``ops`` imports this module at its end and registers these
ops in ``OPS``. ``board`` is a diffed top-level key, so Undo and Redo cover a move. The shot order itself is not
kept here: "Order these" on the board writes it with the ordinary ``reorder`` op.

- ``place_cards``: new positions for some cards (shot or reference ids), one step. Ids that are not on the project
  are refused; positions of shots or references that are gone are dropped at the same time.
- ``board_add_shot``: a picture becomes a new shot with that picture as its still take, placed where it was dropped,
  in one step (one Undo takes the shot, its take and its card off again).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from backend_service.storyboard.models import BOARD_EXTENT, MAX_SHOTS, BoardPos, Id, Line, Project, Shot, new_id
from backend_service.storyboard.ops import ImportTake, OpError, _Args, import_take


class PlaceCards(_Args):
    positions: dict[Id, BoardPos] = Field(min_length=1, max_length=2 * MAX_SHOTS)


class BoardAddShot(_Args):
    still: str = Field(min_length=1, max_length=4096)  # an upload handle, an output file or a local picture
    title: Line = ""
    index: int | None = Field(None, ge=0)
    x: float = Field(0.0, ge=-BOARD_EXTENT, le=BOARD_EXTENT)
    y: float = Field(0.0, ge=-BOARD_EXTENT, le=BOARD_EXTENT)


def _card_ids(p: Project) -> set[str]:
    return {s.id for s in p.shots} | {r.id for r in p.references.all()}


def place_cards(p: Project, folder: Path, a: PlaceCards) -> None:
    known = _card_ids(p)
    unknown = [i for i in a.positions if i not in known]
    if unknown:
        raise OpError(f"No shot or reference '{unknown[0]}' on this project.")
    pos = {k: v for k, v in p.board.pos.items() if k in known}
    pos.update({k: BoardPos(x=round(v.x, 1), y=round(v.y, 1)) for k, v in a.positions.items()})
    p.board.pos = pos


def board_add_shot(p: Project, folder: Path, a: BoardAddShot) -> None:
    if len(p.shots) >= MAX_SHOTS:
        raise OpError(f"A project holds at most {MAX_SHOTS} shots.")
    shot = Shot(id=new_id("s"), title=a.title)
    index = len(p.shots) if a.index is None else min(a.index, len(p.shots))
    p.shots.insert(index, shot)
    import_take(p, folder, ImportTake(shotId=shot.id, still=a.still))
    p.board.pos = {**p.board.pos, shot.id: BoardPos(x=round(a.x, 1), y=round(a.y, 1))}


BOARD_OPS = {
    "place_cards": (PlaceCards, place_cards, True),
    "board_add_shot": (BoardAddShot, board_add_shot, True),
}
BOARD_FILE_ARGS = {"board_add_shot": (("still", {"image"}),)}
