/**
 * The freestyle board's canvas (S-M2): the picture cards laid out freely on a plane that pans and zooms, pictures
 * dropped from the desktop (they go to the caller, with the board point they fell on), and the song's sections as a
 * ruler along the bottom when the project has a song. Input rules are in `useBoardInput`, the view in `useBoardView`.
 */
import { useState, type DragEvent, type RefObject } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardSong } from "../../../types";
import { SongRuler } from "../song/SongRuler";
import { BoardCard } from "./BoardCard";
import { toBoard, type Pos } from "./boardLogic";
import type { BoardPositions } from "./useBoardPositions";
import type { BoardSelection, useBoardInput } from "./useBoardInput";
import type { BoardViewControls } from "./useBoardView";

export interface BoardCanvasProps {
  projectId: string;
  viewport: RefObject<HTMLDivElement>;
  positions: BoardPositions;
  view: BoardViewControls;
  selection: BoardSelection;
  input: ReturnType<typeof useBoardInput>;
  song: StoryboardSong | null;
  onDropPictures: (files: File[], at: Pos) => void;
}

function hasFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

export function BoardCanvas({ projectId, viewport, positions, view, selection, input, song, onDropPictures }: BoardCanvasProps) {
  const { t } = useTranslation("studio");
  const [over, setOver] = useState(false);
  const { cards, pos } = positions;
  const { zoom, x, y } = view.view;
  const current = selection.current && pos[selection.current] ? selection.current : cards[0]?.id ?? null;

  return (
    <div
      ref={viewport}
      className={`board-viewport${over ? " is-drop-target" : ""}${song ? " has-song" : ""}`}
      role="group"
      aria-roledescription={t("board.freestyle.canvasRole", { defaultValue: "board" })}
      aria-label={t("board.freestyle.canvasLabel", { defaultValue: "Picture cards" })}
      aria-describedby="board-keys-hint"
      onPointerDown={input.onBackgroundPointerDown}
      onPointerMove={input.onPointerMove}
      onPointerUp={input.onPointerUp}
      onPointerCancel={input.onPointerUp}
      onDragOver={(event) => {
        if (!hasFiles(event)) return;
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        if (!hasFiles(event)) return;
        event.preventDefault();
        setOver(false);
        const files = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("image/"));
        const rect = viewport.current?.getBoundingClientRect();
        if (files.length > 0) onDropPictures(files, toBoard(view.view, event.clientX - (rect?.left ?? 0), event.clientY - (rect?.top ?? 0)));
      }}
    >
      <div className="board-plane" style={{ transform: `translate(${x}px, ${y}px) scale(${zoom})` }}>
        {cards.map((card) => (
          <BoardCard
            key={card.id}
            projectId={projectId}
            card={card}
            pos={pos[card.id]}
            selected={selection.selected.includes(card.id)}
            current={card.id === current}
            dragging={input.dragging && selection.selected.includes(card.id)}
            onPointerDown={input.onCardPointerDown}
            onKeyDown={input.onCardKeyDown}
            onFocus={selection.setCurrent}
          />
        ))}
      </div>
      {cards.length === 0 ? (
        <p className="board-viewport__empty muted-text">
          {t("board.freestyle.empty", { defaultValue: "No pictures yet. Drop pictures here, or add shots and cast in the other tabs." })}
        </p>
      ) : null}
      {over ? <p className="board-viewport__drop" aria-hidden="true">{t("board.freestyle.dropHere", { defaultValue: "Drop to add the pictures here" })}</p> : null}
      {song ? (
        <div className="board-viewport__song" onPointerDown={(event) => event.stopPropagation()}>
          <SongRuler seconds={song.seconds} sections={song.sections} compact />
        </div>
      ) : null}
    </div>
  );
}
