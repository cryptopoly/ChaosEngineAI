/**
 * The Board tab of a project (S-M2, item 8): a freestyle board. The project's shots (their stills) and its
 * characters, places and styles are picture cards on a plane that pans and zooms; drag them anywhere, select several,
 * and "Order these" turns where they sit into the film's shot order (one `reorder`, one Undo). Pictures dropped on the
 * board, or added from files, become new shots or reference pictures. Each shot card has its own chat (the Plan tab's
 * Rework flow) and the board has one chat (the project's direction chat). With a song attached, its sections run along
 * the bottom as a ruler. Card positions are saved with the project (`place_cards`), so they survive a reload and Undo
 * moves a card back.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardProject } from "../../../types";
import type { PlanChatDeps } from "../PlanTab";
import type { ProjectTabId } from "../projectRoute";
import type { UseStoryboardResult } from "../useStoryboard";
import { BoardAddPictures } from "./BoardAddPictures";
import { BoardCanvas } from "./BoardCanvas";
import { orderShots, sameOrder, type Pos } from "./boardLogic";
import { BoardSide } from "./BoardSide";
import { BoardToolbar } from "./BoardToolbar";
import { useBoardInput } from "./useBoardInput";
import { useBoardPositions } from "./useBoardPositions";
import { useBoardView } from "./useBoardView";
import "../plan.css";
import "../song/song.css";
import "./board.css";

export interface BoardTabProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  chat: PlanChatDeps;
  onTab: (tab: ProjectTabId) => void;
}

export function BoardTab({ story, project, chat, onTab }: BoardTabProps) {
  const { t } = useTranslation("studio");
  const viewport = useRef<HTMLDivElement>(null);
  const positions = useBoardPositions(story, project);
  const view = useBoardView(viewport);
  const [selected, setSelectedRaw] = useState<string[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [multi, setMulti] = useState(false);
  const [adding, setAdding] = useState<{ files: File[]; at: Pos } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const { cards, pos } = positions;

  // Cards that left the project (Undo, another tab, the agent) leave the selection too.
  const known = useMemo(() => new Set(cards.map((c) => c.id)), [cards]);
  const selection = selected.filter((id) => known.has(id));
  const setSelected = (ids: string[]) => setSelectedRaw(ids.filter((id, i) => known.has(id) && ids.indexOf(id) === i));
  const input = useBoardInput(viewport, cards, positions, view, { selected: selection, setSelected, current, setCurrent }, multi);

  // Show every card the first time the board opens.
  const fitted = useRef(false);
  useEffect(() => {
    if (fitted.current || cards.length === 0) return;
    fitted.current = true;
    view.fit(Object.values(pos));
  }, [cards.length, pos, view]);

  const shotIds = project.shots.map((s) => s.id);
  const selectedShots = selection.filter((id) => shotIds.includes(id)).length;
  const nextOrder = orderShots(shotIds, selection, pos);
  const canOrder = shotIds.length > 1 && !sameOrder(nextOrder, shotIds);

  async function order() {
    if (!canOrder) return;
    if (await story.op("reorder", { order: nextOrder })) {
      setNotice(t("board.freestyle.ordered", { defaultValue: "Shot order set from the board. Undo puts it back." }));
    }
  }

  function centre(): Pos {
    const rect = viewport.current?.getBoundingClientRect();
    const v = view.view;
    return { x: ((rect?.width ?? 600) / 2 - v.x) / v.zoom, y: ((rect?.height ?? 400) / 2 - v.y) / v.zoom };
  }

  return (
    <div className="board-tab">
      <div className="board-tab__main">
        <BoardToolbar
          zoom={view.view.zoom}
          multi={multi}
          selectedCount={selection.length}
          selectedShots={selectedShots}
          canOrder={canOrder}
          busy={story.busy}
          onZoomIn={() => view.zoomBy(1.25)}
          onZoomOut={() => view.zoomBy(0.8)}
          onFit={() => view.fit(Object.values(pos))}
          onMulti={setMulti}
          onClear={() => setSelected([])}
          onOrder={() => void order()}
          onAddFiles={(files) => setAdding({ files, at: centre() })}
        />
        <p id="board-keys-hint" className="muted-text board-tab__hint">
          {t("board.freestyle.keysHint", { defaultValue: "Drag cards anywhere. Keys: arrows go to the next card; Space selects; arrows move selected cards (Shift for a bigger step, Alt goes to the next card instead); Escape clears." })}
        </p>
        <BoardCanvas
          projectId={project.id}
          viewport={viewport}
          positions={positions}
          view={view}
          selection={{ selected: selection, setSelected, current, setCurrent }}
          input={input}
          song={project.song ?? null}
          onDropPictures={(files, at) => setAdding({ files, at })}
        />
        {project.song ? null : (
          <p className="muted-text board-tab__hint">
            {t("board.freestyle.noSong", { defaultValue: "Attach a song on the Plan tab to see its sections along the bottom." })}
          </p>
        )}
        <p className="muted-text board-tab__notice" role="status" aria-live="polite">{notice ?? ""}</p>
      </div>
      <BoardSide story={story} project={project} selected={selection} chat={chat} onTab={onTab} onClearSelection={() => setSelected([])} />
      {adding ? (
        <BoardAddPictures
          story={story}
          files={adding.files}
          at={adding.at}
          onCancel={() => setAdding(null)}
          onDone={(failed) => {
            setAdding(null);
            setNotice(
              failed.length > 0
                ? t("board.freestyle.addFailed", { defaultValue: "Not added: {names}", names: failed.join("; ") })
                : t("board.freestyle.added", { defaultValue: "Pictures added. Undo takes the last one off." }),
            );
          }}
        />
      ) : null}
    </div>
  );
}
