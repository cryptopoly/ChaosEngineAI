/**
 * The freestyle board's side column (S-M2): the chat model line when none is loaded, then the chat that fits the
 * selection. One shot selected: that card's chat (the Plan tab's Rework flow for the shot). A character, place or
 * style: its name and a way to Cast. Otherwise: the board chat, which is the project's direction chat (kept with the
 * project, changes applied in one step).
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardProject } from "../../../types";
import { PlanDirection } from "../PlanDirection";
import { PlanModelBar } from "../PlanModelBar";
import type { PlanChatDeps } from "../PlanTab";
import type { ProjectTabId } from "../projectRoute";
import { usePlanStatus } from "../usePlan";
import type { UseStoryboardResult } from "../useStoryboard";
import { BoardCardChat } from "./BoardCardChat";

export interface BoardSideProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  selected: string[];
  chat: PlanChatDeps;
  onTab: (tab: ProjectTabId) => void;
  onClearSelection: () => void;
}

export function BoardSide({ story, project, selected, chat, onTab, onClearSelection }: BoardSideProps) {
  const { t } = useTranslation("studio");
  const { status, reload } = usePlanStatus(project.id, `${chat.loadedChatModel ?? ""}`);
  const [loading, setLoading] = useState(false);
  const bar = useRef<HTMLDivElement>(null);
  const chatModel = status?.chatModel ?? null;
  const canAsk = Boolean(chatModel);
  const showModelMissing = () => {
    void reload();
    bar.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };

  async function load(key: string) {
    if (!chat.onLoadChatModel) return;
    setLoading(true);
    try {
      await chat.onLoadChatModel(key);
    } finally {
      setLoading(false);
      void reload();
    }
  }

  const only = selected.length === 1 ? selected[0] : null;
  const shotIndex = only ? project.shots.findIndex((s) => s.id === only) : -1;
  const shot = shotIndex >= 0 ? project.shots[shotIndex] : null;
  const ref = only && !shot ? [...project.references.characters, ...project.references.places, ...(project.references.styles ?? [])].find((r) => r.id === only) ?? null : null;

  return (
    <aside className="board-side" aria-label={t("board.freestyle.sideLabel", { defaultValue: "Board chat" })}>
      <div ref={bar}>
        <PlanModelBar
          chatModel={chatModel}
          known={status !== null}
          choices={chat.chatModels ?? []}
          preferredKey={chat.preferredChatModel}
          loading={loading}
          onLoad={chat.onLoadChatModel ? (key) => void load(key) : undefined}
        />
      </div>
      {shot ? (
        <BoardCardChat
          key={shot.id}
          story={story}
          projectId={project.id}
          shot={shot}
          number={shotIndex + 1}
          canAsk={canAsk}
          onModelMissing={showModelMissing}
          onBoardChat={onClearSelection}
        />
      ) : (
        <>
          {ref ? (
            <div className="board-side__ref">
              <p>
                <strong>{ref.name || t("board.freestyle.unnamed", { defaultValue: "Unnamed" })}</strong>
                {ref.description ? <span className="muted-text"> · {ref.description}</span> : null}
              </p>
              <button type="button" className="secondary-button plan-quiet-button" onClick={() => onTab("cast")}>
                {t("board.freestyle.openCast", { defaultValue: "Edit in Cast" })}
              </button>
            </div>
          ) : null}
          <PlanDirection story={story} project={project} canAsk={canAsk} onModelMissing={showModelMissing} />
        </>
      )}
    </aside>
  );
}
