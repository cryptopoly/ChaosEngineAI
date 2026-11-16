/**
 * The chat of one shot card on the freestyle board (S-M2): the shot's title and picture prompt, then the Plan tab's
 * "Rework with the model" conversation for that shot (same hook, same view, same one-step "Use this").
 */
import { useTranslation } from "react-i18next";
import type { StoryboardShot } from "../../../types";
import { PlanReworkChat } from "../PlanReworkChat";
import type { UseStoryboardResult } from "../useStoryboard";
import { useRework } from "../useRework";

export interface BoardCardChatProps {
  story: UseStoryboardResult;
  projectId: string;
  shot: StoryboardShot;
  number: number;
  canAsk: boolean;
  onModelMissing: () => void;
  onBoardChat: () => void;
}

export function BoardCardChat({ story, projectId, shot, number, canAsk, onModelMissing, onBoardChat }: BoardCardChatProps) {
  const { t } = useTranslation("studio");
  const rework = useRework(story, projectId, shot, canAsk, onModelMissing);
  return (
    <section className="board-card-chat plan-direction" aria-labelledby="board-card-chat-title">
      <div className="plan-direction__head">
        <div>
          <h3 id="board-card-chat-title">
            {shot.title
              ? t("board.plan.direct.shotTitled", { defaultValue: "Shot {n} ({title})", n: number, title: shot.title })
              : t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })}
          </h3>
          <p className="muted-text board-card-chat__prompt">
            {shot.stillPrompt || t("board.freestyle.noPrompt", { defaultValue: "No picture prompt yet." })}
          </p>
        </div>
        <button type="button" className="secondary-button plan-quiet-button" onClick={onBoardChat}>
          {t("board.freestyle.boardChat", { defaultValue: "Board chat" })}
        </button>
      </div>
      {shot.locked ? (
        <p className="muted-text">{t("board.freestyle.lockedShot", { defaultValue: "This shot is locked. Unlock it on the Storyboard tab to rework it." })}</p>
      ) : (
        <PlanReworkChat id={`board-rework-${shot.id}`} shot={shot} number={number} rework={rework} busy={story.busy} autoFocus={false} />
      )}
    </section>
  );
}
