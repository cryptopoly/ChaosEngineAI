/**
 * The stage rail (audit change 1): the order of the work as one row of stages above the project's tabs, each
 * marked done, now, needs a look, under way, skipped or to do, and under it one sentence and one button for the
 * stage in view (the current one unless the user picked another). The sentence, the status and the action code
 * come from the backend (`guide.stages`); the button opens the tab that does the work, or starts the same batch the
 * Storyboard's run bar would. Stages that do not apply are hidden (no song: Song and Lip sync), except that a
 * Music video start shows and leads with the Song stage until a song is attached.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardGuide, StoryboardGuideAction, StoryboardProject, StoryboardStageId } from "../../../types";
import type { ProjectTabId } from "../projectRoute";
import { modelsMissingReason } from "../storyboardLogic";
import { useProjectEstimate } from "../useProjectEstimate";
import type { UseStoryboardResult } from "../useStoryboard";
import { batchArgs, destinationOf, railStages, shownStage, type StageLook } from "./stageLogic";
import "./guide.css";

export interface StageRailProps {
  guide: StoryboardGuide;
  project: StoryboardProject;
  story: Pick<UseStoryboardResult, "busy" | "batch" | "startBatch" | "resumeBatch">;
  musicVideo?: boolean;
  onTab: (tab: ProjectTabId) => void;
  /** The stage in view at first (tests); the current one when not given. */
  initialStage?: StoryboardStageId | null;
}

function CheckMark() {
  return (
    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="5 12 10 17 19 7" />
    </svg>
  );
}

export function StageRail({ guide, project, story, musicVideo = false, onTab, initialStage = null }: StageRailProps) {
  const { t } = useTranslation("studio");
  const [picked, setPicked] = useState<StoryboardStageId | null>(initialStage);
  const rail = railStages(guide.stages, { musicVideo });
  const shown = shownStage(rail, picked);
  const action = shown?.stage.action ?? null;
  const destination = action ? destinationOf(action) : null;
  const batch = destination?.kind === "batch" ? batchArgs(destination.batch, project, guide) : null;
  const running = story.batch?.state === "queued" || story.batch?.state === "running" || story.batch?.state === "paused";
  // FU-474, as on the run bar: with no model installed for the work, Make stays off and says why.
  const estimate = useProjectEstimate(project.id, project.revision, batch?.shotIds ?? [], batch !== null && batch.what !== "final" && !running);
  const noModel = batch ? modelsMissingReason(estimate) : null;

  const names: Record<StoryboardStageId, string> = {
    song: t("guide.stage.song", { defaultValue: "Song" }),
    look: t("guide.stage.look", { defaultValue: "Look" }),
    shots: t("guide.stage.shots", { defaultValue: "Shots" }),
    drafts: t("guide.stage.drafts", { defaultValue: "Drafts" }),
    final: t("guide.stage.final", { defaultValue: "Final" }),
    lipsync: t("guide.stage.lipsync", { defaultValue: "Lip sync" }),
    sound: t("guide.stage.sound", { defaultValue: "Sound" }),
    check: t("guide.stage.check", { defaultValue: "Check" }),
    export: t("guide.stage.export", { defaultValue: "Export" }),
  };
  const looks: Record<StageLook, string> = {
    done: t("guide.look.done", { defaultValue: "Done" }),
    current: t("guide.look.current", { defaultValue: "Now" }),
    attention: t("guide.look.attention", { defaultValue: "Needs a look" }),
    progress: t("guide.look.progress", { defaultValue: "Under way" }),
    skipped: t("guide.look.skipped", { defaultValue: "Skipped" }),
    todo: t("guide.look.todo", { defaultValue: "To do" }),
  };
  const buttons: Partial<Record<StoryboardGuideAction, string>> = {
    "attach-song": t("guide.action.attachSong", { defaultValue: "Add the song" }),
    "open-cast": t("guide.action.openCast", { defaultValue: "Open Cast" }),
    "cut-plan": t("guide.action.cutPlan", { defaultValue: "Cut the song into shots" }),
    "plan-shots": t("guide.action.planShots", { defaultValue: "Plan the shots" }),
    "open-storyboard": t("guide.action.openStoryboard", { defaultValue: "Open the Storyboard" }),
    "watch-batch": t("guide.action.watchBatch", { defaultValue: "Watch the progress" }),
    "resume-batch": t("guide.action.resumeBatch", { defaultValue: "Resume" }),
    "retry-failed": t("guide.action.retryFailed", { defaultValue: "Try them again" }),
    "make-drafts": t("guide.action.makeDrafts", { defaultValue: "Make drafts" }),
    "make-missing": t("guide.action.makeMissing", { defaultValue: "Make the missing shots" }),
    "make-final": t("guide.action.makeFinal", { defaultValue: "Make them full size" }),
    "choose-final": t("guide.action.chooseFinal", { defaultValue: "Choose the shots" }),
    "make-clips": t("guide.action.makeClips", { defaultValue: "Make the clips" }),
    "remake-stale": t("guide.action.remakeStale", { defaultValue: "Make them again" }),
    "assign-vocals": t("guide.action.assignVocals", { defaultValue: "Give each shot its vocal" }),
    "make-lipsync": t("guide.action.makeLipsync", { defaultValue: "Make the lip-sync clips" }),
    "open-coverage": t("guide.action.openCoverage", { defaultValue: "See which shots sing" }),
    "open-mixer": t("guide.action.openMixer", { defaultValue: "Open the Mixer" }),
    "open-check": t("guide.action.openCheck", { defaultValue: "Check before export" }),
    export: t("guide.action.export", { defaultValue: "Open Export" }),
  };

  function act() {
    if (!destination) return;
    if (destination.kind === "resume") void story.resumeBatch?.();
    else if (destination.kind === "batch" && batch) void story.startBatch(batch.what, batch.shotIds, batch.force, batch.draft);
    onTab(destination.tab);
  }

  // A batch button waits while another batch runs or a change is saving; opening a tab never waits.
  const startsWork = destination?.kind === "batch" || destination?.kind === "resume";
  const disabled = startsWork && (story.busy || (destination?.kind === "batch" && running) || noModel !== null);

  return (
    <nav className="stage-rail" aria-label={t("guide.label", { defaultValue: "Stages of the film" })}>
      <ol className="stage-rail__list">
        {rail.map((item) => {
          const inView = item.stage.id === shown?.stage.id;
          return (
            <li key={item.stage.id}>
              <button
                type="button"
                className={`stage-rail__step stage-rail__step--${item.look}${item.current ? " is-current" : ""}${inView ? " is-shown" : ""}`}
                aria-current={item.current ? "step" : undefined}
                aria-pressed={inView}
                title={item.stage.message}
                onClick={() => setPicked(item.stage.id)}
              >
                <span className="stage-rail__mark" aria-hidden="true">
                  {item.look === "done" ? <CheckMark /> : item.number}
                </span>
                <span className="stage-rail__name">{names[item.stage.id]}</span>
                <span className="stage-rail__look">{looks[item.look]}</span>
              </button>
            </li>
          );
        })}
      </ol>
      {shown ? (
        <div className={`stage-rail__panel stage-rail__panel--${shown.look}`} aria-live="polite">
          <p className="stage-rail__message">
            <strong>{names[shown.stage.id]}</strong>
            {" · "}
            {shown.stage.message}
          </p>
          {noModel ? <p className="muted-text stage-rail__note" role="note">{noModel}</p> : null}
          {action ? (
            <button
              type="button"
              className="primary-button stage-rail__button"
              disabled={disabled}
              title={noModel ?? undefined}
              onClick={act}
            >
              {buttons[action] ?? names[shown.stage.id]}
            </button>
          ) : null}
        </div>
      ) : null}
    </nav>
  );
}
