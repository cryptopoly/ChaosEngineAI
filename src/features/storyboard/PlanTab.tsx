/**
 * The Plan tab of a project (S8, film from an idea): a brief and style pictures, shots the loaded chat model
 * proposes, each shot editable in place with its own "Rework with the model" chat, and a direction chat for
 * the whole film. Everything the model writes is a proposal until the user applies it, and every applied
 * change is an ordinary op, so Undo, Redo and the revision check hold. Nothing renders from here; the
 * Storyboard tab makes the pictures and clips.
 *
 * A tab of its own (not a panel on the Storyboard tab) because planning is a step before the board, it needs
 * room for two conversations, and the board's shot cards stay as they are.
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { proposeStoryboardPlan, type StoryboardProposal } from "../../api/storyboardPlan";
import type { ImageOutputArtifact, StoryboardProject } from "../../types";
import { plannedShotFromProposal } from "./castLock";
import { PlanBrief } from "./PlanBrief";
import { PlanDirection } from "./PlanDirection";
import { planOf, stylesOf } from "./planLogic";
import { PlanModelBar, type ChatModelChoice } from "./PlanModelBar";
import { PlanProposal } from "./PlanProposal";
import { PlanShotList } from "./PlanShotList";
import { SongPanel } from "./song/SongPanel";
import type { ProjectTabId } from "./projectRoute";
import { failureOf, useElapsed, usePlanStatus } from "./usePlan";
import type { UseStoryboardResult } from "./useStoryboard";
import "./plan.css";

/** What the Plan tab needs from the app to offer loading a chat model. */
export interface PlanChatDeps {
  /** Installed chat models the user can load from here. */
  chatModels?: ChatModelChoice[];
  preferredChatModel?: string | null;
  /** The loaded chat model's name, from the workspace (changes when a load finishes). */
  loadedChatModel?: string | null;
  onLoadChatModel?: (key: string) => Promise<void>;
}

export interface PlanTabProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  imageGallery: ImageOutputArtifact[];
  chat: PlanChatDeps;
  onTab: (tab: ProjectTabId) => void;
  /** Open a shot in the shot editor (a lip-sync coverage row's Lengthen or Shorten); without it the Storyboard opens. */
  onOpenShot?: (shotId: string, field?: "length") => void;
}

export function PlanTab({ story, project, imageGallery, chat, onTab, onOpenShot }: PlanTabProps) {
  const { t } = useTranslation("studio");
  const plan = planOf(project);
  const styleKey = stylesOf(project).map((r) => `${r.id}:${r.images.length}:${(r.palette ?? []).join("")}:${r.description.length}`).join("|");
  const statusKey = `${chat.loadedChatModel ?? ""}|${project.defaults.imageModel ?? ""}|${project.brand?.styleUse ?? ""}|${styleKey}`;
  const { status, reload } = usePlanStatus(project.id, statusKey);
  const [proposal, setProposal] = useState<StoryboardProposal | null>(null);
  const [proposing, setProposing] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const elapsed = useElapsed(proposing);
  const modelBar = useRef<HTMLDivElement>(null);
  const toBoard = useRef<HTMLButtonElement>(null);
  const lastAsk = useRef<{ brief: string; filmSeconds: number | null; shots: number | null } | null>(null);
  const chatModel = status?.chatModel ?? null;
  const canAsk = Boolean(chatModel);

  async function propose(brief: string, filmSeconds: number | null, shots: number | null) {
    lastAsk.current = { brief, filmSeconds, shots };
    setProposing(true);
    setProblem(null);
    try {
      setProposal(await proposeStoryboardPlan(project.id, { brief, filmSeconds, shots }));
    } catch (err) {
      const failure = failureOf(err);
      if (failure.code === "no-model") void reload();
      setProblem(failure.message);
    } finally {
      setProposing(false);
    }
  }

  async function use(mode: "append" | "replace") {
    if (!proposal) return;
    const ok = await story.op("apply_plan", {
      mode,
      shots: proposal.shots.map((shot) => plannedShotFromProposal(shot, project)),
    });
    if (ok) {
      setProposal(null);
      // The proposal and its buttons are gone: land on the next step instead of the page.
      requestAnimationFrame(() => toBoard.current?.focus());
    }
  }

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

  const showModelMissing = () => {
    void reload();
    modelBar.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };

  const style = status?.style;
  const styleLine =
    style?.mode === "pictures"
      ? t("board.style.modePictures", { defaultValue: "The style goes in as pictures: {model} takes {n, plural, one {# reference picture} other {# reference pictures}}.", model: style.stillModel ?? "", n: style.pictureSlots })
      : style?.mode === "words"
        ? project.brand?.styleUse === "words"
          ? t("board.style.modeWordsChosen", { defaultValue: "The style goes into each prompt as words, as set in Cast." })
          : t("board.style.modeWords", { defaultValue: "The style goes into each prompt as words: {model} cannot take reference pictures.", model: style.stillModel ?? t("board.style.thisModel", { defaultValue: "this still model" }) })
        : null;

  return (
    <div className="plan-tab">
      <div className="plan-tab__main">
        <div ref={modelBar}>
          <PlanModelBar
            chatModel={chatModel}
            known={status !== null}
            choices={chat.chatModels ?? []}
            preferredKey={chat.preferredChatModel}
            loading={loading}
            onLoad={chat.onLoadChatModel ? (key) => void load(key) : undefined}
          />
        </div>

        <PlanBrief
          story={story}
          project={project}
          imageGallery={imageGallery}
          canPropose={canAsk}
          proposing={proposing}
          elapsed={elapsed}
          modelName={chatModel?.name ?? null}
          hasProposal={proposal != null || project.shots.length > 0}
          onPropose={(brief, filmSeconds, shots) => void propose(brief, filmSeconds, shots)}
          onOpenCast={() => onTab("cast")}
        />

        {problem ? (
          <div className="callout error" role="alert">
            <p>{problem}</p>
          </div>
        ) : null}

        <SongPanel story={story} project={project} onOpenShot={onOpenShot ?? (() => onTab("storyboard"))} />

        {proposal ? (
          <PlanProposal
            proposal={proposal}
            project={project}
            target={plan.filmSeconds}
            busy={story.busy || proposing}
            onUse={(mode) => void use(mode)}
            onAgain={() => lastAsk.current && void propose(lastAsk.current.brief, lastAsk.current.filmSeconds, lastAsk.current.shots)}
            onDiscard={() => setProposal(null)}
          />
        ) : null}

        <section className="plan-shots" aria-labelledby="plan-shots-title">
          <div className="plan-section-head plan-shots__head">
            <div>
              <h3 id="plan-shots-title">{t("board.plan.shots.title", { defaultValue: "Shots" })}</h3>
              {styleLine ? <p className="muted-text plan-shots__style" title={style?.words || undefined}>{styleLine}</p> : null}
            </div>
            {project.shots.length > 0 ? (
              <button type="button" className="primary-button" ref={toBoard} onClick={() => onTab("storyboard")}>
                {t("board.plan.shots.toBoard", { defaultValue: "Make them in Storyboard" })}
              </button>
            ) : null}
          </div>
          {project.shots.length === 0 ? (
            <p className="plan-shots__empty muted-text">
              {t("board.plan.shots.empty", { defaultValue: "No shots yet. Write a brief and propose shots, or add them one by one in Storyboard." })}
            </p>
          ) : (
            <PlanShotList story={story} project={project} canAsk={canAsk} onModelMissing={showModelMissing} />
          )}
        </section>
      </div>

      <aside className="plan-tab__aside">
        <PlanDirection story={story} project={project} canAsk={canAsk} onModelMissing={showModelMissing} />
      </aside>
    </div>
  );
}
