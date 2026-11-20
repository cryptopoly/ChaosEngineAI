/**
 * The Overview tab of a project (D-65): the next step as one button, a checklist that ticks as the work happens,
 * the shots at a glance, and the project settings (open while the project is new, collapsed once it has shots).
 * Cast is optional: the next step never asks for it and its checklist row says so.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardProject, StoryboardTimeline } from "../../types";
import { listFilms } from "./FilmList";
import { previewIsStale, readRenderedRevision } from "./mixer/mixerLogic";
import {
  checklist,
  checklistComplete,
  clipCount,
  nextCardMode,
  nextStep,
  recentActivity,
  stillOnlyCount,
  type ActivityLine,
  type ChecklistId,
  type NextStepId,
} from "./overviewLogic";
import { pausedCounts, pausedProgressOf, pausedReasonText } from "./PausedBatchBar";
import { ProjectDefaultsPanel } from "./ProjectDefaultsPanel";
import { ProjectImage } from "./ProjectMedia";
import { RoughCutButton } from "./preview/RoughCutButton";
import { draftPassTitle, type TFn } from "./RunBar";
import type { ProjectWorkspaceDeps } from "./ProjectShell";
import type { ProjectTabId } from "./projectRoute";
import { batchProgress, filmSeconds, missingShotIds, modelsMissingReason, primaryPass, relativeTime, staleCount, thumbnailOf } from "./storyboardLogic";
import { useProjectEstimate } from "./useProjectEstimate";
import type { UseStoryboardResult } from "./useStoryboard";
import type { UseStoryboardRenderResult } from "./useStoryboardRender";

export interface OverviewTabProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  deps: ProjectWorkspaceDeps;
  render?: UseStoryboardRenderResult;
  onTab: (tab: ProjectTabId) => void;
  onOpenShot: (shotId: string) => void;
  /** Changes when the header asks to open Project settings. */
  settingsRequest?: number;
}

/** Whether a film was ever exported from this project (its exports folder lists one). */
function useHasFilm(projectId: string, version: string | null): boolean {
  const [has, setHas] = useState(false);
  useEffect(() => {
    let live = true;
    listFilms(projectId)
      .then((films) => live && setHas(films.some((film) => film.name.endsWith(".mp4"))))
      .catch(() => live && setHas(false));
    return () => {
      live = false;
    };
  }, [projectId, version]);
  return has;
}

export function OverviewTab({ story, project, timeline, deps, render, onTab, onOpenShot, settingsRequest }: OverviewTabProps) {
  const { t, i18n } = useTranslation("studio");
  const shots = project.shots;
  const activity = recentActivity(project);
  const activityText = (line: ActivityLine): string => {
    const when = relativeTime(line.at, Date.now(), i18n.language);
    if (line.state === "done" && line.failed === 0)
      return t("overview.activity.done", { defaultValue: "{n, plural, one {Batch of # shot finished} other {Batch of # shots finished}} · {when}", n: line.shots, when });
    if (line.state === "done")
      return t("overview.activity.partly", { defaultValue: "Batch finished, {failed} of {n} shots failed · {when}", failed: line.failed, n: line.shots, when });
    if (line.state === "cancelled") return t("overview.activity.cancelled", { defaultValue: "Batch cancelled · {when}", when });
    if (line.state === "interrupted") return t("overview.activity.interrupted", { defaultValue: "Batch interrupted (the app closed) · {when}", when });
    return t("overview.activity.failed", { defaultValue: "Batch failed · {when}", when });
  };
  const clips = clipCount(project);
  const stills = stillOnlyCount(project);
  const stale = staleCount(project);
  const cast = project.references.characters.length + project.references.places.length + (project.references.styles?.length ?? 0);
  const rendered = readRenderedRevision(project.id);
  const previewCurrent = (render?.preview != null || rendered != null) && !previewIsStale(rendered, project.revision);
  const exported = useHasFilm(project.id, render?.exported?.renderId ?? null) || render?.exported != null;
  const facts = { previewCurrent, exported };
  const step = nextStep(project, facts);
  const rows = checklist(project, facts);
  const missingIds = missingShotIds(shots);
  // The same estimate and rule as the run bar (`primaryPass`), so the two never name different primaries.
  const batchState = story.batch?.state ?? null;
  const idle = batchState !== "queued" && batchState !== "running" && batchState !== "paused";
  const estimate = useProjectEstimate(project.id, project.revision, missingIds, idle && step.id === "makeMissing");
  const mode = nextCardMode({ batchState, step: step.id, primary: primaryPass(shots, missingIds.length, estimate) });
  const running = mode === "running";
  const progress = running && story.batch ? batchProgress(story.batch) : null;
  const paused = mode === "paused" && story.batch ? pausedCounts(story.batch, pausedProgressOf(story.batchProgress)) : null;
  // FU-474: with no image or video model installed for the work, Make stays off and says why (the estimate's sentences).
  const noModelReason = step.id === "makeMissing" ? modelsMissingReason(estimate) : null;

  const stepText: Record<NextStepId, { title: string; body: string; button: string }> = {
    writeShots: {
      title: t("overview.next.writeShots.title", { defaultValue: "Write your shots" }),
      body: t("overview.next.writeShots.body", {
        defaultValue: "Add your shots and give each a line of prompt. A cast is optional: a street or a crowd works on its own.",
      }),
      button: t("overview.next.writeShots.button", { defaultValue: "Write your shots" }),
    },
    makeMissing: {
      title: t("overview.next.makeMissing.title", { defaultValue: "{n, plural, one {Make # shot} other {Make # shots}}", n: step.count }),
      body: step.stills
        ? t("overview.next.makeMissing.bodyStills", {
            defaultValue: "{none, plural, =0 {} one {# shot has no take; } other {# shots have no take; }}{stills, plural, one {# has a still but no clip.} other {# have a still but no clip.}} Making them adds what is missing: a still and a clip, or just the clip.",
            none: step.count - step.stills,
            stills: step.stills,
          })
        : t("overview.next.makeMissing.body", {
            defaultValue: "{n, plural, one {# shot has no take yet.} other {# shots have no take yet.}} Making them adds a still, then a clip, for each.",
            n: step.count,
          }),
      button: t("overview.next.makeMissing.button", { defaultValue: "{n, plural, one {Make # missing} other {Make # missing}}", n: step.count }),
    },
    makeClips: {
      title: t("overview.next.makeClips.title", { defaultValue: "Turn the stills into clips" }),
      body: t("overview.next.makeClips.body", {
        defaultValue: "{n, plural, one {# shot has a still but no clip, so the film leaves it out.} other {# shots have a still but no clip, so the film leaves them out.}}",
        n: step.count,
      }),
      button: t("overview.next.makeClips.button", { defaultValue: "Open the Storyboard" }),
    },
    addSound: {
      title: t("overview.next.addSound.title", { defaultValue: "Add voice or music" }),
      body: t("overview.next.addSound.body", { defaultValue: "Optional. Place voice, effects or music in the Mixer, or go straight to a preview." }),
      button: t("overview.next.addSound.button", { defaultValue: "Open the Mixer" }),
    },
    renderPreview: {
      title: t("overview.next.renderPreview.title", { defaultValue: "Render a preview" }),
      body: t("overview.next.renderPreview.body", { defaultValue: "Watch and hear the whole film before you export it." }),
      button: t("overview.next.renderPreview.button", { defaultValue: "Open the Mixer" }),
    },
    export: {
      title: t("overview.next.export.title", { defaultValue: "Export the film" }),
      body: t("overview.next.export.body", { defaultValue: "Write the finished MP4, with captions and an AI tag if you want them." }),
      button: t("overview.next.export.button", { defaultValue: "Open Export" }),
    },
    done: {
      title: t("overview.next.done.title", { defaultValue: "Done" }),
      body: t("overview.next.done.body", { defaultValue: "Export again after a change, or make a new take of any shot." }),
      button: t("overview.next.done.button", { defaultValue: "Open Export" }),
    },
  };
  const rowText: Record<ChecklistId, string> = {
    format: t("overview.check.format", { defaultValue: "Name and format" }),
    cast: t("overview.check.cast", { defaultValue: "Cast (optional)" }),
    shots: t("overview.check.shots", { defaultValue: "Shots" }),
    takes: t("overview.check.takes", { defaultValue: "A clip for every shot" }),
    sound: t("overview.check.sound", { defaultValue: "Sound (optional)" }),
    preview: t("overview.check.preview", { defaultValue: "Preview" }),
    export: t("overview.check.export", { defaultValue: "Export" }),
  };
  const text = stepText[step.id];

  function act() {
    if (step.id === "makeMissing") {
      void story.startBatch("both", missingIds);
    }
    onTab(step.tab);
  }

  function makeDrafts() {
    void story.startBatch("still", missingIds, false, true);
    onTab("storyboard");
  }

  return (
    <div className="overview-tab">
      <section className="overview-tab__next" aria-labelledby="overview-next-title">
        <span className="eyebrow">{t("overview.nextStep", { defaultValue: "Next step" })}</span>
        {running ? (
          <>
            <h3 id="overview-next-title">{t("overview.next.making", { defaultValue: "Making shots" })}</h3>
            <p className="muted-text" role="status">
              {progress
                ? t("overview.next.makingProgress", { defaultValue: "Shot {n} of {total}", n: progress.shotNumber, total: progress.shotCount })
                : t("board.batch.starting", { defaultValue: "Starting…" })}
            </p>
            <div className="overview-tab__actions">
              {story.pauseBatch ? (
                <button
                  type="button"
                  className="secondary-button"
                  disabled={story.pausing}
                  title={t("board.pause.pauseTitle", { defaultValue: "Stops after the step in progress. The rest waits, also across a restart, until you resume." })}
                  onClick={() => void story.pauseBatch?.()}
                >
                  {story.pausing ? t("board.pause.pausing", { defaultValue: "Pausing after this step…" }) : t("board.pause.pause", { defaultValue: "Pause" })}
                </button>
              ) : null}
              <button type="button" className="secondary-button" onClick={() => void story.cancelBatch()}>
                {t("board.batch.cancel", { defaultValue: "Cancel batch" })}
              </button>
            </div>
          </>
        ) : paused && story.batch ? (
          <>
            <h3 id="overview-next-title">{t("board.pause.title", { defaultValue: "Paused" })}</h3>
            <p className="muted-text" role="status">
              {pausedReasonText(t as unknown as (key: string, options: Record<string, unknown>) => string, story.batch.pauseReason ?? null)}{" "}
              {t("board.pause.steps", {
                defaultValue: "{done} of {total} steps done, {left, plural, one {# left} other {# left}}",
                done: paused.done,
                total: paused.total,
                left: paused.left,
              })}
            </p>
            <div className="overview-tab__actions">
              <button type="button" className="primary-button" disabled={story.busy || paused.left === 0} onClick={() => void story.resumeBatch?.()}>
                {t("board.pause.resume", { defaultValue: "Resume" })}
              </button>
              <button type="button" className="secondary-button" disabled={story.busy} onClick={() => void story.cancelBatch()}>
                {t("board.batch.cancel", { defaultValue: "Cancel batch" })}
              </button>
            </div>
          </>
        ) : (
          <>
            <h3 id="overview-next-title">{text.title}</h3>
            <p className="muted-text">{text.body}</p>
            {mode === "drafts" ? (
              <p className="muted-text">
                {t("overview.next.makeMissing.draftsHint", {
                  defaultValue: "The still model is slow. Make quick drafts first, mark the shots you like with Keep, then run the final pass.",
                })}
              </p>
            ) : null}
            {noModelReason ? <p className="muted-text" role="note">{noModelReason}</p> : null}
            <div className="overview-tab__actions">
              {step.id === "writeShots" ? (
                <button type="button" className="primary-button" onClick={() => onTab("plan")}>
                  {t("board.plan.overviewButton", { defaultValue: "Plan from an idea" })}
                </button>
              ) : null}
              {mode === "drafts" ? (
                <button
                  type="button"
                  className="primary-button"
                  disabled={story.busy || noModelReason !== null}
                  title={noModelReason ?? draftPassTitle(t as unknown as TFn, estimate?.passes?.draft)}
                  onClick={makeDrafts}
                >
                  {t("board.passes.draft", { defaultValue: "Make drafts" })}
                </button>
              ) : null}
              <button type="button" className={step.id === "writeShots" || mode === "drafts" ? "secondary-button" : "primary-button"} disabled={story.busy || noModelReason !== null} title={noModelReason ?? undefined} onClick={act}>
                {text.button}
              </button>
              {step.id === "writeShots" ? (
                <button type="button" className="secondary-button" onClick={() => onTab("cast")}>
                  {t("overview.addCast", { defaultValue: "Add a cast (optional)" })}
                </button>
              ) : null}
              {/* Once a clip exists the film can be watched at once, without a render (S9). */}
              {clips > 0 && step.id !== "writeShots" ? <RoughCutButton project={project} timeline={timeline} /> : null}
            </div>
          </>
        )}
        {!checklistComplete(rows) ? (
          <ol className="overview-checklist" aria-label={t("overview.checklist", { defaultValue: "Steps" })}>
            {rows.map((row) => (
              <li key={row.id} className={`overview-checklist__row${row.done ? " is-done" : ""}`}>
                <span className="overview-checklist__mark" aria-hidden="true">
                  {row.done ? (
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><polyline points="5 12 10 17 19 7" /></svg>
                  ) : null}
                </span>
                <button type="button" className="overview-checklist__link" onClick={() => onTab(row.tab)}>
                  {rowText[row.id]}
                </button>
                <span className="sr-only">
                  {row.done ? t("overview.check.doneLabel", { defaultValue: "done" }) : t("overview.check.todoLabel", { defaultValue: "to do" })}
                </span>
              </li>
            ))}
          </ol>
        ) : null}
      </section>

      {shots.length > 0 ? (
        <>
          <p className="overview-tab__progress">
            {t("overview.progressClips", {
              defaultValue: "{shots, plural, one {# shot} other {# shots}} · {clips} with a clip · {stills} still only · {stale} stale · {seconds} s",
              shots: shots.length,
              clips,
              stills,
              stale,
              seconds: filmSeconds(timeline),
            })}
            {" · "}
            {t("overview.cast", { defaultValue: "{n, plural, =0 {no cast} one {# in the cast} other {# in the cast}}", n: cast })}
            {" · "}
            {project.audio.length === 0
              ? t("overview.noSound", { defaultValue: "no sound yet" })
              : t("overview.sound", { defaultValue: "{n, plural, one {# sound clip} other {# sound clips}}", n: project.audio.length })}
          </p>
          <section className="overview-tab__sheet" aria-label={t("overview.sheet", { defaultValue: "Shots" })}>
            <ul className="contact-sheet" style={{ ["--sheet-ratio" as string]: project.defaults.exportPreset.replace(":", " / ") }}>
              {shots.map((shot, index) => {
                const thumb = thumbnailOf(shot);
                return (
                  <li key={shot.id}>
                    <button
                      type="button"
                      className={`contact-sheet__item contact-sheet__item--${shot.status}`}
                      onClick={() => onOpenShot(shot.id)}
                      title={shot.title || t("board.shot.untitled", { defaultValue: "Shot {n}", n: index + 1 })}
                    >
                      {thumb?.kind === "still" ? (
                        <ProjectImage projectId={project.id} rel={thumb.path} version={shot.activeTake} className="contact-sheet__thumb" thumb={256} />
                      ) : (
                        <span className="contact-sheet__thumb contact-sheet__thumb--empty" aria-hidden="true" />
                      )}
                      <span className="contact-sheet__number">{index + 1}</span>
                      {shot.status === "stale" ? (
                        <span className="badge warning contact-sheet__badge">{t("overview.staleBadge", { defaultValue: "Stale" })}</span>
                      ) : null}
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      ) : null}

      {activity.length > 0 || render?.preview || render?.exported ? (
        <section className="overview-tab__activity" aria-labelledby="overview-activity-title">
          <h3 id="overview-activity-title">{t("overview.activity.title", { defaultValue: "Recent activity" })}</h3>
          <ul>
            {render?.exported ? <li>{t("overview.activity.exported", { defaultValue: "Film exported in this session" })}</li> : null}
            {render?.preview ? <li>{t("overview.activity.preview", { defaultValue: "Preview rendered in this session" })}</li> : null}
            {activity.map((line) => (
              <li key={line.id} className={line.state === "done" && line.failed === 0 ? "" : "is-warn"}>{activityText(line)}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <ProjectDefaultsPanel
        project={project}
        busy={story.busy}
        op={story.op}
        imageModels={deps.imageModels}
        videoModels={deps.videoModels}
        videoFit={deps.videoFit}
        defaultOpen={shots.length === 0}
        openRequest={settingsRequest}
      />
    </div>
  );
}
