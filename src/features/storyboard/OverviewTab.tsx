/**
 * The Overview tab of a project (D-65): the shots at a glance, recent activity and the project settings (open
 * while the project is new, collapsed once it has shots). The next step and the order of the work are the stage
 * rail's (audit change 1, `guide/StageRail.tsx`), which sits above every tab, so this tab no longer keeps its own
 * next-step card or checklist.
 */
import { useTranslation } from "react-i18next";
import type { StoryboardProject, StoryboardQuality, StoryboardTimeline } from "../../types";
import { clipCount, recentActivity, stillOnlyCount, type ActivityLine } from "./overviewLogic";
import { ProjectDefaultsPanel } from "./ProjectDefaultsPanel";
import { ProjectImage } from "./ProjectMedia";
import { RoughCutButton } from "./preview/RoughCutButton";
import type { ProjectWorkspaceDeps } from "./ProjectShell";
import type { ProjectTabId } from "./projectRoute";
import { filmSeconds, relativeTime, staleCount, thumbnailOf } from "./storyboardLogic";
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
  /** `guide.quality`, for the Draft tag in the film player. */
  quality?: StoryboardQuality | null;
}

export function OverviewTab({ story, project, timeline, deps, render, onOpenShot, settingsRequest, quality }: OverviewTabProps) {
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

  return (
    <div className="overview-tab">
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
          {/* Once a clip exists the film can be watched at once, without a render (S9). */}
          {clips > 0 ? (
            <div className="overview-tab__actions">
              <RoughCutButton project={project} timeline={timeline} quality={quality} />
            </div>
          ) : null}
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
