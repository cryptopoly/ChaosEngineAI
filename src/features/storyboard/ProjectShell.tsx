/**
 * One open project (D-65): the header with the back link, the name, Undo / Redo and the conflict chip, then
 * five tabs in the order of the work: Overview, Cast, Storyboard, Mixer, Export. The shell owns the project
 * state, so every tab works on this project only, and handles the offline, loading, not-found and error
 * states once for all of them.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { ImageOutputArtifact, VideoOutputArtifact } from "../../types";
import { CastTab } from "./CastTab";
import { ExportTab } from "./ExportTab";
import type { VideoFitSource } from "./formatLogic";
import { MixerTab } from "./mixer/MixerTab";
import { OverviewTab } from "./OverviewTab";
import type { ModelChoice } from "./ProjectDefaultsPanel";
import { setProjectName, type ProjectTabId } from "./projectRoute";
import { ProjectShellHeader } from "./ProjectShellHeader";
import { StoryboardTab } from "./StoryboardTab";
import { useStoryboard } from "./useStoryboard";
import { useStoryboardRender } from "./useStoryboardRender";

/** What the tabs need from the rest of the app. */
export interface ProjectWorkspaceDeps {
  imageGallery: ImageOutputArtifact[];
  videoGallery: VideoOutputArtifact[];
  imageModels: ModelChoice[];
  videoModels: ModelChoice[];
  videoFit?: VideoFitSource;
  /** The app's data directory, to build a path for "Copy path". */
  dataDirectory?: string;
}

export interface ProjectShellProps {
  projectId: string;
  /** The name from the list, shown while the project loads. */
  fallbackName?: string;
  tab: ProjectTabId;
  backendOnline: boolean;
  deps: ProjectWorkspaceDeps;
  onBack: () => void;
  onTab: (tab: ProjectTabId) => void;
}

export type ShellState = "offline" | "notFound" | "error" | "loading" | "ready";

/** What the shell shows instead of (or above) the tabs: handled once here for all five tabs. */
export function shellState(input: { backendOnline: boolean; notFound: boolean; loaded: boolean; error: string | null }): ShellState {
  if (!input.backendOnline) return "offline";
  if (input.notFound) return "notFound";
  if (input.loaded) return "ready";
  return input.error ? "error" : "loading";
}

export function ProjectShell({ projectId, fallbackName, tab, backendOnline, deps, onBack, onTab }: ProjectShellProps) {
  const { t } = useTranslation("studio");
  const story = useStoryboard(backendOnline, projectId);
  const { project, timeline } = story;
  const [focusShotId, setFocusShotId] = useState<string | null>(null);
  // Bumped by the header's format chips: Overview opens Project settings and focuses it.
  const [settingsRequest, setSettingsRequest] = useState(0);
  // One render tracker for the Mixer and Export tabs: the last export, a running render and its clock survive tab switches.
  const render = useStoryboardRender(projectId, backendOnline, () => void story.reload());
  const loadedName = project?.name ?? null;
  useEffect(() => {
    if (loadedName) setProjectName(projectId, loadedName);
  }, [projectId, loadedName]);

  const openShot = (shotId: string | null) => {
    setFocusShotId(shotId);
    onTab("storyboard");
  };

  const state = shellState({ backendOnline, notFound: story.notFound, loaded: Boolean(project && timeline), error: story.error });
  let body: React.ReactNode;
  if (state === "offline") {
    body = <p className="muted-text">{t("storyboard.offline", { defaultValue: "Waiting for the backend…" })}</p>;
  } else if (state === "notFound") {
    body = (
      <div className="empty-state">
        <p className="muted-text">{t("projectShell.notFound", { defaultValue: "This project is not there any more." })}</p>
        <button type="button" className="primary-button" onClick={onBack}>
          {t("projectShell.backToList", { defaultValue: "Back to projects" })}
        </button>
      </div>
    );
  } else if (!project || !timeline) {
    body = state === "error" ? (
      <div className="callout error" role="alert">
        <p>{story.error}</p>
        <button type="button" className="secondary-button" onClick={() => void story.reload()}>
          {t("board.reload", { defaultValue: "Reload" })}
        </button>
      </div>
    ) : (
      <p className="muted-text" role="status">{t("board.loading", { defaultValue: "Loading the project…" })}</p>
    );
  } else {
    body = (
      <>
        {story.error ? (
          <div className="callout error" role="alert">
            <p>{story.error}</p>
          </div>
        ) : null}
        {tab === "overview" ? <OverviewTab story={story} project={project} timeline={timeline} deps={deps} render={render} onTab={onTab} onOpenShot={openShot} settingsRequest={settingsRequest} /> : null}
        {tab === "cast" ? <CastTab story={story} project={project} imageGallery={deps.imageGallery} /> : null}
        {tab === "storyboard" ? (
          <StoryboardTab
            story={story}
            project={project}
            timeline={timeline}
            imageGallery={deps.imageGallery}
            videoGallery={deps.videoGallery}
            focusShotId={focusShotId}
            onFocused={() => setFocusShotId(null)}
          />
        ) : null}
        {tab === "mixer" ? (
          <MixerTab story={story} project={project} timeline={timeline} dataDirectory={deps.dataDirectory} onOpenShot={openShot} render={render} />
        ) : null}
        {tab === "export" ? <ExportTab story={story} project={project} timeline={timeline} dataDirectory={deps.dataDirectory} onTab={onTab} render={render} /> : null}
      </>
    );
  }

  return (
    <div className="content-grid storyboard-page">
      <section className="panel span-2 project-shell">
        <ProjectShellHeader
          name={project?.name ?? fallbackName ?? ""}
          project={project}
          tab={tab}
          busy={story.busy}
          conflict={story.notice === "conflict"}
          onDismissConflict={story.dismissNotice}
          onBack={onBack}
          onRename={(name) => void story.op("edit_project", { name })}
          onUndo={() => void story.undo()}
          onRedo={() => void story.redo()}
          onTab={onTab}
          onOpenSettings={() => {
            setSettingsRequest((n) => n + 1);
            onTab("overview");
          }}
        />
        <div className="project-shell__panel" role="tabpanel" id={`project-panel-${tab}`} aria-labelledby={`project-tab-${tab}`}>
          {body}
        </div>
      </section>
    </div>
  );
}
