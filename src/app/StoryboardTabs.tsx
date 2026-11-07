import { useTranslation } from "react-i18next";
import { Panel } from "../components/Panel";
import { BoardTab } from "../features/storyboard/BoardTab";
import { ProjectsTab } from "../features/storyboard/ProjectsTab";
import { writeOpenProjectId } from "../features/storyboard/storyboardLogic";
import { useStoryboard } from "../features/storyboard/useStoryboard";
import type { AppTabContext } from "./context";

/**
 * Storyboard tabs. Only reachable when HOME_STUDIO_ENABLED is true. The open project id is remembered
 * in localStorage: Projects writes it and moves to the Board, which reads it on mount.
 */
export function StoryProjectsView({ ctx }: { ctx: AppTabContext }) {
  return (
    <ProjectsTab
      backendOnline={ctx.backendOnline}
      onOpenProject={(id) => {
        writeOpenProjectId(id);
        void ctx.setActiveTab("story-board");
      }}
    />
  );
}

export function StoryBoardView({ ctx }: { ctx: AppTabContext }) {
  const story = useStoryboard(ctx.backendOnline);
  const choices = (variants: { id: string; name: string }[]) =>
    variants.map((variant) => ({ id: variant.id, name: variant.name }));
  return (
    <BoardTab
      backendOnline={ctx.backendOnline}
      story={story}
      imageGallery={ctx.imgState.imageOutputs}
      videoGallery={ctx.videoState.videoOutputs}
      imageModels={choices(ctx.imgState.installedImageVariants)}
      videoModels={choices(ctx.videoState.installedVideoVariants)}
      onOpenProjects={() => void ctx.setActiveTab("story-projects")}
      onOpenMixer={() => void ctx.setActiveTab("story-mixer")}
    />
  );
}

/** The Mixer is built separately; until then this tab says what is coming. */
export function StoryMixerView(_props: { ctx: AppTabContext }) {
  const { t } = useTranslation("common");
  return (
    <div className="content-grid">
      <Panel title={t("tabs.storyMixer", { defaultValue: "Mixer" })} className="span-2">
        <div className="empty-state">
          <p className="muted-text">{t("comingSoon.storyboard", { defaultValue: "Storyboard is coming soon." })}</p>
        </div>
      </Panel>
    </div>
  );
}
