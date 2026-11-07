import { BoardTab } from "../features/storyboard/BoardTab";
import { MixerTab } from "../features/storyboard/mixer/MixerTab";
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

export function StoryMixerView({ ctx }: { ctx: AppTabContext }) {
  const story = useStoryboard(ctx.backendOnline);
  return (
    <MixerTab
      backendOnline={ctx.backendOnline}
      story={story}
      dataDirectory={ctx.settings.settingsDraft.dataDirectory}
      onOpenProjects={() => void ctx.setActiveTab("story-projects")}
      onOpenBoard={() => void ctx.setActiveTab("story-board")}
    />
  );
}
