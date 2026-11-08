import { useMemo } from "react";
import type { VideoFitSource } from "../features/storyboard/formatLogic";
import { ProjectListTab } from "../features/storyboard/ProjectListTab";
import { ProjectShell, type ProjectWorkspaceDeps } from "../features/storyboard/ProjectShell";
import { useProjectRoute } from "../features/storyboard/projectRoute";
import type { AppTabContext } from "./context";

/**
 * The Projects entry (D-65): the project list, or one project in its shell. Which one shows comes from the
 * stored route, so the entry reopens where the user left off; galleries and old tab ids write the route
 * before they move here. Only reachable when HOME_STUDIO_ENABLED is true.
 */
function videoFitSource(ctx: AppTabContext): VideoFitSource {
  const status = ctx.videoState.videoRuntimeStatus;
  return {
    variants: ctx.videoState.installedVideoVariants,
    device: status.device ?? null,
    deviceMemoryGb: status.deviceMemoryGb ?? null,
    residentGb: ctx.workspace.runtime.residentModelGb ?? null,
  };
}

const choices = (variants: { id: string; name: string }[]) => variants.map((variant) => ({ id: variant.id, name: variant.name }));

export function StoryboardView({ ctx }: { ctx: AppTabContext }) {
  const [route, go] = useProjectRoute();
  const imageVariants = ctx.imgState.installedImageVariants;
  const videoVariants = ctx.videoState.installedVideoVariants;
  const imageModels = useMemo(() => choices(imageVariants), [imageVariants]);
  const videoModels = useMemo(() => choices(videoVariants), [videoVariants]);
  const videoFit = videoFitSource(ctx);

  if (!route.projectId) {
    return (
      <ProjectListTab
        backendOnline={ctx.backendOnline}
        videoFit={videoFit}
        imageModels={imageModels}
        videoModels={videoModels}
        onOpenProject={(id, created) => go(id, created ? "overview" : undefined)}
      />
    );
  }

  const deps: ProjectWorkspaceDeps = {
    imageGallery: ctx.imgState.imageOutputs,
    videoGallery: ctx.videoState.videoOutputs,
    imageModels,
    videoModels,
    videoFit,
    dataDirectory: ctx.settings.settingsDraft.dataDirectory,
  };
  return (
    <ProjectShell
      key={route.projectId}
      projectId={route.projectId}
      tab={route.tab}
      backendOnline={ctx.backendOnline}
      deps={deps}
      onBack={() => go(null)}
      onTab={(tab) => go(route.projectId, tab)}
    />
  );
}
