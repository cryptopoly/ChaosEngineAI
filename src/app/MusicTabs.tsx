import { MusicLibraryTab } from "../features/music/MusicLibraryTab";
import { MusicModelsPage } from "../features/music/MusicModelsPage";
import { MusicStudioTab } from "../features/music/MusicStudioTab";
import type { AppTabContext } from "./context";

/** The three Music screens. Only reachable when HOME_STUDIO_ENABLED is true. */
export function MusicModelsView({ ctx }: { ctx: AppTabContext }) {
  const { musicState, backendOnline, modelsAvailability, setAreaAvailability, handleOpenExternalUrl } = ctx;
  return (
    <MusicModelsPage
      music={musicState}
      backendOnline={backendOnline}
      availability={modelsAvailability.music}
      onAvailabilityChange={(value) => setAreaAvailability("music", value)}
      onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
    />
  );
}

export function MusicStudioView({ ctx }: { ctx: AppTabContext }) {
  return <MusicStudioTab music={ctx.musicState} backendOnline={ctx.backendOnline} onTabChange={ctx.setActiveTab} />;
}

export function MusicLibraryView({ ctx }: { ctx: AppTabContext }) {
  return <MusicLibraryTab music={ctx.musicState} backendOnline={ctx.backendOnline} onTabChange={ctx.setActiveTab} />;
}
