import { useTranslation } from "react-i18next";
import { Panel } from "../components/Panel";
import type { AppTabContext } from "./context";

/**
 * Placeholder for the Music tabs. Only reachable when HOME_STUDIO_ENABLED is
 * true; the real screens replace these when the Music backend lands.
 */
function MusicPlaceholder({ titleKey, title }: { titleKey: string; title: string }) {
  const { t } = useTranslation("common");
  return (
    <div className="content-grid">
      <Panel title={t(titleKey, { defaultValue: title })} className="span-2">
        <div className="empty-state">
          <p className="muted-text">{t("comingSoon.music", { defaultValue: "Music is coming soon." })}</p>
        </div>
      </Panel>
    </div>
  );
}

export function MusicModelsView(_props: { ctx: AppTabContext }) {
  return <MusicPlaceholder titleKey="tabs.musicModels" title="Music Models" />;
}

export function MusicStudioView(_props: { ctx: AppTabContext }) {
  return <MusicPlaceholder titleKey="tabs.musicStudio" title="Music Studio" />;
}

export function MusicLibraryView(_props: { ctx: AppTabContext }) {
  return <MusicPlaceholder titleKey="tabs.musicLibrary" title="Music Library" />;
}
