import { useTranslation } from "react-i18next";
import { Panel } from "../components/Panel";
import type { AppTabContext } from "./context";

/**
 * Placeholder for the Storyboard tabs. Only reachable when
 * HOME_STUDIO_ENABLED is true; the real screens replace these later.
 */
function StoryboardPlaceholder({ titleKey, title }: { titleKey: string; title: string }) {
  const { t } = useTranslation("common");
  return (
    <div className="content-grid">
      <Panel title={t(titleKey, { defaultValue: title })} className="span-2">
        <div className="empty-state">
          <p className="muted-text">{t("comingSoon.storyboard", { defaultValue: "Storyboard is coming soon." })}</p>
        </div>
      </Panel>
    </div>
  );
}

export function StoryProjectsView(_props: { ctx: AppTabContext }) {
  return <StoryboardPlaceholder titleKey="tabs.storyProjects" title="Projects" />;
}

export function StoryBoardView(_props: { ctx: AppTabContext }) {
  return <StoryboardPlaceholder titleKey="tabs.storyBoard" title="Board" />;
}

export function StoryMixerView(_props: { ctx: AppTabContext }) {
  return <StoryboardPlaceholder titleKey="tabs.storyMixer" title="Mixer" />;
}
