import { useTranslation } from "react-i18next";
import { tabs } from "../constants";
import type { TabId, WorkspaceData } from "../types";

/** Workspace title, caption and platform / cache badges. */
export function WorkspaceHeader({
  activeTab,
  system,
  cacheLabel,
}: {
  activeTab: TabId;
  system: WorkspaceData["system"];
  cacheLabel: string;
}) {
  const { t: tCommon } = useTranslation("common");
  return (
    <header className="workspace-header">
      <div>
        <span className="eyebrow">{tCommon("app.workspace", { defaultValue: "Workspace" })}</span>
        <h2>
          {(() => {
            const tab = tabs.find((entry) => entry.id === activeTab);
            if (!tab) return null;
            return tCommon(tab.labelKey, { defaultValue: tab.label });
          })()}
        </h2>
        <p>
          {(() => {
            const tab = tabs.find((entry) => entry.id === activeTab);
            if (!tab) return null;
            return tCommon(tab.captionKey, { defaultValue: tab.caption });
          })()}
        </p>
      </div>
      <div className="header-badges">
        <span className="badge muted">{system.platform}</span>
        <span className="badge muted">{system.arch}</span>
        <span className="badge accent">{cacheLabel}</span>
      </div>
    </header>
  );
}
