import type { TabId } from "../types";
import { tabs } from "../constants";

interface SidebarProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  platform?: string;
  appVersion?: string;
  backendOnline: boolean;
  engineLabel: string;
  loadedModelName?: string | null;
}

export function Sidebar({
  activeTab,
  onTabChange,
  platform,
  appVersion,
  backendOnline,
  engineLabel,
  loadedModelName,
}: SidebarProps) {
  const visibleTabs = tabs.filter((tab) => {
    if (tab.id === "conversion" && platform && platform !== "Darwin") return false;
    return true;
  });

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-title-row">
          <img src="/logo.svg" alt="ChaosEngineAI" className="brand-logo" />
          <h1>ChaosEngineAI</h1>
        </div>
        <span className="brand-kicker">
          Local AI model runner
          {appVersion ? ` · v${appVersion}` : ""}
        </span>
      </div>

      <nav className="nav-list" aria-label="Primary">
        {visibleTabs.map((tab) => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? "nav-button active" : "nav-button"}
            type="button"
            onClick={() => onTabChange(tab.id)}
          >
            <strong>{tab.label}</strong>
            <span>{tab.caption}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className={`badge ${backendOnline ? "success" : "warning"}`}>
          {backendOnline ? "Backend online" : "Offline"}
        </span>
        <p>{engineLabel}</p>
        <small>{loadedModelName ?? "No model loaded"}</small>
      </div>
    </aside>
  );
}
