import { useMemo } from "react";
import type { SidebarGroupId, TabId } from "../types";
import type { TabConfig } from "../constants";
import { sidebarGroups, tabs as allTabs } from "../constants";
import { useSidebarPrefs } from "../hooks";
import { ChevronIcon, groupIcon, standaloneTabIcon } from "./icons/SidebarIcons";

interface SidebarProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  platform?: string;
  appVersion?: string;
  backendOnline: boolean;
  engineLabel: string;
  loadedModelName?: string | null;
}

type SidebarItem =
  | { kind: "tab"; tab: TabConfig }
  | { kind: "group"; id: SidebarGroupId; label: string; children: TabConfig[] };

function buildItems(visibleTabs: TabConfig[]): SidebarItem[] {
  const items: SidebarItem[] = [];
  const groupsSeen = new Set<SidebarGroupId>();

  for (const tab of visibleTabs) {
    if (!tab.group) {
      items.push({ kind: "tab", tab });
      continue;
    }
    if (groupsSeen.has(tab.group)) continue;
    groupsSeen.add(tab.group);

    const groupDef = sidebarGroups.find((g) => g.id === tab.group);
    if (!groupDef) {
      items.push({ kind: "tab", tab });
      continue;
    }
    const children = visibleTabs.filter((t) => t.group === tab.group);
    items.push({ kind: "group", id: groupDef.id, label: groupDef.label, children });
  }

  return items;
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
  const { collapsedGroups, toggleGroupCollapsed, rememberLastChild } = useSidebarPrefs();

  const visibleTabs = useMemo(
    () =>
      allTabs.filter((tab) => {
        if (tab.id === "conversion" && platform && platform !== "Darwin") return false;
        return true;
      }),
    [platform],
  );

  const items = useMemo(() => buildItems(visibleTabs), [visibleTabs]);
  const activeGroup = visibleTabs.find((t) => t.id === activeTab)?.group;

  function handleTabClick(tab: TabConfig) {
    if (tab.group) rememberLastChild(tab.group, tab.id);
    onTabChange(tab.id);
  }

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
        {items.map((item) => {
          if (item.kind === "tab") {
            const Icon = standaloneTabIcon[item.tab.id];
            const isActive = activeTab === item.tab.id;
            return (
              <button
                key={item.tab.id}
                className={isActive ? "nav-button active" : "nav-button"}
                type="button"
                onClick={() => handleTabClick(item.tab)}
              >
                {Icon ? <Icon className="nav-icon" /> : null}
                <span className="nav-label">
                  <strong>{item.tab.label}</strong>
                  <span>{item.tab.caption}</span>
                </span>
              </button>
            );
          }

          const GroupIcon = groupIcon[item.id];
          const userCollapsed = collapsedGroups.has(item.id);
          const hasActiveChild = activeGroup === item.id;
          const isOpen = hasActiveChild || !userCollapsed;

          return (
            <div key={item.id} className={`nav-group ${isOpen ? "open" : "closed"}`}>
              <button
                type="button"
                className="nav-group-header"
                aria-expanded={isOpen}
                aria-controls={`nav-group-${item.id}`}
                onClick={() => toggleGroupCollapsed(item.id)}
              >
                {GroupIcon ? <GroupIcon className="nav-icon" /> : null}
                <span className="nav-group-label">{item.label}</span>
                <ChevronIcon open={isOpen} className="nav-group-chevron" />
              </button>
              {isOpen ? (
                <div className="nav-group-children" id={`nav-group-${item.id}`} role="group">
                  {item.children.map((child) => {
                    const isActive = activeTab === child.id;
                    return (
                      <button
                        key={child.id}
                        className={isActive ? "nav-child-button active" : "nav-child-button"}
                        type="button"
                        onClick={() => handleTabClick(child)}
                      >
                        <span className="nav-child-dot" aria-hidden />
                        <span className="nav-child-label">{child.shortLabel ?? child.label}</span>
                      </button>
                    );
                  })}
                </div>
              ) : null}
            </div>
          );
        })}
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
