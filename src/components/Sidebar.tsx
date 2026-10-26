import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { SidebarGroupId, SidebarMode, TabId } from "../types";
import type { TabConfig } from "../constants";
import { sidebarGroups, tabs as allTabs } from "../constants";
import { ChevronIcon, groupIcon, standaloneTabIcon } from "./icons/SidebarIcons";

interface SidebarProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  platform?: string;
  appVersion?: string;
  backendOnline: boolean;
  engineLabel: string;
  loadedModelName?: string | null;
  mode: SidebarMode;
  collapsedGroups: Set<SidebarGroupId>;
  onToggleGroupCollapsed: (group: SidebarGroupId) => void;
  lastChildByGroup: Partial<Record<SidebarGroupId, string>>;
  onRememberLastChild: (group: SidebarGroupId, tabId: string) => void;
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

function resolveGroupTarget(
  groupId: SidebarGroupId,
  children: TabConfig[],
  lastChildByGroup: Partial<Record<SidebarGroupId, string>>,
): TabId | null {
  const remembered = lastChildByGroup[groupId];
  if (remembered) {
    const match = children.find((c) => c.id === remembered);
    if (match) return match.id;
  }
  const groupDef = sidebarGroups.find((g) => g.id === groupId);
  if (groupDef) {
    const defaultChild = children.find((c) => c.id === groupDef.defaultChild);
    if (defaultChild) return defaultChild.id;
  }
  return children[0]?.id ?? null;
}

export function Sidebar({
  activeTab,
  onTabChange,
  platform,
  appVersion,
  backendOnline,
  engineLabel,
  loadedModelName,
  mode,
  collapsedGroups,
  onToggleGroupCollapsed,
  lastChildByGroup,
  onRememberLastChild,
}: SidebarProps) {
  const { t } = useTranslation("common");
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
  // FU-042: resolve tab labels / captions through i18next at render
  // time with the English literal as ``defaultValue`` so the source
  // stays usable when a translation hasn't been authored yet.
  const tabLabel = (tab: TabConfig) => t(tab.labelKey, { defaultValue: tab.label });
  const tabCaption = (tab: TabConfig) => t(tab.captionKey, { defaultValue: tab.caption });
  const tabShortLabel = (tab: TabConfig) =>
    tab.shortLabelKey
      ? t(tab.shortLabelKey, { defaultValue: tab.shortLabel ?? tab.label })
      : tab.shortLabel ?? tab.label;
  const groupLabel = (id: SidebarGroupId, fallback: string) =>
    t(`tabGroups.${id}`, { defaultValue: fallback });

  function handleTabClick(tab: TabConfig) {
    if (tab.group) onRememberLastChild(tab.group, tab.id);
    onTabChange(tab.id);
  }

  function handleGroupTabsClick(groupId: SidebarGroupId, children: TabConfig[]) {
    const target = resolveGroupTarget(groupId, children, lastChildByGroup);
    if (target) {
      onRememberLastChild(groupId, target);
      onTabChange(target);
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-title-row">
          <img src="/logo.svg" alt={t("app.name", { defaultValue: "ChaosEngineAI" })} className="brand-logo" />
          <h1>{t("app.name", { defaultValue: "ChaosEngineAI" })}</h1>
        </div>
        <span className="brand-kicker">
          {t("app.tagline", { defaultValue: "Local AI model runner" })}
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
                  <strong>{tabLabel(item.tab)}</strong>
                  <span>{tabCaption(item.tab)}</span>
                </span>
              </button>
            );
          }

          const GroupIcon = groupIcon[item.id];
          const hasActiveChild = activeGroup === item.id;
          const localizedGroupLabel = groupLabel(item.id, item.label);

          // Tabs mode: group header is a single nav button that navigates to defaultChild / lastChild
          if (mode === "tabs") {
            return (
              <button
                key={item.id}
                type="button"
                className={hasActiveChild ? "nav-button active" : "nav-button"}
                onClick={() => handleGroupTabsClick(item.id, item.children)}
              >
                {GroupIcon ? <GroupIcon className="nav-icon" /> : null}
                <span className="nav-label">
                  <strong>{localizedGroupLabel}</strong>
                  <span>
                    {t("tabsCount", {
                      count: item.children.length,
                      defaultValue: "{{count}} tabs",
                    })}
                  </span>
                </span>
              </button>
            );
          }

          // Collapsible mode: group header toggles; children render inline
          const userCollapsed = collapsedGroups.has(item.id);
          const isOpen = hasActiveChild || !userCollapsed;

          return (
            <div key={item.id} className={`nav-group ${isOpen ? "open" : "closed"}`}>
              <button
                type="button"
                className="nav-group-header"
                aria-expanded={isOpen}
                aria-controls={`nav-group-${item.id}`}
                onClick={() => onToggleGroupCollapsed(item.id)}
              >
                {GroupIcon ? <GroupIcon className="nav-icon" /> : null}
                <span className="nav-group-label">{localizedGroupLabel}</span>
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
                        <span className="nav-child-label">{tabShortLabel(child)}</span>
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
          {backendOnline
            ? t("sidebar.backendOnline", { defaultValue: "Backend online" })
            : t("sidebar.backendOffline", { defaultValue: "Offline" })}
        </span>
        <p>{engineLabel}</p>
        <small>
          {loadedModelName ?? t("sidebar.noModelLoaded", { defaultValue: "No model loaded" })}
        </small>
      </div>
    </aside>
  );
}
