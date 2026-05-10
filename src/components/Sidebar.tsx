import { useMemo } from "react";
import type { SidebarGroupId, SidebarMode, TabId } from "../types";
import type { TabConfig } from "../constants";
import { sidebarGroups, tabs as allTabs } from "../constants";
import { TAB_I18N_KEY, GROUP_I18N_KEY } from "../constants/i18nMap";
import { ChevronIcon, groupIcon, standaloneTabIcon } from "./icons/SidebarIcons";
import { useI18n } from "../hooks/useI18n";

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
  const { t } = useI18n();
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
          <img src="/logo.svg" alt="ChaosEngineAI" className="brand-logo" />
          <h1>ChaosEngineAI</h1>
        </div>
        <span className="brand-kicker">
          {t("sidebar.brand.tagline")}
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
                  <strong>{t(TAB_I18N_KEY[item.tab.id] ?? item.tab.label)}</strong>
                  <span>{t(TAB_I18N_KEY[item.tab.id] + ".caption", item.tab.caption)}</span>
                </span>
              </button>
            );
          }

          const GroupIcon = groupIcon[item.id];
          const hasActiveChild = activeGroup === item.id;

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
                  <strong>{t(GROUP_I18N_KEY[item.id] ?? item.label)}</strong>
                  <span>{item.children.length} tabs</span>
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
                <span className="nav-group-label">{t(GROUP_I18N_KEY[item.id] ?? item.label)}</span>
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
                        <span className="nav-child-label">{t(TAB_I18N_KEY[child.id] + ".shortLabel", child.shortLabel ?? child.label)}</span>
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
          {t(backendOnline ? "sidebar.backend.online" : "sidebar.backend.offline")}
        </span>
        <p>{engineLabel}</p>
        <small>{t("sidebar.model.loaded", loadedModelName ?? "No model loaded")}</small>
      </div>
    </aside>
  );
}
