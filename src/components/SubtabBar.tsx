import type { SidebarGroupId, TabId } from "../types";
import type { TabConfig } from "../constants";
import { sidebarGroups, tabs as allTabs } from "../constants";
import { GROUP_I18N_KEY, TAB_I18N_KEY } from "../constants/i18nMap";
import { useI18n } from "../hooks/useI18n";

interface SubtabBarProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  platform?: string;
  onRememberLastChild: (group: SidebarGroupId, tabId: string) => void;
}

export function SubtabBar({ activeTab, onTabChange, platform, onRememberLastChild }: SubtabBarProps) {
  const { t, ti } = useI18n();
  const activeTabConfig = allTabs.find((t) => t.id === activeTab);
  const groupId = activeTabConfig?.group;
  if (!groupId) return null;

  const groupDef = sidebarGroups.find((g) => g.id === groupId);
  if (!groupDef) return null;

  const children: TabConfig[] = allTabs.filter((t) => {
    if (t.group !== groupId) return false;
    if (t.id === "conversion" && platform && platform !== "Darwin") return false;
    return true;
  });

  if (children.length <= 1) return null;

  function handleClick(child: TabConfig) {
    if (!groupId) return;
    onRememberLastChild(groupId, child.id);
    onTabChange(child.id);
  }

  return (
    <div
      className="subtab-bar"
      role="tablist"
      aria-label={ti("subtabBar.ariaLabel", "{{group}} 子标签", {
        group: t(GROUP_I18N_KEY[groupDef.id] ?? "", groupDef.label),
      })}
    >
      {children.map((child) => {
        const isActive = activeTab === child.id;
        const tabKey = TAB_I18N_KEY[child.id];
        const label = t(tabKey ? `${tabKey}.shortLabel` : "", child.shortLabel ?? child.label);
        return (
          <button
            key={child.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={isActive ? "subtab active" : "subtab"}
            onClick={() => handleClick(child)}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
