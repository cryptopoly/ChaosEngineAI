import { useTranslation } from "react-i18next";
import type { SidebarGroupId, TabId } from "../types";
import type { TabConfig } from "../constants";
import { sidebarGroups, tabs as allTabs } from "../constants";

interface SubtabBarProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  platform?: string;
  onRememberLastChild: (group: SidebarGroupId, tabId: string) => void;
}

export function SubtabBar({ activeTab, onTabChange, platform, onRememberLastChild }: SubtabBarProps) {
  const { t } = useTranslation("common");
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

  // FU-042: localised group + per-tab labels via translation keys with
  // English literal fallback.
  const localizedGroupLabel = t(`tabGroups.${groupDef.id}`, { defaultValue: groupDef.label });
  const childLabel = (child: TabConfig) =>
    child.shortLabelKey
      ? t(child.shortLabelKey, { defaultValue: child.shortLabel ?? child.label })
      : child.shortLabel ?? child.label;

  return (
    <div
      className="subtab-bar"
      role="tablist"
      aria-label={t("sidebar.groupTabsAriaLabel", {
        group: localizedGroupLabel,
        defaultValue: "{{group}} tabs",
      })}
    >
      {children.map((child) => {
        const isActive = activeTab === child.id;
        return (
          <button
            key={child.id}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={isActive ? "subtab active" : "subtab"}
            onClick={() => handleClick(child)}
          >
            {childLabel(child)}
          </button>
        );
      })}
    </div>
  );
}
