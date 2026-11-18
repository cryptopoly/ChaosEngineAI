import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { ActiveGeneration, SidebarGroupId, SidebarMode, TabId } from "../types";
import { busyEngineLines } from "../utils/busyEngine";
import type { TabConfig } from "../constants";
import { HOME_STUDIO_ENABLED, tabs as allTabs } from "../constants";
import {
  COMPACT_SIDEBAR_QUERY,
  arrangeRows,
  buildItems,
  resolveGroupTarget,
  trapFocusIndex,
  visibleSidebarTabs,
} from "./sidebarModel";
import { ChevronIcon, MenuIcon, groupIcon, standaloneTabIcon } from "./icons/SidebarIcons";

const FOCUSABLE = "button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex='-1'])";

/** True below 1180 px, where the sidebar is a top bar and a drawer (see styles.css). */
function useCompactLayout(): boolean {
  const query = typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(COMPACT_SIDEBAR_QUERY)
    : null;
  const [compact, setCompact] = useState(() => query?.matches ?? false);
  useEffect(() => {
    if (!query) return;
    const onChange = () => setCompact(query.matches);
    onChange();
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return compact;
}

interface SidebarProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  platform?: string;
  appVersion?: string;
  backendOnline: boolean;
  engineLabel: string;
  loadedModelName?: string | null;
  /** Image, video or music runs in flight; shown where "No model loaded" would mislead. */
  activeGenerations?: ActiveGeneration[];
  mode: SidebarMode;
  collapsedGroups: Set<SidebarGroupId>;
  onToggleGroupCollapsed: (group: SidebarGroupId) => void;
  lastChildByGroup: Partial<Record<SidebarGroupId, string>>;
  onRememberLastChild: (group: SidebarGroupId, tabId: string) => void;
  /** Off (default): only the active group is open. On: every group stays open unless collapsed. */
  keepAllOpen?: boolean;
  /** Groups whose "ready" child should be the click target (model installed, project open). */
  readyGroups?: ReadonlySet<SidebarGroupId>;
  /** Show the Music and Storyboard groups. Defaults to the build-time flag. */
  homeStudioEnabled?: boolean;
  /** Settings > General > "Show experimental tabs" (off by default). */
  showExperimentalTabs?: boolean;
}

export function Sidebar({
  activeTab,
  onTabChange,
  platform,
  appVersion,
  backendOnline,
  engineLabel,
  loadedModelName,
  activeGenerations,
  mode,
  collapsedGroups,
  onToggleGroupCollapsed,
  lastChildByGroup,
  onRememberLastChild,
  keepAllOpen = false,
  readyGroups,
  homeStudioEnabled = HOME_STUDIO_ENABLED,
  showExperimentalTabs = false,
}: SidebarProps) {
  const { t } = useTranslation("common");
  const compact = useCompactLayout();
  const [menuOpen, setMenuOpen] = useState(false);
  const drawerOpen = compact && menuOpen;
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const closeMenu = useCallback((returnFocus: boolean) => {
    setMenuOpen(false);
    if (returnFocus) menuButtonRef.current?.focus();
  }, []);
  useEffect(() => {
    if (!compact) setMenuOpen(false);
  }, [compact]);
  // The open drawer holds the keyboard: focus moves into it, Tab cycles between the menu button and its controls,
  // Escape closes it and gives focus back to the menu button.
  useEffect(() => {
    if (!drawerOpen) return;
    const panel = panelRef.current;
    const trap = () => [menuButtonRef.current, ...(panel ? panel.querySelectorAll<HTMLElement>(FOCUSABLE) : [])]
      .filter((el): el is HTMLElement => !!el);
    const current = panel?.querySelector<HTMLElement>("[aria-current='page']");
    (current ?? trap()[1])?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeMenu(true);
        return;
      }
      if (event.key !== "Tab") return;
      const items = trap();
      const next = trapFocusIndex(items.indexOf(document.activeElement as HTMLElement), items.length, event.shiftKey);
      if (next !== null) {
        event.preventDefault();
        items[next].focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawerOpen, closeMenu]);
  const busyLines = busyEngineLines(t, activeGenerations);
  const visibleTabs = useMemo(
    () => visibleSidebarTabs(allTabs, platform, homeStudioEnabled, showExperimentalTabs),
    [platform, homeStudioEnabled, showExperimentalTabs],
  );

  const rows = useMemo(() => arrangeRows(buildItems(visibleTabs)), [visibleTabs]);
  const activeGroup = visibleTabs.find((t) => t.id === activeTab)?.group;
  // Accordion: one group is open at a time. It follows the active tab, and a
  // header click moves it. Not persisted; the setting only picks the style.
  const [openGroup, setOpenGroup] = useState<SidebarGroupId | null>(activeGroup ?? null);
  useEffect(() => {
    if (activeGroup) setOpenGroup(activeGroup);
  }, [activeGroup]);
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
    if (drawerOpen) closeMenu(true);
  }

  function handleGroupTabsClick(groupId: SidebarGroupId, children: TabConfig[]) {
    const target = resolveGroupTarget(groupId, children, lastChildByGroup, readyGroups);
    if (target) {
      onRememberLastChild(groupId, target);
      onTabChange(target);
      if (drawerOpen) closeMenu(true);
    }
  }

  return (
    <aside className={drawerOpen ? "sidebar sidebar--open" : "sidebar"}>
      <button
        ref={menuButtonRef}
        type="button"
        className="sidebar-menu-button"
        aria-expanded={drawerOpen}
        aria-controls="sidebar-panel"
        aria-label={t("sidebar.menuButton", { defaultValue: "Menu" })}
        onClick={() => (drawerOpen ? closeMenu(true) : setMenuOpen(true))}
      >
        <MenuIcon open={drawerOpen} />
      </button>
      <div className="brand-block">
        <div className="brand-title-row">
          <img src="/logo.svg" alt="" className="brand-logo" />
          <h1>{t("app.name", { defaultValue: "ChaosEngineAI" })}</h1>
        </div>
        <span className="brand-kicker">
          {t("app.tagline", { defaultValue: "Local AI model runner" })}
          {appVersion ? ` · v${appVersion}` : ""}
        </span>
      </div>

      {drawerOpen ? <div className="sidebar-backdrop" aria-hidden onClick={() => closeMenu(true)} /> : null}
      <div className="sidebar-panel" id="sidebar-panel" ref={panelRef}>
        <nav className="nav-list" aria-label={t("sidebar.primaryNavAriaLabel", { defaultValue: "Primary" })}>
          {rows.map((item) => {
            if (item.kind === "section") {
              return (
                <div key={`section-${item.section}`} className="nav-section-label">
                  {item.section === "make"
                    ? t("sidebar.sectionMake", { defaultValue: "Make" })
                    : t("sidebar.sectionRun", { defaultValue: "Run" })}
                </div>
              );
            }
            if (item.kind === "tab") {
              const Icon = standaloneTabIcon[item.tab.id];
              const isActive = activeTab === item.tab.id;
              return (
                <button
                  key={item.tab.id}
                  className={isActive ? "nav-button active" : "nav-button"}
                  type="button"
                  aria-current={isActive ? "page" : undefined}
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
                  aria-current={hasActiveChild ? "page" : undefined}
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
            const isOpen = keepAllOpen
              ? hasActiveChild || !collapsedGroups.has(item.id)
              : openGroup === item.id;
            const toggleGroup = () => {
              if (keepAllOpen) onToggleGroupCollapsed(item.id);
              else setOpenGroup((current) => (current === item.id && !hasActiveChild ? null : item.id));
            };

            return (
              <div key={item.id} className={`nav-group ${isOpen ? "open" : "closed"}`}>
                <button
                  type="button"
                  className="nav-group-header"
                  aria-expanded={isOpen}
                  aria-controls={`nav-group-${item.id}`}
                  onClick={toggleGroup}
                >
                  {GroupIcon ? <GroupIcon className="nav-icon" /> : null}
                  <span className="nav-group-label">{localizedGroupLabel}</span>
                  <ChevronIcon open={isOpen} className="nav-group-chevron" />
                </button>
                {isOpen ? (
                  <div className="nav-group-children" id={`nav-group-${item.id}`} role="group" aria-label={localizedGroupLabel}>
                    {item.children.map((child) => {
                      const isActive = activeTab === child.id;
                      return (
                        <button
                          key={child.id}
                          className={isActive ? "nav-child-button active" : "nav-child-button"}
                          type="button"
                          aria-current={isActive ? "page" : undefined}
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
          {busyLines.length === 0 || loadedModelName ? (
            <small>{loadedModelName ?? t("sidebar.noModelLoaded", { defaultValue: "No model loaded" })}</small>
          ) : null}
          {busyLines.map((line) => (
            <small key={line} role="status">{line}</small>
          ))}
        </div>
      </div>
    </aside>
  );
}
