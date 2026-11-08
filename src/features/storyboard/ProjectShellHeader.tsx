/**
 * The project shell's header (D-65): a back link to the list, the name (click or Enter to rename), the format
 * chips, Undo and Redo, the "Changed elsewhere" chip, and the five tabs as a real tab list (arrow keys, Home,
 * End). Counts ride on the tabs: the Cast size and the stale shots.
 */
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { CloseIcon, RedoIcon, UndoIcon } from "../../components/icons/ActionIcons";
import type { StoryboardProject } from "../../types";
import { RESOLUTION_ALIAS } from "./formatLogic";
import { PROJECT_TABS, type ProjectTabId } from "./projectRoute";
import { nextTabIndex } from "./projectShellLogic";
import { staleCount } from "./storyboardLogic";

export interface ProjectShellHeaderProps {
  name: string;
  project: StoryboardProject | null;
  tab: ProjectTabId;
  busy: boolean;
  conflict: boolean;
  onDismissConflict: () => void;
  onBack: () => void;
  onRename: (name: string) => void;
  onUndo: () => void;
  onRedo: () => void;
  onTab: (tab: ProjectTabId) => void;
  /** The format chips open Overview with Project settings expanded. */
  onOpenSettings?: () => void;
}

function ChevronLeft() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  );
}

function ProjectName({ name, disabled, onRename }: { name: string; disabled: boolean; onRename: (name: string) => void }) {
  const { t } = useTranslation("studio");
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(name);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!editing) setValue(name);
  }, [name, editing]);
  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  function finish(save: boolean) {
    setEditing(false);
    const trimmed = value.trim();
    if (save && trimmed && trimmed !== name) onRename(trimmed);
    else setValue(name);
  }

  if (editing) {
    return (
      <input
        ref={input}
        className="text-input project-shell__name-input"
        value={value}
        maxLength={200}
        aria-label={t("projectShell.nameLabel", { defaultValue: "Project name" })}
        onChange={(event) => setValue(event.target.value)}
        onBlur={() => finish(true)}
        onKeyDown={(event) => {
          if (event.key === "Enter") finish(true);
          if (event.key === "Escape") {
            event.stopPropagation();
            finish(false);
          }
        }}
      />
    );
  }
  return (
    <button
      type="button"
      className="project-shell__name"
      disabled={disabled}
      title={t("projectShell.rename", { defaultValue: "Rename" })}
      onClick={() => setEditing(true)}
    >
      <h2>{name}</h2>
    </button>
  );
}

export function ProjectShellHeader({
  name, project, tab, busy, conflict, onDismissConflict, onBack, onRename, onUndo, onRedo, onTab, onOpenSettings,
}: ProjectShellHeaderProps) {
  const { t } = useTranslation("studio");
  const tabRow = useRef<HTMLDivElement>(null);
  const labels: Record<ProjectTabId, string> = {
    overview: t("projectShell.tab.overview", { defaultValue: "Overview" }),
    cast: t("projectShell.tab.cast", { defaultValue: "Cast" }),
    storyboard: t("projectShell.tab.storyboard", { defaultValue: "Storyboard" }),
    mixer: t("projectShell.tab.mixer", { defaultValue: "Mixer" }),
    export: t("projectShell.tab.export", { defaultValue: "Export" }),
  };
  const castCount = project ? project.references.characters.length + project.references.places.length : 0;
  const stale = project ? staleCount(project) : 0;

  // Keep the active tab in view when the row scrolls (narrow screens).
  useEffect(() => {
    // Only the tab row scrolls: scrollIntoView would also pan the shell and the page.
    const row = tabRow.current;
    const active = row?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (!row || !active) return;
    if (active.offsetLeft < row.scrollLeft) row.scrollLeft = active.offsetLeft;
    else if (active.offsetLeft + active.offsetWidth > row.scrollLeft + row.clientWidth) row.scrollLeft = active.offsetLeft + active.offsetWidth - row.clientWidth;
  }, [tab]);

  function onTabKey(event: KeyboardEvent<HTMLDivElement>) {
    const index = PROJECT_TABS.indexOf(tab);
    const next = nextTabIndex(index, PROJECT_TABS.length, event.key);
    if (next === null) return;
    event.preventDefault();
    onTab(PROJECT_TABS[next]);
    requestAnimationFrame(() => tabRow.current?.querySelector<HTMLElement>(`#project-tab-${PROJECT_TABS[next]}`)?.focus({ preventScroll: true }));
  }

  const d = project?.defaults;
  return (
    <header className="project-shell__header">
      <div className="project-shell__top">
        <button type="button" className="project-shell__back" onClick={onBack}>
          <ChevronLeft />
          {t("projectShell.back", { defaultValue: "Projects" })}
        </button>
        <ProjectName name={name} disabled={busy || !project} onRename={onRename} />
        {d ? (
          <button
            type="button"
            className="project-shell__chips"
            title={t("projectShell.formatTitle", { defaultValue: "Project settings" })}
            onClick={() => (onOpenSettings ? onOpenSettings() : onTab("overview"))}
          >
            <span className="project-shell__chip">{d.exportPreset}</span>
            <span className="project-shell__chip">{RESOLUTION_ALIAS[d.resolution] ?? d.resolution}</span>
            <span className="project-shell__chip">{d.fps} fps</span>
          </button>
        ) : null}
        <div className="project-shell__actions">
          {conflict ? (
            <span className="badge warning conflict-chip" role="status">
              {t("board.conflictNotice", { defaultValue: "Changed elsewhere; showing the latest" })}
              <button
                type="button"
                className="conflict-chip__dismiss"
                aria-label={t("storyboard.dismiss", { defaultValue: "Dismiss" })}
                onClick={onDismissConflict}
              >
                <CloseIcon aria-hidden="true" />
              </button>
            </span>
          ) : null}
          <button type="button" className="secondary-button" disabled={busy || !project || project.undoStack.length === 0} onClick={onUndo}>
            <UndoIcon /> {t("board.undo", { defaultValue: "Undo" })}
          </button>
          <button type="button" className="secondary-button" disabled={busy || !project || project.redoStack.length === 0} onClick={onRedo}>
            <RedoIcon /> {t("board.redo", { defaultValue: "Redo" })}
          </button>
        </div>
      </div>
      <div
        className="project-shell__tabs"
        role="tablist"
        aria-label={t("projectShell.tabsLabel", { defaultValue: "Project" })}
        ref={tabRow}
        onKeyDown={onTabKey}
      >
        {PROJECT_TABS.map((id) => {
          const selected = id === tab;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              id={`project-tab-${id}`}
              aria-selected={selected}
              aria-controls={`project-panel-${id}`}
              tabIndex={selected ? 0 : -1}
              className={`project-shell__tab${selected ? " active" : ""}`}
              onClick={() => onTab(id)}
            >
              {labels[id]}
              {id === "cast" && castCount > 0 ? <span className="project-shell__count">{castCount}</span> : null}
              {id === "storyboard" && stale > 0 ? (
                <span className="project-shell__count project-shell__count--warning">
                  {t("storyProjects.card.stale", { defaultValue: "{n} stale", n: stale })}
                </span>
              ) : null}
            </button>
          );
        })}
      </div>
    </header>
  );
}
