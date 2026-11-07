import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import {
  ALL_PROJECTS,
  NO_PROJECT,
  SEARCH_FROM,
  searchProjects,
  type ProjectOption,
} from "../utils/galleryProject";
import { ChevronDownIcon, SearchIcon } from "./icons/ActionIcons";

export interface ProjectRow {
  id: string;
  label: string;
  count: number;
}

/** The rows of the open list: All projects, No project (when some outputs have none), then the projects matching the search. */
export function projectRows(options: ProjectOption[], query: string, labels: { all: string; none: string }, total: number, untagged: number): ProjectRow[] {
  const searching = query.trim() !== "";
  return [
    ...(searching ? [] : [{ id: ALL_PROJECTS, label: labels.all, count: total }]),
    ...(searching || untagged === 0 ? [] : [{ id: NO_PROJECT, label: labels.none, count: untagged }]),
    ...searchProjects(options, query).map((option) => ({ id: option.id, label: option.name, count: option.count })),
  ];
}

/** The open list: an optional search box and the option rows. Presentational; the dropdown owns focus and keys. */
export function ProjectList({
  id,
  rows,
  value,
  searchable,
  query,
  onQuery,
  onPick,
}: {
  id: string;
  rows: ProjectRow[];
  value: string;
  searchable: boolean;
  query: string;
  onQuery: (query: string) => void;
  onPick: (id: string) => void;
}) {
  const { t } = useTranslation("studio");
  return (
    <>
      {searchable ? (
        <div className="project-filter__search">
          <SearchIcon />
          <input
            type="search"
            data-project-search
            value={query}
            placeholder={t("galleryProject.search", { defaultValue: "Search projects" })}
            aria-label={t("galleryProject.search", { defaultValue: "Search projects" })}
            onChange={(event) => onQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && rows[0]) {
                event.preventDefault();
                onPick(rows[0].id);
              }
            }}
          />
        </div>
      ) : null}
      <div id={id} role="listbox" aria-label={t("galleryProject.filter", { defaultValue: "Project" })} className="project-filter__list">
        {rows.map((row) => (
          <button
            key={row.id}
            type="button"
            role="option"
            aria-selected={row.id === value}
            tabIndex={-1}
            className={`project-filter__option${row.id === value ? " active" : ""}`}
            onClick={() => onPick(row.id)}
          >
            <span className="project-filter__name">{row.label}</span>
            <span className="project-filter__count muted-text">{row.count}</span>
          </button>
        ))}
        {rows.length === 0 ? (
          <p className="project-filter__empty muted-text">{t("galleryProject.noMatch", { defaultValue: "No project matches." })}</p>
        ) : null}
      </div>
    </>
  );
}

/**
 * The gallery toolbar's Project dropdown: All projects, No project, then one row per project with its output
 * count, the project with the newest output first. A search box appears once there are many projects.
 * The list is drawn in a portal so a panel's overflow cannot clip it. Keys follow ActionMenu: the list takes
 * focus on open, Arrow Up/Down, Home and End move, Enter picks, Tab closes, Escape closes and refocuses.
 * Hidden when no output has a project.
 */
export function ProjectFilter({
  options,
  value,
  onChange,
  untagged = 0,
  total = 0,
  showLabel = true,
}: {
  options: ProjectOption[];
  value: string;
  onChange: (value: string) => void;
  /** Outputs without a project; the No project row is shown only when there are some. */
  untagged?: number;
  /** All outputs, for the count on the All projects row. */
  total?: number;
  /** False in a row toolbar where the visible "Project" label would push the button out of line. */
  showLabel?: boolean;
}) {
  const { t } = useTranslation("studio");
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [place, setPlace] = useState<{ top: number; left: number; flip: boolean } | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const uid = useId();
  const labelId = `${uid}-label`;
  const listId = `${uid}-list`;

  function close(refocus: boolean) {
    setOpen(false);
    setQuery("");
    if (refocus) trigger.current?.focus();
  }

  useLayoutEffect(() => {
    if (!open || !trigger.current) return;
    const place = () => {
      const rect = trigger.current?.getBoundingClientRect();
      if (!rect) return;
      const flip = window.innerHeight - rect.bottom < 320 && rect.top > window.innerHeight - rect.bottom;
      setPlace({ top: flip ? rect.top - 4 : rect.bottom + 4, left: Math.max(8, Math.min(rect.left, window.innerWidth - 336)), flip });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open || !place) return;
    const onPointer = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!panel.current?.contains(target) && !trigger.current?.contains(target)) close(false);
    };
    document.addEventListener("mousedown", onPointer);
    return () => document.removeEventListener("mousedown", onPointer);
  }, [open, place]);

  // Focus the search box, else the chosen row, when the list opens.
  useEffect(() => {
    if (!open || !place) return;
    const search = panel.current?.querySelector<HTMLInputElement>("[data-project-search]");
    if (search) search.focus();
    else (panel.current?.querySelector<HTMLElement>('[aria-selected="true"]') ?? panel.current?.querySelector<HTMLElement>('[role="option"]'))?.focus();
  }, [open, place === null]);

  if (options.length === 0) return null;

  const allLabel = t("galleryProject.all", { defaultValue: "All projects" });
  const noneLabel = t("galleryProject.none", { defaultValue: "No project" });
  const current =
    value === ALL_PROJECTS
      ? allLabel
      : value === NO_PROJECT
        ? noneLabel
        : (options.find((option) => option.id === value)?.name ?? allLabel);
  const rows = projectRows(options, query, { all: allLabel, none: noneLabel }, total, untagged);
  const filterLabel = t("galleryProject.filter", { defaultValue: "Project" });

  function pick(id: string) {
    onChange(id);
    close(true);
  }

  function onPanelKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const buttons = Array.from(panel.current?.querySelectorAll<HTMLButtonElement>('[role="option"]') ?? []);
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    switch (event.key) {
      case "Escape":
        event.preventDefault();
        close(true);
        break;
      case "Tab":
        close(false);
        break;
      case "ArrowDown":
        event.preventDefault();
        buttons[(index + 1) % buttons.length]?.focus();
        break;
      case "ArrowUp":
        event.preventDefault();
        if (index <= 0 && panel.current?.querySelector("[data-project-search]")) panel.current.querySelector<HTMLInputElement>("[data-project-search]")?.focus();
        else buttons[(index - 1 + buttons.length) % buttons.length]?.focus();
        break;
      case "Home":
        if (index >= 0) {
          event.preventDefault();
          buttons[0]?.focus();
        }
        break;
      case "End":
        if (index >= 0) {
          event.preventDefault();
          buttons[buttons.length - 1]?.focus();
        }
        break;
    }
  }

  return (
    <div className="project-filter">
      {showLabel ? (
        <span id={labelId} className="project-filter__label">
          {filterLabel}
        </span>
      ) : null}
      <button
        type="button"
        ref={trigger}
        className="text-input project-filter__button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-labelledby={showLabel ? `${labelId} ${uid}-value` : undefined}
        aria-label={showLabel ? undefined : `${filterLabel}: ${current}`}
        title={current}
        onClick={() => (open ? close(false) : setOpen(true))}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" && !open) {
            event.preventDefault();
            setOpen(true);
          }
        }}
      >
        <span id={`${uid}-value`} className="project-filter__current">
          {current}
        </span>
        <ChevronDownIcon />
      </button>
      {open && typeof document !== "undefined"
        ? createPortal(
            <div
              ref={panel}
              className="project-filter__panel"
              style={{
                top: place?.top ?? 0,
                left: place?.left ?? 0,
                transform: place?.flip ? "translateY(-100%)" : undefined,
                visibility: place ? "visible" : "hidden",
              }}
              onKeyDown={onPanelKeyDown}
            >
              <ProjectList
                id={listId}
                rows={rows}
                value={value}
                searchable={options.length >= SEARCH_FROM}
                query={query}
                onQuery={setQuery}
                onPick={pick}
              />
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}

/** What a selected project holds in this gallery; zero or missing counts are left out. */
export interface ProjectCounts {
  stills?: number;
  clips?: number;
  tracks?: number;
  sounds?: number;
  speech?: number;
  transcripts?: number;
}

/** Header over a gallery filtered to one project: its name, what it holds, and a link into the Storyboard. */
export function ProjectHeader({
  name,
  counts,
  onOpen,
}: {
  name: string;
  counts: ProjectCounts;
  /** Opens the project in Storyboard; omit when the Storyboard tab is off. */
  onOpen?: () => void;
}) {
  const { t } = useTranslation("studio");
  const parts = [
    counts.stills ? t("galleryProject.stills", { defaultValue: "{count, plural, one {# still} other {# stills}}", count: counts.stills }) : null,
    counts.clips ? t("galleryProject.clips", { defaultValue: "{count, plural, one {# clip} other {# clips}}", count: counts.clips }) : null,
    counts.tracks ? t("galleryProject.tracks", { defaultValue: "{count, plural, one {# track} other {# tracks}}", count: counts.tracks }) : null,
    counts.sounds ? t("galleryProject.sounds", { defaultValue: "{count, plural, one {# sound} other {# sounds}}", count: counts.sounds }) : null,
    counts.speech ? t("galleryProject.speech", { defaultValue: "{count, plural, one {# speech item} other {# speech items}}", count: counts.speech }) : null,
    counts.transcripts ? t("galleryProject.transcripts", { defaultValue: "{count, plural, one {# transcript} other {# transcripts}}", count: counts.transcripts }) : null,
  ].filter(Boolean);
  return (
    <div className="project-header">
      <div className="project-header__text">
        <strong>{name}</strong>
        <span className="muted-text">{parts.join(" · ")}</span>
      </div>
      {onOpen ? (
        <button type="button" className="secondary-button" onClick={onOpen}>
          {t("galleryProject.open", { defaultValue: "Open in Storyboard" })}
        </button>
      ) : null}
    </div>
  );
}

/** 'From project X' chip on a card made for a project. */
export function ProjectChip({ name }: { name?: string | null }) {
  const { t } = useTranslation("studio");
  if (!name) return null;
  return (
    <span className="badge subtle project-card-badge" title={name}>{t("galleryProject.chip", { defaultValue: "From project {name}", name })}</span>
  );
}
