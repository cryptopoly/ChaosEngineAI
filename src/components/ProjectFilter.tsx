import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  ALL_PROJECTS,
  NO_PROJECT,
  SEARCH_FROM,
  searchProjects,
  type ProjectOption,
} from "../utils/galleryProject";
import { ChevronDownIcon, SearchIcon } from "./icons/ActionIcons";

/**
 * The gallery toolbar's Project dropdown: All projects, No project, then one row per project with its output
 * count, the project with the newest output first. A search box appears once there are many projects.
 * Hidden when no output has a project.
 */
export function ProjectFilter({
  options,
  value,
  onChange,
  untagged = 0,
  total = 0,
  initialOpen = false,
}: {
  options: ProjectOption[];
  value: string;
  onChange: (value: string) => void;
  /** Outputs without a project; the No project row is shown only when there are some. */
  untagged?: number;
  /** All outputs, for the count on the All projects row. */
  total?: number;
  /** Render the list already open; for tests. */
  initialOpen?: boolean;
}) {
  const { t } = useTranslation("studio");
  const [open, setOpen] = useState(initialOpen);
  const [query, setQuery] = useState("");
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    return () => document.removeEventListener("mousedown", onPointer);
  }, [open]);

  if (options.length === 0) return null;

  const searchable = options.length >= SEARCH_FROM;
  const shown = searchProjects(options, query);
  const allLabel = t("galleryProject.all", { defaultValue: "All projects" });
  const noneLabel = t("galleryProject.none", { defaultValue: "No project" });
  const current =
    value === ALL_PROJECTS
      ? allLabel
      : value === NO_PROJECT
        ? noneLabel
        : (options.find((option) => option.id === value)?.name ?? allLabel);

  const rows: { id: string; label: string; count: number }[] = [
    ...(query.trim() ? [] : [{ id: ALL_PROJECTS, label: allLabel, count: total }]),
    ...(untagged > 0 && !query.trim() ? [{ id: NO_PROJECT, label: noneLabel, count: untagged }] : []),
    ...shown.map((option) => ({ id: option.id, label: option.name, count: option.count })),
  ];

  function pick(id: string) {
    onChange(id);
    setOpen(false);
    setQuery("");
  }

  return (
    <div className="project-filter" ref={root}>
      <span className="project-filter__label">{t("galleryProject.filter", { defaultValue: "Project" })}</span>
      <button
        type="button"
        className="text-input project-filter__button"
        aria-haspopup="listbox"
        aria-expanded={open}
        title={current}
        onClick={() => setOpen(!open)}
        onKeyDown={(event) => {
          if (event.key === "Escape") setOpen(false);
        }}
      >
        <span className="project-filter__current">{current}</span>
        <ChevronDownIcon />
      </button>
      {open ? (
        <div
          className="project-filter__panel"
          onKeyDown={(event) => {
            if (event.key === "Escape") setOpen(false);
          }}
        >
          {searchable ? (
            <label className="project-filter__search">
              <SearchIcon />
              <input
                type="search"
                autoFocus
                value={query}
                placeholder={t("galleryProject.search", { defaultValue: "Search projects" })}
                aria-label={t("galleryProject.search", { defaultValue: "Search projects" })}
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
          ) : null}
          <ul role="listbox" aria-label={t("galleryProject.filter", { defaultValue: "Project" })} className="project-filter__list">
            {rows.map((row) => (
              <li key={row.id} role="option" aria-selected={row.id === value}>
                <button
                  type="button"
                  className={`project-filter__option${row.id === value ? " active" : ""}`}
                  onClick={() => pick(row.id)}
                >
                  <span className="project-filter__name">{row.label}</span>
                  <span className="project-filter__count muted-text">{row.count}</span>
                </button>
              </li>
            ))}
            {rows.length === 0 ? (
              <li className="project-filter__empty muted-text">{t("galleryProject.noMatch", { defaultValue: "No project matches." })}</li>
            ) : null}
          </ul>
        </div>
      ) : null}
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
    <span className="badge subtle">{t("galleryProject.chip", { defaultValue: "From project {name}", name })}</span>
  );
}
