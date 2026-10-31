import { useMemo, type CSSProperties, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../Panel";
import type { ModelAvailabilityFilter, ModelEntry } from "./modelEntry";
import type { ResolvedModelStatus } from "./modelStatus";
import {
  buildModelRows,
  hasActiveModelFilters,
  type ModelFilterGroup,
  type ModelRow,
  type ModelSortOption,
  type SortDir,
} from "./modelFilters";
import { ModelFilterBar } from "./ModelFilterBar";
import { modelBucketLabel } from "./ModelStatusChip";

export interface ModelsPageColumn {
  id: string;
  label: string;
  /** Clicking the header sorts by this sort option id. */
  sortId?: string;
}

/** The numbers behind the default subtitle. */
export interface ModelsPageCounts {
  shown: number;
  total: number;
  installed: number;
}

export interface ModelsPageProps<E extends ModelEntry> {
  /** Extra class on the page root ("image-models-page"...). */
  className?: string;
  title: string;
  /** Default: "{shown} of {total} models · {installed} installed", where
   * total counts the rows this machine can show before any filter. A
   * function receives those counts, to add to them (chat adds disk use). */
  subtitle?: string | ((counts: ModelsPageCounts) => string);
  /** Buttons in the panel header (Open Studio...). */
  headerActions?: ReactNode;
  /** Short text under the header. */
  intro?: ReactNode;
  /** Between the filter bar and the table (install banners, Hub results). */
  beforeList?: ReactNode;
  /** After the table (Run from Hugging Face, Import panels). */
  afterList?: ReactNode;

  /** Every row of the area; rows the host cannot run are dropped by the
   * shell unless something of them is on disk. */
  entries: E[];
  resolveStatus: (entry: E) => ResolvedModelStatus;

  search: { value: string; onChange: (value: string) => void; placeholder: string };
  /** When given, the shell filters by search; otherwise ``entries`` is
   * already filtered by the area. */
  matches?: (entry: E, query: string) => boolean;
  groups?: ModelFilterGroup<E>[];
  availability: ModelAvailabilityFilter;
  onAvailabilityChange: (value: ModelAvailabilityFilter) => void;

  /** Rows kept above the others whatever the sort (chat favourites). */
  pinned?: (entry: E) => boolean;

  sortOptions: ModelSortOption<E>[];
  sortId: string;
  sortDir: SortDir;
  defaultSortId: string;
  onSortChange: (id: string, dir: SortDir) => void;
  /** Reset the area's own filters (search, groups); the shell resets
   * availability and sort itself. */
  onClearFilters: () => void;

  columns: ModelsPageColumn[];
  /** CSS grid template for name + columns + status + actions. */
  gridTemplate?: string;
  /** Template used in a window narrower than 1440 px, with the (1-based) columns in
   * ``narrowHide`` left out so the table fits without scrolling sideways. */
  narrowGridTemplate?: string;
  narrowHide?: readonly number[];
  renderRow: (row: ModelRow<E>) => ReactNode;

  /** Text for an empty catalog. */
  emptyCatalog?: ReactNode;
  /** Text when the availability filter leaves nothing (keyed by bucket). */
  emptyByAvailability?: Partial<Record<ModelAvailabilityFilter, ReactNode>>;
  /** Text when search and filters leave nothing. Default: a generic
   * "no models match" line. */
  emptyFilters?: ReactNode;
}

function sortIndicator(active: boolean, dir: SortDir): string {
  if (!active) return "";
  return dir === "asc" ? " ▲" : " ▼";
}

export function ModelsPage<E extends ModelEntry>(props: ModelsPageProps<E>) {
  const { t } = useTranslation("library");
  const {
    className,
    title,
    subtitle,
    headerActions,
    intro,
    beforeList,
    afterList,
    entries,
    resolveStatus,
    search,
    matches,
    groups,
    availability,
    onAvailabilityChange,
    pinned,
    sortOptions,
    sortId,
    sortDir,
    defaultSortId,
    onSortChange,
    onClearFilters,
    columns,
    gridTemplate,
    narrowGridTemplate,
    narrowHide,
    renderRow,
    emptyCatalog,
    emptyByAvailability,
    emptyFilters,
  } = props;

  const sort = sortOptions.find((option) => option.id === sortId) ?? null;
  const defaultSort = sortOptions.find((option) => option.id === defaultSortId) ?? null;

  const { rows, counts, hostVisibleCount } = useMemo(
    () =>
      buildModelRows({
        entries,
        resolveStatus,
        query: matches ? search.value : "",
        matches,
        groups,
        availability,
        sort,
        sortDir,
        pinned,
      }),
    [entries, resolveStatus, matches, search.value, groups, availability, sort, sortDir, pinned],
  );

  const hasActiveFilters = hasActiveModelFilters({
    query: search.value,
    groups,
    availability,
    sortId,
    defaultSortId,
    sortDir,
    defaultSortDir: defaultSort?.defaultDir,
  });

  function clearAll() {
    onClearFilters();
    onAvailabilityChange("all");
    if (defaultSort) onSortChange(defaultSort.id, defaultSort.defaultDir);
  }

  function applyHeaderSort(nextId: string) {
    const option = sortOptions.find((candidate) => candidate.id === nextId);
    if (!option) return;
    if (nextId === sortId) onSortChange(nextId, sortDir === "asc" ? "desc" : "asc");
    else onSortChange(nextId, option.defaultDir);
  }

  const sortLabel = sort ? (sort.summary ? sort.summary(sortDir) : sort.label) : "";
  const summaryChips: Array<{ id: string; text: string }> = [];
  if (search.value.trim()) {
    summaryChips.push({
      id: "search",
      text: t("models.summary.search", { defaultValue: "Search: {query}", query: search.value.trim() }),
    });
  }
  for (const group of groups ?? []) {
    if (group.value === group.defaultValue) continue;
    const option = group.options.find((candidate) => candidate.value === group.value);
    summaryChips.push({ id: group.id, text: `${group.label}: ${option?.label ?? group.value}` });
  }
  if (availability !== "all") {
    summaryChips.push({
      id: "availability",
      text: t("models.summary.availability", {
        defaultValue: "Showing: {bucket}",
        bucket: modelBucketLabel(availability, t),
      }),
    });
  }

  const pageCounts: ModelsPageCounts = { shown: rows.length, total: hostVisibleCount, installed: counts.installed };
  const subtitleText =
    typeof subtitle === "function"
      ? subtitle(pageCounts)
      : subtitle ??
        t("models.subtitle", {
          defaultValue: "{shown} of {total, plural, one {# model} other {# models}} · {installed} installed",
          ...pageCounts,
        });

  const gridStyle = gridTemplate
    ? ({
        ["--media-model-grid-columns" as string]: gridTemplate,
        ...(narrowGridTemplate ? { ["--media-model-grid-columns-narrow" as string]: narrowGridTemplate } : {}),
      } as CSSProperties)
    : undefined;

  let body: ReactNode;
  if (rows.length > 0) {
    body = (
      <div
        className="media-model-table models-table"
        style={gridStyle}
        data-narrow-hide={narrowHide && narrowHide.length > 0 ? narrowHide.join(" ") : undefined}
      >
        <div className="media-model-head" role="row">
          {columns.map((column) =>
            column.sortId ? (
              <button
                key={column.id}
                className="sort-header"
                type="button"
                aria-sort={column.sortId === sortId ? (sortDir === "asc" ? "ascending" : "descending") : undefined}
                onClick={() => applyHeaderSort(column.sortId as string)}
              >
                {column.label}
                {sortIndicator(column.sortId === sortId, sortDir)}
              </button>
            ) : (
              <span key={column.id} className="sort-header">
                {column.label}
              </span>
            ),
          )}
          <span className="sort-header" />
        </div>
        <div className="media-model-rows">{rows.map((row) => renderRow(row))}</div>
      </div>
    );
  } else {
    const catalogEmpty = hostVisibleCount === 0;
    const availabilityOnly = !catalogEmpty && availability !== "all" && counts.all > 0;
    // The area's own sentence ("No video models installed yet") is only
    // true when nothing else narrows the list; with a search or a group
    // filter on, say the filters hide them.
    const otherFiltersActive =
      search.value.trim().length > 0 || (groups ?? []).some((group) => group.value !== group.defaultValue);
    const message = catalogEmpty
      ? emptyCatalog ?? t("models.empty.catalog", { defaultValue: "No models in the catalog yet." })
      : availabilityOnly
        ? (otherFiltersActive ? undefined : emptyByAvailability?.[availability]) ??
          t("models.empty.availability", {
            defaultValue: "No {bucket} models under the current filters.",
            bucket: modelBucketLabel(availability, t).toLowerCase(),
          })
        : emptyFilters ??
          t("models.empty.filters", { defaultValue: "No models match the current filters. Try broadening the filters or search terms." });
    body = (
      <div className="empty-state image-empty-state models-empty">
        <p>{message}</p>
        <div className="button-row">
          {availabilityOnly && availability !== "available" ? (
            <button className="secondary-button" type="button" onClick={() => onAvailabilityChange("available")}>
              {t("models.empty.showAvailable", { defaultValue: "Show available models" })}
            </button>
          ) : null}
          {hasActiveFilters ? (
            <button className="secondary-button" type="button" onClick={clearAll}>
              {t("models.filter.clear", { defaultValue: "Clear Filters" })}
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  return (
    <div className={`models-page${className ? ` ${className}` : ""}`}>
      <Panel title={title} subtitle={subtitleText} actions={headerActions} className="models-page__panel">
        {intro ? <div className="models-page__intro">{intro}</div> : null}
        <ModelFilterBar
          search={search}
          availability={{ value: availability, onChange: onAvailabilityChange, counts }}
          groups={groups}
          sort={{
            options: sortOptions.map((option) => ({ id: option.id, label: option.label })),
            value: sortId,
            onChange: (id) => {
              const option = sortOptions.find((candidate) => candidate.id === id);
              onSortChange(id, option?.defaultDir ?? "desc");
            },
          }}
          hasActiveFilters={hasActiveFilters}
          onClear={clearAll}
        />
        <div className="image-discover-results-summary models-summary">
          <span>
            {t("models.summary.count", {
              defaultValue: "{count, plural, one {# model} other {# models}} · {sortLabel}",
              count: rows.length,
              sortLabel,
            })}
          </span>
          {summaryChips.map((chip) => (
            <span key={chip.id} className="badge muted">
              {chip.text}
            </span>
          ))}
        </div>
      </Panel>
      {beforeList}
      {body}
      {afterList}
    </div>
  );
}
