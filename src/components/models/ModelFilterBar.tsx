import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { MODEL_AVAILABILITY_FILTERS, type ModelAvailabilityFilter, type ModelEntry } from "./modelEntry";
import type { ModelBucketCounts } from "./modelStatus";
import type { ModelFilterGroup } from "./modelFilters";
import { modelBucketLabel } from "./ModelStatusChip";

export interface ModelFilterBarProps<E extends ModelEntry = ModelEntry> {
  search: {
    value: string;
    onChange: (value: string) => void;
    placeholder: string;
    label?: string;
  };
  availability: {
    value: ModelAvailabilityFilter;
    onChange: (value: ModelAvailabilityFilter) => void;
    counts: ModelBucketCounts;
  };
  groups?: ModelFilterGroup<E>[];
  sort?: {
    options: Array<{ id: string; label: string }>;
    value: string;
    onChange: (id: string) => void;
    label?: string;
  };
  hasActiveFilters: boolean;
  onClear: () => void;
  /** Extra controls at the end of the row. */
  extra?: ReactNode;
}

function visibleGroup<E extends ModelEntry>(group: ModelFilterGroup<E>): boolean {
  return !(group.hideBelow && group.options.length < group.hideBelow);
}

export function ModelFilterBar<E extends ModelEntry = ModelEntry>({
  search,
  availability,
  groups = [],
  sort,
  hasActiveFilters,
  onClear,
  extra,
}: ModelFilterBarProps<E>) {
  const { t } = useTranslation("library");
  const selectGroups = groups.filter((group) => group.kind === "select" && visibleGroup(group));
  const chipGroups = groups.filter((group) => group.kind === "chips" && visibleGroup(group));
  return (
    <div className="models-filter-bar">
      <div
        className="models-availability"
        role="group"
        aria-label={t("models.availability.label", { defaultValue: "Show" })}
      >
        {MODEL_AVAILABILITY_FILTERS.map((bucket) => {
          const active = availability.value === bucket;
          const count = availability.counts[bucket];
          return (
            <button
              key={bucket}
              type="button"
              className={`models-availability__option${active ? " is-active" : ""}${count === 0 && !active ? " is-empty" : ""}`}
              aria-pressed={active}
              data-bucket={bucket}
              onClick={() => availability.onChange(bucket)}
            >
              <span>{modelBucketLabel(bucket, t)}</span>
              <span className="models-availability__count">{count}</span>
            </button>
          );
        })}
      </div>

      <div
        className="models-filter-row"
        style={{ ["--models-filter-selects" as string]: String(selectGroups.length + (sort ? 1 : 0)) }}
      >
        <label className="models-filter-search">
          {search.label ?? t("models.filter.search", { defaultValue: "Search" })}
          <input
            className="text-input"
            type="search"
            value={search.value}
            onChange={(event) => search.onChange(event.target.value)}
            placeholder={search.placeholder}
          />
        </label>
        {selectGroups.map((group) => (
          <label key={group.id}>
            {group.label}
            <select
              className="text-input"
              value={group.value}
              onChange={(event) => group.onChange(event.target.value)}
            >
              {group.options.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.count !== undefined ? `${option.label} (${option.count})` : option.label}
                </option>
              ))}
            </select>
          </label>
        ))}
        {sort ? (
          <label>
            {sort.label ?? t("models.filter.sortBy", { defaultValue: "Sort by" })}
            <select className="text-input" value={sort.value} onChange={(event) => sort.onChange(event.target.value)}>
              {sort.options.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        <div className="models-filter-actions">
          {extra}
          <button className="secondary-button" type="button" onClick={onClear} disabled={!hasActiveFilters}>
            {t("models.filter.clear", { defaultValue: "Clear Filters" })}
          </button>
        </div>
      </div>

      {chipGroups.map((group) => (
        <div key={group.id} className="models-chip-group" role="group" aria-label={group.label}>
          <span className="models-chip-group__label">{group.label}</span>
          {group.options.map((option) => {
            const active = group.value === option.value;
            return (
              <button
                key={option.value}
                type="button"
                className={`models-chip${active ? " is-active" : ""}`}
                aria-pressed={active}
                title={option.title}
                style={option.color ? { ["--chip-color" as string]: option.color } : undefined}
                onClick={() => group.onChange(active ? group.defaultValue : option.value)}
              >
                {option.label}
                {option.count !== undefined ? <span className="models-chip__count">{option.count}</span> : null}
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}
