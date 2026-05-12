import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import type { DownloadStatus } from "../../api";
import type {
  ImageModelFamily,
  ImageModelVariant,
  TabId,
} from "../../types";
import {
  compactModelSizeLabel,
  compactReleaseLabel,
  downloadProgressLabel,
  formatReleaseLabel,
  imageDiscoverMemoryEstimate,
  imagePrimarySizeLabel,
  imageSecondarySizeLabel,
} from "../../utils";

type InstalledImageSort = "name" | "provider" | "tasks" | "size" | "ram" | "date" | "status";
type SortDir = "asc" | "desc";
type InstalledImageStatusFilter = "all" | "installed" | "incomplete" | "downloading" | "paused" | "failed";

export interface ImageModelsTabProps {
  installedImageVariants: ImageModelVariant[];
  imageCatalog: ImageModelFamily[];
  activeImageDownloads: Record<string, DownloadStatus>;
  fileRevealLabel: string;
  onActiveTabChange: (tab: TabId) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
  onDeleteImageDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
}

function releaseSortKey(variant: ImageModelVariant): string {
  return variant.releaseDate ?? variant.createdAt ?? variant.lastModified ?? "";
}

function sizeSortKey(variant: ImageModelVariant): number | null {
  const candidates = [variant.onDiskGb, variant.coreWeightsGb, variant.repoSizeGb, variant.sizeGb];
  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  }
  return null;
}

function compareNullableNumberDesc(left: number | null, right: number | null): number {
  const leftKnown = typeof left === "number" && Number.isFinite(left);
  const rightKnown = typeof right === "number" && Number.isFinite(right);
  if (leftKnown && rightKnown) return (right as number) - (left as number);
  if (leftKnown) return -1;
  if (rightKnown) return 1;
  return 0;
}

function compareNullableNumber(left: number | null, right: number | null, dir: SortDir): number {
  const desc = compareNullableNumberDesc(left, right);
  return dir === "desc" ? desc : -desc;
}

function statusSortKey(status: InstalledImageStatusFilter): number {
  if (status === "installed") return 0;
  if (status === "downloading") return 1;
  if (status === "paused") return 2;
  if (status === "failed") return 3;
  if (status === "incomplete") return 4;
  return 5;
}

function defaultSortDir(sort: InstalledImageSort): SortDir {
  return sort === "name" || sort === "provider" || sort === "tasks" ? "asc" : "desc";
}

function imageStatus(variant: ImageModelVariant, downloadState?: DownloadStatus): InstalledImageStatusFilter {
  if (downloadState?.state === "downloading") return "downloading";
  if (downloadState?.state === "cancelled") return "paused";
  if (downloadState?.state === "failed") return "failed";
  if (variant.availableLocally || downloadState?.state === "completed") return "installed";
  return "incomplete";
}

function StatusBadge({ status, downloadState }: { status: InstalledImageStatusFilter; downloadState?: DownloadStatus }) {
  const { t } = useTranslation("library");
  if (status === "installed") return <StatusIcon status="installed" label={t("imageModels.status.installed", { defaultValue: "Installed" })} />;
  if (status === "downloading" && downloadState) return <StatusIcon status="downloading" label={t("imageModels.status.downloading", { defaultValue: "Downloading" })} detail={downloadProgressLabel(downloadState)} />;
  if (status === "paused" && downloadState) return <StatusIcon status="paused" label={t("imageModels.status.paused", { defaultValue: "Paused" })} detail={downloadProgressLabel(downloadState)} />;
  if (status === "failed") return <StatusIcon status="failed" label={t("imageModels.status.failed", { defaultValue: "Failed" })} detail={downloadState?.error ?? t("imageModels.status.downloadFailed", { defaultValue: "Download failed" })} />;
  return <StatusIcon status="incomplete" label={t("imageModels.status.incomplete", { defaultValue: "Incomplete" })} />;
}

function sortIndicator(activeSort: InstalledImageSort, sortDir: SortDir, key: InstalledImageSort): string {
  if (activeSort !== key) return "";
  return sortDir === "asc" ? " \u25B2" : " \u25BC";
}

function sortLabel(sort: InstalledImageSort, sortDir: SortDir, t: (key: string, opts?: Record<string, unknown>) => string): string {
  const direction = sortDir === "asc"
    ? t("imageModels.sort.direction.ascending", { defaultValue: "ascending" })
    : t("imageModels.sort.direction.descending", { defaultValue: "descending" });
  if (sort === "provider") return t("imageModels.sort.providerDir", { defaultValue: "provider {direction}", direction });
  if (sort === "tasks") return t("imageModels.sort.tasksDir", { defaultValue: "tasks {direction}", direction });
  if (sort === "size") return sortDir === "desc"
    ? t("imageModels.sort.largestFirst", { defaultValue: "largest size first" })
    : t("imageModels.sort.smallestFirst", { defaultValue: "smallest size first" });
  if (sort === "ram") return sortDir === "desc"
    ? t("imageModels.sort.highestRam", { defaultValue: "highest RAM/VRAM first" })
    : t("imageModels.sort.lowestRam", { defaultValue: "lowest RAM/VRAM first" });
  if (sort === "status") return t("imageModels.sort.statusDir", { defaultValue: "status {direction}", direction });
  if (sort === "name") return sortDir === "asc"
    ? t("imageModels.sort.nameAsc", { defaultValue: "name A-Z" })
    : t("imageModels.sort.nameDesc", { defaultValue: "name Z-A" });
  return sortDir === "desc"
    ? t("imageModels.sort.newestFirst", { defaultValue: "newest released first" })
    : t("imageModels.sort.oldestFirst", { defaultValue: "oldest released first" });
}

function memoryParts(label: string | null | undefined, pendingLabel: string): { primary: string; secondary: string | null } {
  if (!label) return { primary: pendingLabel, secondary: null };
  const [primary, secondary] = label.split(" @ ");
  if (!secondary) return { primary, secondary: null };
  return { primary: `${primary} @`, secondary };
}

export function ImageModelsTab({
  installedImageVariants,
  imageCatalog,
  activeImageDownloads,
  fileRevealLabel,
  onActiveTabChange,
  onOpenImageStudio,
  onImageDownload,
  onCancelImageDownload,
  onDeleteImageDownload,
  onOpenExternalUrl,
  onRevealPath,
}: ImageModelsTabProps) {
  const { t } = useTranslation("library");
  const pendingMemoryLabel = t("imageModels.memory.pending", { defaultValue: "pending" });
  const [searchInput, setSearchInput] = useState("");
  const [taskFilter, setTaskFilter] = useState<"all" | ImageModelVariant["taskSupport"][number]>("all");
  const [statusFilter, setStatusFilter] = useState<InstalledImageStatusFilter>("all");
  const [sort, setSort] = useState<InstalledImageSort>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const normalizedSearch = searchInput.trim().toLowerCase();
  const hasActiveFilters =
    normalizedSearch.length > 0 || taskFilter !== "all" || statusFilter !== "all" || sort !== "date" || sortDir !== "desc";

  function applySort(nextSort: InstalledImageSort) {
    if (sort === nextSort) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSort(nextSort);
      setSortDir(defaultSortDir(nextSort));
    }
  }

  const rows = useMemo(() => {
    return installedImageVariants
      .map((variant) => {
        const family = imageCatalog.find((item) => item.variants.some((candidate) => candidate.id === variant.id));
        const downloadState = activeImageDownloads[variant.repo];
        const status = imageStatus(variant, downloadState);
        const memoryEstimate = imageDiscoverMemoryEstimate(variant);
        return { variant, family, downloadState, status, memoryEstimate };
      })
      .filter(({ variant, family, status }) => {
        if (taskFilter !== "all" && !variant.taskSupport.includes(taskFilter)) return false;
        if (statusFilter !== "all" && status !== statusFilter) return false;
        if (!normalizedSearch) return true;
        const haystack = [
          variant.name,
          variant.provider,
          variant.repo,
          variant.runtime,
          family?.name ?? "",
          variant.recommendedResolution,
          variant.styleTags.join(" "),
          variant.taskSupport.join(" "),
        ].join(" ").toLowerCase();
        return haystack.includes(normalizedSearch);
      })
      .sort((left, right) => {
        if (sort === "name") {
          const diff = left.variant.name.localeCompare(right.variant.name);
          return sortDir === "asc" ? diff : -diff;
        }
        if (sort === "provider") {
          const diff = left.variant.provider.localeCompare(right.variant.provider);
          if (diff !== 0) return sortDir === "asc" ? diff : -diff;
        }
        if (sort === "tasks") {
          const diff = left.variant.taskSupport.join(" ").localeCompare(right.variant.taskSupport.join(" "));
          if (diff !== 0) return sortDir === "asc" ? diff : -diff;
        }
        if (sort === "size") {
          const diff = compareNullableNumber(sizeSortKey(left.variant), sizeSortKey(right.variant), sortDir);
          if (diff !== 0) return diff;
        } else if (sort === "ram") {
          const diff = compareNullableNumber(left.memoryEstimate?.estimatedPeakGb ?? null, right.memoryEstimate?.estimatedPeakGb ?? null, sortDir);
          if (diff !== 0) return diff;
        } else if (sort === "status") {
          const diff = statusSortKey(left.status) - statusSortKey(right.status);
          if (diff !== 0) return sortDir === "asc" ? diff : -diff;
        }
        const dateDiff = releaseSortKey(right.variant).localeCompare(releaseSortKey(left.variant));
        if (dateDiff !== 0) return sortDir === "desc" ? dateDiff : -dateDiff;
        return left.variant.name.localeCompare(right.variant.name);
      });
  }, [activeImageDownloads, imageCatalog, installedImageVariants, normalizedSearch, sort, sortDir, statusFilter, taskFilter]);

  return (
    <div className="content-grid image-page-grid">
      <Panel
        title={t("common:panels.installedImageModels", { defaultValue: "Installed Image Models" })}
        subtitle={installedImageVariants.length > 0
          ? t("imageModels.subtitle.withLocal", {
              defaultValue: "{rows} of {total, plural, one {# model} other {# models}} with local data",
              rows: rows.length,
              total: installedImageVariants.length,
            })
          : t("imageModels.subtitle.none", { defaultValue: "No image models detected locally yet" })}
        className="span-2"
        actions={
          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-discover")}>
            {t("imageModels.action.browseCatalog", { defaultValue: "Browse Catalog" })}
          </button>
        }
      >
        {installedImageVariants.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>{t("imageModels.empty.noModels", { defaultValue: "Download an image model from Image Discover to get started." })}</p>
          </div>
        ) : (
          <>
            <div className="image-discover-filter-row image-discover-filter-row--wide image-model-filter-row">
              <label className="image-discover-search">
                {t("imageModels.filter.searchLabel", { defaultValue: "Search" })}
                <input
                  className="text-input"
                  type="search"
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                  placeholder={t("imageModels.filter.searchPlaceholder", { defaultValue: "Filter by model, provider, repo, task, or tag..." })}
                />
              </label>
              <label>
                {t("imageModels.filter.taskLabel", { defaultValue: "Task" })}
                <select
                  className="text-input"
                  value={taskFilter}
                  onChange={(event) => setTaskFilter(event.target.value as typeof taskFilter)}
                >
                  <option value="all">{t("imageModels.filter.task.all", { defaultValue: "All tasks" })}</option>
                  <option value="txt2img">{t("imageModels.filter.task.txt2img", { defaultValue: "Text to image" })}</option>
                  <option value="img2img">{t("imageModels.filter.task.img2img", { defaultValue: "Image to image" })}</option>
                  <option value="inpaint">{t("imageModels.filter.task.inpaint", { defaultValue: "Inpaint" })}</option>
                </select>
              </label>
              <label>
                {t("imageModels.filter.statusLabel", { defaultValue: "Status" })}
                <select
                  className="text-input"
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value as InstalledImageStatusFilter)}
                >
                  <option value="all">{t("imageModels.filter.status.all", { defaultValue: "Any status" })}</option>
                  <option value="installed">{t("imageModels.filter.status.installed", { defaultValue: "Installed" })}</option>
                  <option value="incomplete">{t("imageModels.filter.status.incomplete", { defaultValue: "Incomplete" })}</option>
                  <option value="downloading">{t("imageModels.filter.status.downloading", { defaultValue: "Downloading" })}</option>
                  <option value="paused">{t("imageModels.filter.status.paused", { defaultValue: "Paused" })}</option>
                  <option value="failed">{t("imageModels.filter.status.failed", { defaultValue: "Failed" })}</option>
                </select>
              </label>
              <label>
                {t("imageModels.filter.sortByLabel", { defaultValue: "Sort by" })}
                <select
                  className="text-input"
                  value={sort}
                  onChange={(event) => {
                    const nextSort = event.target.value as InstalledImageSort;
                    setSort(nextSort);
                    setSortDir(defaultSortDir(nextSort));
                  }}
                >
                  <option value="name">{t("imageModels.sort.name", { defaultValue: "Name" })}</option>
                  <option value="provider">{t("imageModels.sort.provider", { defaultValue: "Provider" })}</option>
                  <option value="tasks">{t("imageModels.sort.tasks", { defaultValue: "Tasks" })}</option>
                  <option value="date">{t("imageModels.sort.newestReleased", { defaultValue: "Newest released" })}</option>
                  <option value="size">{t("imageModels.sort.largestSize", { defaultValue: "Largest size" })}</option>
                  <option value="ram">{t("imageModels.sort.highestRamVram", { defaultValue: "Highest RAM/VRAM" })}</option>
                  <option value="status">{t("imageModels.sort.status", { defaultValue: "Status" })}</option>
                </select>
              </label>
              <div className="image-discover-filter-actions">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => {
                    setSearchInput("");
                    setTaskFilter("all");
                    setStatusFilter("all");
                    setSort("date");
                    setSortDir("desc");
                  }}
                  disabled={!hasActiveFilters}
                >
                  {t("imageModels.action.clearFilters", { defaultValue: "Clear Filters" })}
                </button>
              </div>
            </div>
            <div className="image-discover-results-summary">
              <span>{t("imageModels.summary.results", {
                defaultValue: "{count, plural, one {# model} other {# models}} · {sortLabel}",
                count: rows.length,
                sortLabel: sortLabel(sort, sortDir, t),
              })}</span>
              {normalizedSearch ? <span className="badge subtle">{t("imageModels.summary.searchChip", { defaultValue: "Search: {query}", query: searchInput.trim() })}</span> : null}
              {taskFilter !== "all" ? <span className="badge muted">{t("imageModels.summary.taskChip", { defaultValue: "Task: {value}", value: taskFilter })}</span> : null}
              {statusFilter !== "all" ? <span className="badge muted">{t("imageModels.summary.statusChip", { defaultValue: "Status: {value}", value: statusFilter })}</span> : null}
            </div>
            {rows.length === 0 ? (
              <div className="empty-state image-empty-state">
                <p>{t("imageModels.empty.noMatches", { defaultValue: "No installed image models match the current filters." })}</p>
              </div>
            ) : (
              <div className="media-model-table media-model-table--image">
                <div className="media-model-head">
                  <button className="sort-header" type="button" onClick={() => applySort("name")}>{t("imageModels.column.model", { defaultValue: "Model" })}{sortIndicator(sort, sortDir, "name")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("provider")}>{t("imageModels.column.provider", { defaultValue: "Provider" })}{sortIndicator(sort, sortDir, "provider")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("tasks")}>{t("imageModels.column.tasks", { defaultValue: "Tasks" })}{sortIndicator(sort, sortDir, "tasks")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("size")}>{t("imageModels.column.size", { defaultValue: "Size" })}{sortIndicator(sort, sortDir, "size")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("ram")}>{t("imageModels.column.ramVram", { defaultValue: "RAM/VRAM" })}{sortIndicator(sort, sortDir, "ram")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("date")}>{t("imageModels.column.released", { defaultValue: "Released" })}{sortIndicator(sort, sortDir, "date")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("status")}>{t("imageModels.column.status", { defaultValue: "Status" })}{sortIndicator(sort, sortDir, "status")}</button>
                  <span className="sort-header"></span>
                </div>
                <div className="media-model-rows">
                  {rows.map(({ variant, family, downloadState, status, memoryEstimate }) => {
                    const isComplete = status === "installed";
                    const isDownloading = status === "downloading";
                    const isPaused = status === "paused";
                    const isDownloadFailed = status === "failed";
                    const isPartial = status === "incomplete";
                    const canDeleteLocalData = Boolean(isComplete || isPaused || isDownloadFailed || isPartial);
                    const secondarySize = imageSecondarySizeLabel(variant);
                    const releaseLabel = compactReleaseLabel(formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt));
                    const primarySizeLabel = imagePrimarySizeLabel(variant);
                    const sizeTitle = [primarySizeLabel, secondarySize].filter(Boolean).join(" / ");
                    const memory = memoryParts(memoryEstimate?.label, pendingMemoryLabel);
                    return (
                      <div key={variant.id} className={`media-model-row-wrap${isComplete ? " downloaded" : ""}`}>
                        <div className="media-model-row">
                          <div className="media-model-name">
                            <strong>{variant.name}</strong>
                            <small>{family?.name ?? variant.provider}</small>
                            <div className="media-model-chip-row">
                              {variant.styleTags.slice(0, 4).map((tag) => (
                                <span key={tag} className="badge subtle">{tag}</span>
                              ))}
                            </div>
                          </div>
                          <span>{variant.provider}</span>
                          <div className="media-model-chip-row">
                            {variant.taskSupport.map((task) => (
                              <span key={task} className="badge muted">{task}</span>
                            ))}
                          </div>
                          <span title={sizeTitle || undefined}>
                            {compactModelSizeLabel(primarySizeLabel)}
                          </span>
                          <span className="media-model-memory" title={memoryEstimate?.title ?? t("imageModels.memory.pendingTitle", { defaultValue: "RAM/VRAM estimate pending until model weight size is known." })}>
                            <span>{memory.primary}</span>
                            {memory.secondary ? <small>{memory.secondary}</small> : null}
                          </span>
                          <span>{releaseLabel ?? t("imageModels.releaseUnknown", { defaultValue: "Unknown" })}</span>
                          <span><StatusBadge status={status} downloadState={downloadState} /></span>
                          <div className="media-model-actions">
                            {isComplete ? (
                              <IconActionButton icon="generate" label={t("imageModels.action.generate", { defaultValue: "Generate" })} buttonStyle="primary" onClick={() => onOpenImageStudio(variant.id)} />
                            ) : isDownloading ? (
                              <IconActionButton icon="pause" label={t("imageModels.action.pauseDownload", { defaultValue: "Pause download" })} onClick={() => onCancelImageDownload(variant.repo)} />
                            ) : (
                              <IconActionButton icon={isDownloadFailed ? "retry" : isPartial ? "resume" : "download"} label={isDownloadFailed
                                ? t("imageModels.action.retryDownload", { defaultValue: "Retry download" })
                                : isPartial
                                  ? t("imageModels.action.resumeDownload", { defaultValue: "Resume download" })
                                  : t("imageModels.action.downloadModel", { defaultValue: "Download model" })} onClick={() => onImageDownload(variant.repo)} />
                            )}
                            {isDownloading || canDeleteLocalData ? (
                              <IconActionButton icon={isDownloading ? "cancel" : "delete"} label={isDownloading
                                ? t("imageModels.action.cancelDownload", { defaultValue: "Cancel download" })
                                : t("imageModels.action.deleteModel", { defaultValue: "Delete model" })} danger onClick={() => onDeleteImageDownload(variant.repo)} />
                            ) : null}
                            {variant.localPath ? (
                              <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(variant.localPath as string)} />
                            ) : null}
                            <IconActionButton icon="modelCard" label={t("imageModels.action.openModelCard", { defaultValue: "Open model card" })} onClick={() => onOpenExternalUrl(variant.link)} />
                          </div>
                        </div>
                        {isDownloadFailed && downloadState?.error ? (
                          <div className="media-model-row-detail callout error">
                            <p>{downloadState.error}</p>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </Panel>
    </div>
  );
}
