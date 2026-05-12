import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import type { DownloadStatus } from "../../api";
import type {
  TabId,
  VideoModelFamily,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../../types";
import {
  compactModelSizeLabel,
  compactReleaseLabel,
  downloadProgressLabel,
  formatReleaseLabel,
  videoDiscoverMemoryEstimate,
  videoDeleteLabelForRepo,
  videoDeleteRepoForVariant,
  videoDownloadStatusForVariant,
  videoPrimarySizeLabel,
  videoSecondarySizeLabel,
} from "../../utils";

type InstalledVideoSort = "name" | "provider" | "tasks" | "size" | "ram" | "date" | "status";
type SortDir = "asc" | "desc";
type InstalledVideoStatusFilter = "all" | "loaded" | "installed" | "incomplete" | "downloading" | "paused" | "failed";

export interface VideoModelsTabProps {
  installedVideoVariants: VideoModelVariant[];
  videoCatalog: VideoModelFamily[];
  activeVideoDownloads: Record<string, DownloadStatus>;
  videoRuntimeStatus: VideoRuntimeStatus;
  videoBusy: boolean;
  videoBusyLabel: string | null;
  loadedVideoVariant: VideoModelVariant | null;
  fileRevealLabel: string;
  onActiveTabChange: (tab: TabId) => void;
  onOpenVideoStudio: (modelId?: string) => void;
  onVideoDownload: (repo: string, modelId?: string) => void;
  onCancelVideoDownload: (repo: string) => void;
  onDeleteVideoDownload: (repo: string) => void;
  onPreloadVideoModel: (variant: VideoModelVariant) => void;
  onUnloadVideoModel: (variant?: VideoModelVariant) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
}

function releaseSortKey(variant: VideoModelVariant): string {
  return variant.releaseDate ?? variant.createdAt ?? variant.lastModified ?? "";
}

function sizeSortKey(variant: VideoModelVariant): number | null {
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

function statusSortKey(status: InstalledVideoStatusFilter): number {
  if (status === "loaded") return 0;
  if (status === "installed") return 1;
  if (status === "downloading") return 2;
  if (status === "paused") return 3;
  if (status === "failed") return 4;
  if (status === "incomplete") return 5;
  return 6;
}

function defaultSortDir(sort: InstalledVideoSort): SortDir {
  return sort === "name" || sort === "provider" || sort === "tasks" ? "asc" : "desc";
}

function videoStatus(
  variant: VideoModelVariant,
  downloadState: DownloadStatus | undefined,
  loadedVideoVariant: VideoModelVariant | null,
): InstalledVideoStatusFilter {
  if (loadedVideoVariant?.id === variant.id) return "loaded";
  if (downloadState?.state === "downloading") return "downloading";
  if (downloadState?.state === "cancelled") return "paused";
  if (downloadState?.state === "failed") return "failed";
  if (variant.availableLocally || downloadState?.state === "completed") return "installed";
  return "incomplete";
}

function StatusBadge({ status, downloadState }: { status: InstalledVideoStatusFilter; downloadState?: DownloadStatus }) {
  const { t } = useTranslation("library");
  if (status === "loaded") return <StatusIcon status="loaded" label={t("videoModels.status.loaded", { defaultValue: "Loaded in memory" })} />;
  if (status === "installed") return <StatusIcon status="installed" label={t("videoModels.status.installed", { defaultValue: "Installed" })} />;
  if (status === "downloading" && downloadState) return <StatusIcon status="downloading" label={t("videoModels.status.downloading", { defaultValue: "Downloading" })} detail={downloadProgressLabel(downloadState)} />;
  if (status === "paused" && downloadState) return <StatusIcon status="paused" label={t("videoModels.status.paused", { defaultValue: "Paused" })} detail={downloadProgressLabel(downloadState)} />;
  if (status === "failed") return <StatusIcon status="failed" label={t("videoModels.status.failed", { defaultValue: "Failed" })} detail={downloadState?.error ?? t("videoModels.status.downloadFailed", { defaultValue: "Download failed" })} />;
  return <StatusIcon status="incomplete" label={t("videoModels.status.incomplete", { defaultValue: "Incomplete" })} />;
}

function sortIndicator(activeSort: InstalledVideoSort, sortDir: SortDir, key: InstalledVideoSort): string {
  if (activeSort !== key) return "";
  return sortDir === "asc" ? " \u25B2" : " \u25BC";
}

function sortLabel(sort: InstalledVideoSort, sortDir: SortDir, t: (key: string, opts?: Record<string, unknown>) => string): string {
  const direction = sortDir === "asc"
    ? t("videoModels.sort.direction.ascending", { defaultValue: "ascending" })
    : t("videoModels.sort.direction.descending", { defaultValue: "descending" });
  if (sort === "provider") return t("videoModels.sort.providerDir", { defaultValue: "provider {direction}", direction });
  if (sort === "tasks") return t("videoModels.sort.tasksDir", { defaultValue: "tasks {direction}", direction });
  if (sort === "size") return sortDir === "desc"
    ? t("videoModels.sort.largestFirst", { defaultValue: "largest size first" })
    : t("videoModels.sort.smallestFirst", { defaultValue: "smallest size first" });
  if (sort === "ram") return sortDir === "desc"
    ? t("videoModels.sort.highestRam", { defaultValue: "highest RAM/VRAM first" })
    : t("videoModels.sort.lowestRam", { defaultValue: "lowest RAM/VRAM first" });
  if (sort === "status") return t("videoModels.sort.statusDir", { defaultValue: "status {direction}", direction });
  if (sort === "name") return sortDir === "asc"
    ? t("videoModels.sort.nameAsc", { defaultValue: "name A-Z" })
    : t("videoModels.sort.nameDesc", { defaultValue: "name Z-A" });
  return sortDir === "desc"
    ? t("videoModels.sort.newestFirst", { defaultValue: "newest released first" })
    : t("videoModels.sort.oldestFirst", { defaultValue: "oldest released first" });
}

function memoryParts(label: string | null | undefined, pendingLabel: string): { primary: string; secondary: string | null } {
  if (!label) return { primary: pendingLabel, secondary: null };
  const [primary, secondary] = label.split(" @ ");
  if (!secondary) return { primary, secondary: null };
  return { primary: `${primary} @`, secondary };
}

export function VideoModelsTab({
  installedVideoVariants,
  videoCatalog,
  activeVideoDownloads,
  videoRuntimeStatus,
  videoBusy,
  videoBusyLabel,
  loadedVideoVariant,
  fileRevealLabel,
  onActiveTabChange,
  onOpenVideoStudio,
  onVideoDownload,
  onCancelVideoDownload,
  onDeleteVideoDownload,
  onPreloadVideoModel,
  onUnloadVideoModel,
  onOpenExternalUrl,
  onRevealPath,
}: VideoModelsTabProps) {
  const { t } = useTranslation("library");
  const pendingMemoryLabel = t("videoModels.memory.pending", { defaultValue: "pending" });
  const [searchInput, setSearchInput] = useState("");
  const [taskFilter, setTaskFilter] = useState<"all" | VideoModelVariant["taskSupport"][number]>("all");
  const [statusFilter, setStatusFilter] = useState<InstalledVideoStatusFilter>("all");
  const [sort, setSort] = useState<InstalledVideoSort>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const normalizedSearch = searchInput.trim().toLowerCase();
  const hasActiveFilters =
    normalizedSearch.length > 0 || taskFilter !== "all" || statusFilter !== "all" || sort !== "date" || sortDir !== "desc";

  function applySort(nextSort: InstalledVideoSort) {
    if (sort === nextSort) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSort(nextSort);
      setSortDir(defaultSortDir(nextSort));
    }
  }

  const rows = useMemo(() => {
    return installedVideoVariants
      .map((variant) => {
        const family = videoCatalog.find((item) =>
          item.variants.some((candidate) => candidate.id === variant.id),
        );
        const downloadState = videoDownloadStatusForVariant(activeVideoDownloads, variant);
        const status = videoStatus(variant, downloadState, loadedVideoVariant);
        const memoryEstimate = videoDiscoverMemoryEstimate(variant);
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
  }, [activeVideoDownloads, installedVideoVariants, loadedVideoVariant, normalizedSearch, sort, sortDir, statusFilter, taskFilter, videoCatalog]);

  return (
    <div className="content-grid image-page-grid">
      <Panel
        title={t("common:panels.installedVideoModels", { defaultValue: "Installed Video Models" })}
        subtitle={installedVideoVariants.length > 0
          ? t("videoModels.subtitle.withLocal", {
              defaultValue: "{rows} of {total, plural, one {# model} other {# models}} with local data",
              rows: rows.length,
              total: installedVideoVariants.length,
            })
          : t("videoModels.subtitle.none", { defaultValue: "No video models detected locally yet" })}
        className="span-2"
        actions={
          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-discover")}>
            {t("videoModels.action.browseCatalog", { defaultValue: "Browse Catalog" })}
          </button>
        }
      >
        {installedVideoVariants.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>{t("videoModels.empty.noModels", { defaultValue: "Download a video model from Video Discover to get started." })}</p>
          </div>
        ) : (
          <>
            <div className="image-discover-filter-row image-discover-filter-row--wide image-model-filter-row">
              <label className="image-discover-search">
                {t("videoModels.filter.searchLabel", { defaultValue: "Search" })}
                <input
                  className="text-input"
                  type="search"
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                  placeholder={t("videoModels.filter.searchPlaceholder", { defaultValue: "Filter by model, provider, repo, task, or tag..." })}
                />
              </label>
              <label>
                {t("videoModels.filter.taskLabel", { defaultValue: "Task" })}
                <select
                  className="text-input"
                  value={taskFilter}
                  onChange={(event) => setTaskFilter(event.target.value as typeof taskFilter)}
                >
                  <option value="all">{t("videoModels.filter.task.all", { defaultValue: "All tasks" })}</option>
                  <option value="txt2video">{t("videoModels.filter.task.txt2video", { defaultValue: "Text to video" })}</option>
                  <option value="img2video">{t("videoModels.filter.task.img2video", { defaultValue: "Image to video" })}</option>
                  <option value="video2video">{t("videoModels.filter.task.video2video", { defaultValue: "Video to video" })}</option>
                </select>
              </label>
              <label>
                {t("videoModels.filter.statusLabel", { defaultValue: "Status" })}
                <select
                  className="text-input"
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value as InstalledVideoStatusFilter)}
                >
                  <option value="all">{t("videoModels.filter.status.all", { defaultValue: "Any status" })}</option>
                  <option value="loaded">{t("videoModels.filter.status.loaded", { defaultValue: "In memory" })}</option>
                  <option value="installed">{t("videoModels.filter.status.installed", { defaultValue: "Installed" })}</option>
                  <option value="incomplete">{t("videoModels.filter.status.incomplete", { defaultValue: "Incomplete" })}</option>
                  <option value="downloading">{t("videoModels.filter.status.downloading", { defaultValue: "Downloading" })}</option>
                  <option value="paused">{t("videoModels.filter.status.paused", { defaultValue: "Paused" })}</option>
                  <option value="failed">{t("videoModels.filter.status.failed", { defaultValue: "Failed" })}</option>
                </select>
              </label>
              <label>
                {t("videoModels.filter.sortByLabel", { defaultValue: "Sort by" })}
                <select
                  className="text-input"
                  value={sort}
                  onChange={(event) => {
                    const nextSort = event.target.value as InstalledVideoSort;
                    setSort(nextSort);
                    setSortDir(defaultSortDir(nextSort));
                  }}
                >
                  <option value="name">{t("videoModels.sort.name", { defaultValue: "Name" })}</option>
                  <option value="provider">{t("videoModels.sort.provider", { defaultValue: "Provider" })}</option>
                  <option value="tasks">{t("videoModels.sort.tasks", { defaultValue: "Tasks" })}</option>
                  <option value="date">{t("videoModels.sort.newestReleased", { defaultValue: "Newest released" })}</option>
                  <option value="size">{t("videoModels.sort.largestSize", { defaultValue: "Largest size" })}</option>
                  <option value="ram">{t("videoModels.sort.highestRamVram", { defaultValue: "Highest RAM/VRAM" })}</option>
                  <option value="status">{t("videoModels.sort.status", { defaultValue: "Status" })}</option>
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
                  {t("videoModels.action.clearFilters", { defaultValue: "Clear Filters" })}
                </button>
              </div>
            </div>
            <div className="image-discover-results-summary">
              <span>{t("videoModels.summary.results", {
                defaultValue: "{count, plural, one {# model} other {# models}} · {sortLabel}",
                count: rows.length,
                sortLabel: sortLabel(sort, sortDir, t),
              })}</span>
              {normalizedSearch ? <span className="badge subtle">{t("videoModels.summary.searchChip", { defaultValue: "Search: {query}", query: searchInput.trim() })}</span> : null}
              {taskFilter !== "all" ? <span className="badge muted">{t("videoModels.summary.taskChip", { defaultValue: "Task: {value}", value: taskFilter })}</span> : null}
              {statusFilter !== "all" ? <span className="badge muted">{t("videoModels.summary.statusChip", { defaultValue: "Status: {value}", value: statusFilter })}</span> : null}
            </div>
            {rows.length === 0 ? (
              <div className="empty-state image-empty-state">
                <p>{t("videoModels.empty.noMatches", { defaultValue: "No installed video models match the current filters." })}</p>
              </div>
            ) : (
              <div className="media-model-table media-model-table--video">
                <div className="media-model-head">
                  <button className="sort-header" type="button" onClick={() => applySort("name")}>{t("videoModels.column.model", { defaultValue: "Model" })}{sortIndicator(sort, sortDir, "name")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("provider")}>{t("videoModels.column.provider", { defaultValue: "Provider" })}{sortIndicator(sort, sortDir, "provider")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("tasks")}>{t("videoModels.column.tasks", { defaultValue: "Tasks" })}{sortIndicator(sort, sortDir, "tasks")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("size")}>{t("videoModels.column.size", { defaultValue: "Size" })}{sortIndicator(sort, sortDir, "size")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("ram")}>{t("videoModels.column.ramVram", { defaultValue: "RAM/VRAM" })}{sortIndicator(sort, sortDir, "ram")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("date")}>{t("videoModels.column.released", { defaultValue: "Released" })}{sortIndicator(sort, sortDir, "date")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("status")}>{t("videoModels.column.status", { defaultValue: "Status" })}{sortIndicator(sort, sortDir, "status")}</button>
                  <span className="sort-header"></span>
                </div>
                <div className="media-model-rows">
                  {rows.map(({ variant, family, downloadState, status, memoryEstimate }) => {
                    const isLoadedInMemory = status === "loaded";
                    const isComplete = status === "loaded" || status === "installed";
                    const isDownloading = status === "downloading";
                    const isPaused = status === "paused";
                    const isDownloadFailed = status === "failed";
                    const isPartial = status === "incomplete";
                    const canDeleteLocalData = Boolean(isComplete || isPaused || isDownloadFailed || isPartial);
                    const localStatusReason = !isComplete && !isDownloading ? variant.localStatusReason : null;
                    const secondarySize = videoSecondarySizeLabel(variant);
                    const releaseLabel = compactReleaseLabel(formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt));
                    const primarySizeLabel = videoPrimarySizeLabel(variant);
                    const sizeTitle = [primarySizeLabel, secondarySize].filter(Boolean).join(" / ");
                    const memory = memoryParts(memoryEstimate?.label, pendingMemoryLabel);
                    const deleteRepo = videoDeleteRepoForVariant(variant, downloadState);
                    const deleteLabel = isDownloading
                      ? t("videoModels.action.cancelDownload", { defaultValue: "Cancel download" })
                      : videoDeleteLabelForRepo(variant, deleteRepo, t("videoModels.action.deleteModel", { defaultValue: "Delete model" }));
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
                          <span className="media-model-memory" title={memoryEstimate?.title ?? t("videoModels.memory.pendingTitle", { defaultValue: "RAM/VRAM estimate pending until model weight size is known." })}>
                            <span>{memory.primary}</span>
                            {memory.secondary ? <small>{memory.secondary}</small> : null}
                          </span>
                          <span>{releaseLabel ?? t("videoModels.releaseUnknown", { defaultValue: "Unknown" })}</span>
                          <span><StatusBadge status={status} downloadState={downloadState} /></span>
                          <div className="media-model-actions">
                            {isComplete ? (
                              <IconActionButton icon="generate" label={t("videoModels.action.generate", { defaultValue: "Generate" })} buttonStyle="primary" onClick={() => onOpenVideoStudio(variant.id)} />
                            ) : isDownloading ? (
                              <IconActionButton icon="pause" label={t("videoModels.action.pauseDownload", { defaultValue: "Pause download" })} onClick={() => onCancelVideoDownload(downloadState?.repo ?? variant.repo)} />
                            ) : (
                              <IconActionButton icon={isDownloadFailed ? "retry" : isPartial ? "resume" : "download"} label={isDownloadFailed
                                ? t("videoModels.action.retryDownload", { defaultValue: "Retry download" })
                                : isPartial
                                  ? t("videoModels.action.resumeDownload", { defaultValue: "Resume download" })
                                  : t("videoModels.action.downloadModel", { defaultValue: "Download model" })} onClick={() => onVideoDownload(variant.repo, variant.id)} />
                            )}
                            {isDownloading || canDeleteLocalData ? (
                              <IconActionButton icon={isDownloading ? "cancel" : "delete"} label={deleteLabel} danger onClick={() => onDeleteVideoDownload(deleteRepo)} />
                            ) : null}
                            {variant.localPath ? (
                              <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(variant.localPath as string)} />
                            ) : null}
                            <IconActionButton icon="modelCard" label={t("videoModels.action.openModelCard", { defaultValue: "Open model card" })} onClick={() => onOpenExternalUrl(variant.link)} />
                          </div>
                        </div>
                        {isDownloadFailed && downloadState?.error ? (
                          <div className="media-model-row-detail callout error">
                            <p>{downloadState.error}</p>
                          </div>
                        ) : localStatusReason ? (
                          <div className="media-model-row-detail callout quiet">
                            <p>{localStatusReason}</p>
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
