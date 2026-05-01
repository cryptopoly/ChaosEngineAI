import { useMemo, useState } from "react";
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

function statusBadge(status: InstalledVideoStatusFilter, downloadState?: DownloadStatus) {
  if (status === "loaded") return <StatusIcon status="loaded" label="Loaded in memory" />;
  if (status === "installed") return <StatusIcon status="installed" label="Installed" />;
  if (status === "downloading" && downloadState) return <StatusIcon status="downloading" label="Downloading" detail={downloadProgressLabel(downloadState)} />;
  if (status === "paused" && downloadState) return <StatusIcon status="paused" label="Paused" detail={downloadProgressLabel(downloadState)} />;
  if (status === "failed") return <StatusIcon status="failed" label="Failed" detail={downloadState?.error ?? "Download failed"} />;
  return <StatusIcon status="incomplete" label="Incomplete" />;
}

function sortIndicator(activeSort: InstalledVideoSort, sortDir: SortDir, key: InstalledVideoSort): string {
  if (activeSort !== key) return "";
  return sortDir === "asc" ? " \u25B2" : " \u25BC";
}

function sortLabel(sort: InstalledVideoSort, sortDir: SortDir): string {
  const direction = sortDir === "asc" ? "ascending" : "descending";
  if (sort === "provider") return `provider ${direction}`;
  if (sort === "tasks") return `tasks ${direction}`;
  if (sort === "size") return sortDir === "desc" ? "largest size first" : "smallest size first";
  if (sort === "ram") return sortDir === "desc" ? "highest RAM/VRAM first" : "lowest RAM/VRAM first";
  if (sort === "status") return `status ${direction}`;
  if (sort === "name") return sortDir === "asc" ? "name A-Z" : "name Z-A";
  return sortDir === "desc" ? "newest released first" : "oldest released first";
}

function memoryParts(label: string | null | undefined): { primary: string; secondary: string | null } {
  if (!label) return { primary: "pending", secondary: null };
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
        title="Installed Video Models"
        subtitle={installedVideoVariants.length > 0
          ? `${rows.length} of ${installedVideoVariants.length} model${installedVideoVariants.length !== 1 ? "s" : ""} with local data`
          : "No video models detected locally yet"}
        className="span-2"
        actions={
          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-discover")}>
            Browse Catalog
          </button>
        }
      >
        {installedVideoVariants.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>Download a video model from Video Discover to get started.</p>
          </div>
        ) : (
          <>
            <div className="image-discover-filter-row image-discover-filter-row--wide image-model-filter-row">
              <label className="image-discover-search">
                Search
                <input
                  className="text-input"
                  type="search"
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                  placeholder="Filter by model, provider, repo, task, or tag..."
                />
              </label>
              <label>
                Task
                <select
                  className="text-input"
                  value={taskFilter}
                  onChange={(event) => setTaskFilter(event.target.value as typeof taskFilter)}
                >
                  <option value="all">All tasks</option>
                  <option value="txt2video">Text to video</option>
                  <option value="img2video">Image to video</option>
                  <option value="video2video">Video to video</option>
                </select>
              </label>
              <label>
                Status
                <select
                  className="text-input"
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value as InstalledVideoStatusFilter)}
                >
                  <option value="all">Any status</option>
                  <option value="loaded">In memory</option>
                  <option value="installed">Installed</option>
                  <option value="incomplete">Incomplete</option>
                  <option value="downloading">Downloading</option>
                  <option value="paused">Paused</option>
                  <option value="failed">Failed</option>
                </select>
              </label>
              <label>
                Sort by
                <select
                  className="text-input"
                  value={sort}
                  onChange={(event) => {
                    const nextSort = event.target.value as InstalledVideoSort;
                    setSort(nextSort);
                    setSortDir(defaultSortDir(nextSort));
                  }}
                >
                  <option value="name">Name</option>
                  <option value="provider">Provider</option>
                  <option value="tasks">Tasks</option>
                  <option value="date">Newest released</option>
                  <option value="size">Largest size</option>
                  <option value="ram">Highest RAM/VRAM</option>
                  <option value="status">Status</option>
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
                  Clear Filters
                </button>
              </div>
            </div>
            <div className="image-discover-results-summary">
              <span>{rows.length} model{rows.length !== 1 ? "s" : ""} · {sortLabel(sort, sortDir)}</span>
              {normalizedSearch ? <span className="badge subtle">Search: {searchInput.trim()}</span> : null}
              {taskFilter !== "all" ? <span className="badge muted">Task: {taskFilter}</span> : null}
              {statusFilter !== "all" ? <span className="badge muted">Status: {statusFilter}</span> : null}
            </div>
            {rows.length === 0 ? (
              <div className="empty-state image-empty-state">
                <p>No installed video models match the current filters.</p>
              </div>
            ) : (
              <div className="media-model-table media-model-table--video">
                <div className="media-model-head">
                  <button className="sort-header" type="button" onClick={() => applySort("name")}>Model{sortIndicator(sort, sortDir, "name")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("provider")}>Provider{sortIndicator(sort, sortDir, "provider")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("tasks")}>Tasks{sortIndicator(sort, sortDir, "tasks")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("size")}>Size{sortIndicator(sort, sortDir, "size")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("ram")}>RAM/VRAM{sortIndicator(sort, sortDir, "ram")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("date")}>Released{sortIndicator(sort, sortDir, "date")}</button>
                  <button className="sort-header" type="button" onClick={() => applySort("status")}>Status{sortIndicator(sort, sortDir, "status")}</button>
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
                    const memory = memoryParts(memoryEstimate?.label);
                    const deleteRepo = videoDeleteRepoForVariant(variant, downloadState);
                    const deleteLabel = isDownloading
                      ? "Cancel download"
                      : videoDeleteLabelForRepo(variant, deleteRepo, "Delete model");
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
                          <span className="media-model-memory" title={memoryEstimate?.title ?? "RAM/VRAM estimate pending until model weight size is known."}>
                            <span>{memory.primary}</span>
                            {memory.secondary ? <small>{memory.secondary}</small> : null}
                          </span>
                          <span>{releaseLabel ?? "Unknown"}</span>
                          <span>{statusBadge(status, downloadState)}</span>
                          <div className="media-model-actions">
                            {isComplete ? (
                              <IconActionButton icon="generate" label="Generate" buttonStyle="primary" onClick={() => onOpenVideoStudio(variant.id)} />
                            ) : isDownloading ? (
                              <IconActionButton icon="pause" label="Pause download" onClick={() => onCancelVideoDownload(downloadState?.repo ?? variant.repo)} />
                            ) : (
                              <IconActionButton icon={isDownloadFailed ? "retry" : isPartial ? "resume" : "download"} label={isDownloadFailed ? "Retry download" : isPartial ? "Resume download" : "Download model"} onClick={() => onVideoDownload(variant.repo, variant.id)} />
                            )}
                            {isDownloading || canDeleteLocalData ? (
                              <IconActionButton icon={isDownloading ? "cancel" : "delete"} label={deleteLabel} danger onClick={() => onDeleteVideoDownload(deleteRepo)} />
                            ) : null}
                            {variant.localPath ? (
                              <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(variant.localPath as string)} />
                            ) : null}
                            <IconActionButton icon="modelCard" label="Open model card" onClick={() => onOpenExternalUrl(variant.link)} />
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
