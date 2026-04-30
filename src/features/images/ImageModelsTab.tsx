import { useMemo, useState } from "react";
import { Panel } from "../../components/Panel";
import type { DownloadStatus } from "../../api";
import type {
  ImageModelFamily,
  ImageModelVariant,
  TabId,
} from "../../types";
import {
  downloadProgressLabel,
  formatReleaseLabel,
  imageDiscoverMemoryEstimate,
  imagePrimarySizeLabel,
  imageSecondarySizeLabel,
} from "../../utils";

type InstalledImageSort = "date" | "size" | "ram" | "name";
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

function imageStatus(variant: ImageModelVariant, downloadState?: DownloadStatus): InstalledImageStatusFilter {
  if (downloadState?.state === "downloading") return "downloading";
  if (downloadState?.state === "cancelled") return "paused";
  if (downloadState?.state === "failed") return "failed";
  if (variant.availableLocally || downloadState?.state === "completed") return "installed";
  return "incomplete";
}

function statusBadge(status: InstalledImageStatusFilter, downloadState?: DownloadStatus) {
  if (status === "installed") return <span className="badge success">Installed</span>;
  if (status === "downloading" && downloadState) return <span className="badge accent">{downloadProgressLabel(downloadState)}</span>;
  if (status === "paused" && downloadState) return <span className="badge warning">{downloadProgressLabel(downloadState)}</span>;
  if (status === "failed") return <span className="badge warning">Download Failed</span>;
  return <span className="badge warning">Incomplete</span>;
}

function sortIndicator(activeSort: InstalledImageSort, key: InstalledImageSort): string {
  return activeSort === key ? " \u25BC" : "";
}

function sortLabel(sort: InstalledImageSort): string {
  if (sort === "size") return "largest size first";
  if (sort === "ram") return "highest RAM/VRAM first";
  if (sort === "name") return "name A-Z";
  return "newest released first";
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
  const [searchInput, setSearchInput] = useState("");
  const [taskFilter, setTaskFilter] = useState<"all" | ImageModelVariant["taskSupport"][number]>("all");
  const [statusFilter, setStatusFilter] = useState<InstalledImageStatusFilter>("all");
  const [sort, setSort] = useState<InstalledImageSort>("date");
  const normalizedSearch = searchInput.trim().toLowerCase();
  const hasActiveFilters =
    normalizedSearch.length > 0 || taskFilter !== "all" || statusFilter !== "all" || sort !== "date";

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
        if (sort === "name") return left.variant.name.localeCompare(right.variant.name);
        if (sort === "size") {
          const diff = compareNullableNumberDesc(sizeSortKey(left.variant), sizeSortKey(right.variant));
          if (diff !== 0) return diff;
        } else if (sort === "ram") {
          const diff = compareNullableNumberDesc(left.memoryEstimate?.estimatedPeakGb ?? null, right.memoryEstimate?.estimatedPeakGb ?? null);
          if (diff !== 0) return diff;
        }
        const dateDiff = releaseSortKey(right.variant).localeCompare(releaseSortKey(left.variant));
        if (dateDiff !== 0) return dateDiff;
        return left.variant.name.localeCompare(right.variant.name);
      });
  }, [activeImageDownloads, imageCatalog, installedImageVariants, normalizedSearch, sort, statusFilter, taskFilter]);

  return (
    <div className="content-grid image-page-grid">
      <Panel
        title="Installed Image Models"
        subtitle={installedImageVariants.length > 0
          ? `${rows.length} of ${installedImageVariants.length} model${installedImageVariants.length !== 1 ? "s" : ""} with local data`
          : "No image models detected locally yet"}
        className="span-2"
        actions={
          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-discover")}>
            Browse Catalog
          </button>
        }
      >
        {installedImageVariants.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>Download an image model from Image Discover to get started.</p>
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
                  <option value="txt2img">Text to image</option>
                  <option value="img2img">Image to image</option>
                  <option value="inpaint">Inpaint</option>
                </select>
              </label>
              <label>
                Status
                <select
                  className="text-input"
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value as InstalledImageStatusFilter)}
                >
                  <option value="all">Any status</option>
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
                  onChange={(event) => setSort(event.target.value as InstalledImageSort)}
                >
                  <option value="date">Newest released</option>
                  <option value="size">Largest size</option>
                  <option value="ram">Highest RAM/VRAM</option>
                  <option value="name">Name A-Z</option>
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
                  }}
                  disabled={!hasActiveFilters}
                >
                  Clear Filters
                </button>
              </div>
            </div>
            <div className="image-discover-results-summary">
              <span>{rows.length} model{rows.length !== 1 ? "s" : ""} · {sortLabel(sort)}</span>
              {normalizedSearch ? <span className="badge subtle">Search: {searchInput.trim()}</span> : null}
              {taskFilter !== "all" ? <span className="badge muted">Task: {taskFilter}</span> : null}
              {statusFilter !== "all" ? <span className="badge muted">Status: {statusFilter}</span> : null}
            </div>
            {rows.length === 0 ? (
              <div className="empty-state image-empty-state">
                <p>No installed image models match the current filters.</p>
              </div>
            ) : (
              <div className="media-model-table media-model-table--image">
                <div className="media-model-head">
                  <button className="sort-header" type="button" onClick={() => setSort("name")}>Model{sortIndicator(sort, "name")}</button>
                  <span className="sort-header">Provider</span>
                  <span className="sort-header">Tasks</span>
                  <button className="sort-header" type="button" onClick={() => setSort("size")}>Size{sortIndicator(sort, "size")}</button>
                  <button className="sort-header" type="button" onClick={() => setSort("ram")}>RAM/VRAM{sortIndicator(sort, "ram")}</button>
                  <span className="sort-header">Spec</span>
                  <button className="sort-header" type="button" onClick={() => setSort("date")}>Date{sortIndicator(sort, "date")}</button>
                  <span className="sort-header">Status</span>
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
                    const releaseLabel = formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt);
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
                          <span title={secondarySize ?? undefined}>
                            {imagePrimarySizeLabel(variant)}
                            {secondarySize ? <small>{secondarySize}</small> : null}
                          </span>
                          <span title={memoryEstimate?.title ?? "RAM/VRAM estimate pending until model weight size is known."}>
                            {memoryEstimate?.label ?? "pending"}
                          </span>
                          <span>{variant.recommendedResolution}</span>
                          <span>{releaseLabel ?? "Unknown"}</span>
                          <span>{statusBadge(status, downloadState)}</span>
                          <div className="media-model-actions">
                            {isComplete ? (
                              <button className="primary-button" type="button" onClick={() => onOpenImageStudio(variant.id)}>
                                Generate
                              </button>
                            ) : isDownloading ? (
                              <button className="secondary-button" type="button" onClick={() => onCancelImageDownload(variant.repo)}>
                                Pause
                              </button>
                            ) : (
                              <button className="secondary-button" type="button" onClick={() => onImageDownload(variant.repo)}>
                                {isDownloadFailed ? "Retry" : isPartial ? "Resume" : "Download"}
                              </button>
                            )}
                            {isDownloading || canDeleteLocalData ? (
                              <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(variant.repo)}>
                                {isDownloading ? "Cancel" : "Delete"}
                              </button>
                            ) : null}
                            {variant.localPath ? (
                              <button
                                className="secondary-button icon-button"
                                type="button"
                                title={fileRevealLabel}
                                onClick={() => onRevealPath(variant.localPath as string)}
                              >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                                  <polyline points="15 3 21 3 21 9" />
                                  <line x1="10" y1="14" x2="21" y2="3" />
                                </svg>
                              </button>
                            ) : null}
                            <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(variant.link)}>
                              Model Card
                            </button>
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
