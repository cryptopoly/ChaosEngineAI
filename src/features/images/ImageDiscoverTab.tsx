import { useMemo, useState } from "react";
import { Panel } from "../../components/Panel";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import type { DownloadStatus } from "../../api";
import type {
  ImageModelVariant,
  TabId,
} from "../../types";
import type {
  DiscoverSort,
  ImageDiscoverTaskFilter,
  ImageDiscoverAccessFilter,
} from "../../types/image";
import {
  compactModelSizeLabel,
  compactReleaseLabel,
  downloadProgressLabel,
  downloadSizeTooltip,
  formatImageAccessError,
  formatImageLicenseLabel,
  formatReleaseLabel,
  imageDiscoverMemoryEstimate,
  imagePrimarySizeLabel,
  imageSecondarySizeLabel,
  isGatedImageAccessError,
} from "../../utils";

type MediaStatusFilter = "all" | "installed" | "not-installed" | "downloading" | "paused" | "failed" | "incomplete";
type SortDir = "asc" | "desc";

export interface ImageDiscoverTabProps {
  combinedImageDiscoverResults: ImageModelVariant[];
  imageDiscoverSearchInput: string;
  onImageDiscoverSearchInputChange: (value: string) => void;
  imageDiscoverTaskFilter: ImageDiscoverTaskFilter;
  onImageDiscoverTaskFilterChange: (value: ImageDiscoverTaskFilter) => void;
  imageDiscoverAccessFilter: ImageDiscoverAccessFilter;
  onImageDiscoverAccessFilterChange: (value: ImageDiscoverAccessFilter) => void;
  imageDiscoverSort: DiscoverSort;
  onImageDiscoverSortChange: (value: DiscoverSort) => void;
  imageDiscoverHasActiveFilters: boolean;
  imageDiscoverSearchQuery: string;
  activeImageDownloads: Record<string, DownloadStatus>;
  selectedImageVariant: ImageModelVariant | null;
  fileRevealLabel: string;
  onActiveTabChange: (tab: TabId) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
  onDeleteImageDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
}

function imageDiscoverSortLabel(sort: DiscoverSort): string {
  if (sort === "name") return "name";
  if (sort === "provider") return "provider";
  if (sort === "tasks") return "tasks";
  if (sort === "size") return "largest size first";
  if (sort === "ram") return "highest RAM/VRAM first";
  if (sort === "likes") return "most liked first";
  if (sort === "downloads") return "most downloads first";
  if (sort === "status") return "status";
  return "newest released first";
}

function sortIndicator(activeSort: DiscoverSort, sortDir: SortDir, key: DiscoverSort): string {
  if (activeSort !== key) return "";
  return sortDir === "asc" ? " \u25B2" : " \u25BC";
}

function defaultSortDir(sort: DiscoverSort): SortDir {
  return sort === "name" || sort === "provider" || sort === "tasks" ? "asc" : "desc";
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

function statusSortKey(status: MediaStatusFilter): number {
  if (status === "installed") return 0;
  if (status === "downloading") return 1;
  if (status === "paused") return 2;
  if (status === "failed") return 3;
  if (status === "incomplete") return 4;
  if (status === "not-installed") return 5;
  return 6;
}

function memoryParts(label: string | null | undefined): { primary: string; secondary: string | null } {
  if (!label) return { primary: "pending", secondary: null };
  const [primary, secondary] = label.split(" @ ");
  if (!secondary) return { primary, secondary: null };
  return { primary: `${primary} @`, secondary };
}

function imageVariantStatus(
  variant: ImageModelVariant,
  downloadState?: DownloadStatus,
): MediaStatusFilter {
  if (variant.availableLocally || downloadState?.state === "completed") return "installed";
  if (downloadState?.state === "downloading") return "downloading";
  if (downloadState?.state === "cancelled") return "paused";
  if (downloadState?.state === "failed") return "failed";
  if (variant.hasLocalData) return "incomplete";
  return "not-installed";
}

function statusBadge(status: MediaStatusFilter, downloadState?: DownloadStatus) {
  const downloadDetail = downloadState
    ? [downloadProgressLabel(downloadState), downloadSizeTooltip(downloadState)].filter(Boolean).join(" / ")
    : null;
  if (status === "installed") return <StatusIcon status="installed" label="Installed" />;
  if (status === "downloading" && downloadState) {
    return <StatusIcon status="downloading" label="Downloading" detail={downloadDetail} />;
  }
  if (status === "paused" && downloadState) {
    return <StatusIcon status="paused" label="Paused" detail={downloadDetail} />;
  }
  if (status === "failed") return <StatusIcon status="failed" label="Failed" detail={downloadState?.error ?? "Download failed"} />;
  if (status === "incomplete") return <StatusIcon status="incomplete" label="Incomplete" />;
  return <StatusIcon status="incomplete" label="Not installed" />;
}

export function ImageDiscoverTab({
  combinedImageDiscoverResults,
  imageDiscoverSearchInput,
  onImageDiscoverSearchInputChange,
  imageDiscoverTaskFilter,
  onImageDiscoverTaskFilterChange,
  imageDiscoverAccessFilter,
  onImageDiscoverAccessFilterChange,
  imageDiscoverSort,
  onImageDiscoverSortChange,
  imageDiscoverHasActiveFilters,
  imageDiscoverSearchQuery,
  activeImageDownloads,
  selectedImageVariant,
  fileRevealLabel,
  onActiveTabChange,
  onOpenImageStudio,
  onImageDownload,
  onCancelImageDownload,
  onDeleteImageDownload,
  onOpenExternalUrl,
  onRevealPath,
}: ImageDiscoverTabProps) {
  const [statusFilter, setStatusFilter] = useState<MediaStatusFilter>("all");
  const [sortDir, setSortDir] = useState<SortDir>(defaultSortDir(imageDiscoverSort));
  const filteredResults = useMemo(
    () =>
      combinedImageDiscoverResults
        .map((variant) => {
          const downloadState = activeImageDownloads[variant.repo];
          const status = imageVariantStatus(variant, downloadState);
          const memoryEstimate = imageDiscoverMemoryEstimate(variant);
          return { variant, status, memoryEstimate };
        })
        .filter(({ status }) => statusFilter === "all" || status === statusFilter)
        .sort((left, right) => {
          if (imageDiscoverSort === "name") {
            const diff = left.variant.name.localeCompare(right.variant.name);
            return sortDir === "asc" ? diff : -diff;
          }
          if (imageDiscoverSort === "provider") {
            const diff = left.variant.provider.localeCompare(right.variant.provider);
            if (diff !== 0) return sortDir === "asc" ? diff : -diff;
          }
          if (imageDiscoverSort === "tasks") {
            const diff = left.variant.taskSupport.join(" ").localeCompare(right.variant.taskSupport.join(" "));
            if (diff !== 0) return sortDir === "asc" ? diff : -diff;
          }
          if (imageDiscoverSort === "size") {
            const diff = compareNullableNumber(sizeSortKey(left.variant), sizeSortKey(right.variant), sortDir);
            if (diff !== 0) return diff;
          } else if (imageDiscoverSort === "ram") {
            const diff = compareNullableNumber(left.memoryEstimate?.estimatedPeakGb ?? null, right.memoryEstimate?.estimatedPeakGb ?? null, sortDir);
            if (diff !== 0) return diff;
          } else if (imageDiscoverSort === "status") {
            const diff = statusSortKey(left.status) - statusSortKey(right.status);
            if (diff !== 0) return sortDir === "asc" ? diff : -diff;
          } else if (imageDiscoverSort === "likes") {
            const diff = compareNullableNumber(left.variant.likes ?? null, right.variant.likes ?? null, sortDir);
            if (diff !== 0) return diff;
          } else if (imageDiscoverSort === "downloads") {
            const diff = compareNullableNumber(left.variant.downloads ?? null, right.variant.downloads ?? null, sortDir);
            if (diff !== 0) return diff;
          }
          const dateDiff = releaseSortKey(right.variant).localeCompare(releaseSortKey(left.variant));
          if (dateDiff !== 0) return sortDir === "desc" ? dateDiff : -dateDiff;
          return left.variant.name.localeCompare(right.variant.name);
        }),
    [activeImageDownloads, combinedImageDiscoverResults, imageDiscoverSort, sortDir, statusFilter],
  );
  const hasActiveFilters = imageDiscoverHasActiveFilters || statusFilter !== "all";

  function applySort(nextSort: DiscoverSort) {
    if (imageDiscoverSort === nextSort) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      onImageDiscoverSortChange(nextSort);
      setSortDir(defaultSortDir(nextSort));
    }
  }

  return (
    <div className="image-discover-stack">
      <Panel
        title="Image Discover"
        subtitle={`${filteredResults.length} of ${combinedImageDiscoverResults.length} models / live Hugging Face metadata`}
      >
        <div className="image-hero">
          <div>
            <h3>Browse and download image models for local generation.</h3>
            <p className="muted-text">
              Download any model to use it in Image Studio. Runtime status lives in the Studio tab.
            </p>
          </div>
          <div className="image-hero-actions">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-models")}>
              Installed Models
            </button>
            <button className="primary-button" type="button" onClick={() => onOpenImageStudio(selectedImageVariant?.id)}>
              Open Studio
            </button>
          </div>
        </div>

        <div className="image-discover-filter-row image-discover-filter-row--wide">
          <label className="image-discover-search">
            Search
            <input
              className="text-input"
              type="search"
              value={imageDiscoverSearchInput}
              onChange={(event) => onImageDiscoverSearchInputChange(event.target.value)}
              placeholder="Search FLUX, SDXL, provider, task, tags, license..."
            />
          </label>
          <label>
            Task
            <select
              className="text-input"
              value={imageDiscoverTaskFilter}
              onChange={(event) => onImageDiscoverTaskFilterChange(event.target.value as ImageDiscoverTaskFilter)}
            >
              <option value="all">All tasks</option>
              <option value="txt2img">Text to image</option>
              <option value="img2img">Image to image</option>
              <option value="inpaint">Inpaint</option>
            </select>
          </label>
          <label>
            Access
            <select
              className="text-input"
              value={imageDiscoverAccessFilter}
              onChange={(event) => onImageDiscoverAccessFilterChange(event.target.value as ImageDiscoverAccessFilter)}
            >
              <option value="all">Open + gated</option>
              <option value="open">Open only</option>
              <option value="gated">Gated only</option>
            </select>
          </label>
          <label>
            Status
            <select
              className="text-input"
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value as MediaStatusFilter)}
            >
              <option value="all">Any status</option>
              <option value="installed">Installed</option>
              <option value="not-installed">Not installed</option>
              <option value="downloading">Downloading</option>
              <option value="paused">Paused</option>
              <option value="failed">Failed</option>
              <option value="incomplete">Incomplete</option>
            </select>
          </label>
          <label>
            Sort by
            <select
              className="text-input"
              value={imageDiscoverSort}
              onChange={(event) => {
                const nextSort = event.target.value as DiscoverSort;
                onImageDiscoverSortChange(nextSort);
                setSortDir(defaultSortDir(nextSort));
              }}
            >
              <option value="name">Name</option>
              <option value="provider">Provider</option>
              <option value="tasks">Tasks</option>
              <option value="release">Newest released</option>
              <option value="size">Largest size</option>
              <option value="ram">Highest RAM/VRAM</option>
              <option value="likes">Most likes</option>
              <option value="downloads">Most downloads</option>
              <option value="status">Status</option>
            </select>
          </label>
          <div className="image-discover-filter-actions">
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                onImageDiscoverSearchInputChange("");
                onImageDiscoverTaskFilterChange("all");
                onImageDiscoverAccessFilterChange("all");
                setStatusFilter("all");
                onImageDiscoverSortChange("release");
                setSortDir("desc");
              }}
              disabled={!hasActiveFilters}
            >
              Clear Filters
            </button>
          </div>
        </div>

        <div className="image-discover-results-summary">
          <span>
            {filteredResults.length} model{filteredResults.length !== 1 ? "s" : ""} · {imageDiscoverSortLabel(imageDiscoverSort)}
          </span>
          {imageDiscoverSearchQuery ? (
            <span className="badge subtle">Search: {imageDiscoverSearchInput.trim()}</span>
          ) : null}
          {imageDiscoverTaskFilter !== "all" ? (
            <span className="badge muted">Task: {imageDiscoverTaskFilter}</span>
          ) : null}
          {imageDiscoverAccessFilter !== "all" ? (
            <span className="badge muted">
              Access: {imageDiscoverAccessFilter === "open" ? "Open only" : "Gated only"}
            </span>
          ) : null}
          {statusFilter !== "all" ? <span className="badge muted">Status: {statusFilter}</span> : null}
        </div>
      </Panel>

      {filteredResults.length === 0 ? (
        <Panel title="Image Models" subtitle="No models match the current filters" className="image-discover-section-panel">
          <div className="empty-state image-empty-state">
            <p>Try broadening the filters or search terms.</p>
          </div>
        </Panel>
      ) : (
        <div className="media-model-table media-model-table--image">
          <div className="media-model-head">
            <button className="sort-header" type="button" onClick={() => applySort("name")}>Model{sortIndicator(imageDiscoverSort, sortDir, "name")}</button>
            <button className="sort-header" type="button" onClick={() => applySort("provider")}>Provider{sortIndicator(imageDiscoverSort, sortDir, "provider")}</button>
            <button className="sort-header" type="button" onClick={() => applySort("tasks")}>Tasks{sortIndicator(imageDiscoverSort, sortDir, "tasks")}</button>
            <button className="sort-header" type="button" onClick={() => applySort("size")}>
              Size{sortIndicator(imageDiscoverSort, sortDir, "size")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("ram")}>
              RAM/VRAM{sortIndicator(imageDiscoverSort, sortDir, "ram")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("release")}>
              Released{sortIndicator(imageDiscoverSort, sortDir, "release")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("status")}>Status{sortIndicator(imageDiscoverSort, sortDir, "status")}</button>
            <span className="sort-header"></span>
          </div>
          <div className="media-model-rows">
            {filteredResults.map(({ variant, status, memoryEstimate }) => {
              const downloadState = activeImageDownloads[variant.repo];
              const isComplete = status === "installed";
              const isDownloading = status === "downloading";
              const isPaused = status === "paused";
              const isDownloadFailed = status === "failed";
              const isPartial = status === "incomplete";
              const isDownloadComplete = downloadState?.state === "completed";
              const hasLocalData = Boolean(variant.hasLocalData || isDownloadComplete || isPaused || isDownloadFailed);
              const friendlyDownloadError = formatImageAccessError(downloadState?.error, variant);
              const needsGatedAccess = isGatedImageAccessError(downloadState?.error);
              const secondarySize = imageSecondarySizeLabel(variant);
              const releaseLabel = compactReleaseLabel(formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt));
              const primarySizeLabel = imagePrimarySizeLabel(variant);
              const sizeTitle = [primarySizeLabel, secondarySize].filter(Boolean).join(" / ");
              const memory = memoryParts(memoryEstimate?.label);
              return (
                <div key={variant.id} className={`media-model-row-wrap${isComplete ? " downloaded" : ""}`}>
                  <div className="media-model-row">
                    <div className="media-model-name">
                      <strong>{variant.name}</strong>
                      <small>{variant.note}</small>
                      <div className="media-model-chip-row">
                        {variant.styleTags.slice(0, 4).map((tag) => (
                          <span key={tag} className="badge subtle">{tag}</span>
                        ))}
                        {typeof variant.gated === "boolean" ? (
                          <span className="badge muted">{variant.gated ? "Gated" : "Open"}</span>
                        ) : null}
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
                    <span>
                      {releaseLabel ?? "Unknown"}
                      {variant.downloadsLabel ? <small>{variant.downloadsLabel}</small> : null}
                      {variant.likesLabel ? <small>{variant.likesLabel}</small> : null}
                      {variant.license ? <small>{formatImageLicenseLabel(variant.license)}</small> : null}
                    </span>
                    <span>{statusBadge(status, downloadState)}</span>
                    <div className="media-model-actions">
                      {isComplete ? (
                        <IconActionButton icon="generate" label="Generate" buttonStyle="primary" onClick={() => onOpenImageStudio(variant.id)} />
                      ) : isDownloading ? (
                        <>
                          <IconActionButton icon="pause" label="Pause download" onClick={() => onCancelImageDownload(variant.repo)} />
                          <IconActionButton icon="cancel" label="Cancel download" danger onClick={() => onDeleteImageDownload(variant.repo)} />
                        </>
                      ) : isPaused ? (
                        <>
                          <IconActionButton icon="resume" label="Resume download" onClick={() => onImageDownload(variant.repo)} />
                          <IconActionButton icon="delete" label="Delete download" danger onClick={() => onDeleteImageDownload(variant.repo)} />
                        </>
                      ) : isDownloadFailed ? (
                        <>
                          <IconActionButton icon="retry" label="Retry download" onClick={() => onImageDownload(variant.repo)} />
                          <IconActionButton icon="delete" label="Delete download" danger onClick={() => onDeleteImageDownload(variant.repo)} />
                        </>
                      ) : (
                        <>
                          <IconActionButton icon={isPartial ? "resume" : "download"} label={isPartial ? "Resume download" : "Download model"} onClick={() => onImageDownload(variant.repo)} />
                          {hasLocalData ? (
                            <IconActionButton icon="delete" label="Delete model" danger onClick={() => onDeleteImageDownload(variant.repo)} />
                          ) : null}
                        </>
                      )}
                      {variant.localPath ? (
                        <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(variant.localPath as string)} />
                      ) : null}
                      <IconActionButton icon="huggingFace" label="Open on Hugging Face" onClick={() => onOpenExternalUrl(variant.link)} />
                    </div>
                  </div>
                  {isDownloadFailed && downloadState?.error ? (
                    <div className="media-model-row-detail callout error">
                      <p>{friendlyDownloadError}</p>
                      {needsGatedAccess ? (
                        <div className="button-row">
                          <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(variant.link)}>
                            Hugging Face
                          </button>
                          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("settings")}>
                            Settings
                          </button>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
