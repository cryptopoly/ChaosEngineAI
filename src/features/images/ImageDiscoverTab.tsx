import { useMemo, useState } from "react";
import { Panel } from "../../components/Panel";
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
  if (sort === "size") return "largest size first";
  if (sort === "ram") return "highest RAM/VRAM first";
  if (sort === "likes") return "most liked first";
  if (sort === "downloads") return "most downloads first";
  return "newest released first";
}

function sortIndicator(activeSort: DiscoverSort, key: DiscoverSort): string {
  return activeSort === key ? " \u25BC" : "";
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
  if (status === "installed") return <span className="badge success">Installed</span>;
  if (status === "downloading" && downloadState) {
    return <span className="badge accent" title={downloadSizeTooltip(downloadState)}>{downloadProgressLabel(downloadState)}</span>;
  }
  if (status === "paused" && downloadState) {
    return <span className="badge warning" title={downloadSizeTooltip(downloadState)}>{downloadProgressLabel(downloadState)}</span>;
  }
  if (status === "failed") return <span className="badge warning">Download Failed</span>;
  if (status === "incomplete") return <span className="badge warning">Incomplete</span>;
  return <span className="badge subtle">Not installed</span>;
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
  const filteredResults = useMemo(
    () =>
      combinedImageDiscoverResults.filter((variant) => {
        if (statusFilter === "all") return true;
        return imageVariantStatus(variant, activeImageDownloads[variant.repo]) === statusFilter;
      }),
    [activeImageDownloads, combinedImageDiscoverResults, statusFilter],
  );
  const hasActiveFilters = imageDiscoverHasActiveFilters || statusFilter !== "all";

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
              onChange={(event) => onImageDiscoverSortChange(event.target.value as DiscoverSort)}
            >
              <option value="release">Newest released</option>
              <option value="size">Largest size</option>
              <option value="ram">Highest RAM/VRAM</option>
              <option value="likes">Most likes</option>
              <option value="downloads">Most downloads</option>
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
            <span className="sort-header">Model</span>
            <span className="sort-header">Provider</span>
            <span className="sort-header">Tasks</span>
            <button className="sort-header" type="button" onClick={() => onImageDiscoverSortChange("size")}>
              Size{sortIndicator(imageDiscoverSort, "size")}
            </button>
            <button className="sort-header" type="button" onClick={() => onImageDiscoverSortChange("ram")}>
              RAM/VRAM{sortIndicator(imageDiscoverSort, "ram")}
            </button>
            <span className="sort-header">Spec</span>
            <button className="sort-header" type="button" onClick={() => onImageDiscoverSortChange("release")}>
              Date{sortIndicator(imageDiscoverSort, "release")}
            </button>
            <span className="sort-header">Status</span>
            <span className="sort-header"></span>
          </div>
          <div className="media-model-rows">
            {filteredResults.map((variant) => {
              const downloadState = activeImageDownloads[variant.repo];
              const status = imageVariantStatus(variant, downloadState);
              const isComplete = status === "installed";
              const isDownloading = status === "downloading";
              const isPaused = status === "paused";
              const isDownloadFailed = status === "failed";
              const isPartial = status === "incomplete";
              const isDownloadComplete = downloadState?.state === "completed";
              const hasLocalData = Boolean(variant.hasLocalData || isDownloadComplete || isPaused || isDownloadFailed);
              const friendlyDownloadError = formatImageAccessError(downloadState?.error, variant);
              const needsGatedAccess = isGatedImageAccessError(downloadState?.error);
              const memoryEstimate = imageDiscoverMemoryEstimate(variant);
              const secondarySize = imageSecondarySizeLabel(variant);
              const releaseLabel = formatReleaseLabel(variant.releaseLabel, variant.releaseDate ?? variant.createdAt);
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
                    <span title={secondarySize ?? undefined}>
                      {imagePrimarySizeLabel(variant)}
                      {secondarySize ? <small>{secondarySize}</small> : null}
                    </span>
                    <span title={memoryEstimate?.title ?? "RAM/VRAM estimate pending until model weight size is known."}>
                      {memoryEstimate?.label ?? "pending"}
                    </span>
                    <span>
                      {variant.recommendedResolution}
                      {variant.pipelineTag ? <small>{variant.pipelineTag}</small> : null}
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
                        <button className="primary-button" type="button" onClick={() => onOpenImageStudio(variant.id)}>
                          Generate
                        </button>
                      ) : isDownloading ? (
                        <>
                          <button className="secondary-button" type="button" onClick={() => onCancelImageDownload(variant.repo)}>
                            Pause
                          </button>
                          <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(variant.repo)}>
                            Cancel
                          </button>
                        </>
                      ) : isPaused ? (
                        <>
                          <button className="secondary-button" type="button" onClick={() => onImageDownload(variant.repo)}>
                            Resume
                          </button>
                          <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(variant.repo)}>
                            Delete
                          </button>
                        </>
                      ) : isDownloadFailed ? (
                        <>
                          <button className="secondary-button" type="button" onClick={() => onImageDownload(variant.repo)}>
                            Retry
                          </button>
                          <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(variant.repo)}>
                            Delete
                          </button>
                        </>
                      ) : (
                        <>
                          <button className="secondary-button" type="button" onClick={() => onImageDownload(variant.repo)}>
                            {isPartial ? "Resume" : "Download"}
                          </button>
                          {hasLocalData ? (
                            <button className="secondary-button danger-button" type="button" onClick={() => onDeleteImageDownload(variant.repo)}>
                              Delete
                            </button>
                          ) : null}
                        </>
                      )}
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
                        Hugging Face
                      </button>
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
