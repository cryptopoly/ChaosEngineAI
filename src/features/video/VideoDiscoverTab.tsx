import { useEffect, useMemo, useState } from "react";
import { InstallLogPanel } from "../../components/InstallLogPanel";
import { Panel } from "../../components/Panel";
import { WanInstallPanel } from "../../components/WanInstallPanel";
import type { DownloadStatus, InstallResult, LongLiveJobState } from "../../api";
import type {
  TabId,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../../types";
import type { DiscoverSort } from "../../types/image";
import type { VideoDiscoverTaskFilter } from "../../types/video";
import {
  downloadProgressLabel,
  downloadSizeTooltip,
  formatReleaseLabel,
  number,
  videoDiscoverMemoryEstimate,
  videoDownloadStatusForVariant,
  videoPrimarySizeLabel,
  videoSecondarySizeLabel,
} from "../../utils";

type MediaStatusFilter = "all" | "installed" | "not-installed" | "downloading" | "paused" | "failed" | "incomplete";

// LongLive ships via a dedicated Python installer (isolated venv + GitHub
// clone + HF weights at Efficient-Large-Model/LongLive-1.3B), not via
// snapshot_download. The catalog repo id ``NVlabs/LongLive-1.3B`` is the
// GitHub org and intentionally does not resolve on Hugging Face — we use
// it purely as a routing key.
function isLongLiveRepo(repo: string | undefined): boolean {
  return repo?.startsWith("NVlabs/LongLive") ?? false;
}

export interface VideoDiscoverTabProps {
  combinedVideoDiscoverResults: VideoModelVariant[];
  videoDiscoverSearchInput: string;
  onVideoDiscoverSearchInputChange: (value: string) => void;
  videoDiscoverTaskFilter: VideoDiscoverTaskFilter;
  onVideoDiscoverTaskFilterChange: (value: VideoDiscoverTaskFilter) => void;
  videoDiscoverSort: DiscoverSort;
  onVideoDiscoverSortChange: (value: DiscoverSort) => void;
  videoDiscoverHasActiveFilters: boolean;
  videoDiscoverSearchQuery: string;
  activeVideoDownloads: Record<string, DownloadStatus>;
  selectedVideoVariant: VideoModelVariant | null;
  fileRevealLabel: string;
  longLiveStatus: VideoRuntimeStatus | null;
  installingLongLive: boolean;
  longLiveJob: LongLiveJobState | null;
  onActiveTabChange: (tab: TabId) => void;
  onOpenVideoStudio: (modelId?: string) => void;
  onVideoDownload: (repo: string, modelId?: string) => void;
  onCancelVideoDownload: (repo: string) => void;
  onDeleteVideoDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
  onRefreshLongLiveStatus: () => void;
  onInstallLongLive: () => Promise<InstallResult>;
}

function videoDiscoverSortLabel(sort: DiscoverSort): string {
  if (sort === "size") return "largest size first";
  if (sort === "ram") return "highest RAM/VRAM first";
  if (sort === "likes") return "most liked first";
  if (sort === "downloads") return "most downloads first";
  return "newest released first";
}

function sortIndicator(activeSort: DiscoverSort, key: DiscoverSort): string {
  return activeSort === key ? " \u25BC" : "";
}

function videoVariantStatus(
  variant: VideoModelVariant,
  downloadState: DownloadStatus | undefined,
  longLiveReady: boolean,
  installingLongLive: boolean,
): MediaStatusFilter {
  if (isLongLiveRepo(variant.repo)) {
    if (longLiveReady) return "installed";
    if (installingLongLive) return "downloading";
    return "not-installed";
  }
  if (variant.availableLocally || downloadState?.state === "completed") return "installed";
  if (downloadState?.state === "downloading") return "downloading";
  if (downloadState?.state === "cancelled") return "paused";
  if (downloadState?.state === "failed") return "failed";
  if (variant.hasLocalData) return "incomplete";
  return "not-installed";
}

function statusBadge(status: MediaStatusFilter, downloadState?: DownloadStatus, longLiveInstalling = false) {
  if (status === "installed") return <span className="badge success">Installed</span>;
  if (longLiveInstalling) return <span className="badge accent">Installing…</span>;
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

export function VideoDiscoverTab({
  combinedVideoDiscoverResults,
  videoDiscoverSearchInput,
  onVideoDiscoverSearchInputChange,
  videoDiscoverTaskFilter,
  onVideoDiscoverTaskFilterChange,
  videoDiscoverSort,
  onVideoDiscoverSortChange,
  videoDiscoverHasActiveFilters,
  videoDiscoverSearchQuery,
  activeVideoDownloads,
  selectedVideoVariant,
  fileRevealLabel,
  longLiveStatus,
  installingLongLive,
  longLiveJob,
  onActiveTabChange,
  onOpenVideoStudio,
  onVideoDownload,
  onCancelVideoDownload,
  onDeleteVideoDownload,
  onOpenExternalUrl,
  onRevealPath,
  onRefreshLongLiveStatus,
  onInstallLongLive,
}: VideoDiscoverTabProps) {
  const hasLongLiveVariant = combinedVideoDiscoverResults.some((variant) =>
    isLongLiveRepo(variant.repo),
  );
  useEffect(() => {
    if (hasLongLiveVariant) onRefreshLongLiveStatus();
  }, [hasLongLiveVariant, onRefreshLongLiveStatus]);

  const [statusFilter, setStatusFilter] = useState<MediaStatusFilter>("all");
  const longLiveReady = longLiveStatus?.realGenerationAvailable ?? false;
  const filteredResults = useMemo(
    () =>
      combinedVideoDiscoverResults.filter((variant) => {
        if (statusFilter === "all") return true;
        const downloadState = videoDownloadStatusForVariant(activeVideoDownloads, variant);
        return videoVariantStatus(variant, downloadState, longLiveReady, installingLongLive) === statusFilter;
      }),
    [activeVideoDownloads, combinedVideoDiscoverResults, installingLongLive, longLiveReady, statusFilter],
  );
  const hasActiveFilters = videoDiscoverHasActiveFilters || statusFilter !== "all";

  return (
    <div className="image-discover-stack">
      <Panel
        title="Video Discover"
        subtitle={`${filteredResults.length} of ${combinedVideoDiscoverResults.length} video models / live Hugging Face metadata`}
      >
        <div className="image-hero">
          <div>
            <h3>Browse and download video models for local generation.</h3>
            <p className="muted-text">
              Download any model to use it in Video Studio. Runtime status lives in the Studio tab.
            </p>
          </div>
          <div className="image-hero-actions">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-models")}>
              Installed Models
            </button>
            <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(selectedVideoVariant?.id)}>
              Open Studio
            </button>
          </div>
        </div>

        {/* FU-025 Phase 9: GUI install action for the Apple-Silicon-only
            Wan MLX runtime. Lists supported raw Wan-AI repos with raw-size
            hints + install button + live progress via InstallLogPanel. */}
        <WanInstallPanel />

        <div className="image-discover-filter-row">
          <label className="image-discover-search">
            Search
            <input
              className="text-input"
              type="search"
              value={videoDiscoverSearchInput}
              onChange={(event) => onVideoDiscoverSearchInputChange(event.target.value)}
              placeholder="Search LTX, Wan, Mochi, provider, tags..."
            />
          </label>
          <label>
            Task
            <select
              className="text-input"
              value={videoDiscoverTaskFilter}
              onChange={(event) => onVideoDiscoverTaskFilterChange(event.target.value as VideoDiscoverTaskFilter)}
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
              value={videoDiscoverSort}
              onChange={(event) => onVideoDiscoverSortChange(event.target.value as DiscoverSort)}
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
                onVideoDiscoverSearchInputChange("");
                onVideoDiscoverTaskFilterChange("all");
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
            {filteredResults.length} model{filteredResults.length !== 1 ? "s" : ""} · {videoDiscoverSortLabel(videoDiscoverSort)}
          </span>
          {videoDiscoverSearchQuery ? (
            <span className="badge subtle">Search: {videoDiscoverSearchInput.trim()}</span>
          ) : null}
          {videoDiscoverTaskFilter !== "all" ? (
            <span className="badge muted">Task: {videoDiscoverTaskFilter}</span>
          ) : null}
          {statusFilter !== "all" ? <span className="badge muted">Status: {statusFilter}</span> : null}
        </div>
      </Panel>

      {filteredResults.length === 0 ? (
        <Panel title="Video Models" subtitle="No models match the current filters" className="image-discover-section-panel">
          <div className="empty-state image-empty-state">
            <p>Try broadening the filters or search terms.</p>
          </div>
        </Panel>
      ) : (
        <div className="media-model-table media-model-table--video">
          <div className="media-model-head">
            <span className="sort-header">Model</span>
            <span className="sort-header">Provider</span>
            <span className="sort-header">Tasks</span>
            <button className="sort-header" type="button" onClick={() => onVideoDiscoverSortChange("size")}>
              Size{sortIndicator(videoDiscoverSort, "size")}
            </button>
            <button className="sort-header" type="button" onClick={() => onVideoDiscoverSortChange("ram")}>
              RAM/VRAM{sortIndicator(videoDiscoverSort, "ram")}
            </button>
            <span className="sort-header">Spec</span>
            <button className="sort-header" type="button" onClick={() => onVideoDiscoverSortChange("release")}>
              Date{sortIndicator(videoDiscoverSort, "release")}
            </button>
            <span className="sort-header">Status</span>
            <span className="sort-header"></span>
          </div>
          <div className="media-model-rows">
            {filteredResults.map((variant) => {
              const isLongLive = isLongLiveRepo(variant.repo);
              const downloadState = videoDownloadStatusForVariant(activeVideoDownloads, variant);
              const status = videoVariantStatus(variant, downloadState, longLiveReady, installingLongLive);
              const isComplete = status === "installed";
              const isDownloading = status === "downloading";
              const isPaused = status === "paused";
              const isDownloadFailed = status === "failed";
              const isPartial = status === "incomplete";
              const isDownloadComplete = downloadState?.state === "completed";
              const canDeleteLocalData = isLongLive
                ? false
                : Boolean(isComplete || isDownloadComplete || isPaused || isDownloadFailed || isPartial);
              const localStatusReason = !isComplete && !isDownloading ? variant.localStatusReason : null;
              const memoryEstimate = videoDiscoverMemoryEstimate(variant);
              const secondarySize = videoSecondarySizeLabel(variant);
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
                      </div>
                    </div>
                    <span>{variant.provider}</span>
                    <div className="media-model-chip-row">
                      {variant.taskSupport.map((task) => (
                        <span key={task} className="badge muted">{task}</span>
                      ))}
                    </div>
                    <span title={secondarySize ?? undefined}>
                      {videoPrimarySizeLabel(variant)}
                      {secondarySize ? <small>{secondarySize}</small> : null}
                    </span>
                    <span title={memoryEstimate?.title ?? "RAM/VRAM estimate pending until model weight size is known."}>
                      {memoryEstimate?.label ?? "pending"}
                    </span>
                    <span>
                      {variant.recommendedResolution}
                      <small>{number(variant.defaultDurationSeconds)}s clip</small>
                    </span>
                    <span>
                      {releaseLabel ?? "Unknown"}
                      {variant.downloadsLabel ? <small>{variant.downloadsLabel}</small> : null}
                      {variant.likesLabel ? <small>{variant.likesLabel}</small> : null}
                    </span>
                    <span>{statusBadge(status, downloadState, isLongLive && installingLongLive && !longLiveReady)}</span>
                    <div className="media-model-actions">
                      {isLongLive ? (
                        isComplete ? (
                          <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(variant.id)}>
                            Generate
                          </button>
                        ) : (
                          <>
                            <button
                              className="secondary-button"
                              type="button"
                              onClick={() => void onInstallLongLive()}
                              disabled={installingLongLive}
                            >
                              {installingLongLive ? "Installing…" : "Install"}
                            </button>
                            <InstallLogPanel job={longLiveJob} variant="longlive" />
                          </>
                        )
                      ) : isComplete ? (
                        <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(variant.id)}>
                          Generate
                        </button>
                      ) : isDownloading ? (
                        <>
                          <button className="secondary-button" type="button" onClick={() => onCancelVideoDownload(downloadState?.repo ?? variant.repo)}>
                            Pause
                          </button>
                          <button className="secondary-button danger-button" type="button" onClick={() => onDeleteVideoDownload(downloadState?.repo ?? variant.repo)}>
                            Cancel
                          </button>
                        </>
                      ) : isPaused ? (
                        <>
                          <button className="secondary-button" type="button" onClick={() => onVideoDownload(variant.repo, variant.id)}>
                            Resume
                          </button>
                          <button className="secondary-button danger-button" type="button" onClick={() => onDeleteVideoDownload(downloadState?.repo ?? variant.repo)}>
                            Delete
                          </button>
                        </>
                      ) : (
                        <button className="secondary-button" type="button" onClick={() => onVideoDownload(variant.repo, variant.id)}>
                          {isDownloadFailed ? "Retry" : isPartial ? "Resume" : "Download"}
                        </button>
                      )}
                      {!isLongLive && !isDownloading && canDeleteLocalData ? (
                        <button className="secondary-button danger-button" type="button" onClick={() => onDeleteVideoDownload(downloadState?.repo ?? variant.repo)}>
                          Delete
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
                  {isLongLive && !isComplete ? (
                    <div className="media-model-row-detail callout quiet">
                      <p>
                        LongLive installs into an isolated venv at <code>~/.chaosengine/longlive</code>.
                        CUDA only, 5-15 min depending on network.
                      </p>
                    </div>
                  ) : isDownloadFailed && downloadState?.error ? (
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
    </div>
  );
}
