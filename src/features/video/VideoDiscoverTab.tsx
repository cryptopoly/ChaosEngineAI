import { Panel } from "../../components/Panel";
import type { DownloadStatus } from "../../api";
import type {
  TabId,
  VideoModelVariant,
} from "../../types";
import type { DiscoverSort } from "../../types/image";
import type { VideoDiscoverTaskFilter } from "../../types/video";
import {
  downloadProgressLabel,
  downloadSizeTooltip,
  formatReleaseLabel,
  number,
  videoPrimarySizeLabel,
  videoSecondarySizeLabel,
} from "../../utils";

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
  onActiveTabChange: (tab: TabId) => void;
  onOpenVideoStudio: (modelId?: string) => void;
  onVideoDownload: (repo: string) => void;
  onCancelVideoDownload: (repo: string) => void;
  onDeleteVideoDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
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
  onActiveTabChange,
  onOpenVideoStudio,
  onVideoDownload,
  onCancelVideoDownload,
  onDeleteVideoDownload,
  onOpenExternalUrl,
  onRevealPath,
}: VideoDiscoverTabProps) {
  return (
    <div className="image-discover-stack">
      <Panel
        title="Video Discover"
        subtitle={`${combinedVideoDiscoverResults.length} video models / live Hugging Face metadata`}
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
            Sort by
            <select
              className="text-input"
              value={videoDiscoverSort}
              onChange={(event) => onVideoDiscoverSortChange(event.target.value as DiscoverSort)}
            >
              <option value="release">Newest released</option>
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
              }}
              disabled={!videoDiscoverHasActiveFilters}
            >
              Clear Filters
            </button>
          </div>
        </div>

        <div className="image-discover-results-summary">
          <span>
            {combinedVideoDiscoverResults.length} model{combinedVideoDiscoverResults.length !== 1 ? "s" : ""} ·{" "}
            {videoDiscoverSort === "likes"
              ? "most liked first"
              : videoDiscoverSort === "downloads"
                ? "most downloads first"
                : "newest released first"}
          </span>
          {videoDiscoverSearchQuery ? (
            <span className="badge subtle">Search: {videoDiscoverSearchInput.trim()}</span>
          ) : null}
          {videoDiscoverTaskFilter !== "all" ? (
            <span className="badge muted">Task: {videoDiscoverTaskFilter}</span>
          ) : null}
        </div>
      </Panel>

      {combinedVideoDiscoverResults.length === 0 ? (
        <Panel title="Video Models" subtitle="No models match the current filters" className="image-discover-section-panel">
          <div className="empty-state image-empty-state">
            <p>Try broadening the filters or search terms.</p>
          </div>
        </Panel>
      ) : (
        <div className="image-discover-grid image-discover-grid--latest">
          {combinedVideoDiscoverResults.map((variant) => {
            const downloadState = activeVideoDownloads[variant.repo];
            const isDownloading = downloadState?.state === "downloading";
            const isPaused = downloadState?.state === "cancelled";
            const isDownloadComplete = downloadState?.state === "completed";
            const isDownloadFailed = downloadState?.state === "failed";
            const isComplete = variant.availableLocally || isDownloadComplete;
            const isPartial = !isComplete && variant.hasLocalData;
            const canDeleteLocalData = Boolean(
              isComplete || isDownloadComplete || isPaused || isDownloadFailed || isPartial,
            );
            return (
              <article key={variant.id} className="image-library-card">
                <div className="image-library-card-head">
                  <div>
                    <h3>{variant.name}</h3>
                    <p>{variant.familyName ?? variant.provider}</p>
                  </div>
                  {isComplete ? (
                    <span className="badge success">Installed</span>
                  ) : isDownloading ? (
                    <span className="badge accent" title={downloadSizeTooltip(downloadState)}>
                      {downloadProgressLabel(downloadState)}
                    </span>
                  ) : isPaused ? (
                    <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                  ) : isDownloadFailed ? (
                    <span className="badge warning">Download Failed</span>
                  ) : isPartial ? (
                    <span className="badge warning">Incomplete</span>
                  ) : null}
                </div>
                <div className="image-library-stats">
                  <span>{videoPrimarySizeLabel(variant)}</span>
                  {videoSecondarySizeLabel(variant) ? <span>{videoSecondarySizeLabel(variant)}</span> : null}
                  <span>{variant.recommendedResolution}</span>
                  <span>{number(variant.defaultDurationSeconds)}s clip</span>
                  {formatReleaseLabel(variant.releaseLabel, variant.releaseDate) ? (
                    <span>{formatReleaseLabel(variant.releaseLabel, variant.releaseDate)}</span>
                  ) : null}
                  {variant.downloadsLabel ? <span>{variant.downloadsLabel}</span> : null}
                  {variant.likesLabel ? <span>{variant.likesLabel}</span> : null}
                  {variant.styleTags.slice(0, 3).map((tag) => (
                    <span key={tag} className="badge subtle">{tag}</span>
                  ))}
                </div>
                <p className="muted-text">{variant.note}</p>
                {isDownloadFailed && downloadState?.error ? (
                  <p className="muted-text" style={{ color: "var(--error, #e26d6d)" }}>{downloadState.error}</p>
                ) : null}
                <div className="button-row">
                  {isComplete ? (
                    <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(variant.id)}>
                      Generate
                    </button>
                  ) : isDownloading ? (
                    <button className="secondary-button" type="button" onClick={() => onCancelVideoDownload(variant.repo)}>
                      Pause
                    </button>
                  ) : isPaused ? (
                    <button className="secondary-button" type="button" onClick={() => onVideoDownload(variant.repo)}>
                      Resume
                    </button>
                  ) : (
                    <button className="secondary-button" type="button" onClick={() => onVideoDownload(variant.repo)}>
                      {isDownloadFailed ? "Retry" : isPartial ? "Resume Download" : "Download"}
                    </button>
                  )}
                  {isDownloading || canDeleteLocalData ? (
                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteVideoDownload(variant.repo)}>
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
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
