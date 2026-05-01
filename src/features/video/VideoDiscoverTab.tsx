import { useEffect, useMemo, useState } from "react";
import { InstallLogPanel } from "../../components/InstallLogPanel";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import { Panel } from "../../components/Panel";
import type { DownloadStatus, InstallResult, LongLiveJobState } from "../../api";
import type {
  TabId,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../../types";
import type { DiscoverSort } from "../../types/image";
import type { VideoDiscoverTaskFilter } from "../../types/video";
import {
  compactModelSizeLabel,
  compactReleaseLabel,
  downloadProgressLabel,
  downloadSizeTooltip,
  formatReleaseLabel,
  videoDiscoverMemoryEstimate,
  videoDeleteLabelForRepo,
  videoDeleteRepoForVariant,
  videoDownloadStatusForVariant,
  videoPrimarySizeLabel,
  videoSecondarySizeLabel,
} from "../../utils";

type MediaStatusFilter = "all" | "installed" | "not-installed" | "downloading" | "paused" | "failed" | "incomplete";
type SortDir = "asc" | "desc";

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
  const downloadDetail = downloadState
    ? [downloadProgressLabel(downloadState), downloadSizeTooltip(downloadState)].filter(Boolean).join(" / ")
    : null;
  if (status === "installed") return <StatusIcon status="installed" label="Installed" />;
  if (longLiveInstalling) return <StatusIcon status="downloading" label="Installing" />;
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
  const [sortDir, setSortDir] = useState<SortDir>(defaultSortDir(videoDiscoverSort));
  const longLiveReady = longLiveStatus?.realGenerationAvailable ?? false;
  const filteredResults = useMemo(
    () =>
      combinedVideoDiscoverResults
        .map((variant) => {
          const downloadState = videoDownloadStatusForVariant(activeVideoDownloads, variant);
          const status = videoVariantStatus(variant, downloadState, longLiveReady, installingLongLive);
          const memoryEstimate = videoDiscoverMemoryEstimate(variant);
          return { variant, status, memoryEstimate };
        })
        .filter(({ status }) => statusFilter === "all" || status === statusFilter)
        .sort((left, right) => {
          if (videoDiscoverSort === "name") {
            const diff = left.variant.name.localeCompare(right.variant.name);
            return sortDir === "asc" ? diff : -diff;
          }
          if (videoDiscoverSort === "provider") {
            const diff = left.variant.provider.localeCompare(right.variant.provider);
            if (diff !== 0) return sortDir === "asc" ? diff : -diff;
          }
          if (videoDiscoverSort === "tasks") {
            const diff = left.variant.taskSupport.join(" ").localeCompare(right.variant.taskSupport.join(" "));
            if (diff !== 0) return sortDir === "asc" ? diff : -diff;
          }
          if (videoDiscoverSort === "size") {
            const diff = compareNullableNumber(sizeSortKey(left.variant), sizeSortKey(right.variant), sortDir);
            if (diff !== 0) return diff;
          } else if (videoDiscoverSort === "ram") {
            const diff = compareNullableNumber(left.memoryEstimate?.estimatedPeakGb ?? null, right.memoryEstimate?.estimatedPeakGb ?? null, sortDir);
            if (diff !== 0) return diff;
          } else if (videoDiscoverSort === "status") {
            const diff = statusSortKey(left.status) - statusSortKey(right.status);
            if (diff !== 0) return sortDir === "asc" ? diff : -diff;
          } else if (videoDiscoverSort === "likes") {
            const diff = compareNullableNumber(left.variant.likes ?? null, right.variant.likes ?? null, sortDir);
            if (diff !== 0) return diff;
          } else if (videoDiscoverSort === "downloads") {
            const diff = compareNullableNumber(left.variant.downloads ?? null, right.variant.downloads ?? null, sortDir);
            if (diff !== 0) return diff;
          }
          const dateDiff = releaseSortKey(right.variant).localeCompare(releaseSortKey(left.variant));
          if (dateDiff !== 0) return sortDir === "desc" ? dateDiff : -dateDiff;
          return left.variant.name.localeCompare(right.variant.name);
        }),
    [
      activeVideoDownloads,
      combinedVideoDiscoverResults,
      installingLongLive,
      longLiveReady,
      sortDir,
      statusFilter,
      videoDiscoverSort,
    ],
  );
  const hasActiveFilters = videoDiscoverHasActiveFilters || statusFilter !== "all";

  function applySort(nextSort: DiscoverSort) {
    if (videoDiscoverSort === nextSort) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      onVideoDiscoverSortChange(nextSort);
      setSortDir(defaultSortDir(nextSort));
    }
  }

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
              onChange={(event) => {
                const nextSort = event.target.value as DiscoverSort;
                onVideoDiscoverSortChange(nextSort);
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
                onVideoDiscoverSearchInputChange("");
                onVideoDiscoverTaskFilterChange("all");
                setStatusFilter("all");
                onVideoDiscoverSortChange("release");
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
            <button className="sort-header" type="button" onClick={() => applySort("name")}>Model{sortIndicator(videoDiscoverSort, sortDir, "name")}</button>
            <button className="sort-header" type="button" onClick={() => applySort("provider")}>Provider{sortIndicator(videoDiscoverSort, sortDir, "provider")}</button>
            <button className="sort-header" type="button" onClick={() => applySort("tasks")}>Tasks{sortIndicator(videoDiscoverSort, sortDir, "tasks")}</button>
            <button className="sort-header" type="button" onClick={() => applySort("size")}>
              Size{sortIndicator(videoDiscoverSort, sortDir, "size")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("ram")}>
              RAM/VRAM{sortIndicator(videoDiscoverSort, sortDir, "ram")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("release")}>
              Released{sortIndicator(videoDiscoverSort, sortDir, "release")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("status")}>Status{sortIndicator(videoDiscoverSort, sortDir, "status")}</button>
            <span className="sort-header"></span>
          </div>
          <div className="media-model-rows">
            {filteredResults.map(({ variant, status, memoryEstimate }) => {
              const isLongLive = isLongLiveRepo(variant.repo);
              const downloadState = videoDownloadStatusForVariant(activeVideoDownloads, variant);
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
                    </span>
                    <span>{statusBadge(status, downloadState, isLongLive && installingLongLive && !longLiveReady)}</span>
                    <div className="media-model-actions">
                      {isLongLive ? (
                        isComplete ? (
                          <IconActionButton icon="generate" label="Generate" buttonStyle="primary" onClick={() => onOpenVideoStudio(variant.id)} />
                        ) : (
                          <>
                            <IconActionButton icon="install" label={installingLongLive ? "Installing" : "Install"} onClick={() => void onInstallLongLive()} disabled={installingLongLive} />
                            <InstallLogPanel job={longLiveJob} variant="longlive" />
                          </>
                        )
                      ) : isComplete ? (
                        <IconActionButton icon="generate" label="Generate" buttonStyle="primary" onClick={() => onOpenVideoStudio(variant.id)} />
                      ) : isDownloading ? (
                        <>
                          <IconActionButton icon="pause" label="Pause download" onClick={() => onCancelVideoDownload(downloadState?.repo ?? variant.repo)} />
                          <IconActionButton icon="cancel" label={deleteLabel} danger onClick={() => onDeleteVideoDownload(deleteRepo)} />
                        </>
                      ) : isPaused ? (
                        <>
                          <IconActionButton icon="resume" label="Resume download" onClick={() => onVideoDownload(variant.repo, variant.id)} />
                          <IconActionButton icon="delete" label={videoDeleteLabelForRepo(variant, deleteRepo, "Delete download")} danger onClick={() => onDeleteVideoDownload(deleteRepo)} />
                        </>
                      ) : (
                        <IconActionButton icon={isDownloadFailed ? "retry" : isPartial ? "resume" : "download"} label={isDownloadFailed ? "Retry download" : isPartial ? "Resume download" : "Download model"} onClick={() => onVideoDownload(variant.repo, variant.id)} />
                      )}
                      {!isLongLive && !isDownloading && !isPaused && canDeleteLocalData ? (
                        <IconActionButton icon="delete" label={deleteLabel} danger onClick={() => onDeleteVideoDownload(deleteRepo)} />
                      ) : null}
                      {variant.localPath ? (
                        <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(variant.localPath as string)} />
                      ) : null}
                      <IconActionButton icon="modelCard" label="Open model card" onClick={() => onOpenExternalUrl(variant.link)} />
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
