import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
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

function videoDiscoverSortLabel(sort: DiscoverSort, t: TFunction<"library">): string {
  if (sort === "name")
    return t("videoDiscover.sortLabel.name", { defaultValue: "name" });
  if (sort === "provider")
    return t("videoDiscover.sortLabel.provider", { defaultValue: "provider" });
  if (sort === "tasks")
    return t("videoDiscover.sortLabel.tasks", { defaultValue: "tasks" });
  if (sort === "size")
    return t("videoDiscover.sortLabel.size", { defaultValue: "largest size first" });
  if (sort === "ram")
    return t("videoDiscover.sortLabel.ram", { defaultValue: "highest RAM/VRAM first" });
  if (sort === "likes")
    return t("videoDiscover.sortLabel.likes", { defaultValue: "most liked first" });
  if (sort === "downloads")
    return t("videoDiscover.sortLabel.downloads", { defaultValue: "most downloads first" });
  if (sort === "status")
    return t("videoDiscover.sortLabel.status", { defaultValue: "status" });
  return t("videoDiscover.sortLabel.release", { defaultValue: "newest released first" });
}

function sortIndicator(activeSort: DiscoverSort, sortDir: SortDir, key: DiscoverSort): string {
  if (activeSort !== key) return "";
  return sortDir === "asc" ? " ▲" : " ▼";
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

function memoryParts(label: string | null | undefined, pendingLabel: string): { primary: string; secondary: string | null } {
  if (!label) return { primary: pendingLabel, secondary: null };
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

function statusBadge(
  status: MediaStatusFilter,
  t: TFunction<"library">,
  downloadState?: DownloadStatus,
  longLiveInstalling = false,
) {
  const downloadDetail = downloadState
    ? [downloadProgressLabel(downloadState), downloadSizeTooltip(downloadState)].filter(Boolean).join(" / ")
    : null;
  if (status === "installed")
    return (
      <StatusIcon
        status="installed"
        label={t("videoDiscover.status.installed", { defaultValue: "Installed" })}
      />
    );
  if (longLiveInstalling)
    return (
      <StatusIcon
        status="downloading"
        label={t("videoDiscover.status.installing", { defaultValue: "Installing" })}
      />
    );
  if (status === "downloading" && downloadState) {
    return (
      <StatusIcon
        status="downloading"
        label={t("videoDiscover.status.downloading", { defaultValue: "Downloading" })}
        detail={downloadDetail}
      />
    );
  }
  if (status === "paused" && downloadState) {
    return (
      <StatusIcon
        status="paused"
        label={t("videoDiscover.status.paused", { defaultValue: "Paused" })}
        detail={downloadDetail}
      />
    );
  }
  if (status === "failed")
    return (
      <StatusIcon
        status="failed"
        label={t("videoDiscover.status.failed", { defaultValue: "Failed" })}
        detail={
          downloadState?.error ?? t("videoDiscover.status.failedDetail", { defaultValue: "Download failed" })
        }
      />
    );
  if (status === "incomplete")
    return (
      <StatusIcon
        status="incomplete"
        label={t("videoDiscover.status.incomplete", { defaultValue: "Incomplete" })}
      />
    );
  return (
    <StatusIcon
      status="incomplete"
      label={t("videoDiscover.status.notInstalled", { defaultValue: "Not installed" })}
    />
  );
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
  const { t } = useTranslation("common");
  const { t: tLib } = useTranslation("library");
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

  const pendingMemoryLabel = tLib("videoDiscover.memory.pending", { defaultValue: "pending" });
  const cancelDownloadLabel = tLib("videoDiscover.action.cancelDownload", { defaultValue: "Cancel download" });
  const deleteModelFallback = tLib("videoDiscover.action.deleteModel", { defaultValue: "Delete model" });
  const deleteDownloadFallback = tLib("videoDiscover.action.deleteDownload", { defaultValue: "Delete download" });

  return (
    <div className="image-discover-stack">
      <Panel
        title={t("tabs.videoDiscover")}
        subtitle={tLib("videoDiscover.subtitle", {
          defaultValue: "{filtered} of {total} video models / live Hugging Face metadata",
          filtered: filteredResults.length,
          total: combinedVideoDiscoverResults.length,
        })}
      >
        <div className="image-hero">
          <div>
            <h3>
              {tLib("videoDiscover.hero.heading", {
                defaultValue: "Browse and download video models for local generation.",
              })}
            </h3>
            <p className="muted-text">
              {tLib("videoDiscover.hero.body", {
                defaultValue:
                  "Download any model to use it in Video Studio. Runtime status lives in the Studio tab.",
              })}
            </p>
          </div>
          <div className="image-hero-actions">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-models")}>
              {tLib("videoDiscover.hero.installedModels", { defaultValue: "Installed Models" })}
            </button>
            <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(selectedVideoVariant?.id)}>
              {tLib("videoDiscover.hero.openStudio", { defaultValue: "Open Studio" })}
            </button>
          </div>
        </div>

        <div className="image-discover-filter-row">
          <label className="image-discover-search">
            {tLib("videoDiscover.filter.search", { defaultValue: "Search" })}
            <input
              className="text-input"
              type="search"
              value={videoDiscoverSearchInput}
              onChange={(event) => onVideoDiscoverSearchInputChange(event.target.value)}
              placeholder={tLib("videoDiscover.filter.searchPlaceholder", {
                defaultValue: "Search LTX, Wan, Mochi, provider, tags...",
              })}
            />
          </label>
          <label>
            {tLib("videoDiscover.filter.task", { defaultValue: "Task" })}
            <select
              className="text-input"
              value={videoDiscoverTaskFilter}
              onChange={(event) => onVideoDiscoverTaskFilterChange(event.target.value as VideoDiscoverTaskFilter)}
            >
              <option value="all">
                {tLib("videoDiscover.task.all", { defaultValue: "All tasks" })}
              </option>
              <option value="txt2video">
                {tLib("videoDiscover.task.txt2video", { defaultValue: "Text to video" })}
              </option>
              <option value="img2video">
                {tLib("videoDiscover.task.img2video", { defaultValue: "Image to video" })}
              </option>
              <option value="video2video">
                {tLib("videoDiscover.task.video2video", { defaultValue: "Video to video" })}
              </option>
            </select>
          </label>
          <label>
            {tLib("videoDiscover.filter.status", { defaultValue: "Status" })}
            <select
              className="text-input"
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value as MediaStatusFilter)}
            >
              <option value="all">
                {tLib("videoDiscover.status.any", { defaultValue: "Any status" })}
              </option>
              <option value="installed">
                {tLib("videoDiscover.status.installed", { defaultValue: "Installed" })}
              </option>
              <option value="not-installed">
                {tLib("videoDiscover.status.notInstalled", { defaultValue: "Not installed" })}
              </option>
              <option value="downloading">
                {tLib("videoDiscover.status.downloading", { defaultValue: "Downloading" })}
              </option>
              <option value="paused">
                {tLib("videoDiscover.status.paused", { defaultValue: "Paused" })}
              </option>
              <option value="failed">
                {tLib("videoDiscover.status.failed", { defaultValue: "Failed" })}
              </option>
              <option value="incomplete">
                {tLib("videoDiscover.status.incomplete", { defaultValue: "Incomplete" })}
              </option>
            </select>
          </label>
          <label>
            {tLib("videoDiscover.filter.sortBy", { defaultValue: "Sort by" })}
            <select
              className="text-input"
              value={videoDiscoverSort}
              onChange={(event) => {
                const nextSort = event.target.value as DiscoverSort;
                onVideoDiscoverSortChange(nextSort);
                setSortDir(defaultSortDir(nextSort));
              }}
            >
              <option value="name">
                {tLib("videoDiscover.sort.name", { defaultValue: "Name" })}
              </option>
              <option value="provider">
                {tLib("videoDiscover.sort.provider", { defaultValue: "Provider" })}
              </option>
              <option value="tasks">
                {tLib("videoDiscover.sort.tasks", { defaultValue: "Tasks" })}
              </option>
              <option value="release">
                {tLib("videoDiscover.sort.release", { defaultValue: "Newest released" })}
              </option>
              <option value="size">
                {tLib("videoDiscover.sort.size", { defaultValue: "Largest size" })}
              </option>
              <option value="ram">
                {tLib("videoDiscover.sort.ram", { defaultValue: "Highest RAM/VRAM" })}
              </option>
              <option value="likes">
                {tLib("videoDiscover.sort.likes", { defaultValue: "Most likes" })}
              </option>
              <option value="downloads">
                {tLib("videoDiscover.sort.downloads", { defaultValue: "Most downloads" })}
              </option>
              <option value="status">
                {tLib("videoDiscover.sort.status", { defaultValue: "Status" })}
              </option>
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
              {tLib("videoDiscover.filter.clear", { defaultValue: "Clear Filters" })}
            </button>
          </div>
        </div>

        <div className="image-discover-results-summary">
          <span>
            {tLib("videoDiscover.summary.count", {
              defaultValue: "{count, plural, one {# model} other {# models}} · {sortLabel}",
              count: filteredResults.length,
              sortLabel: videoDiscoverSortLabel(videoDiscoverSort, tLib),
            })}
          </span>
          {videoDiscoverSearchQuery ? (
            <span className="badge subtle">
              {tLib("videoDiscover.summary.searchBadge", {
                defaultValue: "Search: {query}",
                query: videoDiscoverSearchInput.trim(),
              })}
            </span>
          ) : null}
          {videoDiscoverTaskFilter !== "all" ? (
            <span className="badge muted">
              {tLib("videoDiscover.summary.taskBadge", {
                defaultValue: "Task: {task}",
                task: videoDiscoverTaskFilter,
              })}
            </span>
          ) : null}
          {statusFilter !== "all" ? (
            <span className="badge muted">
              {tLib("videoDiscover.summary.statusBadge", {
                defaultValue: "Status: {status}",
                status: statusFilter,
              })}
            </span>
          ) : null}
        </div>
      </Panel>

      {filteredResults.length === 0 ? (
        <Panel
          title={t("tabs.videoModels")}
          subtitle={t("panels.noModelsMatchFilters", { defaultValue: "No models match the current filters" })}
          className="image-discover-section-panel"
        >
          <div className="empty-state image-empty-state">
            <p>
              {tLib("videoDiscover.empty.body", {
                defaultValue: "Try broadening the filters or search terms.",
              })}
            </p>
          </div>
        </Panel>
      ) : (
        <div className="media-model-table media-model-table--video">
          <div className="media-model-head">
            <button className="sort-header" type="button" onClick={() => applySort("name")}>
              {tLib("videoDiscover.column.model", { defaultValue: "Model" })}
              {sortIndicator(videoDiscoverSort, sortDir, "name")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("provider")}>
              {tLib("videoDiscover.column.provider", { defaultValue: "Provider" })}
              {sortIndicator(videoDiscoverSort, sortDir, "provider")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("tasks")}>
              {tLib("videoDiscover.column.tasks", { defaultValue: "Tasks" })}
              {sortIndicator(videoDiscoverSort, sortDir, "tasks")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("size")}>
              {tLib("videoDiscover.column.size", { defaultValue: "Size" })}
              {sortIndicator(videoDiscoverSort, sortDir, "size")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("ram")}>
              {tLib("videoDiscover.column.ramVram", { defaultValue: "RAM/VRAM" })}
              {sortIndicator(videoDiscoverSort, sortDir, "ram")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("release")}>
              {tLib("videoDiscover.column.released", { defaultValue: "Released" })}
              {sortIndicator(videoDiscoverSort, sortDir, "release")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("status")}>
              {tLib("videoDiscover.column.status", { defaultValue: "Status" })}
              {sortIndicator(videoDiscoverSort, sortDir, "status")}
            </button>
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
              const memory = memoryParts(memoryEstimate?.label, pendingMemoryLabel);
              const deleteRepo = videoDeleteRepoForVariant(variant, downloadState);
              const deleteLabel = isDownloading
                ? cancelDownloadLabel
                : videoDeleteLabelForRepo(variant, deleteRepo, deleteModelFallback);
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
                    <span
                      className="media-model-memory"
                      title={
                        memoryEstimate?.title ??
                        tLib("videoDiscover.memory.pendingTitle", {
                          defaultValue: "RAM/VRAM estimate pending until model weight size is known.",
                        })
                      }
                    >
                      <span>{memory.primary}</span>
                      {memory.secondary ? <small>{memory.secondary}</small> : null}
                    </span>
                    <span>
                      {releaseLabel ?? tLib("videoDiscover.unknown", { defaultValue: "Unknown" })}
                      {variant.downloadsLabel ? <small>{variant.downloadsLabel}</small> : null}
                      {variant.likesLabel ? <small>{variant.likesLabel}</small> : null}
                    </span>
                    <span>{statusBadge(status, tLib, downloadState, isLongLive && installingLongLive && !longLiveReady)}</span>
                    <div className="media-model-actions">
                      {isLongLive ? (
                        isComplete ? (
                          <IconActionButton
                            icon="generate"
                            label={tLib("videoDiscover.action.generate", { defaultValue: "Generate" })}
                            buttonStyle="primary"
                            onClick={() => onOpenVideoStudio(variant.id)}
                          />
                        ) : (
                          <>
                            <IconActionButton
                              icon="install"
                              label={
                                installingLongLive
                                  ? tLib("videoDiscover.action.installing", { defaultValue: "Installing" })
                                  : tLib("videoDiscover.action.install", { defaultValue: "Install" })
                              }
                              onClick={() => void onInstallLongLive()}
                              disabled={installingLongLive}
                            />
                            <InstallLogPanel job={longLiveJob} variant="longlive" />
                          </>
                        )
                      ) : isComplete ? (
                        <IconActionButton
                          icon="generate"
                          label={tLib("videoDiscover.action.generate", { defaultValue: "Generate" })}
                          buttonStyle="primary"
                          onClick={() => onOpenVideoStudio(variant.id)}
                        />
                      ) : isDownloading ? (
                        <>
                          <IconActionButton
                            icon="pause"
                            label={tLib("videoDiscover.action.pauseDownload", { defaultValue: "Pause download" })}
                            onClick={() => onCancelVideoDownload(downloadState?.repo ?? variant.repo)}
                          />
                          <IconActionButton icon="cancel" label={deleteLabel} danger onClick={() => onDeleteVideoDownload(deleteRepo)} />
                        </>
                      ) : isPaused ? (
                        <>
                          <IconActionButton
                            icon="resume"
                            label={tLib("videoDiscover.action.resumeDownload", { defaultValue: "Resume download" })}
                            onClick={() => onVideoDownload(variant.repo, variant.id)}
                          />
                          <IconActionButton
                            icon="delete"
                            label={videoDeleteLabelForRepo(variant, deleteRepo, deleteDownloadFallback)}
                            danger
                            onClick={() => onDeleteVideoDownload(deleteRepo)}
                          />
                        </>
                      ) : (
                        <IconActionButton
                          icon={isDownloadFailed ? "retry" : isPartial ? "resume" : "download"}
                          label={
                            isDownloadFailed
                              ? tLib("videoDiscover.action.retryDownload", { defaultValue: "Retry download" })
                              : isPartial
                                ? tLib("videoDiscover.action.resumeDownload", { defaultValue: "Resume download" })
                                : tLib("videoDiscover.action.downloadModel", { defaultValue: "Download model" })
                          }
                          onClick={() => onVideoDownload(variant.repo, variant.id)}
                        />
                      )}
                      {!isLongLive && !isDownloading && !isPaused && canDeleteLocalData ? (
                        <IconActionButton icon="delete" label={deleteLabel} danger onClick={() => onDeleteVideoDownload(deleteRepo)} />
                      ) : null}
                      {variant.localPath ? (
                        <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(variant.localPath as string)} />
                      ) : null}
                      <IconActionButton
                        icon="modelCard"
                        label={tLib("videoDiscover.action.openModelCard", { defaultValue: "Open model card" })}
                        onClick={() => onOpenExternalUrl(variant.link)}
                      />
                    </div>
                  </div>
                  {isLongLive && !isComplete ? (
                    <div className="media-model-row-detail callout quiet">
                      <p>
                        {tLib("videoDiscover.longLive.installNote", {
                          defaultValue:
                            "LongLive installs into an isolated venv at ~/.chaosengine/longlive. CUDA only, 5-15 min depending on network.",
                        })}
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
