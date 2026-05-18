import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { Panel } from "../../components/Panel";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import type { DownloadStatus } from "../../api";
import type {
  ImageModelVariant,
  SystemStats,
  TabId,
} from "../../types";
import type {
  DiscoverSort,
  ImageDiscoverTaskFilter,
  ImageDiscoverAccessFilter,
} from "../../types/image";
import type { NativeBackendStatus } from "../../types/server";
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
  imageOrVideoVariantPlatformGate,
  isVariantCompatibleWithHost,
} from "../../utils";
import { AcceleratorCard } from "../../components/AcceleratorCard";
import {
  getAccelerator,
  getApplicableAccelerators,
} from "../../components/acceleratorCatalog";

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
  /** FU-056 Phase 3: capability snapshot for the accelerator pills
   * rendered next to each variant. Optional — pre-ready or older
   * backends collapse pills to their "available" form. */
  nativeBackends?: NativeBackendStatus;
  /** FU-056 follow-up: host platform info for hiding MLX-only /
   * CUDA-only variants on the wrong host. */
  hostSystem?: Pick<SystemStats, "platform" | "arch">;
  onActiveTabChange: (tab: TabId) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
  onDeleteImageDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
}

function imageDiscoverSortLabel(sort: DiscoverSort, t: TFunction<"library">): string {
  if (sort === "name")
    return t("imageDiscover.sortLabel.name", { defaultValue: "name" });
  if (sort === "provider")
    return t("imageDiscover.sortLabel.provider", { defaultValue: "provider" });
  if (sort === "tasks")
    return t("imageDiscover.sortLabel.tasks", { defaultValue: "tasks" });
  if (sort === "size")
    return t("imageDiscover.sortLabel.size", { defaultValue: "largest size first" });
  if (sort === "ram")
    return t("imageDiscover.sortLabel.ram", { defaultValue: "highest RAM/VRAM first" });
  if (sort === "likes")
    return t("imageDiscover.sortLabel.likes", { defaultValue: "most liked first" });
  if (sort === "downloads")
    return t("imageDiscover.sortLabel.downloads", { defaultValue: "most downloads first" });
  if (sort === "status")
    return t("imageDiscover.sortLabel.status", { defaultValue: "status" });
  return t("imageDiscover.sortLabel.release", { defaultValue: "newest released first" });
}

function sortIndicator(activeSort: DiscoverSort, sortDir: SortDir, key: DiscoverSort): string {
  if (activeSort !== key) return "";
  return sortDir === "asc" ? " ▲" : " ▼";
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

function memoryParts(label: string | null | undefined, pendingLabel: string): { primary: string; secondary: string | null } {
  if (!label) return { primary: pendingLabel, secondary: null };
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

function statusBadge(
  status: MediaStatusFilter,
  t: TFunction<"library">,
  downloadState?: DownloadStatus,
) {
  const downloadDetail = downloadState
    ? [downloadProgressLabel(downloadState), downloadSizeTooltip(downloadState)].filter(Boolean).join(" / ")
    : null;
  if (status === "installed")
    return (
      <StatusIcon
        status="installed"
        label={t("imageDiscover.status.installed", { defaultValue: "Installed" })}
      />
    );
  if (status === "downloading" && downloadState) {
    return (
      <StatusIcon
        status="downloading"
        label={t("imageDiscover.status.downloading", { defaultValue: "Downloading" })}
        detail={downloadDetail}
      />
    );
  }
  if (status === "paused" && downloadState) {
    return (
      <StatusIcon
        status="paused"
        label={t("imageDiscover.status.paused", { defaultValue: "Paused" })}
        detail={downloadDetail}
      />
    );
  }
  if (status === "failed")
    return (
      <StatusIcon
        status="failed"
        label={t("imageDiscover.status.failed", { defaultValue: "Failed" })}
        detail={
          downloadState?.error ?? t("imageDiscover.status.failedDetail", { defaultValue: "Download failed" })
        }
      />
    );
  if (status === "incomplete")
    return (
      <StatusIcon
        status="incomplete"
        label={t("imageDiscover.status.incomplete", { defaultValue: "Incomplete" })}
      />
    );
  return (
    <StatusIcon
      status="incomplete"
      label={t("imageDiscover.status.notInstalled", { defaultValue: "Not installed" })}
    />
  );
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
  nativeBackends,
  hostSystem,
  onActiveTabChange,
  onOpenImageStudio,
  onImageDownload,
  onCancelImageDownload,
  onDeleteImageDownload,
  onOpenExternalUrl,
  onRevealPath,
}: ImageDiscoverTabProps) {
  const { t } = useTranslation("common");
  const { t: tLib } = useTranslation("library");
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
        .filter(({ variant }) =>
          // FU-056 follow-up: hide mflux-runtime + LTX-2-style apple-
          // only variants on Win/Linux, nunchaku-only rows on Mac.
          // "any"-gated rows pass through (the bulk of the catalog).
          isVariantCompatibleWithHost(
            imageOrVideoVariantPlatformGate(variant),
            hostSystem,
          ),
        )
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
    [activeImageDownloads, combinedImageDiscoverResults, imageDiscoverSort, sortDir, statusFilter, hostSystem],
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

  const accessFilterLabel =
    imageDiscoverAccessFilter === "open"
      ? tLib("imageDiscover.access.openOnly", { defaultValue: "Open only" })
      : tLib("imageDiscover.access.gatedOnly", { defaultValue: "Gated only" });
  const pendingMemoryLabel = tLib("imageDiscover.memory.pending", { defaultValue: "pending" });

  return (
    <div className="image-discover-stack">
      <Panel
        title={t("tabs.imageDiscover")}
        subtitle={tLib("imageDiscover.subtitle", {
          defaultValue: "{filtered} of {total} models / live Hugging Face metadata",
          filtered: filteredResults.length,
          total: combinedImageDiscoverResults.length,
        })}
      >
        <div className="image-hero">
          <div>
            <h3>
              {tLib("imageDiscover.hero.heading", {
                defaultValue: "Browse and download image models for local generation.",
              })}
            </h3>
            <p className="muted-text">
              {tLib("imageDiscover.hero.body", {
                defaultValue:
                  "Download any model to use it in Image Studio. Runtime status lives in the Studio tab.",
              })}
            </p>
          </div>
          <div className="image-hero-actions">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-models")}>
              {tLib("imageDiscover.hero.installedModels", { defaultValue: "Installed Models" })}
            </button>
            <button className="primary-button" type="button" onClick={() => onOpenImageStudio(selectedImageVariant?.id)}>
              {tLib("imageDiscover.hero.openStudio", { defaultValue: "Open Studio" })}
            </button>
          </div>
        </div>

        <div className="image-discover-filter-row image-discover-filter-row--wide">
          <label className="image-discover-search">
            {tLib("imageDiscover.filter.search", { defaultValue: "Search" })}
            <input
              className="text-input"
              type="search"
              value={imageDiscoverSearchInput}
              onChange={(event) => onImageDiscoverSearchInputChange(event.target.value)}
              placeholder={tLib("imageDiscover.filter.searchPlaceholder", {
                defaultValue: "Search FLUX, SDXL, provider, task, tags, license...",
              })}
            />
          </label>
          <label>
            {tLib("imageDiscover.filter.task", { defaultValue: "Task" })}
            <select
              className="text-input"
              value={imageDiscoverTaskFilter}
              onChange={(event) => onImageDiscoverTaskFilterChange(event.target.value as ImageDiscoverTaskFilter)}
            >
              <option value="all">
                {tLib("imageDiscover.task.all", { defaultValue: "All tasks" })}
              </option>
              <option value="txt2img">
                {tLib("imageDiscover.task.txt2img", { defaultValue: "Text to image" })}
              </option>
              <option value="img2img">
                {tLib("imageDiscover.task.img2img", { defaultValue: "Image to image" })}
              </option>
              <option value="inpaint">
                {tLib("imageDiscover.task.inpaint", { defaultValue: "Inpaint" })}
              </option>
            </select>
          </label>
          <label>
            {tLib("imageDiscover.filter.access", { defaultValue: "Access" })}
            <select
              className="text-input"
              value={imageDiscoverAccessFilter}
              onChange={(event) => onImageDiscoverAccessFilterChange(event.target.value as ImageDiscoverAccessFilter)}
            >
              <option value="all">
                {tLib("imageDiscover.access.all", { defaultValue: "Open + gated" })}
              </option>
              <option value="open">
                {tLib("imageDiscover.access.openOnly", { defaultValue: "Open only" })}
              </option>
              <option value="gated">
                {tLib("imageDiscover.access.gatedOnly", { defaultValue: "Gated only" })}
              </option>
            </select>
          </label>
          <label>
            {tLib("imageDiscover.filter.status", { defaultValue: "Status" })}
            <select
              className="text-input"
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value as MediaStatusFilter)}
            >
              <option value="all">
                {tLib("imageDiscover.status.any", { defaultValue: "Any status" })}
              </option>
              <option value="installed">
                {tLib("imageDiscover.status.installed", { defaultValue: "Installed" })}
              </option>
              <option value="not-installed">
                {tLib("imageDiscover.status.notInstalled", { defaultValue: "Not installed" })}
              </option>
              <option value="downloading">
                {tLib("imageDiscover.status.downloading", { defaultValue: "Downloading" })}
              </option>
              <option value="paused">
                {tLib("imageDiscover.status.paused", { defaultValue: "Paused" })}
              </option>
              <option value="failed">
                {tLib("imageDiscover.status.failed", { defaultValue: "Failed" })}
              </option>
              <option value="incomplete">
                {tLib("imageDiscover.status.incomplete", { defaultValue: "Incomplete" })}
              </option>
            </select>
          </label>
          <label>
            {tLib("imageDiscover.filter.sortBy", { defaultValue: "Sort by" })}
            <select
              className="text-input"
              value={imageDiscoverSort}
              onChange={(event) => {
                const nextSort = event.target.value as DiscoverSort;
                onImageDiscoverSortChange(nextSort);
                setSortDir(defaultSortDir(nextSort));
              }}
            >
              <option value="name">
                {tLib("imageDiscover.sort.name", { defaultValue: "Name" })}
              </option>
              <option value="provider">
                {tLib("imageDiscover.sort.provider", { defaultValue: "Provider" })}
              </option>
              <option value="tasks">
                {tLib("imageDiscover.sort.tasks", { defaultValue: "Tasks" })}
              </option>
              <option value="release">
                {tLib("imageDiscover.sort.release", { defaultValue: "Newest released" })}
              </option>
              <option value="size">
                {tLib("imageDiscover.sort.size", { defaultValue: "Largest size" })}
              </option>
              <option value="ram">
                {tLib("imageDiscover.sort.ram", { defaultValue: "Highest RAM/VRAM" })}
              </option>
              <option value="likes">
                {tLib("imageDiscover.sort.likes", { defaultValue: "Most likes" })}
              </option>
              <option value="downloads">
                {tLib("imageDiscover.sort.downloads", { defaultValue: "Most downloads" })}
              </option>
              <option value="status">
                {tLib("imageDiscover.sort.status", { defaultValue: "Status" })}
              </option>
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
              {tLib("imageDiscover.filter.clear", { defaultValue: "Clear Filters" })}
            </button>
          </div>
        </div>

        <div className="image-discover-results-summary">
          <span>
            {tLib("imageDiscover.summary.count", {
              defaultValue: "{count, plural, one {# model} other {# models}} · {sortLabel}",
              count: filteredResults.length,
              sortLabel: imageDiscoverSortLabel(imageDiscoverSort, tLib),
            })}
          </span>
          {imageDiscoverSearchQuery ? (
            <span className="badge subtle">
              {tLib("imageDiscover.summary.searchBadge", {
                defaultValue: "Search: {query}",
                query: imageDiscoverSearchInput.trim(),
              })}
            </span>
          ) : null}
          {imageDiscoverTaskFilter !== "all" ? (
            <span className="badge muted">
              {tLib("imageDiscover.summary.taskBadge", {
                defaultValue: "Task: {task}",
                task: imageDiscoverTaskFilter,
              })}
            </span>
          ) : null}
          {imageDiscoverAccessFilter !== "all" ? (
            <span className="badge muted">
              {tLib("imageDiscover.summary.accessBadge", {
                defaultValue: "Access: {access}",
                access: accessFilterLabel,
              })}
            </span>
          ) : null}
          {statusFilter !== "all" ? (
            <span className="badge muted">
              {tLib("imageDiscover.summary.statusBadge", {
                defaultValue: "Status: {status}",
                status: statusFilter,
              })}
            </span>
          ) : null}
        </div>
      </Panel>

      {filteredResults.length === 0 ? (
        <Panel
          title={t("tabs.imageModels")}
          subtitle={t("panels.noModelsMatchFilters", { defaultValue: "No models match the current filters" })}
          className="image-discover-section-panel"
        >
          <div className="empty-state image-empty-state">
            <p>
              {tLib("imageDiscover.empty.body", {
                defaultValue: "Try broadening the filters or search terms.",
              })}
            </p>
          </div>
        </Panel>
      ) : (
        <div className="media-model-table media-model-table--image">
          <div className="media-model-head">
            <button className="sort-header" type="button" onClick={() => applySort("name")}>
              {tLib("imageDiscover.column.model", { defaultValue: "Model" })}
              {sortIndicator(imageDiscoverSort, sortDir, "name")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("provider")}>
              {tLib("imageDiscover.column.provider", { defaultValue: "Provider" })}
              {sortIndicator(imageDiscoverSort, sortDir, "provider")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("tasks")}>
              {tLib("imageDiscover.column.tasks", { defaultValue: "Tasks" })}
              {sortIndicator(imageDiscoverSort, sortDir, "tasks")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("size")}>
              {tLib("imageDiscover.column.size", { defaultValue: "Size" })}
              {sortIndicator(imageDiscoverSort, sortDir, "size")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("ram")}>
              {tLib("imageDiscover.column.ramVram", { defaultValue: "RAM/VRAM" })}
              {sortIndicator(imageDiscoverSort, sortDir, "ram")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("release")}>
              {tLib("imageDiscover.column.released", { defaultValue: "Released" })}
              {sortIndicator(imageDiscoverSort, sortDir, "release")}
            </button>
            <button className="sort-header" type="button" onClick={() => applySort("status")}>
              {tLib("imageDiscover.column.status", { defaultValue: "Status" })}
              {sortIndicator(imageDiscoverSort, sortDir, "status")}
            </button>
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
              const memory = memoryParts(memoryEstimate?.label, pendingMemoryLabel);
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
                          <span className="badge muted">
                            {variant.gated
                              ? tLib("imageDiscover.access.gated", { defaultValue: "Gated" })
                              : tLib("imageDiscover.access.open", { defaultValue: "Open" })}
                          </span>
                        ) : null}
                        {/* FU-061: tracked-only seeds have no Studio launchable
                            variant — surface a badge + tooltip so users know
                            why the download CTA is disabled. */}
                        {variant.trackedOnly ? (
                          <span
                            className="badge muted"
                            title={tLib("imageDiscover.trackedOnly.tooltip", {
                              defaultValue:
                                "Watching upstream — Studio playback for this family isn't wired yet. Catalog entry is for awareness; download won't unlock Studio.",
                            })}
                          >
                            {tLib("imageDiscover.trackedOnly.badge", { defaultValue: "Watching upstream" })}
                          </span>
                        ) : null}
                        {/* FU-056 Phase 3: read-only accelerator pills.
                            Click-through to install lives in Image Studio's
                            runtime banner so install state stays in one
                            place. */}
                        {getApplicableAccelerators(variant.repo).map((acceleratorId) => {
                          const meta = getAccelerator(acceleratorId);
                          if (!meta) return null;
                          return (
                            <AcceleratorCard
                              key={acceleratorId}
                              meta={meta}
                              capabilities={nativeBackends ?? null}
                              variant="pill"
                            />
                          );
                        })}
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
                        tLib("imageDiscover.memory.pendingTitle", {
                          defaultValue: "RAM/VRAM estimate pending until model weight size is known.",
                        })
                      }
                    >
                      <span>{memory.primary}</span>
                      {memory.secondary ? <small>{memory.secondary}</small> : null}
                    </span>
                    <span>
                      {releaseLabel ?? tLib("imageDiscover.unknown", { defaultValue: "Unknown" })}
                      {variant.downloadsLabel ? <small>{variant.downloadsLabel}</small> : null}
                      {variant.likesLabel ? <small>{variant.likesLabel}</small> : null}
                      {variant.license ? <small>{formatImageLicenseLabel(variant.license)}</small> : null}
                    </span>
                    <span>{statusBadge(status, tLib, downloadState)}</span>
                    <div className="media-model-actions">
                      {variant.trackedOnly ? (
                        // FU-061: tracked-only variant has no Studio playback path.
                        // Disable both Generate + Download CTAs and surface a
                        // tooltip so the user understands why.
                        <IconActionButton
                          icon="download"
                          label={tLib("imageDiscover.action.trackedOnly", {
                            defaultValue: "Tracked only — not yet launchable",
                          })}
                          disabled
                          title={tLib("imageDiscover.trackedOnly.tooltip", {
                            defaultValue:
                              "Watching upstream — Studio playback for this family isn't wired yet. Catalog entry is for awareness; download won't unlock Studio.",
                          })}
                        />
                      ) : isComplete ? (
                        <IconActionButton
                          icon="generate"
                          label={tLib("imageDiscover.action.generate", { defaultValue: "Generate" })}
                          buttonStyle="primary"
                          onClick={() => onOpenImageStudio(variant.id)}
                        />
                      ) : isDownloading ? (
                        <>
                          <IconActionButton
                            icon="pause"
                            label={tLib("imageDiscover.action.pauseDownload", { defaultValue: "Pause download" })}
                            onClick={() => onCancelImageDownload(variant.repo)}
                          />
                          <IconActionButton
                            icon="cancel"
                            label={tLib("imageDiscover.action.cancelDownload", { defaultValue: "Cancel download" })}
                            danger
                            onClick={() => onDeleteImageDownload(variant.repo)}
                          />
                        </>
                      ) : isPaused ? (
                        <>
                          <IconActionButton
                            icon="resume"
                            label={tLib("imageDiscover.action.resumeDownload", { defaultValue: "Resume download" })}
                            onClick={() => onImageDownload(variant.repo)}
                          />
                          <IconActionButton
                            icon="delete"
                            label={tLib("imageDiscover.action.deleteDownload", { defaultValue: "Delete download" })}
                            danger
                            onClick={() => onDeleteImageDownload(variant.repo)}
                          />
                        </>
                      ) : isDownloadFailed ? (
                        <>
                          <IconActionButton
                            icon="retry"
                            label={tLib("imageDiscover.action.retryDownload", { defaultValue: "Retry download" })}
                            onClick={() => onImageDownload(variant.repo)}
                          />
                          <IconActionButton
                            icon="delete"
                            label={tLib("imageDiscover.action.deleteDownload", { defaultValue: "Delete download" })}
                            danger
                            onClick={() => onDeleteImageDownload(variant.repo)}
                          />
                        </>
                      ) : (
                        <>
                          <IconActionButton
                            icon={isPartial ? "resume" : "download"}
                            label={
                              isPartial
                                ? tLib("imageDiscover.action.resumeDownload", { defaultValue: "Resume download" })
                                : tLib("imageDiscover.action.downloadModel", { defaultValue: "Download model" })
                            }
                            onClick={() => onImageDownload(variant.repo)}
                          />
                          {hasLocalData ? (
                            <IconActionButton
                              icon="delete"
                              label={tLib("imageDiscover.action.deleteModel", { defaultValue: "Delete model" })}
                              danger
                              onClick={() => onDeleteImageDownload(variant.repo)}
                            />
                          ) : null}
                        </>
                      )}
                      {variant.localPath ? (
                        <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(variant.localPath as string)} />
                      ) : null}
                      <IconActionButton
                        icon="huggingFace"
                        label={tLib("imageDiscover.action.openHuggingFace", { defaultValue: "Open on Hugging Face" })}
                        onClick={() => onOpenExternalUrl(variant.link)}
                      />
                    </div>
                  </div>
                  {isDownloadFailed && downloadState?.error ? (
                    <div className="media-model-row-detail callout error">
                      <p>{friendlyDownloadError}</p>
                      {needsGatedAccess ? (
                        <div className="button-row">
                          <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(variant.link)}>
                            {tLib("imageDiscover.action.huggingFace", { defaultValue: "Hugging Face" })}
                          </button>
                          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("settings")}>
                            {tLib("imageDiscover.action.settings", { defaultValue: "Settings" })}
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
