/**
 * Video > Models: one page for every video model, installed or not.
 *
 * Replaces VideoDiscoverTab + VideoModelsTab (FU-129) on the shared Models
 * shell (``src/components/models``), following ``ImageModelsPage``. What is
 * specific to video lives here: the download keys and delete targets
 * (R12, see ``adapters/video.ts``), the GGUF-first size sort (R13), the
 * "Loaded in memory" row, and LongLive, which installs through its own
 * installer and reports readiness through a runtime probe (R25).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { DownloadStatus, InstallResult, LongLiveJobState } from "../../api";
import type { VideoModelFamily, VideoModelVariant, VideoRuntimeStatus } from "../../types";
import type { DiscoverSort } from "../../types/image";
import type { VideoDiscoverTaskFilter } from "../../types/video";
import type { NativeBackendStatus } from "../../types/server";
import type { TabRequest } from "../../constants/tabRedirects";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import { InstallLogPanel } from "../../components/InstallLogPanel";
import {
  AcceleratorPills,
  ModelCard,
  ModelDetailList,
  ModelDownloadControls,
  ModelMemoryCell,
  ModelsPage,
  confirmAndDelete,
  fitLabel,
  resolveModelStatus,
  statusSortRank,
  type AcceleratorHost,
  type ConfirmFn,
  type ModelAvailabilityFilter,
  type ModelFilterGroup,
  type ModelRow,
  type ModelSortOption,
  type ModelsPageColumn,
  type SortDir,
} from "../../components/models";
import {
  buildVideoEntries,
  buildVideoRepoUsers,
  hasVisibleLongLiveRow,
  videoDeleteConfirmEntry,
  videoEntryMatchesQuery,
  videoRowActions,
  type VideoModelEntry,
} from "../../components/models/adapters/video";
import {
  formatImageAccessError,
  formatImageLicenseLabel,
  isGatedImageAccessError,
  type VideoDeleteRepoKind,
} from "../../utils";

export interface VideoModelsPageProps {
  /** Curated families (``videoState.videoCatalog``). */
  catalog: VideoModelFamily[];
  /** Tracked "latest" rows (``videoState.latestVideoDiscoverResults``). */
  latest: VideoModelVariant[];
  searchInput: string;
  onSearchInputChange: (value: string) => void;
  taskFilter: VideoDiscoverTaskFilter;
  onTaskFilterChange: (value: VideoDiscoverTaskFilter) => void;
  sort: DiscoverSort;
  onSortChange: (value: DiscoverSort) => void;
  availability: ModelAvailabilityFilter;
  onAvailabilityChange: (value: ModelAvailabilityFilter) => void;
  /** Keyed by variant id first, then by repo (``handleVideoDownload``). */
  activeVideoDownloads: Record<string, DownloadStatus>;
  selectedVideoVariant: VideoModelVariant | null;
  /** Id of the variant the video runtime has in memory; marks that row
   * "Loaded in memory". */
  loadedVideoVariantId?: string | null;
  fileRevealLabel: string;
  nativeBackends?: NativeBackendStatus;
  hostSystem?: AcceleratorHost;
  longLiveStatus: VideoRuntimeStatus | null;
  installingLongLive: boolean;
  longLiveJob: LongLiveJobState | null;
  onActiveTabChange: (tab: TabRequest) => void;
  onOpenVideoStudio: (modelId?: string) => void;
  onVideoDownload: (repo: string, modelId?: string) => void;
  onCancelVideoDownload: (repo: string) => void;
  onDeleteVideoDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
  onRefreshLongLiveStatus: () => void;
  onInstallLongLive: () => Promise<InstallResult>;
  /** Injectable for tests; defaults to ``window.confirm``. */
  confirmDelete?: ConfirmFn;
}

const VIDEO_GRID =
  "minmax(220px, 1.7fr) minmax(95px, 0.9fr) minmax(100px, 0.85fr) minmax(82px, 0.65fr) " +
  "minmax(120px, 1fr) minmax(88px, 0.8fr) minmax(88px, 0.4fr) 204px";
// Below 1440 px the Provider and Tasks columns (2, 3) are left out so the rest fits at 1100 px.
const VIDEO_GRID_NARROW =
  "minmax(190px, 1.6fr) minmax(76px, 0.7fr) minmax(112px, 0.9fr) minmax(100px, 0.75fr) " +
  "minmax(72px, 0.4fr) 188px";

const DEFAULT_SORT: DiscoverSort = "release";

function directionLabel(dir: SortDir, t: TFunction<"library">): string {
  return dir === "asc"
    ? t("videoModels.sort.direction.ascending", { defaultValue: "ascending" })
    : t("videoModels.sort.direction.descending", { defaultValue: "descending" });
}

/** Sort options; ids match the hook's ``DiscoverSort`` so the choice
 * survives a tab switch. Size reads the GGUF file first (R13). */
export function videoSortOptions(t: TFunction<"library">): ModelSortOption<VideoModelEntry>[] {
  const text = (left: string, right: string) => left.localeCompare(right);
  return [
    {
      id: "name",
      label: t("videoDiscover.sort.name", { defaultValue: "Name" }),
      defaultDir: "asc",
      summary: (dir) =>
        dir === "asc"
          ? t("videoModels.sort.nameAsc", { defaultValue: "name A-Z" })
          : t("videoModels.sort.nameDesc", { defaultValue: "name Z-A" }),
      compare: (a, b) => text(a.entry.name, b.entry.name),
    },
    {
      id: "provider",
      label: t("videoDiscover.sort.provider", { defaultValue: "Provider" }),
      defaultDir: "asc",
      summary: (dir) => t("videoModels.sort.providerDir", { defaultValue: "provider {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => text(a.entry.provider, b.entry.provider),
    },
    {
      id: "tasks",
      label: t("videoDiscover.sort.tasks", { defaultValue: "Tasks" }),
      defaultDir: "asc",
      summary: (dir) => t("videoModels.sort.tasksDir", { defaultValue: "tasks {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => text((a.entry.tasks ?? []).join(" "), (b.entry.tasks ?? []).join(" ")),
    },
    {
      id: "release",
      label: t("videoDiscover.sort.release", { defaultValue: "Newest released" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("videoModels.sort.newestFirst", { defaultValue: "newest released first" })
          : t("videoModels.sort.oldestFirst", { defaultValue: "oldest released first" }),
      compare: (a, b) => text(a.entry.releaseSortKey, b.entry.releaseSortKey),
    },
    {
      id: "size",
      label: t("videoDiscover.sort.size", { defaultValue: "Largest size" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("videoModels.sort.largestFirst", { defaultValue: "largest size first" })
          : t("videoModels.sort.smallestFirst", { defaultValue: "smallest size first" }),
      value: (row) => row.entry.sizeGb,
    },
    {
      id: "ram",
      label: t("videoDiscover.sort.ram", { defaultValue: "Highest RAM/VRAM" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("videoModels.sort.highestRam", { defaultValue: "highest RAM/VRAM first" })
          : t("videoModels.sort.lowestRam", { defaultValue: "lowest RAM/VRAM first" }),
      value: (row) => row.entry.memory?.peakGb,
    },
    {
      id: "likes",
      label: t("videoDiscover.sort.likes", { defaultValue: "Most likes" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("videoDiscover.sortLabel.likes", { defaultValue: "most liked first" })
          : t("models.sort.leastLiked", { defaultValue: "least liked first" }),
      value: (row) => row.entry.likes,
    },
    {
      id: "downloads",
      label: t("videoDiscover.sort.downloads", { defaultValue: "Most downloads" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("videoDiscover.sortLabel.downloads", { defaultValue: "most downloads first" })
          : t("models.sort.fewestDownloads", { defaultValue: "fewest downloads first" }),
      value: (row) => row.entry.downloads,
    },
    {
      id: "status",
      label: t("videoDiscover.sort.status", { defaultValue: "Status" }),
      defaultDir: "asc",
      summary: (dir) => t("videoModels.sort.statusDir", { defaultValue: "status {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => statusSortRank(a.status) - statusSortRank(b.status),
    },
  ];
}

function defaultDirFor(sort: DiscoverSort): SortDir {
  return sort === "name" || sort === "provider" || sort === "tasks" || sort === "status" ? "asc" : "desc";
}

/** Label of the delete button: a shared GGUF / text encoder / component
 * download says so, otherwise ``fallback`` (Delete model / download). */
export function videoDeleteLabel(kind: VideoDeleteRepoKind, fallback: string, t: TFunction<"library">): string {
  switch (kind) {
    case "gguf":
      return t("videoModels.action.deleteSharedGguf", { defaultValue: "Delete shared GGUF download" });
    case "textEncoder":
      return t("videoModels.action.deleteSharedTextEncoder", { defaultValue: "Delete shared text encoder download" });
    case "component":
      return t("videoModels.action.deleteSharedComponent", { defaultValue: "Delete shared component download" });
    default:
      return fallback;
  }
}

export function VideoModelsPage({
  catalog,
  latest,
  searchInput,
  onSearchInputChange,
  taskFilter,
  onTaskFilterChange,
  sort,
  onSortChange,
  availability,
  onAvailabilityChange,
  activeVideoDownloads,
  selectedVideoVariant,
  loadedVideoVariantId,
  fileRevealLabel,
  nativeBackends,
  hostSystem,
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
  confirmDelete,
}: VideoModelsPageProps) {
  const { t } = useTranslation("library");
  const { t: tCommon } = useTranslation("common");
  const [sortDir, setSortDir] = useState<SortDir>(() => defaultDirFor(sort));
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const longLiveReady = longLiveStatus?.realGenerationAvailable ?? false;

  const entries = useMemo(
    () =>
      buildVideoEntries(catalog, latest, {
        host: hostSystem,
        loadedVariantId: loadedVideoVariantId,
        longLiveReady,
      }),
    [catalog, latest, hostSystem, loadedVideoVariantId, longLiveReady],
  );
  const familyById = useMemo(() => new Map(catalog.map((family) => [family.id, family])), [catalog]);
  const repoUsers = useMemo(() => buildVideoRepoUsers(entries), [entries]);

  // R25: probe the LongLive runtime once per visit when a LongLive row can
  // show on this host (CUDA only). The callback is read through a ref so a
  // new function from App on every render does not probe again.
  const refreshLongLiveRef = useRef(onRefreshLongLiveStatus);
  useEffect(() => {
    refreshLongLiveRef.current = onRefreshLongLiveStatus;
  });
  const longLiveVisible = hasVisibleLongLiveRow(entries);
  useEffect(() => {
    if (longLiveVisible) refreshLongLiveRef.current();
  }, [longLiveVisible]);

  const matches = useCallback(
    (entry: VideoModelEntry, query: string) => videoEntryMatchesQuery(entry, query, familyById),
    [familyById],
  );
  const resolveStatus = useCallback(
    (entry: VideoModelEntry) =>
      entry.longLive
        ? // LongLive has no download: installing (or not) is all there is.
          resolveModelStatus(entry, { installing: installingLongLive && !longLiveReady })
        : resolveModelStatus(entry, { download: activeVideoDownloads[entry.downloadKey] }),
    [activeVideoDownloads, installingLongLive, longLiveReady],
  );
  const sortOptions = useMemo(() => videoSortOptions(t), [t]);

  const groups: ModelFilterGroup<VideoModelEntry>[] = [
    {
      id: "task",
      label: t("videoDiscover.filter.task", { defaultValue: "Task" }),
      kind: "select",
      value: taskFilter,
      defaultValue: "all",
      options: [
        { value: "all", label: t("videoDiscover.task.all", { defaultValue: "All tasks" }) },
        { value: "txt2video", label: t("videoDiscover.task.txt2video", { defaultValue: "Text to video" }) },
        { value: "img2video", label: t("videoDiscover.task.img2video", { defaultValue: "Image to video" }) },
        { value: "video2video", label: t("videoDiscover.task.video2video", { defaultValue: "Video to video" }) },
      ],
      onChange: (value) => onTaskFilterChange(value as VideoDiscoverTaskFilter),
      predicate: (entry, value) => (entry.tasks ?? []).includes(value),
    },
  ];

  const columns: ModelsPageColumn[] = [
    { id: "model", label: t("videoDiscover.column.model", { defaultValue: "Model" }), sortId: "name" },
    { id: "provider", label: t("videoDiscover.column.provider", { defaultValue: "Provider" }), sortId: "provider" },
    { id: "tasks", label: t("videoDiscover.column.tasks", { defaultValue: "Tasks" }), sortId: "tasks" },
    { id: "size", label: t("videoDiscover.column.size", { defaultValue: "Size" }), sortId: "size" },
    { id: "ram", label: t("videoDiscover.column.ramVram", { defaultValue: "RAM/VRAM" }), sortId: "ram" },
    { id: "released", label: t("videoDiscover.column.released", { defaultValue: "Released" }), sortId: "release" },
    { id: "status", label: t("videoDiscover.column.status", { defaultValue: "Status" }), sortId: "status" },
  ];

  const pendingLabel = t("videoDiscover.memory.pending", { defaultValue: "pending" });
  const pendingTitle = t("videoDiscover.memory.pendingTitle", {
    defaultValue: "RAM/VRAM estimate pending until model weight size is known.",
  });
  const trackedOnlyTitle = t("imageDiscover.trackedOnly.tooltip", {
    defaultValue:
      "Watching upstream — Studio playback for this family isn't wired yet. Catalog entry is for awareness; download won't unlock Studio.",
  });
  const deleteModelLabel = t("models.action.deleteModel", { defaultValue: "Delete model" });
  const deleteDownloadLabel = t("models.action.deleteDownload", { defaultValue: "Delete download" });
  const generateLabel = t("videoDiscover.action.generate", { defaultValue: "Generate" });
  const incompatibleLabel = t("models.card.incompatible", { defaultValue: "Cannot run on this machine" });
  const modelCardLabel = t("videoDiscover.action.openModelCard", { defaultValue: "Open model card" });

  function toggleExpanded(id: string) {
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function renderRow({ entry, status }: ModelRow<VideoModelEntry>) {
    const variant = entry.source;
    const download = status.download;
    const installed = status.state === "installed";
    const hasLocalFiles = status.bucket !== "available";
    const isExpanded = expanded.has(entry.id);
    const rowActions = videoRowActions(entry, download, {
      onVideoDownload,
      onCancelVideoDownload,
      onDeleteVideoDownload,
    });
    const sharedTitle = entry.shared
      ? t("videoModels.sharedRepoTitle", {
          defaultValue:
            "Uses the same download as {count} other variant(s) of {repo}. Deleting this row removes those files too.",
          count: entry.shared.count,
          repo: entry.shared.key,
        })
      : undefined;
    const license = formatImageLicenseLabel(entry.license);
    const friendlyError = formatImageAccessError(download?.error, variant);
    const needsGatedAccess = isGatedImageAccessError(download?.error);
    const longLiveInstalling = entry.longLive && status.state === "downloading";

    const badges = (
      <>
        {(entry.styleTags ?? []).slice(0, 4).map((tag) => (
          <span key={tag} className="badge subtle">{tag}</span>
        ))}
        {typeof entry.gated === "boolean" ? (
          <span className="badge muted">
            {entry.gated
              ? t("imageDiscover.access.gated", { defaultValue: "Gated" })
              : t("imageDiscover.access.open", { defaultValue: "Open" })}
          </span>
        ) : null}
        {entry.trackedOnly ? (
          <span className="badge muted" title={trackedOnlyTitle}>
            {t("imageDiscover.trackedOnly.badge", { defaultValue: "Watching upstream" })}
          </span>
        ) : null}
        <AcceleratorPills ids={entry.accelerators} capabilities={nativeBackends ?? null} />
      </>
    );

    const cells = [
      <>{entry.provider}</>,
      <span className="media-model-chip-row">
        {(entry.tasks ?? []).map((task) => (
          <span key={task} className="badge muted">{task}</span>
        ))}
      </span>,
      <>
        <span title={entry.sizeTitle ?? undefined}>{entry.sizeLabel}</span>
        {entry.shared && hasLocalFiles ? (
          <small className="shared-repo-badge" title={sharedTitle}>
            {t("videoModels.sharedRepo", { defaultValue: "shares storage · +{count}", count: entry.shared.count })}
          </small>
        ) : null}
      </>,
      <ModelMemoryCell memory={entry.memory} fit={entry.fit} pendingLabel={pendingLabel} pendingTitle={pendingTitle} />,
      <>
        {entry.releaseLabel ?? t("videoDiscover.unknown", { defaultValue: "Unknown" })}
        {entry.downloadsLabel ? <small>{entry.downloadsLabel}</small> : null}
        {entry.likesLabel ? <small>{entry.likesLabel}</small> : null}
        {license ? <small>{license}</small> : null}
      </>,
    ];

    const launch = {
      id: "generate",
      icon: "generate" as const,
      label: generateLabel,
      onClick: () => onOpenVideoStudio(entry.id),
      disabled: !entry.hostCompatible,
      title: entry.hostCompatible ? undefined : incompatibleLabel,
    };

    const trailing = (
      <>
        {entry.localPath ? (
          <IconActionButton
            icon="reveal"
            label={fileRevealLabel}
            title={fileRevealLabel}
            onClick={() => onRevealPath(entry.localPath as string)}
          />
        ) : null}
        {entry.link ? (
          <IconActionButton icon="modelCard" label={modelCardLabel} onClick={() => onOpenExternalUrl(entry.link as string)} />
        ) : null}
      </>
    );

    // LongLive installs through its own installer: Generate once the
    // runtime probe says it is ready, otherwise Install. Nothing to pause,
    // cancel or delete (its files live outside the Hugging Face cache).
    const actions = entry.longLive ? (
      <div className="media-model-actions model-download-controls">
        {installed ? (
          <IconActionButton
            icon="generate"
            label={launch.label}
            title={launch.title}
            buttonStyle="primary"
            disabled={launch.disabled}
            onClick={launch.onClick}
          />
        ) : (
          <IconActionButton
            icon="install"
            label={
              installingLongLive
                ? t("videoDiscover.action.installing", { defaultValue: "Installing" })
                : t("videoDiscover.action.install", { defaultValue: "Install" })
            }
            onClick={() => void onInstallLongLive()}
            disabled={installingLongLive}
          />
        )}
        {trailing}
      </div>
    ) : (
      <ModelDownloadControls
        status={status}
        trackedOnly={entry.trackedOnly}
        trackedOnlyTitle={trackedOnlyTitle}
        labels={{
          deleteModel: videoDeleteLabel(rowActions.deleteKind, deleteModelLabel, t),
          deleteDownload: videoDeleteLabel(rowActions.deleteKind, deleteDownloadLabel, t),
        }}
        onDownload={rowActions.download}
        onPause={rowActions.pause}
        onCancel={rowActions.cancel}
        onDeleteModel={() => {
          confirmAndDelete(
            videoDeleteConfirmEntry(entry, rowActions.deleteRepo, repoUsers),
            onDeleteVideoDownload,
            confirmDelete,
          );
        }}
        launch={launch}
        trailing={trailing}
      />
    );

    const statusCell = longLiveInstalling ? (
      <StatusIcon status="downloading" label={t("videoDiscover.status.installing", { defaultValue: "Installing" })} />
    ) : undefined;

    const detail = () => (
      <>
        {entry.note ? <p className="models-card__note">{entry.note}</p> : null}
        <ModelDetailList
          items={[
            { label: t("models.detail.repo", { defaultValue: "Repository" }), value: <code>{entry.repo}</code> },
            variant.ggufRepo && variant.ggufFile
              ? {
                  label: t("models.detail.ggufFile", { defaultValue: "GGUF file" }),
                  value: <code>{`${variant.ggufRepo}/${variant.ggufFile}`}</code>,
                }
              : null,
            variant.textEncoderRepo
              ? {
                  label: t("models.detail.textEncoder", { defaultValue: "Text encoder" }),
                  value: <code>{variant.textEncoderRepo}</code>,
                }
              : null,
            variant.runtime ? { label: t("models.detail.engine", { defaultValue: "Engine" }), value: variant.runtime } : null,
            variant.recommendedResolution
              ? { label: t("models.detail.resolution", { defaultValue: "Recommended size" }), value: variant.recommendedResolution }
              : null,
            typeof variant.sizeGb === "number" && variant.sizeGb > 0
              ? {
                  label: t("models.detail.catalogSize", { defaultValue: "Catalog size" }),
                  value: `${variant.sizeGb.toFixed(1)} GB`,
                }
              : null,
            license ? { label: t("models.detail.license", { defaultValue: "Licence" }), value: license } : null,
            entry.fit
              ? {
                  label: t("models.detail.fit", { defaultValue: "Fit on this machine" }),
                  value: `${fitLabel(entry.fit, t)}${entry.fit.reason ? ` · ${entry.fit.reason}` : ""} (${
                    entry.fit.basis === "measured"
                      ? t("models.fit.measured", { defaultValue: "measured" })
                      : t("models.fit.estimated", { defaultValue: "estimated" })
                  })`,
                }
              : null,
            entry.shared
              ? {
                  label: t("models.detail.sharedWith", { defaultValue: "Shares its download with" }),
                  value: entry.shared.siblingNames.join(", ") || entry.shared.key,
                }
              : null,
            entry.localPath
              ? { label: t("models.detail.localPath", { defaultValue: "On disk" }), value: <code>{entry.localPath}</code> }
              : null,
          ]}
        />
        {entry.link ? (
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(entry.link as string)}>
              {t("models.detail.openModelCard", { defaultValue: "Open model card" })}
            </button>
          </div>
        ) : null}
      </>
    );

    let callouts = null;
    if (entry.longLive && !installed) {
      callouts = (
        <>
          <div className="media-model-row-detail callout quiet">
            <p>
              {t("videoDiscover.longLive.installNote", {
                defaultValue:
                  "LongLive installs into an isolated venv at ~/.chaosengine/longlive. CUDA only, 5-15 min depending on network.",
              })}
            </p>
          </div>
          <InstallLogPanel job={longLiveJob} variant="longlive" />
        </>
      );
    } else if (status.state === "failed" && download?.error) {
      callouts = (
        <div className="media-model-row-detail callout error">
          <p>{friendlyError}</p>
          {needsGatedAccess ? (
            <div className="button-row">
              <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(variant.link)}>
                {t("imageDiscover.action.huggingFace", { defaultValue: "Hugging Face" })}
              </button>
              <button className="secondary-button" type="button" onClick={() => onActiveTabChange("settings")}>
                {t("imageDiscover.action.settings", { defaultValue: "Settings" })}
              </button>
            </div>
          ) : null}
        </div>
      );
    } else if (!installed && status.state !== "downloading" && variant.localStatusReason) {
      callouts = (
        <div className="media-model-row-detail callout quiet">
          <p>{variant.localStatusReason}</p>
        </div>
      );
    }

    return (
      <ModelCard
        key={entry.id}
        entry={entry}
        status={status}
        badges={badges}
        cells={cells}
        statusCell={statusCell}
        actions={actions}
        detail={detail}
        expanded={isExpanded}
        onToggleExpanded={() => toggleExpanded(entry.id)}
        callouts={callouts}
      />
    );
  }

  return (
    <ModelsPage<VideoModelEntry>
      className="video-models-page"
      title={tCommon("tabs.videoModels", { defaultValue: "Video Models" })}
      headerActions={
        <button className="primary-button" type="button" onClick={() => onOpenVideoStudio(selectedVideoVariant?.id)}>
          {t("videoDiscover.hero.openStudio", { defaultValue: "Open Studio" })}
        </button>
      }
      intro={
        <p className="muted-text">
          {t("models.video.intro", {
            defaultValue:
              "Every video model in one list. Download one to use it in Video Studio; filter by what is installed, available or downloading.",
          })}
        </p>
      }
      entries={entries}
      resolveStatus={resolveStatus}
      search={{
        value: searchInput,
        onChange: onSearchInputChange,
        placeholder: t("videoDiscover.filter.searchPlaceholder", {
          defaultValue: "Search LTX, Wan, Mochi, provider, tags...",
        }),
      }}
      matches={matches}
      groups={groups}
      availability={availability}
      onAvailabilityChange={onAvailabilityChange}
      sortOptions={sortOptions}
      sortId={sort}
      sortDir={sortDir}
      defaultSortId={DEFAULT_SORT}
      onSortChange={(id, dir) => {
        onSortChange(id as DiscoverSort);
        setSortDir(dir);
      }}
      onClearFilters={() => {
        onSearchInputChange("");
        onTaskFilterChange("all");
      }}
      columns={columns}
      gridTemplate={VIDEO_GRID}
      narrowGridTemplate={VIDEO_GRID_NARROW}
      narrowHide={[2, 3]}
      renderRow={renderRow}
      emptyCatalog={t("models.video.emptyCatalog", {
        defaultValue: "No video models in the catalog yet. Check that the backend is running.",
      })}
      emptyByAvailability={{
        installed: t("models.video.emptyInstalled", {
          defaultValue: "No video models installed yet. Pick one from the available models to download it.",
        }),
        downloading: t("models.video.emptyDownloading", { defaultValue: "No video model is downloading." }),
        attention: t("models.video.emptyAttention", { defaultValue: "No video model needs attention." }),
      }}
    />
  );
}
