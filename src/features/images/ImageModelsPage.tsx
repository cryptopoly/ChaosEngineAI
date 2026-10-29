/**
 * Images > Models: one page for every image model, installed or not.
 *
 * Replaces ImageDiscoverTab + ImageModelsTab (FU-129). Built on the shared
 * Models shell (``src/components/models``) as the reference port for the
 * Video and Chat areas: this file only holds what is specific to images
 * (columns, filters, the Studio launch, gated-access help).
 */

import { useCallback, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { DownloadStatus } from "../../api";
import type { ImageModelFamily, ImageModelVariant } from "../../types";
import type { DiscoverSort, ImageDiscoverAccessFilter, ImageDiscoverTaskFilter } from "../../types/image";
import type { NativeBackendStatus } from "../../types/server";
import type { TabRequest } from "../../constants/tabRedirects";
import { IconActionButton } from "../../components/ModelActionIcons";
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
  type ConfirmFn,
  type ModelAvailabilityFilter,
  type ModelFilterGroup,
  type ModelRow,
  type ModelSortOption,
  type ModelsPageColumn,
  type SortDir,
} from "../../components/models";
import {
  buildImageEntries,
  imageEntryMatchesQuery,
  type HostLike,
  type ImageModelEntry,
} from "../../components/models/adapters/image";
import { formatImageAccessError, formatImageLicenseLabel, isGatedImageAccessError } from "../../utils";

export interface ImageModelsPageProps {
  /** Curated families (``imgState.imageCatalog``). */
  catalog: ImageModelFamily[];
  /** Tracked "latest" rows (``imgState.latestImageDiscoverResults``). */
  latest: ImageModelVariant[];
  searchInput: string;
  onSearchInputChange: (value: string) => void;
  taskFilter: ImageDiscoverTaskFilter;
  onTaskFilterChange: (value: ImageDiscoverTaskFilter) => void;
  accessFilter: ImageDiscoverAccessFilter;
  onAccessFilterChange: (value: ImageDiscoverAccessFilter) => void;
  sort: DiscoverSort;
  onSortChange: (value: DiscoverSort) => void;
  availability: ModelAvailabilityFilter;
  onAvailabilityChange: (value: ModelAvailabilityFilter) => void;
  activeImageDownloads: Record<string, DownloadStatus>;
  selectedImageVariant: ImageModelVariant | null;
  /** Repo the image runtime has in memory; marks that row "Loaded". */
  loadedImageRepo?: string | null;
  fileRevealLabel: string;
  nativeBackends?: NativeBackendStatus;
  hostSystem?: HostLike;
  onActiveTabChange: (tab: TabRequest) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onImageDownload: (repo: string) => void;
  onCancelImageDownload: (repo: string) => void;
  onDeleteImageDownload: (repo: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
  /** Injectable for tests; defaults to ``window.confirm``. */
  confirmDelete?: ConfirmFn;
}

const IMAGE_GRID =
  "minmax(220px, 1.8fr) minmax(95px, 0.7fr) minmax(100px, 0.75fr) minmax(82px, 0.55fr) " +
  "minmax(120px, 0.8fr) minmax(88px, 0.55fr) minmax(92px, 0.6fr) minmax(220px, 1fr)";

const DEFAULT_SORT: DiscoverSort = "release";

function directionLabel(dir: SortDir, t: TFunction<"library">): string {
  return dir === "asc"
    ? t("imageModels.sort.direction.ascending", { defaultValue: "ascending" })
    : t("imageModels.sort.direction.descending", { defaultValue: "descending" });
}

/** Sort options; ids match the hook's ``DiscoverSort`` so the choice
 * survives a tab switch. */
export function imageSortOptions(t: TFunction<"library">): ModelSortOption<ImageModelEntry>[] {
  const text = (left: string, right: string) => left.localeCompare(right);
  return [
    {
      id: "name",
      label: t("imageDiscover.sort.name", { defaultValue: "Name" }),
      defaultDir: "asc",
      summary: (dir) =>
        dir === "asc"
          ? t("imageModels.sort.nameAsc", { defaultValue: "name A-Z" })
          : t("imageModels.sort.nameDesc", { defaultValue: "name Z-A" }),
      compare: (a, b) => text(a.entry.name, b.entry.name),
    },
    {
      id: "provider",
      label: t("imageDiscover.sort.provider", { defaultValue: "Provider" }),
      defaultDir: "asc",
      summary: (dir) => t("imageModels.sort.providerDir", { defaultValue: "provider {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => text(a.entry.provider, b.entry.provider),
    },
    {
      id: "tasks",
      label: t("imageDiscover.sort.tasks", { defaultValue: "Tasks" }),
      defaultDir: "asc",
      summary: (dir) => t("imageModels.sort.tasksDir", { defaultValue: "tasks {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => text((a.entry.tasks ?? []).join(" "), (b.entry.tasks ?? []).join(" ")),
    },
    {
      id: "release",
      label: t("imageDiscover.sort.release", { defaultValue: "Newest released" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("imageModels.sort.newestFirst", { defaultValue: "newest released first" })
          : t("imageModels.sort.oldestFirst", { defaultValue: "oldest released first" }),
      compare: (a, b) => text(a.entry.releaseSortKey, b.entry.releaseSortKey),
    },
    {
      id: "size",
      label: t("imageDiscover.sort.size", { defaultValue: "Largest size" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("imageModels.sort.largestFirst", { defaultValue: "largest size first" })
          : t("imageModels.sort.smallestFirst", { defaultValue: "smallest size first" }),
      value: (row) => row.entry.sizeGb,
    },
    {
      id: "ram",
      label: t("imageDiscover.sort.ram", { defaultValue: "Highest RAM/VRAM" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("imageModels.sort.highestRam", { defaultValue: "highest RAM/VRAM first" })
          : t("imageModels.sort.lowestRam", { defaultValue: "lowest RAM/VRAM first" }),
      value: (row) => row.entry.memory?.peakGb,
    },
    {
      id: "likes",
      label: t("imageDiscover.sort.likes", { defaultValue: "Most likes" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("imageDiscover.sortLabel.likes", { defaultValue: "most liked first" })
          : t("models.sort.leastLiked", { defaultValue: "least liked first" }),
      value: (row) => row.entry.likes,
    },
    {
      id: "downloads",
      label: t("imageDiscover.sort.downloads", { defaultValue: "Most downloads" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("imageDiscover.sortLabel.downloads", { defaultValue: "most downloads first" })
          : t("models.sort.fewestDownloads", { defaultValue: "fewest downloads first" }),
      value: (row) => row.entry.downloads,
    },
    {
      id: "status",
      label: t("imageDiscover.sort.status", { defaultValue: "Status" }),
      defaultDir: "asc",
      summary: (dir) => t("imageModels.sort.statusDir", { defaultValue: "status {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => statusSortRank(a.status) - statusSortRank(b.status),
    },
  ];
}

function defaultDirFor(sort: DiscoverSort): SortDir {
  return sort === "name" || sort === "provider" || sort === "tasks" || sort === "status" ? "asc" : "desc";
}

export function ImageModelsPage({
  catalog,
  latest,
  searchInput,
  onSearchInputChange,
  taskFilter,
  onTaskFilterChange,
  accessFilter,
  onAccessFilterChange,
  sort,
  onSortChange,
  availability,
  onAvailabilityChange,
  activeImageDownloads,
  selectedImageVariant,
  loadedImageRepo,
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
  confirmDelete,
}: ImageModelsPageProps) {
  const { t } = useTranslation("library");
  const { t: tCommon } = useTranslation("common");
  const [sortDir, setSortDir] = useState<SortDir>(() => defaultDirFor(sort));
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  const entries = useMemo(
    () => buildImageEntries(catalog, latest, { host: hostSystem, loadedRepo: loadedImageRepo }),
    [catalog, latest, hostSystem, loadedImageRepo],
  );
  const familyById = useMemo(() => new Map(catalog.map((family) => [family.id, family])), [catalog]);
  const matches = useCallback(
    (entry: ImageModelEntry, query: string) => imageEntryMatchesQuery(entry, query, familyById),
    [familyById],
  );
  const resolveStatus = useCallback(
    (entry: ImageModelEntry) => resolveModelStatus(entry, { download: activeImageDownloads[entry.downloadKey] }),
    [activeImageDownloads],
  );
  const sortOptions = useMemo(() => imageSortOptions(t), [t]);

  const groups: ModelFilterGroup<ImageModelEntry>[] = [
    {
      id: "task",
      label: t("imageDiscover.filter.task", { defaultValue: "Task" }),
      kind: "select",
      value: taskFilter,
      defaultValue: "all",
      options: [
        { value: "all", label: t("imageDiscover.task.all", { defaultValue: "All tasks" }) },
        { value: "txt2img", label: t("imageDiscover.task.txt2img", { defaultValue: "Text to image" }) },
        { value: "img2img", label: t("imageDiscover.task.img2img", { defaultValue: "Image to image" }) },
        { value: "inpaint", label: t("imageDiscover.task.inpaint", { defaultValue: "Inpaint" }) },
      ],
      onChange: (value) => onTaskFilterChange(value as ImageDiscoverTaskFilter),
      predicate: (entry, value) => (entry.tasks ?? []).includes(value),
    },
    {
      id: "access",
      label: t("imageDiscover.filter.access", { defaultValue: "Access" }),
      kind: "select",
      value: accessFilter,
      defaultValue: "all",
      options: [
        { value: "all", label: t("imageDiscover.access.all", { defaultValue: "Open + gated" }) },
        { value: "open", label: t("imageDiscover.access.openOnly", { defaultValue: "Open only" }) },
        { value: "gated", label: t("imageDiscover.access.gatedOnly", { defaultValue: "Gated only" }) },
      ],
      onChange: (value) => onAccessFilterChange(value as ImageDiscoverAccessFilter),
      predicate: (entry, value) => (value === "gated" ? entry.gated === true : entry.gated !== true),
    },
  ];

  const columns: ModelsPageColumn[] = [
    { id: "model", label: t("imageDiscover.column.model", { defaultValue: "Model" }), sortId: "name" },
    { id: "provider", label: t("imageDiscover.column.provider", { defaultValue: "Provider" }), sortId: "provider" },
    { id: "tasks", label: t("imageDiscover.column.tasks", { defaultValue: "Tasks" }), sortId: "tasks" },
    { id: "size", label: t("imageDiscover.column.size", { defaultValue: "Size" }), sortId: "size" },
    { id: "ram", label: t("imageDiscover.column.ramVram", { defaultValue: "RAM/VRAM" }), sortId: "ram" },
    { id: "released", label: t("imageDiscover.column.released", { defaultValue: "Released" }), sortId: "release" },
    { id: "status", label: t("imageDiscover.column.status", { defaultValue: "Status" }), sortId: "status" },
  ];

  const pendingLabel = t("imageDiscover.memory.pending", { defaultValue: "pending" });
  const pendingTitle = t("imageDiscover.memory.pendingTitle", {
    defaultValue: "RAM/VRAM estimate pending until model weight size is known.",
  });
  const trackedOnlyTitle = t("imageDiscover.trackedOnly.tooltip", {
    defaultValue:
      "Watching upstream — Studio playback for this family isn't wired yet. Catalog entry is for awareness; download won't unlock Studio.",
  });

  function toggleExpanded(id: string) {
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function renderRow({ entry, status }: ModelRow<ImageModelEntry>) {
    const variant = entry.source;
    const download = status.download;
    const hasLocalFiles = status.bucket !== "available";
    const isExpanded = expanded.has(entry.id);
    const sharedTitle = entry.shared
      ? t("imageModels.sharedRepoTitle", {
          defaultValue:
            "Uses the same download as {count} other variant(s) of {repo}. Deleting this row removes those files too.",
          count: entry.shared.count,
          repo: entry.shared.key,
        })
      : undefined;
    const license = formatImageLicenseLabel(entry.license);
    const friendlyError = formatImageAccessError(download?.error, variant);
    const needsGatedAccess = isGatedImageAccessError(download?.error);

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
            {t("imageModels.sharedRepo", { defaultValue: "shares storage · +{count}", count: entry.shared.count })}
          </small>
        ) : null}
      </>,
      <ModelMemoryCell memory={entry.memory} fit={entry.fit} pendingLabel={pendingLabel} pendingTitle={pendingTitle} />,
      <>
        {entry.releaseLabel ?? t("imageDiscover.unknown", { defaultValue: "Unknown" })}
        {entry.downloadsLabel ? <small>{entry.downloadsLabel}</small> : null}
        {entry.likesLabel ? <small>{entry.likesLabel}</small> : null}
        {license ? <small>{license}</small> : null}
      </>,
    ];

    const actions = (
      <ModelDownloadControls
        status={status}
        trackedOnly={entry.trackedOnly}
        trackedOnlyTitle={trackedOnlyTitle}
        onDownload={() => onImageDownload(entry.downloadKey)}
        onPause={() => onCancelImageDownload(download?.repo ?? entry.downloadKey)}
        onCancel={() => onDeleteImageDownload(entry.deleteTarget)}
        onDeleteModel={() => {
          confirmAndDelete(entry, onDeleteImageDownload, confirmDelete);
        }}
        launch={{
          id: "generate",
          icon: "generate",
          label: t("imageDiscover.action.generate", { defaultValue: "Generate" }),
          onClick: () => onOpenImageStudio(entry.id),
          disabled: !entry.hostCompatible,
          title: entry.hostCompatible
            ? undefined
            : t("models.card.incompatible", { defaultValue: "Cannot run on this machine" }),
        }}
        trailing={
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
              <IconActionButton
                icon="huggingFace"
                label={t("imageDiscover.action.openHuggingFace", { defaultValue: "Open on Hugging Face" })}
                onClick={() => onOpenExternalUrl(entry.link as string)}
              />
            ) : null}
          </>
        }
      />
    );

    const detail = (
      <>
        {entry.note ? <p className="models-card__note">{entry.note}</p> : null}
        <ModelDetailList
          items={[
            { label: t("models.detail.repo", { defaultValue: "Repository" }), value: <code>{entry.repo}</code> },
            variant.runtime ? { label: t("models.detail.engine", { defaultValue: "Engine" }), value: variant.runtime } : null,
            variant.recommendedResolution
              ? { label: t("models.detail.resolution", { defaultValue: "Recommended size" }), value: variant.recommendedResolution }
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
            variant.metadataWarning
              ? { label: t("models.detail.warning", { defaultValue: "Note" }), value: variant.metadataWarning }
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

    const callouts =
      status.state === "failed" && download?.error ? (
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
      ) : null;

    return (
      <ModelCard
        key={entry.id}
        entry={entry}
        status={status}
        badges={badges}
        cells={cells}
        actions={actions}
        detail={detail}
        expanded={isExpanded}
        onToggleExpanded={() => toggleExpanded(entry.id)}
        callouts={callouts}
      />
    );
  }

  return (
    <ModelsPage<ImageModelEntry>
      className="image-models-page"
      title={tCommon("tabs.imageModels", { defaultValue: "Image Models" })}
      headerActions={
        <button className="primary-button" type="button" onClick={() => onOpenImageStudio(selectedImageVariant?.id)}>
          {t("imageDiscover.hero.openStudio", { defaultValue: "Open Studio" })}
        </button>
      }
      intro={
        <p className="muted-text">
          {t("models.image.intro", {
            defaultValue:
              "Every image model in one list. Download one to use it in Image Studio; filter by what is installed, available or downloading.",
          })}
        </p>
      }
      entries={entries}
      resolveStatus={resolveStatus}
      search={{
        value: searchInput,
        onChange: onSearchInputChange,
        placeholder: t("imageDiscover.filter.searchPlaceholder", {
          defaultValue: "Search FLUX, SDXL, provider, task, tags, license...",
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
        onAccessFilterChange("all");
      }}
      columns={columns}
      gridTemplate={IMAGE_GRID}
      renderRow={renderRow}
      emptyCatalog={t("models.image.emptyCatalog", {
        defaultValue: "No image models in the catalog yet. Check that the backend is running.",
      })}
      emptyByAvailability={{
        installed: t("models.image.emptyInstalled", {
          defaultValue: "No image models installed yet. Pick one from the available models to download it.",
        }),
        downloading: t("models.image.emptyDownloading", { defaultValue: "No image model is downloading." }),
        attention: t("models.image.emptyAttention", { defaultValue: "No image model needs attention." }),
      }}
    />
  );
}
