/**
 * Chat > Models: every language model in one list, on disk or not.
 *
 * Replaces OnlineModelsTab (Discover) + MyModelsTab (My Models), FU-129.
 * Built on the shared Models shell (``src/components/models``) like
 * Images > Models. The join of catalog and disk scan lives in
 * ``adapters/chat.ts`` (rule at the top of that file); this file holds
 * what is chat-only: the columns, the filters, the launch actions
 * (new thread / use in this chat / load for server, Convert), favourites,
 * and the Hugging Face search results under the table.
 */

import { useCallback, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { DownloadStatus } from "../../api";
import type { HubFileListResponse, HubModel, LibraryItem, ModelFamily, ModelVariant } from "../../types";
import { CAPABILITY_META } from "../../constants";
import { IconActionButton } from "../../components/ModelActionIcons";
import { RunFromHuggingFace } from "../../components/RunFromHuggingFace";
import { ImportModelsPanel } from "../../components/ImportModelsPanel";
import { isMemorySaverStrategy } from "../../components/runtimeSupport";
import {
  FitBadge,
  ModelCard,
  ModelDetailList,
  ModelDownloadControls,
  ModelsPage,
  fitLabel,
  isVisibleOnHost,
  resolveModelStatus,
  statusSortRank,
  type ModelAvailabilityFilter,
  type ModelFilterGroup,
  type ModelLaunchAction,
  type ModelRow,
  type ModelSortOption,
  type ModelsPageColumn,
  type SortDir,
} from "../../components/models";
import {
  CHAT_ACCEL_FILTERS,
  CHAT_MODELS_DEFAULT_SORT,
  buildChatEntries,
  chatContextSortKey,
  chatEntryMatchesQuery,
  makeChatAccelPredicate,
  prepareChatJoin,
  type ChatAccelCompat,
  type ChatModelEntry,
  type ChatModelsSort,
  type HostLike,
} from "../../components/models/adapters/chat";
import { downloadSizeTooltip, number, sizeLabel } from "../../utils";
import { ChatHubResults, sortHubResults } from "./ChatHubResults";
import { memoryFitBucket } from "./memoryFit";

export type ChatLaunchAction = "chat" | "server" | "thread";

export interface ChatModelsPageProps {
  /** Every curated family (``workspace.featuredModels``); host gating is
   * per row, so a model on disk the host cannot run still shows. */
  families: ModelFamily[];
  /** The chat library (``isChatLibraryItem``). */
  library: LibraryItem[];
  activeDownloads: Record<string, DownloadStatus>;

  searchInput: string;
  onSearchInputChange: (value: string) => void;
  searchError: string | null;
  hubResults: HubModel[];
  expandedHubId: string | null;
  onToggleHubExpand: (repo: string) => void;
  hubFileCache: Record<string, HubFileListResponse>;
  hubFileLoading: Record<string, boolean>;
  hubFileError: Record<string, string>;

  capFilter: string | null;
  onCapFilterChange: (value: string | null) => void;
  formatFilter: string | null;
  onFormatFilterChange: (value: string | null) => void;
  accelFilter: string | null;
  onAccelFilterChange: (value: string | null) => void;
  backendFilter: string | null;
  onBackendFilterChange: (value: string | null) => void;
  sort: ChatModelsSort;
  sortDir: SortDir;
  onSortChange: (sort: ChatModelsSort, dir: SortDir) => void;
  availability: ModelAvailabilityFilter;
  onAvailabilityChange: (value: ModelAvailabilityFilter) => void;

  accelCompat: ChatAccelCompat;
  hostSystem?: HostLike;
  /** ``workspace.runtime.loadedModel``: marks its row "Loaded in memory". */
  loadedModel?: { ref?: string | null; path?: string | null; canonicalRepo?: string | null } | null;
  /** Free memory now; only for the fallback fit badge when the backend
   * sends no ``fit`` (R14). */
  availableMemoryGb?: number | null;
  enabledDirectoryCount: number;
  fileRevealLabel: string;
  favoriteModelRefs?: string[];
  onToggleFavoriteModel?: (ref: string) => void;

  onDownloadModel: (repo: string) => void;
  /** Pause. */
  onCancelModelDownload: (repo: string) => void;
  /** Cancel an in-flight download or delete a paused / failed one. */
  onDeleteModelDownload: (repo: string) => void;
  /** Delete local files; the handler asks for confirmation. */
  onDeleteModel: (item: LibraryItem) => void;
  onPrepareCatalogConversion: (variant: ModelVariant) => void;
  onOpenModelSelector: (action: ChatLaunchAction, preselectedKey?: string) => void;
  /** The family Details window (App-level modal). */
  onOpenFamilyDetails: (familyId: string) => void;
  onOpenExternalUrl: (url: string) => void;
  onRevealPath: (path: string) => void;
}

const CHAT_GRID =
  "minmax(240px, 2fr) minmax(90px, 0.7fr) minmax(80px, 0.6fr) minmax(76px, 0.55fr) " +
  "minmax(112px, 0.8fr) minmax(84px, 0.55fr) minmax(64px, 0.45fr) minmax(56px, 0.4fr) minmax(232px, 1fr)";
// Below 1440 px the Backend, Compressed and Context columns (3, 6, 7) are left out so the rest
// fits at 1100 px.
const CHAT_GRID_NARROW =
  "minmax(200px, 1.8fr) minmax(80px, 0.7fr) minmax(70px, 0.55fr) minmax(112px, 0.8fr) " +
  "minmax(64px, 0.5fr) minmax(232px, 1fr)";

function directionLabel(dir: SortDir, t: TFunction<"library">): string {
  return dir === "asc"
    ? t("imageModels.sort.direction.ascending", { defaultValue: "ascending" })
    : t("imageModels.sort.direction.descending", { defaultValue: "descending" });
}

/** Sort options; ids are ``ChatModelsSort`` so ``useModels`` keeps the
 * choice across tab switches. Rows without a number go last. */
export function chatSortOptions(t: TFunction<"library">): ModelSortOption<ChatModelEntry>[] {
  const text = (left: string, right: string) => left.localeCompare(right);
  const numeric = (
    id: ChatModelsSort,
    label: string,
    desc: string,
    asc: string,
    value: (entry: ChatModelEntry) => number | null | undefined,
  ): ModelSortOption<ChatModelEntry> => ({
    id,
    label,
    defaultDir: "desc",
    summary: (dir) => (dir === "desc" ? desc : asc),
    value: (row) => value(row.entry),
  });
  return [
    {
      id: "release",
      label: t("models.chat.sort.release", { defaultValue: "Newest released" }),
      defaultDir: "desc",
      summary: (dir) =>
        dir === "desc"
          ? t("imageModels.sort.newestFirst", { defaultValue: "newest released first" })
          : t("imageModels.sort.oldestFirst", { defaultValue: "oldest released first" }),
      compare: (a, b) => text(a.entry.releaseSortKey, b.entry.releaseSortKey),
    },
    {
      id: "name",
      label: t("models.chat.sort.name", { defaultValue: "Name" }),
      defaultDir: "asc",
      summary: (dir) =>
        dir === "asc"
          ? t("imageModels.sort.nameAsc", { defaultValue: "name A-Z" })
          : t("imageModels.sort.nameDesc", { defaultValue: "name Z-A" }),
      compare: (a, b) => text(a.entry.name, b.entry.name),
    },
    {
      id: "format",
      label: t("models.chat.sort.format", { defaultValue: "Format" }),
      defaultDir: "asc",
      summary: (dir) => t("models.chat.sort.formatDir", { defaultValue: "format {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => text(a.entry.source.displayFormat, b.entry.source.displayFormat),
    },
    {
      id: "backend",
      label: t("models.chat.sort.backend", { defaultValue: "Backend" }),
      defaultDir: "asc",
      summary: (dir) => t("models.chat.sort.backendDir", { defaultValue: "backend {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => text(a.entry.source.displayBackend, b.entry.source.displayBackend),
    },
    numeric(
      "size",
      t("models.chat.sort.size", { defaultValue: "Largest size" }),
      t("imageModels.sort.largestFirst", { defaultValue: "largest size first" }),
      t("imageModels.sort.smallestFirst", { defaultValue: "smallest size first" }),
      (entry) => entry.sizeGb,
    ),
    numeric(
      "ram",
      t("models.chat.sort.ram", { defaultValue: "Highest RAM" }),
      t("models.chat.sort.ramDesc", { defaultValue: "highest RAM first" }),
      t("models.chat.sort.ramAsc", { defaultValue: "lowest RAM first" }),
      (entry) => entry.source.ramGb,
    ),
    numeric(
      "compressed",
      t("models.chat.sort.compressed", { defaultValue: "Highest compressed RAM" }),
      t("models.chat.sort.compressedDesc", { defaultValue: "highest compressed RAM first" }),
      t("models.chat.sort.compressedAsc", { defaultValue: "lowest compressed RAM first" }),
      (entry) => entry.source.compressedGb,
    ),
    numeric(
      "context",
      t("models.chat.sort.context", { defaultValue: "Longest context" }),
      t("models.chat.sort.contextDesc", { defaultValue: "longest context first" }),
      t("models.chat.sort.contextAsc", { defaultValue: "shortest context first" }),
      chatContextSortKey,
    ),
    numeric(
      "modified",
      t("models.chat.sort.modified", { defaultValue: "Recently modified" }),
      t("models.chat.sort.modifiedDesc", { defaultValue: "recently modified first" }),
      t("models.chat.sort.modifiedAsc", { defaultValue: "least recently modified first" }),
      (entry) => entry.source.modifiedAt,
    ),
    {
      id: "status",
      label: t("models.chat.sort.status", { defaultValue: "Status" }),
      defaultDir: "asc",
      summary: (dir) => t("imageModels.sort.statusDir", { defaultValue: "status {direction}", direction: directionLabel(dir, t) }),
      compare: (a, b) => statusSortRank(a.status) - statusSortRank(b.status),
    },
  ];
}

/** The launch actions of an installed row (lead's decision): the primary
 * button starts a new thread with the model; the menu offers "Use in this
 * chat" (re-point the active thread) and "Load for server". All three
 * preselect this row's model in the launch modal and are disabled when
 * the host cannot run it. */
export function chatLaunchActions(
  entry: ChatModelEntry,
  onOpenModelSelector: (action: ChatLaunchAction, preselectedKey?: string) => void,
  t: TFunction<"library">,
): { launch: ModelLaunchAction; overflow: ModelLaunchAction[] } {
  const key = entry.source.launchKey ?? undefined;
  const disabled = !entry.hostCompatible;
  const title = disabled ? t("models.card.incompatible", { defaultValue: "Cannot run on this machine" }) : undefined;
  return {
    launch: {
      id: "thread",
      icon: "chat",
      label: t("models.chat.newThread", { defaultValue: "Chat in a new thread" }),
      onClick: () => onOpenModelSelector("thread", key),
      disabled,
      title,
    },
    overflow: [
      {
        id: "chat",
        icon: "chat",
        label: t("models.chat.useInThisChat", { defaultValue: "Use in this chat" }),
        onClick: () => onOpenModelSelector("chat", key),
        disabled,
        title,
      },
      {
        id: "server",
        icon: "server",
        label: t("onlineModels.action.loadForServer", { defaultValue: "Load for server" }),
        onClick: () => onOpenModelSelector("server", key),
        disabled,
        title,
      },
    ],
  };
}

const ALL = "all";
const toValue = (value: string | null): string => value ?? ALL;
const fromValue = (value: string): string | null => (value === ALL ? null : value);

/** Text-only capability chips (no emoji), coloured like the filter chips. */
function CapabilityBadges({ capabilities, max, t }: { capabilities: string[]; max: number; t: TFunction<"library"> }) {
  if (capabilities.length === 0) return null;
  return (
    <span className="capability-strip">
      {capabilities.slice(0, max).map((capability) => {
        const meta = CAPABILITY_META[capability];
        return (
          <span
            key={capability}
            className="capability-icon"
            title={t(`onlineModels.capability.title.${capability}`, { defaultValue: meta?.title ?? capability })}
            style={meta ? { borderColor: `${meta.color}40`, color: meta.color } : undefined}
          >
            {t(`onlineModels.capability.short.${capability}`, { defaultValue: meta?.shortLabel ?? capability })}
          </span>
        );
      })}
    </span>
  );
}

export function ChatModelsPage({
  families,
  library,
  activeDownloads,
  searchInput,
  onSearchInputChange,
  searchError,
  hubResults,
  expandedHubId,
  onToggleHubExpand,
  hubFileCache,
  hubFileLoading,
  hubFileError,
  capFilter,
  onCapFilterChange,
  formatFilter,
  onFormatFilterChange,
  accelFilter,
  onAccelFilterChange,
  backendFilter,
  onBackendFilterChange,
  sort,
  sortDir,
  onSortChange,
  availability,
  onAvailabilityChange,
  accelCompat,
  hostSystem,
  loadedModel,
  availableMemoryGb,
  enabledDirectoryCount,
  fileRevealLabel,
  favoriteModelRefs,
  onToggleFavoriteModel,
  onDownloadModel,
  onCancelModelDownload,
  onDeleteModelDownload,
  onDeleteModel,
  onPrepareCatalogConversion,
  onOpenModelSelector,
  onOpenFamilyDetails,
  onOpenExternalUrl,
  onRevealPath,
}: ChatModelsPageProps) {
  const { t } = useTranslation("library");
  const { t: tCommon } = useTranslation("common");
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  // The fuzzy match is the costly part; it only depends on the catalog
  // and the disk scan, not on download polling.
  const join = useMemo(() => prepareChatJoin(families, library), [families, library]);
  const entries = useMemo(
    () => buildChatEntries(families, library, activeDownloads, { host: hostSystem, loadedModel, join }),
    [families, library, activeDownloads, hostSystem, loadedModel, join],
  );
  const resolveStatus = useCallback(
    (entry: ChatModelEntry) => resolveModelStatus(entry, { download: entry.source.download }),
    [],
  );
  const favoriteSet = useMemo(() => new Set(favoriteModelRefs ?? []), [favoriteModelRefs]);
  const pinned = useCallback(
    (entry: ChatModelEntry) => Boolean(entry.source.favoriteRef && favoriteSet.has(entry.source.favoriteRef)),
    [favoriteSet],
  );
  const accelPredicate = useMemo(() => makeChatAccelPredicate(accelCompat), [accelCompat]);
  const sortOptions = useMemo(() => chatSortOptions(t), [t]);

  // Chip options and counts come from the rows this host shows that
  // match the search, before the chip filters themselves.
  const searchMatched = useMemo(
    () =>
      entries.filter(
        (entry) => isVisibleOnHost(entry, resolveStatus(entry)) && chatEntryMatchesQuery(entry, searchInput),
      ),
    [entries, resolveStatus, searchInput],
  );

  const capabilityOrder = Object.keys(CAPABILITY_META);
  const presentCapabilities = new Set(searchMatched.flatMap((entry) => entry.capabilities ?? []));
  const formats = [...new Set(searchMatched.map((entry) => entry.source.displayFormat).filter(Boolean))].sort();
  const backends = [...new Set(searchMatched.map((entry) => entry.source.displayBackend).filter(Boolean))].sort();
  const accelChips = CHAT_ACCEL_FILTERS.filter((chip) => accelCompat.appleSilicon !== false || !chip.appleOnly);

  const groups: ModelFilterGroup<ChatModelEntry>[] = [
    {
      id: "capability",
      label: t("models.chat.filter.capability", { defaultValue: "Capability" }),
      kind: "chips",
      value: toValue(capFilter),
      defaultValue: ALL,
      options: [
        { value: ALL, label: t("onlineModels.filter.all", { defaultValue: "All" }) },
        ...capabilityOrder
          .filter((capability) => presentCapabilities.has(capability))
          .map((capability) => ({
            value: capability,
            label: t(`onlineModels.capability.short.${capability}`, { defaultValue: CAPABILITY_META[capability]?.shortLabel ?? capability }),
            title: t(`onlineModels.capability.title.${capability}`, { defaultValue: CAPABILITY_META[capability]?.title ?? capability }),
            color: CAPABILITY_META[capability]?.color,
          })),
      ],
      onChange: (value) => onCapFilterChange(fromValue(value)),
      predicate: (entry, value) => (entry.capabilities ?? []).includes(value),
    },
    {
      id: "format",
      label: t("models.chat.filter.format", { defaultValue: "Format" }),
      kind: "chips",
      value: toValue(formatFilter),
      defaultValue: ALL,
      options: [
        { value: ALL, label: t("onlineModels.filter.allFormats", { defaultValue: "All formats" }) },
        ...formats.map((format) => ({ value: format, label: format })),
      ],
      onChange: (value) => onFormatFilterChange(fromValue(value)),
      predicate: (entry, value) => entry.source.displayFormat === value,
      hideBelow: 3,
    },
    {
      id: "backend",
      label: t("models.chat.filter.backend", { defaultValue: "Backend" }),
      kind: "chips",
      value: toValue(backendFilter),
      defaultValue: ALL,
      options: [
        { value: ALL, label: t("myModels.filter.allBackends", { defaultValue: "All backends" }) },
        ...backends.map((backend) => ({ value: backend, label: backend })),
      ],
      onChange: (value) => onBackendFilterChange(fromValue(value)),
      predicate: (entry, value) => entry.source.displayBackend === value,
      hideBelow: 3,
    },
    {
      id: "acceleration",
      label: t("models.chat.filter.acceleration", { defaultValue: "Acceleration" }),
      kind: "chips",
      value: toValue(accelFilter),
      defaultValue: ALL,
      options: [
        { value: ALL, label: t("onlineModels.filter.allAccel", { defaultValue: "All acceleration" }) },
        ...accelChips.map((chip) => {
          const count = searchMatched.filter((entry) => accelPredicate(entry, chip.id)).length;
          // TurboQuant shrinks the KV cache; it does not speed anything up.
          const memorySaver = isMemorySaverStrategy(chip.id);
          const label = memorySaver
            ? t("onlineModels.accelFilter.memorySaverLabel", { defaultValue: "{label} · memory saver", label: chip.label })
            : chip.label;
          const title = memorySaver
            ? t("onlineModels.accelFilter.memorySaverTooltip", {
                defaultValue:
                  "Show models that can use {label}, a KV-cache memory saver: long chats fit in less RAM, but generation is slower than Native ({count})",
                label: chip.label,
                count,
              })
            : chip.id === "dflash" && count === 0
              ? t("myModels.strategy.dflashEmptyTooltip", {
                  defaultValue:
                    "DFlash speculative-decode drafts only exist for specific base models: "
                    + "Qwen/Qwen3-{4B,8B}, Qwen/Qwen3-Coder-{4B,8B,30B-A3B,Next}, Qwen/Qwen3.5-{4B,7B,9B,14B,27B,35B-A3B}, "
                    + "Qwen/Qwen3.6-35B-A3B, meta-llama/Llama-3.1-8B-Instruct, gpt-oss-{20B,120B}, moonshotai/Kimi-K2.5. "
                    + "Fine-tunes typically don't match. Download a base model from the Models page to enable DFlash.",
                })
              : t("onlineModels.accelFilter.tooltip", {
                  defaultValue: "Show {label}-compatible models ({count})",
                  label: chip.label,
                  count,
                });
          return { value: chip.id, label, count, title, color: chip.color };
        }),
      ],
      onChange: (value) => onAccelFilterChange(fromValue(value)),
      predicate: (entry, value) => accelPredicate(entry, value),
    },
  ];

  const columns: ModelsPageColumn[] = [
    { id: "model", label: t("myModels.column.model", { defaultValue: "Model" }), sortId: "name" },
    { id: "format", label: t("myModels.column.format", { defaultValue: "Format" }), sortId: "format" },
    { id: "backend", label: t("myModels.column.backend", { defaultValue: "Backend" }), sortId: "backend" },
    { id: "size", label: t("myModels.column.size", { defaultValue: "Size" }), sortId: "size" },
    { id: "ram", label: t("myModels.column.ram", { defaultValue: "RAM" }), sortId: "ram" },
    { id: "compressed", label: t("myModels.column.compressed", { defaultValue: "Compressed" }), sortId: "compressed" },
    { id: "context", label: t("myModels.column.context", { defaultValue: "Context" }), sortId: "context" },
    { id: "status", label: t("myModels.column.status", { defaultValue: "Status" }), sortId: "status" },
  ];

  const sortedHubResults = useMemo(() => sortHubResults(hubResults, formatFilter), [hubResults, formatFilter]);
  const hasQuery = searchInput.trim().length > 0;
  const libraryTotalSizeGb = library.reduce((sum, item) => sum + (item.path.startsWith("download://") ? 0 : item.sizeGb), 0);

  function toggleExpanded(id: string) {
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function downloadDetail(download: DownloadStatus): string {
    const size = downloadSizeTooltip(download);
    if (download.state === "cancelled") {
      return size
        ? t("myModels.download.downloadedDetail", { defaultValue: "{size} downloaded.", size })
        : t("myModels.download.paused", { defaultValue: "Download paused." });
    }
    return size
      ? t("myModels.download.downloadedDetail", { defaultValue: "{size} downloaded.", size })
      : t("myModels.download.inProgress", { defaultValue: "Download in progress." });
  }

  function fallbackFitBadge(entry: ChatModelEntry) {
    const variant = entry.source.variant;
    if (entry.source.kind !== "catalog" || !variant) return null;
    const fit = memoryFitBucket(variant, availableMemoryGb);
    if (fit.kind === "unknown") return null;
    const available = availableMemoryGb?.toFixed(1);
    const title =
      fit.kind === "comfortable"
        ? t("onlineModels.memory.comfortable", { defaultValue: "Fits comfortably in {available} GB available", available })
        : fit.kind === "tight"
          ? t("onlineModels.memory.tight", {
              defaultValue: "Fits but tight against {available} GB available — close other apps before loading",
              available,
            })
          : t("onlineModels.memory.over", {
              defaultValue: "Estimated {estimated} GB exceeds {available} GB available — try a smaller quantisation",
              estimated: variant.estimatedMemoryGb?.toFixed?.(1) ?? "?",
              available,
            });
    return (
      <span className={`memory-fit-badge memory-fit-badge--${fit.kind}`} title={title}>
        {t(`onlineModels.memory.${fit.kind}Label`, { defaultValue: fit.label })}
      </span>
    );
  }

  function renderRow({ entry, status }: ModelRow<ChatModelEntry>) {
    const source = entry.source;
    const { variant, family } = source;
    const item = source.kind === "download" ? null : source.item;
    const download = status.download;
    const transferRepo = download?.repo ?? entry.downloadKey;
    const canDownload = Boolean(transferRepo);
    const inTransfer = status.state === "downloading" || status.state === "paused" || status.state === "failed";
    const isExpanded = expanded.has(entry.id);
    const favoriteRef = item ? source.favoriteRef : null;
    const isFavorite = favoriteRef ? favoriteSet.has(favoriteRef) : false;

    const subtitle =
      source.kind === "catalog"
        ? undefined
        : item?.directoryLabel || entry.provider || undefined;

    const badges = (
      <>
        {source.sourceKind ? <span className="badge muted">{source.sourceKind}</span> : null}
        {source.kind === "download" ? (
          <span className="badge muted">{t("models.chat.sourceDownload", { defaultValue: "Download" })}</span>
        ) : null}
        {inTransfer && download && (item || source.kind === "download") ? (
          <span className="badge muted" title={downloadSizeTooltip(download)}>
            {t("myModels.badge.activeDownload", { defaultValue: "Active download" })}
          </span>
        ) : null}
        <CapabilityBadges capabilities={entry.capabilities ?? []} max={5} t={t} />
        {(status.state === "downloading" || status.state === "paused") && download ? (
          <span className="library-download-tag">
            <small className="library-download-reason">{downloadDetail(download)}</small>
          </span>
        ) : null}
        {status.state === "broken" ? (
          <span className="broken-tag">
            <span className="badge warning">{t("myModels.badge.broken", { defaultValue: "BROKEN" })}</span>
            <small className="broken-reason">
              {entry.broken?.reason || t("myModels.status.incompleteReason", { defaultValue: "Incomplete or broken" })}
            </small>
          </span>
        ) : null}
      </>
    );

    const hasFiles = Boolean(item);
    const catalogSize = variant && source.kind === "catalog" && item && variant.sizeGb > 0 ? variant.sizeGb : null;
    const cells = [
      <>
        <span>{source.displayFormat || "-"}</span>
        {source.displayQuantization ? <small>{source.displayQuantization}</small> : null}
      </>,
      <>{source.displayBackend || "-"}</>,
      <>
        <span
          title={
            catalogSize
              ? t("models.chat.sizeTitle", {
                  defaultValue: "{onDisk} on disk; the catalog lists {catalog}",
                  onDisk: entry.sizeLabel ?? "",
                  catalog: sizeLabel(catalogSize),
                })
              : undefined
          }
        >
          {entry.sizeLabel}
        </span>
        {source.paramsB ? <small>{number(source.paramsB)}B</small> : null}
      </>,
      <span
        className="media-model-memory models-memory-cell"
        title={hasFiles ? t("myModels.tooltip.ram", { defaultValue: "Rough resident memory at 8K context (weights + KV + framework)" }) : undefined}
      >
        <span>{source.ramGb != null ? `~${number(source.ramGb)} GB` : "?"}</span>
        {entry.fit ? <FitBadge fit={entry.fit} /> : fallbackFitBadge(entry)}
      </span>,
      <span
        title={hasFiles ? t("myModels.tooltip.compressed", { defaultValue: "Rough resident memory with a compressed KV cache strategy" }) : undefined}
      >
        {source.compressedGb != null ? `~${number(source.compressedGb)} GB` : "?"}
      </span>,
      <>{source.contextWindow ?? ""}</>,
    ];

    const { launch, overflow } = chatLaunchActions(entry, onOpenModelSelector, t);

    const actions = (
      <ModelDownloadControls
        status={status}
        canDownload={canDownload}
        onDownload={() => onDownloadModel(transferRepo)}
        onPause={() => onCancelModelDownload(transferRepo)}
        onCancel={() => onDeleteModelDownload(transferRepo)}
        onDeleteModel={item ? () => onDeleteModel(item) : undefined}
        launch={launch}
        overflow={overflow}
        trailing={
          <>
            {favoriteRef && onToggleFavoriteModel ? (
              <IconActionButton
                icon={isFavorite ? "star" : "starOutline"}
                label={
                  isFavorite
                    ? t("myModels.action.unstarModel", { defaultValue: "Remove from favourites" })
                    : t("myModels.action.starModel", { defaultValue: "Mark as favourite" })
                }
                className={isFavorite ? "action-favorite action-favorite--on" : "action-favorite"}
                onClick={() => onToggleFavoriteModel(favoriteRef)}
              />
            ) : null}
            {status.state === "installed" && source.canConvert && variant ? (
              <IconActionButton
                icon="convert"
                label={t("onlineModels.action.convertModel", { defaultValue: "Convert model" })}
                buttonStyle="primary"
                className="action-convert"
                onClick={() => onPrepareCatalogConversion(variant)}
              />
            ) : null}
            {status.state === "broken" && canDownload && item ? (
              <IconActionButton
                icon="delete"
                label={t("myModels.action.deleteModel", { defaultValue: "Delete model" })}
                danger
                onClick={() => onDeleteModel(item)}
              />
            ) : null}
            {item ? (
              <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(item.path)} />
            ) : null}
          </>
        }
      />
    );

    const detail = () => (
      <>
        {source.kind === "library" && variant ? (
          <p className="muted-text">
            {t("models.chat.closestMatch", { defaultValue: "Closest catalog entry: {name}", name: variant.name })}
          </p>
        ) : null}
        {entry.note ? <p className="models-card__note">{entry.note}</p> : null}
        <ModelDetailList
          items={[
            entry.repo ? { label: t("models.detail.repo", { defaultValue: "Repository" }), value: <code>{entry.repo}</code> } : null,
            entry.releaseLabel ? { label: t("models.chat.detail.released", { defaultValue: "Released" }), value: entry.releaseLabel } : null,
            item ? { label: t("models.detail.localPath", { defaultValue: "On disk" }), value: <code>{item.path}</code> } : null,
            item?.directoryLabel
              ? { label: t("models.chat.detail.directory", { defaultValue: "Directory" }), value: item.directoryLabel }
              : null,
            item?.lastModified
              ? { label: t("models.chat.detail.modified", { defaultValue: "Modified" }), value: item.lastModified }
              : null,
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
          ]}
        />
        {source.kind === "catalog" && family?.readme?.length ? (
          <div className="discover-readme">
            {family.readme.slice(0, 2).map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </div>
        ) : null}
        <div className="button-row">
          {source.kind === "catalog" && family ? (
            <button
              className="secondary-button"
              type="button"
              title={t("onlineModels.action.detailsTooltip", { defaultValue: "Show full details in a focused view" })}
              onClick={() => onOpenFamilyDetails(family.id)}
            >
              {t("onlineModels.action.details", { defaultValue: "Details" })}
            </button>
          ) : null}
          {entry.link ? (
            <button className="secondary-button" type="button" onClick={() => onOpenExternalUrl(entry.link as string)}>
              {t("onlineModels.action.openModelCardHF", { defaultValue: "Open model card on Hugging Face" })}
            </button>
          ) : null}
        </div>
      </>
    );

    const callouts =
      status.state === "failed" && download?.error ? (
        <div className="media-model-row-detail callout error">
          <p>{download.error}</p>
        </div>
      ) : null;

    return (
      <ModelCard
        key={entry.id}
        entry={entry}
        status={status}
        subtitle={subtitle}
        badges={badges}
        cells={cells}
        actions={actions}
        detail={detail}
        expanded={isExpanded}
        onToggleExpanded={() => toggleExpanded(entry.id)}
        callouts={callouts}
        bestFitTitle={t("models.chat.bestFitTitle", {
          defaultValue:
            "The pick in this family for this machine: the newest release that fits comfortably; among sizes released together, the largest that fits.",
        })}
      />
    );
  }

  const hubBlock = (
    <ChatHubResults
      results={sortedHubResults}
      expandedHubId={expandedHubId}
      onToggleHubExpand={onToggleHubExpand}
      hubFileCache={hubFileCache}
      hubFileLoading={hubFileLoading}
      hubFileError={hubFileError}
      activeDownloads={activeDownloads}
      library={library}
      families={families}
      onDownloadModel={onDownloadModel}
      onCancelModelDownload={onCancelModelDownload}
      onDeleteModelDownload={onDeleteModelDownload}
      onOpenModelSelector={onOpenModelSelector}
      onOpenExternalUrl={onOpenExternalUrl}
    />
  );

  return (
    <ModelsPage<ChatModelEntry>
      className="chat-models-page"
      title={tCommon("tabs.chatModels", { defaultValue: "Chat Models" })}
      subtitle={(counts) =>
        t("models.chat.subtitle", {
          defaultValue:
            "{shown} of {total, plural, one {# model} other {# models}} · {installed} installed · {size} on disk · {dirs, plural, one {# directory} other {# directories}}",
          ...counts,
          size: sizeLabel(libraryTotalSizeGb),
          dirs: enabledDirectoryCount,
        })
      }
      intro={
        <p className="muted-text">
          {t("models.chat.intro", {
            defaultValue:
              "Every chat model in one list: the ones on disk, the catalog, and Hugging Face search results. Filter by installed, available or downloading.",
          })}
        </p>
      }
      beforeList={
        <div className="chat-models-tools">
          {searchError ? (
            <div className="callout error">
              <p>{searchError}</p>
            </div>
          ) : null}
          <RunFromHuggingFace />
          <ImportModelsPanel />
        </div>
      }
      afterList={hubBlock}
      entries={entries}
      resolveStatus={resolveStatus}
      search={{
        value: searchInput,
        onChange: onSearchInputChange,
        placeholder: t("models.chat.searchPlaceholder", {
          defaultValue: "Search name, provider, capability, path or format (also searches Hugging Face)...",
        }),
      }}
      matches={chatEntryMatchesQuery}
      groups={groups}
      availability={availability}
      onAvailabilityChange={onAvailabilityChange}
      pinned={favoriteSet.size > 0 ? pinned : undefined}
      sortOptions={sortOptions}
      sortId={sort}
      sortDir={sortDir}
      defaultSortId={CHAT_MODELS_DEFAULT_SORT}
      onSortChange={(id, dir) => onSortChange(id as ChatModelsSort, dir)}
      onClearFilters={() => {
        onSearchInputChange("");
        onCapFilterChange(null);
        onFormatFilterChange(null);
        onBackendFilterChange(null);
        onAccelFilterChange(null);
      }}
      columns={columns}
      gridTemplate={CHAT_GRID}
      narrowGridTemplate={CHAT_GRID_NARROW}
      narrowHide={[3, 6, 7]}
      renderRow={renderRow}
      emptyCatalog={t("models.chat.emptyCatalog", {
        defaultValue: "No chat models in the catalog or on disk yet. Check that the backend is running, or add model directories in Settings.",
      })}
      emptyByAvailability={{
        installed: t("myModels.empty", { defaultValue: "No models found. Add directories in Settings to scan for local models." }),
        downloading: t("models.chat.emptyDownloading", { defaultValue: "No chat model is downloading." }),
        attention: t("models.chat.emptyAttention", { defaultValue: "No chat model needs attention." }),
      }}
      emptyFilters={
        hasQuery && sortedHubResults.length > 0
          ? t("models.chat.emptySearchHub", {
              defaultValue: "No model on disk or in the catalog matches \"{query}\". Results from Hugging Face are below.",
              query: searchInput.trim(),
            })
          : undefined
      }
    />
  );
}
