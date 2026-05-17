import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { DownloadStatus } from "../../api";
import { Panel } from "../../components/Panel";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import type { ModelStatusKind } from "../../components/ModelActionIcons";
import type {
  LibraryItem,
  ModelVariant,
} from "../../types";
import {
  number,
  sizeLabel,
  parseContextK,
  compareOptionalNumber,
  inferHfRepoFromLocalPath,
  downloadProgressLabel,
  downloadSizeTooltip,
  formatReleaseLabel,
} from "../../utils";
import { CAPABILITY_META } from "../../constants";
import { CapabilityStrip } from "../../components/CapabilityStrip";
import { candidateKeys } from "../../components/runtimeSupport";

export interface LibraryRow {
  item: LibraryItem;
  matchedVariant: ModelVariant | null;
  displayFormat: string;
  displayQuantization: string | null;
  displayBackend: string;
  sourceKind: string;
  estimatedRamGb: number | null;
  estimatedCompressedGb: number | null;
}

interface StrategyCompatInfo {
  turboInstalled: boolean;
  turboquantMlxAvailable: boolean;
  dflashSupportedModels: string[];
  mtplxInstalled?: boolean;
  mtplxSupportedModels?: string[];
}

export interface MyModelsTabProps {
  filteredLibraryRows: LibraryRow[];
  libraryTotalSizeGb: number;
  enabledDirectoryCount: number;
  librarySearchInput: string;
  onLibrarySearchInputChange: (value: string) => void;
  libraryCapFilter: string | null;
  onLibraryCapFilterChange: (cap: string | null) => void;
  libraryFormatFilter: string | null;
  onLibraryFormatFilterChange: (fmt: string | null) => void;
  libraryBackendFilter: string | null;
  onLibraryBackendFilterChange: (backend: string | null) => void;
  strategyCompat?: StrategyCompatInfo;
  activeDownloads: Record<string, DownloadStatus>;
  expandedLibraryPath: string | null;
  onExpandedLibraryPathChange: (path: string | null) => void;
  fileRevealLabel: string;
  onDownloadModel: (repo: string) => void;
  onCancelModelDownload: (repo: string) => void;
  onDeleteModelDownload: (repo: string) => void;
  onPrepareLibraryConversion: (item: LibraryItem) => void;
  onOpenModelSelector: (action: "chat" | "server" | "thread", preselectedKey?: string) => void;
  onRevealPath: (path: string) => void;
  onDeleteModel: (item: LibraryItem) => void;
  librarySortKey: "name" | "format" | "backend" | "size" | "ram" | "compressed" | "modified" | "context";
  librarySortDir: "asc" | "desc";
  onLibrarySortKeyChange: (key: "name" | "format" | "backend" | "size" | "ram" | "compressed" | "modified" | "context") => void;
  onLibrarySortDirChange: (dir: "asc" | "desc") => void;
  // FU-052 follow-up: starred model refs + toggle handler. ``favoriteModelRefs``
  // is the persisted list from settings; ``onToggleFavoriteModel`` flips the
  // membership of a single canonical ref and writes the new list back.
  favoriteModelRefs?: string[];
  onToggleFavoriteModel?: (ref: string) => void;
}

function rowFavoriteRef(row: LibraryRow): string | null {
  // Canonical ref used to identify a model for favouriting. Prefer the
  // inferred HF repo (matches what other parts of the UI use to identify
  // a model), fall back to the matched variant repo, then to the local
  // path. Empty strings collapse to ``null``.
  const repo = inferHfRepoFromLocalPath(row.item.path)
    ?? row.matchedVariant?.repo
    ?? (row.item.name.includes("/") ? row.item.name : null);
  if (repo && repo.trim()) return repo.trim();
  if (row.item.path) return row.item.path;
  return null;
}

export function MyModelsTab({
  filteredLibraryRows,
  libraryTotalSizeGb,
  enabledDirectoryCount,
  librarySearchInput,
  onLibrarySearchInputChange,
  libraryCapFilter,
  onLibraryCapFilterChange,
  libraryFormatFilter,
  onLibraryFormatFilterChange,
  libraryBackendFilter,
  onLibraryBackendFilterChange,
  strategyCompat,
  activeDownloads,
  expandedLibraryPath,
  onExpandedLibraryPathChange,
  fileRevealLabel,
  onDownloadModel,
  onCancelModelDownload,
  onDeleteModelDownload,
  onPrepareLibraryConversion,
  onOpenModelSelector,
  onRevealPath,
  onDeleteModel,
  librarySortKey,
  librarySortDir,
  onLibrarySortKeyChange,
  onLibrarySortDirChange,
  favoriteModelRefs,
  onToggleFavoriteModel,
}: MyModelsTabProps) {
  const favoriteRefSet = new Set(favoriteModelRefs ?? []);
  const { t } = useTranslation("library");
  function toggleLibrarySort(key: "name" | "format" | "backend" | "size" | "ram" | "compressed" | "modified" | "context") {
    if (librarySortKey === key) {
      onLibrarySortDirChange(librarySortDir === "asc" ? "desc" : "asc");
    } else {
      onLibrarySortKeyChange(key);
      onLibrarySortDirChange(key === "name" ? "asc" : "desc");
    }
  }

  function sortIndicator(key: string) {
    if (librarySortKey !== key) return "";
    return librarySortDir === "asc" ? " \u25B2" : " \u25BC";
  }

  function libraryDownloadDetail(download: DownloadStatus): string {
    const sizeDetail = downloadSizeTooltip(download);
    if (download.state === "failed") {
      return download.error ?? t("myModels.download.failed", { defaultValue: "Download failed." });
    }
    if (download.state === "cancelled") {
      return sizeDetail
        ? t("myModels.download.downloadedDetail", { defaultValue: "{size} downloaded.", size: sizeDetail })
        : t("myModels.download.paused", { defaultValue: "Download paused." });
    }
    return sizeDetail
      ? t("myModels.download.downloadedDetail", { defaultValue: "{size} downloaded.", size: sizeDetail })
      : t("myModels.download.inProgress", { defaultValue: "Download in progress." });
  }

  function inferredPartialDownload(row: LibraryRow, repo: string | null): DownloadStatus | null {
    if (!repo || !row.item.broken) return null;
    const reason = (row.item.brokenReason ?? "").toLowerCase();
    const isPartialHfCache =
      row.sourceKind.toLowerCase() === "hf cache"
      && (reason.includes("partial blob") || reason.includes("incomplete"));
    if (!isPartialHfCache) return null;

    const downloadedGb = Math.max(0, row.item.sizeGb ?? 0);
    const totalGb = row.matchedVariant?.sizeGb && row.matchedVariant.sizeGb > 0
      ? row.matchedVariant.sizeGb
      : null;
    const progress = totalGb ? Math.max(0, Math.min(1, downloadedGb / totalGb)) : 0;

    return {
      repo,
      state: "cancelled",
      progress,
      downloadedGb,
      totalGb,
      error: null,
    };
  }


  function renderCapabilityFilterBar(
    active: string | null,
    setActive: (cap: string | null) => void,
    capabilities: string[],
  ) {
    const capOrder = Object.keys(CAPABILITY_META);
    const present = new Set(capabilities);
    const uniqueCaps = capOrder.filter((c) => present.has(c));
    return (
      <div className="cap-filter-bar">
        <button
          className={`cap-filter-btn${active === null ? " cap-filter-btn--active" : ""}`}
          type="button"
          onClick={() => setActive(null)}
        >
          {t("myModels.filter.all", { defaultValue: "All" })}
        </button>
        {uniqueCaps.map((cap) => {
          const meta = CAPABILITY_META[cap];
          const localizedTitle = t(`myModels.capability.title.${cap}`, { defaultValue: meta?.title ?? cap });
          const localizedShort = t(`myModels.capability.short.${cap}`, { defaultValue: meta?.shortLabel ?? cap });
          return (
            <button
              key={cap}
              className={`cap-filter-btn${active === cap ? " cap-filter-btn--active" : ""}`}
              type="button"
              onClick={() => setActive(active === cap ? null : cap)}
              title={localizedTitle}
              style={active === cap && meta ? { borderColor: meta.color, color: meta.color, background: `${meta.color}15` } : undefined}
            >
              {meta?.icon ?? ""} {localizedShort}
            </button>
          );
        })}
      </div>
    );
  }

  function renderFormatFilterBar(
    active: string | null,
    setActive: (fmt: string | null) => void,
    formats: string[],
    allLabel?: string,
  ) {
    const resolvedAllLabel = allLabel ?? t("myModels.filter.allFormats", { defaultValue: "All formats" });
    const uniqueFormats = [...new Set(formats)].sort();
    if (uniqueFormats.length < 2) return null;
    return (
      <div className="cap-filter-bar">
        <button
          className={`cap-filter-btn${active === null ? " cap-filter-btn--active" : ""}`}
          type="button"
          onClick={() => setActive(null)}
        >
          {resolvedAllLabel}
        </button>
        {uniqueFormats.map((fmt) => (
          <button
            key={fmt}
            className={`cap-filter-btn${active === fmt ? " cap-filter-btn--active" : ""}`}
            type="button"
            onClick={() => setActive(active === fmt ? null : fmt)}
          >
            {fmt}
          </button>
        ))}
      </div>
    );
  }

  const [strategyFilter, setStrategyFilter] = useState<string | null>(null);

  // ── Strategy compatibility check ──
  function modelSupportsStrategy(row: LibraryRow, strategy: string): boolean {
    const backend = row.displayBackend.toLowerCase();
    const isGGUF = backend.includes("llama") || row.displayFormat === "GGUF";
    const isMLX = backend.includes("mlx") || row.displayFormat === "MLX";
    const modelName = row.item.name;

    switch (strategy) {
      case "dflash": {
        // DFlash requires MLX or vLLM — not available for GGUF/llama.cpp models
        if (isGGUF) return false;
        if (!strategyCompat?.dflashSupportedModels?.length) return false;
        // Use the same candidateKeys matching as the model selection modal
        const modelKeys = candidateKeys([modelName, row.matchedVariant?.repo]);
        return strategyCompat.dflashSupportedModels.some((ref) => {
          const refKeys = candidateKeys([ref]);
          return refKeys.some((k) => modelKeys.includes(k));
        });
      }
      case "turboquant":
        return (isGGUF && !!strategyCompat?.turboInstalled) || (isMLX && !!strategyCompat?.turboquantMlxAvailable);
      case "mtplx": {
        // MTPLX needs baked-in MTP heads from training — much stricter than
        // DFlash (separate drafter). The fuzzy ``matchedVariant.repo`` would
        // attach unrelated Unsloth/Gemma locals to canonical Qwen entries
        // that DO carry MTP heads, falsely listing them under MTPLX. Match
        // on the actual on-disk name only; if a user installs the canonical
        // ``mlx-community/Qwen3.6-27B-4bit`` the directory name itself
        // normalises into the registry. Forks/repacks (e.g. ``-UD-MLX-4bit``)
        // are excluded until explicitly aliased in ``_MTP_ALIASES``.
        if (!isMLX) return false;
        if (!strategyCompat?.mtplxSupportedModels?.length) return false;
        const modelKeys = candidateKeys([modelName]);
        return strategyCompat.mtplxSupportedModels.some((ref) => {
          const refKeys = candidateKeys([ref]);
          return refKeys.some((k) => modelKeys.includes(k));
        });
      }
      default:
        return true;
    }
  }

  // Note: id keeps the canonical strategy slug; label is rendered through
  // ``t()`` at the render site so the chip text follows the active locale.
  const STRATEGY_FILTERS = [
    { id: "dflash", label: "DFlash", color: "#a78bfa" },
    { id: "mtplx", label: "MTPLX", color: "#f472b6" },
    { id: "turboquant", label: "TurboQuant", color: "#60a5fa" },
  ];

  const allLibraryCaps = filteredLibraryRows.flatMap(({ matchedVariant }) => matchedVariant?.capabilities ?? []);
  let capFilteredLibrary = libraryCapFilter
    ? filteredLibraryRows.filter(({ matchedVariant }) => {
        return matchedVariant?.capabilities?.includes(libraryCapFilter!) ?? false;
      })
    : filteredLibraryRows;
  if (libraryFormatFilter) {
    capFilteredLibrary = capFilteredLibrary.filter(({ displayFormat }) => displayFormat === libraryFormatFilter);
  }
  if (libraryBackendFilter) {
    capFilteredLibrary = capFilteredLibrary.filter(({ displayBackend }) => displayBackend === libraryBackendFilter);
  }
  if (strategyFilter) {
    capFilteredLibrary = capFilteredLibrary.filter((row) => modelSupportsStrategy(row, strategyFilter));
  }
  if (favoriteRefSet.size > 0) {
    // Lift starred rows to the top. Preserves the user's chosen sort
    // direction within each band (favourites + non-favourites) so the
    // sort header indicators still mean what they say.
    capFilteredLibrary = [
      ...capFilteredLibrary.filter((row) => {
        const ref = rowFavoriteRef(row);
        return ref ? favoriteRefSet.has(ref) : false;
      }),
      ...capFilteredLibrary.filter((row) => {
        const ref = rowFavoriteRef(row);
        return ref ? !favoriteRefSet.has(ref) : true;
      }),
    ];
  }
  const allLibraryFormats = filteredLibraryRows.map(({ displayFormat }) => displayFormat);
  const allLibraryBackends = filteredLibraryRows.map(({ displayBackend }) => displayBackend);

  return (
    <div className="content-grid">
      <Panel
        title={t("common:tabs.myModels")}
        subtitle={t("myModels.subtitle", {
          defaultValue: "{count} models / {size} on disk / {dirs} directories",
          count: filteredLibraryRows.length,
          size: sizeLabel(libraryTotalSizeGb),
          dirs: enabledDirectoryCount,
        })}
        className="span-2"
        actions={
          <input
            className="text-input discover-search"
            type="search"
            placeholder={t("myModels.searchPlaceholder", { defaultValue: "Filter by name, path, format, quant, or backend..." })}
            value={librarySearchInput}
            onChange={(event) => onLibrarySearchInputChange(event.target.value)}
          />
        }
      >
        {renderCapabilityFilterBar(libraryCapFilter, onLibraryCapFilterChange, allLibraryCaps)}
        {renderFormatFilterBar(libraryFormatFilter, onLibraryFormatFilterChange, allLibraryFormats)}
        {renderFormatFilterBar(libraryBackendFilter, onLibraryBackendFilterChange, allLibraryBackends, t("myModels.filter.allBackends", { defaultValue: "All backends" }))}
        {strategyCompat ? (
          <div className="cap-filter-bar">
            <button
              className={`cap-filter-btn${strategyFilter === null ? " cap-filter-btn--active" : ""}`}
              type="button"
              onClick={() => setStrategyFilter(null)}
            >
              {t("myModels.filter.allStrategies", { defaultValue: "All strategies" })}
            </button>
            {STRATEGY_FILTERS.map((sf) => {
              const count = filteredLibraryRows.filter((row) => modelSupportsStrategy(row, sf.id)).length;
              // DFlash gets a more explanatory tooltip when zero models
              // match — speculative-decode drafts are pinned per family,
              // so users land on "0" often unless they have a base
              // Qwen3 / Llama-3.1 / gpt-oss / Kimi model.
              const tooltip = sf.id === "dflash" && count === 0
                ? t("myModels.strategy.dflashEmptyTooltip", {
                    defaultValue:
                      "DFlash speculative-decode drafts only exist for specific base models: "
                      + "Qwen/Qwen3-{4B,8B}, Qwen/Qwen3-Coder-{4B,8B,30B-A3B,Next}, Qwen/Qwen3.5-{4B,7B,9B,14B,27B,35B-A3B}, "
                      + "Qwen/Qwen3.6-35B-A3B, meta-llama/Llama-3.1-8B-Instruct, gpt-oss-{20B,120B}, moonshotai/Kimi-K2.5. "
                      + "Fine-tunes typically don't match. Download a base model from Discover to enable DFlash.",
                  })
                : t("myModels.strategy.compatTooltip", {
                    defaultValue: "Show models compatible with {label} ({count})",
                    label: sf.label,
                    count,
                  });
              return (
                <button
                  key={sf.id}
                  className={`cap-filter-btn${strategyFilter === sf.id ? " cap-filter-btn--active" : ""}`}
                  type="button"
                  onClick={() => setStrategyFilter(strategyFilter === sf.id ? null : sf.id)}
                  title={tooltip}
                  style={strategyFilter === sf.id ? { borderColor: sf.color, color: sf.color, background: `${sf.color}15` } : undefined}
                >
                  {t("myModels.strategy.chip", { defaultValue: "{label} ({count})", label: sf.label, count })}
                </button>
              );
            })}
          </div>
        ) : null}
        {capFilteredLibrary.length ? (
          <div className="library-full-table">
            <div className="library-head">
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("name")}>{t("myModels.column.model", { defaultValue: "Model" })}{sortIndicator("name")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("format")}>{t("myModels.column.format", { defaultValue: "Format" })}{sortIndicator("format")}</button>
              <span className="sort-header">{t("myModels.column.quant", { defaultValue: "Quant" })}</span>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("backend")}>{t("myModels.column.backend", { defaultValue: "Backend" })}{sortIndicator("backend")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("size")}>{t("myModels.column.size", { defaultValue: "Size" })}{sortIndicator("size")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("ram")}>{t("myModels.column.ram", { defaultValue: "RAM" })}{sortIndicator("ram")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("compressed")}>{t("myModels.column.compressed", { defaultValue: "Compressed" })}{sortIndicator("compressed")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("context")}>{t("myModels.column.context", { defaultValue: "Context" })}{sortIndicator("context")}</button>
              <span className="sort-header">{t("myModels.column.status", { defaultValue: "Status" })}</span>
              <span className="sort-header"></span>
            </div>
            <div className="library-rows">
              {capFilteredLibrary.map(({ item, matchedVariant, displayFormat, displayQuantization, displayBackend, sourceKind, estimatedRamGb, estimatedCompressedGb }) => {
                const isExpanded = expandedLibraryPath === item.path;
                const repo = inferHfRepoFromLocalPath(item.path) ?? matchedVariant?.repo ?? (item.name.includes("/") ? item.name : null);
                const row: LibraryRow = {
                  item,
                  matchedVariant,
                  displayFormat,
                  displayQuantization,
                  displayBackend,
                  sourceKind,
                  estimatedRamGb,
                  estimatedCompressedGb,
                };
                const downloadState = repo
                  ? activeDownloads[repo] ?? inferredPartialDownload(row, repo)
                  : null;
                const isDownloading = downloadState?.state === "downloading";
                const isPaused = downloadState?.state === "cancelled";
                const isDownloadFailed = downloadState?.state === "failed";
                const hasDownloadOverlay = Boolean(isDownloading || isPaused || isDownloadFailed);
                const showBroken = Boolean(item.broken && !hasDownloadOverlay);
                const canRetryBrokenRepo = Boolean(showBroken && repo);
                // Rows synthesised from an in-flight download use a
                // ``download://<repo>`` sentinel path — they have no real
                // file on disk yet, so hide path-only actions.
                const isSyntheticDownloadRow = item.path.startsWith("download://");
                const rowStatus: { kind: ModelStatusKind; label: string; detail?: string | null } = isDownloading && downloadState
                  ? { kind: "downloading", label: t("myModels.status.downloading", { defaultValue: "Downloading" }), detail: downloadProgressLabel(downloadState) }
                  : isPaused && downloadState
                    ? { kind: "paused", label: t("myModels.status.paused", { defaultValue: "Paused" }), detail: downloadProgressLabel(downloadState) }
                    : isDownloadFailed && downloadState
                      ? { kind: "failed", label: t("myModels.status.failed", { defaultValue: "Failed" }), detail: downloadState.error ?? t("myModels.download.failedShort", { defaultValue: "Download failed" }) }
                      : showBroken
                        ? { kind: "incomplete", label: t("myModels.status.incomplete", { defaultValue: "Incomplete" }), detail: item.brokenReason ?? t("myModels.status.incompleteReason", { defaultValue: "Incomplete or broken" }) }
                        : { kind: "installed", label: t("myModels.status.installed", { defaultValue: "Installed" }) };
                const wrapperClassName = [
                  "library-item-wrap",
                  isExpanded ? "expanded" : "",
                  isDownloading ? "download-active" : "",
                  isPaused || isDownloadFailed ? "download-warning" : "",
                ].filter(Boolean).join(" ");
                return (
                  <div key={item.path} className={wrapperClassName}>
                    <div
                      className="library-item-row"
                      role="button"
                      tabIndex={0}
                      onClick={() => onExpandedLibraryPathChange(isExpanded ? null : item.path)}
                    >
                      <div className="library-item-name">
                        <strong>{item.name}</strong>
                        <div className="library-item-meta-row">
                          <span className="badge muted">{sourceKind}</span>
                          {hasDownloadOverlay && downloadState ? (
                            <span className="badge muted" title={downloadSizeTooltip(downloadState)}>{t("myModels.badge.activeDownload", { defaultValue: "Active download" })}</span>
                          ) : null}
                        </div>
                        {matchedVariant ? <CapabilityStrip capabilities={matchedVariant.capabilities} max={5} /> : null}
                        {hasDownloadOverlay && downloadState ? (
                          <span className="library-download-tag">
                            <small className={`library-download-reason${isDownloadFailed ? " error" : ""}`}>
                              {libraryDownloadDetail(downloadState)}
                            </small>
                          </span>
                        ) : null}
                        {showBroken ? (
                          <span className="broken-tag">
                            <span className="badge warning">{t("myModels.badge.broken", { defaultValue: "BROKEN" })}</span>
                            <small className="broken-reason">{item.brokenReason ?? t("myModels.status.incompleteReason", { defaultValue: "Incomplete or broken" })}</small>
                          </span>
                        ) : null}
                      </div>
                      <span>{displayFormat}</span>
                      <span>{displayQuantization ?? "-"}</span>
                      <span>{displayBackend}</span>
                      <span>{sizeLabel(item.sizeGb)}</span>
                      <span title={t("myModels.tooltip.ram", { defaultValue: "Rough resident memory at 8K context (weights + KV + framework)" })}>
                        {estimatedRamGb != null ? `~${number(estimatedRamGb)} GB` : "?"}
                      </span>
                      <span title={t("myModels.tooltip.compressed", { defaultValue: "Rough resident memory with a compressed KV cache strategy" })}>
                        {estimatedCompressedGb != null ? `~${number(estimatedCompressedGb)} GB` : "?"}
                      </span>
                      <span>{matchedVariant?.contextWindow ?? ""}</span>
                      <span className="library-row-status">
                        <StatusIcon status={rowStatus.kind} label={rowStatus.label} detail={rowStatus.detail} />
                      </span>
                      <div className="library-row-actions" onClick={(e) => e.stopPropagation()}>
                        {(() => {
                          const favRef = rowFavoriteRef(row);
                          if (!favRef || !onToggleFavoriteModel) return null;
                          const isFav = favoriteRefSet.has(favRef);
                          return (
                            <IconActionButton
                              icon={isFav ? "star" : "starOutline"}
                              label={isFav
                                ? t("myModels.action.unstarModel", { defaultValue: "Remove from favourites" })
                                : t("myModels.action.starModel", { defaultValue: "Mark as favourite" })}
                              className={isFav ? "action-favorite action-favorite--on" : "action-favorite"}
                              onClick={() => onToggleFavoriteModel(favRef)}
                            />
                          );
                        })()}
                        {hasDownloadOverlay && repo ? (
                          <>
                            {isDownloading ? (
                              <IconActionButton icon="pause" label={t("myModels.action.pauseDownload", { defaultValue: "Pause download" })} onClick={() => onCancelModelDownload(repo)} />
                            ) : (
                              <IconActionButton icon={isDownloadFailed ? "retry" : "resume"} label={isDownloadFailed ? t("myModels.action.retryDownload", { defaultValue: "Retry download" }) : t("myModels.action.resumeDownload", { defaultValue: "Resume download" })} buttonStyle="primary" onClick={() => onDownloadModel(repo)} />
                            )}
                            <IconActionButton icon={isDownloading ? "cancel" : "delete"} label={isDownloading ? t("myModels.action.cancelDownload", { defaultValue: "Cancel download" }) : t("myModels.action.deleteDownload", { defaultValue: "Delete download" })} danger onClick={() => onDeleteModelDownload(repo)} />
                          </>
                        ) : canRetryBrokenRepo ? (
                          <>
                            <IconActionButton icon="retry" label={t("myModels.action.retryDownload", { defaultValue: "Retry download" })} buttonStyle="primary" onClick={() => onDownloadModel(repo!)} />
                            <IconActionButton icon="delete" label={t("myModels.action.deleteDownload", { defaultValue: "Delete download" })} danger onClick={() => onDeleteModelDownload(repo!)} />
                          </>
                        ) : (
                          <>
                            {!item.broken ? (
                              <>
                                <IconActionButton icon="chat" label={t("myModels.action.chatWithModel", { defaultValue: "Chat with model" })} buttonStyle="primary" className="action-chat" onClick={() => onOpenModelSelector("chat", `library:${item.path}`)} />
                                <IconActionButton icon="server" label={t("myModels.action.loadForServer", { defaultValue: "Load for server" })} buttonStyle="primary" className="action-server" onClick={() => onOpenModelSelector("server", `library:${item.path}`)} />
                              </>
                            ) : null}
                          </>
                        )}
                        {!isSyntheticDownloadRow ? (
                          <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(item.path)} />
                        ) : null}
                        {!hasDownloadOverlay ? (
                          <IconActionButton icon="delete" label={t("myModels.action.deleteModel", { defaultValue: "Delete model" })} danger onClick={() => onDeleteModel(item)} />
                        ) : null}
                      </div>
                    </div>
                    {isExpanded ? (
                      <div className="library-item-detail">
                        <div className="library-detail-left">
                          {!isSyntheticDownloadRow ? (
                            <p className="mono-text library-path">{item.path}</p>
                          ) : null}
                          {matchedVariant?.note ? <p className="variant-note">{matchedVariant.note}</p> : null}
                          {formatReleaseLabel(matchedVariant?.releaseLabel, matchedVariant?.releaseDate) ? (
                            <p className="muted-text variant-release-label">
                              {formatReleaseLabel(matchedVariant?.releaseLabel, matchedVariant?.releaseDate)}
                            </p>
                          ) : null}
                        </div>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <p>{t("myModels.empty", { defaultValue: "No models found. Add directories in Settings to scan for local models." })}</p>
          </div>
        )}
      </Panel>
    </div>
  );
}
