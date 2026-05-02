import { useState } from "react";
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
  capabilityMeta,
  parseContextK,
  compareOptionalNumber,
  inferHfRepoFromLocalPath,
  downloadProgressLabel,
  downloadSizeTooltip,
  formatReleaseLabel,
} from "../../utils";
import { CAPABILITY_META } from "../../constants";
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
  chaosengineAvailable: boolean;
  dflashSupportedModels: string[];
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
}: MyModelsTabProps) {
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
      return download.error ?? "Download failed.";
    }
    if (download.state === "cancelled") {
      return sizeDetail ? `${sizeDetail} downloaded.` : "Download paused.";
    }
    return sizeDetail ? `${sizeDetail} downloaded.` : "Download in progress.";
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

  function renderCapabilityIcons(capabilities: string[], max = 5) {
    return (
      <div className="capability-strip">
        {capabilities.slice(0, max).map((capability) => {
          const meta = capabilityMeta(capability);
          const fullMeta = CAPABILITY_META[capability];
          return (
            <span
              className="capability-icon"
              key={capability}
              title={meta.title}
              style={fullMeta ? { borderColor: `${fullMeta.color}40`, color: fullMeta.color } : undefined}
            >
              {fullMeta?.icon ?? ""} {meta.shortLabel}
            </span>
          );
        })}
      </div>
    );
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
          All
        </button>
        {uniqueCaps.map((cap) => {
          const meta = CAPABILITY_META[cap];
          return (
            <button
              key={cap}
              className={`cap-filter-btn${active === cap ? " cap-filter-btn--active" : ""}`}
              type="button"
              onClick={() => setActive(active === cap ? null : cap)}
              title={meta?.title ?? cap}
              style={active === cap && meta ? { borderColor: meta.color, color: meta.color, background: `${meta.color}15` } : undefined}
            >
              {meta?.icon ?? ""} {meta?.shortLabel ?? cap}
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
    allLabel = "All formats",
  ) {
    const uniqueFormats = [...new Set(formats)].sort();
    if (uniqueFormats.length < 2) return null;
    return (
      <div className="cap-filter-bar">
        <button
          className={`cap-filter-btn${active === null ? " cap-filter-btn--active" : ""}`}
          type="button"
          onClick={() => setActive(null)}
        >
          {allLabel}
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
      case "rotorquant":
        return isGGUF && !!strategyCompat?.turboInstalled;
      case "chaosengine":
        return isGGUF && !!strategyCompat?.chaosengineAvailable;
      default:
        return true;
    }
  }

  const STRATEGY_FILTERS = [
    { id: "dflash", label: "DFlash", color: "#a78bfa" },
    { id: "turboquant", label: "TurboQuant", color: "#60a5fa" },
    { id: "rotorquant", label: "RotorQuant", color: "#34d399" },
    { id: "chaosengine", label: "ChaosEngine", color: "#f59e0b" },
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
  const allLibraryFormats = filteredLibraryRows.map(({ displayFormat }) => displayFormat);
  const allLibraryBackends = filteredLibraryRows.map(({ displayBackend }) => displayBackend);

  return (
    <div className="content-grid">
      <Panel
        title="My Models"
        subtitle={`${filteredLibraryRows.length} models / ${sizeLabel(libraryTotalSizeGb)} on disk / ${enabledDirectoryCount} directories`}
        className="span-2"
        actions={
          <input
            className="text-input discover-search"
            type="search"
            placeholder="Filter by name, path, format, quant, or backend..."
            value={librarySearchInput}
            onChange={(event) => onLibrarySearchInputChange(event.target.value)}
          />
        }
      >
        {renderCapabilityFilterBar(libraryCapFilter, onLibraryCapFilterChange, allLibraryCaps)}
        {renderFormatFilterBar(libraryFormatFilter, onLibraryFormatFilterChange, allLibraryFormats)}
        {renderFormatFilterBar(libraryBackendFilter, onLibraryBackendFilterChange, allLibraryBackends, "All backends")}
        {strategyCompat ? (
          <div className="cap-filter-bar">
            <button
              className={`cap-filter-btn${strategyFilter === null ? " cap-filter-btn--active" : ""}`}
              type="button"
              onClick={() => setStrategyFilter(null)}
            >
              All strategies
            </button>
            {STRATEGY_FILTERS.map((sf) => {
              const count = filteredLibraryRows.filter((row) => modelSupportsStrategy(row, sf.id)).length;
              // DFlash gets a more explanatory tooltip when zero models
              // match — speculative-decode drafts are pinned per family,
              // so users land on "0" often unless they have a base
              // Qwen3 / Llama-3.1 / gpt-oss / Kimi model.
              const tooltip = sf.id === "dflash" && count === 0
                ? "DFlash speculative-decode drafts only exist for specific base models: "
                  + "Qwen/Qwen3-{4B,8B}, Qwen/Qwen3-Coder-{4B,8B,30B-A3B,Next}, Qwen/Qwen3.5-{4B,7B,9B,14B,27B,35B-A3B}, "
                  + "Qwen/Qwen3.6-35B-A3B, meta-llama/Llama-3.1-8B-Instruct, gpt-oss-{20B,120B}, moonshotai/Kimi-K2.5. "
                  + "Fine-tunes typically don't match. Download a base model from Discover to enable DFlash."
                : `Show models compatible with ${sf.label} (${count})`;
              return (
                <button
                  key={sf.id}
                  className={`cap-filter-btn${strategyFilter === sf.id ? " cap-filter-btn--active" : ""}`}
                  type="button"
                  onClick={() => setStrategyFilter(strategyFilter === sf.id ? null : sf.id)}
                  title={tooltip}
                  style={strategyFilter === sf.id ? { borderColor: sf.color, color: sf.color, background: `${sf.color}15` } : undefined}
                >
                  {sf.label} ({count})
                </button>
              );
            })}
          </div>
        ) : null}
        {capFilteredLibrary.length ? (
          <div className="library-full-table">
            <div className="library-head">
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("name")}>Model{sortIndicator("name")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("format")}>Format{sortIndicator("format")}</button>
              <span className="sort-header">Quant</span>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("backend")}>Backend{sortIndicator("backend")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("size")}>Size{sortIndicator("size")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("ram")}>RAM{sortIndicator("ram")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("compressed")}>Compressed{sortIndicator("compressed")}</button>
              <button className="sort-header" type="button" onClick={() => toggleLibrarySort("context")}>Context{sortIndicator("context")}</button>
              <span className="sort-header">Status</span>
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
                  ? { kind: "downloading", label: "Downloading", detail: downloadProgressLabel(downloadState) }
                  : isPaused && downloadState
                    ? { kind: "paused", label: "Paused", detail: downloadProgressLabel(downloadState) }
                    : isDownloadFailed && downloadState
                      ? { kind: "failed", label: "Failed", detail: downloadState.error ?? "Download failed" }
                      : showBroken
                        ? { kind: "incomplete", label: "Incomplete", detail: item.brokenReason ?? "Incomplete or broken" }
                        : { kind: "installed", label: "Installed" };
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
                            <span className="badge muted" title={downloadSizeTooltip(downloadState)}>Active download</span>
                          ) : null}
                        </div>
                        {matchedVariant ? renderCapabilityIcons(matchedVariant.capabilities, 5) : null}
                        {hasDownloadOverlay && downloadState ? (
                          <span className="library-download-tag">
                            <small className={`library-download-reason${isDownloadFailed ? " error" : ""}`}>
                              {libraryDownloadDetail(downloadState)}
                            </small>
                          </span>
                        ) : null}
                        {showBroken ? (
                          <span className="broken-tag">
                            <span className="badge warning">BROKEN</span>
                            <small className="broken-reason">{item.brokenReason ?? "Incomplete or broken"}</small>
                          </span>
                        ) : null}
                      </div>
                      <span>{displayFormat}</span>
                      <span>{displayQuantization ?? "-"}</span>
                      <span>{displayBackend}</span>
                      <span>{sizeLabel(item.sizeGb)}</span>
                      <span title="Rough resident memory at 8K context (weights + KV + framework)">
                        {estimatedRamGb != null ? `~${number(estimatedRamGb)} GB` : "?"}
                      </span>
                      <span title="Rough resident memory with a compressed KV cache strategy">
                        {estimatedCompressedGb != null ? `~${number(estimatedCompressedGb)} GB` : "?"}
                      </span>
                      <span>{matchedVariant?.contextWindow ?? ""}</span>
                      <span className="library-row-status">
                        <StatusIcon status={rowStatus.kind} label={rowStatus.label} detail={rowStatus.detail} />
                      </span>
                      <div className="library-row-actions" onClick={(e) => e.stopPropagation()}>
                        {hasDownloadOverlay && repo ? (
                          <>
                            {isDownloading ? (
                              <IconActionButton icon="pause" label="Pause download" onClick={() => onCancelModelDownload(repo)} />
                            ) : (
                              <IconActionButton icon={isDownloadFailed ? "retry" : "resume"} label={isDownloadFailed ? "Retry download" : "Resume download"} buttonStyle="primary" onClick={() => onDownloadModel(repo)} />
                            )}
                            <IconActionButton icon={isDownloading ? "cancel" : "delete"} label={isDownloading ? "Cancel download" : "Delete download"} danger onClick={() => onDeleteModelDownload(repo)} />
                          </>
                        ) : canRetryBrokenRepo ? (
                          <>
                            <IconActionButton icon="retry" label="Retry download" buttonStyle="primary" onClick={() => onDownloadModel(repo!)} />
                            <IconActionButton icon="delete" label="Delete download" danger onClick={() => onDeleteModelDownload(repo!)} />
                          </>
                        ) : (
                          <>
                            {!item.broken && displayFormat !== "MLX" ? (
                              <IconActionButton icon="convert" label="Convert model" buttonStyle="primary" className="action-convert" onClick={() => onPrepareLibraryConversion(item)} />
                            ) : null}
                            {!item.broken ? (
                              <>
                                <IconActionButton icon="chat" label="Chat with model" buttonStyle="primary" className="action-chat" onClick={() => onOpenModelSelector("chat", `library:${item.path}`)} />
                                <IconActionButton icon="server" label="Load for server" buttonStyle="primary" className="action-server" onClick={() => onOpenModelSelector("server", `library:${item.path}`)} />
                              </>
                            ) : null}
                          </>
                        )}
                        {!isSyntheticDownloadRow ? (
                          <IconActionButton icon="reveal" label={fileRevealLabel} title={fileRevealLabel} onClick={() => onRevealPath(item.path)} />
                        ) : null}
                        {!hasDownloadOverlay ? (
                          <IconActionButton icon="delete" label="Delete model" danger onClick={() => onDeleteModel(item)} />
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
            <p>No models found. Add directories in Settings to scan for local models.</p>
          </div>
        )}
      </Panel>
    </div>
  );
}
