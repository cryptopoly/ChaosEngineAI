import { useState } from "react";
import { Panel } from "../../components/Panel";
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
  expandedLibraryPath: string | null;
  onExpandedLibraryPathChange: (path: string | null) => void;
  fileRevealLabel: string;
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
  expandedLibraryPath,
  onExpandedLibraryPathChange,
  fileRevealLabel,
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
              return (
                <button
                  key={sf.id}
                  className={`cap-filter-btn${strategyFilter === sf.id ? " cap-filter-btn--active" : ""}`}
                  type="button"
                  onClick={() => setStrategyFilter(strategyFilter === sf.id ? null : sf.id)}
                  title={`Show models compatible with ${sf.label} (${count})`}
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
              <span className="sort-header"></span>
            </div>
            <div className="library-rows">
              {capFilteredLibrary.map(({ item, matchedVariant, displayFormat, displayQuantization, displayBackend, sourceKind }) => {
                const isExpanded = expandedLibraryPath === item.path;
                return (
                  <div key={item.path} className={`library-item-wrap${isExpanded ? " expanded" : ""}`}>
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
                        </div>
                        {matchedVariant ? renderCapabilityIcons(matchedVariant.capabilities, 5) : null}
                        {item.broken ? (
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
                      <span>{matchedVariant?.estimatedMemoryGb ? `~${number(matchedVariant.estimatedMemoryGb)}GB` : "?"}</span>
                      <span>{matchedVariant?.estimatedCompressedMemoryGb ? `~${number(matchedVariant.estimatedCompressedMemoryGb)}GB` : "?"}</span>
                      <span>{matchedVariant?.contextWindow ?? ""}</span>
                      <div className="library-row-actions" onClick={(e) => e.stopPropagation()}>
                        {displayFormat !== "MLX" ? (
                          <button className="primary-button action-convert" type="button" onClick={() => onPrepareLibraryConversion(item)}>CONVERT</button>
                        ) : null}
                        <button className="primary-button action-chat" type="button" onClick={() => onOpenModelSelector("chat", `library:${item.path}`)}>CHAT</button>
                        <button className="primary-button action-server" type="button" onClick={() => onOpenModelSelector("server", `library:${item.path}`)}>SERVER</button>
                        <button className="secondary-button icon-button" type="button" title={fileRevealLabel} onClick={() => onRevealPath(item.path)}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                            <polyline points="15 3 21 3 21 9" />
                            <line x1="10" y1="14" x2="21" y2="3" />
                          </svg>
                        </button>
                        <button className="secondary-button icon-button danger-button" type="button" title="Delete model" onClick={() => onDeleteModel(item)}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                            <path d="M10 11v6" />
                            <path d="M14 11v6" />
                            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                          </svg>
                        </button>
                      </div>
                    </div>
                    {isExpanded ? (
                      <div className="library-item-detail">
                        <div className="library-detail-left">
                          <p className="mono-text library-path">{item.path}</p>
                          {matchedVariant?.note ? <p className="variant-note">{matchedVariant.note}</p> : null}
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
