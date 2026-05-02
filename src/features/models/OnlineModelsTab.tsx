import { Panel } from "../../components/Panel";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import type { ModelStatusKind } from "../../components/ModelActionIcons";
import type { DownloadStatus } from "../../api";
import type {
  HubFileListResponse,
  HubModel,
  LibraryItem,
  ModelFamily,
  ModelVariant,
} from "../../types";
import {
  number,
  sizeLabel,
  capabilityMeta,
  findLibraryItemForVariant,
  downloadProgressLabel,
  formatReleaseLabel,
  handleActionKeyDown,
} from "../../utils";
import { CAPABILITY_META } from "../../constants";

export interface OnlineModelsTabProps {
  searchResults: ModelFamily[];
  searchInput: string;
  onSearchInputChange: (value: string) => void;
  searchError: string | null;
  localVariantCount: number;
  discoverCapFilter: string | null;
  onDiscoverCapFilterChange: (cap: string | null) => void;
  discoverFormatFilter: string | null;
  onDiscoverFormatFilterChange: (fmt: string | null) => void;
  expandedFamilyId: string | null;
  onExpandedFamilyIdChange: (id: string | null) => void;
  expandedVariantId: string | null;
  onExpandedVariantIdChange: (id: string | null) => void;
  onDetailFamilyIdChange: (id: string) => void;
  library: LibraryItem[];
  activeDownloads: Record<string, DownloadStatus>;
  onDownloadModel: (repo: string) => void;
  onCancelModelDownload: (repo: string) => void;
  onDeleteModelDownload: (repo: string) => void;
  onPrepareCatalogConversion: (variant: ModelVariant) => void;
  onOpenModelSelector: (action: "chat" | "server" | "thread", preselectedKey?: string) => void;
  onOpenExternalUrl: (url: string) => void;
  hubResults: HubModel[];
  expandedHubId: string | null;
  onToggleHubExpand: (repo: string) => void;
  hubFileCache: Record<string, HubFileListResponse>;
  hubFileLoading: Record<string, boolean>;
  hubFileError: Record<string, string>;
  /** Phase 2.14: drives the per-variant fit-in-memory badge. */
  availableMemoryGb?: number | null;
}

/**
 * Phase 2.14: classify whether a variant fits the current host's
 * available memory. Three buckets: comfortable / tight / over.
 *
 * - comfortable: estimated memory ≤ 70% of available
 * - tight: estimated memory ≤ 100% of available
 * - over: estimated memory > available
 *
 * Returns null when neither size nor estimate is known. The hint
 * is optimistic on purpose — TurboQuant / ChaosEngine compression
 * can reclaim ~50% of the listed estimate, so "tight" is still a
 * usable signal rather than a hard block.
 */
export function memoryFitBucket(
  variant: ModelVariant,
  availableMemoryGb: number | null | undefined,
): { kind: "comfortable" | "tight" | "over" | "unknown"; label: string } {
  if (availableMemoryGb == null || availableMemoryGb <= 0) {
    return { kind: "unknown", label: "" };
  }
  const estimate = variant.estimatedMemoryGb ?? variant.sizeGb;
  if (!estimate || estimate <= 0) {
    return { kind: "unknown", label: "" };
  }
  if (estimate <= availableMemoryGb * 0.7) {
    return { kind: "comfortable", label: "Fits" };
  }
  if (estimate <= availableMemoryGb) {
    return { kind: "tight", label: "Tight" };
  }
  return { kind: "over", label: "Too big" };
}

export function OnlineModelsTab({
  searchResults,
  searchInput,
  onSearchInputChange,
  searchError,
  localVariantCount,
  discoverCapFilter,
  onDiscoverCapFilterChange,
  discoverFormatFilter,
  onDiscoverFormatFilterChange,
  expandedFamilyId,
  onExpandedFamilyIdChange,
  expandedVariantId,
  onExpandedVariantIdChange,
  onDetailFamilyIdChange,
  library,
  activeDownloads,
  onDownloadModel,
  onCancelModelDownload,
  onDeleteModelDownload,
  onPrepareCatalogConversion,
  onOpenModelSelector,
  onOpenExternalUrl,
  hubResults,
  expandedHubId,
  onToggleHubExpand,
  hubFileCache,
  hubFileLoading,
  hubFileError,
  availableMemoryGb,
}: OnlineModelsTabProps) {
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

  const allDiscoverCaps = searchResults.flatMap((f) => f.capabilities);
  const allDiscoverFormats = searchResults.flatMap((f) => f.variants.map((v) => v.format));
  const hasActiveQuery = searchInput.trim().length > 0;
  let filteredResults = searchResults;
  if (discoverCapFilter) {
    filteredResults = filteredResults.filter((f) => f.capabilities.includes(discoverCapFilter) || f.variants.some((v) => v.capabilities.includes(discoverCapFilter!)));
  }
  if (discoverFormatFilter) {
    filteredResults = filteredResults.filter((f) => f.variants.some((v) => v.format === discoverFormatFilter));
  }
  const filteredHubResults = [...hubResults]
    .filter((model) => {
      if (discoverFormatFilter && model.format !== discoverFormatFilter) return false;
      return true;
    })
    .sort((left, right) => {
      const leftTime = Date.parse(left.lastModified ?? "") || 0;
      const rightTime = Date.parse(right.lastModified ?? "") || 0;
      if (rightTime !== leftTime) return rightTime - leftTime;
      if (right.downloads !== left.downloads) return right.downloads - left.downloads;
      return right.likes - left.likes;
    });

  function renderCuratedSection() {
    if (filteredResults.length === 0) return null;
    return (
      <section className="discover-section" aria-label="Curated model families">
        <div className="discover-list">
          {filteredResults.map((family) => {
            const isExpanded = expandedFamilyId === family.id;
            const localCount = family.variants.filter((v) => v.availableLocally).length;
            const paramRange = family.variants.length > 1
              ? `${number(Math.min(...family.variants.map((v) => v.paramsB)))}B - ${number(Math.max(...family.variants.map((v) => v.paramsB)))}B`
              : `${number(family.variants[0]?.paramsB ?? 0)}B`;
            const formats = [...new Set(family.variants.map((v) => v.format))];
            // Pick the first variant with an active download to surface in
            // the collapsed header — most families download one variant at
            // a time, so a single badge captures the in-flight state without
            // cluttering the header or forcing the user to expand the card.
            const headerDownload = family.variants
              .map((v) => activeDownloads[v.repo])
              .find((state) => state?.state === "downloading" || state?.state === "cancelled" || state?.state === "failed") ?? null;
            const headerIsDownloading = headerDownload?.state === "downloading";
            const headerIsPaused = headerDownload?.state === "cancelled";
            const headerIsFailed = headerDownload?.state === "failed";
            return (
              <div key={family.id} className={`discover-card${isExpanded ? " expanded" : ""}`}>
                <div
                  className="discover-card-header discover-card-header--interactive"
                  role="button"
                  tabIndex={0}
                  onClick={() => { onExpandedFamilyIdChange(isExpanded ? null : family.id); onExpandedVariantIdChange(null); }}
                  onKeyDown={(event) => handleActionKeyDown(event, () => {
                    onExpandedFamilyIdChange(isExpanded ? null : family.id);
                    onExpandedVariantIdChange(null);
                  })}
                >
                  <div className="discover-card-info">
                    <div className="discover-card-title">
                      <strong>{family.name}</strong>
                      <span className="badge muted">{family.provider}</span>
                      <span className="badge muted">{paramRange}</span>
                      {formats.map((f) => <span key={f} className="badge muted">{f}</span>)}
                      {localCount > 0 ? <StatusIcon status="installed" label={`${localCount} installed`} /> : null}
                      {headerIsDownloading ? (
                        <StatusIcon status="downloading" label="Downloading" detail={headerDownload ? downloadProgressLabel(headerDownload) : null} />
                      ) : headerIsPaused ? (
                        <StatusIcon status="paused" label="Paused" detail={headerDownload ? downloadProgressLabel(headerDownload) : null} />
                      ) : headerIsFailed ? (
                        <StatusIcon status="failed" label="Download failed" detail={headerDownload?.error ?? null} />
                      ) : null}
                    </div>
                    <p>{family.headline}</p>
                    <div className="discover-card-meta">
                      {renderCapabilityIcons(family.capabilities, 8)}
                      <small>{family.variants.length} variants</small>
                      <small>{family.updatedLabel}</small>
                    </div>
                  </div>
                  <div className="discover-card-head-actions">
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDetailFamilyIdChange(family.id);
                      }}
                      title="Show full details in a focused view"
                    >
                      Details
                    </button>
                    <span className="discover-chevron">{isExpanded ? "\u25B2" : "\u25BC"}</span>
                  </div>
                </div>

                {isExpanded ? (
                  <div className="discover-card-body">
                    <p className="discover-summary">{family.summary}</p>
                    <div className="discover-variant-table">
                      <div className="discover-variant-head">
                        <span>Variant</span>
                        <span>Format</span>
                        <span>Backend</span>
                        <span>Params</span>
                        <span>Size</span>
                        <span>RAM</span>
                        <span>Compressed</span>
                        <span>Context</span>
                        <span>Status</span>
                        <span></span>
                      </div>
                      {family.variants.map((variant) => {
                        const matchedLocal = findLibraryItemForVariant(library, variant);
                        const isVariantExpanded = expandedVariantId === variant.id;
                        const downloadState = activeDownloads[variant.repo];
                        const isDownloading = downloadState?.state === "downloading";
                        const isDownloadPaused = downloadState?.state === "cancelled";
                        const isDownloadFailed = downloadState?.state === "failed";
                        const isDownloadComplete = downloadState?.state === "completed";
                        const variantStatus: { kind: ModelStatusKind; label: string; detail?: string | null } = variant.availableLocally || isDownloadComplete
                          ? { kind: "installed", label: variant.availableLocally ? "Installed" : "Download complete" }
                          : isDownloading && downloadState
                            ? { kind: "downloading", label: "Downloading", detail: downloadProgressLabel(downloadState) }
                            : isDownloadPaused && downloadState
                              ? { kind: "paused", label: "Paused", detail: downloadProgressLabel(downloadState) }
                              : isDownloadFailed && downloadState
                                ? { kind: "failed", label: "Failed", detail: downloadState.error ?? "Download failed" }
                                : { kind: "incomplete", label: "Not installed" };
                        return (
                          <div key={variant.id}>
                            <div
                              className={`discover-variant-row${isVariantExpanded ? " expanded" : ""}${variant.availableLocally || isDownloadComplete ? " downloaded" : ""}`}
                              onClick={() => onExpandedVariantIdChange(isVariantExpanded ? null : variant.id)}
                              role="button"
                              tabIndex={0}
                            >
                              <div className="discover-variant-name">
                                <strong>{variant.name}</strong>
                                {renderCapabilityIcons(variant.capabilities, 4)}
                              </div>
                              <span>{variant.format} / {variant.quantization}</span>
                              <span>{variant.backend}</span>
                              <span>{number(variant.paramsB)}B</span>
                              <span>{sizeLabel(variant.sizeGb)}</span>
                              <span>
                                {variant.estimatedMemoryGb ? `~${number(variant.estimatedMemoryGb)}GB` : "?"}
                                {(() => {
                                  const fit = memoryFitBucket(variant, availableMemoryGb);
                                  if (fit.kind === "unknown") return null;
                                  return (
                                    <span
                                      className={`memory-fit-badge memory-fit-badge--${fit.kind}`}
                                      title={
                                        fit.kind === "comfortable"
                                          ? `Fits comfortably in ${availableMemoryGb?.toFixed(1)} GB available`
                                          : fit.kind === "tight"
                                          ? `Fits but tight against ${availableMemoryGb?.toFixed(1)} GB available — close other apps before loading`
                                          : `Estimated ${variant.estimatedMemoryGb?.toFixed?.(1) ?? "?"} GB exceeds ${availableMemoryGb?.toFixed(1)} GB available — try a smaller quantisation`
                                      }
                                    >
                                      {fit.label}
                                    </span>
                                  );
                                })()}
                              </span>
                              <span>{variant.estimatedCompressedMemoryGb ? `~${number(variant.estimatedCompressedMemoryGb)}GB` : "?"}</span>
                              <span>{variant.contextWindow}</span>
                              <span><StatusIcon status={variantStatus.kind} label={variantStatus.label} detail={variantStatus.detail} /></span>
                              <div className="discover-variant-actions" onClick={(e) => e.stopPropagation()}>
                                {variant.availableLocally ? (
                                  <>
                                    {variant.launchMode === "convert" ? (
                                      <IconActionButton icon="convert" label="Convert model" buttonStyle="primary" className="action-convert" onClick={() => onPrepareCatalogConversion(variant)} />
                                    ) : null}
                                    <IconActionButton icon="chat" label="Chat with model" buttonStyle="primary" className="action-chat" onClick={() => onOpenModelSelector("thread", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)} />
                                    <IconActionButton icon="server" label="Load for server" buttonStyle="primary" className="action-server" onClick={() => onOpenModelSelector("server", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)} />
                                  </>
                                ) : isDownloading ? (
                                  <>
                                    <IconActionButton icon="pause" label="Pause download" onClick={() => onCancelModelDownload(variant.repo)} />
                                    <IconActionButton icon="cancel" label="Cancel download" danger onClick={() => onDeleteModelDownload(variant.repo)} />
                                  </>
                                ) : isDownloadPaused ? (
                                  <>
                                    <IconActionButton icon="resume" label="Resume download" onClick={() => onDownloadModel(variant.repo)} />
                                    <IconActionButton icon="delete" label="Delete download" danger onClick={() => onDeleteModelDownload(variant.repo)} />
                                  </>
                                ) : isDownloadFailed ? (
                                  <>
                                    <IconActionButton icon="retry" label="Retry download" onClick={() => onDownloadModel(variant.repo)} />
                                    <IconActionButton icon="delete" label="Delete download" danger onClick={() => onDeleteModelDownload(variant.repo)} />
                                  </>
                                ) : isDownloadComplete ? (
                                  null
                                ) : (
                                  <IconActionButton icon="download" label="Download model" onClick={() => onDownloadModel(variant.repo)} />
                                )}
                              </div>
                            </div>
                            {isDownloadFailed && downloadState?.error ? (
                              <div className="callout error">
                                <p>{downloadState.error}</p>
                              </div>
                            ) : null}
                            {isVariantExpanded ? (
                              <div className="variant-detail-expand">
                                <div className="variant-detail-left">
                                  <p>{variant.note}</p>
                                  {formatReleaseLabel(variant.releaseLabel, variant.releaseDate) ? (
                                    <p className="muted-text variant-release-label">
                                      {formatReleaseLabel(variant.releaseLabel, variant.releaseDate)}
                                    </p>
                                  ) : null}
                                  {matchedLocal ? <p className="mono-text variant-local-path">{matchedLocal.path}</p> : null}
                                  <IconActionButton icon="huggingFace" label="Open model card on Hugging Face" onClick={() => onOpenExternalUrl(variant.link)} />
                                </div>
                              </div>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                    {family.readme.length > 0 ? (
                      <div className="discover-readme">
                        {family.readme.slice(0, 2).map((line, i) => <p key={i}>{line}</p>)}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </section>
    );
  }

  function renderHubSection() {
    if (filteredHubResults.length === 0) return null;
    return (
      <section className="discover-section discover-section--hub" aria-label="Hugging Face Hub results">
        <div className="hub-section-header">
          <span className="eyebrow">HuggingFace Hub</span>
          <p>
            {filteredHubResults.length} live result{filteredHubResults.length !== 1 ? "s" : ""} from huggingface.co, sorted by most recent update
          </p>
        </div>
        <div className="discover-list">
          {filteredHubResults.map((model) => {
            const isExpanded = expandedHubId === model.id;
            const fileData = hubFileCache[model.id];
            const loading = !!hubFileLoading[model.id];
            const errorMsg = hubFileError[model.id];
            const downloadState = activeDownloads[model.repo];
            const isDownloading = downloadState?.state === "downloading";
            const isDownloadPaused = downloadState?.state === "cancelled";
            const isDownloadFailed = downloadState?.state === "failed";
            const isDownloadComplete = downloadState?.state === "completed";
            const hubStatus: { kind: ModelStatusKind; label: string; detail?: string | null } | null = model.availableLocally
              ? { kind: "installed", label: "Installed" }
              : isDownloadComplete
                ? { kind: "downloaded", label: "Download complete" }
                : isDownloading && downloadState
                  ? { kind: "downloading", label: "Downloading", detail: downloadProgressLabel(downloadState) }
                  : isDownloadPaused && downloadState
                    ? { kind: "paused", label: "Paused", detail: downloadProgressLabel(downloadState) }
                    : isDownloadFailed && downloadState
                      ? { kind: "failed", label: "Failed", detail: downloadState.error ?? "Download failed" }
                      : null;
            return (
              <div key={model.id} className={`discover-card${isExpanded ? " expanded" : ""}`}>
                <div
                  className="discover-card-header discover-card-header--interactive"
                  role="button"
                  tabIndex={0}
                  onClick={() => onToggleHubExpand(model.id)}
                  onKeyDown={(event) => handleActionKeyDown(event, () => {
                    onToggleHubExpand(model.id);
                  })}
                >
                  <div className="discover-card-info">
                    <div className="discover-card-title">
                      <strong>{model.name}</strong>
                      <span className="badge muted">{model.provider}</span>
                      <span className={`badge ${model.format === "GGUF" ? "accent" : "muted"}`}>{model.format}</span>
                      {hubStatus ? <StatusIcon status={hubStatus.kind} label={hubStatus.label} detail={hubStatus.detail} /> : null}
                    </div>
                    <div className="discover-card-meta">
                      {formatReleaseLabel(model.releaseLabel, model.createdAt) ? (
                        <small>{formatReleaseLabel(model.releaseLabel, model.createdAt)}</small>
                      ) : null}
                      {model.updatedLabel ? <small>{model.updatedLabel}</small> : null}
                      <small>{model.downloadsLabel}</small>
                      <small>{model.likesLabel}</small>
                    </div>
                  </div>
                  <span className="discover-chevron">{isExpanded ? "\u25B2" : "\u25BC"}</span>
                </div>
                {isExpanded ? (
                  <div className="discover-card-body">
                    {loading ? (
                      <p className="muted-text">Loading file list from Hugging Face...</p>
                    ) : errorMsg ? (
                      <div className="callout error">
                        <p>{errorMsg}</p>
                      </div>
                    ) : fileData ? (
                      <>
                        {fileData.warning ? (
                          <div className="callout quiet">
                            <div className="chip-row">
                              <span className="badge warning">Preview unavailable</span>
                            </div>
                            <p>{fileData.warning}</p>
                          </div>
                        ) : null}
                        <div className="hub-detail-meta">
                          {fileData.license ? <span className="badge muted">License: {fileData.license}</span> : null}
                          {fileData.pipelineTag ? <span className="badge muted">{fileData.pipelineTag}</span> : null}
                          {fileData.totalSizeGb ? <span className="badge muted">{number(fileData.totalSizeGb)} GB total</span> : null}
                          {fileData.lastModified ? <span className="badge muted">Updated {fileData.lastModified.slice(0, 10)}</span> : null}
                        </div>
                        {fileData.tags.length > 0 ? (
                          <div className="hub-detail-tags">
                            {fileData.tags.slice(0, 12).map((tag) => (
                              <span key={tag} className="badge muted hub-tag">{tag}</span>
                            ))}
                            {fileData.tags.length > 12 ? <small className="muted-text">+{fileData.tags.length - 12} more</small> : null}
                          </div>
                        ) : null}
                        {fileData.files.length === 0 ? (
                          <p className="muted-text">File preview is not available for this repo right now.</p>
                        ) : (() => {
                          const weights = fileData.files.filter((f) => f.kind === "weight" || f.kind === "vision_projector");
                          const tokenizer = fileData.files.filter((f) => f.kind === "tokenizer" || f.kind === "config" || f.kind === "template");
                          const other = fileData.files.filter((f) => !weights.includes(f) && !tokenizer.includes(f));
                          const collapseWeightsByDefault = weights.length > 8;
                          const largestWeight = weights[0]?.sizeGb ? `${number(weights[0].sizeGb)} GB` : null;
                          return (
                            <div className="hub-file-groups">
                              {weights.length > 0 ? (
                                collapseWeightsByDefault ? (
                                  <details className="hub-file-group hub-file-group--collapsible">
                                    <summary>
                                      <span className="eyebrow">Weights ({weights.length})</span>
                                      <span className="muted-text">{largestWeight ? `Largest shard ${largestWeight}` : "Show files"}</span>
                                    </summary>
                                    <ul className="hub-file-list">
                                      {weights.map((f) => (
                                        <li key={f.path}>
                                          <code>{f.path}</code>
                                          <span className="muted-text">{f.sizeGb ? `${number(f.sizeGb)} GB` : ""}</span>
                                          {f.kind === "vision_projector" ? <span className="badge muted">vision</span> : null}
                                        </li>
                                      ))}
                                    </ul>
                                  </details>
                                ) : (
                                  <div className="hub-file-group">
                                    <span className="eyebrow">Weights ({weights.length})</span>
                                    <ul className="hub-file-list">
                                      {weights.map((f) => (
                                        <li key={f.path}>
                                          <code>{f.path}</code>
                                          <span className="muted-text">{f.sizeGb ? `${number(f.sizeGb)} GB` : ""}</span>
                                          {f.kind === "vision_projector" ? <span className="badge muted">vision</span> : null}
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                )
                              ) : null}
                              {tokenizer.length > 0 ? (
                                <div className="hub-file-group">
                                  <span className="eyebrow">Config &amp; tokenizer</span>
                                  <ul className="hub-file-list">
                                    {tokenizer.map((f) => (
                                      <li key={f.path}><code>{f.path}</code></li>
                                    ))}
                                  </ul>
                                </div>
                              ) : null}
                              {other.length > 0 ? (
                                <details className="hub-file-extras">
                                  <summary>+{other.length} other files</summary>
                                  <ul className="hub-file-list">
                                    {other.map((f) => (
                                      <li key={f.path}><code>{f.path}</code></li>
                                    ))}
                                  </ul>
                                </details>
                              ) : null}
                            </div>
                          );
                        })()}
                      </>
                    ) : null}
                    {isDownloadFailed && downloadState?.error ? (
                      <div className="callout error">
                        <p>{downloadState.error}</p>
                      </div>
                    ) : null}
                    <div className="button-row">
                      {model.availableLocally ? (
                        <>
                          <IconActionButton icon="chat" label="Chat with model" buttonStyle="primary" className="action-chat" onClick={() => onOpenModelSelector("thread")} />
                          <IconActionButton icon="server" label="Load for server" buttonStyle="primary" className="action-server" onClick={() => onOpenModelSelector("server")} />
                        </>
                      ) : isDownloading ? (
                        <>
                          <IconActionButton icon="pause" label="Pause download" onClick={() => onCancelModelDownload(model.repo)} />
                          <IconActionButton icon="cancel" label="Cancel download" danger onClick={() => onDeleteModelDownload(model.repo)} />
                        </>
                      ) : isDownloadPaused ? (
                        <>
                          <IconActionButton icon="resume" label="Resume download" onClick={() => onDownloadModel(model.repo)} />
                          <IconActionButton icon="delete" label="Delete download" danger onClick={() => onDeleteModelDownload(model.repo)} />
                        </>
                      ) : isDownloadFailed ? (
                        <>
                          <IconActionButton icon="retry" label="Retry download" onClick={() => onDownloadModel(model.repo)} />
                          <IconActionButton icon="delete" label="Delete download" danger onClick={() => onDeleteModelDownload(model.repo)} />
                        </>
                      ) : isDownloadComplete ? (
                        <StatusIcon status="downloaded" label="Download complete" />
                      ) : (
                        <IconActionButton icon="download" label="Download model" buttonStyle="primary" onClick={() => onDownloadModel(model.repo)} />
                      )}
                      <IconActionButton icon="huggingFace" label="Open on Hugging Face" onClick={() => onOpenExternalUrl(model.link)} />
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </section>
    );
  }

  return (
    <div className="content-grid discover-page">
      <Panel
        title="Discover Models"
        subtitle={`${searchResults.length} model families / ${localVariantCount} downloaded locally`}
        className="span-2 discover-panel"
        actions={
          <input
            className="text-input discover-search"
            type="search"
            placeholder="Search by name, provider, or capability..."
            value={searchInput}
            onChange={(event) => onSearchInputChange(event.target.value)}
          />
        }
      >
        {renderCapabilityFilterBar(discoverCapFilter, onDiscoverCapFilterChange, allDiscoverCaps)}
        {renderFormatFilterBar(discoverFormatFilter, onDiscoverFormatFilterChange, allDiscoverFormats)}
        {searchError ? (
          <div className="callout error">
            <p>{searchError}</p>
            <p className="muted-text">Showing the last successful Discover results.</p>
          </div>
        ) : null}
        {filteredResults.length > 0 || filteredHubResults.length > 0 ? (
          <div className="discover-sections">
            {hasActiveQuery ? renderHubSection() : null}
            {renderCuratedSection()}
            {!hasActiveQuery ? renderHubSection() : null}
          </div>
        ) : null}

        {filteredResults.length === 0 && filteredHubResults.length === 0 ? (
          <div className="empty-state">
            <p>{discoverCapFilter ? `No models match the "${CAPABILITY_META[discoverCapFilter]?.shortLabel ?? discoverCapFilter}" filter.` : searchInput ? `No models match "${searchInput}". Try a different search term.` : "Type to search for models."}</p>
          </div>
        ) : null}
      </Panel>
    </div>
  );
}
