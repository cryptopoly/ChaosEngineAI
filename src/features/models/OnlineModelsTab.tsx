import { useTranslation } from "react-i18next";
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
  findLibraryItemForVariant,
  downloadProgressLabel,
  formatReleaseLabel,
  handleActionKeyDown,
} from "../../utils";
import { CAPABILITY_META } from "../../constants";
import { CapabilityStrip } from "../../components/CapabilityStrip";

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
  // ``label`` is the English fallback. The render site re-resolves the
  // localized label via ``t("onlineModels.memory.<kind>Label")`` so it
  // follows the active locale; this string is only used when callers
  // need a sensible default outside i18n context.
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
  const { t } = useTranslation("library");

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
          {t("onlineModels.filter.all", { defaultValue: "All" })}
        </button>
        {uniqueCaps.map((cap) => {
          const meta = CAPABILITY_META[cap];
          const localizedTitle = t(`onlineModels.capability.title.${cap}`, { defaultValue: meta?.title ?? cap });
          const localizedShort = t(`onlineModels.capability.short.${cap}`, { defaultValue: meta?.shortLabel ?? cap });
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
    const resolvedAllLabel = allLabel ?? t("onlineModels.filter.allFormats", { defaultValue: "All formats" });
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
      <section className="discover-section" aria-label={t("onlineModels.section.curatedAria", { defaultValue: "Curated model families" })}>
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
                      {localCount > 0 ? <StatusIcon status="installed" label={t("onlineModels.status.countInstalled", { defaultValue: "{count} installed", count: localCount })} /> : null}
                      {headerIsDownloading ? (
                        <StatusIcon status="downloading" label={t("onlineModels.status.downloading", { defaultValue: "Downloading" })} detail={headerDownload ? downloadProgressLabel(headerDownload) : null} />
                      ) : headerIsPaused ? (
                        <StatusIcon status="paused" label={t("onlineModels.status.paused", { defaultValue: "Paused" })} detail={headerDownload ? downloadProgressLabel(headerDownload) : null} />
                      ) : headerIsFailed ? (
                        <StatusIcon status="failed" label={t("onlineModels.status.downloadFailed", { defaultValue: "Download failed" })} detail={headerDownload?.error ?? null} />
                      ) : null}
                    </div>
                    <p>{family.headline}</p>
                    <div className="discover-card-meta">
                      <CapabilityStrip capabilities={family.capabilities} max={8} />
                      <small>{t("onlineModels.variantCount", { defaultValue: "{count, plural, one {# variant} other {# variants}}", count: family.variants.length })}</small>
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
                      title={t("onlineModels.action.detailsTooltip", { defaultValue: "Show full details in a focused view" })}
                    >
                      {t("onlineModels.action.details", { defaultValue: "Details" })}
                    </button>
                    <span className="discover-chevron">{isExpanded ? "\u25B2" : "\u25BC"}</span>
                  </div>
                </div>

                {isExpanded ? (
                  <div className="discover-card-body">
                    <p className="discover-summary">{family.summary}</p>
                    <div className="discover-variant-table">
                      <div className="discover-variant-head">
                        <span>{t("onlineModels.column.variant", { defaultValue: "Variant" })}</span>
                        <span>{t("onlineModels.column.format", { defaultValue: "Format" })}</span>
                        <span>{t("onlineModels.column.backend", { defaultValue: "Backend" })}</span>
                        <span>{t("onlineModels.column.params", { defaultValue: "Params" })}</span>
                        <span>{t("onlineModels.column.size", { defaultValue: "Size" })}</span>
                        <span>{t("onlineModels.column.ram", { defaultValue: "RAM" })}</span>
                        <span>{t("onlineModels.column.compressed", { defaultValue: "Compressed" })}</span>
                        <span>{t("onlineModels.column.context", { defaultValue: "Context" })}</span>
                        <span>{t("onlineModels.column.status", { defaultValue: "Status" })}</span>
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
                          ? { kind: "installed", label: variant.availableLocally ? t("onlineModels.status.installed", { defaultValue: "Installed" }) : t("onlineModels.status.downloadComplete", { defaultValue: "Download complete" }) }
                          : isDownloading && downloadState
                            ? { kind: "downloading", label: t("onlineModels.status.downloading", { defaultValue: "Downloading" }), detail: downloadProgressLabel(downloadState) }
                            : isDownloadPaused && downloadState
                              ? { kind: "paused", label: t("onlineModels.status.paused", { defaultValue: "Paused" }), detail: downloadProgressLabel(downloadState) }
                              : isDownloadFailed && downloadState
                                ? { kind: "failed", label: t("onlineModels.status.failed", { defaultValue: "Failed" }), detail: downloadState.error ?? t("onlineModels.status.downloadFailed", { defaultValue: "Download failed" }) }
                                : { kind: "incomplete", label: t("onlineModels.status.notInstalled", { defaultValue: "Not installed" }) };
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
                                <CapabilityStrip capabilities={variant.capabilities} max={4} />
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
                                          ? t("onlineModels.memory.comfortable", {
                                              defaultValue: "Fits comfortably in {available} GB available",
                                              available: availableMemoryGb?.toFixed(1),
                                            })
                                          : fit.kind === "tight"
                                          ? t("onlineModels.memory.tight", {
                                              defaultValue: "Fits but tight against {available} GB available — close other apps before loading",
                                              available: availableMemoryGb?.toFixed(1),
                                            })
                                          : t("onlineModels.memory.over", {
                                              defaultValue: "Estimated {estimated} GB exceeds {available} GB available — try a smaller quantisation",
                                              estimated: variant.estimatedMemoryGb?.toFixed?.(1) ?? "?",
                                              available: availableMemoryGb?.toFixed(1),
                                            })
                                      }
                                    >
                                      {t(`onlineModels.memory.${fit.kind}Label`, { defaultValue: fit.label })}
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
                                      <IconActionButton icon="convert" label={t("onlineModels.action.convertModel", { defaultValue: "Convert model" })} buttonStyle="primary" className="action-convert" onClick={() => onPrepareCatalogConversion(variant)} />
                                    ) : null}
                                    <IconActionButton icon="chat" label={t("onlineModels.action.chatWithModel", { defaultValue: "Chat with model" })} buttonStyle="primary" className="action-chat" onClick={() => onOpenModelSelector("thread", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)} />
                                    <IconActionButton icon="server" label={t("onlineModels.action.loadForServer", { defaultValue: "Load for server" })} buttonStyle="primary" className="action-server" onClick={() => onOpenModelSelector("server", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)} />
                                  </>
                                ) : isDownloading ? (
                                  <>
                                    <IconActionButton icon="pause" label={t("onlineModels.action.pauseDownload", { defaultValue: "Pause download" })} onClick={() => onCancelModelDownload(variant.repo)} />
                                    <IconActionButton icon="cancel" label={t("onlineModels.action.cancelDownload", { defaultValue: "Cancel download" })} danger onClick={() => onDeleteModelDownload(variant.repo)} />
                                  </>
                                ) : isDownloadPaused ? (
                                  <>
                                    <IconActionButton icon="resume" label={t("onlineModels.action.resumeDownload", { defaultValue: "Resume download" })} onClick={() => onDownloadModel(variant.repo)} />
                                    <IconActionButton icon="delete" label={t("onlineModels.action.deleteDownload", { defaultValue: "Delete download" })} danger onClick={() => onDeleteModelDownload(variant.repo)} />
                                  </>
                                ) : isDownloadFailed ? (
                                  <>
                                    <IconActionButton icon="retry" label={t("onlineModels.action.retryDownload", { defaultValue: "Retry download" })} onClick={() => onDownloadModel(variant.repo)} />
                                    <IconActionButton icon="delete" label={t("onlineModels.action.deleteDownload", { defaultValue: "Delete download" })} danger onClick={() => onDeleteModelDownload(variant.repo)} />
                                  </>
                                ) : isDownloadComplete ? (
                                  null
                                ) : (
                                  <IconActionButton icon="download" label={t("onlineModels.action.downloadModel", { defaultValue: "Download model" })} onClick={() => onDownloadModel(variant.repo)} />
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
                                  <IconActionButton icon="huggingFace" label={t("onlineModels.action.openModelCardHF", { defaultValue: "Open model card on Hugging Face" })} onClick={() => onOpenExternalUrl(variant.link)} />
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
      <section className="discover-section discover-section--hub" aria-label={t("onlineModels.hub.sectionAria", { defaultValue: "Hugging Face Hub results" })}>
        <div className="hub-section-header">
          <span className="eyebrow">{t("onlineModels.hub.header", { defaultValue: "HuggingFace Hub" })}</span>
          <p>
            {t("onlineModels.hub.liveResults", {
              defaultValue: "{count, plural, one {# live result} other {# live results}} from huggingface.co, sorted by most recent update",
              count: filteredHubResults.length,
            })}
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
              ? { kind: "installed", label: t("onlineModels.status.installed", { defaultValue: "Installed" }) }
              : isDownloadComplete
                ? { kind: "downloaded", label: t("onlineModels.status.downloadComplete", { defaultValue: "Download complete" }) }
                : isDownloading && downloadState
                  ? { kind: "downloading", label: t("onlineModels.status.downloading", { defaultValue: "Downloading" }), detail: downloadProgressLabel(downloadState) }
                  : isDownloadPaused && downloadState
                    ? { kind: "paused", label: t("onlineModels.status.paused", { defaultValue: "Paused" }), detail: downloadProgressLabel(downloadState) }
                    : isDownloadFailed && downloadState
                      ? { kind: "failed", label: t("onlineModels.status.failed", { defaultValue: "Failed" }), detail: downloadState.error ?? t("onlineModels.status.downloadFailed", { defaultValue: "Download failed" }) }
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
                      <p className="muted-text">{t("onlineModels.hub.loading", { defaultValue: "Loading file list from Hugging Face..." })}</p>
                    ) : errorMsg ? (
                      <div className="callout error">
                        <p>{errorMsg}</p>
                      </div>
                    ) : fileData ? (
                      <>
                        {fileData.warning ? (
                          <div className="callout quiet">
                            <div className="chip-row">
                              <span className="badge warning">{t("onlineModels.hub.previewUnavailable", { defaultValue: "Preview unavailable" })}</span>
                            </div>
                            <p>{fileData.warning}</p>
                          </div>
                        ) : null}
                        <div className="hub-detail-meta">
                          {fileData.license ? <span className="badge muted">{t("onlineModels.hub.licenseLabel", { defaultValue: "License: {license}", license: fileData.license })}</span> : null}
                          {fileData.pipelineTag ? <span className="badge muted">{fileData.pipelineTag}</span> : null}
                          {fileData.totalSizeGb ? <span className="badge muted">{t("onlineModels.hub.totalGb", { defaultValue: "{size} GB total", size: number(fileData.totalSizeGb) })}</span> : null}
                          {fileData.lastModified ? <span className="badge muted">{t("onlineModels.hub.updatedDate", { defaultValue: "Updated {date}", date: fileData.lastModified.slice(0, 10) })}</span> : null}
                        </div>
                        {fileData.tags.length > 0 ? (
                          <div className="hub-detail-tags">
                            {fileData.tags.slice(0, 12).map((tag) => (
                              <span key={tag} className="badge muted hub-tag">{tag}</span>
                            ))}
                            {fileData.tags.length > 12 ? <small className="muted-text">{t("onlineModels.hub.moreTags", { defaultValue: "+{count} more", count: fileData.tags.length - 12 })}</small> : null}
                          </div>
                        ) : null}
                        {fileData.files.length === 0 ? (
                          <p className="muted-text">{t("onlineModels.hub.filePreviewUnavailable", { defaultValue: "File preview is not available for this repo right now." })}</p>
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
                                      <span className="eyebrow">{t("onlineModels.hub.weightsHeader", { defaultValue: "Weights ({count})", count: weights.length })}</span>
                                      <span className="muted-text">{largestWeight
                                        ? t("onlineModels.hub.largestShard", { defaultValue: "Largest shard {size}", size: largestWeight })
                                        : t("onlineModels.hub.showFiles", { defaultValue: "Show files" })}</span>
                                    </summary>
                                    <ul className="hub-file-list">
                                      {weights.map((f) => (
                                        <li key={f.path}>
                                          <code>{f.path}</code>
                                          <span className="muted-text">{f.sizeGb ? `${number(f.sizeGb)} GB` : ""}</span>
                                          {f.kind === "vision_projector" ? <span className="badge muted">{t("onlineModels.hub.visionTag", { defaultValue: "vision" })}</span> : null}
                                        </li>
                                      ))}
                                    </ul>
                                  </details>
                                ) : (
                                  <div className="hub-file-group">
                                    <span className="eyebrow">{t("onlineModels.hub.weightsHeader", { defaultValue: "Weights ({count})", count: weights.length })}</span>
                                    <ul className="hub-file-list">
                                      {weights.map((f) => (
                                        <li key={f.path}>
                                          <code>{f.path}</code>
                                          <span className="muted-text">{f.sizeGb ? `${number(f.sizeGb)} GB` : ""}</span>
                                          {f.kind === "vision_projector" ? <span className="badge muted">{t("onlineModels.hub.visionTag", { defaultValue: "vision" })}</span> : null}
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                )
                              ) : null}
                              {tokenizer.length > 0 ? (
                                <div className="hub-file-group">
                                  <span className="eyebrow">{t("onlineModels.hub.configAndTokenizer", { defaultValue: "Config & tokenizer" })}</span>
                                  <ul className="hub-file-list">
                                    {tokenizer.map((f) => (
                                      <li key={f.path}><code>{f.path}</code></li>
                                    ))}
                                  </ul>
                                </div>
                              ) : null}
                              {other.length > 0 ? (
                                <details className="hub-file-extras">
                                  <summary>{t("onlineModels.hub.otherFiles", { defaultValue: "+{count, plural, one {# other file} other {# other files}}", count: other.length })}</summary>
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
                          <IconActionButton icon="chat" label={t("onlineModels.action.chatWithModel", { defaultValue: "Chat with model" })} buttonStyle="primary" className="action-chat" onClick={() => onOpenModelSelector("thread")} />
                          <IconActionButton icon="server" label={t("onlineModels.action.loadForServer", { defaultValue: "Load for server" })} buttonStyle="primary" className="action-server" onClick={() => onOpenModelSelector("server")} />
                        </>
                      ) : isDownloading ? (
                        <>
                          <IconActionButton icon="pause" label={t("onlineModels.action.pauseDownload", { defaultValue: "Pause download" })} onClick={() => onCancelModelDownload(model.repo)} />
                          <IconActionButton icon="cancel" label={t("onlineModels.action.cancelDownload", { defaultValue: "Cancel download" })} danger onClick={() => onDeleteModelDownload(model.repo)} />
                        </>
                      ) : isDownloadPaused ? (
                        <>
                          <IconActionButton icon="resume" label={t("onlineModels.action.resumeDownload", { defaultValue: "Resume download" })} onClick={() => onDownloadModel(model.repo)} />
                          <IconActionButton icon="delete" label={t("onlineModels.action.deleteDownload", { defaultValue: "Delete download" })} danger onClick={() => onDeleteModelDownload(model.repo)} />
                        </>
                      ) : isDownloadFailed ? (
                        <>
                          <IconActionButton icon="retry" label={t("onlineModels.action.retryDownload", { defaultValue: "Retry download" })} onClick={() => onDownloadModel(model.repo)} />
                          <IconActionButton icon="delete" label={t("onlineModels.action.deleteDownload", { defaultValue: "Delete download" })} danger onClick={() => onDeleteModelDownload(model.repo)} />
                        </>
                      ) : isDownloadComplete ? (
                        <StatusIcon status="downloaded" label={t("onlineModels.status.downloadComplete", { defaultValue: "Download complete" })} />
                      ) : (
                        <IconActionButton icon="download" label={t("onlineModels.action.downloadModel", { defaultValue: "Download model" })} buttonStyle="primary" onClick={() => onDownloadModel(model.repo)} />
                      )}
                      <IconActionButton icon="huggingFace" label={t("onlineModels.action.openOnHF", { defaultValue: "Open on Hugging Face" })} onClick={() => onOpenExternalUrl(model.link)} />
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
        title={t("common:panels.discoverModels", { defaultValue: "Discover Models" })}
        subtitle={t("onlineModels.subtitle", {
          defaultValue: "{count} model families / {downloaded} downloaded locally",
          count: searchResults.length,
          downloaded: localVariantCount,
        })}
        className="span-2 discover-panel"
        actions={
          <input
            className="text-input discover-search"
            type="search"
            placeholder={t("onlineModels.searchPlaceholder", { defaultValue: "Search by name, provider, or capability..." })}
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
            <p className="muted-text">{t("onlineModels.searchError.fallback", { defaultValue: "Showing the last successful Discover results." })}</p>
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
            <p>{discoverCapFilter
              ? t("onlineModels.empty.capFilter", {
                  defaultValue: "No models match the \"{cap}\" filter.",
                  cap: t(`onlineModels.capability.short.${discoverCapFilter}`, { defaultValue: CAPABILITY_META[discoverCapFilter]?.shortLabel ?? discoverCapFilter }),
                })
              : searchInput
                ? t("onlineModels.empty.search", {
                    defaultValue: "No models match \"{query}\". Try a different search term.",
                    query: searchInput,
                  })
                : t("onlineModels.empty.prompt", { defaultValue: "Type to search for models." })}</p>
          </div>
        ) : null}
      </Panel>
    </div>
  );
}
