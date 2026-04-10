import { Panel } from "../../components/Panel";
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
  handleActionKeyDown,
} from "../../utils";
import { CAPABILITY_META } from "../../constants";

export interface OnlineModelsTabProps {
  searchResults: ModelFamily[];
  searchInput: string;
  onSearchInputChange: (value: string) => void;
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
}

export function OnlineModelsTab({
  searchResults,
  searchInput,
  onSearchInputChange,
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
  let filteredResults = searchResults;
  if (discoverCapFilter) {
    filteredResults = filteredResults.filter((f) => f.capabilities.includes(discoverCapFilter) || f.variants.some((v) => v.capabilities.includes(discoverCapFilter!)));
  }
  if (discoverFormatFilter) {
    filteredResults = filteredResults.filter((f) => f.variants.some((v) => v.format === discoverFormatFilter));
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
        {filteredResults.length > 0 || hubResults.length > 0 ? (
          <div className="discover-sections">
            {filteredResults.length > 0 ? (
              <section className="discover-section" aria-label="Curated model families">
                <div className="discover-list">
                  {filteredResults.map((family) => {
                    const isExpanded = expandedFamilyId === family.id;
                    const localCount = family.variants.filter((v) => v.availableLocally).length;
                    const paramRange = family.variants.length > 1
                      ? `${number(Math.min(...family.variants.map((v) => v.paramsB)))}B - ${number(Math.max(...family.variants.map((v) => v.paramsB)))}B`
                      : `${number(family.variants[0]?.paramsB ?? 0)}B`;
                    const formats = [...new Set(family.variants.map((v) => v.format))];
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
                              {localCount > 0 ? <span className="badge success">{localCount} downloaded</span> : null}
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
                                      <span>{variant.estimatedMemoryGb ? `~${number(variant.estimatedMemoryGb)}GB` : "?"}</span>
                                      <span>{variant.estimatedCompressedMemoryGb ? `~${number(variant.estimatedCompressedMemoryGb)}GB` : "?"}</span>
                                      <span>{variant.contextWindow}</span>
                                      <div className="discover-variant-actions" onClick={(e) => e.stopPropagation()}>
                                        {variant.availableLocally ? (
                                          <>
                                            {variant.launchMode === "convert" ? (
                                              <button className="primary-button action-convert" type="button" onClick={() => onPrepareCatalogConversion(variant)}>CONVERT</button>
                                            ) : null}
                                            <button className="primary-button action-chat" type="button" onClick={() => onOpenModelSelector("thread", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)}>CHAT</button>
                                            <button className="primary-button action-server" type="button" onClick={() => onOpenModelSelector("server", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`)}>SERVER</button>
                                          </>
                                        ) : isDownloading ? (
                                          <>
                                            <span className="badge accent">{downloadProgressLabel(downloadState).toUpperCase()}</span>
                                            <button className="secondary-button" type="button" onClick={() => onCancelModelDownload(variant.repo)}>PAUSE</button>
                                            <button className="secondary-button danger-button" type="button" onClick={() => onDeleteModelDownload(variant.repo)}>CANCEL</button>
                                          </>
                                        ) : isDownloadPaused ? (
                                          <>
                                            <span className="badge warning">{downloadProgressLabel(downloadState).toUpperCase()}</span>
                                            <button className="secondary-button" type="button" onClick={() => onDownloadModel(variant.repo)}>RESUME</button>
                                            <button className="secondary-button danger-button" type="button" onClick={() => onDeleteModelDownload(variant.repo)}>DELETE</button>
                                          </>
                                        ) : isDownloadFailed ? (
                                          <>
                                            <span className="badge warning">DOWNLOAD FAILED</span>
                                            <button className="secondary-button" type="button" onClick={() => onDownloadModel(variant.repo)}>RETRY</button>
                                            <button className="secondary-button danger-button" type="button" onClick={() => onDeleteModelDownload(variant.repo)}>DELETE</button>
                                          </>
                                        ) : isDownloadComplete ? (
                                          <span className="badge success">DOWNLOAD COMPLETE</span>
                                        ) : (
                                          <button className="secondary-button" type="button" onClick={() => onDownloadModel(variant.repo)}>DOWNLOAD</button>
                                        )}
                                      </div>
                                    </div>
                                    {isVariantExpanded ? (
                                      <div className="variant-detail-expand">
                                        <div className="variant-detail-left">
                                          <p>{variant.note}</p>
                                          {matchedLocal ? <p className="mono-text variant-local-path">{matchedLocal.path}</p> : null}
                                          <a
                                            className="text-link"
                                            href={variant.link}
                                            target="_blank"
                                            rel="noreferrer"
                                            onClick={(event) => {
                                              event.preventDefault();
                                              onOpenExternalUrl(variant.link);
                                            }}
                                          >
                                            Open model card on HuggingFace
                                          </a>
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
            ) : null}

            {hubResults.length > 0 ? (
              <section className="discover-section discover-section--hub" aria-label="Hugging Face Hub results">
                <div className="hub-section-header">
                  <span className="eyebrow">HuggingFace Hub</span>
                  <p>{hubResults.length} results from huggingface.co</p>
                </div>
                <div className="discover-list">
                  {hubResults
                    .filter((model) => {
                      if (discoverFormatFilter && model.format !== discoverFormatFilter) return false;
                      return true;
                    })
                    .map((model) => {
                      const isExpanded = expandedHubId === model.id;
                      const fileData = hubFileCache[model.id];
                      const loading = !!hubFileLoading[model.id];
                      const errorMsg = hubFileError[model.id];
                      const downloadState = activeDownloads[model.repo];
                      const isDownloading = downloadState?.state === "downloading";
                      const isDownloadPaused = downloadState?.state === "cancelled";
                      const isDownloadFailed = downloadState?.state === "failed";
                      const isDownloadComplete = downloadState?.state === "completed";
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
                                {model.availableLocally ? <span className="badge success">Downloaded</span> : null}
                                {!model.availableLocally && isDownloadComplete ? <span className="badge success">Download complete</span> : null}
                              </div>
                              <div className="discover-card-meta">
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
                                    return (
                                      <div className="hub-file-groups">
                                        {weights.length > 0 ? (
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
                              <div className="button-row">
                                {model.availableLocally ? (
                                  <>
                                    <button className="primary-button action-chat" type="button" onClick={() => onOpenModelSelector("thread")}>Chat</button>
                                    <button className="primary-button action-server" type="button" onClick={() => onOpenModelSelector("server")}>Server</button>
                                  </>
                                ) : isDownloading ? (
                                  <>
                                    <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
                                    <button className="secondary-button" type="button" onClick={() => onCancelModelDownload(model.repo)}>
                                      Pause
                                    </button>
                                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteModelDownload(model.repo)}>
                                      Cancel
                                    </button>
                                  </>
                                ) : isDownloadPaused ? (
                                  <>
                                    <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                                    <button
                                      className="secondary-button"
                                      type="button"
                                      onClick={() => onDownloadModel(model.repo)}
                                    >
                                      Resume
                                    </button>
                                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteModelDownload(model.repo)}>
                                      Delete
                                    </button>
                                  </>
                                ) : isDownloadFailed ? (
                                  <>
                                    <span className="badge warning">Download failed</span>
                                    <button className="secondary-button" type="button" onClick={() => onDownloadModel(model.repo)}>
                                      Retry
                                    </button>
                                    <button className="secondary-button danger-button" type="button" onClick={() => onDeleteModelDownload(model.repo)}>
                                      Delete
                                    </button>
                                  </>
                                ) : isDownloadComplete ? (
                                  <span className="badge success">Download complete</span>
                                ) : (
                                  <button
                                    className="primary-button"
                                    type="button"
                                    onClick={() => onDownloadModel(model.repo)}
                                  >
                                    Download
                                  </button>
                                )}
                                <a
                                  className="text-link"
                                  href={model.link}
                                  target="_blank"
                                  rel="noreferrer"
                                  onClick={(event) => {
                                    event.preventDefault();
                                    onOpenExternalUrl(model.link);
                                  }}
                                >
                                  Open on HuggingFace ↗
                                </a>
                              </div>
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                </div>
              </section>
            ) : null}
          </div>
        ) : null}

        {filteredResults.length === 0 && hubResults.length === 0 ? (
          <div className="empty-state">
            <p>{discoverCapFilter ? `No models match the "${CAPABILITY_META[discoverCapFilter]?.shortLabel ?? discoverCapFilter}" filter.` : searchInput ? `No models match "${searchInput}". Try a different search term.` : "Type to search for models."}</p>
          </div>
        ) : null}
      </Panel>
    </div>
  );
}
