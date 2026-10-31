/**
 * Live Hugging Face search results on the chat Models page.
 *
 * Kept as its own block (not table rows): a Hub result has no catalog
 * data, and its card lists the repo's files. Moved unchanged from the old
 * Discover tab, except that "Chat with model" and "Load for server" now
 * preselect this model in the launch modal (R3) instead of falling back
 * to the loaded one.
 */

import { useTranslation } from "react-i18next";
import type { DownloadStatus } from "../../api";
import type { HubFileListResponse, HubModel, LibraryItem, ModelFamily } from "../../types";
import { IconActionButton, StatusIcon } from "../../components/ModelActionIcons";
import type { ModelStatusKind } from "../../components/ModelActionIcons";
import { findLibraryItemForRepo } from "../../components/models/adapters/chat";
import { downloadProgressLabel, formatReleaseLabel, handleActionKeyDown, number } from "../../utils";

/** Format filter, then most recent update, then downloads, then likes. */
export function sortHubResults(hubResults: HubModel[], formatFilter: string | null): HubModel[] {
  return [...hubResults]
    .filter((model) => !formatFilter || model.format === formatFilter)
    .sort((left, right) => {
      const leftTime = Date.parse(left.lastModified ?? "") || 0;
      const rightTime = Date.parse(right.lastModified ?? "") || 0;
      if (rightTime !== leftTime) return rightTime - leftTime;
      if (right.downloads !== left.downloads) return right.downloads - left.downloads;
      return right.likes - left.likes;
    });
}

/** Launch modal preset for a Hub result (R3): the library item that is
 * this repo, else the catalog variant with this repo; ``undefined`` when
 * neither exists (the modal then offers the loaded model, as before). */
export function hubLaunchKey(model: Pick<HubModel, "repo">, library: LibraryItem[], families: ModelFamily[]): string | undefined {
  const item = findLibraryItemForRepo(library, model.repo);
  if (item) return `library:${item.path}`;
  const target = model.repo.toLowerCase();
  for (const family of families) {
    const variant = family.variants.find((candidate) => candidate.repo.toLowerCase() === target);
    if (variant) return `catalog:${variant.id}`;
  }
  return undefined;
}

export interface ChatHubResultsProps {
  /** Already filtered and sorted (``sortHubResults``). */
  results: HubModel[];
  expandedHubId: string | null;
  onToggleHubExpand: (repo: string) => void;
  hubFileCache: Record<string, HubFileListResponse>;
  hubFileLoading: Record<string, boolean>;
  hubFileError: Record<string, string>;
  activeDownloads: Record<string, DownloadStatus>;
  library: LibraryItem[];
  families: ModelFamily[];
  onDownloadModel: (repo: string) => void;
  onCancelModelDownload: (repo: string) => void;
  onDeleteModelDownload: (repo: string) => void;
  onOpenModelSelector: (action: "chat" | "server" | "thread", preselectedKey?: string) => void;
  onOpenExternalUrl: (url: string) => void;
}

export function ChatHubResults({
  results,
  expandedHubId,
  onToggleHubExpand,
  hubFileCache,
  hubFileLoading,
  hubFileError,
  activeDownloads,
  library,
  families,
  onDownloadModel,
  onCancelModelDownload,
  onDeleteModelDownload,
  onOpenModelSelector,
  onOpenExternalUrl,
}: ChatHubResultsProps) {
  const { t } = useTranslation("library");
  if (results.length === 0) return null;
  return (
    <section className="discover-section discover-section--hub chat-hub-results" aria-label={t("onlineModels.hub.sectionAria", { defaultValue: "Hugging Face Hub results" })}>
      <div className="hub-section-header">
        <span className="eyebrow">{t("onlineModels.hub.header", { defaultValue: "HuggingFace Hub" })}</span>
        <p>
          {t("onlineModels.hub.liveResults", {
            defaultValue: "{count, plural, one {# live result} other {# live results}} from huggingface.co, sorted by most recent update",
            count: results.length,
          })}
        </p>
      </div>
      <div className="discover-list">
        {results.map((model) => {
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
          const launchKey = model.availableLocally ? hubLaunchKey(model, library, families) : undefined;
          return (
            <div key={model.id} className={`discover-card${isExpanded ? " expanded" : ""}`} data-hub-id={model.id}>
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
                <span className="discover-chevron" aria-hidden="true">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" focusable="false">
                    <path d={isExpanded ? "m6 15 6-6 6 6" : "m6 9 6 6 6-6"} />
                  </svg>
                </span>
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
                    <HubFileDetail fileData={fileData} />
                  ) : null}
                  {isDownloadFailed && downloadState?.error ? (
                    <div className="callout error">
                      <p>{downloadState.error}</p>
                    </div>
                  ) : null}
                  <div className="button-row">
                    {model.availableLocally ? (
                      <>
                        <IconActionButton icon="chat" label={t("onlineModels.action.chatWithModel", { defaultValue: "Chat with model" })} buttonStyle="primary" className="action-chat" onClick={() => onOpenModelSelector("thread", launchKey)} />
                        <IconActionButton icon="server" label={t("onlineModels.action.loadForServer", { defaultValue: "Load for server" })} buttonStyle="primary" className="action-server" onClick={() => onOpenModelSelector("server", launchKey)} />
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

function HubFileDetail({ fileData }: { fileData: HubFileListResponse }) {
  const { t } = useTranslation("library");
  const weights = fileData.files.filter((f) => f.kind === "weight" || f.kind === "vision_projector");
  const tokenizer = fileData.files.filter((f) => f.kind === "tokenizer" || f.kind === "config" || f.kind === "template");
  const other = fileData.files.filter((f) => !weights.includes(f) && !tokenizer.includes(f));
  const collapseWeightsByDefault = weights.length > 8;
  const largestWeight = weights[0]?.sizeGb ? `${number(weights[0].sizeGb)} GB` : null;
  const weightList = (
    <ul className="hub-file-list">
      {weights.map((f) => (
        <li key={f.path}>
          <code>{f.path}</code>
          <span className="muted-text">{f.sizeGb ? `${number(f.sizeGb)} GB` : ""}</span>
          {f.kind === "vision_projector" ? <span className="badge muted">{t("onlineModels.hub.visionTag", { defaultValue: "vision" })}</span> : null}
        </li>
      ))}
    </ul>
  );
  const weightsHeader = t("onlineModels.hub.weightsHeader", { defaultValue: "Weights ({count})", count: weights.length });
  return (
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
      ) : (
        <div className="hub-file-groups">
          {weights.length > 0 ? (
            collapseWeightsByDefault ? (
              <details className="hub-file-group hub-file-group--collapsible">
                <summary>
                  <span className="eyebrow">{weightsHeader}</span>
                  <span className="muted-text">{largestWeight
                    ? t("onlineModels.hub.largestShard", { defaultValue: "Largest shard {size}", size: largestWeight })
                    : t("onlineModels.hub.showFiles", { defaultValue: "Show files" })}</span>
                </summary>
                {weightList}
              </details>
            ) : (
              <div className="hub-file-group">
                <span className="eyebrow">{weightsHeader}</span>
                {weightList}
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
      )}
    </>
  );
}
