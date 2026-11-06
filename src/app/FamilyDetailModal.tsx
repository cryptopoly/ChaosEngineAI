import { useTranslation } from "react-i18next";
import { CapabilityStrip } from "../components/CapabilityStrip";
import { downloadProgressLabel, findLibraryItemForVariant, number, sizeLabel } from "../utils";
import type { AppTabContext } from "./context";

/** Model family details dialog opened from the Chat Models page. */
export function FamilyDetailModal({ ctx }: { ctx: AppTabContext }) {
  const { t: tCommon } = useTranslation("common");
  const { workspace, chatLibrary, prepareCatalogConversion, openModelSelector, handleOpenExternalUrl } = ctx;
  const {
    detailFamilyId, setDetailFamilyId, activeDownloads,
    handleDownloadModel, handleCancelModelDownload, handleDeleteModelDownload,
  } = ctx.models;
  if (!detailFamilyId) return null;
  const family = workspace.featuredModels.find((f) => f.id === detailFamilyId);
  if (!family) return null;
  const localCount = family.variants.filter((v) => v.availableLocally).length;
  return (
    <div className="modal-overlay" onClick={() => setDetailFamilyId(null)}>
      <div className="modal-content modal-wide detail-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="detail-header-main">
            <h3>{family.name}</h3>
            <div className="detail-header-badges">
              <span className="badge muted">{family.provider}</span>
              {family.variants.map((v) => v.format).filter((f, i, a) => a.indexOf(f) === i).map((f) => (
                <span key={f} className="badge muted">{f}</span>
              ))}
              {localCount > 0 ? <span className="badge success">{tCommon("discoverModal.downloadedCount", { count: localCount, defaultValue: "{count} downloaded" })}</span> : null}
              <span className="badge muted">{family.updatedLabel}</span>
            </div>
            <p className="muted-text">{family.headline}</p>
          </div>
        </div>
        <div className="modal-body detail-modal-body">
          <p className="discover-summary">{family.summary}</p>
          <CapabilityStrip capabilities={family.capabilities} max={12} />
          <div className="detail-variants">
            <span className="eyebrow">{tCommon("discoverModal.variants", { count: family.variants.length, defaultValue: "Variants ({count})" })}</span>
            {family.variants.map((variant) => {
              const matchedLocal = findLibraryItemForVariant(chatLibrary, variant);
              const downloadState = activeDownloads[variant.repo];
              const isDownloading = downloadState?.state === "downloading";
              const isDownloadPaused = downloadState?.state === "cancelled";
              const isDownloadFailed = downloadState?.state === "failed";
              const isDownloadComplete = downloadState?.state === "completed";
              return (
                <div key={variant.id} className={`detail-variant${variant.availableLocally || isDownloadComplete ? " downloaded" : ""}`}>
                  <div className="detail-variant-info">
                    <strong>{variant.name}</strong>
                    <div className="detail-variant-meta">
                      <span>{variant.format}</span>
                      <span>{variant.quantization}</span>
                      <span>{number(variant.paramsB)}B params</span>
                      <span>{sizeLabel(variant.sizeGb)}</span>
                      <span>~{number(variant.estimatedMemoryGb ?? 0)} GB RAM</span>
                      <span>{variant.contextWindow} ctx</span>
                    </div>
                    {variant.note ? <p className="detail-variant-note">{variant.note}</p> : null}
                  </div>
                  <div className="detail-variant-actions">
                    {variant.availableLocally ? (
                      <>
                        {variant.launchMode === "convert" ? (
                          <button className="primary-button action-convert" type="button" onClick={() => { prepareCatalogConversion(variant); setDetailFamilyId(null); }}>{tCommon("discoverModal.convert", { defaultValue: "Convert" })}</button>
                        ) : null}
                        <button className="primary-button action-chat" type="button" onClick={() => { openModelSelector("thread", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`); setDetailFamilyId(null); }}>{tCommon("discoverModal.chat", { defaultValue: "Chat" })}</button>
                        <button className="primary-button action-server" type="button" onClick={() => { openModelSelector("server", matchedLocal ? `library:${matchedLocal.path}` : `catalog:${variant.id}`); setDetailFamilyId(null); }}>{tCommon("discoverModal.server", { defaultValue: "Server" })}</button>
                      </>
                    ) : isDownloading ? (
                      <>
                        <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
                        <button className="secondary-button" type="button" onClick={() => void handleCancelModelDownload(variant.repo)}>{tCommon("discoverModal.pause", { defaultValue: "Pause" })}</button>
                        <button className="secondary-button danger-button" type="button" onClick={() => void handleDeleteModelDownload(variant.repo)}>{tCommon("actions.cancel", { defaultValue: "Cancel" })}</button>
                      </>
                    ) : isDownloadPaused ? (
                      <>
                        <span className="badge warning">{downloadProgressLabel(downloadState)}</span>
                        <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>{tCommon("discoverModal.resume", { defaultValue: "Resume" })}</button>
                        <button className="secondary-button danger-button" type="button" onClick={() => void handleDeleteModelDownload(variant.repo)}>{tCommon("actions.delete", { defaultValue: "Delete" })}</button>
                      </>
                    ) : isDownloadFailed ? (
                      <>
                        <span className="badge warning">{tCommon("discoverModal.downloadFailed", { defaultValue: "Download failed" })}</span>
                        <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>{tCommon("actions.retry", { defaultValue: "Retry" })}</button>
                        <button className="secondary-button danger-button" type="button" onClick={() => void handleDeleteModelDownload(variant.repo)}>{tCommon("actions.delete", { defaultValue: "Delete" })}</button>
                      </>
                    ) : isDownloadComplete ? (
                      <span className="badge success">{tCommon("discoverModal.downloadComplete", { defaultValue: "Download complete" })}</span>
                    ) : (
                      <button className="secondary-button" type="button" onClick={() => void handleDownloadModel(variant.repo)}>{tCommon("actions.download", { defaultValue: "Download" })}</button>
                    )}
                    <a
                      className="text-link"
                      href={variant.link}
                      target="_blank"
                      rel="noreferrer"
                      onClick={(event) => { event.preventDefault(); void handleOpenExternalUrl(variant.link); }}
                    >
                      HF ↗
                    </a>
                  </div>
                </div>
              );
            })}
          </div>
          {family.readme && family.readme.length > 0 ? (
            <div className="detail-readme">
              <span className="eyebrow">{tCommon("discoverModal.overview", { defaultValue: "Overview" })}</span>
              {family.readme.map((line, i) => <p key={i}>{line}</p>)}
            </div>
          ) : null}
        </div>
        <div className="modal-footer">
          <button className="primary-button" type="button" onClick={() => setDetailFamilyId(null)}>{tCommon("actions.close", { defaultValue: "Close" })}</button>
        </div>
      </div>
    </div>
  );
}
