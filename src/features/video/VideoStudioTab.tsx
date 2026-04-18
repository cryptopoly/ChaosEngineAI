import { useEffect, useMemo } from "react";
import { Panel } from "../../components/Panel";
import type { DownloadStatus } from "../../api";
import type {
  TabId,
  TauriBackendInfo,
  VideoModelFamily,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../../types";
import {
  defaultVideoVariantForFamily,
  downloadProgressLabel,
  number,
  sizeLabel,
} from "../../utils";

export interface VideoStudioTabProps {
  videoCatalog: VideoModelFamily[];
  selectedVideoModelId: string;
  onSelectedVideoModelIdChange: (id: string) => void;
  selectedVideoVariant: VideoModelVariant | null;
  selectedVideoFamily: VideoModelFamily | null;
  selectedVideoLoaded: boolean;
  selectedVideoWillLoadOnGenerate: boolean;
  videoRuntimeLoadedDifferentModel: boolean;
  loadedVideoVariant: VideoModelVariant | null;
  videoRuntimeStatus: VideoRuntimeStatus;
  tauriBackend: TauriBackendInfo | null;
  busy: boolean;
  busyAction: string | null;
  videoBusy: boolean;
  videoBusyLabel: string | null;
  backendOnline: boolean;
  activeVideoDownloads: Record<string, DownloadStatus>;
  videoPrompt: string;
  onVideoPromptChange: (value: string) => void;
  videoNegativePrompt: string;
  onVideoNegativePromptChange: (value: string) => void;
  videoUseRandomSeed: boolean;
  onVideoUseRandomSeedChange: (value: boolean) => void;
  videoSeedInput: string;
  onVideoSeedInputChange: (value: string) => void;
  onActiveTabChange: (tab: TabId) => void;
  onPreloadVideoModel: (variant: VideoModelVariant) => void;
  onUnloadVideoModel: (variant?: VideoModelVariant) => void;
  onVideoDownload: (repo: string) => void;
  onGenerateVideo: () => void;
  onOpenExternalUrl: (url: string) => void;
  onRestartServer: () => void;
}

export function VideoStudioTab({
  videoCatalog,
  selectedVideoModelId,
  onSelectedVideoModelIdChange,
  selectedVideoVariant,
  selectedVideoFamily,
  selectedVideoLoaded,
  selectedVideoWillLoadOnGenerate,
  videoRuntimeLoadedDifferentModel,
  loadedVideoVariant,
  videoRuntimeStatus,
  tauriBackend,
  busy,
  busyAction,
  videoBusy,
  videoBusyLabel,
  backendOnline,
  activeVideoDownloads,
  videoPrompt,
  onVideoPromptChange,
  videoNegativePrompt,
  onVideoNegativePromptChange,
  videoUseRandomSeed,
  onVideoUseRandomSeedChange,
  videoSeedInput,
  onVideoSeedInputChange,
  onActiveTabChange,
  onPreloadVideoModel,
  onUnloadVideoModel,
  onVideoDownload,
  onGenerateVideo,
  onOpenExternalUrl,
  onRestartServer,
}: VideoStudioTabProps) {
  // Ensure a valid model is selected once the catalog loads
  useEffect(() => {
    if (selectedVideoModelId) return;
    const fallback = defaultVideoVariantForFamily(videoCatalog[0]);
    if (fallback?.id) onSelectedVideoModelIdChange(fallback.id);
  }, [selectedVideoModelId, videoCatalog, onSelectedVideoModelIdChange]);

  const downloadState = useMemo(
    () => (selectedVideoVariant ? activeVideoDownloads[selectedVideoVariant.repo] : undefined),
    [activeVideoDownloads, selectedVideoVariant],
  );
  const isDownloading = downloadState?.state === "downloading";
  const isDownloaded =
    !!selectedVideoVariant && (selectedVideoVariant.availableLocally || downloadState?.state === "completed");
  const hasPrompt = videoPrompt.trim().length > 0;
  const generationDisabled =
    !selectedVideoVariant
    || !isDownloaded
    || !videoRuntimeStatus.realGenerationAvailable
    || !hasPrompt
    || videoBusy
    || !backendOnline;
  const generateButtonLabel =
    videoBusy && videoBusyLabel?.startsWith("Generating")
      ? videoBusyLabel
      : "Generate video";
  const generateTitle = !selectedVideoVariant
    ? "Choose a video model first."
    : !isDownloaded
      ? `${selectedVideoVariant.name} is not installed locally yet.`
      : !videoRuntimeStatus.realGenerationAvailable
        ? (videoRuntimeStatus.message || "Video runtime is not ready.")
        : !hasPrompt
          ? "Write a prompt before generating."
          : !backendOnline
            ? "Backend is offline."
            : videoBusy
              ? (videoBusyLabel ?? "Busy…")
              : "Start generating this clip.";

  return (
    <div className="content-grid image-page-grid">
      <Panel
        title="Video Studio"
        subtitle={selectedVideoVariant?.name ?? "Choose a video model to get started"}
        className="span-2"
        actions={
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-discover")}>
              Browse Catalog
            </button>
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-models")}>
              Installed Models
            </button>
          </div>
        }
      >
        <div className="callout image-callout image-runtime-callout">
          <p>{videoRuntimeStatus.message}</p>
          <div className="chip-row">
            <span className={`badge ${videoRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
              {videoRuntimeStatus.realGenerationAvailable ? "Real engine ready" : "Fallback active"}
            </span>
            <span className="badge muted">Engine: {videoRuntimeStatus.activeEngine}</span>
            {videoRuntimeStatus.device ? <span className="badge muted">Device: {videoRuntimeStatus.device}</span> : null}
            {loadedVideoVariant ? (
              <span className="badge accent">Loaded: {loadedVideoVariant.name}</span>
            ) : null}
            {(videoRuntimeStatus.missingDependencies ?? []).slice(0, 4).map((dependency) => (
              <span key={dependency} className="badge subtle">{dependency}</span>
            ))}
          </div>
          {!videoRuntimeStatus.realGenerationAvailable && tauriBackend?.managedByTauri ? (
            <div className="image-runtime-actions">
              <button className="secondary-button" type="button" onClick={() => onRestartServer()} disabled={busy}>
                {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
              </button>
            </div>
          ) : null}
        </div>

        <div className="image-studio-grid" style={{ display: "grid", gap: "1rem", gridTemplateColumns: "1fr" }}>
          <label>
            Video Model
            <select
              className="text-input"
              value={selectedVideoModelId}
              onChange={(event) => onSelectedVideoModelIdChange(event.target.value)}
            >
              {videoCatalog.flatMap((family) =>
                family.variants.map((variant) => (
                  <option key={variant.id} value={variant.id}>
                    {variant.name} — {family.name}
                    {variant.availableLocally ? " (installed)" : ""}
                  </option>
                )),
              )}
            </select>
          </label>

          {selectedVideoVariant ? (
            <div className="image-library-stats">
              <span>{sizeLabel(selectedVideoVariant.sizeGb)}</span>
              <span>{selectedVideoVariant.recommendedResolution}</span>
              <span>{number(selectedVideoVariant.defaultDurationSeconds)}s clip</span>
              <span className="badge subtle">{selectedVideoFamily?.name ?? selectedVideoVariant.provider}</span>
              {isDownloaded ? (
                <span className="badge success">Installed</span>
              ) : isDownloading ? (
                <span className="badge accent">{downloadProgressLabel(downloadState)}</span>
              ) : (
                <span className="badge warning">Not downloaded</span>
              )}
              {selectedVideoLoaded ? <span className="badge accent">In Memory</span> : null}
              {videoRuntimeLoadedDifferentModel && loadedVideoVariant ? (
                <span className="badge muted">Loaded model: {loadedVideoVariant.name}</span>
              ) : null}
            </div>
          ) : null}

          <label>
            Prompt
            <textarea
              className="text-input"
              rows={3}
              value={videoPrompt}
              onChange={(event) => onVideoPromptChange(event.target.value)}
              placeholder="A cinematic drone shot of a misty pine forest at dawn..."
            />
          </label>

          <label>
            Negative prompt
            <input
              className="text-input"
              type="text"
              value={videoNegativePrompt}
              onChange={(event) => onVideoNegativePromptChange(event.target.value)}
              placeholder="Optional: things to avoid (low quality, watermark, etc.)"
            />
          </label>

          <div className="button-row">
            <label className="inline-label" style={{ display: "flex", alignItems: "center", gap: ".4rem" }}>
              <input
                type="checkbox"
                checked={videoUseRandomSeed}
                onChange={(event) => onVideoUseRandomSeedChange(event.target.checked)}
              />
              Random seed
            </label>
            {!videoUseRandomSeed ? (
              <input
                className="text-input"
                type="number"
                value={videoSeedInput}
                onChange={(event) => onVideoSeedInputChange(event.target.value)}
                placeholder="Seed (integer)"
                style={{ maxWidth: 200 }}
              />
            ) : null}
          </div>

          <div className="button-row">
            {selectedVideoVariant && !isDownloaded && !isDownloading ? (
              <button
                className="secondary-button"
                type="button"
                disabled={!backendOnline}
                onClick={() => selectedVideoVariant && onVideoDownload(selectedVideoVariant.repo)}
              >
                Download model
              </button>
            ) : null}
            {selectedVideoVariant && isDownloaded && !selectedVideoLoaded ? (
              <button
                className="secondary-button"
                type="button"
                disabled={videoBusy || !videoRuntimeStatus.realGenerationAvailable}
                onClick={() => selectedVideoVariant && onPreloadVideoModel(selectedVideoVariant)}
              >
                {videoBusy && videoBusyLabel?.includes("Loading") ? videoBusyLabel : "Load into memory"}
              </button>
            ) : null}
            {selectedVideoLoaded ? (
              <button
                className="secondary-button"
                type="button"
                disabled={videoBusy}
                onClick={() => selectedVideoVariant && onUnloadVideoModel(selectedVideoVariant)}
              >
                {videoBusy && videoBusyLabel?.includes("Unloading") ? videoBusyLabel : "Unload"}
              </button>
            ) : null}
            <button
              className="primary-button"
              type="button"
              disabled={generationDisabled}
              title={generateTitle}
              onClick={() => onGenerateVideo()}
            >
              {generateButtonLabel}
            </button>
            {selectedVideoVariant ? (
              <button
                className="secondary-button"
                type="button"
                onClick={() => onOpenExternalUrl(selectedVideoVariant.link)}
              >
                Model Card
              </button>
            ) : null}
          </div>

          {selectedVideoWillLoadOnGenerate ? (
            <p className="muted-text">
              The selected model will be loaded into memory on the next generate. First load can take a
              minute for the larger variants.
            </p>
          ) : null}
        </div>
      </Panel>
    </div>
  );
}
