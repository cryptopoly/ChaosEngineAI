import { useEffect, useMemo, useState } from "react";
import { Panel } from "../../components/Panel";
import type { DownloadStatus, InstallResult } from "../../api";
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
  videoWidth: number;
  onVideoWidthChange: (value: number) => void;
  videoHeight: number;
  onVideoHeightChange: (value: number) => void;
  videoNumFrames: number;
  onVideoNumFramesChange: (value: number) => void;
  videoFps: number;
  onVideoFpsChange: (value: number) => void;
  videoSteps: number;
  onVideoStepsChange: (value: number) => void;
  videoGuidance: number;
  onVideoGuidanceChange: (value: number) => void;
  onActiveTabChange: (tab: TabId) => void;
  onPreloadVideoModel: (variant: VideoModelVariant) => void;
  onUnloadVideoModel: (variant?: VideoModelVariant) => void;
  onVideoDownload: (repo: string) => void;
  onGenerateVideo: () => void;
  onOpenExternalUrl: (url: string) => void;
  onRestartServer: () => void;
  onInstallVideoOutputDeps: () => Promise<InstallResult>;
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
  videoWidth,
  onVideoWidthChange,
  videoHeight,
  onVideoHeightChange,
  videoNumFrames,
  onVideoNumFramesChange,
  videoFps,
  onVideoFpsChange,
  videoSteps,
  onVideoStepsChange,
  videoGuidance,
  onVideoGuidanceChange,
  onActiveTabChange,
  onPreloadVideoModel,
  onUnloadVideoModel,
  onVideoDownload,
  onGenerateVideo,
  onOpenExternalUrl,
  onRestartServer,
  onInstallVideoOutputDeps,
}: VideoStudioTabProps) {
  const [installingOutputDeps, setInstallingOutputDeps] = useState(false);
  const missingDependencies = videoRuntimeStatus.missingDependencies ?? [];
  // imageio + imageio-ffmpeg are the two pip packages diffusers video
  // pipelines need to export mp4s. Everything else we surface as a badge;
  // these two get a one-click install button instead because they're the
  // thing that actually blocks "Generate video" from producing an output.
  const mp4EncoderMissing = missingDependencies.some(
    (dep) => dep === "imageio" || dep === "imageio-ffmpeg",
  );
  const otherMissingDependencies = missingDependencies.filter(
    (dep) => dep !== "imageio" && dep !== "imageio-ffmpeg",
  );

  async function handleInstallOutputDeps() {
    if (installingOutputDeps) return;
    setInstallingOutputDeps(true);
    try {
      await onInstallVideoOutputDeps();
    } finally {
      setInstallingOutputDeps(false);
    }
  }
  // Only offer variants the user can actually generate with. We include
  // models that are currently downloading because the user will want to keep
  // them selected while the download finishes. Everything else lives in
  // Discover / My Models.
  const studioFamilies = useMemo(
    () =>
      videoCatalog
        .map((family) => ({
          ...family,
          variants: family.variants.filter((variant) => {
            if (variant.availableLocally) return true;
            if (variant.hasLocalData) return true;
            const downloadState = activeVideoDownloads[variant.repo];
            return downloadState?.state === "downloading" || downloadState?.state === "completed";
          }),
        }))
        .filter((family) => family.variants.length > 0),
    [videoCatalog, activeVideoDownloads],
  );
  const hasAnyInstalled = studioFamilies.length > 0;

  // Ensure a valid model is selected once the catalog loads. Prefer an
  // installed model; fall back to the first catalog entry so the studio
  // still renders a stub when nothing is downloaded yet.
  useEffect(() => {
    if (selectedVideoModelId) {
      const stillValid = videoCatalog.some((family) =>
        family.variants.some((variant) => variant.id === selectedVideoModelId),
      );
      if (stillValid) return;
    }
    const installed = studioFamilies[0]?.variants[0];
    if (installed?.id) {
      onSelectedVideoModelIdChange(installed.id);
      return;
    }
    const fallback = defaultVideoVariantForFamily(videoCatalog[0]);
    if (fallback?.id) onSelectedVideoModelIdChange(fallback.id);
  }, [selectedVideoModelId, videoCatalog, studioFamilies, onSelectedVideoModelIdChange]);

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
            {mp4EncoderMissing ? (
              <span className="badge warning">mp4 encoder missing</span>
            ) : null}
            {otherMissingDependencies.slice(0, 4).map((dependency) => (
              <span key={dependency} className="badge subtle">{dependency}</span>
            ))}
          </div>
          {mp4EncoderMissing ? (
            <div className="image-runtime-actions">
              <p className="muted-text">
                Video generation needs imageio + imageio-ffmpeg to write mp4 files. Install them
                into the backend environment now?
              </p>
              <button
                className="primary-button"
                type="button"
                onClick={() => void handleInstallOutputDeps()}
                disabled={installingOutputDeps || !backendOnline}
              >
                {installingOutputDeps ? "Installing..." : "Install mp4 encoder"}
              </button>
            </div>
          ) : null}
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
            {hasAnyInstalled ? (
              <select
                className="text-input"
                value={selectedVideoModelId}
                onChange={(event) => onSelectedVideoModelIdChange(event.target.value)}
              >
                {studioFamilies.flatMap((family) =>
                  family.variants.map((variant) => {
                    const downloadState = activeVideoDownloads[variant.repo];
                    const isDownloadingVariant = downloadState?.state === "downloading";
                    const suffix = variant.availableLocally
                      ? " (installed)"
                      : isDownloadingVariant
                        ? ` (${downloadProgressLabel(downloadState)})`
                        : " (incomplete)";
                    return (
                      <option key={variant.id} value={variant.id}>
                        {variant.name} — {family.name}
                        {suffix}
                      </option>
                    );
                  }),
                )}
              </select>
            ) : (
              <div className="callout image-callout">
                <p>No video models installed yet. Browse the catalog to download one.</p>
                <div className="button-row">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => onActiveTabChange("video-discover")}
                  >
                    Open Video Discover
                  </button>
                </div>
              </div>
            )}
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

          {/*
            Per-run knobs. We expose these because Wan 2.1 / LTX defaults at
            full resolution + step count can detonate Apple Silicon's MPS
            backend (the attention QK^T matrix scales with width × height ×
            num_frames squared — a 73 GB allocation killed the sidecar at
            832x480 × 96 frames × 50 steps during testing). Letting the user
            dial down resolution / frames / steps is the only way to keep
            consumer hardware in the safe envelope.

            ``numFrames`` step is 4 because Wan-family pipelines require
            ``(num_frames - 1) % 4 == 0``; the parent hook re-snaps on
            generate as a defensive backstop.
          */}
          <div className="field-grid image-field-grid">
            <label>
              Width
              <input
                className="text-input"
                type="number"
                min={256}
                max={2048}
                step={64}
                value={videoWidth}
                onChange={(event) => onVideoWidthChange(Number(event.target.value) || 832)}
              />
            </label>
            <label>
              Height
              <input
                className="text-input"
                type="number"
                min={256}
                max={2048}
                step={64}
                value={videoHeight}
                onChange={(event) => onVideoHeightChange(Number(event.target.value) || 480)}
              />
            </label>
            <label>
              Frames
              <input
                className="text-input"
                type="number"
                min={1}
                max={257}
                step={4}
                value={videoNumFrames}
                onChange={(event) => onVideoNumFramesChange(Number(event.target.value) || 33)}
              />
            </label>
            <label>
              FPS
              <input
                className="text-input"
                type="number"
                min={1}
                max={60}
                value={videoFps}
                onChange={(event) => onVideoFpsChange(Number(event.target.value) || 24)}
              />
            </label>
            <label>
              Steps
              <input
                className="text-input"
                type="number"
                min={1}
                max={100}
                value={videoSteps}
                onChange={(event) => onVideoStepsChange(Number(event.target.value) || 30)}
              />
            </label>
            <label>
              Guidance
              <input
                className="text-input"
                type="number"
                min={1}
                max={20}
                step={0.5}
                value={videoGuidance}
                onChange={(event) => onVideoGuidanceChange(Number(event.target.value) || 5)}
              />
            </label>
          </div>

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
