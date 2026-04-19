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
  assessVideoGenerationSafety,
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

// Numeric input handling that tolerates transient empty states during editing.
// The naive pattern ``onChange={e => setValue(Number(e.target.value) || fallback)}``
// treats an empty string as ``0`` and snaps back to the fallback — which means
// the user can never delete the last digit of a value (they see the default
// reappear). Instead we carry ``NaN`` as "user is mid-edit / field is empty",
// render it as "" in the input, and on blur snap to the fallback if still
// invalid. ``handleVideoGenerate`` + ``clampNumFrames`` defend against any
// ``NaN`` that slips through to the payload.
function onNumericChange(
  event: React.ChangeEvent<HTMLInputElement>,
  setter: (value: number) => void,
): void {
  const raw = event.target.value;
  if (raw === "") {
    setter(Number.NaN);
    return;
  }
  const parsed = Number(raw);
  if (Number.isFinite(parsed)) setter(parsed);
}

function onNumericBlur(
  current: number,
  setter: (value: number) => void,
  fallback: number,
  minimum: number = 1,
): void {
  if (!Number.isFinite(current) || current < minimum) setter(fallback);
}

function displayNumber(value: number): number | string {
  return Number.isFinite(value) ? value : "";
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
  const generateButtonLabel =
    videoBusy && videoBusyLabel?.startsWith("Generating")
      ? videoBusyLabel
      : "Generate video";
  // We compute the disable *reason* (not just the boolean) so the user can see
  // inline why a previous failure might have left the button in a stuck state —
  // the hover-only tooltip wasn't enough ("generate stays disabled after a Wan
  // crash" bug report, April 2026). ``null`` means enabled.
  const generateDisabledReason: string | null = !selectedVideoVariant
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
              : null;
  const generateTitle = generateDisabledReason ?? "Start generating this clip.";
  const generationDisabled = generateDisabledReason !== null;

  // Safety estimate for the chosen width × height × frames against the active
  // device. We surface this *before* the user hits Generate because on Apple
  // Silicon the failure mode is a hard sidecar crash (MPS assertion → Tauri
  // restart loop), not a graceful error — by the time the user sees "Load
  // failed" in the runtime status, the process has already died. See
  // ``assessVideoGenerationSafety`` for the heuristic and the bug it traces.
  //
  // We pass the selected variant's ``sizeGb`` through as ``baseModelFootprintGb``
  // so the estimate accounts for the dominant cost on MPS — weights + text
  // encoder + VAE sitting in unified memory — rather than estimating only
  // the attention kernel peak. Wan 2.1 T2V 1.3B is the key case: its 16 GB
  // on-disk footprint inflates to ~23 GB resident, which is the actual
  // reason it detonates 64 GB Macs at modest frame counts.
  const generationSafety = useMemo(
    () =>
      assessVideoGenerationSafety({
        width: videoWidth,
        height: videoHeight,
        numFrames: videoNumFrames,
        device: videoRuntimeStatus.device,
        deviceMemoryGb: videoRuntimeStatus.deviceMemoryGb,
        baseModelFootprintGb: selectedVideoVariant?.sizeGb,
      }),
    [
      videoWidth,
      videoHeight,
      videoNumFrames,
      videoRuntimeStatus.device,
      videoRuntimeStatus.deviceMemoryGb,
      selectedVideoVariant?.sizeGb,
    ],
  );

  // Format GB with one decimal for small numbers so 2.3 GB / 7.5 GB read
  // clearly, but drop the decimal once we're at 10+ (no user needs "14.0 GB").
  const formatGb = (gb: number): string => (gb >= 10 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`);

  // A concise always-visible capacity label next to the generation knobs so
  // the user can see at a glance how close to their limit they are. We
  // surface it even when ``riskLevel === "safe"`` so it serves as
  // reassurance ("this run wants 3 GB on 32 GB available") rather than only
  // appearing when something is already wrong. When the model-footprint
  // term is known (``modelFootprintGb > 0``), we show a breakdown so the
  // user sees that "the model itself is eating 23 GB" rather than
  // attributing the whole peak to their chosen frame count.
  // Prefer the device the backend reported. When it's missing (probe never
  // came back, "Failed to fetch" sticking) we fall through to the device
  // bucket the safety helper inferred from the host OS — so a Windows
  // RTX 4090 user doesn't see "Apple Silicon" while the backend is
  // unreachable. We tag the inferred case so the user knows it's a guess.
  const inferredDeviceLabel =
    generationSafety.effectiveDevice === "cuda"
      ? "GPU (detected)"
      : generationSafety.effectiveDevice === "cpu"
        ? "CPU (detected)"
        : "Apple Silicon (detected)";
  const deviceLabel = videoRuntimeStatus.device
    ? videoRuntimeStatus.device.toUpperCase().startsWith("CUDA")
      ? "GPU"
      : videoRuntimeStatus.device.toUpperCase() === "MPS"
        ? "Apple Silicon"
        : videoRuntimeStatus.device.toUpperCase()
    : inferredDeviceLabel;
  // Mark the memory figure as a fallback when the backend didn't actually
  // report it — e.g. a stale sidecar that pre-dates the deviceMemoryGb
  // field (we shipped it mid-release cycle) or a platform where detection
  // failed. Without this tag a user on a 64 GB M4 Max sees "16 GB total"
  // and has no way to know the number is inferred, not measured. The "~"
  // prefix + "(default)" suffix reads as "we're guessing" without scaring
  // the user about a real hardware issue.
  const backendReportedMemory =
    videoRuntimeStatus.deviceMemoryGb != null
    && Number.isFinite(videoRuntimeStatus.deviceMemoryGb)
    && videoRuntimeStatus.deviceMemoryGb > 0;
  const memoryLabel = backendReportedMemory
    ? formatGb(generationSafety.deviceMemoryGb)
    : `~${formatGb(generationSafety.deviceMemoryGb)} (default — restart backend for real detection)`;
  const capacityLine =
    generationSafety.modelFootprintGb > 0
      ? `${deviceLabel} · ${memoryLabel} total · model ≈ ${formatGb(generationSafety.modelFootprintGb)}, this run peak ≈ ${formatGb(generationSafety.estimatedPeakGb)}`
      : `${deviceLabel} · ${memoryLabel} total · this run peak ≈ ${formatGb(generationSafety.estimatedPeakGb)}`;

  function handleApplySafeSettings(): void {
    const suggestion = generationSafety.suggestion;
    if (!suggestion) return;
    onVideoWidthChange(suggestion.width);
    onVideoHeightChange(suggestion.height);
    onVideoNumFramesChange(suggestion.numFrames);
  }

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
                value={displayNumber(videoWidth)}
                onChange={(event) => onNumericChange(event, onVideoWidthChange)}
                onBlur={() => onNumericBlur(videoWidth, onVideoWidthChange, 832, 256)}
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
                value={displayNumber(videoHeight)}
                onChange={(event) => onNumericChange(event, onVideoHeightChange)}
                onBlur={() => onNumericBlur(videoHeight, onVideoHeightChange, 480, 256)}
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
                value={displayNumber(videoNumFrames)}
                onChange={(event) => onNumericChange(event, onVideoNumFramesChange)}
                onBlur={() => onNumericBlur(videoNumFrames, onVideoNumFramesChange, 33)}
              />
            </label>
            <label>
              FPS
              <input
                className="text-input"
                type="number"
                min={1}
                max={60}
                value={displayNumber(videoFps)}
                onChange={(event) => onNumericChange(event, onVideoFpsChange)}
                onBlur={() => onNumericBlur(videoFps, onVideoFpsChange, 24)}
              />
            </label>
            <label>
              Steps
              <input
                className="text-input"
                type="number"
                min={1}
                max={100}
                value={displayNumber(videoSteps)}
                onChange={(event) => onNumericChange(event, onVideoStepsChange)}
                onBlur={() => onNumericBlur(videoSteps, onVideoStepsChange, 30)}
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
                value={displayNumber(videoGuidance)}
                onChange={(event) => onNumericChange(event, onVideoGuidanceChange)}
                onBlur={() => onNumericBlur(videoGuidance, onVideoGuidanceChange, 5)}
              />
            </label>
          </div>

          {/*
            Always-on "device capacity" line so the user sees their envelope
            alongside the controls, not only when something's already gone
            wrong. Pairs with the safety callout below when risk rises.
          */}
          <p className="muted-text" aria-live="polite">
            {capacityLine}
          </p>

          {/*
            Pre-flight safety callout. Surfaces the memory-budget heuristic
            before the user hits Generate so they can recover by clicking
            "Use safer settings" rather than triggering a sidecar crash +
            restart loop. Scaled by ``deviceMemoryGb`` so a 64 GB Mac doesn't
            see the same warnings as a 16 GB one, and scaled by the
            selected model's ``sizeGb`` so the estimate reflects the real
            memory pressure (weights + text encoder, not just attention).
            See ``assessVideoGenerationSafety`` in ``src/utils/videos.ts``
            for the heuristic and the bug it traces ("Wan 2.1 T2V 1.3B at
            832×480 × 40 frames" detonation on 64 GB M4 Max, Apr 2026).

            The "Use safer settings" button only shows when a per-request
            tweak can actually recover. When the model itself is too big
            for the device, the heuristic returns ``suggestion: null`` and
            the callout explains that a smaller model is required —
            clicking through to "480×320 × 17 frames" would just produce a
            second crash, which is strictly worse than no button.
          */}
          {generationSafety.riskLevel !== "safe" ? (
            <div
              className={`callout image-callout ${
                generationSafety.riskLevel === "danger" ? "error" : "warning"
              }`}
              role="alert"
            >
              <p>
                <strong>
                  {generationSafety.riskLevel === "danger"
                    ? "Likely to crash the backend"
                    : "Heads up — may struggle on this device"}
                  :
                </strong>{" "}
                {generationSafety.reason}
              </p>
              {generationSafety.suggestion ? (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleApplySafeSettings}
                    disabled={videoBusy}
                    title={`Apply ${generationSafety.suggestion.label}`}
                  >
                    Use safer settings ({generationSafety.suggestion.label})
                  </button>
                </div>
              ) : (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => onActiveTabChange("video-discover")}
                    disabled={videoBusy}
                  >
                    Browse smaller models
                  </button>
                </div>
              )}
            </div>
          ) : null}

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

          {/*
            Make the disable reason visible even when the user isn't hovering
            the button. A failure-recovery flow that left the button stuck
            (real bug, April 2026) was only diagnosable via the tooltip, which
            is easy to miss — this turns the same string into an always-on
            callout so the root cause is obvious at a glance.
          */}
          {generateDisabledReason && !videoBusy ? (
            <p className="muted-text">Generate disabled: {generateDisabledReason}</p>
          ) : null}

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
