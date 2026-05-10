/**
 * Top callout in VideoStudio — runtime status, chip row, and the
 * stack of contextual install actions (CUDA torch, LongLive, mlx-
 * video, mp4 encoder, missing tokenizer deps, GPU bundle bundle).
 *
 * Pulled out of ``VideoStudioTab.tsx`` so the tab keeps composition
 * + form rendering and the dense status/install UI lives in one
 * focused component.
 *
 * Extracted as part of the v0.8.0 Phase 2d-2a refactor.
 */

import { CudaTorchLogPanel } from "../../components/CudaTorchLogPanel";
import { InstallLogPanel } from "../../components/InstallLogPanel";
import { WanRuntimeInstaller } from "../../components/WanRuntimeInstaller";
import type {
  CudaTorchInstallResult,
  GpuBundleJobState,
  LongLiveJobState,
} from "../../api";
import type { VideoModelVariant, VideoRuntimeStatus } from "../../types";


export interface VideoStudioRuntimeBannerProps {
  videoRuntimeStatus: VideoRuntimeStatus;
  loadedVideoVariant: VideoModelVariant | null;
  busy: boolean;
  busyAction: string | null;
  backendOnline: boolean;
  onRestartServer: () => void;
  onInstallCudaTorch?: () => void;
  installingCudaTorch?: boolean;
  cudaTorchResult?: CudaTorchInstallResult | null;
  // Computed flags (derived in the parent — passed in to avoid
  // duplicating logic).
  gpuBundleRestartRequired: boolean;
  isMlxVideoVariant: boolean;
  isAppleSiliconHost: boolean;
  isLongLiveVariant: boolean;
  isWanRepo: boolean;
  selectedRepo: string;
  mp4EncoderMissing: boolean;
  mlxVideoMissing: boolean;
  mlxVideoInstalledScaffold: boolean;
  missingTokenizerDeps: string[];
  otherMissingDependencies: string[];
  // LongLive
  longLiveStatus: VideoRuntimeStatus | null;
  longLiveJob: LongLiveJobState | null;
  installingLongLive: boolean;
  onInstallLongLive: () => void;
  // mlx-video
  mlxVideoStatus: VideoRuntimeStatus | null;
  installingMlxVideo: boolean;
  onInstallMlxVideo: () => void;
  // mp4 / tokenizer / GPU bundle install actions
  installingOutputDeps: boolean;
  installingGpuRuntime: boolean;
  gpuBundleJob: GpuBundleJobState | null;
  onInstallOutputDeps: () => void;
  onInstallTokenizerDeps: () => void;
  onInstallGpuRuntime: () => void;
}


export function VideoStudioRuntimeBanner(props: VideoStudioRuntimeBannerProps) {
  const {
    videoRuntimeStatus,
    loadedVideoVariant,
    busy,
    busyAction,
    backendOnline,
    onRestartServer,
    onInstallCudaTorch,
    installingCudaTorch,
    cudaTorchResult,
    gpuBundleRestartRequired,
    isMlxVideoVariant,
    isAppleSiliconHost,
    isLongLiveVariant,
    isWanRepo,
    selectedRepo,
    mp4EncoderMissing,
    mlxVideoMissing,
    mlxVideoInstalledScaffold,
    missingTokenizerDeps,
    otherMissingDependencies,
    longLiveStatus,
    longLiveJob,
    installingLongLive,
    onInstallLongLive,
    mlxVideoStatus,
    installingMlxVideo,
    onInstallMlxVideo,
    installingOutputDeps,
    installingGpuRuntime,
    gpuBundleJob,
    onInstallOutputDeps,
    onInstallTokenizerDeps,
    onInstallGpuRuntime,
  } = props;

  return (
    <div className="callout image-callout image-runtime-callout compact">
      {/* torchInstallWarning is the loudest signal — when the installed
        * torch wheel doesn't match the host accelerator (e.g. +cpu wheel
        * on a CUDA box) generation silently runs on CPU at a fraction of
        * speed, while every other badge below would otherwise read green
        * ("Real engine ready" / "Device: cuda (expected)"). Render it as
        * the first visible element so users notice before queueing a
        * 5-minute "GPU" run that's actually CPU. */}
      {/* Mirror of the Image Studio callout: same three-state
        * single-banner pattern (post-install restart prompt /
        * GPU acceleration warning / nothing). Keeps the panel
        * uncluttered by never stacking two banners. */}
      {cudaTorchResult?.ok && cudaTorchResult.requiresRestart ? (
        <div className="callout" style={{ marginBottom: "0.6rem" }}>
          <strong>CUDA torch installed.</strong>{" "}
          The running backend still has the old torch in its module cache.
          Restart the backend to activate the new wheel
          {cudaTorchResult.indexUrl
            ? ` (${cudaTorchResult.indexUrl.replace("https://download.pytorch.org/whl/", "")})`
            : ""}
          .
          <div style={{ marginTop: "0.5rem" }}>
            <button
              className="primary-button"
              type="button"
              onClick={() => onRestartServer()}
              disabled={busy}
            >
              {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
            </button>
          </div>
          <CudaTorchLogPanel result={cudaTorchResult ?? null} />
        </div>
      ) : videoRuntimeStatus.torchInstallWarning ? (
        <div className="callout error" style={{ marginBottom: "0.6rem" }}>
          <strong>GPU acceleration not active.</strong>{" "}
          {videoRuntimeStatus.torchInstallWarning}
          {onInstallCudaTorch
            && videoRuntimeStatus.torchInstallWarning.includes("Install CUDA torch") ? (
            <div style={{ marginTop: "0.5rem" }}>
              <button
                className="primary-button"
                type="button"
                onClick={() => onInstallCudaTorch()}
                disabled={Boolean(installingCudaTorch) || !backendOnline}
              >
                {installingCudaTorch ? "Installing CUDA torch..." : "Install CUDA torch"}
              </button>
              <CudaTorchLogPanel result={cudaTorchResult ?? null} />
            </div>
          ) : null}
        </div>
      ) : null}
      <p>{videoRuntimeStatus.message}</p>
      <div className="chip-row">
        <span className={`badge ${videoRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
          {videoRuntimeStatus.realGenerationAvailable ? "Real engine ready" : "Fallback active"}
        </span>
        {videoRuntimeStatus.torchInstallWarning ? (
          <span className="badge danger" title={videoRuntimeStatus.torchInstallWarning}>
            CPU fallback
          </span>
        ) : null}
        {gpuBundleRestartRequired && !videoRuntimeStatus.realGenerationAvailable ? (
          <span className="badge warning">Restart required</span>
        ) : null}
        {/* The "Engine: …" muted chip is suppressed when a more
          * specific engine badge (mlx-video accent / LongLive
          * status) already renders below — they would otherwise
          * report the same activeEngine string twice. We still
          * surface it for diffusers/torch and for fallback states
          * since nothing else announces the engine in those cases. */}
        {isMlxVideoVariant
          && isAppleSiliconHost
          && mlxVideoStatus?.realGenerationAvailable ? null : (
          <span className="badge muted">Engine: {videoRuntimeStatus.activeEngine}</span>
        )}
        {/* Prefer the actual-loaded device; fall back to the predicted
          * expectedDevice computed via nvidia-smi + find_spec (no torch
          * import). With nothing loaded yet, this reads "Device: cuda
          * (expected)" so users can confirm GPU will be used before
          * generate. Mirrors the image studio chip. */}
        {(() => {
          const resolved =
            videoRuntimeStatus.device
            ?? (videoRuntimeStatus.expectedDevice
              ? `${videoRuntimeStatus.expectedDevice} (expected)`
              : null);
          return resolved ? <span className="badge muted">Device: {resolved}</span> : null;
        })()}
        {loadedVideoVariant ? (
          <span className="badge accent">Loaded: {loadedVideoVariant.name}</span>
        ) : null}
        {mp4EncoderMissing ? (
          <span className="badge warning">mp4 encoder missing</span>
        ) : null}
        {missingTokenizerDeps.map((dependency) => (
          <span key={dependency} className="badge warning">{dependency} missing</span>
        ))}
        {otherMissingDependencies.slice(0, 4).map((dependency) => (
          <span key={dependency} className="badge subtle">{dependency}</span>
        ))}
        {isLongLiveVariant && longLiveStatus ? (
          <span
            className={`badge ${
              longLiveStatus.realGenerationAvailable ? "success" : "warning"
            }`}
          >
            {longLiveStatus.realGenerationAvailable
              ? "LongLive ready"
              : "LongLive not installed"}
          </span>
        ) : null}
        {/* mlx-video chip — Apple Silicon only. Four states:
          * missing (warning), scaffold-installed (subtle), ready
          * (success), or active=true when an LTX-2 variant is
          * loaded and routing through mlx-video. Hidden off-platform. */}
        {mlxVideoMissing ? (
          <span className="badge warning">mlx-video not installed</span>
        ) : null}
        {mlxVideoInstalledScaffold ? (
          <span className="badge subtle">mlx-video scaffold</span>
        ) : null}
        {isAppleSiliconHost
          && mlxVideoStatus?.realGenerationAvailable
          && !isMlxVideoVariant ? (
          <span className="badge success">mlx-video ready</span>
        ) : null}
        {isAppleSiliconHost
          && mlxVideoStatus?.realGenerationAvailable
          && isMlxVideoVariant ? (
          <span className="badge accent">Engine: mlx-video</span>
        ) : null}
      </div>
      {isLongLiveVariant && longLiveStatus && !longLiveStatus.realGenerationAvailable ? (
        <div className="image-runtime-actions">
          <p className="muted-text">
            {longLiveStatus.message} LongLive runs in an isolated venv at
            {" "}<code>~/.chaosengine/longlive</code> so its CUDA-specific deps don't
            clash with the main runtime. Install can take 10–20 minutes — pip
            deps, optional flash-attn build, then ~8 GB of HF weights.
          </p>
          <button
            className="primary-button"
            type="button"
            onClick={() => onInstallLongLive()}
            disabled={installingLongLive || !backendOnline}
          >
            {installingLongLive ? "Installing LongLive..." : "Install LongLive"}
          </button>
          <InstallLogPanel job={longLiveJob} variant="longlive" />
        </div>
      ) : null}
      {/* mlx-video install — Apple Silicon only, surfaces when the
        * probe reports the package missing. Once installed the chip
        * flips to the scaffold state and the button hides; the
        * generate path itself lands with FU-009. */}
      {mlxVideoMissing ? (
        <div className="image-runtime-actions">
          <p className="muted-text">
            {mlxVideoStatus?.message ?? "mlx-video not installed."} Adds
            native MLX video generation for Wan2.1 / Wan2.2 / LTX-2 on
            Apple Silicon — faster than diffusers+MPS once the
            generation path lands.
          </p>
          <button
            className="primary-button"
            type="button"
            onClick={() => onInstallMlxVideo()}
            disabled={installingMlxVideo || !backendOnline}
          >
            {installingMlxVideo ? "Installing mlx-video..." : "Install mlx-video"}
          </button>
        </div>
      ) : null}
      {/* FU-025 part 9 (restored UX): surface the Wan MLX runtime
        * convert action when the user picks a Wan-AI variant on
        * Apple Silicon. Shows a "Ready" chip if the converted MLX
        * dir is already on disk, an "Install" button otherwise.
        * Self-contained component — owns its own polling. */}
      {isWanRepo && isAppleSiliconHost && !mlxVideoMissing ? (
        <WanRuntimeInstaller repo={selectedRepo} />
      ) : null}
      {mp4EncoderMissing ? (
        <div className="image-runtime-actions">
          <p className="muted-text">
            Video generation needs imageio + imageio-ffmpeg to write mp4 files. Install them
            into the backend environment now?
          </p>
          <button
            className="primary-button"
            type="button"
            onClick={() => onInstallOutputDeps()}
            disabled={installingOutputDeps || !backendOnline}
          >
            {installingOutputDeps ? "Installing..." : "Install mp4 encoder"}
          </button>
        </div>
      ) : null}
      {missingTokenizerDeps.length > 0 ? (
        <div className="image-runtime-actions">
          <p className="muted-text">
            Some video models load tokenizer / text-encoder packages on demand. The
            following are missing and would block generation: <strong>{missingTokenizerDeps.join(", ")}</strong>.
            Install them now to avoid a mid-generate error.
          </p>
          <button
            className="primary-button"
            type="button"
            onClick={() => onInstallTokenizerDeps()}
            disabled={installingOutputDeps || !backendOnline}
          >
            {installingOutputDeps
              ? "Installing..."
              : `Install tokenizers (${missingTokenizerDeps.length})`}
          </button>
        </div>
      ) : null}
      {gpuBundleRestartRequired && gpuBundleJob ? (
        <>
          <div className="image-runtime-actions">
            <p className="muted-text">
              GPU runtime installed to{" "}
              <code>{gpuBundleJob.targetDir ?? "extras"}</code>. The running backend
              still has its old import cache — click Restart Backend to activate the
              new runtime, then video generation will use it.
            </p>
            <div className="button-row">
              <button
                className="primary-button"
                type="button"
                onClick={() => onRestartServer()}
                disabled={busy}
              >
                {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend to activate"}
              </button>
            </div>
          </div>
          <InstallLogPanel job={gpuBundleJob} />
        </>
      ) : !videoRuntimeStatus.realGenerationAvailable ? (
        <>
          <div className="image-runtime-actions">
            <p className="muted-text">
              Video generation needs the GPU runtime bundle (torch + diffusers + tokenizers,
              ~2.5 GB). Install it once — it writes to a persistent user-local directory so
              subsequent app updates don't re-download it.
            </p>
            <div className="button-row">
              <button
                className="primary-button"
                type="button"
                onClick={() => onInstallGpuRuntime()}
                disabled={installingGpuRuntime || !backendOnline}
              >
                {installingGpuRuntime ? "Installing GPU runtime..." : "Install GPU runtime"}
              </button>
              <button className="secondary-button" type="button" onClick={() => onRestartServer()} disabled={busy}>
                {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
              </button>
            </div>
          </div>
          <InstallLogPanel job={gpuBundleJob} />
        </>
      ) : null}
    </div>
  );
}
