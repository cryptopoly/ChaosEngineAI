import { useEffect, useMemo, useState } from "react";
import { Panel } from "../../components/Panel";
import { InfoTooltip } from "../../components/InfoTooltip";
import { InstallLogPanel } from "../../components/InstallLogPanel";
import type { DownloadStatus, GpuBundleJobState, InstallResult, LongLiveJobState } from "../../api";
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
  videoPrimarySizeLabel,
  videoSecondarySizeLabel,
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
  videoUseNf4: boolean;
  onVideoUseNf4Change: (value: boolean) => void;
  videoEnableLtxRefiner: boolean;
  onVideoEnableLtxRefinerChange: (value: boolean) => void;
  onActiveTabChange: (tab: TabId) => void;
  onPreloadVideoModel: (variant: VideoModelVariant) => void;
  onUnloadVideoModel: (variant?: VideoModelVariant) => void;
  onVideoDownload: (repo: string) => void;
  onGenerateVideo: () => void;
  onOpenExternalUrl: (url: string) => void;
  onRestartServer: () => void;
  onInstallVideoOutputDeps: (packages?: readonly string[]) => Promise<InstallResult>;
  onInstallVideoGpuRuntime: () => Promise<InstallResult>;
  // LongLive (long-form causal video) surface — separate from the main
  // diffusers runtime because LongLive runs via a torchrun subprocess
  // against an isolated venv at ~/.chaosengine/longlive. Null until the
  // user selects a LongLive variant and the status is probed.
  longLiveStatus: VideoRuntimeStatus | null;
  installingLongLive: boolean;
  onRefreshLongLiveStatus: () => void;
  onInstallLongLive: () => Promise<InstallResult>;
  // Live state of the LongLive install job — drives the InstallLogPanel
  // beside the "Install LongLive" button so the user sees per-phase
  // progress (~9 phases over 10-20 minutes) rather than a static spinner.
  longLiveJob: LongLiveJobState | null;
  // mlx-video (Blaizzy) Apple Silicon engine probe (FU-009). Same
  // separation as LongLive — mlx-video runs as an MLX-native subprocess
  // (Wan2.1/2.2/LTX-2) rather than diffusers, so it gets a dedicated
  // chip + install action. Probe returns device="mps" on Apple Silicon
  // and device=null off-platform; chip hides off-platform.
  mlxVideoStatus: VideoRuntimeStatus | null;
  installingMlxVideo: boolean;
  onRefreshMlxVideoStatus: () => void;
  onInstallMlxVideo: () => Promise<InstallResult>;
  // Live state of the GPU bundle install job — drives the InstallLogPanel
  // under the install button so users see per-step pip output.
  gpuBundleJob: GpuBundleJobState | null;
}

// Pipeline-specific tokenizer / text-encoder packages that diffusers loads
// lazily — surfaced by the runtime probe via missingDependencies. Mirrors
// _VIDEO_MODEL_DEPS in backend_service/video_runtime.py so the Studio knows
// which "missing dep" chips it can offer a one-click install for.
const KNOWN_INSTALLABLE_VIDEO_DEPS: ReadonlySet<string> = new Set([
  "imageio",
  "imageio-ffmpeg",
  "tiktoken",
  "sentencepiece",
  "protobuf",
  "ftfy",
]);

// Repos the mlx-video Apple Silicon engine supports natively. Mirrors
// _SUPPORTED_REPOS in backend_service/mlx_video_runtime.py — kept here
// so the Studio can decide when to surface the mlx-video chip without
// an extra capabilities round-trip. See FU-009 in CLAUDE.md.
//
// Today: LTX-2 prince-canuma pre-converted MLX repos only. Wan2.1/2.2
// require an explicit ``mlx_video.models.wan_2.convert`` step on raw HF
// weights (no pre-converted MLX repo today) — until that conversion is
// bundled, Wan paths use diffusers MPS.
const MLX_VIDEO_SUPPORTED_REPOS: ReadonlySet<string> = new Set([
  "prince-canuma/LTX-2-distilled",
  "prince-canuma/LTX-2-dev",
  "prince-canuma/LTX-2.3-distilled",
  "prince-canuma/LTX-2.3-dev",
]);

// Quality presets: common starting points for the denoising step count.
// Frames are deliberately not part of the preset — frame count controls
// clip LENGTH, not image quality, and bundling it into "Draft/High/Max"
// confused users into thinking shorter clips were lower quality. Guidance
// is also omitted because the parent hook sets it per-model (LTX wants 3,
// Hunyuan wants 6, others 5) and presets shouldn't overwrite that.
type VideoQualityPreset = "draft" | "standard" | "high" | "max";
const QUALITY_PRESETS: Record<
  VideoQualityPreset,
  { label: string; sub: string; steps: number }
> = {
  draft: { label: "Draft", sub: "20 steps", steps: 20 },
  standard: { label: "Standard", sub: "30 steps", steps: 30 },
  high: { label: "High", sub: "40 steps", steps: 40 },
  max: { label: "Max", sub: "50 steps", steps: 50 },
};

// Aspect-ratio presets. Concrete resolutions rather than "apply ratio to
// current base" so clicking a pill has zero surprises. Values chosen to
// be safe across LTX / Wan / HunyuanVideo — they're all divisible by 8
// (diffusers requirement) and under the largest-tested resolutions the
// families ship with.
type VideoAspectRatio = "1:1" | "4:3" | "16:9" | "9:16" | "21:9";
const ASPECT_RATIOS: Record<
  VideoAspectRatio,
  { width: number; height: number }
> = {
  "1:1": { width: 512, height: 512 },
  "4:3": { width: 640, height: 480 },
  "16:9": { width: 768, height: 432 },
  "9:16": { width: 432, height: 768 },
  "21:9": { width: 1024, height: 440 },
};

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
  videoUseNf4,
  onVideoUseNf4Change,
  videoEnableLtxRefiner,
  onVideoEnableLtxRefinerChange,
  onActiveTabChange,
  onPreloadVideoModel,
  onUnloadVideoModel,
  onVideoDownload,
  onGenerateVideo,
  onOpenExternalUrl,
  onRestartServer,
  onInstallVideoOutputDeps,
  onInstallVideoGpuRuntime,
  longLiveStatus,
  installingLongLive,
  onRefreshLongLiveStatus,
  onInstallLongLive,
  longLiveJob,
  mlxVideoStatus,
  installingMlxVideo,
  onRefreshMlxVideoStatus,
  onInstallMlxVideo,
  gpuBundleJob,
}: VideoStudioTabProps) {
  const [installingOutputDeps, setInstallingOutputDeps] = useState(false);
  const [installingGpuRuntime, setInstallingGpuRuntime] = useState(false);
  // Opt-in acknowledgement that unlocks Generate when the safety heuristic
  // says "danger". We keep this behind an explicit checkbox because on
  // Apple Silicon an MPS kernel-panic can hard-reset the whole machine
  // (not just kill the sidecar) — one confirmed crash report from a 64 GB
  // M4 Max running Wan 2.2 A14B. The checkbox resets whenever the chosen
  // model, resolution, or frame count change so it's a per-configuration
  // override, not a permanent bypass.
  const [dangerOverrideAck, setDangerOverrideAck] = useState(false);
  const missingDependencies = videoRuntimeStatus.missingDependencies ?? [];
  // imageio + imageio-ffmpeg are the two pip packages diffusers video
  // pipelines need to export mp4s. Everything else we surface as a badge;
  // these two get a dedicated install button because they're the thing that
  // actually blocks "Generate video" from producing an output for ANY model.
  const mp4EncoderMissing = missingDependencies.some(
    (dep) => dep === "imageio" || dep === "imageio-ffmpeg",
  );
  // Tokenizer / text-encoder packages individual pipelines need lazily —
  // tiktoken for LTX-Video, sentencepiece for Wan / HunyuanVideo / CogVideoX
  // / Mochi, plus the protobuf + ftfy support libs. We list them out as a
  // single "Install missing video dependencies" button so the user doesn't
  // get a "tiktoken is required" mid-generate error after waiting on a long
  // model preload.
  const missingTokenizerDeps = missingDependencies.filter(
    (dep) => KNOWN_INSTALLABLE_VIDEO_DEPS.has(dep) && dep !== "imageio" && dep !== "imageio-ffmpeg",
  );
  const otherMissingDependencies = missingDependencies.filter(
    (dep) => !KNOWN_INSTALLABLE_VIDEO_DEPS.has(dep),
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

  async function handleInstallTokenizerDeps() {
    if (installingOutputDeps) return;
    if (missingTokenizerDeps.length === 0) return;
    setInstallingOutputDeps(true);
    try {
      await onInstallVideoOutputDeps(missingTokenizerDeps);
    } finally {
      setInstallingOutputDeps(false);
    }
  }

  // One-click install for the full GPU runtime (torch + diffusers + video
  // deps). Triggered when the probe reports the engine as unavailable —
  // i.e. torch or diffusers is missing from the persistent extras dir.
  async function handleInstallGpuRuntime() {
    if (installingGpuRuntime) return;
    setInstallingGpuRuntime(true);
    try {
      await onInstallVideoGpuRuntime();
    } finally {
      setInstallingGpuRuntime(false);
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

  // Ensure a valid model is selected once the catalog loads. "Valid" means
  // the model is present in ``studioFamilies`` — the installed / in-flight
  // subset the dropdown actually renders options for. Merely being in the
  // full ``videoCatalog`` isn't enough: a ``<select>`` whose ``value``
  // doesn't match any ``<option>`` silently shows the first option
  // visually while React state stays stale, which produces the classic
  // "dropdown says Wan 2.2 but every derived field still says LTX"
  // inconsistency after the previously-selected model is deleted.
  useEffect(() => {
    if (selectedVideoModelId) {
      const reachableFromDropdown = studioFamilies.some((family) =>
        family.variants.some((variant) => variant.id === selectedVideoModelId),
      );
      if (reachableFromDropdown) return;
    }
    const installed = studioFamilies[0]?.variants[0];
    if (installed?.id) {
      onSelectedVideoModelIdChange(installed.id);
      return;
    }
    const fallback = defaultVideoVariantForFamily(videoCatalog[0]);
    if (fallback?.id) onSelectedVideoModelIdChange(fallback.id);
  }, [selectedVideoModelId, videoCatalog, studioFamilies, onSelectedVideoModelIdChange]);

  // Clear the danger-override acknowledgement whenever any input feeding
  // the safety heuristic changes. A user who ticked "generate anyway" for
  // a 720×480 × 33-frame Wan 2.2 run should not have that override still
  // armed when they then bump frames to 161 — the new configuration has
  // its own risk profile and needs its own conscious decision.
  useEffect(() => {
    setDangerOverrideAck(false);
  }, [selectedVideoVariant?.id, videoWidth, videoHeight, videoNumFrames]);

  // Probe LongLive install state whenever the user selects a LongLive
  // variant so the Studio can surface an install callout without the
  // user having to click "generate" to find out the subprocess engine
  // isn't ready yet.
  const isLongLiveVariant =
    selectedVideoVariant?.repo?.startsWith("NVlabs/LongLive") ?? false;
  useEffect(() => {
    if (isLongLiveVariant) onRefreshLongLiveStatus();
  }, [isLongLiveVariant, onRefreshLongLiveStatus]);

  // Same probe-on-select pattern for mlx-video. Backend probe is
  // cheap (find_spec + platform check, no torch import) so refreshing
  // when the user picks a Wan/LTX variant gives the chip up-to-date
  // install state. Off-platform the probe returns ``device=null``
  // so the chip stays hidden — see render gate below.
  const isMlxVideoVariant =
    !!selectedVideoVariant?.repo && MLX_VIDEO_SUPPORTED_REPOS.has(selectedVideoVariant.repo);
  useEffect(() => {
    if (isMlxVideoVariant) onRefreshMlxVideoStatus();
  }, [isMlxVideoVariant, onRefreshMlxVideoStatus]);

  // Apple Silicon detection from the probe result rather than a
  // separate user-agent sniff — backend reports device="mps" or
  // expectedDevice="mps" only on Darwin arm64.
  const isAppleSiliconHost =
    mlxVideoStatus !== null
    && (mlxVideoStatus.device === "mps" || mlxVideoStatus.expectedDevice === "mps");
  const mlxVideoMissing =
    isAppleSiliconHost
    && !mlxVideoStatus.realGenerationAvailable
    && (mlxVideoStatus.missingDependencies ?? []).includes("mlx-video");
  const mlxVideoInstalledScaffold =
    isAppleSiliconHost
    && !mlxVideoStatus.realGenerationAvailable
    && !(mlxVideoStatus.missingDependencies ?? []).includes("mlx-video");

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
  // We defer the danger-safety check until AFTER ``generationSafety`` is
  // computed below — this variable is reassigned a few lines further down
  // to add "danger risk without explicit acknowledgement" to the chain.
  // Keeping the base chain readable here; see ``generateDisabledReason``
  // reassignment after ``generationSafety``.
  let generateDisabledReason: string | null = !selectedVideoVariant
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
        runtimeFootprintGb: selectedVideoVariant?.runtimeFootprintGb,
      }),
    [
      videoWidth,
      videoHeight,
      videoNumFrames,
      videoRuntimeStatus.device,
      videoRuntimeStatus.deviceMemoryGb,
      selectedVideoVariant?.sizeGb,
      selectedVideoVariant?.runtimeFootprintGb,
    ],
  );

  // Danger-level runs are gated behind an explicit acknowledgement because
  // the failure mode on Apple Silicon is a hard MPS kernel panic that can
  // reset the whole machine, not just the sidecar. The base-reason chain
  // above covers "can't generate at all" conditions; this layer covers
  // "could generate but we think it will crash your computer". If the user
  // has ticked the override, we allow the generate — same UX pattern as
  // destructive-operation confirmations elsewhere in the app.
  if (generateDisabledReason === null && generationSafety.riskLevel === "danger" && !dangerOverrideAck) {
    generateDisabledReason =
      "This configuration is likely to crash the backend. Tick \"Generate anyway\" below after reviewing the warning, or lower resolution/frames/model.";
  }
  const generateTitle = generateDisabledReason ?? "Start generating this clip.";
  const generationDisabled = generateDisabledReason !== null;

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
                onClick={() => void onInstallLongLive()}
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
                onClick={() => void onInstallMlxVideo()}
                disabled={installingMlxVideo || !backendOnline}
              >
                {installingMlxVideo ? "Installing mlx-video..." : "Install mlx-video"}
              </button>
            </div>
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
                onClick={() => void handleInstallOutputDeps()}
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
                onClick={() => void handleInstallTokenizerDeps()}
                disabled={installingOutputDeps || !backendOnline}
              >
                {installingOutputDeps
                  ? "Installing..."
                  : `Install ${missingTokenizerDeps.join(" + ")}`}
              </button>
            </div>
          ) : null}
          {!videoRuntimeStatus.realGenerationAvailable ? (
            <>
              <div className="image-runtime-actions">
                {/* Same post-install-awaiting-restart branch Image Studio
                  * uses. After a successful GPU bundle install, the
                  * running backend still can't see the new torch in
                  * extras (PYTHONPATH is snapshotted at spawn). Nudge
                  * the user toward Restart Backend instead of asking
                  * them to install again. */}
                {gpuBundleJob?.phase === "done" && gpuBundleJob.requiresRestart ? (
                  <>
                    <p className="muted-text">
                      GPU runtime installed to{" "}
                      <code>{gpuBundleJob.targetDir ?? "extras"}</code>. The running backend
                      still has its old import cache — click Restart Backend to activate the
                      new runtime, then video generation will use your GPU.
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
                  </>
                ) : (
                  <>
                <p className="muted-text">
                  Video generation needs the GPU runtime bundle (torch + diffusers + tokenizers,
                  ~2.5 GB). Install it once — it writes to a persistent user-local directory so
                  subsequent app updates don't re-download it.
                </p>
                <div className="button-row">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => void handleInstallGpuRuntime()}
                    disabled={installingGpuRuntime || !backendOnline}
                  >
                    {installingGpuRuntime ? "Installing GPU runtime..." : "Install GPU runtime"}
                  </button>
                  <button className="secondary-button" type="button" onClick={() => onRestartServer()} disabled={busy}>
                    {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
                  </button>
                </div>
                  </>
                )}
              </div>
              <InstallLogPanel job={gpuBundleJob} />
            </>
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
              <span>{videoPrimarySizeLabel(selectedVideoVariant)}</span>
              {videoSecondarySizeLabel(selectedVideoVariant) ? (
                <span>{videoSecondarySizeLabel(selectedVideoVariant)}</span>
              ) : null}
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
            <span className="inline-label-text">
              Negative prompt
              <InfoTooltip text="Tells the model what to avoid. A generic prompt is pre-filled and tuned for most video models — clear or edit it if you have a model-specific preference. More specificity usually helps more than it hurts." />
            </span>
            <input
              className="text-input"
              type="text"
              value={videoNegativePrompt}
              onChange={(event) => onVideoNegativePromptChange(event.target.value)}
              placeholder="Optional: things to avoid (low quality, watermark, etc.)"
            />
          </label>

          {/*
            Quality preset pills. Jump straight to Draft/Standard/High/Max
            rather than making users learn what frames/steps mean for each
            model. Guidance stays model-aware (set in the hook) — presets
            intentionally don't overwrite it so LTX-at-3 / Hunyuan-at-6
            survive a preset click. Pill shows "active" when current state
            matches the preset exactly (so a user who tweaks a slider sees
            the active ring drop, confirming they're off-preset).
          */}
          <div className="preset-row">
            <span className="preset-row-label">
              Quality preset
              <InfoTooltip text="Sets the denoising step count. More steps = sharper frames + longer generation time. Frame count (clip length) and guidance stay as set — presets don't touch them." />
            </span>
            {(Object.keys(QUALITY_PRESETS) as VideoQualityPreset[]).map((key) => {
              const preset = QUALITY_PRESETS[key];
              const active = videoSteps === preset.steps;
              return (
                <button
                  key={key}
                  type="button"
                  className={`preset-pill ${active ? "active" : ""}`.trim()}
                  onClick={() => {
                    onVideoStepsChange(preset.steps);
                  }}
                >
                  <span className="preset-pill-label">{preset.label}</span>
                  <span className="preset-pill-sub">{preset.sub}</span>
                </button>
              );
            })}
          </div>

          {/*
            Aspect-ratio preset pills. Fixed resolutions (not "apply ratio
            to current base") so one click is fully deterministic. Values
            are all divisible by 8 and sit inside every supported model's
            tested envelope — safer than letting users pick arbitrary W×H.
          */}
          <div className="preset-row">
            <span className="preset-row-label">
              Aspect ratio
              <InfoTooltip text="Sets Width and Height to a common video shape. All presets are safe on every supported model (≤1024 on the long edge, divisible by 8). Edit Width/Height below for finer control." />
            </span>
            {(Object.keys(ASPECT_RATIOS) as VideoAspectRatio[]).map((key) => {
              const ratio = ASPECT_RATIOS[key];
              const active =
                videoWidth === ratio.width && videoHeight === ratio.height;
              return (
                <button
                  key={key}
                  type="button"
                  className={`preset-pill ${active ? "active" : ""}`.trim()}
                  onClick={() => {
                    onVideoWidthChange(ratio.width);
                    onVideoHeightChange(ratio.height);
                  }}
                >
                  <span className="preset-pill-label">{key}</span>
                  <span className="preset-pill-sub">{ratio.width}×{ratio.height}</span>
                </button>
              );
            })}
          </div>

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
              <span className="inline-label-text">
                Width
                <InfoTooltip text="Horizontal resolution in pixels. Must be divisible by 8. Higher = sharper + slower + more VRAM. Try an Aspect ratio preset above for safe values." />
              </span>
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
              <span className="inline-label-text">
                Height
                <InfoTooltip text="Vertical resolution in pixels. Must be divisible by 8. Higher = sharper + slower + more VRAM. Try an Aspect ratio preset above for safe values." />
              </span>
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
              <span className="inline-label-text">
                Frames
                <InfoTooltip text="How many frames to render. Wan / LTX require (frames-1) to be divisible by 4 — valid values are 1, 5, 9, 13, …, 161. Clip length in seconds = Frames ÷ FPS." />
              </span>
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
              <span className="inline-label-text">
                FPS
                <InfoTooltip text="Frames per second for playback. 24 is cinematic, 30 is smoother. Doesn't affect generation cost — only how fast the clip plays back." />
              </span>
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
              <span className="inline-label-text">
                Steps
                <InfoTooltip text="Denoising steps — how many passes the model makes to clean up noise into an image. More = sharper and more coherent, but linearly slower. 20 is draft quality, 30 is standard, 50+ is high quality with diminishing returns." />
              </span>
              <div className="slider-number-row">
                <input
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={Number.isFinite(videoSteps) ? videoSteps : 30}
                  onChange={(event) => onVideoStepsChange(Number(event.target.value))}
                />
                <input
                  className="text-input"
                  type="number"
                  min={1}
                  max={100}
                  value={displayNumber(videoSteps)}
                  onChange={(event) => onNumericChange(event, onVideoStepsChange)}
                  onBlur={() => onNumericBlur(videoSteps, onVideoStepsChange, 30)}
                />
              </div>
            </label>
            <label>
              <span className="inline-label-text">
                Guidance
                <InfoTooltip text="How strongly the model follows your prompt. Too low = ignores the prompt; too high = rigid or distorted output. Recommended: LTX-Video ≈ 3, Wan ≈ 5, HunyuanVideo ≈ 6. The prompt's 'negative' direction comes from the Negative prompt above." />
              </span>
              <div className="slider-number-row">
                <input
                  type="range"
                  min={1}
                  max={15}
                  step={0.5}
                  value={Number.isFinite(videoGuidance) ? videoGuidance : 5}
                  onChange={(event) => onVideoGuidanceChange(Number(event.target.value))}
                />
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
              </div>
              {selectedVideoVariant?.repo === "Lightricks/LTX-Video" && videoGuidance > 4 ? (
                <p className="caution-text" role="alert">
                  LTX-Video is a flow-matching model — CFG above ~3.5 over-saturates and
                  produces blurred / rainbow output. Lower to 3 for the cleanest results.
                </p>
              ) : null}
            </label>
          </div>

          {Number.isFinite(videoNumFrames) && Number.isFinite(videoFps) && videoFps > 0 ? (
            <p className="muted-text" aria-live="polite">
              Clip length: {(videoNumFrames / videoFps).toFixed(2).replace(/\.?0+$/, "")}s
              {" "}({videoNumFrames} frames ÷ {videoFps} fps)
            </p>
          ) : null}

          {selectedVideoVariant?.repo === "Lightricks/LTX-Video" ? (
            <p className="muted-text">
              Backend auto-tunes LTX decode parameters (frame_rate as model conditioning,
              decode_timestep, decode_noise_scale, guidance_rescale) to the Lightricks
              reference defaults — no extra sliders needed.
            </p>
          ) : null}

          {!isAppleSiliconHost ? (
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={videoUseNf4}
                onChange={(event) => onVideoUseNf4Change(event.target.checked)}
              />
              <span>
                4-bit (NVIDIA NF4) — fits Wan 2.1 14B in &lt;24 GB VRAM via bitsandbytes.
                CUDA only; ignored on CPU.
              </span>
            </label>
          ) : null}

          {selectedVideoVariant?.repo === "Lightricks/LTX-Video" ? (
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={videoEnableLtxRefiner}
                onChange={(event) => onVideoEnableLtxRefinerChange(event.target.checked)}
              />
              <span>
                LTX two-stage spatial upscale — refines through
                LTXLatentUpsamplePipeline. Frame budget +50%.
              </span>
            </label>
          ) : null}

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
              {/*
                Danger-only override. Generate stays disabled until the user
                ticks this box — the checkbox resets on any change to
                variant / width / height / frames so it can't stay armed
                after the configuration shifts (see the dedicated useEffect
                that clears ``dangerOverrideAck``). Users on 128 GB M3 Ultras
                where the heuristic over-warns can still force a run; users
                on 16 GB base Macs get a real speed bump against "just click
                Generate". Warning-level (not danger) still generates freely.
              */}
              {generationSafety.riskLevel === "danger" ? (
                <label
                  className="inline-label"
                  style={{ display: "flex", alignItems: "center", gap: ".4rem", marginTop: ".6rem" }}
                >
                  <input
                    type="checkbox"
                    checked={dangerOverrideAck}
                    onChange={(event) => setDangerOverrideAck(event.target.checked)}
                  />
                  <span>
                    Generate anyway — I accept that the backend may crash and my machine may need to be
                    restarted.
                  </span>
                </label>
              ) : null}
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
