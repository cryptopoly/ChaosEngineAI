import { useDeferredValue, useEffect, useRef, useState } from "react";
import {
  cancelVideoDownload,
  deleteVideoDownload,
  deleteVideoOutput,
  downloadVideoModel,
  generateVideo,
  getGpuBundleStatus,
  getVideoCatalog,
  getVideoDownloadStatus,
  getVideoOutputs,
  getVideoRuntime,
  installPipPackage,
  preloadVideoModel,
  startGpuBundleInstall,
  unloadVideoModel,
} from "../api";
import type { DownloadStatus, GpuBundleJobState, InstallResult } from "../api";

// Duplicate of the same helper in useImageState — both Studio tabs surface
// the same bundle-install progress string through their own busy labels.
// Kept inline here (instead of in ../utils) because the hook files import
// the same api types and the helper is ~20 LOC.
function formatGpuBundleLabel(job: GpuBundleJobState): string {
  const phase = job.phase;
  if (phase === "preflight") return job.message || "Preparing GPU bundle install...";
  if (phase === "downloading") {
    const total = job.packageTotal || 1;
    const pct = Math.max(0, Math.min(100, Math.round(job.percent)));
    const current = job.packageCurrent || job.message || "package";
    return `Installing GPU bundle: ${current} (${job.packageIndex}/${total}, ${pct}%)`;
  }
  if (phase === "verifying") return "Verifying CUDA availability...";
  if (phase === "done") return job.message || "GPU bundle installed.";
  if (phase === "error") return job.error || job.message || "GPU bundle install failed.";
  return job.message || "Working...";
}
import {
  buildDownloadStatusMap,
  defaultVideoVariantForFamily,
  failedDownloadStatus,
  findVideoVariantById,
  findVideoVariantByRepo,
  flattenVideoVariants,
  isTransientNetworkError,
  pendingDownloadStatus,
  videoDiscoverFamilyMatchesQuery,
  videoDiscoverVariantMatchesQuery,
  videoRuntimeErrorStatus,
  videoVariantMatchesDiscoverFilters,
} from "../utils";
import type {
  TabId,
  VideoGenerationPayload,
  VideoModelFamily,
  VideoModelVariant,
  VideoOutputArtifact,
  VideoRuntimeStatus,
} from "../types";
import type { VideoDiscoverTaskFilter } from "../types/video";

const MAX_VIDEO_SEED = 2147483647;

// Default starting point for the Studio sliders. We deliberately choose a
// short clip (~1.4s @ 24fps) and a moderate step count so the *first* generate
// out of the box completes on Apple Silicon unified memory rather than
// detonating Metal with a 70+ GB attention tensor (issue: Wan 2.1 1.3B at
// 832x480 × 96 frames × 50 steps blew up MPS during initial testing).
// Users can dial up via the Studio controls once they know their hardware.
const DEFAULT_VIDEO_NUM_FRAMES = 33;
const DEFAULT_VIDEO_FPS = 24;
const DEFAULT_VIDEO_STEPS = 30;
const DEFAULT_VIDEO_GUIDANCE = 5.0;

// Wan-family pipelines require ``(num_frames - 1) % 4 == 0``. We round to
// the nearest valid value so the user can type any frame count and we still
// hand the backend something it can run.
function clampNumFrames(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_VIDEO_NUM_FRAMES;
  const clamped = Math.max(1, Math.min(257, Math.round(value)));
  // Snap to the nearest n where (n - 1) % 4 == 0 (i.e. 1, 5, 9, 13, ...)
  const remainder = (clamped - 1) % 4;
  if (remainder === 0) return clamped;
  const down = clamped - remainder;
  const up = down + 4;
  return up - clamped < clamped - down ? up : down;
}

/** Parse "832x480" (or similar) into [width, height], falling back to defaults. */
function parseRecommendedResolution(
  value: string | null | undefined,
  defaultWidth: number,
  defaultHeight: number,
): [number, number] {
  if (!value) return [defaultWidth, defaultHeight];
  const match = String(value).trim().match(/^(\d+)\s*[xX\u00d7]\s*(\d+)/);
  if (!match) return [defaultWidth, defaultHeight];
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return [defaultWidth, defaultHeight];
  if (width < 256 || width > 2048 || height < 256 || height > 2048) {
    return [defaultWidth, defaultHeight];
  }
  return [width, height];
}

export function useVideoState(
  backendOnline: boolean,
  setError: (msg: string | null) => void,
  setActiveTab: (tab: TabId) => void,
) {
  const [videoCatalog, setVideoCatalog] = useState<VideoModelFamily[]>([]);
  const [latestVideoDiscoverResults, setLatestVideoDiscoverResults] = useState<VideoModelVariant[]>([]);
  const [videoDiscoverTaskFilter, setVideoDiscoverTaskFilter] = useState<VideoDiscoverTaskFilter>("all");
  const [videoDiscoverSearchInput, setVideoDiscoverSearchInput] = useState("");
  const deferredVideoDiscoverSearch = useDeferredValue(videoDiscoverSearchInput);
  const [selectedVideoModelId, setSelectedVideoModelId] = useState("");
  const [videoPrompt, setVideoPrompt] = useState("");
  const [videoNegativePrompt, setVideoNegativePrompt] = useState("");
  const [videoSeedInput, setVideoSeedInput] = useState("");
  const [videoUseRandomSeed, setVideoUseRandomSeed] = useState(true);
  // Generation knobs the user can tweak in Studio. Defaults are populated
  // from the selected variant's catalog hint when the model changes (see
  // the reset effect below) but stay user-editable thereafter.
  const [videoWidth, setVideoWidth] = useState<number>(832);
  const [videoHeight, setVideoHeight] = useState<number>(480);
  const [videoNumFrames, setVideoNumFrames] = useState<number>(DEFAULT_VIDEO_NUM_FRAMES);
  const [videoFps, setVideoFps] = useState<number>(DEFAULT_VIDEO_FPS);
  const [videoSteps, setVideoSteps] = useState<number>(DEFAULT_VIDEO_STEPS);
  const [videoGuidance, setVideoGuidance] = useState<number>(DEFAULT_VIDEO_GUIDANCE);
  const [videoRuntimeStatus, setVideoRuntimeStatus] = useState<VideoRuntimeStatus>({
    activeEngine: "placeholder",
    realGenerationAvailable: false,
    message: "Video runtime not initialised yet.",
    missingDependencies: [],
  });
  const [videoBusyLabel, setVideoBusyLabel] = useState<string | null>(null);
  const videoBusy = videoBusyLabel !== null;
  const [activeVideoDownloads, setActiveVideoDownloads] = useState<Record<string, DownloadStatus>>({});
  const [videoOutputs, setVideoOutputs] = useState<VideoOutputArtifact[]>([]);
  // Modal state — mirrors useImageState's image-generation modal lifecycle.
  // Opening the modal as soon as ``handleVideoGenerate`` runs lets the user
  // watch the real-time progress bar instead of staring at the studio form
  // wondering whether anything is happening for 60-300s.
  const [showVideoGenerationModal, setShowVideoGenerationModal] = useState(false);
  const [videoGenerationStartedAt, setVideoGenerationStartedAt] = useState<number | null>(null);
  const [videoGenerationError, setVideoGenerationError] = useState<string | null>(null);
  const [videoGenerationArtifact, setVideoGenerationArtifact] = useState<VideoOutputArtifact | null>(null);
  const [videoGenerationRunInfo, setVideoGenerationRunInfo] = useState<{
    modelName: string;
    prompt: string;
    numFrames: number;
    fps: number;
    steps: number;
    needsPipelineLoad: boolean;
  } | null>(null);

  // ── Computed values ─────────────────────────────────────────
  const videoVariants = flattenVideoVariants(videoCatalog);
  const selectedVideoVariant =
    findVideoVariantById(videoCatalog, selectedVideoModelId)
    ?? latestVideoDiscoverResults.find((variant) => variant.id === selectedVideoModelId)
    ?? videoVariants[0]
    ?? null;
  const selectedVideoFamily =
    videoCatalog.find((family) =>
      family.variants.some((variant) => variant.id === selectedVideoVariant?.id),
    ) ?? null;
  const loadedVideoVariant =
    findVideoVariantByRepo(videoCatalog, videoRuntimeStatus.loadedModelRepo)
    ?? (videoRuntimeStatus.loadedModelRepo
      ? latestVideoDiscoverResults.find((variant) => variant.repo === videoRuntimeStatus.loadedModelRepo)
      : null)
    ?? null;
  const selectedVideoLoaded =
    !!selectedVideoVariant
    && !!videoRuntimeStatus.loadedModelRepo
    && videoRuntimeStatus.loadedModelRepo === selectedVideoVariant.repo;
  const selectedVideoWillLoadOnGenerate =
    !!selectedVideoVariant
    && selectedVideoVariant.availableLocally
    && videoRuntimeStatus.realGenerationAvailable
    && !selectedVideoLoaded;
  const videoRuntimeLoadedDifferentModel =
    !!selectedVideoVariant
    && !!loadedVideoVariant
    && loadedVideoVariant.repo !== selectedVideoVariant.repo;

  const installedCatalogVariants = videoVariants.filter(
    (variant) => variant.availableLocally || variant.hasLocalData,
  );
  const installedLatestVariants = latestVideoDiscoverResults.filter(
    (variant) => variant.availableLocally || variant.hasLocalData,
  );
  const seenRepos = new Set(installedCatalogVariants.map((variant) => variant.repo));
  const installedVideoVariants = [
    ...installedCatalogVariants,
    ...installedLatestVariants.filter((variant) => !seenRepos.has(variant.repo)),
  ];

  // Augmented catalog for dropdowns that need to see tracked-but-uncurated entries
  const catalogRepoSet = new Set(videoVariants.map((variant) => variant.repo));
  const latestNotInCatalog = latestVideoDiscoverResults.filter(
    (variant) => !catalogRepoSet.has(variant.repo),
  );
  const videoCatalogWithLatest: VideoModelFamily[] = latestNotInCatalog.length > 0
    ? [
        ...videoCatalog,
        {
          id: "latest-tracked",
          name: "Latest / Tracked",
          provider: "Community",
          headline: "Tracked video models not in the curated catalog",
          summary: "Additional video models tracked by ChaosEngineAI",
          updatedLabel: "Tracked",
          badges: [],
          defaultVariantId: latestNotInCatalog[0]?.id ?? "",
          variants: latestNotInCatalog,
        },
      ]
    : videoCatalog;

  const videoDiscoverSearchQuery = deferredVideoDiscoverSearch.trim().toLowerCase();

  const filteredVideoDiscoverFamilies = videoCatalog
    .map((family) => ({
      ...family,
      variants: family.variants.filter((variant) => {
        if (!videoVariantMatchesDiscoverFilters(variant, videoDiscoverTaskFilter)) return false;
        return (
          videoDiscoverFamilyMatchesQuery(family, videoDiscoverSearchQuery)
          || videoDiscoverVariantMatchesQuery(variant, videoDiscoverSearchQuery)
        );
      }),
    }))
    .filter((family) => family.variants.length > 0);

  const filteredLatestVideoDiscoverResults = latestVideoDiscoverResults.filter(
    (variant) =>
      videoVariantMatchesDiscoverFilters(variant, videoDiscoverTaskFilter)
      && videoDiscoverVariantMatchesQuery(variant, videoDiscoverSearchQuery),
  );

  const combinedVideoDiscoverResults: VideoModelVariant[] = [
    ...filteredVideoDiscoverFamilies.flatMap((family) => {
      const variant = defaultVideoVariantForFamily(family);
      return variant ? [{ ...variant, familyName: variant.familyName ?? family.name }] : [];
    }),
    ...filteredLatestVideoDiscoverResults,
  ];

  const videoDiscoverHasActiveFilters =
    videoDiscoverTaskFilter !== "all" || videoDiscoverSearchQuery.length > 0;

  // ── Selection sync ──────────────────────────────────────────
  useEffect(() => {
    if (!videoCatalog.length) {
      setSelectedVideoModelId("");
      return;
    }
    const variants = flattenVideoVariants(videoCatalog);
    if (variants.some((variant) => variant.id === selectedVideoModelId)) return;
    if (latestVideoDiscoverResults.some((variant) => variant.id === selectedVideoModelId)) return;
    const preferred =
      variants.find((variant) => variant.availableLocally)
      ?? defaultVideoVariantForFamily(videoCatalog[0]);
    setSelectedVideoModelId(preferred?.id ?? "");
  }, [videoCatalog, selectedVideoModelId, latestVideoDiscoverResults]);

  // ── Reset generation knobs when the model changes ───────────
  // We pull the *resolution* hint from the variant catalog (e.g. "832x480")
  // but keep frames / steps at our short, MPS-safe defaults so the first
  // generate succeeds on consumer hardware. The user can dial up via the
  // Studio controls.
  useEffect(() => {
    if (!selectedVideoVariant) return;
    const [w, h] = parseRecommendedResolution(
      selectedVideoVariant.recommendedResolution,
      832,
      480,
    );
    setVideoWidth(w);
    setVideoHeight(h);
    setVideoNumFrames(DEFAULT_VIDEO_NUM_FRAMES);
    setVideoFps(DEFAULT_VIDEO_FPS);
    setVideoSteps(DEFAULT_VIDEO_STEPS);
    setVideoGuidance(DEFAULT_VIDEO_GUIDANCE);
    // Intentionally only depend on the variant ID so we don't clobber the
    // user's edits when unrelated catalog fields refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedVideoVariant?.id]);

  // ── Retry on model change when the runtime is stuck ─────────
  // If the runtime status is ``unavailable`` (usually because a previous
  // fetch rejected with a transport error after a sidecar crash) and the
  // user switches to a different model, re-probe the backend. This gives
  // the user a natural way to say "try again with a smaller model" — they
  // pick LTX in the dropdown and we immediately refetch the runtime state
  // so the Generate button wakes up if the backend is actually fine now.
  // Guarded on ``backendOnline`` so we don't retry while the sidecar is
  // still down (the recovery effect above handles that case).
  useEffect(() => {
    if (!selectedVideoVariant) return;
    if (!backendOnline) return;
    if (videoRuntimeStatus.activeEngine !== "unavailable") return;
    void refreshVideoData();
    // Key on the variant id so we only fire when the user actually changes
    // selection, not on every re-render while the runtime is still unavailable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedVideoVariant?.id]);

  // ── Poll while the runtime is warming up ────────────────────
  // When the sidecar is still importing PyTorch the probe returns either
  // ``activeEngine: "initializing"`` (fast-path from the background warmup
  // worker) or ``"unavailable"`` (when the 30s fetchJson cap fired before
  // warmup finished). In both cases the correct UX is "retry every few
  // seconds and flip to Ready when torch lands in the module cache" — the
  // timeout message promises exactly that, so we need to actually do it.
  // Stops as soon as the engine reports any other state.
  useEffect(() => {
    if (!backendOnline) return;
    const engine = videoRuntimeStatus.activeEngine;
    if (engine !== "initializing" && engine !== "unavailable") return;
    const interval = window.setInterval(() => {
      void refreshVideoData();
    }, 5000);
    return () => window.clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendOnline, videoRuntimeStatus.activeEngine]);

  // ── Download polling ────────────────────────────────────────
  const hasActiveVideoDownloads = Object.values(activeVideoDownloads).some(
    (download) => download.state === "downloading",
  );
  useEffect(() => {
    if (!hasActiveVideoDownloads || !backendOnline) return;
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const statuses = await getVideoDownloadStatus();
          setActiveVideoDownloads(buildDownloadStatusMap(statuses));
          if (statuses.some((status) => status.state === "completed")) {
            void refreshVideoData();
          }
        } catch {
          // keep the last known state until the next poll
        }
      })();
    }, 2000);
    return () => window.clearInterval(interval);
  }, [hasActiveVideoDownloads, backendOnline]);

  // ── Backend recovery ────────────────────────────────────────
  // When the Python sidecar dies (e.g. Wan 2.1 OOMs MPS on a 64 GB Mac →
  // Metal asserts → sidecar process is killed), every in-flight fetch
  // rejects with WebKit's ``TypeError: Load failed`` and the runtime status
  // ends up frozen as ``activeEngine: "unavailable"``. Tauri's managed
  // sidecar restarts shortly after, which flips ``backendOnline`` back to
  // true — but we never re-probe the video runtime, so the Studio stays
  // stuck with "ENGINE: UNAVAILABLE" / "Fallback active" badges and the
  // Generate button disabled even after the user picks a smaller model
  // (e.g. LTX-Video) that would have run fine.
  //
  // Fix: whenever ``backendOnline`` transitions from false to true, fire
  // a one-shot ``refreshVideoData()`` so the runtime status reflects the
  // fresh sidecar's actual capabilities. Using a ref to track the previous
  // value avoids a refresh on first mount (where ``backendOnline`` starts
  // at whatever the backend poll reports — already handled by the initial
  // data-load effect in ``App.tsx``).
  const previousBackendOnlineRef = useRef(backendOnline);
  useEffect(() => {
    const wasOffline = !previousBackendOnlineRef.current;
    previousBackendOnlineRef.current = backendOnline;
    if (backendOnline && wasOffline) {
      void refreshVideoData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendOnline]);

  // ── Data fetching ───────────────────────────────────────────
  async function refreshVideoData() {
    const [catalog, statuses, runtime, outputs] = await Promise.allSettled([
      getVideoCatalog(),
      getVideoDownloadStatus(),
      getVideoRuntime(),
      getVideoOutputs(),
    ]);
    const failures = [catalog, statuses, runtime, outputs].filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );

    if (catalog.status === "fulfilled") {
      setVideoCatalog(catalog.value.families);
      setLatestVideoDiscoverResults(catalog.value.latest ?? []);
    }
    if (statuses.status === "fulfilled") {
      setActiveVideoDownloads(buildDownloadStatusMap(statuses.value));
    }
    if (runtime.status === "fulfilled") {
      setVideoRuntimeStatus(runtime.value);
    } else if (failures.length > 0) {
      setVideoRuntimeStatus(videoRuntimeErrorStatus(failures[0].reason));
    }
    if (outputs.status === "fulfilled") {
      setVideoOutputs(outputs.value);
    }

    if (failures.length > 0) {
      const firstError = failures[0].reason;
      // See useImageState.refreshImageData — swallow transient network
      // errors from background refreshes so a cold-start race doesn't
      // leave a sticky "Failed to fetch" banner on the Chat tab.
      if (!isTransientNetworkError(firstError)) {
        setError(firstError instanceof Error ? firstError.message : "Could not load video runtime data.");
      }
    }
  }

  // ── Download handlers ───────────────────────────────────────
  async function handleVideoDownload(repo: string) {
    try {
      setActiveVideoDownloads((prev) => ({ ...prev, [repo]: pendingDownloadStatus(repo, prev[repo]) }));
      const download = await downloadVideoModel(repo);
      setActiveVideoDownloads((prev) => ({ ...prev, [repo]: download }));
      void refreshVideoData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Video download failed");
      setActiveVideoDownloads((prev) => ({ ...prev, [repo]: failedDownloadStatus(repo, String(err)) }));
    }
  }

  async function handleCancelVideoDownload(repo: string) {
    try {
      const download = await cancelVideoDownload(repo);
      setActiveVideoDownloads((prev) => ({ ...prev, [repo]: download }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not pause video download");
    }
  }

  async function handleDeleteVideoDownload(repo: string) {
    try {
      await deleteVideoDownload(repo);
      const statuses = await getVideoDownloadStatus();
      setActiveVideoDownloads(buildDownloadStatusMap(statuses));
      await refreshVideoData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete video download");
    }
  }

  // ── Runtime handlers ────────────────────────────────────────
  async function handlePreloadVideoModel(variant?: VideoModelVariant | null) {
    if (!variant) {
      setError("Choose an installed video model first.");
      return;
    }
    setVideoBusyLabel(`Loading ${variant.name} into memory...`);
    try {
      const runtime = await preloadVideoModel(variant.id);
      setVideoRuntimeStatus(runtime);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not preload the video model.");
    } finally {
      setVideoBusyLabel(null);
    }
  }

  async function handleUnloadVideoModel(variant?: VideoModelVariant | null) {
    setVideoBusyLabel(
      `Unloading ${variant?.name ?? loadedVideoVariant?.name ?? "video model"} from memory...`,
    );
    try {
      const runtime = await unloadVideoModel(variant?.id);
      setVideoRuntimeStatus(runtime);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not unload the video model.");
    } finally {
      setVideoBusyLabel(null);
    }
  }

  // ── Generation handlers ─────────────────────────────────────
  async function handleVideoGenerate() {
    if (!selectedVideoVariant) {
      setError("Pick a video model before generating.");
      return;
    }
    if (!selectedVideoVariant.availableLocally) {
      setError(`${selectedVideoVariant.name} is not installed locally yet. Download it first.`);
      return;
    }
    if (!videoRuntimeStatus.realGenerationAvailable) {
      setError(videoRuntimeStatus.message || "Video runtime is not ready.");
      return;
    }
    const trimmedPrompt = videoPrompt.trim();
    if (!trimmedPrompt) {
      setError("Write a prompt before generating.");
      return;
    }
    const parsedSeed = videoUseRandomSeed ? null : Number(videoSeedInput);
    if (
      parsedSeed !== null
      && (!Number.isFinite(parsedSeed) || parsedSeed < 0 || parsedSeed > MAX_VIDEO_SEED)
    ) {
      setError("Seed must be a non-negative integer.");
      return;
    }

    // The Studio controls drive every per-run knob. We snap ``numFrames`` to
    // a Wan-compatible value here as a defensive measure — the input field
    // already does this on change but a stale value (e.g. 50 from before the
    // snap landed) would otherwise reach the backend and trip its rounding
    // warning. We also guard against ``NaN`` on every numeric field because
    // the Studio inputs now use ``NaN`` to represent "user is mid-edit / field
    // is empty" — any NaN that reaches here must be replaced with a default
    // so the backend doesn't see ``null`` in the JSON payload.
    const safeNumFrames = clampNumFrames(videoNumFrames);
    const safeSteps = Number.isFinite(videoSteps)
      ? Math.max(1, Math.min(100, Math.round(videoSteps)))
      : DEFAULT_VIDEO_STEPS;
    const safeFps = Number.isFinite(videoFps)
      ? Math.max(1, Math.min(60, Math.round(videoFps)))
      : DEFAULT_VIDEO_FPS;
    const safeGuidance = Number.isFinite(videoGuidance)
      ? Math.max(1, Math.min(20, videoGuidance))
      : DEFAULT_VIDEO_GUIDANCE;
    const safeWidth = Number.isFinite(videoWidth)
      ? Math.max(256, Math.min(2048, Math.round(videoWidth)))
      : 832;
    const safeHeight = Number.isFinite(videoHeight)
      ? Math.max(256, Math.min(2048, Math.round(videoHeight)))
      : 480;

    const payload: VideoGenerationPayload = {
      modelId: selectedVideoVariant.id,
      prompt: trimmedPrompt,
      negativePrompt: videoNegativePrompt.trim() || undefined,
      width: safeWidth,
      height: safeHeight,
      numFrames: safeNumFrames,
      fps: safeFps,
      steps: safeSteps,
      guidance: safeGuidance,
      seed: parsedSeed,
    };

    // The pipeline is "loaded" when the runtime reports the same repo as
    // currently selected. Anything else means we're paying the load cost on
    // this generation, which the modal needs to know to show the right phases.
    const willLoadPipeline =
      videoRuntimeStatus.realGenerationAvailable
      && videoRuntimeStatus.loadedModelRepo !== selectedVideoVariant.repo;

    setShowVideoGenerationModal(true);
    setVideoGenerationStartedAt(Date.now());
    setVideoGenerationError(null);
    setVideoGenerationArtifact(null);
    setVideoGenerationRunInfo({
      modelName: selectedVideoVariant.name,
      prompt: trimmedPrompt,
      numFrames: safeNumFrames,
      fps: safeFps,
      steps: safeSteps,
      needsPipelineLoad: willLoadPipeline,
    });
    setVideoBusyLabel(
      willLoadPipeline
        ? `Loading ${selectedVideoVariant.name} into memory...`
        : `Generating ${safeNumFrames}-frame clip with ${selectedVideoVariant.name}...`,
    );
    setError(null);
    try {
      const response = await generateVideo(payload);
      setVideoOutputs(response.outputs);
      if (response.runtime) setVideoRuntimeStatus(response.runtime);
      setVideoGenerationArtifact(response.artifact);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Video generation failed.";
      const detail = `${message}. Check the Logs tab (filter: video) for backend details.`;
      setError(detail);
      setVideoGenerationError(detail);
      // Resync catalog + runtime in the background. A sidecar crash (e.g. the
      // Wan 2.1 MPS assertion) can leave ``videoRuntimeStatus`` stale, which
      // has been observed to leave the Studio's Generate button in a mystery
      // disabled state after the user dismisses the failure modal. Refreshing
      // restores a known-good view of what the backend actually reports now.
      void refreshVideoData();
    } finally {
      setVideoBusyLabel(null);
      setVideoGenerationStartedAt(null);
    }
  }

  // ── Dependency install ──────────────────────────────────────
  //
  // Diffusers can render frames without imageio, but exporting them as an mp4
  // needs both the `imageio` Python package and its `imageio-ffmpeg` plugin.
  // Individual video pipelines also need pipeline-specific tokenizer packages
  // (tiktoken for LTX-Video, sentencepiece for Wan/HunyuanVideo/CogVideoX).
  // The Video Studio calls this when it detects any are missing so the user
  // can unstick things without dropping to a terminal.
  //
  // ``packages`` defaults to the mp4 pair for backwards compatibility with
  // the original "Install mp4 encoder" button; the Studio passes an explicit
  // list (sourced from the runtime's missingDependencies) for the generic
  // "Install missing video dependencies" path.
  // Full GPU bundle install (torch + diffusers + video deps in one shot).
  // Called from VideoStudioTab's primary "Install GPU support" banner when
  // the probe reports the runtime engine as unavailable (i.e. torch or
  // diffusers missing). Uses the same async backend job as the Image
  // Studio's install — installing once covers both tabs.
  async function handleInstallVideoGpuRuntime(): Promise<InstallResult> {
    setVideoBusyLabel("Starting GPU bundle install...");
    try {
      let job: GpuBundleJobState;
      try {
        job = await startGpuBundleInstall();
      } catch (err) {
        const message = `Failed to start GPU bundle install: ${err instanceof Error ? err.message : String(err)}`;
        setError(message);
        return { ok: false, output: message, capabilities: {} };
      }

      const POLL_MS = 1500;
      const MAX_WAIT_MS = 30 * 60_000;
      const deadline = Date.now() + MAX_WAIT_MS;
      while (!job.done && Date.now() < deadline) {
        setVideoBusyLabel(formatGpuBundleLabel(job));
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
        try {
          job = await getGpuBundleStatus();
        } catch (err) {
          setVideoBusyLabel(
            `Install in progress (status fetch hiccup: ${err instanceof Error ? err.message : "unknown"})`,
          );
        }
      }

      try {
        const runtime = await getVideoRuntime();
        setVideoRuntimeStatus(runtime);
      } catch {
        // Stale status is fine — restart will refresh.
      }

      if (job.phase === "error" || job.error) {
        const message = job.error || job.message || "GPU bundle install failed.";
        setError(message);
        return { ok: false, output: message, capabilities: {} };
      }
      if (!job.done) {
        const message = "GPU bundle install did not finish within 30 minutes.";
        setError(message);
        return { ok: false, output: message, capabilities: {} };
      }

      setError(null);
      const output = job.requiresRestart
        ? `${job.message}\n\nRestart the backend to activate GPU acceleration.`
        : job.message;
      return { ok: true, output, capabilities: {} };
    } finally {
      setVideoBusyLabel(null);
    }
  }

  async function handleInstallVideoOutputDeps(
    packages?: readonly string[],
  ): Promise<InstallResult> {
    const targetPackages =
      packages && packages.length > 0 ? Array.from(packages) : ["imageio", "imageio-ffmpeg"];
    const isMp4Only =
      targetPackages.length === 2
      && targetPackages.includes("imageio")
      && targetPackages.includes("imageio-ffmpeg");
    const friendlyLabel = isMp4Only
      ? "Installing mp4 encoder (imageio + imageio-ffmpeg)..."
      : `Installing video runtime packages (${targetPackages.join(", ")})...`;
    setVideoBusyLabel(friendlyLabel);
    const failures: string[] = [];
    let lastOutput = "";
    try {
      for (const pkg of targetPackages) {
        try {
          const result = await installPipPackage(pkg);
          lastOutput = result.output;
          if (!result.ok) {
            failures.push(`${pkg}: ${result.output.slice(0, 200)}`);
          }
        } catch (err) {
          failures.push(`${pkg}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      // Re-probe regardless — even a partial install can flip one flag.
      try {
        const runtime = await getVideoRuntime();
        setVideoRuntimeStatus(runtime);
      } catch {
        // keep the pre-install status if the probe itself fails
      }
      if (failures.length > 0) {
        const failureSummary = isMp4Only
          ? "mp4 encoder install failed"
          : "Video runtime package install failed";
        const message = `${failureSummary}:\n${failures.join("\n")}`;
        setError(message);
        return { ok: false, output: message, capabilities: {} };
      }
      setError(null);
      return { ok: true, output: lastOutput, capabilities: {} };
    } finally {
      setVideoBusyLabel(null);
    }
  }

  async function handleDeleteVideoOutput(artifactId: string) {
    try {
      const { outputs } = await deleteVideoOutput(artifactId);
      setVideoOutputs(outputs);
      // If the user just deleted the clip currently rendered in the modal,
      // clear it. If the modal has nothing left to show and isn't busy,
      // close it so we don't leave an empty shell on screen.
      if (videoGenerationArtifact?.artifactId === artifactId) {
        setVideoGenerationArtifact(null);
        if (showVideoGenerationModal && !videoBusy) {
          setShowVideoGenerationModal(false);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete video output.");
    }
  }

  // ── Navigation helpers ──────────────────────────────────────
  function openVideoStudio(modelId?: string) {
    if (modelId) setSelectedVideoModelId(modelId);
    setActiveTab("video-studio");
    setError(null);
  }

  return {
    videoCatalog,
    videoCatalogWithLatest,
    setVideoCatalog,
    latestVideoDiscoverResults,
    setLatestVideoDiscoverResults,
    videoDiscoverTaskFilter,
    setVideoDiscoverTaskFilter,
    videoDiscoverSearchInput,
    setVideoDiscoverSearchInput,
    selectedVideoModelId,
    setSelectedVideoModelId,
    videoPrompt,
    setVideoPrompt,
    videoNegativePrompt,
    setVideoNegativePrompt,
    videoSeedInput,
    setVideoSeedInput,
    videoUseRandomSeed,
    setVideoUseRandomSeed,
    videoWidth,
    setVideoWidth,
    videoHeight,
    setVideoHeight,
    videoNumFrames,
    setVideoNumFrames,
    videoFps,
    setVideoFps,
    videoSteps,
    setVideoSteps,
    videoGuidance,
    setVideoGuidance,
    videoRuntimeStatus,
    setVideoRuntimeStatus,
    videoBusyLabel,
    videoBusy,
    showVideoGenerationModal,
    setShowVideoGenerationModal,
    videoGenerationStartedAt,
    videoGenerationError,
    videoGenerationArtifact,
    videoGenerationRunInfo,
    activeVideoDownloads,
    setActiveVideoDownloads,
    videoOutputs,
    setVideoOutputs,
    // Computed
    videoVariants,
    selectedVideoVariant,
    selectedVideoFamily,
    loadedVideoVariant,
    selectedVideoLoaded,
    selectedVideoWillLoadOnGenerate,
    videoRuntimeLoadedDifferentModel,
    installedVideoVariants,
    videoDiscoverSearchQuery,
    filteredVideoDiscoverFamilies,
    filteredLatestVideoDiscoverResults,
    combinedVideoDiscoverResults,
    videoDiscoverHasActiveFilters,
    // Handlers
    refreshVideoData,
    handleVideoDownload,
    handleCancelVideoDownload,
    handleDeleteVideoDownload,
    handlePreloadVideoModel,
    handleUnloadVideoModel,
    handleVideoGenerate,
    handleDeleteVideoOutput,
    handleInstallVideoOutputDeps,
    handleInstallVideoGpuRuntime,
    openVideoStudio,
  };
}
