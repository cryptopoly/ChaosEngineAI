import { useDeferredValue, useEffect, useState } from "react";
import {
  cancelVideoDownload,
  deleteVideoDownload,
  deleteVideoOutput,
  downloadVideoModel,
  generateVideo,
  getVideoCatalog,
  getVideoDownloadStatus,
  getVideoOutputs,
  getVideoRuntime,
  installPipPackage,
  preloadVideoModel,
  unloadVideoModel,
} from "../api";
import type { DownloadStatus, InstallResult } from "../api";
import {
  buildDownloadStatusMap,
  defaultVideoVariantForFamily,
  failedDownloadStatus,
  findVideoVariantById,
  findVideoVariantByRepo,
  flattenVideoVariants,
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
      setError(firstError instanceof Error ? firstError.message : "Could not load video runtime data.");
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

    // Pull defaults from the variant so the frontend doesn't need to know the
    // per-model resolution knobs yet — Phase 11 can expose them as controls.
    const [recommendedWidth, recommendedHeight] = parseRecommendedResolution(
      selectedVideoVariant.recommendedResolution,
      768,
      512,
    );
    const estimatedFps = 24;
    const estimatedFrames = Math.max(
      8,
      Math.round((selectedVideoVariant.defaultDurationSeconds || 4) * estimatedFps),
    );

    const payload: VideoGenerationPayload = {
      modelId: selectedVideoVariant.id,
      prompt: trimmedPrompt,
      negativePrompt: videoNegativePrompt.trim() || undefined,
      width: recommendedWidth,
      height: recommendedHeight,
      numFrames: estimatedFrames,
      fps: estimatedFps,
      steps: 50,
      guidance: 3.0,
      seed: parsedSeed,
    };

    setVideoBusyLabel(`Generating ${estimatedFrames}-frame clip with ${selectedVideoVariant.name}...`);
    setError(null);
    try {
      const response = await generateVideo(payload);
      setVideoOutputs(response.outputs);
      if (response.runtime) setVideoRuntimeStatus(response.runtime);
      setActiveTab("video-gallery");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Video generation failed.");
    } finally {
      setVideoBusyLabel(null);
    }
  }

  // ── Dependency install ──────────────────────────────────────
  //
  // Diffusers can render frames without imageio, but exporting them as an mp4
  // needs both the `imageio` Python package and its `imageio-ffmpeg` plugin.
  // The Video Studio calls this when it detects either is missing so the user
  // can unstick things without dropping to a terminal.
  async function handleInstallVideoOutputDeps(): Promise<InstallResult> {
    setVideoBusyLabel("Installing mp4 encoder (imageio + imageio-ffmpeg)...");
    const failures: string[] = [];
    let lastOutput = "";
    try {
      for (const pkg of ["imageio", "imageio-ffmpeg"] as const) {
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
        const message = `mp4 encoder install failed:\n${failures.join("\n")}`;
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
    videoRuntimeStatus,
    setVideoRuntimeStatus,
    videoBusyLabel,
    videoBusy,
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
    openVideoStudio,
  };
}
