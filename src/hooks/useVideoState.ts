import { useDeferredValue, useEffect, useState } from "react";
import {
  cancelVideoDownload,
  deleteVideoDownload,
  downloadVideoModel,
  getVideoCatalog,
  getVideoDownloadStatus,
  getVideoRuntime,
  preloadVideoModel,
  unloadVideoModel,
} from "../api";
import type { DownloadStatus } from "../api";
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
  VideoModelFamily,
  VideoModelVariant,
  VideoRuntimeStatus,
} from "../types";
import type { VideoDiscoverTaskFilter } from "../types/video";

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
    const [catalog, statuses, runtime] = await Promise.allSettled([
      getVideoCatalog(),
      getVideoDownloadStatus(),
      getVideoRuntime(),
    ]);
    const failures = [catalog, statuses, runtime].filter(
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
    openVideoStudio,
  };
}
