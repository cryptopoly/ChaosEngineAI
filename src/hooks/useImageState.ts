import { useDeferredValue, useEffect, useState } from "react";
import {
  cancelImageDownload,
  deleteImageDownload,
  deleteImageOutput,
  downloadImageModel,
  generateImage,
  getGpuBundleStatus,
  getImageCatalog,
  getImageDownloadStatus,
  getImageOutputs,
  getImageRuntime,
  preloadImageModel,
  startGpuBundleInstall,
  unloadImageModel,
} from "../api";
import type { DownloadStatus, GpuBundleJobState, InstallResult } from "../api";

import { IMAGE_RATIO_PRESETS, IMAGE_QUALITY_PRESETS } from "../constants";
import {
  flattenImageVariants,
  defaultImageVariantForFamily,
  findImageVariantById,
  findImageVariantByRepo,
  imageVariantMatchesDiscoverFilters,
  imageDiscoverVariantMatchesQuery,
  imageDiscoverFamilyMatchesQuery,
  imageRuntimeKind,
  imageOrientation,
  imageArtifactTimestamp,
  imageRuntimeErrorStatus,
  buildDownloadStatusMap,
  pendingDownloadStatus,
  failedDownloadStatus,
  isTransientNetworkError,
} from "../utils";
import type {
  ImageModelFamily,
  ImageModelVariant,
  ImageOutputArtifact,
  ImageQualityPreset,
  ImageRuntimeStatus,
  TabId,
} from "../types";
import type {
  DiscoverSort,
  ImageGalleryRuntimeFilter,
  ImageGalleryOrientationFilter,
  ImageGallerySort,
  ImageDiscoverTaskFilter,
  ImageDiscoverAccessFilter,
} from "../types/image";
import { compareDiscoverVariants } from "../utils";

// Human-friendly label for the GPU bundle install progress. Picks the most
// actionable sentence from the job state so the "Installing..." text in
// ImageStudioTab / VideoStudioTab shows what's actually happening right now
// (downloading torch vs. resolving deps vs. verifying CUDA), not a generic
// spinner with no context.
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

export function useImageState(
  backendOnline: boolean,
  setError: (msg: string | null) => void,
  setActiveTab: (tab: TabId) => void,
) {
  const [imageCatalog, setImageCatalog] = useState<ImageModelFamily[]>([]);
  const [latestImageDiscoverResults, setLatestImageDiscoverResults] = useState<ImageModelVariant[]>([]);
  const [imageDiscoverTaskFilter, setImageDiscoverTaskFilter] = useState<ImageDiscoverTaskFilter>("all");
  const [imageDiscoverAccessFilter, setImageDiscoverAccessFilter] = useState<ImageDiscoverAccessFilter>("all");
  // Default to "release" (most recently released first) so the newest
  // drops surface at the top of the Discover grid. Users can swap to
  // likes / downloads via the sort dropdown.
  const [imageDiscoverSort, setImageDiscoverSort] = useState<DiscoverSort>("release");
  const [imageDiscoverSearchInput, setImageDiscoverSearchInput] = useState("");
  const deferredImageDiscoverSearch = useDeferredValue(imageDiscoverSearchInput);
  const [selectedImageModelId, setSelectedImageModelId] = useState("");
  const [imagePrompt, setImagePrompt] = useState("");
  const [imageNegativePrompt, setImageNegativePrompt] = useState("");
  const [imageQualityPreset, setImageQualityPreset] = useState<ImageQualityPreset>("balanced");
  const [imageRatioId, setImageRatioId] = useState<(typeof IMAGE_RATIO_PRESETS)[number]["id"]>("square");
  const [imageWidth, setImageWidth] = useState(1024);
  const [imageHeight, setImageHeight] = useState(1024);
  const [imageSteps, setImageSteps] = useState(24);
  const [imageGuidance, setImageGuidance] = useState(6);
  const [imageBatchSize, setImageBatchSize] = useState(1);
  const [imageUseRandomSeed, setImageUseRandomSeed] = useState(true);
  const [imageSeedInput, setImageSeedInput] = useState("");
  const [imageOutputs, setImageOutputs] = useState<ImageOutputArtifact[]>([]);
  const [imageGallerySearchInput, setImageGallerySearchInput] = useState("");
  const deferredImageGallerySearch = useDeferredValue(imageGallerySearchInput);
  const [imageGalleryModelFilter, setImageGalleryModelFilter] = useState<string>("all");
  const [imageGalleryRuntimeFilter, setImageGalleryRuntimeFilter] = useState<ImageGalleryRuntimeFilter>("all");
  const [imageGalleryOrientationFilter, setImageGalleryOrientationFilter] = useState<ImageGalleryOrientationFilter>("all");
  const [imageGallerySort, setImageGallerySort] = useState<ImageGallerySort>("newest");
  const [imageRuntimeStatus, setImageRuntimeStatus] = useState<ImageRuntimeStatus>({
    activeEngine: "placeholder",
    realGenerationAvailable: false,
    message: "Image Studio is currently using the placeholder engine on this machine.",
    missingDependencies: [],
  });
  const [imageBusyLabel, setImageBusyLabel] = useState<string | null>(null);
  const imageBusy = imageBusyLabel !== null;
  // Live state from the GPU bundle install job — exposed so ImageStudioTab
  // can render an InstallLogPanel under the install button with per-step
  // pip output. Null until the user clicks Install GPU Runtime; retained
  // (not nulled) after completion so users can still expand the log
  // post-restart to confirm which CUDA index won.
  const [gpuBundleJob, setGpuBundleJob] = useState<GpuBundleJobState | null>(null);
  const [showImageGenerationModal, setShowImageGenerationModal] = useState(false);
  const [imageGenerationStartedAt, setImageGenerationStartedAt] = useState<number | null>(null);
  const [imageGenerationError, setImageGenerationError] = useState<string | null>(null);
  const [imageGenerationArtifacts, setImageGenerationArtifacts] = useState<ImageOutputArtifact[]>([]);
  const [selectedImageGenerationArtifactId, setSelectedImageGenerationArtifactId] = useState<string | null>(null);
  const [imageGenerationRunInfo, setImageGenerationRunInfo] = useState<{
    modelName: string;
    prompt: string;
    batchSize: number;
    steps: number;
    needsPipelineLoad: boolean;
  } | null>(null);
  const [activeImageDownloads, setActiveImageDownloads] = useState<Record<string, DownloadStatus>>({});

  // Computed values
  const imageVariants = flattenImageVariants(imageCatalog);
  const allImageVariantsIncludingLatest = [...imageVariants, ...latestImageDiscoverResults.filter((v) => !imageVariants.some((iv) => iv.repo === v.repo))];
  const selectedImageVariant = findImageVariantById(imageCatalog, selectedImageModelId)
    ?? latestImageDiscoverResults.find((v) => v.id === selectedImageModelId)
    ?? imageVariants[0] ?? null;
  const selectedImageFamily = imageCatalog.find((family) =>
    family.variants.some((variant) => variant.id === selectedImageVariant?.id),
  ) ?? null;
  const loadedImageVariant = findImageVariantByRepo(imageCatalog, imageRuntimeStatus.loadedModelRepo)
    ?? (imageRuntimeStatus.loadedModelRepo ? latestImageDiscoverResults.find((v) => v.repo === imageRuntimeStatus.loadedModelRepo) : null)
    ?? null;
  const selectedImageLoaded =
    !!selectedImageVariant &&
    !!imageRuntimeStatus.loadedModelRepo &&
    imageRuntimeStatus.loadedModelRepo === selectedImageVariant.repo;
  const selectedImageWillLoadOnGenerate =
    !!selectedImageVariant &&
    selectedImageVariant.availableLocally &&
    imageRuntimeStatus.realGenerationAvailable &&
    !selectedImageLoaded;
  const imageRuntimeLoadedDifferentModel =
    !!selectedImageVariant &&
    !!loadedImageVariant &&
    loadedImageVariant.repo !== selectedImageVariant.repo;
  const selectedImageGenerationArtifact =
    imageGenerationArtifacts.find((artifact) => artifact.artifactId === selectedImageGenerationArtifactId) ??
    imageGenerationArtifacts[0] ??
    null;
  const installedCatalogVariants = imageVariants.filter((variant) => variant.availableLocally || variant.hasLocalData);
  const installedLatestVariants = latestImageDiscoverResults.filter((variant) => variant.availableLocally || variant.hasLocalData);
  // Merge, deduplicating by repo
  const seenRepos = new Set(installedCatalogVariants.map((v) => v.repo));
  const installedImageVariants = [
    ...installedCatalogVariants,
    ...installedLatestVariants.filter((v) => !seenRepos.has(v.repo)),
  ];

  // Augment imageCatalog with a synthetic "Latest" family for installed tracked models
  // so they appear in dropdowns (Image Studio, etc.) that iterate imageCatalog families
  const catalogRepoSet = new Set(imageVariants.map((v) => v.repo));
  const latestNotInCatalog = latestImageDiscoverResults.filter((v) => !catalogRepoSet.has(v.repo));
  const imageCatalogWithLatest: ImageModelFamily[] = latestNotInCatalog.length > 0
    ? [
        ...imageCatalog,
        {
          id: "latest-tracked",
          name: "Latest / Tracked",
          provider: "Community",
          headline: "Community and tracked image models",
          summary: "Additional image models tracked by ChaosEngineAI",
          updatedLabel: "Tracked",
          badges: [],
          defaultVariantId: latestNotInCatalog[0]?.id ?? "",
          variants: latestNotInCatalog,
        },
      ]
    : imageCatalog;

  const imageDiscoverSearchQuery = deferredImageDiscoverSearch.trim().toLowerCase();

  const filteredImageDiscoverFamilies = imageCatalog
    .map((family) => ({
      ...family,
      variants: family.variants.filter((variant) => {
        if (!imageVariantMatchesDiscoverFilters(variant, imageDiscoverTaskFilter, imageDiscoverAccessFilter)) return false;
        return (
          imageDiscoverFamilyMatchesQuery(family, imageDiscoverSearchQuery) ||
          imageDiscoverVariantMatchesQuery(variant, imageDiscoverSearchQuery)
        );
      }),
    }))
    .filter((family) => family.variants.length > 0);

  const filteredLatestImageDiscoverResults = latestImageDiscoverResults.filter((variant) =>
    imageVariantMatchesDiscoverFilters(variant, imageDiscoverTaskFilter, imageDiscoverAccessFilter) &&
    imageDiscoverVariantMatchesQuery(variant, imageDiscoverSearchQuery),
  );

  const combinedImageDiscoverResults: ImageModelVariant[] = [
    ...filteredImageDiscoverFamilies.flatMap((family) => {
      const variant = defaultImageVariantForFamily(family);
      return variant ? [{ ...variant, familyName: variant.familyName ?? family.name }] : [];
    }),
    ...filteredLatestImageDiscoverResults,
  ].sort((a, b) => compareDiscoverVariants(imageDiscoverSort, a, b));

  const imageDiscoverHasActiveFilters =
    imageDiscoverTaskFilter !== "all" ||
    imageDiscoverAccessFilter !== "all" ||
    imageDiscoverSearchQuery.length > 0;

  const imageOutputsNewestFirst = [...imageOutputs].sort((left, right) => imageArtifactTimestamp(right) - imageArtifactTimestamp(left));
  const recentImageOutputs = imageOutputsNewestFirst.slice(0, 2);

  const imageGalleryModelOptions = Array.from(
    new Map(
      imageOutputsNewestFirst.map((artifact) => [
        artifact.modelId,
        { id: artifact.modelId, name: artifact.modelName },
      ]),
    ).values(),
  ).sort((left, right) => left.name.localeCompare(right.name));

  const imageGalleryHasActiveFilters =
    imageGalleryModelFilter !== "all" ||
    imageGalleryRuntimeFilter !== "all" ||
    imageGalleryOrientationFilter !== "all" ||
    imageGallerySort !== "newest" ||
    deferredImageGallerySearch.trim().length > 0;

  const filteredImageOutputs = [...imageOutputs].filter((artifact) => {
    if (imageGalleryModelFilter !== "all" && artifact.modelId !== imageGalleryModelFilter) return false;
    if (imageGalleryRuntimeFilter === "diffusers" && imageRuntimeKind(artifact.runtimeLabel) !== "diffusers") return false;
    if (imageGalleryRuntimeFilter === "placeholder" && imageRuntimeKind(artifact.runtimeLabel) !== "placeholder") return false;
    if (imageGalleryRuntimeFilter === "warning" && !artifact.runtimeNote) return false;
    if (
      imageGalleryOrientationFilter !== "all" &&
      imageOrientation(artifact.width, artifact.height) !== imageGalleryOrientationFilter
    ) return false;
    const query = deferredImageGallerySearch.trim().toLowerCase();
    if (!query) return true;
    const haystack = `${artifact.modelName} ${artifact.prompt} ${artifact.runtimeLabel ?? ""} ${artifact.runtimeNote ?? ""}`.toLowerCase();
    return haystack.includes(query);
  }).sort((left, right) => (
    imageGallerySort === "oldest"
      ? imageArtifactTimestamp(left) - imageArtifactTimestamp(right)
      : imageArtifactTimestamp(right) - imageArtifactTimestamp(left)
  ));

  const imageGalleryRealCount = imageOutputs.filter((artifact) => imageRuntimeKind(artifact.runtimeLabel) === "diffusers").length;
  const imageGalleryPlaceholderCount = imageOutputs.filter((artifact) => imageRuntimeKind(artifact.runtimeLabel) === "placeholder").length;
  const imageGalleryWarningCount = imageOutputs.filter((artifact) => Boolean(artifact.runtimeNote)).length;
  const imageGalleryModelCount = imageGalleryModelOptions.length;

  // Image catalog selection sync
  useEffect(() => {
    if (!imageCatalog.length) {
      setSelectedImageModelId("");
      return;
    }
    const variants = flattenImageVariants(imageCatalog);
    // Check both curated and latest variants so tracked models don't get deselected
    if (variants.some((variant) => variant.id === selectedImageModelId)) return;
    if (latestImageDiscoverResults.some((variant) => variant.id === selectedImageModelId)) return;
    const preferred =
      variants.find((variant) => variant.availableLocally) ??
      defaultImageVariantForFamily(imageCatalog[0]);
    setSelectedImageModelId(preferred?.id ?? "");
  }, [imageCatalog, selectedImageModelId, latestImageDiscoverResults]);

  // Image download polling
  const hasActiveImageDownloads = Object.values(activeImageDownloads).some((download) => download.state === "downloading");
  useEffect(() => {
    if (!hasActiveImageDownloads || !backendOnline) return;
    const interval = window.setInterval(() => {
      void (async () => {
        try {
          const statuses = await getImageDownloadStatus();
          setActiveImageDownloads(buildDownloadStatusMap(statuses));
          if (statuses.some((status) => status.state === "completed")) {
            void refreshImageData();
          }
        } catch {
          // keep the last known state until the next poll
        }
      })();
    }, 2000);
    return () => window.clearInterval(interval);
  }, [hasActiveImageDownloads, backendOnline]);

  async function refreshImageData() {
    const [catalog, outputs, statuses, runtime] = await Promise.allSettled([
      getImageCatalog(),
      getImageOutputs(),
      getImageDownloadStatus(),
      getImageRuntime(),
    ]);
    const failures = [catalog, outputs, statuses, runtime].filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );

    if (catalog.status === "fulfilled") {
      setImageCatalog(catalog.value.families);
      setLatestImageDiscoverResults(catalog.value.latest ?? []);
    }
    if (outputs.status === "fulfilled") setImageOutputs(outputs.value);
    if (statuses.status === "fulfilled") setActiveImageDownloads(buildDownloadStatusMap(statuses.value));
    if (runtime.status === "fulfilled") {
      setImageRuntimeStatus(runtime.value);
    } else if (failures.length > 0) {
      setImageRuntimeStatus(imageRuntimeErrorStatus(failures[0].reason));
    }

    if (failures.length > 0) {
      const firstError = failures[0].reason;
      // Swallow transient network errors from background refreshes. These
      // fire on startup before the backend has bound its port, and on
      // Windows the race window is several seconds — long enough that the
      // user sees a sticky "Failed to fetch" banner even though the app
      // is working. Real backend errors (HTTP 4xx/5xx with detail) still
      // surface as usual.
      if (!isTransientNetworkError(firstError)) {
        setError(firstError instanceof Error ? firstError.message : "Could not load image runtime data.");
      }
    }
  }

  async function handleImageDownload(repo: string) {
    try {
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: pendingDownloadStatus(repo, prev[repo]) }));
      const download = await downloadImageModel(repo);
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: download }));
      void refreshImageData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Image download failed");
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: failedDownloadStatus(repo, String(err)) }));
    }
  }

  async function handleCancelImageDownload(repo: string) {
    try {
      const download = await cancelImageDownload(repo);
      setActiveImageDownloads((prev) => ({ ...prev, [repo]: download }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not pause image download");
    }
  }

  async function handleDeleteImageDownload(repo: string) {
    try {
      await deleteImageDownload(repo);
      const statuses = await getImageDownloadStatus();
      setActiveImageDownloads(buildDownloadStatusMap(statuses));
      await refreshImageData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete image download");
    }
  }

  function applyImageRatioPreset(presetId: (typeof IMAGE_RATIO_PRESETS)[number]["id"]) {
    const preset = IMAGE_RATIO_PRESETS.find((item) => item.id === presetId);
    if (!preset) return;
    setImageRatioId(presetId);
    setImageWidth(preset.width);
    setImageHeight(preset.height);
  }

  function applyImageQuality(presetId: ImageQualityPreset) {
    const preset = IMAGE_QUALITY_PRESETS.find((item) => item.id === presetId);
    if (!preset) return;
    setImageQualityPreset(presetId);
    setImageSteps(preset.steps);
    setImageGuidance(preset.guidance);
  }

  function openImageStudio(modelId?: string) {
    if (modelId) setSelectedImageModelId(modelId);
    setActiveTab("image-studio");
    setError(null);
  }

  function openImageGallery(modelId?: string) {
    if (modelId) setImageGalleryModelFilter(modelId);
    setActiveTab("image-gallery");
    setError(null);
  }

  function resetImageGalleryFilters() {
    setImageGallerySearchInput("");
    setImageGalleryModelFilter("all");
    setImageGalleryRuntimeFilter("all");
    setImageGalleryOrientationFilter("all");
    setImageGallerySort("newest");
  }

  function hydrateImageFormFromArtifact(artifact: ImageOutputArtifact, randomizeSeed = false) {
    setSelectedImageModelId(artifact.modelId);
    setImagePrompt(artifact.prompt);
    setImageNegativePrompt(artifact.negativePrompt ?? "");
    setImageWidth(artifact.width);
    setImageHeight(artifact.height);
    setImageSteps(artifact.steps);
    setImageGuidance(artifact.guidance);
    setImageBatchSize(1);
    const ratioPreset = IMAGE_RATIO_PRESETS.find(
      (preset) => preset.width === artifact.width && preset.height === artifact.height,
    );
    if (ratioPreset) setImageRatioId(ratioPreset.id);
    const qualityPreset = IMAGE_QUALITY_PRESETS.find(
      (preset) => preset.steps === artifact.steps && preset.guidance === artifact.guidance,
    );
    if (qualityPreset) setImageQualityPreset(qualityPreset.id);
    setImageUseRandomSeed(randomizeSeed);
    setImageSeedInput(randomizeSeed ? "" : String(artifact.seed));
    openImageStudio(artifact.modelId);
  }

  async function submitImageGeneration(overrides?: {
    modelId?: string;
    prompt?: string;
    negativePrompt?: string;
    width?: number;
    height?: number;
    steps?: number;
    guidance?: number;
    batchSize?: number;
    qualityPreset?: ImageQualityPreset;
    seed?: number | null;
  }) {
    const modelId = overrides?.modelId ?? selectedImageModelId;
    const prompt = (overrides?.prompt ?? imagePrompt).trim();
    if (!modelId) { setError("Choose an image model first."); return; }
    if (!prompt) { setError("Write a prompt before generating."); return; }
    const seed =
      overrides && "seed" in overrides
        ? overrides.seed ?? null
        : imageUseRandomSeed
          ? null
          : (() => {
              const parsed = Number.parseInt(imageSeedInput, 10);
              return Number.isFinite(parsed) ? parsed : null;
            })();

    const selectedVariantForRequest =
      findImageVariantById(imageCatalog, modelId) ??
      (selectedImageVariant?.id === modelId ? selectedImageVariant : null);
    const requestNeedsPipelineLoad =
      !!selectedVariantForRequest &&
      selectedVariantForRequest.availableLocally &&
      imageRuntimeStatus.realGenerationAvailable &&
      imageRuntimeStatus.loadedModelRepo !== selectedVariantForRequest.repo;
    setShowImageGenerationModal(true);
    setImageGenerationStartedAt(Date.now());
    setImageGenerationError(null);
    setImageGenerationArtifacts([]);
    setSelectedImageGenerationArtifactId(null);
    setImageGenerationRunInfo({
      modelName: selectedVariantForRequest?.name ?? "Image model",
      prompt,
      batchSize: overrides?.batchSize ?? imageBatchSize,
      steps: overrides?.steps ?? imageSteps,
      needsPipelineLoad: requestNeedsPipelineLoad,
    });
    setImageBusyLabel(
      requestNeedsPipelineLoad
        ? `Loading ${selectedVariantForRequest?.name ?? "image model"} into memory...`
        : "Generating image...",
    );
    try {
      const response = await generateImage({
        modelId,
        prompt,
        negativePrompt: overrides?.negativePrompt ?? imageNegativePrompt,
        width: overrides?.width ?? imageWidth,
        height: overrides?.height ?? imageHeight,
        steps: overrides?.steps ?? imageSteps,
        guidance: overrides?.guidance ?? imageGuidance,
        batchSize: overrides?.batchSize ?? imageBatchSize,
        qualityPreset: overrides?.qualityPreset ?? imageQualityPreset,
        seed,
      });
      setImageOutputs(response.outputs);
      if (response.runtime) setImageRuntimeStatus(response.runtime);
      setImageGenerationArtifacts(response.artifacts);
      setSelectedImageGenerationArtifactId(response.artifacts[0]?.artifactId ?? null);
      if (seed !== null && !imageUseRandomSeed && !overrides) {
        setImageSeedInput(String(seed));
      }
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Image generation failed.";
      const detail = `${message}. Check the Logs tab (filter: images) for backend details.`;
      setError(detail);
      setImageGenerationError(detail);
    } finally {
      setImageBusyLabel(null);
      setImageGenerationStartedAt(null);
    }
  }

  async function handlePreloadImageModel(variant?: ImageModelVariant | null) {
    if (!variant) { setError("Choose an installed image model first."); return; }
    setImageBusyLabel(`Loading ${variant.name} into memory...`);
    try {
      const runtime = await preloadImageModel(variant.id);
      setImageRuntimeStatus(runtime);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not preload the image model.");
    } finally {
      setImageBusyLabel(null);
    }
  }

  // One-click install for the diffusers image runtime. Kicks off an
  // async ``/api/setup/install-gpu-bundle`` job on the backend (runs in a
  // background thread so the HTTP call returns fast) and then polls the
  // status endpoint at ~1 Hz to surface progress. Writes to a persistent
  // ``~/.chaosengine/extras/site-packages`` tree outside the ephemeral
  // bundled runtime, so updates to ChaosEngineAI itself don't wipe the
  // ~2 GB of CUDA torch that the user just downloaded.
  //
  // Returns an ``InstallResult`` shaped like the old installPipPackage
  // response so the caller UI layer doesn't need to change. The live
  // progress flows through ``imageBusyLabel`` and is picked up by the
  // existing "Installing..." text in ImageStudioTab.
  async function handleInstallImageRuntime(): Promise<InstallResult> {
    setImageBusyLabel("Starting GPU bundle install...");
    try {
      let job: GpuBundleJobState;
      try {
        job = await startGpuBundleInstall();
        setGpuBundleJob(job);
      } catch (err) {
        const message = `Failed to start GPU bundle install: ${err instanceof Error ? err.message : String(err)}`;
        setError(message);
        return { ok: false, output: message, capabilities: {} };
      }

      // Poll until the background job reports done / error. 1.5 s cadence
      // is a compromise: slow enough to not hammer the backend, fast enough
      // that the UI feels live when torch finishes an index attempt.
      const POLL_MS = 1500;
      const MAX_WAIT_MS = 30 * 60_000;
      const deadline = Date.now() + MAX_WAIT_MS;
      while (!job.done && Date.now() < deadline) {
        setImageBusyLabel(formatGpuBundleLabel(job));
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
        try {
          job = await getGpuBundleStatus();
          setGpuBundleJob(job);  // UI log panel re-renders with latest attempts
        } catch (err) {
          setImageBusyLabel(
            `Install in progress (status fetch hiccup: ${err instanceof Error ? err.message : "unknown"})`,
          );
        }
      }

      // Re-probe the image runtime so the badge flips from placeholder -> diffusers.
      try {
        const runtime = await getImageRuntime();
        setImageRuntimeStatus(runtime);
      } catch {
        // Keep the pre-install status; user will see the stale badge until restart.
      }

      if (job.phase === "error" || job.error) {
        const rawMessage = job.error || job.message || "GPU bundle install failed.";
        // Append a hint pointing at the log panel so the toast isn't the
        // only thing the user sees — the real detail is in the expanded
        // attempts list. Also include the target dir so they know where
        // pip was writing (or failing to write) to.
        const hint = job.targetDir
          ? ` See the install log below for per-step pip output. Target: ${job.targetDir}`
          : " See the install log below for per-step pip output.";
        const message = rawMessage + hint;
        setError(message);
        return { ok: false, output: message, capabilities: {} };
      }
      if (!job.done) {
        const message = "GPU bundle install did not finish within 30 minutes. See the install log below.";
        setError(message);
        return { ok: false, output: message, capabilities: {} };
      }

      setError(null);
      const output = job.requiresRestart
        ? `${job.message}\n\nRestart the backend to activate GPU acceleration.`
        : job.message;
      return { ok: true, output, capabilities: {} };
    } finally {
      setImageBusyLabel(null);
    }
  }

  async function handleUnloadImageModel(variant?: ImageModelVariant | null) {
    setImageBusyLabel(`Unloading ${(variant?.name ?? loadedImageVariant?.name ?? "image model")} from memory...`);
    try {
      const runtime = await unloadImageModel(variant?.id);
      setImageRuntimeStatus(runtime);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not unload the image model.");
    } finally {
      setImageBusyLabel(null);
    }
  }

  async function handleDeleteImageArtifact(artifactId: string) {
    try {
      const response = await deleteImageOutput(artifactId);
      setImageOutputs(response.outputs);
      const nextArtifacts = imageGenerationArtifacts.filter((artifact) => artifact.artifactId !== artifactId);
      setImageGenerationArtifacts(nextArtifacts);
      setSelectedImageGenerationArtifactId((current) => {
        if (current && nextArtifacts.some((artifact) => artifact.artifactId === current)) return current;
        return nextArtifacts[0]?.artifactId ?? null;
      });
      if (showImageGenerationModal && nextArtifacts.length === 0 && !imageBusy) {
        setShowImageGenerationModal(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete image output.");
    }
  }

  async function handleVaryImageSeed(artifact: ImageOutputArtifact) {
    const matchedQualityPreset =
      IMAGE_QUALITY_PRESETS.find(
        (preset) => preset.steps === artifact.steps && preset.guidance === artifact.guidance,
      )?.id ?? imageQualityPreset;
    hydrateImageFormFromArtifact(artifact, true);
    await submitImageGeneration({
      modelId: artifact.modelId,
      prompt: artifact.prompt,
      negativePrompt: artifact.negativePrompt ?? "",
      width: artifact.width,
      height: artifact.height,
      steps: artifact.steps,
      guidance: artifact.guidance,
      batchSize: 1,
      qualityPreset: matchedQualityPreset,
      seed: Math.floor(Math.random() * 2147483647),
    });
  }

  function handleUseSameImageSettings(artifact: ImageOutputArtifact, closeModal = false) {
    hydrateImageFormFromArtifact(artifact);
    if (closeModal) setShowImageGenerationModal(false);
  }

  return {
    imageCatalog,
    imageCatalogWithLatest,
    setImageCatalog,
    latestImageDiscoverResults,
    setLatestImageDiscoverResults,
    imageDiscoverTaskFilter,
    setImageDiscoverTaskFilter,
    imageDiscoverAccessFilter,
    setImageDiscoverAccessFilter,
    imageDiscoverSort,
    setImageDiscoverSort,
    imageDiscoverSearchInput,
    setImageDiscoverSearchInput,
    selectedImageModelId,
    setSelectedImageModelId,
    imagePrompt,
    setImagePrompt,
    imageNegativePrompt,
    setImageNegativePrompt,
    imageQualityPreset,
    imageRatioId,
    imageWidth,
    setImageWidth,
    imageHeight,
    setImageHeight,
    imageSteps,
    setImageSteps,
    imageGuidance,
    setImageGuidance,
    imageBatchSize,
    setImageBatchSize,
    imageUseRandomSeed,
    setImageUseRandomSeed,
    imageSeedInput,
    setImageSeedInput,
    imageOutputs,
    setImageOutputs,
    imageGallerySearchInput,
    setImageGallerySearchInput,
    imageGalleryModelFilter,
    setImageGalleryModelFilter,
    imageGalleryRuntimeFilter,
    setImageGalleryRuntimeFilter,
    imageGalleryOrientationFilter,
    setImageGalleryOrientationFilter,
    imageGallerySort,
    setImageGallerySort,
    imageRuntimeStatus,
    setImageRuntimeStatus,
    imageBusyLabel,
    imageBusy,
    showImageGenerationModal,
    setShowImageGenerationModal,
    imageGenerationStartedAt,
    imageGenerationError,
    imageGenerationArtifacts,
    selectedImageGenerationArtifactId,
    setSelectedImageGenerationArtifactId,
    imageGenerationRunInfo,
    activeImageDownloads,
    setActiveImageDownloads,
    // Computed
    imageVariants,
    selectedImageVariant,
    selectedImageFamily,
    loadedImageVariant,
    selectedImageLoaded,
    selectedImageWillLoadOnGenerate,
    imageRuntimeLoadedDifferentModel,
    selectedImageGenerationArtifact,
    installedImageVariants,
    imageDiscoverSearchQuery,
    filteredImageDiscoverFamilies,
    filteredLatestImageDiscoverResults,
    combinedImageDiscoverResults,
    imageDiscoverHasActiveFilters,
    imageOutputsNewestFirst,
    recentImageOutputs,
    imageGalleryModelOptions,
    imageGalleryHasActiveFilters,
    filteredImageOutputs,
    imageGalleryRealCount,
    imageGalleryPlaceholderCount,
    imageGalleryWarningCount,
    imageGalleryModelCount,
    // Handlers
    refreshImageData,
    handleImageDownload,
    handleCancelImageDownload,
    handleDeleteImageDownload,
    applyImageRatioPreset,
    applyImageQuality,
    openImageStudio,
    openImageGallery,
    resetImageGalleryFilters,
    submitImageGeneration,
    handlePreloadImageModel,
    handleUnloadImageModel,
    handleInstallImageRuntime,
    handleDeleteImageArtifact,
    handleVaryImageSeed,
    handleUseSameImageSettings,
    gpuBundleJob,
  };
}
