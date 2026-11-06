import { ImageModelsPage } from "../features/images/ImageModelsPage";
import { ImageStudioTab } from "../features/images/ImageStudioTab";
import { ImageGalleryTab } from "../features/images/ImageGalleryTab";
import type { AppTabContext } from "./context";

export function ImageModelsView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    imgState,
    modelsAvailability,
    setAreaAvailability,
    fileRevealLabel,
    nativeBackends,
    setActiveTab,
    handleOpenExternalUrl,
    handleRevealPath,
  } = ctx;
  return (
    <ImageModelsPage
      catalog={imgState.imageCatalog}
      latest={imgState.latestImageDiscoverResults}
      searchInput={imgState.imageDiscoverSearchInput}
      onSearchInputChange={imgState.setImageDiscoverSearchInput}
      taskFilter={imgState.imageDiscoverTaskFilter}
      onTaskFilterChange={imgState.setImageDiscoverTaskFilter}
      accessFilter={imgState.imageDiscoverAccessFilter}
      onAccessFilterChange={imgState.setImageDiscoverAccessFilter}
      sort={imgState.imageDiscoverSort}
      onSortChange={imgState.setImageDiscoverSort}
      availability={modelsAvailability.image}
      onAvailabilityChange={(value) => setAreaAvailability("image", value)}
      activeImageDownloads={imgState.activeImageDownloads}
      selectedImageVariant={imgState.selectedImageVariant}
      loadedImageRepo={imgState.imageRuntimeStatus.loadedModelRepo ?? null}
      fileRevealLabel={fileRevealLabel}
      nativeBackends={nativeBackends}
      hostSystem={workspace.system}
      onActiveTabChange={setActiveTab}
      onOpenImageStudio={imgState.openImageStudio}
      onImageDownload={(repo) => void imgState.handleImageDownload(repo)}
      onCancelImageDownload={(repo) => void imgState.handleCancelImageDownload(repo)}
      onDeleteImageDownload={(repo) => void imgState.handleDeleteImageDownload(repo)}
      onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
      onRevealPath={(path) => void handleRevealPath(path)}
    />
  );
}

export function ImageStudioView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    imgState,
    tauriBackend,
    busy,
    busyAction,
    backendOnline,
    nativeBackends,
    isAppleSilicon,
    setActiveTab,
    settings,
    installingCudaTorch,
    cudaTorchRawResult,
    handleInstallCudaTorch,
    handleOpenExternalUrl,
    handleRevealPath,
  } = ctx;
  const {
    handleRestartServer,
  } = settings;
  return (
    <ImageStudioTab
      imageCatalog={imgState.imageCatalogWithLatest}
      residentChatModelGb={workspace.runtime.residentModelGb}
      selectedImageModelId={imgState.selectedImageModelId}
      onSelectedImageModelIdChange={imgState.setSelectedImageModelId}
      selectedImageVariant={imgState.selectedImageVariant}
      selectedImageFamily={imgState.selectedImageFamily}
      selectedImageLoaded={imgState.selectedImageLoaded}
      selectedImageWillLoadOnGenerate={imgState.selectedImageWillLoadOnGenerate}
      imageRuntimeLoadedDifferentModel={imgState.imageRuntimeLoadedDifferentModel}
      loadedImageVariant={imgState.loadedImageVariant}
      imageRuntimeStatus={imgState.imageRuntimeStatus}
      tauriBackend={tauriBackend}
      busy={busy}
      busyAction={busyAction}
      imageBusy={imgState.imageBusy}
      imageBusyLabel={imgState.imageBusyLabel}
      backendOnline={backendOnline}
      nativeBackends={nativeBackends}
      activeImageDownloads={imgState.activeImageDownloads}
      imagePrompt={imgState.imagePrompt}
      onImagePromptChange={imgState.setImagePrompt}
      imageNegativePrompt={imgState.imageNegativePrompt}
      imageReferences={imgState.imageReferences}
      onImageReferencesChange={imgState.setImageReferences}
      imageReferenceStrength={imgState.imageReferenceStrength}
      onImageReferenceStrengthChange={imgState.setImageReferenceStrength}
      onImageNegativePromptChange={imgState.setImageNegativePrompt}
      imageQualityPreset={imgState.imageQualityPreset}
      imageDraftMode={imgState.imageDraftMode}
      onImageDraftModeChange={imgState.setImageDraftMode}
      imageSampler={imgState.imageSampler}
      onImageSamplerChange={imgState.setImageSampler}
      imageCacheStrategy={imgState.imageCacheStrategy}
      onImageCacheStrategyChange={imgState.setImageCacheStrategy}
      imageCacheRelL1Thresh={imgState.imageCacheRelL1Thresh}
      onImageCacheRelL1ThreshChange={imgState.setImageCacheRelL1Thresh}
      imageCfgDecay={imgState.imageCfgDecay}
      onImageCfgDecayChange={imgState.setImageCfgDecay}
      imagePreviewVae={imgState.imagePreviewVae}
      onImagePreviewVaeChange={imgState.setImagePreviewVae}
      imageFp8LayerwiseCasting={imgState.imageFp8LayerwiseCasting}
      onImageFp8LayerwiseCastingChange={imgState.setImageFp8LayerwiseCasting}
      isAppleSiliconHost={isAppleSilicon}
      hostSystem={workspace.system}
      imageRatioId={imgState.imageRatioId}
      imageWidth={imgState.imageWidth}
      onImageWidthChange={imgState.setImageWidth}
      imageHeight={imgState.imageHeight}
      onImageHeightChange={imgState.setImageHeight}
      imageSteps={imgState.imageSteps}
      onImageStepsChange={imgState.setImageSteps}
      imageGuidance={imgState.imageGuidance}
      onImageGuidanceChange={imgState.setImageGuidance}
      imageBatchSize={imgState.imageBatchSize}
      onImageBatchSizeChange={imgState.setImageBatchSize}
      imageUseRandomSeed={imgState.imageUseRandomSeed}
      onImageUseRandomSeedChange={imgState.setImageUseRandomSeed}
      imageSeedInput={imgState.imageSeedInput}
      onImageSeedInputChange={imgState.setImageSeedInput}
      imageOutputs={imgState.imageOutputs}
      recentImageOutputs={imgState.recentImageOutputs}
      onActiveTabChange={setActiveTab}
      onOpenImageStudio={imgState.openImageStudio}
      onOpenImageGallery={imgState.openImageGallery}
      onSubmitImageGeneration={() => void imgState.submitImageGeneration()}
      onApplyImageRatioPreset={imgState.applyImageRatioPreset}
      onApplyImageQuality={imgState.applyImageQuality}
      onPreloadImageModel={(variant) => void imgState.handlePreloadImageModel(variant)}
      onUnloadImageModel={(variant) => void imgState.handleUnloadImageModel(variant)}
      onInstallImageRuntime={() => imgState.handleInstallImageRuntime()}
      onInstallCudaTorch={() => void handleInstallCudaTorch()}
      installingCudaTorch={installingCudaTorch}
      cudaTorchResult={cudaTorchRawResult}
      gpuBundleJob={imgState.gpuBundleJob}
      onImageDownload={(repo) => void imgState.handleImageDownload(repo)}
      onCancelImageDownload={(repo) => void imgState.handleCancelImageDownload(repo)}
      onDeleteImageDownload={(repo) => void imgState.handleDeleteImageDownload(repo)}
      onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
      onRestartServer={() => void handleRestartServer()}
      onUseSameImageSettings={imgState.handleUseSameImageSettings}
      onVaryImageSeed={(a) => void imgState.handleVaryImageSeed(a)}
      onRevealPath={(path) => void handleRevealPath(path)}
      onDeleteImageArtifact={(id) => void imgState.handleDeleteImageArtifact(id)}
    />
  );
}

export function ImageGalleryView({ ctx }: { ctx: AppTabContext }) {
  const {
    imgState,
    setActiveTab,
    handleOpenExternalUrl,
    handleRevealPath,
  } = ctx;
  return (
    <ImageGalleryTab
      imageOutputs={imgState.imageOutputs}
      filteredImageOutputs={imgState.filteredImageOutputs}
      imageCatalog={imgState.imageCatalog}
      imageBusy={imgState.imageBusy}
      imageGallerySearchInput={imgState.imageGallerySearchInput}
      onImageGallerySearchInputChange={imgState.setImageGallerySearchInput}
      imageGalleryModelFilter={imgState.imageGalleryModelFilter}
      onImageGalleryModelFilterChange={imgState.setImageGalleryModelFilter}
      imageGalleryRuntimeFilter={imgState.imageGalleryRuntimeFilter}
      onImageGalleryRuntimeFilterChange={imgState.setImageGalleryRuntimeFilter}
      imageGalleryOrientationFilter={imgState.imageGalleryOrientationFilter}
      onImageGalleryOrientationFilterChange={imgState.setImageGalleryOrientationFilter}
      imageGallerySort={imgState.imageGallerySort}
      onImageGallerySortChange={imgState.setImageGallerySort}
      imageGalleryModelOptions={imgState.imageGalleryModelOptions}
      imageGalleryModelCount={imgState.imageGalleryModelCount}
      imageGalleryRealCount={imgState.imageGalleryRealCount}
      imageGalleryPlaceholderCount={imgState.imageGalleryPlaceholderCount}
      imageGalleryWarningCount={imgState.imageGalleryWarningCount}
      imageGalleryHasActiveFilters={imgState.imageGalleryHasActiveFilters}
      onActiveTabChange={setActiveTab}
      onOpenImageStudio={imgState.openImageStudio}
      onResetImageGalleryFilters={imgState.resetImageGalleryFilters}
      onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
      onUseSameImageSettings={imgState.handleUseSameImageSettings}
      onVaryImageSeed={(a) => void imgState.handleVaryImageSeed(a)}
      onRevealPath={(path) => void handleRevealPath(path)}
      onDeleteImageArtifact={(id) => void imgState.handleDeleteImageArtifact(id)}
    />
  );
}
