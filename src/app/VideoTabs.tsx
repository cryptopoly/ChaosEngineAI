import { VideoModelsPage } from "../features/video/VideoModelsPage";
import { VideoStudioTab } from "../features/video/VideoStudioTab";
import { VideoGalleryTab } from "../features/video/VideoGalleryTab";
import type { AppTabContext } from "./context";

export function VideoModelsView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    videoState,
    modelsAvailability,
    setAreaAvailability,
    fileRevealLabel,
    nativeBackends,
    setActiveTab,
    handleOpenExternalUrl,
    handleRevealPath,
  } = ctx;
  return (
    <VideoModelsPage
      catalog={videoState.videoCatalog}
      latest={videoState.latestVideoDiscoverResults}
      searchInput={videoState.videoDiscoverSearchInput}
      onSearchInputChange={videoState.setVideoDiscoverSearchInput}
      taskFilter={videoState.videoDiscoverTaskFilter}
      onTaskFilterChange={videoState.setVideoDiscoverTaskFilter}
      sort={videoState.videoDiscoverSort}
      onSortChange={videoState.setVideoDiscoverSort}
      availability={modelsAvailability.video}
      onAvailabilityChange={(value) => setAreaAvailability("video", value)}
      activeVideoDownloads={videoState.activeVideoDownloads}
      selectedVideoVariant={videoState.selectedVideoVariant}
      // The runtime reports a repo; several rows can share one. The row
      // selected in the Studio is the one that was loaded when its repo
      // matches, otherwise the first row with that repo (as before).
      loadedVideoVariantId={
        videoState.selectedVideoLoaded
          ? videoState.selectedVideoVariant?.id ?? null
          : videoState.loadedVideoVariant?.id ?? null
      }
      fileRevealLabel={fileRevealLabel}
      nativeBackends={nativeBackends}
      hostSystem={workspace.system}
      longLiveStatus={videoState.longLiveStatus}
      installingLongLive={videoState.installingLongLive}
      longLiveJob={videoState.longLiveJob}
      onActiveTabChange={setActiveTab}
      onOpenVideoStudio={videoState.openVideoStudio}
      onVideoDownload={(repo, modelId) => void videoState.handleVideoDownload(repo, modelId)}
      onCancelVideoDownload={(repo) => void videoState.handleCancelVideoDownload(repo)}
      onDeleteVideoDownload={(repo) => void videoState.handleDeleteVideoDownload(repo)}
      onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
      onRevealPath={(path) => void handleRevealPath(path)}
      onRefreshLongLiveStatus={() => void videoState.refreshLongLiveStatus()}
      onInstallLongLive={() => videoState.handleInstallLongLive()}
    />
  );
}

export function VideoStudioView({ ctx }: { ctx: AppTabContext }) {
  const {
    workspace,
    imgState,
    videoState,
    tauriBackend,
    nativeBackends,
    busy,
    busyAction,
    backendOnline,
    setActiveTab,
    settings,
    installingCudaTorch,
    cudaTorchRawResult,
    handleInstallCudaTorch,
    handleOpenExternalUrl,
  } = ctx;
  const {
    handleRestartServer,
  } = settings;
  return (
    <VideoStudioTab
      videoCatalog={videoState.videoCatalogWithLatest}
      residentChatModelGb={workspace.runtime.residentModelGb}
      hostSystem={workspace.system}
      selectedVideoModelId={videoState.selectedVideoModelId}
      onSelectedVideoModelIdChange={videoState.setSelectedVideoModelId}
      selectedVideoVariant={videoState.selectedVideoVariant}
      selectedVideoFamily={videoState.selectedVideoFamily}
      selectedVideoLoaded={videoState.selectedVideoLoaded}
      selectedVideoWillLoadOnGenerate={videoState.selectedVideoWillLoadOnGenerate}
      videoRuntimeLoadedDifferentModel={videoState.videoRuntimeLoadedDifferentModel}
      loadedVideoVariant={videoState.loadedVideoVariant}
      videoRuntimeStatus={videoState.videoRuntimeStatus}
      tauriBackend={tauriBackend}
      nativeBackends={nativeBackends}
      busy={busy}
      busyAction={busyAction}
      videoBusy={videoState.videoBusy}
      videoBusyLabel={videoState.videoBusyLabel}
      backendOnline={backendOnline}
      activeVideoDownloads={videoState.activeVideoDownloads}
      videoPrompt={videoState.videoPrompt}
      onVideoPromptChange={videoState.setVideoPrompt}
      videoNegativePrompt={videoState.videoNegativePrompt}
      videoStartImage={videoState.videoStartImage}
      onVideoStartImageChange={videoState.setVideoStartImage}
      videoEndImage={videoState.videoEndImage}
      onVideoEndImageChange={videoState.setVideoEndImage}
      imageGallery={imgState.imageOutputs}
      onVideoNegativePromptChange={videoState.setVideoNegativePrompt}
      videoUseRandomSeed={videoState.videoUseRandomSeed}
      onVideoUseRandomSeedChange={videoState.setVideoUseRandomSeed}
      videoSeedInput={videoState.videoSeedInput}
      onVideoSeedInputChange={videoState.setVideoSeedInput}
      videoWidth={videoState.videoWidth}
      onVideoWidthChange={videoState.setVideoWidth}
      videoHeight={videoState.videoHeight}
      onVideoHeightChange={videoState.setVideoHeight}
      videoNumFrames={videoState.videoNumFrames}
      onVideoNumFramesChange={videoState.setVideoNumFrames}
      videoFps={videoState.videoFps}
      onVideoFpsChange={videoState.setVideoFps}
      videoSteps={videoState.videoSteps}
      onVideoStepsChange={videoState.setVideoSteps}
      videoGuidance={videoState.videoGuidance}
      onVideoGuidanceChange={videoState.setVideoGuidance}
      videoUseNf4={videoState.videoUseNf4}
      onVideoUseNf4Change={videoState.setVideoUseNf4}
      videoEnableLtxRefiner={videoState.videoEnableLtxRefiner}
      onVideoEnableLtxRefinerChange={videoState.setVideoEnableLtxRefiner}
      videoCfgDecay={videoState.videoCfgDecay}
      onVideoCfgDecayChange={videoState.setVideoCfgDecay}
      videoPreviewVae={videoState.videoPreviewVae}
      onVideoPreviewVaeChange={videoState.setVideoPreviewVae}
      videoFp8LayerwiseCasting={videoState.videoFp8LayerwiseCasting}
      onVideoFp8LayerwiseCastingChange={videoState.setVideoFp8LayerwiseCasting}
      videoCacheStrategy={videoState.videoCacheStrategy}
      onVideoCacheStrategyChange={videoState.setVideoCacheStrategy}
      videoCacheRelL1Thresh={videoState.videoCacheRelL1Thresh}
      onVideoCacheRelL1ThreshChange={videoState.setVideoCacheRelL1Thresh}
      videoStgScale={videoState.videoStgScale}
      onVideoStgScaleChange={videoState.setVideoStgScale}
      videoFastPreview={videoState.videoFastPreview}
      onVideoFastPreviewChange={videoState.setVideoFastPreview}
      onActiveTabChange={setActiveTab}
      onPreloadVideoModel={(variant) => void videoState.handlePreloadVideoModel(variant)}
      onUnloadVideoModel={(variant) => void videoState.handleUnloadVideoModel(variant)}
      onVideoDownload={(repo, modelId) => void videoState.handleVideoDownload(repo, modelId)}
      onGenerateVideo={() => void videoState.handleVideoGenerate()}
      onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
      onRestartServer={() => void handleRestartServer()}
      onInstallVideoOutputDeps={(packages) => videoState.handleInstallVideoOutputDeps(packages)}
      onInstallVideoGpuRuntime={() => videoState.handleInstallVideoGpuRuntime()}
      onInstallCudaTorch={() => void handleInstallCudaTorch()}
      installingCudaTorch={installingCudaTorch}
      cudaTorchResult={cudaTorchRawResult}
      longLiveStatus={videoState.longLiveStatus}
      installingLongLive={videoState.installingLongLive}
      onRefreshLongLiveStatus={() => void videoState.refreshLongLiveStatus()}
      onInstallLongLive={() => videoState.handleInstallLongLive()}
      longLiveJob={videoState.longLiveJob}
      mlxVideoStatus={videoState.mlxVideoStatus}
      installingMlxVideo={videoState.installingMlxVideo}
      onRefreshMlxVideoStatus={() => void videoState.refreshMlxVideoStatus()}
      onInstallMlxVideo={() => videoState.handleInstallMlxVideo()}
      gpuBundleJob={videoState.gpuBundleJob}
    />
  );
}

export function VideoGalleryView({ ctx }: { ctx: AppTabContext }) {
  const {
    videoState,
    setActiveTab,
    handleRevealPath,
  } = ctx;
  return (
    <VideoGalleryTab
      videoOutputs={videoState.videoOutputs}
      imageOutputs={ctx.imgState.imageOutputs}
      videoBusy={videoState.videoBusy}
      onActiveTabChange={setActiveTab}
      onOpenVideoStudio={(modelId) => videoState.openVideoStudio(modelId)}
      onRevealPath={(path) => void handleRevealPath(path)}
      onDeleteVideoArtifact={(id) => void videoState.handleDeleteVideoOutput(id)}
    />
  );
}
