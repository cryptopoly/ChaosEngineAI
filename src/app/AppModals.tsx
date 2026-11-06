import type { Dispatch, SetStateAction } from "react";
import { LaunchModal } from "../components/LaunchModal";
import { ImageGenerationModal } from "../components/ImageGenerationModal";
import { VideoGenerationModal } from "../components/VideoGenerationModal";
import type { AppTabContext } from "./context";
import type { PendingLaunch } from "./launchActions";

/** Model launch picker (Chat / Server / new thread). */
export function LaunchModalHost({
  ctx,
  pendingLaunch,
  launchModelSearch,
  setPendingLaunch,
  setLaunchModelSearch,
  confirmLaunch,
}: {
  ctx: AppTabContext;
  pendingLaunch: PendingLaunch | null;
  launchModelSearch: string;
  setPendingLaunch: Dispatch<SetStateAction<PendingLaunch | null>>;
  setLaunchModelSearch: (value: string) => void;
  confirmLaunch: (key: string) => Promise<void>;
}) {
  const {
    workspace, libraryChatOptions, threadModelOptions, isAppleSilicon, mtplxJob, installingMtplx,
    handleInstallMtplx, tensorfoldControls,
  } = ctx;
  const { launchSettings, preview, installingPackage, installLogs, updateLaunchSetting, handleInstallPackage } = ctx.settings;
  return (
    <LaunchModal
      pendingLaunch={pendingLaunch}
      launchModelSearch={launchModelSearch}
      libraryChatOptions={libraryChatOptions}
      threadModelOptions={threadModelOptions}
      launchSettings={launchSettings}
      preview={preview}
      availableMemoryGb={workspace.system.availableMemoryGb}
      totalMemoryGb={workspace.system.totalMemoryGb}
      gpuVramTotalGb={workspace.system.gpuVramTotalGb}
      availableCacheStrategies={workspace.system.availableCacheStrategies}
      dflashInfo={workspace.system.dflash}
      installingPackage={installingPackage}
      installLogs={installLogs}
      turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
      mtplxSystemInfo={workspace.system.mtplx}
      onInstallMtplx={() => void handleInstallMtplx()}
      installingMtplx={installingMtplx}
      mtplxJob={mtplxJob}
      tensorfold={tensorfoldControls}
      isAppleSilicon={isAppleSilicon}
      onPendingLaunchChange={setPendingLaunch}
      onLaunchModelSearchChange={setLaunchModelSearch}
      onLaunchSettingChange={updateLaunchSetting}
      onConfirmLaunch={(key) => void confirmLaunch(key)}
      onInstallPackage={handleInstallPackage}
    />
  );
}

/** Image and video generation progress / result dialogs. */
export function GenerationModals({ ctx }: { ctx: AppTabContext }) {
  const { imgState, videoState, setActiveTab, handleOpenExternalUrl, handleRevealPath } = ctx;
  return (
    <>
      <ImageGenerationModal
        showImageGenerationModal={imgState.showImageGenerationModal}
        imageBusy={imgState.imageBusy}
        imageGenerationStartedAt={imgState.imageGenerationStartedAt}
        imageGenerationError={imgState.imageGenerationError}
        imageGenerationCancelled={imgState.imageGenerationCancelled}
        imageGenerationCancelling={imgState.imageGenerationCancelling}
        imageGenerationArtifacts={imgState.imageGenerationArtifacts}
        selectedImageGenerationArtifact={imgState.selectedImageGenerationArtifact}
        imageGenerationRunInfo={imgState.imageGenerationRunInfo}
        imageCatalog={imgState.imageCatalog}
        imageSteps={imgState.imageSteps}
        selectedImageVariant={imgState.selectedImageVariant}
        onShowImageGenerationModalChange={imgState.setShowImageGenerationModal}
        onSelectedImageGenerationArtifactIdChange={imgState.setSelectedImageGenerationArtifactId}
        onActiveTabChange={setActiveTab}
        onUseSameSettings={imgState.handleUseSameImageSettings}
        onVarySeed={(a) => void imgState.handleVaryImageSeed(a)}
        onOpenExternalUrl={(url) => void handleOpenExternalUrl(url)}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteArtifact={(id) => void imgState.handleDeleteImageArtifact(id)}
        onCancelGeneration={() => void imgState.handleCancelImageGeneration()}
      />
      <VideoGenerationModal
        showVideoGenerationModal={videoState.showVideoGenerationModal}
        videoBusy={videoState.videoBusy}
        videoGenerationStartedAt={videoState.videoGenerationStartedAt}
        videoGenerationError={videoState.videoGenerationError}
        videoGenerationCancelled={videoState.videoGenerationCancelled}
        videoGenerationCancelling={videoState.videoGenerationCancelling}
        videoGenerationArtifact={videoState.videoGenerationArtifact}
        videoGenerationRunInfo={videoState.videoGenerationRunInfo}
        selectedVideoVariant={videoState.selectedVideoVariant}
        onShowVideoGenerationModalChange={videoState.setShowVideoGenerationModal}
        onActiveTabChange={setActiveTab}
        onRevealPath={(path) => void handleRevealPath(path)}
        onDeleteArtifact={(id) => void videoState.handleDeleteVideoOutput(id)}
        onCancelGeneration={() => void videoState.handleCancelVideoGeneration()}
      />
    </>
  );
}
