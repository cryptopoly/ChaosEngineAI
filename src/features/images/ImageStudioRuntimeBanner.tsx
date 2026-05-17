/**
 * Top callout in ImageStudio — runtime status, chip row, model
 * preload/unload row, and the GPU runtime install action stack.
 *
 * Pulled out of ``ImageStudioTab.tsx`` as part of the v0.8.0
 * Phase 2d-2b refactor.
 */

import { useTranslation } from "react-i18next";
import { CudaTorchLogPanel } from "../../components/CudaTorchLogPanel";
import { InstallLogPanel } from "../../components/InstallLogPanel";
import { TorchUpgradePill } from "../../components/TorchUpgradePill";
import type {
  CudaTorchInstallResult,
  GpuBundleJobState,
} from "../../api";
import type { ImageModelVariant, ImageRuntimeStatus } from "../../types";
import type { NativeBackendStatus } from "../../types/server";
import { ImageStudioBoosters } from "./ImageStudioBoosters";


export interface ImageStudioRuntimeBannerProps {
  imageRuntimeStatus: ImageRuntimeStatus;
  selectedImageVariant: ImageModelVariant | null;
  selectedImageLoaded: boolean;
  selectedImageWillLoadOnGenerate: boolean;
  imageRuntimeLoadedDifferentModel: boolean;
  loadedImageVariant: ImageModelVariant | null;
  busy: boolean;
  busyAction: string | null;
  imageBusy: boolean;
  backendOnline: boolean;
  onRestartServer: () => void;
  onPreloadImageModel: (variant: ImageModelVariant) => void;
  onUnloadImageModel: (variant?: ImageModelVariant) => void;
  onInstallCudaTorch?: () => void;
  installingCudaTorch?: boolean;
  cudaTorchResult?: CudaTorchInstallResult | null;
  installingImageRuntime: boolean;
  gpuBundleJob: GpuBundleJobState | null;
  onInstallImageRuntime: () => void;
  /** FU-056 Phase 3: capability snapshot for the "Performance
   * boosters" sub-section. Optional — collapses to the "available"
   * card state if the backend hasn't probed yet. */
  nativeBackends?: NativeBackendStatus;
}


export function ImageStudioRuntimeBanner(props: ImageStudioRuntimeBannerProps) {
  const { t } = useTranslation("studio");
  const {
    imageRuntimeStatus,
    selectedImageVariant,
    selectedImageLoaded,
    selectedImageWillLoadOnGenerate,
    imageRuntimeLoadedDifferentModel,
    loadedImageVariant,
    busy,
    busyAction,
    imageBusy,
    backendOnline,
    onRestartServer,
    onPreloadImageModel,
    onUnloadImageModel,
    onInstallCudaTorch,
    installingCudaTorch,
    cudaTorchResult,
    installingImageRuntime,
    gpuBundleJob,
    onInstallImageRuntime,
    nativeBackends,
  } = props;

  return (
    <div className="callout image-callout image-runtime-callout">
      {/* torchInstallWarning is the loudest signal -- e.g. +cpu torch
        * wheel on a CUDA host -- so render it as a banner above the
        * chip row. Without this, "Real local generation available" +
        * "Device: cuda (expected)" would still light up green while
        * the user's NVIDIA GPU sits idle and generation runs on CPU
        * at 1/100th speed. */}
      {/* Three states for this slot, all in ONE callout to keep
        * the panel uncluttered (no stacked banners):
        *   (a) install just succeeded but backend still has the
        *       old torch in its module cache -> show "Restart
        *       Backend to activate" with a single primary button
        *   (b) torchInstallWarning is set (the +cpu wheel case
        *       and friends) -> show the warning + Install CUDA
        *       torch button + collapsible log panel
        *   (c) neither -> render nothing (the chip row below
        *       still announces engine / device state)
        *
        * State (a) takes priority because once a successful
        * install lands, the warning is meaningless until the
        * backend reloads -- showing both at once just confuses. */}
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
      ) : imageRuntimeStatus.torchInstallWarning ? (
        <div className="callout error" style={{ marginBottom: "0.6rem" }}>
          <strong>GPU acceleration not active.</strong>{" "}
          {imageRuntimeStatus.torchInstallWarning}
          {/* Inline remedy button + collapsible log. Only renders
            * when the warning is the "+cpu wheel" case (text
            * mentions "Install CUDA torch"); for "torch missing
            * entirely" the larger Install GPU runtime flow below
            * is the right remedy. */}
          {onInstallCudaTorch
            && imageRuntimeStatus.torchInstallWarning.includes("Install CUDA torch") ? (
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
      <div className="chip-row">
        <span className={`badge ${imageRuntimeStatus.realGenerationAvailable ? "success" : "warning"}`}>
          {imageRuntimeStatus.realGenerationAvailable
            ? "Real local generation available"
            : imageRuntimeStatus.activeEngine === "unavailable"
              ? "Runtime unavailable"
              : "Using placeholder outputs"}
        </span>
        {imageRuntimeStatus.torchInstallWarning ? (
          <span className="badge danger" title={imageRuntimeStatus.torchInstallWarning}>
            CPU fallback
          </span>
        ) : null}
        <span className="badge muted">Engine: {imageRuntimeStatus.activeEngine}</span>
        {/* Prefer the actual-loaded device; fall back to the
          * predicted expectedDevice computed cheaply via
          * nvidia-smi + find_spec (no torch import). When
          * nothing is loaded yet the badge reads 'Device: cuda
          * (expected)' which tells the user what will happen on
          * first Generate instead of leaving them to guess. */}
        {(() => {
          const resolved =
            imageRuntimeStatus.device
            ?? (imageRuntimeStatus.expectedDevice
              ? `${imageRuntimeStatus.expectedDevice} (expected)`
              : null);
          return resolved ? <span className="badge muted">Device: {resolved}</span> : null;
        })()}
      </div>
      {/* Torch upgrade nudge — renders only when real generation is
        * working (so the user's stable torch install isn't being
        * second-guessed) AND a newer wheel is on the matching cu{N}
        * index. The pill probes the backend itself on mount; this
        * banner just plugs in the restart-backend callback. */}
      {imageRuntimeStatus.realGenerationAvailable ? (
        <TorchUpgradePill
          backendOnline={backendOnline}
          onRestartBackend={onRestartServer}
          busy={busy}
        />
      ) : null}
      {/* FU-056 Phase 3: per-model accelerator install affordances.
        * Renders nothing when no accelerators apply to the variant
        * (SD1.5 / SDXL / non-DiT) or when real generation isn't
        * available yet (no point installing FLUX accelerators on a
        * box that can't even run FLUX). */}
      {imageRuntimeStatus.realGenerationAvailable ? (
        <ImageStudioBoosters
          selectedVariant={selectedImageVariant}
          nativeBackends={nativeBackends}
        />
      ) : null}
      {selectedImageVariant && imageRuntimeStatus.realGenerationAvailable ? (
        <div className="image-runtime-summary">
          <p className="muted-text">
            {selectedImageLoaded
              ? `${selectedImageVariant.name} is loaded and ready to generate.`
              : imageRuntimeLoadedDifferentModel && loadedImageVariant
                ? `${loadedImageVariant.name} is loaded. Generating with ${selectedImageVariant.name} will swap the pipeline.`
                : selectedImageWillLoadOnGenerate
                  ? `${selectedImageVariant.name} is installed locally but not loaded yet. The first generate will take longer while the diffusion pipeline warms up.`
                  : !selectedImageVariant.availableLocally
                    ? `${selectedImageVariant.name} is not installed locally. Download it from Discover to enable local generation.`
                    : "Model will load on demand when you generate."}
          </p>
          {imageBusy && selectedImageWillLoadOnGenerate ? (
            <p className="busy-indicator"><span className="busy-dot" />{t("runtimeBanner.loadingModel", { defaultValue: "Loading model into memory..." })}</p>
          ) : null}
          {(selectedImageVariant.availableLocally || loadedImageVariant) ? (
            <div className="button-row image-runtime-control-row">
              {selectedImageVariant.availableLocally && !selectedImageLoaded ? (
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => onPreloadImageModel(selectedImageVariant)}
                  disabled={imageBusy || busy || !backendOnline}
                >
                  Preload Model
                </button>
              ) : null}
              {selectedImageLoaded ? (
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => onUnloadImageModel(selectedImageVariant)}
                  disabled={imageBusy || busy || !backendOnline}
                >
                  Unload Model
                </button>
              ) : null}
              {!selectedImageLoaded && loadedImageVariant ? (
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => onUnloadImageModel()}
                  disabled={imageBusy || busy || !backendOnline}
                >
                  Unload {loadedImageVariant.name}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
      {!imageRuntimeStatus.realGenerationAvailable ? (
        <>
          <div className="image-runtime-actions">
            {/* Two display modes for the same not-available state:
              * (a) Fresh / broken install → offer Install GPU runtime.
              * (b) Install just completed this session but backend
              *     hasn't been restarted yet → PYTHONPATH in the
              *     running backend is frozen from spawn time, so
              *     find_spec still can't see the freshly-installed
              *     torch. The user doesn't need to install again,
              *     they need to restart. Keep the restart button
              *     prominent and explain why. */}
            {gpuBundleJob?.phase === "done" && gpuBundleJob.requiresRestart ? (
              <>
                <p className="muted-text">
                  GPU runtime installed to{" "}
                  <code>{gpuBundleJob.targetDir ?? "extras"}</code>. The running backend
                  still has its old import cache — click Restart Backend to activate the
                  new runtime, then image generation will use your GPU.
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
                  Install the GPU image runtime (torch + diffusers + accelerate + transformers,
                  ~2.5 GB) to enable real local generation. Writes to a persistent user-local
                  folder so app updates don't wipe it.
                </p>
                <div className="button-row">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => onInstallImageRuntime()}
                    disabled={installingImageRuntime || !backendOnline}
                  >
                    {installingImageRuntime ? "Installing..." : "Install GPU runtime"}
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
  );
}
