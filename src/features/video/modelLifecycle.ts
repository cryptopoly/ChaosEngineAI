/**
 * Video model preload + unload handlers.
 *
 * Pulled out of ``useVideoState`` so the hook stays focused on
 * orchestration. Both handlers narrow to (variant, deps) and return
 * void since the hook's busy-label / runtime-status state is the
 * only meaningful side effect.
 *
 * Extracted as part of the v0.8.0 Phase 2c-6 refactor.
 */

import { preloadVideoModel, unloadVideoModel } from "../../api";
import type { VideoModelVariant, VideoRuntimeStatus } from "../../types";


export async function handlePreloadVideoModel(
  variant: VideoModelVariant | null | undefined,
  deps: {
    setVideoBusyLabel: (label: string | null) => void;
    setVideoRuntimeStatus: (status: VideoRuntimeStatus) => void;
    setError: (msg: string | null) => void;
  },
): Promise<void> {
  const { setVideoBusyLabel, setVideoRuntimeStatus, setError } = deps;
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


export async function handleUnloadVideoModel(
  variant: VideoModelVariant | null | undefined,
  deps: {
    setVideoBusyLabel: (label: string | null) => void;
    setVideoRuntimeStatus: (status: VideoRuntimeStatus) => void;
    setError: (msg: string | null) => void;
    loadedVideoVariant: VideoModelVariant | null | undefined;
  },
): Promise<void> {
  const { setVideoBusyLabel, setVideoRuntimeStatus, setError, loadedVideoVariant } = deps;
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
