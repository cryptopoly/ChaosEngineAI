import { useEffect, useRef, useState } from "react";
import { LiveProgress, type LiveProgressPhase } from "./LiveProgress";
import { fetchVideoOutputBlobUrl } from "../api";
import { useGenerationProgress } from "../hooks/useGenerationProgress";
import { number, formatImageTimestamp } from "../utils";
import type { TabId, VideoModelVariant, VideoOutputArtifact } from "../types";

export interface VideoGenerationRunInfo {
  modelName: string;
  prompt: string;
  numFrames: number;
  fps: number;
  steps: number;
  needsPipelineLoad: boolean;
}

export interface VideoGenerationModalProps {
  showVideoGenerationModal: boolean;
  videoBusy: boolean;
  videoGenerationStartedAt: number | null;
  videoGenerationError: string | null;
  videoGenerationArtifact: VideoOutputArtifact | null;
  videoGenerationRunInfo: VideoGenerationRunInfo | null;
  selectedVideoVariant: VideoModelVariant | null;
  onShowVideoGenerationModalChange: (show: boolean) => void;
  onActiveTabChange: (tab: TabId) => void;
  onRevealPath: (path: string) => void;
  onDeleteArtifact: (artifactId: string) => void;
}

/**
 * Mirrors ImageGenerationModal: shows a LiveProgress bar driven by the
 * backend ProgressTracker while the diffusers pipeline runs, then swaps to
 * the rendered clip the moment the response lands.
 *
 * Video generation is the longest single operation in the app — easily
 * 60-300s on consumer hardware — so the progress bar matters more here than
 * in any other surface. The phases mirror the backend's ProgressTracker
 * states (loading / encoding / diffusing / decoding / saving) so the
 * realProgress signal can drive the bar in real time.
 */
export function VideoGenerationModal({
  showVideoGenerationModal,
  videoBusy,
  videoGenerationStartedAt,
  videoGenerationError,
  videoGenerationArtifact,
  videoGenerationRunInfo,
  selectedVideoVariant,
  onShowVideoGenerationModalChange,
  onActiveTabChange,
  onRevealPath,
  onDeleteArtifact,
}: VideoGenerationModalProps) {
  // Hook ordering rule: invoke even when hidden so React's render order
  // stays consistent across mounts/unmounts.
  const realProgress = useGenerationProgress("video", videoBusy && Boolean(videoGenerationStartedAt));
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoLoadError, setVideoLoadError] = useState<string | null>(null);
  const activeUrlRef = useRef<string | null>(null);

  // Fetch the mp4 blob whenever a fresh artifact lands. We can't point
  // <video src> directly at the auth-protected /file endpoint — the tag
  // doesn't send custom headers — so we mirror VideoGalleryTab's blob trick.
  useEffect(() => {
    if (!videoGenerationArtifact) {
      setVideoUrl(null);
      setVideoLoadError(null);
      if (activeUrlRef.current) {
        URL.revokeObjectURL(activeUrlRef.current);
        activeUrlRef.current = null;
      }
      return;
    }
    let cancelled = false;
    setVideoLoadError(null);
    fetchVideoOutputBlobUrl(videoGenerationArtifact.artifactId)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        if (activeUrlRef.current) URL.revokeObjectURL(activeUrlRef.current);
        activeUrlRef.current = url;
        setVideoUrl(url);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setVideoLoadError(err instanceof Error ? err.message : "Could not load video.");
      });
    return () => {
      cancelled = true;
    };
  }, [videoGenerationArtifact?.artifactId]);

  // Cleanup the blob URL on unmount so the browser can free the mp4 buffer.
  useEffect(() => {
    return () => {
      if (activeUrlRef.current) {
        URL.revokeObjectURL(activeUrlRef.current);
        activeUrlRef.current = null;
      }
    };
  }, []);

  if (!showVideoGenerationModal) {
    return null;
  }

  const runInfo = videoGenerationRunInfo;
  const steps = runInfo?.steps ?? 50;
  const numFrames = runInfo?.numFrames ?? 0;
  // Estimates are *fallback* — when the backend tracker publishes step counts,
  // LiveProgress drives the bar from the real signal. The estimates still set
  // the proportional widths of each phase block so diffusion doesn't fill the
  // entire bar in one second when an estimate is way off.
  // Video diffusion is heavier per-step than image: ~5-12s/step on Apple
  // Silicon, depending on resolution.
  const diffuseEstimate = Math.max(60, Math.round(steps * 8));
  const decodeEstimate = Math.max(15, Math.round(numFrames * 0.4));
  const videoPhases: LiveProgressPhase[] = [
    ...(runInfo?.needsPipelineLoad
      ? [{ id: "loading", label: "Loading model into memory", estimatedSeconds: 60 }]
      : []),
    { id: "encoding", label: "Encoding prompt", estimatedSeconds: 8 },
    {
      id: "diffusing",
      label: numFrames > 0 ? `Diffusing ${numFrames} frames` : "Diffusing frames",
      estimatedSeconds: diffuseEstimate,
    },
    { id: "decoding", label: "Encoding mp4", estimatedSeconds: decodeEstimate },
    { id: "saving", label: "Saving to gallery", estimatedSeconds: 4 },
  ];

  const artifact = videoGenerationArtifact;
  const clipSeconds = artifact
    ? artifact.clipDurationSeconds || artifact.numFrames / Math.max(1, artifact.fps)
    : 0;

  return (
    <div className="modal-overlay image-result-modal">
      <div className="modal-content" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h3>
            {videoBusy
              ? "Generating video"
              : videoGenerationError
                ? "Video generation failed"
                : "Video ready"}
          </h3>
          {!videoBusy && !videoGenerationError && artifact ? (
            <p>
              {artifact.modelName} · {formatImageTimestamp(artifact.createdAt)}
            </p>
          ) : null}
        </div>
        <div className="modal-body">
          {videoBusy && videoGenerationStartedAt ? (
            <LiveProgress
              title="Generating video"
              subtitle={runInfo?.modelName ?? selectedVideoVariant?.name ?? undefined}
              startedAt={videoGenerationStartedAt}
              accent="image"
              phases={videoPhases}
              realProgress={realProgress}
            />
          ) : videoGenerationError ? (
            <div className="callout error">
              <h3>Video generation failed</h3>
              <p>{videoGenerationError}</p>
              <p className="muted-text">
                Adjust the prompt, resolution, or frame count, then try again. The gallery keeps any earlier successful clips.
              </p>
            </div>
          ) : artifact ? (
            <div className="image-generation-result">
              <div className="image-generation-preview-shell">
                {videoUrl ? (
                  <video
                    className="image-generation-preview"
                    src={videoUrl}
                    controls
                    loop
                    muted
                    playsInline
                    autoPlay
                    preload="metadata"
                  />
                ) : (
                  <div
                    className="image-generation-preview"
                    style={{ display: "grid", placeItems: "center", minHeight: 220 }}
                  >
                    <span className="muted-text">{videoLoadError ?? "Loading clip..."}</span>
                  </div>
                )}
              </div>
              <div className="image-generation-info">
                <div className="chip-row">
                  <span className="badge success">Saved To Gallery</span>
                  {artifact.runtimeLabel ? <span className="badge subtle">{artifact.runtimeLabel}</span> : null}
                </div>
                <div>
                  <h3>{artifact.modelName}</h3>
                  <p className="image-output-prompt">{artifact.prompt}</p>
                  {artifact.runtimeNote ? <p className="muted-text">{artifact.runtimeNote}</p> : null}
                </div>
                <div className="image-output-meta">
                  <span>{artifact.width} x {artifact.height}</span>
                  <span>{artifact.numFrames} frames</span>
                  <span>{artifact.fps} fps</span>
                  <span>{number(clipSeconds)}s clip</span>
                  <span>{artifact.steps} steps</span>
                  <span>CFG {artifact.guidance}</span>
                  <span>Seed {artifact.seed}</span>
                  <span>{number(artifact.durationSeconds)}s render</span>
                </div>
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => {
                      onShowVideoGenerationModalChange(false);
                      onActiveTabChange("video-gallery");
                    }}
                  >
                    Open Gallery
                  </button>
                  {artifact.videoPath ? (
                    <button
                      className="secondary-button"
                      type="button"
                      onClick={() => onRevealPath(artifact.videoPath as string)}
                    >
                      Reveal File
                    </button>
                  ) : null}
                  <button
                    className="secondary-button danger-button"
                    type="button"
                    onClick={() => onDeleteArtifact(artifact.artifactId)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ) : null}
        </div>
        {!videoBusy ? (
          <div className="modal-footer">
            <button
              className="primary-button"
              type="button"
              onClick={() => onShowVideoGenerationModalChange(false)}
            >
              {videoGenerationError ? "Close" : "Done"}
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
