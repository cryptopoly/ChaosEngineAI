import { useEffect, useRef, useState } from "react";
import { Panel } from "../../components/Panel";
import { fetchVideoOutputBlobUrl } from "../../api";
import type { TabId, VideoOutputArtifact } from "../../types";
import { formatImageTimestamp, number } from "../../utils";

export interface VideoGalleryTabProps {
  videoOutputs: VideoOutputArtifact[];
  videoBusy: boolean;
  onActiveTabChange: (tab: TabId) => void;
  onOpenVideoStudio: (modelId?: string) => void;
  onRevealPath: (path: string) => void;
  onDeleteVideoArtifact: (artifactId: string) => void;
}

export function VideoGalleryTab({
  videoOutputs,
  videoBusy,
  onActiveTabChange,
  onOpenVideoStudio,
  onRevealPath,
  onDeleteVideoArtifact,
}: VideoGalleryTabProps) {
  const [expandedArtifactId, setExpandedArtifactId] = useState<string | null>(null);
  const [autoPlayVideos, setAutoPlayVideos] = useState(true);
  const expandedArtifact = videoOutputs.find((artifact) => artifact.artifactId === expandedArtifactId) ?? null;
  const displayedArtifacts = expandedArtifact ? [expandedArtifact] : videoOutputs;

  useEffect(() => {
    if (expandedArtifactId && !videoOutputs.some((artifact) => artifact.artifactId === expandedArtifactId)) {
      setExpandedArtifactId(null);
    }
  }, [expandedArtifactId, videoOutputs]);

  return (
    <div className="content-grid image-page-grid">
      <Panel
        title="Video Gallery"
        subtitle={
          videoOutputs.length > 0
            ? `${videoOutputs.length} saved clip${videoOutputs.length === 1 ? "" : "s"}`
            : "Saved video renders"
        }
        className="span-2"
        actions={
          <div className="button-row video-gallery-actions">
            <label className="video-gallery-autoplay-toggle">
              <input
                type="checkbox"
                checked={autoPlayVideos}
                onChange={(event) => setAutoPlayVideos(event.target.checked)}
              />
              <span>Auto-Play</span>
            </label>
            <button className="secondary-button" type="button" onClick={() => onOpenVideoStudio()}>
              Open Studio
            </button>
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-models")}>
              Installed Models
            </button>
          </div>
        }
      >
        {videoOutputs.length === 0 ? (
          <div className="empty-state image-empty-state">
            <p>
              No video outputs yet. Load a model in the Studio and generate a clip — finished renders will
              land here.
            </p>
            <div className="button-row" style={{ marginTop: 12 }}>
              <button className="secondary-button" type="button" onClick={() => onOpenVideoStudio()}>
                Open Video Studio
              </button>
            </div>
          </div>
        ) : (
          <div className={expandedArtifact ? "video-gallery-expanded" : "image-gallery-grid"}>
            {displayedArtifacts.map((artifact) => (
              <VideoOutputCard
                key={artifact.artifactId}
                artifact={artifact}
                videoBusy={videoBusy}
                expanded={expandedArtifactId === artifact.artifactId}
                autoPlay={autoPlayVideos}
                onToggleExpanded={(artifactId) => {
                  setExpandedArtifactId((current) => (current === artifactId ? null : artifactId));
                }}
                onRevealPath={onRevealPath}
                onDelete={onDeleteVideoArtifact}
              />
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

interface VideoOutputCardProps {
  artifact: VideoOutputArtifact;
  videoBusy: boolean;
  expanded: boolean;
  autoPlay: boolean;
  onToggleExpanded: (artifactId: string) => void;
  onRevealPath: (path: string) => void;
  onDelete: (artifactId: string) => void;
}

/**
 * Fetches the mp4 as an authenticated blob and streams it via an object URL.
 * We can't just hit /file directly as a plain <video src=...> because the
 * backend requires the x-chaosengine-token header — the <video> tag won't
 * send it. Fetching as a blob + createObjectURL sidesteps that cleanly and
 * still lets the browser seek / buffer normally.
 */
function VideoOutputCard({
  artifact,
  videoBusy,
  expanded,
  autoPlay,
  onToggleExpanded,
  onRevealPath,
  onDelete,
}: VideoOutputCardProps) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const activeUrlRef = useRef<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoadError(null);
    fetchVideoOutputBlobUrl(artifact.artifactId)
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
        setLoadError(err instanceof Error ? err.message : "Could not load video.");
      });
    return () => {
      cancelled = true;
      if (activeUrlRef.current) {
        URL.revokeObjectURL(activeUrlRef.current);
        activeUrlRef.current = null;
      }
    };
  }, [artifact.artifactId]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    if (!autoPlay) {
      video.pause();
      return;
    }
    video.muted = true;
    void video.play().catch(() => {
      // Browser autoplay rules can still refuse playback in edge cases.
    });
  }, [autoPlay, videoUrl]);

  const clipSeconds = artifact.clipDurationSeconds || artifact.numFrames / Math.max(1, artifact.fps);
  const previewAspect = artifact.width > 0 && artifact.height > 0 ? `${artifact.width} / ${artifact.height}` : undefined;
  return (
    <article className={`image-output-card video-output-card${expanded ? " video-output-card--expanded" : ""}`}>
      {expanded ? (
        <button
          className="secondary-button video-gallery-collapse-button"
          type="button"
          onClick={() => onToggleExpanded(artifact.artifactId)}
        >
          Collapse
        </button>
      ) : null}
      {videoUrl ? (
        <video
          ref={videoRef}
          className="image-output-preview video-output-preview"
          src={videoUrl}
          controls
          loop
          muted
          playsInline
          autoPlay={autoPlay}
          preload="metadata"
          style={previewAspect ? { aspectRatio: previewAspect } : undefined}
        />
      ) : (
        <div
          className="image-output-preview video-output-preview"
          style={{
            aspectRatio: previewAspect,
            display: "grid",
            placeItems: "center",
            minHeight: 160,
          }}
        >
          <span className="muted-text">{loadError ?? "Loading clip..."}</span>
        </div>
      )}
      <div className="image-output-card-body">
        <div className="image-output-card-head">
          <strong>{artifact.modelName}</strong>
          <span className="badge muted">{formatImageTimestamp(artifact.createdAt)}</span>
        </div>
        {artifact.runtimeLabel ? (
          <div className="chip-row">
            <span className="badge subtle">{artifact.runtimeLabel}</span>
            <span className="badge muted">
              {artifact.width} × {artifact.height}
            </span>
          </div>
        ) : null}
        <p className="image-output-prompt">{artifact.prompt}</p>
        {artifact.runtimeNote ? <p className="muted-text">{artifact.runtimeNote}</p> : null}
        <div className="image-output-meta">
          <span>{artifact.numFrames} frames</span>
          <span>{artifact.fps} fps</span>
          <span>{number(clipSeconds)}s clip</span>
          <span>{artifact.steps} steps</span>
          <span>CFG {artifact.guidance}</span>
          <span>Seed {artifact.seed}</span>
          <span>{number(artifact.durationSeconds)}s render</span>
        </div>
        <div className="button-row">
          {!expanded ? (
            <button
              className="secondary-button"
              type="button"
              onClick={() => onToggleExpanded(artifact.artifactId)}
            >
              Expand
            </button>
          ) : null}
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
            className="secondary-button"
            type="button"
            disabled={videoBusy}
            onClick={() => onDelete(artifact.artifactId)}
          >
            Delete
          </button>
        </div>
      </div>
    </article>
  );
}
