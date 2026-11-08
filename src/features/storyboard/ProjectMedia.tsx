/**
 * Project media (stills and clips) served through the project files route. The route needs the API
 * token, so a file is fetched into an object URL that is revoked when the component goes away.
 */
import { useEffect, useRef, useState, type RefObject } from "react";
import { fetchStoryboardFileBlobUrl } from "../../api";
import type { StoryboardThumbSize } from "../../api/storyboard";

/**
 * The object URL of a project file, or null while it loads or when it cannot be read. With `thumb` a picture
 * comes as a small JPEG. With `lazy`, the fetch waits until `target` is near the screen, so a long grid does
 * not pull every file at once.
 */
export function useProjectFileUrl(
  projectId: string,
  rel: string | null,
  version?: string | null,
  options: { thumb?: StoryboardThumbSize; target?: RefObject<Element | null> } = {},
): string | null {
  const [url, setUrl] = useState<string | null>(null);
  const [near, setNear] = useState(!options.target);
  const { thumb, target } = options;
  useEffect(() => {
    const el = target?.current;
    if (!el || near || typeof IntersectionObserver === "undefined") {
      if (!near) setNear(true);
      return;
    }
    const seen = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setNear(true);
        seen.disconnect();
      }
    }, { rootMargin: "200px" });
    seen.observe(el);
    return () => seen.disconnect();
  }, [target, near]);
  useEffect(() => {
    if (!rel || !near) {
      setUrl(null);
      return;
    }
    let cancelled = false;
    let made: string | null = null;
    setUrl(null);
    fetchStoryboardFileBlobUrl(projectId, rel, thumb)
      .then((next) => {
        if (cancelled) URL.revokeObjectURL(next);
        else {
          made = next;
          setUrl(next);
        }
      })
      .catch(() => {
        if (!cancelled) setUrl(null);
      });
    return () => {
      cancelled = true;
      if (made) URL.revokeObjectURL(made);
    };
  }, [projectId, rel, version, thumb, near]);
  return url;
}

export interface ProjectMediaProps {
  projectId: string;
  rel: string | null;
  version?: string | null;
  className?: string;
  alt?: string;
  /** A small JPEG for grids, fetched when it comes near the screen. */
  thumb?: StoryboardThumbSize;
}

export function ProjectImage({ projectId, rel, version, className, alt = "", thumb }: ProjectMediaProps) {
  const holder = useRef<HTMLSpanElement>(null);
  const url = useProjectFileUrl(projectId, rel, version, thumb ? { thumb, target: holder } : {});
  if (!url) return <span ref={holder} className={`project-media-empty ${className ?? ""}`.trim()} aria-hidden="true" />;
  return <img className={className} src={url} alt={alt} loading="lazy" />;
}

/** A clip as a still frame (metadata only) or, with `controls`, a player. */
export function ProjectVideo({
  projectId,
  rel,
  version,
  className,
  controls,
  autoPlay,
}: ProjectMediaProps & { controls?: boolean; autoPlay?: boolean }) {
  const url = useProjectFileUrl(projectId, rel, version);
  if (!url) return <span className={`project-media-empty ${className ?? ""}`.trim()} aria-hidden="true" />;
  return (
    <video
      className={className}
      src={controls ? url : `${url}#t=0.1`}
      preload="metadata"
      muted={!controls}
      controls={controls}
      autoPlay={autoPlay}
      playsInline
    />
  );
}
