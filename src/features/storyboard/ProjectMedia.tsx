/**
 * Project media (stills and clips) served through the project files route. The route needs the API
 * token, so a file is fetched into an object URL that is revoked when the component goes away.
 */
import { useEffect, useState } from "react";
import { fetchStoryboardFileBlobUrl } from "../../api";

/** The object URL of a project file, or null while it loads or when it cannot be read. */
export function useProjectFileUrl(projectId: string, rel: string | null, version?: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!rel) {
      setUrl(null);
      return;
    }
    let cancelled = false;
    let made: string | null = null;
    setUrl(null);
    fetchStoryboardFileBlobUrl(projectId, rel)
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
  }, [projectId, rel, version]);
  return url;
}

export interface ProjectMediaProps {
  projectId: string;
  rel: string | null;
  version?: string | null;
  className?: string;
  alt?: string;
}

export function ProjectImage({ projectId, rel, version, className, alt = "" }: ProjectMediaProps) {
  const url = useProjectFileUrl(projectId, rel, version);
  if (!url) return <span className={`project-media-empty ${className ?? ""}`.trim()} aria-hidden="true" />;
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
