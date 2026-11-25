/**
 * The before/after media of a finished upscale job: the first frame of the input and of the output (PNG), and
 * the film itself once the person presses Play. The backend answers 404 once it has restarted after the job
 * (its work folder is removed); a URL then stays null and the card falls back to its placeholder.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchVideoUpscaleFileUrl, fetchVideoUpscaleFrameUrl, type VideoUpscaleJob } from "../../api";
import { NO_MEDIA, type UpscaleMediaView } from "./upscaleLogic";

export function useUpscaleMedia(job: Pick<VideoUpscaleJob, "id" | "state"> | null): UpscaleMediaView & { play: () => void } {
  const jobId = job && job.state === "done" ? job.id : null;
  const [frames, setFrames] = useState<{ input: string | null; output: string | null; loading: boolean }>({ input: null, output: null, loading: false });
  const [film, setFilm] = useState<Pick<UpscaleMediaView, "playUrl" | "playState" | "playError">>({ playUrl: null, playState: "idle", playError: null });

  useEffect(() => {
    setFrames({ input: null, output: null, loading: Boolean(jobId) });
    if (!jobId) return undefined;
    let cancelled = false;
    const made: string[] = [];
    const take = (side: "input" | "output") =>
      fetchVideoUpscaleFrameUrl(jobId, side)
        .then((url) => {
          if (url && cancelled) URL.revokeObjectURL(url);
          else if (url) made.push(url);
          return cancelled ? null : url;
        })
        .catch(() => null);
    Promise.all([take("input"), take("output")]).then(([input, output]) => {
      if (!cancelled) setFrames({ input, output, loading: false });
    });
    return () => {
      cancelled = true;
      made.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [jobId]);

  // The film is fetched on request only: it can be gigabytes.
  const current = useRef<string | null>(jobId);
  useEffect(() => {
    current.current = jobId;
    setFilm({ playUrl: null, playState: "idle", playError: null });
    return () => {
      current.current = null;
    };
  }, [jobId]);
  const playUrl = film.playUrl;
  useEffect(() => () => {
    if (playUrl) URL.revokeObjectURL(playUrl);
  }, [playUrl]);

  const play = useCallback(() => {
    if (!jobId) return;
    setFilm({ playUrl: null, playState: "loading", playError: null });
    fetchVideoUpscaleFileUrl(jobId)
      .then((url) => {
        if (current.current !== jobId) {
          if (url) URL.revokeObjectURL(url); // the person moved to another job meanwhile
          return;
        }
        setFilm(url ? { playUrl: url, playState: "idle", playError: null } : { playUrl: null, playState: "missing", playError: null });
      })
      .catch((err: unknown) => {
        if (current.current === jobId) setFilm({ playUrl: null, playState: "error", playError: err instanceof Error ? err.message : String(err) });
      });
  }, [jobId]);

  return jobId ? { inputFrameUrl: frames.input, outputFrameUrl: frames.output, loading: frames.loading, ...film, play } : { ...NO_MEDIA, play };
}
