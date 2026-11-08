/**
 * State for the Mixer's renders: start a preview or an export, poll the render route while one runs,
 * cancel it, and keep the last finished preview and export. The backend runs one render at a time and
 * says so plainly when a second is started; that message is shown as it is.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  StoryboardRevisionConflict,
  cancelStoryboardRender,
  getStoryboardRender,
  startStoryboardRender,
} from "../../api";
import type { StoryboardRenderJob } from "../../types";
import { isMissingVideoTools } from "../videotools/videoToolsLogic";

export const RENDER_POLL_MS = 1000;

export interface UseStoryboardRenderResult {
  /** The newest job, running or finished. */
  job: StoryboardRenderJob | null;
  /** Epoch ms when the running job was first seen, for the elapsed clock. */
  startedAt: number;
  starting: boolean;
  error: string | null;
  /** The last start failed for want of ffmpeg: the screen shows the "Video tools are needed" prompt. */
  needsVideoTools: boolean;
  /** The last finished preview / export, kept while a newer job of the other kind runs. */
  preview: StoryboardRenderJob | null;
  exported: StoryboardRenderJob | null;
  /** Bumps whenever a preview finishes, so the player reloads the (same-named) file. */
  previewVersion: number;
  start: (revision: number, preview: boolean) => Promise<boolean>;
  cancel: () => Promise<void>;
  clearError: () => void;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function useStoryboardRender(
  projectId: string | null,
  enabled: boolean,
  /** Called with the revision of a stale start so the screen can reload the project. */
  onStale?: () => void,
): UseStoryboardRenderResult {
  const [job, setJob] = useState<StoryboardRenderJob | null>(null);
  const [startedAt, setStartedAt] = useState(() => Date.now());
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsVideoTools, setNeedsVideoTools] = useState(false);
  const [preview, setPreview] = useState<StoryboardRenderJob | null>(null);
  const [exported, setExported] = useState<StoryboardRenderJob | null>(null);
  const [previewVersion, setPreviewVersion] = useState(0);
  const seen = useRef<string | null>(null);
  const finished = useRef<string | null>(null);
  const staleRef = useRef(onStale);
  staleRef.current = onStale;

  /** Take a job from the server: remember its start, and keep the last finished one of each kind. */
  const adopt = useCallback((next: StoryboardRenderJob | null) => {
    setJob(next);
    if (!next) return;
    if (next.state === "running" && seen.current !== next.renderId) setStartedAt(Date.now());
    seen.current = next.renderId;
    if (next.state !== "done" || finished.current === next.renderId) return;
    finished.current = next.renderId;
    if (next.preview) {
      setPreview(next);
      setPreviewVersion((v) => v + 1);
    } else {
      setExported(next);
    }
  }, []);

  useEffect(() => {
    setJob(null);
    setPreview(null);
    setExported(null);
    setError(null);
    seen.current = null;
    finished.current = null;
    if (!enabled || !projectId) return;
    let cancelled = false;
    getStoryboardRender(projectId)
      .then((status) => {
        if (!cancelled) adopt(status.render);
      })
      .catch(() => {
        // no render history is not an error
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, projectId, adopt]);

  const running = job?.state === "running";
  useEffect(() => {
    if (!enabled || !projectId || !running) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const status = await getStoryboardRender(projectId);
        if (stopped) return;
        adopt(status.render);
        if (status.render?.state === "running") timer = setTimeout(() => void tick(), RENDER_POLL_MS);
      } catch (err) {
        if (stopped) return;
        setError(message(err));
        timer = setTimeout(() => void tick(), RENDER_POLL_MS * 3);
      }
    };
    timer = setTimeout(() => void tick(), RENDER_POLL_MS);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [enabled, projectId, running, adopt]);

  const start = useCallback(
    async (revision: number, isPreview: boolean) => {
      if (!projectId) return false;
      setStarting(true);
      setError(null);
      setNeedsVideoTools(false);
      try {
        const started = await startStoryboardRender(projectId, revision, isPreview);
        seen.current = null;
        adopt(started.render);
        return true;
      } catch (err) {
        if (err instanceof StoryboardRevisionConflict) staleRef.current?.();
        if (isMissingVideoTools(err)) setNeedsVideoTools(true);
        setError(message(err));
        return false;
      } finally {
        setStarting(false);
      }
    },
    [projectId, adopt],
  );

  const cancel = useCallback(async () => {
    if (!projectId) return;
    try {
      await cancelStoryboardRender(projectId);
      const status = await getStoryboardRender(projectId);
      adopt(status.render);
    } catch (err) {
      setError(message(err));
    }
  }, [projectId, adopt]);

  return {
    job,
    startedAt,
    starting,
    error,
    needsVideoTools,
    preview,
    exported,
    previewVersion,
    start,
    cancel,
    clearError: () => {
      setError(null);
      setNeedsVideoTools(false);
    },
  };
}
