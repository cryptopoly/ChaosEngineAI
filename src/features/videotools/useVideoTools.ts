/**
 * State for the video tools (ffmpeg): the status, the one install action, the optional pinned download and
 * choosing a file. The install uses the existing Setup pip route for `imageio-ffmpeg` on every system; the
 * download is a background job that is polled.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelFfmpegDownload,
  chooseFfmpegFile,
  getFfmpegJob,
  getFfmpegStatus,
  installPipPackage,
  removeFfmpegDownload,
  startFfmpegDownload,
  type FfmpegJob,
  type FfmpegStatus,
} from "../../api";
import { lastLine } from "./videoToolsLogic";

const POLL_MS = 800;

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export interface UseVideoToolsResult {
  status: FfmpegStatus | null;
  busy: boolean;
  percent: number | null;
  error: string | null;
  refresh: () => Promise<FfmpegStatus | null>;
  /** The one-button install (imageio-ffmpeg through Setup's pip route). */
  install: () => Promise<boolean>;
  /** The pinned LGPL download (Windows and Linux). */
  download: () => Promise<boolean>;
  cancel: () => Promise<void>;
  chooseFile: (path: string) => Promise<boolean>;
  remove: () => Promise<void>;
  clearError: () => void;
}

export function useVideoTools(enabled: boolean, onChange?: () => void): UseVideoToolsResult {
  const [status, setStatus] = useState<FfmpegStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [percent, setPercent] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);
  const changed = useRef(onChange);
  changed.current = onChange;

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await getFfmpegStatus();
      if (alive.current) setStatus(next);
      return next;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    if (enabled) void refresh();
  }, [enabled, refresh]);

  const pollJob = useCallback(async (first: FfmpegJob): Promise<FfmpegJob> => {
    let job = first;
    while (alive.current && job.running) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      job = (await getFfmpegJob()).job;
      setPercent(job.percent);
    }
    return job;
  }, []);

  const finish = useCallback(
    async (ok: boolean, failure: string | null) => {
      const next = await refresh();
      if (ok && next?.installed) {
        changed.current?.();
        return true;
      }
      setError(failure ?? "Video tools did not install.");
      return false;
    },
    [refresh],
  );

  const run = useCallback(
    async (work: () => Promise<boolean>) => {
      setBusy(true);
      setError(null);
      setPercent(null);
      try {
        return await work();
      } catch (err) {
        setError(message(err));
        return false;
      } finally {
        if (alive.current) {
          setBusy(false);
          setPercent(null);
        }
      }
    },
    [],
  );

  const install = useCallback(
    () =>
      run(async () => {
        const result = await installPipPackage("imageio-ffmpeg");
        return await finish(result.ok, result.ok ? null : lastLine(result.output));
      }),
    [run, finish],
  );

  const download = useCallback(
    () =>
      run(async () => {
        const job = await pollJob((await startFfmpegDownload()).job);
        return await finish(job.done, job.error);
      }),
    [run, pollJob, finish],
  );

  const chooseFile = useCallback(
    (path: string) =>
      run(async () => {
        const result = await chooseFfmpegFile(path);
        const job = await pollJob(result.job);
        return await finish(result.version !== null || job.done, job.error);
      }),
    [run, pollJob, finish],
  );

  const cancel = useCallback(async () => {
    try {
      await cancelFfmpegDownload();
    } catch {
      // the job ends on its own
    }
  }, []);

  const remove = useCallback(async () => {
    try {
      const next = await removeFfmpegDownload();
      if (alive.current) setStatus(next);
    } catch (err) {
      setError(message(err));
    }
  }, []);

  return { status, busy, percent, error, refresh, install, download, cancel, chooseFile, remove, clearError: () => setError(null) };
}
