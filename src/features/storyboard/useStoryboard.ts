/**
 * State for the open storyboard project: the project and its timeline, edit ops that apply what the
 * server returns (no optimistic updates), undo / redo, and batch polling. A 409 from a stale revision
 * adopts the project the server sent and raises a short notice instead of an error.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  StoryboardApiError,
  cancelStoryboardBatch,
  getStoryboardBatch,
  getStoryboardProject,
  redoStoryboard,
  runStoryboardOp,
  startStoryboardBatch,
  undoStoryboard,
} from "../../api";
import type {
  StoryboardBatchRecord,
  StoryboardBatchWhat,
  StoryboardOpArgs,
  StoryboardOpName,
  StoryboardProject,
  StoryboardProjectPayload,
  StoryboardTimeline,
} from "../../types";
import { classifyFailure, isBatchRunning, readOpenProjectId, writeOpenProjectId } from "./storyboardLogic";

export const BATCH_POLL_MS = 1500;

export type StoryboardNotice = "conflict" | null;

export interface UseStoryboardResult {
  projectId: string | null;
  project: StoryboardProject | null;
  timeline: StoryboardTimeline | null;
  loading: boolean;
  /** Set when a request failed; the message is the server's own words. */
  error: string | null;
  /** True when the project's folder or file is gone (404 on load). */
  notFound: boolean;
  notice: StoryboardNotice;
  dismissNotice: () => void;
  /** True while an op, undo or redo is in flight. */
  busy: boolean;
  batch: StoryboardBatchRecord | null;
  open: (id: string | null) => void;
  reload: () => Promise<void>;
  op: <K extends StoryboardOpName>(name: K, args: StoryboardOpArgs[K]) => Promise<boolean>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  startBatch: (what: StoryboardBatchWhat, shotIds?: string[], force?: boolean) => Promise<boolean>;
  cancelBatch: () => Promise<void>;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function useStoryboard(enabled: boolean): UseStoryboardResult {
  const [projectId, setProjectId] = useState<string | null>(() => readOpenProjectId());
  const [project, setProject] = useState<StoryboardProject | null>(null);
  const [timeline, setTimeline] = useState<StoryboardTimeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [notice, setNotice] = useState<StoryboardNotice>(null);
  const [busy, setBusy] = useState(false);
  const [batch, setBatch] = useState<StoryboardBatchRecord | null>(null);

  const projectRef = useRef<StoryboardProject | null>(null);
  const idRef = useRef<string | null>(projectId);
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  idRef.current = projectId;

  const adopt = useCallback((payload: StoryboardProjectPayload) => {
    projectRef.current = payload.project;
    setProject(payload.project);
    setTimeline(payload.timeline);
    const latest = payload.project.batches[payload.project.batches.length - 1] ?? null;
    setBatch((current) => {
      if (isBatchRunning(latest)) return latest;
      return current && latest && current.id === latest.id ? latest : current;
    });
  }, []);

  const reload = useCallback(async () => {
    const id = idRef.current;
    if (!id) return;
    try {
      adopt(await getStoryboardProject(id));
      setError(null);
      setNotFound(false);
    } catch (err) {
      if (idRef.current !== id) return;
      if (err instanceof StoryboardApiError && err.status === 404) {
        setNotFound(true);
        projectRef.current = null;
        setProject(null);
        setTimeline(null);
      } else {
        setError(message(err));
      }
    }
  }, [adopt]);

  const open = useCallback((id: string | null) => {
    writeOpenProjectId(id);
    idRef.current = id;
    projectRef.current = null;
    setProjectId(id);
    setProject(null);
    setTimeline(null);
    setBatch(null);
    setError(null);
    setNotFound(false);
    setNotice(null);
  }, []);

  useEffect(() => {
    if (!enabled || !projectId) return;
    let cancelled = false;
    setLoading(true);
    void reload().finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [enabled, projectId, reload]);

  /** A stale revision adopts the server's project and raises the notice; any other failure is an error. */
  const handleFailure = useCallback(
    async (err: unknown) => {
      const outcome = classifyFailure(err);
      if (outcome.kind === "conflict") {
        projectRef.current = outcome.project;
        setProject(outcome.project);
        setNotice("conflict");
        await reload();
      } else {
        setError(outcome.message);
      }
    },
    [reload],
  );

  /** Run one change after the previous one finished, always against the newest revision. */
  const mutate = useCallback(
    (run: (id: string, revision: number) => Promise<StoryboardProjectPayload>): Promise<boolean> => {
      const task = queue.current.then(async () => {
        const id = idRef.current;
        const current = projectRef.current;
        if (!id || !current) return false;
        setBusy(true);
        try {
          adopt(await run(id, current.revision));
          setError(null);
          return true;
        } catch (err) {
          await handleFailure(err);
          return false;
        } finally {
          setBusy(false);
        }
      });
      queue.current = task.catch(() => false);
      return task;
    },
    [adopt, handleFailure],
  );

  const op = useCallback(
    <K extends StoryboardOpName>(name: K, args: StoryboardOpArgs[K]) =>
      mutate((id, revision) => runStoryboardOp(id, revision, name, args)),
    [mutate],
  );

  const undo = useCallback(async () => void (await mutate((id, rev) => undoStoryboard(id, rev))), [mutate]);
  const redo = useCallback(async () => void (await mutate((id, rev) => redoStoryboard(id, rev))), [mutate]);

  const startBatch = useCallback(
    async (what: StoryboardBatchWhat, shotIds?: string[], force = false) => {
      const id = idRef.current;
      const current = projectRef.current;
      if (!id || !current) return false;
      try {
        const started = await startStoryboardBatch(id, { revision: current.revision, what, shotIds, force });
        setBatch(started.batch);
        projectRef.current = { ...current, revision: started.revision };
        setError(null);
        return true;
      } catch (err) {
        await handleFailure(err);
        return false;
      }
    },
    [handleFailure],
  );

  const cancelBatch = useCallback(async () => {
    const id = idRef.current;
    if (!id || !batch) return;
    try {
      await cancelStoryboardBatch(id, batch.id);
    } catch (err) {
      setError(message(err));
    }
  }, [batch]);

  // Poll the batch while it runs; reload the project when its revision moved (new takes) and at the end.
  const batchId = batch?.id ?? null;
  const running = isBatchRunning(batch);
  useEffect(() => {
    if (!enabled || !projectId || !batchId || !running) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const status = await getStoryboardBatch(projectId, batchId);
        if (stopped) return;
        setBatch(status.batch);
        const moved = status.revision !== projectRef.current?.revision;
        if (moved || !isBatchRunning(status.batch)) await reload();
        if (!stopped && isBatchRunning(status.batch)) timer = setTimeout(() => void tick(), BATCH_POLL_MS);
      } catch (err) {
        if (!stopped) {
          setError(message(err));
          timer = setTimeout(() => void tick(), BATCH_POLL_MS * 2);
        }
      }
    };
    timer = setTimeout(() => void tick(), BATCH_POLL_MS);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [enabled, projectId, batchId, running, reload]);

  return {
    projectId,
    project,
    timeline,
    loading,
    error,
    notFound,
    notice,
    dismissNotice: () => setNotice(null),
    busy,
    batch,
    open,
    reload,
    op,
    undo,
    redo,
    startBatch,
    cancelBatch,
  };
}
