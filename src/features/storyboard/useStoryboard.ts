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
  pauseStoryboardBatch,
  redoStoryboard,
  resumeStoryboardBatch,
  runStoryboardOp,
  startStoryboardBatch,
  undoStoryboard,
} from "../../api";
import type {
  JobProgress,
  StoryboardBatchRecord,
  StoryboardBatchWhat,
  StoryboardOpArgs,
  StoryboardOpName,
  StoryboardPausedProgress,
  StoryboardProject,
  StoryboardProjectPayload,
  StoryboardTimeline,
} from "../../types";
import { createBusyGate, type BusyGate } from "./busyGate";
import { classifyFailure, isBatchPaused, isBatchRunning } from "./storyboardLogic";
import { createUnsavedTextStore, type UnsavedTextStore } from "./unsavedText";

export const BATCH_POLL_MS = 1500;

export type StoryboardNotice = "conflict" | null;

/** One edit op against the open project; resolves true when it landed. */
export type RunOp = <K extends StoryboardOpName>(name: K, args: StoryboardOpArgs[K]) => Promise<boolean>;

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
  /** Live progress of the running batch (step, elapsed, time left), or a paused batch's steps left; null when none. */
  batchProgress?: JobProgress | StoryboardPausedProgress | null;
  /** True from Pause until the batch stops after its current step (S-M2). */
  pausing?: boolean;
  reload: () => Promise<void>;
  op: <K extends StoryboardOpName>(name: K, args: StoryboardOpArgs[K]) => Promise<boolean>;
  /**
   * Like `op`, but the arguments are worked out from the newest project once the earlier changes have
   * landed (so relative edits add up). Returning null skips the request.
   */
  opFrom: <K extends StoryboardOpName>(
    name: K,
    build: (project: StoryboardProject) => StoryboardOpArgs[K] | null,
  ) => Promise<boolean>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  /** `draft`: Make drafts (the fast still model at a smaller size); `what: "final"` redoes the kept shots. */
  startBatch: (what: StoryboardBatchWhat, shotIds?: string[], force?: boolean, draft?: boolean) => Promise<boolean>;
  cancelBatch: () => Promise<void>;
  /** Stop the running batch after the step in progress; it keeps its remaining steps (S-M1 route). */
  pauseBatch?: () => Promise<void>;
  /** Carry on with the paused batch from its remaining steps. */
  resumeBatch?: () => Promise<boolean>;
  /**
   * Run a project route that answers with the project (attach a song, lyric captions) in the same queue as the
   * ops, against the newest revision; a stale revision adopts the server's project like an op (resolves false).
   * Any other failure rejects, so the caller shows it where it was asked.
   */
  call?: (run: (projectId: string, revision: number) => Promise<StoryboardProjectPayload>) => Promise<boolean>;
  /** Typed text whose save did not land, per field, for this project (S-O, `useUnsavedText`). */
  unsaved?: UnsavedTextStore;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** State for one project. The project shell is keyed by the id, so a different project gets a fresh hook. */
export function useStoryboard(enabled: boolean, projectId: string | null): UseStoryboardResult {
  const [project, setProject] = useState<StoryboardProject | null>(null);
  const [timeline, setTimeline] = useState<StoryboardTimeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [notice, setNotice] = useState<StoryboardNotice>(null);
  const [busy, setBusy] = useState(false);
  // A blur-save made by a mouse press shows busy only after that press's click, so the click is not lost (busyGate.ts).
  const busyGate = useRef<BusyGate | null>(null);
  useEffect(() => {
    const gate = createBusyGate(setBusy);
    busyGate.current = gate;
    return () => {
      gate.dispose();
      busyGate.current = null;
    };
  }, []);
  const [batch, setBatch] = useState<StoryboardBatchRecord | null>(null);
  const [batchProgress, setBatchProgress] = useState<JobProgress | StoryboardPausedProgress | null>(null);
  const [pausing, setPausing] = useState(false);
  const [unsaved] = useState(createUnsavedTextStore);

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
      // A running batch, or one paused (by Pause or a restart) that can still carry on: the bar shows it.
      if (isBatchRunning(latest) || isBatchPaused(latest)) return latest;
      return current && latest && current.id === latest.id ? latest : current;
    });
  }, []);

  const reload = useCallback(async () => {
    const id = idRef.current;
    if (!id) return;
    try {
      const payload = await getStoryboardProject(id);
      if (idRef.current !== id) return;
      adopt(payload);
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
    (
      run: (id: string, revision: number, project: StoryboardProject) => Promise<StoryboardProjectPayload> | null,
      rethrow = false,
    ): Promise<boolean> => {
      const task = queue.current.then(async () => {
        const id = idRef.current;
        const current = projectRef.current;
        if (!id || !current) return false;
        const request = run(id, current.revision, current);
        if (!request) return false;
        const gate = busyGate.current;
        if (gate) gate.begin();
        else setBusy(true);
        try {
          adopt(await request);
          setError(null);
          return true;
        } catch (err) {
          // `call` hands every failure but a stale revision back to its caller, to show where it was asked.
          const kind = classifyFailure(err).kind;
          if (rethrow && kind !== "conflict") throw err;
          // Read by a field whose save this was, right after the change resolves false (useUnsavedText).
          unsaved.lastFailure = kind === "conflict" ? "conflict" : "failed";
          await handleFailure(err);
          return false;
        } finally {
          if (gate) gate.end();
          else setBusy(false);
        }
      });
      queue.current = task.catch(() => false);
      return task;
    },
    [adopt, handleFailure, unsaved],
  );

  const op = useCallback(
    <K extends StoryboardOpName>(name: K, args: StoryboardOpArgs[K]) =>
      mutate((id, revision) => runStoryboardOp(id, revision, name, args)),
    [mutate],
  );

  const opFrom = useCallback(
    <K extends StoryboardOpName>(name: K, build: (project: StoryboardProject) => StoryboardOpArgs[K] | null) =>
      mutate((id, revision, project) => {
        const args = build(project);
        return args ? runStoryboardOp(id, revision, name, args) : null;
      }),
    [mutate],
  );

  const undo = useCallback(async () => void (await mutate((id, rev) => undoStoryboard(id, rev))), [mutate]);
  const redo = useCallback(async () => void (await mutate((id, rev) => redoStoryboard(id, rev))), [mutate]);

  const startBatch = useCallback(
    async (what: StoryboardBatchWhat, shotIds?: string[], force = false, draft = false) => {
      const id = idRef.current;
      const current = projectRef.current;
      if (!id || !current) return false;
      try {
        const started = await startStoryboardBatch(id, { revision: current.revision, what, shotIds, force, draft });
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
      // Nothing polls a paused batch: read the cancelled record and the project back now.
      if (isBatchPaused(batch)) {
        setBatchProgress(null);
        await reload();
      }
    } catch (err) {
      setError(message(err));
    }
  }, [batch, reload]);

  const pauseBatch = useCallback(async () => {
    const id = idRef.current;
    if (!id || !batch || !isBatchRunning(batch)) return;
    try {
      await pauseStoryboardBatch(id, batch.id);
      setPausing(true);
    } catch (err) {
      setError(message(err));
    }
  }, [batch]);

  const resumeBatch = useCallback(async () => {
    const id = idRef.current;
    const current = projectRef.current;
    if (!id || !current || !batch || !isBatchPaused(batch)) return false;
    try {
      const resumed = await resumeStoryboardBatch(id, batch.id, current.revision);
      setBatch(resumed.batch);
      setBatchProgress(null);
      projectRef.current = { ...current, revision: resumed.revision };
      setError(null);
      return true;
    } catch (err) {
      await handleFailure(err);
      return false;
    }
  }, [batch, handleFailure]);

  const call = useCallback(
    (run: (projectId: string, revision: number) => Promise<StoryboardProjectPayload>) => mutate((id, revision) => run(id, revision), true),
    [mutate],
  );

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
        const live = isBatchRunning(status.batch);
        setBatchProgress(live || isBatchPaused(status.batch) ? status.progress ?? null : null);
        if (!live) setPausing(false);
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

  // A paused batch is not polled: read its bar data (steps left, time left) once.
  const paused = isBatchPaused(batch);
  useEffect(() => {
    if (!enabled || !projectId || !batchId || !paused) return;
    let stopped = false;
    getStoryboardBatch(projectId, batchId)
      .then((status) => {
        if (!stopped && isBatchPaused(status.batch)) setBatchProgress(status.progress ?? null);
      })
      .catch(() => undefined);
    return () => {
      stopped = true;
    };
  }, [enabled, projectId, batchId, paused]);

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
    batchProgress,
    pausing,
    reload,
    op,
    opFrom,
    undo,
    redo,
    startBatch,
    cancelBatch,
    pauseBatch,
    resumeBatch,
    call,
    unsaved,
  };
}
