/**
 * FU-191: Keep going for Studio chats. The run lives in the backend (it sends the nudges); this hook follows it
 * for the composer chip, starts, stops and resumes it, and reloads the thread when the backend added a nudge or
 * a reply the window has not streamed itself. Pure helpers are exported for vitest.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchKeepGoing, resumeKeepGoing, startKeepGoing, stopKeepGoing } from "../../api";
import type { KeepGoingGrant } from "../../api";
import type { KeepGoingGpuMode, KeepGoingRun } from "../../types";

export const KEEP_GOING_DEFAULT_CAP = 20;
export const KEEP_GOING_MAX_CAP = 100;
export const KEEP_GOING_DEFAULT_HOURS = 6;
export const KEEP_GOING_MAX_HOURS = 12;
export const KEEP_GOING_POLL_MS = 2000;
const RETRY_MS = 5000;

/** Stop codes the app has its own words for; anything else shows the backend's English reason. */
export const KEEP_GOING_STOP_CODES = [
  "complete",
  "complete_no_project",
  "cap_reached",
  "no_progress",
  "repeated_error",
  "context_full",
  "plan_declined",
  "plan_edit",
  "plan_unanswered",
  "approval_expired",
  "user_stopped",
  "user_message",
  "model_unavailable",
  "turn_failed",
  "app_restart",
  "studio_off",
  "gpu_reload_failed",
] as const;

export function isActive(run: KeepGoingRun | null | undefined): boolean {
  return Boolean(run && run.state !== "stopped");
}

export function clampCap(value: number): number {
  if (!Number.isFinite(value)) return KEEP_GOING_DEFAULT_CAP;
  return Math.max(1, Math.min(KEEP_GOING_MAX_CAP, Math.round(value)));
}

export function clampHours(value: number): number {
  if (!Number.isFinite(value)) return KEEP_GOING_DEFAULT_HOURS;
  return Math.max(1, Math.min(KEEP_GOING_MAX_HOURS, Math.round(value)));
}

/** Hours and minutes left of a grant, or null when unknown. */
export function formatTimeLeft(seconds: number | null | undefined): { hours: number; minutes: number } | null {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return null;
  const minutes = Math.floor(seconds / 60);
  return { hours: Math.floor(minutes / 60), minutes: minutes % 60 };
}

/** The backend saved messages this window does not show (a nudge, or a reply to one) and it is not streaming
 * this chat itself: reload the thread. */
export function needsRefresh(run: KeepGoingRun | null, localCount: number, ownStreaming: boolean): boolean {
  return Boolean(run) && !ownStreaming && run!.messageCount !== localCount;
}

/** The line a screen reader hears when the run changes (state, phase or nudge number). */
export function announcementKey(run: KeepGoingRun | null): string {
  if (!run) return "";
  return `${run.runId}:${run.state}:${run.phase}:${run.nudges}:${run.stopCode ?? ""}`;
}

/** A nudge's text without its first line ("Keep going (automatic message 3 of 20 ...)."): the thread shows
 * that as the bubble's label instead. */
export function nudgeBody(text: string): string {
  const lines = text.split("\n");
  return lines.length > 1 && /^Keep going \(/.test(lines[0]) ? lines.slice(1).join("\n") : text;
}

/**
 * Read one chat's run now, then every 2 s while it is active (5 s after a failed read, e.g. while the backend
 * restarts). A stopped run, or none, is read once. Returns the stop function.
 */
export function followKeepGoing(
  sessionId: string,
  onRun: (run: KeepGoingRun | null) => void,
  fetchRun: (sessionId: string) => Promise<KeepGoingRun | null> = fetchKeepGoing,
): () => void {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const poll = async () => {
    try {
      const next = await fetchRun(sessionId);
      if (stopped) return;
      onRun(next);
      if (isActive(next)) timer = setTimeout(() => void poll(), KEEP_GOING_POLL_MS);
    } catch {
      if (!stopped) timer = setTimeout(() => void poll(), RETRY_MS);
    }
  };
  void poll();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
  };
}

/** A key that changes once per thread change the backend reports, so the window reloads once for each. */
export function refreshKey(run: KeepGoingRun): string {
  return `${run.runId}:${run.messageCount}:${run.turnRunning}`;
}

export interface KeepGoingControls {
  run: KeepGoingRun | null;
  busy: boolean;
  error: string | null;
  /** ``forSession``: a chat made a moment ago, before this hook has its id. */
  start: (cap: number, grant: KeepGoingGrant, hours: number, forSession?: string, gpuMode?: KeepGoingGpuMode) => Promise<void>;
  stop: () => Promise<void>;
  resume: (grant: KeepGoingGrant, hours: number, cap?: number) => Promise<void>;
  studioOff: () => Promise<void>;
}

/**
 * Follow one chat's run (``followKeepGoing``) and act on it. Start is idempotent in the backend, so a double
 * click or a reloaded page never makes two runs.
 */
export function useKeepGoing(
  sessionId: string | null | undefined,
  localCount: number,
  ownStreaming: boolean,
  onRefresh: (sessionId: string) => void,
  api = { fetchKeepGoing, startKeepGoing, stopKeepGoing, resumeKeepGoing },
): KeepGoingControls {
  const [run, setRun] = useState<KeepGoingRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const refreshRef = useRef(onRefresh);
  const refreshedAt = useRef<string>("");
  useEffect(() => {
    refreshRef.current = onRefresh;
  }, [onRefresh]);

  const active = isActive(run);
  useEffect(() => {
    setRun(null);
    setError(null);
    refreshedAt.current = "";
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    return followKeepGoing(sessionId, setRun, api.fetchKeepGoing);
    // `active` restarts polling when a stopped run is started or resumed from this window.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, active]);

  useEffect(() => {
    if (!sessionId || !run || run.sessionId !== sessionId) return;
    if (!needsRefresh(run, localCount, ownStreaming)) return;
    const key = refreshKey(run);
    if (refreshedAt.current === key) return;  // one reload per change the backend reports
    refreshedAt.current = key;
    refreshRef.current(sessionId);
  }, [run, localCount, ownStreaming, sessionId]);

  const act = useCallback(
    async (call: () => Promise<KeepGoingRun | null>) => {
      setBusy(true);
      setError(null);
      try {
        setRun(await call());
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  return {
    run: run && run.sessionId === sessionId ? run : null,
    busy,
    error,
    start: (cap, grant, hours, forSession, gpuMode = "auto") => {
      const id = forSession ?? sessionId;
      return id ? act(() => api.startKeepGoing(id, { cap: clampCap(cap), grant, hours: clampHours(hours), gpuMode })) : Promise.resolve();
    },
    stop: () => (sessionId ? act(() => api.stopKeepGoing(sessionId)) : Promise.resolve()),
    resume: (grant, hours, cap) =>
      sessionId
        ? act(() => api.resumeKeepGoing(sessionId, { grant, hours: clampHours(hours), ...(cap ? { cap: clampCap(cap) } : {}) }))
        : Promise.resolve(),
    studioOff: () => (sessionId ? act(() => api.stopKeepGoing(sessionId, true)) : Promise.resolve()),
  };
}
