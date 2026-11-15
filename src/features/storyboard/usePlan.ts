/**
 * Hooks for the Plan tab (S8): the plan status (the loaded chat model, how the style reaches the still model),
 * refreshed when the project, its still model or the loaded chat model changes, and a seconds counter for a
 * model call in flight.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { getStoryboardPlanStatus, type StoryboardPlanStatus } from "../../api/storyboardPlan";

export function usePlanStatus(projectId: string, refreshKey: string, enabled = true) {
  const [status, setStatus] = useState<StoryboardPlanStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = useRef(true);
  const load = useCallback(async () => {
    try {
      const next = await getStoryboardPlanStatus(projectId);
      if (!live.current) return;
      setStatus(next);
      setError(null);
    } catch (err) {
      if (live.current) setError(err instanceof Error ? err.message : String(err));
    }
  }, [projectId]);
  useEffect(() => {
    live.current = true;
    if (enabled) void load();
    return () => {
      live.current = false;
    };
  }, [load, refreshKey, enabled]);
  return { status, error, reload: load };
}

/** Whole seconds since `active` turned true; 0 while false. */
export function useElapsed(active: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!active) {
      setSeconds(0);
      return;
    }
    const started = Date.now();
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return seconds;
}

/** The message and code of a failed plan call. */
export function failureOf(err: unknown): { message: string; code: string | null } {
  const code = err && typeof err === "object" && "code" in err ? ((err as { code: unknown }).code as string | null) : null;
  return { message: err instanceof Error ? err.message : String(err), code: typeof code === "string" ? code : null };
}
