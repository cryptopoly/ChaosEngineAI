/**
 * Waveform peaks for one project file. Results are kept for the session, keyed by project, file, a
 * key that changes when the file does (its hash) and the bucket count, so a lane of clips and a
 * redraw never ask the server twice. A failure leaves `null`: the clip draws a flat bar.
 */
import { useEffect, useState } from "react";
import { getStoryboardPeaks } from "../../../api";
import type { StoryboardPeaks } from "../../../types";

const cache = new Map<string, StoryboardPeaks>();
const inflight = new Map<string, Promise<StoryboardPeaks>>();

export function peaksKey(projectId: string, rel: string, version: string | null, buckets: number): string {
  return `${projectId}|${rel}|${version ?? ""}|${buckets}`;
}

/** Test hook: forget everything fetched so far. */
export function clearPeaksCache(): void {
  cache.clear();
  inflight.clear();
}

export function loadPeaks(projectId: string, rel: string, version: string | null, buckets: number): Promise<StoryboardPeaks> {
  const key = peaksKey(projectId, rel, version, buckets);
  const known = cache.get(key);
  if (known) return Promise.resolve(known);
  let pending = inflight.get(key);
  if (!pending) {
    pending = getStoryboardPeaks(projectId, rel, buckets)
      .then((result) => {
        cache.set(key, result);
        return result;
      })
      .finally(() => inflight.delete(key));
    inflight.set(key, pending);
  }
  return pending;
}

export interface PeaksState {
  peaks: StoryboardPeaks | null;
  loading: boolean;
  failed: boolean;
}

export function usePeaks(projectId: string, rel: string | null, version: string | null, buckets: number): PeaksState {
  const key = rel ? peaksKey(projectId, rel, version, buckets) : null;
  const [state, setState] = useState<PeaksState>(() => ({
    peaks: key ? (cache.get(key) ?? null) : null,
    loading: !!key && !cache.has(key),
    failed: false,
  }));

  useEffect(() => {
    if (!rel || !key) {
      setState({ peaks: null, loading: false, failed: false });
      return;
    }
    const known = cache.get(key);
    if (known) {
      setState({ peaks: known, loading: false, failed: false });
      return;
    }
    let cancelled = false;
    setState({ peaks: null, loading: true, failed: false });
    loadPeaks(projectId, rel, version, buckets)
      .then((result) => {
        if (!cancelled) setState({ peaks: result, loading: false, failed: false });
      })
      .catch(() => {
        if (!cancelled) setState({ peaks: null, loading: false, failed: true });
      });
    return () => {
      cancelled = true;
    };
  }, [key, projectId, rel, version, buckets]);

  return state;
}
