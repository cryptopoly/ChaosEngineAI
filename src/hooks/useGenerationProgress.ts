import { useEffect, useRef, useState } from "react";
import {
  getBenchmarkProgress,
  getConversionProgress,
  getImageGenerationProgress,
  getMusicProgress,
  getVideoGenerationProgress,
} from "../api";
import type { GenerationProgressSnapshot } from "../types";

const POLL_INTERVAL_MS = 500;
/** Music runs for tens of seconds and its phases are coarse, so it polls once a second. */
const MUSIC_POLL_INTERVAL_MS = 1000;

/**
 * Polls the backend's image / video / music / benchmark / conversion progress endpoint while a job is
 * in flight. The poll only runs while ``active`` is true so we don't pester
 * the backend in the idle case — the modal flips ``active`` on as soon as it
 * fires the generation request, and off again after the response lands.
 *
 * Returns ``null`` whenever the backend reports ``active: false`` so the
 * LiveProgress component knows to fall back to its time-based estimates.
 */
export function useGenerationProgress(
  kind: "image" | "video" | "music" | "benchmark" | "conversion",
  active: boolean,
  /** Poll slower than the default, for a screen that only watches for a run it did not start. */
  intervalMs?: number,
): GenerationProgressSnapshot | null {
  const [snapshot, setSnapshot] = useState<GenerationProgressSnapshot | null>(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;
    if (!active) {
      // Reset so the *next* run starts from a clean slate rather than
      // briefly flashing the previous run's last step.
      setSnapshot(null);
      return;
    }

    const fetcher = {
      image: getImageGenerationProgress,
      video: getVideoGenerationProgress,
      music: getMusicProgress,
      benchmark: getBenchmarkProgress,
      conversion: getConversionProgress,
    }[kind];
    const interval = intervalMs ?? (kind === "music" ? MUSIC_POLL_INTERVAL_MS : POLL_INTERVAL_MS);
    let timer: number | null = null;

    const poll = async () => {
      if (cancelledRef.current) return;
      try {
        const next = await fetcher();
        if (cancelledRef.current) return;
        // A music request that waits behind another heavy job is not active yet but has a queue position to show.
        setSnapshot(next.active || (kind === "music" && next.queue?.waiting) ? next : null);
      } catch {
        // Transient network errors are common during a heavy generation —
        // the worker thread can pause Python's event loop just long enough
        // for a poll to fail. Silently keep the previous snapshot rather
        // than blanking the bar.
      }
      if (cancelledRef.current) return;
      timer = window.setTimeout(poll, interval);
    };

    void poll();

    return () => {
      cancelledRef.current = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [kind, active, intervalMs]);

  return snapshot;
}
