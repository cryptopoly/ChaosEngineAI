import { useEffect, useRef, useState } from "react";
import { getImageGenerationProgress, getVideoGenerationProgress } from "../api";
import type { GenerationProgressSnapshot } from "../types";

const POLL_INTERVAL_MS = 500;

/**
 * Polls the backend's image / video progress endpoint while a generation is
 * in flight. The poll only runs while ``active`` is true so we don't pester
 * the backend in the idle case — the modal flips ``active`` on as soon as it
 * fires the generation request, and off again after the response lands.
 *
 * Returns ``null`` whenever the backend reports ``active: false`` so the
 * LiveProgress component knows to fall back to its time-based estimates.
 */
export function useGenerationProgress(
  kind: "image" | "video",
  active: boolean,
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

    const fetcher = kind === "image" ? getImageGenerationProgress : getVideoGenerationProgress;
    let timer: number | null = null;

    const poll = async () => {
      if (cancelledRef.current) return;
      try {
        const next = await fetcher();
        if (cancelledRef.current) return;
        setSnapshot(next.active ? next : null);
      } catch {
        // Transient network errors are common during a heavy generation —
        // the worker thread can pause Python's event loop just long enough
        // for a poll to fail. Silently keep the previous snapshot rather
        // than blanking the bar.
      }
      if (cancelledRef.current) return;
      timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    };

    void poll();

    return () => {
      cancelledRef.current = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [kind, active]);

  return snapshot;
}
