/**
 * Estimates for one shot's actions in the shot drawer: Make this shot, and Redo clip only / Redo still only (S11).
 * Read again shortly after the project changes (debounced). A kind that cannot be priced comes back with
 * `seconds: null` (and `missing` naming the model to install); a failed request leaves that kind null.
 */
import { useEffect, useState } from "react";
import { estimateStoryboardBatch } from "../../api";
import type { StoryboardEstimate } from "../../api/storyboard";
import type { StoryboardBatchWhat } from "../../types";
import { ESTIMATE_DEBOUNCE_MS } from "./useProjectEstimate";

export type ShotEstimates = Partial<Record<StoryboardBatchWhat, StoryboardEstimate | null>>;

export function useShotEstimates(
  projectId: string,
  revision: number,
  shotId: string,
  kinds: StoryboardBatchWhat[],
  enabled: boolean,
): ShotEstimates {
  const [estimates, setEstimates] = useState<ShotEstimates>({});
  const key = kinds.join(",");
  useEffect(() => {
    if (!enabled || kinds.length === 0) {
      setEstimates({});
      return;
    }
    let live = true;
    const timer = setTimeout(() => {
      void Promise.all(
        kinds.map((what) =>
          estimateStoryboardBatch(projectId, { what, shotIds: [shotId] })
            .then((result) => [what, result] as const)
            .catch(() => [what, null] as const),
        ),
      ).then((pairs) => live && setEstimates(Object.fromEntries(pairs)));
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [projectId, revision, shotId, key, enabled]); // kinds is read through key
  return estimates;
}
