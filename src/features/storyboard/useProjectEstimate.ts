/**
 * The estimate for "Make missing" on the open project (D-65 phase 3), read again shortly after the project
 * changes (debounced), so a new shot or a picked take updates the run bar without a request per keystroke.
 */
import { useEffect, useState } from "react";
import { estimateStoryboardBatch } from "../../api";
import type { StoryboardEstimate } from "../../api/storyboard";

export const ESTIMATE_DEBOUNCE_MS = 400;

export function useProjectEstimate(
  projectId: string,
  revision: number,
  shotIds: string[],
  enabled: boolean,
): StoryboardEstimate | null {
  const [estimate, setEstimate] = useState<StoryboardEstimate | null>(null);
  const key = shotIds.join(",");
  useEffect(() => {
    if (!enabled || shotIds.length === 0) {
      setEstimate(null);
      return;
    }
    let live = true;
    const timer = setTimeout(() => {
      estimateStoryboardBatch(projectId, { what: "both", shotIds })
        .then((result) => live && setEstimate(result))
        .catch(() => live && setEstimate(null));
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [projectId, revision, key, enabled]); // shotIds is read through key
  return estimate;
}
