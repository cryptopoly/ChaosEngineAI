/**
 * The project's lip-sync coverage (`guide.coverage`). The guide block of the project payload (`given`) is the source;
 * only without one is it read from the backend whenever the project's revision changes, so the table outlives a tab
 * switch or a reload. A run's own answer can be shown at once with `show` until the payload catches up.
 */
import { useEffect, useState } from "react";
import { getStoryboardGuide } from "../../../api";
import type { StoryboardGuide, StoryboardLipSyncCoverage } from "../../../types";

export function useSongCoverage(projectId: string, revision: number, hasSong: boolean, given?: StoryboardGuide | null) {
  const [coverage, setCoverage] = useState<{ revision: number; value: StoryboardLipSyncCoverage | null } | null>(null);
  const skip = Boolean(given);

  useEffect(() => {
    if (!hasSong || skip) return;
    let live = true;
    getStoryboardGuide(projectId)
      .then((out) => {
        if (live) setCoverage({ revision: out.revision, value: out.guide.coverage });
      })
      .catch(() => undefined); // keep what is shown; the next revision asks again
    return () => {
      live = false;
    };
  }, [projectId, revision, hasSong, skip]);

  // A run's own answer (shown at once) is newer than the payload's block until the project reaches its revision.
  const ahead = coverage && coverage.revision > revision ? coverage : null;
  const shown = ahead ? ahead.value : given ? given.coverage : (coverage?.value ?? null);
  return {
    coverage: hasSong ? (shown ?? null) : null,
    show: (value: StoryboardLipSyncCoverage | null | undefined, at: number) => {
      if (value !== undefined) setCoverage((prev) => (prev && prev.revision > at ? prev : { revision: at, value }));
    },
  };
}
