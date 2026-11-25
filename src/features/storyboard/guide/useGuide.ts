/**
 * The open project's guide block (stages, quality, coverage, failures, readiness, cast). Every project payload
 * carries it, but the shared project hook keeps only the project and the timeline, so this reads
 * `GET .../guide` again whenever the hook adopts a new copy of the project (an op, an undo, a batch step). The last
 * answer stays on screen while the next one loads, and an answer for an older copy is dropped.
 */
import { useEffect, useState } from "react";
import { getStoryboardGuide } from "../../../api/storyboard";
import type { StoryboardGuide, StoryboardProject } from "../../../types";

export function useGuide(project: StoryboardProject | null, enabled: boolean): StoryboardGuide | null {
  const [guide, setGuide] = useState<{ projectId: string; guide: StoryboardGuide } | null>(null);
  useEffect(() => {
    if (!enabled || !project) return;
    let live = true;
    getStoryboardGuide(project.id)
      .then((answer) => {
        if (live) setGuide({ projectId: project.id, guide: answer.guide });
      })
      .catch(() => {
        // The rail keeps its last answer; the tabs work without it.
      });
    return () => {
      live = false;
    };
  }, [project, enabled]);
  return guide && project && guide.projectId === project.id ? guide.guide : null;
}
