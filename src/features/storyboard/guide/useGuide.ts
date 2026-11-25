/**
 * The open project's guide block (stages, quality, coverage, failures, readiness, cast). Every project payload
 * carries it and the shared project hook keeps it (`story.guide`), so that is what this returns. Only when a payload
 * came without one (an older backend) does it read `GET .../guide` instead, whenever the hook adopts a new copy of
 * the project. The last answer stays on screen while the next one loads, and an answer for an older copy is dropped.
 */
import { useEffect, useState } from "react";
import { getStoryboardGuide } from "../../../api";
import type { StoryboardGuide, StoryboardProject } from "../../../types";

export function useGuide(project: StoryboardProject | null, enabled: boolean, given?: StoryboardGuide | null): StoryboardGuide | null {
  const [guide, setGuide] = useState<{ projectId: string; guide: StoryboardGuide } | null>(null);
  const skip = Boolean(given);
  useEffect(() => {
    if (!enabled || !project || skip) return;
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
  }, [project, enabled, skip]);
  if (given) return given;
  return guide && project && guide.projectId === project.id ? guide.guide : null;
}
