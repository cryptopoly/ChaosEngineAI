/**
 * The Mixer's own transport plays the rough cut (S9 follow-up): the takes in order with the Mixer's sound, from
 * the project files, with no render. The cut is rebuilt on every edit (files already loaded are kept) and leaves
 * the title and end cards out, so its clock is the timeline the Mixer draws. Loop starts it over at the end.
 */
import { useMemo, useRef, type MutableRefObject } from "react";
import type { StoryboardProject, StoryboardTimeline } from "../../../types";
import { buildRoughCut, type RoughCut } from "../preview/roughCutLogic";
import { useRoughCut, type UseRoughCutResult } from "../preview/useRoughCut";

export function useMixerRoughCut(
  project: StoryboardProject,
  timeline: StoryboardTimeline,
  loop: MutableRefObject<boolean>,
): { cut: RoughCut; player: UseRoughCutResult } {
  const cut = useMemo(() => buildRoughCut(project, timeline, { brand: false }), [project, timeline]);
  const self = useRef<UseRoughCutResult | null>(null);
  const player = useRoughCut(project.id, cut, {
    onEnded: () => {
      if (loop.current) setTimeout(() => self.current?.play(), 0);
    },
  });
  self.current = player;
  return { cut, player };
}
