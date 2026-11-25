/**
 * The stage rail's rules (audit changes 1 and 9), without React so they are unit-tested directly. The backend
 * decides each stage's status, sentence and action code (`guide.stages`, `backend_service/storyboard/stages.py`);
 * this file decides only what the app does with them: which stages show, which one is current, how each one
 * reads, and where each action code goes (one of the project's tabs, or the batch the Storyboard's run bar would
 * start).
 */
import type {
  StoryboardBatchWhat,
  StoryboardGuide,
  StoryboardGuideAction,
  StoryboardProject,
  StoryboardStage,
  StoryboardStageId,
  StoryboardStages,
} from "../../../types";
import type { ProjectTabId } from "../projectRoute";
import { finalPassShotIds, missingShotIds } from "../storyboardLogic";

/** The batches a stage button can start: the same ones the Storyboard's run bar and shot cards start. */
export type StageBatch = "drafts" | "missing" | "final" | "stale" | "failed";

/** What a stage's one button does. */
export type StageDestination =
  | { kind: "tab"; tab: ProjectTabId }
  | { kind: "batch"; batch: StageBatch; tab: ProjectTabId }
  | { kind: "resume"; tab: ProjectTabId };

/**
 * Where each action code goes. The song, its cut plan and lip sync live in the Plan tab's Song section, so every
 * song and lip-sync code (and the coverage table's fixes) opens Plan; the check before export is on the Export tab.
 */
export function destinationOf(action: StoryboardGuideAction): StageDestination {
  switch (action) {
    case "attach-song":
    case "cut-plan":
    case "plan-shots":
    case "assign-vocals":
    case "make-lipsync":
    case "open-coverage":
    case "real-speed":
    case "unlock":
    case "lengthen":
    case "shorten":
      return { kind: "tab", tab: "plan" };
    case "open-cast":
      return { kind: "tab", tab: "cast" };
    case "open-storyboard":
    case "watch-batch":
    case "choose-final":
      return { kind: "tab", tab: "storyboard" };
    case "resume-batch":
      return { kind: "resume", tab: "storyboard" };
    case "retry-failed":
      return { kind: "batch", batch: "failed", tab: "storyboard" };
    case "make-drafts":
      return { kind: "batch", batch: "drafts", tab: "storyboard" };
    case "make-missing":
    case "make-clips":
      return { kind: "batch", batch: "missing", tab: "storyboard" };
    case "make-final":
      return { kind: "batch", batch: "final", tab: "storyboard" };
    case "remake-stale":
      return { kind: "batch", batch: "stale", tab: "storyboard" };
    case "open-mixer":
      return { kind: "tab", tab: "mixer" };
    case "open-check":
    case "export":
      return { kind: "tab", tab: "export" };
    default:
      return { kind: "tab", tab: "overview" };
  }
}

/** The actions whose work is in the Plan tab's Song section, below the brief: the button scrolls to it. */
export function opensSongSection(action: StoryboardGuideAction): boolean {
  return action === "attach-song" || action === "cut-plan" || action === "assign-vocals" || action === "make-lipsync" || action === "open-coverage";
}

/**
 * Bring the Song section into view and put focus on its heading. The tab it lives in renders after the click, so
 * this looks for the heading for a few frames. Returns a function that stops the search.
 */
export function focusSongSection(doc: Document = document, frames = 20): () => void {
  let left = frames;
  let handle = 0;
  const look = () => {
    const heading = doc.getElementById("song-panel-title");
    if (heading) {
      heading.focus();
      heading.scrollIntoView?.({ block: "start" });
      return;
    }
    if (--left > 0) handle = requestAnimationFrame(look);
  };
  handle = requestAnimationFrame(look);
  return () => cancelAnimationFrame(handle);
}

/** The arguments of `story.startBatch` for a stage batch, or null when it has no shot to work on. */
export interface StageBatchArgs {
  what: StoryboardBatchWhat;
  shotIds: string[];
  force: boolean;
  draft: boolean;
}

export function batchArgs(
  batch: StageBatch,
  project: Pick<StoryboardProject, "shots">,
  guide: Pick<StoryboardGuide, "quality"> | null | undefined,
): StageBatchArgs | null {
  const shots = project.shots;
  const unlocked = (ids: Set<string>) => shots.filter((s) => ids.has(s.id) && !s.locked).map((s) => s.id);
  let out: StageBatchArgs;
  switch (batch) {
    case "drafts":
      // As "Make drafts" in the run bar: draft stills of the missing shots.
      out = { what: "still", shotIds: missingShotIds(shots), force: false, draft: true };
      break;
    case "missing":
      out = { what: "both", shotIds: missingShotIds(shots), force: false, draft: false };
      break;
    case "final": {
      // "Final pass on kept shots"; with nothing kept and nothing made yet, the shots at full size.
      const kept = finalPassShotIds(shots);
      out = kept.length > 0
        ? { what: "final", shotIds: kept, force: false, draft: false }
        : { what: "both", shotIds: missingShotIds(shots), force: false, draft: false };
      break;
    }
    case "stale": {
      const stale = new Set([...shots.filter((s) => s.status === "stale").map((s) => s.id), ...(guide?.quality.staleShots ?? [])]);
      out = { what: "both", shotIds: unlocked(stale), force: false, draft: false };
      break;
    }
    case "failed":
      // As the shot card's "Try again".
      out = { what: "both", shotIds: unlocked(new Set(shots.filter((s) => s.status === "failed").map((s) => s.id))), force: false, draft: false };
      break;
  }
  return out.shotIds.length > 0 ? out : null;
}

/** How a stage reads on the rail. `skipped`: optional, never started, and the work has moved past it. A current
 *  stage that needs attention reads as `attention` (the rail still marks it current). */
export type StageLook = "done" | "current" | "attention" | "progress" | "skipped" | "todo";

export interface RailStage {
  stage: StoryboardStage;
  look: StageLook;
  current: boolean;
  /** 1-based place on the rail (hidden stages do not count). */
  number: number;
}

export interface RailOptions {
  /** The project was started as a music video: the Song stage shows (and leads) before a song is attached. */
  musicVideo?: boolean;
}

/**
 * The stages the rail shows. A stage whose `applies` is false is hidden (no song: Song and Lip sync), so a film
 * without music reads Look, Shots, Drafts, Final, Sound, Check, Export. A music video start shows Song and Lip sync
 * before the song is attached.
 */
export function visibleStages(stages: StoryboardStage[], options: RailOptions = {}): StoryboardStage[] {
  return stages.filter((stage) => stage.applies || (options.musicVideo === true && (stage.id === "song" || stage.id === "lipsync")));
}

/** The current stage: the backend's, except that a music video start leads with Song until a song is attached. */
export function currentStage(block: StoryboardStages, options: RailOptions = {}): StoryboardStageId | null {
  if (options.musicVideo) {
    const song = block.stages.find((stage) => stage.id === "song");
    if (song && song.status !== "done") return "song";
  }
  return block.current;
}

export function railStages(block: StoryboardStages, options: RailOptions = {}): RailStage[] {
  const shown = visibleStages(block.stages, options);
  const current = currentStage(block, options);
  const currentIndex = current ? shown.findIndex((stage) => stage.id === current) : shown.length;
  return shown.map((stage, index) => {
    const isCurrent = stage.id === current;
    let look: StageLook;
    if (stage.status === "done") look = "done";
    else if (stage.status === "needs-attention") look = "attention";
    else if (isCurrent) look = "current";
    else if (stage.status === "in-progress") look = "progress";
    else if (stage.optional && currentIndex >= 0 && index < currentIndex) look = "skipped";
    else look = "todo";
    return { stage, look, current: isCurrent, number: index + 1 };
  });
}

/** The stage whose sentence and button show under the rail: the one picked, else the current one, else the last. */
export function shownStage(rail: RailStage[], picked: StoryboardStageId | null): RailStage | null {
  return rail.find((item) => item.stage.id === picked) ?? rail.find((item) => item.current) ?? rail[rail.length - 1] ?? null;
}
