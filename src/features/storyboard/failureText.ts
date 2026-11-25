/**
 * Plain-language failures with one action (Storyboard UX audit, change 5). The backend stores a failure code and an
 * action code beside the raw error text (`backend_service/storyboard/failure.py`, `guide.failures`); this turns
 * them into the sentence the card shows and the one button that helps. The raw text stays behind Details.
 *
 * A code a newer backend sends that this list lacks reads as "unknown" with the backend's own sentence.
 */
import type { StoryboardFailure, StoryboardFailureAction, StoryboardFailureCode, StoryboardSize } from "../../types";
import { sizeText, type TFn } from "./qualityText";

export const FAILURE_CODES: readonly StoryboardFailureCode[] = [
  "out-of-memory", "model-missing", "lipsync-model", "download-blocked", "video-tools-missing", "size-over-cap",
  "references-unsupported", "no-prompt", "disk-full", "cancelled", "unknown",
];
export const FAILURE_ACTIONS: readonly StoryboardFailureAction[] = [
  "smaller-size", "open-models", "install-video-tools", "edit-shot", "free-disk", "retry",
];

type FailureLike = Pick<StoryboardFailure, "code" | "kind"> & { message?: string; action?: StoryboardFailureAction | string };

/** The plain sentence for a failure. */
export function failureSentence(t: TFn, failure: FailureLike | null | undefined): string {
  switch (failure?.code) {
    case "out-of-memory":
      return t("board.failure.outOfMemory", { defaultValue: "There was not enough free memory to make this shot at this size." });
    case "model-missing":
      return t("board.failure.modelMissing", { defaultValue: "The model this shot needs is not installed." });
    case "lipsync-model":
      return t("board.failure.lipsyncModel", { defaultValue: "The video model in this project cannot lip-sync this shot's vocal." });
    case "download-blocked":
      return t("board.failure.downloadBlocked", { defaultValue: "The model could not be downloaded. It may need its licence accepted, or the connection is down." });
    case "video-tools-missing":
      return t("board.failure.videoToolsMissing", { defaultValue: "The video tools are not installed." });
    case "size-over-cap":
      return t("board.failure.sizeOverCap", { defaultValue: "This shot is larger than the model can make." });
    case "references-unsupported":
      return t("board.failure.referencesUnsupported", { defaultValue: "The model cannot take this shot's pictures (a start, end or reference picture)." });
    case "no-prompt":
      return t("board.failure.noPrompt", { defaultValue: "This shot has nothing to make it from. Write its prompt first." });
    case "disk-full":
      return t("board.failure.diskFull", { defaultValue: "The disk is full, so the shot could not be saved. Free some space, then try again." });
    case "cancelled":
      return t("board.failure.cancelled", { defaultValue: "The shot was stopped before it finished." });
    case "unknown":
    case undefined:
      return t("board.failure.unknown", { defaultValue: "The shot did not render." });
    default:
      // A code from a newer backend: its own sentence is better than a guess.
      return (failure as FailureLike).message?.trim() || t("board.failure.unknown", { defaultValue: "The shot did not render." });
  }
}

/** The action the button carries: the backend's, or Retry for one this list lacks. */
export function failureActionOf(failure: FailureLike | null | undefined): StoryboardFailureAction {
  const action = failure?.action;
  return action && (FAILURE_ACTIONS as readonly string[]).includes(action) ? (action as StoryboardFailureAction) : "retry";
}

export interface FailureButton {
  /** What the card does when it is pressed: the card maps `retry` and `edit-shot` itself and hands the rest on. */
  action: StoryboardFailureAction;
  label: string;
  title?: string;
}

export interface FailureButtonOptions {
  /** The size a draft is made at (`guide.quality.draftSize`), named on the "smaller size" button. */
  draftSize?: StoryboardSize | null;
  /** Whether the screen can do the action. One it cannot do falls back to Retry, so the button always works. */
  canHandle?: (action: StoryboardFailureAction) => boolean;
}

/** The one button that helps. `free-disk` has no button of its own: the sentence says what to free, Retry follows. */
export function failureButton(t: TFn, failure: FailureLike | null | undefined, options: FailureButtonOptions = {}): FailureButton {
  const retry: FailureButton = { action: "retry", label: t("error.retry", { defaultValue: "Retry" }) };
  const wanted = failureActionOf(failure);
  if (wanted === "retry" || wanted === "free-disk") return retry;
  if (wanted === "edit-shot") {
    return { action: "edit-shot", label: t("board.failure.actionEdit", { defaultValue: "Edit the prompts" }) };
  }
  if (!options.canHandle?.(wanted)) return retry;
  switch (wanted) {
    case "smaller-size":
      return {
        action: wanted,
        label: options.draftSize
          ? t("board.failure.actionSmallerSized", { defaultValue: "Make it at draft size ({size})", size: sizeText(options.draftSize) })
          : t("board.failure.actionSmaller", { defaultValue: "Make it at draft size" }),
        title: t("board.failure.actionSmallerTitle", { defaultValue: "Makes this shot again at a smaller size that needs less memory. It becomes a draft; choose it for final later." }),
      };
    case "open-models":
      return {
        action: wanted,
        label:
          failure?.kind === "clip"
            ? t("board.failure.actionVideoModels", { defaultValue: "Open Video Models" })
            : t("board.failure.actionImageModels", { defaultValue: "Open Image Models" }),
      };
    default:
      return { action: wanted, label: t("board.failure.actionVideoTools", { defaultValue: "Install video tools" }) };
  }
}
