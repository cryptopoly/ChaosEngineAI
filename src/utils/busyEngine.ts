import type { TFunction } from "i18next";
import type { ActiveGeneration } from "../types";

/**
 * One line per run in flight: "Image engine busy: Qwen-Image 2.1 · step 27 of 40". The chat runtime reports "No model
 * loaded" while a studio run holds the GPU in its own engine; this is what to say instead. ``t`` is the "common" namespace.
 */
export function busyEngineLines(t: TFunction, runs: readonly ActiveGeneration[] | undefined): string[] {
  return (runs ?? []).map((run) => {
    const hasSteps = run.totalSteps > 0 && (run.phase === "diffusing" || run.phase === "compose");
    const parts = [run.runLabel || run.message];
    if (hasSteps) {
      parts.push(t("serverTab.engineBusy.step", { defaultValue: "step {step} of {total}", step: Math.min(run.step, run.totalSteps), total: run.totalSteps }));
    }
    const detail = parts.filter(Boolean).join(" · ");
    if (run.kind === "video") {
      return detail
        ? t("serverTab.engineBusy.video", { defaultValue: "Video engine busy: {detail}", detail })
        : t("serverTab.engineBusy.videoBare", { defaultValue: "Video engine busy" });
    }
    if (run.kind === "music") {
      return detail
        ? t("serverTab.engineBusy.music", { defaultValue: "Music engine busy: {detail}", detail })
        : t("serverTab.engineBusy.musicBare", { defaultValue: "Music engine busy" });
    }
    return detail
      ? t("serverTab.engineBusy.image", { defaultValue: "Image engine busy: {detail}", detail })
      : t("serverTab.engineBusy.imageBare", { defaultValue: "Image engine busy" });
  });
}
