/**
 * Words for a recommended chat row (``Recommendation.bestFit`` / ``chatPicks``): its size, the memory it
 * needs and the speed to expect. Shared by the Dashboard and Chat Models so both say the same thing.
 */

import type { TFunction } from "i18next";
import type { ChatPick } from "../../types";
import { number, sizeLabel } from "../../utils";

/** "about 120 tok/s (estimate)" or "233 tok/s, measured on this computer"; ``null`` without a figure. */
export function pickSpeedText(pick: ChatPick, t: TFunction<"dashboard">): string | null {
  if (!pick.tokS || pick.tokS <= 0) return null;
  const value = number(pick.tokS, 0);
  if (pick.tokSBasis === "measured") {
    return t("bestFit.speedMeasured", { defaultValue: "{value} tok/s, measured on this computer", value });
  }
  if (pick.tokSBasis === "rtx4090") {
    return t("bestFit.speedRtx4090", { defaultValue: "{value} tok/s measured on an RTX 4090", value });
  }
  return t("bestFit.speedEstimated", { defaultValue: "about {value} tok/s (estimate)", value });
}

/** "Needs about 10 GB of 24 GB GPU memory" (``gpu`` when the card has its own memory). */
export function pickMemoryText(pick: ChatPick, gpu: boolean, t: TFunction<"dashboard">): string | null {
  const need = pick.fit?.needGb;
  const usable = pick.fit?.usableGb;
  if (need == null || !usable) return null;
  const values = { need: number(need, need >= 10 ? 0 : 1), usable: number(usable, usable >= 10 ? 0 : 1) };
  return gpu
    ? t("bestFit.needsGpu", { defaultValue: "needs about {need} GB of {usable} GB GPU memory", ...values })
    : t("bestFit.needsMemory", { defaultValue: "needs about {need} GB of {usable} GB usable memory", ...values });
}

/** One line under the pick's name: download size, memory, speed. */
export function pickSummary(pick: ChatPick, gpu: boolean, t: TFunction<"dashboard">): string {
  const parts = [
    pick.sizeGb ? t("bestFit.download", { defaultValue: "{size} download", size: sizeLabel(pick.sizeGb) }) : null,
    pickMemoryText(pick, gpu, t),
    pickSpeedText(pick, t),
  ];
  return parts.filter(Boolean).join(" · ");
}

/** A PC whose graphics card has its own memory runs the model there (the Models page judges by the card). */
export function hasDedicatedGpu(system: { gpuVramTotalGb?: number | null; gpuUnifiedMemory?: boolean | null }): boolean {
  return Boolean(system.gpuVramTotalGb) && system.gpuUnifiedMemory === false;
}
