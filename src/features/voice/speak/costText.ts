/** Words for a model's measured cost (card chip, Speak cost line, Models row), from ``costLabel``. */

import { useTranslation } from "react-i18next";
import type { CostLabel } from "./speakLogic";

export interface CostText {
  /** Short chip on a voice card: "3.5 GB, 2.5 s per 10 s". */
  chip: (cost: CostLabel) => string;
  /** The middle of the cost line: "3.5 GB in memory, about 2.5 s per 10 s of speech (measured on ...)". */
  detail: (cost: CostLabel) => string;
}

export function useCostText(): CostText {
  const { t } = useTranslation("studio");
  return {
    chip: (cost) =>
      cost.kind === "kokoro"
        ? t("voice.cost.kokoroChip", { defaultValue: "fast, under 1 GB" })
        : cost.kind === "measured"
          ? t("voice.cost.chip", { defaultValue: "{peak} GB, {seconds} s per 10 s", peak: cost.peakGb, seconds: cost.secondsPer10s })
          : t("voice.cost.unmeasuredChip", { defaultValue: "{gb} GB, not measured", gb: cost.downloadGb }),
    detail: (cost) =>
      cost.kind === "kokoro"
        ? t("voice.cost.kokoroDetail", { defaultValue: "part of the voice pack, under 1 GB" })
        : cost.kind === "measured"
          ? t("voice.cost.detail", {
              defaultValue: "{peak} GB in memory, about {seconds} s per 10 s of speech (measured on {machine})",
              peak: cost.peakGb,
              seconds: cost.secondsPer10s,
              machine: cost.measuredOn,
            })
          : t("voice.cost.unmeasuredDetail", { defaultValue: "speed and memory not measured yet" }),
  };
}
