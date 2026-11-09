/**
 * D-44: "Slow on this Mac" label for a video row, with the measured time (where and when it was timed)
 * or "Expected to be slow" when there is no measurement, and a pointer to the faster row. The caller
 * decides whether it applies (``slowRowInfo``); this only renders.
 */
import { useTranslation } from "react-i18next";
import type { SlowRowInfo } from "../utils/slowRows";
import { formatDuration } from "./RunEstimateStrip";

type TFn = (key: string, options: Record<string, unknown>) => string;

/** One sentence: the measurement or the expectation, then the faster alternative. */
export function useSlowRowText(info: SlowRowInfo, alternativeName: string | null): { label: string; detail: string } {
  const { t } = useTranslation("studio");
  const tf = t as unknown as TFn;
  const label = tf("slowRow.label", { defaultValue: "Slow on this Mac" });
  const basis =
    info.kind === "measured" && info.seconds != null
      ? tf("slowRow.measured", {
          defaultValue: "Measured on {machine}, {date}: about {time} for a {size} clip of {frames, plural, one {# frame} other {# frames}}.",
          machine: info.measuredOn ?? "",
          date: info.measuredDate ?? "",
          time: formatDuration(tf, info.seconds),
          size: info.size ?? "",
          frames: info.frames ?? 0,
        })
      : tf("slowRow.expected", { defaultValue: "Expected to be slow on this Mac. Not timed yet." });
  const alternative = alternativeName
    ? tf("slowRow.alternative", { defaultValue: "Faster here: {name}.", name: alternativeName })
    : "";
  return { label, detail: [basis, alternative].filter(Boolean).join(" ") };
}

/** Small badge for a Models list row; the full sentence is its tooltip. */
export function SlowRowChip({ info, alternativeName }: { info: SlowRowInfo; alternativeName: string | null }) {
  const { label, detail } = useSlowRowText(info, alternativeName);
  return (
    <span className="badge warning" data-testid="slow-row-chip" title={detail}>
      {label}
    </span>
  );
}

/** Note under a Models list row (``detailOnly``: the chip above already says the label) or above Generate in Video Studio. */
export function SlowRowNote({
  info,
  alternativeName,
  detailOnly = false,
}: {
  info: SlowRowInfo;
  alternativeName: string | null;
  detailOnly?: boolean;
}) {
  const { label, detail } = useSlowRowText(info, alternativeName);
  return (
    <p className="muted-text" role="note" data-testid="slow-row-note">
      {detailOnly ? null : <strong>{label}. </strong>}
      {detail}
    </p>
  );
}
