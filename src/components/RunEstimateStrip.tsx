/**
 * Cost before the click: time, peak memory of total, and disk written, with
 * a Measured / Estimated basis, sitting directly above a Studio's primary
 * action. Updates are announced politely. An unknown value reads as a
 * sentence, never as a blank.
 */
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { BusyAdvice } from "../types";

export type EstimateBasis = "measured" | "estimated";
export type EstimateRisk = "safe" | "caution" | "danger";

export interface RunEstimateQueue {
  /** What is running ahead, e.g. "Image Studio". */
  what: string;
  secondsLeft: number | null;
  /** Queue position, used when `what` is unknown. */
  position?: number;
}

export interface RunEstimateStripProps {
  seconds: number | null;
  peakGb: number | null;
  totalGb: number | null;
  diskMb: number | null;
  basis: EstimateBasis | null;
  risk: EstimateRisk;
  queue?: RunEstimateQueue | null;
  /** The machine is short of memory for this run beside what is already loaded (advice only). */
  busy?: BusyAdvice | null;
  /** Shown only at `danger`: the user must accept before running. */
  ack?: { checked: boolean; onChange: (checked: boolean) => void };
  /** Longer facts behind the memory figure (device, model size), shown as its tooltip. */
  memoryTitle?: string;
  action: ReactNode;
}

/** Memory share of total at which the verdict turns caution / danger. */
export const CAUTION_SHARE = 0.7;
export const DANGER_SHARE = 0.92;

export function memoryRisk(peakGb: number | null, totalGb: number | null, fallback: EstimateRisk): EstimateRisk {
  if (peakGb == null || totalGb == null || totalGb <= 0) return fallback;
  const share = peakGb / totalGb;
  if (share > DANGER_SHARE) return "danger";
  if (share >= CAUTION_SHARE) return "caution";
  return "safe";
}

type TFn = (key: string, options: Record<string, unknown>) => string;

function oneDecimal(n: number): string {
  return String(Math.round(n * 10) / 10);
}

export function formatDuration(t: TFn, seconds: number): string {
  const total = Math.max(1, Math.round(seconds));
  if (total < 60) return t("estimate.duration.seconds", { defaultValue: "{n} s", n: total });
  const m = Math.floor(total / 60);
  const s = total % 60;
  if (s === 0) return t("estimate.duration.minutes", { defaultValue: "{n} min", n: m });
  return t("estimate.duration.minutesSeconds", { defaultValue: "{m} min {s} s", m, s });
}

export function formatGb(t: TFn, gb: number): string {
  return t("estimate.size.gb", { defaultValue: "{n} GB", n: oneDecimal(gb) });
}

export function formatDisk(t: TFn, mb: number): string {
  if (mb >= 1000) return formatGb(t, mb / 1000);
  return t("estimate.size.mb", { defaultValue: "{n} MB", n: Math.max(1, Math.round(mb)) });
}

export function RunEstimateStrip({ seconds, peakGb, totalGb, diskMb, basis, risk, queue, busy, ack, memoryTitle, action }: RunEstimateStripProps) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as TFn;
  const memoryBand = memoryRisk(peakGb, totalGb, risk);

  const timeText =
    seconds == null
      ? t("estimate.timeUnknown", { defaultValue: "Time: unknown until the first run" })
      : t("estimate.time", { defaultValue: "About {time}", time: formatDuration(tf, seconds) });
  const memoryText =
    peakGb == null || totalGb == null
      ? t("estimate.memoryUnknown", { defaultValue: "Memory: not estimated" })
      : t("estimate.memory", { defaultValue: "Peak {peak} of {total}", peak: formatGb(tf, peakGb), total: formatGb(tf, totalGb) });
  const diskText =
    diskMb == null
      ? t("estimate.diskUnknown", { defaultValue: "Disk: not estimated" })
      : t("estimate.disk", { defaultValue: "Writes {size}", size: formatDisk(tf, diskMb) });

  let queueText: string | null = null;
  if (queue) {
    queueText =
      queue.what && queue.secondsLeft != null
        ? t("estimate.queueRunning", {
            defaultValue: "{what} is running, about {time} left. Yours starts after it.",
            what: queue.what,
            time: formatDuration(tf, queue.secondsLeft),
          })
        : t("estimate.queuePosition", {
            defaultValue: "Waiting for another job to finish (position {position})",
            position: queue.position ?? 1,
          });
  }

  return (
    <div className={`run-estimate run-estimate--${risk}`}>
      <div className="run-estimate__row">
        <div className="run-estimate__items" aria-live="polite">
          <span className="run-estimate__item run-estimate__item--time">{timeText}</span>
          <span className={`run-estimate__item run-estimate__item--memory run-estimate__item--${memoryBand}`} title={memoryTitle}>{memoryText}</span>
          <span className="run-estimate__item run-estimate__item--disk">{diskText}</span>
          {basis ? (
            <span
              className={`badge subtle run-estimate__basis run-estimate__basis--${basis}`}
              title={
                basis === "measured"
                  ? t("estimate.measuredTitle", { defaultValue: "Timed on this machine for this model and preset." })
                  : t("estimate.estimatedTitle", { defaultValue: "From the model card or a similar machine; not timed here yet." })
              }
            >
              {basis === "measured"
                ? t("estimate.measured", { defaultValue: "Measured" })
                : t("estimate.estimated", { defaultValue: "Estimated" })}
            </span>
          ) : null}
        </div>
        <div className="run-estimate__action">{action}</div>
      </div>
      {queueText ? <p className="run-estimate__queue" aria-live="polite">{queueText}</p> : null}
      {busy && busy.level !== "ok" && busy.message ? <p className="run-estimate__queue run-estimate__busy" aria-live="polite">{busy.message}</p> : null}
      {risk === "danger" && ack ? (
        <label className="check-row run-estimate__ack">
          <input type="checkbox" checked={ack.checked} onChange={(event) => ack.onChange(event.target.checked)} />
          {t("estimate.dangerAck", { defaultValue: "Run anyway. I accept that the backend may crash." })}
        </label>
      ) : null}
    </div>
  );
}
