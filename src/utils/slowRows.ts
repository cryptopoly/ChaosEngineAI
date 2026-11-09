import type { VideoModelVariant } from "../types";
import { isAppleSiliconHost } from "./platform";

/** D-44: what the Models page and Video Studio show for a row that is slow on this machine. */
export interface SlowRowInfo {
  kind: "measured" | "expected";
  /** Only when measured. */
  seconds: number | null;
  measuredOn: string | null;
  measuredDate: string | null;
  size: string | null;
  frames: number | null;
  /** Faster row to point at (catalog id). */
  alternativeId: string | null;
}

/** The "Slow on this Mac" facts for ``variant`` on ``host``, or ``null``. Only the catalog row's own
 * ``slowOn`` field counts, and only on a platform it lists (today: Apple Silicon); CUDA and other
 * hosts get nothing. A measured row without a usable time or a place and date is shown as "expected"
 * rather than with a made-up number. */
export function slowRowInfo(
  variant: Pick<VideoModelVariant, "slowOn"> | null | undefined,
  host: Parameters<typeof isAppleSiliconHost>[0],
): SlowRowInfo | null {
  const slow = variant?.slowOn;
  if (!slow || !Array.isArray(slow.platforms)) return null;
  if (!(slow.platforms.includes("apple-silicon") && isAppleSiliconHost(host))) return null;
  const seconds = typeof slow.seconds === "number" && slow.seconds > 0 ? slow.seconds : null;
  const measured = slow.basis === "measured" && seconds != null && Boolean(slow.measuredOn) && Boolean(slow.measuredDate);
  return {
    kind: measured ? "measured" : "expected",
    seconds: measured ? seconds : null,
    measuredOn: measured ? slow.measuredOn ?? null : null,
    measuredDate: measured ? slow.measuredDate ?? null : null,
    size: measured ? slow.size ?? null : null,
    frames: measured ? slow.frames ?? null : null,
    alternativeId: slow.alternativeId ?? null,
  };
}
