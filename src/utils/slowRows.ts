import type { VideoModelVariant } from "../types";
import { hasNvidiaGpu, imageOrVideoVariantPlatformGate, isAppleSiliconHost, isVariantCompatibleWithHost } from "./platform";

type Host = Parameters<typeof isAppleSiliconHost>[0];

/** The kinds of machine a catalog row can be marked slow on (D-44). */
export type SlowPlatform = "apple-mps" | "cpu-only" | "cuda";

/** Which kind of machine this is, or ``null`` while the host is unknown. Apple Silicon is
 * "apple-mps"; a Windows / Linux box with an NVIDIA GPU is "cuda"; anything else (no NVIDIA card,
 * an Intel Mac) is "cpu-only". */
export function slowPlatformKey(host: Host): SlowPlatform | null {
  if (!host) return null;
  if (isAppleSiliconHost(host)) return "apple-mps";
  if (hasNvidiaGpu(host)) return "cuda";
  return "cpu-only";
}

/** What the Models page and Video Studio show for a row that is slow on this machine. */
export interface SlowRowInfo {
  platform: SlowPlatform;
  kind: "measured" | "expected";
  /** Key of the reason sentence ("pytorch-apple-gpu", "bf16-full-size", "no-gpu-cpu"). */
  reason: string | null;
  /** Only when measured. */
  seconds: number | null;
  measuredOn: string | null;
  measuredDate: string | null;
  size: string | null;
  frames: number | null;
  /** Faster row to point at (catalog id); only a row that runs on this machine is shown. */
  alternativeId: string | null;
}

/** The "Slow on this computer" facts for ``variant`` on ``host``, or ``null`` when no entry of the
 * row's ``slowOn`` list matches this machine (so CUDA machines see nothing unless a measurement says
 * so). A measured entry without a usable time, place and date reads as "expected" rather than with a
 * made-up number. */
export function slowRowInfo(variant: Pick<VideoModelVariant, "slowOn"> | null | undefined, host: Host): SlowRowInfo | null {
  const platform = slowPlatformKey(host);
  if (!platform || !Array.isArray(variant?.slowOn)) return null;
  const slow = variant.slowOn.find((entry) => entry.platform === platform);
  if (!slow) return null;
  const seconds = typeof slow.seconds === "number" && slow.seconds > 0 ? slow.seconds : null;
  const measured = slow.basis === "measured" && seconds != null && Boolean(slow.measuredOn) && Boolean(slow.measuredDate);
  return {
    platform,
    kind: measured ? "measured" : "expected",
    reason: slow.reason ?? null,
    seconds: measured ? seconds : null,
    measuredOn: measured ? slow.measuredOn ?? null : null,
    measuredDate: measured ? slow.measuredDate ?? null : null,
    size: measured ? slow.size ?? null : null,
    frames: measured ? slow.frames ?? null : null,
    alternativeId: slow.alternativeId ?? null,
  };
}

/** Name of the suggested faster row, only when that row is in the catalog and runs on ``host``
 * (the LTX MLX rows exist on Macs only). */
export function slowAlternativeName(
  info: SlowRowInfo | null,
  catalogVariants: VideoModelVariant[],
  host: Host,
): string | null {
  if (!info?.alternativeId) return null;
  const alternative = catalogVariants.find((variant) => variant.id === info.alternativeId);
  if (!alternative) return null;
  return isVariantCompatibleWithHost(imageOrVideoVariantPlatformGate(alternative), host) ? alternative.name : null;
}
