/**
 * The rule both generation Studios share for a run the memory estimate calls dangerous: Generate stays
 * off until the user ticks "Run anyway", and the tick belongs to one configuration only. It is stored
 * with the key of the settings it was given for, so any change to those settings makes it count as
 * unticked again without an effect having to clear it.
 */
import type { EstimateRisk } from "./RunEstimateStrip";

export interface DangerAck {
  key: string;
  checked: boolean;
}

export const NO_ACK: DangerAck = { key: "", checked: false };

/** One key for the settings that feed the safety estimate (model, size, frames ...). */
export function ackKey(parts: readonly (string | number | null | undefined)[]): string {
  return parts.map((part) => String(part ?? "")).join("|");
}

/** True only while the tick was given for exactly these settings. */
export function isAcked(ack: DangerAck, key: string): boolean {
  return ack.checked && ack.key === key;
}

/** Generate is blocked at danger level until the tick is given. Caution and safe runs are never blocked here. */
export function dangerBlocked(risk: EstimateRisk, acked: boolean): boolean {
  return risk === "danger" && !acked;
}
