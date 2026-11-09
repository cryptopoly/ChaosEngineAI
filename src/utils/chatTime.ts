/**
 * Times in Chat: when a message was sent, when a tool call started, how long it has run, and the
 * progress line of a long job. Pure functions over epoch milliseconds (sessions saved before these
 * fields existed have none, and then nothing is shown). Formatting goes through Intl with the app locale.
 */
import { useEffect, useState } from "react";
import type { JobProgress } from "../types";

function safeFormat(locale: string, options: Intl.DateTimeFormatOptions, date: Date): string {
  try {
    return new Intl.DateTimeFormat(locale || undefined, options).format(date);
  } catch {
    return new Intl.DateTimeFormat(undefined, options).format(date);
  }
}

export function isValidTime(ms: unknown): ms is number {
  return typeof ms === "number" && Number.isFinite(ms) && ms > 0;
}

function sameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

/** Short local time ("07:31"); with the date ("9 Nov, 07:31") when it was not today, and the year when not this year. */
export function formatClock(ms: number, locale: string, now: number = Date.now()): string {
  const date = new Date(ms);
  const today = new Date(now);
  const time: Intl.DateTimeFormatOptions = { hour: "2-digit", minute: "2-digit" };
  if (sameDay(date, today)) return safeFormat(locale, time, date);
  const day: Intl.DateTimeFormatOptions =
    date.getFullYear() === today.getFullYear() ? { month: "short", day: "numeric" } : { year: "numeric", month: "short", day: "numeric" };
  return safeFormat(locale, { ...day, ...time }, date);
}

/** Full date and time with seconds, for a tooltip. */
export function formatFullTime(ms: number, locale: string): string {
  return safeFormat(locale, { dateStyle: "full", timeStyle: "medium" }, new Date(ms));
}

function unit(locale: string, unitName: "hour" | "minute" | "second", value: number): string {
  try {
    return new Intl.NumberFormat(locale || undefined, { style: "unit", unit: unitName, unitDisplay: "short" }).format(value);
  } catch {
    return `${value} ${unitName[0]}`;
  }
}

/** "12 sec", "4 min 12 sec", "1 hr 5 min": two units at most, rounded to the second (to the minute from an hour). */
export function formatDuration(seconds: number, locale: string): string {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return unit(locale, "second", total);
  const minutes = Math.floor(total / 60);
  if (minutes < 60) {
    const rest = total % 60;
    return rest ? `${unit(locale, "minute", minutes)} ${unit(locale, "second", rest)}` : unit(locale, "minute", minutes);
  }
  const hours = Math.floor(minutes / 60);
  const restMinutes = minutes % 60;
  return restMinutes ? `${unit(locale, "hour", hours)} ${unit(locale, "minute", restMinutes)}` : unit(locale, "hour", hours);
}

/** Seconds a call has run: its saved elapsed time once finished, otherwise now minus the start. */
export function liveElapsedSeconds(
  call: { startedAt?: number | null; elapsed?: number; running?: boolean },
  now: number,
): number | null {
  if (call.running) return isValidTime(call.startedAt) ? Math.max(0, (now - call.startedAt) / 1000) : null;
  if (typeof call.elapsed === "number" && call.elapsed > 0) return call.elapsed;
  return null;
}

/** Re-renders every second while `active`; returns the current time in ms. */
export function useNow(active: boolean, intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [active, intervalMs]);
  return now;
}

export type EtaBasis = "clips" | "steps" | "catalog";

/** Progress facts the card shows: step text, elapsed, and time left only when the backend had a basis. */
export function progressFacts(progress: JobProgress | null | undefined): {
  step: number | null;
  totalSteps: number | null;
  phase: string | null;
  etaSeconds: number | null;
  etaBasis: EtaBasis | null;
} {
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const eta = num(progress?.etaSeconds);
  const basis = progress?.etaBasis;
  return {
    step: num(progress?.step),
    totalSteps: num(progress?.totalSteps),
    phase: typeof progress?.phase === "string" && progress.phase ? progress.phase : null,
    etaSeconds: eta !== null && eta >= 0 ? eta : null,
    etaBasis: basis === "clips" || basis === "steps" || basis === "catalog" ? basis : null,
  };
}
