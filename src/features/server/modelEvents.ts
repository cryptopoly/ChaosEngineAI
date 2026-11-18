/**
 * Structured model events from ``GET /api/server/events`` (D-88) and the pure
 * helpers the Activity view uses: filters, the per-model timeline, the latest
 * failure, row summaries and the text for Copy and Export.
 */

export type ModelEventKind =
  | "requested"
  | "resolving"
  | "loading"
  | "loaded"
  | "load_failed"
  | "parked"
  | "first_token"
  | "generation"
  | "generation_failed"
  | "idle"
  | "unloaded";

export type ModelEventLevel = "info" | "warning" | "error";

export interface ModelEventDetail {
  command?: string;
  env?: string[];
  stderr?: string[];
  error?: string;
}

export interface ModelEvent {
  id: number;
  ts: number;
  time: string;
  kind: ModelEventKind | string;
  level: ModelEventLevel | string;
  model: string | null;
  modelRef: string | null;
  requestId: string | null;
  engine: string | null;
  cacheStrategy: string | null;
  speculative: string | null;
  message: string;
  data: Record<string, unknown>;
  detail: ModelEventDetail;
}

export interface EventFilter {
  model: string;
  level: string;
  request: string;
  query: string;
}

export const EMPTY_FILTER: EventFilter = { model: "", level: "", request: "", query: "" };

/** Most events kept in memory by the view; the backend ring holds 1000. */
export const MAX_VIEW_EVENTS = 1000;

export function modelKey(event: Pick<ModelEvent, "model" | "modelRef">): string {
  return event.model || event.modelRef || "";
}

/** Merge new events into a list sorted by id, without duplicates, keeping the newest ``cap``. */
export function mergeEvents(current: ModelEvent[], incoming: ModelEvent[], cap = MAX_VIEW_EVENTS): ModelEvent[] {
  if (incoming.length === 0) return current;
  const byId = new Map<number, ModelEvent>();
  for (const event of current) byId.set(event.id, event);
  let changed = false;
  for (const event of incoming) {
    if (!byId.has(event.id)) changed = true;
    byId.set(event.id, event);
  }
  if (!changed) return current;
  const merged = [...byId.values()].sort((a, b) => a.id - b.id);
  return merged.length > cap ? merged.slice(merged.length - cap) : merged;
}

export function modelsOf(events: ModelEvent[]): string[] {
  const seen = new Set<string>();
  for (const event of events) {
    const key = modelKey(event);
    if (key) seen.add(key);
  }
  return [...seen].sort((a, b) => a.localeCompare(b));
}

export function filterEvents(events: ModelEvent[], filter: EventFilter): ModelEvent[] {
  const request = filter.request.trim().toLowerCase();
  const query = filter.query.trim().toLowerCase();
  return events.filter((event) => {
    if (filter.model && modelKey(event) !== filter.model) return false;
    if (filter.level && event.level !== filter.level) return false;
    if (request && !(event.requestId ?? "").toLowerCase().includes(request)) return false;
    if (query) {
      const haystack = [event.kind, event.model, event.modelRef, event.engine, event.message, event.requestId]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  });
}

export function isFailure(event: ModelEvent): boolean {
  return event.level === "error";
}

/** The newest error event, or null. */
export function latestFailure(events: ModelEvent[]): ModelEvent | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (isFailure(events[index])) return events[index];
  }
  return null;
}

// ---------------------------------------------------------------- timeline

export type TimelinePhase = "loading" | "ready" | "generating" | "idle" | "warm" | "failed";

export interface TimelineSegment {
  phase: TimelinePhase;
  start: number;
  end: number;
}

export interface TimelineRow {
  model: string;
  engine: string | null;
  start: number;
  end: number;
  segments: TimelineSegment[];
  /** Still open at the end of the events (loaded or loading now). */
  live: boolean;
  failed: boolean;
}

function numberField(event: ModelEvent, key: string): number | null {
  const value = event.data?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** One row per model: loading, ready, generating, idle and warm phases from its events.
 *
 * ``now`` closes a phase still open at the end (seconds since the epoch). */
export function buildTimeline(events: ModelEvent[], now: number): TimelineRow[] {
  const rows = new Map<string, TimelineRow & { open: TimelineSegment | null }>();
  const sorted = [...events].sort((a, b) => a.ts - b.ts || a.id - b.id);
  for (const event of sorted) {
    const key = modelKey(event);
    if (!key) continue;
    let row = rows.get(key);
    if (!row) {
      row = { model: key, engine: event.engine, start: event.ts, end: event.ts, segments: [], live: false, failed: false, open: null };
      rows.set(key, row);
    }
    if (event.engine) row.engine = event.engine;
    const close = (at: number) => {
      if (row && row.open) {
        row.open.end = Math.max(row.open.start, at);
        row.segments.push(row.open);
        row.open = null;
      }
    };
    const open = (phase: TimelinePhase, at: number) => {
      close(at);
      if (row) row.open = { phase, start: at, end: at };
    };
    switch (event.kind) {
      case "requested":
      case "resolving":
      case "loading":
        if (!row.open || row.open.phase !== "loading") open("loading", event.ts);
        break;
      case "loaded":
        if (!row.open || row.open.phase !== "loading") {
          const seconds = numberField(event, "seconds") ?? 0;
          open("loading", event.ts - seconds);
        }
        open("ready", event.ts);
        break;
      case "load_failed":
        if (row.open) row.open.phase = "failed";
        else row.open = { phase: "failed", start: event.ts, end: event.ts };
        close(event.ts);
        row.failed = true;
        break;
      case "generation":
      case "generation_failed": {
        const seconds = numberField(event, "seconds") ?? 0;
        const start = event.ts - seconds;
        const resting: TimelinePhase = row.open && row.open.phase !== "generating" ? row.open.phase : "ready";
        close(start);
        row.segments.push({ phase: event.kind === "generation_failed" ? "failed" : "generating", start, end: event.ts });
        if (event.kind === "generation_failed") row.failed = true;
        row.open = { phase: resting === "loading" ? "ready" : resting, start: event.ts, end: event.ts };
        break;
      }
      case "idle":
        if (row.open && (row.open.phase === "ready" || row.open.phase === "idle")) row.open.phase = "idle";
        else open("idle", event.ts);
        break;
      case "parked":
        open("warm", event.ts);
        break;
      case "unloaded":
        close(event.ts);
        break;
      default:
        break;
    }
    row.end = Math.max(row.end, event.ts);
  }
  const result: TimelineRow[] = [];
  for (const row of rows.values()) {
    const { open, ...rest } = row;
    if (open) {
      open.end = Math.max(open.start, now);
      rest.segments.push(open);
      rest.live = true;
      rest.end = Math.max(rest.end, now);
    }
    rest.start = rest.segments.length ? Math.min(rest.start, ...rest.segments.map((s) => s.start)) : rest.start;
    result.push(rest);
  }
  return result.sort((a, b) => b.end - a.end);
}

// ---------------------------------------------------------------- text

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

export function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0 s";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds - minutes * 60)} s`;
}

type Translate = (key: string, options?: Record<string, unknown>) => string;

/** One localised line for an event row. Falls back to the backend's English message. */
export function eventSummary(event: ModelEvent, t: Translate): string {
  const seconds = numberField(event, "seconds");
  switch (event.kind) {
    case "requested":
      return t("serverActivity.summary.requested", { backend: String(event.data.backend ?? "auto"), defaultValue: "Load requested ({backend})" });
    case "resolving":
      return t("serverActivity.summary.resolving", { defaultValue: "Choosing an engine" });
    case "loading":
      return t("serverActivity.summary.loading", {
        engine: event.engine ?? "?",
        size: formatBytes(numberField(event, "bytes") ?? 0),
        defaultValue: "Loading on {engine} ({size})",
      });
    case "loaded":
      return event.data.warm
        ? t("serverActivity.summary.loadedWarm", { defaultValue: "Switched to the warm copy" })
        : t("serverActivity.summary.loaded", {
          seconds: formatSeconds(seconds ?? 0),
          memory: formatBytes(numberField(event, "memoryBytes") ?? 0),
          defaultValue: "Loaded in {seconds}, {memory} in memory",
        });
    case "parked":
      return t("serverActivity.summary.parked", { defaultValue: "Kept warm for a quick switch back" });
    case "first_token":
      return t("serverActivity.summary.firstToken", {
        seconds: formatSeconds(numberField(event, "prefillSeconds") ?? 0),
        defaultValue: "First token after {seconds}",
      });
    case "generation":
      return t("serverActivity.summary.generation", {
        tokens: numberField(event, "completionTokens") ?? 0,
        rate: (numberField(event, "tokS") ?? 0).toFixed(1),
        seconds: formatSeconds(seconds ?? 0),
        defaultValue: "{tokens} tokens at {rate} tok/s in {seconds}",
      });
    case "idle":
      return t("serverActivity.summary.idle", { defaultValue: "Idle" });
    case "unloaded":
      return t("serverActivity.summary.unloaded", {
        reason: t(`serverActivity.reason.${String(event.data.reason ?? "user")}`, { defaultValue: String(event.data.reason ?? "user") }),
        defaultValue: "Unloaded ({reason})",
      });
    default:
      return event.message;
  }
}

/** Events as pretty JSON, for Copy and Export. */
export function eventsToJson(events: ModelEvent[]): string {
  return JSON.stringify(events, null, 2);
}

export function exportFilename(date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `model-events-${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}.json`;
}

export function downloadJson(content: string, filename: string): void {
  const blob = new Blob([content], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
