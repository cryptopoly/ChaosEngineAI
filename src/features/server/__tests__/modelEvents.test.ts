import { describe, expect, it } from "vitest";
import {
  EMPTY_FILTER,
  buildTimeline,
  eventSummary,
  exportFilename,
  filterEvents,
  formatBytes,
  formatSeconds,
  latestFailure,
  mergeEvents,
  modelsOf,
  type ModelEvent,
} from "../modelEvents";
import { parseSseChunk } from "../../../hooks/useModelEvents";

let nextId = 1;
function ev(kind: string, ts: number, extra: Partial<ModelEvent> = {}): ModelEvent {
  return {
    id: nextId++,
    ts,
    time: "2026-11-18 08:00:00",
    kind,
    level: "info",
    model: "Qwen",
    modelRef: "qwen/q",
    requestId: "r1",
    engine: "mlx",
    cacheStrategy: "native",
    speculative: null,
    message: kind,
    data: {},
    detail: {},
    ...extra,
  };
}

// i18next-like stub: fills {name} placeholders of the default value.
const t = (_key: string, options?: Record<string, unknown>) =>
  String(options?.defaultValue ?? _key).replace(/\{(\w+)\}/g, (_m, name) => String(options?.[name] ?? ""));

describe("mergeEvents", () => {
  it("keeps one copy per id, in id order, and caps the list", () => {
    const a = ev("loaded", 1);
    const b = ev("idle", 2);
    const merged = mergeEvents([b], [a, b]);
    expect(merged.map((e) => e.id)).toEqual([a.id, b.id]);
    const many = Array.from({ length: 12 }, (_, i) => ev("generation", i));
    expect(mergeEvents([], many, 5).map((e) => e.id)).toEqual(many.slice(-5).map((e) => e.id));
  });

  it("returns the same list when nothing is new", () => {
    const a = ev("loaded", 1);
    const current = [a];
    expect(mergeEvents(current, [a])).toBe(current);
    expect(mergeEvents(current, [])).toBe(current);
  });
});

describe("filters", () => {
  const events = [
    ev("loaded", 1, { model: "A", requestId: "abc123" }),
    ev("load_failed", 2, { model: "B", level: "error", requestId: "zzz", message: "out of memory" }),
    ev("generation", 3, { model: "A", requestId: "def456" }),
  ];

  it("filter by model, level, request id and text", () => {
    expect(filterEvents(events, { ...EMPTY_FILTER, model: "A" })).toHaveLength(2);
    expect(filterEvents(events, { ...EMPTY_FILTER, level: "error" })).toHaveLength(1);
    expect(filterEvents(events, { ...EMPTY_FILTER, request: "DEF" })).toHaveLength(1);
    expect(filterEvents(events, { ...EMPTY_FILTER, query: "memory" })).toHaveLength(1);
    expect(filterEvents(events, EMPTY_FILTER)).toHaveLength(3);
  });

  it("lists models and finds the newest failure", () => {
    expect(modelsOf(events)).toEqual(["A", "B"]);
    expect(latestFailure(events)?.kind).toBe("load_failed");
    expect(latestFailure(events.filter((e) => e.level !== "error"))).toBeNull();
  });
});

describe("buildTimeline", () => {
  it("splits a model's life into loading, ready, generating, idle and warm", () => {
    const events = [
      ev("requested", 100),
      ev("loading", 101),
      ev("loaded", 105, { data: { seconds: 4 } }),
      ev("first_token", 108.5),
      ev("generation", 110, { data: { seconds: 2 } }),
      ev("idle", 110),
      ev("parked", 120),
      ev("unloaded", 130, { data: { reason: "memory" } }),
    ];
    const [row] = buildTimeline(events, 200);
    expect(row.model).toBe("Qwen");
    expect(row.live).toBe(false);
    expect(row.segments.map((s) => [s.phase, s.start, s.end])).toEqual([
      ["loading", 100, 105],
      ["ready", 105, 108],
      ["generating", 108, 110],
      ["idle", 110, 120],
      ["warm", 120, 130],
    ]);
  });

  it("keeps a loaded model open until now and marks failures", () => {
    const rows = buildTimeline(
      [
        ev("loading", 10, { model: "Live" }),
        ev("loaded", 12, { model: "Live", data: { seconds: 2 } }),
        ev("requested", 20, { model: "Broken" }),
        ev("load_failed", 25, { model: "Broken", level: "error" }),
      ],
      40,
    );
    const live = rows.find((r) => r.model === "Live")!;
    expect(live.live).toBe(true);
    expect(live.segments[live.segments.length - 1]).toEqual({ phase: "ready", start: 12, end: 40 });
    const broken = rows.find((r) => r.model === "Broken")!;
    expect(broken.failed).toBe(true);
    expect(broken.segments).toEqual([{ phase: "failed", start: 20, end: 25 }]);
    // newest activity first
    expect(rows[0].model).toBe("Live");
  });

  it("draws a warm switch-back as a short load", () => {
    const [row] = buildTimeline([ev("loaded", 50, { data: { seconds: 0.01, warm: true } })], 60);
    expect(row.segments[0].phase).toBe("loading");
    expect(row.segments[row.segments.length - 1]?.phase).toBe("ready");
  });
});

describe("text helpers", () => {
  it("formats bytes and seconds", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(4.5 * 1024 ** 3)).toBe("4.5 GB");
    expect(formatSeconds(0.25)).toBe("250 ms");
    expect(formatSeconds(8.2)).toBe("8.20 s");
    expect(formatSeconds(75)).toBe("1 min 15 s");
  });

  it("summarises each kind", () => {
    expect(eventSummary(ev("loaded", 1, { data: { seconds: 8.2, memoryBytes: 4 * 1024 ** 3 } }), t)).toBe("Loaded in 8.20 s, 4.0 GB in memory");
    expect(eventSummary(ev("first_token", 1, { data: { prefillSeconds: 0.4 } }), t)).toBe("First token after 400 ms");
    expect(eventSummary(ev("generation", 1, { data: { completionTokens: 120, tokS: 41.27, seconds: 3 } }), t)).toBe("120 tokens at 41.3 tok/s in 3.00 s");
    expect(eventSummary(ev("unloaded", 1, { data: { reason: "swap" } }), t)).toBe("Unloaded (swap)");
    expect(eventSummary(ev("generation_failed", 1, { message: "engine died" }), t)).toBe("engine died");
  });

  it("names export files by date", () => {
    expect(exportFilename(new Date(2026, 10, 18, 8, 5))).toBe("model-events-20261118-0805.json");
  });
});

describe("parseSseChunk", () => {
  it("returns complete events and keeps the unfinished rest", () => {
    const one = JSON.stringify(ev("loaded", 1));
    const { events, rest } = parseSseChunk(`data: ${one}\n\n: keepalive\n\ndata: {"id":`);
    expect(events).toHaveLength(1);
    expect(rest).toBe('data: {"id":');
  });

  it("skips malformed lines", () => {
    expect(parseSseChunk("data: not json\n\n").events).toEqual([]);
  });
});
