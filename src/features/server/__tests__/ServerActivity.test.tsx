import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ConnectPanel, CONNECT_OPEN_KEY, readConnectOpen } from "../ConnectPanel";
import { EventList } from "../EventList";
import { ModelTimeline } from "../ModelTimeline";
import { ServerActivity } from "../ServerActivity";
import { readServerView, SERVER_VIEW_KEY, writeServerView } from "../serverView";
import type { ModelEvent } from "../modelEvents";
import type { ModelEventsState } from "../../../hooks/useModelEvents";

function storage(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => void values.set(key, value),
    removeItem: (key: string) => void values.delete(key),
    values,
  };
}

function stubWindow(initial: Record<string, string> = {}) {
  const localStorage = storage(initial);
  const dispatched: Event[] = [];
  vi.stubGlobal("window", {
    localStorage,
    dispatchEvent: (event: Event) => {
      dispatched.push(event);
      return true;
    },
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  });
  return { localStorage, dispatched };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

let nextId = 1;
function ev(kind: string, ts: number, extra: Partial<ModelEvent> = {}): ModelEvent {
  return {
    id: nextId++,
    ts,
    time: "2026-11-18 08:00:00",
    kind,
    level: "info",
    model: "Qwen3 8B",
    modelRef: "qwen/q",
    requestId: "req-1",
    engine: "mlx",
    cacheStrategy: "native",
    speculative: null,
    message: kind,
    data: {},
    detail: {},
    ...extra,
  };
}

const connectProps = {
  localServerUrl: "http://127.0.0.1:8904/v1",
  apiToken: "tok",
  connectModel: "qwen/q",
  onCopy: () => undefined,
};

describe("ConnectPanel", () => {
  it("is one closed line with the address and Copy by default", () => {
    stubWindow();
    expect(readConnectOpen()).toBe(false);
    const html = renderToStaticMarkup(<ConnectPanel {...connectProps} />);
    expect(html).toContain("Connect your app");
    expect(html).toContain("http://127.0.0.1:8904/v1");
    expect(html).toContain('aria-expanded="false"');
    expect(html).not.toContain("from openai import OpenAI");
  });

  it("opens with the snippets when the user left it open", () => {
    stubWindow({ [CONNECT_OPEN_KEY]: "1" });
    const html = renderToStaticMarkup(<ConnectPanel {...connectProps} />);
    expect(html).toContain('aria-expanded="true"');
    expect(html).toContain("from openai import OpenAI");
  });

  it("reads closed when storage is unavailable", () => {
    vi.stubGlobal("window", {});
    expect(readConnectOpen()).toBe(false);
  });
});

describe("server view memory", () => {
  it("defaults to overview, remembers activity and tells an open page", () => {
    const { localStorage, dispatched } = stubWindow();
    vi.stubGlobal("CustomEvent", class extends Event {
      detail: unknown;
      constructor(type: string, init?: { detail?: unknown }) {
        super(type);
        this.detail = init?.detail;
      }
    });
    expect(readServerView()).toBe("overview");
    writeServerView("activity");
    expect(localStorage.values.get(SERVER_VIEW_KEY)).toBe("activity");
    expect(readServerView()).toBe("activity");
    expect(dispatched).toHaveLength(1);
  });

  it("ignores a stored value it does not know", () => {
    stubWindow({ [SERVER_VIEW_KEY]: "logs" });
    expect(readServerView()).toBe("overview");
  });
});

describe("EventList", () => {
  it("shows rows newest first and the expanded detail with command, output and environment", () => {
    const failed = ev("load_failed", 10, {
      level: "error",
      message: "worker exited",
      detail: { command: "llama-server -m ~/m.gguf --api-key ***", stderr: ["boom"], env: ["CHAOSENGINE_PORT=8904"], error: "RuntimeError: worker exited" },
    });
    const html = renderToStaticMarkup(
      <EventList
        events={[failed, ev("requested", 1)]}
        selected={new Set([failed.id])}
        expandedId={failed.id}
        onToggleSelected={() => undefined}
        onToggleExpanded={() => undefined}
        onCopy={() => undefined}
      />,
    );
    expect(html.indexOf("Load failed")).toBeLessThan(html.indexOf("Requested"));
    expect(html).toContain("level-error");
    expect(html).toContain("llama-server -m ~/m.gguf --api-key ***");
    expect(html).toContain("CHAOSENGINE_PORT=8904");
    expect(html).toContain("boom");
    expect(html).toContain('checked=""');
  });

  it("says what shows here when empty", () => {
    const html = renderToStaticMarkup(
      <EventList events={[]} selected={new Set()} expandedId={null} onToggleSelected={() => undefined} onToggleExpanded={() => undefined} onCopy={() => undefined} />,
    );
    expect(html).toContain("No events match");
  });
});

describe("ModelTimeline", () => {
  it("draws one track per model with its phases", () => {
    const now = Date.now() / 1000;
    const html = renderToStaticMarkup(
      <ModelTimeline
        now={now}
        events={[ev("loading", now - 30), ev("loaded", now - 20, { data: { seconds: 10 } }), ev("generation", now - 5, { data: { seconds: 3 } })]}
      />,
    );
    expect(html).toContain("Qwen3 8B");
    expect(html).toContain("phase-loading");
    expect(html).toContain("phase-generating");
    expect(html).toContain("phase-ready");
    expect(html).toMatch(/aria-label="Qwen3 8B: Loading/);
  });

  it("has an empty state", () => {
    expect(renderToStaticMarkup(<ModelTimeline now={1000} events={[]} />)).toContain("No model activity");
  });
});

describe("ServerActivity", () => {
  const feed = (events: ModelEvent[], extra: Partial<ModelEventsState> = {}): ModelEventsState => ({
    events,
    connected: true,
    paused: false,
    pending: 0,
    pause: () => undefined,
    resume: () => undefined,
    ...extra,
  });

  it("renders the timeline, filters, actions, events and the old log lines", () => {
    const html = renderToStaticMarkup(
      <ServerActivity
        feed={feed([ev("loaded", Date.now() / 1000, { data: { seconds: 2, memoryBytes: 1024 } })])}
        logEntries={[{ ts: "2026-11-18 08:00:00", source: "runtime", level: "info", message: "Model loaded: Qwen3 8B via mlx." }]}
        onCopy={() => undefined}
      />,
    );
    for (const text of ["Model timeline", "All models", "All levels", "Pause", "Latest failure", "Copy selected (0)", "Export selected (0)", "Live", "Log lines", "Model loaded: Qwen3 8B via mlx."]) {
      expect(html).toContain(text);
    }
    // No failure yet: the shortcut is disabled.
    expect(html).toMatch(/<button type="button" class="secondary-button" disabled="">Latest failure<\/button>/);
  });

  it("shows the paused count", () => {
    const html = renderToStaticMarkup(
      <ServerActivity feed={feed([], { paused: true, pending: 3 })} logEntries={[]} onCopy={() => undefined} />,
    );
    expect(html).toContain("Paused, 3 new");
    expect(html).toContain("Resume");
  });
});
