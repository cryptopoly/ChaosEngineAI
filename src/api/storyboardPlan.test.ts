import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { StoryboardApiError } from "./storyboard";
import {
  directStoryboardPlan,
  getStoryboardFonts,
  getStoryboardPlanStatus,
  proposeStoryboardPlan,
  reworkStoryboardShot,
} from "./storyboardPlan";

const API = "http://127.0.0.1:8876";

function stubFetch(handler: (url: string, init: RequestInit) => Partial<Response>) {
  const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
    if (url.endsWith("/api/auth/session")) return { ok: true, status: 200, json: async () => ({ apiToken: "token" }) } as Response;
    return { status: 200, ok: true, ...handler(url, init) } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function body(fetchMock: ReturnType<typeof stubFetch>, path: string): unknown {
  const call = fetchMock.mock.calls.find(([url]) => url === `${API}${path}`);
  if (!call) throw new Error(`no request to ${path}`);
  return JSON.parse(String(call[1]?.body ?? "null"));
}

afterEach(() => vi.unstubAllGlobals());

describe("plan client (S8, S15)", () => {
  it("sends the brief, the length and the shot count only when set", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({ shots: [], note: "", model: "m", seconds: 1, sawPictures: false }) }));
    await proposeStoryboardPlan("p1", { brief: "tea", filmSeconds: 60, shots: null });
    expect(body(fetchMock, "/api/storyboard/projects/p1/plan/propose")).toEqual({ brief: "tea", filmSeconds: 60 });
  });

  it("sends a rework with its history and a direction turn", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({}) }));
    await reworkStoryboardShot("p1", "s_1", "closer", [{ role: "user", text: "warmer" }]);
    expect(body(fetchMock, "/api/storyboard/projects/p1/plan/rework")).toEqual({ shotId: "s_1", message: "closer", history: [{ role: "user", text: "warmer" }] });
    await directStoryboardPlan("p1", "end on the sign");
    expect(body(fetchMock, "/api/storyboard/projects/p1/plan/direct")).toEqual({ message: "end on the sign" });
  });

  it("reads the status and the fonts", async () => {
    stubFetch((url) => ({
      json: async () => (url.endsWith("/fonts") ? { fonts: ["Avenir"] } : { chatModel: null, style: { mode: "none", stillModel: null, pictureSlots: 0, words: "", problem: null } }),
    }));
    expect((await getStoryboardPlanStatus("p1")).chatModel).toBeNull();
    expect(await getStoryboardFonts()).toEqual(["Avenir"]);
  });

  it("carries the no-model code so the tab can offer a load", async () => {
    stubFetch(() => ({
      ok: false,
      status: 409,
      json: async () => ({ detail: { code: "no-model", message: "No chat model is loaded. Load one to plan with it." } }),
      clone() {
        return this as Response;
      },
      text: async () => JSON.stringify({ detail: { code: "no-model", message: "No chat model is loaded. Load one to plan with it." } }),
    }));
    const failure = await proposeStoryboardPlan("p1", { brief: "tea" }).catch((err: unknown) => err);
    expect(failure).toBeInstanceOf(StoryboardApiError);
    expect((failure as StoryboardApiError).code).toBe("no-model");
    expect((failure as StoryboardApiError).message).toContain("No chat model is loaded");
  });
});
