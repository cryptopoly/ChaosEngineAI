import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

// FU-195: a dev page built with VITE_CHAOSENGINE_API_TOKEN, against a backend that restarted with another token.
const BASE = "http://127.0.0.1:8971";

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function authOf(init: RequestInit | undefined): string | null {
  return new Headers(init?.headers ?? {}).get("Authorization");
}

async function loadApi() {
  vi.resetModules();
  return import("./index");
}

describe("apiFetch with a build-time token after a backend restart", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_CHAOSENGINE_API_BASE", BASE);
    vi.stubEnv("VITE_CHAOSENGINE_API_TOKEN", "old");
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("reads the session token once on a 401, retries once, and keeps the new token", async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.endsWith("/api/auth/session")) return json(200, { apiToken: "new" });
      return authOf(init) === "Bearer new" ? json(200, { ok: true }) : json(401, { detail: "no" });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { apiFetch } = await loadApi();

    expect((await apiFetch("/api/workspace")).status).toBe(200);
    expect(fetchMock.mock.calls.map(([url, init]) => [url.replace(BASE, ""), authOf(init)])).toEqual([
      ["/api/workspace", "Bearer old"],
      ["/api/auth/session", null],
      ["/api/workspace", "Bearer new"],
    ]);
    fetchMock.mockClear();
    expect((await apiFetch("/api/workspace")).status).toBe(200);
    expect(fetchMock.mock.calls.map(([, init]) => authOf(init))).toEqual(["Bearer new"]);
    expect(console.warn).toHaveBeenCalledTimes(1);
  });

  it("heals several polls refused at once, each retried once, with one console hint", async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.endsWith("/api/auth/session")) return json(200, { apiToken: "new" });
      return authOf(init) === "Bearer new" ? json(200, { ok: true }) : json(401, { detail: "no" });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { apiFetch } = await loadApi();

    const replies = await Promise.all(["/a", "/b", "/c"].map((path) => apiFetch(path)));
    expect(replies.map((r) => r.status)).toEqual([200, 200, 200]);
    expect(fetchMock.mock.calls.filter(([url]) => !url.endsWith("/api/auth/session"))).toHaveLength(6);
    expect(console.warn).toHaveBeenCalledTimes(1);
  });

  it("does not loop when the backend still refuses: one session read and one retry per request", async () => {
    const fetchMock = vi.fn(async (url: string) =>
      url.endsWith("/api/auth/session") ? json(403, { detail: "origin" }) : json(401, { detail: "no" }));
    vi.stubGlobal("fetch", fetchMock);
    const { apiFetch } = await loadApi();

    expect((await apiFetch("/api/workspace")).status).toBe(401);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(authOf(fetchMock.mock.calls[2][1])).toBe("Bearer old"); // nothing better was found: keep it
    expect(console.warn).not.toHaveBeenCalled();
  });

  it("keeps the build-time token when the backend still uses it", async () => {
    const fetchMock = vi.fn(async () => json(200, { ok: true }));
    vi.stubGlobal("fetch", fetchMock);
    const { apiFetch } = await loadApi();

    await apiFetch("/api/workspace");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(authOf(fetchMock.mock.calls[0][1])).toBe("Bearer old");
  });
});
