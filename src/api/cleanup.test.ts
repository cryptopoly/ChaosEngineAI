import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { deleteCleanupItems, getCleanupItems } from "./cleanup";

const API = "http://127.0.0.1:8876";

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async (url: string) => {
    if (url.endsWith("/api/auth/session")) return { ok: true, status: 200, json: async () => ({ apiToken: "token" }) } as Response;
    return { ok: true, status: 200, json: async () => body } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe("cleanup api", () => {
  it("GETs the list", async () => {
    const fetchMock = stubFetch({ projects: [], threads: [], outputs: [] });
    expect(await getCleanupItems()).toEqual({ projects: [], threads: [], outputs: [] });
    expect(fetchMock.mock.calls.some(([url]) => url === `${API}/api/cleanup/items`)).toBe(true);
  });

  it("POSTs only the three id lists", async () => {
    const fetchMock = stubFetch({ results: [], deleted: 0, refused: 0, freedBytes: 0 });
    await deleteCleanupItems({ projects: ["p"], threads: [], outputs: ["image:a.json"] });
    const call = fetchMock.mock.calls.find(([url]) => url === `${API}/api/cleanup/delete`) as unknown as [string, RequestInit];
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(String(call[1].body))).toEqual({ projects: ["p"], threads: [], outputs: ["image:a.json"] });
  });
});
