import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import { getMfluxInstallStatus, getMfluxStatus, startMfluxInstall } from "./setup";

const API = "http://127.0.0.1:8876";

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    if (url.endsWith("/api/auth/session")) {
      return { ok: true, status: 200, json: async () => ({ apiToken: "token" }) } as Response;
    }
    return { ok: true, status: 200, json: async () => body } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("mflux setup client", () => {
  it("reads the install status", async () => {
    const fetchMock = stubFetch({ installed: true, version: "0.20.0", supported: true, repos: ["Qwen/Qwen-Image-2.1"] });
    const status = await getMfluxStatus();
    expect(status.version).toBe("0.20.0");
    expect(fetchMock.mock.calls.some(([url]) => url === `${API}/api/setup/mflux-status`)).toBe(true);
  });

  it("starts the install with a POST", async () => {
    const fetchMock = stubFetch({ id: "mflux-install", phase: "preflight", done: false });
    const job = await startMfluxInstall();
    expect(job.phase).toBe("preflight");
    const call = fetchMock.mock.calls.find(([url]) => url === `${API}/api/setup/install-mflux`);
    expect(call?.[1]?.method).toBe("POST");
  });

  it("polls the running job", async () => {
    stubFetch({ id: "mflux-install", phase: "installing", done: false, percent: 50 });
    const job = await getMfluxInstallStatus();
    expect(job.percent).toBe(50);
  });
});
