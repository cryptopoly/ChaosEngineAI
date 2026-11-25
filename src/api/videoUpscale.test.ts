import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import {
  fetchVideoUpscaleFileUrl,
  fetchVideoUpscaleFrameUrl,
  getVideoUpscaleModels,
  listVideoUpscaleJobs,
  probeVideoUpscaleInput,
} from "./videoUpscale";

const API = "http://127.0.0.1:8876";

function stubFetch(respond: (url: string, init?: RequestInit) => Partial<Response>) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/api/auth/session")) {
      return { ok: true, status: 200, json: async () => ({ apiToken: "token" }) } as Response;
    }
    return respond(url, init) as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: vi.fn(() => "blob:made"), revokeObjectURL: vi.fn() }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("video upscale client", () => {
  it("posts the path to the probe route and returns what it read", async () => {
    const body = { inputPath: "C:/v/tape.mp4", width: 720, height: 576, fps: 25, durationSeconds: 60, frames: 1500, hasAudio: true, interlaced: null };
    const fetchMock = stubFetch(() => ({ ok: true, status: 200, json: async () => body }));
    await expect(probeVideoUpscaleInput("C:/v/tape.mp4")).resolves.toEqual(body);
    const call = fetchMock.mock.calls.find(([url]) => url === `${API}/api/video/upscale/probe`);
    expect(call?.[1]?.method).toBe("POST");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({ inputPath: "C:/v/tape.mp4" });
  });

  it("shows a probe error word for word", async () => {
    stubFetch(() => ({
      ok: false, status: 400, statusText: "Bad Request",
      json: async () => ({ detail: "File not found: tape.mp4" }),
      text: async () => JSON.stringify({ detail: "File not found: tape.mp4" }),
    }));
    await expect(probeVideoUpscaleInput("C:/v/tape.mp4")).rejects.toThrow("File not found: tape.mp4");
  });

  it("lists the jobs, and reads deinterlaceAvailable from the models", async () => {
    stubFetch((url) => ({
      ok: true, status: 200,
      json: async () => (url.endsWith("/jobs") ? { jobs: [{ id: "a" }, { id: "b" }] } : { models: [], addonInstalled: true, deinterlaceAvailable: false }),
    }));
    expect((await listVideoUpscaleJobs()).map((job) => job.id)).toEqual(["a", "b"]);
    expect((await getVideoUpscaleModels()).deinterlaceAvailable).toBe(false);
  });

  it("fetches a frame or the film with the token, as a blob URL", async () => {
    const fetchMock = stubFetch(() => ({ ok: true, status: 200, blob: async () => new Blob(["x"]) }));
    await expect(fetchVideoUpscaleFrameUrl("job 1", "input")).resolves.toBe("blob:made");
    await expect(fetchVideoUpscaleFileUrl("job 1")).resolves.toBe("blob:made");
    const urls = fetchMock.mock.calls.map(([url]) => url);
    expect(urls).toContain(`${API}/api/video/upscale/jobs/job%201/frame?which=input`);
    expect(urls).toContain(`${API}/api/video/upscale/jobs/job%201/file`);
    const frameCall = fetchMock.mock.calls.find(([url]) => String(url).includes("/frame"));
    expect(new Headers(frameCall?.[1]?.headers).get("Authorization")).toBe("Bearer token");
  });

  it("answers null on a 404 (the backend restarted) and an error on other failures", async () => {
    stubFetch(() => ({ ok: false, status: 404 }));
    await expect(fetchVideoUpscaleFrameUrl("a", "output")).resolves.toBeNull();
    await expect(fetchVideoUpscaleFileUrl("a")).resolves.toBeNull();
    stubFetch(() => ({
      ok: false, status: 500, statusText: "Server Error",
      json: async () => ({ detail: "ffmpeg failed" }),
      text: async () => JSON.stringify({ detail: "ffmpeg failed" }),
    }));
    await expect(fetchVideoUpscaleFileUrl("a")).rejects.toThrow("ffmpeg failed");
  });
});
