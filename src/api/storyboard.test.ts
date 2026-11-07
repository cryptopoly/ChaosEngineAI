import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import {
  StoryboardApiError,
  StoryboardRevisionConflict,
  cancelStoryboardBatch,
  createStoryboardProject,
  deleteStoryboardProject,
  exportStoryboardBundle,
  fetchStoryboardFileSize,
  getStoryboardBatch,
  getStoryboardPeaks,
  getStoryboardRender,
  cancelStoryboardRender,
  importStoryboardBundle,
  listStoryboardProjects,
  parseRevisionConflict,
  redoStoryboard,
  runStoryboardOp,
  startStoryboardBatch,
  startStoryboardRender,
  storyboardFilePath,
  storyboardOps,
  undoStoryboard,
} from "./storyboard";

const API = "http://127.0.0.1:8876";

type Handler = (url: string, init: RequestInit) => Partial<Response>;

function stubFetch(handler: Handler) {
  const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
    if (url.endsWith("/api/auth/session")) {
      return { ok: true, status: 200, json: async () => ({ apiToken: "token" }) } as Response;
    }
    return { status: 200, ok: true, ...handler(url, init) } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function requestTo(fetchMock: ReturnType<typeof stubFetch>, path: string): RequestInit {
  const call = fetchMock.mock.calls.find(([url]) => url === `${API}${path}`);
  if (!call) throw new Error(`no request to ${path}`);
  return call[1] ?? {};
}

const payload = { project: { id: "abc", revision: 3, shots: [] }, timeline: { total: 0, shots: [], audio: [], problems: [] } };

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("storyboard api client", () => {
  it("lists projects", async () => {
    stubFetch(() => ({ json: async () => ({ projects: [{ id: "a", name: "A", revision: 1, shots: 0, createdAt: 1 }] }) }));
    await expect(listStoryboardProjects()).resolves.toHaveLength(1);
  });

  it("creates a project with its defaults", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => payload }));
    await createStoryboardProject("Film", { width: 720, height: 1280, fps: 24, exportPreset: "9:16" });
    const init = requestTo(fetchMock, "/api/storyboard/projects");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      name: "Film",
      defaults: { width: 720, height: 1280, fps: 24, exportPreset: "9:16" },
    });
  });

  it("sends the revision, op name and args with every op", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => payload }));
    await runStoryboardOp("abc", 3, "lock", { shotId: "s_1", locked: true });
    expect(JSON.parse(requestTo(fetchMock, "/api/storyboard/projects/abc/ops").body as string)).toEqual({
      revision: 3,
      op: "lock",
      args: { shotId: "s_1", locked: true },
    });
  });

  it("has a typed wrapper per op that posts the same body", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => payload }));
    await storyboardOps.reorder("abc", 4, { order: ["b", "a"] });
    await storyboardOps.importTake("abc", 5, { shotId: "s_1", clip: "/x/clip.mp4" });
    const bodies = fetchMock.mock.calls
      .filter(([url]) => String(url).endsWith("/ops"))
      .map(([, init]) => JSON.parse((init as RequestInit).body as string));
    expect(bodies).toEqual([
      { revision: 4, op: "reorder", args: { order: ["b", "a"] } },
      { revision: 5, op: "import_take", args: { shotId: "s_1", clip: "/x/clip.mp4" } },
    ]);
  });

  it("sends the revision with undo, redo and delete", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => payload }));
    await undoStoryboard("abc", 7);
    await redoStoryboard("abc", 8);
    await deleteStoryboardProject("abc", 9);
    expect(JSON.parse(requestTo(fetchMock, "/api/storyboard/projects/abc/undo").body as string)).toEqual({ revision: 7 });
    expect(JSON.parse(requestTo(fetchMock, "/api/storyboard/projects/abc/redo").body as string)).toEqual({ revision: 8 });
    expect(requestTo(fetchMock, "/api/storyboard/projects/abc?revision=9").method).toBe("DELETE");
  });

  it("starts, polls and cancels a batch; starts a render", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({ batchId: "b_1", batch: { id: "b_1" }, revision: 4, cancelled: true }) }));
    await startStoryboardBatch("abc", { revision: 3, shotIds: ["s_1"] });
    await getStoryboardBatch("abc", "b_1");
    await cancelStoryboardBatch("abc", "b_1");
    await startStoryboardRender("abc", 3, false);
    expect(JSON.parse(requestTo(fetchMock, "/api/storyboard/projects/abc/batches").body as string)).toEqual({
      revision: 3,
      what: "both",
      shotIds: ["s_1"],
      force: false,
    });
    expect(requestTo(fetchMock, "/api/storyboard/projects/abc/batches/b_1").method).toBe("GET");
    expect(requestTo(fetchMock, "/api/storyboard/projects/abc/batches/b_1/cancel").method).toBe("POST");
    expect(JSON.parse(requestTo(fetchMock, "/api/storyboard/projects/abc/render").body as string)).toEqual({
      revision: 3,
      preview: false,
    });
  });

  it("imports a bundle as raw bytes or as a local path", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => payload }));
    const blob = new Blob(["zip"]);
    await importStoryboardBundle(blob);
    await importStoryboardBundle({ path: "/x/film.chaosfilm" });
    const calls = fetchMock.mock.calls.filter(([url]) => url === `${API}/api/storyboard/import`);
    expect((calls[0][1] as RequestInit).body).toBe(blob);
    expect(new Headers((calls[0][1] as RequestInit).headers).get("Content-Type")).toBe("application/octet-stream");
    expect(JSON.parse((calls[1][1] as RequestInit).body as string)).toEqual({ path: "/x/film.chaosfilm" });
  });

  it("exports a bundle", async () => {
    stubFetch(() => ({ json: async () => ({ file: "exports/film.chaosfilm", bytes: 12 }) }));
    await expect(exportStoryboardBundle("abc")).resolves.toEqual({ file: "exports/film.chaosfilm", bytes: 12 });
  });

  it("builds an encoded project file path", () => {
    expect(storyboardFilePath("abc", "media/s_1/t_1/still.png")).toBe(
      "/api/storyboard/projects/abc/files?path=media%2Fs_1%2Ft_1%2Fstill.png",
    );
  });
});

describe("revision conflicts", () => {
  const current = { id: "abc", revision: 9, shots: [] };

  it("turns a 409 with a project into a typed conflict that carries the fresh project", async () => {
    stubFetch(() => ({
      ok: false,
      status: 409,
      json: async () => ({ detail: { message: "stale", revision: 9, project: current } }),
      clone() {
        return this as unknown as Response;
      },
    }));
    const error = await runStoryboardOp("abc", 3, "lock", { shotId: "s_1" }).catch((err) => err);
    expect(error).toBeInstanceOf(StoryboardRevisionConflict);
    expect((error as StoryboardRevisionConflict).project).toEqual(current);
    expect((error as StoryboardRevisionConflict).revision).toBe(9);
    expect((error as StoryboardRevisionConflict).status).toBe(409);
  });

  it("keeps a 409 without a project (a busy render) as a plain api error", async () => {
    stubFetch(() => ({
      ok: false,
      status: 409,
      json: async () => ({ detail: "A render is already running." }),
      clone() {
        return this as unknown as Response;
      },
    }));
    const error = await startStoryboardRender("abc", 3).catch((err) => err);
    expect(error).toBeInstanceOf(StoryboardApiError);
    expect(error).not.toBeInstanceOf(StoryboardRevisionConflict);
    expect((error as Error).message).toBe("A render is already running.");
  });

  it("parses conflict bodies and ignores other statuses and shapes", () => {
    expect(parseRevisionConflict(409, { detail: { message: "x", project: current } })?.revision).toBe(9);
    expect(parseRevisionConflict(400, { detail: { project: current } })).toBeNull();
    expect(parseRevisionConflict(409, { detail: "text" })).toBeNull();
    expect(parseRevisionConflict(409, null)).toBeNull();
  });

  it("reports other failures with the server's words and status", async () => {
    stubFetch(() => ({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Unknown shot 'x'." }),
      clone() {
        return this as unknown as Response;
      },
    }));
    const error = await runStoryboardOp("abc", 3, "lock", { shotId: "x" }).catch((err) => err);
    expect(error).toBeInstanceOf(StoryboardApiError);
    expect((error as StoryboardApiError).status).toBe(400);
    expect((error as Error).message).toBe("Unknown shot 'x'.");
  });

  it("asks for waveform peaks of a project file with the path encoded and the buckets bounded", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({ peaks: [0, 0.5, 1], durationS: 1.5 }) }));
    const result = await getStoryboardPeaks("abc", "media/audio/a 1.wav", 120);
    expect(result).toEqual({ peaks: [0, 0.5, 1], durationS: 1.5 });
    expect(requestTo(fetchMock, "/api/storyboard/projects/abc/peaks?path=media%2Faudio%2Fa%201.wav&buckets=120").method).toBe("GET");
    await getStoryboardPeaks("abc", "media/audio/a.wav", 5);
    await getStoryboardPeaks("abc", "media/audio/a.wav", 99999);
    await getStoryboardPeaks("abc", "media/audio/a.wav");
    const urls = fetchMock.mock.calls.map(([url]) => url as string);
    expect(urls.some((u) => u.endsWith("a.wav&buckets=32"))).toBe(true);
    expect(urls.some((u) => u.endsWith("a.wav&buckets=4000"))).toBe(true);
    expect(urls.some((u) => u.endsWith("a.wav&buckets=400"))).toBe(true);
  });

  it("raises the server's words when the peaks cannot be read", async () => {
    stubFetch(() => ({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Not a file of this project." }),
      clone() {
        return this as unknown as Response;
      },
    }));
    const error = await getStoryboardPeaks("abc", "media/nope.wav").catch((err) => err);
    expect(error).toBeInstanceOf(StoryboardApiError);
    expect((error as StoryboardApiError).status).toBe(404);
    expect((error as Error).message).toBe("Not a file of this project.");
  });

  it("reads the render status and cancels a render", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({ render: { renderId: "r_1", state: "running" }, cancelled: true }) }));
    expect((await getStoryboardRender("abc")).render?.renderId).toBe("r_1");
    await cancelStoryboardRender("abc");
    expect(requestTo(fetchMock, "/api/storyboard/projects/abc/render").method).toBe("GET");
    expect(requestTo(fetchMock, "/api/storyboard/projects/abc/render/cancel").method).toBe("POST");
  });

  it("reads a file size from the range response, or by reading the file when the header is hidden", async () => {
    const withHeader = stubFetch(() => ({ status: 206, headers: new Headers({ "Content-Range": "bytes 0-0/8303793" }), blob: async () => new Blob(["x"]) }));
    await expect(fetchStoryboardFileSize("abc", "exports/f.mp4")).resolves.toBe(8303793);
    const sent = requestTo(withHeader, "/api/storyboard/projects/abc/files?path=exports%2Ff.mp4");
    expect(new Headers(sent.headers).get("Range")).toBe("bytes=0-0");

    let calls = 0;
    stubFetch((_url, init) => {
      calls += 1;
      const ranged = new Headers(init.headers).get("Range") !== null;
      return { status: ranged ? 206 : 200, headers: new Headers(), blob: async () => new Blob([ranged ? "x" : "twelve bytes"]) };
    });
    await expect(fetchStoryboardFileSize("abc", "exports/f.mp4")).resolves.toBe(12);
    expect(calls).toBe(2);

    stubFetch(() => ({ status: 200, headers: new Headers(), blob: async () => new Blob(["abcd"]) }));
    await expect(fetchStoryboardFileSize("abc", "exports/f.mp4")).resolves.toBe(4);
  });

  it("raises when the file cannot be read for its size", async () => {
    stubFetch(() => ({ ok: false, status: 404, headers: new Headers() }));
    await expect(fetchStoryboardFileSize("abc", "exports/gone.mp4")).rejects.toBeInstanceOf(StoryboardApiError);
  });
});
