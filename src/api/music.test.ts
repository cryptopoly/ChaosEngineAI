import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import {
  MusicApiError,
  cancelMusicGeneration,
  deleteMusicLibraryItems,
  downloadMusicModel,
  estimateMusicRun,
  fetchMusicAudioBlob,
  generateMusic,
  getMusicCatalog,
  getMusicLibrary,
  getMusicProgress,
  getMusicRuntime,
  musicAudioPath,
  saveMusicEmbedTag,
  saveMusicSettings,
  startMusicInstall,
  updateMusicLibraryItem,
} from "./music";

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

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("music api client", () => {
  it("reads the catalog and unwraps the runtime and progress envelopes", async () => {
    stubFetch((url) => {
      if (url.endsWith("/api/music/catalog")) return { json: async () => ({ families: [{ id: "f" }] }) };
      if (url.endsWith("/api/music/runtime")) return { json: async () => ({ runtime: { installed: true } }) };
      return { json: async () => ({ progress: { active: false, phase: "idle" } }) };
    });
    expect((await getMusicCatalog()).families).toHaveLength(1);
    expect((await getMusicRuntime()).installed).toBe(true);
    expect((await getMusicProgress()).phase).toBe("idle");
  });

  it("posts the estimate inputs and sends the bearer token", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({ estimate: { seconds: 5 }, fit: null, totalGb: 64, queue: { running: null, waiting: [] } }) }));
    const response = await estimateMusicRun("m", 60, 2);
    expect(response.estimate.seconds).toBe(5);
    const init = requestTo(fetchMock, "/api/music/estimate");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ modelId: "m", seconds: 60, variations: 2 });
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer token");
  });

  it("posts the generate payload as it is", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({ items: [], seed: 1, jobId: "j", model: "m", seconds: 5 }) }));
    await generateMusic({ caption: "lo-fi", seconds: 30, instrumental: true, variations: 1 });
    expect(JSON.parse(String(requestTo(fetchMock, "/api/music/generate").body))).toEqual({
      caption: "lo-fi",
      seconds: 30,
      instrumental: true,
      variations: 1,
    });
  });

  it("surfaces the backend's plain detail with the status", async () => {
    stubFetch(() => ({
      ok: false,
      status: 503,
      clone: () => ({ json: async () => ({}) }) as Response,
      json: async () => ({ detail: "Not enough free memory for music: unload the image model first." }),
    }));
    const error = await generateMusic({ caption: "x", seconds: 30 }).catch((err: unknown) => err);
    expect(error).toBeInstanceOf(MusicApiError);
    expect((error as MusicApiError).message).toBe("Not enough free memory for music: unload the image model first.");
    expect((error as MusicApiError).status).toBe(503);
  });

  it("surfaces a cancelled run as a 409 'cancelled'", async () => {
    stubFetch(() => ({ ok: false, status: 409, json: async () => ({ detail: "cancelled" }) }));
    await expect(generateMusic({ caption: "x", seconds: 30 })).rejects.toMatchObject({ message: "cancelled", status: 409 });
  });

  it("cancels, and lists the library with its filters in the query", async () => {
    const fetchMock = stubFetch((url) =>
      url.includes("/cancel")
        ? { json: async () => ({ cancelled: true }) }
        : { json: async () => ({ items: [], count: 0, diskBytes: 0 }) },
    );
    expect(await cancelMusicGeneration()).toBe(true);
    await getMusicLibrary("lo fi", "piano");
    expect(fetchMock.mock.calls.some(([url]) => url === `${API}/api/music/library?q=lo+fi&tag=piano`)).toBe(true);
  });

  it("renames, tags and deletes library items", async () => {
    const fetchMock = stubFetch((url) =>
      url.endsWith("/library/delete") ? { json: async () => ({ deleted: ["a"] }) } : { json: async () => ({ id: "1_aaaaaaaa", title: "New" }) },
    );
    expect((await updateMusicLibraryItem("1_aaaaaaaa", { title: "New", tags: ["x"] })).title).toBe("New");
    expect(JSON.parse(String(requestTo(fetchMock, "/api/music/library/1_aaaaaaaa").body))).toEqual({ title: "New", tags: ["x"] });
    expect(await deleteMusicLibraryItems(["a"])).toEqual(["a"]);
    expect(JSON.parse(String(requestTo(fetchMock, "/api/music/library/delete").body))).toEqual({ ids: ["a"] });
  });

  it("fetches a track's audio with the token, so it can play through a blob URL", async () => {
    const wav = new Blob(["fLaC"], { type: "audio/flac" });
    const fetchMock = stubFetch(() => ({ blob: async () => wav }));
    expect(await fetchMusicAudioBlob("1_aaaaaaaa")).toBe(wav);
    expect(musicAudioPath("1_aaaaaaaa")).toBe("/api/music/library/1_aaaaaaaa/audio");
    expect(new Headers(requestTo(fetchMock, "/api/music/library/1_aaaaaaaa/audio").headers).get("Authorization")).toBe("Bearer token");
  });

  it("starts the install, saves the idle minutes, and writes the AI tag switch to the app settings", async () => {
    const fetchMock = stubFetch((url) => (url.endsWith("/api/settings") ? { json: async () => ({ settings: {} }) } : { json: async () => ({ running: true, idleUnloadMinutes: 5 }) }));
    expect((await startMusicInstall()).running).toBe(true);
    await saveMusicSettings({ idleUnloadMinutes: 5 });
    expect(JSON.parse(String(requestTo(fetchMock, "/api/music/settings").body))).toEqual({ idleUnloadMinutes: 5 });
    await saveMusicEmbedTag(false);
    const init = requestTo(fetchMock, "/api/settings");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(String(init.body))).toEqual({ musicEmbedTag: false });
  });

  it("starts a weight download by repo", async () => {
    const fetchMock = stubFetch(() => ({ json: async () => ({ download: { repo: "r", state: "downloading" } }) }));
    expect((await downloadMusicModel("r")).state).toBe("downloading");
    expect(JSON.parse(String(requestTo(fetchMock, "/api/music/download").body))).toEqual({ repo: "r" });
  });
});
