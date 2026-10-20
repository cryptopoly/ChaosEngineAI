import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import {
  deleteGalleryItem,
  getGalleryAudio,
  getKokoroDownloadStatus,
  getVoiceGallery,
  saveGalleryAudio,
  startKokoroDownload,
  synthesizeSpeech,
  transcribeAudio,
} from "./voice";

const API = "http://127.0.0.1:8876";

type Handler = (url: string, init: RequestInit) => Partial<Response>;

/** Route fetch by URL: the auth-session probe always succeeds, every
 * other request goes to the per-test handler. */
function stubFetch(handler: Handler) {
  const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
    if (url.endsWith("/api/auth/session")) {
      return { ok: true, status: 200, json: async () => ({ apiToken: "token" }) } as Response;
    }
    return { status: 200, ...handler(url, init) } as Response;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The RequestInit of the first request made to ``path``. */
function requestTo(fetchMock: ReturnType<typeof stubFetch>, path: string): RequestInit {
  const call = fetchMock.mock.calls.find(([url]) => url === `${API}${path}`);
  if (!call) throw new Error(`no request to ${path}`);
  return call[1] ?? {};
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("voice api client", () => {
  it("posts recorded audio as multipart form data", async () => {
    const fetchMock = stubFetch(() => ({
      ok: true,
      json: async () => ({ text: "hello", duration_s: 0.4 }),
    }));
    const form = new FormData();
    form.append("audio", new Blob(["abc"], { type: "audio/webm" }), "recording.webm");

    await expect(transcribeAudio(form)).resolves.toEqual({ text: "hello", duration_s: 0.4 });

    const init = requestTo(fetchMock, "/api/voice/transcribe");
    expect(init.method).toBe("POST");
    expect(init.body).toBe(form);
  });

  it("surfaces the backend error body when transcription fails", async () => {
    stubFetch(() => ({ ok: false, status: 503, text: async () => "No STT backend available." }));
    await expect(transcribeAudio(new FormData())).rejects.toThrow("No STT backend available.");
  });

  it("falls back to a status message when the error body is empty", async () => {
    stubFetch(() => ({ ok: false, status: 500, text: async () => "" }));
    await expect(transcribeAudio(new FormData())).rejects.toThrow(
      "Transcription failed with status 500",
    );
  });

  it("sends synthesis parameters as JSON and returns the audio blob", async () => {
    const wav = new Blob(["RIFF"], { type: "audio/wav" });
    const fetchMock = stubFetch(() => ({ ok: true, blob: async () => wav }));

    await expect(synthesizeSpeech("Hi there", "bf_emma", 1.25)).resolves.toBe(wav);

    const init = requestTo(fetchMock, "/api/voice/synthesize");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ text: "Hi there", voice: "bf_emma", speed: 1.25 });
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
  });

  it("throws the backend detail when synthesis fails", async () => {
    stubFetch(() => ({ ok: false, status: 400, text: async () => "Unknown voice 'zz'" }));
    await expect(synthesizeSpeech("x", "zz", 1)).rejects.toThrow("Unknown voice 'zz'");
  });

  it("unwraps the gallery item list", async () => {
    const items = [{ id: "1_aaaaaaaa", kind: "transcript", createdAt: 1, text: "t" }];
    stubFetch(() => ({ ok: true, json: async () => ({ items }) }));
    await expect(getVoiceGallery()).resolves.toEqual(items);
  });

  it("base64-encodes binary audio bytes when saving to the gallery", async () => {
    const fetchMock = stubFetch((_url, init) => ({
      ok: true,
      json: async () => ({ id: "1_aaaaaaaa", kind: "audio", createdAt: 1, ...JSON.parse(init.body as string) }),
    }));
    // Includes bytes > 0x7f so a naive text encode would corrupt them.
    const bytes = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x00, 0xff, 0x80, 0x7f]);

    await saveGalleryAudio("spoken", "af_heart", new Blob([bytes]));

    const init = requestTo(fetchMock, "/api/voice/gallery/audio");
    const body = JSON.parse(init.body as string);
    expect(body.text).toBe("spoken");
    expect(body.voice).toBe("af_heart");
    const decoded = Array.from(atob(body.audioBase64), (ch) => ch.charCodeAt(0));
    expect(decoded).toEqual(Array.from(bytes));
  });

  it("deletes gallery items with the DELETE verb", async () => {
    const fetchMock = stubFetch(() => ({ ok: true, json: async () => ({ deleted: true }) }));
    await expect(deleteGalleryItem("1_aaaaaaaa")).resolves.toEqual({ deleted: true });
    const init = requestTo(fetchMock, "/api/voice/gallery/1_aaaaaaaa");
    expect(init.method).toBe("DELETE");
  });

  it("fetches gallery audio as a blob and reports failures with the status", async () => {
    const wav = new Blob(["RIFF"]);
    stubFetch(() => ({ ok: true, blob: async () => wav }));
    await expect(getGalleryAudio("1_aaaaaaaa")).resolves.toBe(wav);

    stubFetch(() => ({ ok: false, status: 404 }));
    await expect(getGalleryAudio("1_aaaaaaaa")).rejects.toThrow("status 404");
  });

  it("starts and polls the kokoro-onnx voice-file download", async () => {
    const fetchMock = stubFetch((url) => ({
      ok: true,
      json: async () =>
        url.endsWith("/status")
          ? { state: "completed", progress: 1, error: null, installed: true }
          : { state: "downloading", progress: 0, error: null },
    }));

    await expect(startKokoroDownload()).resolves.toMatchObject({ state: "downloading" });
    await expect(getKokoroDownloadStatus()).resolves.toMatchObject({ state: "completed", installed: true });

    const startInit = requestTo(fetchMock, "/api/voice/kokoro/download");
    expect(startInit.method).toBe("POST");
  });
});
