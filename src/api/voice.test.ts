import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
  isTauri: vi.fn(() => false),
}));

import {
  deleteLibraryItems,
  getLibrary,
  getLibraryAudio,
  saveSpeech,
  speakStream,
  transcribeBlob,
  startVoiceInstall,
  saveVoiceSettings,
  synthesizeSpeech,
  transcribeAudio,
  speakVoice,
  VoiceApiError,
  parseVoiceErrorInfo,
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

  it("lists the Library with the search and kind as query parameters", async () => {
    const items = [{ id: "1_aaaaaaaa", kind: "transcript", createdAt: 1, preview: "t" }];
    const fetchMock = stubFetch(() => ({ ok: true, json: async () => ({ items }) }));
    await expect(getLibrary("budget review", "transcript")).resolves.toEqual(items);
    requestTo(fetchMock, "/api/voice/library?q=budget+review&kind=transcript");
    await getLibrary();
    requestTo(fetchMock, "/api/voice/library");
  });

  it("sends transcription options as form fields", async () => {
    const fetchMock = stubFetch(() => ({ ok: true, json: async () => ({ text: "" }) }));
    await transcribeBlob(new Blob(["a"]), "talk.mp3", { task: "translate", speakers: true, speakerCount: 3, save: true, words: true });
    const form = requestTo(fetchMock, "/api/voice/transcribe").body as FormData;
    expect([form.get("task"), form.get("speakers"), form.get("speakerCount"), form.get("save"), form.get("words")]).toEqual([
      "translate",
      "true",
      "3",
      "true",
      "true",
    ]);
    expect((form.get("audio") as File).name).toBe("talk.mp3");
  });

  it("saves speech as multipart and bulk-deletes by id", async () => {
    const fetchMock = stubFetch((url) => ({
      ok: true,
      json: async () => (url.endsWith("/delete") ? { deleted: ["1_aaaaaaaa"] } : { id: "1_aaaaaaaa" }),
    }));
    await saveSpeech(new Blob(["RIFF"]), { text: "Hello", voice: "af_heart", speed: 1.2 });
    const form = requestTo(fetchMock, "/api/voice/library/speech").body as FormData;
    expect([form.get("text"), form.get("voice"), form.get("speed")]).toEqual(["Hello", "af_heart", "1.2"]);
    expect([form.get("voiceName"), form.get("voiceKind")]).toEqual([null, null]);
    fetchMock.mockClear();
    await saveSpeech(new Blob(["RIFF"]), { text: "Hi", voice: "qwen3:Ryan", speed: 1, voiceName: "Ryan", voiceKind: "preset" });
    const named = requestTo(fetchMock, "/api/voice/library/speech").body as FormData;
    expect([named.get("voiceName"), named.get("voiceKind")]).toEqual(["Ryan", "preset"]);
    await expect(deleteLibraryItems(["1_aaaaaaaa"])).resolves.toEqual(["1_aaaaaaaa"]);
    expect(JSON.parse(String(requestTo(fetchMock, "/api/voice/library/delete").body))).toEqual({ ids: ["1_aaaaaaaa"] });
  });

  it("fetches Library audio, converting to MP3 on request, and reports failures", async () => {
    const audio = new Blob(["ID3"]);
    const fetchMock = stubFetch(() => ({ ok: true, blob: async () => audio }));
    await expect(getLibraryAudio("1_aaaaaaaa", "mp3")).resolves.toBe(audio);
    requestTo(fetchMock, "/api/voice/library/1_aaaaaaaa/audio?format=mp3");

    stubFetch(() => ({ ok: false, status: 404, text: async () => "" }));
    await expect(getLibraryAudio("1_aaaaaaaa")).rejects.toThrow("status 404");
  });

  it("reads the read-aloud stream sentence by sentence, across chunk boundaries", async () => {
    const lines = [
      JSON.stringify({ index: 0, total: 2, text: "One.", audio: "AAA=", durationS: 1 }),
      JSON.stringify({ index: 1, total: 2, text: "Two.", audio: "BBB=", durationS: 1 }),
      JSON.stringify({ done: true }),
    ].join("\n") + "\n";
    const encoder = new TextEncoder();
    const chunks = [lines.slice(0, 25), lines.slice(25, 90), lines.slice(90)];
    const body = new ReadableStream<Uint8Array<ArrayBuffer>>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    const fetchMock = stubFetch(() => ({ ok: true, body }));
    const seen: string[] = [];
    await speakStream("One. Two.", "af_heart", 1, (s) => seen.push(s.text));
    expect(seen).toEqual(["One.", "Two."]);
    expect(JSON.parse(String(requestTo(fetchMock, "/api/voice/speak/stream").body))).toEqual({
      text: "One. Two.",
      voice: "af_heart",
      speed: 1,
    });
  });

  it("sends any voice with use=call and hands the fallback note to onNote (D-72)", async () => {
    const lines = [
      JSON.stringify({ note: "Ryan could not speak here (not downloaded), so the Kokoro voice reads instead.", voice: "af_heart" }),
      JSON.stringify({ index: 0, total: 1, text: "Hi.", audio: "AAA=", durationS: 1, voice: "af_heart" }),
      JSON.stringify({ done: true }),
    ].join("\n") + "\n";
    const body = new ReadableStream<Uint8Array<ArrayBuffer>>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(lines));
        controller.close();
      },
    });
    const fetchMock = stubFetch(() => ({ ok: true, body }));
    const seen: string[] = [];
    const notes: string[] = [];
    await speakStream("Hi.", "qwen3:Ryan", 1, (s) => seen.push(`${s.voice}:${s.text}`), undefined, { use: "call", onNote: (n) => notes.push(n) });
    expect(seen).toEqual(["af_heart:Hi."]);
    expect(notes).toEqual(["Ryan could not speak here (not downloaded), so the Kokoro voice reads instead."]);
    expect(JSON.parse(String(requestTo(fetchMock, "/api/voice/speak/stream").body))).toEqual({
      text: "Hi.",
      voice: "qwen3:Ryan",
      speed: 1,
      use: "call",
    });
  });

  it("raises an error sent inside the read-aloud stream", async () => {
    const body = new ReadableStream<Uint8Array<ArrayBuffer>>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(JSON.stringify({ error: "no voice", index: 0 }) + "\n"));
        controller.close();
      },
    });
    stubFetch(() => ({ ok: true, body }));
    await expect(speakStream("Hi.", "af_heart", 1, () => {})).rejects.toThrow("no voice");
  });

  it("posts voice settings and install extras", async () => {
    const fetchMock = stubFetch(() => ({ ok: true, json: async () => ({}) }));
    await saveVoiceSettings({ autoRead: true });
    await startVoiceInstall(["chinese"]);
    expect(JSON.parse(String(requestTo(fetchMock, "/api/voice/settings").body))).toEqual({ autoRead: true });
    expect(JSON.parse(String(requestTo(fetchMock, "/api/voice/install").body))).toEqual({ extras: ["chinese"] });
  });
});

describe("voice errors carry a code, a plain sentence and an action", () => {
  const failure = {
    detail: "OutOfMemoryError: CUDA out of memory.",
    code: "gpu_memory",
    message: "This model needs more GPU memory than is free.",
    action: "smaller-model",
    extra: null,
  };

  /** A failed response whose body can be read twice, as a real one can through ``clone()``. */
  const failedResponse = (status: number, body: unknown): Partial<Response> => {
    const make = (): Partial<Response> => ({
      ok: false,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body),
      clone: () => make() as Response,
    });
    return make();
  };

  it("keeps the raw detail as the message and adds the info", async () => {
    stubFetch(() => failedResponse(503, failure));
    const error = await transcribeAudio(new FormData()).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(VoiceApiError);
    const voiceError = error as VoiceApiError;
    expect(voiceError.message).toBe("OutOfMemoryError: CUDA out of memory.");
    expect(voiceError.status).toBe(503);
    expect(voiceError.info).toEqual({
      code: "gpu_memory",
      message: "This model needs more GPU memory than is free.",
      action: "smaller-model",
      extra: null,
    });
  });

  it("reads the same fields from a failed Speak request", async () => {
    stubFetch(() => failedResponse(503, { ...failure, code: "addon_missing", action: "install-addon", extra: "speech-gpu" }));
    const error = (await speakVoice({ text: "Hi", voice: "qwen3:Ryan" }).catch((e: unknown) => e)) as VoiceApiError;
    expect(error).toBeInstanceOf(VoiceApiError);
    expect([error.info?.code, error.info?.action, error.info?.extra]).toEqual(["addon_missing", "install-addon", "speech-gpu"]);
  });

  it("has no info when the response has no code", async () => {
    stubFetch(() => failedResponse(422, { detail: "Invalid input" }));
    const error = (await speakVoice({ text: "", voice: "x" }).catch((e: unknown) => e)) as VoiceApiError;
    expect(error.info).toBeNull();
    expect(error.message).toBe("Invalid input");
  });

  it("carries the fields of an error line inside the read-aloud stream", async () => {
    const line = JSON.stringify({ error: "no voice", code: "model_missing", message: "m", action: "download-model", extra: null, index: 0 });
    const body = new ReadableStream<Uint8Array<ArrayBuffer>>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(`${line}\n`));
        controller.close();
      },
    });
    stubFetch(() => ({ ok: true, body }));
    const error = (await speakStream("Hi.", "af_heart", 1, () => {}).catch((e: unknown) => e)) as VoiceApiError;
    expect(error.message).toBe("no voice");
    expect(error.info?.code).toBe("model_missing");
  });

  it("shows a code it does not know as the generic one and an unknown action as none", () => {
    expect(parseVoiceErrorInfo({ code: "from_the_future", message: "x", action: "dance" })).toEqual({
      code: "unknown",
      message: "x",
      action: "none",
      extra: null,
    });
    expect(parseVoiceErrorInfo({ detail: "x" })).toBeNull();
    expect(parseVoiceErrorInfo(null)).toBeNull();
  });
});
