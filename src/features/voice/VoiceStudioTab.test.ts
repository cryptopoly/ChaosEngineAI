import { afterEach, describe, expect, it, vi } from "vitest";
import { pickDefaultSttModel } from "./VoiceStudioTab";
import type { SttModel } from "../../types";
import { STUDIO_MODE_KEY, readStudioMode, requestedMode, writeStudioMode } from "./voiceStudioMode";

function fakeStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    data,
  };
}

describe("Voice Studio mode", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("opens on Speak the first time", () => {
    vi.stubGlobal("window", { localStorage: fakeStorage() });
    expect(readStudioMode()).toBe("speak");
  });

  it("remembers the last mode used", () => {
    const storage = fakeStorage();
    vi.stubGlobal("window", { localStorage: storage });
    writeStudioMode("transcribe");
    expect(storage.data.get(STUDIO_MODE_KEY)).toBe("transcribe");
    expect(readStudioMode()).toBe("transcribe");
    writeStudioMode("voices");
    expect(readStudioMode()).toBe("voices");
  });

  it("falls back to Speak for a stored value it does not know, or when storage is blocked", () => {
    vi.stubGlobal("window", { localStorage: fakeStorage({ [STUDIO_MODE_KEY]: "karaoke" }) });
    expect(readStudioMode()).toBe("speak");
    const blocked = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };
    vi.stubGlobal("window", { localStorage: blocked });
    expect(readStudioMode()).toBe("speak");
    expect(() => writeStudioMode("voices")).not.toThrow();
  });

  it("a Library transcript opens Transcribe; Use in Speak opens Speak; Browse voices opens Voices", () => {
    expect(requestedMode({ id: "t-1", nonce: 1 })).toBe("transcribe");
    expect(requestedMode({ mode: "speak", voice: "qwen3:Ryan", nonce: 2 })).toBe("speak");
    expect(requestedMode({ mode: "voices", nonce: 3 })).toBe("voices");
    expect(requestedMode({ nonce: 4 })).toBeNull();
  });
});

function makeModel(overrides: Partial<SttModel>): SttModel {
  return {
    id: "org/model",
    name: "Model",
    sizeGb: 0.5,
    installed: false,
    default: false,
    ...overrides,
  };
}

describe("pickDefaultSttModel", () => {
  it("skips rows whose backend package is not installed", () => {
    const models = [
      makeModel({ id: "parakeet", installed: true, backend: "parakeet-mlx", backendInstalled: false }),
      makeModel({ id: "turbo", default: true, installed: true, backend: "mlx-whisper", backendInstalled: true }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("turbo");
  });

  it("prefers the catalog default when it is installed", () => {
    const models = [
      makeModel({ id: "a", installed: true }),
      makeModel({ id: "b", default: true, installed: true }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("b");
  });

  it("falls back to the first installed model when the default is not installed", () => {
    const models = [
      makeModel({ id: "a", default: true, installed: false }),
      makeModel({ id: "b", installed: false }),
      makeModel({ id: "c", installed: true }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("c");
  });

  it("falls back to the catalog default when nothing is installed", () => {
    const models = [
      makeModel({ id: "a", installed: false }),
      makeModel({ id: "b", default: true, installed: false }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("b");
  });

  it("falls back to the first model when there is no default", () => {
    const models = [
      makeModel({ id: "a", installed: false }),
      makeModel({ id: "b", installed: false }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("a");
  });

  it("returns undefined for an empty list", () => {
    expect(pickDefaultSttModel([])).toBeUndefined();
  });
});
