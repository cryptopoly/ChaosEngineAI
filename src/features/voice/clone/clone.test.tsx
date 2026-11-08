import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { CloneModel } from "./cloneApi";
import { canSaveVoice, isSampleFile, pickCloneModel, sampleLengthProblem, wavSeconds } from "./cloneLogic";
import { CloneStatement } from "./CloneStatement";
import { NewVoiceForm } from "./NewVoiceForm";

function model(over: Partial<CloneModel> = {}): CloneModel {
  return {
    key: "qwen3-tts-0.6b",
    name: "Qwen3-TTS 0.6B Base",
    summary: "Faster.",
    repo: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    revision: "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
    sizeGb: 2.52,
    languages: "10 languages",
    default: true,
    licence: { id: "apache-2.0", name: "Apache 2.0", url: "" },
    supported: true,
    installed: false,
    download: { state: "idle", error: null },
    ...over,
  };
}

describe("clone logic", () => {
  it("saves only with basis, name, the permission tick and a sample", () => {
    const details = { basis: "own" as const, speakerName: "Dan", voiceName: "", confirmed: true };
    expect(canSaveVoice(details, true)).toBe(true);
    expect(canSaveVoice(details, false)).toBe(false);
    expect(canSaveVoice({ ...details, basis: null }, true)).toBe(false);
    expect(canSaveVoice({ ...details, speakerName: "  " }, true)).toBe(false);
    expect(canSaveVoice({ ...details, confirmed: false }, true)).toBe(false);
  });

  it("takes WAV, MP3, M4A, FLAC and OGG samples", () => {
    for (const name of ["a.wav", "b.MP3", "c.m4a", "d.flac", "e.ogg"]) expect(isSampleFile(name)).toBe(true);
    for (const name of ["a.webm", "b.mp4", "wav", "c.wav.txt"]) expect(isSampleFile(name)).toBe(false);
  });

  it("checks the 3 to 30 second sample length", () => {
    expect(sampleLengthProblem(null)).toBeNull();
    expect(sampleLengthProblem(2.9)).toBe("short");
    expect(sampleLengthProblem(30.5)).toBe("long");
    expect(sampleLengthProblem(8)).toBeNull();
    expect(wavSeconds(44 + 32000)).toBe(1);
    expect(wavSeconds(10)).toBe(0);
  });

  it("prefers a downloaded default model", () => {
    const small = model();
    const big = model({ key: "qwen3-tts-1.7b", default: false, installed: true });
    expect(pickCloneModel([small, big])?.key).toBe("qwen3-tts-1.7b");
    expect(pickCloneModel([model({ installed: true }), big])?.key).toBe("qwen3-tts-0.6b");
    expect(pickCloneModel([small, model({ key: "x", default: false })])?.key).toBe("qwen3-tts-0.6b");
    expect(pickCloneModel([model({ supported: false })])).toBeNull();
  });
});

describe("clone components", () => {
  it("always shows the plain statement", () => {
    const html = renderToStaticMarkup(<CloneStatement />);
    expect(html).toContain("Only clone your own voice");
    expect(html).toContain("inaudible watermark");
    expect(html).toContain("can be turned off");
  });

  it("states who may be cloned", () => {
    const html = renderToStaticMarkup(<CloneStatement />);
    expect(html).toContain("permission");
    expect(html).toContain("Not public figures, children or people who have died");
  });

  it("starts the new-voice form on one screen with Save disabled", () => {
    const html = renderToStaticMarkup(<NewVoiceForm onCreated={() => {}} onCancel={() => {}} />);
    expect(html).toContain("Whose voice is it?");
    expect(html).toContain("My own voice");
    expect(html).toContain("Someone who gave me permission");
    expect(html).toContain("Voice sample");
    expect(html).toMatch(/type="file"[^>]*accept=".wav,.mp3,.m4a,.flac,.ogg"|accept=".wav,.mp3,.m4a,.flac,.ogg"[^>]*type="file"/);
    expect(html).toContain(">Record<");
    expect(html).toContain("This is my own voice, or the speaker has given me permission to clone it.");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Save voice/);
    // No URL field: samples come only from the microphone or a local file.
    expect(html).not.toMatch(/type="url"/);
  });
});
