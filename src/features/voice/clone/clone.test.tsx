import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { CloneModel } from "./cloneApi";
import { canStartConsent, pickCloneModel, sampleLengthProblem, spacedCode, wavSeconds } from "./cloneLogic";
import { ClonePanel, CloneStatement } from "./ClonePanel";
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
  it("asks for the consent sentence only with basis, name and the statement", () => {
    const form = { basis: "own" as const, speakerName: "Dan", voiceName: "", confirmed: true };
    expect(canStartConsent(form)).toBe(true);
    expect(canStartConsent({ ...form, basis: null })).toBe(false);
    expect(canStartConsent({ ...form, speakerName: "  " })).toBe(false);
    expect(canStartConsent({ ...form, confirmed: false })).toBe(false);
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
    expect(spacedCode("4721")).toBe("4 7 2 1");
  });
});

describe("clone components", () => {
  it("is collapsed by default and always shows the plain statement", () => {
    const html = renderToStaticMarkup(<ClonePanel />);
    expect(html).toContain("<details");
    expect(html).not.toContain("open=\"\"");
    expect(html).toContain("Clone a voice");
    expect(html).toContain("Only copy your own voice");
    expect(html).toContain("cannot be turned off");
    // The body (models, voices) is not mounted while collapsed.
    expect(html).not.toContain("Cloning model");
  });

  it("states who may be cloned", () => {
    const html = renderToStaticMarkup(<CloneStatement />);
    expect(html).toContain("permission");
    expect(html).toContain("Not public figures, children or people who have died");
  });

  it("starts the new-voice form with the consent button disabled", () => {
    const html = renderToStaticMarkup(<NewVoiceForm onCreated={() => {}} onCancel={() => {}} />);
    expect(html).toContain("Whose voice is it?");
    expect(html).toContain("My own voice");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Get the consent sentence/);
    // No URL field: samples come only from the microphone or a local file.
    expect(html).not.toMatch(/type="url"/);
  });
});
