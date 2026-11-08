import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { TtsModelRow, TtsModelsPanel, studioTargetFor } from "../TtsModelsPanel";

describe("TtsModelsPanel", () => {
  it("renders its title before the models load", () => {
    const html = renderToStaticMarkup(<TtsModelsPanel />);
    expect(html).toContain("Speech models");
    expect(html).toContain("Kokoro");
  });
});

describe("TtsModelRow", () => {
  const base = {
    key: "k", name: "Qwen3-TTS 1.7B CustomVoice", summary: "Style.", repo: "r", sizeGb: 3.08, languages: "10 languages", default: false,
    kind: "custom" as const, capabilities: { speak: true, style: true, clone: false, design: false }, supported: true, installed: false,
    download: { state: "idle" as const, error: null },
  };
  it("shows capability chips, the size and a Download button", () => {
    const html = renderToStaticMarkup(<TtsModelRow model={base} onDownload={() => {}} />);
    expect(html).toContain(">Speak<");
    expect(html).toContain(">Style<");
    expect(html).not.toContain(">Clone<");
    expect(html).toContain("3.08 GB");
    expect(html).toContain(">Download<");
  });
  it("shows the measured memory, speed and machine, or says the model is not measured", () => {
    const estimates = { basis: "measured" as const, measuredOn: "Apple M4 Max, 64 GB, 2026-11-08", coldStartSeconds: 2.4, secondsPer10sAudio: 2.45, peakGb: 3.5, idleGb: 3.48 };
    const html = renderToStaticMarkup(<TtsModelRow model={{ ...base, estimates }} onDownload={() => {}} />);
    expect(html).toContain("3.5 GB peak");
    expect(html).toContain("2.5 s per 10 s");
    expect(html).toContain("2.4 s");
    expect(html).toContain("Apple M4 Max, 64 GB, 2026-11-08");
    const bare = renderToStaticMarkup(<TtsModelRow model={base} onDownload={() => {}} />);
    expect(bare).toContain("Not measured yet");
    expect(bare).not.toContain("GB peak");
  });
  it("offers Use in Speak on an installed speaking model and Browse voices on a cloning model", () => {
    expect(renderToStaticMarkup(<TtsModelRow model={{ ...base, installed: true }} onDownload={() => {}} onUse={() => {}} />)).toContain(">Use in Speak<");
    const clone = { ...base, kind: "base" as const, capabilities: { speak: false, style: false, clone: true, design: false }, installed: true };
    expect(renderToStaticMarkup(<TtsModelRow model={clone} onDownload={() => {}} onUse={() => {}} />)).toContain(">Browse voices<");
    expect(renderToStaticMarkup(<TtsModelRow model={base} onDownload={() => {}} onUse={() => {}} />)).not.toContain("Use in Speak");
  });
  it("Use in Speak lands on a preset of that model in the UI language, the design voice, or Voices", () => {
    const caps = { speak: true, style: true, clone: false, design: false };
    const preset = (id: string, language: string) => ({ id, name: id, kind: "preset" as const, language, model: "k", models: ["k"], capabilities: caps, available: true });
    const voices = [preset("qwen3:Serena", "zh"), preset("qwen3:Ryan", "en")];
    expect(studioTargetFor({ ...base, installed: true }, voices, "en-GB")).toEqual({ mode: "speak", voice: "qwen3:Ryan" });
    expect(studioTargetFor({ ...base, installed: true }, voices, "zh-CN")).toEqual({ mode: "speak", voice: "qwen3:Serena" });
    expect(studioTargetFor({ ...base, kind: "design" }, voices, "en")).toEqual({ mode: "speak", voice: "design" });
    expect(studioTargetFor({ ...base, kind: "base", capabilities: { ...caps, clone: true } }, voices, "en")).toEqual({ mode: "voices" });
  });
  it("shows Downloaded without a button once installed", () => {
    const html = renderToStaticMarkup(<TtsModelRow model={{ ...base, installed: true }} onDownload={() => {}} />);
    expect(html).toContain("Downloaded");
    expect(html).not.toContain(">Download<");
  });
});
