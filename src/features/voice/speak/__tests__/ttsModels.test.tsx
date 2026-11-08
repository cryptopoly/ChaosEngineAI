import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { TtsModelRow, TtsModelsPanel } from "../TtsModelsPanel";

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
  it("shows Downloaded without a button once installed", () => {
    const html = renderToStaticMarkup(<TtsModelRow model={{ ...base, installed: true }} onDownload={() => {}} />);
    expect(html).toContain("Downloaded");
    expect(html).not.toContain(">Download<");
  });
});
