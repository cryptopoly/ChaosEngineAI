import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { SttModel, VoiceRuntime } from "../../../types";
import { VoiceProvider } from "../VoiceContext";
import { TranscribePanel } from "./TranscribePanel";

function stt(key: string, name: string, over: Partial<SttModel> = {}): SttModel {
  return { id: `org/${key}`, key, name, languages: "25 languages", sizeGb: 2.5, installed: true, default: false, backend: "mlx-whisper", ...over };
}

function render(models: SttModel[], language = "") {
  const runtime = {
    platform: "Darwin", lane: "mlx", installed: true, speakersAvailable: true, aligner: null, sttModels: models,
    settings: { sttModel: "", language, ttsVoice: "", speed: 1, autoSend: false, autoRead: false, vadSilenceMs: 700, bargeIn: true, idleUnloadMinutes: 10 },
  } as unknown as VoiceRuntime;
  return renderToStaticMarkup(
    <VoiceProvider value={{ runtime, platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>
      <TranscribePanel runtime={runtime} openRequest={null} onSendToChat={() => {}} onOpenModels={() => {}} />
    </VoiceProvider>,
  );
}

describe("Transcribe model and language", () => {
  const parakeet = stt("parakeet-tdt-v3", "Parakeet TDT 0.6B v3", { default: true });
  const whisper = stt("whisper-large-v3-turbo", "Whisper Large v3 Turbo", { installed: false, sizeGb: 0.8 });

  it("says per model whether it finds the language itself or uses your choice", () => {
    const detect = render([parakeet, whisper], "");
    expect(detect).toContain("Parakeet TDT 0.6B v3, finds the language itself");
    expect(detect).toContain("Whisper Large v3 Turbo, finds the language itself");
    const german = render([parakeet, whisper], "de");
    expect(german).toContain("Parakeet TDT 0.6B v3, finds the language itself");
    expect(german).toContain("Whisper Large v3 Turbo, uses your choice");
  });

  it("locks the language select for a model that always finds the language", () => {
    const html = render([parakeet, whisper], "de");
    expect(html).toMatch(/<select[^>]*disabled=""[^>]*>(?:(?!<\/select>).)*Detect automatically/);
    expect(html).toContain("This model finds the language itself.");
  });

  it("offers Remember as default", () => {
    expect(render([parakeet], "")).toContain("Remember as default");
  });

  it("says what an undownloaded pick downloads before the first file, and says nothing for a downloaded one", () => {
    expect(render([whisper], "")).toContain("Downloads 0.8 GB before the first file.");
    expect(render([parakeet], "")).not.toContain("before the first file");
    expect(render([whisper], "")).not.toContain("GB download)");
  });
});
