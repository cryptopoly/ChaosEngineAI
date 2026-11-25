import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { VoiceApiError } from "../../../api/voice";
import type { VoiceRuntime } from "../../../types";
import { VoiceProvider } from "../VoiceContext";
import { describeVoiceError } from "../voiceErrorText";
import type { ClonedVoice, CloneModel } from "./cloneApi";
import { cloneActionBlock, cloneErrorFix, clonedVoiceId } from "./cloneLogic";
import { CloneModelPicker } from "./CloneModelPicker";
import { CloneSaved, CloneVoiceScreen } from "./CloneVoiceScreen";
import { NewVoiceForm } from "./NewVoiceForm";
import { VoiceErrorNotice } from "./VoiceErrorNotice";

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

const runtime = (over: Record<string, unknown> = {}) => ({ installed: true, extras: [], settings: {}, ...over }) as unknown as VoiceRuntime;
const withVoice = (node: ReactNode, rt: VoiceRuntime = runtime()) =>
  renderToStaticMarkup(
    <VoiceProvider value={{ runtime: rt, platform: "Windows", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>{node}</VoiceProvider>,
  );
const t = (key: string, options?: Record<string, unknown>) => String(options?.defaultValue ?? key);
const pc = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: false }] });

describe("Clone a voice: four steps in order (UX audit change 5)", () => {
  it("lists Permission, Sample, Name, Model in that order, with Save after the model", () => {
    const html = withVoice(<NewVoiceForm onCreated={() => {}} />);
    const at = (text: string) => html.indexOf(text);
    const order = [">1. Permission<", ">2. Sample<", ">3. Name<", "4. Model", ">Save voice<"].map(at);
    expect(order.every((i) => i >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    // The permission tick sits with the permission step, before the sample.
    expect(at("This is my own voice, or the speaker has given me permission")).toBeLessThan(at(">2. Sample<"));
    // As a screen of its own there is no Cancel; a form opened over another screen can pass one.
    expect(html).not.toContain(">Cancel<");
    expect(withVoice(<NewVoiceForm onCreated={() => {}} onCancel={() => {}} />)).toContain(">Cancel<");
  });

  it("the model step carries the step number when given a title", () => {
    const html = renderToStaticMarkup(<CloneModelPicker title="4. Model" models={[model()]} selected="" onSelect={() => {}} onDownload={() => {}} />);
    expect(html).toContain("<legend>4. Model</legend>");
    expect(html).not.toContain("Cloning model");
  });

  it("says so when no cloning model can run here, instead of showing nothing", () => {
    const html = renderToStaticMarkup(<CloneModelPicker models={[model({ supported: false })]} selected="" onSelect={() => {}} onDownload={() => {}} />);
    expect(html).toContain("No cloning model runs on this computer.");
    expect(html).not.toContain("<button");
  });
});

describe("CloneVoiceScreen", () => {
  const catalog = { modelsAvailable: true, cloneAllowed: true, refresh: async () => {} };

  it("shows the form when cloning is allowed", () => {
    const html = withVoice(<CloneVoiceScreen catalog={catalog} onSpeakWith={() => {}} />);
    expect(html).toContain("Clone a voice");
    expect(html).toContain(">1. Permission<");
    expect(html).not.toContain("voice-clone__blocked");
  });

  it("replaces the form with the reason and the fix button on a PC without the speech GPU add-on", () => {
    const html = withVoice(<CloneVoiceScreen catalog={{ ...catalog, modelsAvailable: false, cloneAllowed: false }} onSpeakWith={() => {}} />, pc);
    expect(html).toContain("Needs the speech GPU add-on.");
    expect(html).toContain(">Open Voice setup<");
    expect(html).not.toContain(">1. Permission<");
  });

  it("a Mac-only or window-only block is text only", () => {
    const mac = withVoice(<CloneVoiceScreen catalog={{ ...catalog, modelsAvailable: false }} onSpeakWith={() => {}} />);
    expect(mac).toContain("Voice cloning runs on Apple Silicon Macs for now.");
    expect(mac).not.toContain("<button");
    const windowOnly = withVoice(<CloneVoiceScreen catalog={{ ...catalog, cloneAllowed: false }} onSpeakWith={() => {}} />);
    expect(windowOnly).toContain("only be used from the ChaosEngineAI window");
  });

  it("after Save: Voice saved. Speak with it", () => {
    const html = renderToStaticMarkup(<CloneSaved voice={{ id: "abc", name: "Ada" } as ClonedVoice} onSpeakWith={() => {}} onAnother={() => {}} />);
    expect(html).toContain("Voice saved.");
    expect(html).toContain("<strong>Ada</strong>");
    expect(html).toContain(">Speak with it<");
    expect(html).toContain(">Clone another voice<");
    // Speak knows a cloned voice as clone:<id>.
    expect(clonedVoiceId({ id: "abc" })).toBe("clone:abc");
  });
});

describe("greyed actions give their reason as text (UX audit change 5)", () => {
  const ok = { modelsAvailable: true, cloneAllowed: true, runtime: runtime() };
  it("is null while the action can be used", () => {
    expect(cloneActionBlock("clone", ok, t)).toBeNull();
    expect(cloneActionBlock("design", { ...ok, cloneAllowed: false }, t)).toBeNull();
  });
  it("sends a PC without the add-on to Voice setup", () => {
    for (const kind of ["clone", "design"] as const) {
      const block = cloneActionBlock(kind, { modelsAvailable: false, cloneAllowed: false, runtime: pc }, t);
      expect(block?.fix).toBe("setup");
      expect(block?.reason).toContain("speech GPU add-on");
    }
  });
  it("gives the Mac-only and the window-only notes with no button", () => {
    expect(cloneActionBlock("clone", { ...ok, modelsAvailable: false }, t)).toEqual({ reason: "Voice cloning runs on Apple Silicon Macs for now.", fix: null });
    expect(cloneActionBlock("design", { ...ok, modelsAvailable: false }, t)?.reason).toContain("Designing a voice");
    expect(cloneActionBlock("clone", { ...ok, cloneAllowed: false }, t)?.fix).toBeNull();
  });
});

describe("errors in the clone form go through describeVoiceError", () => {
  it("maps each error action to what this form can do", () => {
    expect(cloneErrorFix("retry")).toBe("retry");
    expect(cloneErrorFix("change-sample")).toBe("sample");
    expect(cloneErrorFix("confirm-permission")).toBe("permission");
    for (const action of ["download-model", "smaller-model", "pick-another-model"] as const) expect(cloneErrorFix(action)).toBe("model");
    for (const action of ["install-voice-runtime", "install-addon", "install-ffmpeg", "add-hf-token"] as const) expect(cloneErrorFix(action)).toBe("setup");
    expect(cloneErrorFix("none")).toBeNull();
    expect(cloneErrorFix("pick-another-language")).toBeNull();
  });

  it("shows one sentence, one button and the raw text under Details", () => {
    const error = new VoiceApiError("Reference clip is 1.2 seconds long", 400, { code: "reference_too_short", message: "", action: "change-sample", extra: null });
    const html = renderToStaticMarkup(<VoiceErrorNotice view={describeVoiceError(error)} onAction={() => {}} />);
    expect(html).toContain("The voice sample is too short.");
    expect(html).toContain(">Choose another sample<");
    expect(html).toContain("<summary>Details</summary>");
    expect(html).toContain("Reference clip is 1.2 seconds long");
    expect(html.match(/<button/g)).toHaveLength(1);
  });

  it("shows no button when the screen has nothing to do for the action, and no Details for a plain message", () => {
    const plain = renderToStaticMarkup(<VoiceErrorNotice view={describeVoiceError(new Error("Microphone blocked."))} />);
    expect(plain).toContain("Microphone blocked.");
    expect(plain).not.toContain("<button");
    expect(plain).not.toContain("Details");
    const coded = describeVoiceError(new VoiceApiError("raw", 503, { code: "gpu_memory", message: "", action: "smaller-model", extra: null }));
    expect(renderToStaticMarkup(<VoiceErrorNotice view={coded} />)).not.toContain("<button");
  });
});
