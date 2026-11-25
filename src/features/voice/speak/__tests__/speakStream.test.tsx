import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { VoiceEntry, VoiceModelRow } from "../../../../types";
import { VoiceApiError } from "../../../../api/voice";
import { describeVoiceError } from "../../voiceErrorText";
import { SpeakError, StyleGapNote } from "../SpeakControls";
import { VoiceCanLine } from "../VoiceRow";
import {
  MAX_RECENT_CLIPS,
  addRecent,
  downloadSizeGb,
  errorTarget,
  ffmpegLimits,
  styleGap,
  styleState,
  voiceAbilities,
  voiceControls,
} from "../speakLogic";

const caps = (over: Partial<VoiceEntry["capabilities"]> = {}) => ({ speak: true, style: false, clone: false, design: false, ...over });

function model(over: Partial<VoiceModelRow>): VoiceModelRow {
  return {
    key: "m", name: "M", summary: "", repo: "r", sizeGb: 1, languages: "", default: false, kind: "custom", capabilities: caps(),
    supported: true, installed: false, download: { state: "idle", error: null }, ...over,
  };
}

function voice(over: Partial<VoiceEntry>): VoiceEntry {
  return { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", languageLabel: "English", model: "c06", models: ["c06", "c17"], capabilities: caps({ style: true }), available: true, ...over };
}

const small = model({ key: "c06", name: "0.6B", sizeGb: 2.5 });
const big = model({ key: "c17", name: "1.7B", sizeGb: 3.08, downloadGb: 4.52, capabilities: caps({ style: true }) });
const quick = voice({ id: "af_heart", name: "Heart", kind: "builtin", languageLabel: "English (US)", language: "en-US" });

describe("ffmpeg limits", () => {
  it("leaves everything on while ffmpeg is there", () => {
    expect(ffmpegLimits(quick)).toEqual({ speed: true, pitch: true });
    expect(ffmpegLimits(null)).toEqual({ speed: true, pitch: true });
  });

  it("takes pitch from every voice without ffmpeg, and speed from all but the quick ones", () => {
    expect(ffmpegLimits({ ...quick, needsFfmpeg: true })).toEqual({ speed: true, pitch: false });
    expect(ffmpegLimits(voice({ needsFfmpeg: true }))).toEqual({ speed: false, pitch: false });
  });
});

describe("this voice can", () => {
  const none = voiceControls(null);

  it("lists speed for a quick voice and its language", () => {
    expect(voiceAbilities(quick, none, false)).toEqual({ speed: true, style: false, tags: false, languages: { kind: "own", label: "English (US)" } });
  });

  it("adds style for a voice that takes it, and leaves speed out where ffmpeg is missing", () => {
    expect(voiceAbilities(voice({}), none, true)).toMatchObject({ speed: true, style: true });
    expect(voiceAbilities(voice({ needsFfmpeg: true }), none, true)).toMatchObject({ speed: false, style: true });
  });

  it("uses the model's language list and its sound tags", () => {
    const turbo = model({ capabilities: caps({ tags: true, languageSelect: true }), controls: { tags: ["laugh"], languages: ["de", "en"] } });
    expect(voiceAbilities(voice({}), voiceControls(turbo), false)).toMatchObject({ tags: true, languages: { kind: "select", codes: ["de", "en"] } });
  });

  it("says any language for a designed voice, and nothing for a clone whose language is not recorded", () => {
    expect(voiceAbilities(voice({ kind: "design", language: "auto" }), none, true)?.languages).toEqual({ kind: "any" });
    expect(voiceAbilities(voice({ kind: "clone", language: "auto" }), none, false)?.languages).toBeNull();
    expect(voiceAbilities(null, none, false)).toBeNull();
  });

  it("renders as one line", () => {
    const line = (v: VoiceEntry, style: boolean, controls = none) => renderToStaticMarkup(<VoiceCanLine abilities={voiceAbilities(v, controls, style)} />);
    expect(line(quick, false)).toContain("This voice can: speed. Languages: English (US).");
    expect(line(voice({}), true)).toContain("This voice can: speed, style. Languages: English.");
    expect(line(voice({ needsFfmpeg: true, language: "auto", languageLabel: undefined }), false)).toContain("This voice reads the text as it is.");
    const many = model({ capabilities: caps({ languageSelect: true }), controls: { languages: ["de", "en", "fr", "es"] } });
    expect(line(voice({}), false, voiceControls(many))).toContain("Languages: 4 languages.");
    const some = model({ capabilities: caps({ languageSelect: true }), controls: { languages: ["de", "en"] } });
    expect(line(voice({}), false, voiceControls(some))).toContain("Languages: German, English.");
  });
});

describe("the sentence that replaces an unusable style box", () => {
  it("offers the style-capable natural model to a quick voice, with its catalogue download size", () => {
    const gap = styleGap(styleState(quick, [small, big]), [small, big], true);
    expect(gap).toMatchObject({ kind: "needsNatural", action: "download" });
    expect(gap?.model?.key).toBe("c17");
    expect(downloadSizeGb(big)).toBe(4.5);
    expect(downloadSizeGb(small)).toBe(2.5);
    const html = renderToStaticMarkup(<StyleGapNote gap={gap!} sizeGb={downloadSizeGb(big)} onDownload={() => {}} onChoose={() => {}} />);
    expect(html).toContain("Style needs a natural voice.");
    expect(html).toContain(">Get one, 4.5 GB<");
  });

  it("shows progress while that model downloads, and a Choose button once it is on disk", () => {
    const downloading = { ...big, download: { state: "downloading" as const, error: null } };
    expect(styleGap(styleState(quick, [downloading]), [downloading], true)).toMatchObject({ action: "downloading" });
    const html = renderToStaticMarkup(
      <StyleGapNote gap={styleGap(styleState(quick, [downloading]), [downloading], true)!} sizeGb={4.5} onDownload={() => {}} onChoose={() => {}} />,
    );
    expect(html).toContain("Downloading 1.7B…");
    expect(html).not.toContain("<button");
    const installed = { ...big, installed: true };
    expect(styleGap(styleState(quick, [installed]), [installed], true)).toMatchObject({ kind: "needsNatural", action: "choose" });
  });

  it("has no button when natural voices cannot run here, and none for clones, tags or Expression", () => {
    expect(styleGap(styleState(quick, [big]), [big], false)).toEqual({ kind: "builtinNone", action: null, model: null });
    expect(styleGap(styleState(voice({ kind: "clone" }), []), [], true)).toEqual({ kind: "clone", action: null, model: null });
    expect(styleGap(styleState(voice({ family: "orpheus" }), []), [], true)).toMatchObject({ kind: "tags", action: null });
    expect(styleGap(styleState(voice({ family: "chatterbox" }), []), [], true)).toMatchObject({ kind: "expression", action: null });
  });

  it("offers the 1.7B download to a preset on the small model, and nothing when the style box works", () => {
    const state = styleState(voice({}), [{ ...small, installed: true }, big]);
    expect(styleGap(state, [{ ...small, installed: true }, big], true)).toMatchObject({ kind: "smallModel", action: "download" });
    expect(styleGap(styleState(voice({ kind: "design" }), []), [], true)).toBeNull();
  });
});

describe("recent clips", () => {
  it("keeps the newest five and reports what fell off", () => {
    let list: number[] = [];
    let dropped: number[] = [];
    for (let i = 1; i <= 7; i += 1) {
      const next = addRecent(list, i);
      list = next.list;
      dropped = [...dropped, ...next.dropped];
    }
    expect(MAX_RECENT_CLIPS).toBe(5);
    expect(list).toEqual([7, 6, 5, 4, 3]);
    expect(dropped).toEqual([1, 2]);
  });
});

describe("voice errors on Speak", () => {
  it("leads each action to a place on this screen", () => {
    expect(errorTarget("retry")).toBe("retry");
    expect(errorTarget("install-voice-runtime")).toBe("models");
    expect(errorTarget("install-addon")).toBe("models");
    expect(errorTarget("smaller-model")).toBe("models");
    expect(errorTarget("download-model")).toBe("download");
    expect(errorTarget("install-ffmpeg")).toBe("settings");
    expect(errorTarget("add-hf-token")).toBe("settings");
    expect(errorTarget("pick-another-language")).toBe("picker");
    expect(errorTarget("change-sample")).toBe("clone");
    expect(errorTarget("none")).toBeNull();
  });

  it("shows the sentence, one button and the raw text under Details", () => {
    const view = describeVoiceError(
      new VoiceApiError("CUDA out of memory. Tried to allocate 2.00 GiB", 503, { code: "gpu_memory", message: "m", action: "smaller-model", extra: null }),
    );
    const html = renderToStaticMarkup(<SpeakError error={view} onAction={() => {}} />);
    expect(html).toContain("This model needs more GPU memory than is free.");
    expect(html).toContain(">Choose a smaller model<");
    expect(html).toContain("<summary>Details</summary>");
    expect(html).toContain("CUDA out of memory. Tried to allocate 2.00 GiB");
  });

  it("drops the button when this screen cannot do the action, and the Details when there is no raw text", () => {
    const view = describeVoiceError(new VoiceApiError("The voice worker stopped unexpectedly.", 503, { code: "worker_stopped", message: "m", action: "retry", extra: null }));
    const html = renderToStaticMarkup(<SpeakError error={view} onAction={null} />);
    expect(html).toContain("The voice worker stopped unexpectedly.");
    expect(html).not.toContain("<button");
    expect(html).not.toContain("Details");
  });

  it("shows a plain failure as its text, with no button", () => {
    const html = renderToStaticMarkup(<SpeakError error={describeVoiceError(new Error("Failed to fetch"))} onAction={() => {}} />);
    expect(html).toContain("Failed to fetch");
    expect(html).not.toContain("<button");
  });
});
