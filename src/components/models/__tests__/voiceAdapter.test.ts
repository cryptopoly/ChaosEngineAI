import { describe, expect, it } from "vitest";
import {
  KOKORO_ROW_ID,
  buildVoiceEntries,
  isKokoro,
  kokoroEntry,
  voiceDownloadStatus,
  voiceEntryMatchesQuery,
  voiceFamilyId,
  voiceMakes,
  voiceRowToEntry,
  type VoiceMakes,
} from "../adapters/voice";
import { resolveModelStatus } from "../modelStatus";
import type { VoiceModelRow } from "../../../types";

const caps = { speak: true, style: false, clone: false, design: false };
const fit = { status: "fits", needGb: 3.5, usableGb: 51.8, headroomPct: 93, reason: "Peak memory 3.5 GB while speaking", basis: "measured" };

function row(over: Partial<VoiceModelRow> = {}): VoiceModelRow {
  return {
    key: "qwen3-tts-custom-1.7b", name: "Qwen3-TTS 1.7B CustomVoice", summary: "Nine preset voices.", repo: "mlx-community/x", sizeGb: 3.08,
    languages: "10 languages", default: false, kind: "custom", family: "qwen3-tts", capabilities: { ...caps, style: true }, supported: true,
    installed: true, fit, licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid" },
    estimates: { basis: "measured", peakGb: 3.5, secondsPer10sAudio: 2.45 }, download: { state: "idle", error: null },
    ...over,
  };
}

const LABELS = { quick: "Quick voices", natural: "Natural voices", style: "Style", clone: "Clone", design: "Design", tags: "Sound tags", expression: "Expression", languages: "Language choice" } as Record<VoiceMakes, string>;

describe("voice adapter (D-73 M3b)", () => {
  it("maps a catalog row onto the shared row: key as download and delete target, fit, size and memory", () => {
    const entry = voiceRowToEntry(row({ downloadGb: 3.2 }));
    expect(entry).toMatchObject({ id: "qwen3-tts-custom-1.7b", area: "voice", deleteTarget: "qwen3-tts-custom-1.7b", downloadKey: "qwen3-tts-custom-1.7b", familyId: "qwen3-tts", familyName: "Qwen3-TTS", sizeLabel: "3.20 GB", availableLocally: true });
    expect(entry.fit?.status).toBe("fits");
    expect(entry.memory?.label).toBe("~3.5 GB peak");
    expect(resolveModelStatus(entry).state).toBe("installed");
  });

  it("says what each model makes, from its capabilities", () => {
    expect(voiceMakes(row())).toEqual(["natural", "style"]);
    expect(voiceMakes(row({ kind: "base", capabilities: { ...caps, clone: true } }))).toEqual(["clone"]);
    expect(voiceMakes(row({ kind: "turbo", capabilities: { ...caps, clone: true, tags: true, presets: true } }))).toEqual(["natural", "tags", "clone"]);
    expect(voiceMakes(row({ kind: "voxcpm", capabilities: { ...caps, style: true, design: true, clone: true } }))).toEqual(["design", "clone"]);
  });

  it("puts both Chatterbox models under one family", () => {
    expect(voiceFamilyId("chatterbox-turbo")).toBe("chatterbox");
    expect(voiceFamilyId("chatterbox")).toBe("chatterbox");
    expect(voiceFamilyId(undefined)).toBe("qwen3-tts");
  });

  it("adds Kokoro first, installed with the runtime and never deletable on its own", () => {
    const kokoro = kokoroEntry({ voices: 22, languages: 2, sizeGb: 0.33, peakGb: 0.87, installed: true }, fit);
    const entries = buildVoiceEntries([row()], kokoro);
    expect(entries[0].id).toBe(KOKORO_ROW_ID);
    expect(isKokoro(entries[0].source)).toBe(true);
    expect(resolveModelStatus(entries[0]).state).toBe("installed");
    expect(voiceDownloadStatus(entries[0])).toBeNull();
    expect(buildVoiceEntries([row()], null)).toHaveLength(1);
  });

  it("turns the catalog's download state into the shared status ladder", () => {
    expect(resolveModelStatus(voiceRowToEntry(row({ installed: false })), { download: voiceDownloadStatus(voiceRowToEntry(row({ installed: false }))) }).state).toBe("available");
    const downloading = voiceRowToEntry(row({ installed: false, download: { state: "downloading", error: null } }));
    expect(resolveModelStatus(downloading, { download: voiceDownloadStatus(downloading) }).state).toBe("downloading");
    const failed = voiceRowToEntry(row({ installed: false, download: { state: "failed", error: "No network" } }));
    expect(resolveModelStatus(failed, { download: voiceDownloadStatus(failed) }).state).toBe("failed");
  });

  it("searches name, family, licence, languages and what it makes", () => {
    const entry = voiceRowToEntry(row());
    expect(voiceEntryMatchesQuery(entry, "style", LABELS)).toBe(true);
    expect(voiceEntryMatchesQuery(entry, "apache", LABELS)).toBe(true);
    expect(voiceEntryMatchesQuery(entry, "10 languages", LABELS)).toBe(true);
    expect(voiceEntryMatchesQuery(entry, "orpheus", LABELS)).toBe(false);
  });
});
