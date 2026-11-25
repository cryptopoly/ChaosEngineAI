import { describe, expect, it } from "vitest";
import {
  CLIP_SOUND_MUTED_DB,
  CLIP_SOUND_ON_DB,
  clipSoundDraftOn,
  clipSoundDraftValue,
  clipSoundPatches,
  clipSoundState,
  hasSong,
} from "../clipSound";
import { UPSCALE_METHODS, upscaleMethodNote, upscaleMethodOf, upscaleMethodPatch } from "../upscaleMethod";

const db = (...levels: number[]) => levels.map((nativeSoundDb, i) => ({ id: `s${i + 1}`, nativeSoundDb }));

describe("clip sound under a song", () => {
  it("only talks about the song when the project has one", () => {
    expect(hasSong({ song: null })).toBe(false);
    expect(hasSong({})).toBe(false);
    expect(hasSong({ song: { path: "media/song.flac" } as never })).toBe(true);
  });

  it("is muted when no shot is heard, playing when all are, mixed in between", () => {
    expect(clipSoundState(db(-60, -60))).toBe("muted");
    expect(clipSoundState(db(-45, -60))).toBe("muted");
    expect(clipSoundState(db(0, -6))).toBe("playing");
    expect(clipSoundState(db(0, -60))).toBe("mixed");
  });

  it("turning the switch on sets muted shots to 0 dB and leaves shots already heard alone", () => {
    expect(clipSoundPatches(db(-60, -6, -60), true)).toEqual([
      { shotId: "s1", nativeSoundDb: CLIP_SOUND_ON_DB },
      { shotId: "s3", nativeSoundDb: CLIP_SOUND_ON_DB },
    ]);
    expect(CLIP_SOUND_ON_DB).toBe(0);
  });

  it("turning it off sets heard shots to -60 dB and skips shots already muted", () => {
    expect(clipSoundPatches(db(0, -60, -3), false)).toEqual([
      { shotId: "s1", nativeSoundDb: CLIP_SOUND_MUTED_DB },
      { shotId: "s3", nativeSoundDb: CLIP_SOUND_MUTED_DB },
    ]);
    expect(CLIP_SOUND_MUTED_DB).toBe(-60);
  });

  it("makes no change when the shots already match", () => {
    expect(clipSoundPatches(db(-60, -60), false)).toEqual([]);
    expect(clipSoundPatches(db(0, -6), true)).toEqual([]);
  });

  it("writes -60 or 0 into a shot editor draft and reads the draft's switch from its text", () => {
    expect(clipSoundDraftValue(true)).toBe("0");
    expect(clipSoundDraftValue(false)).toBe("-60");
    expect(clipSoundDraftOn("0")).toBe(true);
    expect(clipSoundDraftOn("-6")).toBe(true);
    expect(clipSoundDraftOn("-60")).toBe(false);
    expect(clipSoundDraftOn("")).toBe(false);
    expect(clipSoundDraftOn("abc")).toBe(false);
  });
});

describe("the upscale method choice", () => {
  const base = { aiLabel: { embedTag: true }, loudnessLufs: null, captionsSrt: false, captionsBurn: false, stems: false, upscale: "1440p" as const };

  it("offers the sharper method first and the standard stretch second", () => {
    expect(UPSCALE_METHODS).toEqual(["fast", "lanczos"]);
  });

  it("reads the stored method and treats a missing one as the standard stretch", () => {
    expect(upscaleMethodOf({ export: base })).toBe("lanczos");
    expect(upscaleMethodOf({ export: { ...base, upscaleMethod: "fast" } as never })).toBe("fast");
    expect(upscaleMethodOf({ export: { ...base, upscaleMethod: "lanczos" } as never })).toBe("lanczos");
  });

  it("writes the choice into the project's export settings", () => {
    expect(upscaleMethodPatch("fast")).toEqual({ export: { upscaleMethod: "fast" } });
    expect(upscaleMethodPatch("lanczos")).toEqual({ export: { upscaleMethod: "lanczos" } });
  });

  it("says something only for the sharper method on a film that is scaled up", () => {
    expect(upscaleMethodNote("none", "fast", false)).toBeNull();
    expect(upscaleMethodNote("1440p", "lanczos", false)).toBeNull();
    expect(upscaleMethodNote("1440p", "fast", false)).toBe("needs-addon");
    expect(upscaleMethodNote("2160p", "fast", true)).toBe("ready");
    expect(upscaleMethodNote("2160p", "fast", null)).toBe("checking");
  });
});
