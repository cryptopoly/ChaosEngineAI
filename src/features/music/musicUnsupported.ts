import type { TFn } from "./musicLogic";

export const UNSUPPORTED_PLATFORM_CODE = "music.unsupportedPlatform";
/** Music runs here (an NVIDIA GPU) but the sounds lane does not yet. */
export const SOUND_UNSUPPORTED_PLATFORM_CODE = "music.soundUnsupportedPlatform";

/** The sentence for a runtime that cannot run music or sounds here: translated by code, else the backend's text. */
export function unsupportedText(t: TFn, reason: string | null | undefined, code: string | null | undefined): string {
  if (code === UNSUPPORTED_PLATFORM_CODE) {
    return t("musicUnsupported.reason", {
      defaultValue:
        "Music generation needs an Apple Silicon Mac or an NVIDIA GPU on Windows or Linux; nothing is installed or downloaded on this computer.",
    });
  }
  if (code === SOUND_UNSUPPORTED_PLATFORM_CODE) {
    return t("musicUnsupported.soundReason", {
      defaultValue:
        "Sound effects run on Apple Silicon Macs in this version. NVIDIA GPU support is planned; nothing is installed or downloaded for sounds on this computer.",
    });
  }
  return reason ?? "";
}

export function unsupportedBadge(t: TFn): string {
  return t("musicUnsupported.badge", { defaultValue: "Not available on this computer yet" });
}
