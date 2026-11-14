import type { TFn } from "./musicLogic";

export const UNSUPPORTED_PLATFORM_CODE = "music.unsupportedPlatform";

/** The sentence for a runtime that cannot run music or sounds here: translated by code, else the backend's text. */
export function unsupportedText(t: TFn, reason: string | null | undefined, code: string | null | undefined): string {
  if (code === UNSUPPORTED_PLATFORM_CODE) {
    return t("musicUnsupported.reason", {
      defaultValue:
        "Music generation runs on Apple Silicon Macs in this version. Windows and Linux support (NVIDIA GPUs) is planned; nothing is installed or downloaded on this computer.",
    });
  }
  return reason ?? "";
}

export function unsupportedBadge(t: TFn): string {
  return t("musicUnsupported.badge", { defaultValue: "Not available on this computer yet" });
}
