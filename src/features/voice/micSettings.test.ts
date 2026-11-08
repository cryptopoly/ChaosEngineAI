import { describe, expect, it } from "vitest";
import { micErrorMessage } from "./audio";
import { micSettingsUrl, micSettingsUrlForError } from "./micSettings";

describe("micSettingsUrl", () => {
  it("picks the OS privacy page", () => {
    expect(micSettingsUrl("Darwin")).toBe("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone");
    expect(micSettingsUrl("Windows")).toBe("ms-settings:privacy-microphone");
  });

  it("has no page for Linux or an unknown platform", () => {
    expect(micSettingsUrl("Linux")).toBeNull();
    expect(micSettingsUrl("")).toBeNull();
  });
});

describe("micSettingsUrlForError", () => {
  const denied = new DOMException("no", "NotAllowedError");

  it("offers the page only for the blocked error", () => {
    expect(micSettingsUrlForError(micErrorMessage(denied, "Darwin"), "Darwin")).toContain("Privacy_Microphone");
    expect(micSettingsUrlForError(micErrorMessage(denied, "Windows"), "Windows")).toBe("ms-settings:privacy-microphone");
  });

  it("offers nothing for other errors, Linux, or no error", () => {
    expect(micSettingsUrlForError(micErrorMessage(new DOMException("x", "NotFoundError"), "Darwin"), "Darwin")).toBeNull();
    expect(micSettingsUrlForError(micErrorMessage(denied, "Linux"), "Linux")).toBeNull();
    expect(micSettingsUrlForError(null, "Darwin")).toBeNull();
  });
});
