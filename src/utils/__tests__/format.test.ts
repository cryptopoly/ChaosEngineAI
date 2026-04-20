import { describe, expect, it } from "vitest";

import {
  formatImageAccessError,
  formatImageLicenseLabel,
  formatImageTimestamp,
  formatReleaseLabel,
  imageRuntimeErrorStatus,
  isGatedImageAccessError,
  number,
  signedDelta,
  sizeLabel,
} from "../format";

describe("number()", () => {
  it("formats zero", () => {
    expect(number(0)).toBe("0.0");
  });

  it("formats a fractional value with default digits", () => {
    expect(number(1.5)).toBe("1.5");
  });

  it("formats a large integer with default digits", () => {
    expect(number(100)).toBe("100.0");
  });

  it("rounds to the requested number of digits", () => {
    expect(number(3.14159, 2)).toBe("3.14");
  });

  it("pads with trailing zeros when needed", () => {
    expect(number(7, 3)).toBe("7.000");
  });
});

describe("sizeLabel()", () => {
  it("returns a formatted GB label for positive values", () => {
    expect(sizeLabel(4.2)).toBe("4.2 GB");
  });

  it("returns 'Unknown' for zero", () => {
    expect(sizeLabel(0)).toBe("Unknown");
  });

  it("returns 'Unknown' for negative values", () => {
    expect(sizeLabel(-1)).toBe("Unknown");
  });
});

describe("signedDelta()", () => {
  it("prefixes positive values with +", () => {
    expect(signedDelta(2.5)).toBe("+2.5");
  });

  it("does not prefix negative values", () => {
    expect(signedDelta(-1.3)).toBe("-1.3");
  });

  it("does not prefix zero", () => {
    expect(signedDelta(0)).toBe("0.0");
  });

  it("appends a suffix when provided", () => {
    expect(signedDelta(5, 0, "%")).toBe("+5%");
  });
});

describe("formatImageTimestamp()", () => {
  it("formats a valid ISO date string", () => {
    const result = formatImageTimestamp("2024-06-15T14:30:00Z");
    // The exact output depends on locale, but it should not be the raw string
    expect(result).not.toBe("2024-06-15T14:30:00Z");
    expect(result.length).toBeGreaterThan(0);
  });

  it("returns the original string for an invalid date", () => {
    expect(formatImageTimestamp("not-a-date")).toBe("not-a-date");
  });
});

describe("formatImageLicenseLabel()", () => {
  it("returns null for falsy input", () => {
    expect(formatImageLicenseLabel(null)).toBeNull();
    expect(formatImageLicenseLabel(undefined)).toBeNull();
    expect(formatImageLicenseLabel("")).toBeNull();
  });

  it("capitalizes words and replaces hyphens/underscores with spaces", () => {
    expect(formatImageLicenseLabel("apache-2.0")).toBe("Apache 2.0");
    expect(formatImageLicenseLabel("mit_license")).toBe("Mit License");
  });
});

describe("isGatedImageAccessError()", () => {
  it("returns false for null/undefined", () => {
    expect(isGatedImageAccessError(null)).toBe(false);
    expect(isGatedImageAccessError(undefined)).toBe(false);
  });

  it("detects gated repo messages", () => {
    expect(isGatedImageAccessError("Cannot access gated repo")).toBe(true);
    expect(isGatedImageAccessError("gated repo requires auth")).toBe(true);
    expect(isGatedImageAccessError("not on the authorized list")).toBe(true);
    expect(isGatedImageAccessError("Access to model is restricted")).toBe(true);
  });

  it("returns false for unrelated errors", () => {
    expect(isGatedImageAccessError("network timeout")).toBe(false);
  });
});

describe("formatImageAccessError()", () => {
  it("returns empty string for falsy message", () => {
    expect(formatImageAccessError(null)).toBe("");
    expect(formatImageAccessError(undefined)).toBe("");
  });

  it("returns original message for non-gated errors", () => {
    expect(formatImageAccessError("some other error")).toBe("some other error");
  });

  it("returns gated-access guidance for gated errors", () => {
    const result = formatImageAccessError("Cannot access gated repo", {
      name: "SDXL",
      link: "https://hf.co/sdxl",
    });
    expect(result).toContain("SDXL");
    expect(result).toContain("gated on Hugging Face");
  });
});

describe("imageRuntimeErrorStatus()", () => {
  it("extracts message from Error instances", () => {
    const status = imageRuntimeErrorStatus(new Error("boom"));
    expect(status.activeEngine).toBe("unavailable");
    expect(status.realGenerationAvailable).toBe(false);
    expect(status.message).toBe("boom");
  });

  it("uses a fallback message for non-Error values", () => {
    const status = imageRuntimeErrorStatus("string error");
    expect(status.message).toBe("Image runtime unavailable.");
  });
});

describe("formatReleaseLabel()", () => {
  it("prefers an already-formatted backend label", () => {
    expect(formatReleaseLabel("Released Aug 2024", "2024-08-01")).toBe("Released Aug 2024");
  });

  it("parses a YYYY-MM curated shorthand", () => {
    expect(formatReleaseLabel(null, "2024-08")).toBe("Released Aug 2024");
  });

  it("parses a YYYY-MM-DD curated shorthand", () => {
    expect(formatReleaseLabel(null, "2025-02-15")).toBe("Released Feb 2025");
  });

  it("parses a full ISO datetime from the Hugging Face API", () => {
    expect(formatReleaseLabel(null, "2024-11-12T09:30:00.000Z")).toBe("Released Nov 2024");
  });

  it("returns null when both inputs are empty or invalid", () => {
    expect(formatReleaseLabel(null, null)).toBeNull();
    expect(formatReleaseLabel(null, "not-a-date")).toBeNull();
    expect(formatReleaseLabel("", "")).toBeNull();
  });
});
