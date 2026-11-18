import { describe, expect, it } from "vitest";
import { maskSecret, maskSecretsIn } from "../secretMask";

const KEY = "ce_live_7f3a9b2c1d0e4f5a6b7c8d9e0f1a2b3c";

describe("maskSecret", () => {
  it("shows dots and only the last 4 characters", () => {
    expect(maskSecret(KEY)).toBe("••••••••2b3c");
    expect(maskSecret(KEY)).not.toContain(KEY.slice(0, 4));
  });

  it("shows only dots for a short secret", () => {
    expect(maskSecret("tok")).toBe("••••••••");
    expect(maskSecret("12345678")).toBe("••••••••");
  });
});

describe("maskSecretsIn", () => {
  it("masks every occurrence in a code sample and leaves the rest", () => {
    const sample = `curl -H 'Authorization: Bearer ${KEY}' http://127.0.0.1:8876/v1/models # ${KEY}`;
    const shown = maskSecretsIn(sample, [KEY]);
    expect(shown).not.toContain(KEY);
    expect(shown).toBe("curl -H 'Authorization: Bearer ••••••••2b3c' http://127.0.0.1:8876/v1/models # ••••••••2b3c");
  });

  it("masks several secrets, the longer one whole, and skips empty ones", () => {
    const read = `${KEY}-read`;
    const shown = maskSecretsIn(`a=${KEY} b=${read}`, [KEY, read, null, undefined, ""]);
    expect(shown).not.toContain(KEY);
    expect(shown).toBe("a=••••••••2b3c b=••••••••read");
  });
});
