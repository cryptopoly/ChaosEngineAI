import { describe, expect, it } from "vitest";
import { friendlyValidationDetail } from "./validationErrors";

describe("friendlyValidationDetail", () => {
  it("explains an out-of-range number in plain words", () => {
    const detail = [{ type: "less_than_equal", loc: ["body", "steps"], msg: "Input should be less than or equal to 100", input: 248, ctx: { le: 100 } }];
    expect(friendlyValidationDetail(detail)).toBe("Steps must be at most 100 (you entered 248).");
  });

  it("covers minimums, long text and missing fields", () => {
    expect(friendlyValidationDetail([{ type: "greater_than_equal", loc: ["body", "width"], input: 100, ctx: { ge: 256 } }])).toBe("Width must be at least 256 (you entered 100).");
    expect(friendlyValidationDetail([{ type: "string_too_long", loc: ["body", "prompt"], ctx: { max_length: 4000 } }])).toBe("Prompt is too long (at most 4000 characters).");
    expect(friendlyValidationDetail([{ type: "missing", loc: ["body", "modelId"] }])).toBe("Model id is required.");
  });

  it("ignores anything that is not a validation list", () => {
    expect(friendlyValidationDetail("plain text")).toBeNull();
    expect(friendlyValidationDetail([])).toBeNull();
    expect(friendlyValidationDetail(null)).toBeNull();
  });
});
