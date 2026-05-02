import { describe, expect, it } from "vitest";
import { lastLines, tidyReasoningForDisplay } from "../ReasoningPanel";

describe("tidyReasoningForDisplay", () => {
  it("returns empty for empty input", () => {
    expect(tidyReasoningForDisplay("")).toBe("");
  });

  it("strips leading whitespace + newlines", () => {
    expect(tidyReasoningForDisplay("\n\n   Okay let me think.")).toBe("Okay let me think.");
  });

  it("collapses the first paragraph break to a single newline", () => {
    // Models often emit:  "Okay, the user wants...\n\nLet me explore..."
    // which renders as two paragraphs with a tall margin between them.
    // We collapse the very first \n\n to a single newline.
    const input = "Okay, the user wants X.\n\nLet me explore Y.";
    expect(tidyReasoningForDisplay(input)).toBe("Okay, the user wants X.\nLet me explore Y.");
  });

  it("preserves mid-stream paragraph breaks beyond the first", () => {
    const input = "First.\n\nSecond.\n\nThird.";
    // Only the first \n\n collapses; subsequent paragraph breaks stay.
    expect(tidyReasoningForDisplay(input)).toBe("First.\nSecond.\n\nThird.");
  });

  it("leaves single-line content alone", () => {
    expect(tidyReasoningForDisplay("just one line")).toBe("just one line");
  });

  it("leaves content with no leading whitespace + no early gap alone", () => {
    expect(tidyReasoningForDisplay("Hi.\nLow.")).toBe("Hi.\nLow.");
  });
});

describe("lastLines", () => {
  it("returns empty when there are no non-empty lines", () => {
    expect(lastLines("\n\n   \n", 2)).toBe("");
  });

  it("returns the last N lines joined with a separator", () => {
    expect(lastLines("first\nsecond\nthird\nfourth", 2)).toBe("third · fourth");
  });

  it("returns fewer when the source has fewer than N lines", () => {
    expect(lastLines("only one", 2)).toBe("only one");
  });

  it("trims whitespace inside lines and skips empties", () => {
    expect(lastLines("  alpha  \n\n  beta  ", 2)).toBe("alpha · beta");
  });
});
