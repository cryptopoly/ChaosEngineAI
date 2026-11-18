import { describe, expect, it } from "vitest";
import { findConversionSource } from "../conversionSource";
import type { LibraryItem } from "../../types";

function makeItem(overrides: Partial<LibraryItem>): LibraryItem {
  return {
    name: "org/model",
    path: "/models/org-model",
    format: "Transformers",
    sourceKind: "HF cache",
    quantization: "BF16",
    backend: "mlx",
    sizeGb: 4,
    lastModified: "2026-11-18 08:00",
    actions: [],
    ...overrides,
  } as LibraryItem;
}

describe("findConversionSource", () => {
  const gguf = makeItem({ name: "org/model-GGUF", path: "/hub/models--org--model-GGUF", format: "GGUF" });
  const other = makeItem({ name: "org/model-GGUF-2", path: "/hub/models--org--model-GGUF-2", format: "GGUF" });
  const folder = makeItem({});

  it("finds the item when the draft path is the item's own folder", () => {
    expect(findConversionSource([gguf, folder], "/models/org-model")).toBe(folder);
  });

  it("finds a GGUF item picked as one weights file inside its folder", () => {
    const file = "/hub/models--org--model-GGUF/snapshots/abc/model-Q4_K_M.gguf";
    expect(findConversionSource([folder, gguf], file)).toBe(gguf);
  });

  it("does not match a folder whose name only starts the same", () => {
    const file = "/hub/models--org--model-GGUF-2/snapshots/abc/model.gguf";
    expect(findConversionSource([gguf, other], file)).toBe(other);
  });

  it("handles Windows separators and an empty draft", () => {
    const win = makeItem({ path: "C:\\models\\qwen" });
    expect(findConversionSource([win], "C:\\models\\qwen\\model.gguf")).toBe(win);
    expect(findConversionSource([win, gguf], "")).toBeNull();
    expect(findConversionSource([gguf], "/elsewhere/model.gguf")).toBeNull();
  });
});
