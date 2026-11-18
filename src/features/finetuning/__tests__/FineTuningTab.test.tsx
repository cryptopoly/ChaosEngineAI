import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { AdapterCard, type Adapter } from "../FineTuningTab";

/** The row shape GET /api/adapters really returns (no id, no size_mb). */
function row(over: Partial<Adapter> = {}): Adapter {
  return {
    name: "my-lora",
    path: "/Users/someone/Models/my-lora",
    base_model: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    created: "2026-11-18 07:40",
    size_bytes: 3 * 1048576,
    config: {},
    ...over,
  };
}

describe("AdapterCard", () => {
  it("renders a backend row; the tab used to crash on size_mb.toFixed when any adapter was on disk", () => {
    const html = renderToStaticMarkup(<AdapterCard adapter={row()} />);
    expect(html).toContain("my-lora");
    expect(html).toContain("Base: mlx-community/Qwen2.5-0.5B-Instruct-4bit");
    expect(html).toContain("Size: 3.0 MB");
    expect(html).toContain("Created: 2026-11-18 07:40");
  });

  it("leaves the Created line out when the backend has no date, and survives a missing size", () => {
    const html = renderToStaticMarkup(<AdapterCard adapter={row({ created: undefined, size_bytes: undefined as never })} />);
    expect(html).not.toContain("Created:");
    expect(html).toContain("Size: 0.0 MB");
  });
});
