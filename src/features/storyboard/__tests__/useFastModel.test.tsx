import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { StoryboardStillModels } from "../../../api/storyboard";
import { UseFastModelButton } from "../UseFastModelButton";

const models: StoryboardStillModels = {
  models: [
    { id: "qwen", name: "Qwen-Image 2.1", fit: null, territoryRestricted: false, secondsPerStill: 600 },
    { id: "iris", name: "Iris-3B Turbo", fit: null, territoryRestricted: false, secondsPerStill: 15 },
  ],
  fast: "iris",
  quality: "qwen",
  width: 1280,
  height: 720,
};
const noop = () => undefined;

describe("Use the fast model button", () => {
  it("names the fast model and its time per still when the project uses a slower one", () => {
    const html = renderToStaticMarkup(<UseFastModelButton models={models} saved="qwen" disabled={false} onUse={noop} />);
    expect(html).toContain(">Use the fast model</button>");
    expect(html).toContain("Iris-3B Turbo, about 15 s per still");
  });

  it("is not there on Automatic or the fast model", () => {
    expect(renderToStaticMarkup(<UseFastModelButton models={models} saved={null} disabled={false} onUse={noop} />)).toBe("");
    expect(renderToStaticMarkup(<UseFastModelButton models={models} saved="iris" disabled={false} onUse={noop} />)).toBe("");
    expect(renderToStaticMarkup(<UseFastModelButton models={null} saved="qwen" disabled={false} onUse={noop} />)).toBe("");
  });
});
