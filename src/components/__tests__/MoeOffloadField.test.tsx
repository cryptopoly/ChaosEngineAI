import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MoeOffloadField, moeCpuLayersForLoad } from "../MoeOffloadField";

const OFFLOAD = { blocks: 40, moeLayers: 40, expertGb: 16.9, autoCpuLayers: 21, gpuGb: 10.9, ramGb: 9.1, freeGpuGb: 12 };

describe("moeCpuLayersForLoad (FU-550)", () => {
  it("sends the count only for the model it was chosen for", () => {
    const settings = { moeCpuLayers: 20, moeCpuLayersFor: "ggml-org/Qwen3.6-35B-A3B-GGUF" };
    expect(moeCpuLayersForLoad(settings, "ggml-org/Qwen3.6-35B-A3B-GGUF")).toBe(20);
    expect(moeCpuLayersForLoad(settings, "other/model")).toBeNull();
  });

  it("is automatic (null) when nothing was chosen, and keeps an explicit zero", () => {
    expect(moeCpuLayersForLoad({}, "m")).toBeNull();
    expect(moeCpuLayersForLoad({ moeCpuLayers: null, moeCpuLayersFor: "m" }, "m")).toBeNull();
    expect(moeCpuLayersForLoad({ moeCpuLayers: 0, moeCpuLayersFor: "m" }, "m")).toBe(0);
  });
});

describe("MoeOffloadField", () => {
  it("shows automatic with the count it would pick and the GPU / RAM split", () => {
    const markup = renderToStaticMarkup(
      <MoeOffloadField offload={OFFLOAD} modelRef="m" value={null} onChange={() => undefined} />,
    );
    expect(markup).toContain("Experts in system RAM");
    expect(markup).toContain("Automatic (about 21 of 40 layers)");
    expect(markup).toContain("about 10.9 GB on the GPU, 9.1 GB in system RAM");
  });

  it("shows a chosen count and none", () => {
    const chosen = renderToStaticMarkup(<MoeOffloadField offload={OFFLOAD} modelRef="m" value={20} onChange={() => undefined} />);
    expect(chosen).toContain("20 of 40 layers");
    const none = renderToStaticMarkup(<MoeOffloadField offload={OFFLOAD} modelRef="m" value={0} onChange={() => undefined} />);
    expect(none).toContain("None, all on the GPU");
  });
});
