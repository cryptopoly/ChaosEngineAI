import { describe, expect, it } from "vitest";
import type { GenerationMetrics, PerfTelemetry } from "../../types";
import { buildPerfChips } from "../ChatPerfStrip";

function makeTelemetry(overrides: Partial<PerfTelemetry> = {}): PerfTelemetry {
  return { ...overrides };
}

describe("buildPerfChips", () => {
  it("returns empty when nothing is set", () => {
    expect(buildPerfChips(makeTelemetry(), null)).toEqual([]);
  });

  it("renders tok/s when positive", () => {
    const chips = buildPerfChips(makeTelemetry(), 42.5);
    expect(chips[0].label).toBe("42.5 tok/s");
  });

  it("flags slow tok/s as warn / alert", () => {
    expect(buildPerfChips(makeTelemetry(), 4)[0].tone).toBe("warn");
    expect(buildPerfChips(makeTelemetry(), 0.3)[0].tone).toBe("alert");
  });

  it("renders CPU + memory when present", () => {
    const chips = buildPerfChips(
      makeTelemetry({ cpuPercent: 45, availableMemoryGb: 12 }),
      null,
    );
    expect(chips.find((c) => c.key === "cpu")?.label).toBe("CPU 45%");
    expect(chips.find((c) => c.key === "mem")?.label).toBe("12.0 GB free");
  });

  it("flags high CPU as warn", () => {
    const chips = buildPerfChips(makeTelemetry({ cpuPercent: 95 }), null);
    expect(chips[0].tone).toBe("warn");
  });

  it("flags low memory as alert / warn", () => {
    const alert = buildPerfChips(makeTelemetry({ availableMemoryGb: 1 }), null);
    expect(alert[0].tone).toBe("alert");
    const warn = buildPerfChips(makeTelemetry({ availableMemoryGb: 3 }), null);
    expect(warn[0].tone).toBe("warn");
  });

  it("renders thermal state with appropriate tone", () => {
    expect(buildPerfChips(makeTelemetry({ thermalState: "nominal" }), null)[0].tone).toBe("default");
    expect(buildPerfChips(makeTelemetry({ thermalState: "moderate" }), null)[0].tone).toBe("warn");
    expect(buildPerfChips(makeTelemetry({ thermalState: "critical" }), null)[0].tone).toBe("alert");
  });

  it("omits zero / null tok/s", () => {
    expect(buildPerfChips(makeTelemetry({ cpuPercent: 50 }), 0)).toHaveLength(1);
    expect(buildPerfChips(makeTelemetry({ cpuPercent: 50 }), null)).toHaveLength(1);
  });

  it("composes a full chip set when all fields present", () => {
    const chips = buildPerfChips(
      makeTelemetry({
        cpuPercent: 30,
        gpuPercent: 80,
        availableMemoryGb: 16,
        thermalState: "nominal",
      }),
      40,
    );
    const keys = chips.map((c) => c.key).sort();
    expect(keys).toEqual(["cpu", "gpu", "mem", "thermal", "toks"]);
  });
});

describe("ChatPerfStrip integration shape", () => {
  it("metrics interface accepts perfTelemetry", () => {
    const metrics: GenerationMetrics = {
      finishReason: "stop",
      promptTokens: 5,
      completionTokens: 10,
      totalTokens: 15,
      tokS: 30,
      runtimeNote: null,
      perfTelemetry: { cpuPercent: 25, thermalState: "nominal" },
    };
    expect(metrics.perfTelemetry?.cpuPercent).toBe(25);
  });
});
