import { describe, expect, it } from "vitest";
import type { SystemStats } from "../../types";
import { filterTextStrategies } from "../kvStrategyFilter";

type Strategy = NonNullable<SystemStats["availableCacheStrategies"]>[number];

function makeStrategy(overrides: Partial<Strategy>): Strategy {
  return {
    id: overrides.id ?? "test",
    name: overrides.name ?? "Test",
    available: overrides.available ?? true,
    bitRange: overrides.bitRange ?? null,
    defaultBits: overrides.defaultBits ?? null,
    supportsFp16Layers: overrides.supportsFp16Layers ?? false,
    appliesTo: overrides.appliesTo ?? ["text"],
    ...overrides,
  } as Strategy;
}

const NATIVE = makeStrategy({ id: "native", name: "Native f16" });
const ROTORQUANT = makeStrategy({ id: "rotorquant", name: "RotorQuant", requiredLlamaBinary: "turbo" });
const TURBOQUANT = makeStrategy({ id: "turboquant", name: "TurboQuant", requiredLlamaBinary: "turbo" });
const CHAOSENGINE = makeStrategy({ id: "chaosengine", name: "ChaosEngine" });
const TRIATTENTION = makeStrategy({ id: "triattention", name: "TriAttention" });
const TEACACHE = makeStrategy({ id: "teacache", name: "TeaCache", appliesTo: ["image", "video"] });

const ALL = [NATIVE, ROTORQUANT, TURBOQUANT, CHAOSENGINE, TRIATTENTION, TEACACHE];

describe("filterTextStrategies", () => {
  it("returns empty for null input", () => {
    expect(filterTextStrategies(undefined, "mlx")).toEqual([]);
  });

  it("drops diffusion-only strategies for any text engine", () => {
    const out = filterTextStrategies(ALL, "mlx").map((s) => s.id);
    expect(out).not.toContain("teacache");
  });

  it("MLX engine: only native / turboquant / triattention", () => {
    const out = filterTextStrategies(ALL, "mlx").map((s) => s.id);
    expect(out.sort()).toEqual(["native", "triattention", "turboquant"]);
  });

  it("mlx_worker engine: same set as mlx", () => {
    const out = filterTextStrategies(ALL, "mlx_worker").map((s) => s.id);
    expect(out.sort()).toEqual(["native", "triattention", "turboquant"]);
  });

  it("llamacpp engine: native + rotorquant + turboquant + chaosengine", () => {
    const out = filterTextStrategies(ALL, "llamacpp").map((s) => s.id);
    expect(out.sort()).toEqual(["chaosengine", "native", "rotorquant", "turboquant"]);
  });

  it("vllm engine: native + triattention only", () => {
    const out = filterTextStrategies(ALL, "vllm").map((s) => s.id);
    expect(out.sort()).toEqual(["native", "triattention"]);
  });

  it("unknown engine: keeps all text strategies (safe default)", () => {
    const out = filterTextStrategies(ALL, "made-up").map((s) => s.id);
    expect(out).toContain("native");
    expect(out).not.toContain("teacache");
  });

  it("missing engine: keeps all text strategies", () => {
    const out = filterTextStrategies(ALL, null).map((s) => s.id);
    expect(out).not.toContain("teacache");
    expect(out.length).toBeGreaterThan(0);
  });

  it("case-insensitive engine match", () => {
    const out = filterTextStrategies(ALL, "MLX").map((s) => s.id);
    expect(out).toContain("native");
    expect(out).not.toContain("rotorquant");
  });

  it("missing appliesTo defaults to text (back-compat)", () => {
    const noAppliesTo = makeStrategy({ id: "native", name: "Native (legacy shape)" });
    delete (noAppliesTo as { appliesTo?: string[] }).appliesTo;
    // With no engine constraint, the missing appliesTo entry survives.
    const out = filterTextStrategies([noAppliesTo], null).map((s) => s.id);
    expect(out).toContain("native");
  });
});
