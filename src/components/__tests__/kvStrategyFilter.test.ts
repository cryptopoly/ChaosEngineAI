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
const FBCACHE = makeStrategy({ id: "fbcache", name: "First Block Cache", appliesTo: ["image", "video"] });

const ALL = [NATIVE, ROTORQUANT, TURBOQUANT, CHAOSENGINE, TRIATTENTION, TEACACHE, FBCACHE];

describe("filterTextStrategies", () => {
  it("returns empty for null input", () => {
    expect(filterTextStrategies(undefined, "mlx")).toEqual([]);
  });

  it("drops diffusion-only strategies (TeaCache, FBCache) for any text engine", () => {
    const out = filterTextStrategies(ALL, "mlx").map((s) => s.id);
    expect(out).not.toContain("teacache");
    expect(out).not.toContain("fbcache");
  });

  it("MLX engine: only native + turboquant (matches launch-settings modal)", () => {
    // RotorQuant + ChaosEngine require llama.cpp / vLLM substrate;
    // TriAttention requires vLLM. STRATEGY_ENGINE_SUPPORT in
    // runtimeSupport.ts is the single source of truth; the chip
    // mirrors the modal verdict so users don't see options the
    // modal would mark N/A.
    const out = filterTextStrategies(ALL, "mlx").map((s) => s.id);
    expect(out.sort()).toEqual(["native", "turboquant"]);
  });

  it("llama.cpp engine: native + rotorquant + turboquant + chaosengine", () => {
    const out = filterTextStrategies(ALL, "llama.cpp").map((s) => s.id);
    expect(out.sort()).toEqual(["chaosengine", "native", "rotorquant", "turboquant"]);
  });

  it("gguf substring matches the llama.cpp set (engine label can be 'gguf')", () => {
    const out = filterTextStrategies(ALL, "gguf").map((s) => s.id);
    expect(out.sort()).toEqual(["chaosengine", "native", "rotorquant", "turboquant"]);
  });

  it("vllm engine: full set including triattention (matches modal)", () => {
    // ``STRATEGY_ENGINE_SUPPORT`` lists rotorquant / chaosengine /
    // turboquant as vLLM-compatible alongside triattention, so the
    // chip mirrors the modal and shows them all. Diffusion-only
    // strategies (TeaCache / FBCache) stay out via layer 1.
    const out = filterTextStrategies(ALL, "vllm").map((s) => s.id);
    expect(out.sort()).toEqual([
      "chaosengine",
      "native",
      "rotorquant",
      "triattention",
      "turboquant",
    ]);
  });

  it("unknown engine: keeps all compatible text strategies (safe default)", () => {
    // ``isStrategyCompatible`` returns true for unknown engines so a
    // freshly-loaded substrate doesn't accidentally hide everything.
    const out = filterTextStrategies(ALL, "made-up").map((s) => s.id);
    expect(out).toContain("native");
    expect(out).not.toContain("teacache");
  });

  it("missing engine: keeps every available text strategy", () => {
    const out = filterTextStrategies(ALL, null).map((s) => s.id);
    expect(out).not.toContain("teacache");
    expect(out.length).toBeGreaterThan(0);
  });

  it("drops unavailable non-native strategies entirely (matches modal N/A badge)", () => {
    const unavailableTriattention = makeStrategy({
      id: "triattention",
      name: "TriAttention",
      available: false,
    });
    // vLLM substrate would normally accept TriAttention; flagging it
    // ``available: false`` (no pip wheel installed) should hide it.
    const out = filterTextStrategies([NATIVE, unavailableTriattention], "vllm").map(
      (s) => s.id,
    );
    expect(out).toEqual(["native"]);
  });

  it("native survives even when its ``available`` flag is false", () => {
    // Defensive: native f16 has no install dependency; if a future
    // backend regression flips the flag we still want the user to be
    // able to fall back to it without the chip going empty.
    const nativeFalse = makeStrategy({
      id: "native",
      name: "Native f16",
      available: false,
    });
    const out = filterTextStrategies([nativeFalse], "mlx").map((s) => s.id);
    expect(out).toEqual(["native"]);
  });

  it("missing appliesTo defaults to text (back-compat)", () => {
    const noAppliesTo = makeStrategy({ id: "native", name: "Native (legacy shape)" });
    delete (noAppliesTo as { appliesTo?: string[] }).appliesTo;
    const out = filterTextStrategies([noAppliesTo], null).map((s) => s.id);
    expect(out).toContain("native");
  });
});
