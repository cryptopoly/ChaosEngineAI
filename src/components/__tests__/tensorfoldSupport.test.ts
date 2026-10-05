import { describe, expect, it } from "vitest";

import {
  canAddTensorfoldExtra,
  isTensorfoldRepo,
  resolveTensorfoldSupport,
  tensorfoldHasExtra,
  tensorfoldEngaged,
  tensorfoldLaunchInfoFor,
  tensorfoldMemoryShortfall,
  tensorfoldModelList,
  type TensorfoldInfo,
} from "../tensorfoldSupport";
import {
  isStrategyCompatible,
  sanitizeSpeculativeSelection,
  strategyIncompatReason,
} from "../runtimeSupport";

const FLASH = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP";
const DENSE = "Vontra/Qwen3.8-27B-MLX-4bit";
const GEMMA = "mlx-community/gemma-4-26b-a4b-it-4bit";

function makeInfo(overrides: Partial<TensorfoldInfo> = {}): TensorfoldInfo {
  return {
    available: true,
    version: "0.5.0",
    supportedModels: [DENSE, FLASH, GEMMA],
    exclusiveModels: [FLASH],
    visionModels: [DENSE],
    extras: [],
    minMemoryGb: { [FLASH]: 128 },
    ...overrides,
  };
}

describe("resolveTensorfoldSupport()", () => {
  it("returns null without registry info or candidates", () => {
    expect(resolveTensorfoldSupport(undefined, [DENSE])).toBeNull();
    expect(resolveTensorfoldSupport(null, [DENSE])).toBeNull();
    expect(resolveTensorfoldSupport(makeInfo(), [])).toBeNull();
    expect(resolveTensorfoldSupport(makeInfo(), [null, undefined, "  "])).toBeNull();
  });

  it("matches the exact repo id, ignoring case", () => {
    expect(resolveTensorfoldSupport(makeInfo(), [DENSE.toUpperCase()])?.repo).toBe(DENSE);
    expect(resolveTensorfoldSupport(makeInfo(), [null, DENSE])?.repo).toBe(DENSE);
  });

  it("does not fuzzy-match a different conversion of the same model", () => {
    // DFlash / MTPLX match by normalised name; TensorFold is tested with
    // specific conversions, so these must not count.
    expect(resolveTensorfoldSupport(makeInfo(), ["mlx-community/Qwen3.8-27B-4bit"])).toBeNull();
    expect(resolveTensorfoldSupport(makeInfo(), ["Qwen3.8-27B-MLX-4bit"])).toBeNull();
    expect(resolveTensorfoldSupport(makeInfo(), ["lmstudio-community/gemma-4-26b-a4b-it-4bit"])).toBeNull();
  });

  it("splits exclusive from tested checkpoints and carries the memory floor", () => {
    const flash = resolveTensorfoldSupport(makeInfo(), [FLASH]);
    expect(flash).toEqual({ tier: "exclusive", repo: FLASH, minMemoryGb: 128, vision: false });
    const dense = resolveTensorfoldSupport(makeInfo(), [DENSE]);
    expect(dense).toEqual({ tier: "tested", repo: DENSE, minMemoryGb: null, vision: true });
  });

  it("marks only the checkpoints with a vision tower as able to read images", () => {
    expect(resolveTensorfoldSupport(makeInfo(), [DENSE])?.vision).toBe(true);
    expect(resolveTensorfoldSupport(makeInfo(), [GEMMA])?.vision).toBe(false);
    // An older backend that does not publish visionModels offers no image input.
    expect(resolveTensorfoldSupport(makeInfo({ visionModels: undefined }), [DENSE])?.vision).toBe(false);
  });
});

describe("tensorfoldMemoryShortfall()", () => {
  const flash = resolveTensorfoldSupport(makeInfo(), [FLASH]);
  const dense = resolveTensorfoldSupport(makeInfo(), [DENSE]);

  it("flags a Mac below the stated minimum", () => {
    expect(tensorfoldMemoryShortfall(flash, 64)).toBe(true);
    expect(tensorfoldMemoryShortfall(flash, 127.9)).toBe(true);
  });

  it("passes at or above the minimum", () => {
    expect(tensorfoldMemoryShortfall(flash, 128)).toBe(false);
    expect(tensorfoldMemoryShortfall(flash, 512)).toBe(false);
  });

  it("never warns without a stated minimum or an unknown memory size", () => {
    expect(tensorfoldMemoryShortfall(dense, 8)).toBe(false);
    expect(tensorfoldMemoryShortfall(null, 8)).toBe(false);
    expect(tensorfoldMemoryShortfall(flash, 0)).toBe(false);
    expect(tensorfoldMemoryShortfall(flash, undefined)).toBe(false);
  });
});

describe("tensorfoldEngaged()", () => {
  it("is false unless TensorFold is installed", () => {
    const support = resolveTensorfoldSupport(makeInfo(), [FLASH])!;
    expect(tensorfoldEngaged({ available: false, support }, true)).toBe(false);
    expect(tensorfoldEngaged(null, true)).toBe(false);
  });

  it("always runs exclusive checkpoints, whatever the toggle says", () => {
    const support = resolveTensorfoldSupport(makeInfo(), [FLASH])!;
    expect(tensorfoldEngaged({ available: true, support }, false)).toBe(true);
    expect(tensorfoldEngaged({ available: true, support }, true)).toBe(true);
  });

  it("runs tested checkpoints only while speculative decoding is ticked (mirrors the backend rule)", () => {
    const support = resolveTensorfoldSupport(makeInfo(), [DENSE])!;
    expect(tensorfoldEngaged({ available: true, support }, true)).toBe(true);
    expect(tensorfoldEngaged({ available: true, support }, false)).toBe(false);
  });
});

describe("tensorfoldLaunchInfoFor()", () => {
  it("is null for models TensorFold does not serve", () => {
    expect(tensorfoldLaunchInfoFor({ info: makeInfo() }, ["mlx-community/Llama-3.2-3B-Instruct-4bit"])).toBeNull();
    expect(tensorfoldLaunchInfoFor(undefined, [DENSE])).toBeNull();
    expect(tensorfoldLaunchInfoFor({}, [DENSE])).toBeNull();
  });

  it("carries install state and version for a served model", () => {
    const launch = tensorfoldLaunchInfoFor({ info: makeInfo({ available: false, version: null }) }, [null, DENSE]);
    expect(launch?.available).toBe(false);
    expect(launch?.support.repo).toBe(DENSE);
    const installed = tensorfoldLaunchInfoFor({ info: makeInfo() }, [DENSE]);
    expect(installed).toMatchObject({ available: true, version: "0.5.0" });
  });

  it("carries the installed extras, defaulting to none for an older backend", () => {
    const withExtras = tensorfoldLaunchInfoFor({ info: makeInfo({ extras: ["vision"] }) }, [DENSE]);
    expect(withExtras?.extras).toEqual(["vision"]);
    const legacy = tensorfoldLaunchInfoFor({ info: makeInfo({ extras: undefined }) }, [DENSE]);
    expect(legacy?.extras).toEqual([]);
  });
});

describe("optional extras (image input, structured output)", () => {
  const launch = (infoOverrides: Partial<TensorfoldInfo>, repo = DENSE) => {
    const result = tensorfoldLaunchInfoFor({ info: makeInfo(infoOverrides) }, [repo]);
    if (!result) throw new Error("expected a launch info");
    return result;
  };

  it("reports which extras are already installed", () => {
    expect(tensorfoldHasExtra(launch({ extras: ["vision"] }), "vision")).toBe(true);
    expect(tensorfoldHasExtra(launch({ extras: ["vision"] }), "grammar")).toBe(false);
    expect(tensorfoldHasExtra(null, "vision")).toBe(false);
  });

  it("offers image support only for a checkpoint with a vision tower", () => {
    expect(canAddTensorfoldExtra(launch({}), "vision")).toBe(true);
    expect(canAddTensorfoldExtra(launch({}, GEMMA), "vision")).toBe(false);
    expect(canAddTensorfoldExtra(launch({}, FLASH), "vision")).toBe(false);
  });

  it("offers structured output for any served checkpoint", () => {
    expect(canAddTensorfoldExtra(launch({}, GEMMA), "grammar")).toBe(true);
    expect(canAddTensorfoldExtra(launch({}, FLASH), "grammar")).toBe(true);
  });

  it("offers nothing that is already installed, or before TensorFold itself is", () => {
    expect(canAddTensorfoldExtra(launch({ extras: ["vision", "grammar"] }), "vision")).toBe(false);
    expect(canAddTensorfoldExtra(launch({ extras: ["vision", "grammar"] }), "grammar")).toBe(false);
    expect(canAddTensorfoldExtra(launch({ available: false, version: null }), "vision")).toBe(false);
    expect(canAddTensorfoldExtra(null, "grammar")).toBe(false);
  });
});

describe("tensorfoldModelList() / isTensorfoldRepo()", () => {
  it("lists the registry's checkpoints", () => {
    expect(tensorfoldModelList(makeInfo())).toEqual([DENSE, FLASH, GEMMA]);
    expect(tensorfoldModelList(undefined)).toEqual([]);
  });

  it("tests membership exactly", () => {
    const models = tensorfoldModelList(makeInfo());
    expect(isTensorfoldRepo(models, [GEMMA.toLowerCase()])).toBe(true);
    expect(isTensorfoldRepo(models, ["mlx-community/gemma-4-26b-a4b-it-5bit"])).toBe(false);
    expect(isTensorfoldRepo([], [DENSE])).toBe(false);
    expect(isTensorfoldRepo(undefined, [DENSE])).toBe(false);
  });
});

describe("sanitizeSpeculativeSelection() with TensorFold", () => {
  const noDflash = { available: false, mlxAvailable: false, vllmAvailable: false, supportedModels: [] as string[] };

  it("keeps the speculative flag for a checkpoint TensorFold serves, even without a DFlash draft", () => {
    const result = sanitizeSpeculativeSelection({
      dflashInfo: noDflash,
      tensorfoldInfo: makeInfo(),
      selectedBackend: "mlx",
      canonicalRepo: DENSE,
      modelRef: "Qwen3.8-27B-MLX-4bit",
      modelName: "Qwen3.8 27B",
      speculativeDecoding: true,
      treeBudget: 64,
    });
    expect(result.speculativeDecoding).toBe(true);
    // DDTree is a DFlash feature and never applies to TensorFold
    expect(result.treeBudget).toBe(0);
  });

  it("still clears the flag for models neither engine can speed up", () => {
    const result = sanitizeSpeculativeSelection({
      dflashInfo: noDflash,
      tensorfoldInfo: makeInfo(),
      selectedBackend: "mlx",
      canonicalRepo: "mlx-community/Llama-3.2-3B-Instruct-4bit",
      modelRef: "Llama-3.2-3B-Instruct-4bit",
      modelName: "Llama 3.2 3B",
      speculativeDecoding: true,
      treeBudget: 0,
    });
    expect(result.speculativeDecoding).toBe(false);
  });

  it("is unchanged when the snapshot has no TensorFold info (older backend)", () => {
    const result = sanitizeSpeculativeSelection({
      dflashInfo: noDflash,
      selectedBackend: "mlx",
      canonicalRepo: DENSE,
      modelRef: DENSE,
      modelName: "x",
      speculativeDecoding: true,
      treeBudget: 0,
    });
    expect(result.speculativeDecoding).toBe(false);
  });
});

describe("cache strategies on the tensorfold backend", () => {
  it("only native applies: TensorFold keeps its own KV caches", () => {
    expect(isStrategyCompatible("native", "tensorfold")).toBe(true);
    expect(isStrategyCompatible("turboquant", "tensorfold")).toBe(false);
    expect(strategyIncompatReason("turboquant", "tensorfold")).toContain("tensorfold");
  });
});
