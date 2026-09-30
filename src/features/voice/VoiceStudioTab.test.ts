import { describe, expect, it } from "vitest";
import { pickDefaultSttModel } from "./VoiceStudioTab";
import type { SttModel } from "../../types";

function makeModel(overrides: Partial<SttModel>): SttModel {
  return {
    id: "org/model",
    name: "Model",
    sizeGb: 0.5,
    installed: false,
    default: false,
    ...overrides,
  };
}

describe("pickDefaultSttModel", () => {
  it("prefers the catalog default when it is installed", () => {
    const models = [
      makeModel({ id: "a", installed: true }),
      makeModel({ id: "b", default: true, installed: true }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("b");
  });

  it("falls back to the first installed model when the default is not installed", () => {
    const models = [
      makeModel({ id: "a", default: true, installed: false }),
      makeModel({ id: "b", installed: false }),
      makeModel({ id: "c", installed: true }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("c");
  });

  it("falls back to the catalog default when nothing is installed", () => {
    const models = [
      makeModel({ id: "a", installed: false }),
      makeModel({ id: "b", default: true, installed: false }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("b");
  });

  it("falls back to the first model when there is no default", () => {
    const models = [
      makeModel({ id: "a", installed: false }),
      makeModel({ id: "b", installed: false }),
    ];
    expect(pickDefaultSttModel(models)?.id).toBe("a");
  });

  it("returns undefined for an empty list", () => {
    expect(pickDefaultSttModel([])).toBeUndefined();
  });
});
