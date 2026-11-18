import { describe, expect, it } from "vitest";

import {
  estimateLibraryItemCompressedGb,
  estimateLibraryItemResidentGb,
  findCatalogVariantForLibraryItem,
  findLibraryItemForVariant,
  isChatLibraryItem,
  libraryVariantMatchScore,
} from "../library";
import type { LibraryItem, ModelFamily, ModelVariant } from "../../types";

function makeVariant(overrides: Partial<ModelVariant> & { id: string; name: string; repo: string }): ModelVariant {
  const { id, name, repo, ...rest } = overrides;
  return {
    id,
    familyId: "qwen",
    name,
    repo,
    link: `https://example.com/${id}`,
    paramsB: 0,
    sizeGb: 0,
    format: "GGUF",
    quantization: "Q4_K_M",
    capabilities: [],
    note: "",
    contextWindow: "128K",
    estimatedMemoryGb: null,
    estimatedCompressedMemoryGb: null,
    availableLocally: false,
    launchMode: "direct",
    backend: "llama.cpp",
    ...rest,
  };
}

function makeFamily(variants: ModelVariant[]): ModelFamily {
  return {
    id: "qwen-family",
    name: "Qwen",
    provider: "Qwen",
    headline: "",
    summary: "",
    description: "",
    updatedLabel: "",
    popularityLabel: "",
    likesLabel: "",
    badges: [],
    capabilities: [],
    defaultVariantId: variants[0]?.id ?? "",
    variants,
    readme: [],
  };
}

function makeItem(overrides: Partial<LibraryItem> & { name: string; path: string }): LibraryItem {
  const { name, path, ...rest } = overrides;
  return {
    name,
    path,
    format: "GGUF",
    quantization: "Q4_K_M",
    backend: "llama.cpp",
    modelType: "text",
    sizeGb: 20.6,
    lastModified: "2026-04-15T10:00:00Z",
    actions: [],
    ...rest,
  };
}

describe("findCatalogVariantForLibraryItem()", () => {
  it("prefers matching parameter scale over matching quantization alone", () => {
    const qwen9 = makeVariant({
      id: "lmstudio-community/Qwen3.5-9B-GGUF",
      name: "Qwen3.5 9B GGUF",
      repo: "lmstudio-community/Qwen3.5-9B-GGUF",
      paramsB: 9,
      sizeGb: 5.8,
      format: "GGUF",
      quantization: "Q4_K_M",
    });
    const qwen35 = makeVariant({
      id: "Qwen/Qwen3.5-35B-A3B-FP8",
      name: "Qwen3.5 35B A3B",
      repo: "Qwen/Qwen3.5-35B-A3B-FP8",
      paramsB: 35,
      sizeGb: 22.8,
      format: "Transformers",
      quantization: "FP8",
      launchMode: "convert",
      backend: "auto",
    });

    const item = makeItem({
      name: "Qwen3.5-35B-A3B-GGUF",
      path: "/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf",
    });

    const matched = findCatalogVariantForLibraryItem([makeFamily([qwen9, qwen35])], item);
    expect(matched?.id).toBe(qwen35.id);
  });

  it("still matches the 9B GGUF variant for a real 9B GGUF file", () => {
    const qwen9 = makeVariant({
      id: "lmstudio-community/Qwen3.5-9B-GGUF",
      name: "Qwen3.5 9B GGUF",
      repo: "lmstudio-community/Qwen3.5-9B-GGUF",
      paramsB: 9,
      sizeGb: 5.8,
      format: "GGUF",
      quantization: "Q4_K_M",
    });
    const qwen35 = makeVariant({
      id: "Qwen/Qwen3.5-35B-A3B-FP8",
      name: "Qwen3.5 35B A3B",
      repo: "Qwen/Qwen3.5-35B-A3B-FP8",
      paramsB: 35,
      sizeGb: 22.8,
      format: "Transformers",
      quantization: "FP8",
      launchMode: "convert",
      backend: "auto",
    });

    const item = makeItem({
      name: "Qwen3.5-9B-GGUF",
      path: "/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
      sizeGb: 5.8,
    });

    const matched = findCatalogVariantForLibraryItem([makeFamily([qwen9, qwen35])], item);
    expect(matched?.id).toBe(qwen9.id);
  });

  it("does not match a row that shares only format and quantization", () => {
    // A 0.6B GGUF used to borrow a 39.8B coding model's size and badges in the Chat model picker.
    const flint = makeVariant({
      id: "bartowski/RecursiveSI_flint-GGUF-q4km",
      name: "Flint GGUF Q4_K_M",
      repo: "bartowski/RecursiveSI_flint-GGUF",
      paramsB: 39.8,
      format: "GGUF",
      quantization: "Q4_K_M",
    });
    const item = makeItem({
      name: "lmstudio-community/Qwen3-0.6B-GGUF",
      path: "/hf/hub/models--lmstudio-community--Qwen3-0.6B-GGUF",
      sizeGb: 0.45,
    });

    expect(findCatalogVariantForLibraryItem([makeFamily([flint])], item)).toBeNull();
  });

  it("does not match the same family at a different parameter scale", () => {
    const qwen27 = makeVariant({
      id: "unsloth/Qwen3.8-27B-GGUF",
      name: "Qwen3.8 27B GGUF",
      repo: "unsloth/Qwen3.8-27B-GGUF",
      paramsB: 27,
      format: "GGUF",
      quantization: "Q4_K_M",
    });
    const item = makeItem({
      name: "lmstudio-community/Qwen3-0.6B-GGUF",
      path: "/hf/hub/models--lmstudio-community--Qwen3-0.6B-GGUF",
      sizeGb: 0.45,
    });

    expect(findCatalogVariantForLibraryItem([makeFamily([qwen27])], item)).toBeNull();
  });
});

describe("a Hugging Face cache folder is matched by its own repo name", () => {
  const cache = "/Users/me/.cache/huggingface/hub";
  const flashNext = makeVariant({
    id: "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    name: "Qwen3.8 Flash Next MLX 4-bit MTP",
    repo: "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    format: "MLX",
    quantization: "4-bit",
    backend: "tensorfold",
  });
  const qwen27 = makeVariant({
    id: "lmstudio-community/Qwen3.8-27B-GGUF",
    name: "Qwen3.8 27B GGUF",
    repo: "lmstudio-community/Qwen3.8-27B-GGUF",
    paramsB: 27,
  });
  const base27 = makeVariant({
    id: "Qwen/Qwen3.6-27B",
    name: "Qwen3.6 27B",
    repo: "Qwen/Qwen3.6-27B",
    paramsB: 27,
    format: "Transformers",
    quantization: "BF16",
    launchMode: "convert",
    backend: "mlx",
  });
  const families = [makeFamily([flashNext, qwen27, base27])];
  const mlxItem = (repo: string) => makeItem({
    name: repo,
    path: `${cache}/models--${repo.replace("/", "--")}`,
    format: "MLX",
    quantization: "4-bit",
    backend: "mlx",
  });

  it("does not hand a small Qwen the repo of another Qwen row that only shares words, format and bits", () => {
    for (const repo of ["mlx-community/Qwen3-0.6B-4bit", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "mlx-community/Qwen3-8B-4bit"]) {
      const item = mlxItem(repo);
      expect(libraryVariantMatchScore(item, flashNext), repo).toBe(0);
      expect(findCatalogVariantForLibraryItem(families, item), repo).toBeNull();
      expect(findLibraryItemForVariant([item], flashNext), repo).toBeNull();
    }
  });

  it("still matches the same repo", () => {
    const item = mlxItem("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP");
    expect(findCatalogVariantForLibraryItem(families, item)?.id).toBe(flashNext.id);
  });

  it("still matches the same model under another uploader, and a conversion of a base repo", () => {
    const gguf = makeItem({ name: "unsloth/Qwen3.8-27B-GGUF", path: `${cache}/models--unsloth--Qwen3.8-27B-GGUF` });
    expect(findCatalogVariantForLibraryItem(families, gguf)?.id).toBe(qwen27.id);
    const converted = mlxItem("mlx-community/Qwen3.6-27B-4bit");
    expect(findCatalogVariantForLibraryItem(families, converted)?.id).toBe(base27.id);
    expect(findLibraryItemForVariant([converted], base27)?.path).toBe(converted.path);
  });

  it("keeps the looser match for a file outside the cache, which names no repo", () => {
    const loose = makeItem({ name: "Qwen3.8-27B-Q4_K_M", path: "/models/qwen/Qwen3.8-27B-Q4_K_M.gguf" });
    expect(findCatalogVariantForLibraryItem(families, loose)?.id).toBe(qwen27.id);
  });
});

describe("isChatLibraryItem()", () => {
  it("keeps text models", () => {
    expect(isChatLibraryItem(makeItem({ name: "Qwen3-8B", path: "/models/Qwen3-8B" }))).toBe(true);
  });

  it("excludes explicit non-chat model types", () => {
    expect(isChatLibraryItem(makeItem({ name: "Flux", path: "/models/Flux", modelType: "image" }))).toBe(false);
    expect(isChatLibraryItem(makeItem({ name: "LTX-Video", path: "/models/LTX-Video", modelType: "video" }))).toBe(false);
    expect(isChatLibraryItem(makeItem({ name: "Qwen-Draft", path: "/models/Qwen-Draft", modelType: "draft" }))).toBe(false);
    expect(isChatLibraryItem(makeItem({ name: "dinov3-encoder", path: "/models/dinov3.safetensors", modelType: "other" }))).toBe(false);
  });

  it("keeps speech, transcription and image-only models out of the chat list", () => {
    for (const repo of ["mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", "prince-canuma/Kokoro-82M", "mlx-community/whisper-small-mlx", "speridlabs/iris-3b", "ResembleAI/chatterbox", "canopylabs/orpheus-3b-0.1-ft", "openbmb/VoxCPM2", "FunAudioLLM/S3TokenizerV2"]) {
      const dir = repo.replace("/", "--");
      expect(isChatLibraryItem(makeItem({ name: repo, path: `/hf/models--${dir}/snapshots/1`, modelType: null })), repo).toBe(false);
    }
    expect(isChatLibraryItem(makeItem({ name: "Vontra/Qwen3.8-27B-MLX-4bit", path: "/hf/models--Vontra--Qwen3.8-27B-MLX-4bit/snapshots/1" }))).toBe(true);
  });

  it("filters legacy cached LTX-2 entries without modelType", () => {
    expect(isChatLibraryItem(makeItem({
      name: "prince-canuma/LTX-2.3-dev",
      path: "/hf/models--prince-canuma--LTX-2.3-dev/snapshots/1234",
      modelType: null,
    }))).toBe(false);
  });

  it("filters stale LTX-2 entries previously misclassified as text", () => {
    expect(isChatLibraryItem(makeItem({
      name: "prince-canuma/LTX-2-distilled",
      path: "/hf/models--prince-canuma--LTX-2-distilled/snapshots/1234",
      modelType: "text",
    }))).toBe(false);
  });

  it("filters stale Sana image entries previously misclassified as text", () => {
    expect(isChatLibraryItem(makeItem({
      name: "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
      path: "/hf/models--Efficient-Large-Model--Sana_Sprint_1.6B_1024px_diffusers/snapshots/1234",
      modelType: "text",
    }))).toBe(false);
  });

  it("excludes tracked-latest image families (ERNIE-Image, Nucleus, Z-Image, HiDream, GLM-Image)", () => {
    for (const repo of [
      "baidu/ERNIE-Image",
      "baidu/ERNIE-Image-Turbo",
      "NucleusAI/Nucleus-Image",
      "Tongyi-MAI/Z-Image-Turbo",
      "HiDream-ai/HiDream-I1-Full",
      "zai-org/GLM-Image",
    ]) {
      expect(isChatLibraryItem(makeItem({
        name: repo,
        path: `/hf/models--${repo.replace("/", "--")}/snapshots/1234`,
        modelType: "text", // simulate stale entry missing the image classification
      }))).toBe(false);
    }
  });
});

describe("estimateLibraryItemResidentGb()", () => {
  it("scales with actual on-disk size, not a stale catalog guess", () => {
    const tiny = estimateLibraryItemResidentGb(makeItem({ name: "a", path: "/a", sizeGb: 0.9 }));
    const small = estimateLibraryItemResidentGb(makeItem({ name: "b", path: "/b", sizeGb: 1.4 }));
    const medium = estimateLibraryItemResidentGb(makeItem({ name: "c", path: "/c", sizeGb: 15.3 }));
    const big = estimateLibraryItemResidentGb(makeItem({ name: "d", path: "/d", sizeGb: 67 }));

    expect(tiny).not.toBeNull();
    expect(small).not.toBeNull();
    expect(medium).not.toBeNull();
    expect(big).not.toBeNull();

    // The broken behaviour before this fix: three differently-sized Qwen models
    // all rendered as ~76.6 GB because they fell back to the catalog flagship.
    // Now every size maps to a distinct estimate that tracks the disk footprint.
    expect(tiny! < small!).toBe(true);
    expect(small! < medium!).toBe(true);
    expect(medium! < big!).toBe(true);
  });

  it("gives a sane ballpark for a typical 8B BF16 model", () => {
    // 15.3 GB weights on disk. Resident with small KV + framework overhead
    // should land in the mid-teens, not in the 70s.
    const ram = estimateLibraryItemResidentGb(makeItem({
      name: "Qwen3-8B",
      path: "/hf/models--Qwen--Qwen3-8B",
      sizeGb: 15.3,
    }));
    expect(ram).not.toBeNull();
    expect(ram!).toBeGreaterThan(15);
    expect(ram!).toBeLessThan(18);
  });

  it("gives a sane ballpark for a tiny 0.5B model", () => {
    // 0.9 GB on disk should produce a sub-2 GB estimate, never the old 76.6 GB.
    const ram = estimateLibraryItemResidentGb(makeItem({
      name: "Qwen2.5-0.5B-Instruct",
      path: "/hf/models--Qwen--Qwen2.5-0.5B-Instruct",
      sizeGb: 0.9,
    }));
    expect(ram).not.toBeNull();
    expect(ram!).toBeLessThan(2);
  });

  it("ignores non-positive or non-finite sizeGb and falls back to the catalog estimate", () => {
    const matched = makeVariant({
      id: "Qwen/Qwen3-8B",
      name: "Qwen3 8B",
      repo: "Qwen/Qwen3-8B",
      paramsB: 8,
      estimatedMemoryGb: 16,
    });

    expect(estimateLibraryItemResidentGb(makeItem({ name: "x", path: "/x", sizeGb: 0 }), matched)).toBe(16);
    expect(estimateLibraryItemResidentGb(makeItem({ name: "x", path: "/x", sizeGb: -5 }), matched)).toBe(16);
    expect(estimateLibraryItemResidentGb(makeItem({ name: "x", path: "/x", sizeGb: Number.NaN }), matched)).toBe(16);
  });

  it("returns null when both on-disk size and catalog fallback are missing", () => {
    const ram = estimateLibraryItemResidentGb(makeItem({ name: "x", path: "/x", sizeGb: 0 }));
    expect(ram).toBeNull();
  });
});

describe("estimateLibraryItemCompressedGb()", () => {
  it("is slightly below the uncompressed estimate at short contexts", () => {
    const item = makeItem({ name: "Qwen3-8B", path: "/hf/qwen3-8b", sizeGb: 15.3 });
    const uncompressed = estimateLibraryItemResidentGb(item)!;
    const compressed = estimateLibraryItemCompressedGb(item)!;
    expect(compressed).toBeLessThan(uncompressed);
    // At 8K context the KV term is small; the delta should be small too — that's
    // the honest signal that compression shows its value at long contexts.
    expect(uncompressed - compressed).toBeLessThan(1);
  });

  it("scales with the on-disk size, same as uncompressed", () => {
    const small = estimateLibraryItemCompressedGb(makeItem({ name: "a", path: "/a", sizeGb: 0.9 }));
    const big = estimateLibraryItemCompressedGb(makeItem({ name: "b", path: "/b", sizeGb: 67 }));
    expect(small).not.toBeNull();
    expect(big).not.toBeNull();
    expect(big! > small!).toBe(true);
  });
});
