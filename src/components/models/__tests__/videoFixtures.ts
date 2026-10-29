import type { DownloadStatus } from "../../../api";
import type { VideoModelFamily, VideoModelVariant } from "../../../types";

export const MAC = { platform: "Darwin", arch: "arm64", gpuVramTotalGb: null };
export const CUDA_PC = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24 };

export function makeVideoVariant(overrides: Partial<VideoModelVariant> = {}): VideoModelVariant {
  return {
    id: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    familyId: "wan-2-1",
    name: "Wan 2.1 T2V 1.3B",
    provider: "Wan-AI",
    repo: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    link: "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    runtime: "diffusers WanPipeline",
    styleTags: ["general", "fast"],
    taskSupport: ["txt2video"],
    sizeGb: 16.4,
    recommendedResolution: "832x480",
    defaultDurationSeconds: 5,
    note: "Small Wan 2.1 text-to-video model.",
    availableLocally: false,
    hasLocalData: false,
    estimatedGenerationSeconds: 60,
    releaseDate: "2025-02",
    ...overrides,
  };
}

export function makeVideoFamily(overrides: Partial<VideoModelFamily> = {}, variants?: VideoModelVariant[]): VideoModelFamily {
  return {
    id: "wan-2-1",
    name: "Wan 2.1",
    provider: "Wan-AI",
    headline: "Open video models",
    summary: "Wan 2.1 text-to-video models",
    updatedLabel: "Updated",
    badges: [],
    defaultVariantId: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    variants: variants ?? [makeVideoVariant()],
    ...overrides,
  };
}

/** Base Wan 2.1 14B pipeline plus three GGUF quants of it, as in the
 * catalog: every quant loads the base repo and one file of the shared
 * GGUF repo. */
export function wan14bRows(): VideoModelVariant[] {
  const base = makeVideoVariant({
    id: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    name: "Wan 2.1 14B",
    repo: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    availableLocally: true,
    hasLocalData: true,
    localDataRepos: ["Wan-AI/Wan2.1-T2V-14B-Diffusers"],
    primaryLocalRepo: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    localPath: "/hf/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/abc",
    sizeGb: 57,
    releaseDate: "2025-02",
  });
  const quant = (suffix: string, file: string, fileGb: number, extra: Partial<VideoModelVariant> = {}) =>
    makeVideoVariant({
      id: `city96/Wan2.1-T2V-14B-gguf-${suffix}`,
      name: `Wan 2.1 14B · GGUF ${suffix}`,
      repo: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
      ggufRepo: "city96/Wan2.1-T2V-14B-gguf",
      ggufFile: file,
      ggufFileGb: fileGb,
      onDiskGb: 120,
      styleTags: ["general", "gguf"],
      sizeGb: 30,
      releaseDate: "2025-02",
      ...extra,
    });
  return [
    base,
    quant("q4km", "wan2.1-t2v-14b-Q4_K_M.gguf", 10.1, {
      sharedRepoSiblings: 2,
      sharedRepoSiblingIds: ["city96/Wan2.1-T2V-14B-gguf-q6k", "city96/Wan2.1-T2V-14B-gguf-q8"],
      sharedRepoKey: "city96/Wan2.1-T2V-14B-gguf",
    }),
    quant("q6k", "wan2.1-t2v-14b-Q6_K.gguf", 12.5),
    quant("q8", "wan2.1-t2v-14b-Q8_0.gguf", 15.9),
  ];
}

export function makeLongLiveVariant(overrides: Partial<VideoModelVariant> = {}): VideoModelVariant {
  return makeVideoVariant({
    id: "NVlabs/LongLive-1.3B",
    familyId: "longlive",
    name: "LongLive 1.3B",
    provider: "NVlabs",
    repo: "NVlabs/LongLive-1.3B",
    link: "https://huggingface.co/Efficient-Large-Model/LongLive-1.3B",
    runtime: "LongLive subprocess (torchrun)",
    styleTags: ["general", "long-form"],
    sizeGb: 10,
    note: "CUDA only. Click Install LongLive to set up the isolated venv.",
    releaseDate: "2025-09",
    ...overrides,
  });
}

export function downloadState(repo: string, state: DownloadStatus["state"], extra: Partial<DownloadStatus> = {}): DownloadStatus {
  return { repo, state, progress: 0.4, downloadedGb: 4, totalGb: 10, error: null, ...extra };
}
