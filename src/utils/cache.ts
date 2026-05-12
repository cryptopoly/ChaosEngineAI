import i18next from "i18next";

export function parseContextK(ctx: string | undefined | null): number {
  if (!ctx) return 0;
  const upper = ctx.toUpperCase();
  if (upper.endsWith("M")) return parseFloat(ctx) * 1000;
  if (upper.endsWith("K")) return parseFloat(ctx);
  return parseFloat(ctx) / 1024;
}

export function estimateArchFromParams(paramsB: number): { numLayers: number; hiddenSize: number; numHeads: number; numKvHeads: number } {
  // Modern 7B+ models standardize on Grouped Query Attention with 8 KV heads.
  // Sub-4B models often still use Multi-Head Attention (kv = full heads).
  if (paramsB <= 1.5) return { numLayers: 22, hiddenSize: 2048, numHeads: 32, numKvHeads: 32 };
  if (paramsB <= 4) return { numLayers: 26, hiddenSize: 3072, numHeads: 24, numKvHeads: 24 };
  if (paramsB <= 9) return { numLayers: 32, hiddenSize: 4096, numHeads: 32, numKvHeads: 8 };
  if (paramsB <= 16) return { numLayers: 40, hiddenSize: 5120, numHeads: 40, numKvHeads: 8 };
  if (paramsB <= 35) return { numLayers: 60, hiddenSize: 6656, numHeads: 52, numKvHeads: 8 };
  if (paramsB <= 50) return { numLayers: 64, hiddenSize: 7168, numHeads: 56, numKvHeads: 8 };
  return { numLayers: 80, hiddenSize: 8192, numHeads: 64, numKvHeads: 8 };
}

export function estimateParamsBFromDisk(diskGb: number, bitsPerWeight: number): number {
  if (!diskGb || !bitsPerWeight) return 0;
  return (diskGb * 8) / bitsPerWeight;
}

export function detectBitsPerWeight(haystack: string): number {
  const text = haystack.toLowerCase();
  const match = text.match(/(\d)[\s-]?bit|q(\d)/);
  if (match) {
    const bits = Number(match[1] ?? match[2]);
    if (bits >= 2 && bits <= 8) return bits + 0.5;
  }
  if (/bf16|fp16|float16|f16/.test(text)) return 16;
  if (/fp32|float32|f32/.test(text)) return 32;
  return 16;
}

export function compareOptionalNumber(left: number | null | undefined, right: number | null | undefined, dir: 1 | -1) {
  const leftKnown = typeof left === "number" && Number.isFinite(left);
  const rightKnown = typeof right === "number" && Number.isFinite(right);
  if (leftKnown && rightKnown) return dir * ((left as number) - (right as number));
  if (leftKnown && !rightKnown) return -1;
  if (!leftKnown && rightKnown) return 1;
  return 0;
}

export interface CacheFitStatus {
  label: string;
  className: string;
  advice: string | null;
}

/** ``gpuVramGb`` is the binding constraint on chat KV-cache fit when an
 * NVIDIA discrete card is present. llama.cpp puts the KV cache on the GPU
 * with ``-ngl 999`` (the default for offload-capable models), so on a
 * 24 GB 4090 a 60 GB f16 cache fails far before system RAM starts to
 * matter -- it OOMs the GPU first, and CPU spillover via
 * ``--no-kv-offload`` only buys headroom up to system RAM. The pre-VRAM
 * version of this check looked only at ``totalGb`` (system RAM, 64 GB
 * on the user's machine) and reported "may exceed RAM" while completely
 * missing the much tighter VRAM ceiling. Pass null on Apple Silicon
 * (unified memory) and on machines without a discrete GPU. */
export function getCacheFitStatus(
  optimizedCacheGb: number,
  diskSizeGb: number,
  totalGb: number,
  bits: number,
  gpuVramGb?: number | null,
): CacheFitStatus {
  // Use total system memory because loading a new chat model unloads the old
  // one. Keep a reserve for the OS and other desktop apps.
  const usable = totalGb > 0 ? totalGb * 0.80 : 0;
  if (usable <= 0) {
    return {
      label: i18next.t("runtime:cacheFit.mayNotFit", { defaultValue: "May not fit" }),
      className: "warning",
      advice: null,
    };
  }

  if (diskSizeGb > usable) {
    return {
      label: i18next.t("runtime:cacheFit.modelMayNotFit", { defaultValue: "Model may not fit" }),
      className: "warning",
      advice: i18next.t("runtime:cacheAdvice.modelExceedsRam", {
        defaultValue:
          "Model weights alone exceed estimated usable RAM. Pick a smaller model or a more aggressive quantisation.",
      }),
    };
  }

  // VRAM check fires BEFORE the system-RAM check when a discrete GPU is
  // present. llama.cpp's default for GGUF on CUDA is full GPU offload
  // including the KV cache; spillover to CPU is opt-in (--no-kv-offload),
  // and even then it's bottlenecked by PCIe transfers per token. So if
  // the cache won't fit in VRAM we tell the user the right thing to fix
  // (compressed cache or lower context) rather than waiting for system
  // RAM to also fill up.
  const vramUsable = gpuVramGb && gpuVramGb > 0 ? gpuVramGb * 0.85 : 0;
  if (vramUsable > 0 && optimizedCacheGb > vramUsable) {
    const cacheGbStr = optimizedCacheGb >= 10 ? optimizedCacheGb.toFixed(0) : optimizedCacheGb.toFixed(1);
    const vramGbStr = gpuVramGb && gpuVramGb >= 10 ? gpuVramGb.toFixed(0) : (gpuVramGb ?? 0).toFixed(1);
    const cacheKindHint = bits <= 0
      ? "a full native f16 KV cache"
      : "the selected KV cache";
    return {
      label: "Cache won't fit GPU",
      className: "warning",
      advice: (
        `${cacheKindHint} at this context is ~${cacheGbStr} GB, larger than the `
        + `${vramGbStr} GB of GPU VRAM available. llama.cpp will spill to system RAM `
        + "(slow PCIe transfers per token) or fail to allocate. Lower context, drop "
        + "FP16 layers, or pick a compressed strategy (RotorQuant / TurboQuant) so "
        + "the cache fits in VRAM."
      ),
    };
  }

  const totalNeeded = optimizedCacheGb + diskSizeGb;
  const ratio = totalNeeded / usable;
  if (ratio < 0.7) {
    return {
      label: i18next.t("runtime:cacheFit.fitsEasily", { defaultValue: "Fits easily" }),
      className: "success",
      advice: null,
    };
  }
  if (ratio < 0.95) {
    return {
      label: i18next.t("runtime:cacheFit.tightFit", { defaultValue: "Tight fit" }),
      className: "warning",
      advice: null,
    };
  }

  const advice = bits <= 0
    ? i18next.t("runtime:cacheAdvice.nativeF16TooBig", {
        defaultValue:
          "The model can load, but a full native f16 cache at this context may exceed system RAM as the thread fills. Lower context, or pick a compressed strategy.",
      })
    : i18next.t("runtime:cacheAdvice.compressedTooBig", {
        defaultValue:
          "The model can load, but the selected context cache may exceed system RAM as the thread fills. Lower context or reduce FP16 layers.",
      });
  return {
    label: i18next.t("runtime:cacheFit.contextMayNotFit", { defaultValue: "Full context may not fit" }),
    className: "warning",
    advice,
  };
}
