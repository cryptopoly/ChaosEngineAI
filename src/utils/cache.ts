export function parseContextK(ctx: string | undefined | null): number {
  if (!ctx) return 0;
  const upper = ctx.toUpperCase();
  if (upper.endsWith("M")) return parseFloat(ctx) * 1000;
  if (upper.endsWith("K")) return parseFloat(ctx);
  return parseFloat(ctx) / 1024;
}

export function estimateArchFromParams(paramsB: number): { numLayers: number; hiddenSize: number; numHeads: number } {
  if (paramsB <= 1.5) return { numLayers: 22, hiddenSize: 2048, numHeads: 32 };
  if (paramsB <= 4) return { numLayers: 26, hiddenSize: 3072, numHeads: 24 };
  if (paramsB <= 9) return { numLayers: 32, hiddenSize: 4096, numHeads: 32 };
  if (paramsB <= 16) return { numLayers: 40, hiddenSize: 5120, numHeads: 40 };
  if (paramsB <= 35) return { numLayers: 60, hiddenSize: 6656, numHeads: 52 };
  if (paramsB <= 50) return { numLayers: 64, hiddenSize: 7168, numHeads: 56 };
  return { numLayers: 80, hiddenSize: 8192, numHeads: 64 };
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
