/**
 * Turn a FastAPI validation error (a list of {loc, msg, type, ctx, input}) into one plain sentence per
 * field, e.g. "Steps must be at most 100 (you entered 248)". Anything that is not that shape returns null.
 */

const FIELD_NAMES: Record<string, string> = {
  steps: "Steps",
  guidance: "Guidance",
  width: "Width",
  height: "Height",
  numFrames: "Frames",
  fps: "Frames per second",
  seed: "Seed",
  batchSize: "Images",
  prompt: "Prompt",
  negativePrompt: "Negative prompt",
  imageStrength: "Image strength",
  referenceImagePaths: "Reference images",
  startImagePath: "Start image",
  endImagePath: "End image",
};

function fieldName(loc: unknown): string {
  const parts = Array.isArray(loc) ? loc.filter((part) => typeof part === "string") as string[] : [];
  const last = parts[parts.length - 1] ?? "value";
  if (FIELD_NAMES[last]) return FIELD_NAMES[last];
  const spaced = last.replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function describe(item: Record<string, unknown>): string {
  const field = fieldName(item.loc);
  const ctx = (item.ctx ?? {}) as Record<string, unknown>;
  const entered = item.input !== undefined && typeof item.input !== "object" ? ` (you entered ${String(item.input)})` : "";
  switch (item.type) {
    case "less_than_equal":
      return `${field} must be at most ${ctx.le}${entered}.`;
    case "less_than":
      return `${field} must be less than ${ctx.lt}${entered}.`;
    case "greater_than_equal":
      return `${field} must be at least ${ctx.ge}${entered}.`;
    case "greater_than":
      return `${field} must be more than ${ctx.gt}${entered}.`;
    case "string_too_long":
      return `${field} is too long (at most ${ctx.max_length} characters).`;
    case "string_too_short":
      return `${field} is too short.`;
    case "missing":
      return `${field} is required.`;
    case "too_long":
      return `${field} has too many items (at most ${ctx.max_length}).`;
    default: {
      const msg = typeof item.msg === "string" ? item.msg : "is not valid";
      return `${field}: ${msg}.`;
    }
  }
}

export function friendlyValidationDetail(detail: unknown): string | null {
  if (!Array.isArray(detail) || detail.length === 0) return null;
  const items = detail.filter((item): item is Record<string, unknown> => !!item && typeof item === "object" && "loc" in item);
  if (items.length === 0) return null;
  return items.slice(0, 3).map(describe).join(" ");
}
