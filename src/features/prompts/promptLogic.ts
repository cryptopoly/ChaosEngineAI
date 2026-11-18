/** A saved prompt template's fields that search reads. */
export interface SearchableTemplate {
  name: string;
  category: string;
  tags: string[];
  systemPrompt: string;
}

/**
 * The backend stores `createdAt` / `updatedAt` as epoch seconds (a float); the page used to print that
 * number as it came. Seconds, milliseconds and ISO strings are accepted; anything else gives "".
 */
export function formatTemplateTime(value: unknown, locale?: string): string {
  let ms: number;
  if (typeof value === "number") {
    ms = value < 1e11 ? value * 1000 : value;
  } else if (typeof value === "string" && value.trim()) {
    const numeric = Number(value);
    ms = Number.isFinite(numeric) ? (numeric < 1e11 ? numeric * 1000 : numeric) : Date.parse(value);
  } else {
    return "";
  }
  if (!Number.isFinite(ms) || ms <= 0) return "";
  return new Date(ms).toLocaleString(locale, { dateStyle: "medium", timeStyle: "short" });
}

/** Search over name, category, tags and the prompt text itself (the route's `query` reads the text too). */
export function matchesTemplateSearch(template: SearchableTemplate, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return (
    (template.name ?? "").toLowerCase().includes(q) ||
    (template.category ?? "").toLowerCase().includes(q) ||
    (template.tags ?? []).some((tag) => tag.toLowerCase().includes(q)) ||
    (template.systemPrompt ?? "").toLowerCase().includes(q)
  );
}

/**
 * What the editor's two JSON boxes hold. An empty box is fine (no variables, no samplers); text that is not
 * the expected JSON shape is reported instead of being dropped on save, which is what used to happen.
 */
export function jsonBoxProblem(text: string, shape: "array" | "object"): boolean {
  if (!text.trim()) return false;
  try {
    const parsed: unknown = JSON.parse(text);
    if (shape === "array") return !Array.isArray(parsed);
    return !parsed || typeof parsed !== "object" || Array.isArray(parsed);
  } catch {
    return true;
  }
}
