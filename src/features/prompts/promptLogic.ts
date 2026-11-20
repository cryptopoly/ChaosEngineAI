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

/** A template's declared variable: the fill form shows a checkbox, a number box or a text box for it. */
export interface PromptVariable {
  name: string;
  type: "string" | "number" | "boolean";
  default?: string | number | boolean | null;
  description?: string;
}

/** A saved template as `GET /api/prompts` returns it. */
export interface PromptTemplate extends SearchableTemplate {
  id: string;
  fewShotExamples?: Array<{ role: string; content: string }>;
  variables?: PromptVariable[];
  presetSamplers?: Record<string, unknown> | null;
  presetModelRef?: string | null;
  /** Epoch seconds from the backend. */
  createdAt: number | string;
  updatedAt: number | string;
}

export type VariableValues = Record<string, string | number | boolean>;

const PLACEHOLDER_PATTERN = /\{\{\s*([A-Za-z0-9_-]+)\s*\}\}/g;

/**
 * Replace `{{name}}` placeholders with the filled values, as the backend's `apply_variables` does. A name with no
 * value stays as the literal placeholder so the gap shows.
 */
export function applyVariables(text: string, values: Record<string, string | number | boolean | null | undefined>): string {
  if (!text) return text;
  return text.replace(PLACEHOLDER_PATTERN, (placeholder, name: string) => {
    if (!(name in values)) return placeholder;
    const value = values[name];
    if (value == null) return "";
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  });
}

/** The fill form's starting values: each variable's default, or an empty value of its type. */
export function seedVariableValues(variables: PromptVariable[] | undefined): VariableValues {
  const seed: VariableValues = {};
  for (const variable of variables ?? []) {
    const fallback = variable.type === "boolean" ? false : variable.type === "number" ? 0 : "";
    seed[variable.name] = (variable.default ?? fallback) as string | number | boolean;
  }
  return seed;
}

/** The composer text after inserting a prompt: the prompt alone, or after what is already typed, on a new paragraph. */
export function insertIntoDraft(draft: string, text: string): string {
  if (!draft.trim()) return text;
  return `${draft.trimEnd()}\n\n${text}`;
}

/** A name for a template saved from the composer: the first line of the message, cut at 48 characters. */
export function nameFromDraft(draft: string): string {
  const firstLine = draft.trim().split(/\r?\n/, 1)[0] ?? "";
  return firstLine.length > 48 ? `${firstLine.slice(0, 47).trimEnd()}…` : firstLine;
}

/** The editor's variables box, read back: a JSON array of objects with a name; anything else gives no variables. */
export function parseVariablesBox(text: string): PromptVariable[] {
  if (!text.trim()) return [];
  try {
    const parsed: unknown = JSON.parse(text);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (v): v is PromptVariable => Boolean(v) && typeof v === "object" && typeof (v as PromptVariable).name === "string",
    );
  } catch {
    return [];
  }
}
