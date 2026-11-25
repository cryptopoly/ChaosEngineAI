/**
 * The export's upscale method: "Sharper" runs each shot through the AI upscaler (the Upscale add-on), "Standard
 * stretch" is the plain resize. The setting is `export.upscaleMethod` ("fast" or "lanczos"). Pure rules, no React.
 */
import type { StoryboardExportSettings } from "../../../types";

export type UpscaleMethod = "lanczos" | "fast";
/** The order the choice shows in: the sharper one first. */
export const UPSCALE_METHODS: UpscaleMethod[] = ["fast", "lanczos"];

/** The method the project uses; a project saved before the field existed (or an old payload) stretches the standard way. */
export function upscaleMethodOf(project: { export: StoryboardExportSettings }): UpscaleMethod {
  // The TypeScript export type does not carry `upscaleMethod` yet (backend `models.py` has it): read it as optional.
  const stored = (project.export as StoryboardExportSettings & { upscaleMethod?: UpscaleMethod }).upscaleMethod;
  return stored === "fast" ? "fast" : "lanczos";
}

/** The `edit_project` arguments for the choice. */
export function upscaleMethodPatch(method: UpscaleMethod): { export: { upscaleMethod: UpscaleMethod } } {
  return { export: { upscaleMethod: method } };
}

/**
 * What to say under the choice. null: nothing (the film is not scaled up, or the standard stretch is chosen).
 * "needs-addon": the sharper method is chosen and the add-on is not installed, so the export would be refused.
 * "ready": the add-on is installed. "checking": its state is not known yet.
 */
export function upscaleMethodNote(
  upscale: string,
  method: UpscaleMethod,
  addonInstalled: boolean | null,
): "needs-addon" | "ready" | "checking" | null {
  if (upscale === "none" || method !== "fast") return null;
  if (addonInstalled === null) return "checking";
  return addonInstalled ? "ready" : "needs-addon";
}
