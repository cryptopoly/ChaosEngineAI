/**
 * Image Studio + Gallery preset / navigation helpers.
 *
 * Five small, synchronous helpers pulled out of ``useImageState`` so the
 * hook can stay focused on data fetching + generation lifecycle. Each
 * takes only the setters it actually mutates so call-sites in the hook
 * stay narrow:
 *
 * * ``applyRatioPreset`` — flip the ratio dropdown + push the matching
 *   width/height into the form.
 * * ``applyQualityPreset`` — flip the quality dropdown + push the matching
 *   steps/guidance into the form.
 * * ``openStudio`` — switch to the Image Studio tab, optionally selecting
 *   a model first; clears the global error banner so a stale error from
 *   another tab doesn't follow the user across.
 * * ``openGallery`` — same shape but lands on the Image Gallery tab,
 *   optionally pre-filtering by model id.
 * * ``resetGalleryFilters`` — restore every gallery filter dropdown +
 *   the search input + the sort to its default.
 *
 * Extracted from ``src/hooks/useImageState.ts`` as part of the v0.8.0
 * Phase 2c-5 refactor. Pure setter calls — no async, no API.
 */

import { IMAGE_RATIO_PRESETS, IMAGE_QUALITY_PRESETS } from "../../constants";
import type {
  ImageQualityPreset,
  TabId,
} from "../../types";
import type {
  ImageGalleryOrientationFilter,
  ImageGalleryRuntimeFilter,
  ImageGallerySort,
} from "../../types/image";

type RatioPresetId = (typeof IMAGE_RATIO_PRESETS)[number]["id"];

interface RatioDeps {
  setImageRatioId: (id: RatioPresetId) => void;
  setImageWidth: (px: number) => void;
  setImageHeight: (px: number) => void;
}

export function applyRatioPreset(presetId: RatioPresetId, deps: RatioDeps): void {
  const preset = IMAGE_RATIO_PRESETS.find((item) => item.id === presetId);
  if (!preset) return;
  deps.setImageRatioId(presetId);
  deps.setImageWidth(preset.width);
  deps.setImageHeight(preset.height);
}

interface QualityDeps {
  setImageQualityPreset: (id: ImageQualityPreset) => void;
  setImageSteps: (steps: number) => void;
  setImageGuidance: (guidance: number) => void;
}

export function applyQualityPreset(
  presetId: ImageQualityPreset,
  deps: QualityDeps,
): void {
  const preset = IMAGE_QUALITY_PRESETS.find((item) => item.id === presetId);
  if (!preset) return;
  deps.setImageQualityPreset(presetId);
  deps.setImageSteps(preset.steps);
  deps.setImageGuidance(preset.guidance);
}

/**
 * FU-472: the steps and guidance to set when a row is picked. A distilled or few-step row (``defaultSteps`` in
 * the catalog: FLUX 3 Klein 8, schnell 4) starts at its own values; FLUX 3 Klein opened at the Balanced
 * preset's 24 steps (13 s) where 8 is recommended (7.0 s measured). A row without its own values goes back to
 * the quality preset when the previous row had set its own, and is otherwise left alone (``null``).
 */
export function stepsForPickedVariant(
  variant: { defaultSteps?: number | null; cfgOverride?: number | null } | null | undefined,
  presetId: ImageQualityPreset,
  previousHadOwnSteps: boolean,
): { steps: number; guidance: number | null } | null {
  const own = variant?.defaultSteps;
  if (typeof own === "number" && Number.isFinite(own) && own > 0) {
    const cfg = variant?.cfgOverride;
    return { steps: own, guidance: typeof cfg === "number" && Number.isFinite(cfg) ? cfg : null };
  }
  if (!previousHadOwnSteps) return null;
  const preset = IMAGE_QUALITY_PRESETS.find((item) => item.id === presetId) ?? IMAGE_QUALITY_PRESETS[1];
  return { steps: preset.steps, guidance: preset.guidance };
}

interface OpenStudioDeps {
  setSelectedImageModelId: (id: string) => void;
  setActiveTab: (tab: TabId) => void;
  setError: (msg: string | null) => void;
}

export function openStudio(modelId: string | undefined, deps: OpenStudioDeps): void {
  if (modelId) deps.setSelectedImageModelId(modelId);
  deps.setActiveTab("image-studio");
  deps.setError(null);
}

interface OpenGalleryDeps {
  setImageGalleryModelFilter: (id: string) => void;
  setActiveTab: (tab: TabId) => void;
  setError: (msg: string | null) => void;
}

export function openGallery(modelId: string | undefined, deps: OpenGalleryDeps): void {
  if (modelId) deps.setImageGalleryModelFilter(modelId);
  deps.setActiveTab("image-gallery");
  deps.setError(null);
}

interface ResetGalleryDeps {
  setImageGallerySearchInput: (value: string) => void;
  setImageGalleryModelFilter: (value: string) => void;
  setImageGalleryRuntimeFilter: (value: ImageGalleryRuntimeFilter) => void;
  setImageGalleryOrientationFilter: (value: ImageGalleryOrientationFilter) => void;
  setImageGallerySort: (value: ImageGallerySort) => void;
}

export function resetGalleryFilters(deps: ResetGalleryDeps): void {
  deps.setImageGallerySearchInput("");
  deps.setImageGalleryModelFilter("all");
  deps.setImageGalleryRuntimeFilter("all");
  deps.setImageGalleryOrientationFilter("all");
  deps.setImageGallerySort("newest");
}
