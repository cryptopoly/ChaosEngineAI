/**
 * Image gallery / artifact-driven action helpers.
 *
 * Four helpers pulled out of ``useImageState`` so the hook can stay
 * focused on selection + generation lifecycle. Each takes the setters /
 * callbacks it actually mutates as a deps object:
 *
 * * ``hydrateFormFromArtifact`` — push every field from a saved artifact
 *   (model, prompt, dimensions, steps/guidance, seed, ratio + quality
 *   preset matches) back into the Studio form, then optionally jump to
 *   the Image Studio tab. ``randomizeSeed`` flips the form to "random
 *   each run" and clears the seed input — the entry-point for "Vary
 *   seed".
 * * ``deleteArtifact`` — DELETE the saved output, prune the in-flight
 *   modal's artifact list, and shift the selected artifact id forward.
 *   When the modal becomes empty + idle the modal closes.
 * * ``varyImageSeed`` — full "Vary seed" flow: hydrates the form (with
 *   randomize=true), then submits a fresh generation with the artifact's
 *   knobs and a random seed. Picks the matching quality preset off the
 *   artifact's steps + guidance to keep the dropdown honest.
 * * ``useSameImageSettings`` — load every form field from an artifact
 *   without submitting. Optionally closes the generation modal so the
 *   user lands back in the Studio with a pre-filled form.
 *
 * Extracted from ``src/hooks/useImageState.ts`` as part of the v0.8.0
 * Phase 2c-5 refactor.
 */

import type { Dispatch, SetStateAction } from "react";
import { deleteImageOutput } from "../../api";
import { IMAGE_QUALITY_PRESETS, IMAGE_RATIO_PRESETS } from "../../constants";
import type {
  ImageOutputArtifact,
  ImageQualityPreset,
} from "../../types";

type RatioPresetId = (typeof IMAGE_RATIO_PRESETS)[number]["id"];

interface HydrateDeps {
  setSelectedImageModelId: (id: string) => void;
  setImagePrompt: (text: string) => void;
  setImageNegativePrompt: (text: string) => void;
  setImageWidth: (px: number) => void;
  setImageHeight: (px: number) => void;
  setImageSteps: (steps: number) => void;
  setImageGuidance: (guidance: number) => void;
  setImageBatchSize: (size: number) => void;
  setImageRatioId: (id: RatioPresetId) => void;
  setImageQualityPreset: (id: ImageQualityPreset) => void;
  setImageUseRandomSeed: (random: boolean) => void;
  setImageSeedInput: (text: string) => void;
  openImageStudio: (modelId?: string) => void;
}

export function hydrateFormFromArtifact(
  artifact: ImageOutputArtifact,
  randomizeSeed: boolean,
  deps: HydrateDeps,
): void {
  deps.setSelectedImageModelId(artifact.modelId);
  deps.setImagePrompt(artifact.prompt);
  deps.setImageNegativePrompt(artifact.negativePrompt ?? "");
  deps.setImageWidth(artifact.width);
  deps.setImageHeight(artifact.height);
  deps.setImageSteps(artifact.steps);
  deps.setImageGuidance(artifact.guidance);
  deps.setImageBatchSize(1);
  const ratioPreset = IMAGE_RATIO_PRESETS.find(
    (preset) => preset.width === artifact.width && preset.height === artifact.height,
  );
  if (ratioPreset) deps.setImageRatioId(ratioPreset.id);
  const qualityPreset = IMAGE_QUALITY_PRESETS.find(
    (preset) => preset.steps === artifact.steps && preset.guidance === artifact.guidance,
  );
  if (qualityPreset) deps.setImageQualityPreset(qualityPreset.id);
  deps.setImageUseRandomSeed(randomizeSeed);
  deps.setImageSeedInput(randomizeSeed ? "" : String(artifact.seed));
  deps.openImageStudio(artifact.modelId);
}

interface DeleteArtifactDeps {
  imageGenerationArtifacts: ImageOutputArtifact[];
  showImageGenerationModal: boolean;
  imageBusy: boolean;
  setImageOutputs: (outputs: ImageOutputArtifact[]) => void;
  setImageGenerationArtifacts: (artifacts: ImageOutputArtifact[]) => void;
  setSelectedImageGenerationArtifactId: Dispatch<SetStateAction<string | null>>;
  setShowImageGenerationModal: (open: boolean) => void;
  setError: (msg: string | null) => void;
}

export async function deleteArtifact(
  artifactId: string,
  deps: DeleteArtifactDeps,
): Promise<void> {
  try {
    const response = await deleteImageOutput(artifactId);
    deps.setImageOutputs(response.outputs);
    const nextArtifacts = deps.imageGenerationArtifacts.filter(
      (artifact) => artifact.artifactId !== artifactId,
    );
    deps.setImageGenerationArtifacts(nextArtifacts);
    deps.setSelectedImageGenerationArtifactId((current) => {
      if (current && nextArtifacts.some((artifact) => artifact.artifactId === current)) return current;
      return nextArtifacts[0]?.artifactId ?? null;
    });
    if (deps.showImageGenerationModal && nextArtifacts.length === 0 && !deps.imageBusy) {
      deps.setShowImageGenerationModal(false);
    }
  } catch (err) {
    deps.setError(err instanceof Error ? err.message : "Could not delete image output.");
  }
}

interface SubmitOverrides {
  modelId?: string;
  prompt?: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance?: number;
  batchSize?: number;
  qualityPreset?: ImageQualityPreset;
  seed?: number | null;
}

interface VarySeedDeps {
  imageQualityPreset: ImageQualityPreset;
  hydrateFormFromArtifact: (artifact: ImageOutputArtifact, randomizeSeed: boolean) => void;
  submitImageGeneration: (overrides?: SubmitOverrides) => Promise<void>;
}

export async function varyImageSeed(
  artifact: ImageOutputArtifact,
  deps: VarySeedDeps,
): Promise<void> {
  const matchedQualityPreset =
    IMAGE_QUALITY_PRESETS.find(
      (preset) => preset.steps === artifact.steps && preset.guidance === artifact.guidance,
    )?.id ?? deps.imageQualityPreset;
  deps.hydrateFormFromArtifact(artifact, true);
  await deps.submitImageGeneration({
    modelId: artifact.modelId,
    prompt: artifact.prompt,
    negativePrompt: artifact.negativePrompt ?? "",
    width: artifact.width,
    height: artifact.height,
    steps: artifact.steps,
    guidance: artifact.guidance,
    batchSize: 1,
    qualityPreset: matchedQualityPreset,
    seed: Math.floor(Math.random() * 2147483647),
  });
}

interface UseSameSettingsDeps {
  hydrateFormFromArtifact: (artifact: ImageOutputArtifact) => void;
  setShowImageGenerationModal: (open: boolean) => void;
}

export function useSameImageSettings(
  artifact: ImageOutputArtifact,
  closeModal: boolean,
  deps: UseSameSettingsDeps,
): void {
  deps.hydrateFormFromArtifact(artifact);
  if (closeModal) deps.setShowImageGenerationModal(false);
}
