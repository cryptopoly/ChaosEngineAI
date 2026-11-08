/**
 * The Sounds mode's form and run: the form, the estimate (refreshed 300 ms after an input changes), the
 * run lifecycle, progress polling (the same music progress tracker) and cancel. A finished run adds its sounds
 * to the session list; the Library list is refreshed because the backend saves every sound.
 */
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { cancelMusicGeneration, estimateMusicRun, generateSound, MusicApiError } from "../api";
import type { MusicEstimateResponse, MusicLibraryItem, MusicModelVariant, MusicRuntimeStatus } from "../types";
import {
  INITIAL_SOUND_RUN,
  buildSoundPayload,
  clampSoundSeconds,
  defaultSoundForm,
  soundLimits,
  soundRunReducer,
  validateSoundForm,
  type SoundForm,
} from "../features/music/soundsLogic";
import { useGenerationProgress } from "./useGenerationProgress";

const ESTIMATE_DEBOUNCE_MS = 300;

export interface UseSoundsStudioInput {
  backendOnline: boolean;
  variant: MusicModelVariant | null;
  runtime: MusicRuntimeStatus | null;
  refreshRuntime: () => Promise<void>;
  refreshLibrary: () => Promise<void>;
}

export function useSoundsStudio({ backendOnline, variant, runtime, refreshRuntime, refreshLibrary }: UseSoundsStudioInput) {
  const [form, setFormState] = useState<SoundForm>(() => defaultSoundForm(variant));
  const [estimate, setEstimate] = useState<MusicEstimateResponse | null>(null);
  const [run, dispatch] = useReducer(soundRunReducer, INITIAL_SOUND_RUN);
  const progress = useGenerationProgress("music", run.running);
  const formTouched = useRef(false);
  const variantId = variant?.id ?? null;
  const limits = soundLimits(variant);

  // A new model (or the catalog arriving) resets an untouched form; a touched one keeps its text and settings but its
  // length is brought inside the model's own range (Small stops at 11 s, the full model at 47 s).
  useEffect(() => {
    if (!variant) return;
    if (!formTouched.current) {
      setFormState(defaultSoundForm(variant));
      return;
    }
    setFormState((previous) => {
      const seconds = clampSoundSeconds(previous.seconds, variant);
      return seconds === previous.seconds ? previous : { ...previous, seconds };
    });
  }, [variant]);

  const setForm = useCallback((patch: Partial<SoundForm>) => {
    formTouched.current = true;
    dispatch({ type: "dismiss" });
    setFormState((previous) => ({ ...previous, ...patch }));
  }, []);

  const replaceForm = useCallback((next: SoundForm) => {
    formTouched.current = true;
    dispatch({ type: "dismiss" });
    setFormState(next);
  }, []);

  const { seconds, variations } = form;
  const loadedModel = runtime?.sound?.loadedModel ?? null;
  useEffect(() => {
    setEstimate(null);
  }, [variantId]);
  useEffect(() => {
    if (!backendOnline || !variantId || run.running || !(seconds >= limits.min && seconds <= limits.max)) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      estimateMusicRun(variantId, seconds, variations, controller.signal)
        .then(setEstimate)
        .catch(() => {
          if (!controller.signal.aborted) setEstimate(null);
        });
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [backendOnline, variantId, seconds, variations, loadedModel, run.running, limits.min, limits.max]);

  const generate = useCallback(
    async (formOverride?: SoundForm) => {
      const next = formOverride ?? form;
      if (!variantId || validateSoundForm(next, variant).length > 0 || run.running) return;
      dispatch({ type: "start", at: Date.now(), form: next });
      try {
        const response = await generateSound(buildSoundPayload(next, variantId));
        dispatch({
          type: "succeeded",
          results: response.items.map((item: MusicLibraryItem) => ({ item, jobSeconds: response.seconds, seed: item.seed ?? response.seed })),
        });
      } catch (err) {
        dispatch({ type: "failed", message: err instanceof Error ? err.message : String(err), status: err instanceof MusicApiError ? err.status : null });
      }
      void refreshLibrary();
      void refreshRuntime();
    },
    [form, variant, variantId, run.running, refreshLibrary, refreshRuntime],
  );

  const cancel = useCallback(async () => {
    dispatch({ type: "cancelRequested" });
    try {
      await cancelMusicGeneration();
    } catch {
      // The run reports its own end; a failed cancel call changes nothing.
    }
  }, []);

  const dismissError = useCallback(() => dispatch({ type: "dismiss" }), []);
  const forgetResults = useCallback((ids: string[]) => dispatch({ type: "forget", ids }), []);

  return { form, setForm, replaceForm, estimate, run, running: run.running, progress, generate, cancel, dismissError, forgetResults };
}

export type UseSoundsStudio = ReturnType<typeof useSoundsStudio>;
