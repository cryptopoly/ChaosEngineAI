/**
 * The Music Studio's form and run: the form with the spec's defaults, the estimate (refreshed 300 ms after
 * an input changes), the generation lifecycle (a reducer), progress polling and cancel. A finished run adds
 * its tracks to the session list; the Library list is refreshed because the backend saves every track.
 */
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { cancelMusicGeneration, estimateMusicRun, generateMusic, MusicApiError } from "../api";
import type { MusicEstimateResponse, MusicLibraryItem, MusicModelVariant, MusicRuntimeStatus } from "../types";
import {
  INITIAL_GENERATION,
  SECONDS_MAX,
  SECONDS_MIN,
  buildPayload,
  classifyMusicError,
  defaultForm,
  generationReducer,
  validateForm,
  type MusicForm,
} from "../features/music/musicLogic";
import { useGenerationProgress } from "./useGenerationProgress";

const ESTIMATE_DEBOUNCE_MS = 300;

export interface UseMusicStudioInput {
  backendOnline: boolean;
  variant: MusicModelVariant | null;
  runtime: MusicRuntimeStatus | null;
  refreshRuntime: () => Promise<void>;
  refreshLibrary: () => Promise<void>;
}

export function useMusicStudio({ backendOnline, variant, runtime, refreshRuntime, refreshLibrary }: UseMusicStudioInput) {
  const [form, setFormState] = useState<MusicForm>(() => defaultForm(variant));
  const [estimate, setEstimate] = useState<MusicEstimateResponse | null>(null);
  const [generation, dispatch] = useReducer(generationReducer, INITIAL_GENERATION);
  const running = generation.phase === "running";
  const progress = useGenerationProgress("music", running);
  const formTouched = useRef(false);
  const variantId = variant?.id ?? null;

  // The first form follows the model's own defaults until the user changes something.
  useEffect(() => {
    if (!formTouched.current && variant) setFormState(defaultForm(variant));
  }, [variant]);

  // An error from the last run goes away with the next edit (it described the old settings).
  const setForm = useCallback((patch: Partial<MusicForm>) => {
    formTouched.current = true;
    dispatch({ type: "dismiss" });
    setFormState((previous) => ({ ...previous, ...patch }));
  }, []);

  const replaceForm = useCallback((next: MusicForm) => {
    formTouched.current = true;
    dispatch({ type: "dismiss" });
    setFormState(next);
  }, []);

  // The estimate follows the inputs that change it; it also refreshes when the model loads or a run ends.
  const seconds = form.seconds;
  const variations = form.variations;
  const loadedModel = runtime?.loadedModel ?? null;
  // Another model's estimate is not this model's: drop it until the new one comes back.
  useEffect(() => {
    setEstimate(null);
  }, [variantId]);
  useEffect(() => {
    if (!backendOnline || !variantId || running || !(seconds >= SECONDS_MIN && seconds <= SECONDS_MAX)) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      estimateMusicRun(variantId, seconds, variations, controller.signal)
        .then(setEstimate)
        .catch(() => {
          // No estimate then ("unknown until the first run"), never the last one's.
          if (!controller.signal.aborted) setEstimate(null);
        });
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [backendOnline, variantId, seconds, variations, loadedModel, running]);

  const generate = useCallback(
    async (formOverride?: MusicForm) => {
      const run = formOverride ?? form;
      if (!variantId || validateForm(run).length > 0 || running) return;
      dispatch({ type: "start", at: Date.now(), form: run });
      try {
        const response = await generateMusic(buildPayload(run, variantId));
        dispatch({
          type: "succeeded",
          results: response.items.map((item: MusicLibraryItem) => ({
            item,
            jobSeconds: response.seconds,
            seed: item.seed ?? response.seed,
          })),
        });
      } catch (err) {
        const status = err instanceof MusicApiError ? err.status : null;
        dispatch({ type: "failed", error: classifyMusicError(err instanceof Error ? err.message : String(err), status) });
      }
      void refreshLibrary();
      void refreshRuntime();
    },
    [form, variantId, running, refreshLibrary, refreshRuntime],
  );

  const cancel = useCallback(async () => {
    dispatch({ type: "cancelRequested" });
    try {
      await cancelMusicGeneration();
    } catch {
      // The run itself reports its end; a failed cancel call changes nothing.
    }
  }, []);

  const dismissError = useCallback(() => dispatch({ type: "dismiss" }), []);
  const forgetResults = useCallback((ids: string[]) => dispatch({ type: "forget", ids }), []);

  return { form, setForm, replaceForm, estimate, generation, running, progress, generate, cancel, dismissError, forgetResults };
}

export type UseMusicStudio = ReturnType<typeof useMusicStudio>;
