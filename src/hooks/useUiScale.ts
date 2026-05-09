import { useCallback, useState } from "react";

export type UiScale = 0.85 | 0.9 | 0.95 | 1;

const STORAGE_KEY = "chaosengine.uiScale.v1";
const DEFAULT_UI_SCALE: UiScale = 0.9;
const VALID_UI_SCALES = new Set<number>([0.85, 0.9, 0.95, 1]);

function readStoredScale(): UiScale {
  if (typeof window === "undefined") return DEFAULT_UI_SCALE;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  const value = raw ? Number(raw) : DEFAULT_UI_SCALE;
  return VALID_UI_SCALES.has(value) ? value as UiScale : DEFAULT_UI_SCALE;
}

function writeStoredScale(scale: UiScale) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, String(scale));
  } catch {
    /* ignore quota / privacy mode errors */
  }
}

export function useUiScale() {
  const [uiScale, setUiScaleState] = useState<UiScale>(readStoredScale);

  // Persist inline on change so the read-then-write-back-identical-value
  // round-trip on mount is skipped entirely.
  const setUiScale = useCallback((scale: UiScale) => {
    setUiScaleState(scale);
    writeStoredScale(scale);
  }, []);

  return { uiScale, setUiScale };
}
