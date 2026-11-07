/**
 * All Music state for the three screens, built once in `App` next to the other area hooks: the catalog,
 * runtime and install (`useMusicRuntime`), the Library (`useMusicLibrary`) and the Studio form and run
 * (`useMusicStudio`), plus the two moves between screens ("Open in Studio", "Make another like this").
 */
import { useCallback, useMemo, useState } from "react";
import type { TabRequest } from "../constants";
import type { MusicLibraryItem, SidebarGroupId } from "../types";
import { formFromItem } from "../features/music/musicLogic";
import { soundFormFromItem } from "../features/music/soundsLogic";
import { useMusicLibrary } from "./useMusicLibrary";
import { useMusicRuntime } from "./useMusicRuntime";
import { useMusicStudio } from "./useMusicStudio";
import { useSoundsStudio } from "./useSoundsStudio";

/** The Music Studio's two modes: songs and instrumentals, or short sound effects and loops. */
export type StudioMode = "music" | "sounds";

export function useMusicState(
  backendOnline: boolean,
  setError: (message: string | null) => void,
  setActiveTab: (request: TabRequest) => void,
) {
  const runtimeState = useMusicRuntime(backendOnline, setError);
  const library = useMusicLibrary(backendOnline, setError);
  const { selectedVariant, refreshRuntime, setSelectedModelId } = runtimeState;
  const studio = useMusicStudio({
    backendOnline,
    variant: selectedVariant,
    runtime: runtimeState.runtime,
    refreshRuntime,
    refreshLibrary: library.refreshLibrary,
  });
  const { replaceForm } = studio;
  const [studioMode, setStudioMode] = useState<StudioMode>("music");
  const sounds = useSoundsStudio({
    backendOnline,
    variant: runtimeState.soundVariant,
    runtime: runtimeState.runtime,
    refreshRuntime,
    refreshLibrary: library.refreshLibrary,
  });
  const { replaceForm: replaceSoundForm } = sounds;

  /** Music Models > "Open in Studio": select the model (a sound row opens the Sounds mode) and go. */
  const openStudio = useCallback(
    (modelId?: string) => {
      if (modelId && runtimeState.soundVariants.some((variant) => variant.id === modelId)) setStudioMode("sounds");
      else {
        if (modelId) setSelectedModelId(modelId);
        setStudioMode("music");
      }
      setActiveTab("music-studio");
    },
    [setSelectedModelId, setActiveTab, runtimeState.soundVariants],
  );

  /** Music Library > "Make another like this": the item's settings in the Studio form. */
  const makeAnotherLike = useCallback(
    (item: MusicLibraryItem) => {
      if (item.kind === "sound") {
        replaceSoundForm(soundFormFromItem(item));
        setStudioMode("sounds");
        setActiveTab("music-studio");
        return;
      }
      setStudioMode("music");
      const variant = runtimeState.variants.find((entry) => entry.id === item.model) ?? selectedVariant;
      if (variant && variant.id !== selectedVariant?.id) setSelectedModelId(variant.id);
      replaceForm(formFromItem(item, variant));
      setActiveTab("music-studio");
    },
    [runtimeState.variants, selectedVariant, setSelectedModelId, replaceForm, replaceSoundForm, setActiveTab],
  );

  /** Groups the sidebar opens on their ready child: Music once a music model is installed. */
  const readyGroups = useMemo<ReadonlySet<SidebarGroupId>>(
    () => new Set<SidebarGroupId>(runtimeState.hasInstalledModel ? ["music"] : []),
    [runtimeState.hasInstalledModel],
  );

  return { ...runtimeState, library, studio, sounds, studioMode, setStudioMode, openStudio, makeAnotherLike, readyGroups };
}

export type UseMusicState = ReturnType<typeof useMusicState>;
