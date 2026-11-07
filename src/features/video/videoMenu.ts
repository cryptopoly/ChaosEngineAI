/** The Video Studio's "More actions" items: which exist for the selected model, and when they are off. */
import type { ActionMenuItem } from "../../components/ActionMenu";
import type { VideoModelVariant } from "../../types";

type TFn = (key: string, options: Record<string, unknown>) => string;

export interface VideoMenuState {
  selected: VideoModelVariant | null;
  loadedVariant: VideoModelVariant | null;
  isDownloaded: boolean;
  selectedLoaded: boolean;
  videoBusy: boolean;
  /** The runtime can really load and run a model (it is off when the engine or its packages are missing). */
  realGenerationAvailable: boolean;
}

export interface VideoMenuActions {
  openLink: (url: string) => void;
  preload: (variant: VideoModelVariant) => void;
  unload: (variant?: VideoModelVariant) => void;
}

export function videoMenuItems(t: TFn, state: VideoMenuState, actions: VideoMenuActions): ActionMenuItem[] {
  const { selected, loadedVariant } = state;
  if (!selected) return [];
  const items: ActionMenuItem[] = [
    { id: "model-card", label: t("studioKit.modelCard", { defaultValue: "Model card" }), onSelect: () => actions.openLink(selected.link) },
  ];
  if (state.isDownloaded && !state.selectedLoaded) {
    items.push({
      id: "load",
      label: t("videoStudio.actions.loadIntoMemory", { defaultValue: "Load into memory" }),
      onSelect: () => actions.preload(selected),
      disabled: state.videoBusy || !state.realGenerationAvailable,
    });
  }
  if (state.selectedLoaded) {
    items.push({
      id: "unload",
      label: t("videoStudio.actions.unload", { defaultValue: "Unload" }),
      onSelect: () => actions.unload(selected),
      disabled: state.videoBusy,
    });
  }
  if (!state.selectedLoaded && loadedVariant) {
    items.push({
      id: "unload-named",
      label: t("videoStudio.actions.unloadNamed", { defaultValue: "Unload {name}", name: loadedVariant.name }),
      onSelect: () => actions.unload(),
      disabled: state.videoBusy,
    });
  }
  return items;
}
