/** The Image Studio's "More actions" items: which exist for the selected model, and when they are off. */
import type { ActionMenuItem } from "../../components/ActionMenu";
import type { ImageModelVariant } from "../../types";

type TFn = (key: string, options: Record<string, unknown>) => string;

export interface ImageMenuState {
  selected: ImageModelVariant | null;
  loadedVariant: ImageModelVariant | null;
  selectedLoaded: boolean;
  /** The runtime can really load and run a model. Without it the load and unload items are not offered. */
  realGenerationAvailable: boolean;
  /** A load, unload or run is under way, the backend is busy, or it is offline. */
  menuDisabled: boolean;
}

export interface ImageMenuActions {
  openLink: (url: string) => void;
  preload: (variant: ImageModelVariant) => void;
  unload: (variant?: ImageModelVariant) => void;
}

export function imageMenuItems(t: TFn, state: ImageMenuState, actions: ImageMenuActions): ActionMenuItem[] {
  const { selected, loadedVariant } = state;
  if (!selected) return [];
  const items: ActionMenuItem[] = [
    { id: "model-card", label: t("studioKit.modelCard", { defaultValue: "Model card" }), onSelect: () => actions.openLink(selected.link) },
  ];
  if (!state.realGenerationAvailable) return items;
  if (selected.availableLocally && !state.selectedLoaded) {
    items.push({
      id: "load",
      label: t("runtimeBanner.preloadModel", { defaultValue: "Preload Model" }),
      onSelect: () => actions.preload(selected),
      disabled: state.menuDisabled,
    });
  }
  if (state.selectedLoaded) {
    items.push({
      id: "unload",
      label: t("runtimeBanner.unloadModel", { defaultValue: "Unload Model" }),
      onSelect: () => actions.unload(selected),
      disabled: state.menuDisabled,
    });
  }
  if (!state.selectedLoaded && loadedVariant) {
    items.push({
      id: "unload-named",
      label: t("runtimeBanner.unloadNamed", { name: loadedVariant.name, defaultValue: "Unload {name}" }),
      onSelect: () => actions.unload(),
      disabled: state.menuDisabled,
    });
  }
  return items;
}
