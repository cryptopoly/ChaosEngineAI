/**
 * TEMPORARY (FU-129 cut-over). Remove when the Chat and Video Models pages
 * are ported to ``ModelsPage``.
 *
 * Until then ``chat-models`` and ``video-models`` still render the old
 * Discover and My Models tabs; this two-button switch in the workspace
 * header picks which one, driven by the area's availability filter so the
 * redirect presets ("available" from an old Discover link, "installed"
 * from an old My Models link) open the matching old view.
 */

import { useTranslation } from "react-i18next";
import type { ModelAvailabilityFilter } from "./modelEntry";

export type LegacyModelsView = "installed" | "catalog";

/** "available" shows the old Discover view; anything else the old My
 * Models view, as the sidebar did before (its default child was My Models). */
export function legacyModelsView(availability: ModelAvailabilityFilter): LegacyModelsView {
  return availability === "available" ? "catalog" : "installed";
}

export function LegacyModelsSwitch({
  view,
  onViewChange,
}: {
  view: LegacyModelsView;
  onViewChange: (view: LegacyModelsView) => void;
}) {
  const { t } = useTranslation("library");
  return (
    <div className="models-availability models-legacy-switch" role="group" aria-label={t("models.legacy.label", { defaultValue: "View" })}>
      <button
        type="button"
        className={`models-availability__option${view === "installed" ? " is-active" : ""}`}
        aria-pressed={view === "installed"}
        onClick={() => onViewChange("installed")}
      >
        {t("models.legacy.installed", { defaultValue: "Installed" })}
      </button>
      <button
        type="button"
        className={`models-availability__option${view === "catalog" ? " is-active" : ""}`}
        aria-pressed={view === "catalog"}
        onClick={() => onViewChange("catalog")}
      >
        {t("models.legacy.catalog", { defaultValue: "Browse catalog" })}
      </button>
    </div>
  );
}
