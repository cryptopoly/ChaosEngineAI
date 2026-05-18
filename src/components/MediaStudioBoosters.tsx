import { useCallback, useState } from "react";

import { AcceleratorCard } from "./AcceleratorCard";
import {
  type AcceleratorId,
  getAccelerator,
  getApplicableAccelerators,
} from "./acceleratorCatalog";
import { installPipPackage } from "../api";
import type { NativeBackendStatus } from "../types/server";

/**
 * Shared "Performance boosters" section (FU-056 Phase 3 + Phase 4).
 *
 * Sits inside ``ImageStudioRuntimeBanner`` and
 * ``VideoStudioRuntimeBanner`` between the torch-upgrade pill and the
 * model-load summary. Renders the accelerator cards that apply to the
 * currently-selected variant — typically nunchaku + sageattention on
 * FLUX / SD3.5 / Qwen-Image, sageattention on video DiTs, plus
 * triattention specifically on Wan 2.1 1.3B for the LongLive bonus.
 *
 * The component takes a minimal ``{repo, name}`` slice of the variant
 * rather than a concrete ``ImageModelVariant`` / ``VideoModelVariant``
 * type — both shapes carry those two fields and the booster logic
 * doesn't need anything else. Keeps one source of truth for the
 * install / overlay / re-probe dance.
 *
 * Self-contained install state: clicking "Install" calls
 * ``installPipPackage`` directly, captures the result, and overlays
 * the install response's ``capabilities`` payload onto the parent-
 * provided ``nativeBackends`` so the card flips to "Installed v…"
 * without waiting for the next workspace refetch.
 *
 * Renders nothing in two cases:
 *   - The selected variant has no applicable accelerators (SD1.5 /
 *     SDXL / non-DiT). The whole section folds away rather than
 *     rendering an empty "Performance boosters" header.
 *   - ``selectedVariant === null`` — same reason.
 *
 * Callers should additionally gate the render on
 * ``runtimeStatus.realGenerationAvailable`` so accelerators don't
 * surface on a box that can't even run the pipeline yet.
 */

/** Minimal structural shape of a Studio variant. Both
 * ``ImageModelVariant`` and ``VideoModelVariant`` carry these fields,
 * so the component accepts either. */
export interface MediaStudioBoostersVariant {
  repo: string;
  name?: string;
}

export interface MediaStudioBoostersProps {
  /** The variant currently chosen in the Studio drop-down. Determines
   * which accelerators are applicable via ``getApplicableAccelerators``. */
  selectedVariant: MediaStudioBoostersVariant | null;
  /** Parent-provided capability snapshot — usually
   * ``workspace.runtime.nativeBackends``. ``undefined`` (older
   * backends) collapses every card to its "available" form. */
  nativeBackends?: NativeBackendStatus;
}

interface InstallState {
  installing: boolean;
  error: string | null;
  output: string | null;
}

const EMPTY_INSTALL_STATE: InstallState = {
  installing: false,
  error: null,
  output: null,
};

export function MediaStudioBoosters({
  selectedVariant,
  nativeBackends,
}: MediaStudioBoostersProps) {
  // Hold a local capabilities overlay so a fresh install flips the
  // card state immediately. The parent's ``nativeBackends`` is the
  // authoritative source; we just merge install responses on top.
  const [localCaps, setLocalCaps] = useState<NativeBackendStatus | null>(null);
  const [installStates, setInstallStates] = useState<Record<string, InstallState>>({});

  const handleInstall = useCallback(async (pipPackage: string) => {
    const existing = installStates[pipPackage];
    if (existing?.installing) return;

    setInstallStates((prev) => ({
      ...prev,
      [pipPackage]: { installing: true, error: null, output: null },
    }));

    try {
      const result = await installPipPackage(pipPackage);
      if (result.ok) {
        // ``install-package`` re-probes capabilities server-side and
        // returns the fresh snapshot; we slot it onto the local
        // overlay so the card flips without a parent refetch.
        setLocalCaps((result.capabilities as unknown as NativeBackendStatus) ?? null);
        setInstallStates((prev) => ({
          ...prev,
          [pipPackage]: {
            installing: false,
            error: null,
            output: result.output ?? null,
          },
        }));
      } else {
        setInstallStates((prev) => ({
          ...prev,
          [pipPackage]: {
            installing: false,
            error: "Install command exited non-zero.",
            output: result.output ?? null,
          },
        }));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setInstallStates((prev) => ({
        ...prev,
        [pipPackage]: { installing: false, error: message, output: null },
      }));
    }
  }, [installStates]);

  const repo = selectedVariant?.repo;
  const applicable: AcceleratorId[] = getApplicableAccelerators(repo);

  if (applicable.length === 0) return null;

  // Merge parent caps with the local install overlay. The overlay
  // wins per-field so a freshly-installed accelerator's flag flips
  // green even before the parent re-fetches the workspace.
  const mergedCaps: NativeBackendStatus | null = localCaps
    ? { ...(nativeBackends ?? {}), ...localCaps } as NativeBackendStatus
    : (nativeBackends ?? null);

  return (
    <section className="media-studio-boosters">
      <header className="media-studio-boosters-header">
        <strong style={{ fontSize: "0.92rem" }}>Performance boosters</strong>
        <span className="muted-text" style={{ fontSize: "0.78rem", marginLeft: 8 }}>
          for {selectedVariant?.name ?? "the selected model"}
        </span>
      </header>
      <div className="media-studio-boosters-stack" style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 8 }}>
        {applicable.map((acceleratorId) => {
          const meta = getAccelerator(acceleratorId);
          if (!meta) return null;
          const state = installStates[meta.pipPackage] ?? EMPTY_INSTALL_STATE;
          return (
            <AcceleratorCard
              key={acceleratorId}
              meta={meta}
              capabilities={mergedCaps}
              variant="card"
              installing={state.installing}
              installError={state.error}
              installOutput={state.output}
              onInstall={handleInstall}
            />
          );
        })}
      </div>
    </section>
  );
}
