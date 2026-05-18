import { useTranslation } from "react-i18next";
import {
  dflashPackageFor,
  resolveDflashSupport,
} from "../../components/runtimeSupport";
import type { SystemStats } from "../../types";

/**
 * Inline nudge bar above the prompt textarea (FU-056 Phase 5).
 *
 * Renders only when:
 *   - The currently-loaded model has a registered DFlash draft
 *     (the model ref / canonical repo matches an entry in
 *     ``dflashInfo.supportedModels``), AND
 *   - The DFlash pip package isn't installed for the active
 *     backend yet (i.e. ``dflashInfo.available === false`` for the
 *     backend-appropriate variant), AND
 *   - The user is on a backend that actually supports DFlash
 *     (MLX or vLLM — not GGUF / llama.cpp).
 *
 * Click installs the right pip package for the backend
 * (``dflash-mlx`` on MLX, ``dflash`` on vLLM/CUDA) via the parent's
 * ``onInstallPackage`` callback. After install the parent
 * refreshes capabilities, ``dflashInfo.available`` flips ``true``,
 * and the hint folds away — no manual dismissal needed.
 *
 * This is the chat-surface twin of the Image / Video Studio
 * "Performance boosters" cards: discoverable from the actual
 * generation surface, no need to drill into Launch settings.
 */

export interface ChatComposerDflashHintProps {
  /** Aggregate DFlash signal from ``SystemStats``. Optional — when
   * the backend probe hasn't reported yet, the hint stays hidden. */
  dflashInfo?: SystemStats["dflash"];
  /** Active engine string (``"mlx"`` / ``"vllm"`` / ``"gguf"`` /
   * ``"llama.cpp"``). Drives both the visibility gate (GGUF hides
   * the hint entirely — DFlash isn't supported there) and the pip
   * package picker (vLLM → ``dflash``, MLX → ``dflash-mlx``). */
  loadedModelEngine?: string | null;
  /** Currently-loaded model identifiers. Any of the three are
   * matched against ``dflashInfo.supportedModels`` via the
   * existing ``resolveDflashSupport`` helper. */
  loadedModelRef?: string | null;
  loadedModelCanonicalRepo?: string | null;
  loadedModelName?: string | null;
  /** Dispatcher — called with ``"dflash-mlx"`` or ``"dflash"`` per
   * ``dflashPackageFor(loadedModelEngine)``. Parent owns the install
   * lifecycle (same pattern as the Studio runtime banners). */
  onInstallPackage?: (pipPackage: string) => void;
  /** Which package install is currently in flight, if any.
   * Drives the "Installing..." button label + disabled state. */
  installingPackage?: string | null;
}

export function ChatComposerDflashHint({
  dflashInfo,
  loadedModelEngine,
  loadedModelRef,
  loadedModelCanonicalRepo,
  loadedModelName,
  onInstallPackage,
  installingPackage,
}: ChatComposerDflashHintProps) {
  const { t } = useTranslation("runtime");

  // Bail early on the cheap rejection paths so we don't waste a
  // resolveDflashSupport call when there's nothing to render.
  if (!dflashInfo || !onInstallPackage) return null;
  // Already installed → nothing to nudge about.
  if (dflashInfo.available) return null;

  const support = resolveDflashSupport({
    dflashInfo,
    selectedBackend: loadedModelEngine ?? null,
    modelRef: loadedModelRef ?? null,
    canonicalRepo: loadedModelCanonicalRepo ?? null,
    modelName: loadedModelName ?? null,
  });

  // ``modelSupported`` is strictly true when the loaded model has
  // a registered draft. ``null`` means "unknown / empty supported
  // list" — don't nudge in that case, the user might be on an
  // unrelated model.
  if (support.modelSupported !== true) return null;

  const pkg = dflashPackageFor(loadedModelEngine);
  const inFlight = installingPackage === pkg;

  return (
    <div
      className="composer-dflash-hint"
      role="status"
      aria-live="polite"
    >
      <span className="composer-dflash-hint-icon" aria-hidden="true">⚡</span>
      <span className="composer-dflash-hint-text">
        {t("dflash.composerHint", {
          defaultValue:
            "DFlash speculative decoding can ~2× this model with no quality loss.",
        })}
      </span>
      <button
        type="button"
        className="composer-dflash-hint-button"
        disabled={installingPackage != null}
        onClick={() => onInstallPackage(pkg)}
      >
        {inFlight
          ? t("dflash.installing", { defaultValue: "Installing..." })
          : t("dflash.installButton", { defaultValue: "Install DFlash" })}
      </button>
    </div>
  );
}
