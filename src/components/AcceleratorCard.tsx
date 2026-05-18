import { useState } from "react";

import type { NativeBackendStatus } from "../types/server";
import {
  type AcceleratorMeta,
  isPlatformCompatible,
} from "./acceleratorCatalog";

/**
 * Reusable card for the six CUDA-side accelerators (FU-056 Phase 2).
 *
 * Three placement variants share one component so the per-feature
 * surfaces in Phases 3–6 stay in sync without re-implementing the
 * three states (idle / installing / installed / failed) per surface:
 *
 *   - ``card`` (default) — full-width banner with title, speedup
 *     claim, applies-to copy, size pill, primary action. Lives in the
 *     Diagnostics Boost Pack and the Image / Video Studio runtime
 *     banners.
 *   - ``pill`` — compact horizontal chip with "🚀 Label +Nx [Install]"
 *     copy. Lives on catalog variant cards in the Discover / Models
 *     tabs.
 *   - ``row`` — table-friendly form with name + applies-to + status +
 *     action laid out as columns. Used by the Diagnostics Boost Pack
 *     to render all six accelerators in one scannable view.
 *
 * State ownership: the *parent* owns the install lifecycle (which
 * package is in flight, success / failure of the most recent attempt,
 * output captured from the install pipe). The card itself only owns
 * the "log expanded?" toggle. This mirrors the
 * ``CudaTorchLogPanel`` / ``TorchUpgradePill`` contract — keeps the
 * card cheap to render in many places without each instance
 * duplicating polling work.
 */

export interface AcceleratorCardProps {
  /** Catalog row for the accelerator this card represents. */
  meta: AcceleratorMeta;
  /** Live capability snapshot. Used to read ``meta.capabilityField``
   * + ``meta.versionField`` for installed-state display. */
  capabilities: NativeBackendStatus | null;
  /** Card layout density. Defaults to ``"card"``. */
  variant?: "card" | "pill" | "row";
  /** True while *this specific* accelerator's install is in flight.
   * The parent owns this state; the card just renders accordingly. */
  installing?: boolean;
  /** Last error message from a failed install attempt. ``null`` after
   * a successful install or before any attempt. */
  installError?: string | null;
  /** Captured pip output from the last install (success or fail).
   * Surfaced inside a collapsible ``<details>`` so success runs stay
   * compact but failures expose the diagnostic. */
  installOutput?: string | null;
  /** Fired when the user clicks Install / Retry. Parent should call
   * ``installPipPackage(meta.pipPackage)`` then ``refreshWorkspace()``.
   *
   * Optional: when omitted, the card renders **read-only** — status
   * pill + meta only, no action button. Used by discovery surfaces
   * (the Image Models / Discover tabs) where the install action lives
   * in a sibling surface (the Image Studio runtime banner) so the
   * install state stays in one place rather than scattered. */
  onInstall?: (pipPackage: string) => void;
  /** Optional click handler for the platform-mismatch tooltip — lets
   * the parent surface a "this won't run on your hardware" toast. */
  onPlatformMismatch?: (meta: AcceleratorMeta) => void;
  /** Force-show the card even when ``platformGate`` says it's
   * incompatible. The Diagnostics Boost Pack uses this so users can
   * see every accelerator; per-feature surfaces leave it false. */
  showIncompatible?: boolean;
}

/** Exported for unit-test reach: ``true`` iff capabilities reports
 * this accelerator's flag as ``=== true``. Older backends without
 * FU-056 fields read as ``false`` (the fields are optional on the
 * shared TS interface). Never throws. */
export function readInstalled(
  meta: AcceleratorMeta,
  capabilities: NativeBackendStatus | null,
): boolean {
  if (!capabilities) return false;
  const value = capabilities[meta.capabilityField];
  return value === true;
}

/** Exported for unit-test reach: returns the version string when the
 * backend exposed it, else ``null``. ``"0.0.0"`` and other zero-prefix
 * versions count as present — we don't filter on semver shape. */
export function readVersion(
  meta: AcceleratorMeta,
  capabilities: NativeBackendStatus | null,
): string | null {
  if (!capabilities) return null;
  const value = capabilities[meta.versionField];
  return typeof value === "string" && value.length > 0 ? value : null;
}

/** Exported for unit-test reach: human-readable platform requirement. */
export function platformLabel(gate: AcceleratorMeta["platformGate"]): string {
  switch (gate) {
    case "cuda":
      return "CUDA only";
    case "apple-silicon":
      return "Apple Silicon only";
    case "any":
      return "Cross-platform";
  }
}

/** Exported for unit-test reach: maps the (installed / installing /
 * failed / idle, sync / async) matrix onto the button copy. Returns
 * ``null`` when no action button should render (i.e. the install is
 * already complete). */
export function actionLabelFor(args: {
  installed: boolean;
  installing: boolean;
  hasError: boolean;
  installMode: AcceleratorMeta["installMode"];
}): string | null {
  if (args.installed) return null;
  if (args.installing) return "Installing…";
  if (args.hasError) return "Retry";
  return args.installMode === "async" ? "Install (background)" : "Install";
}

export function AcceleratorCard(props: AcceleratorCardProps) {
  const {
    meta,
    capabilities,
    variant = "card",
    installing = false,
    installError = null,
    installOutput = null,
    onInstall,
    onPlatformMismatch,
    showIncompatible = false,
  } = props;

  const installed = readInstalled(meta, capabilities);
  const version = readVersion(meta, capabilities);
  const compatible = capabilities ? isPlatformCompatible(meta, capabilities) : true;
  const [logOpen, setLogOpen] = useState<boolean>(Boolean(installError));

  // When the affordance is shown on a platform that physically can't
  // run the accelerator and the surface isn't a "show everything"
  // diagnostic — hide it. Cleaner than rendering a disabled card the
  // user can't act on.
  if (!compatible && !showIncompatible) {
    return null;
  }

  // Read-only mode: when no ``onInstall`` is wired we render the card
  // as a passive informational element — no Install button, no Retry,
  // no platform-mismatch toast. The discovery surfaces use this so
  // they don't accidentally become install dispatchers.
  const readOnly = onInstall === undefined;

  const handleInstall = () => {
    if (readOnly) return;
    if (!compatible) {
      onPlatformMismatch?.(meta);
      return;
    }
    onInstall(meta.pipPackage);
  };

  const statusBadge = (() => {
    if (installed) {
      return (
        <span className="accelerator-card-status accelerator-card-status-installed">
          {version ? `✓ v${version}` : "✓ Installed"}
        </span>
      );
    }
    if (installing) {
      return (
        <span className="accelerator-card-status accelerator-card-status-installing">
          Installing…
        </span>
      );
    }
    if (installError) {
      return (
        <span className="accelerator-card-status accelerator-card-status-failed">
          Install failed
        </span>
      );
    }
    return null;
  })();

  const actionLabel = actionLabelFor({
    installed,
    installing,
    hasError: Boolean(installError),
    installMode: meta.installMode,
  });

  if (variant === "pill") {
    return (
      <span
        className={
          "accelerator-card accelerator-card-pill" +
          (installed ? " accelerator-card-installed" : "") +
          (!compatible ? " accelerator-card-incompatible" : "")
        }
        data-accelerator-id={meta.id}
      >
        <span className="accelerator-card-pill-label">
          {installed ? "✓ " : "🚀 "}
          {meta.shortLabel}
        </span>
        {!installed && !readOnly && (
          <button
            type="button"
            className="accelerator-card-action accelerator-card-action-pill"
            onClick={handleInstall}
            disabled={installing}
            aria-label={`Install ${meta.label}`}
          >
            {actionLabel}
          </button>
        )}
      </span>
    );
  }

  if (variant === "row") {
    return (
      <tr
        className={
          "accelerator-card-row" +
          (installed ? " accelerator-card-installed" : "") +
          (!compatible ? " accelerator-card-incompatible" : "")
        }
        data-accelerator-id={meta.id}
      >
        <td className="accelerator-card-row-label">
          <strong>{meta.label}</strong>
          <span className="accelerator-card-row-applies">{meta.appliesTo}</span>
        </td>
        <td className="accelerator-card-row-size">{meta.sizeOnDiskLabel}</td>
        <td className="accelerator-card-row-platform">{platformLabel(meta.platformGate)}</td>
        <td className="accelerator-card-row-status">{statusBadge}</td>
        <td className="accelerator-card-row-action">
          {!readOnly && actionLabel && (
            <button
              type="button"
              className="accelerator-card-action"
              onClick={handleInstall}
              disabled={installing || !compatible}
              title={!compatible ? `Requires: ${platformLabel(meta.platformGate)}` : undefined}
            >
              {actionLabel}
            </button>
          )}
        </td>
      </tr>
    );
  }

  // Default: full card.
  return (
    <section
      className={
        "accelerator-card" +
        (installed ? " accelerator-card-installed" : "") +
        (!compatible ? " accelerator-card-incompatible" : "")
      }
      data-accelerator-id={meta.id}
    >
      <header className="accelerator-card-header">
        <h3 className="accelerator-card-title">
          {installed ? "✓ " : "🚀 "}
          {meta.label}
        </h3>
        {statusBadge}
      </header>

      <p className="accelerator-card-claim">{meta.speedupClaim}</p>
      <p className="accelerator-card-applies">
        <span className="accelerator-card-applies-label">Applies to:</span>{" "}
        {meta.appliesTo}
      </p>

      <div className="accelerator-card-meta">
        <span className="accelerator-card-meta-item">{meta.sizeOnDiskLabel}</span>
        <span className="accelerator-card-meta-item">{platformLabel(meta.platformGate)}</span>
        <span className="accelerator-card-meta-item">
          {meta.installMode === "async" ? "Background install" : "Quick install"}
        </span>
        <span className="accelerator-card-meta-item accelerator-card-meta-follow-up">
          {meta.followUp}
        </span>
      </div>

      {!readOnly && actionLabel && (
        <div className="accelerator-card-actions">
          <button
            type="button"
            className="accelerator-card-action accelerator-card-action-primary"
            onClick={handleInstall}
            disabled={installing || !compatible}
            title={!compatible ? `Requires: ${platformLabel(meta.platformGate)}` : undefined}
          >
            {actionLabel}
          </button>
        </div>
      )}

      {installError && (
        <details
          className="accelerator-card-log"
          open={logOpen}
          onToggle={(event) => setLogOpen((event.target as HTMLDetailsElement).open)}
        >
          <summary className="accelerator-card-log-summary">
            Install failure — show output
          </summary>
          <p className="accelerator-card-log-error">{installError}</p>
          {installOutput && (
            <pre className="accelerator-card-log-output">{installOutput}</pre>
          )}
        </details>
      )}

      {installed && installOutput && !installError && (
        <details className="accelerator-card-log accelerator-card-log-success">
          <summary className="accelerator-card-log-summary">
            Install output
          </summary>
          <pre className="accelerator-card-log-output">{installOutput}</pre>
        </details>
      )}
    </section>
  );
}
