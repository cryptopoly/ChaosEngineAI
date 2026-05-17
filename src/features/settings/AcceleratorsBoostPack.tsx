import { useCallback, useEffect, useState } from "react";

import { AcceleratorCard } from "../../components/AcceleratorCard";
import { ACCELERATOR_CATALOG } from "../../components/acceleratorCatalog";
import { installPipPackage, refreshCapabilities } from "../../api";
import type { NativeBackendStatus } from "../../types/server";

/**
 * The Diagnostics tab's "Boost Pack" section (FU-056 Phase 6).
 *
 * Single panel listing every CUDA-side accelerator the catalog
 * registers, with current install state + one-click install. The
 * "everything in one place" surface for users who want to see the
 * full accelerator landscape; per-feature surfaces (Phases 3-5)
 * inherit the same ``AcceleratorCard`` component but show only the
 * accelerators relevant to that tab.
 *
 * State ownership
 * ---------------
 * This panel self-probes capabilities on mount (``refreshCapabilities``
 * hits ``/api/setup/refresh-capabilities``) and re-probes after each
 * successful install so the Installed pills flip without a parent
 * refetch. Per-accelerator install state lives in ``installStates``
 * keyed by ``pipPackage`` — the card itself stays stateless beyond
 * its "log expanded" toggle.
 *
 * The panel intentionally renders **every** entry in
 * ``ACCELERATOR_CATALOG`` regardless of platform (``showIncompatible``
 * is true). The user-experience choice here: this is the diagnostics
 * surface, the user wants visibility into what exists across the
 * ecosystem, not just what their current box can run. Per-feature
 * surfaces will gate by platform so wrong-platform affordances don't
 * appear next to a FLUX model card.
 */

export interface AcceleratorsBoostPackProps {
  /** Set false until the backend health check has cleared.
   * Capabilities fetch needs the backend up. */
  backendOnline: boolean;
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

export function AcceleratorsBoostPack({ backendOnline }: AcceleratorsBoostPackProps) {
  const [capabilities, setCapabilities] = useState<NativeBackendStatus | null>(null);
  const [capError, setCapError] = useState<string | null>(null);
  const [installStates, setInstallStates] = useState<Record<string, InstallState>>({});

  const probe = useCallback(async () => {
    if (!backendOnline) return;
    try {
      const next = await refreshCapabilities();
      // ``refreshCapabilities`` returns a generic ``Record`` because
      // it serves several consumers; the FU-056 Phase 1 fields are
      // optional on ``NativeBackendStatus`` so this cast is safe even
      // when the backend is older than the frontend.
      setCapabilities(next as unknown as NativeBackendStatus);
      setCapError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setCapError(message);
    }
  }, [backendOnline]);

  useEffect(() => {
    if (backendOnline) {
      void probe();
    }
  }, [backendOnline, probe]);

  const handleInstall = useCallback(
    async (pipPackage: string) => {
      // Guard against double-clicks on the same accelerator. Other
      // accelerators can still install concurrently — the backend's
      // ``/api/setup/install-package`` endpoint serialises pip writes
      // for us at the OS-FS layer.
      const existing = installStates[pipPackage];
      if (existing?.installing) return;

      setInstallStates((prev) => ({
        ...prev,
        [pipPackage]: { installing: true, error: null, output: null },
      }));

      try {
        const result = await installPipPackage(pipPackage);
        if (result.ok) {
          setInstallStates((prev) => ({
            ...prev,
            [pipPackage]: {
              installing: false,
              error: null,
              output: result.output ?? null,
            },
          }));
          await probe();
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
          [pipPackage]: {
            installing: false,
            error: message,
            output: null,
          },
        }));
      }
    },
    [installStates, probe],
  );

  // Ordered: Apple Silicon, CUDA, cross-platform — but really, all
  // surfaced together. The user can scan the platformGate column to
  // decide what their box supports. Keep the catalog order verbatim
  // so the table is stable across runs.
  const rows = ACCELERATOR_CATALOG;

  return (
    <section className="accelerators-boost-pack" style={{ marginTop: 18 }}>
      <header className="accelerators-boost-pack-header">
        <h3 style={{ margin: 0, fontSize: "0.98rem", fontWeight: 600 }}>
          Boost Pack
        </h3>
        <p className="muted-text" style={{ margin: "4px 0 0", fontSize: "0.84rem" }}>
          Optional accelerators for image, video, and chat inference. Each is a
          single pip install away — click Install on the rows your hardware
          supports.
        </p>
      </header>

      {capError ? (
        <p
          className="muted-text"
          style={{ color: "rgb(252, 165, 165)", margin: "8px 0 0", fontSize: "0.82rem" }}
        >
          Could not read accelerator capabilities: {capError}
        </p>
      ) : null}

      {!backendOnline ? (
        <p className="muted-text" style={{ margin: "8px 0 0", fontSize: "0.82rem" }}>
          Backend offline — start the sidecar to read accelerator state.
        </p>
      ) : null}

      <table className="accelerators-boost-pack-table" style={{ width: "100%", marginTop: 10, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid rgba(255, 255, 255, 0.06)", fontSize: "0.76rem", color: "var(--muted)" }}>
            <th style={{ padding: "6px 10px", textAlign: "left", fontWeight: 500 }}>Accelerator</th>
            <th style={{ padding: "6px 10px", textAlign: "left", fontWeight: 500 }}>Size</th>
            <th style={{ padding: "6px 10px", textAlign: "left", fontWeight: 500 }}>Platform</th>
            <th style={{ padding: "6px 10px", textAlign: "right", fontWeight: 500 }}>Status</th>
            <th style={{ padding: "6px 10px", textAlign: "right", fontWeight: 500 }}>Action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((meta) => {
            const state = installStates[meta.pipPackage] ?? EMPTY_INSTALL_STATE;
            return (
              <AcceleratorCard
                key={meta.id}
                meta={meta}
                capabilities={capabilities}
                variant="row"
                installing={state.installing}
                installError={state.error}
                installOutput={state.output}
                onInstall={handleInstall}
                showIncompatible
              />
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
