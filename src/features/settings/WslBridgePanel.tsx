import { useCallback, useEffect, useRef, useState } from "react";

import { InstallLogPanel } from "../../components/InstallLogPanel";
import {
  getVllmWslInstallStatus,
  refreshCapabilities,
  startVllmWslInstall,
  type VllmWslJobState,
} from "../../api";
import type { NativeBackendStatus } from "../../types/server";

/**
 * Windows-only Setup panel that surfaces the WSL2 vLLM bridge state
 * + one-click installer (FU-056 Phase 8).
 *
 * vLLM ships no native Windows wheels, so users on RTX Windows boxes
 * can't get to the vLLM lane without dropping to PowerShell. This
 * panel makes the install one click: behind the scenes it spawns a
 * background job that creates an isolated venv inside the user's
 * default WSL distro at ``~/.chaosengine/vllm-venv``, pip-installs
 * vllm (~2 GB), and verifies the import works.
 *
 * Five state buckets:
 *
 *   1. **Not Windows** — render nothing. The caller already gates by
 *      ``platform === "win32"`` but the bail here is defensive.
 *   2. **WSL2 not installed** — surface the official ``wsl --install``
 *      command with a copy-paste hint + a link to Microsoft's docs.
 *      The user reboots, re-opens ChaosEngineAI, the panel flips to
 *      bucket 3.
 *   3. **WSL2 ready, CUDA not visible inside WSL** — the NVIDIA WSL
 *      driver kicker isn't installed on the Windows host. Surface a
 *      link to the NVIDIA WSL guide; we can't install drivers from
 *      inside the app.
 *   4. **WSL2 + CUDA ready, vLLM not installed** — the install
 *      button. Background-job pattern (start → poll status) same as
 *      LongLive / GPU bundle.
 *   5. **vLLM ready** — green pill with the version. The install
 *      button collapses to "Reinstall" so a user who hit a half-baked
 *      build can recover without dropping to PowerShell.
 *
 * Self-contained: probes capabilities on mount, polls install status
 * at 1.5 Hz when a job is in flight, refreshes capabilities on
 * completion so the parent's workspace state catches up.
 */

export interface WslBridgePanelProps {
  /** Set false until backend health check has cleared. Probe + install
   * both need the backend up. */
  backendOnline: boolean;
}

const POLL_INTERVAL_MS = 1500;
// Pulled out as a constant so the link in the "WSL2 not installed"
// bucket points at the live Microsoft doc page rather than burying
// the URL inline. Bump when MS retires this page (unlikely soon).
const WSL_INSTALL_DOCS_URL = "https://learn.microsoft.com/en-us/windows/wsl/install";
const NVIDIA_WSL_DOCS_URL = "https://docs.nvidia.com/cuda/wsl-user-guide/";

export function WslBridgePanel({ backendOnline }: WslBridgePanelProps) {
  const [caps, setCaps] = useState<NativeBackendStatus | null>(null);
  const [capsError, setCapsError] = useState<string | null>(null);
  const [job, setJob] = useState<VllmWslJobState | null>(null);
  const [starting, setStarting] = useState(false);
  const pollTimer = useRef<number | null>(null);

  // ----- One-shot capability probe on mount / backend-online toggle -----
  const probe = useCallback(async () => {
    if (!backendOnline) return;
    try {
      const next = await refreshCapabilities();
      setCaps(next as unknown as NativeBackendStatus);
      setCapsError(null);
    } catch (err) {
      setCapsError(err instanceof Error ? err.message : String(err));
    }
  }, [backendOnline]);

  useEffect(() => {
    void probe();
  }, [probe]);

  // ----- Install status poll loop -----
  // Polls at 1.5 Hz while the job is in flight; stops on done / error.
  // Re-probes capabilities on completion so the bucket switches without
  // a parent refetch.
  useEffect(() => {
    if (!job) return;
    if (job.done || job.phase === "done" || job.phase === "error") {
      if (pollTimer.current) {
        window.clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
      if (job.phase === "done") {
        void probe();
      }
      return;
    }
    if (pollTimer.current) return;
    pollTimer.current = window.setInterval(() => {
      void (async () => {
        try {
          const next = await getVllmWslInstallStatus();
          setJob(next);
        } catch {
          // Best-effort poll; swallow errors so we don't disrupt the UI.
          // The next tick will retry.
        }
      })();
    }, POLL_INTERVAL_MS);
    return () => {
      if (pollTimer.current) {
        window.clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [job, probe]);

  const handleInstall = useCallback(async () => {
    if (starting) return;
    setStarting(true);
    try {
      const next = await startVllmWslInstall();
      setJob(next);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // Synthesize an error job so the InstallLogPanel can surface the
      // failure even when the start endpoint itself bailed (e.g. user
      // managed to click on a non-Windows host through a stale UI).
      setJob({
        id: "vllm-wsl-error",
        phase: "error",
        message,
        packageCurrent: null,
        packageIndex: 0,
        packageTotal: 0,
        percent: 0,
        targetDir: null,
        error: message,
        startedAt: Date.now() / 1000,
        finishedAt: Date.now() / 1000,
        attempts: [],
        done: true,
      });
    } finally {
      setStarting(false);
    }
  }, [starting]);

  // ----- Bucket selection -----
  // The panel itself only renders on Windows; macOS / Linux callers
  // shouldn't see it. Bail defensively if a caller forgets to gate.
  const isWindows = typeof navigator !== "undefined"
    && navigator.userAgent.toLowerCase().includes("windows");
  if (!isWindows) return null;

  if (!backendOnline) {
    return (
      <section className="wsl-bridge-panel" style={{ marginTop: 18 }}>
        <header>
          <strong style={{ fontSize: "0.95rem" }}>WSL2 vLLM bridge</strong>
        </header>
        <p className="muted-text" style={{ fontSize: "0.84rem", margin: "4px 0 0" }}>
          Backend offline — start the sidecar to probe WSL state.
        </p>
      </section>
    );
  }

  if (capsError) {
    return (
      <section className="wsl-bridge-panel" style={{ marginTop: 18 }}>
        <header>
          <strong style={{ fontSize: "0.95rem" }}>WSL2 vLLM bridge</strong>
        </header>
        <p className="muted-text" style={{ color: "rgb(252, 165, 165)", fontSize: "0.82rem", margin: "4px 0 0" }}>
          Could not read WSL state: {capsError}
        </p>
      </section>
    );
  }

  if (!caps) {
    return (
      <section className="wsl-bridge-panel" style={{ marginTop: 18 }}>
        <header>
          <strong style={{ fontSize: "0.95rem" }}>WSL2 vLLM bridge</strong>
        </header>
        <p className="muted-text" style={{ fontSize: "0.84rem", margin: "4px 0 0" }}>
          Probing WSL state...
        </p>
      </section>
    );
  }

  const wsl2 = caps.wsl2Available === true;
  const wslCuda = caps.wslCudaAvailable === true;
  const vllmInstalled = caps.wslVllmAvailable === true;
  const distro = caps.wslDistroName ?? "WSL";
  const vllmVersion = caps.wslVllmVersion ?? null;

  // Common chrome — same header on every bucket so the panel reads as
  // one section the user can find by name.
  const header = (
    <header className="wsl-bridge-panel-header" style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
      <strong style={{ fontSize: "0.95rem" }}>WSL2 vLLM bridge</strong>
      {wsl2 && wslCuda && vllmInstalled ? (
        <span
          className="badge subtle"
          style={{
            background: "rgba(80, 180, 100, 0.22)",
            color: "#8fd99e",
            padding: "2px 8px",
            borderRadius: 10,
            fontSize: "0.72rem",
            fontWeight: 600,
          }}
        >
          ✓ Ready{vllmVersion ? ` · v${vllmVersion}` : ""}
        </span>
      ) : null}
    </header>
  );

  // Bucket 1: WSL2 not installed at all.
  if (!wsl2) {
    return (
      <section className="wsl-bridge-panel" style={{ marginTop: 18 }}>
        {header}
        <p className="muted-text" style={{ fontSize: "0.84rem", margin: "6px 0 0" }}>
          WSL2 isn't installed on this Windows host. vLLM ships no native
          Windows wheels, so the practical path is to run vLLM inside a
          WSL2 Ubuntu distro.
        </p>
        <p className="muted-text" style={{ fontSize: "0.82rem", margin: "8px 0 0" }}>
          Open an admin PowerShell and run:
        </p>
        <pre style={{
          margin: "4px 0",
          padding: "8px 10px",
          background: "rgba(0, 0, 0, 0.35)",
          borderRadius: 6,
          fontSize: "0.8rem",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        }}>wsl --install</pre>
        <p className="muted-text" style={{ fontSize: "0.78rem", margin: 0 }}>
          Reboot when prompted, then reopen ChaosEngineAI.{" "}
          <a href={WSL_INSTALL_DOCS_URL} target="_blank" rel="noreferrer">Microsoft docs</a>
        </p>
      </section>
    );
  }

  // Bucket 2: WSL2 up but CUDA not reachable inside it.
  if (!wslCuda) {
    return (
      <section className="wsl-bridge-panel" style={{ marginTop: 18 }}>
        {header}
        <p className="muted-text" style={{ fontSize: "0.84rem", margin: "6px 0 0" }}>
          WSL2 is installed ({distro}), but{" "}
          <code style={{ fontSize: "0.82em" }}>nvidia-smi</code> isn't
          reachable inside the distro. Install the NVIDIA WSL driver
          kicker on Windows — it lets CUDA passthrough from your GPU
          driver into WSL.
        </p>
        <p className="muted-text" style={{ fontSize: "0.78rem", margin: "8px 0 0" }}>
          <a href={NVIDIA_WSL_DOCS_URL} target="_blank" rel="noreferrer">NVIDIA WSL guide ↗</a>
        </p>
      </section>
    );
  }

  // Bucket 3+: ready to install / already installed.
  const installRunning = job != null
    && (job.phase === "preflight" || job.phase === "installing");
  const buttonLabel = installRunning
    ? job?.message || "Installing..."
    : vllmInstalled
      ? "Reinstall vLLM in WSL"
      : "Install vLLM in WSL";

  return (
    <section className="wsl-bridge-panel" style={{ marginTop: 18 }}>
      {header}
      <p className="muted-text" style={{ fontSize: "0.84rem", margin: "6px 0 0" }}>
        {vllmInstalled
          ? `vLLM ${vllmVersion ?? ""} is installed in ${distro} at ~/.chaosengine/vllm-venv. The desktop app can route vLLM model loads through this venv.`
          : `WSL2 (${distro}) + CUDA passthrough are ready. Install vLLM into an isolated venv at ~/.chaosengine/vllm-venv (~2 GB download, ~5-15 min on a warm box).`}
      </p>
      <div className="button-row" style={{ marginTop: 10 }}>
        <button
          type="button"
          className="secondary-button"
          onClick={() => void handleInstall()}
          disabled={starting || installRunning}
        >
          {buttonLabel}
        </button>
      </div>
      <InstallLogPanel job={job} variant="vllm-wsl" />
    </section>
  );
}
