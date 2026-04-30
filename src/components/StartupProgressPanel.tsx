import type { TauriBackendInfo } from "../types";

interface Props {
  elapsedSeconds: number;
  backendOnline: boolean;
  tauriBackend: TauriBackendInfo | null;
}

// First launch of a packaged ChaosEngineAI build does three heavy things
// before the UI has anything to render:
//
//   1. Tauri extracts the bundled ~280 MB runtime tarball into a
//      manifest-hash-suffixed cache dir. Cold SSD + gunzip = 5-15 s.
//   2. The Rust shell spawns the Python sidecar. Python 3.11 imports
//      the core FastAPI app. Heavier image/video/cache runtimes stay lazy
//      until their routes are used.
//   3. The FastAPI server finishes binding its port and answers
//      /api/workspace, which releases the splash.
//
// Subsequent launches (runtime already extracted, Python modules in
// the OS page cache) are 2-5 s end to end. Showing a static
// "Loading workspace state..." for 30 s on first launch made the app
// feel hung. This panel turns the wait into a phased narrative driven
// off elapsed wall time + the tauri-side backend info.
export function StartupProgressPanel(props: Props) {
  const { elapsedSeconds, backendOnline, tauriBackend } = props;
  const startupError = tauriBackend?.startupError;

  if (startupError) {
    return (
      <div className="loading-state loading-state-error">
        <div className="loading-state-title">Backend failed to start</div>
        <div className="loading-state-detail">{startupError}</div>
      </div>
    );
  }

  const phase = pickPhase(elapsedSeconds, backendOnline, tauriBackend);
  return (
    <div className="loading-state loading-state-progress">
      <div className="loading-state-spinner" aria-hidden="true" />
      <div className="loading-state-title">{phase.title}</div>
      <div className="loading-state-detail">{phase.detail}</div>
      <div className="loading-state-elapsed">
        {elapsedSeconds}s elapsed
        {elapsedSeconds > 45 ? " — first launches can take up to a minute" : ""}
      </div>
    </div>
  );
}

function pickPhase(
  elapsedSeconds: number,
  backendOnline: boolean,
  tauriBackend: TauriBackendInfo | null,
): { title: string; detail: string } {
  // Backend is up; we're just waiting on the workspace payload. Usually
  // a fraction of a second — only visible when getWorkspace does heavy
  // work on a fresh install (catalog scan, disk probe).
  if (backendOnline) {
    return {
      title: "Loading workspace state",
      detail: "Backend is up — fetching your models, sessions, and settings.",
    };
  }

  // The Rust shell exposes `started=true` once the sidecar process has
  // been spawned. Before that we're in extraction / boot territory.
  const sidecarSpawned = tauriBackend?.started === true;

  if (!sidecarSpawned && elapsedSeconds < 4) {
    return {
      title: "Starting backend sidecar",
      detail: "Launching the ChaosEngineAI runtime.",
    };
  }
  if (!sidecarSpawned && elapsedSeconds < 15) {
    return {
      title: "Extracting embedded runtime",
      detail:
        "First launch only — unpacking the bundled Python runtime and llama.cpp " +
        "into the app cache.",
    };
  }
  if (elapsedSeconds < 25) {
    return {
      title: "Starting Python runtime",
      detail: "Loading the core API and restoring workspace state.",
    };
  }
  if (elapsedSeconds < 45) {
    return {
      title: "Waiting for backend",
      detail:
        "The sidecar is still binding its API port and checking local runtime state.",
    };
  }
  return {
    title: "Still loading",
    detail:
      "Cold-start imports are taking longer than usual. If this stalls for " +
      "more than two minutes, quit and reopen — a stale manifest can force a " +
      "re-extract.",
  };
}
