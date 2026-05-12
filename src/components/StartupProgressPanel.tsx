import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
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
  const { t } = useTranslation("common");
  const { elapsedSeconds, backendOnline, tauriBackend } = props;
  const startupError = tauriBackend?.startupError;

  if (startupError) {
    return (
      <div className="loading-state loading-state-error">
        <div className="loading-state-title">
          {t("startupProgress.error.title", { defaultValue: "Backend failed to start" })}
        </div>
        <div className="loading-state-detail">{startupError}</div>
      </div>
    );
  }

  const phase = pickPhase(t, elapsedSeconds, backendOnline, tauriBackend);
  return (
    <div className="loading-state loading-state-progress">
      <div className="loading-state-spinner" aria-hidden="true" />
      <div className="loading-state-title">{phase.title}</div>
      <div className="loading-state-detail">{phase.detail}</div>
      <div className="loading-state-elapsed">
        {elapsedSeconds > 45
          ? t("startupProgress.elapsedSlow", {
              defaultValue: "{seconds}s elapsed — first launches can take up to a minute",
              seconds: elapsedSeconds,
            })
          : t("startupProgress.elapsed", {
              defaultValue: "{seconds}s elapsed",
              seconds: elapsedSeconds,
            })}
      </div>
    </div>
  );
}

function pickPhase(
  t: TFunction,
  elapsedSeconds: number,
  backendOnline: boolean,
  tauriBackend: TauriBackendInfo | null,
): { title: string; detail: string } {
  // Backend is up; we're just waiting on the workspace payload. Usually
  // a fraction of a second — only visible when getWorkspace does heavy
  // work on a fresh install (catalog scan, disk probe).
  if (backendOnline) {
    return {
      title: t("startupProgress.phase.loadingWorkspace.title", { defaultValue: "Loading workspace state" }),
      detail: t("startupProgress.phase.loadingWorkspace.detail", {
        defaultValue: "Backend is up — fetching your models, sessions, and settings.",
      }),
    };
  }

  // The Rust shell exposes `started=true` once the sidecar process has
  // been spawned. Before that we're in extraction / boot territory.
  const sidecarSpawned = tauriBackend?.started === true;

  if (!sidecarSpawned && elapsedSeconds < 4) {
    return {
      title: t("startupProgress.phase.startingSidecar.title", { defaultValue: "Starting backend sidecar" }),
      detail: t("startupProgress.phase.startingSidecar.detail", {
        defaultValue: "Launching the ChaosEngineAI runtime.",
      }),
    };
  }
  if (!sidecarSpawned && elapsedSeconds < 15) {
    return {
      title: t("startupProgress.phase.extractingRuntime.title", { defaultValue: "Extracting embedded runtime" }),
      detail: t("startupProgress.phase.extractingRuntime.detail", {
        defaultValue:
          "First launch only — unpacking the bundled Python runtime and llama.cpp into the app cache.",
      }),
    };
  }
  if (elapsedSeconds < 25) {
    return {
      title: t("startupProgress.phase.startingPython.title", { defaultValue: "Starting Python runtime" }),
      detail: t("startupProgress.phase.startingPython.detail", {
        defaultValue: "Loading the core API and restoring workspace state.",
      }),
    };
  }
  if (elapsedSeconds < 45) {
    return {
      title: t("startupProgress.phase.waitingBackend.title", { defaultValue: "Waiting for backend" }),
      detail: t("startupProgress.phase.waitingBackend.detail", {
        defaultValue:
          "The sidecar is still binding its API port and checking local runtime state.",
      }),
    };
  }
  return {
    title: t("startupProgress.phase.stillLoading.title", { defaultValue: "Still loading" }),
    detail: t("startupProgress.phase.stillLoading.detail", {
      defaultValue:
        "Cold-start imports are taking longer than usual. If this stalls for more than two minutes, quit and reopen — a stale manifest can force a re-extract.",
    }),
  };
}
