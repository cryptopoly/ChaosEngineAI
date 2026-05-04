/**
 * WanInstallPanel — FU-025 Phase 9 UI.
 *
 * Lists every Wan-AI raw repo the mlx-video convert pipeline supports.
 * Per row:
 *   - "Converted" badge if the MLX artifacts are already on disk.
 *   - "Install" button otherwise → POSTs to /api/setup/install-mlx-video-wan
 *     and starts polling /api/setup/install-mlx-video-wan/status.
 *   - InstallLogPanel underneath shows live progress while a job runs.
 *
 * Apple Silicon only — backend preflight rejects other platforms with
 * a clean error string surfaced into the panel.
 */

import { useCallback, useEffect, useState } from "react";

import {
  getWanInstallStatus,
  getWanInventory,
  startWanInstall,
  type WanInstallJobState,
  type WanInventory,
  type WanInventoryItem,
} from "../api";
import { InstallLogPanel } from "./InstallLogPanel";

const POLL_INTERVAL_MS = 1500;
const _RUNNING_PHASES: ReadonlyArray<WanInstallJobState["phase"]> = [
  "preflight",
  "downloading",
  "converting",
  "verifying",
];

function isJobRunning(job: WanInstallJobState | null): boolean {
  if (!job) return false;
  return _RUNNING_PHASES.includes(job.phase);
}

function formatSize(gb: number | null): string {
  if (gb == null) return "?";
  if (gb >= 50) return `~${gb.toFixed(0)} GB`;
  return `~${gb.toFixed(1)} GB`;
}

export function WanInstallPanel() {
  const [inventory, setInventory] = useState<WanInventory | null>(null);
  const [job, setJob] = useState<WanInstallJobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pendingRepo, setPendingRepo] = useState<string | null>(null);

  const refreshInventory = useCallback(async () => {
    try {
      const data = await getWanInventory();
      setInventory(data);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, []);

  // Initial load + status poll
  useEffect(() => {
    void refreshInventory();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    async function pollStatus() {
      try {
        const status = await getWanInstallStatus();
        if (cancelled) return;
        setJob(status);
        if (isJobRunning(status)) {
          timer = setTimeout(() => void pollStatus(), POLL_INTERVAL_MS);
        } else if (status.done && status.phase === "done") {
          // Job finished successfully — inventory may have flipped to
          // converted. Refresh once.
          void refreshInventory();
        }
      } catch {
        // Soft-fail status poll — backend may have restarted; the next
        // user action triggers another cycle.
      }
    }
    void pollStatus();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [refreshInventory]);

  const handleInstall = async (repo: string) => {
    setError(null);
    setPendingRepo(repo);
    try {
      const initial = await startWanInstall(repo);
      setJob(initial);
      // Spin up a status poll for this run.
      const tick = async () => {
        try {
          const status = await getWanInstallStatus();
          setJob(status);
          if (isJobRunning(status)) {
            setTimeout(() => void tick(), POLL_INTERVAL_MS);
          } else {
            void refreshInventory();
            setPendingRepo(null);
          }
        } catch {
          setPendingRepo(null);
        }
      };
      setTimeout(() => void tick(), POLL_INTERVAL_MS);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setPendingRepo(null);
    }
  };

  const renderRow = (item: WanInventoryItem) => {
    const isThisRepoRunning = isJobRunning(job) && job?.repo === item.repo;
    const isDifferentRepoRunning = isJobRunning(job) && job?.repo !== item.repo;
    const showLog = isThisRepoRunning || (job?.repo === item.repo && job?.done);

    return (
      <div className="wan-install-row" key={item.repo}>
        <div className="wan-install-row-meta">
          <strong>{item.repo}</strong>
          <small>raw download {formatSize(item.approxRawSizeGb)}</small>
          {item.converted ? (
            <span className="badge accent">Converted</span>
          ) : item.status.note ? (
            <small className="muted">{item.status.note}</small>
          ) : null}
        </div>
        <div className="wan-install-row-actions">
          {item.converted ? (
            <span className="badge muted">Ready · routes to mlx-video</span>
          ) : (
            <button
              className="secondary-button"
              type="button"
              disabled={isThisRepoRunning || isDifferentRepoRunning || pendingRepo === item.repo}
              onClick={() => void handleInstall(item.repo)}
              title={
                isDifferentRepoRunning
                  ? `Another Wan install is running (${job?.repo}). Wait or cancel it first.`
                  : "Download raw weights + convert to MLX (5-30 min depending on model size)."
              }
            >
              {isThisRepoRunning ? "Installing..." : "Install"}
            </button>
          )}
        </div>
        {showLog && job ? (
          <InstallLogPanel
            job={{
              id: job.id,
              phase: job.phase === "converting" ? "downloading" : job.phase,
              message: job.message,
              packageCurrent: job.packageCurrent,
              packageIndex: job.packageIndex,
              packageTotal: job.packageTotal,
              percent: job.percent,
              targetDir: job.outputDir,
              error: job.error,
              startedAt: job.startedAt,
              finishedAt: job.finishedAt,
              attempts: job.attempts,
              done: job.done,
            }}
            variant="longlive"
          />
        ) : null}
      </div>
    );
  };

  if (!inventory) {
    return (
      <section className="wan-install-panel">
        <h3>Wan MLX runtime</h3>
        <p className="muted">Loading Wan inventory…</p>
        {error ? <p className="error">{error}</p> : null}
      </section>
    );
  }

  return (
    <section className="wan-install-panel">
      <header>
        <h3>Wan MLX runtime (Apple Silicon)</h3>
        <p className="muted">
          Convert raw Wan-AI checkpoints to MLX format so video generation
          runs natively via mlx-video instead of diffusers MPS.
          Converted output: <code>{inventory.convertRoot}</code>.
          Raw downloads cache to <code>{inventory.rawRoot}</code>.
        </p>
      </header>

      {error ? <p className="error">{error}</p> : null}

      <div className="wan-install-rows">
        {inventory.items.map(renderRow)}
      </div>
    </section>
  );
}
