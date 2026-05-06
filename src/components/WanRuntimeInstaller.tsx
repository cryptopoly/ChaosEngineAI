/**
 * Wan MLX runtime install action — restored Setup-tab UX surface for
 * the FU-025 backend endpoints (``startWanInstall`` /
 * ``getWanInventory`` / ``getWanInstallStatus``).
 *
 * Scoped to a single repo at a time so it can render contextually
 * inside VideoStudioTab — when the user picks a Wan-AI variant, this
 * component checks if the converted MLX dir is on disk and either
 * shows a ``Ready`` chip or an ``Install`` button. The install kicks
 * off the same background-job pattern LongLive uses (preflight →
 * download-raw → convert → verify) and polls status at 1.5 Hz.
 *
 * Apple Silicon only — backend preflight rejects other platforms with
 * a clean error string that we surface inline.
 */
import { useCallback, useEffect, useState } from "react";
import {
  getWanInstallStatus,
  getWanInventory,
  startWanInstall,
  type WanInstallJobState,
  type WanInventoryItem,
} from "../api";

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

function formatSize(gb: number | null | undefined): string {
  if (gb == null) return "?";
  if (gb >= 50) return `~${gb.toFixed(0)} GB`;
  return `~${gb.toFixed(1)} GB`;
}

export interface WanRuntimeInstallerProps {
  repo: string;
}

export function WanRuntimeInstaller({ repo }: WanRuntimeInstallerProps) {
  const [item, setItem] = useState<WanInventoryItem | null>(null);
  const [job, setJob] = useState<WanInstallJobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [convertRoot, setConvertRoot] = useState<string | null>(null);

  const refreshInventory = useCallback(async () => {
    try {
      const inventory = await getWanInventory();
      const match = inventory.items.find((it) => it.repo === repo) ?? null;
      setItem(match);
      setConvertRoot(inventory.convertRoot);
      setError(null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }, [repo]);

  useEffect(() => {
    void refreshInventory();
    let timer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    async function pollStatus() {
      try {
        const status = await getWanInstallStatus();
        if (cancelled) return;
        // Only show the running job if it's targeting THIS repo —
        // another Wan repo's install would otherwise overwrite our
        // local state and confuse the panel copy.
        if (status.repo === repo || !status.repo) {
          setJob(status);
        }
        if (isJobRunning(status)) {
          timer = setTimeout(() => void pollStatus(), POLL_INTERVAL_MS);
        } else if (status.done && status.phase === "done") {
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
  }, [repo, refreshInventory]);

  const handleInstall = async () => {
    setError(null);
    setPending(true);
    try {
      const initial = await startWanInstall(repo);
      setJob(initial);
      const tick = async () => {
        try {
          const status = await getWanInstallStatus();
          setJob(status);
          if (isJobRunning(status)) {
            setTimeout(() => void tick(), POLL_INTERVAL_MS);
          } else {
            void refreshInventory();
            setPending(false);
          }
        } catch {
          setPending(false);
        }
      };
      setTimeout(() => void tick(), POLL_INTERVAL_MS);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setPending(false);
    }
  };

  if (item == null) {
    if (error) return <p className="caution-text">Wan inventory: {error}</p>;
    return null;
  }

  const isThisRunning = isJobRunning(job) && job?.repo === repo;
  const isOtherRunning = isJobRunning(job) && job?.repo !== repo && job?.repo != null;
  const showLog = isThisRunning || (job?.repo === repo && job?.done);
  const installDisabled = isThisRunning || isOtherRunning || pending || item.converted;

  return (
    <div className="wan-runtime-installer">
      <div className="wan-runtime-installer__row">
        <div className="wan-runtime-installer__meta">
          <strong>Wan MLX runtime</strong>
          <small>
            {item.converted
              ? `Converted · routes via mlx-video native`
              : `Raw download ${formatSize(item.approxRawSizeGb)} → MLX convert (5-30 min)`}
          </small>
          {item.status.note && !item.converted ? (
            <small className="muted">{item.status.note}</small>
          ) : null}
          {convertRoot && !item.converted ? (
            <small className="muted">
              Output: <code>{convertRoot}</code>
            </small>
          ) : null}
        </div>
        <div className="wan-runtime-installer__actions">
          {item.converted ? (
            <span className="badge accent">Ready</span>
          ) : (
            <button
              className="secondary-button"
              type="button"
              disabled={installDisabled}
              onClick={() => void handleInstall()}
              title={
                isOtherRunning
                  ? `Another Wan install is running (${job?.repo}). Wait or cancel it first.`
                  : "Download raw weights + convert to MLX"
              }
            >
              {isThisRunning ? "Installing..." : pending ? "Starting..." : "Install"}
            </button>
          )}
        </div>
      </div>
      {error ? <p className="caution-text">{error}</p> : null}
      {showLog && job ? (
        <div className="wan-runtime-installer__log">
          <div className="wan-runtime-installer__log-header">
            <span>{job.phase}</span>
            <span>{Math.round(job.percent)}%</span>
          </div>
          <p className="wan-runtime-installer__log-message">{job.message}</p>
          {job.error ? <p className="caution-text">{job.error}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
