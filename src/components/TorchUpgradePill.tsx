import { useCallback, useEffect, useRef, useState } from "react";

import {
  checkTorchUpgradeAvailable,
  getTorchUpgradeStatus,
  startTorchUpgrade,
  type TorchUpgradeAvailability,
  type TorchUpgradeJobState,
} from "../api";

/**
 * Compact pill surfacing the "newer torch is available" path.
 *
 * Self-contained: on mount it calls ``/api/setup/torch-upgrade-available``
 * once. If the response is ``available: false`` (Apple Silicon, no
 * extras, CPU wheel, already at latest, etc.) the component renders
 * nothing — the user's stable setup stays uncluttered. Only when an
 * actual upgrade exists does the pill appear.
 *
 * Three display states:
 *
 *   1. **Available** — single-line summary plus a primary ``Upgrade``
 *      button and a collapsible explainer that lists what'll be
 *      reinstalled (only present on minor / major bumps where the
 *      ABI-dependent deps need ``--force-reinstall``).
 *
 *   2. **In progress** — once the user confirms, kicks off the
 *      background job and polls ``/api/setup/upgrade-torch/status`` at
 *      1.5 Hz. Renders the latest message + the current phase.
 *      Each new attempt scrolls into a small log block below.
 *
 *   3. **Done / error** — surfaces the final outcome with a Restart
 *      Backend prompt on success, or the error message + a "rolled
 *      back" note on failure so the user knows their previous torch
 *      is still there.
 *
 * Designed for inline placement in the Image / Video Studio runtime
 * banners — sits below the chip row, above the model summary. Not
 * exposed elsewhere because the banners are where users go when they
 * want to confirm their GPU stack is healthy, so it's the right place
 * to nudge them about a newer wheel.
 */
export interface TorchUpgradePillProps {
  /** Whether the backend health check has cleared. The detection probe
   * spawns a pip subprocess, which would fail spuriously if the backend
   * isn't up yet — keep the pill silent until the backend is online. */
  backendOnline: boolean;
  /** Fired when the upgrade reports success + ``requiresRestart``. The
   * caller plumbs this through the existing Restart Backend button so
   * the same notification stack handles the prompt. */
  onRestartBackend: () => void;
  /** Disables the Restart Backend button while another action is busy. */
  busy: boolean;
}

const POLL_INTERVAL_MS = 1500;

export function TorchUpgradePill({ backendOnline, onRestartBackend, busy }: TorchUpgradePillProps) {
  const [availability, setAvailability] = useState<TorchUpgradeAvailability | null>(null);
  const [job, setJob] = useState<TorchUpgradeJobState | null>(null);
  const [starting, setStarting] = useState(false);
  // ``hidden`` is a one-session dismissal — if the user clicks Dismiss the
  // pill stays gone until the next backend restart. We deliberately don't
  // persist this to localStorage: on the next session we want the user
  // to see the upgrade option again (the latest version on the index
  // may have changed in the meantime).
  const [hidden, setHidden] = useState(false);

  // ----- One-shot availability probe -----
  // Runs once when the backend comes online. We don't re-probe on every
  // window focus because the answer is stable for a backend session
  // (the underlying torch wheel can't change without a restart).
  useEffect(() => {
    if (!backendOnline) return;
    let cancelled = false;
    (async () => {
      try {
        const result = await checkTorchUpgradeAvailable();
        if (!cancelled) setAvailability(result);
      } catch {
        // Silent fail — the detection probe is best-effort, and we'd
        // rather not show a "torch upgrade check failed" error in the
        // banner. The pill just stays hidden.
        if (!cancelled) setAvailability({ available: false, reason: "index-query-failed" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [backendOnline]);

  // ----- Poll the job status while a run is in flight -----
  // Single interval — the dependency on ``job?.done`` collapses the
  // effect once the job reaches a terminal state, so we don't keep
  // polling after success. No need for a separate "is polling" flag.
  const pollTimerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!job || job.done) {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }
    pollTimerRef.current = window.setInterval(async () => {
      try {
        const next = await getTorchUpgradeStatus();
        setJob(next);
      } catch {
        // Network blip — keep polling; the next tick will likely succeed.
      }
    }, POLL_INTERVAL_MS);
    return () => {
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [job?.id, job?.done]);

  const handleUpgrade = useCallback(async () => {
    if (starting) return;
    setStarting(true);
    try {
      const initial = await startTorchUpgrade({ rebuildDependents: true });
      setJob(initial);
    } catch (err) {
      // Synthesize a job-shaped error so the user sees what went wrong
      // without us having to render a parallel error UI.
      const message = err instanceof Error ? err.message : String(err);
      setJob({
        id: "torch-upgrade-error",
        phase: "error",
        message,
        currentVersion: availability?.available ? availability.current : null,
        targetVersion: availability?.available ? availability.latest : null,
        upgradeType: availability?.available ? availability.upgradeType : null,
        indexUrl: availability?.available ? availability.indexUrl : null,
        rebuildDependents: true,
        rebuiltPackages: [],
        rolledBack: false,
        rollbackPath: null,
        cudaVerified: null,
        requiresRestart: false,
        error: message,
        startedAt: Date.now() / 1000,
        finishedAt: Date.now() / 1000,
        attempts: [{ ok: false, output: message }],
        done: true,
      });
    } finally {
      setStarting(false);
    }
  }, [availability, starting]);

  if (!backendOnline || hidden) return null;
  // Loading + no-upgrade-available are both "render nothing" — keeps the
  // chip row uncluttered. The job state takes priority once a run
  // begins, so a Dismiss after starting won't actually hide it.
  if (job === null && (availability === null || !availability.available)) return null;

  // ----- Job-in-flight + terminal-state rendering -----
  if (job !== null) {
    return (
      <div className="torch-upgrade-pill torch-upgrade-pill-active">
        <TorchUpgradeJobStatus
          job={job}
          onRestartBackend={onRestartBackend}
          onClose={() => {
            setJob(null);
            // After a successful upgrade, hide the pill for this
            // session — re-running the detection probe would show
            // ``already-latest`` anyway.
            if (job.done && !job.error) setHidden(true);
          }}
          restartBusy={busy}
        />
      </div>
    );
  }

  // ----- Available, not yet started -----
  if (availability === null || !availability.available) return null;
  const isMajorMinor = availability.upgradeType === "minor" || availability.upgradeType === "major";
  return (
    <div className="torch-upgrade-pill torch-upgrade-pill-available">
      <div className="torch-upgrade-pill-summary">
        <strong>Torch upgrade available:</strong>{" "}
        <code>{availability.current}</code> → <code>{availability.latest}</code>{" "}
        <span className={`torch-upgrade-badge torch-upgrade-badge-${availability.upgradeType}`}>
          {availability.upgradeType}
        </span>
      </div>
      <div className="torch-upgrade-pill-actions">
        <button
          className="primary-button"
          type="button"
          onClick={handleUpgrade}
          disabled={starting}
        >
          {starting ? "Starting…" : "Upgrade torch"}
        </button>
        <button
          className="link-button"
          type="button"
          onClick={() => setHidden(true)}
        >
          Dismiss
        </button>
      </div>
      {isMajorMinor && availability.rebuildPackages.length > 0 ? (
        <details className="torch-upgrade-pill-details">
          <summary>
            Will also rebuild {availability.rebuildPackages.length}{" "}
            ABI-dependent package{availability.rebuildPackages.length === 1 ? "" : "s"}
          </summary>
          <p className="muted-text">
            Torch {availability.upgradeType} bumps break the C++ ABI, so wheels
            for these packages won't load until they're reinstalled against the
            new torch:
          </p>
          <ul className="torch-upgrade-pill-pkg-list">
            {availability.rebuildPackages.map((pkg) => (
              <li key={pkg}>
                <code>{pkg}</code>
              </li>
            ))}
          </ul>
          <p className="muted-text">
            If the upgrade fails verification, your previous torch{" "}
            (<code>{availability.current}</code>) is automatically restored
            from a sibling rollback directory — no re-download needed.
          </p>
        </details>
      ) : (
        <details className="torch-upgrade-pill-details">
          <summary>What happens when I upgrade?</summary>
          <p className="muted-text">
            Patch bumps keep the torch C++ ABI stable, so only the torch wheel
            itself is reinstalled. Your current torch is moved to a sibling
            rollback directory first; if the new wheel fails CUDA verification,
            the previous one is restored automatically. Restart Backend
            activates the new wheel.
          </p>
        </details>
      )}
    </div>
  );
}

interface TorchUpgradeJobStatusProps {
  job: TorchUpgradeJobState;
  onRestartBackend: () => void;
  onClose: () => void;
  restartBusy: boolean;
}

function TorchUpgradeJobStatus({ job, onRestartBackend, onClose, restartBusy }: TorchUpgradeJobStatusProps) {
  const inFlight = !job.done;
  const success = job.done && !job.error && job.cudaVerified === true;
  const failed = job.done && (job.error !== null || job.cudaVerified === false);

  return (
    <>
      <div className="torch-upgrade-pill-summary">
        <strong>
          {inFlight ? "Upgrading torch" : success ? "Torch upgrade complete" : "Torch upgrade failed"}
        </strong>
        {job.currentVersion && job.targetVersion ? (
          <>
            {": "}
            <code>{job.currentVersion}</code> → <code>{job.targetVersion}</code>
          </>
        ) : null}
      </div>
      <div className="torch-upgrade-pill-message">{job.message}</div>
      {success && job.requiresRestart ? (
        <div className="torch-upgrade-pill-actions">
          <button
            className="primary-button"
            type="button"
            onClick={onRestartBackend}
            disabled={restartBusy}
          >
            {restartBusy ? "Restarting…" : "Restart Backend to activate"}
          </button>
          <button className="link-button" type="button" onClick={onClose}>
            Dismiss
          </button>
        </div>
      ) : null}
      {failed ? (
        <div className="torch-upgrade-pill-actions">
          {job.rolledBack ? (
            <span className="muted-text">
              Previous torch restored from rollback — generation continues on the prior version.
            </span>
          ) : job.rollbackPath ? (
            <span className="muted-text">
              Rollback dir kept at <code>{job.rollbackPath}</code> for manual recovery.
            </span>
          ) : null}
          <button className="link-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>
      ) : null}
      {job.attempts.length > 0 ? (
        <details className="torch-upgrade-pill-details" open={failed}>
          <summary>
            Install log ({job.attempts.length}{" "}
            {job.attempts.length === 1 ? "step" : "steps"})
          </summary>
          <pre className="torch-upgrade-pill-terminal">
            {job.attempts
              .map((attempt) => {
                const marker = attempt.ok ? "[ OK ]" : "[FAIL]";
                const label = attemptLabel(attempt);
                const body = attempt.output
                  ? "\n" +
                    attempt.output
                      .split(/\r?\n/)
                      .map((line) => `       ${line}`)
                      .join("\n")
                  : "";
                return `${marker} ${label}${body}`;
              })
              .join("\n\n")}
          </pre>
        </details>
      ) : null}
    </>
  );
}

function attemptLabel(attempt: TorchUpgradeJobState["attempts"][number]): string {
  switch (attempt.phase) {
    case "rollback-prepare":
      return "Move existing torch to rollback dir";
    case "install":
      return "pip install torch (new version)";
    case "deps":
      return "pip install torch (transitive deps)";
    case "constraint":
      return "Pin torch in constraints.txt";
    case "rebuild":
      return attempt.package ? `pip install ${attempt.package} (--force-reinstall)` : "Rebuild ABI-dependent package";
    case "verify":
      return "Verify torch.cuda.is_available() in subprocess";
    case "cleanup":
      return "Prune old rollback dirs";
    case "rollback-restore":
      return "Restore previous torch from rollback";
    default:
      return attempt.phase ?? "step";
  }
}
