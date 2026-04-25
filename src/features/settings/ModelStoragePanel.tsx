import { useCallback, useEffect, useRef, useState } from "react";
import { Panel } from "../../components/Panel";
import {
  getModelMoveStatus,
  getStorageSettings,
  startModelMove,
  updateHfCachePath,
  type ModelMoveJobState,
  type StorageSettingsSnapshot,
} from "../../api";

export interface ModelStoragePanelProps {
  backendOnline: boolean;
  onRestartServer: () => void;
  busyAction: string | null;
  onPickDirectory: (currentPath?: string) => Promise<string | null>;
}

// Formats bytes as GB/MB/KB — same units we use everywhere else. Unlike
// ``fmtBytes`` in DiagnosticsPanel this one returns a friendlier "—" for
// missing values so the panel's stat rows always have something to render.
function fmtBytes(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function fmtPhase(phase: ModelMoveJobState["phase"]): string {
  switch (phase) {
    case "preflight":
      return "Checking sizes + free space";
    case "copying":
      return "Copying";
    case "cleanup":
      return "Removing old copy";
    case "done":
      return "Done";
    case "error":
      return "Failed";
    default:
      return "";
  }
}

// The move worker runs in a background thread; we poll at 1 Hz while the
// job is active to surface per-file progress. Stops polling when the job
// reaches ``done`` or ``error`` — no point hammering the API after that.
function useMoveJobPoll(
  active: boolean,
  onUpdate: (job: ModelMoveJobState) => void,
) {
  const activeRef = useRef(active);
  activeRef.current = active;
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const job = await getModelMoveStatus();
        if (cancelled) return;
        onUpdate(job);
      } catch {
        // Transient network / auth — swallow and try again next tick.
      }
    };
    void tick();
    const handle = window.setInterval(() => {
      if (cancelled || !activeRef.current) return;
      void tick();
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);
}

export function ModelStoragePanel({
  backendOnline,
  onRestartServer,
  busyAction,
  onPickDirectory,
}: ModelStoragePanelProps) {
  const [snapshot, setSnapshot] = useState<StorageSettingsSnapshot | null>(null);
  const [draftPath, setDraftPath] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startingMove, setStartingMove] = useState(false);

  const refresh = useCallback(async () => {
    if (!backendOnline) return;
    setLoading(true);
    setError(null);
    try {
      const next = await getStorageSettings();
      setSnapshot(next);
      // Only seed the input on first load or after a save — don't clobber
      // the user's in-progress edit on a background refresh.
      setDraftPath((prev) => (prev === "" ? next.configuredPath : prev));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [backendOnline]);

  useEffect(() => {
    if (backendOnline && snapshot === null) {
      void refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendOnline]);

  const moveJob = snapshot?.moveJob;
  const moveActive =
    moveJob !== undefined
    && (moveJob.phase === "preflight"
      || moveJob.phase === "copying"
      || moveJob.phase === "cleanup");

  // Poll the job state while it's running so the progress bar keeps moving
  // without the user having to click refresh.
  useMoveJobPoll(moveActive, (job) => {
    setSnapshot((prev) => (prev ? { ...prev, moveJob: job } : prev));
  });

  async function handleBrowse() {
    const next = await onPickDirectory(draftPath);
    if (next) setDraftPath(next);
  }

  async function handleSavePath() {
    if (saving) return;
    setSaving(true);
    setError(null);
    setSavedMessage(null);
    try {
      const result = await updateHfCachePath(draftPath.trim());
      setSavedMessage(
        result.restartRequired
          ? "Path saved. Restart the backend for new downloads to land on the new drive."
          : "Path saved.",
      );
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleReset() {
    if (saving) return;
    setDraftPath("");
    setSaving(true);
    setError(null);
    setSavedMessage(null);
    try {
      const result = await updateHfCachePath("");
      setSavedMessage(
        result.restartRequired
          ? "Reset to default. Restart the backend to use the default location."
          : "Reset to default.",
      );
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleMove() {
    if (!snapshot) return;
    const target = (draftPath || snapshot.configuredPath || "").trim();
    if (!target) {
      setError("Set a destination path first (type or Browse above).");
      return;
    }
    const bytes = snapshot.currentHubSizeBytes;
    const confirmed = window.confirm(
      `Move ${fmtBytes(bytes)} of models from ${snapshot.effectiveHubPath} to ${target}?\n\n`
        + "The source tree will be deleted after a successful copy. On a spinning HDD this can take 15-45 min for 100 GB.\n\n"
        + "Close any running generations before starting.",
    );
    if (!confirmed) return;
    setStartingMove(true);
    setError(null);
    try {
      const job = await startModelMove(target, true);
      setSnapshot((prev) => (prev ? { ...prev, moveJob: job } : prev));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setStartingMove(false);
    }
  }

  const configuredPath = snapshot?.configuredPath ?? "";
  const effectivePath = snapshot?.effectivePath ?? "";
  const restartRequired =
    !!snapshot
    && draftPath.trim() === configuredPath
    && configuredPath !== ""
    && effectivePath !== ""
    && !effectivePath.startsWith(configuredPath)
    && !configuredPath.startsWith(effectivePath);
  const pathDirty = draftPath.trim() !== configuredPath;
  const moveDone = moveJob?.phase === "done";
  const moveError = moveJob?.phase === "error";
  const showMoveButton =
    !!snapshot
    && snapshot.currentHubSizeBytes > 0
    && !moveActive
    && pathDirty;

  return (
    <Panel
      title="Model Storage"
      subtitle="Redirect Hugging Face model downloads to a different drive. Useful when your system drive is filling up with 14-100 GB image and video checkpoints."
    >
      {!snapshot && loading ? (
        <p className="muted-text">Reading storage info...</p>
      ) : null}

      {!snapshot && !loading && !error ? (
        <p className="muted-text">
          {backendOnline ? "Loading storage info..." : "Backend offline — start the sidecar to configure storage."}
        </p>
      ) : null}

      {error ? (
        <p className="muted-text" style={{ color: "rgb(252, 165, 165)" }}>{error}</p>
      ) : null}

      {snapshot ? (
        <div className="control-stack">
          <div className="field-label-row">
            <label className="field-label">Current cache location</label>
          </div>
          <div className="directory-add-row">
            <input
              className="text-input directory-add-path mono-text"
              type="text"
              readOnly
              value={snapshot.effectiveHubPath}
            />
            <span className="badge muted">{fmtBytes(snapshot.currentHubSizeBytes)} in cache</span>
            <span className="badge muted">{fmtBytes(snapshot.currentFreeBytes)} free</span>
          </div>

          <div className="field-label-row" style={{ marginTop: 14 }}>
            <label className="field-label">Override path (requires restart)</label>
            {!configuredPath ? <span className="badge muted">using default</span> : null}
          </div>
          <div className="directory-add-row">
            <input
              className="text-input directory-add-path mono-text"
              type="text"
              placeholder="D:\AI\huggingface or ~/ai-models"
              value={draftPath}
              onChange={(event) => setDraftPath(event.target.value)}
              disabled={saving || moveActive}
            />
            <button
              className="secondary-button"
              type="button"
              onClick={() => void handleBrowse()}
              disabled={saving || moveActive}
            >
              Browse...
            </button>
            <button
              className="primary-button"
              type="button"
              onClick={() => void handleSavePath()}
              disabled={saving || moveActive || !pathDirty}
            >
              {saving ? "Saving..." : "Save path"}
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => void handleReset()}
              disabled={saving || moveActive || !configuredPath}
            >
              Reset to default
            </button>
          </div>

          {savedMessage ? (
            <p className="help-text" style={{ color: "rgb(134, 239, 172)" }}>{savedMessage}</p>
          ) : null}
          {restartRequired ? (
            <div className="button-row" style={{ marginTop: 4 }}>
              <button
                className="secondary-button"
                type="button"
                onClick={() => onRestartServer()}
                disabled={Boolean(busyAction)}
              >
                {busyAction === "Restarting server..." ? "Restarting..." : "Restart backend now"}
              </button>
            </div>
          ) : null}

          {showMoveButton ? (
            <div style={{ marginTop: 14 }}>
              <p className="help-text" style={{ margin: "0 0 6px" }}>
                You have {fmtBytes(snapshot.currentHubSizeBytes)} of models at the old location.
                Move them so they keep working from the new path. On a spinning HDD, budget
                ~1 minute per 6 GB.
              </p>
              <div className="button-row">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void handleMove()}
                  disabled={startingMove || !backendOnline}
                >
                  {startingMove ? "Starting..." : "Move existing models to new path"}
                </button>
              </div>
            </div>
          ) : null}

          {moveJob && (moveActive || moveDone || moveError) ? (
            <div className="callout image-callout" style={{ marginTop: 14 }}>
              <div className="field-label-row">
                <strong>{fmtPhase(moveJob.phase)}</strong>
                {moveActive ? (
                  <span className="badge muted">{moveJob.percent.toFixed(1)}%</span>
                ) : null}
              </div>
              <p className="muted-text" style={{ margin: "4px 0" }}>{moveJob.message}</p>
              {moveActive ? (
                <>
                  <div className="progress-bar" style={{ margin: "6px 0" }}>
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${Math.max(1, Math.min(100, moveJob.percent))}%` }}
                    />
                  </div>
                  <p className="muted-text" style={{ fontSize: 12 }}>
                    {fmtBytes(moveJob.bytesCopied)} / {fmtBytes(moveJob.bytesTotal)} ·
                    {" "}{moveJob.filesCopied.toLocaleString()} / {moveJob.filesTotal.toLocaleString()} files
                    {moveJob.currentEntry ? <> · <code>{moveJob.currentEntry}</code></> : null}
                  </p>
                </>
              ) : null}
              {moveDone ? (
                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => onRestartServer()}
                    disabled={Boolean(busyAction)}
                  >
                    {busyAction === "Restarting server..." ? "Restarting..." : "Restart backend"}
                  </button>
                </div>
              ) : null}
              {moveError && moveJob.error ? (
                <p className="mono-text" style={{ color: "rgb(252, 165, 165)", fontSize: 12 }}>
                  {moveJob.error}
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
    </Panel>
  );
}
