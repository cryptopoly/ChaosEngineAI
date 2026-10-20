import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import {
  fetchDiagnosticsSnapshot,
  deleteModelPath,
  fetchStorageChildren,
  fetchStorageTop,
  installCudaTorch,
  installPipPackage,
  reextractRuntime,
  type CudaTorchInstallResult,
  type DiagnosticsSnapshot,
  type InstallResult,
  type StorageTopEntry,
  type StorageTopResponse,
} from "../../api";
import { AcceleratorsBoostPack } from "./AcceleratorsBoostPack";
import { WslBridgePanel } from "./WslBridgePanel";

// In-app troubleshooting panel. Surfaces OS, hardware, runtime paths,
// GPU state, env vars, and the backend log tail without asking users to
// run PowerShell commands. The Copy button formats everything as Markdown
// so users can paste a single block into a support thread.
//
// Design bias: observability first, actions sparingly. The only action
// is "Re-extract runtime" (deletes %TEMP% extraction on next restart);
// the rest is read-only. Empty / missing values are rendered as "—" so
// users can distinguish "not applicable here" from "broken".

export interface DiagnosticsPanelProps {
  backendOnline: boolean;
  onRestartServer: () => void;
  busyAction: string | null;
}

export function DiagnosticsPanel({ backendOnline, onRestartServer, busyAction }: DiagnosticsPanelProps) {
  const { t } = useTranslation("diagnostics");
  const [snapshot, setSnapshot] = useState<DiagnosticsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<number | null>(null);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");
  const [reextractBusy, setReextractBusy] = useState(false);
  const [reextractMessage, setReextractMessage] = useState<string | null>(null);
  const [cudaBusy, setCudaBusy] = useState(false);
  const [cudaResult, setCudaResult] = useState<CudaTorchInstallResult | null>(null);
  const [mlxVideoBusy, setMlxVideoBusy] = useState(false);
  const [mlxVideoResult, setMlxVideoResult] = useState<InstallResult | null>(null);

  const refresh = useCallback(async () => {
    if (!backendOnline) return;
    setLoading(true);
    setError(null);
    try {
      const next = await fetchDiagnosticsSnapshot();
      setSnapshot(next);
      setLastFetchedAt(Date.now());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [backendOnline]);

  useEffect(() => {
    // One-shot fetch when the panel first becomes reachable. Users can
    // hit Refresh to pull again; we deliberately don't auto-poll so the
    // torch probe subprocess doesn't fire every second.
    if (backendOnline && snapshot === null && !loading) {
      void refresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendOnline]);

  async function handleCopy() {
    if (!snapshot) return;
    const markdown = formatSnapshotAsMarkdown(snapshot);
    try {
      await navigator.clipboard.writeText(markdown);
      setCopyStatus("copied");
      window.setTimeout(() => setCopyStatus("idle"), 2500);
    } catch {
      setCopyStatus("failed");
      window.setTimeout(() => setCopyStatus("idle"), 2500);
    }
  }

  async function handleInstallCuda() {
    if (cudaBusy) return;
    const confirmed = window.confirm(
      "Reinstall CUDA torch into the extras directory? This downloads ~2 GB, takes 1-5 minutes, and replaces any stale or CPU-only torch currently installed. Use this when the GPU is detected but torch imports stuck as CPU.",
    );
    if (!confirmed) return;
    setCudaBusy(true);
    setCudaResult(null);
    try {
      const result = await installCudaTorch();
      setCudaResult(result);
      if (result.ok) {
        // Re-probe so the snapshot refreshes with the new torch state.
        await refresh();
      }
    } catch (err) {
      setCudaResult({
        ok: false,
        output: err instanceof Error ? err.message : String(err),
        indexUrl: null,
        attempts: [],
        requiresRestart: false,
        pythonExecutable: "",
        pythonVersion: null,
        noWheelForPython: false,
        capabilities: {},
      });
    } finally {
      setCudaBusy(false);
    }
  }

  async function handleInstallMlxVideo() {
    if (mlxVideoBusy) return;
    const confirmed = window.confirm(
      "Reinstall mlx-video from the Blaizzy/mlx-video git source? This pulls the package fresh from GitHub (replaces any stale or PyPI-shadow install) and takes 1-2 minutes. Use this on Apple Silicon to enable LTX-2 native generation, or after a failed install left the package partially staged.",
    );
    if (!confirmed) return;
    setMlxVideoBusy(true);
    setMlxVideoResult(null);
    try {
      const result = await installPipPackage("mlx-video");
      setMlxVideoResult(result);
      if (result.ok) {
        await refresh();
      }
    } catch (err) {
      setMlxVideoResult({
        ok: false,
        output: err instanceof Error ? err.message : String(err),
        capabilities: {},
      });
    } finally {
      setMlxVideoBusy(false);
    }
  }

  async function handleReextract() {
    if (reextractBusy) return;
    const confirmed = window.confirm(
      "Delete the cached embedded runtime extraction? The backend will re-extract from scratch on the next restart. Use this when the runtime tree looks corrupted (rare — usually after a crashed install).",
    );
    if (!confirmed) return;
    setReextractBusy(true);
    setReextractMessage(null);
    try {
      const result = await reextractRuntime();
      if (result.error) {
        setReextractMessage(`Re-extract failed: ${result.error}`);
      } else if (result.deleted) {
        setReextractMessage(
          `Runtime cache deleted at ${result.path}. Click Restart Backend to re-extract.`,
        );
      } else {
        setReextractMessage(
          `No cached runtime to delete at ${result.path ?? "(unknown path)"} — next restart will extract fresh.`,
        );
      }
    } catch (err) {
      setReextractMessage(`Re-extract failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setReextractBusy(false);
    }
  }

  const copyLabel = copyStatus === "copied"
    ? "Copied!"
    : copyStatus === "failed"
      ? "Copy failed"
      : "Copy diagnostics";

  return (
    <Panel
      title={t("title", { defaultValue: "Diagnostics" })}
      subtitle={t("subtitleSnapshot", {
        defaultValue: "System snapshot for troubleshooting. Copy-paste the markdown to share with support when something misbehaves.",
      })}
      actions={
        <div className="button-row">
          <button
            className="secondary-button"
            type="button"
            onClick={() => void refresh()}
            disabled={!backendOnline || loading}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
          <button
            className="primary-button"
            type="button"
            onClick={() => void handleCopy()}
            disabled={!snapshot}
          >
            {copyLabel}
          </button>
        </div>
      }
    >
      {error ? (
        <p className="muted-text" style={{ color: "rgb(252, 165, 165)" }}>
          Could not fetch diagnostics: {error}
        </p>
      ) : null}

      {!snapshot && loading ? (
        <p className="muted-text">Gathering system info...</p>
      ) : null}

      {!snapshot && !loading && !error ? (
        <p className="muted-text">
          {backendOnline ? "Click Refresh to fetch diagnostics." : "Backend offline — start the sidecar first."}
        </p>
      ) : null}

      {snapshot ? (
        <div className="diagnostics-body">
          <DiagnosticsHeader snapshot={snapshot} lastFetchedAt={lastFetchedAt} />
          <DiagnosticsSection title="OS" rows={[
            ["System", snapshot.os.system as string],
            ["Release", snapshot.os.release as string],
            ["Version", snapshot.os.version as string],
            ["Machine", snapshot.os.machine as string],
            ["Platform", snapshot.os.platform as string],
          ]} />
          <DiagnosticsSection title="Hardware" rows={[
            ["CPU logical cores", String(snapshot.hardware.cpu?.logicalCount ?? "—")],
            ["CPU physical cores", String(snapshot.hardware.cpu?.physicalCount ?? "—")],
            ["Memory total", snapshot.hardware.memory.totalGb != null ? `${snapshot.hardware.memory.totalGb} GB` : "—"],
            ["Memory available", snapshot.hardware.memory.availableGb != null ? `${snapshot.hardware.memory.availableGb} GB` : "—"],
            ["Swap used / total", `${fmtGb(snapshot.hardware.swap.usedGb)} / ${fmtGb(snapshot.hardware.swap.totalGb)}`],
            ["GPU", (snapshot.hardware.gpu.gpuName as string) || "—"],
            ["VRAM total", (snapshot.hardware.gpu.vramTotalGb as number | null) != null ? `${snapshot.hardware.gpu.vramTotalGb} GB` : "—"],
            ["Driver", (snapshot.hardware.gpu.driverVersion as string) || "—"],
            ["System CUDA (driver)", (snapshot.hardware.gpu.systemCudaVersion as string) || "—"],
          ]} />
          <DiagnosticsSection title="Python" rows={[
            ["Executable", snapshot.python.executable],
            ["Version", snapshot.python.version ?? "—"],
            ["Implementation", snapshot.python.implementation],
            ["Prefix", snapshot.python.prefix],
            ["Base prefix", snapshot.python.basePrefix],
            ["cwd", snapshot.python.cwd ?? "—"],
          ]} />
          <DiagnosticsSection title="Runtime" rows={[
            ["Engine", snapshot.runtime.engineLabel ?? "—"],
            ["Loaded model", (snapshot.runtime.loadedModel?.name as string) ?? "none"],
            ["Warm pool size", String(snapshot.runtime.warmPoolCount ?? 0)],
            ["llama-server", snapshot.runtime.llamaServerPath ?? "not set"],
            ["llama-server-turbo", snapshot.runtime.llamaServerTurboPath ?? "not set"],
            ["llama-cli", snapshot.runtime.llamaCliPath ?? "not set"],
          ]} />
          <DiagnosticsSection title="GPU packages" rows={[
            ["torch", fmtPresent(snapshot.gpu.torchFindSpec)],
            ["diffusers", fmtPresent(snapshot.gpu.diffusersFindSpec)],
            ["accelerate", fmtPresent(snapshot.gpu.accelerateFindSpec)],
            ["transformers", fmtPresent(snapshot.gpu.transformersFindSpec)],
            ["imageio", fmtPresent(snapshot.gpu.imageioFindSpec)],
            ["imageio-ffmpeg", fmtPresent(snapshot.gpu.ffmpegFindSpec)],
            ["sentencepiece", fmtPresent(snapshot.gpu.sentencepieceFindSpec)],
            ["tiktoken", fmtPresent(snapshot.gpu.tiktokenFindSpec)],
            ["protobuf", fmtPresent(snapshot.gpu.protobufFindSpec)],
            ["ftfy", fmtPresent(snapshot.gpu.ftfyFindSpec)],
          ]} />
          {snapshot.gpu.torchSubprocess ? (
            <DiagnosticsSection title="torch runtime" rows={Object.entries(snapshot.gpu.torchSubprocess).map(
              ([k, v]) => [k, String(v)],
            )} />
          ) : null}
          <DiagnosticsSection title="Extras (GPU install target)" rows={[
            ["Path", snapshot.extras.path],
            ["Exists", snapshot.extras.exists ? "yes" : "no"],
            ["Size", snapshot.extras.sizeBytes != null ? fmtBytes(snapshot.extras.sizeBytes) : "—"],
            ["Free on volume", snapshot.extras.freeBytes != null ? fmtBytes(snapshot.extras.freeBytes) : "—"],
            ["Top-level entries",
              snapshot.extras.topLevelEntries.length
                ? snapshot.extras.topLevelEntries.slice(0, 20).join(", ")
                : "(empty)"],
          ]} />
          <DiagnosticsSection title="Environment" rows={
            Object.entries(snapshot.environment).map(([k, v]) => [k, v ?? "(unset)"])
          } />

          <details className="install-log-panel" style={{ marginTop: 16 }}>
            <summary className="install-log-summary">
              Backend log tail ({snapshot.logs.tailLines.length} lines)
              {snapshot.logs.path ? <> · <code style={{ fontSize: "0.7rem" }}>{snapshot.logs.path}</code></> : null}
            </summary>
            <div className="install-log-body">
              <pre className="install-log-output" style={{ maxHeight: 420 }}>
                {snapshot.logs.tailLines.join("\n") || "(no log content)"}
              </pre>
            </div>
          </details>

          {(() => {
            // Platform-aware install buttons. Apple Silicon hosts get the
            // mlx-video reinstall (LTX-2 native engine); CUDA hosts get the
            // torch reinstall. Showing both to everyone is noise; showing
            // neither hides recovery actions when a prior install broke.
            const osSystem = String(snapshot.os.system ?? "").toLowerCase();
            const osMachine = String(snapshot.os.machine ?? "").toLowerCase();
            const isAppleSilicon = osSystem === "darwin" && (osMachine === "arm64" || osMachine === "aarch64");
            const isCudaHost = osSystem === "windows" || osSystem === "linux";
            return (
              <>
                {isAppleSilicon ? (
                  <div className="diagnostics-actions" style={{ marginTop: 18 }}>
                    <p className="muted-text" style={{ margin: "0 0 8px" }}>
                      Reinstall mlx-video from the Blaizzy/mlx-video git source.
                      Use this on Apple Silicon when LTX-2 generation fails with
                      ``ModuleNotFoundError: No module named 'mlx_video.ltx_2'``
                      (the unrelated PyPI 0.1.0 package shadowed the real one),
                      or when a prior install left the package partially staged.
                    </p>
                    <div className="button-row">
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void handleInstallMlxVideo()}
                        disabled={!backendOnline || mlxVideoBusy}
                      >
                        {mlxVideoBusy ? "Installing mlx-video..." : "Install / reinstall mlx-video"}
                      </button>
                    </div>
                    {mlxVideoResult ? (
                      <div style={{ marginTop: 8 }}>
                        <p className="muted-text" style={{ margin: 0 }}>
                          {mlxVideoResult.ok
                            ? "mlx-video installed. Restart the backend to activate."
                            : "Install failed — see the pip output below."}
                        </p>
                        {mlxVideoResult.output ? (
                          <details style={{ marginTop: 4 }}>
                            <summary className="muted-text" style={{ cursor: "pointer" }}>pip output</summary>
                            <pre className="install-log-output" style={{ maxHeight: 240 }}>{mlxVideoResult.output}</pre>
                          </details>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {isCudaHost ? (
                  <div className="diagnostics-actions" style={{ marginTop: 18 }}>
                    <p className="muted-text" style={{ margin: "0 0 8px" }}>
                      Reinstall CUDA torch into the extras directory. Use this on Windows/Linux when an NVIDIA
                      GPU is detected but torch imports as CPU-only, or when a prior install left stale /
                      mismatched torch files in extras. The main-page banner offers the same action but can
                      be dismissed — this button is always available.
                    </p>
                    <div className="button-row">
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void handleInstallCuda()}
                        disabled={!backendOnline || cudaBusy}
                      >
                        {cudaBusy ? "Installing CUDA torch..." : "Install / reinstall CUDA torch"}
                      </button>
                    </div>
                    {cudaResult ? (
                      <div style={{ marginTop: 8 }}>
                        <p className="muted-text" style={{ margin: 0 }}>
                          {cudaResult.ok
                            ? `Installed from ${cudaResult.indexUrl ?? "(unknown index)"}. Restart the backend to activate.`
                            : cudaResult.noWheelForPython
                              ? `No CUDA torch wheel exists for Python ${cudaResult.pythonVersion ?? "(unknown)"}. Ship ChaosEngineAI against a supported Python (3.13 recommended).`
                              : "Install failed — see the log tail below for the full pip output."}
                        </p>
                        {cudaResult.output ? (
                          <details style={{ marginTop: 4 }}>
                            <summary className="muted-text" style={{ cursor: "pointer" }}>pip output</summary>
                            <pre className="install-log-output" style={{ maxHeight: 240 }}>{cudaResult.output}</pre>
                          </details>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </>
            );
          })()}

          <div className="diagnostics-actions" style={{ marginTop: 18 }}>
            <p className="muted-text" style={{ margin: "0 0 8px" }}>
              Advanced: delete the cached embedded runtime and re-extract on next restart. Use this
              only when the runtime tree looks corrupted (rare — usually after a crashed install).
            </p>
            <div className="button-row">
              <button
                className="secondary-button"
                type="button"
                onClick={() => void handleReextract()}
                disabled={!backendOnline || reextractBusy}
              >
                {reextractBusy ? "Deleting..." : "Re-extract runtime"}
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={() => onRestartServer()}
                disabled={Boolean(busyAction)}
              >
                {busyAction === "Restarting server..." ? "Restarting..." : "Restart Backend"}
              </button>
            </div>
            {reextractMessage ? (
              <p className="muted-text" style={{ marginTop: 8 }}>{reextractMessage}</p>
            ) : null}
          </div>
        </div>
      ) : null}
    </Panel>
  );
}

// Settings > Accelerators. The Boost Pack and the WSL bridge used to sit at
// the bottom of the Diagnostics panel, below the system snapshot; refreshing
// the snapshot grew that panel and pushed them off screen with no way to scroll
// to them, so they now have their own tab.
export function AcceleratorsPanel({ backendOnline }: { backendOnline: boolean }) {
  return (
    <Panel
      title="Accelerators"
      subtitle="Optional speed-ups for image, video and chat inference, and the Windows WSL bridge."
    >
      <AcceleratorsBoostPack backendOnline={backendOnline} />
      <WslBridgePanel backendOnline={backendOnline} />
    </Panel>
  );
}

// Settings > Disk usage (FU-055), for the same reason.
export function DiskUsagePanel({ backendOnline }: { backendOnline: boolean }) {
  return (
    <Panel
      title="Disk usage"
      subtitle="Which model repos take the space, so you know what to delete."
    >
      <StorageTopSection backendOnline={backendOnline} />
    </Panel>
  );
}

// ---- Sub-components -----------------------------------------------

function StorageTopSection({ backendOnline }: { backendOnline: boolean }) {
  // FU-055: in-app top-disk-consumers view. Walks every enabled model dir
  // and reports the biggest repo dirs, sorted by actual byte size. Closes
  // the Stuff Diver gap on HF cache symlink layouts.
  const [data, setData] = useState<StorageTopResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [freedBytes, setFreedBytes] = useState(0);

  const scan = useCallback(async () => {
    if (!backendOnline) return;
    setBusy(true);
    setErr(null);
    setFreedBytes(0);
    try {
      const next = await fetchStorageTop(20);
      setData(next);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [backendOnline]);

  return (
    <div className="diagnostics-section">
      <h3>Top 20 model repos</h3>
      <p className="muted-text" style={{ marginBottom: 8 }}>
        Walks every enabled model directory and lists the biggest repos on disk.
        Counts blob bytes once (Hugging Face's ``snapshots/`` are symlinks to
        ``blobs/`` so naive scanners over-count).
      </p>
      <button
        className="secondary-button"
        type="button"
        onClick={() => void scan()}
        disabled={!backendOnline || busy}
      >
        {busy ? "Scanning..." : data ? "Rescan" : "Scan disk usage"}
      </button>
      {err ? <p className="muted-text" style={{ color: "#f87171", marginTop: 8 }}>{err}</p> : null}
      {data ? (
        <div style={{ marginTop: 12 }}>
          <p className="muted-text">
            Total across all enabled directories:{" "}
            <strong>{data.totalGb !== null ? `${data.totalGb.toFixed(1)} GB` : "—"}</strong>
            {" · "}
            {data.entries.length} repos shown / scanned{" "}
            {data.scannedDirectories.length} root dir(s)
            {freedBytes > 0 ? (
              <>
                {" · "}
                <strong>{(freedBytes / 1e9).toFixed(1)} GB freed since scan</strong>
              </>
            ) : null}
          </p>
          <div className="storage-top-table">
            <div className="storage-top-head">
              <span>Repo</span>
              <span>Size</span>
              <span>Source</span>
              <span>Path</span>
              <span />
            </div>
            {data.entries.map((entry) => (
              <StorageRow
                key={entry.path}
                entry={entry}
                depth={0}
                onDeleted={(removed) => setFreedBytes((n) => n + removed.sizeBytes)}
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function StorageRow({
  entry,
  depth,
  onDeleted,
}: {
  entry: StorageTopEntry;
  depth: number;
  onDeleted: (entry: StorageTopEntry) => void;
}) {
  const [open, setOpen] = useState(false);
  const [gone, setGone] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [children, setChildren] = useState<StorageTopEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (!next || children !== null) return;
    setLoading(true);
    setErr(null);
    try {
      setChildren(await fetchStorageChildren(entry.path));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const remove = async () => {
    const size = entry.sizeGb !== null ? `${entry.sizeGb.toFixed(1)} GB` : "unknown size";
    const confirmed = window.confirm(
      `Permanently delete ${entry.path} (${size})?\n\nThis removes it from disk and cannot be undone.`,
    );
    if (!confirmed) return;
    setDeleting(true);
    setErr(null);
    try {
      await deleteModelPath(entry.path);
      setGone(true);
      onDeleted(entry);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  };

  if (gone) return null;

  return (
    <>
      <div className="storage-top-row">
        <span className="mono-text" style={{ paddingLeft: depth * 16 }}>
          {entry.expandable ? (
            <button
              type="button"
              className="storage-top-toggle"
              onClick={() => void toggle()}
              aria-expanded={open}
              aria-label={open ? `Collapse ${entry.repoLabel}` : `Expand ${entry.repoLabel}`}
            >
              {open ? "▾" : "▸"}
            </button>
          ) : (
            <span className="storage-top-toggle" aria-hidden="true" />
          )}
          {entry.repoLabel}
        </span>
        <span><strong>{entry.sizeGb !== null ? `${entry.sizeGb.toFixed(1)} GB` : "—"}</strong></span>
        <span className="muted-text">{entry.sourceKind || "—"}</span>
        <span className="mono-text muted-text storage-top-path" title={entry.path}>
          {entry.path}
        </span>
        {entry.deletable === false ? (
          <span className="muted-text" title="Shared Hugging Face store: many repos link into it">shared</span>
        ) : (
          <button
            type="button"
            className="secondary-button storage-top-delete"
            onClick={() => void remove()}
            disabled={deleting}
          >
            {deleting ? "Deleting…" : "Delete"}
          </button>
        )}
      </div>
      {open && loading ? <div className="storage-top-row muted-text">Reading…</div> : null}
      {open && err ? <div className="storage-top-row" style={{ color: "#f87171" }}>{err}</div> : null}
      {open && children?.map((child) => (
        <StorageRow key={child.path} entry={child} depth={depth + 1} onDeleted={onDeleted} />
      ))}
      {open && children?.length === 0 ? (
        <div className="storage-top-row muted-text" style={{ paddingLeft: 12 + (depth + 1) * 16 }}>Empty</div>
      ) : null}
    </>
  );
}

function DiagnosticsHeader({ snapshot, lastFetchedAt }: { snapshot: DiagnosticsSnapshot; lastFetchedAt: number | null }) {
  const timestamp = lastFetchedAt ? new Date(lastFetchedAt).toLocaleString() : "—";
  return (
    <div className="diagnostics-header">
      <div>
        <strong>ChaosEngineAI v{snapshot.app.appVersion}</strong>
        <span className="muted-text" style={{ marginLeft: 8 }}>
          captured {timestamp}
        </span>
      </div>
    </div>
  );
}

type Row = [string, string];

function DiagnosticsSection({ title, rows }: { title: string; rows: Row[] }) {
  return (
    <details className="diagnostics-section" open>
      <summary className="diagnostics-section-title">{title}</summary>
      <table className="diagnostics-table">
        <tbody>
          {rows.map(([key, value], idx) => (
            <tr key={`${key}-${idx}`}>
              <td className="diagnostics-key">{key}</td>
              <td className="diagnostics-value">{value || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

// ---- Helpers ------------------------------------------------------

function fmtGb(value: number | null | undefined): string {
  return value != null ? `${value} GB` : "—";
}

function fmtBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function fmtPresent(flag: boolean): string {
  return flag ? "installed" : "missing";
}

// Turn the JSON snapshot into a paste-friendly Markdown document.
// Kept close to how the panel renders visually so users don't have to
// read two different formats.
function formatSnapshotAsMarkdown(snapshot: DiagnosticsSnapshot): string {
  const lines: string[] = [];
  lines.push(`# ChaosEngineAI Diagnostics`);
  lines.push("");
  lines.push(`- App version: \`${snapshot.app.appVersion}\``);
  lines.push(`- Captured at: ${new Date(snapshot.generatedAt * 1000).toISOString()}`);
  lines.push("");

  lines.push(`## OS`);
  lines.push(renderKvTable({
    System: snapshot.os.system,
    Release: snapshot.os.release,
    Version: snapshot.os.version,
    Machine: snapshot.os.machine,
    Platform: snapshot.os.platform,
    Processor: snapshot.os.processor,
  }));

  lines.push(`## Hardware`);
  lines.push(renderKvTable({
    "CPU logical": snapshot.hardware.cpu?.logicalCount,
    "CPU physical": snapshot.hardware.cpu?.physicalCount,
    "Memory total (GB)": snapshot.hardware.memory.totalGb,
    "Memory available (GB)": snapshot.hardware.memory.availableGb,
    "Swap used (GB)": snapshot.hardware.swap.usedGb,
    "GPU name": snapshot.hardware.gpu.gpuName,
    "VRAM total (GB)": snapshot.hardware.gpu.vramTotalGb,
    "Driver version": snapshot.hardware.gpu.driverVersion,
    "System CUDA": snapshot.hardware.gpu.systemCudaVersion,
    "nvidia-smi on PATH": snapshot.hardware.gpu.nvidiaSmiOnPath,
  }));
  if (snapshot.hardware.disks?.length) {
    lines.push("### Disks");
    for (const disk of snapshot.hardware.disks) {
      lines.push(`- \`${disk.mount}\` (${disk.fstype}): ${disk.freeGb} GB free / ${disk.totalGb} GB total`);
    }
    lines.push("");
  }

  lines.push(`## Python`);
  lines.push(renderKvTable({
    Executable: snapshot.python.executable,
    Version: snapshot.python.version,
    Implementation: snapshot.python.implementation,
    Prefix: snapshot.python.prefix,
    "Base prefix": snapshot.python.basePrefix,
    cwd: snapshot.python.cwd,
  }));

  lines.push(`## Runtime`);
  lines.push(renderKvTable({
    Engine: snapshot.runtime.engineLabel,
    "Loaded model": snapshot.runtime.loadedModel?.name ?? "none",
    "Warm pool": snapshot.runtime.warmPoolCount,
    "llama-server": snapshot.runtime.llamaServerPath,
    "llama-server-turbo": snapshot.runtime.llamaServerTurboPath,
    "llama-cli": snapshot.runtime.llamaCliPath,
  }));

  lines.push(`## GPU packages (find_spec)`);
  lines.push(renderKvTable({
    torch: snapshot.gpu.torchFindSpec,
    diffusers: snapshot.gpu.diffusersFindSpec,
    accelerate: snapshot.gpu.accelerateFindSpec,
    transformers: snapshot.gpu.transformersFindSpec,
    imageio: snapshot.gpu.imageioFindSpec,
    "imageio-ffmpeg": snapshot.gpu.ffmpegFindSpec,
    sentencepiece: snapshot.gpu.sentencepieceFindSpec,
    tiktoken: snapshot.gpu.tiktokenFindSpec,
    protobuf: snapshot.gpu.protobufFindSpec,
    ftfy: snapshot.gpu.ftfyFindSpec,
  }));

  if (snapshot.gpu.torchSubprocess) {
    lines.push(`## torch runtime (subprocess probe)`);
    lines.push(renderKvTable(snapshot.gpu.torchSubprocess));
  }

  lines.push(`## Extras (GPU install target)`);
  lines.push(renderKvTable({
    Path: snapshot.extras.path,
    Exists: snapshot.extras.exists,
    "Size (bytes)": snapshot.extras.sizeBytes,
    "Free (bytes)": snapshot.extras.freeBytes,
    Entries: snapshot.extras.topLevelEntries.join(", "),
  }));

  lines.push(`## Environment`);
  lines.push(renderKvTable(snapshot.environment as Record<string, unknown>));

  lines.push(`## Backend log tail (${snapshot.logs.tailLines.length} lines)`);
  if (snapshot.logs.path) {
    lines.push(`Path: \`${snapshot.logs.path}\``);
    lines.push("");
  }
  lines.push("```");
  lines.push(snapshot.logs.tailLines.join("\n") || "(no log content)");
  lines.push("```");

  return lines.join("\n");
}

function renderKvTable(data: Record<string, unknown>): string {
  const rows: string[] = ["| Key | Value |", "|---|---|"];
  for (const [key, value] of Object.entries(data)) {
    const display = value == null || value === "" ? "—" : String(value);
    // Escape pipes so markdown tables don't break.
    const escaped = display.replace(/\|/g, "\\|");
    rows.push(`| ${key} | ${escaped} |`);
  }
  rows.push("");
  return rows.join("\n");
}
