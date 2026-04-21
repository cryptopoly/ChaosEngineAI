import { useCallback, useEffect, useState } from "react";
import { Panel } from "../../components/Panel";
import {
  fetchDiagnosticsSnapshot,
  reextractRuntime,
  type DiagnosticsSnapshot,
} from "../../api";

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
  const [snapshot, setSnapshot] = useState<DiagnosticsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<number | null>(null);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");
  const [reextractBusy, setReextractBusy] = useState(false);
  const [reextractMessage, setReextractMessage] = useState<string | null>(null);

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
      title="Diagnostics"
      subtitle="System snapshot for troubleshooting. Copy-paste the markdown to share with support when something misbehaves."
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

// ---- Sub-components -----------------------------------------------

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
