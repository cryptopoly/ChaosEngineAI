import { VideoToolsRow } from "../videotools/VideoToolsRow";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import {
  fetchDiagnosticsSnapshot,
  deleteHfRepo,
  deleteModelPath,
  fetchStorageChildren,
  fetchStorageTop,
  type DiagnosticsSnapshot,
  type StorageTopEntry,
  type StorageTopResponse,
} from "../../api";
import { AcceleratorsBoostPack } from "./AcceleratorsBoostPack";
import { DiagnosticsRecoveryActions } from "./DiagnosticsRecoveryActions";
import { DiagnosticsSnapshotView } from "./DiagnosticsSnapshotView";
import { LlamaCudaPanel } from "./LlamaCudaPanel";
import type { SystemStats } from "../../types/system";
import { WslBridgePanel } from "./WslBridgePanel";

// In-app troubleshooting panel. Surfaces OS, hardware, runtime paths,
// GPU state, env vars, and the backend log tail without asking users to
// run PowerShell commands. The Copy button formats everything as Markdown
// so users can paste a single block into a support thread.
//
// Design bias: observability first, actions sparingly. The read-only
// snapshot lives in DiagnosticsSnapshotView and the few recovery buttons
// in DiagnosticsRecoveryActions. Empty / missing values are rendered as
// "—" so users can distinguish "not applicable here" from "broken".
//
// Every label on screen goes through t() (diagnostics.json "panel").
// The copied markdown stays in English on purpose: it is read by
// support, not by the user.

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

  const copyLabel = copyStatus === "copied"
    ? t("panel.copied", { defaultValue: "Copied!" })
    : copyStatus === "failed"
      ? t("panel.copyFailed", { defaultValue: "Copy failed" })
      : t("panel.copy", { defaultValue: "Copy diagnostics" });

  return (
    <Panel
      className="panel-scroll"
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
            {loading
              ? t("panel.refreshing", { defaultValue: "Refreshing..." })
              : t("panel.refresh", { defaultValue: "Refresh" })}
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
          {t("panel.fetchError", { defaultValue: "Could not fetch diagnostics: {error}", error })}
        </p>
      ) : null}

      {!snapshot && loading ? (
        <p className="muted-text">{t("panel.gathering", { defaultValue: "Gathering system info..." })}</p>
      ) : null}

      {!snapshot && !loading && !error ? (
        <p className="muted-text">
          {backendOnline
            ? t("panel.clickRefresh", { defaultValue: "Click Refresh to fetch diagnostics." })
            : t("panel.backendOffline", { defaultValue: "Backend offline — start the sidecar first." })}
        </p>
      ) : null}

      {snapshot ? (
        <div className="diagnostics-body">
          <DiagnosticsSnapshotView snapshot={snapshot} lastFetchedAt={lastFetchedAt} />
          <DiagnosticsRecoveryActions
            os={snapshot.os}
            backendOnline={backendOnline}
            onRestartServer={onRestartServer}
            busyAction={busyAction}
            onRefresh={refresh}
          />
        </div>
      ) : null}
    </Panel>
  );
}

// Settings > Accelerators. The Boost Pack and the WSL bridge used to sit at
// the bottom of the Diagnostics panel, below the system snapshot; refreshing
// the snapshot grew that panel and pushed them off screen with no way to scroll
// to them, so they now have their own tab.
export function AcceleratorsPanel({
  backendOnline,
  hostSystem,
}: {
  backendOnline: boolean;
  hostSystem?: Pick<SystemStats, "platform" | "arch" | "gpuVramTotalGb">;
}) {
  const { t } = useTranslation("diagnostics");
  return (
    <Panel
      className="panel-scroll"
      title={t("accelerators.title", { defaultValue: "Accelerators" })}
      subtitle={t("accelerators.subtitle", {
        defaultValue: "Optional speed-ups for image, video and chat inference, and the Windows WSL bridge.",
      })}
    >
      <VideoToolsRow backendOnline={backendOnline} />
      <AcceleratorsBoostPack backendOnline={backendOnline} hostSystem={hostSystem} />
      <LlamaCudaPanel backendOnline={backendOnline} />
      <WslBridgePanel backendOnline={backendOnline} />
    </Panel>
  );
}

// Settings > Disk usage (FU-055), for the same reason.
export function DiskUsagePanel({ backendOnline }: { backendOnline: boolean }) {
  const { t } = useTranslation("diagnostics");
  return (
    <Panel
      className="panel-scroll"
      title={t("diskUsage.title", { defaultValue: "Disk usage" })}
      subtitle={t("diskUsage.subtitle", { defaultValue: "Which model repos take the space, so you know what to delete." })}
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
  const { t } = useTranslation("diagnostics");
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
      <h3>{t("diskUsage.topTitle", { defaultValue: "Top 20 model repos" })}</h3>
      <p className="muted-text" style={{ marginBottom: 8 }}>
        {t("diskUsage.topHint", {
          defaultValue:
            "Walks every enabled model directory and lists the biggest repos on disk. Counts blob bytes once (Hugging Face's snapshots/ are symlinks to blobs/ so naive scanners over-count).",
        })}
      </p>
      <button
        className="secondary-button"
        type="button"
        onClick={() => void scan()}
        disabled={!backendOnline || busy}
      >
        {busy
          ? t("diskUsage.scanning", { defaultValue: "Scanning…" })
          : data
            ? t("diskUsage.rescan", { defaultValue: "Rescan" })
            : t("diskUsage.scan", { defaultValue: "Scan disk usage" })}
      </button>
      {err ? <p className="muted-text" style={{ color: "#f87171", marginTop: 8 }}>{err}</p> : null}
      {data ? (
        <div style={{ marginTop: 12 }}>
          <p className="muted-text">
            {t("diskUsage.total", { defaultValue: "Total across all enabled directories:" })}{" "}
            <strong>{data.totalGb !== null ? `${data.totalGb.toFixed(1)} GB` : "—"}</strong>
            {" · "}
            {t("diskUsage.shown", {
              defaultValue: "{repos} repos shown / scanned {dirs} root dir(s)",
              repos: data.entries.length,
              dirs: data.scannedDirectories.length,
            })}
            {freedBytes > 0 ? (
              <>
                {" · "}
                <strong>
                  {t("diskUsage.freed", { defaultValue: "{gb} GB freed since scan", gb: (freedBytes / 1e9).toFixed(1) })}
                </strong>
              </>
            ) : null}
          </p>
          <div className="storage-top-table">
            <div className="storage-top-head">
              <span>{t("diskUsage.colRepo", { defaultValue: "Repo" })}</span>
              <span>{t("diskUsage.colSize", { defaultValue: "Size" })}</span>
              <span>{t("diskUsage.colSource", { defaultValue: "Source" })}</span>
              <span>{t("diskUsage.colPath", { defaultValue: "Path" })}</span>
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
  const { t } = useTranslation("diagnostics");
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
    const size = entry.sizeGb !== null
      ? `${entry.sizeGb.toFixed(1)} GB`
      : t("diskUsage.unknownSize", { defaultValue: "unknown size" });
    const confirmed = window.confirm(
      t("diskUsage.deleteConfirm", {
        defaultValue: "Permanently delete {name} ({size})?\n\nThis removes it from disk and cannot be undone.",
        name: entry.hfRepo ?? entry.path,
        size,
      }),
    );
    if (!confirmed) return;
    setDeleting(true);
    setErr(null);
    try {
      if (entry.hfRepo) await deleteHfRepo(entry.hfRepo);
      else await deleteModelPath(entry.path);
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
              aria-label={open
                ? t("diskUsage.collapse", { defaultValue: "Collapse {name}", name: entry.repoLabel })
                : t("diskUsage.expand", { defaultValue: "Expand {name}", name: entry.repoLabel })}
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
          <span
            className="muted-text"
            title={t("diskUsage.sharedTitle", { defaultValue: "Shared Hugging Face store: many repos link into it" })}
          >
            {t("diskUsage.shared", { defaultValue: "shared" })}
          </span>
        ) : (
          <button
            type="button"
            className="secondary-button storage-top-delete"
            onClick={() => void remove()}
            disabled={deleting}
          >
            {deleting
              ? t("diskUsage.deleting", { defaultValue: "Deleting…" })
              : t("diskUsage.delete", { defaultValue: "Delete" })}
          </button>
        )}
      </div>
      {open && loading ? (
        <div className="storage-top-row muted-text">{t("diskUsage.reading", { defaultValue: "Reading…" })}</div>
      ) : null}
      {open && err ? <div className="storage-top-row" style={{ color: "#f87171" }}>{err}</div> : null}
      {open && children?.map((child) => (
        <StorageRow key={child.path} entry={child} depth={depth + 1} onDeleted={onDeleted} />
      ))}
      {open && children?.length === 0 ? (
        <div className="storage-top-row muted-text" style={{ paddingLeft: 12 + (depth + 1) * 16 }}>
          {t("diskUsage.empty", { defaultValue: "Empty" })}
        </div>
      ) : null}
    </>
  );
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
