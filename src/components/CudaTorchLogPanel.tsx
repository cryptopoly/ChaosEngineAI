import { useEffect, useRef } from "react";
import type { CudaTorchInstallResult } from "../api";

// Collapsible terminal-style log for the inline "Install CUDA torch"
// action in Image / Video Studio. Mirrors the visual shape of
// InstallLogPanel (single scrollable <pre>, [ OK ]/[FAIL] markers per
// attempt, target-dir / Python meta line) but keyed off the
// CudaTorchInstallResult shape returned by /api/setup/install-cuda-torch
// rather than the GpuBundleJobState progress lifecycle. The endpoint
// is synchronous -- it walks cu124/cu126/cu128/cu121 in order and
// returns the full attempts array on completion -- so there's no
// streaming to drive an in-progress phase. We expose only the final
// result, but we still want the per-index pip output visible for
// debugging because users hitting "No CUDA wheel for this Python" or
// resolver clashes need to see which index failed and why.
//
// Collapsed by default on success; auto-opens on failure so the user
// doesn't have to click to find out what went wrong.

interface CudaTorchLogPanelProps {
  result: CudaTorchInstallResult | null;
}

export function CudaTorchLogPanel({ result }: CudaTorchLogPanelProps) {
  const scrollRef = useRef<HTMLPreElement | null>(null);
  const attemptCount = result?.attempts.length ?? 0;
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [attemptCount]);

  if (!result) return null;

  const openByDefault = !result.ok;
  const summary = result.ok
    ? `Install complete — see log${result.indexUrl ? ` (${shortIndex(result.indexUrl)})` : ""}`
    : `Install failed — see log${result.attempts.length > 0 ? ` (${result.attempts.length} attempt${result.attempts.length === 1 ? "" : "s"})` : ""}`;

  return (
    <details className="install-log-panel" open={openByDefault} style={{ marginTop: "0.5rem" }}>
      <summary className="install-log-summary">{summary}</summary>
      <div className="install-log-body">
        {renderMeta(result)}
        <pre ref={scrollRef} className="install-log-terminal">
          {renderTerminal(result)}
        </pre>
      </div>
    </details>
  );
}

function renderMeta(result: CudaTorchInstallResult): React.ReactNode {
  const fragments: string[] = [];
  if (result.targetDir) fragments.push(`Target: ${result.targetDir}`);
  if (result.pythonVersion) fragments.push(`Python ${result.pythonVersion}`);
  if (result.indexUrl) fragments.push(`CUDA index: ${result.indexUrl}`);
  if (result.noWheelForPython) fragments.push("No CUDA wheel for this Python");
  if (result.requiresRestart) fragments.push("Restart Backend to activate");
  if (fragments.length === 0) return null;
  return <div className="install-log-meta">{fragments.join(" · ")}</div>;
}

function renderTerminal(result: CudaTorchInstallResult): string {
  const lines: string[] = [];
  for (const attempt of result.attempts) {
    const marker = attempt.ok ? "[ OK ]" : "[FAIL]";
    lines.push(`${marker} torch (from ${attempt.indexUrl})`);
    if (attempt.output) {
      const body = filterPipNoise(attempt.output);
      if (body) {
        for (const bodyLine of body.split(/\r?\n/)) {
          lines.push(`       ${bodyLine}`);
        }
      }
    }
    lines.push("");
  }
  // Some failure modes (e.g. no extras dir resolvable) come back with
  // empty attempts but a populated top-level output -- show that so
  // users aren't staring at a blank panel.
  if (result.attempts.length === 0 && result.output) {
    const body = filterPipNoise(result.output);
    if (body) {
      for (const bodyLine of body.split(/\r?\n/)) {
        lines.push(bodyLine);
      }
    }
  }
  return lines.join("\n").trimEnd() || "(no output captured)";
}

function shortIndex(url: string): string {
  return url.replace("https://download.pytorch.org/whl/", "");
}

// Trim pip's noisy resolver complaints + cap the displayed log at the
// last 80 lines so the panel doesn't scroll to the bottom of the
// universe when torch downloads ~2.5 GB. Mirror of the helper in
// InstallLogPanel -- copied rather than shared so this panel has no
// runtime dependency on the GPU-bundle job shape.
const PIP_NOISE_PATTERNS = [
  /^ERROR: pip's dependency resolver does not currently take into account/i,
  /^\w[\w-]+\s+[\d.]+\s+requires\s+[\w-]+(?:[<>=!~].+)?, which is not installed\.$/i,
];

function filterPipNoise(output: string): string {
  const lines = output.split(/\r?\n/);
  const filtered: string[] = [];
  let inNoiseBlock = false;
  for (const line of lines) {
    const isNoiseHeader = PIP_NOISE_PATTERNS[0].test(line);
    const isNoiseDetail = PIP_NOISE_PATTERNS[1].test(line.trim());
    if (isNoiseHeader) {
      inNoiseBlock = true;
      continue;
    }
    if (inNoiseBlock && (isNoiseDetail || line.trim() === "")) {
      if (isNoiseDetail) continue;
      inNoiseBlock = false;
      continue;
    }
    inNoiseBlock = false;
    filtered.push(line);
  }
  if (filtered.length > 80) {
    const kept = filtered.slice(-80);
    return `... (${filtered.length - 80} earlier lines omitted)\n${kept.join("\n")}`;
  }
  return filtered.join("\n");
}
