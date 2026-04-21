import { useEffect, useRef } from "react";
import type { GpuBundleJobState } from "../api";

interface InstallLogPanelProps {
  job: GpuBundleJobState | null;
}

// Single scrollable terminal rendering the GPU bundle install progress.
// Previously this was a stack of per-step <details> cards; they were
// OK on small installs but on the full 13-package bundle the whole
// panel ran off the bottom of the Studio and users lost their place
// when pip output streamed in mid-scroll. The user asked for a
// "fixed-width terminal with a step counter" — this is that.
//
// Auto-scrolls to the bottom whenever new attempts land, so you can
// leave it visible and watch the install tail like a ``tail -f``.

export function InstallLogPanel({ job }: InstallLogPanelProps) {
  const scrollRef = useRef<HTMLPreElement | null>(null);

  // Auto-scroll to the newest output whenever attempts grow. We don't
  // scroll on final-message updates (phase transitions) because those
  // can fire while the user is scrolled up reading earlier output;
  // yanking them back is disrespectful of their attention.
  const attemptCount = job?.attempts.length ?? 0;
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [attemptCount]);

  if (!job || job.phase === "idle") return null;
  const hasOutput = attemptCount > 0 || Boolean(job.message) || Boolean(job.error);
  if (!hasOutput) return null;

  const openByDefault = job.phase === "error" || Boolean(job.error);
  const stepLabel = formatStepCounter(job);
  const statusLabel = formatStatusLabel(job);

  return (
    <details className="install-log-panel" open={openByDefault}>
      <summary className="install-log-summary">{statusLabel}</summary>
      <div className="install-log-body">
        <InstallLogMeta job={job} />
        <div className="install-log-step-line">{stepLabel}</div>
        <pre ref={scrollRef} className="install-log-terminal">
          {renderTerminal(job)}
        </pre>
        {job.message && (job.phase === "done" || job.phase === "error") ? (
          <div className="install-log-final">
            <strong>Final status:</strong> {job.message}
          </div>
        ) : null}
      </div>
    </details>
  );
}

function InstallLogMeta({ job }: { job: GpuBundleJobState }) {
  // Line of context shown above the terminal. The target dir is
  // load-bearing: if the install appears to "succeed" but the app
  // still shows CPU, it's almost always because the backend wasn't
  // restarted (PYTHONPATH on the running process is fixed at spawn).
  const fragments: string[] = [];
  if (job.targetDir) fragments.push(`Target: ${job.targetDir}`);
  if (job.pythonVersion) fragments.push(`Python ${job.pythonVersion}`);
  if (job.indexUrlUsed) fragments.push(`CUDA index: ${job.indexUrlUsed}`);
  if (job.cudaVerified === true) fragments.push("CUDA verified");
  if (job.cudaVerified === false && job.phase === "done") fragments.push("CUDA verification failed");
  if (job.noWheelForPython) fragments.push("No CUDA wheel for this Python");
  if (fragments.length === 0) return null;
  return <div className="install-log-meta">{fragments.join(" · ")}</div>;
}

function formatStatusLabel(job: GpuBundleJobState): string {
  if (job.phase === "error" || job.error) return "Install failed — see log";
  if (job.phase === "done") return "Install complete — see log";
  if (job.phase === "preflight") return "Install starting…";
  if (job.phase === "verifying") return "Verifying CUDA…";
  return "Install in progress";
}

function formatStepCounter(job: GpuBundleJobState): string {
  // Packages-complete counter. The backend tracks packages via
  // packageIndex / packageTotal; torch also has a two-pass install
  // (CUDA-index walk for the wheel + dep-pass for transitive deps)
  // that fires in the same packageIndex=1 slot. We count finished
  // packages as attempts whose ok=true AND whose shape is a unique
  // package row (not the deps sub-pass or the cuda verify step).
  const done = job.attempts.filter(
    (a) => a.ok && a.phase !== "deps" && a.phase !== "verify",
  ).length;
  const total = job.packageTotal || Math.max(done, 1);
  const current = job.packageCurrent ?? "(waiting)";
  const percent = Math.max(0, Math.min(100, Math.round(job.percent)));
  if (job.phase === "error" || job.phase === "done") {
    return `Final: ${done}/${total} packages · ${percent}%`;
  }
  return `Step ${done}/${total}: ${current} · ${percent}%`;
}

function renderTerminal(job: GpuBundleJobState): string {
  // One big string of per-attempt sections, each prefixed with a
  // status marker so you can scan down the left edge for failures.
  // pip's own output is indented two spaces — keeps our marker visible.
  const lines: string[] = [];
  for (const attempt of job.attempts) {
    const marker = attempt.ok ? "[ OK ]" : "[FAIL]";
    lines.push(`${marker} ${attemptLabel(attempt)}`);
    if (attempt.output) {
      const body = filterPipNoise(attempt.output);
      if (body) {
        for (const bodyLine of body.split(/\r?\n/)) {
          lines.push(`       ${bodyLine}`);
        }
      }
    }
    lines.push(""); // blank line between attempts for legibility
  }
  if (job.phase !== "done" && job.phase !== "error") {
    const spinner = job.message || "working…";
    lines.push(`[....] ${spinner}`);
  }
  return lines.join("\n");
}

function attemptLabel(attempt: GpuBundleJobState["attempts"][number]): string {
  // Attempts from the worker come in four shapes:
  //   - torch CUDA swap: { indexUrl, ok, output }
  //   - torch deps pass: { indexUrl, phase: "deps", ok, output }
  //   - per-package pip: { package, ok, output }
  //   - cuda verify:     { phase: "verify", ok, output }
  if (attempt.phase === "verify") return "Verify torch.cuda.is_available()";
  if (attempt.phase === "deps" && attempt.indexUrl) return `torch deps (from ${attempt.indexUrl})`;
  if (attempt.indexUrl) return `torch (from ${attempt.indexUrl})`;
  if (attempt.package) return `pip install ${attempt.package}`;
  if (attempt.phase) return attempt.phase;
  return "step";
}

// Regexes that identify pip's dep-resolver warnings. These fire when
// something in the ambient env declares a dep that isn't satisfied —
// cosmetic for our case (user's .venv had turboquant-mlx-full from
// earlier testing, which declares an mlx>= constraint that will never
// be met on Windows). Surfacing them confused the user because they
// look like errors. We drop them from the DISPLAYED log but leave them
// intact in job.attempts[].output for support / backend debugging.
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
      // stay in the block through the detail lines
      if (isNoiseDetail) continue;
      // empty line after the block — end of it
      inNoiseBlock = false;
      continue;
    }
    inNoiseBlock = false;
    filtered.push(line);
  }
  // Keep only the tail — pip's download output can be thousands of
  // lines for torch. 80 lines is plenty to see the critical parts
  // (version resolved, Successfully installed …, any error stack).
  if (filtered.length > 80) {
    const kept = filtered.slice(-80);
    return `... (${filtered.length - 80} earlier lines omitted)\n${kept.join("\n")}`;
  }
  return filtered.join("\n");
}
