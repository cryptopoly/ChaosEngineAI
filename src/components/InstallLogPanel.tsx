import { useEffect, useRef } from "react";
import type { GpuBundleJobState, LongLiveJobState } from "../api";

// The panel renders either kind of install job — GPU bundle or LongLive.
// They share the core fields (phase / message / attempts / progress
// counters / targetDir) and differ only in optional metadata. Treating
// the prop as a union keeps both Studio surfaces using one component
// instead of duplicating the auto-scroll, pip-noise filter, and
// terminal layout.
export type InstallJobState = GpuBundleJobState | LongLiveJobState;

// Optional fields read by the meta line. ``GpuBundleJobState`` has these;
// ``LongLiveJobState`` doesn't. Centralised here so the meta renderer
// can pluck whichever subset is present without a ``in`` ladder at the
// call site.
interface InstallJobMetaFields {
  pythonVersion?: string | null;
  indexUrlUsed?: string | null;
  cudaVerified?: boolean | null;
  noWheelForPython?: boolean;
}

interface InstallLogPanelProps {
  job: InstallJobState | null;
  // Title shown in the collapsed summary. Defaults to the GPU bundle
  // wording so existing call sites don't need to pass it.
  variant?: "gpu-bundle" | "longlive";
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

export function InstallLogPanel({ job, variant = "gpu-bundle" }: InstallLogPanelProps) {
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
  const statusLabel = formatStatusLabel(job, variant);

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

function InstallLogMeta({ job }: { job: InstallJobState }) {
  // Line of context shown above the terminal. The target dir is
  // load-bearing: if the install appears to "succeed" but the app
  // still shows CPU, it's almost always because the backend wasn't
  // restarted (PYTHONPATH on the running process is fixed at spawn).
  const fragments: string[] = [];
  if (job.targetDir) fragments.push(`Target: ${job.targetDir}`);
  // GPU-bundle-only fields. Reading via a typed-narrowed alias keeps
  // both job shapes flowing through this component without runtime
  // ``in`` checks per field.
  const meta = job as InstallJobState & InstallJobMetaFields;
  if (meta.pythonVersion) fragments.push(`Python ${meta.pythonVersion}`);
  if (meta.indexUrlUsed) fragments.push(`CUDA index: ${meta.indexUrlUsed}`);
  if (meta.cudaVerified === true) fragments.push("CUDA verified");
  if (meta.cudaVerified === false && job.phase === "done") fragments.push("CUDA verification failed");
  if (meta.noWheelForPython) fragments.push("No CUDA wheel for this Python");
  if (fragments.length === 0) return null;
  return <div className="install-log-meta">{fragments.join(" · ")}</div>;
}

function formatStatusLabel(job: InstallJobState, variant: "gpu-bundle" | "longlive"): string {
  const noun = variant === "longlive" ? "LongLive install" : "Install";
  if (job.phase === "error" || job.error) return `${noun} failed — see log`;
  if (job.phase === "done") return `${noun} complete — see log`;
  if (job.phase === "preflight") return `${noun} starting…`;
  if (job.phase === "verifying") return "Verifying CUDA…";
  return `${noun} in progress`;
}

function formatStepCounter(job: InstallJobState): string {
  // Packages-complete counter. The backend tracks packages via
  // packageIndex / packageTotal; torch also has a two-pass install
  // (CUDA-index walk for the wheel + dep-pass for transitive deps)
  // that fires in the same packageIndex=1 slot. Count logical packages,
  // not attempt rows, so cleanup / constraint / repair / verify entries
  // can show in the terminal without inflating "Final: n/n packages".
  const nonPackagePhases = new Set([
    "constraint",
    "deps",
    "torch-cleanup",
    "torch-repair",
    "verify",
  ]);
  const packagesDone = new Set<string>();
  let phaseStepsDone = 0;
  for (const attempt of job.attempts) {
    if (!attempt.ok || nonPackagePhases.has(attempt.phase ?? "")) continue;
    if (attempt.package) {
      packagesDone.add(attempt.package);
    } else if (attempt.indexUrl) {
      packagesDone.add("torch");
    } else if (attempt.phase) {
      phaseStepsDone += 1;
    }
  }
  const done = packagesDone.size > 0 ? packagesDone.size : phaseStepsDone;
  const total = Math.max(job.packageTotal || 0, done, 1);
  const current = job.packageCurrent ?? "(waiting)";
  const percent = Math.max(0, Math.min(100, Math.round(job.percent)));
  if (job.phase === "error" || job.phase === "done") {
    return `Final: ${done}/${total} packages · ${percent}%`;
  }
  return `Step ${done}/${total}: ${current} · ${percent}%`;
}

function renderTerminal(job: InstallJobState): string {
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

function attemptLabel(attempt: InstallJobState["attempts"][number]): string {
  // Attempts from the worker come in four shapes:
  //   - torch CUDA swap: { indexUrl, ok, output }
  //   - torch deps pass: { indexUrl, phase: "deps", ok, output }
  //   - per-package pip: { package, ok, output }
  //   - cuda verify:     { phase: "verify", ok, output }
  if (attempt.phase === "verify") return "Verify torch.cuda.is_available()";
  if (attempt.phase === "deps" && attempt.indexUrl) return `torch deps (from ${attempt.indexUrl})`;
  if (attempt.phase === "torch-cleanup") return "Clean stale torch files";
  if (attempt.phase === "torch-repair") return "Repair CUDA torch wheel";
  if (attempt.phase === "constraint") return "Pin torch version";
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
