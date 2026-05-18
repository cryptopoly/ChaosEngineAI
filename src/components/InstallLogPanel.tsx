import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { GpuBundleJobState, LongLiveJobState, MtplxJobState, VllmWslJobState } from "../api";

// The panel renders any background install job — GPU bundle, LongLive,
// MTPLX, or WSL vLLM. All share the core fields (phase / message /
// attempts / progress counters / targetDir). Treating the prop as a
// union keeps all surfaces using one component without duplicating
// auto-scroll, pip-noise filter, and terminal layout.
export type InstallJobState =
  | GpuBundleJobState
  | LongLiveJobState
  | MtplxJobState
  | VllmWslJobState;

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
  variant?: "gpu-bundle" | "longlive" | "mtplx" | "vllm-wsl";
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
  const { t } = useTranslation("setup");
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
  const stepLabel = formatStepCounter(job, t);
  const statusLabel = formatStatusLabel(job, variant, t);

  return (
    <details className="install-log-panel" open={openByDefault}>
      <summary className="install-log-summary">{statusLabel}</summary>
      <div className="install-log-body">
        <InstallLogMeta job={job} t={t} />
        <div className="install-log-step-line">{stepLabel}</div>
        <pre ref={scrollRef} className="install-log-terminal">
          {renderTerminal(job, t)}
        </pre>
        {job.message && (job.phase === "done" || job.phase === "error") ? (
          <div className="install-log-final">
            <strong>{t("installLog.finalStatus", { defaultValue: "Final status:" })}</strong> {job.message}
          </div>
        ) : null}
      </div>
    </details>
  );
}

function InstallLogMeta({ job, t }: { job: InstallJobState; t: TFunction }) {
  // Line of context shown above the terminal. The target dir is
  // load-bearing: if the install appears to "succeed" but the app
  // still shows CPU, it's almost always because the backend wasn't
  // restarted (PYTHONPATH on the running process is fixed at spawn).
  const fragments: string[] = [];
  if (job.targetDir) fragments.push(t("installLog.meta.target", { dir: job.targetDir, defaultValue: `Target: ${job.targetDir}` }));
  // GPU-bundle-only fields. Reading via a typed-narrowed alias keeps
  // both job shapes flowing through this component without runtime
  // ``in`` checks per field.
  const meta = job as InstallJobState & InstallJobMetaFields;
  if (meta.pythonVersion) fragments.push(t("installLog.meta.python", { version: meta.pythonVersion, defaultValue: `Python ${meta.pythonVersion}` }));
  if (meta.indexUrlUsed) fragments.push(t("installLog.meta.cudaIndex", { url: meta.indexUrlUsed, defaultValue: `CUDA index: ${meta.indexUrlUsed}` }));
  if (meta.cudaVerified === true) fragments.push(t("installLog.meta.cudaVerified", { defaultValue: "CUDA verified" }));
  if (meta.cudaVerified === false && job.phase === "done") fragments.push(t("installLog.meta.cudaVerificationFailed", { defaultValue: "CUDA verification failed" }));
  if (meta.noWheelForPython) fragments.push(t("installLog.meta.noWheelForPython", { defaultValue: "No CUDA wheel for this Python" }));
  if (fragments.length === 0) return null;
  return <div className="install-log-meta">{fragments.join(" · ")}</div>;
}

function formatStatusLabel(job: InstallJobState, variant: "gpu-bundle" | "longlive" | "mtplx" | "vllm-wsl", t: TFunction): string {
  const noun = variant === "longlive"
    ? t("installLog.statusNoun.longlive", { defaultValue: "LongLive install" })
    : variant === "mtplx"
    ? t("installLog.statusNoun.mtplx", { defaultValue: "MTPLX install" })
    : variant === "vllm-wsl"
    ? t("installLog.statusNoun.vllmWsl", { defaultValue: "vLLM-in-WSL install" })
    : t("installLog.statusNoun.gpuBundle", { defaultValue: "Install" });
  if (job.phase === "error" || job.error) return t("installLog.status.failed", { noun, defaultValue: `${noun} failed — see log` });
  if (job.phase === "done") return t("installLog.status.complete", { noun, defaultValue: `${noun} complete — see log` });
  if (job.phase === "preflight") return t("installLog.status.starting", { noun, defaultValue: `${noun} starting…` });
  if (job.phase === "verifying") return t("installLog.status.verifyingCuda", { defaultValue: "Verifying CUDA…" });
  return t("installLog.status.inProgress", { noun, defaultValue: `${noun} in progress` });
}

function formatStepCounter(job: InstallJobState, t: TFunction): string {
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
  const current = job.packageCurrent ?? t("installLog.step.waiting", { defaultValue: "(waiting)" });
  const percent = Math.max(0, Math.min(100, Math.round(job.percent)));
  if (job.phase === "error" || job.phase === "done") {
    return t("installLog.step.final", { done, total, percent, defaultValue: `Final: ${done}/${total} packages · ${percent}%` });
  }
  return t("installLog.step.current", { done, total, current, percent, defaultValue: `Step ${done}/${total}: ${current} · ${percent}%` });
}

function renderTerminal(job: InstallJobState, t: TFunction): string {
  // One big string of per-attempt sections, each prefixed with a
  // status marker so you can scan down the left edge for failures.
  // pip's own output is indented two spaces — keeps our marker visible.
  const lines: string[] = [];
  for (const attempt of job.attempts) {
    const marker = attempt.ok ? "[ OK ]" : "[FAIL]";
    lines.push(`${marker} ${attemptLabel(attempt, t)}`);
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
    const spinner = job.message || t("installLog.terminal.working", { defaultValue: "working…" });
    lines.push(`[....] ${spinner}`);
  }
  return lines.join("\n");
}

function attemptLabel(attempt: InstallJobState["attempts"][number], t: TFunction): string {
  // Attempts from the worker come in four shapes:
  //   - torch CUDA swap: { indexUrl, ok, output }
  //   - torch deps pass: { indexUrl, phase: "deps", ok, output }
  //   - per-package pip: { package, ok, output }
  //   - cuda verify:     { phase: "verify", ok, output }
  if (attempt.phase === "verify") return t("installLog.attempt.verifyCuda", { defaultValue: "Verify torch.cuda.is_available()" });
  if (attempt.phase === "deps" && attempt.indexUrl) return t("installLog.attempt.torchDeps", { url: attempt.indexUrl, defaultValue: `torch deps (from ${attempt.indexUrl})` });
  if (attempt.phase === "torch-cleanup") return t("installLog.attempt.torchCleanup", { defaultValue: "Clean stale torch files" });
  if (attempt.phase === "torch-repair") return t("installLog.attempt.torchRepair", { defaultValue: "Repair CUDA torch wheel" });
  if (attempt.phase === "constraint") return t("installLog.attempt.constraint", { defaultValue: "Pin torch version" });
  if (attempt.indexUrl) return t("installLog.attempt.torchFromIndex", { url: attempt.indexUrl, defaultValue: `torch (from ${attempt.indexUrl})` });
  if (attempt.package) return t("installLog.attempt.pipInstall", { pkg: attempt.package, defaultValue: `pip install ${attempt.package}` });
  if (attempt.phase) return attempt.phase;
  return t("installLog.attempt.step", { defaultValue: "step" });
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
