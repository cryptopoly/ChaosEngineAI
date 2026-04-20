import type { GpuBundleJobState } from "../api";

interface InstallLogPanelProps {
  job: GpuBundleJobState | null;
}

// Terminal-like expandable panel that renders the per-step output of the
// GPU bundle install. Sits under the "Install GPU runtime" button in both
// Image Studio and Video Studio so a failed install surfaces the real pip
// output (which CUDA index was tried, which package crashed, which
// ``No matching distribution`` line blew things up) instead of collapsing
// into a single useless toast.
//
// Auto-expands on error so the user doesn't have to click to see what
// went wrong. On success it stays collapsed so the UI doesn't get noisy.
export function InstallLogPanel({ job }: InstallLogPanelProps) {
  if (!job || job.phase === "idle") return null;
  const hasOutput = job.attempts.length > 0 || Boolean(job.message) || Boolean(job.error);
  if (!hasOutput) return null;

  const openByDefault = job.phase === "error" || Boolean(job.error);
  const attemptCount = job.attempts.length;
  const summaryLabel = (() => {
    if (job.phase === "error" || job.error) return "Install failed — see log";
    if (job.phase === "done") return `Install log (${attemptCount} step${attemptCount === 1 ? "" : "s"})`;
    if (job.phase === "downloading" || job.phase === "preflight") return `Install in progress (${attemptCount} step${attemptCount === 1 ? "" : "s"} so far)`;
    if (job.phase === "verifying") return "Verifying — install log";
    return `Install log (${attemptCount} step${attemptCount === 1 ? "" : "s"})`;
  })();

  return (
    <details className="install-log-panel" open={openByDefault}>
      <summary className="install-log-summary">{summaryLabel}</summary>
      <div className="install-log-body">
        <InstallLogMeta job={job} />
        {job.attempts.length === 0 ? (
          <div className="install-log-empty">No steps recorded yet — worker is still setting up.</div>
        ) : (
          <ol className="install-log-attempts">
            {job.attempts.map((attempt, index) => (
              <InstallLogAttempt
                key={`${attempt.package ?? attempt.indexUrl ?? attempt.phase ?? "step"}-${index}`}
                attempt={attempt}
              />
            ))}
          </ol>
        )}
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
  // Line of context shown above the per-step log — everything the user
  // might want to paste into a support thread. The target dir in
  // particular is load-bearing: if the install appears to "succeed" but
  // the app still shows CPU, it's almost always because they restarted
  // without closing, which skips the PYTHONPATH reload.
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

function InstallLogAttempt({ attempt }: { attempt: GpuBundleJobState["attempts"][number] }) {
  const label = attemptLabel(attempt);
  const status = attempt.ok ? "ok" : "fail";
  const statusText = attempt.ok ? "OK" : "FAIL";
  return (
    <li className={`install-log-attempt install-log-attempt--${status}`}>
      <div className="install-log-attempt-header">
        <span className={`install-log-status install-log-status--${status}`}>{statusText}</span>
        <span className="install-log-attempt-label">{label}</span>
      </div>
      {attempt.output ? (
        <pre className="install-log-output">{trimOutput(attempt.output)}</pre>
      ) : null}
    </li>
  );
}

function attemptLabel(attempt: GpuBundleJobState["attempts"][number]): string {
  // Attempts from the worker come in three shapes:
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

function trimOutput(output: string): string {
  // Pip outputs can be thousands of lines — the backend already truncates
  // to the last 2000 chars but that still can render as a wall of text.
  // Keep the last ~60 lines which is plenty to see the actual error.
  const lines = output.split(/\r?\n/);
  if (lines.length <= 60) return output;
  const kept = lines.slice(-60);
  return `... (${lines.length - 60} earlier lines omitted)\n${kept.join("\n")}`;
}
