import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import {
  getLlamaCudaInstallStatus,
  getLlamaCudaStatus,
  startLlamaCudaInstall,
} from "../../api";
import type { LlamaCudaJobState, LlamaCudaStatus } from "../../api";

/**
 * Settings > Accelerators: install the NVIDIA CUDA build of llama-server on
 * Linux (x86_64 and ARM64, e.g. DGX Spark). Shown only when the backend sees
 * a Linux host with a working NVIDIA driver; the bundled build stays in use
 * until the CUDA build lists a CUDA device.
 */
export function LlamaCudaPanel({ backendOnline }: { backendOnline: boolean }) {
  const { t } = useTranslation("common");
  const [status, setStatus] = useState<LlamaCudaStatus | null>(null);
  const [job, setJob] = useState<LlamaCudaJobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getLlamaCudaStatus());
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    if (backendOnline) void refresh();
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, [backendOnline, refresh]);

  const poll = useCallback(() => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = window.setInterval(async () => {
      try {
        const next = await getLlamaCudaInstallStatus();
        setJob(next);
        if (next.done) {
          if (timer.current !== null) window.clearInterval(timer.current);
          timer.current = null;
          await refresh();
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }, 1500);
  }, [refresh]);

  const install = useCallback(async () => {
    setError(null);
    try {
      setJob(await startLlamaCudaInstall());
      poll();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [poll]);

  if (!backendOnline || !status?.supported) return null;

  const running = job !== null && !job.done;
  const state = status.inUse
    ? t("llamaCuda.inUse", { defaultValue: "In use: GGUF models run on the NVIDIA GPU." })
    : status.installed
      ? t("llamaCuda.installedNoDevice", {
          defaultValue: "Installed, but it found no CUDA device. Check the NVIDIA driver (nvidia-smi).",
        })
      : t("llamaCuda.notInstalled", {
          defaultValue: "Not installed. GGUF models use the bundled build.",
        });

  return (
    <section className="accelerators-boost-pack" style={{ marginTop: 18 }}>
      <header className="accelerators-boost-pack-header">
        <h3 style={{ margin: 0, fontSize: "0.98rem", fontWeight: 600 }}>
          {t("llamaCuda.title", { defaultValue: "CUDA llama-server (NVIDIA)" })}
        </h3>
        <p className="muted-text" style={{ margin: "4px 0 0", fontSize: "0.84rem" }}>
          {t("llamaCuda.description", {
            cuda: status.cudaVersion ?? "",
            tag: status.tag,
            defaultValue:
              "Runs GGUF chat models on your NVIDIA GPU with CUDA {cuda} (llama.cpp {tag}). Downloads about 680 MB, including the CUDA runtime (0.9 GB on disk).",
          })}
        </p>
      </header>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 10 }}>
        <span className="muted-text" style={{ flex: 1, fontSize: "0.84rem" }}>
          {running ? `${job?.message || ""} ${Math.round(job?.percent ?? 0)}%` : state}
        </span>
        {!status.inUse ? (
          <button type="button" className="secondary-button" disabled={running} onClick={() => void install()}>
            {status.installed
              ? t("llamaCuda.reinstall", { defaultValue: "Reinstall" })
              : t("llamaCuda.install", { defaultValue: "Install" })}
          </button>
        ) : null}
      </div>
      {job?.done && job.phase === "done" ? (
        <p className="muted-text" style={{ margin: "6px 0 0", fontSize: "0.82rem" }}>{job.message}</p>
      ) : null}
      {error || job?.error ? (
        <p className="muted-text" style={{ color: "rgb(252, 165, 165)", margin: "6px 0 0", fontSize: "0.82rem" }}>
          {error || job?.error}
        </p>
      ) : null}
    </section>
  );
}
