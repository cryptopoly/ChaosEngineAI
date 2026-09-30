import { useEffect, useState } from "react";
import { Panel } from "../../components/Panel";
import { installPipPackage, downloadModel, getDownloadStatus, startKokoroDownload, getKokoroDownloadStatus } from "../../api";
import type { DownloadStatus, KokoroDownloadStatus } from "../../api";
import type { VoiceRuntime, SttModel, TtsVoice } from "../../types";

export interface VoiceModelsTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  onRefreshVoiceRuntime: () => void;
}

function InstallButton({
  packageKeys,
  label,
  onDone,
}: {
  packageKeys: string[];
  label: string;
  onDone: () => void;
}) {
  const [state, setState] = useState<"idle" | "installing" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function handleInstall() {
    setState("installing");
    setErrorMsg("");
    try {
      for (const packageKey of packageKeys) {
        const result = await installPipPackage(packageKey);
        if (!result.ok) {
          setErrorMsg(`${packageKey}: ${result.output?.slice(0, 200) ?? "Install failed"}`);
          setState("error");
          return;
        }
      }
      onDone();
    } catch (e) {
      setErrorMsg(String(e));
      setState("error");
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <button
        className="action-btn primary"
        onClick={handleInstall}
        disabled={state === "installing"}
        style={{ alignSelf: "flex-start" }}
      >
        {state === "installing" ? `Installing ${label}…` : `Install ${label}`}
      </button>
      {state === "error" && (
        <p className="muted-text" style={{ fontSize: "0.75rem", color: "var(--color-error)" }}>
          {errorMsg}
        </p>
      )}
    </div>
  );
}

/** Downloadable-model card shared by the STT model grid and the TTS
 * voice-model row — Download button, progress poll, Installed badge. */
function ModelDownloadCard({ model, onInstalled }: { model: SttModel; onInstalled: () => void }) {
  const [status, setStatus] = useState<DownloadStatus | null>(null);

  useEffect(() => {
    if (status?.state !== "downloading") return;
    const interval = window.setInterval(() => {
      void getDownloadStatus().then((all) => {
        const mine = all.find((d) => d.repo === model.id);
        if (!mine) return;
        setStatus(mine);
        if (mine.state === "completed") onInstalled();
      });
    }, 2000);
    return () => window.clearInterval(interval);
  }, [status, model.id, onInstalled]);

  async function handleDownload() {
    try {
      const result = await downloadModel(model.id);
      setStatus(result);
    } catch (e) {
      setStatus({
        repo: model.id,
        state: "failed",
        progress: 0,
        downloadedGb: 0,
        totalGb: null,
        error: String(e),
      });
    }
  }

  const downloading = status?.state === "downloading";

  return (
    <article className="image-library-card">
      <div className="image-library-card-head">
        <div>
          <h3 style={{ fontSize: "0.9rem" }}>{model.name}</h3>
          <p className="muted-text" style={{ fontSize: "0.75rem" }}>{model.id}</p>
        </div>
        <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
          {model.default && <span className="badge subtle">Default</span>}
          {model.installed ? (
            <span className="badge success">Installed</span>
          ) : (
            <span className="badge muted">Not installed</span>
          )}
        </div>
      </div>
      <div className="image-library-stats" style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span>{model.sizeGb} GB</span>
        {!model.installed && (
          downloading ? (
            <span className="muted-text">{Math.round((status?.progress ?? 0) * 100)}%</span>
          ) : (
            <button className="action-btn" onClick={handleDownload}>Download</button>
          )
        )}
      </div>
      {status?.state === "failed" && (
        <p className="muted-text" style={{ fontSize: "0.75rem", color: "var(--color-error)" }}>
          {status.error}
        </p>
      )}
    </article>
  );
}

/** Download card for the kokoro-onnx GitHub-release voice files
 * (non-Apple TTS lane) — same look as ModelDownloadCard but polls the
 * dedicated /api/voice/kokoro endpoints instead of the HF machinery. */
function KokoroOnnxCard({ installed, onInstalled }: { installed: boolean; onInstalled: () => void }) {
  const [status, setStatus] = useState<KokoroDownloadStatus | null>(null);

  useEffect(() => {
    if (status?.state !== "downloading") return;
    const interval = window.setInterval(() => {
      void getKokoroDownloadStatus().then((next) => {
        setStatus(next);
        if (next.state === "completed") onInstalled();
      }).catch(() => {});
    }, 2000);
    return () => window.clearInterval(interval);
  }, [status, onInstalled]);

  async function handleDownload() {
    try {
      setStatus(await startKokoroDownload());
    } catch (e) {
      setStatus({ state: "failed", progress: 0, error: String(e) });
    }
  }

  const downloading = status?.state === "downloading";

  return (
    <article className="image-library-card">
      <div className="image-library-card-head">
        <div>
          <h3 style={{ fontSize: "0.9rem" }}>Kokoro ONNX voice files</h3>
          <p className="muted-text" style={{ fontSize: "0.75rem" }}>
            kokoro-v1.0.onnx + voices-v1.0.bin
          </p>
        </div>
        {installed ? (
          <span className="badge success">Installed</span>
        ) : (
          <span className="badge muted">Not installed</span>
        )}
      </div>
      <div className="image-library-stats" style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span>0.34 GB</span>
        {!installed && (
          downloading ? (
            <span className="muted-text">{Math.round((status?.progress ?? 0) * 100)}%</span>
          ) : (
            <button className="action-btn" onClick={handleDownload}>Download</button>
          )
        )}
      </div>
      {status?.state === "failed" && (
        <p className="muted-text" style={{ fontSize: "0.75rem", color: "var(--color-error)" }}>
          {status.error}
        </p>
      )}
    </article>
  );
}

export function VoiceModelsTab({ voiceRuntime, backendOnline, onRefreshVoiceRuntime }: VoiceModelsTabProps) {
  const sttModels = voiceRuntime?.sttModels ?? [];
  const ttsVoices = voiceRuntime?.ttsVoices ?? [];
  const sttBackend = voiceRuntime?.sttBackend ?? null;
  const ttsBackend = voiceRuntime?.ttsBackend ?? null;
  const sttInstallPackages = voiceRuntime?.sttInstallPackages ?? null;
  const ttsInstallPackages = voiceRuntime?.ttsInstallPackages ?? null;

  return (
    <div className="content-grid image-page-grid">
      {/* ── STT Models ──────────────────────────────────────────────────── */}
      <Panel
        title="Speech-to-Text Models"
        subtitle={sttBackend ? `Backend: ${sttBackend}` : "No STT backend detected"}
        className="span-2"
      >
        {!backendOnline ? (
          <div className="empty-state">
            <p className="muted-text">Backend offline — connect to see model status.</p>
          </div>
        ) : sttInstallPackages ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <p className="muted-text">
              STT backend not installed. Click below to install the platform-appropriate package.
            </p>
            <InstallButton
              packageKeys={sttInstallPackages}
              label={sttInstallPackages.join(" + ")}
              onDone={onRefreshVoiceRuntime}
            />
          </div>
        ) : (
          <div className="image-library-grid">
            {sttModels.map((model: SttModel) => (
              <ModelDownloadCard key={model.id} model={model} onInstalled={onRefreshVoiceRuntime} />
            ))}
          </div>
        )}
      </Panel>

      {/* ── TTS Engine ──────────────────────────────────────────────────── */}
      <Panel
        title="Text-to-Speech Engine"
        subtitle={ttsBackend ? `Backend: ${ttsBackend}` : "No TTS backend detected"}
        className="span-2"
      >
        {!backendOnline ? (
          <div className="empty-state">
            <p className="muted-text">Backend offline — connect to see TTS status.</p>
          </div>
        ) : ttsInstallPackages ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <p className="muted-text">
              TTS backend not installed. Click below to install the platform-appropriate package.
            </p>
            <InstallButton
              packageKeys={ttsInstallPackages}
              label={ttsInstallPackages.join(" + ")}
              onDone={onRefreshVoiceRuntime}
            />
          </div>
        ) : (
          <>
            {voiceRuntime?.ttsModelRepo ? (
              <div className="image-library-grid" style={{ marginBottom: 12 }}>
                <ModelDownloadCard
                  model={{
                    id: voiceRuntime.ttsModelRepo,
                    name: "Kokoro 82M (voice model)",
                    sizeGb: 0.35,
                    installed: voiceRuntime.ttsModelInstalled,
                    default: true,
                  }}
                  onInstalled={onRefreshVoiceRuntime}
                />
              </div>
            ) : voiceRuntime ? (
              <div className="image-library-grid" style={{ marginBottom: 12 }}>
                <KokoroOnnxCard
                  installed={voiceRuntime.ttsModelInstalled}
                  onInstalled={onRefreshVoiceRuntime}
                />
              </div>
            ) : null}
            {ttsVoices.length > 0 && (
              <div className="image-library-grid">
                {ttsVoices.map((voice: TtsVoice) => (
                  <article key={voice.id} className="image-library-card">
                    <div className="image-library-card-head">
                      <div>
                        <h3 style={{ fontSize: "0.9rem" }}>{voice.name}</h3>
                        <p className="muted-text" style={{ fontSize: "0.75rem" }}>{voice.id}</p>
                      </div>
                      <span className="badge muted">{voice.language}</span>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </>
        )}
      </Panel>
    </div>
  );
}
