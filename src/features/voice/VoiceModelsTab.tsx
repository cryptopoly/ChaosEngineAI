import { Panel } from "../../components/Panel";
import type { VoiceRuntime, SttModel, TtsVoice } from "../../types";

export interface VoiceModelsTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
}

export function VoiceModelsTab({ voiceRuntime, backendOnline }: VoiceModelsTabProps) {
  const sttModels = voiceRuntime?.sttModels ?? [];
  const ttsVoices = voiceRuntime?.ttsVoices ?? [];
  const sttBackend = voiceRuntime?.sttBackend ?? null;
  const ttsBackend = voiceRuntime?.ttsBackend ?? null;

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
        ) : sttModels.length === 0 ? (
          <div className="empty-state">
            <p className="muted-text">
              No STT models listed. Install{" "}
              <code>mlx-whisper</code> (Apple Silicon) or{" "}
              <code>faster-whisper</code> (other platforms) to enable transcription.
            </p>
          </div>
        ) : (
          <div className="image-library-grid">
            {sttModels.map((model: SttModel) => (
              <article key={model.id} className="image-library-card">
                <div className="image-library-card-head">
                  <div>
                    <h3 style={{ fontSize: "0.9rem" }}>{model.name}</h3>
                    <p className="muted-text" style={{ fontSize: "0.75rem" }}>
                      {model.id}
                    </p>
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
                <div className="image-library-stats">
                  <span>{model.sizeGb} GB</span>
                </div>
              </article>
            ))}
          </div>
        )}

        {!voiceRuntime?.sttAvailable && backendOnline && (
          <div className="notice-banner" style={{ marginTop: 12 }}>
            <p>
              Install <code>mlx-whisper</code> (Apple Silicon) or <code>faster-whisper</code>{" "}
              to enable transcription.
            </p>
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
        ) : (
          <>
            {!voiceRuntime?.ttsAvailable && (
              <div className="notice-banner" style={{ marginBottom: 12 }}>
                <p>
                  Install <code>mlx-audio</code> (Apple Silicon) or <code>kokoro-onnx</code>{" "}
                  to enable text-to-speech synthesis.
                </p>
              </div>
            )}
            {ttsVoices.length > 0 && (
              <div className="image-library-grid">
                {ttsVoices.map((voice: TtsVoice) => (
                  <article key={voice.id} className="image-library-card">
                    <div className="image-library-card-head">
                      <div>
                        <h3 style={{ fontSize: "0.9rem" }}>{voice.name}</h3>
                        <p className="muted-text" style={{ fontSize: "0.75rem" }}>
                          {voice.id}
                        </p>
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
