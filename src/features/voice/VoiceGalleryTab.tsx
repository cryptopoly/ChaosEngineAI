import { Panel } from "../../components/Panel";

export function VoiceGalleryTab() {
  return (
    <div className="content-grid image-page-grid">
      <Panel title="Voice Gallery" subtitle="Saved transcripts and audio clips" className="span-2">
        <div className="empty-state">
          <p>
            <span className="badge warning" style={{ marginRight: 8 }}>Coming soon</span>
            Saved transcripts and generated audio clips will appear here. Recordings and
            synthesised speech are saved to{" "}
            <code>~/.chaosengine/voice/</code> and surfaced with filters by date and voice.
          </p>
          <p className="muted-text" style={{ marginTop: 12 }}>
            Use Voice Studio to record and transcribe audio, or generate speech from text.
            Saved outputs will land here automatically once the persistence layer ships.
          </p>
        </div>
      </Panel>
    </div>
  );
}
