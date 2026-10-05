import { useEffect, useState } from "react";
import { Panel } from "../../components/Panel";
import { getVoiceGallery, deleteGalleryItem, getGalleryAudio } from "../../api";
import type { VoiceGalleryItem } from "../../api";

export interface VoiceGalleryTabProps {
  backendOnline: boolean;
}

function formatDate(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

function GalleryCard({ item, onDeleted }: { item: VoiceGalleryItem; onDeleted: (id: string) => void }) {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loadingAudio, setLoadingAudio] = useState(false);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  async function handlePlay() {
    if (audioUrl) return;
    setLoadingAudio(true);
    try {
      const blob = await getGalleryAudio(item.id);
      setAudioUrl(URL.createObjectURL(blob));
    } finally {
      setLoadingAudio(false);
    }
  }

  async function handleDelete() {
    await deleteGalleryItem(item.id);
    onDeleted(item.id);
  }

  return (
    <article className="image-library-card">
      <div className="image-library-card-head">
        <div>
          <span className="badge subtle" style={{ marginRight: 6 }}>
            {item.kind === "audio" ? `Audio · ${item.voice ?? ""}` : "Transcript"}
          </span>
          <p className="muted-text" style={{ fontSize: "0.75rem" }}>{formatDate(item.createdAt)}</p>
        </div>
        <button className="action-btn" onClick={handleDelete}>Delete</button>
      </div>
      <p style={{ fontSize: "0.85rem", marginTop: 6 }}>{item.text}</p>
      {item.kind === "audio" && (
        audioUrl ? (
          <audio controls src={audioUrl} style={{ width: "100%", marginTop: 8 }} />
        ) : (
          <button className="action-btn" onClick={handlePlay} disabled={loadingAudio} style={{ marginTop: 8 }}>
            {loadingAudio ? "Loading…" : "Play"}
          </button>
        )
      )}
    </article>
  );
}

export function VoiceGalleryTab({ backendOnline }: VoiceGalleryTabProps) {
  const [items, setItems] = useState<VoiceGalleryItem[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!backendOnline) {
      setItems([]);
      setLoaded(false);
      return;
    }
    void getVoiceGallery()
      .then((result) => setItems(result))
      .catch(() => {})
      .finally(() => setLoaded(true));
  }, [backendOnline]);

  function handleDeleted(id: string) {
    setItems((current) => current.filter((item) => item.id !== id));
  }

  return (
    <div className="content-grid image-page-grid">
      <Panel title="Voice Gallery" subtitle="Saved transcripts and audio clips" className="span-2">
        {!backendOnline ? (
          <div className="empty-state">
            <p className="muted-text">Backend offline — connect to see saved items.</p>
          </div>
        ) : !loaded ? (
          <div className="empty-state">
            <p className="muted-text">Loading…</p>
          </div>
        ) : items.length === 0 ? (
          <div className="empty-state">
            <p className="muted-text">
              Nothing saved yet. Use Voice Studio to save a transcript or generated audio clip.
            </p>
          </div>
        ) : (
          <div className="image-library-grid">
            {items.map((item) => (
              <GalleryCard key={item.id} item={item} onDeleted={handleDeleted} />
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
