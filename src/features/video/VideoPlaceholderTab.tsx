import { Panel } from "../../components/Panel";

export type VideoPlaceholderVariant = "models" | "discover" | "studio" | "gallery";

interface PlannedEngine {
  name: string;
  provider: string;
  note: string;
  size: string;
}

const plannedEngines: PlannedEngine[] = [
  {
    name: "LTX-Video",
    provider: "Lightricks",
    size: "~2 GB",
    note: "Fast, lightweight — designed for consumer hardware. Good first target.",
  },
  {
    name: "Wan 2.2",
    provider: "Alibaba",
    size: "~14 GB",
    note: "Strong text-to-video quality. Runs on 24GB+ VRAM or Apple Silicon with unified memory.",
  },
  {
    name: "HunyuanVideo",
    provider: "Tencent",
    size: "~25 GB",
    note: "High fidelity, longer clips. Needs 40GB+ class hardware.",
  },
  {
    name: "Mochi 1",
    provider: "Genmo",
    size: "~10 GB",
    note: "Open-weight with competitive motion quality. Apache 2.0 licence.",
  },
];

const copy: Record<VideoPlaceholderVariant, { title: string; subtitle: string; body: string }> = {
  models: {
    title: "Installed Video Models",
    subtitle: "Models on this machine",
    body: "No video models detected locally yet. Once the video engine ships, downloaded weights will show up here alongside their quantisation, size on disk, and runtime target.",
  },
  discover: {
    title: "Discover Video Models",
    subtitle: "Browse and download video generators",
    body: "The catalogue below previews the engines we plan to support first. Download buttons will light up once the video runtime is wired in.",
  },
  studio: {
    title: "Video Studio",
    subtitle: "Prompt, generate, and iterate",
    body: "Studio will host the prompt editor, aspect ratio and duration controls, seed pinning, and per-frame previews. Sit tight — no engine loaded yet.",
  },
  gallery: {
    title: "Video Gallery",
    subtitle: "Saved outputs and filters",
    body: "Finished renders will land here with filters by model, duration, and resolution. Drag-to-reorder and favourites are on the roadmap.",
  },
};

interface VideoPlaceholderTabProps {
  variant: VideoPlaceholderVariant;
}

export function VideoPlaceholderTab({ variant }: VideoPlaceholderTabProps) {
  const { title, subtitle, body } = copy[variant];
  const showCatalog = variant === "discover" || variant === "models";

  return (
    <div className="content-grid image-page-grid">
      <Panel title={title} subtitle={subtitle} className="span-2">
        <div className="empty-state">
          <p>
            <span className="badge warning" style={{ marginRight: 8 }}>Coming soon</span>
            {body}
          </p>
          <p className="muted-text" style={{ marginTop: 12 }}>
            Local video generation is on the ChaosEngineAI roadmap. This tab is a placeholder so we can ship
            the routing and UX shell early — the inference engine will slot in behind it.
          </p>
        </div>
      </Panel>

      {showCatalog ? (
        <Panel
          title="Planned engines"
          subtitle="Candidates we're evaluating for the first release"
          className="span-2"
        >
          <div className="image-library-grid">
            {plannedEngines.map((engine) => (
              <article key={engine.name} className="image-library-card">
                <div className="image-library-card-head">
                  <div>
                    <h3>{engine.name}</h3>
                    <p>{engine.provider}</p>
                  </div>
                  <span className="badge subtle">Planned</span>
                </div>
                <div className="image-library-stats">
                  <span>{engine.size}</span>
                </div>
                <p className="muted-text">{engine.note}</p>
              </article>
            ))}
          </div>
        </Panel>
      ) : null}
    </div>
  );
}
