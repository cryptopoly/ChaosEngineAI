import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";

export type VideoPlaceholderVariant = "models" | "discover" | "studio" | "gallery";

interface PlannedEngine {
  name: string;
  provider: string;
  note: string;
  size: string;
}

function plannedEngineList(t: TFunction<"common">): PlannedEngine[] {
  return [
    {
      name: "LTX-Video",
      provider: "Lightricks",
      size: "~2 GB",
      note: t("videoPlaceholderTab.engines.ltxVideo", {
        defaultValue: "Fast, lightweight — designed for consumer hardware. Good first target.",
      }),
    },
    {
      name: "Wan 2.2",
      provider: "Alibaba",
      size: "~14 GB",
      note: t("videoPlaceholderTab.engines.wan22", {
        defaultValue: "Strong text-to-video quality. Runs on 24GB+ VRAM or Apple Silicon with unified memory.",
      }),
    },
    {
      name: "HunyuanVideo",
      provider: "Tencent",
      size: "~25 GB",
      note: t("videoPlaceholderTab.engines.hunyuanVideo", {
        defaultValue: "High fidelity, longer clips. Needs 40GB+ class hardware.",
      }),
    },
    {
      name: "Mochi 1",
      provider: "Genmo",
      size: "~10 GB",
      note: t("videoPlaceholderTab.engines.mochi1", {
        defaultValue: "Open-weight with competitive motion quality. Apache 2.0 licence.",
      }),
    },
  ];
}

function placeholderCopy(
  variant: VideoPlaceholderVariant,
  t: TFunction<"common">,
): { title: string; subtitle: string; body: string } {
  switch (variant) {
    case "models":
      return {
        title: t("videoPlaceholderTab.models.title", { defaultValue: "Installed Video Models" }),
        subtitle: t("videoPlaceholderTab.models.subtitle", { defaultValue: "Models on this machine" }),
        body: t("videoPlaceholderTab.models.body", {
          defaultValue: "No video models detected locally yet. Once the video engine ships, downloaded weights will show up here alongside their quantisation, size on disk, and runtime target.",
        }),
      };
    case "discover":
      return {
        title: t("videoPlaceholderTab.discover.title", { defaultValue: "Discover Video Models" }),
        subtitle: t("videoPlaceholderTab.discover.subtitle", { defaultValue: "Browse and download video generators" }),
        body: t("videoPlaceholderTab.discover.body", {
          defaultValue: "The catalogue below previews the engines we plan to support first. Download buttons will light up once the video runtime is wired in.",
        }),
      };
    case "studio":
      return {
        title: t("videoPlaceholderTab.studio.title", { defaultValue: "Video Studio" }),
        subtitle: t("videoPlaceholderTab.studio.subtitle", { defaultValue: "Prompt, generate, and iterate" }),
        body: t("videoPlaceholderTab.studio.body", {
          defaultValue: "Studio will host the prompt editor, aspect ratio and duration controls, seed pinning, and per-frame previews. Sit tight — no engine loaded yet.",
        }),
      };
    case "gallery":
      return {
        title: t("videoPlaceholderTab.gallery.title", { defaultValue: "Video Gallery" }),
        subtitle: t("videoPlaceholderTab.gallery.subtitle", { defaultValue: "Saved outputs and filters" }),
        body: t("videoPlaceholderTab.gallery.body", {
          defaultValue: "Finished renders will land here with filters by model, duration, and resolution. Drag-to-reorder and favourites are on the roadmap.",
        }),
      };
  }
}

interface VideoPlaceholderTabProps {
  variant: VideoPlaceholderVariant;
}

export function VideoPlaceholderTab({ variant }: VideoPlaceholderTabProps) {
  const { t } = useTranslation("common");
  const { title, subtitle, body } = placeholderCopy(variant, t);
  const plannedEngines = plannedEngineList(t);
  const showCatalog = variant === "discover" || variant === "models";

  return (
    <div className="content-grid image-page-grid">
      <Panel title={title} subtitle={subtitle} className="span-2">
        <div className="empty-state">
          <p>
            <span className="badge warning" style={{ marginRight: 8 }}>{t("videoPlaceholderTab.comingSoon", { defaultValue: "Coming soon" })}</span>
            {body}
          </p>
          <p className="muted-text" style={{ marginTop: 12 }}>
            {t("videoPlaceholderTab.roadmap", {
              defaultValue: "Local video generation is on the ChaosEngineAI roadmap. This tab is a placeholder so we can ship the routing and UX shell early — the inference engine will slot in behind it.",
            })}
          </p>
        </div>
      </Panel>

      {showCatalog ? (
        <Panel
          title={t("panels.plannedEngines", { defaultValue: "Planned engines" })}
          subtitle={t("panels.plannedEnginesSubtitle", {
            defaultValue: "Candidates we're evaluating for the first release",
          })}
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
                  <span className="badge subtle">{t("videoPlaceholderTab.planned", { defaultValue: "Planned" })}</span>
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
