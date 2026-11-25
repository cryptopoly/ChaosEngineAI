/**
 * "Ready to export?" (audit change 6): the project's open points in three groups, each line a plain sentence with at
 * most one button that opens the place where it is fixed. The groups come from `guide.readiness`; the lines the app
 * adds (loudness, captions, the AI tag, the preview) sit under "For your information".
 */
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardProject } from "../../types";
import type { LocalLine, ReadinessFix, ReadinessGroup, ReadinessGroupId, ReadinessLine } from "./exportReadiness";
import "./exportCheck.css";

export interface ReadinessCheckProps {
  project: Pick<StoryboardProject, "shots">;
  groups: ReadinessGroup[];
  /** The Rough cut button, shown beside the title. */
  actions?: ReactNode;
  onFix: (fix: ReadinessFix) => void;
}

function LocalText({ line }: { line: LocalLine }) {
  const { t } = useTranslation("studio");
  switch (line.id) {
    case "upscale":
      return (
        <>
          {line.method === "fast"
            ? t("exportCheck.line.upscaleSharp", { defaultValue: "It is scaled up to {size} at the end with the AI upscaler.", size: line.size })
            : t("exportTab.line.upscale", { defaultValue: "It is scaled up to {size} at the end: more pixels, not more detail.", size: line.size })}
        </>
      );
    case "loudness":
      return (
        <>
          {line.lufs != null
            ? t("exportTab.line.loudness", { defaultValue: "The sound is levelled to {lufs} LUFS.", lufs: line.lufs })
            : t("exportTab.line.loudnessOff", { defaultValue: "The sound is left as mixed (no loudness target)." })}
        </>
      );
    case "captions":
      if (line.count === 0) return <>{t("exportTab.line.noCaptions", { defaultValue: "No captions." })}</>;
      return (
        <>
          {line.how === "srt+burn"
            ? t("exportTab.line.captionsBoth", { defaultValue: "{n, plural, one {# caption} other {# captions}}, burnt in and written as an SRT file.", n: line.count })
            : line.how === "srt"
              ? t("exportTab.line.captionsSrt", { defaultValue: "{n, plural, one {# caption} other {# captions}}, written as an SRT file.", n: line.count })
              : line.how === "burn"
                ? t("exportTab.line.captionsBurn", { defaultValue: "{n, plural, one {# caption} other {# captions}}, burnt into the picture.", n: line.count })
                : t("exportTab.line.captionsUnused", { defaultValue: "{n, plural, one {# caption} other {# captions}} not exported: tick SRT file or Burn in below.", n: line.count })}
        </>
      );
    case "aiTag":
      return (
        <>
          {line.cloned
            ? t("exportTab.line.cloned", { defaultValue: "A cloned voice is used, so the AI tag is always written." })
            : line.on
              ? t("exportTab.line.aiTagOn", { defaultValue: "A machine-readable AI tag is written into the file." })
              : t("exportTab.line.aiTagOff", { defaultValue: "No AI tag is written." })}
        </>
      );
    default:
      return <>{t("exportTab.line.preview", { defaultValue: "The preview is out of date (that does not stop an export)." })}</>;
  }
}

/** The open-place for a local line: only the preview points anywhere (the Mixer, where the preview is made). */
function localFix(line: LocalLine): ReadinessFix | null {
  return line.id === "preview" ? { kind: "mixer", tab: "mixer", shotId: null } : null;
}

export function ReadinessCheck({ project, groups, actions, onFix }: ReadinessCheckProps) {
  const { t } = useTranslation("studio");
  const titles: Record<ReadinessGroupId, string> = {
    stops: t("exportCheck.stops", { defaultValue: "Stops the export" }),
    worthFixing: t("exportCheck.worthFixing", { defaultValue: "Worth fixing" }),
    forInfo: t("exportCheck.forInfo", { defaultValue: "For your information" }),
  };
  const none: Record<ReadinessGroupId, string> = {
    stops: t("exportCheck.stopsNone", { defaultValue: "Nothing stops the export." }),
    worthFixing: t("exportCheck.worthFixingNone", { defaultValue: "Nothing to fix." }),
    forInfo: "",
  };

  const fixLabel = (fix: ReadinessFix): string => {
    switch (fix.kind) {
      case "plan":
        return t("exportCheck.fix.plan", { defaultValue: "Open Plan" });
      case "song":
        return t("exportCheck.fix.song", { defaultValue: "Open the song step" });
      case "cast":
        return t("exportCheck.fix.cast", { defaultValue: "Open Cast" });
      case "mixer":
        return t("exportTab.openMixer", { defaultValue: "Open the Mixer" });
      case "shot": {
        const index = fix.shotId ? project.shots.findIndex((shot) => shot.id === fix.shotId) : -1;
        return index >= 0
          ? t("exportCheck.fix.shot", { defaultValue: "Open shot {n}", n: index + 1 })
          : t("exportTab.openStoryboard", { defaultValue: "Open in Storyboard" });
      }
      default:
        return t("exportTab.openStoryboard", { defaultValue: "Open in Storyboard" });
    }
  };

  const mark = (group: ReadinessGroupId): string => (group === "stops" ? "×" : group === "worthFixing" ? "!" : "i");

  return (
    <section className="export-check" aria-labelledby="export-check-title">
      <div className="export-check__head">
        <h4 id="export-check-title">{t("exportCheck.title", { defaultValue: "Ready to export?" })}</h4>
        {actions}
      </div>
      {groups.map((group) => (
        <div key={group.id} className={`export-check__group export-check__group--${group.id}`} data-group={group.id}>
          <h5 className="export-check__group-title">{titles[group.id]}</h5>
          {group.lines.length === 0 ? (
            none[group.id] ? <p className="muted-text export-check__none">{none[group.id]}</p> : null
          ) : (
            <ul>
              {group.lines.map((line: ReadinessLine, index) => {
                const fix = line.source === "guide" ? line.fix : localFix(line.local);
                return (
                  <li key={line.source === "guide" ? `${line.code}-${index}` : line.local.id} className="export-check__line" data-code={line.source === "guide" ? line.code : line.local.id}>
                    <span className="export-check__mark" aria-hidden="true">{mark(group.id)}</span>
                    <span className="export-check__text">{line.source === "guide" ? line.message : <LocalText line={line.local} />}</span>
                    {fix ? (
                      <button type="button" className="mixer__link" onClick={() => onFix(fix)}>
                        {fixLabel(fix)}
                      </button>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      ))}
    </section>
  );
}
