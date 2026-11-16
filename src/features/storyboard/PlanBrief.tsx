/**
 * The brief on the Plan tab (S8): what the film is, how long it should be, how many shots, and the style
 * pictures the model should follow. The brief and the length save as ops (so Undo works) when they change;
 * the pictures join the project's styles (S15), so the Cast tab and every shot use the same look. "Propose
 * shots" asks the loaded chat model; while it writes, the button shows how long it has taken.
 */
import { useEffect, useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import { uploadStoryboardPicture } from "../../api";
import { PlusIcon } from "../../components/icons/ActionIcons";
import type { ImageOutputArtifact, StoryboardProject } from "../../types";
import { PaletteSwatches } from "./PaletteSwatches";
import { MAX_PICTURES, picturesFrom } from "./pickerLogic";
import { LENGTH_CHOICES, SHOT_COUNT_CHOICES, lengthLabel, planOf, projectPalette, styleWithRoom, stylesOf } from "./planLogic";
import { PlanSuggestion } from "./PlanSuggestion";
import { ProjectImage } from "./ProjectMedia";
import { ReferenceImagePicker } from "./ReferenceImagePicker";
import type { UseStoryboardResult } from "./useStoryboard";

export interface PlanBriefProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  imageGallery: ImageOutputArtifact[];
  canPropose: boolean;
  proposing: boolean;
  elapsed: number;
  modelName: string | null;
  /** A proposal waits below, or shots exist: "Use these shots" or "Make them in Storyboard" is the main action then, so Propose steps back. */
  hasProposal?: boolean;
  onPropose: (brief: string, filmSeconds: number | null, shots: number | null) => void;
  onOpenCast: () => void;
}

export function PlanBrief({ story, project, imageGallery, canPropose, proposing, elapsed, modelName, hasProposal, onPropose, onOpenCast }: PlanBriefProps) {
  const { t } = useTranslation("studio");
  const plan = planOf(project);
  const [brief, setBrief] = useState(plan.brief);
  const [count, setCount] = useState<number | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [over, setOver] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const saved = useRef(plan.brief);
  useEffect(() => {
    if (plan.brief !== saved.current) {
      saved.current = plan.brief;
      setBrief(plan.brief); // an Undo or another window changed it
    }
  }, [plan.brief]);

  const styles = stylesOf(project);
  const pictures = styles.flatMap((ref) => ref.images.map((image, index) => ({ ref, image, index })));
  const palette = projectPalette(project);

  function saveBrief() {
    if (brief === saved.current) return;
    saved.current = brief;
    void story.op("set_plan", { brief });
  }

  function setLength(seconds: number | null) {
    void story.op("set_plan", seconds ? { filmSeconds: seconds } : { clearFilmSeconds: true });
  }

  /** One picture into the first style with room, or a new style named "Look". */
  async function addPicture(source: string): Promise<boolean> {
    const target = styleWithRoom(project);
    if (target) return await story.op("edit_reference", { refId: target.id, addImage: source });
    return await story.op("add_reference", {
      kind: "style",
      name: t("board.style.defaultName", { defaultValue: "Look" }),
      images: [source],
    });
  }

  async function addFiles(files: File[]) {
    setProblem(null);
    try {
      for (const file of files) {
        const { handle } = await uploadStoryboardPicture(file);
        if (!(await addPicture(handle))) break;
      }
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setOver(false);
    const files = picturesFrom(event.dataTransfer?.files, 8);
    if (files.length > 0) void addFiles(files);
  }

  const lengthText = (seconds: number) => {
    const { minutes, seconds: rest } = lengthLabel(seconds);
    if (minutes === 0) return t("board.plan.length.seconds", { defaultValue: "{n} s", n: rest });
    if (rest === 0) return t("board.plan.length.minutes", { defaultValue: "{n} min", n: minutes });
    return t("board.plan.length.minutesSeconds", { defaultValue: "{m} min {s} s", m: minutes, s: rest });
  };
  const trimmed = brief.trim();

  return (
    <section className="plan-brief" aria-labelledby="plan-brief-title">
      <div className="plan-section-head">
        <h3 id="plan-brief-title">{t("board.plan.brief.title", { defaultValue: "The idea" })}</h3>
        <p className="muted-text">
          {t("board.plan.brief.subtitle", { defaultValue: "Say what the film is for and how it should feel. The model proposes the shots; you edit them before anything renders." })}
        </p>
      </div>

      <label className="field plan-brief__text">
        <span>{t("board.plan.brief.label", { defaultValue: "Brief" })}</span>
        <textarea
          className="text-area"
          rows={4}
          maxLength={4000}
          value={brief}
          placeholder={t("board.plan.brief.placeholder", { defaultValue: "A one-minute advert for a small tea shop: calm mornings, loose-leaf tea poured slowly, regulars who feel at home. End on the shop sign." })}
          onChange={(event) => setBrief(event.target.value)}
          onBlur={saveBrief}
        />
        {brief.length > 3600 ? (
          <span className="muted-text plan-brief__count">{t("board.plan.brief.count", { defaultValue: "{n} of 4000 characters", n: brief.length })}</span>
        ) : null}
      </label>

      <div className="plan-brief__options">
        <div className="plan-brief__option" role="group" aria-labelledby="plan-length-label">
          <span id="plan-length-label" className="plan-brief__option-label">{t("board.plan.length.label", { defaultValue: "Length" })}</span>
          <div className="plan-chips">
            <button
              type="button"
              className={`plan-chip${plan.filmSeconds === null ? " is-active" : ""}`}
              aria-pressed={plan.filmSeconds === null}
              disabled={story.busy}
              onClick={() => setLength(null)}
            >
              {t("board.plan.length.any", { defaultValue: "Any" })}
            </button>
            {LENGTH_CHOICES.map((seconds) => (
              <button
                key={seconds}
                type="button"
                className={`plan-chip${plan.filmSeconds === seconds ? " is-active" : ""}`}
                aria-pressed={plan.filmSeconds === seconds}
                disabled={story.busy}
                onClick={() => setLength(seconds)}
              >
                {lengthText(seconds)}
              </button>
            ))}
          </div>
        </div>
        <label className="plan-brief__option">
          <span className="plan-brief__option-label">{t("board.plan.count.label", { defaultValue: "Shots" })}</span>
          <select className="text-input plan-brief__count-select" value={count ?? ""} onChange={(event) => setCount(event.target.value ? Number(event.target.value) : null)}>
            <option value="">{t("board.plan.count.auto", { defaultValue: "4 to 8, the model chooses" })}</option>
            {SHOT_COUNT_CHOICES.map((n) => (
              <option key={n} value={n}>
                {t("board.plan.count.value", { defaultValue: "{n} shots", n })}
              </option>
            ))}
          </select>
        </label>
      </div>
      <PlanSuggestion brief={brief} target={plan.filmSeconds} lengthText={lengthText} onUseLength={setLength} />

      <div
        className={`plan-pictures${over ? " is-over" : ""}`}
        onDragOver={(event) => {
          if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={(event) => {
          if (event.currentTarget === event.target) setOver(false);
        }}
        onDrop={onDrop}
      >
        <div className="plan-pictures__head">
          <span className="plan-brief__option-label">{t("board.plan.pictures.label", { defaultValue: "Style pictures" })}</span>
          <span className="muted-text">
            {t("board.plan.pictures.hint", { defaultValue: "Screenshots or photos of the look you want. Drop them here; nothing is fetched from the web." })}
          </span>
        </div>
        <ul className="plan-pictures__grid">
          {pictures.map(({ ref, image, index }) => (
            <li key={`${ref.id}-${image.path}`}>
              <ProjectImage
                projectId={project.id}
                rel={image.path}
                version={image.hash}
                thumb={256}
                className="plan-pictures__thumb"
                alt={t("board.style.pictureAlt", { defaultValue: "{name}, picture {n}", name: ref.name || t("board.style.defaultName", { defaultValue: "Look" }), n: index + 1 })}
              />
            </li>
          ))}
          {styles.length < 20 || styleWithRoom(project) ? (
            <li>
              <button type="button" className="plan-pictures__add" disabled={story.busy} onClick={() => setPickerOpen(true)}>
                <PlusIcon aria-hidden="true" />
                <span>{t("board.plan.pictures.add", { defaultValue: "Add picture" })}</span>
              </button>
            </li>
          ) : null}
        </ul>
        {palette.length > 0 ? (
          <div className="plan-pictures__palette">
            <PaletteSwatches colours={palette} size="sm" label={t("board.style.paletteLabel", { defaultValue: "Colours read from the pictures" })} />
            <button type="button" className="plan-link" onClick={onOpenCast}>
              {t("board.plan.pictures.edit", { defaultValue: "Edit the style in Cast" })}
            </button>
          </div>
        ) : null}
        {problem ? <p className="plan-error" role="alert">{problem}</p> : null}
      </div>

      <div className="plan-brief__actions">
        <button
          type="button"
          className={hasProposal ? "secondary-button" : "primary-button"}
          disabled={!canPropose || proposing || !trimmed}
          aria-describedby="plan-propose-hint"
          onClick={() => {
            saveBrief();
            onPropose(trimmed, plan.filmSeconds, count);
          }}
        >
          {proposing
            ? t("board.plan.propose.working", { defaultValue: "Writing shots… {n} s", n: elapsed })
            : project.shots.length > 0
              ? t("board.plan.propose.again", { defaultValue: "Propose new shots" })
              : t("board.plan.propose.button", { defaultValue: "Propose shots" })}
        </button>
        <span id="plan-propose-hint" className="muted-text" aria-live="polite">
          {proposing
            ? t("board.plan.propose.wait", { defaultValue: "{model} is writing. A large model can take a minute or two.", model: modelName ?? "" })
            : !trimmed
              ? t("board.plan.propose.needBrief", { defaultValue: "Write a few words first." })
              : !canPropose
                ? t("board.plan.propose.needModel", { defaultValue: "Load a chat model first." })
                : t("board.plan.propose.safe", { defaultValue: "Nothing changes until you choose to use the shots." })}
        </span>
      </div>

      {pickerOpen ? (
        <ReferenceImagePicker
          title={t("board.plan.pictures.pickerTitle", { defaultValue: "Add a style picture" })}
          project={project}
          imageGallery={imageGallery}
          room={MAX_PICTURES}
          onPick={addPicture}
          onClose={() => setPickerOpen(false)}
        />
      ) : null}
    </section>
  );
}
