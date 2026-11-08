/**
 * The Cast tab of a project (D-65): characters and places, each a name, a few words and up to four pictures.
 * Set once, used by shots to keep a look. Optional: a project with no cast works (a street, a crowd), and
 * nothing elsewhere waits for one. Dropping pictures on empty space adds a character named after the file.
 */
import { useEffect, useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import { uploadStoryboardPicture } from "../../api";
import { PlusIcon } from "../../components/icons/ActionIcons";
import type { ImageOutputArtifact, StoryboardProject, StoryboardReference } from "../../types";
import { MAX_PICTURES, nameFromFile, picturesFrom, usedByCount } from "./pickerLogic";
import { ReferenceCard } from "./ReferenceCard";
import { ReferenceImagePicker } from "./ReferenceImagePicker";
import type { UseStoryboardResult } from "./useStoryboard";

export interface CastTabProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  imageGallery: ImageOutputArtifact[];
}

type Kind = "character" | "place";

export function CastTab({ story, project, imageGallery }: CastTabProps) {
  const { t } = useTranslation("studio");
  const { busy } = story;
  const [pickerFor, setPickerFor] = useState<StoryboardReference | null>(null);
  const [addedId, setAddedId] = useState<string | null>(null);
  const [dropOver, setDropOver] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const { characters, places } = project.references;
  const known = useRef<Set<string> | null>(null);
  // The reference as the project has it now (a name typed just before opening the picker lands meanwhile).
  const picking = pickerFor ? ([...characters, ...places].find((r) => r.id === pickerFor.id) ?? pickerFor) : null;

  async function add(kind: Kind, extra: { name?: string; images?: string[] } = {}) {
    known.current = new Set(allIds);
    return await story.op("add_reference", { kind, ...extra });
  }

  // A card just added gets the focus: it is the one whose id was not there before the add.
  const allIds = [...characters, ...places].map((r) => r.id);
  const idsKey = allIds.join(",");
  useEffect(() => {
    if (!known.current) return;
    const fresh = allIds.find((id) => !known.current?.has(id));
    if (fresh) {
      setAddedId(fresh);
      known.current = null;
    }
  }, [idsKey]); // allIds is derived from idsKey

  async function uploadAll(files: File[]): Promise<string[]> {
    const handles: string[] = [];
    for (const file of files) handles.push((await uploadStoryboardPicture(file)).handle);
    return handles;
  }

  async function addFiles(reference: StoryboardReference, files: File[]) {
    setProblem(null);
    try {
      for (const handle of await uploadAll(files)) {
        await story.op("edit_reference", { refId: reference.id, addImage: handle });
      }
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    }
  }

  async function onTabDrop(event: DragEvent) {
    event.preventDefault();
    setDropOver(false);
    const files = picturesFrom(event.dataTransfer?.files, 8);
    if (files.length === 0) return;
    setProblem(null);
    try {
      for (const file of files) {
        const [handle] = await uploadAll([file]);
        await add("character", { name: nameFromFile(file.name), images: [handle] });
      }
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    }
  }

  const groups: { kind: Kind; title: string; add: string; items: StoryboardReference[] }[] = [
    { kind: "character", title: t("board.references.characters", { defaultValue: "Characters" }), add: t("board.references.addCharacter", { defaultValue: "Add character" }), items: characters },
    { kind: "place", title: t("board.references.places", { defaultValue: "Places" }), add: t("board.references.addPlace", { defaultValue: "Add place" }), items: places },
  ];

  return (
    <div
      className={`cast-tab${dropOver ? " cast-tab--over" : ""}`}
      onDragOver={(event) => {
        if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
        event.preventDefault();
        setDropOver(true);
      }}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDropOver(false);
      }}
      onDrop={(event) => void onTabDrop(event)}
    >
      <div className="cast-tab__head">
        <div>
          <h3>{t("cast.title", { defaultValue: "Cast and places" })}</h3>
          <p className="muted-text">
            {t("cast.subtitle", {
              defaultValue: "Optional. A name, a few words and up to {max} pictures; shots that use it keep the same look. A film without characters works too.",
              max: MAX_PICTURES,
            })}
          </p>
        </div>
      </div>
      {problem ? (
        <div className="callout error" role="alert">
          <p>{problem}</p>
        </div>
      ) : null}
      {groups.map((group) => (
        <section key={group.kind} className="cast-tab__group" aria-label={group.title}>
          <div className="cast-tab__group-head">
            <h4>{group.title}</h4>
            <button type="button" className="secondary-button" disabled={busy} onClick={() => void add(group.kind)}>
              <PlusIcon /> {group.add}
            </button>
          </div>
          {group.items.length === 0 ? (
            <p className="muted-text">
              {t("cast.noneShort", { defaultValue: "None yet." })}
            </p>
          ) : (
            <div className="cast-tab__cards">
              {group.items.map((reference) => (
                <ReferenceCard
                  key={reference.id}
                  projectId={project.id}
                  reference={reference}
                  usedBy={usedByCount(project, reference.id)}
                  disabled={busy}
                  autoFocus={reference.id === addedId}
                  onEdit={(patch) => void story.op("edit_reference", { refId: reference.id, ...patch })}
                  onRemove={() => void story.op("remove_reference", { refId: reference.id })}
                  onRemovePicture={(index) => void story.op("edit_reference", { refId: reference.id, removeImage: index })}
                  onOpenPicker={() => setPickerFor(reference)}
                  onFiles={(files) => void addFiles(reference, files)}
                />
              ))}
            </div>
          )}
        </section>
      ))}
      <p className="cast-tab__drop muted-text">
        {t("cast.dropHint", { defaultValue: "Drop pictures anywhere on this tab to add a character named after each file." })}
      </p>
      {picking ? (
        <ReferenceImagePicker
          title={t("cast.pickerTitle", { defaultValue: "Add a picture of {name}", name: picking.name || t("cast.unnamed", { defaultValue: "Unnamed" }) })}
          project={project}
          imageGallery={imageGallery}
          room={MAX_PICTURES - picking.images.length}
          onPick={(source) => story.op("edit_reference", { refId: picking.id, addImage: source })}
          onClose={() => setPickerFor(null)}
        />
      ) : null}
    </div>
  );
}
