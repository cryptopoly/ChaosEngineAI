/** Project-level references (characters and places) that shots can use to keep a look. */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { PlusIcon, TrashIcon } from "../../components/icons/ActionIcons";
import { Disclosure } from "../../components/Disclosure";
import type { StoryboardOpArgs, StoryboardOpName, StoryboardProject, StoryboardReference } from "../../types";
import { PathField } from "./PathField";
import { ProjectImage } from "./ProjectMedia";

export type RunOp = <K extends StoryboardOpName>(name: K, args: StoryboardOpArgs[K]) => Promise<boolean>;

function ReferenceRow({
  projectId,
  reference,
  disabled,
  onEdit,
  onRemove,
}: {
  projectId: string;
  reference: StoryboardReference;
  disabled: boolean;
  onEdit: (patch: { name?: string; description?: string; image?: string }) => void;
  onRemove: () => void;
}) {
  const { t } = useTranslation("studio");
  const [name, setName] = useState(reference.name);
  const [description, setDescription] = useState(reference.description);
  useEffect(() => setName(reference.name), [reference.name]);
  useEffect(() => setDescription(reference.description), [reference.description]);

  return (
    <li className="reference-row">
      <ProjectImage projectId={projectId} rel={reference.image} version={reference.hash} className="reference-row__thumb" />
      <div className="reference-row__fields">
        <input
          type="text"
          value={name}
          maxLength={200}
          disabled={disabled}
          placeholder={t("board.references.name", { defaultValue: "Name" })}
          aria-label={t("board.references.name", { defaultValue: "Name" })}
          onChange={(event) => setName(event.target.value)}
          onBlur={() => name !== reference.name && onEdit({ name })}
        />
        <textarea
          rows={2}
          value={description}
          maxLength={4000}
          disabled={disabled}
          placeholder={t("board.references.description", { defaultValue: "What it looks like" })}
          aria-label={t("board.references.description", { defaultValue: "What it looks like" })}
          onChange={(event) => setDescription(event.target.value)}
          onBlur={() => description !== reference.description && onEdit({ description })}
        />
        <PathField
          label={t("board.references.image", { defaultValue: "Reference image" })}
          buttonLabel={t("board.references.setImage", { defaultValue: "Set image" })}
          extensions={["png", "jpg", "jpeg", "webp"]}
          disabled={disabled}
          onSubmit={(path) => onEdit({ image: path })}
        />
      </div>
      <button
        type="button"
        className="secondary-button icon-button"
        disabled={disabled}
        aria-label={t("board.references.remove", { defaultValue: "Remove reference" })}
        title={t("board.references.remove", { defaultValue: "Remove reference" })}
        onClick={onRemove}
      >
        <TrashIcon />
      </button>
    </li>
  );
}

export function ReferencesPanel({ project, busy, op }: { project: StoryboardProject; busy: boolean; op: RunOp }) {
  const { t } = useTranslation("studio");
  const { characters, places } = project.references;
  const groups: { kind: "character" | "place"; title: string; add: string; items: StoryboardReference[] }[] = [
    {
      kind: "character",
      title: t("board.references.characters", { defaultValue: "Characters" }),
      add: t("board.references.addCharacter", { defaultValue: "Add character" }),
      items: characters,
    },
    {
      kind: "place",
      title: t("board.references.places", { defaultValue: "Places" }),
      add: t("board.references.addPlace", { defaultValue: "Add place" }),
      items: places,
    },
  ];

  return (
    <Disclosure
      id="story-references"
      title={t("board.references.title", { defaultValue: "References" })}
      summary={t("board.references.summary", {
        defaultValue: "{characters, plural, one {# character} other {# characters}} · {places, plural, one {# place} other {# places}}",
        characters: characters.length,
        places: places.length,
      })}
    >
      <div className="references-panel">
        {groups.map((group) => (
          <section key={group.kind} className="references-panel__group">
            <div className="references-panel__head">
              <h4>{group.title}</h4>
              <button
                type="button"
                className="secondary-button"
                disabled={busy}
                onClick={() => void op("add_reference", { kind: group.kind })}
              >
                <PlusIcon /> {group.add}
              </button>
            </div>
            {group.items.length === 0 ? (
              <p className="muted-text">{t("board.references.none", { defaultValue: "None yet." })}</p>
            ) : (
              <ul className="reference-list">
                {group.items.map((reference) => (
                  <ReferenceRow
                    key={reference.id}
                    projectId={project.id}
                    reference={reference}
                    disabled={busy}
                    onEdit={(patch) => void op("edit_reference", { refId: reference.id, ...patch })}
                    onRemove={() => void op("remove_reference", { refId: reference.id })}
                  />
                ))}
              </ul>
            )}
          </section>
        ))}
      </div>
    </Disclosure>
  );
}
