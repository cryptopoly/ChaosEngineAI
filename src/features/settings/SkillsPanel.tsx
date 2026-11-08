/**
 * D-62: Settings > Skills. Import a skill folder or .zip (by path or the desktop file dialog), see its
 * description and files, switch it on for Chat, delete it. New skills start off; scripts never run.
 */
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { deleteSkill, importSkill, listSkills, setSkillEnabled } from "../../api/skills";
import type { Skill } from "../../api/skills";
import { Panel } from "../../components/Panel";
import { notifySkillsChanged } from "../chat/skillsChat";

export function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
}

async function pickPath(directory: boolean, title: string): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open(
    directory ? { directory: true, multiple: false, title } : { multiple: false, title, filters: [{ name: "Zip", extensions: ["zip"] }] },
  );
  return typeof selected === "string" && selected ? selected : null;
}

export interface SkillCardProps {
  skill: Skill;
  busy: boolean;
  confirming: boolean;
  onToggle: (enabled: boolean) => void;
  onAskDelete: () => void;
  onCancelDelete: () => void;
  onDelete: () => void;
}

/** One imported skill: name, switch, description, files and skipped files, delete with a confirm step. */
export function SkillCard({ skill, busy, confirming, onToggle, onAskDelete, onCancelDelete, onDelete }: SkillCardProps) {
  const { t } = useTranslation("common");
  return (
    <li className="integration-card skills-card" aria-label={skill.name}>
      <div className="skills-card__head">
        <strong>{skill.name}</strong>
        <label className="check-row">
          <input
            type="checkbox"
            checked={skill.enabled}
            disabled={busy}
            onChange={(event) => onToggle(event.target.checked)}
          />
          {t("settingsTab.skills.enabled", { defaultValue: "Available in Chat" })}
        </label>
      </div>
      <p className="skills-card__description">{skill.description}</p>
      <details>
        <summary>
          {t("settingsTab.skills.files", { count: skill.files.length, defaultValue: "{count, plural, one {# file} other {# files}}" })}
        </summary>
        <ul className="skills-files">
          {skill.files.map((file) => (
            <li key={file.path}>
              <span className="mono-text">{file.path}</span> <span className="muted-text">{formatBytes(file.bytes)}</span>
            </li>
          ))}
        </ul>
        {skill.skipped.length > 0 ? (
          <>
            <p className="muted-text">{t("settingsTab.skills.skippedTitle", { defaultValue: "Skipped (not imported):" })}</p>
            <ul className="skills-files">
              {skill.skipped.map((file) => (
                <li key={file.path}>
                  <span className="mono-text">{file.path}</span> <span className="muted-text">{file.reason}</span>
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </details>
      <div className="skills-card__actions">
        {confirming ? (
          <>
            <button
              className="danger-button"
              type="button"
              disabled={busy}
              onClick={onDelete}
            >
              {t("settingsTab.skills.confirmDelete", { name: skill.name, defaultValue: "Delete {name} for good" })}
            </button>
            <button className="secondary-button" type="button" onClick={onCancelDelete}>
              {t("settingsTab.skills.cancel", { defaultValue: "Cancel" })}
            </button>
          </>
        ) : (
          <button className="secondary-button" type="button" disabled={busy} onClick={onAskDelete}>
            {t("settingsTab.skills.delete", { defaultValue: "Delete" })}
          </button>
        )}
      </div>
    </li>
  );
}

export function SkillsPanel({ backendOnline }: { backendOnline: boolean }) {
  const { t } = useTranslation("common");
  const [skills, setSkills] = useState<Skill[]>([]);
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imported, setImported] = useState<Skill | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setSkills(await listSkills());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    if (backendOnline) void refresh();
  }, [backendOnline, refresh]);

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      await refresh();
      notifySkillsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function choose(directory: boolean) {
    try {
      const picked = await pickPath(directory, t("settingsTab.skills.pickTitle", { defaultValue: "Choose a skill" }));
      if (picked) setPath(picked);
    } catch {
      setError(t("settingsTab.skills.noDialog", { defaultValue: "The file dialog is not available here. Type the path instead." }));
    }
  }

  const doImport = () =>
    run(async () => {
      setImported(await importSkill(path.trim()));
      setPath("");
    });

  return (
    <Panel
      title={t("settingsTab.skills.title", { defaultValue: "Skills" })}
      subtitle={t("settingsTab.skills.subtitle", {
        defaultValue: "Teach the chat model a procedure: a folder with a SKILL.md and optional text files, or a .zip of one.",
      })}
      className="settings-integrations-panel skills-panel"
    >
      <div className="control-stack">
        <p className="muted-text" role="note">
          {t("settingsTab.skills.safety", {
            defaultValue:
              "Scripts in a skill are never run. The model only reads a skill's text, as reference. New skills start switched off: read the description and file list before you switch one on.",
          })}
        </p>
        <div className="skills-import-row">
          <input
            type="text"
            value={path}
            placeholder={t("settingsTab.skills.pathPlaceholder", { defaultValue: "/path/to/skill-folder or skill.zip" })}
            aria-label={t("settingsTab.skills.pathLabel", { defaultValue: "Skill folder or .zip path" })}
            disabled={busy || !backendOnline}
            onChange={(event) => setPath(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && path.trim()) void doImport();
            }}
          />
          <button className="secondary-button" type="button" disabled={busy} onClick={() => void choose(true)}>
            {t("settingsTab.skills.chooseFolder", { defaultValue: "Choose folder" })}
          </button>
          <button className="secondary-button" type="button" disabled={busy} onClick={() => void choose(false)}>
            {t("settingsTab.skills.chooseZip", { defaultValue: "Choose .zip" })}
          </button>
          <button className="primary-button" type="button" disabled={busy || !backendOnline || !path.trim()} onClick={() => void doImport()}>
            {t("settingsTab.skills.import", { defaultValue: "Import" })}
          </button>
        </div>
        {error ? <p className="error-text" role="alert">{error}</p> : null}
        {imported ? (
          <p className="muted-text" role="status">
            {t("settingsTab.skills.imported", {
              name: imported.name,
              files: imported.files.length,
              skipped: imported.skipped.length,
              defaultValue:
                "Imported {name}: {files, plural, one {# file} other {# files}} kept, {skipped, plural, =0 {nothing skipped} one {# skipped} other {# skipped}}. It is off until you switch it on.",
            })}
          </p>
        ) : null}
        {skills.length === 0 ? (
          <p className="muted-text">{t("settingsTab.skills.empty", { defaultValue: "No skills imported yet." })}</p>
        ) : (
          <ul className="skills-list" aria-label={t("settingsTab.skills.listLabel", { defaultValue: "Imported skills" })}>
            {skills.map((skill) => (
              <SkillCard
                key={skill.id}
                skill={skill}
                busy={busy}
                confirming={confirmDelete === skill.id}
                onToggle={(enabled) => void run(async () => { await setSkillEnabled(skill.id, enabled); })}
                onAskDelete={() => setConfirmDelete(skill.id)}
                onCancelDelete={() => setConfirmDelete(null)}
                onDelete={() => void run(async () => { await deleteSkill(skill.id); setConfirmDelete(null); })}
              />
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}
