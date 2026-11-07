import { useTranslation } from "react-i18next";
import { ALL_PROJECTS, type ProjectOption } from "../utils/galleryProject";

/** The gallery toolbar's Project filter (same label + select pattern as Model and Frame). Hidden when no output has a project. */
export function ProjectFilter({
  options,
  value,
  onChange,
}: {
  options: ProjectOption[];
  value: string;
  onChange: (value: string) => void;
}) {
  const { t } = useTranslation("studio");
  if (options.length === 0) return null;
  return (
    <label>
      {t("galleryProject.filter", { defaultValue: "Project" })}
      <select className="text-input" value={value} onChange={(event) => onChange(event.target.value)}>
        <option value={ALL_PROJECTS}>{t("galleryProject.all", { defaultValue: "All projects" })}</option>
        {options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.name}
          </option>
        ))}
      </select>
    </label>
  );
}

/** 'From project X' chip on a card made by a Storyboard batch. */
export function ProjectChip({ name }: { name?: string | null }) {
  const { t } = useTranslation("studio");
  if (!name) return null;
  return (
    <span className="badge subtle">{t("galleryProject.chip", { defaultValue: "From project {name}", name })}</span>
  );
}
