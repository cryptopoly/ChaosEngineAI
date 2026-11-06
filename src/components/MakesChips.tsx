/** What a model makes, as one to three small chips. Labels come from i18n. */
import { useTranslation } from "react-i18next";

export type MakesKind = "songs" | "instrumental" | "sfx" | "loops";

const DEFAULT_LABELS: Record<MakesKind, string> = {
  songs: "Songs with vocals",
  instrumental: "Instrumental",
  sfx: "Sound effects",
  loops: "Loops",
};

export function MakesChips({ makes }: { makes: MakesKind[] }) {
  const { t } = useTranslation("studio");
  if (makes.length === 0) return null;
  return (
    <ul className="makes-chips">
      {makes.map((kind) => (
        <li key={kind} className="makes-chip">
          {t(`models.makes.${kind}`, { defaultValue: DEFAULT_LABELS[kind] })}
        </li>
      ))}
    </ul>
  );
}
