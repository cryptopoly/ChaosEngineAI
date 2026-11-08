import { useTranslation } from "react-i18next";

/** The plain statement shown at the top of the panel, always. */
export function CloneStatement() {
  const { t } = useTranslation("studio");
  return (
    <p className="voice-clone__statement">
      {t("voice.clone.statement", {
        defaultValue:
          "Only clone your own voice or the voice of someone who has given you permission. Not public figures, children or people who have died. Every clip made with a cloned voice is labelled as AI-made inside the file and carries an inaudible watermark; neither can be turned off.",
      })}
    </p>
  );
}
