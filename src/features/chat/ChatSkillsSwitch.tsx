/**
 * D-62: the composer's per-chat Skills switch. Shown only when at least one skill is switched on in
 * Settings; when on, the chat model sees those skills' names and descriptions and can read their text.
 */
import { useTranslation } from "react-i18next";
import { useChatSkillsEnabled, useEnabledSkillSummary, writeChatSkillsEnabled } from "./skillsChat";

export function ChatSkillsSwitch({ sessionId }: { sessionId: string | null | undefined }) {
  const { t } = useTranslation("chat");
  const enabled = useChatSkillsEnabled(sessionId);
  const summary = useEnabledSkillSummary();
  const { count } = summary;
  if (!sessionId || count === 0) return null;
  const ratings = t("skillsChat.switch.ratings", {
    high: summary.high,
    medium: summary.medium,
    low: summary.low,
    defaultValue: "Security scan: {high} High, {medium} Medium, {low} Low.",
  });
  return (
    <button
      className={`secondary-button composer-studio-toggle${enabled ? " active-toggle" : ""}`}
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={() => writeChatSkillsEnabled(sessionId, !enabled)}
      title={`${
        enabled
          ? t("skillsChat.switch.onTooltip", {
              count,
              defaultValue: "Skills on for this chat: the model sees the names of {count, plural, one {# skill} other {# skills}} and reads one when it fits. Scripts in skills never run.",
            })
          : t("skillsChat.switch.offTooltip", {
              defaultValue: "Offer the skills switched on in Settings to the model in this chat",
            })
      } ${ratings}`}
    >
      {enabled ? t("skillsChat.switch.on", { defaultValue: "Skills ON" }) : t("skillsChat.switch.label", { defaultValue: "Skills" })}
    </button>
  );
}
