/**
 * D-68: a skill's security scan in Settings > Skills. The rating chip (monochrome icon plus a text label, so
 * the rating never rests on colour alone) and the expandable list of findings: file, line, rule and a short
 * explanation. The scan is local and rule-based, a heuristic and not a guarantee; the copy says so.
 */
import { useTranslation } from "react-i18next";
import type { SkillScan, SkillScanRating } from "../../api/skills";

function ScanIcon({ rating }: { rating: SkillScanRating }) {
  return (
    <svg className="skills-scan-chip__icon" width="14" height="14" viewBox="0 0 16 16" fill="none"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {rating === "high" ? (
        <>
          <path d="M8 1.8 14.6 13.5H1.4Z" />
          <path d="M8 6v3.6" />
          <path d="M8 11.6v.1" />
        </>
      ) : (
        <>
          <path d="M8 1.5 13.5 3.5v4.2c0 3.3-2.3 5.8-5.5 6.8-3.2-1-5.5-3.5-5.5-6.8V3.5Z" />
          {rating === "medium" ? (
            <>
              <path d="M8 5v3.4" />
              <path d="M8 10.6v.1" />
            </>
          ) : (
            <path d="m5.6 8 1.7 1.7 3.2-3.4" />
          )}
        </>
      )}
    </svg>
  );
}

export function useRatingLabel() {
  const { t } = useTranslation("common");
  return (rating: SkillScanRating) =>
    rating === "high"
      ? t("settingsTab.skills.scan.high", { defaultValue: "High" })
      : rating === "medium"
        ? t("settingsTab.skills.scan.medium", { defaultValue: "Medium" })
        : t("settingsTab.skills.scan.low", { defaultValue: "Low" });
}

export function SkillScanChip({ scan }: { scan: SkillScan }) {
  const { t } = useTranslation("common");
  const label = useRatingLabel();
  return (
    <span
      className={`skills-scan-chip skills-scan-chip--${scan.rating}`}
      title={t("settingsTab.skills.scan.chipTitle", {
        defaultValue: "Security scan by local rules. A heuristic, not a guarantee.",
      })}
    >
      <ScanIcon rating={scan.rating} />
      {t("settingsTab.skills.scan.chip", { rating: label(scan.rating), defaultValue: "Scan: {rating}" })}
    </span>
  );
}

/** The chip and the findings. ``open`` expands the list (the import preview does, a card does for High). */
export function SkillScanReport({ scan, open = false }: { scan: SkillScan; open?: boolean }) {
  const { t } = useTranslation("common");
  const label = useRatingLabel();
  return (
    <div className="skills-scan">
      <details open={open}>
        <summary>
          <SkillScanChip scan={scan} />{" "}
          <span className="muted-text">
            {t("settingsTab.skills.scan.findings", {
              count: scan.findings.length,
              defaultValue: "{count, plural, =0 {no findings} one {# finding} other {# findings}}",
            })}
          </span>
        </summary>
        {scan.findings.length > 0 ? (
          <ul className="skills-scan__findings">
            {scan.findings.map((finding, index) => (
              <li key={`${finding.file}:${finding.line}:${finding.rule}:${index}`} className="skills-scan__finding">
                <span className={`skills-scan__severity skills-scan__severity--${finding.severity}`}>{label(finding.severity)}</span>{" "}
                <span className="mono-text">
                  {finding.line > 0
                    ? t("settingsTab.skills.scan.where", { file: finding.file, line: finding.line, defaultValue: "{file}, line {line}" })
                    : finding.file}
                </span>{" "}
                <span className="muted-text mono-text">{finding.rule}</span>
                <div>{finding.message}</div>
                {finding.excerpt ? <div className="mono-text muted-text skills-scan__excerpt">{finding.excerpt}</div> : null}
              </li>
            ))}
          </ul>
        ) : null}
        {scan.truncated ? (
          <p className="muted-text">
            {t("settingsTab.skills.scan.truncated", { count: scan.findings.length, defaultValue: "Only the first {count} findings are shown." })}
          </p>
        ) : null}
        <p className="muted-text skills-scan__note">
          {t("settingsTab.skills.scan.disclaimer", {
            defaultValue:
              "A heuristic, not a guarantee: fixed rules look for risky wording, hidden characters, links, encoded text and bundled scripts. It runs on this computer and nothing leaves it. Low means no rule matched, not that the skill is safe.",
          })}
        </p>
      </details>
    </div>
  );
}
