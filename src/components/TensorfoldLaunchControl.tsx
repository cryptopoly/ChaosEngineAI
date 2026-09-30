import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TensorfoldJobState } from "../api";
import { InstallLogPanel } from "./InstallLogPanel";
import {
  tensorfoldMemoryShortfall,
  type TensorfoldLaunchInfo,
} from "./tensorfoldSupport";

interface TensorfoldLaunchControlProps {
  launch: TensorfoldLaunchInfo;
  /** The shared speculative-decoding launch flag the backend reads. */
  speculativeDecoding: boolean;
  onSpeculativeChange: (enabled: boolean) => void;
  onInstall?: () => void;
  installing?: boolean;
  job?: TensorfoldJobState | null;
  totalMemoryGb?: number;
}

/**
 * Launch-settings block for the TensorFold engine (exact speculative
 * decoding, Apple Silicon only — the caller hides it elsewhere).
 *
 * Shown only for checkpoints the backend registry says TensorFold serves
 * (FU-034: no control for a model that no install could help). It binds to
 * the same ``speculativeDecoding`` flag DFlash / MTPLX use; the backend
 * picks the lane, so this block replaces those toggles whenever TensorFold
 * is installed rather than showing two boxes for one flag.
 */
export function TensorfoldLaunchControl({
  launch,
  speculativeDecoding,
  onSpeculativeChange,
  onInstall,
  installing = false,
  job = null,
  totalMemoryGb,
}: TensorfoldLaunchControlProps) {
  const { t } = useTranslation("runtime");
  const [expanded, setExpanded] = useState(false);
  const { available, support, version } = launch;
  const exclusive = support.tier === "exclusive";
  const shortfall = tensorfoldMemoryShortfall(support, totalMemoryGb);

  // A finished install is the moment the user asked for exact drafts, so
  // turn them on once; they can untick it afterwards and it stays off.
  const previousPhase = useRef<string | null>(job?.phase ?? null);
  useEffect(() => {
    const phase = job?.phase ?? null;
    if (phase === "done" && previousPhase.current !== null && previousPhase.current !== "done") {
      onSpeculativeChange(true);
    }
    previousPhase.current = phase;
  }, [job?.phase, onSpeculativeChange]);

  return (
    <>
      <div className="check-row">
        <label
          className="check-row"
          style={{ margin: 0 }}
          title={t("tensorfold.tooltip", {
            defaultValue:
              "TensorFold: exact speculative decoding. Drafts from the model's MTP head and/or a draft model are verified against the target model, so replies are identical to serial decoding — only faster.",
          })}
        >
          <input
            type="checkbox"
            checked={speculativeDecoding && available}
            disabled={!available}
            onChange={(event) => onSpeculativeChange(event.target.checked)}
          />
          <span>{t("tensorfold.label", { defaultValue: "TensorFold" })}</span>
        </label>
        {exclusive ? (
          <span className="muted-text" style={{ fontSize: "0.85em" }}>
            {t("tensorfold.requiredNote", { defaultValue: "required for this model" })}
          </span>
        ) : null}
        {!available && onInstall ? (
          <button
            type="button"
            className="cache-strategy-install-btn"
            disabled={installing}
            onClick={onInstall}
          >
            {installing
              ? t("tensorfold.installing", { defaultValue: "Installing..." })
              : t("tensorfold.installButton", { defaultValue: "Install TensorFold" })}
          </button>
        ) : null}
        <button
          type="button"
          className="cache-strategy-info-btn"
          onClick={() => setExpanded((open) => !open)}
          title={t("tensorfold.aboutTitle", { defaultValue: "About TensorFold speculative decoding" })}
        >
          i
        </button>
      </div>
      {shortfall && support.minMemoryGb ? (
        <p className="warning-text" style={{ margin: "4px 0" }}>
          {t("tensorfold.memoryShortfall", {
            defaultValue: "This checkpoint needs a Mac with at least {{needed}} GB of memory; this one has {{have}} GB.",
            needed: support.minMemoryGb,
            have: Math.round(totalMemoryGb ?? 0),
          })}
        </p>
      ) : null}
      {/* Show the terminal during the install and on error; hide it once
          done so a finished install does not reappear on every modal open. */}
      {job && job.phase !== "idle" && job.phase !== "done" ? (
        <InstallLogPanel job={job} variant="tensorfold" />
      ) : null}
      {expanded ? (
        <div className="cache-strategy-info-panel" style={{ marginTop: 4 }}>
          <p>
            {t("tensorfold.body", {
              defaultValue:
                "TensorFold is a local inference engine with hand-written kernels per model family. It drafts several tokens at a time " +
                "(from the model's own MTP head and, for some families, a small draft model) and checks every draft against the target " +
                "model, so the reply is byte-identical to one-token-at-a-time decoding.",
            })}
          </p>
          <div className="cache-strategy-meta">
            <span className="cache-strategy-meta-label">{t("tensorfold.requiresLabel", { defaultValue: "Requires:" })}</span>
            <span>
              {t("tensorfold.requiresBody", {
                defaultValue: "Apple Silicon and a separate TensorFold Python environment (installed from this panel). Serves only the checkpoints it is tested with.",
              })}
            </span>
          </div>
          {support.minMemoryGb ? (
            <div className="cache-strategy-meta">
              <span className="cache-strategy-meta-label">{t("tensorfold.memoryLabel", { defaultValue: "Memory:" })}</span>
              <span>
                {t("tensorfold.memoryBody", {
                  defaultValue: "{{needed}} GB of unified memory or more.",
                  needed: support.minMemoryGb,
                })}
              </span>
            </div>
          ) : null}
          <div className="cache-strategy-meta">
            <span className="cache-strategy-meta-label">{t("tensorfold.statusLabel", { defaultValue: "Status:" })}</span>
            <span>
              {available
                ? t("tensorfold.statusInstalled", {
                    defaultValue: "Installed{{version}} — active when speculative decoding is ticked.",
                    version: version ? ` (${version})` : "",
                  })
                : t("tensorfold.statusNotInstalled", {
                    defaultValue: "Not installed. Click Install TensorFold to set up its separate environment.",
                  })}
            </span>
          </div>
          {exclusive ? (
            <div className="cache-strategy-meta">
              <span className="cache-strategy-meta-label">{t("tensorfold.exclusiveLabel", { defaultValue: "Note:" })}</span>
              <span>
                {t("tensorfold.exclusiveBody", {
                  defaultValue: "No other engine in the app can load this model, so it always runs on TensorFold; the box switches its drafting on and off.",
                })}
              </span>
            </div>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
