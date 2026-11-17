/**
 * FU-191: the composer's Keep going control and chips (Studio chats only). Turning it on asks two things in the
 * same panel: how many nudges at most, and how renders are approved for this run (automatically for up to a
 * visible number of hours, or at each plan card). Nothing is preselected for renders, so the wider permission
 * is always an explicit click. While a run is active the chip shows the counter, the phase and the grant, with
 * Stop; a stopped run shows its reason, with Resume when it can go on. Also hosts the chat's older "Renders
 * auto-approved" chip (2 hours), which stays as it was when Keep going is off.
 */
import { useEffect, useId, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { KeepGoingGrant } from "../../api";
import type { KeepGoingGpuMode, KeepGoingRun } from "../../types";
import {
  KEEP_GOING_DEFAULT_CAP,
  KEEP_GOING_DEFAULT_HOURS,
  KEEP_GOING_MAX_CAP,
  KEEP_GOING_MAX_HOURS,
  KEEP_GOING_STOP_CODES,
  announcementKey,
  formatTimeLeft,
  isActive,
  useKeepGoing,
} from "./keepGoing";
import { AUTO_APPROVE_HOURS, useStudioAutoApprove, writeStudioAutoApprove } from "./studioChat";

export function KeepGoingIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d="M17 2l4 4-4 4" />
      <path d="M3 11v-1a4 4 0 0 1 4-4h14" />
      <path d="M7 22l-4-4 4-4" />
      <path d="M21 13v1a4 4 0 0 1-4 4H3" />
    </svg>
  );
}

/** The label on a nudge in the thread: written by the app, not by the user. */
export function KeepGoingEyebrow({ nudge, cap }: { nudge?: number | null; cap?: number | null }) {
  const { t } = useTranslation("chat");
  return (
    <span className="keep-going-eyebrow">
      <KeepGoingIcon />
      {typeof nudge === "number" && typeof cap === "number"
        ? t("keepGoing.thread.labelCount", { n: nudge, cap, defaultValue: "Keep going: app message {n} of {cap}" })
        : t("keepGoing.thread.label", { defaultValue: "Keep going: app message" })}
    </span>
  );
}

export interface ChatKeepGoingProps {
  sessionId: string | null | undefined;
  studioTools: boolean;
  messageCount: number;
  /** This window streams the chat's turn itself (its frames arrive here; no reload needed). */
  ownStreaming: boolean;
  onRefresh: (sessionId: string) => void;
  /** A new chat has no id yet: make one so the run can be armed before the brief is sent. */
  onCreateSession?: () => void | string | Promise<void | string>;
}

type Mode = "start" | "resume";

export function ChatKeepGoing({ sessionId, studioTools, messageCount, ownStreaming, onRefresh, onCreateSession }: ChatKeepGoingProps) {
  const { t } = useTranslation("chat");
  const controls = useKeepGoing(sessionId, messageCount, ownStreaming, onRefresh);
  const { run } = controls;
  const autoApprove = useStudioAutoApprove(sessionId);
  const [panel, setPanel] = useState<Mode | null>(null);
  const [dismissed, setDismissed] = useState<string | null>(null);
  const active = isActive(run);

  // Studio off ends the run (it needs the studio tools).
  const wasStudio = useRef(studioTools);
  useEffect(() => {
    if (wasStudio.current && !studioTools && active) void controls.studioOff();
    wasStudio.current = studioTools;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studioTools, active]);

  if (!studioTools || (!sessionId && !onCreateSession)) return null;
  const showStopped = run && run.state === "stopped" && dismissed !== run.runId + String(run.stoppedAt);
  return (
    <>
      {(
        <div className="keep-going">
          {!active ? (
            <button
              type="button"
              className="secondary-button composer-studio-toggle keep-going__toggle"
              aria-haspopup="dialog"
              aria-expanded={panel === "start"}
              onClick={() => setPanel(panel === "start" ? null : "start")}
              title={t("keepGoing.toggleTooltip", {
                defaultValue: "The app sends the next step after each turn until the film is done or a stop rule applies.",
              })}
            >
              <KeepGoingIcon />
              <span>{t("keepGoing.label", { defaultValue: "Keep going" })}</span>
            </button>
          ) : null}
          {panel ? (
            <KeepGoingPanel
              mode={panel}
              run={run}
              busy={controls.busy}
              onCancel={() => setPanel(null)}
              onSubmit={async (cap, grant, hours, gpuMode) => {
                if (panel === "start") {
                  const id = sessionId ?? (await onCreateSession?.()) ?? null;
                  if (id) await controls.start(cap, grant, hours, id, gpuMode);
                }
                else await controls.resume(grant, hours, cap);
                setPanel(null);
              }}
            />
          ) : null}
        </div>
      )}
      {active && run ? <ActiveChip run={run} busy={controls.busy} onStop={() => void controls.stop()} /> : null}
      {showStopped && run ? (
        <span className="keep-going-chip keep-going-chip--stopped">
          <KeepGoingIcon />
          <span className="keep-going-chip__text">
            {t("keepGoing.stopped", { defaultValue: "Keep going stopped:" })} {stopReason(t, run)}
          </span>
          {run.resumable ? (
            <button type="button" className="keep-going-chip__button" onClick={() => setPanel("resume")}
              aria-haspopup="dialog" aria-expanded={panel === "resume"}>
              {t("keepGoing.resume", { defaultValue: "Resume" })}
            </button>
          ) : null}
          <button type="button" className="keep-going-chip__button keep-going-chip__dismiss"
            onClick={() => setDismissed(run.runId + String(run.stoppedAt))}
            aria-label={t("keepGoing.dismiss", { defaultValue: "Hide this note" })}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
              strokeLinecap="round" aria-hidden="true" focusable="false"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </span>
      ) : null}
      {!active && autoApprove ? (
        <span className="composer-studio-auto" role="status">
          <span title={t("studioChat.auto.tooltip", {
            hours: AUTO_APPROVE_HOURS,
            defaultValue: "Renders in this chat run without a plan card. This ends after {hours} hours or when the app restarts.",
          })}>
            {t("studioChat.auto.label", { defaultValue: "Renders auto-approved" })}
          </span>
          <button type="button" className="composer-studio-auto__off" onClick={() => writeStudioAutoApprove(sessionId, false)}>
            {t("studioChat.auto.off", { defaultValue: "Turn off" })}
          </button>
        </span>
      ) : null}
      {controls.error ? <span className="keep-going-error" role="alert">{controls.error}</span> : null}
      <span className="sr-only" aria-live="polite" aria-atomic="true" key={announcementKey(run)}>
        {run ? announce(t, run) : ""}
      </span>
    </>
  );
}

type T = ReturnType<typeof useTranslation>["t"];

export function stopReason(t: T, run: KeepGoingRun): string {
  const code = run.stopCode ?? "";
  if (!(KEEP_GOING_STOP_CODES as readonly string[]).includes(code)) return run.stopReason ?? "";
  return t(`keepGoing.stop.${code}`, {
    cap: run.cap,
    minutes: run.stallMinutes ?? 20,
    detail: run.stopDetail ?? "",
    defaultValue: run.stopReason ?? code,
  });
}

function phaseText(t: T, run: KeepGoingRun): string {
  switch (run.phase) {
    case "armed":
      return t("keepGoing.phase.armed", { defaultValue: "send your brief" });
    case "awaiting_approval":
      return t("keepGoing.phase.awaitingApproval", { defaultValue: "waiting for your approval" });
    case "waiting_job":
      return t("keepGoing.phase.waitingJob", { defaultValue: "waiting for a job to finish" });
    case "nudging":
      return t("keepGoing.phase.nudging", { defaultValue: "sending the next step" });
    case "gpu_swap":
      return t("keepGoing.phase.gpuSwap", { defaultValue: "chat model unloaded while a render runs" });
    default:
      return t("keepGoing.phase.turn", { defaultValue: "the model is working" });
  }
}

function announce(t: T, run: KeepGoingRun): string {
  if (run.state === "stopped") return `${t("keepGoing.stopped", { defaultValue: "Keep going stopped:" })} ${stopReason(t, run)}`;
  return `${t("keepGoing.counter", { n: run.nudges, cap: run.cap, defaultValue: "nudge {n} of {cap}" })}, ${phaseText(t, run)}`;
}

function grantText(t: T, run: KeepGoingRun): string | null {
  if (!run.grant) return null;
  if (run.grant.mode === "manual") return t("keepGoing.grant.manualChip", { defaultValue: "you approve each plan" });
  const left = formatTimeLeft(run.grant.secondsLeft);
  if (!left) return t("keepGoing.grant.autoChip", { defaultValue: "renders auto-approved" });
  return t("keepGoing.grant.autoChipLeft", {
    hours: left.hours,
    minutes: left.minutes,
    defaultValue: "renders auto-approved, {hours} h {minutes} min left",
  });
}

export function ActiveChip({ run, busy, onStop }: { run: KeepGoingRun; busy: boolean; onStop: () => void }) {
  const { t } = useTranslation("chat");
  const grant = grantText(t, run);
  return (
    <span className={`keep-going-chip keep-going-chip--active${run.grant?.mode === "auto" ? " keep-going-chip--auto" : ""}`}>
      <KeepGoingIcon />
      <span className="keep-going-chip__text">
        <strong>{t("keepGoing.label", { defaultValue: "Keep going" })}</strong>
        {run.state === "armed" ? null : (
          <span>{t("keepGoing.counter", { n: run.nudges, cap: run.cap, defaultValue: "nudge {n} of {cap}" })}</span>
        )}
        <span>{phaseText(t, run)}</span>
        {run.waitingOn ? (
          <span className="keep-going-chip__waiting">
            {t("keepGoing.waitingOn", {
              label: run.waitingOn.label,
              minutes: run.waitingOn.minutes,
              defaultValue: "waiting on {label} ({minutes} min)",
            })}
          </span>
        ) : null}
        {grant ? <span className="keep-going-chip__grant">{grant}</span> : null}
      </span>
      <button type="button" className="keep-going-chip__button keep-going-chip__stop" onClick={onStop} disabled={busy}>
        <svg width="10" height="10" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="5" y="5" width="14" height="14" rx="2" fill="currentColor" /></svg>
        {t("keepGoing.stopButton", { defaultValue: "Stop" })}
      </button>
    </span>
  );
}

export function KeepGoingPanel({ mode, run, busy, onCancel, onSubmit }: {
  mode: Mode;
  run: KeepGoingRun | null;
  busy: boolean;
  onCancel: () => void;
  onSubmit: (cap: number, grant: KeepGoingGrant, hours: number, gpuMode: KeepGoingGpuMode) => Promise<void>;
}) {
  const { t } = useTranslation("chat");
  const id = useId();
  const [cap, setCap] = useState(mode === "resume" && run ? run.cap : KEEP_GOING_DEFAULT_CAP);
  const [grant, setGrant] = useState<KeepGoingGrant | null>(null);
  const [hours, setHours] = useState(KEEP_GOING_DEFAULT_HOURS);
  const [gpuMode, setGpuMode] = useState<KeepGoingGpuMode>(run?.gpuMode ?? "auto");
  const first = useRef<HTMLInputElement>(null);
  useEffect(() => {
    first.current?.focus();
  }, []);
  const title = mode === "start"
    ? t("keepGoing.panel.title", { defaultValue: "Keep going" })
    : t("keepGoing.panel.resumeTitle", { defaultValue: "Resume Keep going" });
  return (
    <div
      className="keep-going-panel"
      role="dialog"
      aria-modal="false"
      aria-labelledby={`${id}-title`}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onCancel();
        }
      }}
    >
      <strong id={`${id}-title`} className="keep-going-panel__title">{title}</strong>
      <p className="keep-going-panel__hint">
        {mode === "start"
          ? t("keepGoing.panel.hint", {
              defaultValue: "After each turn the app reads the project and sends the next step, until the film is done or a stop rule applies. Your next message is the brief.",
            })
          : t("keepGoing.panel.resumeHint", {
              defaultValue: "The next step is rebuilt from your brief and the saved project. Choose again how renders are approved.",
            })}
      </p>
      <label className="keep-going-panel__row" htmlFor={`${id}-cap`}>
        <span>{t("keepGoing.panel.cap", { defaultValue: "Stop after at most this many nudges" })}</span>
        <input
          ref={first}
          id={`${id}-cap`}
          type="number"
          min={1}
          max={KEEP_GOING_MAX_CAP}
          value={cap}
          onChange={(event) => setCap(Number(event.target.value))}
        />
      </label>
      <fieldset className="keep-going-panel__grant">
        <legend>{t("keepGoing.grant.legend", { defaultValue: "How are renders approved in this run?" })}</legend>
        <label className="keep-going-panel__choice">
          <input type="radio" name={`${id}-grant`} checked={grant === "manual"} onChange={() => setGrant("manual")} />
          <span>{t("keepGoing.grant.manual", { defaultValue: "I approve each plan card (the run waits for me)" })}</span>
        </label>
        <label className="keep-going-panel__choice">
          <input type="radio" name={`${id}-grant`} checked={grant === "auto"} onChange={() => setGrant("auto")} />
          <span>{t("keepGoing.grant.auto", { defaultValue: "Approve renders automatically for this run" })}</span>
        </label>
        <label className={`keep-going-panel__row keep-going-panel__hours${grant === "auto" ? "" : " is-disabled"}`} htmlFor={`${id}-hours`}>
          <span>{t("keepGoing.grant.hours", { max: KEEP_GOING_MAX_HOURS, defaultValue: "for up to this many hours (at most {max})" })}</span>
          <input
            id={`${id}-hours`}
            type="number"
            min={1}
            max={KEEP_GOING_MAX_HOURS}
            value={hours}
            disabled={grant !== "auto"}
            onChange={(event) => setHours(Number(event.target.value))}
          />
        </label>
        <small className="keep-going-panel__note">
          {t("keepGoing.grant.note", {
            defaultValue: "Automatic approval ends when the run stops, when the hours run out (the run then pauses) or when the app restarts.",
          })}
        </small>
      </fieldset>
      {mode === "start" ? (
        <label className="keep-going-panel__row" htmlFor={`${id}-gpu`}>
          <span>
            {t("keepGoing.gpu.label", { defaultValue: "Free GPU memory while rendering" })}
            <small className="keep-going-panel__note keep-going-panel__note--block">
              {t("keepGoing.gpu.hint", {
                defaultValue: "Auto unloads the chat model during a render only when both do not fit, then loads it again.",
              })}
            </small>
          </span>
          <select id={`${id}-gpu`} value={gpuMode} onChange={(event) => setGpuMode(event.target.value as KeepGoingGpuMode)}>
            <option value="auto">{t("keepGoing.gpu.auto", { defaultValue: "Auto" })}</option>
            <option value="always">{t("keepGoing.gpu.always", { defaultValue: "Always" })}</option>
            <option value="never">{t("keepGoing.gpu.never", { defaultValue: "Never" })}</option>
          </select>
        </label>
      ) : null}
      <div className="keep-going-panel__actions">
        <button type="button" className="secondary-button" onClick={onCancel}>
          {t("keepGoing.panel.cancel", { defaultValue: "Cancel" })}
        </button>
        <button
          type="button"
          className="primary-button"
          disabled={busy || grant === null}
          title={grant === null ? t("keepGoing.grant.pick", { defaultValue: "Choose how renders are approved first" }) : undefined}
          onClick={() => grant && void onSubmit(cap, grant, hours, gpuMode)}
        >
          {mode === "start"
            ? t("keepGoing.panel.start", { defaultValue: "Turn on" })
            : t("keepGoing.resume", { defaultValue: "Resume" })}
        </button>
      </div>
    </div>
  );
}
