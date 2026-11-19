import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { formatDuration } from "../../components/RunEstimateStrip";
import type { StudioPlan, StudioPlanEstimate, StudioPlanItem, StudioTier } from "../../types";
import { formatSeconds, submitPlanDecision, type PlanAction } from "./studioChat";

interface StudioPlanCardProps {
  plan: StudioPlan;
  sessionId: string | undefined;
  /** "Edit by chat": the calls are declined; the composer gets focus so the user can say what to change. */
  onEditByChat?: () => void;
}

const KIND_LABELS: Record<string, string> = {
  image: "Image",
  video: "Video",
  speech: "Speech",
  music: "Music",
  sound: "Sound",
  storyboard: "Storyboard",
  film: "Film",
  captions: "Captions",
};

/** Every argument that changes what gets made or what it costs, as short labelled facts. */
function ItemFacts({ item, tier }: { item: StudioPlanItem; tier?: StudioTier }) {
  const { t } = useTranslation("chat");
  // A tier picked on the card shows that tier's size and time; the rest of the call is unchanged.
  const picked = tier && tier !== item.tier ? item.tierOptions?.[tier] : undefined;
  const time = formatSeconds((picked ? picked.estimatedSeconds : item.estimatedSeconds) ?? null);
  const size = picked?.size ?? item.size;
  const facts: string[] = [];
  if (item.model) facts.push(item.model);
  if (size) facts.push(size);
  if (typeof item.frames === "number") facts.push(t("studioChat.plan.frames", { count: item.frames, defaultValue: "{count} frames" }));
  if (typeof item.fps === "number") facts.push(t("studioChat.plan.fps", { fps: item.fps, defaultValue: "{fps} fps" }));
  if (typeof item.steps === "number") facts.push(t("studioChat.plan.steps", { count: item.steps, defaultValue: "{count} steps" }));
  if (typeof item.variations === "number" && item.variations > 1) {
    facts.push(t("studioChat.plan.variations", { count: item.variations, defaultValue: "{count} takes" }));
  }
  if (item.instrumental) facts.push(t("studioChat.plan.instrumental", { defaultValue: "instrumental" }));
  if (typeof item.shots === "number") facts.push(t("studioChat.plan.shots", { count: item.shots, defaultValue: "{count} shots" }));
  if (item.what) facts.push(t("studioChat.plan.what", { what: item.what, defaultValue: "makes: {what}" }));
  if (item.preview) facts.push(t("studioChat.plan.preview", { defaultValue: "preview" }));
  if (item.force) facts.push(t("studioChat.plan.force", { defaultValue: "includes locked shots" }));
  if (time) {
    facts.push(item.estimateAtRecommended && !picked
      ? t("studioChat.plan.timeRecommended", { time, defaultValue: "about {time} at the recommended size" })
      : t("studioChat.plan.time", { time, defaultValue: "about {time}" }));
  }
  if (typeof item.peakGb === "number") facts.push(t("studioChat.plan.memory", { gb: item.peakGb.toFixed(0), defaultValue: "about {gb} GB" }));
  return (
    <>
      {facts.map((fact, i) => <span key={i}>{fact}</span>)}
      {item.inputs?.length ? (
        <span className="studio-plan-card__inputs">
          {t("studioChat.plan.inputs", { files: item.inputs.join(", "), defaultValue: "input images: {files}" })}
        </span>
      ) : null}
    </>
  );
}

type TFn = (key: string, options: Record<string, unknown>) => string;

const TIERS: StudioTier[] = ["draft", "standard", "final"];

/** FU-241: Draft, Standard or Final for the single images and clips on the card, with each one's size and time. */
function QualityPicker({ item, tier, onPick, disabled }: {
  item: StudioPlanItem;
  tier: StudioTier;
  onPick: (tier: StudioTier) => void;
  disabled: boolean;
}) {
  const { t } = useTranslation("chat");
  const labels: Record<StudioTier, string> = {
    draft: t("studioChat.plan.tierDraft", { defaultValue: "Draft" }),
    standard: t("studioChat.plan.tierStandard", { defaultValue: "Standard" }),
    final: t("studioChat.plan.tierFinal", { defaultValue: "Final" }),
  };
  return (
    <div className="studio-plan-card__quality" role="radiogroup" aria-label={t("studioChat.plan.qualityLabel", { defaultValue: "Quality" })}>
      {TIERS.map((name) => {
        const option = item.tierOptions?.[name];
        const time = formatSeconds(option?.estimatedSeconds ?? null);
        return (
          <button key={name} type="button" role="radio" aria-checked={tier === name} disabled={disabled}
            className={`secondary-button studio-plan-card__tier${tier === name ? " is-active" : ""}`}
            onClick={() => onPick(name)}>
            <strong>{labels[name]}</strong>
            <small>{[option?.size, time].filter(Boolean).join(" · ")}</small>
          </button>
        );
      })}
    </div>
  );
}

/**
 * S7: time per shot (still and clip) and the total before a storyboard render is approved. The time, unknown
 * and basis words are the Storyboard run bar's own (studio namespace), so both screens say the same thing.
 */
function StoryboardEstimate({ estimate }: { estimate: StudioPlanEstimate }) {
  const { t } = useTranslation("chat");
  const { t: ts } = useTranslation("studio");
  const tf = ts as unknown as TFn;
  const about = (seconds: number | null | undefined) =>
    typeof seconds === "number" && seconds > 0
      ? ts("estimate.time", { defaultValue: "About {time}", time: formatDuration(tf, seconds) })
      : t("studioChat.plan.unknownTime", { defaultValue: "time unknown" });
  const basis = estimate.basis;
  return (
    <div className="studio-plan-card__estimate">
      <span className="studio-plan-card__estimate-label">{t("studioChat.plan.perShot", { defaultValue: "Time per shot" })}</span>
      <ul className="studio-plan-card__estimate-shots">
        {estimate.shots.map((shot, index) => (
          <li key={shot.id}>
            <span className="studio-plan-card__estimate-title">{shot.title || t("studioChat.plan.shotNumber", { n: index + 1, defaultValue: "Shot {n}" })}</span>
            {shot.still !== undefined ? <span>{t("studioChat.plan.shotStill", { time: about(shot.still), defaultValue: "still: {time}" })}</span> : null}
            {shot.clip !== undefined ? <span>{t("studioChat.plan.shotClip", { time: about(shot.clip), defaultValue: "clip: {time}" })}</span> : null}
          </li>
        ))}
        {estimate.moreShots ? (
          <li className="muted-text">{t("studioChat.plan.moreShots", { count: estimate.moreShots, defaultValue: "and {count} more" })}</li>
        ) : null}
      </ul>
      <div className="studio-plan-card__estimate-total">
        <strong>{t("studioChat.plan.total", { defaultValue: "Total" })}</strong>
        <span>
          {estimate.seconds != null && estimate.seconds > 0
            ? ts("estimate.time", { defaultValue: "About {time}", time: formatDuration(tf, estimate.seconds) })
            : ts("estimate.timeUnknown", { defaultValue: "Time: unknown until the first run" })}
        </span>
        {basis ? (
          <span
            className={`badge subtle run-estimate__basis run-estimate__basis--${basis}`}
            title={basis === "measured"
              ? ts("estimate.measuredTitle", { defaultValue: "Timed on this machine for this model and preset." })
              : ts("estimate.estimatedTitle", { defaultValue: "From the model card or a similar machine; not timed here yet." })}
          >
            {basis === "measured"
              ? ts("estimate.measured", { defaultValue: "Measured" })
              : ts("estimate.estimated", { defaultValue: "Estimated" })}
          </span>
        ) : null}
      </div>
      {estimate.missing.length ? (
        <ul className="studio-plan-card__estimate-missing">
          {estimate.missing.map((line) => <li key={line}>{line}</li>)}
        </ul>
      ) : null}
    </div>
  );
}

function ItemPrompt({ item }: { item: StudioPlanItem }) {
  const { t } = useTranslation("chat");
  const [open, setOpen] = useState(false);
  return (
    <span className="studio-plan-card__prompt">
      {open && item.promptFull ? item.promptFull : item.prompt || item.tool}
      {item.promptFull ? (
        <button type="button" className="studio-plan-card__more" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? t("studioChat.plan.less", { defaultValue: "less" }) : t("studioChat.plan.more", { defaultValue: "more" })}
        </button>
      ) : null}
      {item.lyrics ? <span className="studio-plan-card__lyrics">{item.lyrics}</span> : null}
    </span>
  );
}

/**
 * D-60: what the chat model wants to make, before anything renders. Approval is only these buttons:
 * Approve all, Approve this (one row, the rest declined), Deny, Edit by chat. The decision covers the
 * listed calls of this turn; "Always allow renders in this chat" (off by default) skips the card for this
 * chat until it is turned off, for at most a few hours and never past an app restart.
 */
export function StudioPlanCard({ plan, sessionId, onEditByChat }: StudioPlanCardProps) {
  const { t } = useTranslation("chat");
  const [alwaysAllow, setAlwaysAllow] = useState(false);
  const rendered = plan.items.find((i) => i.tierOptions);
  const [tier, setTier] = useState<StudioTier>(rendered?.tier ?? "standard");
  const [keepTier, setKeepTier] = useState(false);
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const headingRef = useRef<HTMLElement>(null);
  const pending = plan.status === "pending" && !sent;
  const many = plan.items.length > 1;
  const chatId = plan.sessionId ?? sessionId;

  // A new plan takes focus so keyboard and screen-reader users find it.
  useEffect(() => {
    if (plan.status === "pending") headingRef.current?.focus();
  }, [plan.planId, plan.status]);

  async function choose(action: PlanAction) {
    setSending(true);
    setError(null);
    try {
      await submitPlanDecision(plan, action, alwaysAllow, chatId, rendered ? { tier, keep: keepTier } : undefined);
      setSent(true);
      if (action.type === "edit") onEditByChat?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSending(false);
    }
  }

  const statusText =
    plan.status === "unanswered"
      ? t("studioChat.plan.unanswered", { defaultValue: "Not answered. Nothing was made." })
      : plan.status === "edit"
      ? t("studioChat.plan.edited", { defaultValue: "You asked to change the plan. Nothing was made." })
      : plan.status === "decided"
      ? t("studioChat.plan.decided", {
          approved: Object.values(plan.decisions ?? {}).filter((d) => d === "approve").length,
          total: plan.items.length,
          defaultValue: "Approved {approved} of {total}.",
        })
      : sent
      ? t("studioChat.plan.sent", { defaultValue: "Sent. Waiting for the model..." })
      : null;

  const memory = plan.memory;
  const free = typeof memory?.freeGb === "number" ? memory.freeGb.toFixed(0) : null;
  const memoryLine = free === null
    ? null
    : memory?.chatModel
    ? t("studioChat.plan.memoryLoaded", { model: memory.chatModel, gb: free, defaultValue: "{model} stays loaded: about {gb} GB free." })
    : t("studioChat.plan.memoryFree", { gb: free, defaultValue: "About {gb} GB free." });

  return (
    <section className="studio-plan-card" aria-label={t("studioChat.plan.title", { defaultValue: "Plan to approve" })}>
      <header className="studio-plan-card__header">
        <strong ref={headingRef} tabIndex={-1}>{t("studioChat.plan.title", { defaultValue: "Plan to approve" })}</strong>
        <small>
          {t("studioChat.plan.subtitle", {
            defaultValue: "Nothing renders until you approve. This covers these items in this turn only.",
          })}
        </small>
      </header>
      <ol className="studio-plan-card__items">
        {plan.items.map((item, index) => {
          const decision = plan.decisions?.[item.callId];
          const kind = t(`studioChat.kind.${item.kind}`, { defaultValue: KIND_LABELS[item.kind] ?? item.kind });
          return (
            <li key={`${item.callId}-${index}`} className="studio-plan-card__item" data-decision={decision ?? "pending"}>
              <div className="studio-plan-card__item-main">
                <span className="studio-plan-card__kind">{kind}</span>
                <ItemPrompt item={item} />
              </div>
              <div className="studio-plan-card__meta">
                <ItemFacts item={item} tier={tier} />
                {decision ? (
                  <span className={`studio-plan-card__decision studio-plan-card__decision--${decision}`}>
                    {decision === "approve"
                      ? t("studioChat.plan.approved", { defaultValue: "Approved" })
                      : t("studioChat.plan.declined", { defaultValue: "Declined" })}
                  </span>
                ) : null}
              </div>
              {item.estimate ? <StoryboardEstimate estimate={item.estimate} /> : null}
              {pending && item.tierOptions ? <QualityPicker item={item} tier={tier} onPick={setTier} disabled={sending} /> : null}
              {item.qualityNotes?.length ? (
                <ul className="studio-plan-card__quality-notes">{item.qualityNotes.map((note) => <li key={note}>{note}</li>)}</ul>
              ) : null}
              {pending && many ? (
                <button type="button" className="secondary-button studio-plan-card__one" disabled={sending}
                  aria-label={t("studioChat.plan.approveThisLabel", {
                    kind, prompt: item.prompt.slice(0, 60), defaultValue: "Approve only this {kind}: {prompt}",
                  })}
                  onClick={() => void choose({ type: "approveOne", callId: item.callId })}>
                  {t("studioChat.plan.approveThis", { defaultValue: "Approve this" })}
                </button>
              ) : null}
            </li>
          );
        })}
      </ol>
      {memoryLine ? (
        <p className={`studio-plan-card__memory studio-plan-card__memory--${memory?.level ?? "ok"}`} role={memory?.level === "risky" ? "alert" : undefined}>
          {memoryLine}
          {memory?.advice ? ` ${memory.advice}` : ""}
        </p>
      ) : null}
      {pending ? (
        <>
          {rendered ? (
            <label className="studio-plan-card__always">
              <input type="checkbox" checked={keepTier} onChange={(e) => setKeepTier(e.target.checked)} />
              {t("studioChat.plan.keepTier", { defaultValue: "Use for the rest of this chat" })}
            </label>
          ) : null}
          <label className="studio-plan-card__always">
            <input type="checkbox" checked={alwaysAllow} onChange={(e) => setAlwaysAllow(e.target.checked)} />
            {t("studioChat.plan.alwaysAllow", { defaultValue: "Always allow renders in this chat" })}
          </label>
          <div className="studio-plan-card__actions">
            <button type="button" className="primary-button" disabled={sending} onClick={() => void choose({ type: "approveAll" })}>
              {many
                ? t("studioChat.plan.approveAll", { defaultValue: "Approve all" })
                : t("studioChat.plan.approve", { defaultValue: "Approve" })}
            </button>
            <button type="button" className="secondary-button" disabled={sending} onClick={() => void choose({ type: "deny" })}>
              {t("studioChat.plan.deny", { defaultValue: "Deny" })}
            </button>
            <button type="button" className="secondary-button" disabled={sending} onClick={() => void choose({ type: "edit" })}>
              {t("studioChat.plan.edit", { defaultValue: "Edit by chat" })}
            </button>
          </div>
        </>
      ) : null}
      {statusText ? <p className="studio-plan-card__status" role="status">{statusText}</p> : null}
      {error ? <p className="studio-plan-card__error" role="alert">{error}</p> : null}
    </section>
  );
}
