/**
 * Take compare (S10): two takes of one shot side by side, stills or clips, with the clips played in step (one
 * Play, one scrubber, sound from one side), what made each take (pass, models, seeds, size) with the differences
 * marked, and "Use this one" (the pick_take op). A modal dialog: focus stays inside, Escape closes it, and on a
 * narrow screen the panes stack and the dialog opens as a sheet from the bottom.
 */
import { useEffect, useRef, useState, type RefObject } from "react";
import { useTranslation } from "react-i18next";
import { CloseIcon, PauseIcon, PlayIcon } from "../../components/icons/ActionIcons";
import { useDialogFocus } from "../../hooks/useDialogFocus";
import type { StoryboardShot, StoryboardSize, StoryboardTake, StoryboardTakeQuality } from "../../types";
import { ProjectImage, useProjectFileUrl } from "./ProjectMedia";
import { QualityTag } from "./QualityTag";
import { takeQualityOf } from "./qualityText";
import {
  compareRows,
  compareSummary,
  compareView,
  defaultComparePair,
  syncTime,
  type CompareKey,
} from "./remakeLogic";

type Side = "left" | "right";
type Sound = Side | "off";

export interface TakeCompareProps {
  projectId: string;
  shot: StoryboardShot;
  /** One-based shot number, for the title. */
  number: number;
  disabled?: boolean;
  /** Makes a take the active one; resolves true when it landed. */
  onPick: (takeId: string) => Promise<boolean> | boolean | void;
  onClose: () => void;
  /** takeId to what it was made at (`guide.quality.takes[shotId]`), for the Draft tag. */
  quality?: Record<string, StoryboardTakeQuality>;
  /** The film's size, named in the Draft tag's explanation. */
  target?: StoryboardSize | null;
}

function aspectOf(take: StoryboardTake | undefined): string {
  const w = Number(take?.settings?.width);
  const h = Number(take?.settings?.height);
  return w > 0 && h > 0 ? `${w} / ${h}` : "16 / 9";
}

function formatTime(seconds: number): string {
  return (Math.round((Number.isFinite(seconds) ? seconds : 0) * 10) / 10).toFixed(1);
}

function CompareVideo({ projectId, take, videoRef, muted, label, onMeta }: {
  projectId: string;
  take: StoryboardTake;
  videoRef: RefObject<HTMLVideoElement>;
  muted: boolean;
  label: string;
  onMeta: () => void;
}) {
  const url = useProjectFileUrl(projectId, take.clip, take.hashes.clip);
  if (!url) return <span className="project-media-empty take-compare__media" aria-hidden="true" />;
  return (
    <video
      ref={videoRef}
      className="take-compare__media"
      src={url}
      preload="auto"
      muted={muted}
      playsInline
      aria-label={label}
      onLoadedMetadata={onMeta}
    />
  );
}

export function TakeCompare({ projectId, shot, number, disabled, onPick, onClose, quality, target }: TakeCompareProps) {
  const { t } = useTranslation("studio");
  const dialog = useRef<HTMLDivElement>(null);
  useDialogFocus(dialog, { onEscape: onClose });
  const pair = defaultComparePair(shot);
  const [ids, setIds] = useState<Record<Side, string>>(() => ({ left: pair?.[0] ?? "", right: pair?.[1] ?? "" }));
  const left = shot.takes.find((take) => take.id === ids.left);
  const right = shot.takes.find((take) => take.id === ids.right);
  const [view, setView] = useState<"clip" | "still">(() => (left && right ? compareView(left, right) : "still"));
  const [sound, setSound] = useState<Sound>("left");
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [announce, setAnnounce] = useState("");
  const [picking, setPicking] = useState<string | null>(null);
  const videos: Record<Side, RefObject<HTMLVideoElement>> = { left: useRef<HTMLVideoElement>(null), right: useRef<HTMLVideoElement>(null) };
  const numberOf = (id: string) => shot.takes.findIndex((take) => take.id === id) + 1;

  /** The longer clip leads, so neither is cut short; the other follows it. */
  function leaderSide(): Side {
    const l = videos.left.current?.duration ?? 0;
    const r = videos.right.current?.duration ?? 0;
    return (Number.isFinite(r) ? r : 0) > (Number.isFinite(l) ? l : 0) ? "right" : "left";
  }

  function readDuration() {
    const lengths = [videos.left.current?.duration, videos.right.current?.duration].filter(
      (value): value is number => typeof value === "number" && Number.isFinite(value),
    );
    setDuration(lengths.length ? Math.max(...lengths) : 0);
  }

  function pauseBoth() {
    videos.left.current?.pause();
    videos.right.current?.pause();
    setPlaying(false);
  }

  function seekBoth(seconds: number) {
    for (const side of ["left", "right"] as Side[]) {
      const video = videos[side].current;
      if (!video) continue;
      const end = Number.isFinite(video.duration) ? video.duration : seconds;
      video.currentTime = Math.min(seconds, end);
    }
    setTime(seconds);
  }

  function playBoth() {
    const lead = videos[leaderSide()].current;
    if (lead && lead.duration && lead.currentTime >= lead.duration - 0.05) seekBoth(0);
    for (const side of ["left", "right"] as Side[]) void videos[side].current?.play().catch(() => undefined);
    setPlaying(true);
  }

  // While playing, the follower is kept with the leader; when the leader ends, both go back to the start and play on.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      const leadSide = leaderSide();
      const lead = videos[leadSide].current;
      const follow = videos[leadSide === "left" ? "right" : "left"].current;
      if (lead) {
        if (lead.ended) {
          seekBoth(0);
          void lead.play().catch(() => undefined);
          void follow?.play().catch(() => undefined);
        } else if (follow) {
          const next = syncTime(lead.currentTime, follow.currentTime, follow.duration);
          if (next !== null) follow.currentTime = next;
          if (follow.paused && !follow.ended && lead.currentTime < (follow.duration || 0) - 0.05) {
            void follow.play().catch(() => undefined);
          }
        }
        setTime(lead.currentTime);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]); // the refs are read live

  // A different pair or view starts again from the top.
  useEffect(() => {
    pauseBoth();
    setTime(0);
    setDuration(0);
  }, [ids.left, ids.right, view]);

  if (!left || !right) return null;

  const hasClips = Boolean(left.clip || right.clip);
  const hasStills = Boolean(left.still || right.still);
  const clipsShown = view === "clip" && hasClips;
  const rows = compareRows(left, right);
  const summary = compareSummary(left, right);
  const takes: Record<Side, StoryboardTake> = { left, right };

  const factLabels: Record<CompareKey, string> = {
    pass: t("board.compare.facts.pass", { defaultValue: "Pass" }),
    stillModel: t("board.compare.facts.stillModel", { defaultValue: "Still model" }),
    stillSeed: t("board.compare.facts.stillSeed", { defaultValue: "Still seed" }),
    clipModel: t("board.compare.facts.clipModel", { defaultValue: "Clip model" }),
    clipSeed: t("board.compare.facts.clipSeed", { defaultValue: "Clip seed" }),
    size: t("board.compare.facts.size", { defaultValue: "Size" }),
    length: t("board.compare.facts.length", { defaultValue: "Length (s)" }),
  };
  const passLabel = (take: StoryboardTake) =>
    take.renderPass === "draft"
      ? t("board.compare.draft", { defaultValue: "Draft" })
      : t("board.compare.final", { defaultValue: "Final" });
  const factValue = (key: CompareKey, value: string | null, take: StoryboardTake) => {
    // A still-only take (a draft, say) has no clip facts to record: say so instead of "Not recorded".
    if (value === null && !take.clip && (key === "clipModel" || key === "clipSeed" || key === "length"))
      return t("board.compare.noClip", { defaultValue: "No clip" });
    if (value === null) return t("board.compare.notRecorded", { defaultValue: "Not recorded" });
    if (key === "pass") return value === "draft" ? t("board.compare.draft", { defaultValue: "Draft" }) : t("board.compare.final", { defaultValue: "Final" });
    return value;
  };
  const summaryText = {
    sameStill: t("board.compare.summary.sameStill", { defaultValue: "Same still, a different clip." }),
    sameClip: t("board.compare.summary.sameClip", { defaultValue: "Same clip, a different still. Remake the clip to match the new still." }),
    identical: t("board.compare.summary.identical", { defaultValue: "These takes hold the same still and clip." }),
    different: t("board.compare.summary.different", { defaultValue: "A different still and clip." }),
  }[summary];
  const takeName = (take: StoryboardTake) => t("board.takes.take", { defaultValue: "Take {n}", n: numberOf(take.id) });

  async function use(take: StoryboardTake) {
    setPicking(take.id);
    try {
      const ok = await onPick(take.id);
      if (ok !== false) {
        setAnnounce(t("board.compare.picked", { defaultValue: "{take} is now the shot's take.", take: takeName(take) }));
      }
    } finally {
      setPicking(null);
    }
  }

  function pane(side: Side) {
    const take = takes[side];
    const other = takes[side === "left" ? "right" : "left"];
    const active = take.id === shot.activeTake;
    const sideLabel = side === "left" ? t("board.compare.left", { defaultValue: "Left" }) : t("board.compare.right", { defaultValue: "Right" });
    const selectId = `take-compare-${shot.id}-${side}`;
    return (
      <section className={`take-compare__pane${active ? " take-compare__pane--active" : ""}`} aria-labelledby={`${selectId}-label`}>
        <div className="take-compare__pane-head">
          <label id={`${selectId}-label`} htmlFor={selectId} className="sr-only">
            {t("board.compare.pickSide", { defaultValue: "{side} take", side: sideLabel })}
          </label>
          <select
            id={selectId}
            className="text-input take-compare__select"
            value={take.id}
            onChange={(event) => setIds((current) => ({ ...current, [side]: event.target.value }))}
          >
            {shot.takes.map((item, index) => (
              <option key={item.id} value={item.id} disabled={item.id === other.id}>
                {t("board.compare.option", {
                  defaultValue: "Take {n} · {pass}{active, select, yes { · in use} other {}}",
                  n: index + 1,
                  pass: passLabel(item),
                  active: item.id === shot.activeTake ? "yes" : "no",
                })}
              </option>
            ))}
          </select>
          {take.renderPass === "draft" ? (
            <QualityTag quality={takeQualityOf(take, quality?.[take.id])} target={target} variant="inline" />
          ) : (
            <span className="badge success">{passLabel(take)}</span>
          )}
          {active ? <span className="badge accent">{t("board.compare.inUse", { defaultValue: "In use" })}</span> : null}
        </div>
        <div className="take-compare__frame" style={{ aspectRatio: aspectOf(take) }}>
          {clipsShown && take.clip ? (
            <CompareVideo
              projectId={projectId}
              take={take}
              videoRef={videos[side]}
              muted={sound !== side}
              label={t("board.compare.clipOf", { defaultValue: "Clip of {take}", take: takeName(take) })}
              onMeta={readDuration}
            />
          ) : take.still && (view === "still" || !take.clip) ? (
            <ProjectImage
              projectId={projectId}
              rel={take.still}
              version={take.hashes.still}
              className="take-compare__media"
              alt={t("board.compare.stillOf", { defaultValue: "Still of {take}", take: takeName(take) })}
            />
          ) : (
            <span className="take-compare__none">
              {view === "clip"
                ? t("board.compare.noClip", { defaultValue: "No clip in this take yet." })
                : t("board.compare.noStill", { defaultValue: "No still in this take." })}
            </span>
          )}
          {clipsShown && !take.clip && take.still ? (
            <span className="take-compare__note">{t("board.compare.stillShown", { defaultValue: "No clip yet: its still is shown." })}</span>
          ) : null}
        </div>
        {active ? (
          <button type="button" className="secondary-button take-compare__use" disabled>
            {t("board.compare.inUseButton", { defaultValue: "In use" })}
          </button>
        ) : (
          <button
            type="button"
            className="primary-button take-compare__use"
            disabled={disabled || picking !== null}
            onClick={() => void use(take)}
          >
            {t("board.compare.use", { defaultValue: "Use this one" })}
          </button>
        )}
      </section>
    );
  }

  return (
    <div className="modal-overlay picker-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div
        ref={dialog}
        className="modal-content take-compare"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`take-compare-title-${shot.id}`}
      >
        <div className="take-compare__head">
          <div className="modal-header">
            <h3 id={`take-compare-title-${shot.id}`}>
              {t("board.compare.title", { defaultValue: "Compare takes · Shot {n}", n: number })}
            </h3>
            <p>{summaryText}</p>
          </div>
          <button
            type="button"
            className="secondary-button icon-button"
            aria-label={t("storyboard.close", { defaultValue: "Close" })}
            title={t("storyboard.close", { defaultValue: "Close" })}
            onClick={onClose}
          >
            <CloseIcon />
          </button>
        </div>

        <div className="modal-body take-compare__body">
          {hasClips && hasStills ? (
            <div className="take-compare__views" role="group" aria-label={t("board.compare.viewLabel", { defaultValue: "Show" })}>
              {(["clip", "still"] as const).map((item) => (
                <button
                  key={item}
                  type="button"
                  aria-pressed={view === item}
                  className={`take-compare__view${view === item ? " active" : ""}`}
                  onClick={() => setView(item)}
                >
                  {item === "clip" ? t("board.compare.clips", { defaultValue: "Clips" }) : t("board.compare.stills", { defaultValue: "Stills" })}
                </button>
              ))}
            </div>
          ) : null}

          <div className="take-compare__grid">
            {pane("left")}
            {pane("right")}
          </div>

          {clipsShown ? (
            <div className="take-compare__transport" role="group" aria-label={t("board.compare.transport", { defaultValue: "Play both clips" })}>
              <button
                type="button"
                className="secondary-button"
                aria-pressed={playing}
                onClick={() => (playing ? pauseBoth() : playBoth())}
              >
                {playing ? <PauseIcon /> : <PlayIcon />}{" "}
                {playing ? t("board.compare.pause", { defaultValue: "Pause" }) : t("board.compare.playBoth", { defaultValue: "Play both" })}
              </button>
              <input
                type="range"
                className="take-compare__scrub"
                min={0}
                max={duration || 0}
                step={0.01}
                value={Math.min(time, duration || 0)}
                disabled={!duration}
                aria-label={t("board.compare.position", { defaultValue: "Position in both clips" })}
                aria-valuetext={t("board.compare.time", { defaultValue: "{time} of {length} s", time: formatTime(time), length: formatTime(duration) })}
                onChange={(event) => seekBoth(Number(event.target.value))}
              />
              <span className="take-compare__time" aria-hidden="true">
                {t("board.compare.time", { defaultValue: "{time} of {length} s", time: formatTime(time), length: formatTime(duration) })}
              </span>
              <label className="take-compare__sound">
                <span>{t("board.compare.sound", { defaultValue: "Sound" })}</span>
                <select className="text-input" value={sound} onChange={(event) => setSound(event.target.value as Sound)}>
                  <option value="left">{t("board.compare.left", { defaultValue: "Left" })}</option>
                  <option value="right">{t("board.compare.right", { defaultValue: "Right" })}</option>
                  <option value="off">{t("board.compare.soundOff", { defaultValue: "Off" })}</option>
                </select>
              </label>
            </div>
          ) : null}

          <table className="take-compare__facts">
            <caption className="sr-only">{t("board.compare.factsCaption", { defaultValue: "What made each take" })}</caption>
            <thead>
              <tr>
                <th scope="col"><span className="sr-only">{t("board.compare.setting", { defaultValue: "Setting" })}</span></th>
                <th scope="col">{takeName(left)}</th>
                <th scope="col">{takeName(right)}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key} className={row.differs ? "take-compare__differs" : undefined}>
                  <th scope="row">
                    {factLabels[row.key]}
                    {row.differs ? <span className="sr-only"> {t("board.compare.differs", { defaultValue: "(differs)" })}</span> : null}
                  </th>
                  <td>{factValue(row.key, row.a, left)}</td>
                  <td>{factValue(row.key, row.b, right)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="sr-only" role="status" aria-live="polite">{announce}</p>
      </div>
    </div>
  );
}
