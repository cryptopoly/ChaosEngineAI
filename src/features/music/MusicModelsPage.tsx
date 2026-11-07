/**
 * Music > Models: every music model, installed or not, on the shared Models shell (`ModelsPage`) through the
 * music adapter. Above the list: the callout for a machine that cannot run music yet, the music pack install
 * banner (only while the pack is missing or installing) and the first-run line. Row detail carries the
 * measured speed, the licence lines and the model card link; the licence chip opens the licence screen.
 */
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { LicenceChip, type ModelLicence } from "../../components/LicenceChip";
import { LicenceScreen } from "../../components/LicenceScreen";
import { IconActionButton } from "../../components/ModelActionIcons";
import { MakesChips } from "../../components/MakesChips";
import {
  ModelCard,
  ModelDetailList,
  ModelDownloadControls,
  ModelMemoryCell,
  ModelsPage,
  confirmAndDelete,
  fitLabel,
  fitSortRank,
  resolveModelStatus,
  statusSortRank,
  type ModelAvailabilityFilter,
  type ModelFilterGroup,
  type ModelRow,
  type ModelSortOption,
  type ModelsPageColumn,
  type SortDir,
} from "../../components/models";
import { buildMusicEntries, measuredSpeed, musicEntryMatchesQuery, type MusicModelEntry } from "../../components/models/adapters/music";
import type { UseMusicState } from "../../hooks/useMusicState";
import { MusicPackBanner } from "./MusicPackBanner";
import { SoundPackNotice } from "./SoundPackNotice";
import { isLicenceAccepted, rememberLicenceAccepted } from "./musicLicence";

export interface MusicModelsPageProps {
  music: UseMusicState;
  backendOnline: boolean;
  availability: ModelAvailabilityFilter;
  onAvailabilityChange: (value: ModelAvailabilityFilter) => void;
  onOpenExternalUrl: (url: string) => void;
}

// Columns: model, makes, licence, size, fit, status, actions. Below 1440 px the makes and licence columns
// (2, 3) drop and their chips show under the name instead (see `.music-models__inline`).
const MUSIC_GRID =
  "minmax(220px, 1.7fr) minmax(150px, 1.1fr) minmax(170px, 1.2fr) minmax(76px, 0.6fr) " +
  "minmax(130px, 1fr) minmax(100px, 0.8fr) 168px";
const MUSIC_GRID_NARROW = "minmax(190px, 1.6fr) minmax(70px, 0.6fr) minmax(130px, 1fr) minmax(100px, 0.8fr) 150px";

export function musicSortOptions(t: (key: string, options?: Record<string, unknown>) => string): ModelSortOption<MusicModelEntry>[] {
  return [
    {
      id: "fit",
      label: t("musicModels.sort.fit", { defaultValue: "Best fit first" }),
      defaultDir: "asc",
      summary: () => t("musicModels.sort.fitSummary", { defaultValue: "best fit first" }),
      compare: (a, b) =>
        Number(Boolean(b.entry.bestFit)) - Number(Boolean(a.entry.bestFit)) || fitSortRank(a.entry.fit) - fitSortRank(b.entry.fit),
    },
    {
      id: "size",
      label: t("musicModels.sort.size", { defaultValue: "Size" }),
      defaultDir: "asc",
      summary: (dir) =>
        dir === "asc"
          ? t("musicModels.sort.smallestFirst", { defaultValue: "smallest first" })
          : t("musicModels.sort.largestFirst", { defaultValue: "largest first" }),
      value: (row) => row.entry.sizeGb,
    },
    {
      id: "name",
      label: t("musicModels.sort.name", { defaultValue: "Name" }),
      defaultDir: "asc",
      summary: () => t("musicModels.sort.nameSummary", { defaultValue: "name A-Z" }),
      compare: (a, b) => a.entry.name.localeCompare(b.entry.name),
    },
    {
      id: "status",
      label: t("musicModels.sort.status", { defaultValue: "Status" }),
      defaultDir: "asc",
      summary: () => t("musicModels.sort.statusSummary", { defaultValue: "installed first" }),
      compare: (a, b) => statusSortRank(a.status) - statusSortRank(b.status),
    },
  ];
}

export function MusicModelsPage({ music, backendOnline, availability, onAvailabilityChange, onOpenExternalUrl }: MusicModelsPageProps) {
  const { t } = useTranslation("studio");
  const [search, setSearch] = useState("");
  const [makesFilter, setMakesFilter] = useState("all");
  const [licenceFilter, setLicenceFilter] = useState("all");
  const [sortId, setSortId] = useState("fit");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [licenceFor, setLicenceFor] = useState<MusicModelEntry | null>(null);
  const [acceptedTick, setAcceptedTick] = useState(0);

  const { runtime, families, downloads } = music;
  const unavailable = runtime?.unavailableReason ?? null;
  const entries = useMemo(
    () => buildMusicEntries(families, { loadedEngineModel: runtime?.loadedModel ?? null }),
    [families, runtime?.loadedModel],
  );
  const sortOptions = useMemo(() => musicSortOptions(t), [t]);
  const tf = t as unknown as (key: string, options?: Record<string, unknown>) => string;

  const groups: ModelFilterGroup<MusicModelEntry>[] = [
    {
      id: "makes",
      label: t("musicModels.filter.makes", { defaultValue: "Makes" }),
      kind: "select",
      value: makesFilter,
      defaultValue: "all",
      options: [
        { value: "all", label: t("musicModels.filter.makesAll", { defaultValue: "Everything" }) },
        { value: "songs", label: t("models.makes.songs", { defaultValue: "Songs with vocals" }) },
        { value: "instrumental", label: t("models.makes.instrumental", { defaultValue: "Instrumental" }) },
        { value: "sfx", label: t("models.makes.sfx", { defaultValue: "Sound effects" }) },
      ],
      onChange: setMakesFilter,
      predicate: (entry, value) => entry.source.makes.includes(value as never),
    },
    {
      id: "licence",
      label: t("musicModels.filter.licence", { defaultValue: "Licence" }),
      kind: "select",
      value: licenceFilter,
      defaultValue: "all",
      options: [
        { value: "all", label: t("musicModels.filter.licenceAny", { defaultValue: "Any licence" }) },
        { value: "commercial", label: t("musicModels.filter.commercialOnly", { defaultValue: "Commercial use ok" }) },
      ],
      onChange: setLicenceFilter,
      predicate: (entry) => entry.source.licence?.commercial === "yes",
    },
  ];

  const columns: ModelsPageColumn[] = [
    { id: "model", label: t("musicModels.column.model", { defaultValue: "Model" }), sortId: "name" },
    { id: "makes", label: t("musicModels.column.makes", { defaultValue: "Makes" }) },
    { id: "licence", label: t("musicModels.column.licence", { defaultValue: "Licence" }) },
    { id: "size", label: t("musicModels.column.size", { defaultValue: "Size" }), sortId: "size" },
    { id: "fit", label: t("musicModels.column.fit", { defaultValue: "Fit" }), sortId: "fit" },
    { id: "status", label: t("musicModels.column.status", { defaultValue: "Status" }), sortId: "status" },
  ];

  function toggleExpanded(id: string) {
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function licenceOf(entry: MusicModelEntry): ModelLicence | null {
    const licence = entry.source.licence;
    if (!licence) return null;
    return { ...licence, summary: licence.summaryLines?.[0] ?? licence.name };
  }

  function renderRow({ entry, status }: ModelRow<MusicModelEntry>) {
    const variant = entry.source;
    const licence = licenceOf(entry);
    const needsLicence = Boolean(licence?.gated) && !isLicenceAccepted(variant.id, licence?.id ?? "", acceptedTick);
    const download = status.download;
    const speed = measuredSpeed(variant);
    const openLicence = licence ? () => setLicenceFor(entry) : undefined;
    const makes = <MakesChips makes={variant.makes} />;
    const licenceChip = licence ? <LicenceChip licence={licence} onOpen={openLicence} /> : null;

    const badges = (
      <>
        <span className="music-models__inline">
          {makes}
          {licenceChip}
        </span>
        {variant.platforms.map((platform) => (
          <span key={platform} className="badge muted">
            {platform === "apple-silicon"
              ? t("musicModels.platform.appleSilicon", { defaultValue: "Apple Silicon" })
              : platform === "cuda"
                ? t("musicModels.platform.cuda", { defaultValue: "CUDA" })
                : platform === "cpu"
                  ? t("musicModels.platform.cpu", { defaultValue: "CPU" })
                  : platform}
          </span>
        ))}
      </>
    );

    const cells = [
      makes,
      licenceChip,
      <span title={entry.sizeTitle ?? undefined}>{entry.sizeLabel}</span>,
      <ModelMemoryCell
        memory={entry.memory}
        fit={entry.fit}
        pendingLabel={t("musicModels.memory.pending", { defaultValue: "not estimated" })}
        pendingTitle={t("musicModels.memory.pendingTitle", { defaultValue: "No memory figure for this model yet." })}
      />,
    ];

    const statusCell = needsLicence ? (
      <span className="badge warning">{t("licence.status.needs", { defaultValue: "Needs licence" })}</span>
    ) : undefined;

    const actions = needsLicence ? (
      <div className="media-model-actions model-download-controls">
        <button type="button" className="primary-button" onClick={openLicence}>
          {t("licence.action.accept", { defaultValue: "Accept licence" })}
        </button>
      </div>
    ) : (
      <ModelDownloadControls
        status={status}
        canDownload={!unavailable}
        onDownload={() => void music.startDownload(entry.downloadKey)}
        onPause={() => void music.pauseDownload(download?.repo ?? entry.downloadKey)}
        onCancel={() => void music.deleteDownload(entry.deleteTarget)}
        onDeleteModel={() => {
          confirmAndDelete(entry, (repo) => void music.deleteDownload(repo));
        }}
        launch={{
          id: "studio",
          icon: "generate",
          label: t("musicModels.action.openInStudio", { defaultValue: "Open in Studio" }),
          onClick: () => music.openStudio(entry.id),
        }}
        trailing={
          entry.link ? (
            <IconActionButton
              icon="huggingFace"
              label={t("musicModels.action.openHuggingFace", { defaultValue: "Open on Hugging Face" })}
              onClick={() => onOpenExternalUrl(entry.link as string)}
            />
          ) : null
        }
      />
    );

    const detail = () => (
      <>
        {entry.note ? <p className="models-card__note">{entry.note}</p> : null}
        <ModelDetailList
          items={[
            {
              label: t("musicModels.detail.speed", { defaultValue: "Speed" }),
              value: speed.measured
                ? t("models.speed.measured", { defaultValue: "About {rate} s of audio per second on this machine", rate: speed.rate })
                : t("models.speed.unknown", { defaultValue: "Speed not measured yet" }),
            },
            entry.fit
              ? {
                  label: t("musicModels.detail.fit", { defaultValue: "Fit on this machine" }),
                  value: `${fitLabel(entry.fit, tf as never)}${entry.fit.reason ? ` · ${entry.fit.reason}` : ""}`,
                }
              : null,
            licence
              ? {
                  label: t("musicModels.detail.licence", { defaultValue: "Licence" }),
                  value: (
                    <ul className="music-models__licence-lines">
                      {(licence.summaryLines ?? [licence.name]).map((line) => (
                        <li key={line}>{line}</li>
                      ))}
                    </ul>
                  ),
                }
              : null,
            variant.sizeNote ? { label: t("musicModels.detail.size", { defaultValue: "Download" }), value: variant.sizeNote } : null,
            variant.licence?.attribution
              ? { label: t("musicModels.detail.attribution", { defaultValue: "Attribution" }), value: variant.licence.attribution }
              : null,
            (variant.requires ?? []).length > 0
              ? { label: t("musicModels.detail.requires", { defaultValue: "Also needs" }), value: (variant.requires ?? []).join(", ") }
              : null,
            { label: t("musicModels.detail.repo", { defaultValue: "Repository" }), value: <code>{entry.repo}</code> },
          ]}
        />
        <div className="button-row">
          {licence?.url ? (
            <button type="button" className="secondary-button" onClick={() => onOpenExternalUrl(licence.url as string)}>
              {t("licence.screen.readFull", { defaultValue: "Read the full licence" })}
            </button>
          ) : null}
          {entry.link ? (
            <button type="button" className="secondary-button" onClick={() => onOpenExternalUrl(entry.link as string)}>
              {t("musicModels.detail.openModelCard", { defaultValue: "Open model card" })}
            </button>
          ) : null}
        </div>
      </>
    );

    const downloadCallout =
      status.state === "failed" && download?.error ? (
        <div className="media-model-row-detail callout error">
          <p>{t("musicModels.downloadFailed", { defaultValue: "The download did not finish." })}</p>
          <p className="muted-text">{download.error}</p>
          <details className="debug-details">
            <summary>{t("error.technicalDetails", { defaultValue: "Technical details" })}</summary>
            <pre className="install-log-terminal">{download.error}</pre>
          </details>
        </div>
      ) : null;
    // A sound row also needs the separate sounds pack, offered once the licence is accepted (or the weights exist).
    const callouts =
      variant.kind === "sound" && !needsLicence ? (
        <>
          {downloadCallout}
          <SoundPackNotice music={music} />
        </>
      ) : (
        downloadCallout
      );

    return (
      <ModelCard
        key={entry.id}
        entry={entry}
        status={status}
        badges={badges}
        cells={cells}
        statusCell={statusCell}
        actions={actions}
        detail={detail}
        expanded={expanded.has(entry.id)}
        onToggleExpanded={() => toggleExpanded(entry.id)}
        callouts={callouts}
      />
    );
  }

  const installedCount = entries.filter((entry) => entry.availableLocally && entry.source.kind !== "sound").length;
  const recommended = music.selectedVariant;
  const firstRun =
    backendOnline && music.catalogLoaded && !unavailable && installedCount === 0 && recommended && !downloads[recommended.repo] ? (
      <div className="callout quiet music-models__first-run">
        <p>
          {t("musicModels.empty.firstRun", {
            defaultValue: "Pick a music model to start. {name} ({size}, {licence}) fits this machine.",
            name: recommended.name,
            size: `${recommended.sizeGb.toFixed(1)} GB`,
            licence: recommended.licence?.name ?? "",
          })}
        </p>
        <div className="button-row">
          <button type="button" className="primary-button" onClick={() => void music.startDownload(recommended.repo)}>
            {t("musicModels.empty.firstRunAction", { defaultValue: "Download {name}", name: recommended.name })}
          </button>
        </div>
      </div>
    ) : null;

  const openStudioButton = (
    <button type="button" className="primary-button" onClick={() => music.openStudio(music.selectedVariant?.id)}>
      {t("musicModels.openStudio", { defaultValue: "Open Studio" })}
    </button>
  );

  const licence = licenceFor ? licenceOf(licenceFor) : null;

  return (
    <>
      <ModelsPage<MusicModelEntry>
        className="music-models-page"
        title={t("musicModels.title", { defaultValue: "Music Models" })}
        headerActions={openStudioButton}
        intro={
          <p className="muted-text">
            {t("musicModels.intro", {
              defaultValue: "Music models run on this computer. Download one, then open the Studio to make a track.",
            })}
          </p>
        }
        beforeList={
          <>
            {unavailable ? (
              <div className="callout warning music-models__unavailable" role="status">
                <p>{t("musicModels.unavailable", { defaultValue: "Music needs an Apple Silicon Mac for now." })}</p>
                <p className="muted-text">{unavailable}</p>
              </div>
            ) : null}
            {!unavailable ? <MusicPackBanner music={music} /> : null}
            {music.downloadPollError ? (
              <div className="callout warning" role="alert">
                <p>{t("musicModels.downloadStatusLost", { defaultValue: "Download progress can no longer be read. Reopen this page to check again." })}</p>
                <p className="muted-text">{music.downloadPollError}</p>
              </div>
            ) : null}
            {firstRun}
          </>
        }
        entries={entries}
        resolveStatus={(entry) => resolveModelStatus(entry, { download: downloads[entry.downloadKey] })}
        search={{
          value: search,
          onChange: setSearch,
          placeholder: t("musicModels.search", { defaultValue: "Search models, licence, what they make" }),
        }}
        matches={musicEntryMatchesQuery}
        groups={groups}
        availability={availability}
        onAvailabilityChange={onAvailabilityChange}
        sortOptions={sortOptions}
        sortId={sortId}
        sortDir={sortDir}
        defaultSortId="fit"
        onSortChange={(id, dir) => {
          setSortId(id);
          setSortDir(dir);
        }}
        onClearFilters={() => {
          setSearch("");
          setMakesFilter("all");
          setLicenceFilter("all");
        }}
        columns={columns}
        gridTemplate={MUSIC_GRID}
        narrowGridTemplate={MUSIC_GRID_NARROW}
        narrowHide={[2, 3]}
        renderRow={renderRow}
        emptyCatalog={t("musicModels.empty.catalog", { defaultValue: "No music models in the catalog yet." })}
        emptyByAvailability={{
          installed: t("musicModels.empty.installed", { defaultValue: "No music models installed yet. Pick one from the available models to download it." }),
          downloading: t("musicModels.empty.downloading", { defaultValue: "No music model is downloading." }),
          attention: t("musicModels.empty.attention", { defaultValue: "No music model needs attention." }),
        }}
      />
      {licenceFor && licence ? (
        <LicenceScreen
          entry={{ name: licenceFor.name }}
          licence={licence}
          accepted={isLicenceAccepted(licenceFor.source.id, licence.id, acceptedTick)}
          onAccept={() => {
            rememberLicenceAccepted(licenceFor.source.id, licence.id);
            setAcceptedTick((tick) => tick + 1);
            setLicenceFor(null);
          }}
          onDismiss={() => setLicenceFor(null)}
          onOpenHub={() => onOpenExternalUrl(licenceFor.link ?? licenceFor.source.link)}
          onOpenFull={() => onOpenExternalUrl(licence.url ?? licenceFor.source.link)}
        />
      ) : null}
    </>
  );
}
