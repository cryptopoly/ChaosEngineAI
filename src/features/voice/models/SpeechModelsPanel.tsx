/**
 * Voice > Models > Speech models on the shared Models shell (D-73 M3b): Kokoro as the pinned first row (an add-on of
 * the voice runtime since D-75, "Not installed" when left out), then every speech model grouped by family, with Makes, Size, Fit,
 * Status and the shared actions (Use in Speak or Browse voices, Download, Delete with a confirm, Licence). The full
 * measured cost sits in the row's detail. A first-run callout offers a natural voice while none is downloaded.
 */
import { Fragment, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  ModelCard,
  ModelDetailList,
  ModelDownloadControls,
  ModelMemoryCell,
  ModelsPage,
  fitLabel,
  resolveModelStatus,
  type ModelAvailabilityFilter,
  type ModelFilterGroup,
  type ModelRow,
  type ModelsPageColumn,
  type SortDir,
} from "../../../components/models";
import {
  KOKORO_ROW_ID,
  VOICE_FAMILY_ORDER,
  buildVoiceEntries,
  isKokoro,
  kokoroEntry,
  voiceDownloadStatus,
  voiceEntryMatchesQuery,
  type VoiceMakes,
  type VoiceModelEntry,
} from "../../../components/models/adapters/voice";
import type { VoiceModelRow, VoiceRuntime } from "../../../types";
import { getSpeechModelDeletePlan } from "../clone/cloneApi";
import { InstallVoiceCard } from "../speak/InstallVoiceCard";
import { useCostText } from "../speak/costText";
import { costLabel, recommendedNaturalModel, showInstallCard } from "../speak/speakLogic";
import type { VoiceCatalog } from "../speak/useVoiceCatalog";
import { quickVoiceSummary } from "../voicePackState";
import type { StudioTarget } from "../VoiceModelsTab";
import { collapsedFamilyRows, familyHeadingTracker, speechDeleteConfirmText, studioTargetFor, voiceSortOptions } from "./voiceModelsLogic";

const VOICE_GRID = "minmax(220px, 1.7fr) minmax(160px, 1.2fr) minmax(76px, 0.5fr) minmax(130px, 1fr) minmax(100px, 0.8fr) 168px";
const VOICE_GRID_NARROW = "minmax(190px, 1.7fr) minmax(70px, 0.5fr) minmax(130px, 1fr) minmax(100px, 0.8fr) 150px";

// i18n-extract: voice.models.makes.quick
// i18n-extract: voice.models.makes.natural
// i18n-extract: voice.models.makes.style
// i18n-extract: voice.models.makes.clone
// i18n-extract: voice.models.makes.design
// i18n-extract: voice.models.makes.tags
// i18n-extract: voice.models.makes.expression
// i18n-extract: voice.models.makes.languages
const MAKES_LABELS: Record<VoiceMakes, string> = {
  quick: "Quick voices",
  natural: "Natural voices",
  style: "Style",
  clone: "Clone",
  design: "Design",
  tags: "Sound tags",
  expression: "Expression",
  languages: "Language choice",
};

// i18n-extract: voice.models.family.qwen3-tts
// i18n-extract: voice.models.family.chatterbox
// i18n-extract: voice.models.family.voxcpm2
// i18n-extract: voice.models.family.orpheus
const FAMILY_BLURBS: Record<string, string> = {
  "qwen3-tts": "Natural voices, style instructions, voice design and cloning",
  chatterbox: "Sound tags, the Expression dial and cloning",
  voxcpm2: "Voice design and cloning in 30 languages",
  orpheus: "Eight expressive English voices with sound tags",
};

export function VoiceMakesChips({ makes }: { makes: string[] }) {
  const { t } = useTranslation("studio");
  if (makes.length === 0) return null;
  return (
    <ul className="makes-chips">
      {makes.map((kind) => (
        <li key={kind} className="makes-chip">
          {t(`voice.models.makes.${kind}`, { defaultValue: MAKES_LABELS[kind as VoiceMakes] ?? kind })}
        </li>
      ))}
    </ul>
  );
}

export interface SpeechModelsPanelProps {
  catalog: VoiceCatalog;
  runtime: VoiceRuntime;
  availability: ModelAvailabilityFilter;
  onAvailabilityChange: (value: ModelAvailabilityFilter) => void;
  onOpenStudio?: (target?: StudioTarget) => void;
  onOpenExternalUrl?: (url: string) => void;
}

/** The Kokoro quick voices' download (the MLX checkpoint or the ONNX files), shown while they are not installed. */
const KOKORO_DOWNLOAD_GB = 0.33;

export function SpeechModelsPanel({ catalog, runtime, availability, onAvailabilityChange, onOpenStudio, onOpenExternalUrl }: SpeechModelsPanelProps) {
  const { t, i18n } = useTranslation("studio");
  const costs = useCostText();
  const [search, setSearch] = useState("");
  const [makesFilter, setMakesFilter] = useState("all");
  const [sortId, setSortId] = useState("family");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [openFamilies, setOpenFamilies] = useState<Set<string>>(() => new Set());
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const tf = t as unknown as (key: string, options?: Record<string, unknown>) => string;

  const quick = quickVoiceSummary(runtime.ttsVoices);
  const kokoroPart = runtime.parts?.find((part) => part.id === "kokoro")?.sizeGb ?? null;
  const entries = useMemo(() => {
    const kokoro = runtime.installed
      ? kokoroEntry(
          {
            voices: quick.count,
            languages: quick.languages,
            // 0 GB on disk when Kokoro is left out (an add-on, D-75): show its download size instead.
            sizeGb: kokoroPart || KOKORO_DOWNLOAD_GB,
            peakGb: catalog.kokoro?.peakGb ?? null,
            installed: runtime.ttsModelInstalled,
          },
          catalog.kokoro?.fit,
        )
      : null;
    return buildVoiceEntries(catalog.models, kokoro);
  }, [catalog.models, catalog.kokoro, runtime.installed, runtime.ttsModelInstalled, quick.count, quick.languages, kokoroPart]);
  const collapsible = useMemo(() => collapsedFamilyRows(entries), [entries]);
  // A search, a filter or another sort lists every matching row; the collapse is for the plain family view.
  const plainView = search.trim() === "" && makesFilter === "all" && availability === "all" && sortId === "family";
  const catalogOrder = useMemo(() => catalog.models.map((m) => m.key), [catalog.models]);
  const sortOptions = useMemo(() => voiceSortOptions(tf, catalogOrder), [tf, catalogOrder]);
  const makesLabels = Object.fromEntries(
    Object.entries(MAKES_LABELS).map(([kind, label]) => [kind, t(`voice.models.makes.${kind}`, { defaultValue: label })]),
  ) as Record<VoiceMakes, string>;

  const groups: ModelFilterGroup<VoiceModelEntry>[] = [
    {
      id: "makes",
      label: t("voice.models.filter.makes", { defaultValue: "Makes" }),
      kind: "select",
      value: makesFilter,
      defaultValue: "all",
      options: [
        { value: "all", label: t("voice.models.filter.makesAll", { defaultValue: "Everything" }) },
        ...(["natural", "style", "tags", "clone", "design"] as VoiceMakes[]).map((kind) => ({ value: kind, label: makesLabels[kind] })),
      ],
      onChange: setMakesFilter,
      predicate: (entry, value) => (entry.capabilities ?? []).includes(value),
    },
  ];

  const columns: ModelsPageColumn[] = [
    { id: "model", label: t("voice.models.column.model", { defaultValue: "Model" }), sortId: "name" },
    { id: "makes", label: t("voice.models.column.makes", { defaultValue: "Makes" }) },
    { id: "size", label: t("voice.models.column.size", { defaultValue: "Size" }), sortId: "size" },
    { id: "fit", label: t("voice.models.column.fit", { defaultValue: "Fit" }), sortId: "fit" },
    { id: "status", label: t("voice.models.column.status", { defaultValue: "Status" }), sortId: "status" },
  ];

  const toggleExpanded = (id: string) =>
    setExpanded((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleFamily = (id: string) =>
    setOpenFamilies((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const remove = async (entry: VoiceModelEntry) => {
    if (!catalog.remove || isKokoro(entry.source)) return;
    setError(null);
    try {
      const plan = await getSpeechModelDeletePlan(entry.deleteTarget);
      if (!window.confirm(speechDeleteConfirmText(plan, entry.sizeGb, tf))) return;
      setBusyKey(entry.id);
      await catalog.remove(entry.deleteTarget);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyKey(null);
    }
  };

  const openLicence = (row: VoiceModelRow) => {
    const url = row.licence?.url;
    if (!url) return;
    if (onOpenExternalUrl) onOpenExternalUrl(url);
    else window.open(url, "_blank", "noopener");
  };

  const headingFor = familyHeadingTracker();

  function familyHeading(entry: VoiceModelEntry) {
    const family = VOICE_FAMILY_ORDER.find((f) => f.id === entry.familyId);
    if (!family || family.id === "kokoro") return null;
    const members = catalog.models.filter((m) => entries.some((e) => e.id === m.key && e.familyId === family.id));
    const licence = [...new Set(members.map((m) => m.licence?.name).filter(Boolean))].join(", ");
    return (
      <div className="voice-models__family" role="row">
        <strong>{family.name}</strong>
        <span className="muted-text">
          {t(`voice.models.family.${family.id}`, { defaultValue: FAMILY_BLURBS[family.id] ?? "" })}
          {licence ? ` · ${licence}` : ""}
        </span>
        {plainView && collapsible.has(family.id) ? (
          <button
            type="button"
            className="link-button voice-models__sizes"
            aria-expanded={openFamilies.has(family.id)}
            onClick={() => toggleFamily(family.id)}
          >
            {openFamilies.has(family.id)
              ? t("voice.models.hideSizes", { defaultValue: "Hide sizes" })
              : t("voice.models.showSizes", { defaultValue: "Show {count} sizes", count: collapsible.get(family.id)?.total ?? 0 })}
          </button>
        ) : null}
      </div>
    );
  }

  function renderRow({ entry, status }: ModelRow<VoiceModelEntry>) {
    const source = entry.source;
    const fold = plainView ? collapsible.get(entry.familyId ?? "") : undefined;
    if (fold && !openFamilies.has(entry.familyId ?? "") && !fold.shown.has(entry.id)) return null;
    const heading = sortId === "family" && headingFor(entry) ? familyHeading(entry) : null;
    const makes = <VoiceMakesChips makes={entry.capabilities ?? []} />;
    const cells = [
      makes,
      <span title={entry.sizeTitle ?? undefined}>{entry.sizeLabel ?? "—"}</span>,
      <ModelMemoryCell
        memory={entry.memory}
        fit={entry.fit}
        pendingLabel={t("voice.models.memory.pending", { defaultValue: "not measured" })}
        pendingTitle={t("voice.models.memory.pendingTitle", { defaultValue: "No memory figure for this model yet." })}
      />,
    ];
    const badges = <span className="voice-models__inline">{makes}</span>;

    if (isKokoro(source)) {
      return (
        <ModelCard
            key={entry.id}
            entry={entry}
            status={status}
            subtitle={t("voice.models.kokoroLineAddon", {
              defaultValue: "{count} quick voices in {languages} languages. Fast on any machine; an add-on of the voice runtime.",
              count: source.voices,
              languages: source.languages,
            })}
            badges={badges}
            cells={cells}
            statusCell={
              source.installed ? (
                <span className="badge muted">{t("voice.models.partOfRuntime", { defaultValue: "Part of the runtime" })}</span>
              ) : (
                <span className="badge muted">{t("voice.models.kokoroNotInstalled", { defaultValue: "Not installed" })}</span>
              )
            }
            actions={
              <div className="media-model-actions model-download-controls">
                {!source.installed ? (
                  <span className="muted-text">
                    {t("voice.models.kokoroAddHint", { defaultValue: "Tick it under Voice runtime > Add-ons." })}
                  </span>
                ) : onOpenStudio ? (
                  <button type="button" className="secondary-button" onClick={() => onOpenStudio({ mode: "voices" })}>
                    {t("voice.quickVoices.browse", { defaultValue: "Browse voices" })}
                  </button>
                ) : null}
              </div>
            }
          />
      );
    }

    const row = source;
    const target = studioTargetFor(row, catalog.voices, i18n.language);
    const launchLabel =
      target.mode === "voices"
        ? t("voice.models.browseVoices", { defaultValue: "Browse voices" })
        : t("voice.models.useInSpeak", { defaultValue: "Use in Speak" });
    const downloading = row.download.state === "downloading";
    const actions = !row.supported && !row.installed ? (
      <div className="media-model-actions model-download-controls">
        <span className="muted-text">{t("voice.speak.appleOnly", { defaultValue: "Apple Silicon Macs only for now." })}</span>
      </div>
    ) : downloading ? (
      <div className="media-model-actions model-download-controls">
        <span className="muted-text" aria-live="polite">
          {row.download.phase === "watermark"
            ? t("voice.clone.installingWatermark", { defaultValue: "Installing the watermark component…" })
            : t("voice.clone.downloading", { defaultValue: "Downloading…" })}
        </span>
        {catalog.cancelDownload ? (
          <button type="button" className="secondary-button" onClick={() => void catalog.cancelDownload?.(row.key)}>
            {t("voice.models.cancelDownload", { defaultValue: "Cancel" })}
          </button>
        ) : null}
      </div>
    ) : busyKey === entry.id ? (
      <div className="media-model-actions model-download-controls">
        <span className="muted-text" aria-live="polite">{t("voice.models.deleting", { defaultValue: "Deleting…" })}</span>
      </div>
    ) : (
      <ModelDownloadControls
        status={status}
        labels={{ download: t("voice.cost.download", { defaultValue: "Download {gb} GB", gb: (entry.sizeGb ?? row.sizeGb).toFixed(2) }) }}
        onDownload={() => void catalog.download(row.key)}
        onPause={() => {}}
        onCancel={() => void catalog.refresh()}
        onDeleteModel={catalog.remove ? () => void remove(entry) : undefined}
        launch={
          onOpenStudio
            ? { id: "use", icon: "generate", label: launchLabel, onClick: () => onOpenStudio(target) }
            : null
        }
        overflow={
          row.licence?.url
            ? [{ id: "licence", icon: "modelCard", label: t("voice.models.licence", { defaultValue: "Licence: {name}", name: row.licence.name }), onClick: () => openLicence(row) }]
            : undefined
        }
      />
    );

    const cost = costLabel(row);
    const detail = () => (
      <>
        {entry.note ? <p className="models-card__note">{entry.note}</p> : null}
        <ModelDetailList
          items={[
            { label: t("voice.models.detail.languages", { defaultValue: "Languages" }), value: row.languages },
            { label: t("voice.models.detail.cost", { defaultValue: "Memory and speed" }), value: costs.detail(cost) },
            cost.kind === "measured" && cost.coldStartS != null
              ? { label: t("voice.models.cost.coldStart", { defaultValue: "Cold start" }), value: t("voice.models.cost.seconds", { defaultValue: "{seconds} s", seconds: cost.coldStartS }) }
              : null,
            entry.fit
              ? { label: t("voice.models.detail.fit", { defaultValue: "Fit on this machine" }), value: `${fitLabel(entry.fit, tf as never)}${entry.fit.reason ? ` · ${entry.fit.reason}` : ""}` }
              : null,
            row.licence ? { label: t("voice.models.detail.licence", { defaultValue: "Licence" }), value: row.licence.name } : null,
            { label: t("voice.models.detail.repo", { defaultValue: "Repository" }), value: <code>{row.repo}</code> },
          ]}
        />
      </>
    );

    return (
      <Fragment key={entry.id}>
        {heading}
        <ModelCard
          entry={entry}
          status={status}
          subtitle={entry.note ?? undefined}
          badges={badges}
          cells={cells}
          actions={actions}
          detail={detail}
          expanded={expanded.has(entry.id)}
          onToggleExpanded={() => toggleExpanded(entry.id)}
          callouts={
            row.download.error ? (
              <div className="media-model-row-detail callout error">
                <p className="voice-error" role="alert">{row.download.error}</p>
              </div>
            ) : null
          }
        />
      </Fragment>
    );
  }

  const installModel = showInstallCard(catalog.models, catalog.modelsAvailable) ? recommendedNaturalModel(catalog.models) : null;

  return (
    <div className="span-2 voice-models__speech">
      <ModelsPage<VoiceModelEntry>
        className="voice-models-page"
        title={t("voice.models.ttsTitle", { defaultValue: "Speech models" })}
        intro={
          <p className="muted-text">
            {t("voice.models.ttsIntro", { defaultValue: "Turn text into speech: Speak, read-aloud and calls. Pick a voice on Studio > Voices." })}
          </p>
        }
        beforeList={
          <>
            {installModel ? <InstallVoiceCard model={installModel} onDownload={() => void catalog.download(installModel.key)} /> : null}
            {catalog.error ? <p className="voice-error" role="alert">{catalog.error}</p> : null}
            {error ? <p className="voice-error" role="alert">{error}</p> : null}
          </>
        }
        entries={entries}
        resolveStatus={(entry) => resolveModelStatus(entry, { download: voiceDownloadStatus(entry) })}
        search={{ value: search, onChange: setSearch, placeholder: t("voice.models.search", { defaultValue: "Search models, languages, what they make" }) }}
        matches={(entry, query) => voiceEntryMatchesQuery(entry, query, makesLabels)}
        groups={groups}
        availability={availability}
        onAvailabilityChange={onAvailabilityChange}
        pinned={(entry) => entry.id === KOKORO_ROW_ID}
        sortOptions={sortOptions}
        sortId={sortId}
        sortDir={sortDir}
        defaultSortId="family"
        onSortChange={(id, dir) => {
          setSortId(id);
          setSortDir(dir);
        }}
        onClearFilters={() => {
          setSearch("");
          setMakesFilter("all");
        }}
        columns={columns}
        gridTemplate={VOICE_GRID}
        narrowGridTemplate={VOICE_GRID_NARROW}
        narrowHide={[2]}
        renderRow={renderRow}
        emptyCatalog={t("voice.models.ttsNone", { defaultValue: "No speech models are listed." })}
        emptyByAvailability={{
          installed: t("voice.models.emptyInstalled", { defaultValue: "No speech model downloaded yet. The quick voices come with the runtime." }),
          downloading: t("voice.models.emptyDownloading", { defaultValue: "No speech model is downloading." }),
          attention: t("voice.models.emptyAttention", { defaultValue: "No speech model needs attention." }),
        }}
      />
    </div>
  );
}
