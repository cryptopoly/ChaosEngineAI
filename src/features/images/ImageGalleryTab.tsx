import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { ImageOutputCard } from "../../components/ImageOutputCard";
import type {
  ImageModelFamily,
  ImageOutputArtifact,
  TabId,
} from "../../types";
import type {
  ImageGalleryRuntimeFilter,
  ImageGalleryOrientationFilter,
  ImageGallerySort,
} from "../../types/image";

export interface ImageGalleryTabProps {
  imageOutputs: ImageOutputArtifact[];
  filteredImageOutputs: ImageOutputArtifact[];
  imageCatalog: ImageModelFamily[];
  imageBusy: boolean;
  imageGallerySearchInput: string;
  onImageGallerySearchInputChange: (value: string) => void;
  imageGalleryModelFilter: string;
  onImageGalleryModelFilterChange: (value: string) => void;
  imageGalleryRuntimeFilter: ImageGalleryRuntimeFilter;
  onImageGalleryRuntimeFilterChange: (value: ImageGalleryRuntimeFilter) => void;
  imageGalleryOrientationFilter: ImageGalleryOrientationFilter;
  onImageGalleryOrientationFilterChange: (value: ImageGalleryOrientationFilter) => void;
  imageGallerySort: ImageGallerySort;
  onImageGallerySortChange: (value: ImageGallerySort) => void;
  imageGalleryModelOptions: { id: string; name: string }[];
  imageGalleryModelCount: number;
  imageGalleryRealCount: number;
  imageGalleryPlaceholderCount: number;
  imageGalleryWarningCount: number;
  imageGalleryHasActiveFilters: boolean;
  onActiveTabChange: (tab: TabId) => void;
  onOpenImageStudio: (modelId?: string) => void;
  onResetImageGalleryFilters: () => void;
  onOpenExternalUrl: (url: string) => void;
  onUseSameImageSettings: (artifact: ImageOutputArtifact, closeModal?: boolean) => void;
  onVaryImageSeed: (artifact: ImageOutputArtifact) => void;
  onRevealPath: (path: string) => void;
  onDeleteImageArtifact: (id: string) => void;
}

export function ImageGalleryTab({
  imageOutputs,
  filteredImageOutputs,
  imageCatalog,
  imageBusy,
  imageGallerySearchInput,
  onImageGallerySearchInputChange,
  imageGalleryModelFilter,
  onImageGalleryModelFilterChange,
  imageGalleryRuntimeFilter,
  onImageGalleryRuntimeFilterChange,
  imageGalleryOrientationFilter,
  onImageGalleryOrientationFilterChange,
  imageGallerySort,
  onImageGallerySortChange,
  imageGalleryModelOptions,
  imageGalleryModelCount,
  imageGalleryRealCount,
  imageGalleryPlaceholderCount,
  imageGalleryWarningCount,
  imageGalleryHasActiveFilters,
  onActiveTabChange,
  onOpenImageStudio,
  onResetImageGalleryFilters,
  onOpenExternalUrl,
  onUseSameImageSettings,
  onVaryImageSeed,
  onRevealPath,
  onDeleteImageArtifact,
}: ImageGalleryTabProps) {
  const { t } = useTranslation("common");
  const { t: tStudio } = useTranslation("studio");
  return (
    <div className="content-grid image-page-grid">
      <Panel
        title={t("tabs.imageGallery")}
        subtitle={imageOutputs.length > 0
          ? tStudio("imageGallery.subtitle.count", {
              defaultValue:
                "{filtered} of {total, plural, one {# saved output} other {# saved outputs}}",
              filtered: filteredImageOutputs.length,
              total: imageOutputs.length,
            })
          : tStudio("imageGallery.subtitle.empty", {
              defaultValue: "Saved generations, filters, and quick reuse actions",
            })}
        className="span-2"
        actions={
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => onOpenImageStudio()}>
              {tStudio("imageGallery.actions.studio", { defaultValue: "Studio" })}
            </button>
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-models")}>
              {tStudio("imageGallery.actions.installedModels", { defaultValue: "Installed Models" })}
            </button>
          </div>
        }
      >
        <div className="image-studio-hero">
          <div>
            <span className="eyebrow">
              {tStudio("imageGallery.hero.eyebrow", { defaultValue: "Saved Outputs" })}
            </span>
            <h3>
              {tStudio("imageGallery.hero.heading", {
                defaultValue: "Browse, filter, and reuse generated images",
              })}
            </h3>
            <p className="muted-text">
              {tStudio("imageGallery.hero.body", {
                defaultValue:
                  "Keep Image Studio focused on prompting and generation, then use Image Gallery to search old outputs, compare models, and jump back into Studio with the same settings.",
              })}
            </p>
          </div>
          <div className="image-studio-hero-stats">
            <span className="badge muted">
              {tStudio("imageGallery.stats.saved", {
                defaultValue: "{count, plural, one {# saved} other {# saved}}",
                count: imageOutputs.length,
              })}
            </span>
            <span className="badge muted">
              {tStudio("imageGallery.stats.modelsUsed", {
                defaultValue: "{count, plural, one {# model used} other {# models used}}",
                count: imageGalleryModelCount,
              })}
            </span>
            {imageGalleryRealCount > 0 ? (
              <span className="badge success">
                {tStudio("imageGallery.stats.realEngine", {
                  defaultValue: "{count} real engine",
                  count: imageGalleryRealCount,
                })}
              </span>
            ) : null}
            {imageGalleryPlaceholderCount > 0 ? (
              <span className="badge warning">
                {tStudio("imageGallery.stats.placeholder", {
                  defaultValue: "{count} placeholder",
                  count: imageGalleryPlaceholderCount,
                })}
              </span>
            ) : null}
            {imageGalleryWarningCount > 0 ? (
              <span className="badge subtle">
                {tStudio("imageGallery.stats.withNotes", {
                  defaultValue: "{count} with notes",
                  count: imageGalleryWarningCount,
                })}
              </span>
            ) : null}
          </div>
        </div>

        <div className="image-gallery-toolbar">
          <label className="image-gallery-search">
            {tStudio("imageGallery.toolbar.search", { defaultValue: "Search" })}
            <input
              className="text-input"
              type="search"
              placeholder={tStudio("imageGallery.toolbar.searchPlaceholder", {
                defaultValue: "Prompt, model, runtime note",
              })}
              value={imageGallerySearchInput}
              onChange={(event) => onImageGallerySearchInputChange(event.target.value)}
            />
          </label>
          <label>
            {tStudio("imageGallery.toolbar.model", { defaultValue: "Model" })}
            <select
              className="text-input"
              value={imageGalleryModelFilter}
              onChange={(event) => onImageGalleryModelFilterChange(event.target.value)}
            >
              <option value="all">
                {tStudio("imageGallery.toolbar.allModels", { defaultValue: "All models" })}
              </option>
              {imageGalleryModelOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            {tStudio("imageGallery.toolbar.runtime", { defaultValue: "Runtime" })}
            <select
              className="text-input"
              value={imageGalleryRuntimeFilter}
              onChange={(event) => onImageGalleryRuntimeFilterChange(event.target.value as ImageGalleryRuntimeFilter)}
            >
              <option value="all">
                {tStudio("imageGallery.toolbar.allRuntimes", { defaultValue: "All runtimes" })}
              </option>
              <option value="diffusers">
                {tStudio("imageGallery.toolbar.runtimeReal", { defaultValue: "Real engine" })}
              </option>
              <option value="placeholder">
                {tStudio("imageGallery.toolbar.runtimePlaceholder", { defaultValue: "Placeholder" })}
              </option>
              <option value="warning">
                {tStudio("imageGallery.toolbar.runtimeWithNotes", { defaultValue: "With notes" })}
              </option>
            </select>
          </label>
          <label>
            {tStudio("imageGallery.toolbar.frame", { defaultValue: "Frame" })}
            <select
              className="text-input"
              value={imageGalleryOrientationFilter}
              onChange={(event) => onImageGalleryOrientationFilterChange(event.target.value as ImageGalleryOrientationFilter)}
            >
              <option value="all">
                {tStudio("imageGallery.toolbar.allFrames", { defaultValue: "All frames" })}
              </option>
              <option value="square">
                {tStudio("imageGallery.toolbar.frameSquare", { defaultValue: "Square" })}
              </option>
              <option value="portrait">
                {tStudio("imageGallery.toolbar.framePortrait", { defaultValue: "Portrait" })}
              </option>
              <option value="landscape">
                {tStudio("imageGallery.toolbar.frameLandscape", { defaultValue: "Landscape" })}
              </option>
            </select>
          </label>
          <label>
            {tStudio("imageGallery.toolbar.sort", { defaultValue: "Sort" })}
            <select
              className="text-input"
              value={imageGallerySort}
              onChange={(event) => onImageGallerySortChange(event.target.value as ImageGallerySort)}
            >
              <option value="newest">
                {tStudio("imageGallery.toolbar.sortNewest", { defaultValue: "Newest first" })}
              </option>
              <option value="oldest">
                {tStudio("imageGallery.toolbar.sortOldest", { defaultValue: "Oldest first" })}
              </option>
            </select>
          </label>
        </div>

        {imageGalleryHasActiveFilters ? (
          <div className="button-row image-gallery-toolbar-actions">
            <span className="muted-text">
              {tStudio("imageGallery.matching", {
                defaultValue:
                  "Showing {count, plural, one {# matching output.} other {# matching outputs.}}",
                count: filteredImageOutputs.length,
              })}
            </span>
            <button className="secondary-button" type="button" onClick={onResetImageGalleryFilters}>
              {tStudio("imageGallery.toolbar.clearFilters", { defaultValue: "Clear Filters" })}
            </button>
          </div>
        ) : null}
      </Panel>

      <Panel
        title={t("panels.savedOutputs", { defaultValue: "Saved Outputs" })}
        subtitle={filteredImageOutputs.length > 0
          ? tStudio("imageGallery.savedSubtitle.ready", {
              defaultValue:
                "{count, plural, one {# image ready to browse} other {# images ready to browse}}",
              count: filteredImageOutputs.length,
            })
          : imageOutputs.length > 0
            ? tStudio("imageGallery.savedSubtitle.noneMatch", {
                defaultValue: "No saved outputs match the current filters",
              })
            : tStudio("imageGallery.savedSubtitle.startGenerating", {
                defaultValue: "Generate in Image Studio to start building the gallery",
              })}
        className="span-2 image-gallery-page-panel"
      >
        {filteredImageOutputs.length === 0 ? (
          <div className="empty-state image-empty-state">
            <div className="image-empty-state-copy">
              <p>
                {imageOutputs.length === 0
                  ? tStudio("imageGallery.empty.firstGeneration", {
                      defaultValue:
                        "Generate a prompt in Image Studio to create the first saved image artifact for this branch.",
                    })
                  : tStudio("imageGallery.empty.noMatches", {
                      defaultValue:
                        "No saved images match the current filters yet. Try broadening the search or clearing one of the filters.",
                    })}
              </p>
              <div className="button-row">
                {imageOutputs.length === 0 ? (
                  <button className="secondary-button" type="button" onClick={() => onOpenImageStudio()}>
                    {tStudio("imageGallery.empty.openStudio", { defaultValue: "Open Studio" })}
                  </button>
                ) : (
                  <button className="secondary-button" type="button" onClick={onResetImageGalleryFilters}>
                    {tStudio("imageGallery.toolbar.clearFilters", { defaultValue: "Clear Filters" })}
                  </button>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="image-gallery-grid">
            {filteredImageOutputs.map((artifact) => (
              <ImageOutputCard
                key={artifact.artifactId}
                artifact={artifact}
                imageCatalog={imageCatalog}
                imageBusy={imageBusy}
                onOpenExternalUrl={(url) => onOpenExternalUrl(url)}
                onUseSameSettings={onUseSameImageSettings}
                onVarySeed={(a) => onVaryImageSeed(a)}
                onRevealPath={(path) => onRevealPath(path)}
                onDelete={(id) => onDeleteImageArtifact(id)}
                onNavigateSettings={() => onActiveTabChange("settings")}
              />
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
