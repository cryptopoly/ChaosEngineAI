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
  return (
    <div className="content-grid image-page-grid">
      <Panel
        title="Image Gallery"
        subtitle={imageOutputs.length > 0
          ? `${filteredImageOutputs.length} of ${imageOutputs.length} saved outputs`
          : "Saved generations, filters, and quick reuse actions"}
        className="span-2"
        actions={
          <div className="button-row">
            <button className="secondary-button" type="button" onClick={() => onOpenImageStudio()}>
              Studio
            </button>
            <button className="secondary-button" type="button" onClick={() => onActiveTabChange("image-models")}>
              Installed
            </button>
          </div>
        }
      >
        <div className="image-studio-hero">
          <div>
            <span className="eyebrow">Saved Outputs</span>
            <h3>Browse, filter, and reuse generated images</h3>
            <p className="muted-text">
              Keep Image Studio focused on prompting and generation, then use Image Gallery to search old outputs,
              compare models, and jump back into Studio with the same settings.
            </p>
          </div>
          <div className="image-studio-hero-stats">
            <span className="badge muted">{imageOutputs.length} saved</span>
            <span className="badge muted">{imageGalleryModelCount} models used</span>
            {imageGalleryRealCount > 0 ? <span className="badge success">{imageGalleryRealCount} real engine</span> : null}
            {imageGalleryPlaceholderCount > 0 ? <span className="badge warning">{imageGalleryPlaceholderCount} placeholder</span> : null}
            {imageGalleryWarningCount > 0 ? <span className="badge subtle">{imageGalleryWarningCount} with notes</span> : null}
          </div>
        </div>

        <div className="image-gallery-toolbar">
          <label className="image-gallery-search">
            Search
            <input
              className="text-input"
              type="search"
              placeholder="Prompt, model, runtime note"
              value={imageGallerySearchInput}
              onChange={(event) => onImageGallerySearchInputChange(event.target.value)}
            />
          </label>
          <label>
            Model
            <select
              className="text-input"
              value={imageGalleryModelFilter}
              onChange={(event) => onImageGalleryModelFilterChange(event.target.value)}
            >
              <option value="all">All models</option>
              {imageGalleryModelOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Runtime
            <select
              className="text-input"
              value={imageGalleryRuntimeFilter}
              onChange={(event) => onImageGalleryRuntimeFilterChange(event.target.value as ImageGalleryRuntimeFilter)}
            >
              <option value="all">All runtimes</option>
              <option value="diffusers">Real engine</option>
              <option value="placeholder">Placeholder</option>
              <option value="warning">With notes</option>
            </select>
          </label>
          <label>
            Frame
            <select
              className="text-input"
              value={imageGalleryOrientationFilter}
              onChange={(event) => onImageGalleryOrientationFilterChange(event.target.value as ImageGalleryOrientationFilter)}
            >
              <option value="all">All frames</option>
              <option value="square">Square</option>
              <option value="portrait">Portrait</option>
              <option value="landscape">Landscape</option>
            </select>
          </label>
          <label>
            Sort
            <select
              className="text-input"
              value={imageGallerySort}
              onChange={(event) => onImageGallerySortChange(event.target.value as ImageGallerySort)}
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
            </select>
          </label>
        </div>

        {imageGalleryHasActiveFilters ? (
          <div className="button-row image-gallery-toolbar-actions">
            <span className="muted-text">
              Showing {filteredImageOutputs.length} matching output{filteredImageOutputs.length === 1 ? "" : "s"}.
            </span>
            <button className="secondary-button" type="button" onClick={onResetImageGalleryFilters}>
              Clear Filters
            </button>
          </div>
        ) : null}
      </Panel>

      <Panel
        title="Saved Outputs"
        subtitle={filteredImageOutputs.length > 0
          ? `${filteredImageOutputs.length} image${filteredImageOutputs.length === 1 ? "" : "s"} ready to browse`
          : imageOutputs.length > 0
            ? "No saved outputs match the current filters"
            : "Generate in Image Studio to start building the gallery"}
        className="span-2 image-gallery-page-panel"
      >
        {filteredImageOutputs.length === 0 ? (
          <div className="empty-state image-empty-state">
            <div className="image-empty-state-copy">
              <p>
                {imageOutputs.length === 0
                  ? "Generate a prompt in Image Studio to create the first saved image artifact for this branch."
                  : "No saved images match the current filters yet. Try broadening the search or clearing one of the filters."}
              </p>
              <div className="button-row">
                {imageOutputs.length === 0 ? (
                  <button className="secondary-button" type="button" onClick={() => onOpenImageStudio()}>
                    Open Studio
                  </button>
                ) : (
                  <button className="secondary-button" type="button" onClick={onResetImageGalleryFilters}>
                    Clear Filters
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
