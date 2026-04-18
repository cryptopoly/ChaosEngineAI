import { Panel } from "../../components/Panel";
import type { TabId } from "../../types";

export interface VideoGalleryTabProps {
  onActiveTabChange: (tab: TabId) => void;
}

export function VideoGalleryTab({ onActiveTabChange }: VideoGalleryTabProps) {
  return (
    <div className="content-grid image-page-grid">
      <Panel
        title="Video Gallery"
        subtitle="Saved renders"
        className="span-2"
        actions={
          <button className="secondary-button" type="button" onClick={() => onActiveTabChange("video-studio")}>
            Open Studio
          </button>
        }
      >
        <div className="empty-state image-empty-state">
          <p>
            <span className="badge warning" style={{ marginRight: 8 }}>Coming soon</span>
            No video outputs yet. Once generation ships, finished renders will land here with filters by
            model, duration, and resolution.
          </p>
          <p className="muted-text" style={{ marginTop: 12 }}>
            The runtime already supports preload/unload — you can warm a model up from Video Models while we
            finish the generation loop.
          </p>
        </div>
      </Panel>
    </div>
  );
}
