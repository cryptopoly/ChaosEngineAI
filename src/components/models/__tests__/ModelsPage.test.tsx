import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ModelEntry } from "../modelEntry";
import { resolveModelStatus } from "../modelStatus";
import type { ModelFilterGroup, ModelSortOption } from "../modelFilters";
import { ModelsPage, type ModelsPageProps } from "../ModelsPage";
import { ModelCard, defaultModelSubtitle } from "../ModelCard";
import { ModelDownloadControls } from "../ModelDownloadControls";

function entry(overrides: Partial<ModelEntry> & { id: string }): ModelEntry {
  return {
    area: "chat",
    name: overrides.id,
    provider: "Org",
    repo: `org/${overrides.id}`,
    downloadKey: `org/${overrides.id}`,
    deleteTarget: `org/${overrides.id}`,
    availableLocally: false,
    hasLocalData: false,
    hostCompatible: true,
    releaseSortKey: "",
    source: null,
    ...overrides,
  };
}

const sortOptions: ModelSortOption[] = [
  { id: "release", label: "Newest", defaultDir: "desc", compare: (a, b) => a.entry.releaseSortKey.localeCompare(b.entry.releaseSortKey) },
  { id: "name", label: "Name", defaultDir: "asc", compare: (a, b) => a.entry.name.localeCompare(b.entry.name) },
];

function render(overrides: Partial<ModelsPageProps<ModelEntry>> = {}): string {
  const props: ModelsPageProps<ModelEntry> = {
    title: "Chat Models",
    entries: [],
    resolveStatus: (item) => resolveModelStatus(item),
    search: { value: "", onChange: vi.fn(), placeholder: "Search" },
    availability: "all",
    onAvailabilityChange: vi.fn(),
    sortOptions,
    sortId: "release",
    sortDir: "desc",
    defaultSortId: "release",
    onSortChange: vi.fn(),
    onClearFilters: vi.fn(),
    columns: [{ id: "model", label: "Model", sortId: "name" }, { id: "format", label: "Format" }],
    renderRow: ({ entry: item }) => <div key={item.id} data-row={item.id}>{item.name}</div>,
    ...overrides,
  };
  return renderToStaticMarkup(<ModelsPage {...props} />);
}

describe("ModelsPage", () => {
  it("says the catalog is empty when there is nothing to show", () => {
    expect(render()).toContain("No models in the catalog yet.");
    expect(render({ emptyCatalog: "Backend offline" })).toContain("Backend offline");
  });

  it("renders the rows, a sortable header and the default subtitle", () => {
    const markup = render({
      entries: [entry({ id: "a", availableLocally: true }), entry({ id: "b" })],
    });
    expect(markup).toContain('data-row="a"');
    expect(markup).toContain('data-row="b"');
    expect(markup).toContain("2 of 2 models · 1 installed");
    expect(markup).toContain(">Model</button>");
    expect(markup).toContain('<span class="sort-header">Format</span>');
  });

  it("lists active filters in the summary and enables Clear Filters", () => {
    const group: ModelFilterGroup = {
      id: "format",
      label: "Format",
      kind: "chips",
      value: "gguf",
      defaultValue: "all",
      options: [
        { value: "all", label: "All" },
        { value: "gguf", label: "GGUF", count: 1 },
      ],
      onChange: vi.fn(),
      predicate: (item, value) => item.id.includes(value),
    };
    const markup = render({
      entries: [entry({ id: "gguf-model" }), entry({ id: "mlx-model" })],
      search: { value: "model", onChange: vi.fn(), placeholder: "Search" },
      matches: (item, query) => item.id.includes(query),
      groups: [group],
      availability: "available",
    });
    expect(markup).toContain('data-row="gguf-model"');
    expect(markup).not.toContain('data-row="mlx-model"');
    expect(markup).toContain("Search: model");
    expect(markup).toContain("Format: GGUF");
    expect(markup).toContain("Showing: Available");
    expect(markup).not.toMatch(/<button class="secondary-button" type="button" disabled="">Clear Filters/);
  });

  it("disables Clear Filters at the defaults", () => {
    const markup = render({ entries: [entry({ id: "a" })] });
    expect(markup).toMatch(/<button class="secondary-button" type="button" disabled="">Clear Filters/);
  });

  it("explains an empty bucket and offers the available models", () => {
    const markup = render({
      entries: [entry({ id: "a" })],
      availability: "installed",
      emptyByAvailability: { installed: "Nothing installed yet." },
    });
    expect(markup).toContain("Nothing installed yet.");
    expect(markup).toContain("Show available models");
  });

  it("renders the slots around the list", () => {
    const markup = render({
      entries: [entry({ id: "a" })],
      intro: <p>Intro text</p>,
      beforeList: <section data-slot="hub">Hub results</section>,
      afterList: <section data-slot="import">Import</section>,
      headerActions: <button type="button">Open Studio</button>,
    });
    expect(markup).toContain("Intro text");
    expect(markup.indexOf('data-slot="hub"')).toBeLessThan(markup.indexOf('data-row="a"'));
    expect(markup.indexOf('data-row="a"')).toBeLessThan(markup.indexOf('data-slot="import"'));
    expect(markup).toContain("Open Studio");
  });
});

describe("ModelCard and ModelDownloadControls", () => {
  it("builds the subtitle from provider and family without repeats", () => {
    expect(defaultModelSubtitle({ provider: "Qwen", familyName: "Qwen 3.6" })).toBe("Qwen · Qwen 3.6");
    expect(defaultModelSubtitle({ provider: "FLUX", familyName: "flux" })).toBe("FLUX");
    expect(defaultModelSubtitle({ provider: "Org", familyName: null })).toBe("Org");
  });

  it("puts a primary launch and an overflow menu on an installed row (chat: thread + use in this chat)", () => {
    const item = entry({ id: "qwen", availableLocally: true, hasLocalData: true });
    const markup = renderToStaticMarkup(
      <ModelDownloadControls
        status={resolveModelStatus(item)}
        onDownload={vi.fn()}
        onPause={vi.fn()}
        onCancel={vi.fn()}
        onDeleteModel={vi.fn()}
        launch={{ id: "thread", icon: "chat", label: "Chat in a new thread", onClick: vi.fn() }}
        overflow={[
          { id: "chat", icon: "chat", label: "Use in this chat", onClick: vi.fn() },
          { id: "server", icon: "server", label: "Load for server", onClick: vi.fn() },
        ]}
      />,
    );
    expect(markup).toContain('aria-label="Chat in a new thread"');
    expect(markup).toContain("primary-button");
    expect(markup).toContain('class="model-action-menu"');
    expect(markup).toContain(">Use in this chat</button>");
    expect(markup).toContain(">Load for server</button>");
    expect(markup).toContain('aria-label="Delete model"');
  });

  it("hides the launch actions on a row that is not installed", () => {
    const markup = renderToStaticMarkup(
      <ModelDownloadControls
        status={resolveModelStatus(entry({ id: "x" }))}
        onDownload={vi.fn()}
        onPause={vi.fn()}
        onCancel={vi.fn()}
        launch={{ id: "thread", icon: "chat", label: "Chat", onClick: vi.fn() }}
        overflow={[{ id: "chat", icon: "chat", label: "Use in this chat", onClick: vi.fn() }]}
      />,
    );
    expect(markup).toContain('aria-label="Download model"');
    expect(markup).not.toContain('aria-label="Chat"');
    expect(markup).not.toContain("model-action-menu");
  });

  it("marks a row the host cannot run", () => {
    const item = entry({ id: "mlx", hostCompatible: false, availableLocally: true, hasLocalData: true });
    const markup = renderToStaticMarkup(
      <ModelCard entry={item} status={resolveModelStatus(item)} cells={[]} actions={<div />} />,
    );
    expect(markup).toContain("Cannot run on this machine");
    expect(markup).toContain("models-card--incompatible");
  });
});
