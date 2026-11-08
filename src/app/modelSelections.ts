import type { ChatSession, LibraryItem, WorkspaceData } from "../types";
import type { ChatModelOption } from "../types/chat";
import type { useBenchmarks, useSettings } from "../hooks";
import {
  findVariantForReference,
  findLibraryItemForVariant,
  findCatalogVariantForLibraryItem,
  libraryItemFormat,
} from "../utils";

/**
 * Which model option each surface (chat thread, server, benchmark, conversion)
 * is pointing at, resolved from the local library and the current drafts.
 */
export function deriveModelSelections({
  workspace,
  chatLibrary,
  threadModelOptions,
  libraryChatOptions,
  activeChat,
  serverModelKey,
  benchmarkModelKey,
  benchmarkDraft,
  conversionDraft,
}: {
  workspace: WorkspaceData;
  chatLibrary: LibraryItem[];
  threadModelOptions: ChatModelOption[];
  libraryChatOptions: ChatModelOption[];
  activeChat: ChatSession | undefined;
  serverModelKey: string;
  benchmarkModelKey: string;
  benchmarkDraft: ReturnType<typeof useBenchmarks>["benchmarkDraft"];
  conversionDraft: ReturnType<typeof useSettings>["conversionDraft"];
}) {
  const activeThreadOption =
    threadModelOptions.find((option) => option.modelRef === activeChat?.modelRef && option.path === (activeChat?.modelPath ?? undefined)) ??
    threadModelOptions.find((option) => option.modelRef === activeChat?.modelRef) ??
    null;
  const loadedModelOption =
    threadModelOptions.find(
      (option) =>
        option.modelRef === workspace.runtime.loadedModel?.ref &&
        option.path === (workspace.runtime.loadedModel?.path ?? undefined),
    ) ??
    threadModelOptions.find((option) => option.modelRef === workspace.runtime.loadedModel?.ref) ??
    null;
  const selectedServerOptionBase =
    threadModelOptions.find((option) => option.key === serverModelKey) ?? loadedModelOption ?? activeThreadOption ?? null;
  const selectedServerOption = (() => {
    if (!selectedServerOptionBase || selectedServerOptionBase.source !== "catalog") return selectedServerOptionBase;
    const variant = findVariantForReference(workspace.featuredModels, selectedServerOptionBase.modelRef, selectedServerOptionBase.model);
    if (!variant) return selectedServerOptionBase;
    const localItem = findLibraryItemForVariant(chatLibrary, variant);
    if (!localItem) return selectedServerOptionBase;
    return libraryChatOptions.find((option) => option.path === localItem.path) ?? selectedServerOptionBase;
  })();
  const convertibleLibrary = chatLibrary.filter((item) => libraryItemFormat(item) !== "MLX");
  const conversionSource = convertibleLibrary.find((item) => item.path === conversionDraft.path) ?? null;
  const conversionVariant =
    (conversionSource ? findCatalogVariantForLibraryItem(workspace.featuredModels, conversionSource) : null) ??
    findVariantForReference(workspace.featuredModels, conversionDraft.modelRef, conversionDraft.modelRef) ??
    null;
  const benchmarkOption =
    threadModelOptions.find((option) => option.key === benchmarkModelKey) ??
    threadModelOptions.find((option) => option.modelRef === benchmarkDraft.modelRef && option.path === benchmarkDraft.path) ??
    activeThreadOption ??
    loadedModelOption ??
    threadModelOptions[0] ??
    null;
  return {
    activeThreadOption,
    loadedModelOption,
    selectedServerOption,
    convertibleLibrary,
    conversionSource,
    conversionVariant,
    benchmarkOption,
  };
}
