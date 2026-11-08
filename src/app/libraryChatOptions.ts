import type { LibraryItem, ModelFamily, SystemStats } from "../types";
import type { ChatModelOption } from "../types/chat";
import {
  number,
  findCatalogVariantForLibraryItem,
  libraryItemFormat,
  libraryItemQuantization,
  libraryItemBackend,
  inferHfRepoFromLocalPath,
  resolveCapabilities,
  isVariantCompatibleWithHost,
  chatVariantPlatformGate,
} from "../utils";

/**
 * Chat model options built from the local library only. Catalog-only entries
 * would let the user pick a model that isn't downloaded yet, which then 500s
 * on Load.
 *
 * FU-056 follow-up: also filter by host platform. MLX-backed options
 * (``backend === "mlx"``) only run on Apple Silicon; vLLM options
 * (``backend === "vllm"``) only on CUDA hosts. Without this filter, Windows
 * users see MLX rows in every model picker that would error on load.
 */
export function buildLibraryChatOptions(
  chatLibrary: LibraryItem[],
  featuredModels: ModelFamily[],
  system: SystemStats,
): ChatModelOption[] {
  return chatLibrary
    .filter((item) => !item.broken)
    .map((item) => {
      const matched = findCatalogVariantForLibraryItem(featuredModels, item);
      const displayFormat = libraryItemFormat(item, matched);
      const displayQuantization = libraryItemQuantization(item, matched);
      const canonicalRepo = matched?.repo ?? inferHfRepoFromLocalPath(item.path);
      return {
        key: `library:${item.path}`,
        label: item.name,
        detail: `${displayFormat} / ${number(item.loadSizeGb ?? item.sizeGb)} GB`,
        group: "Local library",
        model: item.name,
        modelRef: item.name,
        canonicalRepo,
        source: "library",
        path: item.path,
        backend: libraryItemBackend(item, matched),
        paramsB: matched?.paramsB,
        sizeGb: item.loadSizeGb ?? item.sizeGb,
        contextWindow: matched?.contextWindow,
        format: displayFormat,
        quantization: displayQuantization ?? undefined,
        maxContext: item.maxContext ?? matched?.maxContext ?? null,
        // Phase 2.11: resolve typed capabilities so the picker can show
        // capability badges per option without re-deriving in each view.
        capabilities: resolveCapabilities(canonicalRepo ?? item.name, matched?.capabilities ?? null),
      };
    })
    .filter((option) =>
      isVariantCompatibleWithHost(
        chatVariantPlatformGate(option),
        system,
      ),
    );
}
