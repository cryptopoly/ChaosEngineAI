/**
 * Shared parts of the Models pages (Chat, Images, Video): one page per
 * area replaces the old Discover + My Models pair (FU-129).
 *
 * An area page builds ``ModelEntry`` rows with an adapter
 * (``adapters/image.ts``), resolves each row's status with
 * ``resolveModelStatus``, and hands everything to ``ModelsPage``, which
 * filters, counts, sorts and lays out the rows; the area renders each row
 * with ``ModelCard`` + ``ModelDownloadControls``. See
 * ``src/features/images/ImageModelsPage.tsx`` for the reference port.
 */

export * from "./fit";
export * from "./modelEntry";
export * from "./modelStatus";
export * from "./modelFilters";
export * from "./downloadActions";
export * from "./deleteConfirm";
export { FitBadge, BestFitRibbon, fitLabel, fitTooltip } from "./FitBadge";
export { ModelStatusChip, modelStatusLabel, modelBucketLabel, modelStatusDetail } from "./ModelStatusChip";
export { ModelDownloadControls, ModelActionMenu } from "./ModelDownloadControls";
export type { ModelDownloadControlsProps, ModelLaunchAction } from "./ModelDownloadControls";
export { ModelFilterBar } from "./ModelFilterBar";
export type { ModelFilterBarProps } from "./ModelFilterBar";
export { ModelCard, ModelMemoryCell, ModelDetailList, defaultModelSubtitle } from "./ModelCard";
export type { ModelCardProps } from "./ModelCard";
export { AcceleratorPills, acceleratorsForHost } from "./AcceleratorPills";
export type { AcceleratorHost } from "./AcceleratorPills";
export { ModelsPage } from "./ModelsPage";
export type { ModelsPageColumn, ModelsPageProps } from "./ModelsPage";
export { LegacyModelsSwitch } from "./LegacyModelsSwitch";
