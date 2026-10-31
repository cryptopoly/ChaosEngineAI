import type { SystemStats } from "../../types";
import type { NativeBackendStatus } from "../../types/server";
import { AcceleratorCard } from "../AcceleratorCard";
import {
  getAccelerator,
  getApplicableAcceleratorsForHost,
  type AcceleratorId,
} from "../acceleratorCatalog";

export type AcceleratorHost = Pick<SystemStats, "platform" | "arch" | "gpuVramTotalGb"> | null | undefined;

/** The accelerators worth a pill for ``variant`` on ``host``: those whose
 * hardware this machine has and that patch the engine the variant runs on
 * (no CUDA boosters on a Mac, none for mflux / sd.cpp / mlx-video rows). */
export function acceleratorsForHost(
  variant: { repo: string; runtime?: string | null },
  host: AcceleratorHost,
): AcceleratorId[] {
  return getApplicableAcceleratorsForHost({ repo: variant.repo, runtime: variant.runtime ?? undefined }, host);
}

/** Read-only pills; installing lives in the Studio runtime banner and in
 * Settings > Accelerators, so install state stays in one place. */
export function AcceleratorPills({
  ids,
  capabilities,
}: {
  ids: AcceleratorId[] | undefined;
  capabilities?: NativeBackendStatus | null;
}) {
  if (!ids || ids.length === 0) return null;
  return (
    <>
      {ids.map((id) => {
        const meta = getAccelerator(id);
        if (!meta) return null;
        return <AcceleratorCard key={id} meta={meta} capabilities={capabilities ?? null} variant="pill" />;
      })}
    </>
  );
}
