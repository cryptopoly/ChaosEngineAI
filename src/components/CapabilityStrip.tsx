import { CAPABILITY_META } from "../constants";
import { capabilityMeta } from "../utils";


interface CapabilityStripProps {
  capabilities: string[];
  max?: number;
}

/**
 * Capability badge strip used by the model picker / catalog rows.
 *
 * Each capability slug renders as a coloured pill driven by
 * ``CAPABILITY_META`` (icon + accent colour) plus the localised
 * short-label from ``capabilityMeta``. Three call sites (App,
 * MyModelsTab, OnlineModelsTab) shared an identical inline
 * implementation before this component was extracted.
 */
export function CapabilityStrip({ capabilities, max = 5 }: CapabilityStripProps) {
  return (
    <div className="capability-strip">
      {capabilities.slice(0, max).map((capability) => {
        const meta = capabilityMeta(capability);
        const fullMeta = CAPABILITY_META[capability];
        return (
          <span
            className="capability-icon"
            key={capability}
            title={meta.title}
            style={
              fullMeta
                ? { borderColor: `${fullMeta.color}40`, color: fullMeta.color }
                : undefined
            }
          >
            {fullMeta?.icon ?? ""} {meta.shortLabel}
          </span>
        );
      })}
    </div>
  );
}
