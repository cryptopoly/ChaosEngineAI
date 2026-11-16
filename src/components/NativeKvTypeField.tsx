import { useTranslation } from "react-i18next";
import { SliderField } from "./SliderField";

// FU-178: the native strategy on llama.cpp can store the KV cache as q8_0 or
// q4_0. The setting rides on cacheBits (0 = f16, 8 = q8_0, 4 = q4_0), the
// values the backend's NativeStrategy maps to --cache-type-k / -v.
export const NATIVE_KV_TYPES: ReadonlyArray<{ bits: number; label: string }> = [
  { bits: 0, label: "f16" },
  { bits: 8, label: "q8_0" },
  { bits: 4, label: "q4_0" },
];

export function nativeKvIndex(cacheBits: number): number {
  const index = NATIVE_KV_TYPES.findIndex((entry) => entry.bits === cacheBits);
  return index < 0 ? 0 : index;
}

export function NativeKvTypeField(props: { cacheBits: number; onChange: (bits: number) => void }) {
  const { t } = useTranslation("runtime");
  const index = nativeKvIndex(props.cacheBits);
  return (
    <div>
      <SliderField
        label={t("controls.kvCacheTypeGguf", { defaultValue: "KV cache type" })}
        value={index}
        min={0}
        max={NATIVE_KV_TYPES.length - 1}
        step={1}
        ticks={NATIVE_KV_TYPES.map((entry, i) => ({ value: i, label: entry.label }))}
        formatValue={(i) => NATIVE_KV_TYPES[i]?.label ?? "f16"}
        onChange={(i) => props.onChange(NATIVE_KV_TYPES[i]?.bits ?? 0)}
      />
      <small className="muted-text">
        {t("controls.kvCacheTypeGgufHint", {
          defaultValue: "q8_0 holds the KV cache in about half the memory of f16, q4_0 in about a quarter. Decode is a little slower.",
        })}
      </small>
    </div>
  );
}
