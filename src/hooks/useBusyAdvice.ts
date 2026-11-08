import { useEffect, useState } from "react";
import { fetchBusyAdvice } from "../api";
import type { BusyAdvice } from "../types";

/**
 * Busy-machine advice for a model of ``footprintGb`` (a Studio's estimated peak), read when the figure
 * changes. A Studio job waits its turn behind other heavy jobs, so only the "risky" verdict (resident
 * models plus this peak would not fit) is worth a note there; "slower" is chat's concern.
 */
export function useBusyAdvice(footprintGb: number | null): BusyAdvice | null {
  const [advice, setAdvice] = useState<BusyAdvice | null>(null);
  useEffect(() => {
    if (footprintGb == null || footprintGb <= 0) {
      setAdvice(null);
      return;
    }
    let live = true;
    void fetchBusyAdvice(footprintGb).then((result) => {
      if (live) setAdvice(result?.level === "risky" ? result : null);
    });
    return () => {
      live = false;
    };
  }, [footprintGb]);
  return advice;
}
