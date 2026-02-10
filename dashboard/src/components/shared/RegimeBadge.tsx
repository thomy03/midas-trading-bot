import { REGIME_BG, Regime } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function RegimeBadge({ regime }: { regime: string }) {
  const upper = regime.toUpperCase() as Regime;
  const cls = REGIME_BG[upper] ?? REGIME_BG.RANGE;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-bold tracking-wider",
        cls
      )}
    >
      {upper}
    </span>
  );
}
