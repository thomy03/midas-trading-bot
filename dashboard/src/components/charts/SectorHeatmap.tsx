import type { SectorPerformance } from "@/api/types";
import { cn } from "@/lib/utils";

interface Props {
  sectors: SectorPerformance[];
  metric?: "perf_1d" | "perf_1w" | "perf_1m" | "perf_ytd";
}

function perfColor(val: number): string {
  if (val > 2) return "bg-green-500/40 text-green-300";
  if (val > 0.5) return "bg-green-500/20 text-green-400";
  if (val > -0.5) return "bg-gray-500/20 text-gray-400";
  if (val > -2) return "bg-red-500/20 text-red-400";
  return "bg-red-500/40 text-red-300";
}

export function SectorHeatmap({ sectors, metric = "perf_1d" }: Props) {
  if (sectors.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-sm text-gray-500">
        No sector data
      </div>
    );
  }

  return (
    <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4">
      {sectors.map((s) => {
        const val = s[metric];
        return (
          <div
            key={s.etf}
            className={cn(
              "rounded-lg px-2 py-2 text-center",
              perfColor(val)
            )}
          >
            <div className="text-[10px] font-medium opacity-70">{s.etf}</div>
            <div className="text-sm font-bold">
              {val > 0 ? "+" : ""}
              {val.toFixed(1)}%
            </div>
            <div className="truncate text-[9px] opacity-60">{s.name}</div>
          </div>
        );
      })}
    </div>
  );
}
