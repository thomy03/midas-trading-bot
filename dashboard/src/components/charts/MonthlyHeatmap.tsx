import type { PortfolioHistory } from "@/api/types";
import { cn } from "@/lib/utils";

interface Props {
  data: PortfolioHistory;
}

function monthColor(val: number): string {
  if (val > 5) return "bg-green-500/50 text-green-200";
  if (val > 2) return "bg-green-500/30 text-green-300";
  if (val > 0) return "bg-green-500/15 text-green-400";
  if (val > -2) return "bg-red-500/15 text-red-400";
  if (val > -5) return "bg-red-500/30 text-red-300";
  return "bg-red-500/50 text-red-200";
}

export function MonthlyHeatmap({ data }: Props) {
  const daily = data.daily_history;
  const dates = Object.keys(daily).sort();

  if (dates.length < 2) {
    return (
      <div className="flex h-24 items-center justify-center text-sm text-gray-500">
        Not enough data
      </div>
    );
  }

  // Group by month
  const months: Record<string, { start: number; end: number }> = {};
  for (const d of dates) {
    const key = d.slice(0, 7); // YYYY-MM
    if (!months[key]) months[key] = { start: daily[d].open, end: daily[d].close };
    else months[key].end = daily[d].close;
  }

  const monthEntries = Object.entries(months).map(([key, { start, end }]) => ({
    label: key.slice(5) + "/" + key.slice(2, 4), // MM/YY
    pct: start > 0 ? ((end / start - 1) * 100) : 0,
  }));

  return (
    <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-6">
      {monthEntries.map((m) => (
        <div
          key={m.label}
          className={cn("rounded-lg px-2 py-1.5 text-center", monthColor(m.pct))}
        >
          <div className="text-[9px] opacity-60">{m.label}</div>
          <div className="text-xs font-bold">
            {m.pct > 0 ? "+" : ""}
            {m.pct.toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
}
