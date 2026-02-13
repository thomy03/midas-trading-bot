import { StrategyBreakdown } from "@/api/types";
import { formatCurrency, formatPct } from "@/lib/utils";

interface Props {
  strategies?: { [key: string]: StrategyBreakdown };
}

const STRATEGY_META: Record<string, { icon: string; label: string; color: string }> = {
  aggressive: { icon: "⚡", label: "Aggressive", color: "text-red-400" },
  moderate: { icon: "🛡️", label: "Moderate", color: "text-green-400" },
};

export function StrategySplit({ strategies }: Props) {
  if (!strategies) return null;

  return (
    <div className="mt-2 rounded-xl bg-white/[0.03] border border-white/5 px-3 py-2 space-y-1">
      {Object.entries(strategies).map(([key, s]) => {
        const meta = STRATEGY_META[key] || { icon: "📊", label: key, color: "text-gray-400" };
        return (
          <div key={key} className="flex items-center justify-between text-xs">
            <span className={`${meta.color} font-medium`}>
              {meta.icon} {meta.label}
            </span>
            <div className="flex items-center gap-3">
              <span className="text-gray-300 font-medium">{formatCurrency(s.equity)}</span>
              <span className={s.return_pct >= 0 ? "text-green-400" : "text-red-400"}>
                {s.return_pct >= 0 ? "+" : ""}{formatPct(s.return_pct)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function StrategyBadge({ strategyId }: { strategyId?: string }) {
  if (!strategyId) return null;
  const meta = STRATEGY_META[strategyId] || { icon: "📊", label: strategyId, color: "text-gray-400" };
  return (
    <span className={`text-[9px] ${meta.color} font-medium`}>
      {meta.icon} {meta.label}
    </span>
  );
}
