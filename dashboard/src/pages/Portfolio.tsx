import { usePortfolioPositions, usePortfolioSummary } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { PnlBadge } from "@/components/shared/PnlBadge";
import { MiniGauge } from "@/components/shared/GaugeRing";
import { DonutChart } from "@/components/charts/DonutChart";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { PILLAR_COLORS } from "@/lib/constants";
import { formatCurrency, formatPct } from "@/lib/utils";
import { StrategySplit, StrategyBadge } from "@/components/shared/StrategySplit";
import { Wallet, ChevronDown, ChevronUp, TrendingUp, ShieldCheck } from "lucide-react";
import { useState } from "react";

export default function Portfolio() {
  const { data: positions, isLoading } = usePortfolioPositions();
  const { data: summary } = usePortfolioSummary();
  const [expanded, setExpanded] = useState<string | null>(null);

  if (isLoading) return <LoadingSkeleton rows={5} />;

  const posArr = positions?.positions ?? [];

  const donutData = posArr.map((p) => ({
    name: p.symbol,
    value: p.allocation_pct,
  }));
  if (positions && positions.cash > 0 && positions.total_value > 0) {
    donutData.push({
      name: "Cash",
      value: (positions.cash / positions.total_value) * 100,
    });
  }

  const totalPnl = posArr.reduce((acc, p) => acc + (p.pnl_amount ?? 0), 0);

  return (
    <div className="space-y-4">
      {/* Hero */}
      <Card variant="accent" className="fade-up">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
              <Wallet size={12} className="inline mr-1" />
              Portfolio
            </div>
            <div className="text-3xl font-bold text-white tracking-tight">
              {positions ? formatCurrency(positions.total_value) : "--"}
            </div>
            {summary && <PnlBadge value={summary.unrealized_pnl} mode="currency" size="sm" />}
          </div>
          <div className="text-right">
            <div className="text-[10px] text-gray-500">{posArr.length} positions</div>
            <div className="text-xs text-gray-400 mt-1">
              Cash: {positions ? formatCurrency(positions.cash) : "--"}
            </div>
          </div>
        </div>
        {summary?.strategies && <StrategySplit strategies={summary.strategies} />}
      </Card>

      {/* Allocation */}
      {donutData.length > 0 && (
        <Card className="fade-up fade-up-1">
          <CardHeader>
            <CardTitle>Allocation</CardTitle>
          </CardHeader>
          <DonutChart data={donutData} />
          <div className="mt-3 flex flex-wrap justify-center gap-x-4 gap-y-1">
            {donutData.slice(0, 8).map((d) => (
              <span key={d.name} className="flex items-center gap-1.5 text-[10px] text-gray-400">
                <div className="h-2 w-2 rounded-full bg-gold" />
                {d.name} <span className="text-gray-500">{d.value.toFixed(1)}%</span>
              </span>
            ))}
          </div>
        </Card>
      )}

      {/* Positions */}
      <div className="fade-up fade-up-2">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-300">Open Positions</h2>
          <span className="text-[10px] text-gray-500">{posArr.length} active</span>
        </div>

        {posArr.length === 0 ? (
          <Card>
            <div className="py-8 text-center">
              <ShieldCheck size={32} className="mx-auto text-gray-600 mb-2" />
              <p className="text-sm text-gray-500">No open positions</p>
              <p className="text-[10px] text-gray-600 mt-1">Midas is waiting for the right opportunity</p>
            </div>
          </Card>
        ) : (
          <div className="space-y-2">
            {posArr.map((p) => {
              const isOpen = expanded === p.symbol;
              return (
                <Card
                  key={p.symbol}
                  className="cursor-pointer transition-all duration-200 hover:border-white/10"
                  onClick={() => setExpanded(isOpen ? null : p.symbol)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-white">{p.symbol}</span>
                        <StrategyBadge strategyId={p.strategy_id} />
                        {p.sector && (
                          <span className="rounded-md bg-white/5 px-1.5 py-0.5 text-[9px] text-gray-500">
                            {p.sector}
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-gray-500 mt-0.5">
                        {p.company_name || ""}
                      </div>
                    </div>
                    <div className="text-right flex items-center gap-2">
                      <div>
                        <div className="text-sm font-semibold text-white">
                          {formatCurrency(p.position_value)}
                        </div>
                        <PnlBadge value={p.pnl_percent ?? 0} mode="pct" size="sm" />
                      </div>
                      {isOpen ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
                    </div>
                  </div>

                  {isOpen && (
                    <div className="mt-3 pt-3 border-t border-white/5 space-y-3">
                      {/* Trade details */}
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-500">Entry</span>
                          <span className="text-white font-medium">${p.entry_price.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Shares</span>
                          <span className="text-white font-medium">{p.shares}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Alloc</span>
                          <span className="text-white font-medium">{formatPct(p.allocation_pct, 1)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500">Score</span>
                          <span className="text-white font-medium">{p.score_at_entry?.toFixed(0) ?? "--"}</span>
                        </div>
                        {p.stop_loss && (
                          <div className="flex justify-between">
                            <span className="text-gray-500">Stop Loss</span>
                            <span className="text-red-400 font-medium">${p.stop_loss.toFixed(2)}</span>
                          </div>
                        )}
                        {p.take_profit && (
                          <div className="flex justify-between">
                            <span className="text-gray-500">Take Profit</span>
                            <span className="text-green-400 font-medium">${p.take_profit.toFixed(2)}</span>
                          </div>
                        )}
                      </div>

                      {/* Pillar gauges */}
                      {(p.pillar_technical || p.pillar_fundamental) && (
                        <div className="grid grid-cols-2 gap-3">
                          <MiniGauge value={p.pillar_technical ?? 0} max={25} color={PILLAR_COLORS.technical} label="Technical" />
                          <MiniGauge value={p.pillar_fundamental ?? 0} max={25} color={PILLAR_COLORS.fundamental} label="Fundamental" />
                        </div>
                      )}

                      {/* Reasoning */}
                      {p.reasoning && (
                        <div className="rounded-xl bg-purple-500/5 border border-purple-500/10 p-3">
                          <div className="text-[10px] text-purple-400 uppercase tracking-wider mb-1">
                            🧠 Entry Reasoning
                          </div>
                          <p className="text-[11px] text-gray-300 leading-relaxed">
                            {p.reasoning}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
