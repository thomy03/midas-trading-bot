import { usePortfolioPositions, usePortfolioSummary } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { PnlBadge } from "@/components/shared/PnlBadge";
import { PillarBar } from "@/components/shared/PillarBar";
import { DonutChart } from "@/components/charts/DonutChart";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { formatCurrency, formatPct } from "@/lib/utils";
import { useState } from "react";

export default function Portfolio() {
  const { data: positions, isLoading } = usePortfolioPositions();
  const { data: summary } = usePortfolioSummary();
  const [expanded, setExpanded] = useState<string | null>(null);

  if (isLoading) return <LoadingSkeleton rows={5} />;

  const posArr = positions?.positions ?? [];

  // Donut data
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

  return (
    <div className="space-y-3">
      {/* Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio</CardTitle>
          {summary && (
            <PnlBadge
              value={summary.unrealized_pnl}
              mode="currency"
              size="md"
            />
          )}
        </CardHeader>
        <div className="text-2xl font-bold text-white">
          {positions ? formatCurrency(positions.total_value) : "--"}
        </div>
        <div className="mt-1 text-xs text-gray-500">
          {posArr.length} positions | Cash:{" "}
          {positions ? formatCurrency(positions.cash) : "--"}
        </div>
      </Card>

      {/* Allocation Donut */}
      {donutData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Allocation</CardTitle>
          </CardHeader>
          <DonutChart data={donutData} />
          <div className="mt-2 flex flex-wrap gap-2">
            {donutData.slice(0, 6).map((d) => (
              <span key={d.name} className="text-[10px] text-gray-400">
                {d.name} {d.value.toFixed(1)}%
              </span>
            ))}
          </div>
        </Card>
      )}

      {/* Positions List */}
      <div className="space-y-2">
        {posArr.length === 0 && (
          <Card>
            <div className="py-4 text-center text-sm text-gray-500">
              No open positions
            </div>
          </Card>
        )}
        {posArr.map((p) => (
          <Card
            key={p.symbol}
            className="cursor-pointer active:bg-surface-2"
            onClick={() =>
              setExpanded(expanded === p.symbol ? null : p.symbol)
            }
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium text-white">{p.symbol}</div>
                <div className="text-[10px] text-gray-500">
                  {p.company_name || p.sector || ""}
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-white">
                  {formatCurrency(p.position_value)}
                </div>
                <PnlBadge
                  value={p.pnl_percent ?? 0}
                  mode="pct"
                  size="sm"
                />
              </div>
            </div>

            {expanded === p.symbol && (
              <div className="mt-3 space-y-2 border-t border-border pt-3">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Entry:</span>{" "}
                    <span className="text-white">
                      ${p.entry_price.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Shares:</span>{" "}
                    <span className="text-white">{p.shares}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Alloc:</span>{" "}
                    <span className="text-white">
                      {formatPct(p.allocation_pct, 1)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Score:</span>{" "}
                    <span className="text-white">
                      {p.score_at_entry?.toFixed(0) ?? "--"}
                    </span>
                  </div>
                </div>
                <PillarBar
                  technical={p.pillar_technical}
                  fundamental={p.pillar_fundamental}
                  sentiment={p.pillar_sentiment}
                  news={p.pillar_news}
                />
                {p.reasoning && (
                  <p className="text-[11px] text-gray-400">{p.reasoning}</p>
                )}
              </div>
            )}
          </Card>
        ))}
      </div>
    </div>
  );
}
