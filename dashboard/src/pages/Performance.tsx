import { usePortfolioHistory, usePerformance } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityCurve } from "@/components/charts/EquityCurve";
import { MonthlyHeatmap } from "@/components/charts/MonthlyHeatmap";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { formatPct } from "@/lib/utils";

export default function Performance() {
  const { data: rawHistory, isLoading: hLoading } = usePortfolioHistory();
  const { data: perf, isLoading: pLoading } = usePerformance();

  // Handle API returning {error: ...} instead of real data
  const history = rawHistory && !('error' in rawHistory) ? rawHistory : null;

  if (hLoading || pLoading) return <LoadingSkeleton rows={4} />;

  return (
    <div className="space-y-3">
      {/* Equity Curve */}
      <Card>
        <CardHeader>
          <CardTitle>Equity Curve</CardTitle>
          {history && (
            <span className="text-xs text-gray-500">
              {history.days_tracked ?? 0} days
            </span>
          )}
        </CardHeader>
        {history ? (
          <EquityCurve data={history} />
        ) : (
          <div className="flex h-48 items-center justify-center text-sm text-gray-500">
            No history data yet. Start the agent to track performance.
          </div>
        )}
      </Card>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <Card>
          <div className="text-xs text-gray-500">Total Return</div>
          <div className="mt-1 text-xl font-bold">
            <span className={(history?.total_return_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}>
              {history?.total_return_pct != null ? formatPct(history.total_return_pct) : "--"}
            </span>
          </div>
        </Card>
        <Card>
          <div className="text-xs text-gray-500">Alpha vs SPY</div>
          <div className="mt-1 text-xl font-bold">
            <span className={(history?.alpha_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}>
              {history?.alpha_pct != null ? formatPct(history.alpha_pct) : "--"}
            </span>
          </div>
        </Card>
        <Card>
          <div className="text-xs text-gray-500">Sharpe Ratio</div>
          <div className="mt-1 text-xl font-bold text-white">
            {perf?.sharpe_ratio != null ? perf.sharpe_ratio.toFixed(2) : "--"}
          </div>
        </Card>
        <Card>
          <div className="text-xs text-gray-500">Max Drawdown</div>
          <div className="mt-1 text-xl font-bold text-red-400">
            {perf?.max_drawdown != null ? formatPct(-perf.max_drawdown) : "--"}
          </div>
        </Card>
      </div>

      {/* Win/Loss Stats */}
      {perf && perf.total_trades > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Trade Stats</CardTitle>
          </CardHeader>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div>
              <div className="text-lg font-bold text-white">
                {perf.total_trades}
              </div>
              <div className="text-gray-500">Total</div>
            </div>
            <div>
              <div className="text-lg font-bold text-green-400">
                {((perf.win_rate ?? 0) * 100).toFixed(0)}%
              </div>
              <div className="text-gray-500">Win Rate</div>
            </div>
            <div>
              <div className="text-lg font-bold">
                {perf.profit_factor != null ? perf.profit_factor.toFixed(2) : "--"}
              </div>
              <div className="text-gray-500">Profit Factor</div>
            </div>
          </div>
        </Card>
      )}

      {/* Monthly Returns */}
      {history && (
        <Card>
          <CardHeader>
            <CardTitle>Monthly Returns</CardTitle>
          </CardHeader>
          <MonthlyHeatmap data={history} />
        </Card>
      )}

      {/* No data message */}
      {!history && (!perf || perf.total_trades === 0) && (
        <Card>
          <div className="flex h-32 items-center justify-center text-sm text-gray-500">
            📊 No performance data yet. The agent needs to make trades to generate stats.
          </div>
        </Card>
      )}
    </div>
  );
}
