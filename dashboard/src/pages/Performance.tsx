import { usePortfolioHistory, usePerformance } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { EquityCurve } from "@/components/charts/EquityCurve";
import { MonthlyHeatmap } from "@/components/charts/MonthlyHeatmap";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { formatPct } from "@/lib/utils";
import { BarChart3, TrendingUp, TrendingDown, Target, Award } from "lucide-react";

export default function Performance() {
  const { data: rawHistory, isLoading: hLoading } = usePortfolioHistory();
  const { data: perf, isLoading: pLoading } = usePerformance();
  const history = rawHistory && !('error' in rawHistory) ? rawHistory : null;

  if (hLoading || pLoading) return <LoadingSkeleton rows={4} />;

  return (
    <div className="space-y-4">
      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-3 fade-up">
        <Card variant="accent">
          <TrendingUp size={16} className="text-green-400 mb-2" />
          <div className="text-[10px] text-gray-500 uppercase tracking-wider">Total Return</div>
          <div className="text-2xl font-bold mt-1">
            <span className={(history?.total_return_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}>
              {history?.total_return_pct != null ? formatPct(history.total_return_pct) : "--"}
            </span>
          </div>
        </Card>
        <Card variant="accent">
          <Award size={16} className="text-gold mb-2" />
          <div className="text-[10px] text-gray-500 uppercase tracking-wider">Alpha vs SPY</div>
          <div className="text-2xl font-bold mt-1">
            <span className={(history?.alpha_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}>
              {history?.alpha_pct != null ? formatPct(history.alpha_pct) : "--"}
            </span>
          </div>
        </Card>
        <Card>
          <BarChart3 size={16} className="text-blue-400 mb-2" />
          <div className="text-[10px] text-gray-500 uppercase tracking-wider">Sharpe Ratio</div>
          <div className="text-2xl font-bold text-white mt-1">
            {perf?.sharpe_ratio != null ? perf.sharpe_ratio.toFixed(2) : "--"}
          </div>
        </Card>
        <Card>
          <TrendingDown size={16} className="text-red-400 mb-2" />
          <div className="text-[10px] text-gray-500 uppercase tracking-wider">Max Drawdown</div>
          <div className="text-2xl font-bold text-red-400 mt-1">
            {perf?.max_drawdown != null ? formatPct(-perf.max_drawdown) : "--"}
          </div>
        </Card>
      </div>

      {/* Equity Curve */}
      <Card className="fade-up fade-up-1">
        <CardHeader>
          <CardTitle>📈 Equity Curve</CardTitle>
          {history && <span className="text-[10px] text-gray-500">{history.days_tracked ?? 0} days</span>}
        </CardHeader>
        {history ? (
          <EquityCurve data={history} />
        ) : (
          <div className="flex h-44 items-center justify-center text-sm text-gray-500">
            <BarChart3 size={24} className="mr-2 text-gray-600" />
            Start trading to see your equity curve
          </div>
        )}
      </Card>

      {/* Trade Stats */}
      {perf && perf.total_trades > 0 && (
        <Card className="fade-up fade-up-2">
          <CardHeader>
            <CardTitle>
              <Target size={14} className="inline text-blue-400 mr-1" />
              Trade Statistics
            </CardTitle>
          </CardHeader>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-white">{perf.total_trades}</div>
              <div className="text-[10px] text-gray-500">Total Trades</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-400">
                {((perf.win_rate ?? 0) * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-gray-500">Win Rate</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400">
                {perf.profit_factor != null ? perf.profit_factor.toFixed(2) : "--"}
              </div>
              <div className="text-[10px] text-gray-500">Profit Factor</div>
            </div>
          </div>
          
          {/* Win/Loss bar */}
          <div className="mt-4">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] text-gray-500">W/L Distribution</span>
            </div>
            <div className="h-3 w-full rounded-full bg-white/5 overflow-hidden flex">
              <div 
                className="h-full bg-green-500/60 transition-all duration-700"
                style={{ width: `${(perf.win_rate ?? 0) * 100}%` }}
              />
              <div 
                className="h-full bg-red-500/60 transition-all duration-700"
                style={{ width: `${(1 - (perf.win_rate ?? 0)) * 100}%` }}
              />
            </div>
            <div className="flex justify-between mt-1 text-[10px]">
              <span className="text-green-400">Avg Win: ${perf.avg_win?.toFixed(0) ?? "--"}</span>
              <span className="text-red-400">Avg Loss: ${perf.avg_loss?.toFixed(0) ?? "--"}</span>
            </div>
          </div>
        </Card>
      )}

      {/* Monthly Returns */}
      {history && (
        <Card className="fade-up fade-up-3">
          <CardHeader>
            <CardTitle>Monthly Returns</CardTitle>
          </CardHeader>
          <MonthlyHeatmap data={history} />
        </Card>
      )}

      {/* Empty state */}
      {!history && (!perf || perf.total_trades === 0) && (
        <Card className="fade-up fade-up-2">
          <div className="flex flex-col items-center justify-center py-8">
            <BarChart3 size={40} className="text-gray-700 mb-3" />
            <p className="text-sm text-gray-500">No performance data yet</p>
            <p className="text-[10px] text-gray-600 mt-1">The agent needs to make trades to generate stats</p>
          </div>
        </Card>
      )}
    </div>
  );
}
