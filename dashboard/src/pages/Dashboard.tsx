import { usePortfolioSummary, useAgentStatus, useAlerts } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PnlBadge } from "@/components/shared/PnlBadge";
import { RegimeBadge } from "@/components/shared/RegimeBadge";
import { CardSkeleton } from "@/components/shared/LoadingSkeleton";
import { formatCurrency } from "@/lib/utils";
import { Activity, TrendingUp, AlertTriangle, Flame } from "lucide-react";

export default function Dashboard() {
  const { data: portfolio, isLoading: pLoading } = usePortfolioSummary();
  const { data: agent, isLoading: aLoading } = useAgentStatus();
  const { data: alerts } = useAlerts(3);

  if (pLoading || aLoading) {
    return (
      <div className="space-y-3">
        <CardSkeleton />
        <CardSkeleton />
        <CardSkeleton />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Portfolio Value */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Value</CardTitle>
          {portfolio && (
            <PnlBadge value={portfolio.unrealized_pnl} mode="currency" size="sm" />
          )}
        </CardHeader>
        <div className="text-3xl font-bold text-white">
          {portfolio ? formatCurrency(portfolio.total_capital) : "--"}
        </div>
        <div className="mt-1 flex items-center gap-3 text-xs text-gray-500">
          <span>
            Invested: {portfolio ? formatCurrency(portfolio.invested_capital) : "--"}
          </span>
          <span>
            Cash: {portfolio ? formatCurrency(portfolio.available_capital) : "--"}
          </span>
        </div>
      </Card>

      {/* Stats Row */}
      <div className="grid grid-cols-2 gap-3">
        <Card>
          <div className="flex items-center gap-2 text-gray-400">
            <Activity size={14} />
            <span className="text-xs">Positions</span>
          </div>
          <div className="mt-1 text-xl font-bold text-white">
            {portfolio?.open_positions ?? 0}
          </div>
        </Card>
        <Card>
          <div className="flex items-center gap-2 text-gray-400">
            <TrendingUp size={14} />
            <span className="text-xs">Regime</span>
          </div>
          <div className="mt-1">
            {agent ? (
              <RegimeBadge regime={agent.market_regime} />
            ) : (
              <span className="text-sm text-gray-500">--</span>
            )}
          </div>
        </Card>
      </div>

      {/* Agent Metrics */}
      {agent?.metrics && (
        <Card>
          <CardHeader>
            <CardTitle>Agent Metrics</CardTitle>
            <Badge variant={agent.running ? "success" : "default"}>
              {agent.running ? "Running" : "Stopped"}
            </Badge>
          </CardHeader>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div>
              <div className="text-lg font-bold text-white">
                {agent.metrics.cycles ?? 0}
              </div>
              <div className="text-gray-500">Cycles</div>
            </div>
            <div>
              <div className="text-lg font-bold text-white">
                {agent.metrics.signals ?? 0}
              </div>
              <div className="text-gray-500">Signals</div>
            </div>
            <div>
              <div className="text-lg font-bold text-white">
                {agent.metrics.trades ?? 0}
              </div>
              <div className="text-gray-500">Trades</div>
            </div>
          </div>
        </Card>
      )}

      {/* Hot Symbols */}
      {agent?.hot_symbols && agent.hot_symbols.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>
              <Flame size={14} className="inline text-orange-400" /> Hot Symbols
            </CardTitle>
          </CardHeader>
          <div className="flex flex-wrap gap-1.5">
            {agent.hot_symbols.map((s) => (
              <Badge key={s} variant="warning">
                {s}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      {/* Recent Alerts */}
      {alerts && alerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>
              <AlertTriangle size={14} className="inline text-yellow-400" />{" "}
              Recent Alerts
            </CardTitle>
          </CardHeader>
          <div className="space-y-2">
            {alerts.slice(0, 5).map((a) => (
              <div
                key={a.id}
                className="flex items-center justify-between rounded-lg bg-surface-2 px-3 py-2"
              >
                <div>
                  <span className="font-medium text-white">{a.symbol}</span>
                  <span className="ml-2 text-xs text-gray-500">
                    {a.timeframe}
                  </span>
                </div>
                <Badge
                  variant={
                    a.recommendation.includes("BUY") ? "success" : "default"
                  }
                >
                  {a.recommendation}
                </Badge>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
