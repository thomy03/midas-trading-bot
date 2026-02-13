import { usePortfolioSummary, useAgentStatus, useAlerts } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PnlBadge } from "@/components/shared/PnlBadge";
import { RegimeBadge } from "@/components/shared/RegimeBadge";
import { CardSkeleton } from "@/components/shared/LoadingSkeleton";
import { GaugeRing } from "@/components/shared/GaugeRing";
import { formatCurrency } from "@/lib/utils";
import { StrategySplit } from "@/components/shared/StrategySplit";
import { Activity, TrendingUp, AlertTriangle, Flame, Brain, Zap, Timer } from "lucide-react";

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

  const metrics = agent?.metrics;
  const brief = agent?.intelligence_brief;

  return (
    <div className="space-y-4">
      {/* Hero: Portfolio Value */}
      <Card variant="accent" className="fade-up">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Portfolio Value</div>
            <div className="text-3xl font-bold text-white tracking-tight">
              {portfolio ? formatCurrency(portfolio.total_capital) : "--"}
            </div>
            {portfolio && (
              <div className="mt-1">
                <PnlBadge value={portfolio.unrealized_pnl} mode="currency" size="sm" />
              </div>
            )}
          </div>
          <div className="flex flex-col items-end gap-1">
            <div className={`h-3 w-3 rounded-full ${agent?.running ? "bg-green-500 pulse-glow" : "bg-gray-600"}`} />
            <span className="text-[10px] text-gray-500">
              {agent?.running ? "Live" : "Offline"}
            </span>
          </div>
        </div>
        {portfolio?.strategies && <StrategySplit strategies={portfolio.strategies} />}
        <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <div className="h-1.5 w-1.5 rounded-full bg-blue-400" />
            Invested: {portfolio ? formatCurrency(portfolio.invested_capital) : "--"}
          </span>
          <span className="flex items-center gap-1">
            <div className="h-1.5 w-1.5 rounded-full bg-green-400" />
            Cash: {portfolio ? formatCurrency(portfolio.available_capital) : "--"}
          </span>
        </div>
      </Card>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-3 fade-up fade-up-1">
        <Card className="text-center">
          <Activity size={16} className="mx-auto text-blue-400 mb-1" />
          <div className="text-xl font-bold text-white">{portfolio?.open_positions ?? 0}</div>
          <div className="text-[10px] text-gray-500">Positions</div>
        </Card>
        <Card className="text-center">
          <TrendingUp size={16} className="mx-auto text-purple-400 mb-1" />
          {agent ? <RegimeBadge regime={agent.market_regime} /> : <span className="text-sm text-gray-500">--</span>}
          <div className="text-[10px] text-gray-500 mt-1">Regime</div>
        </Card>
        <Card className="text-center">
          <Timer size={16} className="mx-auto text-gold mb-1" />
          <div className="text-xl font-bold text-white">{metrics?.cycles ?? 0}</div>
          <div className="text-[10px] text-gray-500">Cycles</div>
        </Card>
      </div>

      {/* Agent Brain Activity */}
      {metrics && (
        <Card className="fade-up fade-up-2">
          <CardHeader>
            <CardTitle>
              <Brain size={14} className="inline text-purple-400 mr-1" />
              Midas Brain
            </CardTitle>
            <Badge variant={agent?.running ? "success" : "default"}>
              {agent?.phase ?? "idle"}
            </Badge>
          </CardHeader>
          <div className="flex items-center justify-around">
            <GaugeRing 
              value={metrics.signals ?? 0} 
              max={Math.max(metrics.signals ?? 1, 1)} 
              color="#7c3aed" 
              label="Signals" 
              size={64} 
            />
            <GaugeRing 
              value={metrics.trades ?? 0} 
              max={Math.max(metrics.signals ?? 1, 1)} 
              color="#22c55e" 
              label="Trades" 
              size={64} 
            />
            <GaugeRing 
              value={metrics.screens ?? 0} 
              max={Math.max(metrics.screens ?? 1, 1)} 
              color="#3b82f6" 
              label="Screens" 
              size={64} 
            />
          </div>
        </Card>
      )}

      {/* Intelligence Brief */}
      {brief?.reasoning_summary && (
        <Card variant="accent" className="fade-up fade-up-3">
          <CardHeader>
            <CardTitle>
              <Brain size={14} className="inline text-purple-400 mr-1" />
              Midas Thinks...
            </CardTitle>
            <span className="text-[10px] text-gray-500">
              {brief.timestamp ? new Date(brief.timestamp).toLocaleTimeString() : ""}
            </span>
          </CardHeader>
          <p className="text-sm text-gray-300 leading-relaxed italic">
            "{brief.reasoning_summary}"
          </p>
        </Card>
      )}

      {/* Hot Symbols */}
      {agent?.hot_symbols && agent.hot_symbols.length > 0 && (
        <Card className="fade-up fade-up-3">
          <CardHeader>
            <CardTitle>
              <Flame size={14} className="inline text-orange-400 mr-1" />
              Hot Symbols
            </CardTitle>
          </CardHeader>
          <div className="flex flex-wrap gap-2">
            {agent.hot_symbols.map((s) => (
              <span key={s} className="rounded-lg bg-orange-500/10 border border-orange-500/20 px-3 py-1 text-xs font-medium text-orange-300">
                {s}
              </span>
            ))}
          </div>
        </Card>
      )}

      {/* Recent Alerts */}
      {alerts && alerts.length > 0 && (
        <Card className="fade-up fade-up-4">
          <CardHeader>
            <CardTitle>
              <Zap size={14} className="inline text-yellow-400 mr-1" />
              Latest Signals
            </CardTitle>
            <span className="text-[10px] text-gray-500">{alerts.length} recent</span>
          </CardHeader>
          <div className="space-y-2">
            {alerts.slice(0, 5).map((a) => (
              <div
                key={a.id}
                className="flex items-center justify-between rounded-xl bg-white/[0.03] px-3 py-2.5 transition-colors hover:bg-white/[0.06]"
              >
                <div>
                  <span className="font-semibold text-white">{a.symbol}</span>
                  <span className="ml-2 text-[10px] text-gray-500">{a.timeframe}</span>
                </div>
                <Badge
                  variant={a.recommendation.includes("BUY") ? "success" : a.recommendation.includes("SELL") ? "danger" : "default"}
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
