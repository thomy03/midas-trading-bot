import { useAgentStatus, useAlerts } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { Brain, AlertCircle, Zap, TrendingUp } from "lucide-react";

export default function Signals() {
  const { data: agent, isLoading: aLoading } = useAgentStatus();
  const { data: alerts, isLoading: alLoading } = useAlerts(7);

  if (aLoading || alLoading) return <LoadingSkeleton rows={4} />;

  const brief = agent?.intelligence_brief;

  return (
    <div className="space-y-3">
      {/* Intelligence Brief */}
      {brief && (
        <Card>
          <CardHeader>
            <CardTitle>
              <Brain size={14} className="inline text-purple-400" /> Intelligence
              Brief
            </CardTitle>
            <span className="text-[10px] text-gray-500">
              {brief.timestamp
                ? new Date(brief.timestamp).toLocaleTimeString()
                : ""}
            </span>
          </CardHeader>
          {brief.reasoning_summary && (
            <p className="text-sm text-gray-300">{brief.reasoning_summary}</p>
          )}
        </Card>
      )}

      {/* Market Events */}
      {brief?.market_events && brief.market_events.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>
              <AlertCircle size={14} className="inline text-blue-400" /> Market
              Events
            </CardTitle>
          </CardHeader>
          <div className="space-y-1.5">
            {brief.market_events.map((evt, i) => (
              <div
                key={i}
                className="rounded-lg bg-surface-2 px-3 py-2 text-xs text-gray-300"
              >
                {typeof evt === "string" ? evt : JSON.stringify(evt)}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Megatrends */}
      {brief?.megatrends && brief.megatrends.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>
              <TrendingUp size={14} className="inline text-gold" /> Megatrends
            </CardTitle>
          </CardHeader>
          <div className="flex flex-wrap gap-1.5">
            {brief.megatrends.map((t, i) => (
              <Badge key={i} variant="info">
                {typeof t === "string" ? t : JSON.stringify(t)}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      {/* Portfolio Alerts */}
      {brief?.portfolio_alerts && brief.portfolio_alerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>
              <Zap size={14} className="inline text-yellow-400" /> Portfolio
              Alerts
            </CardTitle>
          </CardHeader>
          <div className="space-y-1.5">
            {brief.portfolio_alerts.map((a, i) => (
              <div
                key={i}
                className="rounded-lg bg-yellow-500/10 px-3 py-2 text-xs text-yellow-300"
              >
                {typeof a === "string" ? a : JSON.stringify(a)}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Recent Signals */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Signals (7d)</CardTitle>
          <span className="text-xs text-gray-500">
            {alerts?.length ?? 0} alerts
          </span>
        </CardHeader>
        {alerts && alerts.length > 0 ? (
          <div className="space-y-2">
            {alerts.map((a) => (
              <div
                key={a.id}
                className="flex items-center justify-between rounded-lg bg-surface-2 px-3 py-2"
              >
                <div>
                  <span className="font-medium text-white">{a.symbol}</span>
                  <span className="ml-2 text-[10px] text-gray-500">
                    ${(a.current_price ?? 0).toFixed(2)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {a.confidence_score && (
                    <span className="text-xs text-gray-400">
                      {(a.confidence_score ?? 0).toFixed(0)}
                    </span>
                  )}
                  <Badge
                    variant={
                      a.recommendation.includes("BUY")
                        ? "success"
                        : a.recommendation.includes("SELL")
                        ? "danger"
                        : "default"
                    }
                  >
                    {a.recommendation}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-4 text-center text-sm text-gray-500">
            No signals yet
          </div>
        )}
      </Card>
    </div>
  );
}
