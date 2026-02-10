import { useAgentStatus, useHealth, useLearningWeights } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { RegimeBadge } from "@/components/shared/RegimeBadge";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { PILLAR_COLORS } from "@/lib/constants";

export default function Settings() {
  const { data: agent, isLoading: aLoading } = useAgentStatus();
  const { data: health } = useHealth();
  const { data: weights } = useLearningWeights();

  if (aLoading) return <LoadingSkeleton rows={4} />;

  return (
    <div className="space-y-3">
      {/* Agent Status */}
      <Card>
        <CardHeader>
          <CardTitle>Agent Status</CardTitle>
          <Badge variant={agent?.running ? "success" : "default"}>
            {agent?.running ? "Running" : "Stopped"}
          </Badge>
        </CardHeader>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <span className="text-gray-500">Phase:</span>{" "}
            <span className="text-white">{agent?.phase ?? "--"}</span>
          </div>
          <div>
            <span className="text-gray-500">Regime:</span>{" "}
            {agent ? <RegimeBadge regime={agent.market_regime} /> : "--"}
          </div>
          <div>
            <span className="text-gray-500">Errors:</span>{" "}
            <span className="text-white">
              {agent?.metrics?.errors ?? 0}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Avg Cycle:</span>{" "}
            <span className="text-white">
              {agent?.metrics?.avg_cycle_ms
                ? `${agent.metrics.avg_cycle_ms.toFixed(0)}ms`
                : "--"}
            </span>
          </div>
        </div>
      </Card>

      {/* API Health */}
      <Card>
        <CardHeader>
          <CardTitle>API Health</CardTitle>
          <Badge variant={health?.status === "healthy" ? "success" : "danger"}>
            {health?.status ?? "unknown"}
          </Badge>
        </CardHeader>
        <div className="text-xs text-gray-500">
          Version: {health?.version ?? "--"} | Last check:{" "}
          {health?.timestamp
            ? new Date(health.timestamp).toLocaleTimeString()
            : "--"}
        </div>
      </Card>

      {/* Pillar Weights */}
      {weights?.weights && (
        <Card>
          <CardHeader>
            <CardTitle>Pillar Weights</CardTitle>
          </CardHeader>
          <div className="space-y-3">
            {Object.entries(weights.weights).map(([pillar, data]) => (
              <div key={pillar}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span
                    className="font-medium capitalize"
                    style={{
                      color:
                        PILLAR_COLORS[pillar as keyof typeof PILLAR_COLORS] ??
                        "#9ca3af",
                    }}
                  >
                    {pillar}
                  </span>
                  <span className="text-gray-400">
                    {(data.weight * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-surface-3">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${data.weight * 100}%`,
                      backgroundColor:
                        PILLAR_COLORS[pillar as keyof typeof PILLAR_COLORS] ??
                        "#6b7280",
                    }}
                  />
                </div>
                {data.top_indicators && data.top_indicators.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {data.top_indicators.slice(0, 3).map((ind) => (
                      <span
                        key={ind.name}
                        className="text-[9px] text-gray-500"
                      >
                        {ind.name} ({(ind.accuracy * 100).toFixed(0)}%)
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Profile Info */}
      <Card>
        <CardHeader>
          <CardTitle>About</CardTitle>
        </CardHeader>
        <div className="space-y-1 text-xs text-gray-400">
          <div>Midas Trading Bot v8</div>
          <div>5 Pillars + ML Gate + Regime Detection</div>
          <div>Dashboard v1.0</div>
        </div>
      </Card>
    </div>
  );
}
