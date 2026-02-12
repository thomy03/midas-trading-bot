import { useAgentStatus, useAlerts } from "@/api/hooks";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MiniGauge } from "@/components/shared/GaugeRing";
import { ThoughtProcess, buildThoughtSteps } from "@/components/shared/ThoughtProcess";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { PILLAR_COLORS } from "@/lib/constants";
import { Brain, Zap, TrendingUp, ChevronDown, ChevronUp, AlertCircle } from "lucide-react";
import { useState } from "react";

function useSignals(limit = 50) {
  return useQuery({
    queryKey: ["signals", limit],
    queryFn: () => apiFetch<{ signals: any[]; total: number }>(`/api/v1/signals?limit=${limit}`),
    refetchInterval: 30_000,
  });
}

export default function Signals() {
  const { data: agent, isLoading: aLoading } = useAgentStatus();
  const { data: alerts, isLoading: alLoading } = useAlerts(7);
  const { data: signalsData } = useSignals();
  const [expandedId, setExpandedId] = useState<number | null>(null);

  if (aLoading || alLoading) return <LoadingSkeleton rows={4} />;

  const brief = agent?.intelligence_brief;
  const signals = signalsData?.signals ?? [];
  // Use alerts as fallback if no signals endpoint data
  const displayItems = signals.length > 0 ? signals : (alerts ?? []);

  return (
    <div className="space-y-4">
      {/* Intelligence Brief */}
      {brief?.reasoning_summary && (
        <Card variant="accent" className="fade-up">
          <CardHeader>
            <CardTitle>
              <Brain size={14} className="inline text-purple-400 mr-1" />
              Midas Reasoning
            </CardTitle>
            <span className="text-[10px] text-gray-500">
              {brief.timestamp ? new Date(brief.timestamp).toLocaleTimeString() : ""}
            </span>
          </CardHeader>
          <p className="text-sm text-gray-300 leading-relaxed italic">
            "{brief.reasoning_summary}"
          </p>

          {/* Market Events */}
          {brief.market_events && brief.market_events.length > 0 && (
            <div className="mt-3 space-y-1.5">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider">Market Events</div>
              {brief.market_events.map((evt, i) => (
                <div key={i} className="flex items-start gap-2 rounded-lg bg-white/[0.03] px-3 py-2">
                  <AlertCircle size={12} className="text-blue-400 mt-0.5 shrink-0" />
                  <span className="text-xs text-gray-300">
                    {typeof evt === "string" ? evt : JSON.stringify(evt)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Megatrends */}
          {brief.megatrends && brief.megatrends.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {brief.megatrends.map((t, i) => (
                <span key={i} className="rounded-lg bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 text-[10px] text-purple-300">
                  {typeof t === "string" ? t : JSON.stringify(t)}
                </span>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* V8.1 Architecture Explanation */}
      <Card className="fade-up fade-up-1">
        <CardHeader>
          <CardTitle>
            <Zap size={14} className="inline text-gold mr-1" />
            How Midas Decides (V8.1)
          </CardTitle>
        </CardHeader>
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="rounded-xl bg-blue-500/8 border border-blue-500/15 p-2.5">
            <div className="font-semibold text-blue-400 mb-1">📊 Technical (55%)</div>
            <div className="text-gray-400">Momentum, trend, patterns</div>
          </div>
          <div className="rounded-xl bg-green-500/8 border border-green-500/15 p-2.5">
            <div className="font-semibold text-green-400 mb-1">📈 Fundamental (45%)</div>
            <div className="text-gray-400">Value, growth, quality</div>
          </div>
          <div className="rounded-xl bg-pink-500/8 border border-pink-500/15 p-2.5">
            <div className="font-semibold text-pink-400 mb-1">🤖 ML Gate</div>
            <div className="text-gray-400">Blocks &lt;40, boosts &gt;60</div>
          </div>
          <div className="rounded-xl bg-purple-500/8 border border-purple-500/15 p-2.5">
            <div className="font-semibold text-purple-400 mb-1">🧠 Orchestrator</div>
            <div className="text-gray-400">LLM reasoning + sentiment</div>
          </div>
        </div>
      </Card>

      {/* Signals List */}
      <div className="fade-up fade-up-2">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-300">
            <Zap size={14} className="inline text-yellow-400 mr-1" />
            Recent Signals
          </h2>
          <span className="text-[10px] text-gray-500">{displayItems.length} signals</span>
        </div>

        {displayItems.length === 0 ? (
          <Card>
            <div className="py-8 text-center">
              <Brain size={32} className="mx-auto text-gray-600 mb-2" />
              <p className="text-sm text-gray-500">Midas is analyzing markets...</p>
              <p className="text-[10px] text-gray-600 mt-1">Signals will appear here when detected</p>
            </div>
          </Card>
        ) : (
          <div className="space-y-2">
            {displayItems.map((item: any, idx: number) => {
              const id = item.id ?? idx;
              const isExpanded = expandedId === id;
              const symbol = item.symbol ?? "???";
              const rec = item.recommendation ?? item.decision ?? "UNKNOWN";
              const score = item.confidence_score ?? item.score_at_entry ?? 0;
              const tech = item.pillar_technical ?? item.technical_score ?? 0;
              const fund = item.pillar_fundamental ?? item.fundamental_score ?? 0;

              return (
                <Card
                  key={id}
                  className="cursor-pointer transition-all duration-200 hover:border-white/10"
                  onClick={() => setExpandedId(isExpanded ? null : id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex flex-col items-center">
                        <div className={`text-lg font-bold ${score > 65 ? "text-green-400" : score > 50 ? "text-yellow-400" : "text-gray-400"}`}>
                          {score > 0 ? score.toFixed(0) : "--"}
                        </div>
                        <div className="text-[8px] text-gray-600">score</div>
                      </div>
                      <div>
                        <div className="font-semibold text-white">{symbol}</div>
                        <div className="text-[10px] text-gray-500">
                          ${(item.current_price ?? item.price ?? 0).toFixed(2)}
                          {item.timeframe && <span className="ml-1">{item.timeframe}</span>}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant={rec.includes("BUY") || rec.includes("STRONG") ? "success" : rec.includes("SELL") ? "danger" : "default"}
                      >
                        {rec}
                      </Badge>
                      {isExpanded ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
                    </div>
                  </div>

                  {/* Pillar mini-gauges */}
                  {(tech > 0 || fund > 0) && (
                    <div className="mt-3 grid grid-cols-2 gap-3">
                      <MiniGauge value={tech} max={25} color={PILLAR_COLORS.technical} label="Technical" />
                      <MiniGauge value={fund} max={25} color={PILLAR_COLORS.fundamental} label="Fundamental" />
                    </div>
                  )}

                  {/* Expanded: Thought Process */}
                  {isExpanded && (
                    <div className="mt-4 pt-3 border-t border-white/5">
                      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-3">
                        🧠 Midas Thought Process
                      </div>
                      <ThoughtProcess {...buildThoughtSteps(item)} />
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
