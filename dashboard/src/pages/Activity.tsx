import { useEffect, useState, useCallback } from "react";
import { Activity as ActivityIcon, RefreshCw, Brain, TrendingUp } from "lucide-react";

const API = import.meta.env.VITE_API_URL || "";

interface ActivityEntry {
  timestamp: string;
  type: string;
  symbol: string;
  details: string;
  scores: Record<string, any>;
  reasoning: string;
  strategy: string;
  decision: string;
  agent: string;
}

const TYPE_BADGES: Record<string, string> = {
  scan_start: "bg-blue-500/20 text-blue-300",
  scan_result: "bg-gray-500/20 text-gray-300",
  signal_found: "bg-green-500/20 text-green-300",
  signal_rejected: "bg-orange-500/20 text-orange-300",
  trade_queued: "bg-yellow-500/20 text-yellow-300",
  trade_executed: "bg-emerald-500/20 text-emerald-300",
  regime_change: "bg-purple-500/20 text-purple-300",
  pending_confirmation: "bg-orange-500/20 text-orange-300",
};

function ScoreBar({ label, value, max = 100 }: { label: string; value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct >= 70 ? "bg-green-500" : pct >= 50 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-14 text-gray-500 shrink-0">{label}</span>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-right text-gray-400 font-mono">{typeof value === 'number' ? value.toFixed(1) : value}</span>
    </div>
  );
}

function AgentBadge({ agent }: { agent: string }) {
  if (agent === "llm") return <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-amber-500/20 text-amber-400">🧠 LLM</span>;
  if (agent === "nollm") return <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-blue-500/20 text-blue-400">📊 NoLLM</span>;
  return null;
}

export default function Activity() {
  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [agentFilter, setAgentFilter] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);

  const fetchActivities = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: "100" });
      if (filter) params.set("type", filter);
      if (agentFilter) params.set("agent", agentFilter);
      const res = await fetch(`${API}/api/v1/activity?${params}`);
      const data = await res.json();
      setActivities(data.activities || []);
    } catch {
      setActivities([]);
    }
    setLoading(false);
  }, [filter, agentFilter]);

  useEffect(() => {
    fetchActivities();
    const interval = setInterval(fetchActivities, 30000);
    return () => clearInterval(interval);
  }, [fetchActivities]);

  const types = ["", "scan_result", "signal_found", "signal_rejected", "trade_executed"];
  const agents = ["", "llm", "nollm"];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <ActivityIcon size={22} className="text-gold" />
          Activity Feed
        </h1>
        <button onClick={fetchActivities} className="p-2 rounded-lg bg-white/5 hover:bg-white/10">
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* Agent filter */}
      <div className="flex gap-2">
        {agents.map((a) => (
          <button
            key={a}
            onClick={() => setAgentFilter(a)}
            className={`px-3 py-1 rounded-full text-xs whitespace-nowrap transition-all ${
              agentFilter === a ? "bg-gold/20 text-gold" : "bg-white/5 text-gray-400 hover:bg-white/10"
            }`}
          >
            {a === "" ? "All Agents" : a === "llm" ? "🧠 LLM" : "📊 NoLLM"}
          </button>
        ))}
      </div>

      {/* Type filters */}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {types.map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`px-3 py-1 rounded-full text-xs whitespace-nowrap transition-all ${
              filter === t ? "bg-gold/20 text-gold" : "bg-white/5 text-gray-400 hover:bg-white/10"
            }`}
          >
            {t || "All Types"}
          </button>
        ))}
      </div>

      {/* Activity list */}
      <div className="space-y-2">
        {activities.length === 0 && !loading && (
          <div className="text-center text-gray-500 py-12">No activities yet</div>
        )}
        {activities.map((a, i) => {
          const s = a.scores || {};
          const hasIntel = s.intel_adj != null && s.intel_adj !== 0;
          const hasNews = s.intel_news && s.intel_news.length > 0;
          const hasIntelSummary = !!s.intel_summary;
          const hasIntelReasoning = !!s.intel_reasoning;

          return (
            <div
              key={i}
              className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-3 cursor-pointer hover:bg-white/[0.05] transition-all"
              onClick={() => setExpanded(expanded === i ? null : i)}
            >
              {/* Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 flex-wrap">
                  <AgentBadge agent={a.agent} />
                  <span className={`px-2 py-0.5 rounded-md text-[10px] font-medium ${TYPE_BADGES[a.type] || "bg-white/10 text-gray-300"}`}>
                    {a.type}
                  </span>
                  {a.symbol && <span className="font-mono font-bold text-sm">{a.symbol}</span>}
                  {a.decision && (
                    <span className={`text-xs font-medium ${
                      a.decision.includes("PENDING") ? "text-orange-400" :
                      a.decision.includes("BUY") ? "text-green-400" :
                      a.decision === "SKIP" || a.decision === "HOLD" ? "text-gray-400" : "text-yellow-400"
                    }`}>
                      {a.decision}
                    </span>
                  )}
                  {hasIntel && (
                    <span className={`text-[10px] font-medium ${s.intel_adj > 0 ? "text-cyan-400" : "text-orange-400"}`}>
                      🧠 {s.intel_adj > 0 ? "+" : ""}{s.intel_adj.toFixed(1)}
                    </span>
                  )}
                  {hasNews && (
                    <span className="text-[10px] text-cyan-500/60">📰 {s.intel_news.length}</span>
                  )}
                </div>
                <span className="text-[10px] text-gray-600">
                  {new Date(a.timestamp).toLocaleTimeString()}
                </span>
              </div>

              {a.details && <p className="text-xs text-gray-500 mt-1">{a.details}</p>}

              {/* Score summary */}
              {s.total > 0 && (
                <div className="mt-2 flex items-center gap-3 text-xs">
                  <TrendingUp size={12} className="text-gray-500" />
                  <span className={`font-bold ${s.total >= 70 ? "text-green-400" : s.total >= 55 ? "text-yellow-400" : "text-red-400"}`}>
                    {s.total?.toFixed(1)}/100
                  </span>
                  <span className="text-gray-600">T:{s.technical?.toFixed(0)} F:{s.fundamental?.toFixed(0)}</span>
                  {s.ml_gate === "confirmed" && <span className="text-green-400/60 text-[10px]">✅ ML</span>}
                </div>
              )}

              {/* Expanded details */}
              {expanded === i && (
                <div className="mt-3 space-y-3 border-t border-white/[0.06] pt-3">
                  {s.technical != null && (
                    <div className="space-y-1.5">
                      <ScoreBar label="Technical" value={s.technical} />
                      <ScoreBar label="Funda" value={s.fundamental || 0} />
                      {s.ml_gate && (
                        <p className="text-[10px] mt-1">
                          {s.ml_gate === "confirmed"
                            ? <span className="text-green-400">🤖 ML Gate: ✅ confirmed (+5 pts)</span>
                            : <span className="text-gray-500">🤖 ML Gate: — no confirmation</span>
                          }
                        </p>
                      )}
                    </div>
                  )}

                  {/* Intelligence section (LLM only) */}
                  {(hasIntel || hasIntelSummary || hasIntelReasoning || hasNews) && (
                    <div className="p-2.5 rounded-lg bg-cyan-500/5 border border-cyan-500/10 space-y-2">
                      <div className="flex items-center gap-1.5">
                        <Brain size={12} className="text-cyan-400" />
                        <span className="text-[11px] font-medium text-cyan-400">Intelligence Orchestrator</span>
                      </div>
                      {hasIntel && (
                        <p className="text-xs">
                          <span className={s.intel_adj > 0 ? "text-cyan-400" : "text-orange-400"}>
                            Score adjustment: {s.intel_adj > 0 ? "+" : ""}{s.intel_adj.toFixed(1)} pts
                          </span>
                        </p>
                      )}
                      {hasIntelReasoning && (
                        <p className="text-[11px] text-gray-300">{s.intel_reasoning}</p>
                      )}
                      {hasIntelSummary && (
                        <div>
                          <p className="text-[10px] text-cyan-400/60 mb-0.5">Macro Brief:</p>
                          <p className="text-[10px] text-gray-400 leading-relaxed">{s.intel_summary}</p>
                        </div>
                      )}
                      {hasNews && (
                        <div>
                          <p className="text-[10px] text-cyan-400/60 mb-0.5">Related News ({s.intel_news.length}):</p>
                          <ul className="space-y-0.5">
                            {s.intel_news.map((n: string, j: number) => (
                              <li key={j} className="text-[10px] text-gray-400">📰 {n}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}

                  {/* No intelligence for NoLLM */}
                  {a.agent === "nollm" && (
                    <div className="p-2 rounded-lg bg-gray-500/5 border border-gray-500/10">
                      <p className="text-[10px] text-gray-500">📊 NoLLM agent — pure technical + fundamental scoring (no intelligence orchestrator)</p>
                    </div>
                  )}

                  {a.reasoning && (
                    <div>
                      <p className="text-[10px] text-gray-600 mb-1">Analysis Reasoning:</p>
                      <p className="text-xs text-gray-400 whitespace-pre-wrap leading-relaxed">{a.reasoning}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
