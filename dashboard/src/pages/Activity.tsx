import { useEffect, useState, useCallback } from "react";
import { Activity as ActivityIcon, RefreshCw, Filter } from "lucide-react";

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
}

const TYPE_COLORS: Record<string, string> = {
  scan_start: "text-blue-400",
  scan_result: "text-gray-400",
  signal_found: "text-green-400",
  signal_rejected: "text-red-400",
  trade_queued: "text-yellow-400",
  trade_executed: "text-emerald-400",
  regime_change: "text-purple-400",
  pillar_analysis: "text-cyan-400",
};

const TYPE_BADGES: Record<string, string> = {
  scan_start: "bg-blue-500/20 text-blue-300",
  scan_result: "bg-gray-500/20 text-gray-300",
  signal_found: "bg-green-500/20 text-green-300",
  signal_rejected: "bg-orange-500/20 text-orange-300",
  trade_queued: "bg-yellow-500/20 text-yellow-300",
  trade_executed: "bg-emerald-500/20 text-emerald-300",
  regime_change: "bg-purple-500/20 text-purple-300",
};

function ScoreBar({ label, value, max = 25 }: { label: string; value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct >= 70 ? "bg-green-500" : pct >= 50 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-12 text-gray-500">{label}</span>
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 text-right text-gray-400">{typeof value === 'number' ? value.toFixed(1) : value}</span>
    </div>
  );
}

export default function Activity() {
  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);

  const fetchActivities = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: "100" });
      if (filter) params.set("type", filter);
      const res = await fetch(`${API}/api/v1/activity?${params}`);
      const data = await res.json();
      setActivities(data.activities || []);
    } catch {
      setActivities([]);
    }
    setLoading(false);
  }, [filter]);

  useEffect(() => {
    fetchActivities();
    const interval = setInterval(fetchActivities, 30000);
    return () => clearInterval(interval);
  }, [fetchActivities]);

  const types = ["", "scan_result", "signal_found", "signal_rejected", "trade_queued", "trade_executed"];

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

      {/* Filters */}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {types.map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`px-3 py-1 rounded-full text-xs whitespace-nowrap transition-all ${
              filter === t ? "bg-gold/20 text-gold" : "bg-white/5 text-gray-400 hover:bg-white/10"
            }`}
          >
            {t || "All"}
          </button>
        ))}
      </div>

      {/* Activity list */}
      <div className="space-y-2">
        {activities.length === 0 && !loading && (
          <div className="text-center text-gray-500 py-12">No activities yet</div>
        )}
        {activities.map((a, i) => (
          <div
            key={i}
            className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-3 cursor-pointer hover:bg-white/[0.05] transition-all"
            onClick={() => setExpanded(expanded === i ? null : i)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
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
              </div>
              <span className="text-[10px] text-gray-600">
                {new Date(a.timestamp).toLocaleTimeString()}
              </span>
            </div>

            {a.details && <p className="text-xs text-gray-500 mt-1">{a.details}</p>}

            {/* Scores */}
            {a.scores && a.scores.total > 0 && (
              <div className="mt-2 flex items-center gap-3 text-xs">
                <span className="text-gray-500">Score:</span>
                <span className={`font-bold ${a.scores.total >= 70 ? "text-green-400" : a.scores.total >= 55 ? "text-yellow-400" : "text-red-400"}`}>
                  {a.scores.total?.toFixed(1)}/100
                </span>
              </div>
            )}

            {/* Expanded details */}
            {expanded === i && (
              <div className="mt-3 space-y-2 border-t border-white/[0.06] pt-3">
                {a.scores && Object.keys(a.scores).length > 0 && (
                  <div className="space-y-1">
                    <ScoreBar label="Tech" value={a.scores.technical || 0} />
                    <ScoreBar label="Funda" value={a.scores.fundamental || 0} />
                    {a.scores.ml_gate && (
                      <p className="text-[10px] mt-1">
                        {a.scores.ml_gate === "confirmed" 
                          ? <span className="text-green-400">🤖 ML Gate: ✅ confirmed (+5 pts)</span>
                          : <span className="text-gray-500">🤖 ML Gate: — no confirmation</span>
                        }
                      </p>
                    )}
                    {a.scores.intel_adj != null && a.scores.intel_adj != 0 && (
                      <p className="text-[10px] mt-1">
                        <span className={a.scores.intel_adj > 0 ? "text-cyan-400" : "text-orange-400"}>
                          🧠 Orchestrator: {a.scores.intel_adj > 0 ? "+" : ""}{Number(a.scores.intel_adj).toFixed(1)} pts
                        </span>
                      </p>
                    )}
                    {a.scores.intel_summary && (
                      <div className="mt-1 p-2 rounded bg-cyan-500/5 border border-cyan-500/10">
                        <p className="text-[10px] text-cyan-400/80">🧠 Intelligence Brief:</p>
                        <p className="text-[10px] text-gray-400 mt-0.5">{a.scores.intel_summary}</p>
                      </div>
                    )}
                  </div>
                )}
                {a.reasoning && (
                  <div className="mt-2">
                    <p className="text-[10px] text-gray-600 mb-1">Reasoning:</p>
                    <p className="text-xs text-gray-400 whitespace-pre-wrap leading-relaxed">{a.reasoning}</p>
                  </div>
                )}
                {a.strategy && (
                  <p className="text-[10px] text-gray-600">Strategy: <span className="text-gray-400">{a.strategy}</span></p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
