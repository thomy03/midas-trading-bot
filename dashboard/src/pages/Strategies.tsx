import React from "react";
import { useAgent } from "@/contexts/AgentContext";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { formatPct, formatCurrency } from "@/lib/utils";
import { Brain, Shield, ChevronRight, Zap, ShieldCheck, Trophy, TrendingUp, TrendingDown, Target } from "lucide-react";

interface StrategyData {
  id: string;
  name: string;
  color: string;
  equity: number;
  cash: number;
  return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  max_drawdown: number;
  open_positions: number;
  signals_evaluated: number;
  signals_accepted: number;
  signals_rejected: number;
  equity_curve: { date: string; equity: number }[];
  recent_trades: any[];
  positions: any[];
  agent?: string;
  strategy_id?: string;
}

interface ComparisonResponse {
  strategies: StrategyData[];
  updated_at: string;
}

function useStrategyComparison() {
  return useQuery({
    queryKey: ["strategies", "comparison"],
    queryFn: () => apiFetch<ComparisonResponse>("/api/v1/strategies/comparison"),
    refetchInterval: 30_000,
  });
}

function isAggressive(s: StrategyData) {
  return s.id?.includes("aggressive") || s.strategy_id?.includes("aggressive");
}
function isLLM(s: StrategyData) {
  return s.agent === "llm" || s.id?.startsWith("llm_");
}

/* ─── Agent Toggle ─── */
function AgentToggle({ active, onChange }: { active: "llm" | "nollm"; onChange: (v: "llm" | "nollm") => void }) {
  return (
    <div className="flex rounded-xl bg-white/[0.04] border border-white/[0.08] p-1 gap-1">
      <button
        onClick={() => onChange("llm")}
        className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-sm font-medium transition-all duration-200 ${
          active === "llm"
            ? "bg-purple-500/20 text-purple-300 border border-purple-500/30 shadow-lg shadow-purple-500/10"
            : "text-gray-500 hover:text-gray-300 hover:bg-white/[0.03]"
        }`}
      >
        <Brain size={16} />
        <span>LLM Agent</span>
      </button>
      <button
        onClick={() => onChange("nollm")}
        className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-sm font-medium transition-all duration-200 ${
          active === "nollm"
            ? "bg-blue-500/20 text-blue-300 border border-blue-500/30 shadow-lg shadow-blue-500/10"
            : "text-gray-500 hover:text-gray-300 hover:bg-white/[0.03]"
        }`}
      >
        <Shield size={16} />
        <span>No-LLM</span>
      </button>
    </div>
  );
}

/* ─── Agent Summary ─── */
function AgentSummary({ strategies, agent }: { strategies: StrategyData[]; agent: "llm" | "nollm" }) {
  const totalEquity = strategies.reduce((s, st) => s + st.equity, 0);
  const totalInitial = strategies.length * 15000;
  const totalReturn = totalInitial > 0 ? ((totalEquity - totalInitial) / totalInitial) * 100 : 0;
  const totalPos = strategies.reduce((s, st) => s + st.open_positions, 0);
  const totalTrades = strategies.reduce((s, st) => s + st.total_trades, 0);
  const totalWins = strategies.reduce((s, st) => s + st.winning_trades, 0);
  const winRate = totalTrades > 0 ? (totalWins / totalTrades) * 100 : 0;

  const borderColor = agent === "llm" ? "border-purple-500/20" : "border-blue-500/20";
  const accentColor = agent === "llm" ? "text-purple-400" : "text-blue-400";

  return (
    <div className={`rounded-2xl border ${borderColor} p-4`} style={{ background: "rgba(255,255,255,0.02)" }}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {agent === "llm" 
            ? <Brain size={18} className="text-purple-400" />
            : <Shield size={18} className="text-blue-400" />
          }
          <div>
            <div className="text-xs font-bold text-white">
              {agent === "llm" ? "Grok + Gemini Reasoning" : "Pure Technical + Fundamental"}
            </div>
            <div className="text-[10px] text-gray-500">
              {strategies.length} strategies · ${(totalInitial / 1000).toFixed(0)}K capital
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className={`text-lg font-bold ${totalReturn >= 0 ? "text-green-400" : "text-red-400"}`}>
            {formatPct(totalReturn)}
          </div>
          <div className="text-[10px] text-gray-500">{formatCurrency(totalEquity)}</div>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2 text-center">
        {[
          { label: "Positions", value: totalPos.toString(), color: "text-white" },
          { label: "Trades", value: totalTrades.toString(), color: "text-white" },
          { label: "Win Rate", value: winRate > 0 ? `${winRate.toFixed(0)}%` : "--", color: winRate >= 50 ? "text-green-400" : "text-white" },
          { label: "P&L", value: formatCurrency(strategies.reduce((s, st) => s + st.total_pnl, 0)), color: strategies.reduce((s, st) => s + st.total_pnl, 0) >= 0 ? "text-green-400" : "text-red-400" },
        ].map(kpi => (
          <div key={kpi.label}>
            <div className={`text-sm font-bold ${kpi.color}`}>{kpi.value}</div>
            <div className="text-[9px] text-gray-500">{kpi.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Strategy Card (detailed) ─── */
function StrategyCard({ s, onClick }: { s: StrategyData; onClick: () => void }) {
  const aggressive = isAggressive(s);

  return (
    <div
      className="rounded-2xl bg-white/[0.03] border border-white/[0.06] p-4 cursor-pointer hover:bg-white/[0.06] transition-all duration-200 active:scale-[0.98]"
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {aggressive
            ? <Zap size={16} className="text-red-400" />
            : <ShieldCheck size={16} className="text-green-400" />
          }
          <span className="text-sm font-bold text-white">
            {aggressive ? "Aggressive" : "Moderate"}
          </span>
          <span className={`text-[9px] px-2 py-0.5 rounded-full ${aggressive ? "bg-red-500/10 text-red-400 border border-red-500/20" : "bg-green-500/10 text-green-400 border border-green-500/20"}`}>
            {aggressive ? "min 70" : "min 78"}
          </span>
        </div>
        <ChevronRight size={16} className="text-gray-600" />
      </div>

      {/* Equity & Return */}
      <div className="flex items-end justify-between mb-3">
        <div>
          <div className="text-[10px] text-gray-500 uppercase">Equity</div>
          <div className="text-xl font-bold text-white">{formatCurrency(s.equity)}</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] text-gray-500 uppercase">Return</div>
          <div className={`text-xl font-bold ${s.return_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
            {formatPct(s.return_pct)}
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-2 text-center border-t border-white/[0.06] pt-3">
        <div>
          <div className="text-xs font-semibold text-white">{s.open_positions}</div>
          <div className="text-[9px] text-gray-500">Open</div>
        </div>
        <div>
          <div className="text-xs font-semibold text-white">{s.total_trades}</div>
          <div className="text-[9px] text-gray-500">Trades</div>
        </div>
        <div>
          <div className="text-xs font-semibold text-white">
            {s.win_rate > 0 ? `${s.win_rate.toFixed(0)}%` : "--"}
          </div>
          <div className="text-[9px] text-gray-500">Win Rate</div>
        </div>
        <div>
          <div className={`text-xs font-semibold ${s.total_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
            {formatCurrency(s.total_pnl)}
          </div>
          <div className="text-[9px] text-gray-500">P&L</div>
        </div>
      </div>

      {/* Signals */}
      <div className="flex items-center gap-3 mt-2 text-[10px] text-gray-500">
        <span>Signals: {s.signals_accepted}/{s.signals_evaluated} accepted</span>
        {s.max_drawdown > 0 && <span>· DD: -{s.max_drawdown.toFixed(1)}%</span>}
      </div>
    </div>
  );
}

/* ─── Comparison Row ─── */
function ComparisonRow({ label, llm, nollm, format = "number", invert = false }: {
  label: string;
  llm: number;
  nollm: number;
  format?: "number" | "pct" | "currency";
  invert?: boolean;
}) {
  const better = invert ? (llm < nollm ? "llm" : "nollm") : (llm > nollm ? "llm" : "nollm");
  const fmt = (v: number) => {
    if (format === "pct") return v > 0 ? `${v.toFixed(1)}%` : "--";
    if (format === "currency") return formatCurrency(v);
    return v.toString();
  };

  return (
    <div className="flex items-center justify-between py-2 border-b border-white/[0.04] last:border-0">
      <span className="text-xs text-gray-400 w-24">{label}</span>
      <div className="flex items-center gap-4">
        <span className={`text-xs font-medium ${better === "llm" && llm !== nollm ? "text-purple-400" : "text-gray-300"}`}>
          {fmt(llm)} {better === "llm" && llm !== nollm ? "✦" : ""}
        </span>
        <span className="text-[10px] text-gray-600">vs</span>
        <span className={`text-xs font-medium ${better === "nollm" && llm !== nollm ? "text-blue-400" : "text-gray-300"}`}>
          {fmt(nollm)} {better === "nollm" && llm !== nollm ? "✦" : ""}
        </span>
      </div>
    </div>
  );
}

/* ─── Main Page ─── */
export default function Strategies() {
  const navigate = useNavigate();
  const { data, isLoading } = useStrategyComparison();
  const { agent: activeAgent, setAgent: setActiveAgent } = useAgent();

  if (isLoading) return <LoadingSkeleton rows={6} />;
  const strategies = data?.strategies ?? [];

  if (strategies.length === 0) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-2 text-sm text-gray-500">
        <Target size={32} className="text-gray-600" />
        <p>No strategy data yet.</p>
        <p className="text-[10px]">Agents will populate once market opens.</p>
      </div>
    );
  }

  const llmStrats = strategies.filter(s => isLLM(s));
  const noLlmStrats = strategies.filter(s => !isLLM(s));
  const currentStrats = activeAgent === "llm" ? llmStrats : noLlmStrats;
  const otherStrats = activeAgent === "llm" ? noLlmStrats : llmStrats;

  // Head-to-head comparison data (aggressive vs aggressive, moderate vs moderate)
  const llmAgg = llmStrats.find(s => isAggressive(s));
  const llmMod = llmStrats.find(s => !isAggressive(s));
  const nollmAgg = noLlmStrats.find(s => isAggressive(s));
  const nollmMod = noLlmStrats.find(s => !isAggressive(s));

  return (
    <div className="space-y-4 pb-20">
      {/* Title */}
      <div className="fade-up">
        <h2 className="text-lg font-bold text-white">Strategies</h2>
        <p className="text-[11px] text-gray-500 mt-0.5">
          Compare LLM vs No-LLM · Aggressive vs Moderate
        </p>
      </div>

      {/* Agent Toggle */}
      <div className="fade-up fade-up-1">
        <AgentToggle active={activeAgent} onChange={setActiveAgent} />
      </div>

      {/* Agent Summary */}
      <div className="fade-up fade-up-1">
        <AgentSummary strategies={currentStrats} agent={activeAgent} />
      </div>

      {/* Strategy Cards */}
      <div className="space-y-3 fade-up fade-up-2">
        {[...currentStrats]
          .sort((a, b) => (isAggressive(a) ? -1 : 1))
          .map(s => (
            <StrategyCard
              key={s.id}
              s={s}
              onClick={() => navigate(`/strategies/${s.id}`)}
            />
          ))}
      </div>

      {/* Head-to-Head Comparison */}
      {llmStrats.length > 0 && noLlmStrats.length > 0 && (
        <Card className="fade-up fade-up-3">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Trophy size={14} className="text-yellow-400" />
              Head-to-Head: 🧠 LLM vs 📊 No-LLM
            </CardTitle>
          </CardHeader>

          {/* Aggressive comparison */}
          {llmAgg && nollmAgg && (
            <div className="mb-4">
              <div className="flex items-center gap-2 mb-2">
                <Zap size={12} className="text-red-400" />
                <span className="text-xs font-semibold text-gray-300">Aggressive</span>
              </div>
              <div className="rounded-xl bg-white/[0.02] p-3">
                <ComparisonRow label="Return" llm={llmAgg.return_pct} nollm={nollmAgg.return_pct} format="pct" />
                <ComparisonRow label="Win Rate" llm={llmAgg.win_rate} nollm={nollmAgg.win_rate} format="pct" />
                <ComparisonRow label="Trades" llm={llmAgg.total_trades} nollm={nollmAgg.total_trades} />
                <ComparisonRow label="P&L" llm={llmAgg.total_pnl} nollm={nollmAgg.total_pnl} format="currency" />
                <ComparisonRow label="Max DD" llm={llmAgg.max_drawdown} nollm={nollmAgg.max_drawdown} format="pct" invert />
              </div>
            </div>
          )}

          {/* Moderate comparison */}
          {llmMod && nollmMod && (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <ShieldCheck size={12} className="text-green-400" />
                <span className="text-xs font-semibold text-gray-300">Moderate</span>
              </div>
              <div className="rounded-xl bg-white/[0.02] p-3">
                <ComparisonRow label="Return" llm={llmMod.return_pct} nollm={nollmMod.return_pct} format="pct" />
                <ComparisonRow label="Win Rate" llm={llmMod.win_rate} nollm={nollmMod.win_rate} format="pct" />
                <ComparisonRow label="Trades" llm={llmMod.total_trades} nollm={nollmMod.total_trades} />
                <ComparisonRow label="P&L" llm={llmMod.total_pnl} nollm={nollmMod.total_pnl} format="currency" />
                <ComparisonRow label="Max DD" llm={llmMod.max_drawdown} nollm={nollmMod.max_drawdown} format="pct" invert />
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
