import { useQuery } from "@tanstack/react-query";
import { useParams, useNavigate } from "react-router-dom";
import { apiFetch } from "@/api/client";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { formatPct, formatCurrency } from "@/lib/utils";

interface Position {
  symbol: string;
  shares: number;
  entry_price: number;
  current_price: number;
  entry_date: string;
  stop_loss: number;
  take_profit: number;
  pnl_pct: number;
  pnl_amount: number;
  score_at_entry: number;
  reasoning: string;
  company_name?: string;
  sector?: string;
  pillar_technical?: number;
  pillar_fundamental?: number;
}

interface ClosedTrade {
  symbol: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
  pnl_pct: number;
  entry_date: string;
  exit_date: string;
  reason: string;
  hold_days?: number;
}

interface StrategyDetail {
  id: string;
  strategy_id: string;
  agent: string;
  name: string;
  color: string;
  equity: number;
  cash: number;
  initial_capital: number;
  return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number;
  max_drawdown: number;
  signals_evaluated: number;
  signals_accepted: number;
  signals_rejected: number;
  positions: Position[];
  closed_trades: ClosedTrade[];
  equity_curve: { date: string; equity: number }[];
  profile: {
    description?: string;
    min_score?: number;
    max_positions?: number;
    position_size_pct?: number;
    use_ml_gate?: boolean;
    pillar_weights?: Record<string, number>;
  };
  error?: string;
}

export default function StrategyDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { data, isLoading } = useQuery({
    queryKey: ["strategy", id],
    queryFn: () => apiFetch<StrategyDetail>(`/api/v1/strategies/${id}`),
    refetchInterval: 30_000,
    enabled: !!id,
  });

  if (isLoading) return <LoadingSkeleton rows={8} />;
  if (!data || data.error) {
    return (
      <div className="p-4 text-center text-red-400">
        {data?.error || "Strategy not found"}
      </div>
    );
  }

  const s = data;
  const winRate = s.total_trades > 0 ? ((s.winning_trades / s.total_trades) * 100).toFixed(1) : "--";
  const acceptRate = s.signals_evaluated > 0 ? ((s.signals_accepted / s.signals_evaluated) * 100).toFixed(1) : "--";

  return (
    <div className="space-y-3 pb-20">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate("/strategies")}
          className="rounded-lg bg-gray-800 px-3 py-1.5 text-sm text-gray-300 hover:bg-gray-700"
        >
          ← Back
        </button>
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-4 w-4 rounded-full"
            style={{ backgroundColor: s.color }}
          />
          <h1 className="text-lg font-bold text-white">{s.name}</h1>
        </div>
        <span className="rounded bg-gray-800 px-2 py-0.5 text-xs text-gray-400">
          {s.agent === "llm" ? "🤖 LLM" : "📊 No-LLM"}
        </span>
      </div>

      {/* Profile Info */}
      {s.profile?.description && (
        <Card>
          <div className="text-xs text-gray-400 space-y-1">
            <p>{s.profile.description}</p>
            <div className="flex flex-wrap gap-3 mt-2">
              <span>Min Score: <b className="text-white">{s.profile.min_score}</b></span>
              <span>Max Pos: <b className="text-white">{s.profile.max_positions}</b></span>
              <span>Size: <b className="text-white">{((s.profile.position_size_pct || 0) <= 1 ? ((s.profile.position_size_pct || 0) * 100).toFixed(0) : (s.profile.position_size_pct || 0).toFixed(0))}%</b></span>
              <span>ML Gate: <b className="text-white">{s.profile.use_ml_gate ? "✅" : "❌"}</b></span>
            </div>
            {s.profile.pillar_weights && (
              <div className="flex flex-wrap gap-2 mt-1">
                {Object.entries(s.profile.pillar_weights).map(([k, v]) => (
                  <span key={k} className="rounded bg-gray-800 px-2 py-0.5">
                    {k}: <b className="text-white">{((v as number) * 100).toFixed(0)}%</b>
                  </span>
                ))}
              </div>
            )}
          </div>
        </Card>
      )}

      {/* KPI Row */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: "Equity", value: formatCurrency(s.equity), color: "text-white" },
          { label: "Return", value: formatPct(s.return_pct), color: s.return_pct >= 0 ? "text-green-400" : "text-red-400" },
          { label: "Win Rate", value: winRate + "%", color: "text-white" },
          { label: "Cash", value: formatCurrency(s.cash), color: "text-gray-300" },
        ].map((kpi) => (
          <Card key={kpi.label}>
            <div className="text-center">
              <div className="text-[10px] text-gray-500 uppercase">{kpi.label}</div>
              <div className={`text-sm font-bold ${kpi.color}`}>{kpi.value}</div>
            </div>
          </Card>
        ))}
      </div>

      {/* Stats */}
      <Card>
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div>
            <span className="text-gray-500">Trades</span>
            <div className="font-bold text-white">{s.total_trades}</div>
          </div>
          <div>
            <span className="text-gray-500">Wins / Losses</span>
            <div className="font-bold">
              <span className="text-green-400">{s.winning_trades}</span>
              {" / "}
              <span className="text-red-400">{s.losing_trades}</span>
            </div>
          </div>
          <div>
            <span className="text-gray-500">P&L</span>
            <div className={`font-bold ${s.total_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
              {formatCurrency(s.total_pnl)}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Max DD</span>
            <div className="font-bold text-red-400">
              {s.max_drawdown > 0 ? `-${s.max_drawdown.toFixed(1)}%` : "--"}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Signals</span>
            <div className="font-bold text-white">
              {s.signals_accepted}/{s.signals_evaluated}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Accept Rate</span>
            <div className="font-bold text-white">{acceptRate}%</div>
          </div>
        </div>
      </Card>

      {/* Equity Curve */}
      {s.equity_curve.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">📈 Equity Curve</CardTitle>
          </CardHeader>
          <EquityCurve data={s.equity_curve} color={s.color} />
        </Card>
      )}

      {/* Open Positions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            📂 Open Positions ({s.positions.length})
          </CardTitle>
        </CardHeader>
        {s.positions.length === 0 ? (
          <div className="py-4 text-center text-xs text-gray-500">No open positions</div>
        ) : (
          <div className="space-y-2">
            {s.positions.map((p, i) => (
              <PositionCard key={i} position={p} />
            ))}
          </div>
        )}
      </Card>

      {/* Closed Trades */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            📋 Closed Trades ({s.closed_trades.length})
          </CardTitle>
        </CardHeader>
        {s.closed_trades.length === 0 ? (
          <div className="py-4 text-center text-xs text-gray-500">No closed trades yet</div>
        ) : (
          <div className="space-y-1">
            {[...s.closed_trades].reverse().map((t, i) => (
              <div
                key={i}
                className="flex items-center justify-between border-b border-gray-800 py-2 text-xs"
              >
                <div>
                  <div className="font-semibold text-white">{t.symbol}</div>
                  <div className="text-gray-500">
                    {t.entry_date?.slice(0, 10)} → {t.exit_date?.slice(0, 10)}
                    {t.hold_days ? ` (${t.hold_days}d)` : ""}
                  </div>
                </div>
                <div className="text-center text-gray-500">
                  <div>${(t.entry_price ?? 0).toFixed(2)} → ${(t.exit_price ?? 0).toFixed(2)}</div>
                  <div>{t.reason || ""}</div>
                </div>
                <div className="text-right">
                  <div className={`font-bold ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {formatCurrency(t.pnl ?? 0)}
                  </div>
                  <div className={(t.pnl_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}>
                    {formatPct(t.pnl_pct ?? 0)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function PositionCard({ position: p }: { position: Position }) {
  const [expanded, setExpanded] = React.useState(false);

  return (
    <div
      className="rounded-lg border border-gray-800 p-3 cursor-pointer hover:border-gray-600 transition-colors"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-white text-sm">{p.symbol}</span>
            {p.company_name && (
              <span className="text-xs text-gray-500">{p.company_name}</span>
            )}
          </div>
          <div className="text-xs text-gray-500">
            x{p.shares} @ ${(p.entry_price ?? 0).toFixed(2)}
            {p.sector && <span className="ml-2">• {p.sector}</span>}
          </div>
        </div>
        <div className="text-right">
          <div className={`text-sm font-bold ${(p.pnl_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
            {formatPct(p.pnl_pct ?? 0)}
          </div>
          <div className={`text-xs ${(p.pnl_amount ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
            {formatCurrency(p.pnl_amount ?? 0)}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="mt-3 space-y-2 border-t border-gray-800 pt-3">
          {/* Entry details */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Entry</span>
              <div className="text-white">${(p.entry_price ?? 0).toFixed(2)}</div>
              <div className="text-gray-600">{p.entry_date?.slice(0, 10)}</div>
            </div>
            <div>
              <span className="text-gray-500">Stop Loss</span>
              <div className="text-red-400">${(p.stop_loss ?? 0).toFixed(2)}</div>
            </div>
            <div>
              <span className="text-gray-500">Take Profit</span>
              <div className="text-green-400">${(p.take_profit ?? 0).toFixed(2)}</div>
            </div>
          </div>

          {/* Score & Pillars */}
          <div className="text-xs">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-gray-500">Score:</span>
              <span className="font-bold text-yellow-400">{(p.score_at_entry ?? 0).toFixed(1)}</span>
            </div>
            {(p.pillar_technical !== undefined || p.pillar_fundamental !== undefined) && (
              <div className="flex gap-3">
                {p.pillar_technical !== undefined && (
                  <span>Tech: <b className="text-blue-400">{p.pillar_technical.toFixed(1)}</b></span>
                )}
                {p.pillar_fundamental !== undefined && (
                  <span>Funda: <b className="text-purple-400">{p.pillar_fundamental.toFixed(1)}</b></span>
                )}
              </div>
            )}
          </div>

          {/* Reasoning */}
          {p.reasoning && (
            <div className="text-xs">
              <div className="text-gray-500 mb-1">Reasoning:</div>
              <pre className="whitespace-pre-wrap rounded bg-gray-900 p-2 text-gray-300 text-[10px] leading-relaxed max-h-48 overflow-y-auto">
                {p.reasoning}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EquityCurve({ data, color }: { data: { date: string; equity: number }[]; color: string }) {
  if (data.length < 2) return null;
  
  const w = 700;
  const h = 150;
  const equities = data.map((d) => d.equity);
  const minE = Math.min(...equities) * 0.995;
  const maxE = Math.max(...equities) * 1.005;

  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((d.equity - minE) / (maxE - minE)) * h;
    return `${x},${y}`;
  });

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-36">
      <rect width={w} height={h} fill="#111" rx="8" />
      <polyline points={points.join(" ")} fill="none" stroke={color} strokeWidth="2" />
    </svg>
  );
}

import React from "react";
