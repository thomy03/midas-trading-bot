import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { formatPct } from "@/lib/utils";

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
}

interface ComparisonResponse {
  strategies: StrategyData[];
  updated_at: string;
  error?: string;
}

function useStrategyComparison() {
  return useQuery({
    queryKey: ["strategies", "comparison"],
    queryFn: () => apiFetch<ComparisonResponse>("/api/v1/strategies/comparison"),
    refetchInterval: 30_000,
  });
}

function EquityMiniChart({ strategies }: { strategies: StrategyData[] }) {
  if (!strategies.some((s) => s.equity_curve.length > 1)) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-gray-500">
        Equity curves will appear once the agent starts trading...
      </div>
    );
  }

  // Simple SVG line chart
  const maxLen = Math.max(...strategies.map((s) => s.equity_curve.length));
  const allEquities = strategies.flatMap((s) => s.equity_curve.map((e) => e.equity));
  const minE = Math.min(...allEquities) * 0.995;
  const maxE = Math.max(...allEquities) * 1.005;
  const w = 700;
  const h = 200;

  return (
    <svg viewBox={"0 0 " + w + " " + h} className="w-full h-48">
      <rect width={w} height={h} fill="#111" rx="8" />
      {/* Grid */}
      {[0.25, 0.5, 0.75].map((pct) => (
        <line
          key={pct}
          x1={0}
          y1={h * pct}
          x2={w}
          y2={h * pct}
          stroke="#333"
          strokeDasharray="4"
        />
      ))}
      {/* Starting capital line */}
      {(() => {
        const y = h - ((15000 - minE) / (maxE - minE)) * h;
        return (
          <line x1={0} y1={y} x2={w} y2={y} stroke="#555" strokeDasharray="2" />
        );
      })()}
      {/* Lines for each strategy */}
      {strategies.map((s) => {
        if (s.equity_curve.length < 2) return null;
        const points = s.equity_curve.map((e, i) => {
          const x = (i / (maxLen - 1)) * w;
          const y = h - ((e.equity - minE) / (maxE - minE)) * h;
          return x + "," + y;
        });
        return (
          <polyline
            key={s.id}
            points={points.join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth="2"
          />
        );
      })}
    </svg>
  );
}

export default function Strategies() {
  const { data, isLoading } = useStrategyComparison();

  if (isLoading) return <LoadingSkeleton rows={6} />;

  const strategies = data?.strategies ?? [];

  if (strategies.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-gray-500">
        No strategy data yet. Start the agent to begin tracking.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Equity Curves */}
      <Card>
        <CardHeader>
          <CardTitle>📈 Strategy Comparison</CardTitle>
          <div className="flex flex-wrap gap-3 mt-1">
            {strategies.map((s) => (
              <span key={s.id} className="flex items-center gap-1 text-xs">
                <span
                  className="inline-block h-3 w-3 rounded-full"
                  style={{ backgroundColor: s.color }}
                />
                {s.name}
              </span>
            ))}
          </div>
        </CardHeader>
        <EquityMiniChart strategies={strategies} />
      </Card>

      {/* Rankings */}
      <Card>
        <CardHeader>
          <CardTitle>🏆 Ranking</CardTitle>
        </CardHeader>
        <div className="space-y-2">
          {strategies.map((s, idx) => (
            <div
              key={s.id}
              className="flex items-center justify-between rounded-lg border border-gray-800 p-3"
            >
              <div className="flex items-center gap-3">
                <span className="text-lg font-bold text-gray-500">
                  #{idx + 1}
                </span>
                <span
                  className="inline-block h-4 w-4 rounded-full"
                  style={{ backgroundColor: s.color }}
                />
                <div>
                  <div className="text-sm font-semibold text-white">{s.name}</div>
                  <div className="text-xs text-gray-500">
                    {s.open_positions} positions | {s.total_trades} trades
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div
                  className={
                    "text-lg font-bold " +
                    (s.return_pct >= 0 ? "text-green-400" : "text-red-400")
                  }
                >
                  {formatPct(s.return_pct)}
                </div>
                <div className="text-xs text-gray-500">
                  ${s.equity.toLocaleString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-3">
        {strategies.map((s) => (
          <Card key={s.id}>
            <div className="flex items-center gap-2 mb-2">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              <span className="text-xs font-semibold text-white">{s.name}</span>
            </div>
            <div className="grid grid-cols-2 gap-1 text-xs">
              <div>
                <span className="text-gray-500">Win Rate</span>
                <div className="font-bold text-white">{s.win_rate}%</div>
              </div>
              <div>
                <span className="text-gray-500">Max DD</span>
                <div className="font-bold text-red-400">
                  {s.max_drawdown > 0 ? "-" + s.max_drawdown.toFixed(1) + "%" : "--"}
                </div>
              </div>
              <div>
                <span className="text-gray-500">P&L</span>
                <div
                  className={
                    "font-bold " +
                    (s.total_pnl >= 0 ? "text-green-400" : "text-red-400")
                  }
                >
                  ${s.total_pnl.toFixed(0)}
                </div>
              </div>
              <div>
                <span className="text-gray-500">Signals</span>
                <div className="font-bold text-white">
                  {s.signals_accepted}/{s.signals_evaluated}
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Open Positions by Strategy */}
      {strategies.filter((s) => s.positions.length > 0).map((s) => (
        <Card key={s.id + "-pos"}>
          <CardHeader>
            <div className="flex items-center gap-2">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              <CardTitle className="text-sm">{s.name} — Positions</CardTitle>
            </div>
          </CardHeader>
          <div className="space-y-1 text-xs">
            {s.positions.map((p: any, i: number) => (
              <div
                key={i}
                className="flex justify-between border-b border-gray-800 py-1"
              >
                <span className="font-semibold text-white">{p.symbol}</span>
                <span className="text-gray-500">x{p.shares} @${(p.entry_price ?? 0).toFixed(2)}</span>
                <span
                  className={
                    (p.pnl_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"
                  }
                >
                  {formatPct(p.pnl_pct ?? 0)}
                </span>
              </div>
            ))}
          </div>
        </Card>
      ))}

      {/* Recent Trades */}
      {strategies.filter((s) => s.recent_trades.length > 0).map((s) => (
        <Card key={s.id + "-trades"}>
          <CardHeader>
            <div className="flex items-center gap-2">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              <CardTitle className="text-sm">{s.name} — Recent Trades</CardTitle>
            </div>
          </CardHeader>
          <div className="space-y-1 text-xs">
            {s.recent_trades.slice(-10).reverse().map((t: any, i: number) => (
              <div
                key={i}
                className="flex justify-between border-b border-gray-800 py-1"
              >
                <span className="font-semibold text-white">{t.symbol}</span>
                <span className="text-gray-500">{t.reason}</span>
                <span
                  className={
                    (t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"
                  }
                >
                  ${(t.pnl ?? 0).toFixed(0)} ({formatPct(t.pnl_pct ?? 0)})
                </span>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}
