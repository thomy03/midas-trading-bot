import { useTrades } from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PnlBadge } from "@/components/shared/PnlBadge";
import { PillarBar } from "@/components/shared/PillarBar";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";
import { useState } from "react";

export default function Journal() {
  const { data: trades, isLoading } = useTrades("closed");
  const [expanded, setExpanded] = useState<string | null>(null);

  if (isLoading) return <LoadingSkeleton rows={6} />;

  const tradeList = trades ?? [];

  return (
    <div className="space-y-3">
      <Card>
        <CardHeader>
          <CardTitle>Trade Journal</CardTitle>
          <span className="text-xs text-gray-500">
            {tradeList.length} closed trades
          </span>
        </CardHeader>
      </Card>

      {tradeList.length === 0 ? (
        <Card>
          <div className="py-8 text-center text-sm text-gray-500">
            No closed trades yet
          </div>
        </Card>
      ) : (
        <div className="space-y-2">
          {tradeList.map((t) => (
            <Card
              key={t.trade_id}
              className="cursor-pointer active:bg-surface-2"
              onClick={() =>
                setExpanded(expanded === t.trade_id ? null : t.trade_id)
              }
            >
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-medium text-white">{t.symbol}</span>
                  <span className="ml-2 text-[10px] text-gray-500">
                    {t.entry_date
                      ? new Date(t.entry_date).toLocaleDateString()
                      : ""}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <PnlBadge
                    value={t.pnl_pct ?? 0}
                    mode="pct"
                    size="sm"
                  />
                  <Badge
                    variant={(t.pnl ?? 0) >= 0 ? "success" : "danger"}
                  >
                    ${(t.pnl ?? 0).toFixed(0)}
                  </Badge>
                </div>
              </div>

              {expanded === t.trade_id && (
                <div className="mt-3 space-y-2 border-t border-border pt-3">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">Entry:</span>{" "}
                      <span className="text-white">
                        ${(t.entry_price ?? 0).toFixed(2)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Shares:</span>{" "}
                      <span className="text-white">{t.shares}</span>
                    </div>
                    {t.score_at_entry && (
                      <div>
                        <span className="text-gray-500">Score:</span>{" "}
                        <span className="text-white">
                          {(t.score_at_entry ?? 0).toFixed(0)}
                        </span>
                      </div>
                    )}
                    {t.sector && (
                      <div>
                        <span className="text-gray-500">Sector:</span>{" "}
                        <span className="text-white">{t.sector}</span>
                      </div>
                    )}
                  </div>
                  {(t.pillar_technical || t.pillar_fundamental) && (
                    <PillarBar
                      technical={t.pillar_technical}
                      fundamental={t.pillar_fundamental}
                      sentiment={t.pillar_sentiment}
                      news={t.pillar_news}
                    />
                  )}
                  {t.reasoning && (
                    <p className="text-[11px] text-gray-400">{t.reasoning}</p>
                  )}
                </div>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
