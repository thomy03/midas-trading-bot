import {
  useSectors,
  useMarketBreadth,
  useEconomicEvents,
} from "@/api/hooks";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SectorHeatmap } from "@/components/charts/SectorHeatmap";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { LoadingSkeleton } from "@/components/shared/LoadingSkeleton";

export default function Market() {
  const { data: sectors, isLoading: sLoading } = useSectors();
  const { data: breadth } = useMarketBreadth();
  const { data: events } = useEconomicEvents(14);

  if (sLoading) return <LoadingSkeleton rows={4} />;

  return (
    <div className="space-y-3">
      {/* Market Breadth */}
      {breadth && (
        <Card>
          <CardHeader>
            <CardTitle>Market Breadth</CardTitle>
            <span className="text-xs text-gray-500">
              {breadth.total} stocks
            </span>
          </CardHeader>
          <div className="flex items-center gap-2">
            <div className="flex-1">
              <div className="flex h-3 overflow-hidden rounded-full bg-surface-3">
                <div
                  className="bg-green-500"
                  style={{
                    width: `${(breadth.advancing / breadth.total) * 100}%`,
                  }}
                />
                <div
                  className="bg-red-500"
                  style={{
                    width: `${(breadth.declining / breadth.total) * 100}%`,
                  }}
                />
              </div>
            </div>
            <div className="flex gap-3 text-xs">
              <span className="text-green-400">{breadth.advancing} adv</span>
              <span className="text-red-400">{breadth.declining} dec</span>
            </div>
          </div>
          <div className="mt-1 text-xs text-gray-500">
            Avg: {breadth.avg_performance > 0 ? "+" : ""}
            {(breadth.avg_performance ?? 0).toFixed(2)}% | Ratio:{" "}
            {(breadth.breadth_ratio ?? 0).toFixed(2)}
          </div>
        </Card>
      )}

      {/* Sector Heatmap */}
      {sectors && sectors.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Sector Performance</CardTitle>
          </CardHeader>
          <Tabs defaultValue="perf_1d">
            <TabsList className="mb-3">
              <TabsTrigger value="perf_1d">1D</TabsTrigger>
              <TabsTrigger value="perf_1w">1W</TabsTrigger>
              <TabsTrigger value="perf_1m">1M</TabsTrigger>
              <TabsTrigger value="perf_ytd">YTD</TabsTrigger>
            </TabsList>
            <TabsContent value="perf_1d">
              <SectorHeatmap sectors={sectors} metric="perf_1d" />
            </TabsContent>
            <TabsContent value="perf_1w">
              <SectorHeatmap sectors={sectors} metric="perf_1w" />
            </TabsContent>
            <TabsContent value="perf_1m">
              <SectorHeatmap sectors={sectors} metric="perf_1m" />
            </TabsContent>
            <TabsContent value="perf_ytd">
              <SectorHeatmap sectors={sectors} metric="perf_ytd" />
            </TabsContent>
          </Tabs>
        </Card>
      )}

      {/* Economic Calendar */}
      {events && events.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Economic Calendar</CardTitle>
            <span className="text-xs text-gray-500">Next 14 days</span>
          </CardHeader>
          <div className="space-y-1.5">
            {events.slice(0, 10).map((e) => (
              <div
                key={e.event_id}
                className="flex items-center justify-between rounded-lg bg-surface-2 px-3 py-2"
              >
                <div>
                  <div className="text-xs font-medium text-white">
                    {e.title}
                  </div>
                  <div className="text-[10px] text-gray-500">
                    {new Date(e.date).toLocaleDateString()}
                    {e.symbol && ` | ${e.symbol}`}
                  </div>
                </div>
                <Badge
                  variant={
                    e.impact === "high"
                      ? "danger"
                      : e.impact === "medium"
                      ? "warning"
                      : "default"
                  }
                >
                  {e.impact}
                </Badge>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
