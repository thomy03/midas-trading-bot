import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";
import { useAgent } from "@/contexts/AgentContext";
import type {
  PortfolioSummary,
  PositionsResponse,
  AgentStatus,
  PortfolioHistory,
  Trade,
  PerformanceStats,
  SectorPerformance,
  MarketBreadth,
  EconomicEvent,
  AlertItem,
  HealthResponse,
  LearningWeights,
} from "./types";

export function usePortfolioSummary() {
  const { agent } = useAgent();
  return useQuery({
    queryKey: ["portfolio", "summary", agent],
    queryFn: () => apiFetch<PortfolioSummary>(`/api/v1/portfolio/summary?agent=${agent}`),
  });
}

export function usePortfolioPositions() {
  const { agent } = useAgent();
  return useQuery({
    queryKey: ["portfolio", "positions", agent],
    queryFn: () => apiFetch<PositionsResponse>(`/api/v1/portfolio/positions?agent=${agent}`),
  });
}

export function useAgentStatus() {
  const { agent } = useAgent();
  return useQuery({
    queryKey: ["agent", "status", agent],
    queryFn: () => apiFetch<AgentStatus>(`/api/v1/agent/status?agent=${agent}`),
  });
}

export function usePortfolioHistory() {
  const { agent } = useAgent();
  return useQuery({
    queryKey: ["portfolio", "history", agent],
    queryFn: () => apiFetch<PortfolioHistory>(`/api/v1/portfolio/history?agent=${agent}`),
    refetchInterval: 60_000,
  });
}

export function useTrades(status?: "open" | "closed") {
  const { agent } = useAgent();
  const params = new URLSearchParams({ agent });
  if (status) params.set("status", status);
  return useQuery({
    queryKey: ["trades", status ?? "all", agent],
    queryFn: () => apiFetch<Trade[]>(`/api/v1/trades?${params.toString()}`),
  });
}

export function usePerformance() {
  const { agent } = useAgent();
  return useQuery({
    queryKey: ["trades", "performance", agent],
    queryFn: () => apiFetch<PerformanceStats>(`/api/v1/trades/performance?agent=${agent}`),
  });
}

export function useAlerts(days = 7) {
  const { agent } = useAgent();
  return useQuery({
    queryKey: ["alerts", days, agent],
    queryFn: () => apiFetch<AlertItem[]>(`/api/v1/alerts?days=${days}&agent=${agent}`),
  });
}

// Global hooks (no agent param)
export function useSectors() {
  return useQuery({
    queryKey: ["sectors"],
    queryFn: () => apiFetch<SectorPerformance[]>("/api/v1/sectors"),
    refetchInterval: 60_000,
  });
}

export function useMarketBreadth() {
  return useQuery({
    queryKey: ["market", "breadth"],
    queryFn: () => apiFetch<MarketBreadth>("/api/v1/market/breadth"),
    refetchInterval: 60_000,
  });
}

export function useEconomicEvents(daysAhead = 30) {
  return useQuery({
    queryKey: ["calendar", "events", daysAhead],
    queryFn: () =>
      apiFetch<EconomicEvent[]>(
        `/api/v1/calendar/events?days_ahead=${daysAhead}`
      ),
    refetchInterval: 300_000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiFetch<HealthResponse>("/health"),
    refetchInterval: 60_000,
  });
}

export function useLearningWeights() {
  return useQuery({
    queryKey: ["learning", "weights"],
    queryFn: () => apiFetch<LearningWeights>("/api/v1/learning/weights"),
    refetchInterval: 120_000,
  });
}
