import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";
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
  return useQuery({
    queryKey: ["portfolio", "summary"],
    queryFn: () => apiFetch<PortfolioSummary>("/api/v1/portfolio/summary"),
  });
}

export function usePortfolioPositions() {
  return useQuery({
    queryKey: ["portfolio", "positions"],
    queryFn: () => apiFetch<PositionsResponse>("/api/v1/portfolio/positions"),
  });
}

export function useAgentStatus() {
  return useQuery({
    queryKey: ["agent", "status"],
    queryFn: () => apiFetch<AgentStatus>("/api/v1/agent/status"),
  });
}

export function usePortfolioHistory() {
  return useQuery({
    queryKey: ["portfolio", "history"],
    queryFn: () => apiFetch<PortfolioHistory>("/api/v1/portfolio/history"),
    refetchInterval: 60_000,
  });
}

export function useTrades(status?: "open" | "closed") {
  const params = status ? `?status=${status}` : "";
  return useQuery({
    queryKey: ["trades", status ?? "all"],
    queryFn: () => apiFetch<Trade[]>(`/api/v1/trades${params}`),
  });
}

export function usePerformance() {
  return useQuery({
    queryKey: ["trades", "performance"],
    queryFn: () => apiFetch<PerformanceStats>("/api/v1/trades/performance"),
  });
}

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

export function useAlerts(days = 7) {
  return useQuery({
    queryKey: ["alerts", days],
    queryFn: () => apiFetch<AlertItem[]>(`/api/v1/alerts?days=${days}`),
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
