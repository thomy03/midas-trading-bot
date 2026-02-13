export interface StrategyBreakdown {
  equity: number;
  cash: number;
  invested: number;
  positions: number;
  pnl: number;
  return_pct: number;
}

export interface PortfolioSummary {
  total_capital: number;
  available_capital: number;
  invested_capital: number;
  open_positions: number;
  unrealized_pnl: number;
  strategies?: { [key: string]: StrategyBreakdown };
}

export interface Position {
  symbol: string;
  entry_price: number;
  shares: number;
  position_value: number;
  allocation_pct: number;
  pnl_amount?: number;
  pnl_percent?: number;
  stop_loss?: number;
  take_profit?: number;
  score_at_entry?: number;
  pillar_technical?: number;
  pillar_fundamental?: number;
  pillar_sentiment?: number;
  pillar_news?: number;
  reasoning?: string;
  company_name?: string;
  sector?: string;
  industry?: string;
  entry_date?: string;
  strategy_id?: string;
}

export interface PositionsResponse {
  positions: Position[];
  cash: number;
  total_value: number;
  total_capital: number;
  positions_count: number;
}

export interface AgentStatus {
  running: boolean;
  phase: string;
  market_regime: string;
  llm_enabled?: boolean;
  metrics: {
    started_at?: string;
    cycles?: number;
    screens?: number;
    signals?: number;
    trades?: number;
    errors?: number;
    avg_cycle_ms?: number;
  };
  hot_symbols: string[];
  intelligence_brief?: IntelligenceBrief | null;
}

export interface IntelligenceBrief {
  reasoning_summary: string;
  portfolio_alerts: string[];
  market_events: string[];
  megatrends: string[];
  timestamp: string;
}

export interface PortfolioHistory {
  start_date: string;
  end_date: string;
  days_tracked: number;
  start_value: number;
  end_value: number;
  total_return_pct: number;
  spy_return_pct: number;
  alpha_pct: number;
  daily_history: Record<
    string,
    {
      open: number;
      high: number;
      low: number;
      close: number;
      spy_open: number;
      spy_close: number;
    }
  >;
}

export interface Trade {
  trade_id: string;
  symbol: string;
  entry_date: string;
  entry_price: number;
  shares: number;
  status: string;
  pnl?: number;
  pnl_pct?: number;
  stop_loss?: number;
  take_profit?: number;
  score_at_entry?: number;
  pillar_technical?: number;
  pillar_fundamental?: number;
  pillar_sentiment?: number;
  pillar_news?: number;
  reasoning?: string;
  position_value?: number;
  company_name?: string;
  sector?: string;
  industry?: string;
}

export interface PerformanceStats {
  total_trades: number;
  open_trades: number;
  closed_trades: number;
  total_pnl: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  profit_factor?: number;
  sharpe_ratio?: number;
  max_drawdown: number;
}

export interface SectorPerformance {
  name: string;
  etf: string;
  perf_1d: number;
  perf_1w: number;
  perf_1m: number;
  perf_ytd: number;
}

export interface MarketBreadth {
  advancing: number;
  declining: number;
  total: number;
  breadth_ratio: number;
  avg_performance: number;
}

export interface EconomicEvent {
  event_id: string;
  event_type: string;
  title: string;
  date: string;
  impact: string;
  symbol?: string;
}

export interface AlertItem {
  id: number;
  symbol: string;
  timeframe: string;
  current_price: number;
  support_level: number;
  recommendation: string;
  confidence_score?: number;
  created_at: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

export interface LearningWeights {
  status: string;
  weights: Record<
    string,
    {
      weight: number;
      top_indicators: { name: string; weight: number; accuracy: number }[];
    }
  >;
}
