export const PILLAR_COLORS = {
  technical: '#3b82f6',    // blue
  fundamental: '#10b981',  // green
  sentiment: '#f59e0b',    // amber
  news: '#8b5cf6',         // violet
  ml: '#ec4899',           // pink
} as const

export const REGIME_COLORS = {
  BULL: '#22c55e',     // green
  BEAR: '#ef4444',     // red
  RANGE: '#f59e0b',    // amber
  VOLATILE: '#8b5cf6', // violet
} as const

export const REGIME_LABELS = {
  BULL: 'Bull',
  BEAR: 'Bear',
  RANGE: 'Range',
  VOLATILE: 'Volatile',
} as const

export const SIGNAL_COLORS = {
  BUY: '#22c55e',
  SELL: '#ef4444',
  HOLD: '#6b7280',
} as const

export type Regime = keyof typeof REGIME_COLORS
export type SignalType = keyof typeof SIGNAL_COLORS

export const REGIME_BG: Record<string, string> = {
  BULL: 'bg-green-500/20 text-green-400 border-green-500/30',
  BEAR: 'bg-red-500/20 text-red-400 border-red-500/30',
  RANGE: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  VOLATILE: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  UNKNOWN: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
}

export const SECTOR_ETFS: Record<string, string> = {
  Technology: "XLK",
  Healthcare: "XLV",
  Financials: "XLF",
  "Consumer Disc.": "XLY",
  Communication: "XLC",
  Industrials: "XLI",
  "Consumer Staples": "XLP",
  Energy: "XLE",
  Utilities: "XLU",
  "Real Estate": "XLRE",
  Materials: "XLB",
}
