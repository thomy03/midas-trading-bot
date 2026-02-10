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
  BULL: '🐂 Bull',
  BEAR: '🐻 Bear',
  RANGE: '📊 Range',
  VOLATILE: '⚡ Volatile',
} as const

export const SIGNAL_COLORS = {
  BUY: '#22c55e',
  SELL: '#ef4444',
  HOLD: '#6b7280',
} as const

export type Regime = keyof typeof REGIME_COLORS
export type SignalType = keyof typeof SIGNAL_COLORS

export const REGIME_BG = {
  BULL: 'bg-green-100 text-green-800',
  BEAR: 'bg-red-100 text-red-800',
  RANGE: 'bg-amber-100 text-amber-800',
  VOLATILE: 'bg-violet-100 text-violet-800',
} as const
