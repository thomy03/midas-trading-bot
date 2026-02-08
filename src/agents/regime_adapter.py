"""
Regime-Adaptive Strategy Module
Adjusts stock universe and strategy based on market regime
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"
    VOLATILE = "volatile"


@dataclass
class RegimeConfig:
    """Configuration for each regime"""
    # Universe filters
    market_cap_min: int
    volume_min: int
    
    # Stock type preferences
    allow_small_caps: bool
    allow_growth_stocks: bool
    prefer_dividends: bool
    prefer_defensive: bool
    
    # Scoring adjustments
    momentum_weight: float
    fundamental_weight: float
    news_weight: float
    
    # Risk parameters
    position_size_max: float
    stop_loss_pct: float


# Regime-specific configurations
REGIME_CONFIGS = {
    MarketRegime.BULL: RegimeConfig(
        # Universe: WIDER - include small caps and growth
        market_cap_min=100_000_000,      # 00M (vs B in BEAR)
        volume_min=200_000,
        allow_small_caps=True,
        allow_growth_stocks=True,
        prefer_dividends=False,
        prefer_defensive=False,
        # Scoring: favor momentum and news
        momentum_weight=1.3,
        fundamental_weight=0.8,
        news_weight=1.2,
        # Risk: more aggressive
        position_size_max=0.10,
        stop_loss_pct=0.08,
    ),
    MarketRegime.BEAR: RegimeConfig(
        # Universe: NARROW - only blue chips
        market_cap_min=1_000_000_000,    # B minimum
        volume_min=500_000,
        allow_small_caps=False,
        allow_growth_stocks=False,
        prefer_dividends=True,
        prefer_defensive=True,
        # Scoring: favor fundamentals
        momentum_weight=0.7,
        fundamental_weight=1.4,
        news_weight=0.8,
        # Risk: conservative
        position_size_max=0.05,
        stop_loss_pct=0.05,
    ),
    MarketRegime.RANGE: RegimeConfig(
        # Universe: MODERATE
        market_cap_min=300_000_000,
        volume_min=300_000,
        allow_small_caps=False,
        allow_growth_stocks=True,
        prefer_dividends=False,
        prefer_defensive=False,
        # Scoring: balanced
        momentum_weight=1.0,
        fundamental_weight=1.0,
        news_weight=1.0,
        # Risk: moderate
        position_size_max=0.08,
        stop_loss_pct=0.06,
    ),
    MarketRegime.VOLATILE: RegimeConfig(
        # Universe: VERY NARROW - only mega caps
        market_cap_min=2_000_000_000,
        volume_min=1_000_000,
        allow_small_caps=False,
        allow_growth_stocks=False,
        prefer_dividends=True,
        prefer_defensive=True,
        # Scoring: favor stability
        momentum_weight=0.5,
        fundamental_weight=1.5,
        news_weight=0.6,
        # Risk: very conservative
        position_size_max=0.04,
        stop_loss_pct=0.04,
    ),
}

# Defensive sectors (preferred in BEAR/VOLATILE)
DEFENSIVE_SECTORS = [
    'Consumer Staples', 'Utilities', 'Healthcare', 
    'Consumer Defensive', 'Real Estate'
]

# Growth sectors (preferred in BULL)
GROWTH_SECTORS = [
    'Technology', 'Consumer Cyclical', 'Communication Services',
    'Industrials', 'Financial Services'
]


class RegimeAdapter:
    """Adapts trading strategy based on market regime"""
    
    def __init__(self):
        self.current_regime = MarketRegime.RANGE
        self.regime_confidence = 0.0
        self.last_detection = None
        
    def detect_regime(self):
        """Detect current market regime using SPY and VIX"""
        try:
            # Get SPY data
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="3mo")
            
            if spy_hist.empty:
                logger.warning("[REGIME] No SPY data, defaulting to RANGE")
                return MarketRegime.RANGE, 0.5
            
            close = spy_hist['Close']
            
            # Calculate indicators
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            
            current_price = close.iloc[-1]
            price_vs_ema20 = (current_price / ema20.iloc[-1] - 1) * 100
            price_vs_ema50 = (current_price / ema50.iloc[-1] - 1) * 100
            
            # Trend (20-day return)
            trend_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100
            
            # Volatility
            returns = close.pct_change()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # Get VIX
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="5d")
                vix_level = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20
            except:
                vix_level = 20
            
            # Regime detection logic
            if vix_level > 30 or volatility > 35:
                regime = MarketRegime.VOLATILE
                confidence = 0.85
            elif price_vs_ema50 > 3 and trend_20d > 3 and vix_level < 20:
                regime = MarketRegime.BULL
                confidence = min(0.9, 0.6 + trend_20d / 30)
            elif price_vs_ema50 < -3 and trend_20d < -3:
                regime = MarketRegime.BEAR
                confidence = min(0.9, 0.6 + abs(trend_20d) / 30)
            else:
                regime = MarketRegime.RANGE
                confidence = 0.6
            
            self.current_regime = regime
            self.regime_confidence = confidence
            
            logger.info(
                f"[REGIME] Detected: {regime.value.upper()} (conf={confidence:.0%}) | "
                f"SPY: {price_vs_ema50:+.1f}% vs EMA50, Trend: {trend_20d:+.1f}%, VIX: {vix_level:.1f}"
            )
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"[REGIME] Detection error: {e}")
            return MarketRegime.RANGE, 0.5
    
    def get_config(self, regime: MarketRegime = None) -> RegimeConfig:
        """Get configuration for current or specified regime"""
        regime = regime or self.current_regime
        return REGIME_CONFIGS.get(regime, REGIME_CONFIGS[MarketRegime.RANGE])
    
    def filter_by_regime(self, symbols: List[str], regime: MarketRegime = None) -> List[str]:
        """Filter symbols based on regime preferences"""
        regime = regime or self.current_regime
        config = self.get_config(regime)
        
        filtered = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Market cap filter
                market_cap = info.get('marketCap', 0)
                if market_cap < config.market_cap_min:
                    continue
                
                # Sector preference
                sector = info.get('sector', '')
                
                if config.prefer_defensive and sector not in DEFENSIVE_SECTORS:
                    # In BEAR/VOLATILE, skip non-defensive unless very high score
                    continue
                    
                if not config.allow_growth_stocks and sector in GROWTH_SECTORS:
                    # In BEAR, limit growth exposure
                    if market_cap < 10_000_000_000:  # Allow mega-cap growth
                        continue
                
                filtered.append(symbol)
                
            except Exception as e:
                # If we can't get info, include it (will be filtered by other checks)
                filtered.append(symbol)
        
        logger.info(f"[REGIME] Filtered {len(symbols)} -> {len(filtered)} symbols for {regime.value} regime")
        return filtered
    
    def adjust_score(self, base_score: float, pillar_scores: Dict[str, float], regime: MarketRegime = None) -> float:
        """Adjust final score based on regime weights"""
        regime = regime or self.current_regime
        config = self.get_config(regime)
        
        # Apply regime-specific weights
        adjusted = base_score
        
        technical = pillar_scores.get('technical', 0)
        fundamental = pillar_scores.get('fundamental', 0)
        news = pillar_scores.get('news', 0)
        
        # Weight adjustments
        adjustment = (
            (technical * (config.momentum_weight - 1.0)) +
            (fundamental * (config.fundamental_weight - 1.0)) +
            (news * (config.news_weight - 1.0))
        ) / 3
        
        adjusted += adjustment
        
        return max(0, min(100, adjusted))


# Singleton instance
_adapter = None

def get_regime_adapter() -> RegimeAdapter:
    global _adapter
    if _adapter is None:
        _adapter = RegimeAdapter()
    return _adapter
