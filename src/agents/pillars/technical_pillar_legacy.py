"""
Technical Pillar - Technical analysis using indicator library.

Analyzes:
- Trend indicators (EMA, MACD, ADX)
- Momentum indicators (RSI, Stochastic)
- Volume indicators (OBV, VWAP, Volume Ratio)
- Volatility (ATR, Bollinger Bands)
- Support/Resistance levels

Contributes 25% to final decision score.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base import BasePillar, PillarScore
from src.indicators import (
    get_indicator_registry,
    compute_indicator,
    RSI, EMA, MACD, ADX, Supertrend,
    OBV, VWAP, VolumeRatio, CMF,
    ATR, BollingerBands,
    SupportResistance
)

logger = logging.getLogger(__name__)


class TechnicalPillar(BasePillar):
    """
    Technical analysis pillar.

    Uses the indicator library to compute a comprehensive
    technical score based on multiple indicator categories.
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(weight)
        self.registry = get_indicator_registry()

        # Indicator weights within technical analysis
        self.category_weights = {
            'trend': 0.30,      # EMA, MACD, ADX
            'momentum': 0.25,   # RSI, Stochastic
            'volume': 0.25,     # OBV, Volume Ratio
            'volatility': 0.20  # ATR, Bollinger
        }

    def get_name(self) -> str:
        return "Technical"

    async def analyze(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> PillarScore:
        """
        Perform technical analysis on a symbol.

        Args:
            symbol: Stock symbol
            data: Must contain 'df' with OHLCV DataFrame

        Returns:
            PillarScore with technical analysis result
        """
        df = data.get('df')
        if df is None or len(df) < 50:
            logger.warning(f"[TECHNICAL] {symbol}: Insufficient data ({len(df) if df is not None else 0} rows, need 50+)")
            return self._create_score(
                score=0,
                reasoning=f"Insufficient data for technical analysis ({len(df) if df is not None else 0} rows)",
                data_quality=0.0
            )

        logger.info(f"[TECHNICAL] {symbol}: Analyzing {len(df)} data points...")

        factors = []
        category_scores = {}

        # 1. Trend Analysis (30%)
        trend_score, trend_factors = self._analyze_trend(df)
        category_scores['trend'] = trend_score
        factors.extend(trend_factors)

        # 2. Momentum Analysis (25%)
        momentum_score, momentum_factors = self._analyze_momentum(df)
        category_scores['momentum'] = momentum_score
        factors.extend(momentum_factors)

        # 3. Volume Analysis (25%)
        volume_score, volume_factors = self._analyze_volume(df)
        category_scores['volume'] = volume_score
        factors.extend(volume_factors)

        # 4. Volatility Analysis (20%)
        volatility_score, volatility_factors = self._analyze_volatility(df)
        category_scores['volatility'] = volatility_score
        factors.extend(volatility_factors)

        # Calculate weighted total
        total_score = sum(
            category_scores[cat] * self.category_weights[cat]
            for cat in category_scores
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(category_scores, factors)

        # Calculate confidence based on indicator agreement
        confidence = self._calculate_confidence(factors)

        # V4.4: Verbose logging
        signal = "bullish" if total_score > 20 else "bearish" if total_score < -20 else "neutral"
        logger.info(f"[TECHNICAL] {symbol}: Score={(total_score + 100) / 2:.1f}/100 ({signal}) | Trend={category_scores.get('trend', 0):.0f} Momentum={category_scores.get('momentum', 0):.0f} Volume={category_scores.get('volume', 0):.0f} Volatility={category_scores.get('volatility', 0):.0f}")

        return self._create_score(
            score=(total_score + 100) / 2,  # Normalized to 0-100 (50=neutral)
            reasoning=reasoning,
            factors=factors,
            confidence=confidence
        )

    def _analyze_trend(self, df: pd.DataFrame) -> tuple[float, List[Dict]]:
        """Analyze trend indicators"""
        factors = []
        scores = []

        try:
            # EMA Analysis (20/50/200)
            df = compute_indicator('EMA', df, period=20)
            df = compute_indicator('EMA', df, period=50)
            df = compute_indicator('EMA', df, period=200)

            price = df['Close'].iloc[-1]
            # EMA indicator uses 'EMA' for period=20, 'EMA_{period}' otherwise
            ema20 = df['EMA'].iloc[-1] if 'EMA' in df.columns else None
            ema50 = df['EMA_50'].iloc[-1] if 'EMA_50' in df.columns else None
            ema200 = df['EMA_200'].iloc[-1] if 'EMA_200' in df.columns and not pd.isna(df['EMA_200'].iloc[-1]) else None

            if ema20 is None or ema50 is None:
                raise ValueError("EMA computation failed")

            # EMA alignment score
            if ema200:
                if price > ema20 > ema50 > ema200:
                    ema_score = 80
                    ema_msg = "Strong uptrend (price > EMA20 > EMA50 > EMA200)"
                elif price < ema20 < ema50 < ema200:
                    ema_score = -80
                    ema_msg = "Strong downtrend (price < EMA20 < EMA50 < EMA200)"
                elif price > ema50:
                    ema_score = 30
                    ema_msg = "Moderate uptrend (above EMA50)"
                elif price < ema50:
                    ema_score = -30
                    ema_msg = "Moderate downtrend (below EMA50)"
                else:
                    ema_score = 0
                    ema_msg = "Mixed trend"
            else:
                # Without EMA200
                if price > ema20 > ema50:
                    ema_score = 60
                    ema_msg = "Uptrend (price > EMA20 > EMA50)"
                elif price < ema20 < ema50:
                    ema_score = -60
                    ema_msg = "Downtrend (price < EMA20 < EMA50)"
                else:
                    ema_score = 0
                    ema_msg = "Consolidating"

            factors.append({
                'indicator': 'EMA',
                'score': ema_score,
                'message': ema_msg,
                'values': {'price': price, 'ema20': ema20, 'ema50': ema50}
            })
            scores.append(ema_score)

        except Exception as e:
            logger.warning(f"EMA analysis failed: {e}")

        try:
            # MACD Analysis
            df = compute_indicator('MACD', df)
            macd = df['MACD'].iloc[-1]
            signal = df['MACD_SIGNAL'].iloc[-1]
            hist = df['MACD_HIST'].iloc[-1]
            prev_hist = df['MACD_HIST'].iloc[-2] if len(df) > 1 else hist

            if macd > signal and hist > 0:
                if hist > prev_hist:
                    macd_score = 70
                    macd_msg = "MACD bullish and strengthening"
                else:
                    macd_score = 40
                    macd_msg = "MACD bullish but weakening"
            elif macd < signal and hist < 0:
                if hist < prev_hist:
                    macd_score = -70
                    macd_msg = "MACD bearish and strengthening"
                else:
                    macd_score = -40
                    macd_msg = "MACD bearish but weakening"
            else:
                macd_score = 0
                macd_msg = "MACD neutral"

            factors.append({
                'indicator': 'MACD',
                'score': macd_score,
                'message': macd_msg,
                'values': {'macd': macd, 'signal': signal, 'histogram': hist}
            })
            scores.append(macd_score)

        except Exception as e:
            logger.warning(f"MACD analysis failed: {e}")

        try:
            # ADX Analysis (trend strength)
            df = compute_indicator('ADX', df)
            adx = df['ADX'].iloc[-1]
            plus_di = df['PLUS_DI'].iloc[-1]
            minus_di = df['MINUS_DI'].iloc[-1]

            if adx > 25:
                if plus_di > minus_di:
                    adx_score = min(80, 30 + adx)
                    adx_msg = f"Strong uptrend (ADX={adx:.1f})"
                else:
                    adx_score = max(-80, -30 - adx)
                    adx_msg = f"Strong downtrend (ADX={adx:.1f})"
            else:
                adx_score = 0
                adx_msg = f"Weak trend (ADX={adx:.1f})"

            factors.append({
                'indicator': 'ADX',
                'score': adx_score,
                'message': adx_msg,
                'values': {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
            })
            scores.append(adx_score)

        except Exception as e:
            logger.warning(f"ADX analysis failed: {e}")

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors

    def _analyze_momentum(self, df: pd.DataFrame) -> tuple[float, List[Dict]]:
        """Analyze momentum indicators"""
        factors = []
        scores = []

        try:
            # RSI Analysis
            df = compute_indicator('RSI', df, period=14)
            rsi = df['RSI'].iloc[-1]
            prev_rsi = df['RSI'].iloc[-2] if len(df) > 1 else rsi

            if rsi < 30:
                rsi_score = 60  # Oversold = bullish opportunity
                rsi_msg = f"Oversold (RSI={rsi:.1f})"
            elif rsi > 70:
                rsi_score = -60  # Overbought = bearish risk
                rsi_msg = f"Overbought (RSI={rsi:.1f})"
            elif rsi > 50 and rsi > prev_rsi:
                rsi_score = 30
                rsi_msg = f"Bullish momentum (RSI={rsi:.1f})"
            elif rsi < 50 and rsi < prev_rsi:
                rsi_score = -30
                rsi_msg = f"Bearish momentum (RSI={rsi:.1f})"
            else:
                rsi_score = 0
                rsi_msg = f"Neutral (RSI={rsi:.1f})"

            factors.append({
                'indicator': 'RSI',
                'score': rsi_score,
                'message': rsi_msg,
                'values': {'rsi': rsi}
            })
            scores.append(rsi_score)

        except Exception as e:
            logger.warning(f"RSI analysis failed: {e}")

        try:
            # Stochastic Analysis
            df = compute_indicator('Stochastic', df)
            k = df['STOCH_K'].iloc[-1]
            d = df['STOCH_D'].iloc[-1]

            if k < 20 and d < 20:
                stoch_score = 50
                stoch_msg = f"Oversold (K={k:.1f}, D={d:.1f})"
            elif k > 80 and d > 80:
                stoch_score = -50
                stoch_msg = f"Overbought (K={k:.1f}, D={d:.1f})"
            elif k > d:
                stoch_score = 20
                stoch_msg = f"Bullish crossover (K > D)"
            elif k < d:
                stoch_score = -20
                stoch_msg = f"Bearish crossover (K < D)"
            else:
                stoch_score = 0
                stoch_msg = "Neutral"

            factors.append({
                'indicator': 'Stochastic',
                'score': stoch_score,
                'message': stoch_msg,
                'values': {'k': k, 'd': d}
            })
            scores.append(stoch_score)

        except Exception as e:
            logger.warning(f"Stochastic analysis failed: {e}")

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors

    def _analyze_volume(self, df: pd.DataFrame) -> tuple[float, List[Dict]]:
        """Analyze volume indicators"""
        factors = []
        scores = []

        try:
            # Volume Ratio
            df = compute_indicator('VolumeRatio', df, period=20)
            vol_ratio = df['VOLUME_RATIO'].iloc[-1]
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]

            if vol_ratio > 2.0:
                if price_change > 0:
                    vol_score = 70
                    vol_msg = f"High volume buying ({vol_ratio:.1f}x avg)"
                else:
                    vol_score = -70
                    vol_msg = f"High volume selling ({vol_ratio:.1f}x avg)"
            elif vol_ratio > 1.5:
                if price_change > 0:
                    vol_score = 40
                    vol_msg = f"Above avg volume up ({vol_ratio:.1f}x)"
                else:
                    vol_score = -40
                    vol_msg = f"Above avg volume down ({vol_ratio:.1f}x)"
            elif vol_ratio < 0.5:
                vol_score = 0
                vol_msg = f"Low volume ({vol_ratio:.1f}x) - weak conviction"
            else:
                vol_score = 10 if price_change > 0 else -10
                vol_msg = f"Normal volume ({vol_ratio:.1f}x)"

            factors.append({
                'indicator': 'VolumeRatio',
                'score': vol_score,
                'message': vol_msg,
                'values': {'ratio': vol_ratio, 'price_change': price_change}
            })
            scores.append(vol_score)

        except Exception as e:
            logger.warning(f"Volume ratio analysis failed: {e}")

        try:
            # OBV trend
            df = compute_indicator('OBV', df)
            obv = df['OBV'].iloc[-1]
            obv_prev = df['OBV'].iloc[-5] if len(df) > 5 else obv
            obv_change = (obv - obv_prev) / abs(obv_prev) if obv_prev != 0 else 0

            price = df['Close'].iloc[-1]
            price_prev = df['Close'].iloc[-5] if len(df) > 5 else price
            price_change = (price - price_prev) / price_prev

            if obv_change > 0 and price_change > 0:
                obv_score = 40
                obv_msg = "OBV confirms uptrend"
            elif obv_change < 0 and price_change < 0:
                obv_score = -40
                obv_msg = "OBV confirms downtrend"
            elif obv_change > 0 and price_change < 0:
                obv_score = 50
                obv_msg = "Bullish OBV divergence (accumulation)"
            elif obv_change < 0 and price_change > 0:
                obv_score = -50
                obv_msg = "Bearish OBV divergence (distribution)"
            else:
                obv_score = 0
                obv_msg = "OBV neutral"

            factors.append({
                'indicator': 'OBV',
                'score': obv_score,
                'message': obv_msg,
                'values': {'obv_change_pct': obv_change * 100}
            })
            scores.append(obv_score)

        except Exception as e:
            logger.warning(f"OBV analysis failed: {e}")

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors

    def _analyze_volatility(self, df: pd.DataFrame) -> tuple[float, List[Dict]]:
        """Analyze volatility indicators"""
        factors = []
        scores = []

        try:
            # ATR analysis
            df = compute_indicator('ATR', df, period=14)
            atr = df['ATR'].iloc[-1]
            atr_pct = df['ATR_PERCENT'].iloc[-1]
            atr_avg = df['ATR'].rolling(20).mean().iloc[-1]
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1

            if atr_ratio > 1.5:
                atr_score = -20  # High volatility = risk
                atr_msg = f"High volatility ({atr_ratio:.1f}x normal)"
            elif atr_ratio < 0.7:
                atr_score = 20  # Low volatility = potential breakout
                atr_msg = f"Low volatility - potential breakout"
            else:
                atr_score = 0
                atr_msg = f"Normal volatility"

            factors.append({
                'indicator': 'ATR',
                'score': atr_score,
                'message': atr_msg,
                'values': {'atr': atr, 'atr_pct': atr_pct, 'atr_ratio': atr_ratio}
            })
            scores.append(atr_score)

        except Exception as e:
            logger.warning(f"ATR analysis failed: {e}")

        try:
            # Bollinger Bands
            df = compute_indicator('BollingerBands', df)
            price = df['Close'].iloc[-1]
            upper = df['BB_UPPER'].iloc[-1]
            lower = df['BB_LOWER'].iloc[-1]
            bb_pct = df['BB_PERCENT'].iloc[-1]

            if price < lower:
                bb_score = 50
                bb_msg = f"Below lower BB (oversold)"
            elif price > upper:
                bb_score = -50
                bb_msg = f"Above upper BB (overbought)"
            elif bb_pct < 0.2:
                bb_score = 30
                bb_msg = f"Near lower BB (potential bounce)"
            elif bb_pct > 0.8:
                bb_score = -30
                bb_msg = f"Near upper BB (potential pullback)"
            else:
                bb_score = 0
                bb_msg = f"Within bands (%B={bb_pct:.2f})"

            factors.append({
                'indicator': 'BollingerBands',
                'score': bb_score,
                'message': bb_msg,
                'values': {'bb_percent': bb_pct}
            })
            scores.append(bb_score)

        except Exception as e:
            logger.warning(f"Bollinger analysis failed: {e}")

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors

    def _generate_reasoning(
        self,
        category_scores: Dict[str, float],
        factors: List[Dict]
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []

        # Overall assessment
        total = sum(category_scores.get(c, 0) * self.category_weights.get(c, 0.25)
                    for c in category_scores)

        if total > 50:
            parts.append("Technical outlook: STRONGLY BULLISH")
        elif total > 20:
            parts.append("Technical outlook: BULLISH")
        elif total > -20:
            parts.append("Technical outlook: NEUTRAL")
        elif total > -50:
            parts.append("Technical outlook: BEARISH")
        else:
            parts.append("Technical outlook: STRONGLY BEARISH")

        # Category summaries
        for cat, score in category_scores.items():
            direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
            parts.append(f"- {cat.capitalize()}: {direction} ({score:+.0f})")

        # Key factors
        key_factors = sorted(factors, key=lambda x: abs(x.get('score', 0)), reverse=True)[:3]
        if key_factors:
            parts.append("\nKey signals:")
            for f in key_factors:
                parts.append(f"  * {f.get('message', 'N/A')}")

        return "\n".join(parts)

    def _calculate_confidence(self, factors: List[Dict]) -> float:
        """Calculate confidence based on indicator agreement"""
        if not factors:
            return 0.5

        scores = [f.get('score', 0) for f in factors]
        if not scores:
            return 0.5

        # Check if indicators agree
        bullish = sum(1 for s in scores if s > 20)
        bearish = sum(1 for s in scores if s < -20)
        total = len(scores)

        # Higher confidence if indicators agree
        agreement = max(bullish, bearish) / total if total > 0 else 0.5

        # Scale to 0.5-1.0 range
        return 0.5 + (agreement * 0.5)


# Singleton
_technical_pillar: Optional[TechnicalPillar] = None


def get_technical_pillar() -> TechnicalPillar:
    """Get or create the TechnicalPillar singleton"""
    global _technical_pillar
    if _technical_pillar is None:
        _technical_pillar = TechnicalPillar()
    return _technical_pillar
