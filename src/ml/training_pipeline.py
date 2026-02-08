"""
ML Training Pipeline - Generates training data from backtester results.

Runs the V6 backtester in Mode B/C on historical data, extracts:
1. 40 features from MLPillar._extract_features() style extraction
2. Market regime at signal time
3. Trade outcome: profitable (1) or loss (0)
4. PnL% for sample weighting

Stores in data/ml_training_data.parquet
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training data generation"""
    train_start: str = '2015-01-01'
    train_end: str = '2022-01-01'
    validation_start: str = '2022-01-01'
    validation_end: str = '2024-01-01'
    test_start: str = '2024-01-01'
    test_end: str = '2025-12-31'
    output_dir: str = 'data'
    min_data_points: int = 100


# Feature names (matching MLPillar's 40 features)
FEATURE_NAMES = [
    # Trend (10)
    'ema_cross_20_50', 'ema_cross_50_200', 'macd_histogram',
    'macd_signal_cross', 'adx_value', 'adx_direction',
    'supertrend_signal', 'aroon_oscillator', 'price_vs_ema20',
    'price_vs_ema50',
    # Momentum (10)
    'rsi_14', 'rsi_slope', 'stoch_k', 'stoch_d',
    'williams_r', 'cci_20', 'roc_10', 'momentum_10',
    'rsi_oversold', 'rsi_overbought',
    # Volume (8)
    'volume_ratio_20', 'obv_trend', 'obv_slope',
    'cmf_20', 'mfi_14', 'volume_trend_5d',
    'volume_breakout', 'price_volume_trend',
    # Volatility (6)
    'atr_percent', 'atr_ratio', 'bb_width', 'bb_percent',
    'volatility_20d', 'volatility_expansion',
    # Regime (6)
    'spy_above_ema50', 'vix_level', 'vix_percentile',
    'sector_momentum', 'market_breadth', 'correlation_spy'
]


class FeatureExtractor:
    """Extracts 40 ML features from OHLCV data at a specific date index.
    Replicates the MLPillar._extract_features() logic but works on historical data
    without lookahead bias."""

    def extract_at(
        self,
        df: pd.DataFrame,
        date_loc: int,
        spy_df: pd.DataFrame = None,
        vix_df: pd.DataFrame = None
    ) -> Optional[Dict[str, float]]:
        """Extract all 40 features at a specific date index.

        Args:
            df: OHLCV DataFrame for the symbol
            date_loc: Integer location in the DataFrame index (no lookahead)
            spy_df: SPY data for regime features
            vix_df: VIX data for regime features

        Returns:
            Dict of feature_name -> value, or None if insufficient data
        """
        if date_loc < 100 or date_loc >= len(df):
            return None

        # Use data only up to date_loc (no lookahead)
        data = df.iloc[:date_loc + 1]

        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

            features = {}

            # === TREND FEATURES ===
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean() if len(data) >= 200 else ema50

            features['ema_cross_20_50'] = 1.0 if ema20.iloc[-1] > ema50.iloc[-1] else -1.0
            features['ema_cross_50_200'] = 1.0 if ema50.iloc[-1] > ema200.iloc[-1] else -1.0
            features['price_vs_ema20'] = (close.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1] * 100
            features['price_vs_ema50'] = (close.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1] * 100

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            features['macd_histogram'] = float(macd.iloc[-1] - signal.iloc[-1])
            features['macd_signal_cross'] = 1.0 if macd.iloc[-1] > signal.iloc[-1] else -1.0

            # ADX
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()

            features['adx_value'] = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 25.0
            features['adx_direction'] = 1.0 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1.0

            # Supertrend (simplified)
            upper_band = (high + low) / 2 + (2 * atr)
            features['supertrend_signal'] = 1.0 if close.iloc[-1] > upper_band.iloc[-1] else -1.0

            # Aroon
            if len(data) >= 14:
                aroon_up = (
                    (14 - (14 - high.rolling(14).apply(lambda x: x.argmax(), raw=True))) / 14
                ) * 100
                aroon_down = (
                    (14 - (14 - low.rolling(14).apply(lambda x: x.argmin(), raw=True))) / 14
                ) * 100
                features['aroon_oscillator'] = float(aroon_up.iloc[-1] - aroon_down.iloc[-1])
            else:
                features['aroon_oscillator'] = 0.0

            # === MOMENTUM FEATURES ===
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss_s
            rsi = 100 - (100 / (1 + rs))

            features['rsi_14'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            features['rsi_slope'] = (
                float(rsi.iloc[-1] - rsi.iloc[-5])
                if len(rsi) > 5 and not pd.isna(rsi.iloc[-5])
                else 0.0
            )
            features['rsi_oversold'] = 1.0 if features['rsi_14'] < 30 else 0.0
            features['rsi_overbought'] = 1.0 if features['rsi_14'] > 70 else 0.0

            # Stochastic
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            hh_ll_range = highest_high - lowest_low
            hh_ll_range = hh_ll_range.replace(0, 1)
            stoch_k = ((close - lowest_low) / hh_ll_range) * 100
            stoch_d = stoch_k.rolling(3).mean()
            features['stoch_k'] = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
            features['stoch_d'] = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50.0

            # Williams %R
            if highest_high.iloc[-1] != lowest_low.iloc[-1]:
                features['williams_r'] = float(
                    -100 * (highest_high.iloc[-1] - close.iloc[-1])
                    / (highest_high.iloc[-1] - lowest_low.iloc[-1])
                )
            else:
                features['williams_r'] = -50.0

            # CCI
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            if not pd.isna(mad.iloc[-1]) and mad.iloc[-1] != 0:
                features['cci_20'] = float(
                    (tp.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mad.iloc[-1])
                )
            else:
                features['cci_20'] = 0.0

            # ROC & Momentum
            features['roc_10'] = (
                float(((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100)
                if len(close) > 10
                else 0.0
            )
            features['momentum_10'] = (
                float(close.iloc[-1] - close.iloc[-10])
                if len(close) > 10
                else 0.0
            )

            # === VOLUME FEATURES ===
            avg_vol_20 = volume.rolling(20).mean()
            features['volume_ratio_20'] = (
                float(volume.iloc[-1] / avg_vol_20.iloc[-1])
                if avg_vol_20.iloc[-1] > 0
                else 1.0
            )
            features['volume_trend_5d'] = (
                float((volume.iloc[-1] - volume.iloc[-5]) / volume.iloc[-5])
                if len(volume) > 5 and volume.iloc[-5] > 0
                else 0.0
            )
            features['volume_breakout'] = 1.0 if features['volume_ratio_20'] > 2.0 else 0.0

            # OBV
            obv = (np.sign(close.diff()) * volume).cumsum()
            features['obv_trend'] = (
                1.0 if len(obv) > 20 and obv.iloc[-1] > obv.iloc[-20] else -1.0
            )
            features['obv_slope'] = (
                float((obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) * 100)
                if len(obv) > 5 and obv.iloc[-5] != 0
                else 0.0
            )

            # CMF
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
            mfv = mfm * volume
            vol_sum = volume.rolling(20).sum()
            features['cmf_20'] = (
                float(mfv.rolling(20).sum().iloc[-1] / vol_sum.iloc[-1])
                if vol_sum.iloc[-1] > 0
                else 0.0
            )

            # MFI
            tp2 = (high + low + close) / 3
            mf = tp2 * volume
            pos_mf = mf.where(tp2 > tp2.shift(1), 0).rolling(14).sum()
            neg_mf = mf.where(tp2 < tp2.shift(1), 0).rolling(14).sum()
            features['mfi_14'] = (
                float(100 - (100 / (1 + pos_mf.iloc[-1] / neg_mf.iloc[-1])))
                if neg_mf.iloc[-1] > 0
                else 50.0
            )

            # Price Volume Trend
            features['price_volume_trend'] = (
                features['volume_ratio_20'] * np.sign(close.iloc[-1] - close.iloc[-2])
            )

            # === VOLATILITY FEATURES ===
            atr_series = atr
            features['atr_percent'] = (
                float(atr_series.iloc[-1] / close.iloc[-1] * 100)
                if not pd.isna(atr_series.iloc[-1])
                else 2.0
            )
            atr_avg = atr_series.rolling(20).mean()
            features['atr_ratio'] = (
                float(atr_series.iloc[-1] / atr_avg.iloc[-1])
                if len(atr_series) > 20 and atr_avg.iloc[-1] > 0
                else 1.0
            )

            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_upper = sma20 + (2 * std20)
            bb_lower = sma20 - (2 * std20)
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            features['bb_width'] = (
                float(bb_range / sma20.iloc[-1] * 100) if sma20.iloc[-1] > 0 else 0.0
            )
            features['bb_percent'] = (
                float((close.iloc[-1] - bb_lower.iloc[-1]) / bb_range) if bb_range > 0 else 0.5
            )

            returns = close.pct_change()
            vol_20 = returns.rolling(20).std()
            features['volatility_20d'] = (
                float(vol_20.iloc[-1] * np.sqrt(252) * 100)
                if not pd.isna(vol_20.iloc[-1])
                else 20.0
            )
            features['volatility_expansion'] = 1.0 if features['atr_ratio'] > 1.5 else 0.0

            # === REGIME FEATURES ===
            current_date = df.index[date_loc]

            if spy_df is not None and len(spy_df) > 50:
                spy_slice = spy_df.loc[:current_date]
                if len(spy_slice) >= 50:
                    spy_close = spy_slice['Close']
                    spy_ema50 = spy_close.ewm(span=50, adjust=False).mean()
                    features['spy_above_ema50'] = (
                        1.0 if spy_close.iloc[-1] > spy_ema50.iloc[-1] else -1.0
                    )

                    features['sector_momentum'] = (
                        float(spy_close.iloc[-1] / spy_close.iloc[-20] - 1) * 100
                        if len(spy_close) > 20
                        else 0.0
                    )

                    # Correlation with SPY
                    if len(close) >= 60 and len(spy_close) >= 60:
                        aligned = pd.concat(
                            [returns.iloc[-60:], spy_close.pct_change().iloc[-60:]],
                            axis=1
                        ).dropna()
                        if len(aligned) > 20:
                            features['correlation_spy'] = float(
                                aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                            )
                        else:
                            features['correlation_spy'] = 0.5
                    else:
                        features['correlation_spy'] = 0.5
                else:
                    features['spy_above_ema50'] = 0.0
                    features['sector_momentum'] = 0.0
                    features['correlation_spy'] = 0.5
            else:
                features['spy_above_ema50'] = 0.0
                features['sector_momentum'] = 0.0
                features['correlation_spy'] = 0.5

            if vix_df is not None:
                vix_slice = vix_df.loc[:current_date]
                if not vix_slice.empty:
                    vix_val = float(vix_slice['Close'].iloc[-1])
                    features['vix_level'] = vix_val
                    # VIX percentile over last 252 days
                    if len(vix_slice) >= 252:
                        vix_hist = vix_slice['Close'].iloc[-252:]
                        features['vix_percentile'] = float((vix_hist < vix_val).mean() * 100)
                    else:
                        features['vix_percentile'] = 50.0
                else:
                    features['vix_level'] = 20.0
                    features['vix_percentile'] = 50.0
            else:
                features['vix_level'] = 20.0
                features['vix_percentile'] = 50.0

            # Not available in backtest without broad market data
            features['market_breadth'] = 50.0

            # Replace NaN/inf with defaults
            for key in features:
                val = features[key]
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    features[key] = 0.0

            return features

        except Exception as e:
            logger.warning(f"Feature extraction error at loc {date_loc}: {e}")
            return None


class TrainingDataGenerator:
    """Generates ML training data by running backtest and extracting features + labels."""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.feature_extractor = FeatureExtractor()

    def generate(
        self,
        symbols: List[str],
        period: str = 'train',
        forward_returns_days: int = 20
    ) -> pd.DataFrame:
        """Generate training data for a specific period.

        For each symbol on each weekly scan date:
        1. Extract 40 features (no lookahead)
        2. Calculate forward return over next N days (label)
        3. Detect regime at that point

        Args:
            symbols: List of stock symbols
            period: 'train', 'validation', or 'test'
            forward_returns_days: Days ahead to measure outcome

        Returns:
            DataFrame with features, labels, and metadata
        """
        # Determine date range
        if period == 'train':
            start, end = self.config.train_start, self.config.train_end
        elif period == 'validation':
            start, end = self.config.validation_start, self.config.validation_end
        else:
            start, end = self.config.test_start, self.config.test_end

        # Buffer for indicators
        buffer_start = (pd.Timestamp(start) - timedelta(days=365)).strftime('%Y-%m-%d')
        # Buffer end for forward returns
        buffer_end = (
            pd.Timestamp(end) + timedelta(days=forward_returns_days + 30)
        ).strftime('%Y-%m-%d')

        logger.info(f"Generating {period} data: {start} to {end}, {len(symbols)} symbols")

        # Fetch SPY and VIX
        spy_df = self._fetch_data('SPY', buffer_start, buffer_end)
        vix_df = self._fetch_data('^VIX', buffer_start, buffer_end)

        all_records = []

        for i, symbol in enumerate(symbols):
            if (i + 1) % 20 == 0:
                logger.info(
                    f"Processing {i+1}/{len(symbols)}: {symbol} "
                    f"({len(all_records)} records so far)"
                )

            df = self._fetch_data(symbol, buffer_start, buffer_end)
            if df is None or len(df) < 200:
                continue

            # Build trading day index within our period
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            period_mask = (df.index >= start_ts) & (df.index <= end_ts)
            period_indices = df.index[period_mask]

            # Weekly sampling (every 5 trading days)
            for j, date in enumerate(period_indices):
                if j % 5 != 0:
                    continue

                date_loc = df.index.get_loc(date)
                if date_loc < 100:
                    continue

                # Extract features (no lookahead - uses data up to date_loc)
                features = self.feature_extractor.extract_at(df, date_loc, spy_df, vix_df)
                if features is None:
                    continue

                # Calculate forward return (label)
                future_loc = min(date_loc + forward_returns_days, len(df) - 1)
                if future_loc <= date_loc:
                    continue

                current_price = float(df['Close'].iloc[date_loc])
                future_price = float(df['Close'].iloc[future_loc])
                forward_return = (future_price - current_price) / current_price

                # Detect regime
                regime = self._detect_regime_at(spy_df, vix_df, date)

                # Create record
                record = {
                    'symbol': symbol,
                    'date': date,
                    'regime': regime,
                    'forward_return': forward_return,
                    'label': 1 if forward_return > 0.01 else 0,  # Profitable if > 1% (covers costs)
                    'forward_return_raw': forward_return,
                    'current_price': current_price,
                    **features
                }
                all_records.append(record)

        df_result = pd.DataFrame(all_records)
        logger.info(f"Generated {len(df_result)} training records for {period}")

        if len(df_result) > 0:
            pos_rate = df_result['label'].mean()
            logger.info(f"Positive rate: {pos_rate:.1%}")

        return df_result

    def generate_all_periods(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Generate training data for all 3 periods."""
        results = {}
        for period in ['train', 'validation', 'test']:
            results[period] = self.generate(symbols, period)
        return results

    def save(self, df: pd.DataFrame, filename: str = 'ml_training_data.parquet'):
        """Save training data to parquet."""
        output_path = os.path.join(self.config.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
        return output_path

    def _fetch_data(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data via yfinance."""
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            return None

    def _detect_regime_at(
        self,
        spy_df: Optional[pd.DataFrame],
        vix_df: Optional[pd.DataFrame],
        date: pd.Timestamp
    ) -> str:
        """Detect market regime at a specific date (no lookahead)."""
        try:
            if spy_df is None:
                return 'RANGE'

            spy_slice = spy_df.loc[:date]
            if len(spy_slice) < 50:
                return 'RANGE'

            close = spy_slice['Close']
            ema50 = close.ewm(span=50, adjust=False).mean()
            current_price = float(close.iloc[-1])
            ema50_val = float(ema50.iloc[-1])
            price_vs_ema50 = (current_price / ema50_val - 1) * 100

            trend_20d = 0
            if len(close) >= 20:
                trend_20d = (float(close.iloc[-1]) / float(close.iloc[-20]) - 1) * 100

            returns = close.pct_change()
            vol_val = returns.rolling(20).std().iloc[-1]
            volatility = (
                float(vol_val) * np.sqrt(252) * 100 if not pd.isna(vol_val) else 20.0
            )

            vix_level = 20.0
            if vix_df is not None:
                vix_slice = vix_df.loc[:date]
                if not vix_slice.empty:
                    vix_level = float(vix_slice['Close'].iloc[-1])

            if vix_level > 30 or volatility > 35:
                return 'VOLATILE'
            elif price_vs_ema50 > 3 and trend_20d > 3 and vix_level < 20:
                return 'BULL'
            elif price_vs_ema50 < -3 and trend_20d < -3:
                return 'BEAR'
            else:
                return 'RANGE'

        except Exception:
            return 'RANGE'


def generate_training_data(
    symbols: List[str],
    config: TrainingConfig = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function to generate all training data."""
    generator = TrainingDataGenerator(config)
    results = generator.generate_all_periods(symbols)

    # Save all periods
    for period, df in results.items():
        if len(df) > 0:
            generator.save(df, f'ml_{period}_data.parquet')

    return results
