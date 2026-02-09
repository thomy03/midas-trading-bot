"""
V7.1 Regime ML Predictor

A fundamentally different approach from V7's trade-level ML:
- V7 ML tried to predict "will this trade be profitable?" using the same
  technical features as the scoring system -> AUC=0.50 (random)
- V7.1 ML predicts "will the MARKET be up in 20 days?" using macro features
  (VIX, breadth, sector rotation) that the scoring system does NOT use

If AUC < 0.55 on OOS test, the model is disabled (conservative approach).

Usage:
    predictor = RegimeMLPredictor()
    predictor.train(spy_df, vix_df, symbol_data)
    prediction = predictor.predict(spy_df, vix_df, symbol_data, date)
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RegimeMLConfig:
    """Configuration for regime ML model"""
    forward_days: int = 20           # Predict market direction N days ahead
    min_auc: float = 0.55            # Minimum AUC to keep model active
    n_estimators: int = 100          # RandomForest trees
    max_depth: int = 5               # Shallow trees to prevent overfitting
    min_samples_leaf: int = 50       # Large leaf size for stability
    model_path: str = 'models/regime_ml_v71.joblib'
    history_path: str = 'data/regime_ml_history.json'


class RegimeMLFeatureExtractor:
    """Extracts MACRO features for regime prediction.

    Key difference from V7: these features are market-level, not stock-level.
    They capture conditions the scoring system cannot see.
    """

    def extract(
        self,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame],
        symbol_data: Dict[str, pd.DataFrame],
        date_loc: int,
        spy_date: pd.Timestamp = None
    ) -> Optional[Dict[str, float]]:
        """Extract macro features at a specific date.

        Args:
            spy_df: SPY OHLCV data
            vix_df: VIX data
            symbol_data: Dict of symbol -> OHLCV DataFrames
            date_loc: Index location in spy_df
            spy_date: The actual date (for symbol_data lookups)

        Returns:
            Dict of feature_name -> value, or None if insufficient data
        """
        if date_loc < 60:
            return None

        features = {}
        spy_close = spy_df['Close']

        # ── VIX Features ──
        if vix_df is not None and len(vix_df) > date_loc:
            vix_close = vix_df['Close']
            vix_now = float(vix_close.iloc[date_loc])
            features['vix_level'] = vix_now

            # VIX percentile (where current VIX falls in 1-year range)
            lookback = min(252, date_loc)
            vix_window = vix_close.iloc[date_loc - lookback:date_loc + 1].values
            features['vix_percentile'] = float(
                np.sum(vix_window <= vix_now) / len(vix_window)
            )

            # VIX 5-day change
            if date_loc >= 5:
                vix_5d = float(vix_close.iloc[date_loc - 5])
                features['vix_change_5d'] = (vix_now - vix_5d) / vix_5d if vix_5d > 0 else 0
            else:
                features['vix_change_5d'] = 0

            # VIX 20-day change
            if date_loc >= 20:
                vix_20d = float(vix_close.iloc[date_loc - 20])
                features['vix_change_20d'] = (vix_now - vix_20d) / vix_20d if vix_20d > 0 else 0
            else:
                features['vix_change_20d'] = 0

            # VIX term structure proxy: current vs 20d MA
            vix_ma20 = float(vix_close.iloc[max(0, date_loc-19):date_loc+1].mean())
            features['vix_vs_ma20'] = (vix_now / vix_ma20 - 1) if vix_ma20 > 0 else 0
        else:
            features['vix_level'] = 20.0
            features['vix_percentile'] = 0.5
            features['vix_change_5d'] = 0
            features['vix_change_20d'] = 0
            features['vix_vs_ma20'] = 0

        # ── SPY Trend Features ──
        spy_now = float(spy_close.iloc[date_loc])

        # Price vs EMA50
        ema50 = spy_close.ewm(span=50, adjust=False).mean()
        features['spy_vs_ema50'] = (spy_now / float(ema50.iloc[date_loc]) - 1)

        # Price vs EMA200
        if date_loc >= 200:
            ema200 = spy_close.ewm(span=200, adjust=False).mean()
            features['spy_vs_ema200'] = (spy_now / float(ema200.iloc[date_loc]) - 1)
        else:
            features['spy_vs_ema200'] = 0

        # SPY momentum at multiple timeframes
        for period_name, period in [('5d', 5), ('20d', 20), ('60d', 60)]:
            if date_loc >= period:
                past = float(spy_close.iloc[date_loc - period])
                features[f'spy_return_{period_name}'] = (spy_now - past) / past if past > 0 else 0
            else:
                features[f'spy_return_{period_name}'] = 0

        # SPY realized volatility (20d annualized)
        if date_loc >= 21:
            spy_rets = spy_close.pct_change().iloc[date_loc-20:date_loc+1]
            features['spy_vol_20d'] = float(spy_rets.std()) * np.sqrt(252)
        else:
            features['spy_vol_20d'] = 0.15

        # SPY drawdown from 52-week high
        lookback_252 = min(252, date_loc)
        spy_high = float(spy_close.iloc[date_loc - lookback_252:date_loc + 1].max())
        features['spy_drawdown'] = (spy_now - spy_high) / spy_high if spy_high > 0 else 0

        # ── Market Breadth Features ──
        if symbol_data and spy_date is not None:
            above_ema50 = 0
            above_ema20 = 0
            positive_20d = 0
            total = 0

            for sym, df in symbol_data.items():
                if spy_date not in df.index:
                    continue
                sloc = df.index.get_loc(spy_date)
                if sloc < 50:
                    continue
                total += 1
                close_val = float(df['Close'].iloc[sloc])
                ema50_val = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[sloc])
                ema20_val = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[sloc])

                if close_val > ema50_val:
                    above_ema50 += 1
                if close_val > ema20_val:
                    above_ema20 += 1
                if sloc >= 20:
                    past_close = float(df['Close'].iloc[sloc - 20])
                    if close_val > past_close:
                        positive_20d += 1

            if total > 0:
                features['breadth_ema50'] = above_ema50 / total
                features['breadth_ema20'] = above_ema20 / total
                features['breadth_positive_20d'] = positive_20d / total
            else:
                features['breadth_ema50'] = 0.5
                features['breadth_ema20'] = 0.5
                features['breadth_positive_20d'] = 0.5
        else:
            features['breadth_ema50'] = 0.5
            features['breadth_ema20'] = 0.5
            features['breadth_positive_20d'] = 0.5

        # ── Sector Rotation Features ──
        # Use dispersion of returns across symbols as a regime signal
        if symbol_data and spy_date is not None and total > 5:
            rets_20d = []
            for sym, df in symbol_data.items():
                if spy_date not in df.index:
                    continue
                sloc = df.index.get_loc(spy_date)
                if sloc >= 20:
                    r = (float(df['Close'].iloc[sloc]) / float(df['Close'].iloc[sloc - 20])) - 1
                    rets_20d.append(r)
            if len(rets_20d) >= 5:
                features['return_dispersion'] = float(np.std(rets_20d))
                features['return_skew'] = float(pd.Series(rets_20d).skew())
            else:
                features['return_dispersion'] = 0.02
                features['return_skew'] = 0
        else:
            features['return_dispersion'] = 0.02
            features['return_skew'] = 0

        return features


# Ordered list of feature names for consistent model input
REGIME_FEATURE_NAMES = [
    'vix_level', 'vix_percentile', 'vix_change_5d', 'vix_change_20d', 'vix_vs_ma20',
    'spy_vs_ema50', 'spy_vs_ema200',
    'spy_return_5d', 'spy_return_20d', 'spy_return_60d',
    'spy_vol_20d', 'spy_drawdown',
    'breadth_ema50', 'breadth_ema20', 'breadth_positive_20d',
    'return_dispersion', 'return_skew'
]


class RegimeMLPredictor:
    """ML model for market regime prediction.

    Predicts: "Will SPY be higher in 20 days?" (binary classification)
    Uses macro features that the scoring system does NOT use.
    """

    def __init__(self, config: RegimeMLConfig = None):
        self.config = config or RegimeMLConfig()
        self.model = None
        self.scaler = None
        self.auc_score = 0.0
        self.is_active = False
        self._extractor = RegimeMLFeatureExtractor()

    def train(
        self,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame],
        symbol_data: Dict[str, pd.DataFrame],
        train_end_loc: int = None
    ) -> Dict[str, float]:
        """Train the regime prediction model.

        Args:
            spy_df: Full SPY OHLCV data
            vix_df: Full VIX data
            symbol_data: Dict of symbol -> full OHLCV data
            train_end_loc: Index location to split train/test.
                          If None, uses 80/20 split.

        Returns:
            Dict with training metrics (auc, accuracy, etc.)
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score, accuracy_score
        except ImportError:
            logger.warning("scikit-learn not available - regime ML disabled")
            return {'error': 'sklearn not available'}

        n = len(spy_df)
        forward = self.config.forward_days

        # Generate features and targets
        logger.info(f"Generating regime ML features ({n} data points)...")
        features_list = []  # type: List[Dict[str, float]]
        targets = []  # type: List[int]
        dates = []  # type: List[pd.Timestamp]

        spy_close = spy_df['Close']

        for loc in range(60, n - forward):
            spy_date = spy_df.index[loc]
            feat = self._extractor.extract(spy_df, vix_df, symbol_data, loc, spy_date)
            if feat is None:
                continue

            # Target: SPY return over forward_days > 0
            future_price = float(spy_close.iloc[loc + forward])
            current_price = float(spy_close.iloc[loc])
            target = 1 if future_price > current_price else 0

            features_list.append(feat)
            targets.append(target)
            dates.append(spy_date)

        if len(features_list) < 200:
            logger.warning(f"Too few samples ({len(features_list)}) for regime ML")
            return {'error': 'insufficient data'}

        # Convert to arrays
        X = np.array([[f.get(name, 0) for name in REGIME_FEATURE_NAMES]
                       for f in features_list])
        y = np.array(targets)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Train/test split (time-based, no shuffling)
        if train_end_loc is None:
            split = int(len(X) * 0.8)
        else:
            # Find the index in our features_list that corresponds to train_end_loc
            split = sum(1 for d in dates if spy_df.index.get_loc(d) < train_end_loc)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(X_test) < 50:
            logger.warning("Test set too small for reliable AUC estimation")
            return {'error': 'test set too small'}

        logger.info(f"Training regime ML: {len(X_train)} train, {len(X_test)} test")
        logger.info(f"Target distribution - train: {y_train.mean():.2%} positive, "
                     f"test: {y_test.mean():.2%} positive")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train RandomForest (shallow trees, large leaves -> less overfitting)
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        self.auc_score = auc
        self.is_active = auc >= self.config.min_auc

        # Feature importance
        importances = dict(zip(REGIME_FEATURE_NAMES, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

        metrics = {
            'auc': auc,
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'is_active': self.is_active,
            'top_features': {k: round(v, 4) for k, v in top_features}
        }

        if self.is_active:
            logger.info(f"Regime ML ACTIVE: AUC={auc:.3f}, Accuracy={accuracy:.1%}")
            logger.info(f"Top features: {[f'{k}={v:.3f}' for k, v in top_features]}")
            self._save_model()
        else:
            logger.warning(f"Regime ML DISABLED: AUC={auc:.3f} < {self.config.min_auc} threshold")
            self.model = None
            self.scaler = None

        # Save history
        self._save_history(metrics)

        return metrics

    def predict(
        self,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame],
        symbol_data: Dict[str, pd.DataFrame],
        date_loc: int,
        spy_date: pd.Timestamp = None
    ) -> Tuple[float, bool]:
        """Predict market direction probability.

        Returns:
            (probability_bullish 0-1, is_prediction_valid)
        """
        if not self.is_active or self.model is None:
            return 0.5, False

        feat = self._extractor.extract(spy_df, vix_df, symbol_data, date_loc, spy_date)
        if feat is None:
            return 0.5, False

        X = np.array([[feat.get(name, 0) for name in REGIME_FEATURE_NAMES]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0]
        bull_prob = proba[1] if len(proba) > 1 else proba[0]

        return float(bull_prob), True

    def load_model(self) -> bool:
        """Load a previously trained model."""
        try:
            import joblib
            if os.path.exists(self.config.model_path):
                saved = joblib.load(self.config.model_path)
                self.model = saved['model']
                self.scaler = saved['scaler']
                self.auc_score = saved.get('auc', 0)
                self.is_active = self.auc_score >= self.config.min_auc
                logger.info(f"Loaded regime ML model (AUC={self.auc_score:.3f}, active={self.is_active})")
                return True
        except Exception as e:
            logger.warning(f"Failed to load regime ML model: {e}")
        return False

    def _save_model(self):
        """Save trained model to disk."""
        try:
            import joblib
            os.makedirs(os.path.dirname(self.config.model_path) or '.', exist_ok=True)
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'auc': self.auc_score,
                'feature_names': REGIME_FEATURE_NAMES
            }, self.config.model_path)
            logger.info(f"Saved regime ML model to {self.config.model_path}")
        except Exception as e:
            logger.warning(f"Failed to save regime ML model: {e}")

    def _save_history(self, metrics: Dict):
        """Append training metrics to history file."""
        try:
            history = []
            if os.path.exists(self.config.history_path):
                with open(self.config.history_path, 'r') as f:
                    history = json.load(f)

            entry = {
                'timestamp': datetime.now().isoformat(),
                **{k: v for k, v in metrics.items() if k != 'top_features'},
                'top_features': metrics.get('top_features', {})
            }
            history.append(entry)

            os.makedirs(os.path.dirname(self.config.history_path) or '.', exist_ok=True)
            with open(self.config.history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save regime ML history: {e}")
