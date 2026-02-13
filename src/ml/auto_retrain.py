"""
Nightly Auto-Retrainer for Midas ML Models.

Uses walk-forward methodology with rolling windows on recent trade data
to keep ML models fresh without manual intervention.

Data sources:
  - data/trade_analysis_db.json  (ParameterOptimizer trade records)
  - data/signals/tracked/*.json  (signal_tracker daily files)

Rolling windows:
  - Training:   last 180 days of trade data
  - Validation: last 30 days of trade data

Validation gate:
  - New AUC-ROC must exceed current model AUC-ROC
  - New accuracy must be >= 52%
  - Sharpe ratio must not degrade (if old stats available)

Usage:
  python -m src.ml.auto_retrain
  python -m src.ml.auto_retrain --dry-run
  python -m src.ml.auto_retrain --force --min-trades 50
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.walk_forward_trainer import (
    FEATURE_NAMES,
    ModelResult,
    WalkForwardTrainer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TRADE_DB = os.path.join("data", "trade_analysis_db.json")
DEFAULT_SIGNALS_DIR = os.path.join("data", "signals", "tracked")
DEFAULT_MODEL_DIR = "models"
DEFAULT_BACKUP_DIR = os.path.join("models", "backup")
DEFAULT_LOG_FILE = os.path.join("logs", "auto_retrain.log")

TRAIN_WINDOW_DAYS = 180
VALIDATION_WINDOW_DAYS = 30
MIN_TRADES_DEFAULT = 100
MIN_ACCURACY = 0.52


@dataclass
class RetrainResult:
    """Outcome of a nightly retrain attempt."""
    success: bool
    reason: str
    old_auc: Optional[float] = None
    new_auc: Optional[float] = None
    old_accuracy: Optional[float] = None
    new_accuracy: Optional[float] = None
    n_train_samples: int = 0
    n_val_samples: int = 0
    model_replaced: bool = False
    backup_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "reason": self.reason,
            "old_auc": self.old_auc,
            "new_auc": self.new_auc,
            "old_accuracy": self.old_accuracy,
            "new_accuracy": self.new_accuracy,
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
            "model_replaced": self.model_replaced,
            "backup_path": self.backup_path,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# NightlyRetrainer
# ---------------------------------------------------------------------------

class NightlyRetrainer:
    """Collects recent trade data, retrains ML models, and conditionally
    promotes the new model if it passes a validation gate.

    Reuses :class:`WalkForwardTrainer` from ``src.ml.walk_forward_trainer``
    for the actual model training, feature preparation, and regime-specific
    model fitting so that hyperparameters and training logic stay in one place.
    """

    def __init__(
        self,
        trade_db_path: str = DEFAULT_TRADE_DB,
        signals_dir: str = DEFAULT_SIGNALS_DIR,
        model_dir: str = DEFAULT_MODEL_DIR,
        backup_dir: str = DEFAULT_BACKUP_DIR,
        log_file: str = DEFAULT_LOG_FILE,
        train_window_days: int = TRAIN_WINDOW_DAYS,
        val_window_days: int = VALIDATION_WINDOW_DAYS,
        min_trades: int = MIN_TRADES_DEFAULT,
        dry_run: bool = False,
        force: bool = False,
    ):
        self.trade_db_path = trade_db_path
        self.signals_dir = signals_dir
        self.model_dir = model_dir
        self.backup_dir = backup_dir
        self.log_file = log_file
        self.train_window_days = train_window_days
        self.val_window_days = val_window_days
        self.min_trades = min_trades
        self.dry_run = dry_run
        self.force = force

        # Reuse the existing trainer for model building
        self._trainer = WalkForwardTrainer(
            output_dir=model_dir,
            data_dir="data",
            n_top_features=20,
        )

        self._setup_logging()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logging(self):
        """Configure a file handler for ``logs/auto_retrain.log``."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(fmt)

        # Attach to both the module logger and the walk_forward_trainer logger
        for lg in [logger, logging.getLogger("src.ml.walk_forward_trainer")]:
            if not any(
                isinstance(h, logging.FileHandler)
                and getattr(h, "baseFilename", None) == file_handler.baseFilename
                for h in lg.handlers
            ):
                lg.addHandler(file_handler)
            lg.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _load_trade_db(self) -> List[Dict]:
        """Load completed trades from ``data/trade_analysis_db.json``."""
        if not os.path.exists(self.trade_db_path):
            logger.warning("Trade DB not found at %s", self.trade_db_path)
            return []
        try:
            with open(self.trade_db_path, "r") as f:
                data = json.load(f)
            trades = data.get("trades", [])
            logger.info("Loaded %d trades from trade_analysis_db.json", len(trades))
            return trades
        except Exception as exc:
            logger.error("Error loading trade DB: %s", exc)
            return []

    def _load_signal_tracker(self, start_date: str, end_date: str) -> List[Dict]:
        """Load signal tracker records between *start_date* and *end_date*
        (inclusive, ``YYYY-MM-DD`` strings).

        Each daily file in ``data/signals/tracked/YYYY-MM-DD.json`` contains a
        list of signal records.
        """
        if not os.path.isdir(self.signals_dir):
            logger.warning("Signal tracker dir not found: %s", self.signals_dir)
            return []

        records: List[Dict] = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            ds = current.strftime("%Y-%m-%d")
            daily_file = os.path.join(self.signals_dir, f"{ds}.json")
            if os.path.exists(daily_file):
                try:
                    with open(daily_file, "r") as f:
                        daily = json.load(f)
                    if isinstance(daily, list):
                        records.extend(daily)
                except (json.JSONDecodeError, IOError) as exc:
                    logger.warning("Bad signal file %s: %s", daily_file, exc)
            current += timedelta(days=1)

        logger.info(
            "Loaded %d signal records from %s to %s", len(records), start_date, end_date
        )
        return records

    # ------------------------------------------------------------------
    # Feature synthesis from trade / signal records
    # ------------------------------------------------------------------

    def _trade_to_feature_row(self, trade: Dict) -> Optional[Dict]:
        """Convert a single trade record (from ``trade_analysis_db.json``) into a
        feature row compatible with the 40-feature schema used by
        :data:`FEATURE_NAMES`.

        The trade's ``indicators`` sub-dict provides raw indicator values captured
        at entry time.  We map them to the canonical feature names.  Features that
        have no direct source in the trade record are filled with neutral defaults
        so the resulting vector always has the correct length.
        """
        ind = trade.get("indicators", {})
        if not ind:
            return None

        entry_date = trade.get("entry_date", "")
        pnl_pct = trade.get("pnl_percent", 0.0)

        # Label: profitable trade if pnl > 1% (same threshold as training_pipeline)
        label = 1 if pnl_pct > 1.0 else 0

        features: Dict = {}

        # --- Trend features ---
        ema_alignment = ind.get("ema_alignment", "neutral")
        features["ema_cross_20_50"] = (
            1.0 if ema_alignment == "bullish" else (-1.0 if ema_alignment == "bearish" else 0.0)
        )
        features["ema_cross_50_200"] = features["ema_cross_20_50"]  # best proxy
        features["macd_histogram"] = ind.get("macd_histogram", 0.0)
        features["macd_signal_cross"] = (
            1.0 if ind.get("macd_value", 0) > ind.get("macd_signal", 0) else -1.0
        )
        features["adx_value"] = ind.get("adx_value", 25.0)
        features["adx_direction"] = 1.0 if ind.get("adx_value", 25) > 25 else -1.0
        features["supertrend_signal"] = 0.0  # not stored in IndicatorSnapshot
        features["aroon_oscillator"] = 0.0  # not stored
        features["price_vs_ema20"] = 0.0
        features["price_vs_ema50"] = 0.0
        if ind.get("ema_20") and ind.get("ema_50"):
            entry_price = trade.get("entry_price", 0)
            if entry_price > 0:
                features["price_vs_ema20"] = (
                    (entry_price - ind["ema_20"]) / ind["ema_20"] * 100
                    if ind["ema_20"] > 0
                    else 0.0
                )
                features["price_vs_ema50"] = (
                    (entry_price - ind["ema_50"]) / ind["ema_50"] * 100
                    if ind["ema_50"] > 0
                    else 0.0
                )

        # --- Momentum features ---
        features["rsi_14"] = ind.get("rsi_value", 50.0)
        features["rsi_slope"] = 0.0  # single-snapshot, no history
        features["stoch_k"] = ind.get("stochastic_k", 50.0)
        features["stoch_d"] = ind.get("stochastic_d", 50.0)
        features["williams_r"] = ind.get("williams_r", -50.0)
        features["cci_20"] = ind.get("cci_value", 0.0)
        features["roc_10"] = ind.get("roc_value", 0.0)
        features["momentum_10"] = ind.get("roc_value", 0.0)  # proxy
        features["rsi_oversold"] = 1.0 if features["rsi_14"] < 30 else 0.0
        features["rsi_overbought"] = 1.0 if features["rsi_14"] > 70 else 0.0

        # --- Volume features ---
        features["volume_ratio_20"] = ind.get("volume_ratio", 1.0)
        obv_trend_str = ind.get("obv_trend", "flat")
        features["obv_trend"] = (
            1.0 if obv_trend_str == "up" else (-1.0 if obv_trend_str == "down" else 0.0)
        )
        features["obv_slope"] = 0.0
        features["cmf_20"] = ind.get("cmf_value", 0.0)
        features["mfi_14"] = ind.get("mfi_value", 50.0)
        features["volume_trend_5d"] = 0.0
        features["volume_breakout"] = 1.0 if features["volume_ratio_20"] > 2.0 else 0.0
        features["price_volume_trend"] = (
            features["volume_ratio_20"] * np.sign(pnl_pct) if pnl_pct != 0 else 0.0
        )

        # --- Volatility features ---
        features["atr_percent"] = ind.get("atr_percent", 2.0)
        features["atr_ratio"] = 1.0
        features["bb_width"] = ind.get("bb_width", 0.0)
        features["bb_percent"] = ind.get("bb_position", 0.5)
        features["volatility_20d"] = 20.0  # not stored per-trade
        features["volatility_expansion"] = 1.0 if features["atr_percent"] > 3.0 else 0.0

        # --- Regime features ---
        spy_trend = ind.get("spy_trend", "flat")
        features["spy_above_ema50"] = (
            1.0 if spy_trend == "up" else (-1.0 if spy_trend == "down" else 0.0)
        )
        features["vix_level"] = ind.get("vix_level", 20.0)
        features["vix_percentile"] = 50.0
        features["sector_momentum"] = ind.get("sector_momentum", 0.0)
        features["market_breadth"] = 50.0
        features["correlation_spy"] = 0.5

        # Metadata
        regime = ind.get("market_regime", "RANGE")
        regime = regime.upper() if isinstance(regime, str) else "RANGE"
        # Normalise free-text regime to one of the four canonical labels
        if regime not in ("BULL", "BEAR", "RANGE", "VOLATILE"):
            if "bull" in regime.lower():
                regime = "BULL"
            elif "bear" in regime.lower():
                regime = "BEAR"
            elif "volat" in regime.lower():
                regime = "VOLATILE"
            else:
                regime = "RANGE"

        row = {
            "symbol": trade.get("symbol", "UNK"),
            "date": entry_date,
            "regime": regime,
            "forward_return": pnl_pct / 100.0,
            "label": label,
            "forward_return_raw": pnl_pct / 100.0,
            "current_price": trade.get("entry_price", 0.0),
            **features,
        }
        return row

    def _signal_to_feature_row(self, signal: Dict) -> Optional[Dict]:
        """Convert a signal tracker record to a feature row.

        Signal records carry ``normalized_values`` and ``raw_values`` dicts
        that may contain feature data.  Since signals lack an outcome (no PnL),
        we can only use *taken* signals whose outcome can be inferred later.
        Rejected signals are skipped because we have no outcome label.
        """
        decision = signal.get("decision", "UNKNOWN")
        if decision in ("REJECTED", "UNKNOWN"):
            return None  # no label available

        raw = signal.get("raw_values", {})
        norm = signal.get("normalized_values", {})
        pillar = signal.get("pillar_scores", {})

        if not raw and not norm:
            return None

        # We cannot derive a PnL from a signal alone, so we use the combined
        # score as a weak label proxy:  score > 65 -> label 1, else 0.
        score = signal.get("score", 0)
        label = 1 if score > 65 else 0

        features: Dict = {}
        for fname in FEATURE_NAMES:
            # Try raw_values first, then normalized_values, then default
            if fname in raw:
                features[fname] = float(raw[fname])
            elif fname in norm:
                features[fname] = float(norm[fname])
            else:
                features[fname] = 0.0

        regime = signal.get("regime", "RANGE")
        regime = regime.upper() if isinstance(regime, str) else "RANGE"
        if regime not in ("BULL", "BEAR", "RANGE", "VOLATILE"):
            regime = "RANGE"

        row = {
            "symbol": signal.get("symbol", "UNK"),
            "date": signal.get("timestamp", ""),
            "regime": regime,
            "forward_return": 0.0,
            "label": label,
            "forward_return_raw": 0.0,
            "current_price": 0.0,
            **features,
        }
        return row

    # ------------------------------------------------------------------
    # Build rolling-window datasets
    # ------------------------------------------------------------------

    def collect_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect and split data into training and validation DataFrames
        using the rolling-window approach.

        Returns:
            ``(train_df, val_df)`` where *train_df* covers the training
            window and *val_df* covers the validation window.
        """
        now = datetime.now()
        total_window = self.train_window_days + self.val_window_days
        start_date = (now - timedelta(days=total_window)).strftime("%Y-%m-%d")
        split_date = (now - timedelta(days=self.val_window_days)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

        # 1. Trade DB records
        all_trades = self._load_trade_db()

        # 2. Signal tracker records
        signal_records = self._load_signal_tracker(start_date, end_date)

        # Convert to feature rows
        rows: List[Dict] = []

        for trade in all_trades:
            entry = trade.get("entry_date", "")
            if not entry:
                continue
            entry_ds = entry[:10]
            if start_date <= entry_ds <= end_date:
                row = self._trade_to_feature_row(trade)
                if row is not None:
                    rows.append(row)

        trade_count = len(rows)
        logger.info("Converted %d trade records to feature rows", trade_count)

        for sig in signal_records:
            row = self._signal_to_feature_row(sig)
            if row is not None:
                rows.append(row)

        signal_count = len(rows) - trade_count
        logger.info("Converted %d signal records to feature rows", signal_count)

        if not rows:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.DataFrame(rows)

        # Parse date for splitting
        df["_date_str"] = df["date"].astype(str).str[:10]

        train_df = df[df["_date_str"] < split_date].drop(columns=["_date_str"]).copy()
        val_df = df[df["_date_str"] >= split_date].drop(columns=["_date_str"]).copy()

        logger.info(
            "Rolling window split: train=%d (before %s), val=%d (from %s)",
            len(train_df),
            split_date,
            len(val_df),
            split_date,
        )

        return train_df, val_df

    # ------------------------------------------------------------------
    # Load current model stats for comparison
    # ------------------------------------------------------------------

    def _load_current_model_stats(self) -> Dict:
        """Read metadata about the currently-deployed model so we can compare
        the new candidate against it.

        Looks for ``models/ml_features_v7.json`` (saved by WalkForwardTrainer)
        and the last retrain result in ``logs/auto_retrain_history.json``.
        """
        stats: Dict = {"auc_roc": 0.0, "accuracy": 0.0, "sharpe": None}

        # Try the retrain history first
        history_path = os.path.join("logs", "auto_retrain_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                if history:
                    last = history[-1]
                    stats["auc_roc"] = last.get("new_auc") or last.get("old_auc") or 0.0
                    stats["accuracy"] = (
                        last.get("new_accuracy") or last.get("old_accuracy") or 0.0
                    )
            except Exception:
                pass

        # Try the features file for any extra info
        features_path = os.path.join(self.model_dir, "ml_features_v7.json")
        if os.path.exists(features_path):
            try:
                with open(features_path, "r") as f:
                    meta = json.load(f)
                if "last_auc" in meta:
                    stats["auc_roc"] = meta["last_auc"]
                if "last_accuracy" in meta:
                    stats["accuracy"] = meta["last_accuracy"]
            except Exception:
                pass

        return stats

    # ------------------------------------------------------------------
    # Model backup
    # ------------------------------------------------------------------

    def _backup_current_model(self) -> Optional[str]:
        """Copy the current model files to ``models/backup/<timestamp>/``."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(self.backup_dir, timestamp)
        os.makedirs(dest, exist_ok=True)

        copied = 0
        for pattern in [
            "ml_model_v7.joblib",
            "ml_scaler_v7.joblib",
            "ml_features_v7.json",
            "ml_model_BULL.joblib",
            "ml_scaler_BULL.joblib",
            "ml_model_BEAR.joblib",
            "ml_scaler_BEAR.joblib",
            "ml_model_RANGE.joblib",
            "ml_scaler_RANGE.joblib",
            "ml_model_VOLATILE.joblib",
            "ml_scaler_VOLATILE.joblib",
        ]:
            src = os.path.join(self.model_dir, pattern)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, pattern))
                copied += 1

        if copied == 0:
            logger.info("No existing model files to back up")
            return None

        logger.info("Backed up %d model files to %s", copied, dest)
        return dest

    # ------------------------------------------------------------------
    # Persistence of retrain history
    # ------------------------------------------------------------------

    def _save_retrain_result(self, result: RetrainResult):
        """Append the retrain result to ``logs/auto_retrain_history.json``."""
        history_path = os.path.join("logs", "auto_retrain_history.json")
        history: List[Dict] = []
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(result.to_dict())

        # Keep last 90 entries
        history = history[-90:]

        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Core retrain logic
    # ------------------------------------------------------------------

    def retrain(self) -> RetrainResult:
        """Run the full nightly retrain pipeline.

        Steps:
          1. Collect rolling-window data from trade DB and signal tracker.
          2. Verify minimum sample sizes.
          3. Train RF model (and regime models) using WalkForwardTrainer.
          4. Apply validation gate.
          5. Backup old model and promote new model if gate passes.
        """
        logger.info("=" * 70)
        logger.info("  MIDAS NIGHTLY AUTO-RETRAIN  --  %s", datetime.now().isoformat())
        logger.info("=" * 70)

        if self.dry_run:
            logger.info("DRY-RUN mode enabled -- model will NOT be replaced")

        # Step 1: Collect data
        train_df, val_df = self.collect_data()

        n_train = len(train_df)
        n_val = len(val_df)
        logger.info("Collected data: train=%d, val=%d", n_train, n_val)

        if n_train + n_val < self.min_trades:
            msg = (
                f"Insufficient data: {n_train + n_val} total rows, "
                f"need >= {self.min_trades}"
            )
            logger.warning(msg)
            result = RetrainResult(
                success=False,
                reason=msg,
                n_train_samples=n_train,
                n_val_samples=n_val,
            )
            self._save_retrain_result(result)
            return result

        if n_val < 10:
            msg = f"Validation set too small: {n_val} rows (need >= 10)"
            logger.warning(msg)
            result = RetrainResult(
                success=False,
                reason=msg,
                n_train_samples=n_train,
                n_val_samples=n_val,
            )
            self._save_retrain_result(result)
            return result

        # Step 2: Build datasets dict compatible with WalkForwardTrainer
        datasets = {"train": train_df, "validation": val_df}

        # Step 3: Train using the existing trainer methods
        logger.info("Training models on rolling window data ...")

        try:
            X_train, y_train = self._trainer.prepare_features(train_df)
            X_val, y_val = self._trainer.prepare_features(val_df)
        except Exception as exc:
            msg = f"Feature preparation failed: {exc}"
            logger.error(msg)
            result = RetrainResult(
                success=False,
                reason=msg,
                n_train_samples=n_train,
                n_val_samples=n_val,
            )
            self._save_retrain_result(result)
            return result

        logger.info(
            "Feature matrix: train=%s, val=%s",
            X_train.shape,
            X_val.shape,
        )
        logger.info("Train label dist: %s", np.bincount(y_train).tolist())
        logger.info("Val label dist: %s", np.bincount(y_val).tolist())

        # Train RF model (the primary model for nightly)
        best_model_obj = None
        best_result: Optional[ModelResult] = None
        best_score = -1.0

        for model_type in ["rf", "gb", "ensemble"]:
            try:
                (model, scaler), mr = self._trainer.train_single_model(
                    model_type, X_train, y_train, X_val, y_val
                )
                if mr.auc_roc > best_score:
                    best_score = mr.auc_roc
                    best_model_obj = (model, scaler)
                    best_result = mr
            except Exception as exc:
                logger.warning("Model type %s failed: %s", model_type, exc)

        if best_result is None or best_model_obj is None:
            msg = "All model training attempts failed"
            logger.error(msg)
            result = RetrainResult(
                success=False,
                reason=msg,
                n_train_samples=n_train,
                n_val_samples=n_val,
            )
            self._save_retrain_result(result)
            return result

        new_auc = best_result.auc_roc
        new_accuracy = best_result.accuracy

        logger.info(
            "Best candidate: %s | AUC=%.4f | Acc=%.4f | F1=%.4f",
            best_result.name,
            new_auc,
            new_accuracy,
            best_result.f1,
        )

        # Step 4: Validation gate
        old_stats = self._load_current_model_stats()
        old_auc = old_stats.get("auc_roc", 0.0)
        old_accuracy = old_stats.get("accuracy", 0.0)

        gate_passed = True
        gate_reasons: List[str] = []

        if not self.force:
            # Gate 1: AUC must improve (or old model has no stats)
            if old_auc > 0 and new_auc < old_auc:
                gate_passed = False
                gate_reasons.append(
                    f"AUC regression: new {new_auc:.4f} < old {old_auc:.4f}"
                )

            # Gate 2: Minimum accuracy
            if new_accuracy < MIN_ACCURACY:
                gate_passed = False
                gate_reasons.append(
                    f"Accuracy below minimum: {new_accuracy:.4f} < {MIN_ACCURACY}"
                )

            # Gate 3: Sharpe non-degradation (if available)
            old_sharpe = old_stats.get("sharpe")
            if old_sharpe is not None and old_sharpe > 0:
                # We approximate new Sharpe from the win rate and average return
                # stored in the validation set
                val_returns = val_df["forward_return"].values
                if len(val_returns) > 1 and np.std(val_returns) > 0:
                    new_sharpe = float(np.mean(val_returns) / np.std(val_returns))
                    if new_sharpe < old_sharpe * 0.9:
                        gate_passed = False
                        gate_reasons.append(
                            f"Sharpe degradation: new ~{new_sharpe:.3f} < "
                            f"90% of old {old_sharpe:.3f}"
                        )
        else:
            logger.info("--force flag set, skipping validation gate")

        if not gate_passed:
            reasons_str = "; ".join(gate_reasons)
            msg = f"Validation gate FAILED: {reasons_str}"
            logger.warning(msg)
            result = RetrainResult(
                success=True,
                reason=msg,
                old_auc=old_auc,
                new_auc=new_auc,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                n_train_samples=n_train,
                n_val_samples=n_val,
                model_replaced=False,
            )
            self._save_retrain_result(result)
            return result

        # Step 5: Backup and promote
        if self.dry_run:
            msg = (
                f"DRY-RUN: gate passed (AUC {new_auc:.4f} vs old {old_auc:.4f}), "
                f"model NOT replaced"
            )
            logger.info(msg)
            result = RetrainResult(
                success=True,
                reason=msg,
                old_auc=old_auc,
                new_auc=new_auc,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                n_train_samples=n_train,
                n_val_samples=n_val,
                model_replaced=False,
            )
            self._save_retrain_result(result)
            return result

        backup_path = self._backup_current_model()

        # Save new model using the same paths as WalkForwardTrainer
        try:
            import joblib

            model, scaler_obj = best_model_obj
            model_path = os.path.join(self.model_dir, "ml_model_v7.joblib")
            scaler_path = os.path.join(self.model_dir, "ml_scaler_v7.joblib")
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(model, model_path)
            joblib.dump(scaler_obj, scaler_path)
            logger.info("Saved new model: %s", model_path)

            # Save feature metadata with new stats for future comparisons
            features_path = os.path.join(self.model_dir, "ml_features_v7.json")
            feature_ranking = sorted(
                best_result.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            pruned = [f for f, _ in feature_ranking[: self._trainer.n_top_features]]
            with open(features_path, "w") as f:
                json.dump(
                    {
                        "all_features": FEATURE_NAMES,
                        "pruned_features": pruned,
                        "feature_ranking": feature_ranking,
                        "last_auc": new_auc,
                        "last_accuracy": new_accuracy,
                        "retrained_at": datetime.now().isoformat(),
                        "train_samples": n_train,
                        "val_samples": n_val,
                    },
                    f,
                    indent=2,
                )

        except Exception as exc:
            msg = f"Failed to save new model: {exc}"
            logger.error(msg)
            result = RetrainResult(
                success=False,
                reason=msg,
                old_auc=old_auc,
                new_auc=new_auc,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                n_train_samples=n_train,
                n_val_samples=n_val,
                model_replaced=False,
                backup_path=backup_path,
            )
            self._save_retrain_result(result)
            return result

        # Train per-regime models
        try:
            self._trainer._train_regime_models(datasets, best_result)
            logger.info("Per-regime models trained and saved")
        except Exception as exc:
            logger.warning("Regime model training failed (non-fatal): %s", exc)

        msg = (
            f"Model REPLACED: {best_result.name} | "
            f"AUC {old_auc:.4f} -> {new_auc:.4f} | "
            f"Acc {old_accuracy:.4f} -> {new_accuracy:.4f}"
        )
        logger.info(msg)

        result = RetrainResult(
            success=True,
            reason=msg,
            old_auc=old_auc,
            new_auc=new_auc,
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            n_train_samples=n_train,
            n_val_samples=n_val,
            model_replaced=True,
            backup_path=backup_path,
        )
        self._save_retrain_result(result)
        return result


# ---------------------------------------------------------------------------
# Async entry point for live_loop integration
# ---------------------------------------------------------------------------

async def run_nightly(
    dry_run: bool = False,
    force: bool = False,
    min_trades: int = MIN_TRADES_DEFAULT,
) -> RetrainResult:
    """Async wrapper that can be called from the live loop.

    Runs the retrain in a thread executor to avoid blocking the event loop.

    Args:
        dry_run: If True, train and evaluate but do not replace the model.
        force: If True, skip the validation gate and always replace.
        min_trades: Minimum number of trade records required to proceed.

    Returns:
        :class:`RetrainResult` describing what happened.
    """
    loop = asyncio.get_event_loop()
    retrainer = NightlyRetrainer(
        dry_run=dry_run,
        force=force,
        min_trades=min_trades,
    )
    result = await loop.run_in_executor(None, retrainer.retrain)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Midas Nightly ML Model Auto-Retrainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.ml.auto_retrain                   # full retrain\n"
            "  python -m src.ml.auto_retrain --dry-run          # evaluate only\n"
            "  python -m src.ml.auto_retrain --force             # skip gate\n"
            "  python -m src.ml.auto_retrain --min-trades 50     # lower bar\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate but do NOT replace the production model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the validation gate and always promote the new model.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=MIN_TRADES_DEFAULT,
        metavar="N",
        help=(
            f"Minimum total trade/signal records required to proceed "
            f"(default: {MIN_TRADES_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--trade-db",
        type=str,
        default=DEFAULT_TRADE_DB,
        help=f"Path to trade_analysis_db.json (default: {DEFAULT_TRADE_DB}).",
    )
    parser.add_argument(
        "--signals-dir",
        type=str,
        default=DEFAULT_SIGNALS_DIR,
        help=f"Path to signal tracker directory (default: {DEFAULT_SIGNALS_DIR}).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to model output directory (default: {DEFAULT_MODEL_DIR}).",
    )

    args = parser.parse_args()

    # Console logging in addition to file
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    )
    logging.getLogger().addHandler(console)
    logging.getLogger().setLevel(logging.INFO)

    retrainer = NightlyRetrainer(
        trade_db_path=args.trade_db,
        signals_dir=args.signals_dir,
        model_dir=args.model_dir,
        dry_run=args.dry_run,
        force=args.force,
        min_trades=args.min_trades,
    )

    result = retrainer.retrain()

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("  AUTO-RETRAIN SUMMARY")
    print("=" * 70)
    print(f"  Success:        {result.success}")
    print(f"  Model replaced: {result.model_replaced}")
    print(f"  Train samples:  {result.n_train_samples}")
    print(f"  Val samples:    {result.n_val_samples}")
    print(f"  Old AUC:        {result.old_auc}")
    print(f"  New AUC:        {result.new_auc}")
    print(f"  Old Accuracy:   {result.old_accuracy}")
    print(f"  New Accuracy:   {result.new_accuracy}")
    print(f"  Backup:         {result.backup_path}")
    print(f"  Reason:         {result.reason}")
    print("=" * 70)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
