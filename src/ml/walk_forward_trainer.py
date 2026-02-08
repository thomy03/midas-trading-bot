"""
Walk-Forward ML Trainer for Midas V7.

Trains and evaluates ML models using walk-forward methodology:
- Train: 2015-2021 (7 years)
- Validation: 2022-2023 (2 years) - hyperparameter tuning
- Test: 2024-2025 (1 year) - final untouchable evaluation

Models tested:
- RandomForest (existing in ml_pillar.py)
- GradientBoosting (existing in adaptive_ml_gate.py)
- Ensemble (RF + GB vote)

Per-regime models: Trains specialized models for each regime.

Saves best model to models/ml_model_v7.joblib
"""

import numpy as np
import pandas as pd
import logging
import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from sklearn.model_selection import GridSearchCV
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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


@dataclass
class ModelResult:
    """Results from training a single model."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    train_accuracy: float
    feature_importances: Dict[str, float]
    best_params: Dict
    confusion_matrix: list  # 2x2 list
    regime: str = 'ALL'  # Which regime this model is for


@dataclass
class WalkForwardResult:
    """Complete walk-forward training results."""
    best_model_name: str
    best_model_regime: str  # 'ALL' or specific regime
    overall_accuracy: float
    overall_auc: float
    models: List[ModelResult]
    regime_models: Dict[str, ModelResult]
    feature_importance_ranking: List[Tuple[str, float]]
    pruned_features: List[str]  # Features kept after pruning
    report: str


class WalkForwardTrainer:
    """Trains ML models using walk-forward methodology.

    The trainer loads pre-split datasets (train/validation/test), trains
    multiple model architectures (RandomForest, GradientBoosting, Ensemble),
    selects the best by AUC-ROC, trains per-regime specialists, prunes
    features to the top N, and persists all artifacts to disk.
    """

    def __init__(
        self,
        output_dir: str = 'models',
        data_dir: str = 'data',
        n_top_features: int = 20
    ):
        """Initialize the walk-forward trainer.

        Args:
            output_dir: Directory where trained models and reports are saved.
            data_dir: Directory containing parquet training data files.
            n_top_features: Number of features to retain after pruning.
        """
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.n_top_features = n_top_features
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load training data from parquet files.

        Looks for ml_train_data.parquet, ml_validation_data.parquet, and
        ml_test_data.parquet in self.data_dir.

        Returns:
            Dictionary mapping period name ('train', 'validation', 'test')
            to the corresponding DataFrame. Missing periods are omitted.
        """
        datasets = {}
        for period in ['train', 'validation', 'test']:
            path = os.path.join(self.data_dir, f'ml_{period}_data.parquet')
            if os.path.exists(path):
                datasets[period] = pd.read_parquet(path)
                logger.info(f"Loaded {period}: {len(datasets[period])} records")
            else:
                logger.warning(f"Missing {period} data at {path}")
        return datasets

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and label vector y from DataFrame.

        Args:
            df: DataFrame containing feature columns and a 'label' column.
            feature_names: Optional list of feature column names to use.
                Defaults to the global FEATURE_NAMES list.

        Returns:
            Tuple of (X, y) where X is a 2D float64 array and y is an
            integer label array. Missing values and infinities are replaced
            with 0.
        """
        features = feature_names or FEATURE_NAMES
        # Filter to available features
        available = [f for f in features if f in df.columns]

        X = df[available].fillna(0).values.astype(np.float64)
        y = df['label'].values.astype(int)

        # Replace any remaining inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y

    def train_single_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        regime: str = 'ALL'
    ) -> Tuple[object, ModelResult]:
        """Train a single model and evaluate on validation set.

        Args:
            model_type: One of 'rf' (RandomForest), 'gb' (GradientBoosting),
                or 'ensemble' (soft-voting RF + GB).
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Validation feature matrix.
            y_val: Validation labels.
            regime: Market regime label for this model ('ALL' or a specific
                regime like 'BULL', 'BEAR', etc.).

        Returns:
            Tuple of ((model, scaler), ModelResult) where model is the fitted
            sklearn estimator and scaler is the fitted StandardScaler.

        Raises:
            RuntimeError: If sklearn is not installed.
            ValueError: If model_type is not recognized.
        """
        if not ML_AVAILABLE:
            raise RuntimeError("sklearn not available")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Create model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                min_samples_split=20,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            name = 'RandomForest'
        elif model_type == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=10,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            name = 'GradientBoosting'
        elif model_type == 'ensemble':
            rf = RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_leaf=10,
                random_state=42, n_jobs=-1, class_weight='balanced'
            )
            gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                learning_rate=0.05, subsample=0.8, random_state=42
            )
            model = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb)],
                voting='soft'
            )
            name = 'Ensemble_RF_GB'
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        model.fit(X_train_scaled, y_train)

        # Evaluate on train and validation sets
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        val_proba = (
            model.predict_proba(X_val_scaled)[:, 1]
            if hasattr(model, 'predict_proba')
            else val_pred.astype(float)
        )

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_prec = precision_score(y_val, val_pred, zero_division=0)
        val_recall = recall_score(y_val, val_pred, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)

        try:
            val_auc = roc_auc_score(y_val, val_proba)
        except ValueError:
            val_auc = 0.5

        cm = confusion_matrix(y_val, val_pred).tolist()

        # Feature importances
        importances = {}
        if hasattr(model, 'feature_importances_'):
            for fname, imp in zip(FEATURE_NAMES[:X_train.shape[1]], model.feature_importances_):
                importances[fname] = float(imp)
        elif hasattr(model, 'estimators_'):
            # VotingClassifier - average importances from sub-models
            for est_name, est in model.named_estimators_.items():
                if hasattr(est, 'feature_importances_'):
                    for fname, imp in zip(FEATURE_NAMES[:X_train.shape[1]], est.feature_importances_):
                        importances[fname] = importances.get(fname, 0) + float(imp)
            n_est = len(model.named_estimators_)
            importances = {k: v / n_est for k, v in importances.items()}

        result = ModelResult(
            name=name,
            accuracy=val_acc,
            precision=val_prec,
            recall=val_recall,
            f1=val_f1,
            auc_roc=val_auc,
            train_accuracy=train_acc,
            feature_importances=importances,
            best_params={},
            confusion_matrix=cm,
            regime=regime
        )

        logger.info(
            f"[{name}] regime={regime} | Val acc={val_acc:.3f} | "
            f"AUC={val_auc:.3f} | F1={val_f1:.3f} | "
            f"Train acc={train_acc:.3f}"
        )

        return (model, scaler), result

    def train_all_models(
        self,
        datasets: Dict[str, pd.DataFrame]
    ) -> WalkForwardResult:
        """Train all model types and select the best.

        Trains RandomForest, GradientBoosting, and an Ensemble on the full
        training set, evaluates on validation, selects the best by AUC-ROC,
        then trains per-regime specialist models using the same architecture
        as the best overall model. Finally, evaluates on the held-out test
        set if available.

        Args:
            datasets: Dictionary with 'train', 'validation', and optionally
                'test' DataFrames as returned by load_data().

        Returns:
            WalkForwardResult containing the best model metadata, all
            per-model results, regime models, feature rankings, and a
            human-readable report.

        Raises:
            ValueError: If train or validation datasets are missing.
            RuntimeError: If all model training attempts fail.
        """
        if 'train' not in datasets or 'validation' not in datasets:
            raise ValueError("Need train and validation datasets")

        X_train, y_train = self.prepare_features(datasets['train'])
        X_val, y_val = self.prepare_features(datasets['validation'])

        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Validation data: {X_val.shape[0]} samples")
        logger.info(f"Train label distribution: {np.bincount(y_train)}")
        logger.info(f"Val label distribution: {np.bincount(y_val)}")

        # Train each model type
        all_results = []
        best_model = None
        best_scaler = None
        best_score = -1
        best_result = None

        for model_type in ['rf', 'gb', 'ensemble']:
            try:
                (model, scaler), result = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val
                )
                all_results.append(result)

                # Select best by AUC-ROC (most robust metric)
                if result.auc_roc > best_score:
                    best_score = result.auc_roc
                    best_model = model
                    best_scaler = scaler
                    best_result = result
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")

        if best_model is None:
            raise RuntimeError("All model training failed")

        # Feature importance analysis from best model
        feature_ranking = sorted(
            best_result.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Prune features: keep top N
        pruned_features = [f for f, _ in feature_ranking[:self.n_top_features]]
        logger.info(f"Top {self.n_top_features} features: {pruned_features}")

        # Train per-regime models
        regime_models = self._train_regime_models(datasets, best_result)

        # Save best overall model
        model_path = os.path.join(self.output_dir, 'ml_model_v7.joblib')
        scaler_path = os.path.join(self.output_dir, 'ml_scaler_v7.joblib')
        joblib.dump(best_model, model_path)
        joblib.dump(best_scaler, scaler_path)
        logger.info(f"Saved best model ({best_result.name}): {model_path}")

        # Save feature list
        features_path = os.path.join(self.output_dir, 'ml_features_v7.json')
        with open(features_path, 'w') as f:
            json.dump({
                'all_features': FEATURE_NAMES,
                'pruned_features': pruned_features,
                'feature_ranking': feature_ranking,
            }, f, indent=2)

        # Test set evaluation (if available)
        test_report = self._evaluate_test_set(datasets, best_model, best_scaler, best_result)

        # Generate report
        report = self._generate_report(
            best_result, all_results, regime_models, feature_ranking, test_report
        )

        return WalkForwardResult(
            best_model_name=best_result.name,
            best_model_regime='ALL',
            overall_accuracy=best_result.accuracy,
            overall_auc=best_result.auc_roc,
            models=all_results,
            regime_models=regime_models,
            feature_importance_ranking=feature_ranking,
            pruned_features=pruned_features,
            report=report
        )

    def _train_regime_models(
        self,
        datasets: Dict[str, pd.DataFrame],
        best_result: ModelResult
    ) -> Dict[str, ModelResult]:
        """Train specialized models for each market regime.

        Args:
            datasets: The full datasets dictionary with 'train' and 'validation'.
            best_result: The best overall ModelResult, used to determine which
                model architecture to use for regime-specific training.

        Returns:
            Dictionary mapping regime name to its ModelResult.
        """
        regime_models = {}

        # Determine best model type string from the result name
        best_type = 'rf'
        if best_result.name == 'GradientBoosting':
            best_type = 'gb'
        elif best_result.name == 'Ensemble_RF_GB':
            best_type = 'ensemble'

        for regime in ['BULL', 'BEAR', 'RANGE', 'VOLATILE']:
            train_regime = datasets['train'][datasets['train']['regime'] == regime]
            val_regime = datasets['validation'][datasets['validation']['regime'] == regime]

            if len(train_regime) < 50 or len(val_regime) < 20:
                logger.warning(
                    f"Not enough data for regime {regime}: "
                    f"train={len(train_regime)}, val={len(val_regime)}"
                )
                continue

            X_r_train, y_r_train = self.prepare_features(train_regime)
            X_r_val, y_r_val = self.prepare_features(val_regime)

            try:
                (regime_model, regime_scaler), regime_result = self.train_single_model(
                    best_type, X_r_train, y_r_train, X_r_val, y_r_val, regime=regime
                )
                regime_models[regime] = regime_result

                # Save regime-specific model
                model_path = os.path.join(self.output_dir, f'ml_model_{regime}.joblib')
                scaler_path = os.path.join(self.output_dir, f'ml_scaler_{regime}.joblib')
                joblib.dump(regime_model, model_path)
                joblib.dump(regime_scaler, scaler_path)
                logger.info(f"Saved regime model: {model_path}")

            except Exception as e:
                logger.warning(f"Failed to train regime model for {regime}: {e}")

        return regime_models

    def _evaluate_test_set(
        self,
        datasets: Dict[str, pd.DataFrame],
        best_model: object,
        best_scaler: StandardScaler,
        best_result: ModelResult
    ) -> str:
        """Evaluate the best model on the held-out test set.

        Args:
            datasets: The full datasets dictionary; 'test' key is optional.
            best_model: The fitted best sklearn estimator.
            best_scaler: The fitted StandardScaler for the best model.
            best_result: The best model's validation ModelResult (used
                to compute the retention ratio).

        Returns:
            A formatted string containing test-set metrics, or an empty
            string if no test data is available.
        """
        if 'test' not in datasets or len(datasets['test']) == 0:
            return ""

        X_test, y_test = self.prepare_features(datasets['test'])
        X_test_scaled = best_scaler.transform(X_test)
        test_pred = best_model.predict(X_test_scaled)
        test_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        test_acc = accuracy_score(y_test, test_pred)
        test_auc = (
            roc_auc_score(y_test, test_proba)
            if len(np.unique(y_test)) > 1
            else 0.5
        )

        test_report = (
            f"\n=== TEST SET (OOS) ===\n"
            f"Accuracy: {test_acc:.3f}\n"
            f"AUC-ROC: {test_auc:.3f}\n"
        )
        test_report += classification_report(y_test, test_pred)

        # Retention ratio: how much validation accuracy is retained on test
        retention = test_acc / best_result.accuracy if best_result.accuracy > 0 else 0
        test_report += f"\nRetention ratio (test/val): {retention:.2%}\n"
        if retention < 0.6:
            test_report += "WARNING: Possible overfitting (retention < 60%)\n"

        logger.info(
            f"TEST SET: acc={test_acc:.3f}, AUC={test_auc:.3f}, "
            f"retention={retention:.2%}"
        )

        return test_report

    def _generate_report(
        self,
        best: ModelResult,
        all_results: List[ModelResult],
        regime_models: Dict[str, ModelResult],
        feature_ranking: List[Tuple[str, float]],
        test_report: str
    ) -> str:
        """Generate human-readable training report.

        Args:
            best: The best overall ModelResult.
            all_results: List of all model results for comparison.
            regime_models: Dictionary of regime-specific ModelResults.
            feature_ranking: Sorted list of (feature_name, importance) tuples.
            test_report: Pre-formatted test-set evaluation string (may be empty).

        Returns:
            A multi-line string summarizing the full training run.
        """
        lines = [
            "=" * 70,
            "        MIDAS V7 - ML WALK-FORWARD TRAINING REPORT",
            "=" * 70,
            f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nBest Model: {best.name}",
            f"Validation Accuracy: {best.accuracy:.3f}",
            f"Validation AUC-ROC: {best.auc_roc:.3f}",
            f"Validation F1: {best.f1:.3f}",
            f"Validation Precision: {best.precision:.3f}",
            f"Validation Recall: {best.recall:.3f}",
            f"Train Accuracy: {best.train_accuracy:.3f}",
            f"Confusion Matrix: {best.confusion_matrix}",
            "\n--- All Models Comparison ---",
        ]

        for r in all_results:
            lines.append(
                f"  {r.name}: acc={r.accuracy:.3f}, "
                f"AUC={r.auc_roc:.3f}, F1={r.f1:.3f}"
            )

        lines.append("\n--- Per-Regime Models ---")
        if regime_models:
            for regime, r in regime_models.items():
                lines.append(
                    f"  {regime}: {r.name} acc={r.accuracy:.3f}, "
                    f"AUC={r.auc_roc:.3f}"
                )
        else:
            lines.append("  No regime-specific models trained (insufficient data)")

        lines.append(f"\n--- Top 15 Features ---")
        for i, (fname, imp) in enumerate(feature_ranking[:15]):
            lines.append(f"  {i+1}. {fname}: {imp:.4f}")

        if test_report:
            lines.append(test_report)

        lines.append("=" * 70)
        return "\n".join(lines)


def train_v7_model(
    symbols: List[str] = None,
    data_dir: str = 'data',
    output_dir: str = 'models'
) -> WalkForwardResult:
    """Convenience function to run the full walk-forward training pipeline.

    If training data doesn't exist yet and symbols are provided, generates
    the data first using training_pipeline.generate_training_data().

    Args:
        symbols: List of ticker symbols. Required only if training data
            has not been pre-generated.
        data_dir: Path to the directory containing parquet data files.
        output_dir: Path to the directory where models and reports are saved.

    Returns:
        WalkForwardResult with the best model info, all results, and report.

    Raises:
        ValueError: If no training data exists and no symbols are provided.
    """
    trainer = WalkForwardTrainer(output_dir=output_dir, data_dir=data_dir)

    # Load data
    datasets = trainer.load_data()

    if not datasets or 'train' not in datasets:
        if symbols is None:
            raise ValueError(
                "No training data found and no symbols provided. "
                "Run training_pipeline first."
            )

        # Generate training data
        from .training_pipeline import generate_training_data
        datasets = generate_training_data(symbols)

    # Train
    result = trainer.train_all_models(datasets)

    # Save report
    report_path = os.path.join(output_dir, 'v7_training_report.txt')
    with open(report_path, 'w') as f:
        f.write(result.report)
    logger.info(f"Report saved to {report_path}")

    return result
