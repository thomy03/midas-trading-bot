"""
Optuna Hyperparameter Optimizer for Midas V7 ML Pipeline.

Uses Optuna's Bayesian optimization (TPE sampler) to find optimal
hyperparameters for RandomForest, GradientBoosting, and feature selection.

Integrates with WalkForwardTrainer: uses the same data splits, feature
definitions, and data preparation logic so results are directly comparable.

Usage:
    # Programmatic
    from src.ml.optuna_optimizer import OptunaMLOptimizer
    optimizer = OptunaMLOptimizer()
    results = optimizer.run_full_optimization()

    # CLI
    python -m src.ml.optuna_optimizer --n-trials 100 --timeout 1800
"""

import numpy as np
import pandas as pd
import logging
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )
    OPTUNA_VIS_AVAILABLE = True
except ImportError:
    OPTUNA_VIS_AVAILABLE = False

# Import project-level constants and trainer
from .walk_forward_trainer import FEATURE_NAMES, WalkForwardTrainer

# ---------------------------------------------------------------------------
# Baseline parameters (current hard-coded values in walk_forward_trainer.py)
# ---------------------------------------------------------------------------
BASELINE_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "min_samples_split": 20,
    "max_features": "sqrt",
    "class_weight": "balanced",
}

BASELINE_GB_PARAMS: Dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_samples_leaf": 10,
    "subsample": 0.8,
}


class OptunaMLOptimizer:
    """Bayesian hyperparameter optimization for the Midas ML pipeline.

    Wraps three Optuna studies -- RandomForest tuning, GradientBoosting
    tuning, and feature selection -- behind a clean API that mirrors the
    data loading / preparation already implemented in WalkForwardTrainer.

    Parameters
    ----------
    data_dir : str
        Directory containing parquet data files.
    output_dir : str
        Directory for saving models, results, and the Optuna SQLite DB.
    n_trials : int
        Default number of trials for each study.
    timeout : int
        Default wall-clock timeout (seconds) for each study.
    study_name : str
        Base name for Optuna studies. Individual studies will be suffixed
        with ``_rf``, ``_gb``, and ``_features``.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "models",
        n_trials: int = 100,
        timeout: int = 1800,
        study_name: str = "midas_ml",
        random_state: int = 42,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required. Install with: pip install 'optuna>=3.0.0'"
            )
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.random_state = random_state

        os.makedirs(output_dir, exist_ok=True)

        # Will be populated by _load_data()
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

        # Will be populated by optimize_*() methods
        self._best_rf_params: Optional[Dict] = None
        self._best_gb_params: Optional[Dict] = None
        self._best_features: Optional[List[str]] = None

        # Reuse WalkForwardTrainer for consistent data handling
        self._trainer = WalkForwardTrainer(
            output_dir=output_dir, data_dir=data_dir
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load and prepare train / validation splits from parquet files.

        Delegates to WalkForwardTrainer.load_data() and
        WalkForwardTrainer.prepare_features() so the data pipeline is
        identical to the one used for regular training.
        """
        if self._X_train is not None:
            return  # already loaded

        datasets = self._trainer.load_data()
        if "train" not in datasets or "validation" not in datasets:
            raise ValueError(
                "Both train and validation parquet files are required. "
                "Run training_pipeline first."
            )

        self._X_train, self._y_train = self._trainer.prepare_features(
            datasets["train"]
        )
        self._X_val, self._y_val = self._trainer.prepare_features(
            datasets["validation"]
        )

        # Determine which features are actually available in the data
        available = [
            f for f in FEATURE_NAMES if f in datasets["train"].columns
        ]
        self._feature_names = available

        logger.info(
            "Data loaded: train=%d samples, val=%d samples, features=%d",
            self._X_train.shape[0],
            self._X_val.shape[0],
            len(self._feature_names),
        )
        logger.info(
            "Train label distribution: %s",
            dict(zip(*np.unique(self._y_train, return_counts=True))),
        )
        logger.info(
            "Val   label distribution: %s",
            dict(zip(*np.unique(self._y_val, return_counts=True))),
        )

    # ------------------------------------------------------------------
    # Helper: evaluate a model on the validation set
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Fit *model* on (X_train, y_train) and return AUC-ROC on (X_val, y_val)."""
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_v = scaler.transform(X_val)

        model.fit(X_tr, y_train)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_v)[:, 1]
        else:
            proba = model.predict(X_v).astype(float)

        try:
            return roc_auc_score(y_val, proba)
        except ValueError:
            return 0.5

    # ------------------------------------------------------------------
    # Helper: SQLite storage URL
    # ------------------------------------------------------------------

    def _storage_url(self) -> str:
        db_path = os.path.join(self.output_dir, "optuna_study.db")
        return f"sqlite:///{db_path}"

    # ------------------------------------------------------------------
    # 1. RandomForest optimization
    # ------------------------------------------------------------------

    def optimize_rf(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Optimize RandomForest hyperparameters using Optuna.

        Search space:
            n_estimators    [50, 500]
            max_depth       [3, 15]
            min_samples_leaf [5, 50]
            min_samples_split [2, 20]
            max_features    'sqrt' | 'log2' | float in [0.3, 0.8]
            class_weight    'balanced' | 'balanced_subsample' | None

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        self._load_data()
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout

        study_name = f"{self.study_name}_rf"
        logger.info(
            "Starting RF optimization: n_trials=%d, timeout=%ds",
            n_trials,
            timeout,
        )

        # Compute baseline AUC
        baseline_model = RandomForestClassifier(
            random_state=self.random_state, n_jobs=-1, **BASELINE_RF_PARAMS
        )
        baseline_auc = self._evaluate(
            baseline_model,
            self._X_train,
            self._y_train,
            self._X_val,
            self._y_val,
        )
        logger.info("RF baseline AUC-ROC: %.4f", baseline_auc)

        def objective(trial: "optuna.Trial") -> float:
            # --- max_features ---
            mf_type = trial.suggest_categorical(
                "max_features_type", ["sqrt", "log2", "float"]
            )
            if mf_type == "float":
                max_features: Any = trial.suggest_float(
                    "max_features_frac", 0.3, 0.8
                )
            else:
                max_features = mf_type

            # --- class_weight ---
            cw = trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", "none"]
            )
            class_weight: Any = None if cw == "none" else cw

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 5, 50
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "max_features": max_features,
                "class_weight": class_weight,
                "random_state": self.random_state,
                "n_jobs": -1,
            }

            model = RandomForestClassifier(**params)
            auc = self._evaluate(
                model,
                self._X_train,
                self._y_train,
                self._X_val,
                self._y_val,
            )

            # Report intermediate value for pruning
            trial.report(auc, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return auc

        study = optuna.create_study(
            study_name=study_name,
            storage=self._storage_url(),
            load_if_exists=True,
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best = study.best_params
        # Reconstruct clean param dict
        mf_type = best.pop("max_features_type", "sqrt")
        mf_frac = best.pop("max_features_frac", None)
        best["max_features"] = mf_frac if mf_type == "float" else mf_type

        cw_raw = best.pop("class_weight", "balanced")
        best["class_weight"] = None if cw_raw == "none" else cw_raw

        self._best_rf_params = best

        logger.info("=" * 60)
        logger.info("RF OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info("Best AUC-ROC: %.4f (baseline: %.4f)", study.best_value, baseline_auc)
        logger.info(
            "Improvement : %+.4f (%+.2f%%)",
            study.best_value - baseline_auc,
            (study.best_value - baseline_auc) / max(baseline_auc, 1e-9) * 100,
        )
        logger.info("Best params :")
        for k, v in best.items():
            logger.info("  %-22s = %s", k, v)
        logger.info("Trials completed: %d", len(study.trials))

        self._save_visualizations(study, "rf")

        return best

    # ------------------------------------------------------------------
    # 2. GradientBoosting optimization
    # ------------------------------------------------------------------

    def optimize_gb(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Optimize GradientBoosting hyperparameters using Optuna.

        Search space:
            n_estimators     [50, 300]
            max_depth        [3, 10]
            learning_rate    [0.01, 0.3] (log scale)
            subsample        [0.5, 1.0]
            min_samples_leaf [5, 30]

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        self._load_data()
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout

        study_name = f"{self.study_name}_gb"
        logger.info(
            "Starting GB optimization: n_trials=%d, timeout=%ds",
            n_trials,
            timeout,
        )

        # Compute baseline AUC
        baseline_model = GradientBoostingClassifier(
            random_state=self.random_state, **BASELINE_GB_PARAMS
        )
        baseline_auc = self._evaluate(
            baseline_model,
            self._X_train,
            self._y_train,
            self._X_val,
            self._y_val,
        )
        logger.info("GB baseline AUC-ROC: %.4f", baseline_auc)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 5, 30
                ),
                "random_state": self.random_state,
            }

            model = GradientBoostingClassifier(**params)
            auc = self._evaluate(
                model,
                self._X_train,
                self._y_train,
                self._X_val,
                self._y_val,
            )

            trial.report(auc, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return auc

        study = optuna.create_study(
            study_name=study_name,
            storage=self._storage_url(),
            load_if_exists=True,
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best = study.best_params
        self._best_gb_params = best

        logger.info("=" * 60)
        logger.info("GB OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info("Best AUC-ROC: %.4f (baseline: %.4f)", study.best_value, baseline_auc)
        logger.info(
            "Improvement : %+.4f (%+.2f%%)",
            study.best_value - baseline_auc,
            (study.best_value - baseline_auc) / max(baseline_auc, 1e-9) * 100,
        )
        logger.info("Best params :")
        for k, v in best.items():
            logger.info("  %-22s = %s", k, v)
        logger.info("Trials completed: %d", len(study.trials))

        self._save_visualizations(study, "gb")

        return best

    # ------------------------------------------------------------------
    # 3. Feature selection
    # ------------------------------------------------------------------

    def optimize_features(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> List[str]:
        """Select the optimal feature subset via Optuna.

        Each feature is represented as a boolean trial parameter. A
        constraint enforces that between 15 and 35 features are selected.
        The objective maximises AUC-ROC of a RandomForest trained on
        the selected subset.

        If RF params have already been optimised (via ``optimize_rf``),
        those params are used; otherwise the baseline defaults are used.

        Returns
        -------
        list[str]
            Names of the features in the best subset.
        """
        self._load_data()
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout

        study_name = f"{self.study_name}_features"
        logger.info(
            "Starting feature selection: n_trials=%d, timeout=%ds, "
            "candidate features=%d",
            n_trials,
            timeout,
            len(self._feature_names),
        )

        # Pick the model params to use for evaluation
        rf_params = dict(self._best_rf_params or BASELINE_RF_PARAMS)
        rf_params["random_state"] = self.random_state
        rf_params["n_jobs"] = -1

        # Baseline: all features
        baseline_model = RandomForestClassifier(**rf_params)
        baseline_auc = self._evaluate(
            baseline_model,
            self._X_train,
            self._y_train,
            self._X_val,
            self._y_val,
        )
        logger.info("Feature selection baseline AUC-ROC (all features): %.4f", baseline_auc)

        feature_names = self._feature_names

        def objective(trial: "optuna.Trial") -> float:
            # Each feature is a boolean
            mask = [
                trial.suggest_categorical(f"use_{fname}", [True, False])
                for fname in feature_names
            ]
            selected_count = sum(mask)

            # Enforce bounds (15-35 features)
            if selected_count < 15 or selected_count > 35:
                return 0.0  # penalty -- effectively pruned

            indices = [i for i, m in enumerate(mask) if m]
            X_tr_sub = self._X_train[:, indices]
            X_v_sub = self._X_val[:, indices]

            model = RandomForestClassifier(**rf_params)
            auc = self._evaluate(
                model, X_tr_sub, self._y_train, X_v_sub, self._y_val
            )

            trial.report(auc, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return auc

        study = optuna.create_study(
            study_name=study_name,
            storage=self._storage_url(),
            load_if_exists=True,
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Extract best feature subset
        best_mask = [
            study.best_params.get(f"use_{fname}", True)
            for fname in feature_names
        ]
        best_features = [
            fname for fname, m in zip(feature_names, best_mask) if m
        ]
        self._best_features = best_features

        logger.info("=" * 60)
        logger.info("FEATURE SELECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(
            "Best AUC-ROC: %.4f (baseline all-features: %.4f)",
            study.best_value,
            baseline_auc,
        )
        logger.info(
            "Improvement : %+.4f (%+.2f%%)",
            study.best_value - baseline_auc,
            (study.best_value - baseline_auc) / max(baseline_auc, 1e-9) * 100,
        )
        logger.info(
            "Selected %d / %d features:", len(best_features), len(feature_names)
        )
        for fname in best_features:
            logger.info("  + %s", fname)
        logger.info("Trials completed: %d", len(study.trials))

        dropped = [f for f in feature_names if f not in best_features]
        if dropped:
            logger.info("Dropped %d features:", len(dropped))
            for fname in dropped:
                logger.info("  - %s", fname)

        self._save_visualizations(study, "features")

        return best_features

    # ------------------------------------------------------------------
    # 4. Full optimization pipeline
    # ------------------------------------------------------------------

    def run_full_optimization(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run RF, GB, and feature selection optimizations in sequence.

        Returns
        -------
        dict
            Keys: ``best_rf_params``, ``best_gb_params``, ``best_features``,
            plus metadata like ``baseline_rf_auc``, ``baseline_gb_auc``,
            ``optimized_rf_auc``, ``optimized_gb_auc``, and
            ``timestamp``.
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        self._load_data()

        start_time = time.time()
        logger.info("=" * 70)
        logger.info("    MIDAS V7 - FULL OPTUNA HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 70)
        logger.info("n_trials=%d per study, timeout=%ds per study", n_trials, timeout)

        # ------ Baselines ------
        baseline_rf = RandomForestClassifier(
            random_state=self.random_state, n_jobs=-1, **BASELINE_RF_PARAMS
        )
        baseline_rf_auc = self._evaluate(
            baseline_rf, self._X_train, self._y_train, self._X_val, self._y_val
        )

        baseline_gb = GradientBoostingClassifier(
            random_state=self.random_state, **BASELINE_GB_PARAMS
        )
        baseline_gb_auc = self._evaluate(
            baseline_gb, self._X_train, self._y_train, self._X_val, self._y_val
        )

        logger.info("Baseline RF  AUC-ROC: %.4f", baseline_rf_auc)
        logger.info("Baseline GB  AUC-ROC: %.4f", baseline_gb_auc)

        # ------ Step 1: RF ------
        logger.info("-" * 60)
        logger.info("STEP 1/3: Optimizing RandomForest")
        logger.info("-" * 60)
        best_rf = self.optimize_rf(n_trials=n_trials, timeout=timeout)

        # ------ Step 2: GB ------
        logger.info("-" * 60)
        logger.info("STEP 2/3: Optimizing GradientBoosting")
        logger.info("-" * 60)
        best_gb = self.optimize_gb(n_trials=n_trials, timeout=timeout)

        # ------ Step 3: Features ------
        logger.info("-" * 60)
        logger.info("STEP 3/3: Optimizing Feature Selection")
        logger.info("-" * 60)
        best_features = self.optimize_features(n_trials=n_trials, timeout=timeout)

        # ------ Evaluate optimized models ------
        opt_rf = RandomForestClassifier(
            random_state=self.random_state, n_jobs=-1, **best_rf
        )
        opt_rf_auc = self._evaluate(
            opt_rf, self._X_train, self._y_train, self._X_val, self._y_val
        )

        opt_gb = GradientBoostingClassifier(
            random_state=self.random_state, **best_gb
        )
        opt_gb_auc = self._evaluate(
            opt_gb, self._X_train, self._y_train, self._X_val, self._y_val
        )

        elapsed = time.time() - start_time

        # ------ Summary ------
        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "n_trials_per_study": n_trials,
            "timeout_per_study": timeout,
            "baseline_rf_params": BASELINE_RF_PARAMS,
            "baseline_gb_params": BASELINE_GB_PARAMS,
            "baseline_rf_auc": round(baseline_rf_auc, 6),
            "baseline_gb_auc": round(baseline_gb_auc, 6),
            "best_rf_params": _serialize_params(best_rf),
            "best_gb_params": _serialize_params(best_gb),
            "best_features": best_features,
            "optimized_rf_auc": round(opt_rf_auc, 6),
            "optimized_gb_auc": round(opt_gb_auc, 6),
            "rf_improvement": round(opt_rf_auc - baseline_rf_auc, 6),
            "gb_improvement": round(opt_gb_auc - baseline_gb_auc, 6),
            "n_features_selected": len(best_features),
            "n_features_total": len(self._feature_names),
        }

        # Save JSON
        results_path = os.path.join(self.output_dir, "optuna_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", results_path)

        # Print final summary
        logger.info("=" * 70)
        logger.info("    OPTIMIZATION COMPLETE  (%.1f seconds)", elapsed)
        logger.info("=" * 70)
        logger.info("")
        logger.info("  RandomForest AUC-ROC:       %.4f -> %.4f (%+.4f)",
                     baseline_rf_auc, opt_rf_auc, opt_rf_auc - baseline_rf_auc)
        logger.info("  GradientBoosting AUC-ROC:   %.4f -> %.4f (%+.4f)",
                     baseline_gb_auc, opt_gb_auc, opt_gb_auc - baseline_gb_auc)
        logger.info("  Features selected:           %d / %d",
                     len(best_features), len(self._feature_names))
        logger.info("")
        logger.info("  Best RF params:")
        for k, v in best_rf.items():
            logger.info("    %-24s = %s", k, v)
        logger.info("  Best GB params:")
        for k, v in best_gb.items():
            logger.info("    %-24s = %s", k, v)
        logger.info("")
        logger.info("  Artifacts:")
        logger.info("    Results JSON : %s", results_path)
        logger.info("    Optuna DB    : %s",
                     os.path.join(self.output_dir, "optuna_study.db"))

        return results

    # ------------------------------------------------------------------
    # 5. Apply best params to WalkForwardTrainer and retrain
    # ------------------------------------------------------------------

    def apply_best_params(
        self,
        results: Optional[Dict[str, Any]] = None,
    ) -> "WalkForwardTrainer":
        """Apply optimized hyperparameters and retrain via WalkForwardTrainer.

        If *results* is not supplied, the method attempts to load them from
        ``models/optuna_results.json``.

        This monkey-patches ``WalkForwardTrainer.train_single_model`` so
        the optimized parameters are used instead of the hard-coded
        defaults, then runs the full ``train_all_models`` pipeline
        (including per-regime models and test-set evaluation).

        Returns
        -------
        WalkForwardTrainer
            The trainer instance (models have been saved to disk).
        """
        if results is None:
            results_path = os.path.join(self.output_dir, "optuna_results.json")
            if not os.path.exists(results_path):
                raise FileNotFoundError(
                    f"No results found at {results_path}. "
                    "Run run_full_optimization() first."
                )
            with open(results_path) as f:
                results = json.load(f)

        best_rf = results.get("best_rf_params", BASELINE_RF_PARAMS)
        best_gb = results.get("best_gb_params", BASELINE_GB_PARAMS)
        best_features = results.get("best_features")

        logger.info("Applying optimized params and retraining ...")
        logger.info("RF params : %s", best_rf)
        logger.info("GB params : %s", best_gb)
        if best_features:
            logger.info("Features  : %d selected", len(best_features))

        trainer = WalkForwardTrainer(
            output_dir=self.output_dir,
            data_dir=self.data_dir,
        )
        datasets = trainer.load_data()

        if not datasets or "train" not in datasets:
            raise ValueError("Training data not found. Run training_pipeline first.")

        # If we have an optimized feature subset, override prepare_features
        if best_features:
            original_prepare = trainer.prepare_features

            def prepare_with_selected(df, feature_names=None):
                return original_prepare(df, feature_names=best_features)

            trainer.prepare_features = prepare_with_selected

        # Override train_single_model to inject optimized params
        original_train_single = trainer.train_single_model

        def patched_train_single(
            model_type, X_train, y_train, X_val, y_val, regime="ALL"
        ):
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("sklearn not available")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            if model_type == "rf":
                params = dict(best_rf)
                params["random_state"] = 42
                params["n_jobs"] = -1
                # Ensure class_weight is proper type
                if params.get("class_weight") == "none":
                    params["class_weight"] = None
                model = RandomForestClassifier(**params)
                name = "RandomForest_Optuna"
            elif model_type == "gb":
                params = dict(best_gb)
                params["random_state"] = 42
                model = GradientBoostingClassifier(**params)
                name = "GradientBoosting_Optuna"
            else:
                # Ensemble or unknown -- delegate to original
                return original_train_single(
                    model_type, X_train, y_train, X_val, y_val, regime
                )

            model.fit(X_train_scaled, y_train)

            # Import metrics locally to stay self-contained
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, confusion_matrix, roc_auc_score as _roc_auc,
            )
            from .walk_forward_trainer import ModelResult

            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            val_proba = (
                model.predict_proba(X_val_scaled)[:, 1]
                if hasattr(model, "predict_proba")
                else val_pred.astype(float)
            )

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            val_prec = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

            try:
                val_auc = _roc_auc(y_val, val_proba)
            except ValueError:
                val_auc = 0.5

            cm = confusion_matrix(y_val, val_pred).tolist()

            importances = {}
            feat_names = best_features or FEATURE_NAMES
            if hasattr(model, "feature_importances_"):
                for fname, imp in zip(
                    feat_names[: X_train.shape[1]],
                    model.feature_importances_,
                ):
                    importances[fname] = float(imp)

            result = ModelResult(
                name=name,
                accuracy=val_acc,
                precision=val_prec,
                recall=val_recall,
                f1=val_f1,
                auc_roc=val_auc,
                train_accuracy=train_acc,
                feature_importances=importances,
                best_params=params,
                confusion_matrix=cm,
                regime=regime,
            )

            logger.info(
                "[%s] regime=%s | Val acc=%.3f | AUC=%.3f | F1=%.3f | "
                "Train acc=%.3f",
                name, regime, val_acc, val_auc, val_f1, train_acc,
            )

            return (model, scaler), result

        trainer.train_single_model = patched_train_single

        # Run full training
        wf_result = trainer.train_all_models(datasets)

        # Save report
        report_path = os.path.join(self.output_dir, "v7_optuna_training_report.txt")
        with open(report_path, "w") as f:
            f.write(wf_result.report)
        logger.info("Optuna-tuned training report saved to %s", report_path)

        return trainer

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _save_visualizations(
        self, study: "optuna.Study", suffix: str
    ) -> None:
        """Attempt to save Optuna visualization HTML files."""
        if not OPTUNA_VIS_AVAILABLE:
            logger.debug(
                "optuna.visualization not available; skipping plots."
            )
            return

        vis_dir = os.path.join(self.output_dir, "optuna_plots")
        os.makedirs(vis_dir, exist_ok=True)

        try:
            fig = plot_optimization_history(study)
            fig.write_html(
                os.path.join(vis_dir, f"optimization_history_{suffix}.html")
            )
        except Exception as e:
            logger.debug("Could not save optimization_history plot: %s", e)

        try:
            fig = plot_param_importances(study)
            fig.write_html(
                os.path.join(vis_dir, f"param_importances_{suffix}.html")
            )
        except Exception as e:
            logger.debug("Could not save param_importances plot: %s", e)

        try:
            fig = plot_slice(study)
            fig.write_html(
                os.path.join(vis_dir, f"slice_{suffix}.html")
            )
        except Exception as e:
            logger.debug("Could not save slice plot: %s", e)


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------

def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy / non-JSON-serializable types in a param dict."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main() -> None:
    """CLI entry point: ``python -m src.ml.optuna_optimizer``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Midas V7 -- Optuna ML Hyperparameter Optimization",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per study (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds per study (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="midas_ml",
        help="Base name for Optuna studies (default: midas_ml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with parquet training data (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for output artifacts (default: models)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="After optimization, apply best params and retrain.",
    )
    parser.add_argument(
        "--rf-only",
        action="store_true",
        help="Only optimize RandomForest.",
    )
    parser.add_argument(
        "--gb-only",
        action="store_true",
        help="Only optimize GradientBoosting.",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only optimize feature selection.",
    )

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    optimizer = OptunaMLOptimizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
    )

    # Determine what to run
    run_all = not (args.rf_only or args.gb_only or args.features_only)

    if run_all:
        results = optimizer.run_full_optimization(
            n_trials=args.n_trials, timeout=args.timeout
        )
    else:
        results = {}
        if args.rf_only:
            results["best_rf_params"] = optimizer.optimize_rf(
                n_trials=args.n_trials, timeout=args.timeout
            )
        if args.gb_only:
            results["best_gb_params"] = optimizer.optimize_gb(
                n_trials=args.n_trials, timeout=args.timeout
            )
        if args.features_only:
            results["best_features"] = optimizer.optimize_features(
                n_trials=args.n_trials, timeout=args.timeout
            )

    # Optionally retrain with best params
    if args.apply:
        print("\n--- Applying best parameters and retraining ---\n")
        optimizer.apply_best_params(results if run_all else None)

    print("\nDone.")


if __name__ == "__main__":
    main()
