"""
ML Module - V7 Machine Learning components.

- training_pipeline: Generate ML training data from backtester
- walk_forward_trainer: Train models with walk-forward methodology
- threshold_optimizer: Grid search for optimal thresholds
- auto_retrain: Nightly auto-retraining with validation gate
"""

from .training_pipeline import (
    TrainingDataGenerator,
    TrainingConfig,
    FeatureExtractor,
    FEATURE_NAMES,
    generate_training_data,
)

from .walk_forward_trainer import (
    WalkForwardTrainer,
    WalkForwardResult,
    ModelResult,
    train_v7_model,
)

from .threshold_optimizer import (
    ThresholdOptimizer,
    OptimizerReport,
    optimize_thresholds,
)

from .auto_retrain import (
    NightlyRetrainer,
    RetrainResult,
    run_nightly,
)

try:
    from .optuna_optimizer import OptunaMLOptimizer
    _OPTUNA_EXPORTS = ['OptunaMLOptimizer']
except ImportError:
    _OPTUNA_EXPORTS = []

__all__ = [
    'TrainingDataGenerator',
    'TrainingConfig',
    'FeatureExtractor',
    'FEATURE_NAMES',
    'generate_training_data',
    'WalkForwardTrainer',
    'WalkForwardResult',
    'ModelResult',
    'train_v7_model',
    'ThresholdOptimizer',
    'OptimizerReport',
    'optimize_thresholds',
    'NightlyRetrainer',
    'RetrainResult',
    'run_nightly',
] + _OPTUNA_EXPORTS
