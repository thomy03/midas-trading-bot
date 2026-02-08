"""
ML Module - V7 Machine Learning components.

- training_pipeline: Generate ML training data from backtester
- walk_forward_trainer: Train models with walk-forward methodology
- threshold_optimizer: Grid search for optimal thresholds
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
]
