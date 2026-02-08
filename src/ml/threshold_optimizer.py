"""
Threshold Optimizer - Grid search on scoring thresholds.

Optimizes:
1. Buy/sell thresholds (entry/exit scores)
2. ML Gate parameters (volatility threshold, boost/block thresholds)
3. Pillar weights per regime

Uses validation period (2022-2023) for tuning.
Reports best parameters and performance improvement.
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from a single parameter combination."""
    params: Dict
    sharpe: float
    cagr: float
    max_dd: float
    total_trades: int
    win_rate: float
    profit_factor: float


@dataclass
class OptimizerReport:
    """Complete optimization report."""
    best_thresholds: Dict
    best_ml_gate: Dict
    best_regime_weights: Dict[str, Dict[str, float]]
    all_results: List[OptimizationResult]
    improvement_vs_baseline: Dict[str, float]
    report_text: str


class ThresholdOptimizer:
    """Grid search optimizer for trading thresholds."""

    def __init__(self, output_dir: str = 'models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def optimize_thresholds(
        self,
        symbols: List[str],
        start_date: str = '2022-01-01',
        end_date: str = '2024-01-01',
        initial_capital: float = 100_000
    ) -> OptimizerReport:
        """Run grid search on buy/sell thresholds and ML Gate params.

        Tests combinations on the validation period.
        """
        from src.backtesting.v6_backtester import (
            V6Backtester, V6BacktestConfig, TransactionCosts,
            ScoringMode, BacktestDataManager
        )

        logger.info(f"Starting threshold optimization: {start_date} to {end_date}")

        # Pre-fetch data once
        data_mgr = BacktestDataManager()
        buffer_start = (pd.Timestamp(start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

        logger.info("Pre-fetching data...")
        data_mgr.get_spy_data(buffer_start, end_date)
        data_mgr.get_vix_data(buffer_start, end_date)
        for sym in symbols:
            data_mgr.get_ohlcv(sym, buffer_start, end_date)
            data_mgr.get_fundamentals(sym)

        # Define search space
        buy_thresholds = [60, 62, 65, 68, 70, 72, 75]
        sell_thresholds = [35, 38, 40, 42, 45]

        ml_vol_thresholds = [0.02, 0.025, 0.03, 0.035, 0.04]
        ml_boost_thresholds = [0.55, 0.60, 0.65, 0.70]
        ml_block_thresholds = [0.35, 0.40, 0.45]
        ml_boost_points = [3, 5, 7, 10]

        # Phase 1: Optimize buy/sell thresholds
        logger.info("Phase 1: Optimizing buy/sell thresholds...")
        threshold_results = []

        for buy_t, sell_t in product(buy_thresholds, sell_thresholds):
            if sell_t >= buy_t:
                continue  # sell must be below buy

            config = V6BacktestConfig(
                initial_capital=initial_capital,
                buy_threshold=buy_t,
                sell_threshold=sell_t,
                scoring_mode=ScoringMode.THREE_PILLARS_VIX,
                costs=TransactionCosts()
            )

            bt = V6Backtester(config, data_mgr=data_mgr)
            result = bt.run(symbols, start_date, end_date)
            m = result.metrics

            years = max(0.5, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
            cagr = ((1 + m.total_return / 100) ** (1 / years) - 1) * 100

            opt_result = OptimizationResult(
                params={'buy_threshold': buy_t, 'sell_threshold': sell_t},
                sharpe=m.sharpe_ratio,
                cagr=cagr,
                max_dd=m.max_drawdown,
                total_trades=m.total_trades,
                win_rate=m.win_rate,
                profit_factor=m.profit_factor
            )
            threshold_results.append(opt_result)
            logger.info(f"  buy={buy_t}, sell={sell_t}: Sharpe={m.sharpe_ratio:.2f}, CAGR={cagr:.1f}%, trades={m.total_trades}")

        # Find best threshold combo (maximize Sharpe, require >= 50 trades)
        valid_results = [r for r in threshold_results if r.total_trades >= 50]
        if not valid_results:
            valid_results = threshold_results

        best_threshold = max(valid_results, key=lambda r: r.sharpe)
        best_buy = best_threshold.params['buy_threshold']
        best_sell = best_threshold.params['sell_threshold']
        logger.info(f"Best thresholds: buy={best_buy}, sell={best_sell}, Sharpe={best_threshold.sharpe:.2f}")

        # Phase 2: Optimize ML Gate with best thresholds
        logger.info("Phase 2: Optimizing ML Gate parameters...")
        ml_gate_results = []

        for vol_t, boost_t, block_t, boost_p in product(
            ml_vol_thresholds[:3],  # Reduce search space
            ml_boost_thresholds[:3],
            ml_block_thresholds[:2],
            ml_boost_points[:3]
        ):
            config = V6BacktestConfig(
                initial_capital=initial_capital,
                buy_threshold=best_buy,
                sell_threshold=best_sell,
                ml_gate_enabled=True,
                ml_gate_volatility_threshold=vol_t,
                ml_boost_threshold=boost_t,
                ml_block_threshold=block_t,
                ml_boost_points=boost_p,
                scoring_mode=ScoringMode.THREE_PILLARS_VIX,
                costs=TransactionCosts()
            )

            bt = V6Backtester(config, data_mgr=data_mgr)
            result = bt.run(symbols, start_date, end_date)
            m = result.metrics

            years = max(0.5, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
            cagr = ((1 + m.total_return / 100) ** (1 / years) - 1) * 100

            opt_result = OptimizationResult(
                params={
                    'ml_vol_threshold': vol_t,
                    'ml_boost_threshold': boost_t,
                    'ml_block_threshold': block_t,
                    'ml_boost_points': boost_p
                },
                sharpe=m.sharpe_ratio,
                cagr=cagr,
                max_dd=m.max_drawdown,
                total_trades=m.total_trades,
                win_rate=m.win_rate,
                profit_factor=m.profit_factor
            )
            ml_gate_results.append(opt_result)

        valid_ml = [r for r in ml_gate_results if r.total_trades >= 30]
        if not valid_ml:
            valid_ml = ml_gate_results
        best_ml_gate = max(valid_ml, key=lambda r: r.sharpe)
        logger.info(f"Best ML Gate: {best_ml_gate.params}, Sharpe={best_ml_gate.sharpe:.2f}")

        # Run baseline for comparison
        baseline_config = V6BacktestConfig(
            initial_capital=initial_capital,
            scoring_mode=ScoringMode.THREE_PILLARS_VIX,
            costs=TransactionCosts()
        )
        bt_baseline = V6Backtester(baseline_config, data_mgr=data_mgr)
        baseline_result = bt_baseline.run(symbols, start_date, end_date)
        baseline_sharpe = baseline_result.metrics.sharpe_ratio
        years = max(0.5, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
        baseline_cagr = ((1 + baseline_result.metrics.total_return / 100) ** (1 / years) - 1) * 100

        # Generate report
        improvement = {
            'sharpe_improvement': best_threshold.sharpe - baseline_sharpe,
            'cagr_improvement': best_threshold.cagr - baseline_cagr,
            'baseline_sharpe': baseline_sharpe,
            'baseline_cagr': baseline_cagr,
            'optimized_sharpe': best_threshold.sharpe,
            'optimized_cagr': best_threshold.cagr,
        }

        report_text = self._generate_report(
            best_threshold, best_ml_gate, baseline_sharpe, baseline_cagr,
            threshold_results, ml_gate_results
        )

        # Save optimized config
        optimized_config = {
            'buy_threshold': best_buy,
            'sell_threshold': best_sell,
            'ml_gate_volatility_threshold': best_ml_gate.params.get('ml_vol_threshold', 0.03),
            'ml_boost_threshold': best_ml_gate.params.get('ml_boost_threshold', 0.60),
            'ml_block_threshold': best_ml_gate.params.get('ml_block_threshold', 0.40),
            'ml_boost_points': best_ml_gate.params.get('ml_boost_points', 5),
        }

        config_path = os.path.join(self.output_dir, 'optimized_thresholds_v7.json')
        with open(config_path, 'w') as f:
            json.dump(optimized_config, f, indent=2)
        logger.info(f"Saved optimized config to {config_path}")

        return OptimizerReport(
            best_thresholds={'buy_threshold': best_buy, 'sell_threshold': best_sell},
            best_ml_gate=best_ml_gate.params,
            best_regime_weights={},  # Could be extended
            all_results=threshold_results + ml_gate_results,
            improvement_vs_baseline=improvement,
            report_text=report_text
        )

    def _generate_report(
        self,
        best_threshold: OptimizationResult,
        best_ml_gate: OptimizationResult,
        baseline_sharpe: float,
        baseline_cagr: float,
        threshold_results: List[OptimizationResult],
        ml_gate_results: List[OptimizationResult]
    ) -> str:
        """Generate optimization report."""
        lines = [
            "=" * 70,
            "        MIDAS V7 - THRESHOLD OPTIMIZATION REPORT",
            "=" * 70,
            f"\n--- Baseline (V6.2 defaults) ---",
            f"Sharpe: {baseline_sharpe:.2f}",
            f"CAGR: {baseline_cagr:.1f}%",
            f"\n--- Best Thresholds ---",
            f"Buy threshold: {best_threshold.params['buy_threshold']}",
            f"Sell threshold: {best_threshold.params['sell_threshold']}",
            f"Sharpe: {best_threshold.sharpe:.2f} (delta: {best_threshold.sharpe - baseline_sharpe:+.2f})",
            f"CAGR: {best_threshold.cagr:.1f}% (delta: {best_threshold.cagr - baseline_cagr:+.1f}%)",
            f"Win Rate: {best_threshold.win_rate:.1f}%",
            f"Trades: {best_threshold.total_trades}",
            f"Max DD: -{best_threshold.max_dd:.1f}%",
            f"\n--- Best ML Gate ---",
        ]
        for k, v in best_ml_gate.params.items():
            lines.append(f"  {k}: {v}")
        lines.extend([
            f"Sharpe: {best_ml_gate.sharpe:.2f}",
            f"CAGR: {best_ml_gate.cagr:.1f}%",
            f"\n--- Top 5 Threshold Combos (by Sharpe) ---",
        ])

        sorted_t = sorted(threshold_results, key=lambda r: r.sharpe, reverse=True)[:5]
        for r in sorted_t:
            lines.append(
                f"  buy={r.params['buy_threshold']}, sell={r.params['sell_threshold']}: "
                f"Sharpe={r.sharpe:.2f}, CAGR={r.cagr:.1f}%, trades={r.total_trades}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


def optimize_thresholds(
    symbols: List[str],
    start_date: str = '2022-01-01',
    end_date: str = '2024-01-01',
    output_dir: str = 'models'
) -> OptimizerReport:
    """Convenience function for threshold optimization."""
    optimizer = ThresholdOptimizer(output_dir=output_dir)
    return optimizer.optimize_thresholds(symbols, start_date, end_date)
