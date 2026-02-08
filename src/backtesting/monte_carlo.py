"""
Monte Carlo Simulation - Statistical validation of backtest results.

Takes completed trades from a backtest and:
1. Reshuffles trade order 1000 times
2. Recalculates equity curve for each shuffle
3. Produces confidence intervals for CAGR, Sharpe, Max DD
4. Tests if strategy is statistically significant

If the 5th percentile Sharpe > 0, the strategy is statistically significant.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    n_simulations: int

    # CAGR distribution
    cagr_mean: float
    cagr_median: float
    cagr_p5: float      # 5th percentile
    cagr_p25: float
    cagr_p75: float
    cagr_p95: float

    # Sharpe distribution
    sharpe_mean: float
    sharpe_median: float
    sharpe_p5: float
    sharpe_p25: float
    sharpe_p75: float
    sharpe_p95: float

    # Max Drawdown distribution
    maxdd_mean: float
    maxdd_median: float
    maxdd_p5: float      # Best case (lowest DD)
    maxdd_p95: float     # Worst case (highest DD)

    # Statistical significance
    is_significant: bool       # Sharpe p5 > 0
    prob_profitable: float     # % of sims with positive CAGR
    prob_beat_spy: float       # % of sims beating SPY CAGR

    report: str


class MonteCarloSimulator:
    """Monte Carlo simulation for backtest validation."""

    def __init__(
        self,
        n_simulations: int = 1000,
        initial_capital: float = 100_000,
        spy_cagr: float = 10.0,  # Benchmark S&P 500 CAGR
        seed: int = 42
    ):
        self.n_simulations = n_simulations
        self.initial_capital = initial_capital
        self.spy_cagr = spy_cagr
        self.seed = seed

    def run(
        self,
        trade_pnls: List[float],
        trade_durations: List[int] = None,
        total_days: int = 2520  # ~10 years
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on trade P&L sequence.

        Args:
            trade_pnls: List of trade P&L as dollar amounts
            trade_durations: Optional list of trade durations in days
            total_days: Total trading days in the backtest period

        Returns:
            MonteCarloResult with distribution statistics
        """
        if not trade_pnls:
            return self._empty_result()

        rng = np.random.RandomState(self.seed)
        pnls = np.array(trade_pnls)
        n_trades = len(pnls)

        logger.info(f"Running {self.n_simulations} Monte Carlo simulations on {n_trades} trades")

        years = total_days / 252.0

        cagrs = []
        sharpes = []
        max_dds = []

        for i in range(self.n_simulations):
            # Shuffle trade order
            shuffled = rng.permutation(pnls)

            # Build equity curve
            equity = np.zeros(n_trades + 1)
            equity[0] = self.initial_capital
            for j, pnl in enumerate(shuffled):
                equity[j + 1] = equity[j] + pnl

            # Calculate metrics
            final_capital = equity[-1]
            total_return = (final_capital - self.initial_capital) / self.initial_capital

            # CAGR
            if total_return > -1:
                cagr = ((1 + total_return) ** (1 / years) - 1) * 100
            else:
                cagr = -100.0
            cagrs.append(cagr)

            # Approximate daily returns from equity curve
            # Distribute trades evenly across trading days
            daily_equity = np.interp(
                np.linspace(0, n_trades, total_days),
                np.arange(n_trades + 1),
                equity
            )
            daily_returns = np.diff(daily_equity) / daily_equity[:-1]
            daily_returns = daily_returns[np.isfinite(daily_returns)]

            # Sharpe
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                excess = daily_returns - 0.02 / 252  # Risk-free rate
                sharpe = np.sqrt(252) * np.mean(excess) / np.std(excess)
            else:
                sharpe = 0.0
            sharpes.append(sharpe)

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak * 100
            max_dd = np.max(drawdown)
            max_dds.append(max_dd)

        # Calculate percentiles
        cagrs = np.array(cagrs)
        sharpes = np.array(sharpes)
        max_dds = np.array(max_dds)

        is_significant = float(np.percentile(sharpes, 5)) > 0
        prob_profitable = float(np.mean(cagrs > 0))
        prob_beat_spy = float(np.mean(cagrs > self.spy_cagr))

        result = MonteCarloResult(
            n_simulations=self.n_simulations,
            cagr_mean=float(np.mean(cagrs)),
            cagr_median=float(np.median(cagrs)),
            cagr_p5=float(np.percentile(cagrs, 5)),
            cagr_p25=float(np.percentile(cagrs, 25)),
            cagr_p75=float(np.percentile(cagrs, 75)),
            cagr_p95=float(np.percentile(cagrs, 95)),
            sharpe_mean=float(np.mean(sharpes)),
            sharpe_median=float(np.median(sharpes)),
            sharpe_p5=float(np.percentile(sharpes, 5)),
            sharpe_p25=float(np.percentile(sharpes, 25)),
            sharpe_p75=float(np.percentile(sharpes, 75)),
            sharpe_p95=float(np.percentile(sharpes, 95)),
            maxdd_mean=float(np.mean(max_dds)),
            maxdd_median=float(np.median(max_dds)),
            maxdd_p5=float(np.percentile(max_dds, 5)),
            maxdd_p95=float(np.percentile(max_dds, 95)),
            is_significant=is_significant,
            prob_profitable=prob_profitable,
            prob_beat_spy=prob_beat_spy,
            report=''
        )
        result.report = self._generate_report(result)

        logger.info(f"Monte Carlo complete: significant={is_significant}, "
                    f"prob_profitable={prob_profitable:.1%}, "
                    f"prob_beat_SPY={prob_beat_spy:.1%}")

        return result

    def _generate_report(self, r: MonteCarloResult) -> str:
        """Generate report string."""
        significance = "YES - Statistically significant" if r.is_significant else "NO - NOT significant"
        lines = [
            "=" * 60,
            "        MONTE CARLO SIMULATION RESULTS",
            "=" * 60,
            f"Simulations: {r.n_simulations}",
            "",
            "--- CAGR Distribution ---",
            f"  Mean:      {r.cagr_mean:+.1f}%",
            f"  Median:    {r.cagr_median:+.1f}%",
            f"  5th pctl:  {r.cagr_p5:+.1f}%  (worst case)",
            f"  25th pctl: {r.cagr_p25:+.1f}%",
            f"  75th pctl: {r.cagr_p75:+.1f}%",
            f"  95th pctl: {r.cagr_p95:+.1f}%  (best case)",
            "",
            "--- Sharpe Distribution ---",
            f"  Mean:      {r.sharpe_mean:.2f}",
            f"  Median:    {r.sharpe_median:.2f}",
            f"  5th pctl:  {r.sharpe_p5:.2f}  (worst case)",
            f"  95th pctl: {r.sharpe_p95:.2f}  (best case)",
            "",
            "--- Max Drawdown Distribution ---",
            f"  Mean:      -{r.maxdd_mean:.1f}%",
            f"  Median:    -{r.maxdd_median:.1f}%",
            f"  Best case: -{r.maxdd_p5:.1f}%  (5th pctl)",
            f"  Worst case: -{r.maxdd_p95:.1f}%  (95th pctl)",
            "",
            "--- Statistical Significance ---",
            f"  Sharpe p5 > 0: {significance}",
            f"  Prob. profitable: {r.prob_profitable:.1%}",
            f"  Prob. beat S&P 500: {r.prob_beat_spy:.1%}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _empty_result(self) -> MonteCarloResult:
        return MonteCarloResult(
            n_simulations=0,
            cagr_mean=0, cagr_median=0, cagr_p5=0, cagr_p25=0, cagr_p75=0, cagr_p95=0,
            sharpe_mean=0, sharpe_median=0, sharpe_p5=0, sharpe_p25=0, sharpe_p75=0, sharpe_p95=0,
            maxdd_mean=0, maxdd_median=0, maxdd_p5=0, maxdd_p95=0,
            is_significant=False, prob_profitable=0, prob_beat_spy=0,
            report="No trades to simulate"
        )


def run_monte_carlo(
    trade_pnls: List[float],
    n_simulations: int = 1000,
    initial_capital: float = 100_000,
    total_days: int = 2520
) -> MonteCarloResult:
    """Convenience function for Monte Carlo simulation."""
    sim = MonteCarloSimulator(
        n_simulations=n_simulations,
        initial_capital=initial_capital
    )
    return sim.run(trade_pnls, total_days=total_days)
