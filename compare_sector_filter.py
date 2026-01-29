#!/usr/bin/env python
"""
Sector Filter Comparison Script
Compare strategy performance WITH vs WITHOUT sector momentum filter

Uses 2022 (bear market with corrections) for more signal generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.backtesting import PortfolioSimulator, BacktestConfig
from src.data.market_data import market_data_fetcher


def run_comparison():
    """Run backtest comparison: WITH vs WITHOUT sector filter"""

    # Diversified symbols (US + EU)
    symbols = [
        # US Tech
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        # US Value
        'JPM', 'JNJ', 'XOM', 'PG', 'KO',
        # EU
        'MC.PA', 'SAP.DE', 'SIE.DE', 'BNP.PA', 'TTE.PA', 'AIR.PA'
    ]

    print("=" * 80)
    print("SECTOR FILTER COMPARISON")
    print("=" * 80)
    print(f"\nSymbols: {len(symbols)}")

    # Load data
    print("\nLoading data...")
    data_weekly = market_data_fetcher.get_batch_historical_data(
        symbols, period='5y', interval='1wk', batch_size=25
    )
    data_daily = market_data_fetcher.get_batch_historical_data(
        symbols, period='5y', interval='1d', batch_size=25
    )
    print(f"  Loaded: {len(data_weekly)} weekly, {len(data_daily)} daily")

    # Common configuration
    base_config = dict(
        initial_capital=10000,
        max_positions=10,
        position_size_pct=0.10,
        use_adaptive_exit=True,
        chandelier_atr_period=22,
        chandelier_multiplier=3.0,
        max_hold_days=90,
        use_daily_fallback=True,
        min_ema_conditions_for_fallback=2,
        min_confidence_score=35,  # Lower threshold for more signals
        use_enhanced_detector=False,
        precision_mode='low'
    )

    # Test periods (years with different market conditions)
    test_periods = [
        ('2022-01-01', '2022-12-31', '2022 (Bear Market)'),
        ('2023-01-01', '2023-12-31', '2023 (Recovery)'),
        ('2024-01-01', '2024-12-15', '2024 (Bull Market)'),
    ]

    results = []

    for start_date, end_date, period_name in test_periods:
        print(f"\n{'='*80}")
        print(f"PERIOD: {period_name}")
        print(f"{'='*80}")

        # Test 1: WITHOUT sector filter
        print(f"\n[1/2] Running WITHOUT sector filter...")
        config1 = BacktestConfig(**base_config, use_sector_filter=False)
        sim1 = PortfolioSimulator(config1)
        result1 = sim1.run_simulation(
            all_data=data_weekly,
            start_date=start_date,
            end_date=end_date,
            all_data_daily=data_daily
        )

        # Test 2: WITH sector filter
        print(f"\n[2/2] Running WITH sector filter...")
        config2 = BacktestConfig(**base_config, use_sector_filter=True, min_sector_momentum=0.0)
        sim2 = PortfolioSimulator(config2)
        result2 = sim2.run_simulation(
            all_data=data_weekly,
            start_date=start_date,
            end_date=end_date,
            all_data_daily=data_daily
        )

        results.append({
            'period': period_name,
            'without': result1,
            'with': result2
        })

        # Quick summary
        m1, m2 = result1.metrics, result2.metrics
        print(f"\n{period_name} Summary:")
        print(f"  WITHOUT: {result1.signals_detected} signals, {len(result1.trades)} trades, {m1.total_return:+.2f}%, Win {m1.win_rate:.0f}%")
        print(f"  WITH:    {result2.signals_detected} signals, {len(result2.trades)} trades, {m2.total_return:+.2f}%, Win {m2.win_rate:.0f}%, Blocked {result2.signals_skipped_sector}")

    # ========== FINAL COMPARISON ==========
    print("\n" + "=" * 100)
    print("FINAL COMPARISON: WITHOUT vs WITH SECTOR FILTER")
    print("=" * 100)

    print(f"\n{'Period':<20} {'Metric':<15} {'WITHOUT':>12} {'WITH':>12} {'Diff':>12} {'Better':>10}")
    print("-" * 85)

    total_return_without = 0
    total_return_with = 0
    total_trades_without = 0
    total_trades_with = 0
    total_wins_without = 0
    total_wins_with = 0

    for r in results:
        m1 = r['without'].metrics
        m2 = r['with'].metrics

        # Return comparison
        ret_diff = m2.total_return - m1.total_return
        ret_better = "WITH" if ret_diff > 0 else "WITHOUT" if ret_diff < 0 else "EQUAL"
        print(f"{r['period']:<20} {'Return %':<15} {m1.total_return:>+11.2f}% {m2.total_return:>+11.2f}% {ret_diff:>+11.2f}% {ret_better:>10}")

        # Win rate comparison
        wr_diff = m2.win_rate - m1.win_rate
        wr_better = "WITH" if wr_diff > 0 else "WITHOUT" if wr_diff < 0 else "EQUAL"
        print(f"{'':<20} {'Win Rate %':<15} {m1.win_rate:>11.1f}% {m2.win_rate:>11.1f}% {wr_diff:>+11.1f}% {wr_better:>10}")

        # Trades
        trades1 = len(r['without'].trades)
        trades2 = len(r['with'].trades)
        blocked = r['with'].signals_skipped_sector
        print(f"{'':<20} {'Trades':<15} {trades1:>12} {trades2:>12} {trades2-trades1:>+12} {'(blocked: ' + str(blocked) + ')'}")

        # Drawdown
        dd_diff = m2.max_drawdown - m1.max_drawdown
        dd_better = "WITH" if dd_diff < 0 else "WITHOUT" if dd_diff > 0 else "EQUAL"
        print(f"{'':<20} {'Max DD %':<15} {m1.max_drawdown:>11.2f}% {m2.max_drawdown:>11.2f}% {dd_diff:>+11.2f}% {dd_better:>10}")

        print("-" * 85)

        # Accumulate totals
        total_return_without += m1.total_return
        total_return_with += m2.total_return
        total_trades_without += trades1
        total_trades_with += trades2

        # Count wins
        for t in r['without'].trades:
            if t.profit_loss > 0:
                total_wins_without += 1
        for t in r['with'].trades:
            if t.profit_loss > 0:
                total_wins_with += 1

    # Overall summary
    print(f"\n{'TOTAL':<20} {'Return %':<15} {total_return_without:>+11.2f}% {total_return_with:>+11.2f}% {total_return_with - total_return_without:>+11.2f}%")

    overall_wr_without = (total_wins_without / total_trades_without * 100) if total_trades_without > 0 else 0
    overall_wr_with = (total_wins_with / total_trades_with * 100) if total_trades_with > 0 else 0
    print(f"{'':<20} {'Win Rate %':<15} {overall_wr_without:>11.1f}% {overall_wr_with:>11.1f}% {overall_wr_with - overall_wr_without:>+11.1f}%")
    print(f"{'':<20} {'Total Trades':<15} {total_trades_without:>12} {total_trades_with:>12}")

    # Conclusion
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)

    if total_return_with > total_return_without:
        print(f"  The sector filter IMPROVED cumulative returns by {total_return_with - total_return_without:+.2f}%")
    else:
        print(f"  The sector filter REDUCED cumulative returns by {total_return_with - total_return_without:.2f}%")

    if overall_wr_with > overall_wr_without:
        print(f"  The sector filter IMPROVED overall win rate by {overall_wr_with - overall_wr_without:+.1f}%")
    else:
        print(f"  The sector filter REDUCED overall win rate by {overall_wr_with - overall_wr_without:.1f}%")

    blocked_total = sum(r['with'].signals_skipped_sector for r in results)
    print(f"  Total signals blocked by sector filter: {blocked_total}")

    return results


if __name__ == '__main__':
    run_comparison()
