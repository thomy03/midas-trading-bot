#!/usr/bin/env python
"""
Compare Strategy vs Benchmarks (SPY, QQQ)
Compares our trading strategy against buy-and-hold SPY and QQQ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datetime import datetime
from src.backtesting import PortfolioSimulator, BacktestConfig
from src.data.market_data import market_data_fetcher


def calculate_benchmark_return(symbol: str, start_date: str, end_date: str) -> dict:
    """Calculate buy-and-hold return for a benchmark"""
    data = market_data_fetcher.get_historical_data(symbol, period='5y', interval='1d')

    if data is None or data.empty:
        return {'return': 0, 'max_drawdown': 0}

    # Convert index to timezone-naive for comparison
    if data.index.tz is not None:
        data = data.copy()
        data.index = data.index.tz_localize(None)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Filter to date range
    mask = (data.index >= start_dt) & (data.index <= end_dt)
    period_data = data[mask]

    if len(period_data) < 2:
        return {'return': 0, 'max_drawdown': 0}

    # Buy and hold return
    start_price = period_data['Close'].iloc[0]
    end_price = period_data['Close'].iloc[-1]
    total_return = ((end_price / start_price) - 1) * 100

    # Calculate max drawdown
    cummax = period_data['Close'].cummax()
    drawdown = (period_data['Close'] - cummax) / cummax * 100
    max_drawdown = abs(drawdown.min())

    return {
        'return': total_return,
        'max_drawdown': max_drawdown
    }


def run_comparison():
    """Run full comparison: Strategy vs SPY vs QQQ"""

    # Trading symbols
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        'JPM', 'JNJ', 'XOM', 'PG', 'KO',
        'MC.PA', 'SAP.DE', 'SIE.DE', 'BNP.PA', 'TTE.PA', 'AIR.PA'
    ]

    print("=" * 100)
    print("STRATEGY vs BENCHMARKS COMPARISON")
    print("=" * 100)
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

    # Strategy configuration
    config = BacktestConfig(
        initial_capital=10000,
        max_positions=10,
        position_size_pct=0.10,
        use_adaptive_exit=True,
        chandelier_atr_period=22,
        chandelier_multiplier=3.0,
        max_hold_days=90,
        use_daily_fallback=True,
        min_ema_conditions_for_fallback=2,
        min_confidence_score=35,
        use_enhanced_detector=False,
        precision_mode='low',
        use_sector_filter=False  # Without sector filter for now
    )

    # Test periods
    test_periods = [
        ('2022-01-01', '2022-12-31', '2022 (Bear)'),
        ('2023-01-01', '2023-12-31', '2023 (Recovery)'),
        ('2024-01-01', '2024-12-15', '2024 (Bull)'),
    ]

    results = []

    for start_date, end_date, period_name in test_periods:
        print(f"\n{'='*100}")
        print(f"PERIOD: {period_name}")
        print(f"{'='*100}")

        # Calculate benchmark returns
        print("\nCalculating benchmarks...")
        spy_result = calculate_benchmark_return('SPY', start_date, end_date)
        qqq_result = calculate_benchmark_return('QQQ', start_date, end_date)

        print(f"  SPY: {spy_result['return']:+.2f}% (Max DD: {spy_result['max_drawdown']:.2f}%)")
        print(f"  QQQ: {qqq_result['return']:+.2f}% (Max DD: {qqq_result['max_drawdown']:.2f}%)")

        # Run strategy
        print("\nRunning strategy...")
        simulator = PortfolioSimulator(config)
        result = simulator.run_simulation(
            all_data=data_weekly,
            start_date=start_date,
            end_date=end_date,
            all_data_daily=data_daily
        )

        print(f"  Strategy: {result.metrics.total_return:+.2f}% (Max DD: {result.metrics.max_drawdown:.2f}%)")
        print(f"  Trades: {len(result.trades)}, Win Rate: {result.metrics.win_rate:.1f}%")

        results.append({
            'period': period_name,
            'strategy': result,
            'spy': spy_result,
            'qqq': qqq_result
        })

    # ========== FINAL COMPARISON ==========
    print("\n" + "=" * 100)
    print("FINAL COMPARISON: STRATEGY vs BENCHMARKS")
    print("=" * 100)

    print(f"\n{'Period':<20} {'Strategy':>12} {'SPY':>12} {'QQQ':>12} {'vs SPY':>12} {'vs QQQ':>12}")
    print("-" * 85)

    total_strategy = 0
    total_spy = 0
    total_qqq = 0

    for r in results:
        strat_ret = r['strategy'].metrics.total_return
        spy_ret = r['spy']['return']
        qqq_ret = r['qqq']['return']

        vs_spy = strat_ret - spy_ret
        vs_qqq = strat_ret - qqq_ret

        print(f"{r['period']:<20} {strat_ret:>+11.2f}% {spy_ret:>+11.2f}% {qqq_ret:>+11.2f}% {vs_spy:>+11.2f}% {vs_qqq:>+11.2f}%")

        total_strategy += strat_ret
        total_spy += spy_ret
        total_qqq += qqq_ret

    print("-" * 85)
    print(f"{'CUMULATIVE':<20} {total_strategy:>+11.2f}% {total_spy:>+11.2f}% {total_qqq:>+11.2f}% {total_strategy - total_spy:>+11.2f}% {total_strategy - total_qqq:>+11.2f}%")

    # Strategy metrics summary
    print("\n" + "=" * 100)
    print("STRATEGY METRICS BY PERIOD")
    print("=" * 100)

    print(f"\n{'Period':<20} {'Return':>10} {'Win Rate':>10} {'Trades':>10} {'Profit F.':>10} {'Max DD':>10}")
    print("-" * 75)

    total_trades = 0
    total_wins = 0

    for r in results:
        m = r['strategy'].metrics
        trades = len(r['strategy'].trades)
        wins = sum(1 for t in r['strategy'].trades if t.profit_loss > 0)

        print(f"{r['period']:<20} {m.total_return:>+9.2f}% {m.win_rate:>9.1f}% {trades:>10} {m.profit_factor:>10.2f} {m.max_drawdown:>9.2f}%")

        total_trades += trades
        total_wins += wins

    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print("-" * 75)
    print(f"{'TOTAL':<20} {total_strategy:>+9.2f}% {overall_wr:>9.1f}% {total_trades:>10}")

    # Conclusion
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)

    if total_strategy > total_spy:
        print(f"  Strategy OUTPERFORMS SPY by {total_strategy - total_spy:+.2f}%")
    else:
        print(f"  Strategy UNDERPERFORMS SPY by {total_strategy - total_spy:.2f}%")

    if total_strategy > total_qqq:
        print(f"  Strategy OUTPERFORMS QQQ by {total_strategy - total_qqq:+.2f}%")
    else:
        print(f"  Strategy UNDERPERFORMS QQQ by {total_strategy - total_qqq:.2f}%")

    print(f"\n  Total trades executed: {total_trades}")
    print(f"  Overall win rate: {overall_wr:.1f}%")

    return results


if __name__ == '__main__':
    run_comparison()
