#!/usr/bin/env python
"""
Sector Momentum Filter - Comparison Backtest
Compare strategy WITH and WITHOUT sector momentum filter

Period: 2024-01-01 to 2024-12-15 (real data available)
Uses the same 30 diversified symbols as run_fast_backtest.py
"""

import sys
import os

sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datetime import datetime
from src.backtesting import PortfolioSimulator, BacktestConfig
from src.data.market_data import market_data_fetcher


def run_backtest(config: BacktestConfig, data_weekly: dict, data_daily: dict,
                 start_date: str, end_date: str, name: str):
    """Run a single backtest with given config"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {name}")
    print(f"{'='*80}")

    def progress_callback(current, total, msg):
        if current % 10 == 0 or current == total:
            pct = current / total * 100 if total > 0 else 0
            print(f"  [{pct:5.1f}%] Day {current}/{total}: {msg}", flush=True)

    simulator = PortfolioSimulator(config)
    result = simulator.run_simulation(
        all_data=data_weekly,
        start_date=start_date,
        end_date=end_date,
        all_data_daily=data_daily,
        progress_callback=progress_callback
    )

    return result


def print_comparison(result_without: 'SimulationResult', result_with: 'SimulationResult'):
    """Print side-by-side comparison"""
    m_without = result_without.metrics
    m_with = result_with.metrics

    print("\n" + "=" * 100)
    print("COMPARISON: WITHOUT vs WITH SECTOR FILTER")
    print("=" * 100)

    def fmt_diff(without_val, with_val, pct=False, higher_better=True):
        diff = with_val - without_val
        if pct:
            sign = "+" if diff >= 0 else ""
            indicator = "+" if (diff > 0 and higher_better) or (diff < 0 and not higher_better) else "-" if diff != 0 else "="
            return f"{sign}{diff:.2f}% [{indicator}]"
        else:
            sign = "+" if diff >= 0 else ""
            indicator = "+" if (diff > 0 and higher_better) or (diff < 0 and not higher_better) else "-" if diff != 0 else "="
            return f"{sign}{diff:.2f} [{indicator}]"

    print(f"\n{'Metric':<25} {'WITHOUT Sector':<20} {'WITH Sector':<20} {'Difference':<20}")
    print("-" * 85)

    # Performance metrics
    print(f"{'Total Return (%)':<25} {m_without.total_return:>+18.2f}% {m_with.total_return:>+18.2f}% {fmt_diff(m_without.total_return, m_with.total_return, pct=True):<20}")
    print(f"{'Final Capital ($)':<25} ${m_without.final_capital:>16,.2f} ${m_with.final_capital:>16,.2f} ${m_with.final_capital - m_without.final_capital:>+.2f}")
    print(f"{'Win Rate (%)':<25} {m_without.win_rate:>18.1f}% {m_with.win_rate:>18.1f}% {fmt_diff(m_without.win_rate, m_with.win_rate, pct=True):<20}")
    print(f"{'Profit Factor':<25} {m_without.profit_factor:>19.2f} {m_with.profit_factor:>19.2f} {fmt_diff(m_without.profit_factor, m_with.profit_factor):<20}")
    print(f"{'Max Drawdown (%)':<25} {m_without.max_drawdown:>18.2f}% {m_with.max_drawdown:>18.2f}% {fmt_diff(m_without.max_drawdown, m_with.max_drawdown, pct=True, higher_better=False):<20}")
    print(f"{'Sharpe Ratio':<25} {m_without.sharpe_ratio:>19.2f} {m_with.sharpe_ratio:>19.2f} {fmt_diff(m_without.sharpe_ratio, m_with.sharpe_ratio):<20}")
    print(f"{'Avg Win ($)':<25} ${m_without.avg_win:>17.2f} ${m_with.avg_win:>17.2f} ${m_with.avg_win - m_without.avg_win:>+.2f}")
    print(f"{'Avg Loss ($)':<25} ${m_without.avg_loss:>17.2f} ${m_with.avg_loss:>17.2f} ${m_with.avg_loss - m_without.avg_loss:>+.2f}")

    # Signal statistics
    print(f"\n{'--- Signal Statistics ---':<85}")
    print(f"{'Signals Detected':<25} {result_without.signals_detected:>19} {result_with.signals_detected:>19}")
    print(f"{'Signals Taken':<25} {result_without.signals_taken:>19} {result_with.signals_taken:>19}")
    print(f"{'Trades Closed':<25} {len(result_without.trades):>19} {len(result_with.trades):>19}")
    print(f"{'Skipped (Capital)':<25} {result_without.signals_skipped_capital:>19} {result_with.signals_skipped_capital:>19}")
    print(f"{'Skipped (Max Pos)':<25} {result_without.signals_skipped_max_positions:>19} {result_with.signals_skipped_max_positions:>19}")
    print(f"{'Skipped (Sector)':<25} {'N/A':>19} {result_with.signals_skipped_sector:>19}")

    # Conclusion
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)

    return_diff = m_with.total_return - m_without.total_return
    if return_diff > 0:
        print(f"  Sector filter IMPROVED returns by {return_diff:+.2f}%")
    else:
        print(f"  Sector filter REDUCED returns by {return_diff:.2f}%")

    drawdown_diff = m_with.max_drawdown - m_without.max_drawdown
    if drawdown_diff < 0:
        print(f"  Sector filter REDUCED drawdown by {abs(drawdown_diff):.2f}%")
    else:
        print(f"  Sector filter INCREASED drawdown by {drawdown_diff:.2f}%")

    win_rate_diff = m_with.win_rate - m_without.win_rate
    if win_rate_diff > 0:
        print(f"  Sector filter IMPROVED win rate by {win_rate_diff:+.1f}%")
    else:
        print(f"  Sector filter REDUCED win rate by {win_rate_diff:.1f}%")


def main():
    print("=" * 80, flush=True)
    print("SECTOR MOMENTUM FILTER - COMPARISON BACKTEST", flush=True)
    print("=" * 80, flush=True)

    # Common configuration
    base_config = dict(
        initial_capital=10000,
        max_positions=15,
        position_size_pct=0.10,
        # Adaptive exit
        use_adaptive_exit=True,
        chandelier_atr_period=22,
        chandelier_multiplier=3.0,
        max_hold_days=120,
        # Dual-timeframe
        use_daily_fallback=True,
        min_ema_conditions_for_fallback=2,
        # Filters
        min_confidence_score=40,
        require_volume_confirmation=False,
        min_volume_ratio=0.8,
        # Standard detector (faster)
        use_enhanced_detector=False,
        precision_mode='low'
    )

    # 30 diversified symbols
    symbols = [
        # US Tech (10)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        # US Value/Industrial (5)
        'JPM', 'JNJ', 'XOM', 'PG', 'KO',
        # CAC40 (8)
        'MC.PA', 'OR.PA', 'AI.PA', 'SAN.PA', 'BNP.PA',
        'TTE.PA', 'AIR.PA', 'SU.PA',
        # DAX (7)
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE',
        'BAS.DE', 'BMW.DE', 'MBG.DE'
    ]

    # Period: 2024 (real data available)
    start_date = '2024-01-01'
    end_date = '2024-12-15'

    print(f"\nSymbols: {len(symbols)}", flush=True)
    print(f"Period: {start_date} to {end_date}", flush=True)

    # Load data once for both tests
    print("\n" + "-" * 80, flush=True)
    print("Loading market data...", flush=True)

    data_weekly = market_data_fetcher.get_batch_historical_data(
        symbols, period='5y', interval='1wk', batch_size=30
    )
    print(f"  Weekly: {len(data_weekly)} symbols loaded", flush=True)

    data_daily = market_data_fetcher.get_batch_historical_data(
        symbols, period='5y', interval='1d', batch_size=30
    )
    print(f"  Daily: {len(data_daily)} symbols loaded", flush=True)

    # ========== TEST 1: WITHOUT SECTOR FILTER ==========
    config_without = BacktestConfig(
        **base_config,
        use_sector_filter=False
    )

    result_without = run_backtest(
        config=config_without,
        data_weekly=data_weekly,
        data_daily=data_daily,
        start_date=start_date,
        end_date=end_date,
        name="WITHOUT Sector Filter"
    )

    # ========== TEST 2: WITH SECTOR FILTER ==========
    config_with = BacktestConfig(
        **base_config,
        use_sector_filter=True,
        min_sector_momentum=0.0  # Sector must be >= SPY performance
    )

    result_with = run_backtest(
        config=config_with,
        data_weekly=data_weekly,
        data_daily=data_daily,
        start_date=start_date,
        end_date=end_date,
        name="WITH Sector Filter (vs SPY >= 0%)"
    )

    # ========== COMPARISON ==========
    print_comparison(result_without, result_with)

    # Save trades to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if result_without.trades:
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'entry_date': t.entry_date,
            'entry_price': t.entry_price,
            'exit_date': t.exit_date,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'profit_loss': t.profit_loss,
            'profit_loss_pct': t.profit_loss_pct,
            'exit_reason': t.exit_reason
        } for t in result_without.trades])
        trades_df.to_csv(f"backtest_no_sector_{timestamp}.csv", index=False)
        print(f"\nTrades (without filter) saved: backtest_no_sector_{timestamp}.csv")

    if result_with.trades:
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'entry_date': t.entry_date,
            'entry_price': t.entry_price,
            'exit_date': t.exit_date,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'profit_loss': t.profit_loss,
            'profit_loss_pct': t.profit_loss_pct,
            'exit_reason': t.exit_reason
        } for t in result_with.trades])
        trades_df.to_csv(f"backtest_with_sector_{timestamp}.csv", index=False)
        print(f"Trades (with filter) saved: backtest_with_sector_{timestamp}.csv")

    print("\n" + "=" * 80, flush=True)
    print("COMPARISON BACKTEST COMPLETE", flush=True)
    print("=" * 80 + "\n", flush=True)

    return result_without, result_with


if __name__ == '__main__':
    main()
