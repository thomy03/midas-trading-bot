#!/usr/bin/env python
"""
Full Backtest - 4000+ assets with dual-timeframe
Weekly primary + Daily fallback (if EMAs bullish)

Usage: python run_full_backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timedelta
from src.backtesting import PortfolioSimulator, BacktestConfig
from src.data.market_data import market_data_fetcher

def main():
    print("\n" + "="*70)
    print("FULL BACKTEST - DUAL TIMEFRAME (Weekly + Daily Fallback)")
    print("="*70)

    # Configuration with ADAPTIVE EXIT SYSTEM
    config = BacktestConfig(
        initial_capital=10000,
        max_positions=20,  # Increased from 10
        position_size_pct=0.10,  # 10% per position
        min_confidence_score=55,
        require_volume_confirmation=True,
        min_volume_ratio=1.0,
        # ADAPTIVE EXIT (replaces fixed trailing/take-profit)
        use_adaptive_exit=True,  # Chandelier Exit + Bollinger Squeeze + Volume
        chandelier_atr_period=22,
        chandelier_multiplier=3.0,
        max_hold_days=120,  # Increased safety net
        # Legacy exits disabled
        use_trailing_stop=False,
        take_profit_pct=0.0,
        # Strategy
        use_enhanced_detector=True,
        precision_mode='medium',
        # DUAL TIMEFRAME
        use_daily_fallback=True,
        min_ema_conditions_for_fallback=2
    )

    print(f"\nConfiguration:")
    print(f"  Capital: ${config.initial_capital:,}")
    print(f"  Max positions: {config.max_positions}")
    print(f"  Dual-timeframe: {config.use_daily_fallback}")
    print(f"  Min EMA conditions for daily fallback: {config.min_ema_conditions_for_fallback}")
    print(f"\n  EXIT SYSTEM: {'ADAPTIVE' if config.use_adaptive_exit else 'LEGACY'}")
    if config.use_adaptive_exit:
        print(f"    - Chandelier Exit: ATR({config.chandelier_atr_period}) x {config.chandelier_multiplier}")
        print(f"    - Bollinger Squeeze + Volume confirmation")
        print(f"    - Max hold days: {config.max_hold_days} (safety net)")

    # Gather all symbols
    print("\n" + "-"*70)
    print("Loading symbols...")

    all_symbols = []

    # NASDAQ (~4000)
    nasdaq = market_data_fetcher.get_nasdaq_tickers()
    print(f"  NASDAQ: {len(nasdaq)} symbols")
    all_symbols.extend(nasdaq)

    # S&P 500
    sp500 = market_data_fetcher.get_sp500_tickers()
    print(f"  S&P 500: {len(sp500)} symbols")
    all_symbols.extend(sp500)

    # Europe
    europe = market_data_fetcher.get_european_tickers()
    print(f"  Europe: {len(europe)} symbols")
    all_symbols.extend(europe)

    # CAC40
    cac40 = market_data_fetcher.get_cac40_tickers()
    print(f"  CAC40: {len(cac40)} symbols")
    all_symbols.extend(cac40)

    # DAX
    dax = market_data_fetcher.get_dax_tickers()
    print(f"  DAX: {len(dax)} symbols")
    all_symbols.extend(dax)

    # Remove duplicates
    all_symbols = list(set(all_symbols))
    print(f"\nTotal unique symbols: {len(all_symbols)}")

    # Date range - 2025 (full year)
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 15)

    print(f"\nPeriod: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Load data
    print("\n" + "-"*70)
    print("Loading market data (this may take 15-30 minutes)...")

    # Weekly data (primary)
    print("\n[1/2] Loading WEEKLY data...")
    data_weekly = market_data_fetcher.prefetch_all_markets(
        markets=['NASDAQ', 'SP500', 'EUROPE', 'CAC40', 'DAX'],
        period='5y',
        interval='1wk',
        batch_size=100,
        exclude_crypto=True,
        progress_callback=lambda cur, tot, sym, st: print(f"  [{cur}/{tot}] {sym}: {st}")
    )
    print(f"  -> {len(data_weekly)} symbols loaded")

    # Daily data (fallback)
    print("\n[2/2] Loading DAILY data...")
    data_daily = market_data_fetcher.prefetch_all_markets(
        markets=['NASDAQ', 'SP500', 'EUROPE', 'CAC40', 'DAX'],
        period='5y',
        interval='1d',
        batch_size=100,
        exclude_crypto=True,
        progress_callback=lambda cur, tot, sym, st: print(f"  [{cur}/{tot}] {sym}: {st}")
    )
    print(f"  -> {len(data_daily)} symbols loaded")

    # Run simulation
    print("\n" + "-"*70)
    print("Running DUAL-TIMEFRAME simulation...")
    print("  Weekly signal check first")
    print("  Daily fallback if weekly EMAs bullish (>=2/3 conditions)")
    print("-"*70)

    simulator = PortfolioSimulator(config)

    def progress_cb(current, total, msg):
        pct = current / total * 100 if total > 0 else 0
        print(f"  [{pct:5.1f}%] Day {current}/{total}: {msg}")

    result = simulator.run_simulation(
        all_data=data_weekly,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        progress_callback=progress_cb,
        all_data_daily=data_daily
    )

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n--- Signal Statistics ---")
    print(f"  Signals detected:     {result.signals_detected}")
    print(f"  Signals taken:        {result.signals_taken}")
    print(f"  Skipped (no capital): {result.signals_skipped_capital}")
    print(f"  Skipped (max pos):    {result.signals_skipped_max_positions}")

    print(f"\n--- Performance ---")
    m = result.metrics
    print(f"  Initial Capital:  ${config.initial_capital:,.2f}")
    print(f"  Final Capital:    ${m.final_capital:,.2f}")
    print(f"  Total Return:     {m.total_return:+.2f}%")
    print(f"  Total Trades:     {m.total_trades}")
    print(f"  Win Rate:         {m.win_rate:.1f}%")
    print(f"  Profit Factor:    {m.profit_factor:.2f}")
    print(f"  Max Drawdown:     {m.max_drawdown:.2f}%")
    print(f"  Sharpe Ratio:     {m.sharpe_ratio:.2f}")
    print(f"  Avg Win:          ${m.avg_win:.2f}")
    print(f"  Avg Loss:         ${m.avg_loss:.2f}")

    # Save results
    output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if result.trades:
        import pandas as pd
        trades_df = pd.DataFrame([{
            'symbol': t.symbol,
            'entry_date': t.entry_date,
            'entry_price': t.entry_price,
            'exit_date': t.exit_date,
            'exit_price': t.exit_price,
            'shares': t.shares,
            'profit_loss': t.profit_loss,
            'profit_loss_pct': t.profit_loss_pct,
            'exit_reason': t.exit_reason,
            'signal_strength': t.signal_strength,
            'confidence_score': t.confidence_score
        } for t in result.trades])
        trades_df.to_csv(output_file, index=False)
        print(f"\n  Trades saved to: {output_file}")

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70 + "\n")

    return result


if __name__ == '__main__':
    main()
