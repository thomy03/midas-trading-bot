#!/usr/bin/env python
"""
Fast Backtest - 30 diversified symbols
Period: 2025-01-01 to 2025-12-15
Uses standard detector (faster) with adaptive exit
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


def main():
    print("=" * 80, flush=True)
    print("FAST BACKTEST - 30 Diversified Symbols (2025)", flush=True)
    print("=" * 80, flush=True)

    # Configuration with adaptive exit but standard (faster) detector
    config = BacktestConfig(
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
        # Filters - more permissive
        min_confidence_score=40,
        require_volume_confirmation=False,
        min_volume_ratio=0.8,
        # Use STANDARD detector (faster than enhanced)
        use_enhanced_detector=False,
        precision_mode='low'  # Fastest mode
    )

    print(f"\nConfiguration:", flush=True)
    print(f"  Capital: ${config.initial_capital:,}", flush=True)
    print(f"  Max positions: {config.max_positions}", flush=True)
    print(f"  Exit system: ADAPTIVE (Chandelier ATR x{config.chandelier_multiplier})", flush=True)
    print(f"  Detector: STANDARD (fast)", flush=True)
    print(f"  Dual-timeframe: {config.use_daily_fallback}", flush=True)

    # 30 diversified symbols - mix of US Tech, Value, CAC40, DAX
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

    print(f"\nSymbols: {len(symbols)}", flush=True)
    print("  US Tech: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, NFLX, CRM", flush=True)
    print("  US Value: JPM, JNJ, XOM, PG, KO", flush=True)
    print("  CAC40: MC.PA, OR.PA, AI.PA, SAN.PA, BNP.PA, TTE.PA, AIR.PA, SU.PA", flush=True)
    print("  DAX: SAP.DE, SIE.DE, ALV.DE, DTE.DE, BAS.DE, BMW.DE, MBG.DE", flush=True)

    # Period
    start_date = '2025-01-01'
    end_date = '2025-12-15'

    print(f"\nPeriod: {start_date} to {end_date}", flush=True)

    # Load data
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

    # Run simulation
    print("\n" + "-" * 80, flush=True)
    print("Running DUAL-TIMEFRAME simulation...", flush=True)
    print("-" * 80, flush=True)

    def progress_callback(current, total, msg):
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

    # ========== RESULTS ==========
    print("\n" + "=" * 80, flush=True)
    print("RESULTS", flush=True)
    print("=" * 80, flush=True)

    print(f"\n--- Signal Statistics ---", flush=True)
    print(f"  Signals detected:     {result.signals_detected}", flush=True)
    print(f"  Signals taken:        {result.signals_taken}", flush=True)
    print(f"  Skipped (capital):    {result.signals_skipped_capital}", flush=True)
    print(f"  Skipped (max pos):    {result.signals_skipped_max_positions}", flush=True)
    print(f"  Trades closed:        {len(result.trades)}", flush=True)

    # Metrics
    m = result.metrics
    print(f"\n--- Performance ---", flush=True)
    print(f"  Initial Capital:  ${config.initial_capital:,.2f}", flush=True)
    print(f"  Final Capital:    ${m.final_capital:,.2f}", flush=True)
    print(f"  Total Return:     {m.total_return:+.2f}%", flush=True)
    print(f"  Win Rate:         {m.win_rate:.1f}%", flush=True)
    print(f"  Profit Factor:    {m.profit_factor:.2f}", flush=True)
    print(f"  Max Drawdown:     {m.max_drawdown:.2f}%", flush=True)
    print(f"  Sharpe Ratio:     {m.sharpe_ratio:.2f}", flush=True)
    print(f"  Avg Win:          ${m.avg_win:.2f}", flush=True)
    print(f"  Avg Loss:         ${m.avg_loss:.2f}", flush=True)

    # Trade details
    if result.trades:
        print("\n" + "=" * 80, flush=True)
        print("TRADE DETAILS", flush=True)
        print("=" * 80, flush=True)

        # Exit reasons count
        exit_reasons = {}
        for t in result.trades:
            reason = t.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print("\nExit reasons:", flush=True)
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}", flush=True)

        # Trade table
        print("\n" + "-" * 120, flush=True)
        header = f"{'Symbol':<10} {'Entry':<12} {'Exit':<12} {'Days':<6} {'Entry$':<10} {'Exit$':<10} {'P&L':<12} {'P&L%':<8} {'Reason':<20}"
        print(header, flush=True)
        print("-" * 120, flush=True)

        for t in result.trades:
            try:
                entry_str = t.entry_date.strftime('%Y-%m-%d') if hasattr(t.entry_date, 'strftime') else str(t.entry_date)[:10]
            except:
                entry_str = str(t.entry_date)[:10]

            try:
                exit_str = t.exit_date.strftime('%Y-%m-%d') if hasattr(t.exit_date, 'strftime') else str(t.exit_date)[:10]
            except:
                exit_str = str(t.exit_date)[:10]

            try:
                d1 = pd.to_datetime(t.entry_date)
                d2 = pd.to_datetime(t.exit_date)
                hold_days = (d2 - d1).days
            except:
                hold_days = 0

            pnl_sign = "+" if t.profit_loss >= 0 else ""
            reason = t.exit_reason or 'unknown'

            line = f"{t.symbol:<10} {entry_str:<12} {exit_str:<12} {hold_days:<6} ${t.entry_price:<9.2f} ${t.exit_price:<9.2f} {pnl_sign}${t.profit_loss:<10.2f} {t.profit_loss_pct:+.1f}%    {reason:<20}"
            print(line, flush=True)

        # Stats by symbol
        print("\n" + "=" * 80, flush=True)
        print("STATS BY SYMBOL", flush=True)
        print("=" * 80, flush=True)

        symbol_stats = {}
        for t in result.trades:
            if t.symbol not in symbol_stats:
                symbol_stats[t.symbol] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            symbol_stats[t.symbol]['trades'] += 1
            symbol_stats[t.symbol]['pnl'] += t.profit_loss
            if t.profit_loss > 0:
                symbol_stats[t.symbol]['wins'] += 1

        print(f"\n{'Symbol':<10} {'Trades':<8} {'Wins':<8} {'Win%':<10} {'Total P&L':<15}", flush=True)
        print("-" * 55, flush=True)

        for sym, stats in sorted(symbol_stats.items(), key=lambda x: -x[1]['pnl']):
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            pnl_sign = "+" if stats['pnl'] >= 0 else ""
            line = f"{sym:<10} {stats['trades']:<8} {stats['wins']:<8} {win_rate:<9.1f}% {pnl_sign}${stats['pnl']:<14.2f}"
            print(line, flush=True)

        # Save to CSV
        output_file = f"backtest_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
            'signal_strength': getattr(t, 'signal_strength', None),
            'confidence_score': getattr(t, 'confidence_score', None)
        } for t in result.trades])
        trades_df.to_csv(output_file, index=False)
        print(f"\nTrades saved: {output_file}", flush=True)

    else:
        print("\nNo trades closed during this period.", flush=True)

    # Open positions
    if hasattr(result, 'open_positions') and result.open_positions:
        print("\n" + "=" * 80, flush=True)
        print(f"OPEN POSITIONS ({len(result.open_positions)})", flush=True)
        print("=" * 80, flush=True)

        for pos in result.open_positions:
            print(f"  {pos.symbol}: {pos.shares} shares @ ${pos.entry_price:.2f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("BACKTEST COMPLETE", flush=True)
    print("=" * 80 + "\n", flush=True)

    return result


if __name__ == '__main__':
    main()
