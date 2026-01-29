#!/usr/bin/env python
"""
Backtest Détaillé - Affiche les trades, positions, sorties et P&L
"""
import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datetime import datetime
from src.backtesting import PortfolioSimulator, BacktestConfig
from src.data.market_data import market_data_fetcher


def main():
    print("=" * 80, flush=True)
    print("BACKTEST DETAILLE - Systeme de Sortie Adaptatif", flush=True)
    print("=" * 80, flush=True)

    # Configuration avec système de sortie adaptatif
    config = BacktestConfig(
        initial_capital=10000,
        max_positions=10,
        position_size_pct=0.10,
        # Sortie adaptative
        use_adaptive_exit=True,
        chandelier_atr_period=22,
        chandelier_multiplier=3.0,
        max_hold_days=120,
        # Dual-timeframe
        use_daily_fallback=True,
        min_ema_conditions_for_fallback=2,
        # Filtres
        min_confidence_score=40,
        require_volume_confirmation=False
    )

    print(f"\nConfiguration:", flush=True)
    print(f"  Capital: ${config.initial_capital:,}", flush=True)
    print(f"  Max positions: {config.max_positions}", flush=True)
    print(f"  Exit system: ADAPTIVE (Chandelier ATR x{config.chandelier_multiplier})", flush=True)
    print(f"  Dual-timeframe: {config.use_daily_fallback}", flush=True)

    # Symboles de test
    symbols = [
        # US Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        # CAC40
        'MC.PA', 'OR.PA', 'AI.PA', 'SAN.PA', 'BNP.PA', 'TTE.PA', 'AIR.PA'
    ]
    print(f"\nSymboles: {len(symbols)}", flush=True)
    for s in symbols:
        print(f"  - {s}", flush=True)

    # Charger données
    print("\n" + "-" * 80, flush=True)
    print("Chargement des données...", flush=True)

    data_weekly = market_data_fetcher.get_batch_historical_data(
        symbols, period='5y', interval='1wk', batch_size=20
    )
    print(f"  Weekly: {len(data_weekly)} symboles charges", flush=True)

    data_daily = market_data_fetcher.get_batch_historical_data(
        symbols, period='5y', interval='1d', batch_size=20
    )
    print(f"  Daily: {len(data_daily)} symboles charges", flush=True)

    # Periode de simulation
    start_date = '2024-01-01'
    end_date = '2024-12-15'

    print("\n" + "-" * 80, flush=True)
    print(f"Simulation: {start_date} to {end_date}", flush=True)
    print("-" * 80, flush=True)

    # Callback de progression
    def progress_callback(current, total, msg):
        if current % 20 == 0 or current == total:
            pct = current / total * 100 if total > 0 else 0
            print(f"  [{pct:5.1f}%] Jour {current}/{total}: {msg}", flush=True)

    # Lancer simulation
    simulator = PortfolioSimulator(config)
    result = simulator.run_simulation(
        all_data=data_weekly,
        start_date=start_date,
        end_date=end_date,
        all_data_daily=data_daily,
        progress_callback=progress_callback
    )

    # ========== RESULTATS ==========
    print("\n" + "=" * 80, flush=True)
    print("RESUME", flush=True)
    print("=" * 80, flush=True)

    print(f"\n--- Statistiques Signaux ---", flush=True)
    print(f"  Signaux detectes:     {result.signals_detected}", flush=True)
    print(f"  Signaux pris:         {result.signals_taken}", flush=True)
    print(f"  Skipped (capital):    {result.signals_skipped_capital}", flush=True)
    print(f"  Skipped (max pos):    {result.signals_skipped_max_positions}", flush=True)
    print(f"  Trades fermes:        {len(result.trades)}", flush=True)

    # Metriques
    m = result.metrics
    print(f"\n--- Performance ---", flush=True)
    print(f"  Capital initial:  ${config.initial_capital:,.2f}", flush=True)
    print(f"  Capital final:    ${m.final_capital:,.2f}", flush=True)
    print(f"  Total Return:     {m.total_return:+.2f}%", flush=True)
    print(f"  Win Rate:         {m.win_rate:.1f}%", flush=True)
    print(f"  Profit Factor:    {m.profit_factor:.2f}", flush=True)
    print(f"  Max Drawdown:     {m.max_drawdown:.2f}%", flush=True)
    print(f"  Sharpe Ratio:     {m.sharpe_ratio:.2f}", flush=True)
    print(f"  Avg Win:          ${m.avg_win:.2f}", flush=True)
    print(f"  Avg Loss:         ${m.avg_loss:.2f}", flush=True)

    # Detail des trades
    if result.trades:
        print("\n" + "=" * 80, flush=True)
        print("DETAIL DES TRADES", flush=True)
        print("=" * 80, flush=True)

        # Compter par raison de sortie
        exit_reasons = {}
        for t in result.trades:
            reason = t.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print("\nRaisons de sortie:", flush=True)
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}", flush=True)

        # Tableau des trades
        print("\n" + "-" * 120, flush=True)
        header = f"{'Symbol':<10} {'Entry':<12} {'Exit':<12} {'Days':<6} {'Entry$':<10} {'Exit$':<10} {'P&L':<12} {'P&L%':<8} {'Reason':<20}"
        print(header, flush=True)
        print("-" * 120, flush=True)

        for t in result.trades:
            # Formater les dates
            try:
                entry_str = t.entry_date.strftime('%Y-%m-%d') if hasattr(t.entry_date, 'strftime') else str(t.entry_date)[:10]
            except:
                entry_str = str(t.entry_date)[:10]

            try:
                exit_str = t.exit_date.strftime('%Y-%m-%d') if hasattr(t.exit_date, 'strftime') else str(t.exit_date)[:10]
            except:
                exit_str = str(t.exit_date)[:10]

            # Calculer la duree
            try:
                from datetime import datetime
                if hasattr(t.entry_date, 'days'):
                    hold_days = (t.exit_date - t.entry_date).days
                else:
                    d1 = pd.to_datetime(t.entry_date)
                    d2 = pd.to_datetime(t.exit_date)
                    hold_days = (d2 - d1).days
            except:
                hold_days = 0

            # Format P&L
            pnl_sign = "+" if t.profit_loss >= 0 else ""
            reason = t.exit_reason or 'unknown'

            line = f"{t.symbol:<10} {entry_str:<12} {exit_str:<12} {hold_days:<6} ${t.entry_price:<9.2f} ${t.exit_price:<9.2f} {pnl_sign}${t.profit_loss:<10.2f} {t.profit_loss_pct:+.1f}%    {reason:<20}"
            print(line, flush=True)

        # Stats par symbole
        print("\n" + "=" * 80, flush=True)
        print("STATS PAR SYMBOLE", flush=True)
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

        # Sauvegarder en CSV
        output_file = f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        print(f"\nTrades sauvegardes: {output_file}", flush=True)

    else:
        print("\nAucun trade ferme dans cette periode.", flush=True)

    # Positions ouvertes
    if hasattr(result, 'open_positions') and result.open_positions:
        print("\n" + "=" * 80, flush=True)
        print(f"POSITIONS OUVERTES ({len(result.open_positions)})", flush=True)
        print("=" * 80, flush=True)

        for pos in result.open_positions:
            print(f"  {pos.symbol}: {pos.shares} actions @ ${pos.entry_price:.2f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("BACKTEST TERMINE", flush=True)
    print("=" * 80 + "\n", flush=True)

    return result


if __name__ == '__main__':
    main()
