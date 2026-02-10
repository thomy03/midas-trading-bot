"""
V8 Macro Regime Backtest - Compare regime detection with/without macro signals.

Tests whether adding yield curve, credit spreads, dollar strength, and
defensive rotation improves regime detection and overall backtest performance.
"""

import json
import time
import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING)


def run_config(label, config, symbols, shared_data_mgr, shared_scorer):
    from src.backtesting.v6_backtester import V6Backtester

    start_date = '2015-01-01'
    end_date = '2025-01-01'

    bt = V6Backtester(config, data_mgr=shared_data_mgr)
    bt.tech_scorer = shared_scorer
    result = bt.run(symbols, start_date, end_date)
    m = result.metrics
    years = max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
    cagr = ((1 + m.total_return / 100) ** (1 / years) - 1) * 100

    exits = {}
    for t in result.trades:
        exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

    return {
        'label': label,
        'cagr': round(cagr, 2),
        'sharpe': round(m.sharpe_ratio, 2),
        'sortino': round(m.sortino_ratio, 2),
        'max_dd': round(m.max_drawdown, 2),
        'win_rate': round(m.win_rate, 1),
        'pf': round(m.profit_factor, 2),
        'trades': m.total_trades,
        'regime_history': result.regime_history,
        'exits': exits,
    }


def main():
    from src.backtesting.v6_backtester import (
        V6Backtester, V6BacktestConfig, BacktestDataManager, BacktestTechnicalScorer,
        make_profile_config, SYMBOLS_DIVERSIFIED, SYMBOLS_TECH_HEAVY
    )
    import copy

    print("=" * 70)
    print("V8 MACRO REGIME BACKTEST")
    print("Comparing regime detection WITH and WITHOUT macro signals")
    print("=" * 70)

    shared_data_mgr = BacktestDataManager()
    shared_scorer = BacktestTechnicalScorer()

    results = []

    for profile_name in ['growth', 'safe']:
        base_config, symbols = make_profile_config(profile_name)

        # Precompute indicators once
        buffer_start = '2014-01-01'
        end_date = '2025-01-01'
        symbol_data = {}
        for sym in symbols:
            df = shared_data_mgr.get_ohlcv(sym, buffer_start, end_date)
            if df is not None and len(df) >= 100:
                symbol_data[sym] = df
                shared_scorer.precompute(sym, df)

        print("\n%s PROFILE (%d symbols)" % (profile_name.upper(), len(symbols)))
        print("-" * 50)

        # Without macro
        t0 = time.time()
        cfg_no_macro = copy.deepcopy(base_config)
        cfg_no_macro.macro_regime_enabled = False
        r1 = run_config(
            '%s (no macro)' % profile_name,
            cfg_no_macro, symbols, shared_data_mgr, shared_scorer
        )
        print("  %-25s CAGR=%6.2f%%  Sharpe=%5.2f  MaxDD=%5.1f%%  Trades=%d  [%.0fs]" % (
            r1['label'], r1['cagr'], r1['sharpe'], r1['max_dd'], r1['trades'], time.time() - t0))
        results.append(r1)

        # With macro
        t0 = time.time()
        cfg_macro = copy.deepcopy(base_config)
        cfg_macro.macro_regime_enabled = True
        r2 = run_config(
            '%s (+ macro)' % profile_name,
            cfg_macro, symbols, shared_data_mgr, shared_scorer
        )
        print("  %-25s CAGR=%6.2f%%  Sharpe=%5.2f  MaxDD=%5.1f%%  Trades=%d  [%.0fs]" % (
            r2['label'], r2['cagr'], r2['sharpe'], r2['max_dd'], r2['trades'], time.time() - t0))
        results.append(r2)

        # Delta
        d_cagr = r2['cagr'] - r1['cagr']
        d_sharpe = r2['sharpe'] - r1['sharpe']
        d_dd = r2['max_dd'] - r1['max_dd']
        print("  DELTA:                    CAGR=%+.2f%%  Sharpe=%+.2f  MaxDD=%+.1f%%" % (
            d_cagr, d_sharpe, d_dd))

        # Regime distribution
        print("  Regime (no macro): %s" % r1.get('regime_history', {}))
        print("  Regime (+ macro):  %s" % r2.get('regime_history', {}))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print("  %-25s CAGR=%6.2f%%  Sharpe=%5.2f  MaxDD=%5.1f%%" % (
            r['label'], r['cagr'], r['sharpe'], r['max_dd']))

    # Save results
    with open('macro_regime_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to macro_regime_results.json")


if __name__ == '__main__':
    main()
