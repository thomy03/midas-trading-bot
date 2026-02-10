"""
V8 Thematic ETF Momentum Backtest - Measure impact of ETF momentum bonus.

Tests whether adding thematic ETF momentum (SMH, XBI, XLK, etc. vs SPY)
as a scoring bonus improves backtest performance.
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
        'exits': exits,
    }


def main():
    from src.backtesting.v6_backtester import (
        V6Backtester, V6BacktestConfig, BacktestDataManager, BacktestTechnicalScorer,
        make_profile_config, SYMBOLS_DIVERSIFIED, SYMBOLS_TECH_HEAVY
    )
    import copy

    print("=" * 70)
    print("V8 THEMATIC ETF MOMENTUM BACKTEST")
    print("Comparing scoring WITH and WITHOUT thematic ETF momentum bonus")
    print("=" * 70)

    shared_data_mgr = BacktestDataManager()
    shared_scorer = BacktestTechnicalScorer()

    results = []

    for profile_name in ['growth', 'safe']:
        base_config, symbols = make_profile_config(profile_name)

        # Precompute indicators once
        buffer_start = '2014-01-01'
        end_date = '2025-01-01'
        for sym in symbols:
            df = shared_data_mgr.get_ohlcv(sym, buffer_start, end_date)
            if df is not None and len(df) >= 100:
                shared_scorer.precompute(sym, df)

        # Pre-fetch ETF data for thematic bonus (direct import to bypass __init__.py)
        import importlib.util as _ilu
        _orch_path = 'src/intelligence/intelligence_orchestrator.py'
        _spec = _ilu.spec_from_file_location("intelligence_orchestrator", _orch_path)
        _orch_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_orch_mod)
        THEME_ETF_PROXIES = _orch_mod.THEME_ETF_PROXIES

        print("\nPre-fetching thematic ETF data...")
        for etf_sym in THEME_ETF_PROXIES:
            shared_data_mgr.get_ohlcv(etf_sym, buffer_start, end_date)
        # Also need SPY
        shared_data_mgr.get_ohlcv('SPY', buffer_start, end_date)

        print("\n%s PROFILE (%d symbols)" % (profile_name.upper(), len(symbols)))
        print("-" * 50)

        # Without thematic ETF
        t0 = time.time()
        cfg_no_etf = copy.deepcopy(base_config)
        cfg_no_etf.thematic_etf_enabled = False
        r1 = run_config(
            '%s (no thematic)' % profile_name,
            cfg_no_etf, symbols, shared_data_mgr, shared_scorer
        )
        print("  %-25s CAGR=%6.2f%%  Sharpe=%5.2f  MaxDD=%5.1f%%  Trades=%d  [%.0fs]" % (
            r1['label'], r1['cagr'], r1['sharpe'], r1['max_dd'], r1['trades'], time.time() - t0))
        results.append(r1)

        # With thematic ETF
        t0 = time.time()
        cfg_etf = copy.deepcopy(base_config)
        cfg_etf.thematic_etf_enabled = True
        r2 = run_config(
            '%s (+ thematic)' % profile_name,
            cfg_etf, symbols, shared_data_mgr, shared_scorer
        )
        print("  %-25s CAGR=%6.2f%%  Sharpe=%5.2f  MaxDD=%5.1f%%  Trades=%d  [%.0fs]" % (
            r2['label'], r2['cagr'], r2['sharpe'], r2['max_dd'], r2['trades'], time.time() - t0))
        results.append(r2)

        # With both macro + thematic
        t0 = time.time()
        cfg_both = copy.deepcopy(base_config)
        cfg_both.thematic_etf_enabled = True
        cfg_both.macro_regime_enabled = True
        r3 = run_config(
            '%s (macro+thematic)' % profile_name,
            cfg_both, symbols, shared_data_mgr, shared_scorer
        )
        print("  %-25s CAGR=%6.2f%%  Sharpe=%5.2f  MaxDD=%5.1f%%  Trades=%d  [%.0fs]" % (
            r3['label'], r3['cagr'], r3['sharpe'], r3['max_dd'], r3['trades'], time.time() - t0))
        results.append(r3)

        # Deltas
        print("\n  DELTA (thematic only):    CAGR=%+.2f%%  Sharpe=%+.2f  MaxDD=%+.1f%%" % (
            r2['cagr'] - r1['cagr'], r2['sharpe'] - r1['sharpe'], r2['max_dd'] - r1['max_dd']))
        print("  DELTA (macro+thematic):   CAGR=%+.2f%%  Sharpe=%+.2f  MaxDD=%+.1f%%" % (
            r3['cagr'] - r1['cagr'], r3['sharpe'] - r1['sharpe'], r3['max_dd'] - r1['max_dd']))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print("  %-25s CAGR=%6.2f%%  Sharpe=%5.2f  MaxDD=%5.1f%%" % (
            r['label'], r['cagr'], r['sharpe'], r['max_dd']))

    # Save results
    with open('thematic_etf_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to thematic_etf_results.json")


if __name__ == '__main__':
    main()
