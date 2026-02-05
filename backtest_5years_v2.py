#!/usr/bin/env python3
"""
MIDAS Backtest V2 - 5 Years Technical Signals
==============================================
Simplified backtest - iterates per symbol then merges results.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    close = df['Close'].values
    
    # EMAs
    df['ema_20'] = pd.Series(close).ewm(span=20, adjust=False).mean().values
    df['ema_50'] = pd.Series(close).ewm(span=50, adjust=False).mean().values
    df['ema_200'] = pd.Series(close).ewm(span=200, adjust=False).mean().values
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().values
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd'] = (exp1 - exp2).values
    df['macd_signal'] = pd.Series(df['macd']).ewm(span=9, adjust=False).mean().values
    
    # ATR
    high = df['High'].values
    low = df['Low'].values
    high_low = high - low
    high_close = np.abs(high[1:] - close[:-1])
    low_close = np.abs(low[1:] - close[:-1])
    tr = np.zeros(len(close))
    tr[1:] = np.maximum(np.maximum(high_low[1:], high_close), low_close)
    df['atr'] = pd.Series(tr).rolling(14).mean().values
    
    return df

def get_signal_score(row_idx: int, df: pd.DataFrame) -> int:
    """Calculate signal score for a given row index."""
    if row_idx < 200:  # Need enough history for EMA200
        return 0
    
    close = df['Close'].iloc[row_idx]
    ema_20 = df['ema_20'].iloc[row_idx]
    ema_50 = df['ema_50'].iloc[row_idx]
    ema_200 = df['ema_200'].iloc[row_idx]
    rsi = df['rsi'].iloc[row_idx]
    macd = df['macd'].iloc[row_idx]
    macd_signal = df['macd_signal'].iloc[row_idx]
    prev_macd = df['macd'].iloc[row_idx - 1]
    prev_macd_signal = df['macd_signal'].iloc[row_idx - 1]
    
    if pd.isna(ema_200) or pd.isna(rsi):
        return 0
    
    score = 0
    
    # EMA trend (30%)
    if ema_20 > ema_50 > ema_200:
        score += 30
    elif ema_20 > ema_50:
        score += 20
    
    # RSI (25%)
    if 30 <= rsi <= 50:
        score += 25
    elif 50 <= rsi <= 70:
        score += 15
    
    # MACD (25%)
    if macd > macd_signal and prev_macd <= prev_macd_signal:
        score += 25
    elif macd > 0 and macd > macd_signal:
        score += 15
    
    # Price vs EMA (20%)
    if close > ema_20 > ema_50:
        score += 20
    
    return score

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    initial_capital: float = 1000,
    stop_loss_atr_mult: float = 2.0,
    take_profit_pct: float = 0.20,
    max_hold_weeks: int = 12
) -> List[Trade]:
    """Backtest a single symbol."""
    trades = []
    position = None
    capital = initial_capital
    
    for i in range(200, len(df)):
        date = df.index[i]
        date_str = str(date)[:10]
        close = df['Close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        # Check exit conditions if in position
        if position is not None:
            entry_price = position['entry_price']
            pnl_pct = (close - entry_price) / entry_price
            weeks_held = (i - position['entry_idx'])
            
            exit_reason = None
            
            # Stop loss
            if close <= position['stop_loss']:
                exit_reason = 'stop_loss'
            # Take profit
            elif pnl_pct >= take_profit_pct:
                exit_reason = 'take_profit'
            # Max hold
            elif weeks_held >= max_hold_weeks:
                exit_reason = 'max_hold'
            
            if exit_reason:
                pnl = (close - entry_price) * position['shares']
                trades.append(Trade(
                    symbol=symbol,
                    entry_date=position['entry_date'],
                    entry_price=entry_price,
                    exit_date=date_str,
                    exit_price=close,
                    shares=position['shares'],
                    pnl=pnl,
                    pnl_pct=pnl_pct * 100,
                    exit_reason=exit_reason
                ))
                capital += close * position['shares']
                position = None
        
        # Check entry if no position
        if position is None:
            score = get_signal_score(i, df)
            
            if score >= 50:
                shares = int(capital / close)
                if shares > 0:
                    cost = shares * close
                    capital -= cost
                    stop_loss = close - (atr * stop_loss_atr_mult) if not pd.isna(atr) else close * 0.9
                    
                    position = {
                        'entry_price': close,
                        'entry_date': date_str,
                        'entry_idx': i,
                        'shares': shares,
                        'stop_loss': stop_loss
                    }
    
    # Close remaining position
    if position is not None:
        close = df['Close'].iloc[-1]
        pnl = (close - position['entry_price']) * position['shares']
        pnl_pct = (close - position['entry_price']) / position['entry_price']
        trades.append(Trade(
            symbol=symbol,
            entry_date=position['entry_date'],
            entry_price=position['entry_price'],
            exit_date=str(df.index[-1])[:10],
            exit_price=close,
            shares=position['shares'],
            pnl=pnl,
            pnl_pct=pnl_pct * 100,
            exit_reason='end_of_backtest'
        ))
    
    return trades

def run_backtest(
    symbols: List[str],
    start_date: str = '2020-01-01',
    end_date: str = '2025-12-31',
    capital_per_symbol: float = 1000
):
    """Run backtest across all symbols."""
    
    print("=" * 70)
    print("MIDAS BACKTEST V2 - 5 Years Technical Signals")
    print("=" * 70)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"Capital per symbol: ${capital_per_symbol:,}")
    
    # Load and backtest each symbol
    print("\n" + "-" * 70)
    print("Running backtests...")
    
    all_trades = []
    symbols_tested = 0
    
    for sym in symbols:
        try:
            data = yf.download(sym, start=start_date, end=end_date, interval='1wk', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) < 200:
                print(f"  ✗ {sym}: Not enough data ({len(data)} weeks)")
                continue
            
            data = calculate_indicators(data)
            trades = backtest_symbol(sym, data, initial_capital=capital_per_symbol)
            
            if trades:
                all_trades.extend(trades)
                total_pnl = sum(t.pnl for t in trades)
                print(f"  ✓ {sym}: {len(trades)} trades, P&L: ${total_pnl:+,.0f}")
            else:
                print(f"  - {sym}: No trades")
            
            symbols_tested += 1
            
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if not all_trades:
        print("\n❌ No trades generated!")
        return
    
    total_initial = capital_per_symbol * symbols_tested
    total_pnl = sum(t.pnl for t in all_trades)
    total_return = total_pnl / total_initial * 100
    
    winning = [t for t in all_trades if t.pnl > 0]
    losing = [t for t in all_trades if t.pnl <= 0]
    win_rate = len(winning) / len(all_trades) * 100
    
    avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
    avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0
    
    print(f"\n📊 Performance")
    print(f"  Symbols tested:   {symbols_tested}")
    print(f"  Initial Capital:  ${total_initial:,.0f}")
    print(f"  Total P&L:        ${total_pnl:+,.0f}")
    print(f"  Total Return:     {total_return:+.1f}%")
    
    print(f"\n📈 Trades")
    print(f"  Total Trades:     {len(all_trades)}")
    print(f"  Winning:          {len(winning)} ({win_rate:.1f}%)")
    print(f"  Losing:           {len(losing)}")
    print(f"  Avg Win:          {avg_win:+.1f}%")
    print(f"  Avg Loss:         {avg_loss:.1f}%")
    
    # Exit reasons
    exit_reasons = {}
    for t in all_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    print(f"\n🚪 Exit Reasons")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    # Best/worst
    print(f"\n🏆 Top 5 Trades")
    for t in sorted(all_trades, key=lambda x: x.pnl, reverse=True)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f})")
    
    print(f"\n💀 Worst 5 Trades")
    for t in sorted(all_trades, key=lambda x: x.pnl)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f})")
    
    # By symbol
    print(f"\n📊 By Symbol")
    symbol_stats = {}
    for t in all_trades:
        if t.symbol not in symbol_stats:
            symbol_stats[t.symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
        symbol_stats[t.symbol]['trades'] += 1
        symbol_stats[t.symbol]['pnl'] += t.pnl
        if t.pnl > 0:
            symbol_stats[t.symbol]['wins'] += 1
    
    for sym in sorted(symbol_stats.keys(), key=lambda x: symbol_stats[x]['pnl'], reverse=True):
        s = symbol_stats[sym]
        wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
        print(f"  {sym:8}: {s['trades']:2} trades, WR: {wr:5.1f}%, P&L: ${s['pnl']:+,.0f}")
    
    # Save results
    results = {
        'summary': {
            'period': f"{start_date} to {end_date}",
            'symbols_tested': symbols_tested,
            'initial_capital': total_initial,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': len(all_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss
        },
        'trades': [
            {
                'symbol': t.symbol,
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason
            }
            for t in all_trades
        ],
        'by_symbol': symbol_stats
    }
    
    with open('midas_backtest_5years.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to midas_backtest_5years.json")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    symbols = [
        # US Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        # US Value
        'JPM', 'JNJ', 'XOM', 'PG', 'KO', 'WMT', 'DIS', 'V', 'MA', 'HD',
        # CAC40
        'MC.PA', 'OR.PA', 'TTE.PA', 'AIR.PA', 'SAN.PA',
        # DAX
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'MBG.DE'
    ]
    
    run_backtest(symbols, start_date='2020-01-01', end_date='2025-12-31')
