#!/usr/bin/env python3
"""
MIDAS Backtest - 5 Years Technical Signals
==========================================
Backtest simplifié pour track record Substack.
Teste les signaux techniques (EMA cross, RSI, MACD) sur 5 ans.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
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

@dataclass
class BacktestResult:
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: List[Dict]

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    # EMAs
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR for stop loss
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['atr'] = ranges.max(axis=1).rolling(14).mean()
    
    return df

def generate_signal(row, prev_row) -> Optional[str]:
    """Generate BUY/SELL signal based on technical indicators."""
    try:
        # Extract scalar values
        ema_20 = float(row['ema_20'])
        ema_50 = float(row['ema_50'])
        ema_200 = float(row['ema_200'])
        rsi = float(row['rsi'])
        macd = float(row['macd'])
        macd_signal = float(row['macd_signal'])
        macd_hist = float(row['macd_hist'])
        close = float(row['Close'])
        prev_macd = float(prev_row['macd'])
        prev_macd_signal = float(prev_row['macd_signal'])
        
        score = 0
        
        # EMA trend (weight: 30%)
        if ema_20 > ema_50 > ema_200:
            score += 30  # Strong uptrend
        elif ema_20 > ema_50:
            score += 20  # Moderate uptrend
        elif ema_20 < ema_50 < ema_200:
            score -= 30  # Strong downtrend
        
        # RSI (weight: 25%)
        if 30 <= rsi <= 50:  # Oversold recovering
            score += 25
        elif 50 <= rsi <= 70:  # Bullish momentum
            score += 15
        elif rsi > 80:  # Overbought
            score -= 20
        elif rsi < 25:  # Deeply oversold - risky
            score += 10
        
        # MACD (weight: 25%)
        if macd > macd_signal and prev_macd <= prev_macd_signal:
            score += 25  # Bullish crossover
        elif macd > 0 and macd_hist > 0:
            score += 15  # Bullish momentum
        elif macd < macd_signal and prev_macd >= prev_macd_signal:
            score -= 25  # Bearish crossover
        
        # Price vs EMA (weight: 20%)
        if close > ema_20 > ema_50:
            score += 20
        elif close < ema_20 < ema_50:
            score -= 20
        
        # Signal thresholds
        if score >= 50:
            return 'BUY'
        elif score <= -40:
            return 'SELL'
        return None
    except Exception:
        return None

def run_backtest(
    symbols: List[str],
    start_date: str = '2021-01-01',
    end_date: str = '2025-12-31',
    initial_capital: float = 10000,
    max_positions: int = 10,
    position_size_pct: float = 0.10,
    stop_loss_atr_mult: float = 2.0,
    take_profit_pct: float = 0.20,
    max_hold_weeks: int = 12
) -> BacktestResult:
    """Run backtest simulation."""
    
    print("=" * 70)
    print("MIDAS BACKTEST - 5 Years Technical Signals")
    print("=" * 70)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"Capital: ${initial_capital:,}")
    print(f"Max positions: {max_positions}")
    print(f"Position size: {position_size_pct*100}%")
    print(f"Stop loss: {stop_loss_atr_mult}x ATR")
    print(f"Take profit: {take_profit_pct*100}%")
    
    # Load data
    print("\n" + "-" * 70)
    print("Loading data...")
    
    all_data = {}
    for sym in symbols:
        try:
            data = yf.download(sym, start=start_date, end=end_date, interval='1wk', progress=False)
            # Flatten MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if len(data) >= 52:  # At least 1 year of data
                data = calculate_indicators(data)
                all_data[sym] = data
                print(f"  ✓ {sym}: {len(data)} weeks")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
    
    print(f"\n{len(all_data)} symbols loaded")
    
    # Simulation
    print("\n" + "-" * 70)
    print("Running simulation...")
    
    capital = initial_capital
    positions = {}  # {symbol: {entry_price, entry_date, shares, stop_loss}}
    trades = []
    equity_curve = []
    
    # Get all unique dates
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)
    
    for i, date in enumerate(all_dates):
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        
        # Check existing positions for exit
        positions_to_close = []
        for sym, pos in positions.items():
            if sym not in all_data or date not in all_data[sym].index:
                continue
            
            row = all_data[sym].loc[date]
            current_price = row['Close']
            entry_price = pos['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            weeks_held = (date - pd.to_datetime(pos['entry_date'])).days // 7
            
            exit_reason = None
            
            # Stop loss
            if current_price <= pos['stop_loss']:
                exit_reason = 'stop_loss'
            # Take profit
            elif pnl_pct >= take_profit_pct:
                exit_reason = 'take_profit'
            # Max hold time
            elif weeks_held >= max_hold_weeks:
                exit_reason = 'max_hold'
            
            if exit_reason:
                pnl = (current_price - entry_price) * pos['shares']
                capital += current_price * pos['shares']
                
                trades.append(Trade(
                    symbol=sym,
                    entry_date=pos['entry_date'],
                    entry_price=entry_price,
                    exit_date=date_str,
                    exit_price=current_price,
                    shares=pos['shares'],
                    pnl=pnl,
                    pnl_pct=pnl_pct * 100,
                    exit_reason=exit_reason
                ))
                positions_to_close.append(sym)
        
        for sym in positions_to_close:
            del positions[sym]
        
        # Check for new entries
        if len(positions) < max_positions:
            for sym, df in all_data.items():
                if sym in positions:
                    continue
                if date not in df.index:
                    continue
                if i < 1:  # Need previous row
                    continue
                
                prev_date = all_dates[i-1]
                if prev_date not in df.index:
                    continue
                
                row = df.loc[date]
                prev_row = df.loc[prev_date]
                
                # Skip if indicators not ready
                if pd.isna(row['ema_200']) or pd.isna(row['atr']):
                    continue
                
                signal = generate_signal(row, prev_row)
                
                if signal == 'BUY' and len(positions) < max_positions:
                    # Calculate position size
                    position_value = capital * position_size_pct
                    shares = int(position_value / row['Close'])
                    
                    if shares > 0 and shares * row['Close'] <= capital:
                        cost = shares * row['Close']
                        capital -= cost
                        
                        positions[sym] = {
                            'entry_price': row['Close'],
                            'entry_date': date_str,
                            'shares': shares,
                            'stop_loss': row['Close'] - (row['atr'] * stop_loss_atr_mult)
                        }
        
        # Calculate portfolio value
        portfolio_value = capital
        for sym, pos in positions.items():
            if sym in all_data and date in all_data[sym].index:
                portfolio_value += all_data[sym].loc[date]['Close'] * pos['shares']
        
        equity_curve.append({
            'date': date_str,
            'value': portfolio_value,
            'positions': len(positions)
        })
        
        if i % 52 == 0:  # Progress every year
            print(f"  {date_str}: ${portfolio_value:,.0f} ({len(positions)} pos)")
    
    # Close remaining positions at end
    final_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        if sym in all_data and final_date in all_data[sym].index:
            current_price = all_data[sym].loc[final_date]['Close']
            pnl = (current_price - pos['entry_price']) * pos['shares']
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            capital += current_price * pos['shares']
            
            trades.append(Trade(
                symbol=sym,
                entry_date=pos['entry_date'],
                entry_price=pos['entry_price'],
                exit_date=final_date.strftime('%Y-%m-%d'),
                exit_price=current_price,
                shares=pos['shares'],
                pnl=pnl,
                pnl_pct=pnl_pct * 100,
                exit_reason='end_of_backtest'
            ))
    
    # Calculate metrics
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
    
    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for point in equity_curve:
        if point['value'] > peak:
            peak = point['value']
        dd = (peak - point['value']) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Sharpe ratio (simplified)
    returns = []
    for i in range(1, len(equity_curve)):
        ret = (equity_curve[i]['value'] - equity_curve[i-1]['value']) / equity_curve[i-1]['value']
        returns.append(ret)
    
    if returns:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    
    result = BacktestResult(
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return_pct=total_return,
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        trades=trades,
        equity_curve=equity_curve
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Performance")
    print(f"  Initial Capital:  ${initial_capital:,.2f}")
    print(f"  Final Capital:    ${final_capital:,.2f}")
    print(f"  Total Return:     {total_return:+.2f}%")
    print(f"  Max Drawdown:     {max_dd:.2f}%")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")
    
    print(f"\n📈 Trades")
    print(f"  Total Trades:     {len(trades)}")
    print(f"  Winning:          {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"  Losing:           {len(losing_trades)}")
    print(f"  Avg Win:          {avg_win:+.2f}%")
    print(f"  Avg Loss:         {avg_loss:.2f}%")
    
    # Exit reasons
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    print(f"\n🚪 Exit Reasons")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    # Top trades
    print(f"\n🏆 Top 5 Trades")
    for t in sorted(trades, key=lambda x: x.pnl, reverse=True)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f})")
    
    print(f"\n💀 Worst 5 Trades")
    for t in sorted(trades, key=lambda x: x.pnl)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f})")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    
    return result

def save_results(result: BacktestResult, filename: str = 'backtest_results.json'):
    """Save results to JSON."""
    data = {
        'summary': {
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return_pct': result.total_return_pct,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'max_drawdown_pct': result.max_drawdown_pct,
            'sharpe_ratio': result.sharpe_ratio
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
            for t in result.trades
        ],
        'equity_curve': result.equity_curve
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {filename}")

if __name__ == '__main__':
    # Diversified portfolio
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
    
    result = run_backtest(
        symbols=symbols,
        start_date='2021-01-01',
        end_date='2025-12-31',
        initial_capital=10000,
        max_positions=10,
        position_size_pct=0.10
    )
    
    save_results(result, 'midas_backtest_5years.json')
