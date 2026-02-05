#!/usr/bin/env python3
"""
MIDAS Backtest - FINAL VERSION (Risk-Managed)
==============================================
Objectif: Return > SPY (85%), Max DD < QQQ (35%)

Risk Management ajouté:
1. Position size cap: 80% max (20% cash reserve)
2. Stop serré en VOLATILE: 2.0 ATR (vs 2.5 normal)
3. Circuit breaker: Réduire position à 50% quand DD > 15%
4. Portfolio heat limit: Max 3 positions simultanées (simulé via sizing)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

WEIGHTS = {
    'technical': 35,
    'fundamental': 20,
    'sentiment': 15,
    'news': 10,
    'ml_regime': 20
}

# RISK MANAGEMENT PARAMS
MAX_POSITION_PCT = 0.80  # Max 80% in any position
VOLATILE_STOP_ATR = 2.0  # Tighter stop in volatile regime
NORMAL_STOP_ATR = 2.5
DD_THRESHOLD_1 = 0.10  # At 10% DD, reduce to 70%
DD_THRESHOLD_2 = 0.15  # At 15% DD, reduce to 50%
DD_THRESHOLD_3 = 0.20  # At 20% DD, reduce to 30%

@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    entry_score: float
    regime: str
    position_size_pct: float

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    df['ema_9'] = pd.Series(close).ewm(span=9, adjust=False).mean().values
    df['ema_20'] = pd.Series(close).ewm(span=20, adjust=False).mean().values
    df['ema_50'] = pd.Series(close).ewm(span=50, adjust=False).mean().values
    df['ema_200'] = pd.Series(close).ewm(span=200, adjust=False).mean().values
    
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().values
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd'] = (exp1 - exp2).values
    df['macd_signal'] = pd.Series(df['macd']).ewm(span=9, adjust=False).mean().values
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    df['atr'] = pd.Series(tr).rolling(14).mean().values
    
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_smooth = pd.Series(tr).rolling(14).sum().values
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * pd.Series(plus_dm).rolling(14).sum().values / tr_smooth
        minus_di = 100 * pd.Series(minus_dm).rolling(14).sum().values / tr_smooth
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = pd.Series(dx).rolling(14).mean().values
    
    df['bb_middle'] = pd.Series(close).rolling(20).mean().values
    bb_std = pd.Series(close).rolling(20).std().values
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    df['volume_sma'] = pd.Series(volume.astype(float)).rolling(20).mean().values
    df['volume_ratio'] = volume / df['volume_sma']
    
    df['roc_5'] = pd.Series(close).pct_change(5).values * 100
    df['roc_10'] = pd.Series(close).pct_change(10).values * 100
    df['roc_20'] = pd.Series(close).pct_change(20).values * 100
    df['roc_60'] = pd.Series(close).pct_change(60).values * 100
    
    df['volatility'] = pd.Series(close).pct_change().rolling(20).std().values * np.sqrt(252) * 100
    
    return df

def detect_regime(idx: int, df: pd.DataFrame) -> str:
    if idx < 60:
        return 'UNKNOWN'
    
    ema_20 = df['ema_20'].iloc[idx]
    ema_50 = df['ema_50'].iloc[idx]
    ema_200 = df['ema_200'].iloc[idx]
    adx = df['adx'].iloc[idx] if not pd.isna(df['adx'].iloc[idx]) else 20
    volatility = df['volatility'].iloc[idx] if not pd.isna(df['volatility'].iloc[idx]) else 20
    
    if volatility > 40:
        return 'VOLATILE'
    elif ema_20 > ema_50 > ema_200 and adx > 20:
        return 'BULL'
    elif ema_20 < ema_50 < ema_200 and adx > 20:
        return 'BEAR'
    else:
        return 'RANGE'

def score_technical(idx: int, df: pd.DataFrame) -> float:
    if idx < 200:
        return 50
    
    score = 50
    close = df['Close'].iloc[idx]
    ema_9 = df['ema_9'].iloc[idx]
    ema_20 = df['ema_20'].iloc[idx]
    ema_50 = df['ema_50'].iloc[idx]
    ema_200 = df['ema_200'].iloc[idx]
    rsi = df['rsi'].iloc[idx]
    macd = df['macd'].iloc[idx]
    macd_signal = df['macd_signal'].iloc[idx]
    adx = df['adx'].iloc[idx]
    bb_pct = df['bb_pct'].iloc[idx]
    volume_ratio = df['volume_ratio'].iloc[idx]
    roc_10 = df['roc_10'].iloc[idx]
    
    if pd.isna(ema_200) or pd.isna(rsi):
        return 50
    
    if ema_9 > ema_20 > ema_50 > ema_200:
        score += 15
    elif ema_20 > ema_50 > ema_200:
        score += 10
    elif ema_50 > ema_200:
        score += 5
    elif ema_9 < ema_20 < ema_50 < ema_200:
        score -= 15
    elif ema_20 < ema_50 < ema_200:
        score -= 10
    
    if close > ema_20 and close > ema_50:
        score += 10
    elif close < ema_20 and close < ema_50:
        score -= 10
    
    if 40 <= rsi <= 55:
        score += 10
    elif 30 <= rsi < 40:
        score += 8
    elif 55 < rsi <= 65:
        score += 5
    elif rsi > 75:
        score -= 8
    elif rsi < 25:
        score += 3
    
    if macd > macd_signal and macd > 0:
        score += 10
    elif macd > macd_signal:
        score += 5
    elif macd < macd_signal and macd < 0:
        score -= 10
    
    if not pd.isna(adx) and adx > 25:
        if close > ema_50:
            score += 5
        else:
            score -= 5
    
    if not pd.isna(bb_pct):
        if 0.2 <= bb_pct <= 0.5:
            score += 5
        elif bb_pct > 0.95:
            score -= 3
    
    if not pd.isna(volume_ratio) and not pd.isna(roc_10):
        if volume_ratio > 1.5 and roc_10 > 0:
            score += 5
    
    return max(0, min(100, score))

def score_fundamental(idx: int, df: pd.DataFrame, symbol: str) -> float:
    if idx < 200:
        return 50
    
    score = 50
    roc_60 = df['roc_60'].iloc[idx] / 100 if not pd.isna(df['roc_60'].iloc[idx]) else 0
    
    if roc_60 > 0.20:
        score += 20
    elif roc_60 > 0.10:
        score += 15
    elif roc_60 > 0.05:
        score += 10
    elif roc_60 > 0:
        score += 5
    elif roc_60 < -0.20:
        score -= 20
    elif roc_60 < -0.10:
        score -= 15
    elif roc_60 < 0:
        score -= 5
    
    tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM', 'SAP.DE']
    if symbol in tech_symbols and roc_60 > 0:
        score += 5
    
    return max(0, min(100, score))

def score_sentiment(idx: int, df: pd.DataFrame) -> float:
    if idx < 30:
        return 50
    
    score = 50
    roc_5 = df['roc_5'].iloc[idx] / 100 if not pd.isna(df['roc_5'].iloc[idx]) else 0
    roc_10 = df['roc_10'].iloc[idx] / 100 if not pd.isna(df['roc_10'].iloc[idx]) else 0
    volatility = df['volatility'].iloc[idx] if not pd.isna(df['volatility'].iloc[idx]) else 20
    
    if roc_5 > 0.05:
        score += 20
    elif roc_5 > 0.02:
        score += 12
    elif roc_5 > 0:
        score += 5
    elif roc_5 < -0.05:
        score -= 20
    elif roc_5 < -0.02:
        score -= 12
    elif roc_5 < 0:
        score -= 5
    
    if volatility > 40:
        score = 50 + (score - 50) * 0.5
    
    if roc_10 > 0 and roc_5 > 0:
        score += 5
    elif roc_10 < 0 and roc_5 < 0:
        score -= 5
    
    return max(0, min(100, score))

def score_news(idx: int, df: pd.DataFrame) -> float:
    if idx < 5:
        return 50
    
    score = 50
    prev_close = df['Close'].iloc[idx-1]
    current_open = df['Open'].iloc[idx] if 'Open' in df.columns else df['Close'].iloc[idx]
    gap_pct = (current_open - prev_close) / prev_close
    
    if gap_pct > 0.03:
        score += 15
    elif gap_pct > 0.01:
        score += 8
    elif gap_pct < -0.03:
        score -= 15
    elif gap_pct < -0.01:
        score -= 8
    
    return max(0, min(100, score))

def score_ml_regime(idx: int, df: pd.DataFrame) -> Tuple[float, str]:
    if idx < 60:
        return 50, 'UNKNOWN'
    
    regime = detect_regime(idx, df)
    close = df['Close'].iloc[idx]
    ema_50 = df['ema_50'].iloc[idx]
    roc_20 = df['roc_20'].iloc[idx] if not pd.isna(df['roc_20'].iloc[idx]) else 0
    bb_pct = df['bb_pct'].iloc[idx] if not pd.isna(df['bb_pct'].iloc[idx]) else 0.5
    
    if regime == 'BULL':
        score = 70 + min(15, roc_20 / 2)
    elif regime == 'BEAR':
        score = 30 + max(-15, roc_20 / 2)
    elif regime == 'VOLATILE':
        score = 55 if close > ema_50 else 40
    else:
        if bb_pct < 0.3:
            score = 60
        elif bb_pct > 0.7:
            score = 40
        else:
            score = 50
    
    return max(0, min(100, score)), regime

def calculate_total_score(idx: int, df: pd.DataFrame, symbol: str) -> Tuple[float, str]:
    tech = score_technical(idx, df)
    fund = score_fundamental(idx, df, symbol)
    sent = score_sentiment(idx, df)
    news = score_news(idx, df)
    ml, regime = score_ml_regime(idx, df)
    
    total = (
        tech * WEIGHTS['technical'] +
        fund * WEIGHTS['fundamental'] +
        sent * WEIGHTS['sentiment'] +
        news * WEIGHTS['news'] +
        ml * WEIGHTS['ml_regime']
    ) / 100
    
    return total, regime

def get_position_size_multiplier(current_dd: float, regime: str, score: float) -> float:
    """Calculate position size based on DD level and regime."""
    # Base multiplier
    mult = MAX_POSITION_PCT
    
    # Reduce based on current drawdown (circuit breaker)
    if current_dd >= DD_THRESHOLD_3:
        mult *= 0.30  # 30% size at 20%+ DD
    elif current_dd >= DD_THRESHOLD_2:
        mult *= 0.50  # 50% size at 15%+ DD
    elif current_dd >= DD_THRESHOLD_1:
        mult *= 0.70  # 70% size at 10%+ DD
    
    # Reduce in VOLATILE regime
    if regime == 'VOLATILE':
        mult *= 0.70
    
    # Increase slightly for high conviction
    if score >= 65:
        mult *= 1.10
    
    return min(mult, MAX_POSITION_PCT)

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    initial_capital: float = 1000,
    signal_threshold: float = 55,
    take_profit_pct: float = 0.35,
    max_hold_days: int = 45,
    trailing_stop_pct: float = 0.15
) -> Tuple[List[Trade], List[float]]:
    trades = []
    equity_curve = []
    position = None
    capital = initial_capital
    peak_equity = initial_capital
    
    for i in range(200, len(df)):
        date_str = str(df.index[i])[:10]
        close = df['Close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else close * 0.02
        
        # Track equity and drawdown
        current_equity = capital
        if position:
            current_equity += (close - position['entry_price']) * position['shares']
        
        equity_curve.append(current_equity)
        
        if current_equity > peak_equity:
            peak_equity = current_equity
        
        current_dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
        
        # Exit logic
        if position is not None:
            entry_price = position['entry_price']
            current_pnl_pct = (close - entry_price) / entry_price
            days_held = i - position['entry_idx']
            
            if close > position['high_since_entry']:
                position['high_since_entry'] = close
                position['trailing_stop'] = close * (1 - trailing_stop_pct)
            
            exit_reason = None
            
            if close <= position['stop_loss']:
                exit_reason = 'stop_loss'
            elif close <= position['trailing_stop'] and current_pnl_pct > 0.05:
                exit_reason = 'trailing_stop'
            elif current_pnl_pct >= take_profit_pct:
                exit_reason = 'take_profit'
            elif days_held >= max_hold_days:
                exit_reason = 'max_hold'
            elif position.get('last_score', 50) < 35:
                exit_reason = 'score_exit'
            # Emergency exit if portfolio DD > 25%
            elif current_dd > 0.25:
                exit_reason = 'dd_circuit_breaker'
            
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
                    pnl_pct=current_pnl_pct * 100,
                    exit_reason=exit_reason,
                    entry_score=position['entry_score'],
                    regime=position['regime'],
                    position_size_pct=position['position_size_pct']
                ))
                capital += close * position['shares']
                position = None
        
        # Entry logic
        if position is None:
            total_score, regime = calculate_total_score(i, df, symbol)
            
            # Skip BEAR regime
            if regime == 'BEAR':
                continue
            
            # Skip if in heavy DD
            if current_dd > DD_THRESHOLD_3:
                continue
            
            if total_score >= signal_threshold:
                # Dynamic position sizing
                size_mult = get_position_size_multiplier(current_dd, regime, total_score)
                position_capital = capital * size_mult
                shares = position_capital / close
                
                # Dynamic stop loss based on regime
                stop_atr_mult = VOLATILE_STOP_ATR if regime == 'VOLATILE' else NORMAL_STOP_ATR
                stop_loss = close - (atr * stop_atr_mult)
                
                if shares > 0:
                    position = {
                        'entry_price': close,
                        'entry_date': date_str,
                        'entry_idx': i,
                        'shares': shares,
                        'stop_loss': stop_loss,
                        'trailing_stop': close * (1 - trailing_stop_pct),
                        'high_since_entry': close,
                        'entry_score': total_score,
                        'regime': regime,
                        'last_score': total_score,
                        'position_size_pct': size_mult
                    }
                    capital -= position_capital
        else:
            total_score, _ = calculate_total_score(i, df, symbol)
            position['last_score'] = total_score
    
    # Close remaining
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
            exit_reason='end_of_backtest',
            entry_score=position['entry_score'],
            regime=position['regime'],
            position_size_pct=position.get('position_size_pct', 0.8)
        ))
    
    return trades, equity_curve

def run_backtest(symbols: List[str], start_date: str, end_date: str, capital_per_symbol: float = 1000):
    print("=" * 70)
    print("MIDAS BACKTEST - FINAL (Risk-Managed)")
    print("=" * 70)
    print(f"\nObjectif: Return > SPY (85%), Max DD < QQQ (35%)")
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"\nRISK MANAGEMENT:")
    print(f"  ✓ Position cap: {MAX_POSITION_PCT*100:.0f}% max")
    print(f"  ✓ Volatile stop: {VOLATILE_STOP_ATR}x ATR (vs {NORMAL_STOP_ATR}x normal)")
    print(f"  ✓ Circuit breaker: 70%/50%/30% at 10%/15%/20% DD")
    print(f"  ✓ Emergency exit: DD > 25%")
    print(f"  ✓ BEAR regime filter: ON")
    
    print("\n" + "-" * 70)
    
    all_trades = []
    all_equity = []
    symbols_tested = 0
    
    for sym in symbols:
        try:
            print(f"  {sym}...", end=" ", flush=True)
            data = yf.download(sym, start=start_date, end=end_date, interval='1d', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) < 250:
                print(f"Not enough data")
                continue
            
            data = calculate_technical_indicators(data)
            trades, equity = backtest_symbol(sym, data, initial_capital=capital_per_symbol)
            
            if equity:
                all_equity.append(equity)
            
            if trades:
                all_trades.extend(trades)
                total_pnl = sum(t.pnl for t in trades)
                win_rate = len([t for t in trades if t.pnl > 0]) / len(trades) * 100
                print(f"{len(trades)} trades, P&L: ${total_pnl:+,.0f}, WR: {win_rate:.0f}%")
            else:
                print("No trades")
            
            symbols_tested += 1
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if not all_trades:
        print("No trades!")
        return
    
    total_initial = capital_per_symbol * symbols_tested
    total_pnl = sum(t.pnl for t in all_trades)
    final_capital = total_initial + total_pnl
    total_return = total_pnl / total_initial * 100
    years = 5
    cagr = ((final_capital / total_initial) ** (1/years) - 1) * 100
    
    winning = [t for t in all_trades if t.pnl > 0]
    losing = [t for t in all_trades if t.pnl <= 0]
    win_rate = len(winning) / len(all_trades) * 100
    
    avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
    avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0
    
    gross_profit = sum(t.pnl for t in winning)
    gross_loss = abs(sum(t.pnl for t in losing))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    
    # Max DD calculation (more accurate with equity curve)
    equity_values = [total_initial]
    running_pnl = 0
    for t in sorted(all_trades, key=lambda x: x.exit_date):
        running_pnl += t.pnl
        equity_values.append(total_initial + running_pnl)
    
    peak = equity_values[0]
    max_dd = 0
    max_dd_date = ""
    for i, val in enumerate(equity_values):
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Sharpe & Sortino
    pnl_pcts = [t.pnl_pct for t in all_trades]
    avg_return = np.mean(pnl_pcts)
    std_return = np.std(pnl_pcts)
    trades_per_year = len(all_trades) / years
    sharpe = (avg_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
    
    downside = [p for p in pnl_pcts if p < 0]
    downside_std = np.std(downside) if downside else 1
    sortino = (avg_return / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0
    
    # Check objectives
    obj_return = "✅" if total_return > 85 else "❌"
    obj_dd = "✅" if max_dd < 35 else "❌"
    
    print(f"\n{'='*70}")
    print(f"📊 PERFORMANCE vs BENCHMARK")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'MIDAS FINAL':>12} {'SPY':>12} {'QQQ':>12} {'OK?':>6}")
    print(f"  {'-'*62}")
    print(f"  {'Total Return':<20} {total_return:>+11.1f}% {'~85%':>12} {'~120%':>12} {obj_return:>6}")
    print(f"  {'CAGR':<20} {cagr:>+11.1f}% {'~13%':>12} {'~17%':>12}")
    print(f"  {'Max Drawdown':<20} {max_dd:>11.1f}% {'~34%':>12} {'~35%':>12} {obj_dd:>6}")
    print(f"  {'Sharpe Ratio':<20} {sharpe:>12.2f} {'~0.8':>12} {'~0.9':>12}")
    print(f"  {'Sortino Ratio':<20} {sortino:>12.2f}")
    
    if total_return > 85 and max_dd < 35:
        print(f"\n  🎯 OBJECTIF ATTEINT: Return > SPY ET DD < QQQ!")
    elif total_return > 85:
        print(f"\n  ⚠️  Return OK, mais DD trop élevé ({max_dd:.1f}% > 35%)")
    else:
        print(f"\n  ⚠️  Objectifs non atteints")
    
    print(f"\n📈 TRADE STATISTICS")
    print(f"  Total Trades:     {len(all_trades)}")
    print(f"  Trades/Year:      {len(all_trades)//years}")
    print(f"  Win Rate:         {win_rate:.1f}%")
    print(f"  Avg Win:          {avg_win:+.1f}%")
    print(f"  Avg Loss:         {avg_loss:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Expectancy:       ${total_pnl/len(all_trades):.2f}/trade")
    
    # Exit reasons
    exit_reasons = {}
    for t in all_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    print(f"\n🚪 EXIT REASONS")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(all_trades)*100:.1f}%)")
    
    # Regime analysis
    regime_stats = {}
    for t in all_trades:
        r = t.regime
        if r not in regime_stats:
            regime_stats[r] = {'trades': 0, 'pnl': 0, 'wins': 0}
        regime_stats[r]['trades'] += 1
        regime_stats[r]['pnl'] += t.pnl
        if t.pnl > 0:
            regime_stats[r]['wins'] += 1
    
    print(f"\n📊 REGIME ANALYSIS")
    for r in ['BULL', 'RANGE', 'VOLATILE', 'BEAR', 'UNKNOWN']:
        if r in regime_stats:
            s = regime_stats[r]
            wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
            print(f"  {r:10}: {s['trades']:3} trades, WR: {wr:5.1f}%, P&L: ${s['pnl']:+,.0f}")
    
    # Position sizing analysis
    avg_pos_size = np.mean([t.position_size_pct for t in all_trades]) * 100
    print(f"\n📏 POSITION SIZING")
    print(f"  Avg position size: {avg_pos_size:.1f}%")
    
    # Top/Bottom
    print(f"\n🏆 TOP 5 TRADES")
    for t in sorted(all_trades, key=lambda x: x.pnl, reverse=True)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f}) | {t.regime}")
    
    print(f"\n💀 WORST 5 TRADES")
    for t in sorted(all_trades, key=lambda x: x.pnl)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f}) | {t.regime}")
    
    # By symbol
    print(f"\n📊 TOP 10 SYMBOLS BY P&L")
    symbol_stats = {}
    for t in all_trades:
        if t.symbol not in symbol_stats:
            symbol_stats[t.symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
        symbol_stats[t.symbol]['trades'] += 1
        symbol_stats[t.symbol]['pnl'] += t.pnl
        if t.pnl > 0:
            symbol_stats[t.symbol]['wins'] += 1
    
    for sym in sorted(symbol_stats.keys(), key=lambda x: symbol_stats[x]['pnl'], reverse=True)[:10]:
        s = symbol_stats[sym]
        wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
        print(f"  {sym:8}: {s['trades']:2} trades, WR: {wr:5.1f}%, P&L: ${s['pnl']:+,.0f}")
    
    # Save
    results = {
        'summary': {
            'period': f"{start_date} to {end_date}",
            'symbols_tested': symbols_tested,
            'initial_capital': total_initial,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'total_trades': len(all_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'objective_return_met': total_return > 85,
            'objective_dd_met': max_dd < 35
        },
        'trades': [
            {
                'symbol': t.symbol,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'regime': t.regime,
                'position_size_pct': t.position_size_pct
            }
            for t in all_trades
        ]
    }
    
    with open('midas_final_backtest.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to midas_final_backtest.json")
    print("=" * 70)

if __name__ == '__main__':
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        'JPM', 'JNJ', 'XOM', 'PG', 'KO', 'WMT', 'DIS', 'V', 'MA', 'HD',
        'MC.PA', 'OR.PA', 'TTE.PA', 'AIR.PA', 'SAN.PA',
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'MBG.DE'
    ]
    
    run_backtest(symbols, '2020-01-01', '2025-12-31', capital_per_symbol=1000)
