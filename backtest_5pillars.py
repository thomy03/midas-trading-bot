#!/usr/bin/env python3
"""
MIDAS Backtest - 5 Pillars Simulation (Daily)
==============================================
Simulates all 5 pillars for realistic backtesting.

Pillars:
1. Technical (35%) - Real data: EMA, RSI, MACD, ADX, Bollinger
2. Fundamental (20%) - Proxy: sector performance, size, long-term momentum
3. Sentiment (15%) - Simulated: volatility + momentum correlation
4. News (10%) - Simulated: earnings proximity, price gaps
5. ML/Regime (20%) - Market regime detection: bull/bear/range/volatile
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Pillar weights (must sum to 100)
WEIGHTS = {
    'technical': 35,
    'fundamental': 20,
    'sentiment': 15,
    'news': 10,
    'ml_regime': 20
}

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
    pillar_scores: Dict[str, float]

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # EMAs
    df['ema_9'] = pd.Series(close).ewm(span=9, adjust=False).mean().values
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
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ADX
    df['atr'] = calculate_atr(high, low, close, 14)
    df['adx'] = calculate_adx(high, low, close, 14)
    
    # Bollinger Bands
    df['bb_middle'] = pd.Series(close).rolling(20).mean().values
    bb_std = pd.Series(close).rolling(20).std().values
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume
    df['volume_sma'] = pd.Series(volume.astype(float)).rolling(20).mean().values
    df['volume_ratio'] = volume / df['volume_sma']
    
    # Momentum
    df['roc_10'] = pd.Series(close).pct_change(10).values * 100
    df['roc_20'] = pd.Series(close).pct_change(20).values * 100
    
    # Volatility
    df['volatility'] = pd.Series(close).pct_change().rolling(20).std().values * np.sqrt(252) * 100
    
    return df

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (simplified)."""
    tr = calculate_atr(high, low, close, 1)
    
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = pd.Series(tr).rolling(period).sum().values
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum().values / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum().values / tr_smooth
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = pd.Series(dx).rolling(period).mean().values
    return adx

# ============================================================
# PILLAR 1: TECHNICAL (35%)
# ============================================================
def score_technical(idx: int, df: pd.DataFrame) -> float:
    """Score based on technical indicators (0-100)."""
    if idx < 200:
        return 50  # Neutral
    
    score = 50  # Start neutral
    
    close = df['Close'].iloc[idx]
    ema_9 = df['ema_9'].iloc[idx]
    ema_20 = df['ema_20'].iloc[idx]
    ema_50 = df['ema_50'].iloc[idx]
    ema_200 = df['ema_200'].iloc[idx]
    rsi = df['rsi'].iloc[idx]
    macd = df['macd'].iloc[idx]
    macd_signal = df['macd_signal'].iloc[idx]
    macd_hist = df['macd_hist'].iloc[idx]
    adx = df['adx'].iloc[idx]
    bb_pct = df['bb_pct'].iloc[idx]
    volume_ratio = df['volume_ratio'].iloc[idx]
    roc_10 = df['roc_10'].iloc[idx]
    
    # Skip if NaN
    if pd.isna(ema_200) or pd.isna(rsi) or pd.isna(adx):
        return 50
    
    # EMA alignment (+/- 15)
    if ema_9 > ema_20 > ema_50 > ema_200:
        score += 15  # Perfect bullish alignment
    elif ema_20 > ema_50 > ema_200:
        score += 10
    elif ema_50 > ema_200:
        score += 5
    elif ema_9 < ema_20 < ema_50 < ema_200:
        score -= 15  # Perfect bearish
    elif ema_20 < ema_50 < ema_200:
        score -= 10
    
    # Price vs EMAs (+/- 10)
    if close > ema_20 and close > ema_50:
        score += 10
    elif close < ema_20 and close < ema_50:
        score -= 10
    
    # RSI (+/- 10)
    if 40 <= rsi <= 60:
        score += 5  # Neutral momentum, room to run
    elif 30 <= rsi < 40:
        score += 10  # Oversold, potential reversal
    elif 60 < rsi <= 70:
        score += 3  # Bullish momentum
    elif rsi > 75:
        score -= 5  # Overbought risk
    elif rsi < 25:
        score += 5  # Deeply oversold
    
    # MACD (+/- 10)
    if macd > macd_signal and macd_hist > 0:
        score += 10
    elif macd > 0 and macd > macd_signal:
        score += 5
    elif macd < macd_signal and macd_hist < 0:
        score -= 10
    
    # ADX trend strength (+/- 5)
    if adx > 25:
        # Strong trend - boost direction
        if close > ema_50:
            score += 5
        else:
            score -= 5
    
    # Bollinger position (+/- 5)
    if 0.2 <= bb_pct <= 0.5:
        score += 5  # Lower half, room to grow
    elif bb_pct > 0.95:
        score -= 3  # Near upper band
    elif bb_pct < 0.05:
        score += 3  # Near lower band, potential bounce
    
    # Volume confirmation (+/- 5)
    if volume_ratio > 1.5 and roc_10 > 0:
        score += 5  # Volume confirming uptrend
    elif volume_ratio > 1.5 and roc_10 < 0:
        score -= 3  # Volume on downside
    
    return max(0, min(100, score))

# ============================================================
# PILLAR 2: FUNDAMENTAL (20%)
# Simulated via long-term momentum and sector proxy
# ============================================================
def score_fundamental(idx: int, df: pd.DataFrame, symbol: str) -> float:
    """Fundamental score based on proxies."""
    if idx < 200:
        return 50
    
    score = 50
    
    # Long-term momentum as fundamental proxy (profitable companies trend up)
    roc_60 = df['Close'].iloc[idx] / df['Close'].iloc[idx-60] - 1 if idx >= 60 else 0
    roc_120 = df['Close'].iloc[idx] / df['Close'].iloc[idx-120] - 1 if idx >= 120 else 0
    
    # 3-month momentum (+/- 15)
    if roc_60 > 0.15:
        score += 15
    elif roc_60 > 0.05:
        score += 10
    elif roc_60 > 0:
        score += 5
    elif roc_60 < -0.15:
        score -= 15
    elif roc_60 < -0.05:
        score -= 10
    elif roc_60 < 0:
        score -= 5
    
    # 6-month trend (+/- 10)
    if roc_120 > 0.20:
        score += 10
    elif roc_120 > 0:
        score += 5
    elif roc_120 < -0.20:
        score -= 10
    elif roc_120 < 0:
        score -= 5
    
    # Sector bonus for growth stocks (tech)
    tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM', 'SAP.DE']
    if symbol in tech_symbols:
        # Tech gets slight boost in growth environments
        if roc_60 > 0:
            score += 5
    
    return max(0, min(100, score))

# ============================================================
# PILLAR 3: SENTIMENT (15%)
# Simulated via volatility and momentum correlation
# ============================================================
def score_sentiment(idx: int, df: pd.DataFrame) -> float:
    """Sentiment score simulated from price action."""
    if idx < 30:
        return 50
    
    score = 50
    
    # Recent momentum correlates with sentiment
    roc_5 = df['Close'].iloc[idx] / df['Close'].iloc[idx-5] - 1 if idx >= 5 else 0
    roc_10 = df['roc_10'].iloc[idx] if not pd.isna(df['roc_10'].iloc[idx]) else 0
    volatility = df['volatility'].iloc[idx] if not pd.isna(df['volatility'].iloc[idx]) else 20
    
    # Short-term momentum = sentiment proxy (+/- 20)
    if roc_5 > 0.05:
        score += 20  # Very bullish sentiment
    elif roc_5 > 0.02:
        score += 15
    elif roc_5 > 0:
        score += 8
    elif roc_5 < -0.05:
        score -= 20  # Very bearish sentiment
    elif roc_5 < -0.02:
        score -= 15
    elif roc_5 < 0:
        score -= 8
    
    # Volatility adjustment
    # High volatility = uncertain sentiment
    if volatility > 40:
        score = 50 + (score - 50) * 0.5  # Compress towards neutral
    
    # Trend continuation (sentiment follows trend)
    if roc_10 > 0 and roc_5 > 0:
        score += 5  # Consistent positive sentiment
    elif roc_10 < 0 and roc_5 < 0:
        score -= 5  # Consistent negative
    
    return max(0, min(100, score))

# ============================================================
# PILLAR 4: NEWS (10%)
# Simulated via price gaps and volatility spikes
# ============================================================
def score_news(idx: int, df: pd.DataFrame) -> float:
    """News score simulated from price gaps."""
    if idx < 5:
        return 50
    
    score = 50
    
    # Gap detection (news causes gaps)
    prev_close = df['Close'].iloc[idx-1]
    current_open = df['Open'].iloc[idx] if 'Open' in df.columns else df['Close'].iloc[idx]
    gap_pct = (current_open - prev_close) / prev_close
    
    # Today's range vs average
    today_range = (df['High'].iloc[idx] - df['Low'].iloc[idx]) / df['Close'].iloc[idx]
    avg_range = df['atr'].iloc[idx] / df['Close'].iloc[idx] if not pd.isna(df['atr'].iloc[idx]) else 0.02
    range_ratio = today_range / avg_range if avg_range > 0 else 1
    
    # Gap scoring (+/- 15)
    if gap_pct > 0.03:
        score += 15  # Big positive gap = good news
    elif gap_pct > 0.01:
        score += 8
    elif gap_pct < -0.03:
        score -= 15  # Big negative gap = bad news
    elif gap_pct < -0.01:
        score -= 8
    
    # High range day with positive close = news catalyst (+/- 10)
    day_return = (df['Close'].iloc[idx] - df['Open'].iloc[idx]) / df['Open'].iloc[idx] if 'Open' in df.columns else 0
    if range_ratio > 1.5:
        if day_return > 0:
            score += 10
        else:
            score -= 10
    
    return max(0, min(100, score))

# ============================================================
# PILLAR 5: ML/REGIME (20%)
# Market regime detection
# ============================================================
def detect_regime(idx: int, df: pd.DataFrame, spy_df: pd.DataFrame = None) -> Tuple[str, float]:
    """Detect market regime and return score."""
    if idx < 60:
        return 'UNKNOWN', 50
    
    # Use stock's own data for regime
    ema_20 = df['ema_20'].iloc[idx]
    ema_50 = df['ema_50'].iloc[idx]
    ema_200 = df['ema_200'].iloc[idx]
    close = df['Close'].iloc[idx]
    volatility = df['volatility'].iloc[idx] if not pd.isna(df['volatility'].iloc[idx]) else 20
    adx = df['adx'].iloc[idx] if not pd.isna(df['adx'].iloc[idx]) else 20
    roc_20 = df['roc_20'].iloc[idx] if not pd.isna(df['roc_20'].iloc[idx]) else 0
    
    # Regime detection
    if volatility > 35:
        regime = 'VOLATILE'
        # In volatile regime, be more cautious
        if close > ema_50:
            score = 55  # Slight bullish bias
        else:
            score = 40  # Bearish bias
    elif ema_20 > ema_50 > ema_200 and adx > 20:
        regime = 'BULL'
        score = 70 + min(15, roc_20)  # Bull market, boost score
    elif ema_20 < ema_50 < ema_200 and adx > 20:
        regime = 'BEAR'
        score = 30 + max(-15, roc_20)  # Bear market, reduce score
    else:
        regime = 'RANGE'
        # Range-bound, mean reversion
        bb_pct = df['bb_pct'].iloc[idx] if not pd.isna(df['bb_pct'].iloc[idx]) else 0.5
        if bb_pct < 0.3:
            score = 60  # Near lower band, potential bounce
        elif bb_pct > 0.7:
            score = 40  # Near upper band, potential pullback
        else:
            score = 50
    
    return regime, max(0, min(100, score))

def score_ml_regime(idx: int, df: pd.DataFrame) -> float:
    """ML/Regime pillar score."""
    regime, score = detect_regime(idx, df)
    return score

# ============================================================
# COMBINED SCORING
# ============================================================
def calculate_total_score(idx: int, df: pd.DataFrame, symbol: str) -> Tuple[float, Dict[str, float]]:
    """Calculate combined score from all 5 pillars."""
    
    tech = score_technical(idx, df)
    fund = score_fundamental(idx, df, symbol)
    sent = score_sentiment(idx, df)
    news = score_news(idx, df)
    ml = score_ml_regime(idx, df)
    
    pillar_scores = {
        'technical': tech,
        'fundamental': fund,
        'sentiment': sent,
        'news': news,
        'ml_regime': ml
    }
    
    # Weighted average
    total = (
        tech * WEIGHTS['technical'] +
        fund * WEIGHTS['fundamental'] +
        sent * WEIGHTS['sentiment'] +
        news * WEIGHTS['news'] +
        ml * WEIGHTS['ml_regime']
    ) / 100
    
    return total, pillar_scores

# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    initial_capital: float = 1000,
    signal_threshold: float = 50,
    stop_loss_atr_mult: float = 2.0,
    take_profit_pct: float = 0.25,
    max_hold_days: int = 30,
    trailing_stop_pct: float = 0.10
) -> Tuple[List[Trade], List[Dict]]:
    """Backtest a single symbol with 5-pillar scoring."""
    trades = []
    equity_curve = []
    position = None
    capital = initial_capital
    
    for i in range(200, len(df)):
        date = df.index[i]
        date_str = str(date)[:10]
        close = df['Close'].iloc[i]
        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else close * 0.02
        
        # Track equity
        current_equity = capital
        if position:
            current_equity += (close - position['entry_price']) * position['shares']
        
        equity_curve.append({
            'date': date_str,
            'equity': current_equity,
            'position': 1 if position else 0
        })
        
        # Check exit conditions if in position
        if position is not None:
            entry_price = position['entry_price']
            current_pnl_pct = (close - entry_price) / entry_price
            days_held = i - position['entry_idx']
            
            # Update trailing stop
            if close > position['high_since_entry']:
                position['high_since_entry'] = close
                position['trailing_stop'] = close * (1 - trailing_stop_pct)
            
            exit_reason = None
            
            # Stop loss (ATR-based)
            if close <= position['stop_loss']:
                exit_reason = 'stop_loss'
            # Trailing stop
            elif close <= position['trailing_stop']:
                exit_reason = 'trailing_stop'
            # Take profit
            elif current_pnl_pct >= take_profit_pct:
                exit_reason = 'take_profit'
            # Max hold
            elif days_held >= max_hold_days:
                exit_reason = 'max_hold'
            # Exit on very bearish score
            elif position.get('last_score', 50) < 35:
                exit_reason = 'score_exit'
            
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
                    pillar_scores=position['pillar_scores']
                ))
                capital += close * position['shares']
                position = None
        
        # Check entry if no position
        if position is None:
            total_score, pillar_scores = calculate_total_score(i, df, symbol)
            
            if total_score >= signal_threshold:
                shares = capital / close
                if shares > 0:
                    stop_loss = close - (atr * stop_loss_atr_mult)
                    
                    position = {
                        'entry_price': close,
                        'entry_date': date_str,
                        'entry_idx': i,
                        'shares': shares,
                        'stop_loss': stop_loss,
                        'trailing_stop': close * (1 - trailing_stop_pct),
                        'high_since_entry': close,
                        'entry_score': total_score,
                        'pillar_scores': pillar_scores,
                        'last_score': total_score
                    }
                    capital = 0  # All in
        else:
            # Update score for position management
            total_score, _ = calculate_total_score(i, df, symbol)
            position['last_score'] = total_score
    
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
            exit_reason='end_of_backtest',
            entry_score=position['entry_score'],
            pillar_scores=position['pillar_scores']
        ))
    
    return trades, equity_curve

# ============================================================
# MAIN
# ============================================================
def run_backtest(
    symbols: List[str],
    start_date: str = '2020-01-01',
    end_date: str = '2025-12-31',
    capital_per_symbol: float = 1000,
    signal_threshold: float = 50
):
    """Run full 5-pillar backtest."""
    
    print("=" * 70)
    print("MIDAS BACKTEST - 5 PILLARS (Daily)")
    print("=" * 70)
    print(f"\nPeriod: {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)}")
    print(f"Signal threshold: {signal_threshold}")
    print(f"Capital per symbol: ${capital_per_symbol:,}")
    print(f"\nPillar weights: {WEIGHTS}")
    
    print("\n" + "-" * 70)
    print("Running backtests...")
    
    all_trades = []
    all_equity = {}
    symbols_tested = 0
    
    for sym in symbols:
        try:
            print(f"  Loading {sym}...", end=" ", flush=True)
            data = yf.download(sym, start=start_date, end=end_date, interval='1d', progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) < 250:
                print(f"Not enough data ({len(data)} days)")
                continue
            
            data = calculate_technical_indicators(data)
            trades, equity = backtest_symbol(
                sym, data, 
                initial_capital=capital_per_symbol,
                signal_threshold=signal_threshold
            )
            
            all_equity[sym] = equity
            
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
    
    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if not all_trades:
        print("\n❌ No trades generated!")
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
    
    # Max Drawdown
    combined_equity = []
    for sym, eq in all_equity.items():
        for point in eq:
            combined_equity.append(point)
    
    # Simplified max DD calculation
    equity_values = [total_initial]
    running_pnl = 0
    for t in sorted(all_trades, key=lambda x: x.exit_date):
        running_pnl += t.pnl
        equity_values.append(total_initial + running_pnl)
    
    peak = equity_values[0]
    max_dd = 0
    for val in equity_values:
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
    
    downside_returns = [p for p in pnl_pcts if p < 0]
    downside_std = np.std(downside_returns) if downside_returns else 1
    sortino = (avg_return / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0
    
    print(f"\n📊 PERFORMANCE")
    print(f"  Initial Capital:  ${total_initial:,.0f}")
    print(f"  Final Capital:    ${final_capital:,.0f}")
    print(f"  Total Return:     {total_return:+.1f}%")
    print(f"  CAGR:             {cagr:+.1f}%")
    
    print(f"\n📈 TRADE STATISTICS")
    print(f"  Total Trades:     {len(all_trades)}")
    print(f"  Trades/Year:      {len(all_trades)//years}")
    print(f"  Win Rate:         {win_rate:.1f}%")
    print(f"  Avg Win:          {avg_win:+.1f}%")
    print(f"  Avg Loss:         {avg_loss:.1f}%")
    print(f"  Best Trade:       {max(pnl_pcts):+.1f}%")
    print(f"  Worst Trade:      {min(pnl_pcts):+.1f}%")
    
    print(f"\n📉 RISK METRICS")
    print(f"  Max Drawdown:     {max_dd:.1f}%")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Sharpe Ratio:     {sharpe:.2f}")
    print(f"  Sortino Ratio:    {sortino:.2f}")
    print(f"  Expectancy:       ${total_pnl/len(all_trades):.2f}/trade")
    
    # Exit reasons
    exit_reasons = {}
    for t in all_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    print(f"\n🚪 EXIT REASONS")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(all_trades)*100:.1f}%)")
    
    # Pillar analysis
    print(f"\n🎯 PILLAR ANALYSIS (avg entry scores)")
    pillar_avgs = {p: [] for p in WEIGHTS.keys()}
    for t in all_trades:
        for p, score in t.pillar_scores.items():
            pillar_avgs[p].append(score)
    
    for p in WEIGHTS.keys():
        avg = np.mean(pillar_avgs[p]) if pillar_avgs[p] else 50
        print(f"  {p:12}: {avg:.1f}")
    
    # Top/Bottom trades
    print(f"\n🏆 TOP 5 TRADES")
    for t in sorted(all_trades, key=lambda x: x.pnl, reverse=True)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f}) | Score: {t.entry_score:.0f}")
    
    print(f"\n💀 WORST 5 TRADES")
    for t in sorted(all_trades, key=lambda x: x.pnl)[:5]:
        print(f"  {t.symbol}: {t.entry_date} → {t.exit_date} | {t.pnl_pct:+.1f}% (${t.pnl:+,.0f}) | Score: {t.entry_score:.0f}")
    
    # Save results
    results = {
        'summary': {
            'period': f"{start_date} to {end_date}",
            'symbols_tested': symbols_tested,
            'initial_capital': total_initial,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'cagr': cagr,
            'total_trades': len(all_trades),
            'trades_per_year': len(all_trades) // years,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'expectancy': total_pnl / len(all_trades)
        },
        'trades': [
            {
                'symbol': t.symbol,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'entry_score': t.entry_score,
                'pillar_scores': t.pillar_scores
            }
            for t in all_trades
        ]
    }
    
    with open('midas_5pillars_backtest.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to midas_5pillars_backtest.json")
    print("=" * 70)

if __name__ == '__main__':
    symbols = [
        # US Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'CRM',
        # US Value/Finance
        'JPM', 'JNJ', 'XOM', 'PG', 'KO', 'WMT', 'DIS', 'V', 'MA', 'HD',
        # CAC40
        'MC.PA', 'OR.PA', 'TTE.PA', 'AIR.PA', 'SAN.PA',
        # DAX
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'MBG.DE'
    ]
    
    run_backtest(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2025-12-31',
        capital_per_symbol=1000,
        signal_threshold=50
    )
