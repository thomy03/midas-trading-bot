#!/usr/bin/env python3
"""
MIDAS Massive Backtest - 10 Years, Full Nasdaq
================================================
Backtests the 5-pillar system on 10 years of data with 3000+ Nasdaq stocks.

Features:
- 10-year historical data (2015-2025)
- All Nasdaq stocks (~3000)
- Market regime detection (bull/bear/range/volatile)
- Performance tracking by regime
- Parallel processing for speed
- SQLite storage for results

Author: Jarvis for Thomas
Created: 2026-02-05
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import json
import sqlite3
from pathlib import Path
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("/root/tradingbot-github/data")
BACKTEST_DB = DATA_DIR / "massive_backtest.db"
RESULTS_FILE = DATA_DIR / "backtest_10y_results.json"

# Configuration
CONFIG = {
    "start_date": "2015-01-01",
    "end_date": "2025-01-01",
    "initial_capital": 100000,
    "max_position_pct": 0.05,      # Max 5% per position
    "max_positions": 30,           # Max concurrent positions
    "stop_loss_pct": 0.10,         # 8% stop loss
    "take_profit_pct": 0.20,       # 15% take profit
    "min_score": 50,               # Min score to enter
    "holding_days_min": 2,
    "holding_days_max": 30,
    "batch_size": 50,              # Symbols per batch (memory management)
}

# Pillar weights
WEIGHTS = {
    'technical': 0.35,
    'fundamental': 0.20,
    'sentiment': 0.15,
    'news': 0.10,
    'ml_regime': 0.20
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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
    market_regime: str
    pillar_scores: Dict[str, float]


@dataclass
class RegimeStats:
    regime: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    best_trade_pct: float
    worst_trade_pct: float


# =============================================================================
# NASDAQ SYMBOLS
# =============================================================================

def get_nasdaq_symbols() -> List[str]:
    """Get all Nasdaq-listed symbols."""
    try:
        # Try to get from stored file first
        symbols_file = DATA_DIR / "nasdaq_symbols.json"
        if symbols_file.exists():
            with open(symbols_file) as f:
                data = json.load(f)
                if data.get('updated', '') > (datetime.now() - timedelta(days=30)).isoformat():
                    return data['symbols']
        
        # Fetch from Nasdaq FTP
        logger.info("Fetching Nasdaq symbols...")
        url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
        df = pd.read_csv(url, sep="|")
        
        symbols = df['Symbol'].dropna().tolist()
        # Filter valid symbols (letters only, 1-5 chars)
        # Filter: only letters, 1-5 chars, NO warrants/units/rights
        symbols = [s for s in symbols if s.isalpha() and 1 <= len(s) <= 5]
        # Remove warrants (ending W), units (ending U), rights (ending R after 4+ chars)
        symbols = [s for s in symbols if not (
            s.endswith('W') or 
            s.endswith('U') or 
            (len(s) >= 4 and s.endswith('R')) or
            len(s) == 5  # Most 5-letter symbols are SPACs/special
        )]
        
        # Save for later
        with open(symbols_file, 'w') as f:
            json.dump({
                'symbols': symbols,
                'updated': datetime.now().isoformat(),
                'count': len(symbols)
            }, f)
        
        logger.info(f"Found {len(symbols)} Nasdaq symbols")
        return symbols
        
    except Exception as e:
        logger.warning(f"Error fetching Nasdaq symbols: {e}")
        # Fallback to a reasonable set
        logger.info("Using curated symbol list for reliable backtest")
        return get_fallback_symbols()


def get_fallback_symbols() -> List[str]:
    """Curated list of stocks that existed throughout 2015-2025."""
    # Major stocks that have 10+ years of history
    major = [
        # FAANG + Mag 7
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
        # Tech
        "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "LRCX", "AMAT", "KLAC",
        "ASML", "TSM", "NXPI", "MRVL", "ON", "ADI", "MCHP", "SWKS",
        # Software
        "CRM", "ORCL", "ADBE", "NOW", "INTU", "SNOW", "PLTR", "DDOG",
        "ZS", "CRWD", "OKTA", "NET", "MDB", "PANW", "FTNT", "SPLK",
        # Internet
        "NFLX", "PYPL", "SHOP", "SQ", "ABNB", "UBER", "LYFT", "DASH",
        "SNAP", "PINS", "RBLX", "COIN", "HOOD", "SOFI",
        # Biotech
        "AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA", "BNTX", "ILMN",
        "ISRG", "DXCM", "ALGN", "IDXX",
        # Healthcare
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "UNH", "CVS",
        # Consumer
        "COST", "WMT", "TGT", "HD", "LOW", "SBUX", "MCD", "NKE",
        "LULU", "DECK", "ETSY", "EBAY", "W", "CHWY",
        # EV / Energy
        "LCID", "RIVN", "NIO", "XPEV", "LI", "ENPH", "SEDG", "FSLR",
        # Finance
        "V", "MA", "AXP", "GS", "MS", "JPM", "BAC", "C", "WFC",
        # Industrial
        "CAT", "DE", "BA", "LMT", "RTX", "GE", "HON",
        # Telecom/Media
        "T", "VZ", "TMUS", "CMCSA", "DIS", "NFLX", "WBD", "PARA",
    ]
    
    # Mid-caps and small caps with history
    midcaps = [
        "ROKU", "TTD", "TWLO", "ZM", "DOCU", "WDAY", "HUBS",
        "DKNG", "PENN", "MGM", "LVS", "WYNN",
        "GME", "BB", "NOK",
        "SMCI", "DELL", "HPE", "HPQ", "NTAP", "WDC", "STX",
        # More established mid-caps
        "TEAM", "ZS", "CRWD", "NET", "DDOG", "MDB", "VEEV",
        "SPLK", "PANW", "FTNT", "CYBR", "OKTA",
        # Biotech established
        "ALNY", "SGEN", "BMRN", "INCY", "EXEL", "JAZZ", "NBIX",
        # Finance
        "MKTX", "CBOE", "NDAQ", "ICE", "CME", "SCHW", "IBKR",
        # Retail
        "ROST", "TJX", "DLTR", "DG", "FIVE", "ULTA", "ORLY", "AZO",
        # Industrial
        "ODFL", "JBHT", "CHRW", "EXPD", "XPO", "UBER", "LYFT",
        # More tech
        "SNPS", "CDNS", "ANSS", "KEYS", "MANH", "EPAM", "GLOB",
        # Healthcare
        "VEEV", "HOLX", "TECH", "BIO", "A", "TMO", "DHR", "IQV",
        # Communications
        "GOOGL", "META", "SNAP", "PINS", "MTCH",
        # Semiconductors
        "MRVL", "SWKS", "QRVO", "WOLF", "CRUS", "SLAB", "MPWR",
        # Software
        "ADSK", "ANSYS", "CDNS", "PTC", "FICO", "PAYC", "PCTY",
    ]
    
    # Total ~200 quality stocks with history
    return list(set(major + midcaps))


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators."""
    if len(df) < 200:
        return df
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values.astype(float)
    
    # EMAs
    df['ema_9'] = pd.Series(close).ewm(span=9, adjust=False).mean().values
    df['ema_20'] = pd.Series(close).ewm(span=20, adjust=False).mean().values
    df['ema_50'] = pd.Series(close).ewm(span=50, adjust=False).mean().values
    df['ema_200'] = pd.Series(close).ewm(span=200, adjust=False).mean().values
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50).values
    
    # MACD
    exp1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(close).ewm(span=26, adjust=False).mean()
    df['macd'] = (exp1 - exp2).values
    df['macd_signal'] = pd.Series(df['macd']).ewm(span=9, adjust=False).mean().values
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ADX (simplified)
    tr = np.maximum(high - low, 
                   np.maximum(np.abs(high - np.roll(close, 1)),
                             np.abs(low - np.roll(close, 1))))
    df['atr'] = pd.Series(tr).rolling(14).mean().values
    
    # Calculate +DI and -DI
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    atr14 = pd.Series(tr).rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr14
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = pd.Series(dx).rolling(14).mean().fillna(25).values
    
    # Bollinger Bands
    df['bb_middle'] = pd.Series(close).rolling(20).mean().values
    bb_std = pd.Series(close).rolling(20).std().values
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_pct'] = np.where(
        (df['bb_upper'] - df['bb_lower']) > 0,
        (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']),
        0.5
    )
    
    # Volume
    df['volume_sma'] = pd.Series(volume).rolling(20).mean().values
    df['volume_ratio'] = np.where(df['volume_sma'] > 0, volume / df['volume_sma'], 1)
    
    # Momentum
    df['roc_10'] = pd.Series(close).pct_change(10).values * 100
    df['roc_20'] = pd.Series(close).pct_change(20).values * 100
    df['roc_60'] = pd.Series(close).pct_change(60).values * 100
    
    # Volatility
    df['volatility'] = pd.Series(close).pct_change().rolling(20).std().values * np.sqrt(252) * 100
    
    return df


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

def detect_market_regime(spy_data: pd.DataFrame, date: str) -> str:
    """
    Detect market regime based on SPY data at given date.
    
    Returns: 'BULL', 'BEAR', 'RANGE', or 'VOLATILE'
    """
    try:
        # Get data up to this date
        mask = spy_data.index <= pd.Timestamp(date)
        data = spy_data[mask].tail(60)  # Last 60 days
        
        if len(data) < 20:
            return "RANGE"
        
        close = data["Close"].values.flatten()
        
        # Calculate metrics
        sma_20 = np.mean(close[-20:])
        sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
        current = close[-1]
        
        # Trend (20-day return)
        ret_20d = (close[-1] / close[-20] - 1) * 100
        
        # Volatility
        vol = np.std(np.diff(close) / close[:-1]) * np.sqrt(252) * 100
        
        # VIX proxy (ATR-based)
        high = data["High"].values.flatten()
        low = data["Low"].values.flatten()
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                 np.abs(low - np.roll(close, 1))))
        atr_pct = (np.mean(tr[-14:]) / current) * 100
        
        # Classify regime - more sensitive thresholds
        if vol > 25 or atr_pct > 2.0:
            return "VOLATILE"
        elif ret_20d > 2 and current > sma_20:  # Easier BULL detection
            return "BULL"
        elif ret_20d < -2 and current < sma_20:  # Easier BEAR detection
            return "BEAR"
        else:
            return "RANGE"
            
    except Exception as e:
        return "RANGE"


# =============================================================================
# PILLAR SCORING (Simplified for backtest)
# =============================================================================

def calculate_pillar_scores(row: pd.Series, regime: str) -> Dict[str, float]:
    """Calculate all pillar scores for a given day."""
    
    scores = {}
    
    # 1. TECHNICAL (35%)
    tech_score = 0
    
    # Trend (EMA alignment) - STRONGER signals
    if row.get('Close', 0) > row.get('ema_20', 0) > row.get('ema_50', 0) > row.get('ema_200', 0):
        tech_score += 40  # Perfect uptrend alignment
    elif row.get('Close', 0) > row.get('ema_20', 0) > row.get('ema_50', 0):
        tech_score += 30
    elif row.get('Close', 0) > row.get('ema_20', 0):
        tech_score += 15
    elif row.get('Close', 0) < row.get('ema_20', 0) < row.get('ema_50', 0):
        tech_score -= 25
    
    # RSI
    rsi = row.get('rsi', 50)
    if 30 <= rsi <= 70:
        if rsi < 40:
            tech_score += 15  # Oversold but not extreme
        elif rsi > 60:
            tech_score -= 10  # Getting overbought
    elif rsi < 30:
        tech_score += 20  # Very oversold
    elif rsi > 70:
        tech_score -= 20  # Very overbought
    
    # MACD
    if row.get('macd_hist', 0) > 0:
        tech_score += 15
    else:
        tech_score -= 10
    
    # Bollinger
    bb_pct = row.get('bb_pct', 0.5)
    if bb_pct < 0.2:
        tech_score += 15  # Near lower band
    elif bb_pct > 0.8:
        tech_score -= 15  # Near upper band
    
    # Volume confirmation
    vol_ratio = row.get('volume_ratio', 1)
    if vol_ratio > 1.5:
        tech_score += 10
    
    # ADX (trend strength)
    adx = row.get('adx', 25)
    if adx > 25:
        tech_score = tech_score * 1.2  # Strengthen signal in trending market
    
    scores['technical'] = max(-100, min(100, tech_score))
    
    # 2. FUNDAMENTAL (20%) - Simulated based on long-term momentum
    fund_score = 0
    roc_60 = row.get('roc_60', 0)
    if roc_60 > 20:
        fund_score = 30
    elif roc_60 > 10:
        fund_score = 20
    elif roc_60 > 0:
        fund_score = 10
    elif roc_60 > -10:
        fund_score = -10
    else:
        fund_score = -30
    
    scores['fundamental'] = fund_score
    
    # 3. SENTIMENT (15%) - Simulated based on volatility and momentum
    sent_score = 0
    vol = row.get('volatility', 20)
    roc_10 = row.get('roc_10', 0)
    
    # Positive momentum + moderate volatility = bullish sentiment
    if roc_10 > 5 and vol < 40:
        sent_score = 25
    elif roc_10 > 0 and vol < 30:
        sent_score = 15
    elif roc_10 < -5:
        sent_score = -20
    
    scores['sentiment'] = sent_score
    
    # 4. NEWS (10%) - Simulated based on gap detection
    news_score = 0
    # In real backtest, we'd track earnings. Here, approximate with gaps
    open_price = row.get('Open', row.get('Close', 0))
    prev_close = row.get('prev_close', open_price)
    if prev_close > 0:
        gap_pct = (open_price - prev_close) / prev_close * 100
        if gap_pct > 3:
            news_score = 20  # Positive gap
        elif gap_pct < -3:
            news_score = -20  # Negative gap
    
    scores['news'] = news_score
    
    # 5. ML/REGIME (20%) - Based on market regime alignment
    ml_score = 0
    
    if regime == "BULL":
        # In bull market, BE AGGRESSIVE on trend-following
        if row.get('Close', 0) > row.get('ema_50', 0):
            ml_score = 40  # Strong bonus for being in uptrend
        elif row.get('Close', 0) > row.get('ema_20', 0):
            ml_score = 25  # Still bullish if above short-term
        else:
            ml_score = 0   # Neutral, not negative in bull market
    elif regime == "BEAR":
        # In bear market, be very selective
        if rsi < 30 and row.get('macd_hist', 0) > 0:
            ml_score = 20  # Oversold reversal
        else:
            ml_score = -30
    elif regime == "VOLATILE":
        # In volatile market, require strong signals
        if tech_score > 30:
            ml_score = 15
        else:
            ml_score = -20
    else:  # RANGE
        # Mean reversion works better
        if bb_pct < 0.3 and rsi < 40:
            ml_score = 25
        elif bb_pct > 0.7 and rsi > 60:
            ml_score = -25
    
    scores['ml_regime'] = ml_score
    
    return scores


def calculate_final_score(pillar_scores: Dict[str, float]) -> float:
    """Calculate weighted final score (0-100 scale)."""
    weighted = sum(pillar_scores.get(p, 0) * w for p, w in WEIGHTS.items())
    # Convert from -100/+100 to 0-100
    return (weighted + 100) / 2


# =============================================================================
# BACKTESTER
# =============================================================================

class MassiveBacktester:
    """
    Runs massive backtest on thousands of stocks over 10 years.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[str, float]] = []
        self.regime_trades: Dict[str, List[Trade]] = {
            "BULL": [], "BEAR": [], "RANGE": [], "VOLATILE": []
        }
        
        # Initialize database
        self._init_db()
        
        # SPY data for regime detection
        self.spy_data = None
        
    def _init_db(self):
        """Initialize SQLite database for results."""
        conn = sqlite3.connect(BACKTEST_DB)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                entry_date TEXT,
                entry_price REAL,
                exit_date TEXT,
                exit_price REAL,
                shares REAL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                entry_score REAL,
                market_regime TEXT,
                pillar_scores TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS equity_curve (
                date TEXT PRIMARY KEY,
                equity REAL,
                regime TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS regime_stats (
                regime TEXT PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                avg_pnl_pct REAL,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_spy_data(self):
        """Load SPY data for regime detection with retry."""
        logger.info("Loading SPY data for regime detection...")
        for attempt in range(3):
            try:
                self.spy_data = yf.download(
                    "SPY",
                    start=self.config["start_date"],
                    end=self.config["end_date"],
                    progress=False
                )
                if len(self.spy_data) > 0:
                    logger.info(f"Loaded {len(self.spy_data)} days of SPY data")
                    return
                time.sleep(2)
            except Exception as e:
                logger.warning(f"SPY download attempt {attempt+1} failed: {e}")
                time.sleep(2)
        
        # If still failed, use fallback
        logger.warning("Using QQQ as fallback for regime detection")
        self.spy_data = yf.download(
            "QQQ",
            start=self.config["start_date"],
            end=self.config["end_date"],
            progress=False
        )
        logger.info(f"Loaded {len(self.spy_data)} days of QQQ data (fallback)")
    
    def _download_batch(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Download data for a batch of symbols."""
        data = {}
        
        try:
            batch_str = " ".join(symbols)
            df = yf.download(
                batch_str,
                start=self.config["start_date"],
                end=self.config["end_date"],
                progress=False,
                threads=True,
                group_by='ticker'
            )
            
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        symbol_df = df
                    else:
                        symbol_df = df[symbol].dropna()
                    
                    if len(symbol_df) >= 200:
                        # Add prev_close for gap detection
                        symbol_df = symbol_df.copy()
                        symbol_df['prev_close'] = symbol_df['Close'].shift(1)
                        data[symbol] = calculate_indicators(symbol_df)
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Batch download error: {e}")
        
        return data
    
    def run_backtest(self, symbols: List[str] = None, max_symbols: int = None):
        """
        Run the full backtest.
        
        Args:
            symbols: List of symbols (None = all Nasdaq)
            max_symbols: Limit number of symbols (for testing)
        """
        if symbols is None:
            symbols = json.load(open("data/clean_symbols.json"))["symbols"]
        
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        logger.info(f"Starting backtest: {len(symbols)} symbols, "
                   f"{self.config['start_date']} to {self.config['end_date']}")
        
        # Load SPY for regime detection
        self._load_spy_data()
        
        # Initialize portfolio
        capital = self.config["initial_capital"]
        positions = {}  # symbol -> {shares, entry_price, entry_date, entry_score, regime}
        
        # Process symbols in batches
        all_data = {}
        batch_size = self.config["batch_size"]
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"Downloading batch {i//batch_size + 1}/{len(symbols)//batch_size + 1} "
                       f"({len(batch)} symbols)...")
            
            batch_data = self._download_batch(batch)
            all_data.update(batch_data)
            
            time.sleep(0.5)  # Rate limiting
        
        logger.info(f"Downloaded data for {len(all_data)} symbols")
        
        # Get all trading dates from SPY
        trading_dates = self.spy_data.index.tolist()
        
        # Simulate day by day
        for i, date in enumerate(trading_dates[200:]):  # Skip first 200 for indicators
            date_str = date.strftime("%Y-%m-%d")
            
            # Progress logging
            if i % 250 == 0:
                logger.info(f"Processing {date_str}... Capital: ${capital:,.0f}, "
                           f"Positions: {len(positions)}, Trades: {len(self.trades)}")
            
            # Detect market regime
            regime = detect_market_regime(self.spy_data, date_str)
            
            # Check exits first
            for symbol in list(positions.keys()):
                if symbol not in all_data:
                    continue
                
                df = all_data[symbol]
                if date not in df.index:
                    continue
                
                row = df.loc[date]
                pos = positions[symbol]
                
                current_price = row['Close']
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                hold_days = (date - pd.Timestamp(pos['entry_date'])).days
                
                # Check exit conditions
                exit_reason = None
                
                if pnl_pct <= -self.config["stop_loss_pct"] * 100:
                    exit_reason = "stop_loss"
                elif pnl_pct >= self.config["take_profit_pct"] * 100:
                    exit_reason = "take_profit"
                elif hold_days >= self.config["holding_days_max"]:
                    exit_reason = "time_exit"
                
                if exit_reason:
                    pnl = (current_price - pos['entry_price']) * pos['shares']
                    capital += pos['shares'] * current_price
                    
                    trade = Trade(
                        symbol=symbol,
                        entry_date=pos['entry_date'],
                        entry_price=pos['entry_price'],
                        exit_date=date_str,
                        exit_price=current_price,
                        shares=pos['shares'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        entry_score=pos['entry_score'],
                        market_regime=pos['regime'],
                        pillar_scores=pos.get('pillar_scores', {})
                    )
                    
                    self.trades.append(trade)
                    self.regime_trades[pos['regime']].append(trade)
                    del positions[symbol]
            
            # Check entries
            if len(positions) < self.config["max_positions"]:
                # Score all symbols
                candidates = []
                
                for symbol, df in all_data.items():
                    if symbol in positions:
                        continue
                    if date not in df.index:
                        continue
                    
                    row = df.loc[date]
                    
                    # Calculate scores
                    pillar_scores = calculate_pillar_scores(row, regime)
                    final_score = calculate_final_score(pillar_scores)
                    
                    if final_score >= self.config["min_score"]:
                        candidates.append({
                            "symbol": symbol,
                            "score": final_score,
                            "price": row['Close'],
                            "pillar_scores": pillar_scores
                        })
                
                # Sort by score and take top candidates
                candidates.sort(key=lambda x: x['score'], reverse=True)
                slots_available = self.config["max_positions"] - len(positions)
                
                for cand in candidates[:slots_available]:
                    # Position sizing
                    max_position = capital * self.config["max_position_pct"]
                    shares = int(max_position / cand['price'])
                    
                    if shares > 0 and shares * cand['price'] <= capital:
                        capital -= shares * cand['price']
                        
                        positions[cand['symbol']] = {
                            "shares": shares,
                            "entry_price": cand['price'],
                            "entry_date": date_str,
                            "entry_score": cand['score'],
                            "regime": regime,
                            "pillar_scores": cand['pillar_scores']
                        }
            
            # Record equity
            portfolio_value = capital + sum(
                all_data[s].loc[date]['Close'] * p['shares']
                for s, p in positions.items()
                if s in all_data and date in all_data[s].index
            )
            self.equity_curve.append((date_str, portfolio_value))
        
        # Close remaining positions at end
        for symbol, pos in positions.items():
            if symbol in all_data:
                df = all_data[symbol]
                if len(df) > 0:
                    last_price = df['Close'].iloc[-1]
                    pnl_pct = (last_price - pos['entry_price']) / pos['entry_price'] * 100
                    pnl = (last_price - pos['entry_price']) * pos['shares']
                    
                    trade = Trade(
                        symbol=symbol,
                        entry_date=pos['entry_date'],
                        entry_price=pos['entry_price'],
                        exit_date=self.config["end_date"],
                        exit_price=last_price,
                        shares=pos['shares'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="end_of_backtest",
                        entry_score=pos['entry_score'],
                        market_regime=pos['regime'],
                        pillar_scores=pos.get('pillar_scores', {})
                    )
                    
                    self.trades.append(trade)
                    self.regime_trades[pos['regime']].append(trade)
        
        logger.info(f"Backtest complete: {len(self.trades)} trades")
        
        # Save results
        self._save_results()
        
        return self.generate_report()
    
    def _save_results(self):
        """Save results to database and JSON."""
        conn = sqlite3.connect(BACKTEST_DB)
        c = conn.cursor()
        
        # Clear old data
        c.execute('DELETE FROM trades')
        c.execute('DELETE FROM equity_curve')
        c.execute('DELETE FROM regime_stats')
        
        # Save trades
        for trade in self.trades:
            c.execute('''
                INSERT INTO trades 
                (symbol, entry_date, entry_price, exit_date, exit_price, shares,
                 pnl, pnl_pct, exit_reason, entry_score, market_regime, pillar_scores)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.symbol, trade.entry_date, trade.entry_price,
                trade.exit_date, trade.exit_price, trade.shares,
                trade.pnl, trade.pnl_pct, trade.exit_reason,
                trade.entry_score, trade.market_regime,
                json.dumps(trade.pillar_scores)
            ))
        
        # Save equity curve
        for date, equity in self.equity_curve:
            c.execute('''
                INSERT INTO equity_curve (date, equity) VALUES (?, ?)
            ''', (date, equity))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(self.trades)} trades to database")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive backtest report."""
        
        if not self.trades:
            return {"error": "No trades executed"}
        
        # Overall stats
        total_pnl = sum(t.pnl for t in self.trades)
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl < 0]
        
        # Equity curve analysis
        if self.equity_curve:
            equities = [e[1] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for e in equities:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            
            total_return = (equities[-1] / equities[0] - 1) * 100
            
            # Sharpe ratio (simplified)
            returns = np.diff(equities) / equities[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            total_return = 0
            max_dd = 0
            sharpe = 0
        
        # Regime breakdown
        regime_stats = {}
        for regime, trades in self.regime_trades.items():
            if trades:
                wins = [t for t in trades if t.pnl > 0]
                regime_stats[regime] = {
                    "total_trades": len(trades),
                    "winning_trades": len(wins),
                    "win_rate": len(wins) / len(trades) * 100,
                    "avg_pnl_pct": np.mean([t.pnl_pct for t in trades]),
                    "total_pnl": sum(t.pnl for t in trades),
                    "best_trade": max(t.pnl_pct for t in trades),
                    "worst_trade": min(t.pnl_pct for t in trades),
                }
            else:
                regime_stats[regime] = {"total_trades": 0}
        
        report = {
            "summary": {
                "period": f"{self.config['start_date']} to {self.config['end_date']}",
                "initial_capital": self.config["initial_capital"],
                "final_capital": self.equity_curve[-1][1] if self.equity_curve else 0,
                "total_return_pct": total_return,
                "total_pnl": total_pnl,
                "total_trades": len(self.trades),
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": len(winning) / len(self.trades) * 100 if self.trades else 0,
                "avg_win_pct": np.mean([t.pnl_pct for t in winning]) if winning else 0,
                "avg_loss_pct": np.mean([t.pnl_pct for t in losing]) if losing else 0,
                "max_drawdown_pct": max_dd,
                "sharpe_ratio": sharpe,
                "profit_factor": abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing)) if losing and sum(t.pnl for t in losing) != 0 else 0,
            },
            "by_regime": regime_stats,
            "by_exit_reason": {},
            "top_winners": [],
            "top_losers": [],
        }
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            if t.exit_reason not in exit_reasons:
                exit_reasons[t.exit_reason] = {"count": 0, "total_pnl": 0}
            exit_reasons[t.exit_reason]["count"] += 1
            exit_reasons[t.exit_reason]["total_pnl"] += t.pnl
        report["by_exit_reason"] = exit_reasons
        
        # Top trades
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl_pct, reverse=True)
        report["top_winners"] = [
            {"symbol": t.symbol, "pnl_pct": t.pnl_pct, "regime": t.market_regime}
            for t in sorted_trades[:10]
        ]
        report["top_losers"] = [
            {"symbol": t.symbol, "pnl_pct": t.pnl_pct, "regime": t.market_regime}
            for t in sorted_trades[-10:]
        ]
        
        # Save report
        with open(RESULTS_FILE, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# =============================================================================
# MAIN
# =============================================================================

def print_report(report: Dict):
    """Pretty print the backtest report."""
    
    print("\n" + "="*60)
    print("MIDAS BACKTEST REPORT - 10 YEARS")
    print("="*60)
    
    s = report["summary"]
    print(f"\n📅 Period: {s['period']}")
    print(f"💰 Initial: ${s['initial_capital']:,.0f}")
    print(f"💎 Final: ${s['final_capital']:,.0f}")
    print(f"📈 Total Return: {s['total_return_pct']:.1f}%")
    print(f"📊 Total P&L: ${s['total_pnl']:,.0f}")
    
    print(f"\n📋 TRADE STATISTICS")
    print(f"   Total trades: {s['total_trades']}")
    print(f"   Win rate: {s['win_rate']:.1f}%")
    print(f"   Avg win: +{s['avg_win_pct']:.1f}%")
    print(f"   Avg loss: {s['avg_loss_pct']:.1f}%")
    print(f"   Max drawdown: {s['max_drawdown_pct']:.1f}%")
    print(f"   Sharpe ratio: {s['sharpe_ratio']:.2f}")
    print(f"   Profit factor: {s['profit_factor']:.2f}")
    
    print(f"\n🎯 BY MARKET REGIME")
    for regime, stats in report["by_regime"].items():
        if stats.get("total_trades", 0) > 0:
            print(f"   {regime}:")
            print(f"      Trades: {stats['total_trades']}")
            print(f"      Win rate: {stats['win_rate']:.1f}%")
            print(f"      Avg P&L: {stats['avg_pnl_pct']:.1f}%")
            print(f"      Total P&L: ${stats['total_pnl']:,.0f}")
    
    print(f"\n🏆 TOP WINNERS")
    for t in report["top_winners"][:5]:
        print(f"   {t['symbol']}: +{t['pnl_pct']:.1f}% ({t['regime']})")
    
    print(f"\n💀 TOP LOSERS")
    for t in report["top_losers"][:5]:
        print(f"   {t['symbol']}: {t['pnl_pct']:.1f}% ({t['regime']})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    # Parse args
    max_symbols = None
    if len(sys.argv) > 1:
        try:
            max_symbols = int(sys.argv[1])
        except:
            pass
    
    # Run backtest
    backtester = MassiveBacktester()
    report = backtester.run_backtest(max_symbols=max_symbols)
    
    # Print results
    print_report(report)
