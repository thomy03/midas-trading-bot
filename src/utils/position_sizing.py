"""
Position Sizing Calculator - Risk-based position sizing

Calculates optimal position size based on:
- Available capital (minus open positions)
- Risk per trade (varies by market cap)
- Stop-loss fixed at lowest price corresponding to lowest RSI peak of trendline
- EUR/USD conversion for European accounts trading US stocks

Usage with IBKR FX conversion:
    from src.utils.position_sizing import AsyncPositionSizer

    sizer = AsyncPositionSizer(
        total_capital=1500,  # EUR
        account_currency='EUR',
        ibkr_client=ib
    )

    # Calculate with FX conversion
    position = await sizer.calculate_with_fx(
        df=df,
        entry_price=175.0,  # USD
        symbol='AAPL',
        market_cap=3000
    )
    print(f"Buy {position.shares} shares, FX rate: {position.fx_rate}")
"""

import json
import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of position sizing calculation"""
    shares: int  # Number of shares to buy
    position_value: float  # Total position value in currency
    stop_loss: float  # Stop-loss price (fixed at lowest RSI peak price)
    risk_amount: float  # Max loss if stop hit
    risk_pct: float  # Risk % used (varies by market cap)
    entry_price: float  # Entry price used
    stop_source: str  # Description of how stop was determined


@dataclass
class PositionSizeWithFX(PositionSize):
    """
    Position sizing result with FX conversion information.

    Used when trading US stocks from a EUR account.
    All prices are in the stock's currency (USD), but capital is converted.
    """
    fx_rate: float = 1.0  # EUR/USD rate used (e.g., 1.08)
    account_currency: str = 'EUR'  # Account currency
    stock_currency: str = 'USD'  # Stock trading currency
    capital_in_stock_currency: float = 0.0  # Capital converted to USD
    risk_in_account_currency: float = 0.0  # Risk amount in EUR

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'shares': self.shares,
            'position_value': self.position_value,
            'stop_loss': self.stop_loss,
            'risk_amount': self.risk_amount,
            'risk_pct': self.risk_pct,
            'entry_price': self.entry_price,
            'stop_source': self.stop_source,
            'fx_rate': self.fx_rate,
            'account_currency': self.account_currency,
            'stock_currency': self.stock_currency,
            'capital_in_stock_currency': self.capital_in_stock_currency,
            'risk_in_account_currency': self.risk_in_account_currency
        }


@dataclass
class OpenPosition:
    """Tracks an open position"""
    symbol: str
    shares: int
    entry_price: float
    entry_date: str  # ISO format
    stop_loss: float
    position_value: float

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'stop_loss': self.stop_loss,
            'position_value': self.position_value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'OpenPosition':
        return cls(
            symbol=data['symbol'],
            shares=data['shares'],
            entry_price=data['entry_price'],
            entry_date=data['entry_date'],
            stop_loss=data['stop_loss'],
            position_value=data['position_value']
        )


class PortfolioTracker:
    """
    Tracks open positions and calculates available capital

    Usage:
        tracker = PortfolioTracker(total_capital=10000)
        tracker.add_position('AAPL', shares=10, entry_price=150, stop_loss=145)
        available = tracker.get_available_capital()
    """

    def __init__(self, total_capital: float = 10000.0, portfolio_file: str = None):
        """
        Initialize portfolio tracker

        Args:
            total_capital: Total trading capital
            portfolio_file: Path to JSON file for persistence (optional)
        """
        self.total_capital = total_capital
        self.positions: List[OpenPosition] = []
        self.portfolio_file = portfolio_file or 'data/portfolio.json'
        self._load_positions()

    def _load_positions(self):
        """Load positions from JSON file if exists"""
        try:
            path = Path(self.portfolio_file)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.total_capital = data.get('total_capital', self.total_capital)
                    self.positions = [
                        OpenPosition.from_dict(p) for p in data.get('positions', [])
                    ]
        except Exception:
            self.positions = []

    def _save_positions(self):
        """Save positions to JSON file"""
        try:
            path = Path(self.portfolio_file)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'total_capital': self.total_capital,
                'updated': datetime.now().isoformat(),
                'positions': [p.to_dict() for p in self.positions]
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save portfolio: {e}")

    def add_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
        entry_date: str = None
    ) -> OpenPosition:
        """
        Add a new open position

        Args:
            symbol: Stock symbol
            shares: Number of shares
            entry_price: Entry price per share
            stop_loss: Stop-loss price
            entry_date: Entry date (ISO format, defaults to now)

        Returns:
            Created OpenPosition
        """
        position = OpenPosition(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            entry_date=entry_date or datetime.now().strftime('%Y-%m-%d'),
            stop_loss=stop_loss,
            position_value=shares * entry_price
        )
        self.positions.append(position)
        self._save_positions()
        return position

    def close_position(self, symbol: str) -> Optional[OpenPosition]:
        """
        Close a position by symbol

        Args:
            symbol: Stock symbol to close

        Returns:
            Closed position or None if not found
        """
        for i, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                closed = self.positions.pop(i)
                self._save_positions()
                return closed
        return None

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        """Get position by symbol"""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def get_invested_capital(self) -> float:
        """Get total capital currently invested"""
        return sum(p.position_value for p in self.positions)

    def get_available_capital(self) -> float:
        """Get capital available for new positions"""
        return max(0, self.total_capital - self.get_invested_capital())

    def update_total_capital(self, new_capital: float):
        """Update total capital"""
        self.total_capital = new_capital
        self._save_positions()

    def get_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'total_capital': self.total_capital,
            'invested_capital': self.get_invested_capital(),
            'available_capital': self.get_available_capital(),
            'num_positions': len(self.positions),
            'positions': [p.to_dict() for p in self.positions]
        }

    def clear_all_positions(self):
        """Clear all positions (use with caution)"""
        self.positions = []
        self._save_positions()


def get_risk_pct_by_market_cap(market_cap: float) -> float:
    """
    Get risk percentage based on market cap

    Larger companies = lower volatility = higher risk % allowed

    Args:
        market_cap: Market cap in billions USD

    Returns:
        Risk percentage (0.01 to 0.05)
    """
    if market_cap is None or market_cap <= 0:
        return 0.015  # Default conservative 1.5%

    # Market cap tiers (in billions)
    if market_cap >= 200:  # Mega cap (>200B): AAPL, MSFT, NVDA, etc.
        return 0.04  # 4% risk
    elif market_cap >= 50:  # Large cap (50-200B)
        return 0.03  # 3% risk
    elif market_cap >= 10:  # Mid cap (10-50B)
        return 0.025  # 2.5% risk
    elif market_cap >= 2:  # Small cap (2-10B)
        return 0.02  # 2% risk
    else:  # Micro cap (<2B)
        return 0.015  # 1.5% risk - more conservative


def find_stop_loss_from_rsi_trendline(
    df: pd.DataFrame,
    rsi_peaks_indices: List[int],
    fallback_pct: float = 0.05
) -> tuple:
    """
    Find stop-loss price at the moment when RSI was at its lowest during the trendline period

    The stop-loss is the LOW price at the bar where RSI reached its MINIMUM
    during the RSI trendline period (from first peak to END OF DATA/breakout).

    IMPORTANT: The period extends from first peak to the end of the data (the breakout),
    NOT just to the last peak. The trendline continues beyond the last peak until broken.

    Args:
        df: DataFrame with OHLCV data
        rsi_peaks_indices: List of indices where RSI peaks form the trendline
        fallback_pct: Fallback stop distance as % of price if no peaks provided

    Returns:
        (stop_loss_price, stop_source_description)
    """
    if df is None or len(df) == 0:
        return (0, "No data available")

    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
    else:
        close = df['Close']
        low = df['Low']

    current_price = float(close.iloc[-1])

    # If no RSI peaks provided, use fallback
    if not rsi_peaks_indices or len(rsi_peaks_indices) == 0:
        stop_loss = current_price * (1 - fallback_pct)
        return (stop_loss, f"Fallback: {fallback_pct*100:.0f}% below entry")

    # Filter valid indices
    valid_indices = [i for i in rsi_peaks_indices if 0 <= i < len(df)]

    if not valid_indices:
        stop_loss = current_price * (1 - fallback_pct)
        return (stop_loss, f"Fallback: {fallback_pct*100:.0f}% below entry")

    # Get the range of the RSI trendline period (first peak to END OF DATA/breakout)
    # The trendline continues beyond the last peak until broken
    start_idx = min(valid_indices)
    end_idx = len(df) - 1  # Use end of data (breakout), NOT max(valid_indices)

    # Calculate RSI for the period
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero: add small epsilon to loss
    rs = gain / (loss + np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))

    # Find where RSI was at its MINIMUM during the trendline period
    period_rsi = rsi.iloc[start_idx:end_idx + 1]

    # Get the index where RSI was lowest
    rsi_min_idx = period_rsi.idxmin()

    # Get the position index (integer) for iloc access
    if hasattr(rsi_min_idx, 'strftime'):
        # It's a datetime index, need to find its position
        rsi_min_pos = df.index.get_loc(rsi_min_idx)
    else:
        rsi_min_pos = rsi_min_idx

    # Stop-loss is the LOW price at the bar where RSI was minimum
    stop_loss = float(low.iloc[rsi_min_pos])

    # Get the date and RSI value for description
    try:
        if hasattr(rsi_min_idx, 'strftime'):
            stop_date = rsi_min_idx.strftime('%Y-%m-%d')
        else:
            stop_date = str(rsi_min_idx)
        rsi_value = float(period_rsi.min())
    except (ValueError, TypeError, AttributeError):
        stop_date = "unknown"
        rsi_value = 0

    return (stop_loss, f"Low at RSI minimum {rsi_value:.0f} ({stop_date})")


def calculate_position_size(
    df: pd.DataFrame,
    entry_price: float,
    capital: float = 10000.0,
    market_cap: float = None,
    rsi_peaks_indices: List[int] = None,
    max_position_pct: float = 0.25,
    risk_pct_override: float = None
) -> PositionSize:
    """
    Calculate position size based on risk management

    Formula:
    - Stop Loss = Lowest price at lowest RSI peak of trendline
    - Risk % = Based on market cap (larger = higher %)
    - Risk Amount = Available Capital * Risk %
    - Position Size = Risk Amount / (Entry - Stop Loss)

    Args:
        df: DataFrame with OHLCV data
        entry_price: Intended entry price
        capital: Available trading capital (after existing positions)
        market_cap: Market cap in billions for risk % calculation
        rsi_peaks_indices: Indices of RSI peaks forming the trendline
        max_position_pct: Max single position as % of capital
        risk_pct_override: Force a specific risk % (overrides market cap calc)

    Returns:
        PositionSize with all calculated values
    """
    # Determine risk percentage
    if risk_pct_override is not None:
        risk_pct = risk_pct_override
    else:
        risk_pct = get_risk_pct_by_market_cap(market_cap)

    # Find stop-loss from RSI trendline
    stop_loss, stop_source = find_stop_loss_from_rsi_trendline(
        df, rsi_peaks_indices, fallback_pct=0.05
    )

    # Validate stop-loss
    if stop_loss <= 0 or stop_loss >= entry_price:
        # Invalid stop, use 5% fallback
        stop_loss = entry_price * 0.95
        stop_source = "Fallback: 5% below entry (invalid RSI stop)"

    # Validate stop-loss is reasonable (2-20% from entry)
    stop_distance_pct = (entry_price - stop_loss) / entry_price * 100
    if stop_distance_pct < 2.0:
        # Stop too tight, use 3% minimum
        stop_loss = entry_price * 0.97
        stop_source = f"Adjusted: 3% below entry (original {stop_distance_pct:.1f}% too tight)"
    elif stop_distance_pct > 20.0:
        # Stop too wide, use 15% maximum
        stop_loss = entry_price * 0.85
        stop_source = f"Adjusted: 15% below entry (original {stop_distance_pct:.1f}% too wide)"

    # Calculate stop distance
    stop_distance = entry_price - stop_loss

    # Risk amount in currency
    risk_amount = capital * risk_pct

    # Position size calculation
    if stop_distance > 0:
        shares = risk_amount / stop_distance
    else:
        shares = 0

    # Round down to whole shares
    shares = int(shares)

    # Ensure at least 1 share if calculation is valid
    if shares < 1 and risk_amount > 0 and stop_distance > 0:
        shares = 1

    # Calculate position value
    position_value = shares * entry_price

    # Check if position exceeds max allowed
    max_position_value = capital * max_position_pct
    if position_value > max_position_value:
        shares = int(max_position_value / entry_price)
        position_value = shares * entry_price

    # Recalculate actual risk amount
    actual_risk = shares * stop_distance

    return PositionSize(
        shares=shares,
        position_value=round(position_value, 2),
        stop_loss=round(stop_loss, 2),
        risk_amount=round(actual_risk, 2),
        risk_pct=risk_pct,
        entry_price=round(entry_price, 2),
        stop_source=stop_source
    )


def format_position_recommendation(pos: PositionSize, symbol: str) -> str:
    """
    Format position sizing as readable recommendation

    Args:
        pos: PositionSize result
        symbol: Stock symbol

    Returns:
        Formatted string recommendation
    """
    return (
        f"Position Sizing {symbol}:\n"
        f"  Achat: {pos.shares} actions @ {pos.entry_price:.2f}EUR\n"
        f"  Valeur: {pos.position_value:.2f}EUR\n"
        f"  Stop Loss: {pos.stop_loss:.2f}EUR ({pos.stop_source})\n"
        f"  Risque: {pos.risk_amount:.2f}EUR ({pos.risk_pct*100:.1f}% du capital)"
    )


class PositionSizer:
    """
    Position sizing calculator with portfolio tracking

    Usage:
        sizer = PositionSizer(total_capital=10000)

        # Calculate position
        position = sizer.calculate(
            df=df,
            entry_price=150.0,
            symbol='AAPL',
            market_cap=3000,  # 3 trillion
            rsi_peaks_indices=[10, 25, 40]  # From trendline detector
        )

        # Track position
        sizer.open_position('AAPL', position)

        # Check available capital
        available = sizer.get_available_capital()
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        max_position_pct: float = 0.25,
        portfolio_file: str = None
    ):
        """
        Initialize position sizer with portfolio tracking

        Args:
            total_capital: Total trading capital
            max_position_pct: Max single position as % of capital
            portfolio_file: Path to portfolio JSON file
        """
        self.max_position_pct = max_position_pct
        self.portfolio = PortfolioTracker(
            total_capital=total_capital,
            portfolio_file=portfolio_file
        )

    def calculate(
        self,
        df: pd.DataFrame,
        entry_price: float,
        symbol: str = "",
        market_cap: float = None,
        rsi_peaks_indices: List[int] = None,
        risk_pct_override: float = None
    ) -> PositionSize:
        """
        Calculate position size for a trade

        Args:
            df: DataFrame with OHLCV data
            entry_price: Intended entry price
            symbol: Stock symbol (for logging)
            market_cap: Market cap in billions for risk calculation
            rsi_peaks_indices: RSI peak indices from trendline detector
            risk_pct_override: Force specific risk % (optional)

        Returns:
            PositionSize with all calculated values
        """
        # Use available capital (after existing positions)
        available_capital = self.portfolio.get_available_capital()

        return calculate_position_size(
            df=df,
            entry_price=entry_price,
            capital=available_capital,
            market_cap=market_cap,
            rsi_peaks_indices=rsi_peaks_indices,
            max_position_pct=self.max_position_pct,
            risk_pct_override=risk_pct_override
        )

    def open_position(self, symbol: str, position: PositionSize) -> OpenPosition:
        """
        Record a new position from calculated sizing

        Args:
            symbol: Stock symbol
            position: PositionSize from calculate()

        Returns:
            Created OpenPosition
        """
        return self.portfolio.add_position(
            symbol=symbol,
            shares=position.shares,
            entry_price=position.entry_price,
            stop_loss=position.stop_loss
        )

    def close_position(self, symbol: str) -> Optional[OpenPosition]:
        """Close a position by symbol"""
        return self.portfolio.close_position(symbol)

    def get_available_capital(self) -> float:
        """Get capital available for new positions"""
        return self.portfolio.get_available_capital()

    def get_total_capital(self) -> float:
        """Get total capital"""
        return self.portfolio.total_capital

    def update_capital(self, new_capital: float):
        """Update total capital"""
        self.portfolio.update_total_capital(new_capital)

    def get_portfolio_summary(self) -> Dict:
        """Get full portfolio summary"""
        return self.portfolio.get_summary()

    def get_max_shares(self, price: float) -> int:
        """Get maximum shares based on available capital and max position size"""
        available = self.get_available_capital()
        max_value = min(
            available,
            self.portfolio.total_capital * self.max_position_pct
        )
        return int(max_value / price) if price > 0 else 0


# Default instance with 10,000EUR capital
position_sizer = PositionSizer(total_capital=10000)


# =============================================================================
# ASYNC POSITION SIZER WITH EUR/USD CONVERSION (IBKR)
# =============================================================================

class FXRateCache:
    """
    Cache for FX rates to avoid excessive IBKR requests.

    IBKR charges for FX quotes, and rates don't change every second.
    Caches rates for a configurable duration (default 5 minutes).
    """

    def __init__(self, cache_duration_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}  # pair -> (rate, timestamp)
        self._cache_duration = cache_duration_seconds
        self._lock = asyncio.Lock()

    async def get(self, pair: str) -> Optional[float]:
        """Get cached rate if valid"""
        async with self._lock:
            if pair in self._cache:
                rate, timestamp = self._cache[pair]
                if (datetime.now() - timestamp).total_seconds() < self._cache_duration:
                    return rate
            return None

    async def set(self, pair: str, rate: float):
        """Cache a rate"""
        async with self._lock:
            self._cache[pair] = (rate, datetime.now())

    def clear(self):
        """Clear all cached rates"""
        self._cache = {}


# Global FX cache
_fx_cache = FXRateCache()


class AsyncPositionSizer:
    """
    Async position sizer with EUR/USD conversion via IBKR.

    For European accounts trading US stocks, this properly converts capital
    to USD before calculating position sizes, avoiding ~10% sizing errors.

    Usage:
        from ib_insync import IB

        ib = IB()
        await ib.connectAsync('127.0.0.1', 7497, clientId=1)

        sizer = AsyncPositionSizer(
            total_capital=1500,  # EUR
            account_currency='EUR',
            ibkr_client=ib
        )

        position = await sizer.calculate_with_fx(
            df=df,
            entry_price=175.0,  # USD
            symbol='AAPL',
            market_cap=3000
        )

        print(f"Buy {position.shares} @ ${position.entry_price}")
        print(f"FX rate: {position.fx_rate} EUR/USD")
        print(f"Risk: {position.risk_in_account_currency:.2f} EUR")
    """

    def __init__(
        self,
        total_capital: float = 1500.0,
        account_currency: str = 'EUR',
        ibkr_client: Any = None,
        max_position_pct: float = 0.25,
        portfolio_file: str = None,
        fallback_fx_rate: float = 1.08  # Fallback if IBKR unavailable
    ):
        """
        Initialize async position sizer with FX support.

        Args:
            total_capital: Total capital in account currency
            account_currency: Account base currency ('EUR', 'GBP', 'CHF', etc.)
            ibkr_client: IB-insync client instance (or None for offline mode)
            max_position_pct: Max single position as % of capital
            portfolio_file: Path to portfolio JSON file
            fallback_fx_rate: FX rate to use if IBKR unavailable
        """
        self.account_currency = account_currency.upper()
        self.ibkr = ibkr_client
        self.fallback_fx_rate = fallback_fx_rate
        self.max_position_pct = max_position_pct

        # Standard portfolio tracker (in account currency)
        self.portfolio = PortfolioTracker(
            total_capital=total_capital,
            portfolio_file=portfolio_file or f'data/portfolio_{account_currency.lower()}.json'
        )

        logger.info(
            f"AsyncPositionSizer initialized: {total_capital} {account_currency}, "
            f"IBKR={'connected' if ibkr_client else 'offline'}"
        )

    async def get_fx_rate(self, base: str = 'EUR', quote: str = 'USD') -> float:
        """
        Get FX rate from IBKR with caching.

        Args:
            base: Base currency (e.g., 'EUR')
            quote: Quote currency (e.g., 'USD')

        Returns:
            Exchange rate (e.g., 1.08 for EUR/USD)
        """
        pair = f"{base}{quote}"

        # Check cache first
        cached_rate = await _fx_cache.get(pair)
        if cached_rate is not None:
            logger.debug(f"FX rate {pair} from cache: {cached_rate}")
            return cached_rate

        # Try IBKR if available
        if self.ibkr and self.ibkr.isConnected():
            try:
                # Import here to avoid issues if ib_insync not installed
                from ib_insync import Forex

                fx_contract = Forex(pair)

                # Request ticker
                ticker = await self.ibkr.reqTickersAsync(fx_contract)

                if ticker and len(ticker) > 0:
                    # Use midpoint of bid/ask
                    rate = ticker[0].midpoint()

                    if rate and rate > 0:
                        await _fx_cache.set(pair, rate)
                        logger.info(f"FX rate {pair} from IBKR: {rate:.4f}")
                        return rate

            except Exception as e:
                logger.warning(f"IBKR FX request failed for {pair}: {e}")

        # Fallback to default rate
        logger.warning(f"Using fallback FX rate for {pair}: {self.fallback_fx_rate}")
        return self.fallback_fx_rate

    def _detect_stock_currency(self, symbol: str) -> str:
        """
        Detect stock trading currency from symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'AIR.PA', 'VOW3.DE')

        Returns:
            Currency code ('USD', 'EUR', 'GBP', etc.)
        """
        symbol_upper = symbol.upper()

        # European exchanges
        if symbol_upper.endswith('.PA') or symbol_upper.endswith('.AS'):
            return 'EUR'  # Paris (Euronext), Amsterdam
        elif symbol_upper.endswith('.DE') or symbol_upper.endswith('.F'):
            return 'EUR'  # Frankfurt (Xetra), Frankfurt floor
        elif symbol_upper.endswith('.MI'):
            return 'EUR'  # Milan
        elif symbol_upper.endswith('.MC'):
            return 'EUR'  # Madrid
        elif symbol_upper.endswith('.BR'):
            return 'EUR'  # Brussels
        elif symbol_upper.endswith('.L'):
            return 'GBP'  # London
        elif symbol_upper.endswith('.SW'):
            return 'CHF'  # Swiss
        elif symbol_upper.endswith('.TO'):
            return 'CAD'  # Toronto
        elif symbol_upper.endswith('.AX'):
            return 'AUD'  # Australia
        elif symbol_upper.endswith('.HK'):
            return 'HKD'  # Hong Kong
        elif symbol_upper.endswith('.T'):
            return 'JPY'  # Tokyo

        # Default: US stocks
        return 'USD'

    async def calculate_with_fx(
        self,
        df: pd.DataFrame,
        entry_price: float,
        symbol: str = "",
        market_cap: float = None,
        rsi_peaks_indices: List[int] = None,
        risk_pct_override: float = None,
        stock_currency: str = None
    ) -> PositionSizeWithFX:
        """
        Calculate position size with FX conversion.

        Converts account capital to stock currency before calculating,
        ensuring accurate position sizing for cross-currency trades.

        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price in stock's currency
            symbol: Stock symbol
            market_cap: Market cap in billions for risk calculation
            rsi_peaks_indices: RSI peak indices from trendline detector
            risk_pct_override: Force specific risk % (optional)
            stock_currency: Override auto-detected stock currency

        Returns:
            PositionSizeWithFX with FX conversion details
        """
        # Detect or use provided stock currency
        if stock_currency is None:
            stock_currency = self._detect_stock_currency(symbol)

        # Get FX rate if needed
        fx_rate = 1.0
        if self.account_currency != stock_currency:
            fx_rate = await self.get_fx_rate(self.account_currency, stock_currency)

        # Available capital in account currency
        available_capital_account = self.portfolio.get_available_capital()

        # Convert to stock currency
        available_capital_stock = available_capital_account * fx_rate

        logger.debug(
            f"Capital conversion: {available_capital_account:.2f} {self.account_currency} "
            f"-> {available_capital_stock:.2f} {stock_currency} (rate: {fx_rate:.4f})"
        )

        # Determine risk percentage
        if risk_pct_override is not None:
            risk_pct = risk_pct_override
        else:
            risk_pct = get_risk_pct_by_market_cap(market_cap)

        # Find stop-loss from RSI trendline
        stop_loss, stop_source = find_stop_loss_from_rsi_trendline(
            df, rsi_peaks_indices, fallback_pct=0.05
        )

        # Validate stop-loss (same logic as sync version)
        if stop_loss <= 0 or stop_loss >= entry_price:
            stop_loss = entry_price * 0.95
            stop_source = "Fallback: 5% below entry (invalid RSI stop)"

        stop_distance_pct = (entry_price - stop_loss) / entry_price * 100
        if stop_distance_pct < 2.0:
            stop_loss = entry_price * 0.97
            stop_source = f"Adjusted: 3% below entry (original {stop_distance_pct:.1f}% too tight)"
        elif stop_distance_pct > 20.0:
            stop_loss = entry_price * 0.85
            stop_source = f"Adjusted: 15% below entry (original {stop_distance_pct:.1f}% too wide)"

        # Calculate in stock currency
        stop_distance = entry_price - stop_loss
        risk_amount_stock = available_capital_stock * risk_pct

        # Position size in stock currency
        if stop_distance > 0:
            shares = risk_amount_stock / stop_distance
        else:
            shares = 0

        shares = int(shares)

        if shares < 1 and risk_amount_stock > 0 and stop_distance > 0:
            shares = 1

        # Position value in stock currency
        position_value = shares * entry_price

        # Check max position
        max_position_value = available_capital_stock * self.max_position_pct
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price

        # Actual risk in stock currency
        actual_risk_stock = shares * stop_distance

        # Convert risk back to account currency
        risk_in_account = actual_risk_stock / fx_rate if fx_rate > 0 else actual_risk_stock

        logger.info(
            f"Position sizing {symbol}: {shares} shares @ {entry_price:.2f} {stock_currency}, "
            f"SL={stop_loss:.2f}, Risk={risk_in_account:.2f} {self.account_currency} "
            f"(FX: {fx_rate:.4f})"
        )

        return PositionSizeWithFX(
            shares=shares,
            position_value=round(position_value, 2),
            stop_loss=round(stop_loss, 2),
            risk_amount=round(actual_risk_stock, 2),
            risk_pct=risk_pct,
            entry_price=round(entry_price, 2),
            stop_source=stop_source,
            fx_rate=round(fx_rate, 4),
            account_currency=self.account_currency,
            stock_currency=stock_currency,
            capital_in_stock_currency=round(available_capital_stock, 2),
            risk_in_account_currency=round(risk_in_account, 2)
        )

    async def open_position(
        self,
        symbol: str,
        position: PositionSizeWithFX
    ) -> OpenPosition:
        """
        Record a new position from calculated sizing.

        Stores in account currency for portfolio tracking.

        Args:
            symbol: Stock symbol
            position: PositionSizeWithFX from calculate_with_fx()

        Returns:
            Created OpenPosition
        """
        # Convert position value to account currency for portfolio tracking
        position_value_account = position.position_value / position.fx_rate if position.fx_rate > 0 else position.position_value

        return self.portfolio.add_position(
            symbol=symbol,
            shares=position.shares,
            entry_price=position.entry_price,
            stop_loss=position.stop_loss
        )

    def close_position(self, symbol: str) -> Optional[OpenPosition]:
        """Close a position by symbol"""
        return self.portfolio.close_position(symbol)

    def get_available_capital(self) -> float:
        """Get capital available in account currency"""
        return self.portfolio.get_available_capital()

    async def get_available_capital_in_usd(self) -> float:
        """Get capital available converted to USD"""
        capital_account = self.portfolio.get_available_capital()
        if self.account_currency == 'USD':
            return capital_account
        fx_rate = await self.get_fx_rate(self.account_currency, 'USD')
        return capital_account * fx_rate

    def get_total_capital(self) -> float:
        """Get total capital in account currency"""
        return self.portfolio.total_capital

    def update_capital(self, new_capital: float):
        """Update total capital in account currency"""
        self.portfolio.update_total_capital(new_capital)

    def get_portfolio_summary(self) -> Dict:
        """Get full portfolio summary"""
        return self.portfolio.get_summary()


# =============================================================================
# HELPER FUNCTIONS FOR SYNC/ASYNC COMPATIBILITY
# =============================================================================

def get_fx_rate_sync(
    base: str = 'EUR',
    quote: str = 'USD',
    fallback: float = 1.08
) -> float:
    """
    Synchronous fallback for FX rate (uses hardcoded approximate rates).

    For accurate rates, use AsyncPositionSizer with IBKR connection.

    Args:
        base: Base currency
        quote: Quote currency
        fallback: Default fallback rate

    Returns:
        Approximate exchange rate
    """
    # Approximate rates (updated periodically)
    rates = {
        'EURUSD': 1.08,
        'GBPUSD': 1.27,
        'USDCHF': 0.88,
        'USDJPY': 150.0,
        'USDCAD': 1.36,
        'AUDUSD': 0.66,
        'EURGBP': 0.85,
        'EURCHF': 0.95
    }

    pair = f"{base.upper()}{quote.upper()}"
    inverse = f"{quote.upper()}{base.upper()}"

    if pair in rates:
        return rates[pair]
    elif inverse in rates:
        return 1.0 / rates[inverse]
    else:
        logger.warning(f"Unknown FX pair {pair}, using fallback {fallback}")
        return fallback


def convert_capital(
    amount: float,
    from_currency: str,
    to_currency: str,
    fx_rate: float = None
) -> float:
    """
    Convert capital between currencies.

    Args:
        amount: Amount to convert
        from_currency: Source currency
        to_currency: Target currency
        fx_rate: Optional rate to use (fetches sync if None)

    Returns:
        Converted amount
    """
    if from_currency.upper() == to_currency.upper():
        return amount

    if fx_rate is None:
        fx_rate = get_fx_rate_sync(from_currency, to_currency)

    return amount * fx_rate
