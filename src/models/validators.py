"""
Pydantic validators and models for TradingBot

This module provides data validation for:
- Screening requests and parameters
- Alert results
- Ticker information
- Configuration validation
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class TimeframeEnum(str, Enum):
    """Valid timeframe values"""
    WEEKLY = "weekly"
    DAILY = "daily"


class RecommendationEnum(str, Enum):
    """Valid recommendation values"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    OBSERVE = "OBSERVE"


class MarketEnum(str, Enum):
    """Valid market values"""
    NASDAQ = "NASDAQ"
    SP500 = "SP500"
    EUROPE = "EUROPE"
    ASIA_ADR = "ASIA_ADR"
    CUSTOM = "CUSTOM"


def validate_symbol(symbol: str) -> str:
    """
    Validate and clean a stock symbol

    Args:
        symbol: Raw symbol string

    Returns:
        Cleaned and uppercased symbol

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol:
        raise ValueError("Symbol cannot be empty")

    cleaned = symbol.strip().upper()

    # Basic validation: only alphanumeric, dots, and hyphens
    if not all(c.isalnum() or c in '.-' for c in cleaned):
        raise ValueError(f"Invalid symbol format: {symbol}")

    if len(cleaned) > 10:
        raise ValueError(f"Symbol too long: {symbol}")

    return cleaned


def validate_timeframe(timeframe: str) -> str:
    """
    Validate timeframe value

    Args:
        timeframe: Timeframe string

    Returns:
        Validated timeframe

    Raises:
        ValueError: If timeframe is invalid
    """
    valid = ['weekly', 'daily', '1wk', '1d']
    if timeframe.lower() not in valid:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid}")

    # Normalize to standard format
    if timeframe.lower() == '1wk':
        return 'weekly'
    if timeframe.lower() == '1d':
        return 'daily'

    return timeframe.lower()


class TickerInfo(BaseModel):
    """Model for ticker information"""
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None

    @validator('symbol', pre=True)
    def validate_ticker_symbol(cls, v):
        return validate_symbol(v)

    class Config:
        extra = 'ignore'


class ScreeningRequest(BaseModel):
    """Model for screening request parameters"""
    symbol: str
    lookback_days: int = Field(default=365, ge=30, le=730)
    timeframe: str = Field(default='weekly')
    include_rsi: bool = Field(default=True)

    @validator('symbol', pre=True)
    def validate_request_symbol(cls, v):
        return validate_symbol(v)

    @validator('timeframe')
    def validate_request_timeframe(cls, v):
        return validate_timeframe(v)

    class Config:
        extra = 'forbid'


class ScreeningConfig(BaseModel):
    """Model for screening configuration"""
    ema_periods: List[int] = Field(default=[24, 38, 62])
    zone_tolerance: float = Field(default=8.0, ge=1.0, le=20.0)
    min_market_cap: float = Field(default=100.0, ge=0)
    min_daily_volume: float = Field(default=750000.0, ge=0)
    max_stocks: int = Field(default=700, ge=1, le=2000)

    @validator('ema_periods')
    def validate_ema_periods(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 EMA periods required")
        if not all(isinstance(p, int) and p > 0 for p in v):
            raise ValueError("EMA periods must be positive integers")
        return sorted(v)

    class Config:
        extra = 'ignore'


class AlertResult(BaseModel):
    """Model for screening alert result"""
    symbol: str
    company_name: Optional[str] = None
    timeframe: str
    current_price: float = Field(ge=0)
    support_level: float = Field(ge=0)
    distance_to_support_pct: float
    ema_24: Optional[float] = None
    ema_38: Optional[float] = None
    ema_62: Optional[float] = None
    ema_alignment: Optional[str] = None
    recommendation: str
    crossover_info: Optional[Dict[str, Any]] = None
    rsi_breakout: Optional[bool] = None
    rsi_trendline: Optional[bool] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=100)
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('symbol', pre=True)
    def validate_alert_symbol(cls, v):
        return validate_symbol(v)

    @validator('timeframe')
    def validate_alert_timeframe(cls, v):
        return validate_timeframe(v)

    @validator('recommendation')
    def validate_recommendation(cls, v):
        valid = ['STRONG_BUY', 'BUY', 'WATCH', 'OBSERVE']
        if v.upper() not in valid:
            raise ValueError(f"Invalid recommendation: {v}. Must be one of {valid}")
        return v.upper()

    @root_validator
    def validate_price_levels(cls, values):
        """Ensure support level is below or at current price"""
        current = values.get('current_price', 0)
        support = values.get('support_level', 0)

        if support > current * 1.1:  # Allow 10% tolerance
            raise ValueError(f"Support level ({support}) should not be significantly above current price ({current})")

        return values

    class Config:
        extra = 'ignore'


class TickerListData(BaseModel):
    """Model for ticker list JSON file"""
    updated: str
    source: str
    market: str
    count: int = Field(ge=0)
    tickers: List[TickerInfo]

    @validator('updated')
    def validate_updated_date(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f"Invalid date format: {v}")
        return v

    @validator('count')
    def validate_count_matches(cls, v, values):
        tickers = values.get('tickers', [])
        if len(tickers) != v:
            # Warning but don't fail - just update the count
            return len(tickers)
        return v

    class Config:
        extra = 'ignore'


class NotificationConfig(BaseModel):
    """Model for notification configuration"""
    telegram_enabled: bool = True
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    email_enabled: bool = False
    email_from: Optional[str] = None
    email_to: Optional[str] = None
    smtp_server: Optional[str] = None
    smtp_port: int = Field(default=587, ge=1, le=65535)

    @root_validator
    def validate_notification_config(cls, values):
        """Ensure required fields are present when channel is enabled"""
        if values.get('telegram_enabled'):
            if not values.get('telegram_token') or not values.get('telegram_chat_id'):
                # Just disable if not configured, don't raise error
                values['telegram_enabled'] = False

        if values.get('email_enabled'):
            required = ['email_from', 'email_to', 'smtp_server']
            if not all(values.get(f) for f in required):
                values['email_enabled'] = False

        return values

    class Config:
        extra = 'ignore'


# Utility functions for validation
def validate_symbols_list(symbols: List[str]) -> List[str]:
    """
    Validate a list of symbols

    Args:
        symbols: List of symbol strings

    Returns:
        List of cleaned and validated symbols

    Raises:
        ValueError: If any symbol is invalid
    """
    if not symbols:
        return []

    validated = []
    for sym in symbols:
        try:
            validated.append(validate_symbol(sym))
        except ValueError as e:
            # Skip invalid symbols with warning
            pass

    return list(set(validated))  # Remove duplicates


def validate_screening_parameters(
    symbol: str = None,
    symbols: List[str] = None,
    timeframe: str = 'weekly',
    lookback_days: int = 365
) -> Dict[str, Any]:
    """
    Validate screening parameters and return cleaned values

    Args:
        symbol: Single symbol (optional)
        symbols: List of symbols (optional)
        timeframe: Timeframe for analysis
        lookback_days: Number of days to look back

    Returns:
        Dictionary with validated parameters

    Raises:
        ValueError: If parameters are invalid
    """
    result = {
        'symbols': [],
        'timeframe': validate_timeframe(timeframe),
        'lookback_days': max(30, min(730, lookback_days))
    }

    if symbol:
        result['symbols'].append(validate_symbol(symbol))

    if symbols:
        result['symbols'].extend(validate_symbols_list(symbols))

    # Remove duplicates
    result['symbols'] = list(set(result['symbols']))

    if not result['symbols']:
        raise ValueError("At least one valid symbol is required")

    return result
