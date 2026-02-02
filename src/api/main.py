"""
TradingBot V3 REST API - Main Application

FastAPI-based REST API for market screening, alerts, and portfolio management.

Run with: uvicorn src.api.main:app --reload
"""
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.api.websocket import (
    connection_manager,
    handle_websocket_message,
    AlertMessage,
    ProgressMessage
)

from config.settings import CAPITAL
from src.screening.screener import market_screener
from src.data.market_data import market_data_fetcher
from src.database.db_manager import db_manager
from src.utils.watchlist_manager import watchlist_manager
from src.utils.trade_tracker import trade_tracker
from src.utils.sector_heatmap import sector_heatmap_builder
from src.utils.economic_calendar import economic_calendar, EventType


# ==================== Pydantic Models ====================

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class StockScreenRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=50)
    save_to_db: bool = False


class StockScreenResponse(BaseModel):
    symbol: str
    company_name: str
    timeframe: str
    current_price: float
    support_level: float
    distance_to_support_pct: float
    recommendation: str
    confidence_score: Optional[float] = None
    has_rsi_breakout: bool = False
    has_earnings_soon: bool = False
    earnings_warning: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    symbol: str
    timeframe: str
    current_price: float
    support_level: float
    recommendation: str
    confidence_score: Optional[float] = None
    created_at: str


class WatchlistCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    symbols: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class WatchlistResponse(BaseModel):
    id: str
    name: str
    symbols: List[str]
    created_at: str


class TradeOpen(BaseModel):
    symbol: str
    entry_price: float = Field(..., gt=0)
    shares: int = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    target_price: Optional[float] = None
    notes: Optional[str] = None


class TradeClose(BaseModel):
    exit_price: float = Field(..., gt=0)
    exit_reason: Optional[str] = None


class TradeResponse(BaseModel):
    trade_id: str
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    status: str
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


class PerformanceResponse(BaseModel):
    total_trades: int
    open_trades: int
    closed_trades: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: float


class SectorPerformanceResponse(BaseModel):
    name: str
    etf: str
    perf_1d: float
    perf_1w: float
    perf_1m: float
    perf_ytd: float


class MarketBreadthResponse(BaseModel):
    advancing: int
    declining: int
    total: int
    breadth_ratio: float
    avg_performance: float


class EconomicEventResponse(BaseModel):
    event_id: str
    event_type: str
    title: str
    date: str
    impact: str
    symbol: Optional[str] = None


class PortfolioSummary(BaseModel):
    total_capital: float
    available_capital: float
    invested_capital: float
    open_positions: int
    unrealized_pnl: float


# ==================== API Key Security ====================

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key() -> Optional[str]:
    """Get API key from environment"""
    return os.getenv("TRADINGBOT_API_KEY")


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Verify API key if configured"""
    expected_key = get_api_key()

    # If no API key is configured, allow all requests (development mode)
    if expected_key is None:
        return "development"

    if api_key is None or api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return api_key


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting TradingBot API...")
    yield
    # Shutdown
    print("Shutting down TradingBot API...")


# ==================== App Configuration ====================

app = FastAPI(
    title="TradingBot V3 API",
    description="REST API for market screening, alerts, and portfolio management",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health & Info ====================

@app.get("/", tags=["Info"])
async def root():
    """API root endpoint"""
    return {
        "name": "TradingBot V3 API",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        timestamp=datetime.now().isoformat()
    )


# ==================== Screening Endpoints ====================

@app.post("/api/v1/screen", response_model=List[StockScreenResponse], tags=["Screening"])
async def screen_stocks(
    request: StockScreenRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Screen multiple stocks for trading signals.

    Returns alerts for stocks with valid signals (STRONG_BUY, BUY, WATCH).
    """
    try:
        alerts = market_screener.screen_multiple_stocks(
            [{'symbol': s, 'name': s} for s in request.symbols]
        )

        if request.save_to_db and alerts:
            background_tasks.add_task(db_manager.save_alerts, alerts)

        return [
            StockScreenResponse(
                symbol=a['symbol'],
                company_name=a.get('company_name', a['symbol']),
                timeframe=a['timeframe'],
                current_price=a['current_price'],
                support_level=a['support_level'],
                distance_to_support_pct=a['distance_to_support_pct'],
                recommendation=a['recommendation'],
                confidence_score=a.get('confidence_score'),
                has_rsi_breakout=a.get('has_rsi_breakout', False),
                has_earnings_soon=a.get('has_earnings_soon', False),
                earnings_warning=a.get('earnings_warning')
            )
            for a in alerts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/screen/{symbol}", response_model=Optional[StockScreenResponse], tags=["Screening"])
async def screen_single_stock(
    symbol: str,
    api_key: str = Depends(verify_api_key)
):
    """Screen a single stock for trading signals."""
    try:
        alert = market_screener.screen_single_stock(symbol.upper())

        if alert is None:
            return None

        return StockScreenResponse(
            symbol=alert['symbol'],
            company_name=alert.get('company_name', alert['symbol']),
            timeframe=alert['timeframe'],
            current_price=alert['current_price'],
            support_level=alert['support_level'],
            distance_to_support_pct=alert['distance_to_support_pct'],
            recommendation=alert['recommendation'],
            confidence_score=alert.get('confidence_score'),
            has_rsi_breakout=alert.get('has_rsi_breakout', False),
            has_earnings_soon=alert.get('has_earnings_soon', False),
            earnings_warning=alert.get('earnings_warning')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Alerts Endpoints ====================

@app.get("/api/v1/alerts", response_model=List[AlertResponse], tags=["Alerts"])
async def get_recent_alerts(
    days: int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=50, ge=1, le=200),
    api_key: str = Depends(verify_api_key)
):
    """Get recent alerts from the database."""
    try:
        alerts = db_manager.get_recent_alerts(days=days)[:limit]

        return [
            AlertResponse(
                id=a.id,
                symbol=a.symbol,
                timeframe=a.timeframe,
                current_price=a.current_price,
                support_level=a.support_level,
                recommendation=a.recommendation,
                confidence_score=getattr(a, 'confidence_score', None),
                created_at=a.created_at.isoformat() if a.created_at else ""
            )
            for a in alerts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts/{symbol}", response_model=List[AlertResponse], tags=["Alerts"])
async def get_symbol_alerts(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365),
    api_key: str = Depends(verify_api_key)
):
    """Get alerts for a specific symbol."""
    try:
        alerts = db_manager.get_alerts_by_symbol(symbol.upper(), days=days)

        return [
            AlertResponse(
                id=a.id,
                symbol=a.symbol,
                timeframe=a.timeframe,
                current_price=a.current_price,
                support_level=a.support_level,
                recommendation=a.recommendation,
                confidence_score=getattr(a, 'confidence_score', None),
                created_at=a.created_at.isoformat() if a.created_at else ""
            )
            for a in alerts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Watchlist Endpoints ====================

@app.get("/api/v1/watchlists", response_model=List[WatchlistResponse], tags=["Watchlists"])
async def get_watchlists(api_key: str = Depends(verify_api_key)):
    """Get all watchlists."""
    try:
        watchlists = watchlist_manager.get_all_watchlists()

        return [
            WatchlistResponse(
                id=w.id,
                name=w.name,
                symbols=w.symbols,
                created_at=w.created_at.isoformat() if w.created_at else ""
            )
            for w in watchlists
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/watchlists", response_model=WatchlistResponse, tags=["Watchlists"])
async def create_watchlist(
    watchlist: WatchlistCreate,
    api_key: str = Depends(verify_api_key)
):
    """Create a new watchlist."""
    try:
        wl = watchlist_manager.create_watchlist(
            name=watchlist.name,
            symbols=watchlist.symbols,
            description=watchlist.description
        )

        return WatchlistResponse(
            id=wl.id,
            name=wl.name,
            symbols=wl.symbols,
            created_at=wl.created_at.isoformat() if wl.created_at else ""
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/watchlists/{watchlist_id}", tags=["Watchlists"])
async def delete_watchlist(
    watchlist_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete a watchlist."""
    try:
        success = watchlist_manager.delete_watchlist(watchlist_id)
        if not success:
            raise HTTPException(status_code=404, detail="Watchlist not found")
        return {"status": "deleted", "id": watchlist_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/watchlists/{watchlist_id}/symbols/{symbol}", tags=["Watchlists"])
async def add_symbol_to_watchlist(
    watchlist_id: str,
    symbol: str,
    api_key: str = Depends(verify_api_key)
):
    """Add a symbol to a watchlist."""
    try:
        success = watchlist_manager.add_symbol(watchlist_id, symbol.upper())
        if not success:
            raise HTTPException(status_code=404, detail="Watchlist not found")
        return {"status": "added", "symbol": symbol.upper()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Portfolio & Trades ====================

@app.get("/api/v1/portfolio/summary", response_model=PortfolioSummary, tags=["Portfolio"])
async def get_portfolio_summary(api_key: str = Depends(verify_api_key)):
    """Get portfolio summary."""
    try:
        summary = market_screener.get_portfolio_summary()
        return PortfolioSummary(
            total_capital=summary.get('total_capital', CAPITAL),
            available_capital=summary.get('available_capital', CAPITAL),
            invested_capital=summary.get('invested_capital', 0),
            open_positions=summary.get('open_positions', 0),
            unrealized_pnl=summary.get('unrealized_pnl', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trades", response_model=List[TradeResponse], tags=["Trades"])
async def get_trades(
    status: Optional[str] = Query(None, pattern="^(open|closed)$"),
    api_key: str = Depends(verify_api_key)
):
    """Get all trades, optionally filtered by status."""
    try:
        if status == "open":
            trades = trade_tracker.get_open_trades()
        elif status == "closed":
            trades = trade_tracker.get_closed_trades()
        else:
            trades = trade_tracker.get_all_trades()

        return [
            TradeResponse(
                trade_id=t.trade_id,
                symbol=t.symbol,
                entry_date=t.entry_date,
                entry_price=t.entry_price,
                shares=t.shares,
                status=t.status,
                pnl=t.pnl if t.status == 'closed' else None,
                pnl_pct=t.pnl_pct if t.status == 'closed' else None
            )
            for t in trades
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/trades", response_model=TradeResponse, tags=["Trades"])
async def open_trade(
    trade: TradeOpen,
    api_key: str = Depends(verify_api_key)
):
    """Open a new trade."""
    try:
        t = trade_tracker.open_trade(
            symbol=trade.symbol.upper(),
            entry_price=trade.entry_price,
            shares=trade.shares,
            stop_loss=trade.stop_loss,
            target_price=trade.target_price,
            notes=trade.notes
        )

        return TradeResponse(
            trade_id=t.trade_id,
            symbol=t.symbol,
            entry_date=t.entry_date,
            entry_price=t.entry_price,
            shares=t.shares,
            status=t.status,
            pnl=None,
            pnl_pct=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/trades/{trade_id}/close", response_model=TradeResponse, tags=["Trades"])
async def close_trade(
    trade_id: str,
    close_data: TradeClose,
    api_key: str = Depends(verify_api_key)
):
    """Close an existing trade."""
    try:
        t = trade_tracker.close_trade(
            trade_id=trade_id,
            exit_price=close_data.exit_price,
            exit_reason=close_data.exit_reason
        )

        if t is None:
            raise HTTPException(status_code=404, detail="Trade not found")

        return TradeResponse(
            trade_id=t.trade_id,
            symbol=t.symbol,
            entry_date=t.entry_date,
            entry_price=t.entry_price,
            shares=t.shares,
            status=t.status,
            pnl=t.pnl,
            pnl_pct=t.pnl_pct
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trades/performance", response_model=PerformanceResponse, tags=["Trades"])
async def get_performance(api_key: str = Depends(verify_api_key)):
    """Get trading performance metrics."""
    try:
        summary = trade_tracker.get_performance_summary()

        return PerformanceResponse(
            total_trades=summary['total_trades'],
            open_trades=summary['open_trades'],
            closed_trades=summary['closed_trades'],
            total_pnl=summary['total_pnl'],
            win_rate=summary['win_rate'],
            avg_win=summary['avg_win'],
            avg_loss=summary['avg_loss'],
            profit_factor=summary.get('profit_factor'),
            sharpe_ratio=summary.get('sharpe_ratio'),
            max_drawdown=summary.get('max_drawdown', 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Market Data ====================

@app.get("/api/v1/sectors", response_model=List[SectorPerformanceResponse], tags=["Market"])
async def get_sector_performance(api_key: str = Depends(verify_api_key)):
    """Get sector performance data."""
    try:
        sectors = sector_heatmap_builder.get_sector_performance(use_cache=True)

        return [
            SectorPerformanceResponse(
                name=s.name,
                etf=s.etf,
                perf_1d=s.perf_1d,
                perf_1w=s.perf_1w,
                perf_1m=s.perf_1m,
                perf_ytd=s.perf_ytd
            )
            for s in sectors
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/breadth", response_model=MarketBreadthResponse, tags=["Market"])
async def get_market_breadth(api_key: str = Depends(verify_api_key)):
    """Get market breadth indicators."""
    try:
        breadth = sector_heatmap_builder.get_market_breadth()

        return MarketBreadthResponse(
            advancing=breadth['advancing'],
            declining=breadth['declining'],
            total=breadth['total'],
            breadth_ratio=breadth['breadth_ratio'],
            avg_performance=breadth['avg_performance']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/rotation", tags=["Market"])
async def get_sector_rotation(api_key: str = Depends(verify_api_key)):
    """Get sector rotation signal."""
    try:
        rotation = sector_heatmap_builder.get_sector_rotation_signal()
        return rotation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Economic Calendar ====================

@app.get("/api/v1/calendar/events", response_model=List[EconomicEventResponse], tags=["Calendar"])
async def get_economic_events(
    days_ahead: int = Query(default=30, ge=1, le=90),
    event_type: Optional[str] = Query(None, pattern="^(fomc|cpi|jobs|earnings)$"),
    api_key: str = Depends(verify_api_key)
):
    """Get upcoming economic events."""
    try:
        event_types = None
        if event_type:
            event_types = [EventType(event_type)]

        events = economic_calendar.get_upcoming_events(
            days_ahead=days_ahead,
            event_types=event_types,
            include_earnings=(event_type == 'earnings' or event_type is None)
        )

        return [
            EconomicEventResponse(
                event_id=e.event_id,
                event_type=e.event_type.value,
                title=e.title,
                date=e.date.isoformat(),
                impact=e.impact.value,
                symbol=e.symbol
            )
            for e in events
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/calendar/earnings/{symbol}", tags=["Calendar"])
async def get_earnings_date(
    symbol: str,
    api_key: str = Depends(verify_api_key)
):
    """Get earnings date for a specific symbol."""
    try:
        earnings = economic_calendar.get_earnings_date(symbol.upper())

        if earnings is None:
            return {"symbol": symbol.upper(), "earnings_date": None, "days_until": None}

        return {
            "symbol": symbol.upper(),
            "earnings_date": earnings['earnings_date'].isoformat() if earnings.get('earnings_date') else None,
            "days_until": earnings.get('days_until'),
            "warning": economic_calendar.get_earnings_warning(symbol.upper())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Stock Data ====================

@app.get("/api/v1/stock/{symbol}/price", tags=["Stock Data"])
async def get_stock_price(
    symbol: str,
    api_key: str = Depends(verify_api_key)
):
    """Get current stock price and basic info."""
    try:
        df = market_data_fetcher.get_historical_data(symbol.upper(), period='5d', interval='1d')

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        current_price = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

        return {
            "symbol": symbol.upper(),
            "price": current_price,
            "change": change,
            "change_pct": change_pct,
            "volume": int(df['Volume'].iloc[-1]),
            "high": float(df['High'].iloc[-1]),
            "low": float(df['Low'].iloc[-1]),
            "date": df.index[-1].strftime('%Y-%m-%d')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stock/{symbol}/history", tags=["Stock Data"])
async def get_stock_history(
    symbol: str,
    period: str = Query(default="1y", pattern="^(1mo|3mo|6mo|1y|2y|5y)$"),
    interval: str = Query(default="1d", pattern="^(1d|1wk)$"),
    api_key: str = Depends(verify_api_key)
):
    """Get historical stock data."""
    try:
        df = market_data_fetcher.get_historical_data(
            symbol.upper(),
            period=period,
            interval=interval
        )

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")

        # Convert to JSON-serializable format
        data = []
        for idx, row in df.iterrows():
            data.append({
                "date": idx.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })

        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "count": len(data),
            "data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WebSocket Endpoints ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Connect and receive:
    - New alerts as they are detected
    - Screening progress updates
    - Price updates for watched symbols

    Send commands:
    - {"action": "subscribe", "topics": ["alerts", "progress", "prices"]}
    - {"action": "unsubscribe", "topics": ["progress"]}
    - {"action": "watch", "symbol": "AAPL"}
    - {"action": "unwatch", "symbol": "AAPL"}
    - {"action": "ping"}
    - {"action": "status"}
    """
    client_id = await connection_manager.connect(websocket)

    try:
        while True:
            # Receive and process messages
            data = await websocket.receive_json()
            await handle_websocket_message(client_id, data, connection_manager)

    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        connection_manager.disconnect(client_id)
        print(f"WebSocket error for {client_id}: {e}")


@app.get("/ws/status", tags=["WebSocket"])
async def websocket_status(api_key: str = Depends(verify_api_key)):
    """Get WebSocket connection status."""
    return {
        "active_connections": connection_manager.get_connection_count(),
        "subscriptions": {
            topic: len(clients)
            for topic, clients in connection_manager.subscriptions.items()
        }
    }


# Helper function to broadcast alerts from screening
async def broadcast_screening_alert(alert: dict):
    """Broadcast a new alert via WebSocket."""
    message = AlertMessage(
        symbol=alert.get('symbol', ''),
        recommendation=alert.get('recommendation', ''),
        current_price=alert.get('current_price', 0),
        support_level=alert.get('support_level', 0),
        confidence_score=alert.get('confidence_score')
    )
    await connection_manager.broadcast_alert(message)


async def broadcast_screening_progress(total: int, completed: int, current_symbol: str):
    """Broadcast screening progress via WebSocket."""
    message = ProgressMessage(
        total=total,
        completed=completed,
        current_symbol=current_symbol
    )
    await connection_manager.broadcast_progress(message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
