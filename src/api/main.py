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

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# Portfolio history tracking
try:
    from src.utils.portfolio_tracker import record_snapshot
    PORTFOLIO_TRACKER = True
except ImportError:
    PORTFOLIO_TRACKER = False
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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    score_at_entry: Optional[float] = None
    pillar_technical: Optional[float] = None
    pillar_fundamental: Optional[float] = None
    pillar_sentiment: Optional[float] = None
    pillar_news: Optional[float] = None
    reasoning: Optional[str] = None
    position_value: Optional[float] = None
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


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

# V8.1 signals endpoint
try:
    from src.api.signals_endpoint import register_signals_endpoint
    register_signals_endpoint(app)
except Exception:
    pass

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Static Files (Dashboard SPA) ====================

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dashboard", "dist")

if os.path.isdir(DASHBOARD_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DASHBOARD_DIR, "assets")), name="static-assets")


# ==================== Health & Info ====================

@app.get("/api/info", tags=["Info"])
async def api_info():
    """API info endpoint"""
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

@app.get("/api/v1/portfolio/summary", tags=["Portfolio"])
async def get_portfolio_summary(agent: str = "llm", api_key: str = Depends(verify_api_key)):
    """Get portfolio summary (reads from portfolio.json for paper trading)."""
    import json
    from pathlib import Path
    
    try:
        data_dir = "data" if agent == "llm" else "data-nollm"
        suffix = "llm" if agent == "llm" else "nollm"
        ms_path = Path(f"{data_dir}/multi_strategy_{suffix}.json")

        # Try multi_strategy file first
        if ms_path.exists():
            with open(ms_path, "r") as f:
                ms_data = json.load(f)
            total_cash = 0
            total_invested = 0
            total_pnl = 0
            total_positions = 0
            for sid, sdata in ms_data.get("strategies", {}).items():
                cash = sdata.get("cash", 15000)
                total_cash += cash
                for p in sdata.get("positions", []):
                    cur = p.get("current_price", p.get("entry_price", 0))
                    entry = p.get("entry_price", 0)
                    shares = p.get("shares", 0)
                    total_invested += cur * shares
                    total_pnl += (cur - entry) * shares
                    total_positions += 1
            # Build per-strategy breakdown
            strategies_breakdown = {}
            for sid2, sdata2 in ms_data.get("strategies", {}).items():
                s_cash = sdata2.get("cash", 15000)
                s_invested = 0
                s_pnl = 0
                s_positions = 0
                for p2 in sdata2.get("positions", []):
                    cur2 = p2.get("current_price", p2.get("entry_price", 0))
                    entry2 = p2.get("entry_price", 0)
                    shares2 = p2.get("shares", 0)
                    s_invested += cur2 * shares2
                    s_pnl += (cur2 - entry2) * shares2
                    s_positions += 1
                s_equity = s_cash + s_invested
                initial = sdata2.get("initial_capital", 15000)
                s_return_pct = round(((s_equity - initial) / initial) * 100, 2) if initial > 0 else 0
                strategies_breakdown[sid2] = {
                    "equity": round(s_equity, 2),
                    "cash": round(s_cash, 2),
                    "invested": round(s_invested, 2),
                    "positions": s_positions,
                    "pnl": round(s_pnl, 2),
                    "return_pct": s_return_pct
                }
            return {
                "total_capital": round(total_cash + total_invested, 2),
                "available_capital": round(total_cash, 2),
                "invested_capital": round(total_invested, 2),
                "open_positions": total_positions,
                "unrealized_pnl": round(total_pnl, 2),
                "strategies": strategies_breakdown
            }

        # Fallback to portfolio.json
        portfolio_path = Path(f"{data_dir}/portfolio.json")
        if portfolio_path.exists():
            with open(portfolio_path, 'r') as f:
                data = json.load(f)
            
            total_capital = data.get('total_capital', CAPITAL)
            positions = data.get('positions', [])
            invested = sum(p.get('position_value', p.get('shares', 0) * p.get('entry_price', 0)) for p in positions)
            unrealized_pnl = sum(p.get('pnl_amount', 0) for p in positions)
            
            # Record snapshot for tracking history
            if PORTFOLIO_TRACKER:
                try:
                    record_snapshot()
                except: pass
            return PortfolioSummary(
                total_capital=total_capital,
                available_capital=data.get('cash', total_capital - invested),
                invested_capital=invested,
                open_positions=len(positions),
                unrealized_pnl=unrealized_pnl
            )
        else:
            # Fallback to market_screener
            summary = market_screener.get_portfolio_summary()
            # Record snapshot for tracking history
            if PORTFOLIO_TRACKER:
                try:
                    record_snapshot()
                except: pass
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
    agent: str = "llm",
    api_key: str = Depends(verify_api_key)
):
    """Get all trades, optionally filtered by status (reads from portfolio.json for paper trading)."""
    import json
    from pathlib import Path
    from datetime import datetime
    
    try:
        trades = []
        
        data_dir = "data" if agent == "llm" else "data-nollm"
        suffix = "llm" if agent == "llm" else "nollm"

        # Read open positions from multi_strategy file
        if status != "closed":
            ms_path = Path(f"{data_dir}/multi_strategy_{suffix}.json")
            if ms_path.exists():
                with open(ms_path, "r") as f:
                    ms_data = json.load(f)
                for sid, sdata in ms_data.get("strategies", {}).items():
                    for i, p in enumerate(sdata.get("positions", [])):
                        entry = p.get("entry_price", 0)
                        cur = p.get("current_price", entry)
                        shares = p.get("shares", 0)
                        trades.append(TradeResponse(
                            trade_id=f"{agent}_{sid}_{p.get('symbol', '')}_{i}",
                            symbol=p.get("symbol", ""),
                            entry_date=p.get("entry_date", datetime.now().isoformat()),
                            entry_price=entry,
                            shares=shares,
                            status="open",
                            pnl=round((cur - entry) * shares, 2),
                            pnl_pct=round(((cur - entry) / entry) * 100, 2) if entry > 0 else 0,
                            stop_loss=p.get("stop_loss"),
                            take_profit=p.get("take_profit"),
                            score_at_entry=p.get("score_at_entry"),
                            reasoning=p.get("reasoning"),
                            position_value=round(cur * shares, 2),
                            company_name=p.get("company_name"),
                            sector=p.get("sector"),
                        ))

            portfolio_path = Path(f"{data_dir}/portfolio.json")
            if portfolio_path.exists():
                with open(portfolio_path, 'r') as f:
                    data = json.load(f)
                for i, p in enumerate(data.get('positions', [])):
                    trades.append(TradeResponse(
                        trade_id=f"paper_{p['symbol']}_{i}",
                        symbol=p['symbol'],
                        entry_date=p.get('entry_date', datetime.now().isoformat()),
                        entry_price=p['entry_price'],
                        shares=p.get('shares', p.get('quantity', 0)),
                        status='open',
                        pnl=p.get('pnl_amount'),
                        pnl_pct=p.get('pnl_percent'),
                        stop_loss=p.get('stop_loss'),
                        take_profit=p.get('take_profit'),
                        score_at_entry=p.get('score_at_entry'),
                        pillar_technical=p.get('pillar_technical'),
                        pillar_fundamental=p.get('pillar_fundamental'),
                        pillar_sentiment=p.get('pillar_sentiment'),
                        pillar_news=p.get('pillar_news'),
                        reasoning=p.get('reasoning'),
                        position_value=p.get('position_value'),
                        company_name=p.get('company_name'),
                        sector=p.get('sector'),
                        industry=p.get('industry')
                    ))
        
        # Read closed trades from trades_history.json
        if status != "open":
            history_path = Path(f"{data_dir}/trades_history.json")
            if history_path.exists():
                with open(history_path, 'r') as f:
                    data = json.load(f)
                for i, t in enumerate(data.get('trades', [])):
                    trades.append(TradeResponse(
                        trade_id=f"closed_{t['symbol']}_{i}",
                        symbol=t['symbol'],
                        entry_date=t.get('entry_date', ''),
                        entry_price=t['entry_price'],
                        shares=t.get('quantity', 0),
                        status='closed',
                        pnl=t.get('pnl_amount'),
                        pnl_pct=t.get('pnl_percent')
                    ))
        
        return trades
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
async def get_performance(agent: str = "llm", api_key: str = Depends(verify_api_key)):
    """Get trading performance metrics."""
    import json as _json
    from pathlib import Path as _Path
    try:
        data_dir = "data" if agent == "llm" else "data-nollm"
        suffix = "llm" if agent == "llm" else "nollm"
        ms_path = _Path(f"{data_dir}/multi_strategy_{suffix}.json")

        if ms_path.exists():
            with open(ms_path, "r") as f:
                ms_data = _json.load(f)
            total_trades = 0
            winning = 0
            losing = 0
            total_pnl = 0.0
            wins_sum = 0.0
            losses_sum = 0.0
            max_dd = 0.0
            open_pos = 0
            for sid, sdata in ms_data.get("strategies", {}).items():
                total_trades += sdata.get("total_trades", 0)
                winning += sdata.get("winning_trades", 0)
                losing += sdata.get("losing_trades", 0)
                total_pnl += sdata.get("total_pnl", 0)
                open_pos += len(sdata.get("positions", []))
                dd = abs(sdata.get("max_drawdown", 0))
                if dd > max_dd:
                    max_dd = dd
                for t in sdata.get("closed_trades", []):
                    pnl = t.get("pnl_amount", t.get("pnl", 0))
                    if pnl > 0:
                        wins_sum += pnl
                    else:
                        losses_sum += abs(pnl)
            closed = total_trades
            win_rate = (winning / closed * 100) if closed > 0 else 0
            avg_win = (wins_sum / winning) if winning > 0 else 0
            avg_loss = (losses_sum / losing) if losing > 0 else 0
            pf = (wins_sum / losses_sum) if losses_sum > 0 else None
            return PerformanceResponse(
                total_trades=total_trades + open_pos,
                open_trades=open_pos,
                closed_trades=closed,
                total_pnl=round(total_pnl, 2),
                win_rate=round(win_rate, 1),
                avg_win=round(avg_win, 2),
                avg_loss=round(avg_loss, 2),
                profit_factor=round(pf, 2) if pf else None,
                sharpe_ratio=None,
                max_drawdown=round(max_dd, 2)
            )

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
    """Get current stock price - LIVE, no cache."""
    import yfinance as yf
    from datetime import datetime
    
    try:
        ticker = yf.Ticker(symbol.upper())
        
        # Get 1-minute data for real-time price (NO CACHE)
        df = ticker.history(period="1d", interval="1m")
        
        if df is None or df.empty:
            # Fallback to daily
            df = ticker.history(period="5d", interval="1d")
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found")
        
        current_price = float(df["Close"].iloc[-1])
        
        # Get previous close for daily change
        df_daily = ticker.history(period="5d", interval="1d")
        if df_daily is not None and len(df_daily) > 1:
            prev_close = float(df_daily["Close"].iloc[-2])
        else:
            prev_close = current_price
        
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
        
        return {
            "symbol": symbol.upper(),
            "price": current_price,
            "change": change,
            "change_pct": change_pct,
            "volume": int(df["Volume"].iloc[-1]) if "Volume" in df else 0,
            "high": float(df["High"].iloc[-1]),
            "low": float(df["Low"].iloc[-1]),
            "date": df.index[-1].strftime("%Y-%m-%d %H:%M")
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


# ==================== Analysis Endpoint (V5.3) ====================

@app.post("/api/v1/analyze/{symbol}", tags=["Analysis"])
async def analyze_symbol_endpoint(symbol: str, background_tasks: BackgroundTasks):
    """
    Analyze a symbol using all pillars (Technical, Fundamental, Sentiment, News, ML).
    
    Returns:
    - finalScore: 0-100 overall score
    - decision: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    - pillars: Individual pillar scores and details
    - summary: AI-generated summary
    """
    from src.api.analyze_endpoint import analyze_symbol
    
    try:
        result = await analyze_symbol(symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analyze/{symbol}", tags=["Analysis"])
async def analyze_symbol_get(symbol: str):
    """GET version of analyze endpoint for convenience."""
    from src.api.analyze_endpoint import analyze_symbol
    
    try:
        result = await analyze_symbol(symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LEARNING / FEEDBACK ENDPOINTS
# =============================================================================

@app.get("/api/v1/learning/status", tags=["Learning"])
async def get_learning_status():
    """Get current learning/feedback status and pillar weights."""
    try:
        from src.learning.multi_pillar_feedback import get_multi_pillar_feedback
        mpf = get_multi_pillar_feedback()
        summary = mpf.get_summary()
        return {
            "status": "ok",
            "learning": summary
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/v1/learning/feedback", tags=["Learning"])
async def run_feedback_loop():
    """
    Manually trigger the feedback loop.
    Analyzes yesterday's market movers and adjusts indicator weights.
    """
    try:
        from src.learning.multi_pillar_feedback import run_multi_pillar_feedback
        result = await run_multi_pillar_feedback()
        return {
            "status": "ok",
            "result": result
        }
    except Exception as e:
        logger.error(f"Feedback loop error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/v1/learning/weights", tags=["Learning"])
async def get_pillar_weights():
    """Get current pillar weights."""
    try:
        from src.learning.multi_pillar_feedback import get_multi_pillar_feedback
        mpf = get_multi_pillar_feedback()
        weights = {}
        for pillar in ["technical", "fundamental", "sentiment", "news"]:
            if pillar in mpf.weights:
                weights[pillar] = {
                    "weight": mpf.weights[pillar].get("pillar_weight", 0.25),
                    "top_indicators": [
                        {"name": k, "weight": v["weight"], "accuracy": v["accuracy"]}
                        for k, v in sorted(
                            mpf.weights[pillar].get("indicators", {}).items(),
                            key=lambda x: x[1]["accuracy"],
                            reverse=True
                        )[:5]
                    ]
                }
        return {"status": "ok", "weights": weights}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==================== Agent Status Endpoint ====================

@app.get("/api/v1/agent/status", tags=["Agent"])
async def get_agent_status(agent: str = "llm", api_key: str = Depends(verify_api_key)):
    """Get live agent status, regime, metrics, hot symbols, and intelligence brief."""
    import json
    from pathlib import Path

    _llm_disabled = os.environ.get('DISABLE_LLM', '').strip() in ('1', 'true', 'yes')
    result = {
        "running": False,
        "phase": "unknown",
        "market_regime": "RANGE",
        "metrics": {},
        "hot_symbols": [],
        "intelligence_brief": None,
        "llm_enabled": not _llm_disabled,
    }

    data_dir = "data" if agent == "llm" else "data-nollm"

    # Read agent state
    state_path = Path(f"{data_dir}/agent_state.json")
    if state_path.exists():
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            result["running"] = state.get("running", False)
            result["phase"] = state.get("session", state.get("phase", "unknown"))
            result["market_regime"] = state.get("market_regime", "RANGE")
            result["metrics"] = state.get("metrics", {})
            result["hot_symbols"] = state.get("hot_symbols", [])
        except Exception:
            pass

    # Read intelligence brief
    brief_path = Path(f"{data_dir}/intelligence_brief.json")
    if brief_path.exists():
        try:
            with open(brief_path, "r") as f:
                result["intelligence_brief"] = json.load(f)
        except Exception:
            pass

    return result


# ==================== Portfolio History Endpoint ====================

@app.get("/api/v1/portfolio/history", tags=["Portfolio"])
async def get_portfolio_history(agent: str = "llm", api_key: str = Depends(verify_api_key)):
    """Get portfolio equity curve (daily history vs SPY)."""
    try:
        from src.utils.portfolio_tracker import get_performance_stats
        stats = get_performance_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Portfolio Positions Endpoint ====================

@app.get("/api/v1/portfolio/positions", tags=["Portfolio"])
async def get_portfolio_positions(agent: str = "llm", api_key: str = Depends(verify_api_key)):
    """Get detailed positions with allocation percentages and pillar scores."""
    import json
    from pathlib import Path

    data_dir = "data" if agent == "llm" else "data-nollm"
    suffix = "llm" if agent == "llm" else "nollm"
    ms_path = Path(f"{data_dir}/multi_strategy_{suffix}.json")

    # Try multi_strategy file first
    if ms_path.exists():
        try:
            with open(ms_path, "r") as f:
                ms_data = json.load(f)
            all_positions = []
            total_cash = 0
            for sid, sdata in ms_data.get("strategies", {}).items():
                total_cash += sdata.get("cash", 15000)
                for p in sdata.get("positions", []):
                    p["strategy"] = sdata.get("name", sid)
                    all_positions.append(p)
            total_invested = sum(
                p.get("current_price", p.get("entry_price", 0)) * p.get("shares", 0)
                for p in all_positions
            )
            total_value = total_cash + total_invested
            enriched = []
            for p in all_positions:
                pos_value = p.get("current_price", p.get("entry_price", 0)) * p.get("shares", 0)
                enriched.append({
                    **p,
                    "position_value": pos_value,
                    "allocation_pct": round((pos_value / total_value * 100), 2) if total_value > 0 else 0
                })
            return {
                "positions": enriched,
                "cash": round(total_cash, 2),
                "total_value": round(total_value, 2),
                "total_capital": round(total_value, 2),
                "positions_count": len(enriched)
            }
        except Exception:
            pass

    portfolio_path = Path(f"{data_dir}/portfolio.json")
    if not portfolio_path.exists():
        return {"positions": [], "total_value": 0}

    try:
        with open(portfolio_path, "r") as f:
            data = json.load(f)

        positions = data.get("positions", [])
        cash = data.get("cash", 0)
        total_capital = data.get("total_capital", cash)

        total_invested = sum(
            p.get("position_value", p.get("shares", 0) * p.get("entry_price", 0))
            for p in positions
        )
        total_value = cash + total_invested

        enriched = []
        for p in positions:
            pos_value = p.get("position_value", p.get("shares", 0) * p.get("entry_price", 0))
            enriched.append({
                **p,
                "position_value": pos_value,
                "allocation_pct": (pos_value / total_value * 100) if total_value > 0 else 0,
            })

        return {
            "positions": enriched,
            "cash": cash,
            "total_value": total_value,
            "total_capital": total_capital,
            "positions_count": len(positions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SPA Catch-All (MUST be last) ====================

# ============================================================
# V8.1: Multi-Strategy Comparison Endpoints
# ============================================================

@app.get("/api/v1/strategies/comparison", tags=["Strategies"])
async def get_strategy_comparison():
    """Get comparison of all 8 strategies (2 agents x 4 profiles)."""
    import json, os
    all_strategies = []
    
    # Agent WITH LLM
    llm_file = "data/multi_strategy_llm.json"
    if os.path.exists(llm_file):
        try:
            with open(llm_file, "r") as f:
                data = json.load(f)
            for sid, sdata in data.get("strategies", {}).items():
                sdata["id"] = f"llm_{sid}"
                sdata["name"] = f"🤖 {sdata.get('name', sid)}"
                sdata["agent"] = "llm"
                all_strategies.append(sdata)
        except Exception as e:
            logger.warning(f"Error reading LLM strategies: {e}")
    
    # Agent WITHOUT LLM
    nollm_file = "data-nollm/multi_strategy_nollm.json"
    if os.path.exists(nollm_file):
        try:
            with open(nollm_file, "r") as f:
                data = json.load(f)
            for sid, sdata in data.get("strategies", {}).items():
                sdata["id"] = f"nollm_{sid}"
                sdata["name"] = f"📊 {sdata.get('name', sid)}"
                sdata["agent"] = "nollm"
                all_strategies.append(sdata)
        except Exception as e:
            logger.warning(f"Error reading No-LLM strategies: {e}")
    
    # Fallback: try default file if no dual-agent data
    if not all_strategies:
        try:
            from src.agents.multi_strategy_tracker import get_multi_tracker
            tracker = get_multi_tracker()
            return tracker.get_comparison()
        except Exception as e:
            logger.error(f"Strategy comparison error: {e}")
    
    # Sort by return
    all_strategies.sort(key=lambda x: x.get("return_pct", ((x.get("cash", 15000) + sum(p.get("current_price", 0) * p.get("shares", 0) for p in x.get("positions", []))) - 15000) / 150), reverse=True)
    
    # Compute return_pct and other derived fields if missing
    for s in all_strategies:
        equity = s.get("cash", 15000)
        for p in s.get("positions", []):
            equity += p.get("current_price", 0) * p.get("shares", 0)
        initial = s.get("initial_capital", 15000)
        s["equity"] = round(equity, 2)
        s["return_pct"] = round(((equity - initial) / initial) * 100, 2) if initial > 0 else 0
        s["open_positions"] = len(s.get("positions", []))
        total = s.get("total_trades", 0)
        wins = s.get("winning_trades", 0)
        s["win_rate"] = round((wins / total) * 100, 1) if total > 0 else 0
        if "recent_trades" not in s:
            s["recent_trades"] = s.get("closed_trades_list", [])[-10:] if isinstance(s.get("closed_trades_list"), list) else []
    
    return {
        "strategies": all_strategies,
        "agents": {"llm": "🤖 With LLMs", "nollm": "📊 Without LLMs"},
        "total_strategies": len(all_strategies),
        "updated_at": datetime.now().isoformat()
    }

@app.post("/api/v1/strategies/reset", tags=["Strategies"])
async def reset_strategies():
    """Reset all virtual strategy portfolios."""
    try:
        from src.agents.multi_strategy_tracker import get_multi_tracker
        tracker = get_multi_tracker()
        tracker.reset()
        return {"status": "ok", "message": "All strategies reset to initial capital"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/v1/strategies/profiles", tags=["Strategies"])
async def get_strategy_profiles():
    """Get the 4 strategy profile definitions."""
    try:
        from config.strategies import get_all_profiles
        profiles = get_all_profiles()
        return {
            pid: {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "color": p.color,
                "min_score": p.min_score,
                "max_positions": p.max_positions,
                "position_size_pct": p.position_size_pct,
                "use_ml_gate": p.use_ml_gate,
                "pillar_weights": p.pillar_weights,
            }
            for pid, p in profiles.items()
        }
    except Exception as e:
        return {"error": str(e)}



@app.get("/api/v1/strategies/{strategy_id}", tags=["Strategies"])
async def get_strategy_detail(strategy_id: str):
    """Get detailed info for a single strategy (positions, trades, reasoning)."""
    import json, os
    
    # Parse agent and strategy from id (e.g. llm_aggressive_ml or nollm_moderate_no_ml)
    if strategy_id.startswith('llm_'):
        agent = 'llm'
        sid = strategy_id[4:]
        data_file = 'data/multi_strategy_llm.json'
    elif strategy_id.startswith('nollm_'):
        agent = 'nollm'
        sid = strategy_id[6:]
        data_file = 'data-nollm/multi_strategy_nollm.json'
    else:
        return {"error": "Invalid strategy_id. Use llm_ or nollm_ prefix."}
    
    if not os.path.exists(data_file):
        return {"error": f"Data file not found: {data_file}"}
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        sdata = data.get('strategies', {}).get(sid)
        if not sdata:
            return {"error": f"Strategy {sid} not found in {agent} agent"}
        
        # Enrich positions with P&L
        positions = sdata.get('positions', [])
        for p in positions:
            entry = p.get('entry_price', 0)
            current = p.get('current_price', entry)
            if entry > 0:
                p['pnl_pct'] = round(((current - entry) / entry) * 100, 2)
                p['pnl_amount'] = round((current - entry) * p.get('shares', 0), 2)
            else:
                p['pnl_pct'] = 0
                p['pnl_amount'] = 0
        
        # Compute equity
        equity = sdata.get('cash', 15000)
        for p in positions:
            equity += p.get('current_price', 0) * p.get('shares', 0)
        
        initial = sdata.get('initial_capital', 15000)
        
        # Get strategy profile info
        try:
            from config.strategies import get_all_profiles
            profile = get_all_profiles().get(sid)
            profile_info = {
                'description': profile.description if profile else '',
                'min_score': profile.min_score if profile else 0,
                'max_positions': profile.max_positions if profile else 0,
                'position_size_pct': profile.position_size_pct if profile else 0,
                'use_ml_gate': profile.use_ml_gate if profile else False,
                'pillar_weights': profile.pillar_weights if profile else {},
            } if profile else {}
        except:
            profile_info = {}
        
        return {
            'id': strategy_id,
            'strategy_id': sid,
            'agent': agent,
            'name': sdata.get('name', sid),
            'color': sdata.get('color', '#888'),
            'equity': round(equity, 2),
            'cash': round(sdata.get('cash', 15000), 2),
            'initial_capital': initial,
            'return_pct': round(((equity - initial) / initial) * 100, 2) if initial > 0 else 0,
            'total_trades': sdata.get('total_trades', 0),
            'winning_trades': sdata.get('winning_trades', 0),
            'losing_trades': sdata.get('losing_trades', 0),
            'total_pnl': sdata.get('total_pnl', 0),
            'max_drawdown': sdata.get('max_drawdown', 0),
            'signals_evaluated': sdata.get('signals_evaluated', 0),
            'signals_accepted': sdata.get('signals_accepted', 0),
            'signals_rejected': sdata.get('signals_rejected', 0),
            'positions': positions,
            'closed_trades': sdata.get('closed_trades', []),
            'equity_curve': sdata.get('equity_curve', []),
            'profile': profile_info,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}




# ==================== Activity Logs Endpoint ====================

@app.get("/api/v1/activity", tags=["Activity"])
async def get_activity_logs(
    limit: int = Query(default=50, ge=1, le=500),
    type: Optional[str] = Query(None, alias="type"),
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    agent: Optional[str] = None,
):
    """Get bot activity logs with optional filters."""
    try:
        from src.utils.activity_logger import get_activities
        entries = get_activities(limit=limit, activity_type=type, strategy=strategy, symbol=symbol, agent=agent)
        return {"activities": entries, "count": len(entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/{path:path}", include_in_schema=False)
async def serve_spa(path: str):
    """Serve the React SPA for any non-API route."""
    # Never intercept API routes
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail=f"API route not found: /{path}")
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.isdir(DASHBOARD_DIR) and os.path.isfile(index_path):
        return FileResponse(index_path)
    return {
        "name": "TradingBot V3 API",
        "version": "3.0.0",
        "docs": "/docs",
        "dashboard": "not built - run: cd dashboard && npm run build",
    }


