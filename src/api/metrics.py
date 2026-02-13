"""
Midas Trading Bot - Prometheus Metrics Module

Defines Prometheus metrics for monitoring trades, portfolio performance,
API latency, and intelligence source health.

Usage:
    from src.api.metrics import setup_metrics, record_trade, set_win_rate, ...
    setup_metrics(app)  # call once during FastAPI startup
"""

import time
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    make_asgi_app,
)
from starlette.requests import Request
from starlette.routing import Mount


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Trades
midas_trades_total = Counter(
    "midas_trades_total",
    "Total number of trades executed",
    labelnames=["agent_id", "strategy", "direction"],
)

# Win rate per agent/strategy
midas_win_rate = Gauge(
    "midas_win_rate",
    "Current win rate (0-100) for an agent/strategy",
    labelnames=["agent_id", "strategy"],
)

# Drawdown
midas_drawdown_pct = Gauge(
    "midas_drawdown_pct",
    "Current drawdown percentage for an agent",
    labelnames=["agent_id"],
)

# Portfolio value
midas_portfolio_value = Gauge(
    "midas_portfolio_value",
    "Current portfolio value in USD",
    labelnames=["agent_id", "strategy"],
)

# Cycle duration (how long one bot cycle takes)
midas_cycle_duration_seconds = Histogram(
    "midas_cycle_duration_seconds",
    "Duration of a single bot cycle in seconds",
    labelnames=["agent_id"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

# API request latency
midas_api_request_duration_seconds = Histogram(
    "midas_api_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["endpoint", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# Intelligence source health (1 = ok, 0 = failed)
midas_intelligence_source_status = Gauge(
    "midas_intelligence_source_status",
    "Intelligence source status (1=ok, 0=failed)",
    labelnames=["source"],
)

# RSS scraper gauge
midas_rss_articles_scraped = Gauge(
    "midas_rss_articles_scraped",
    "Number of RSS articles scraped in the latest cycle",
)


# ---------------------------------------------------------------------------
# Convenience recording helpers
# ---------------------------------------------------------------------------

def record_trade(agent_id: str, strategy: str, direction: str) -> None:
    """Increment the trade counter."""
    midas_trades_total.labels(
        agent_id=agent_id, strategy=strategy, direction=direction
    ).inc()


def set_win_rate(agent_id: str, strategy: str, rate: float) -> None:
    """Set the current win rate gauge (0-100)."""
    midas_win_rate.labels(agent_id=agent_id, strategy=strategy).set(rate)


def set_drawdown(agent_id: str, pct: float) -> None:
    """Set the current drawdown percentage gauge."""
    midas_drawdown_pct.labels(agent_id=agent_id).set(pct)


def set_portfolio_value(agent_id: str, strategy: str, value: float) -> None:
    """Set the current portfolio value gauge."""
    midas_portfolio_value.labels(agent_id=agent_id, strategy=strategy).set(value)


def observe_cycle_duration(agent_id: str, duration_seconds: float) -> None:
    """Record a bot-cycle duration observation."""
    midas_cycle_duration_seconds.labels(agent_id=agent_id).observe(duration_seconds)


def set_intelligence_source_status(source: str, ok: bool) -> None:
    """Set intelligence source status (True=1, False=0)."""
    midas_intelligence_source_status.labels(source=source).set(1 if ok else 0)


def set_rss_articles_scraped(count: int) -> None:
    """Set the number of RSS articles scraped."""
    midas_rss_articles_scraped.set(count)


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------

def setup_metrics(app) -> None:
    """
    Mount the /metrics endpoint and add request-duration middleware.

    Call this once after creating the FastAPI app instance::

        from src.api.metrics import setup_metrics
        setup_metrics(app)
    """

    # Mount the prometheus ASGI app at /metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Middleware to time every HTTP request
    @app.middleware("http")
    async def _track_request_duration(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        # Avoid recording /metrics itself to prevent self-referential noise
        path = request.url.path
        if not path.startswith("/metrics"):
            midas_api_request_duration_seconds.labels(
                endpoint=path, method=request.method
            ).observe(elapsed)

        return response
