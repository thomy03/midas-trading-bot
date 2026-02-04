"""Dashboard pages package

Each page module exposes a render() function that displays the page content.

Usage:
    from src.dashboard.pages import PAGE_MODULES
    page_module = PAGE_MODULES.get(page_name)
    if page_module:
        page_module.render()
"""

from . import (
    home,
    pro_chart,
    chart_analyzer,
    signals_history,
    screening,
    scheduler,
    backtesting,
    watchlists,
    portfolio,
    performance,
    sector_map,
    calendar_page,
    trendline_analysis,
    intelligence,
    trend_discovery,
    alerts_history,
    settings,
)

PAGE_MODULES = {
    "🏠 Home": home,
    "📈 Pro Chart": pro_chart,
    "📊 Chart Analyzer": chart_analyzer,
    "📈 Signaux Historiques": signals_history,
    "🔍 Screening": screening,
    "⏰ Scheduler": scheduler,
    "🔬 Backtesting": backtesting,
    "📋 Watchlists": watchlists,
    "💼 Portfolio": portfolio,
    "📊 Performance": performance,
    "🗺️ Sector Map": sector_map,
    "📅 Calendar": calendar_page,
    "🎯 Trendline Analysis": trendline_analysis,
    "🧠 Intelligence": intelligence,
    "🔮 Trend Discovery": trend_discovery,
    "🚨 Alerts History": alerts_history,
    "⚙️ Settings": settings,
}

PAGE_NAMES = list(PAGE_MODULES.keys())

__all__ = ["PAGE_MODULES", "PAGE_NAMES"]
