"""
Portfolio Performance Tracker
Enregistre la valeur du portefeuille toutes les 5 minutes pour construire l historique.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yfinance as yf

logger = logging.getLogger(__name__)

HISTORY_FILE = Path("/app/data/portfolio_history.json")
PORTFOLIO_FILE = Path("/app/data/portfolio.json")


def load_history() -> Dict:
    """Charge l historique existant."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except:
            pass
    return {"snapshots": [], "daily_summary": {}}


def save_history(history: Dict):
    """Sauvegarde l historique."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def get_current_price(symbol: str) -> Optional[float]:
    """Recupere le prix actuel via yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="5m")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Could not get price for {symbol}: {e}")
    return None


def get_spy_price() -> Optional[float]:
    """Recupere le prix SPY pour benchmark."""
    return get_current_price("SPY")


def take_snapshot() -> Dict:
    """Prend un snapshot du portefeuille avec prix live."""
    
    # Charger le portefeuille
    if not PORTFOLIO_FILE.exists():
        return {"error": "No portfolio file"}
    
    with open(PORTFOLIO_FILE) as f:
        portfolio = json.load(f)
    
    positions = portfolio.get("positions", [])
    cash = portfolio.get("cash", 0)
    
    # Calculer la valeur avec prix live
    total_value = cash
    position_details = []
    
    for pos in positions:
        symbol = pos["symbol"]
        shares = pos["shares"]
        entry_price = pos["entry_price"]
        
        current_price = get_current_price(symbol)
        if current_price is None:
            current_price = entry_price  # Fallback
        
        position_value = shares * current_price
        pnl = (current_price - entry_price) * shares
        pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        
        total_value += position_value
        position_details.append({
            "symbol": symbol,
            "shares": shares,
            "entry_price": entry_price,
            "current_price": current_price,
            "value": position_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
    
    # SPY pour benchmark
    spy_price = get_spy_price()
    
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "total_value": total_value,
        "cash": cash,
        "invested": total_value - cash,
        "positions_count": len(positions),
        "spy_price": spy_price,
        "positions": position_details
    }
    
    return snapshot


def record_snapshot():
    """Enregistre un snapshot dans l historique."""
    history = load_history()
    snapshot = take_snapshot()
    
    if "error" not in snapshot:
        # Ajouter au snapshots (garder les 2000 derniers = ~7 jours a 5min)
        history["snapshots"].append(snapshot)
        if len(history["snapshots"]) > 2000:
            history["snapshots"] = history["snapshots"][-2000:]
        
        # Mettre a jour le summary quotidien
        date_key = datetime.now().strftime("%Y-%m-%d")
        if date_key not in history["daily_summary"]:
            history["daily_summary"][date_key] = {
                "open": snapshot["total_value"],
                "high": snapshot["total_value"],
                "low": snapshot["total_value"],
                "close": snapshot["total_value"],
                "spy_open": snapshot["spy_price"],
                "spy_close": snapshot["spy_price"]
            }
        else:
            day = history["daily_summary"][date_key]
            day["close"] = snapshot["total_value"]
            day["high"] = max(day["high"], snapshot["total_value"])
            day["low"] = min(day["low"], snapshot["total_value"])
            day["spy_close"] = snapshot["spy_price"]
        
        save_history(history)
        logger.info(f"Portfolio snapshot: ${snapshot['total_value']:.2f}")
    
    return snapshot


def get_performance_stats() -> Dict:
    """Calcule les stats de performance."""
    history = load_history()
    daily = history.get("daily_summary", {})
    
    if not daily:
        return {"error": "No history yet"}
    
    dates = sorted(daily.keys())
    first_day = daily[dates[0]]
    last_day = daily[dates[-1]]
    
    # Performance totale
    start_value = first_day["open"]
    end_value = last_day["close"]
    total_return = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0
    
    # Performance SPY
    spy_start = first_day.get("spy_open", 0)
    spy_end = last_day.get("spy_close", 0)
    spy_return = ((spy_end / spy_start) - 1) * 100 if spy_start > 0 else 0
    
    # Alpha
    alpha = total_return - spy_return
    
    return {
        "start_date": dates[0],
        "end_date": dates[-1],
        "days_tracked": len(dates),
        "start_value": start_value,
        "end_value": end_value,
        "total_return_pct": total_return,
        "spy_return_pct": spy_return,
        "alpha_pct": alpha,
        "daily_history": daily
    }


if __name__ == "__main__":
    # Test
    snapshot = record_snapshot()
    print(json.dumps(snapshot, indent=2))
    
    stats = get_performance_stats()
    print("\nPerformance Stats:")
    print(json.dumps(stats, indent=2))
