"""
Backtest Discovery Mode - Détection de pépites historiques
Objectif: Valider si nos critères auraient détecté les futures stars ET évité les crashs
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
OUTPUT_FILE = "/root/tradingbot-github/data/backtest_discovery_results.json"

# Stocks à tester - Mix de winners et losers historiques
WINNERS = [
    # Tech giants quand ils étaient petits
    "NVDA", "AMD", "TSLA", "META", "SHOP", "SQ", "ROKU", "TTD", "CRWD", "NET",
    "DDOG", "ZS", "SNOW", "PLTR", "ABNB", "COIN", "AFRM", "HOOD", "RBLX", "U"
]

LOSERS = [
    # Stocks qui ont crashé après un hype
    "PTON", "ZM", "DOCU", "BYND", "SPCE", "NKLA", "WISH", "CLOV", "SOFI", "LCID",
    "RIVN", "DWAC", "BBBY", "AMC", "GME", "PLBY", "WKHS", "RIDE", "QS", "LAZR"
]

ALL_STOCKS = WINNERS + LOSERS

def get_stock_data(symbol, start_date, end_date):
    """Récupère les données historiques d un stock"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d")
        info = ticker.info
        return df, info
    except Exception as e:
        logger.warning(f"Erreur {symbol}: {e}")
        return None, None

def get_google_trends(symbol, timeframe="today 5-y"):
    """Récupère les données Google Trends"""
    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload([symbol], timeframe=timeframe, geo="US")
        data = pytrends.interest_over_time()
        time.sleep(2)  # Rate limiting
        return data
    except Exception as e:
        logger.warning(f"Google Trends erreur {symbol}: {e}")
        return None

def calculate_momentum(df, window=20):
    """Calcule le momentum sur une fenêtre"""
    if df is None or len(df) < window:
        return None
    returns = df["Close"].pct_change(window).iloc[-1]
    return returns * 100

def calculate_volume_surge(df, window=20):
    """Calcule si le volume est en hausse vs moyenne"""
    if df is None or len(df) < window:
        return None
    avg_vol = df["Volume"].rolling(window).mean().iloc[-1]
    current_vol = df["Volume"].iloc[-1]
    return (current_vol / avg_vol - 1) * 100 if avg_vol > 0 else 0

def calculate_rsi(df, period=14):
    """Calcule le RSI"""
    if df is None or len(df) < period + 1:
        return None
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_trends_momentum(trends_data, symbol, weeks=4):
    """Calcule l augmentation de l intérêt Google Trends"""
    if trends_data is None or trends_data.empty:
        return None
    col = symbol if symbol in trends_data.columns else trends_data.columns[0]
    if len(trends_data) < weeks:
        return None
    recent = trends_data[col].iloc[-weeks:].mean()
    previous = trends_data[col].iloc[-weeks*2:-weeks].mean() if len(trends_data) >= weeks*2 else recent
    return ((recent / previous) - 1) * 100 if previous > 0 else 0

def analyze_stock(symbol, is_winner):
    """Analyse complète d un stock"""
    logger.info(f"Analyse de {symbol} ({'WINNER' if is_winner else 'LOSER'})...")
    
    # Données sur 5 ans
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    df, info = get_stock_data(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    trends = get_google_trends(symbol)
    
    if df is None or df.empty:
        return None
    
    # Calculer les métriques à différents points dans le temps
    results = {
        "symbol": symbol,
        "is_winner": is_winner,
        "current_price": df["Close"].iloc[-1] if not df.empty else None,
        "price_5y_ago": df["Close"].iloc[0] if not df.empty else None,
        "total_return_5y": ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100 if not df.empty else None,
        "max_price": df["Close"].max() if not df.empty else None,
        "min_price": df["Close"].min() if not df.empty else None,
        "volatility": df["Close"].pct_change().std() * 100 if not df.empty else None,
        "momentum_20d": calculate_momentum(df, 20),
        "momentum_60d": calculate_momentum(df, 60),
        "volume_surge": calculate_volume_surge(df, 20),
        "rsi": calculate_rsi(df),
        "trends_momentum_4w": calculate_trends_momentum(trends, symbol, 4),
        "trends_current": trends[symbol].iloc[-1] if trends is not None and symbol in trends.columns else None,
        "trends_max": trends[symbol].max() if trends is not None and symbol in trends.columns else None,
        "market_cap": info.get("marketCap") if info else None,
        "pe_ratio": info.get("trailingPE") if info else None,
        "revenue_growth": info.get("revenueGrowth") if info else None,
        "profit_margin": info.get("profitMargins") if info else None,
    }
    
    return results

def run_backtest():
    """Lance le backtest complet"""
    logger.info("=" * 60)
    logger.info("BACKTEST DISCOVERY MODE - Début")
    logger.info("=" * 60)
    
    results = []
    
    for symbol in ALL_STOCKS:
        is_winner = symbol in WINNERS
        result = analyze_stock(symbol, is_winner)
        if result:
            results.append(result)
        time.sleep(1)  # Rate limiting
    
    # Analyse des résultats
    df_results = pd.DataFrame(results)
    
    winners_df = df_results[df_results["is_winner"] == True]
    losers_df = df_results[df_results["is_winner"] == False]
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_stocks": len(results),
        "winners_count": len(winners_df),
        "losers_count": len(losers_df),
        "winners_avg_return": winners_df["total_return_5y"].mean(),
        "losers_avg_return": losers_df["total_return_5y"].mean(),
        "winners_avg_trends": winners_df["trends_current"].mean(),
        "losers_avg_trends": losers_df["trends_current"].mean(),
        "winners_avg_momentum": winners_df["momentum_20d"].mean(),
        "losers_avg_momentum": losers_df["momentum_20d"].mean(),
        "discriminating_factors": [],
        "raw_results": results
    }
    
    # Trouver les facteurs discriminants
    for col in ["momentum_20d", "volume_surge", "rsi", "trends_momentum_4w", "trends_current"]:
        w_mean = winners_df[col].mean() if col in winners_df else None
        l_mean = losers_df[col].mean() if col in losers_df else None
        if w_mean and l_mean and abs(w_mean - l_mean) > 5:
            analysis["discriminating_factors"].append({
                "factor": col,
                "winners_avg": w_mean,
                "losers_avg": l_mean,
                "difference": w_mean - l_mean
            })
    
    # Sauvegarder
    with open(OUTPUT_FILE, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("BACKTEST TERMINÉ")
    logger.info(f"Résultats sauvegardés dans {OUTPUT_FILE}")
    logger.info(f"Winners avg return: {analysis[winners_avg_return]:.1f}%")
    logger.info(f"Losers avg return: {analysis[losers_avg_return]:.1f}%")
    logger.info("=" * 60)
    
    return analysis

if __name__ == "__main__":
    run_backtest()
