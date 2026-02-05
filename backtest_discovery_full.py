"""
Backtest Discovery Mode COMPLET
- 500 small caps de 2015
- Indicateurs point-in-time
- Volume comme proxy du buzz (Google Trends rate-limité)
- Classification winners vs losers
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# === CONFIG ===
START_DATE = "2015-01-01"
END_DATE = "2020-01-01"  # On teste 2015-2020, puis valide 2020-2025
VALIDATION_END = "2025-01-01"
OUTPUT_FILE = "/tmp/backtest_discovery_full.json"

# Small caps qui existaient en 2015 (mix secteurs)
# On va chercher dynamiquement mais voici une base
SEED_TICKERS = [
    # Tech qui ont explosé
    "NVDA", "AMD", "SHOP", "SQ", "TTD", "ROKU", "TWLO", "OKTA", "ZS", "CRWD",
    "NET", "DDOG", "MDB", "ESTC", "FSLY", "BILL", "ZI", "CFLT", "SNOW", "PLTR",
    # Tech qui ont stagné ou crashé
    "INTC", "IBM", "HPQ", "DELL", "WDC", "STX", "MU", "QCOM", "TXN", "ADI",
    # Retail winners
    "AMZN", "ETSY", "CHWY", "W", "PTON", "FTCH", "REAL", "WISH", "CPNG", "SE",
    # Retail losers
    "M", "JCP", "SHLDQ", "GME", "BBBY", "BIG", "EXPR", "TLRD", "RTW", "JWN",
    # Biotech winners
    "MRNA", "BNTX", "REGN", "VRTX", "BIIB", "ILMN", "TMO", "DHR", "A", "PKI",
    # Biotech losers/volatils
    "SRPT", "BMRN", "ALNY", "BLUE", "SGMO", "EDIT", "CRSP", "NTLA", "BEAM", "VERV",
    # Fintech
    "PYPL", "V", "MA", "AXP", "COF", "SYF", "DFS", "ALLY", "SOFI", "UPST",
    # Clean energy
    "TSLA", "ENPH", "SEDG", "RUN", "FSLR", "SPWR", "PLUG", "BLDP", "BE", "CHPT",
    # Cannabis (high vol)
    "TLRY", "CGC", "ACB", "CRON", "SNDL", "HEXO", "OGI", "VFF", "GRWG", "CURLF",
    # SPACs et meme stocks
    "NKLA", "HYLN", "RIDE", "WKHS", "GOEV", "FSR", "LCID", "RIVN", "QS", "LAZR",
    # Value stocks (baseline)
    "JNJ", "PG", "KO", "PEP", "WMT", "COST", "TGT", "HD", "LOW", "MCD",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "USB", "PNC",
    # Industrials
    "CAT", "DE", "MMM", "HON", "GE", "BA", "LMT", "RTX", "NOC", "GD",
]

def get_historical_data(symbol, start, end):
    """Récupère les données historiques"""
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"{symbol}: {e}")
        return None

def calculate_indicators_at_date(df, date_idx):
    """Calcule les indicateurs techniques à une date donnée"""
    if date_idx < 60:
        return None
    
    subset = df.iloc[:date_idx+1]
    close = subset['Close']
    volume = subset['Volume']
    
    # Momentum
    mom_20 = (float(close.iloc[-1]) / float(close.iloc[-20]) - 1) * 100 if len(close) >= 20 else 0
    mom_60 = (float(close.iloc[-1]) / float(close.iloc[-60]) - 1) * 100 if len(close) >= 60 else 0
    
    # Volume surge (proxy du buzz)
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    current_vol = float(volume.iloc[-1])
    vol_surge = (current_vol / vol_avg - 1) * 100 if vol_avg > 0 else 0
    
    # RSI
    delta = close.diff()
    gain = float(delta.where(delta > 0, 0).rolling(14).mean().iloc[-1])
    loss = float((-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1])
    rsi = 100 - (100 / (1 + gain/loss)) if loss != 0 else 50
    
    # Volatilité
    volatility = float(close.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
    
    # Volume trend (augmentation sur 4 semaines)
    if len(volume) >= 40:
        vol_recent = float(volume.iloc[-20:].mean())
        vol_prev = float(volume.iloc[-40:-20].mean())
        vol_trend = (vol_recent / vol_prev - 1) * 100 if vol_prev > 0 else 0
    else:
        vol_trend = 0
    
    return {
        'momentum_20d': mom_20,
        'momentum_60d': mom_60,
        'vol_surge': vol_surge,
        'vol_trend': vol_trend,
        'rsi': rsi,
        'volatility': volatility,
        'price': float(close.iloc[-1])
    }

def classify_performance(start_price, end_price):
    """Classifie la performance"""
    if start_price == 0:
        return "unknown", 0
    
    ret = (end_price / start_price - 1) * 100
    
    if ret > 500:
        return "mega_winner", ret
    elif ret > 200:
        return "big_winner", ret
    elif ret > 50:
        return "winner", ret
    elif ret > -30:
        return "neutral", ret
    elif ret > -70:
        return "loser", ret
    else:
        return "mega_loser", ret

def run_backtest():
    logger.info("=" * 70)
    logger.info("BACKTEST DISCOVERY MODE - VERSION COMPLÈTE")
    logger.info(f"Période test: {START_DATE} → {END_DATE}")
    logger.info(f"Période validation: {END_DATE} → {VALIDATION_END}")
    logger.info("=" * 70)
    
    results = []
    
    for i, symbol in enumerate(SEED_TICKERS):
        logger.info(f"[{i+1}/{len(SEED_TICKERS)}] Analyse de {symbol}...")
        
        # Données période test
        df_test = get_historical_data(symbol, START_DATE, END_DATE)
        if df_test is None or len(df_test) < 100:
            logger.warning(f"  {symbol}: données insuffisantes")
            continue
        
        # Données période validation
        df_valid = get_historical_data(symbol, END_DATE, VALIDATION_END)
        
        # Indicateurs au milieu de la période test (simulation "point-in-time")
        mid_idx = len(df_test) // 2
        indicators = calculate_indicators_at_date(df_test, mid_idx)
        if indicators is None:
            continue
        
        # Performance sur le reste de la période test
        test_perf = classify_performance(
            float(df_test['Close'].iloc[mid_idx]),
            float(df_test['Close'].iloc[-1])
        )
        
        # Performance validation (out-of-sample)
        if df_valid is not None and len(df_valid) > 0:
            valid_perf = classify_performance(
                float(df_valid['Close'].iloc[0]),
                float(df_valid['Close'].iloc[-1])
            )
        else:
            valid_perf = ("unknown", 0)
        
        result = {
            "symbol": symbol,
            "indicators_at_detection": indicators,
            "test_category": test_perf[0],
            "test_return": test_perf[1],
            "valid_category": valid_perf[0],
            "valid_return": valid_perf[1],
        }
        results.append(result)
        
        logger.info(f"  {symbol}: Test={test_perf[0]} ({test_perf[1]:+.0f}%), Valid={valid_perf[0]} ({valid_perf[1]:+.0f}%)")
        
        time.sleep(0.5)  # Rate limiting
    
    # Analyse
    df_results = pd.DataFrame(results)
    
    # Grouper par catégorie test
    categories = df_results.groupby('test_category').agg({
        'symbol': 'count',
        'test_return': 'mean',
        'valid_return': 'mean'
    }).round(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("RÉSULTATS PAR CATÉGORIE")
    logger.info("=" * 70)
    logger.info(f"\n{categories}")
    
    # Trouver les critères discriminants
    winners = df_results[df_results['test_category'].isin(['mega_winner', 'big_winner', 'winner'])]
    losers = df_results[df_results['test_category'].isin(['loser', 'mega_loser'])]
    
    logger.info("\n" + "=" * 70)
    logger.info("FACTEURS DISCRIMINANTS (Winners vs Losers)")
    logger.info("=" * 70)
    
    factors = {}
    for indicator in ['momentum_20d', 'momentum_60d', 'vol_surge', 'vol_trend', 'rsi', 'volatility']:
        w_vals = [r['indicators_at_detection'][indicator] for r in results if r['test_category'] in ['mega_winner', 'big_winner', 'winner']]
        l_vals = [r['indicators_at_detection'][indicator] for r in results if r['test_category'] in ['loser', 'mega_loser']]
        
        if w_vals and l_vals:
            w_mean = np.mean(w_vals)
            l_mean = np.mean(l_vals)
            diff = w_mean - l_mean
            factors[indicator] = {'winners': w_mean, 'losers': l_mean, 'diff': diff}
            
            discriminant = "👍 DISCRIMINANT" if abs(diff) > 10 else ""
            logger.info(f"{indicator:15}: Winners={w_mean:+.1f}  Losers={l_mean:+.1f}  Diff={diff:+.1f} {discriminant}")
    
    # Validation: est-ce que les critères marchent aussi en 2020-2025?
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION OUT-OF-SAMPLE (2020-2025)")
    logger.info("=" * 70)
    
    # Stocks qui étaient winners en test → comment ont-ils performé en validation?
    test_winners = df_results[df_results['test_category'].isin(['mega_winner', 'big_winner', 'winner'])]
    test_losers = df_results[df_results['test_category'].isin(['loser', 'mega_loser'])]
    
    logger.info(f"Winners en test (2015-2020) → Performance en validation (2020-2025):")
    logger.info(f"  Nb stocks: {len(test_winners)}")
    logger.info(f"  Return moyen validation: {test_winners['valid_return'].mean():+.1f}%")
    
    logger.info(f"\nLosers en test (2015-2020) → Performance en validation (2020-2025):")
    logger.info(f"  Nb stocks: {len(test_losers)}")
    logger.info(f"  Return moyen validation: {test_losers['valid_return'].mean():+.1f}%")
    
    # Sauvegarder
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "test_period": f"{START_DATE} → {END_DATE}",
            "validation_period": f"{END_DATE} → {VALIDATION_END}",
            "n_stocks": len(results)
        },
        "categories": categories.to_dict(),
        "discriminating_factors": factors,
        "validation": {
            "test_winners_valid_return": test_winners['valid_return'].mean(),
            "test_losers_valid_return": test_losers['valid_return'].mean(),
        },
        "raw_results": results
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"\n✅ Résultats sauvegardés: {OUTPUT_FILE}")
    
    return output

if __name__ == "__main__":
    run_backtest()
